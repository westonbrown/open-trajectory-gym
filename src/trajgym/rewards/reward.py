"""CTF reward function for online GRPO training.

Physics-inspired scoring (8 configurable signals, sum to 1.0 + penalty):

  Primary (v9 production weights):
  - Flag capture (0.85): Boundary condition -- did the trajectory reach the goal?
  - Efficiency (0.10): Least action -- step_ratio × action_novelty × temporal_decay.
  - Format compliance (0.05): Signal fidelity -- valid instrument readings only.

  Available but zeroed in v9 (reintroduce when >10% flag capture on hard challenges):
  - Recovery (0.0): Resilience -- trajectory pivots after stuck runs.
  - Progression (0.0): Phase space ordering -- RECON->ENUM->EXPLOIT trajectory.
  - Cognitive (0.0): Words-per-action -- optimal reasoning density (~42 WPA).
  - Exploration (0.0): Exponentially-decayed novelty -- early diversity > late.
  - Uniqueness (0.0): Information entropy -- repeated observations carry zero bits.
  - Hallucination penalty (0.0): Energy loss -- false claims decay trajectory progress.

Design principles:
  - **Principle of least action**: Efficiency score = step_ratio × novelty × time_decay.
    step_ratio = min(optimal/actual, 1.0) penalizes long trajectories.
    novelty = unique_actions/total_actions penalizes redundant commands.
    time_decay = exp(-λ × excess_steps) exponentially discounts late progress.
    Combined: the optimal trajectory reaches the flag via the shortest non-redundant path.
  - **Flag-dominant for RLOO**: v7 showed 4-signal reward compressed RLOO advantages
    to ±0.12 → zero gradient. v9 keeps flag at 0.85 for a solve/fail gap of ~0.90.
    Process signals (0.15 total) create within-group variance without compressing
    between-group gap.
  - **Temporal discounting on efficiency**: Earlier discoveries are exponentially
    more valuable (λ=0.3). A model that identifies the vulnerability at step 3 is
    better than one that finds it at step 15.
  - **Information specificity**: Only known CTF instruments carry signal.
    Set-based lookup, no regex.
  - **Online**: environment verification ("Correct! Flag verified") is authoritative.

Evolution:
  v7: flag=0.50 + 3 process signals → RLOO gap 0.48, policy_loss=0 (no learning)
  v8: flag=1.0 (pure binary) → RLOO gap 1.0, grad_norm=0.14, BUT crashed on
      all-solve batches (zero variance → SkyRL dynamic_sampling empty array)
  v9: flag=0.85 + efficiency(least-action)=0.10 + format=0.05 → RLOO gap ~0.90,
      within-group variance from efficiency's 3 components prevents zero-variance crash

"""

import collections
import json
import random
import re
import threading
from typing import Any

# Placeholder flag strings that should not be used for matching.
# "CHECK" is used by PortSwigger challenges (appears in system prompt).
_FLAG_PLACEHOLDERS = frozenset(
    {
        "content_of_flag_here",
        "FLAG{...}",
        "",
        "CHECK",
        # Dummy/fake flags found in GRPO data (poisoned ground truth)
        "gctf{DUMMY_FLAG}",
        "flag{this_is_fake_flag}",
    }
)

# ---------------------------------------------------------------------------
# Phase classification for skill progression (set-based, NO regex).
# Classify commands by first token (binary name) of shell commands.
# ---------------------------------------------------------------------------
_RECON_BINARIES = frozenset(
    {
        "nmap",
        "masscan",
        "ping",
        "traceroute",
        "whois",
        "dig",
        "nslookup",
        "host",
        "arp-scan",
        "netdiscover",
        "ftp",
        "smbclient",
        "smbmap",
        "rpcclient",
        "snmpwalk",
        "enum4linux",
    }
)
_ENUM_BINARIES = frozenset(
    {
        "curl",
        "wget",
        "gobuster",
        "ffuf",
        "dirb",
        "dirsearch",
        "nikto",
        "wpscan",
        "whatweb",
        "ls",
        "cat",
        "head",
        "tail",
        "find",
        "grep",
        "egrep",
        "fgrep",
        "strings",
        "file",
        "id",
        "whoami",
        "ps",
        "env",
        "uname",
        "hostname",
        "ip",
        "ifconfig",
        "netstat",
        "ss",
        "wc",
        "sort",
        "uniq",
        "less",
        "more",
        "xxd",
        "hexdump",
        "objdump",
        "readelf",
        "cd",
        "echo",
        "which",
        "sed",
        "awk",
        "apt",
        "apt-get",
        "pip",
        "pip3",
        "export",
        "mkdir",
        "cp",
        "mv",
        "rm",
        "printf",
        "crackmapexec",
        "fls",
        "tesseract",
        "unzip",
        "tar",
        "gunzip",
    }
)
_EXPLOIT_BINARIES = frozenset(
    {
        "sqlmap",
        "hydra",
        "john",
        "hashcat",
        "python",
        "python3",
        "python2",
        "ruby",
        "perl",
        "gcc",
        "g++",
        "make",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "msfconsole",
        "msfvenom",
        "chmod",
        "chown",
        "gdb",
        "ltrace",
        "strace",
        "pwntools",
        "sshpass",
        "bash",
        "node",
        "java",
        "docker",
        "php",
        "socat",
    }
)
# Tool names (not shell commands) that map directly to phases.
_TOOL_NAME_PHASES = {
    "web_search": "recon",
    "WebFetch": "recon",
    "read_file": "enum",
    "Read": "enum",
    "read": "enum",
    "grep": "enum",
    "file_search": "enum",
    "python_code": "exploit",
    "python3_code": "exploit",
    "apply_patch": "exploit",
    "Write": "exploit",
    "Edit": "exploit",
    "flag_found": "flag",
}

# ---------------------------------------------------------------------------
# Known instrument registry -- information specificity filter.
#
# Includes all phase-classified tools + shell/exec wrappers + session
# management + orchestration tools found in real BoxPwnr traces.
# ---------------------------------------------------------------------------
_KNOWN_TOOL_NAMES = frozenset(
    set(_TOOL_NAME_PHASES.keys())
    | {
        "shell_command",
        "exec_command",
        "execute_command",
        "shell",
        "bash",
        "Bash",
        "write_stdin",
        "tmux_send_keys",
        "tmux_create_session",
        "tmux_read_output",
        # Session/orchestration (no phase, but still known instruments)
        "close_session",
        "list_sessions",
        "update_plan",
        "Task",
        "TaskOutput",
        "TaskStop",
    }
)

# Shell wrapper tool names (used for action fingerprinting in recovery).
_SHELL_WRAPPERS = frozenset(
    {
        "shell_command",
        "exec_command",
        "execute_command",
        "shell",
        "bash",
        "Bash",
    }
)


class Reward:
    """CTF reward for online GRPO training.

    Compatible with both SkyRL (via TrajGymTextEnv) and TRL-style trainers.

    The ``__call__`` signature matches the standard expectation:
        reward_fn(completions, prompts=None, **kwargs) -> list[float]

    Extra metadata (``ground_truth_flag``, ``optimal_steps``) is forwarded
    via ``**kwargs`` by the trainer when the dataset contains those columns.
    """

    # Trainers may access reward_func.__name__ for logging.
    __name__ = "reward_score"

    # GDPO (Group-Decoupled Policy Optimization) buffer
    _gdpo_stats: dict[str, collections.deque] = {
        "flag": collections.deque(maxlen=256),
        "efficiency": collections.deque(maxlen=256),
        "progression": collections.deque(maxlen=256),
        "exploration": collections.deque(maxlen=256),
        "uniqueness": collections.deque(maxlen=256),
        "format": collections.deque(maxlen=256),
        "recovery": collections.deque(maxlen=256),
        "cognitive": collections.deque(maxlen=256),
        "hallucination": collections.deque(maxlen=256),
    }
    _gdpo_lock = threading.Lock()

    def __init__(
        self,
        flag_weight: float = 0.40,
        efficiency_weight: float = 0.15,
        progression_weight: float = 0.08,
        exploration_weight: float = 0.05,
        uniqueness_weight: float = 0.05,
        format_weight: float = 0.10,
        recovery_weight: float = 0.09,
        cognitive_weight: float = 0.08,
        hallucination_penalty: float = 0.20,
        noise_range: float = 0.01,
        exploration_gamma: float = 0.95,
        seed: int | None = None,
        use_gdpo: bool = False,
    ) -> None:
        """Initialize reward configuration.

        The 8 primary weights must sum to 1.0.
        """
        total = (
            flag_weight
            + efficiency_weight
            + progression_weight
            + exploration_weight
            + uniqueness_weight
            + format_weight
            + recovery_weight
            + cognitive_weight
        )
        if abs(total - 1.0) > 1e-6:
            if total > 0:
                import logging

                _log = logging.getLogger(__name__)
                _log.warning(
                    "Reward weights sum to %.4f, auto-normalizing to 1.0 "
                    "(flag=%.3f eff=%.3f prog=%.3f expl=%.3f uniq=%.3f "
                    "fmt=%.3f rec=%.3f cog=%.3f)",
                    total,
                    flag_weight,
                    efficiency_weight,
                    progression_weight,
                    exploration_weight,
                    uniqueness_weight,
                    format_weight,
                    recovery_weight,
                    cognitive_weight,
                )
                scale = 1.0 / total
                flag_weight *= scale
                efficiency_weight *= scale
                progression_weight *= scale
                exploration_weight *= scale
                uniqueness_weight *= scale
                format_weight *= scale
                recovery_weight *= scale
                cognitive_weight *= scale
            else:
                raise ValueError(f"Reward weights must sum to 1.0, got {total:.4f}")
        self.flag_weight = flag_weight
        self.efficiency_weight = efficiency_weight
        self.progression_weight = progression_weight
        self.exploration_weight = exploration_weight
        self.uniqueness_weight = uniqueness_weight
        self.format_weight = format_weight
        self.recovery_weight = recovery_weight
        self.cognitive_weight = cognitive_weight
        self.hallucination_penalty = hallucination_penalty
        self.noise_range = noise_range
        self.exploration_gamma = exploration_gamma
        self.use_gdpo = use_gdpo
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: list[Any],
        prompts: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Score a batch of completions.

        Args:
            completions: List of completions. Each element is either a raw
                string or a list of message dicts (ChatML).
            prompts: (unused) kept for TRL compatibility.
            **kwargs: May contain ``ground_truth_flag``, ``optimal_steps``,
                and ``metadata`` lists forwarded from dataset columns.

        Returns:
            List of float reward values, one per completion.
        """
        n = len(completions)
        ground_truth_flags: list[str | None] = kwargs.get(
            "ground_truth_flag", [None] * n
        )
        optimal_steps_list: list[int | None] = kwargs.get("optimal_steps", [None] * n)
        metadata_list: list[dict[str, Any] | None] = kwargs.get("metadata", [None] * n)

        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            gt_flag = ground_truth_flags[idx] if idx < len(ground_truth_flags) else None
            opt_steps = (
                optimal_steps_list[idx] if idx < len(optimal_steps_list) else None
            )
            meta = metadata_list[idx] if idx < len(metadata_list) else None

            score, _ = self._score_one(completion, gt_flag, opt_steps, meta)
            rewards.append(score)

        return rewards

    def compute_with_breakdown(
        self,
        completions: list[Any],
        prompts: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[float, dict[str, float]]]:
        """Score a batch of completions, returning per-signal breakdowns.

        Same interface as ``__call__`` but returns a list of
        ``(total_reward, breakdown_dict)`` tuples. The breakdown dict
        contains the raw (pre-noise) weighted contribution of each signal.

        This method does NOT modify the existing ``__call__`` contract.
        """
        n = len(completions)
        ground_truth_flags: list[str | None] = kwargs.get(
            "ground_truth_flag", [None] * n
        )
        optimal_steps_list: list[int | None] = kwargs.get("optimal_steps", [None] * n)
        metadata_list: list[dict[str, Any] | None] = kwargs.get("metadata", [None] * n)

        results: list[tuple[float, dict[str, float]]] = []
        for idx, completion in enumerate(completions):
            gt_flag = ground_truth_flags[idx] if idx < len(ground_truth_flags) else None
            opt_steps = (
                optimal_steps_list[idx] if idx < len(optimal_steps_list) else None
            )
            meta = metadata_list[idx] if idx < len(metadata_list) else None

            score, breakdown = self._score_one(completion, gt_flag, opt_steps, meta)
            results.append((score, breakdown))

        return results

    def _score_one(
        self,
        completion: Any,
        gt_flag: str | None,
        opt_steps: int | None,
        meta: dict[str, Any] | None,
    ) -> tuple[float, dict[str, float]]:
        """Score a single completion. Returns (total_score, breakdown_dict).

        The breakdown dict contains raw signal values (before weighting)
        keyed by signal name, plus the weighted contributions.
        """
        text, tool_calls = self._extract(completion)

        task_category = (
            meta.get("task_category", "web") if isinstance(meta, dict) else "web"
        )

        flag_sc = self._flag_score(text, gt_flag, tool_calls=tool_calls)

        # Compute process signals (all ungated for dual-mode support).
        eff_sc = self._efficiency_score(
            len(tool_calls),
            opt_steps,
            flag_found=(flag_sc >= 1.0),
            tool_calls=tool_calls,
        )
        prog_sc = self._progression_score(tool_calls)
        expl_sc = self._exploration_score(tool_calls)
        uniq_sc = self._uniqueness_score(tool_calls)
        fmt_sc = self._format_score(tool_calls)
        hall_sc = self._hallucination_score(tool_calls, flag_sc, gt_flag)
        recov_sc = self._recovery_score(tool_calls)
        cog_sc = self._cognitive_score(text, tool_calls)

        # Minimum work gate: prevent reward hacking via very short episodes.
        # If fewer than 3 tool calls, cap easily-gamed process signals.
        if len(tool_calls) < 3:
            _SHORT_CAP = 0.3
            fmt_sc = min(fmt_sc, _SHORT_CAP)
            uniq_sc = min(uniq_sc, _SHORT_CAP)
            recov_sc = min(recov_sc, _SHORT_CAP)

        # Entropy-scaled format: modulate format by information density.
        # Low uniqueness = low entropy = less format credit.
        info_density = max(uniq_sc, 0.5) if tool_calls else 0.0
        fmt_effective = fmt_sc * info_density

        # Hallucination as energy loss: wrong flag submission decays
        # process signals to 30% (v8 fix: full zeroing made flag_found
        # EV-negative for small models, discouraging all flag attempts).
        if hall_sc < 0:
            _HALL_DECAY = 0.3
            fmt_effective *= _HALL_DECAY
            expl_sc *= _HALL_DECAY
            prog_sc *= _HALL_DECAY
            recov_sc *= _HALL_DECAY
            cog_sc *= _HALL_DECAY
            eff_sc *= _HALL_DECAY
            uniq_sc *= _HALL_DECAY

        raw_signals = {
            "flag": flag_sc,
            "efficiency": eff_sc,
            "progression": prog_sc,
            "exploration": expl_sc,
            "uniqueness": uniq_sc,
            "format": fmt_effective,
            "recovery": recov_sc,
            "cognitive": cog_sc,
            "hallucination": hall_sc,
        }

        if self.use_gdpo:
            with self._gdpo_lock:
                normalized_signals = {}
                for k, v in raw_signals.items():
                    self._gdpo_stats[k].append(v)
                    history = list(self._gdpo_stats[k])
                    if len(history) > 1:
                        mean_val = sum(history) / len(history)
                        variance = sum((x - mean_val) ** 2 for x in history) / len(
                            history
                        )
                        std_val = (variance**0.5) + 1e-4
                        normalized_signals[k] = (v - mean_val) / std_val
                    else:
                        normalized_signals[k] = v - 0.5  # Rough center if no history
        else:
            normalized_signals = raw_signals

        # Dynamic weight redistribution:
        # Non-web challenges don't follow RECON->ENUM->EXPLOIT.
        current_prog_weight = self.progression_weight
        current_cog_weight = self.cognitive_weight
        current_eff_weight = self.efficiency_weight

        if str(task_category).lower() in (
            "crypto",
            "rev",
            "reverse",
            "forensics",
            "pwn",
        ):
            # Disable strict progression and redistribute its weight
            redistribute = current_prog_weight
            current_prog_weight = 0.0
            current_cog_weight += redistribute * 0.5
            current_eff_weight += redistribute * 0.5

        score = (
            self.flag_weight * normalized_signals["flag"]
            + current_eff_weight * normalized_signals["efficiency"]
            + current_prog_weight * normalized_signals["progression"]
            + self.exploration_weight * normalized_signals["exploration"]
            + self.uniqueness_weight * normalized_signals["uniqueness"]
            + self.format_weight * normalized_signals["format"]
            + self.recovery_weight * normalized_signals["recovery"]
            + current_cog_weight * normalized_signals["cognitive"]
            + normalized_signals["hallucination"]
        )

        # Guarantee variance for GRPO
        noise = self._rng.uniform(-self.noise_range, self.noise_range)
        score += noise

        # Build breakdown: raw signal values + their weighted contributions
        breakdown = {
            "flag": raw_signals["flag"],
            "efficiency": raw_signals["efficiency"],
            "progression": raw_signals["progression"],
            "exploration": raw_signals["exploration"],
            "uniqueness": raw_signals["uniqueness"],
            "format": raw_signals["format"],
            "recovery": raw_signals["recovery"],
            "cognitive": raw_signals["cognitive"],
            "hallucination": raw_signals["hallucination"],
            # Weighted contributions (what each signal added to the total)
            "flag_weighted": self.flag_weight * normalized_signals["flag"],
            "efficiency_weighted": current_eff_weight
            * normalized_signals["efficiency"],
            "progression_weighted": current_prog_weight
            * normalized_signals["progression"],
            "exploration_weighted": self.exploration_weight
            * normalized_signals["exploration"],
            "uniqueness_weighted": self.uniqueness_weight
            * normalized_signals["uniqueness"],
            "format_weighted": self.format_weight * normalized_signals["format"],
            "recovery_weighted": self.recovery_weight * normalized_signals["recovery"],
            "cognitive_weighted": current_cog_weight * normalized_signals["cognitive"],
            "hallucination_weighted": normalized_signals["hallucination"],
            "noise": noise,
        }

        return score, breakdown

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    # Verification strings emitted by ToolExecutor flag_found handler.
    _FLAG_VERIFIED = "Correct! Flag verified"
    _FLAG_REJECTED = "Incorrect submission"

    def _flag_score(
        self,
        text: str,
        ground_truth: str | None,
        metadata_success: bool | None = None,
        tool_calls: list[dict[str, str]] | None = None,
    ) -> float:
        """Score flag capture (online mode).

        Priority order:
          1. Environment verification signal in tool response text
             ("Correct! Flag verified" -> 1.0, "Incorrect submission" -> 0.0)
          2. Exact match against ground_truth_flag -> 1.0
             (only if flag_found was called — prevents false rewards from
             flags appearing incidentally in tool output)
          3. Pattern match: FLAG{<alphanumeric>} -> 0.1
          4. No flag -> 0.0

        Note: metadata_success is accepted for API compatibility but ignored.
        Flag score requires explicit submission via flag_found/submit_flag
        (environment verification) or exact text match against ground truth.
        The old metadata_success shortcut was removed because episode_done
        (which feeds metadata.success) fires when ground-truth appears
        anywhere in tool output, even if the model never called flag_found.
        """
        # 1. Environment verification: ToolExecutor responds with
        #    "Correct! Flag verified: <flag>" on success, or
        #    "Incorrect submission: <flag>" on failure. This is the
        #    strongest signal (environment-verified).
        if self._FLAG_VERIFIED in text:
            return 1.0
        if self._FLAG_REJECTED in text:
            # Environment explicitly rejected — wrong flag was submitted.
            return 0.0

        # 2. Exact match against ground truth — only if flag_found was called.
        #    Without this guard, the ground truth appearing anywhere in tool
        #    output (e.g. reading source code that contains the flag, or
        #    browsing benchmark solution files) gives a false 1.0 reward.
        #    Check both tool_calls list and text for flag_found presence.
        has_flag_submission = "flag_found" in text or any(
            tc.get("name") in ("flag_found", "submit_flag")
            for tc in (tool_calls or [])
        )
        if (
            ground_truth
            and self._is_real_flag(ground_truth)
            and ground_truth in text
            and has_flag_submission
        ):
            return 1.0

        # 3. Pattern match: FLAG{<alphanumeric>} (excludes FLAG{...})
        if re.search(r"FLAG\{[a-zA-Z0-9_-]+\}", text):
            return 0.1

        return 0.0

    @staticmethod
    def _is_real_flag(flag: str) -> bool:
        """Return True if the flag is a real value, not a placeholder or template."""
        if not flag:
            return False
        if flag in _FLAG_PLACEHOLDERS:
            return False
        flag_lower = flag.lower()
        # Detect template text or dummy flags
        for poison in (
            "content_of_flag_here",
            "dummy_flag",
            "fake_flag",
            "this_is_fake",
            "placeholder",
        ):
            if poison in flag_lower:
                return False
        return True

    def _uniqueness_score(self, tool_calls: list[dict[str, str]]) -> float:
        """Score command diversity (0.0 - 1.0). No regex.

        Information entropy: ratio of unique commands to total commands.
        Returns 0.0 for no tool calls, 0.5 for non-extractable commands.
        """
        if not tool_calls:
            return 0.0

        commands: list[str] = []
        for tc in tool_calls:
            cmd = self._extract_command(tc)
            if cmd:
                commands.append(cmd)

        if not commands:
            return 0.5  # Neutral for non-command tool calls

        return len(set(commands)) / len(commands)

    @staticmethod
    def _extract_command(tc: dict[str, str]) -> str:
        """Extract the command string from a tool call's arguments.

        Handles common BoxPwnr argument schemas:
          - {"command": "..."} (shell_command, exec_command)
          - {"code": "..."} (python_code)
          - {"content": "..."} (flag_found)
          - {"query": "..."} (web_search, grep)
          - {"path": "..."} (read_file)
          - Plain string arguments
        """
        args_str = tc.get("arguments", "")
        if not args_str:
            return ""

        # Try JSON first
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            if isinstance(args, dict):
                # Try common argument keys in priority order
                for key in (
                    "command",
                    "code",
                    "content",
                    "query",
                    "path",
                    "pattern",
                    "search_query",
                    "stdin",
                ):
                    val = args.get(key)
                    if val and isinstance(val, str):
                        return val.strip()
                # Fallback: use first non-empty string value
                for val in args.values():
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            elif isinstance(args, str):
                return args.strip()
        except (json.JSONDecodeError, TypeError):
            # Not JSON -- use raw string
            if isinstance(args_str, str):
                return args_str.strip()

        return ""

    def _efficiency_score(
        self,
        actual_steps: int,
        optimal_steps: int | None,
        flag_found: bool = False,
        tool_calls: list[dict[str, str]] | None = None,
    ) -> float:
        """Principle of least action: step_ratio × action_novelty × temporal_decay.

        Three physics-inspired components:
          1. Step ratio (classical efficiency): min(optimal/actual, 1.0)
             — the shortest path through solution space.
          2. Action novelty (information redundancy): unique_fingerprints / total
             — repeated commands are wasted energy; the least-action path
             uses each technique exactly once.
          3. Temporal decay (time cost): exp(-λ × excess_steps / optimal)
             — steps beyond the optimal horizon are exponentially penalized,
             favoring trajectories that discover solutions earlier.

        Combined: score = step_ratio × novelty × time_decay

        Returns 0.0 for zero steps, 0.3 (weak prior) without metadata.
        Non-flag completions capped at 0.3. Fewer than 3 steps = 0.0.
        """
        import math

        if actual_steps == 0:
            return 0.0
        if actual_steps < 3:
            return 0.0
        opt = optimal_steps or 10

        # Component 1: Step ratio (classical least-action efficiency)
        step_ratio = min(opt / actual_steps, 1.0)

        # Component 2: Action novelty — unique action fingerprints / total
        # Repeated commands = wasted energy in information space.
        # When tool_calls not provided, default to 1.0 (no redundancy penalty).
        if tool_calls and len(tool_calls) > 0:
            fingerprints = [self._action_fingerprint(tc) for tc in tool_calls]
            novelty = len(set(fingerprints)) / len(fingerprints)
        else:
            novelty = 1.0

        # Component 3: Temporal decay — exponentially penalize excess steps
        # beyond the optimal horizon. λ=0.3 means 2x optimal → 0.74 decay,
        # 3x optimal → 0.55, 4x optimal → 0.37.
        excess = max(0, actual_steps - opt) / max(opt, 1)
        time_decay = math.exp(-0.3 * excess)

        score = step_ratio * novelty * time_decay

        if not flag_found:
            return min(score, 0.3)
        return score

    def _format_score(self, tool_calls: list[dict[str, str]]) -> float:
        """Signal fidelity: valid instrument readings from known tools only.

        Scoring per known tool call:
          - Valid JSON args: 1.0
          - Invalid JSON args: 0.5 (partial signal, learning)
          - Unknown tool name: 0.0 (no signal)
        """
        if not tool_calls:
            return 0.0

        valid = 0
        known_count = 0
        for tc in tool_calls:
            name = tc.get("name", "")
            if not self._is_known_tool(name):
                continue

            known_count += 1
            if tc["arguments"]:
                args = tc["arguments"]
                if isinstance(args, dict):
                    valid += 1
                else:
                    try:
                        json.loads(args)
                        valid += 1
                    except (json.JSONDecodeError, TypeError):
                        valid += 0.5

        if known_count == 0:
            return 0.0
        return min(valid / known_count, 1.0)

    @staticmethod
    def _is_known_tool(name: str) -> bool:
        """Check if a tool name is a recognized CTF instrument."""
        return name in _KNOWN_TOOL_NAMES

    def _progression_score(self, tool_calls: list[dict[str, str]]) -> float:
        """Phase space trajectory: RECON->ENUM->EXPLOIT ordering.

        Scoring: 0.6 for phase presence + 0.4 for correct ordering.
        """
        if not tool_calls:
            return 0.0

        # Build deduplicated phase sequence
        phases: list[str] = []
        for tc in tool_calls:
            phase = self._classify_phase(tc)
            if phase and (not phases or phases[-1] != phase):
                phases.append(phase)

        if not phases:
            return 0.0

        has_recon = "recon" in phases
        has_enum = "enum" in phases
        has_exploit = "exploit" in phases

        # Phase presence (0.0 - 0.6)
        presence = 0.2 * has_recon + 0.2 * has_enum + 0.2 * has_exploit

        # Order adherence (0.0 - 0.4)
        order = 0.0
        if has_recon and has_enum and phases.index("recon") < phases.index("enum"):
            order += 0.2
        if has_enum and has_exploit and phases.index("enum") < phases.index("exploit"):
            order += 0.2

        return min(presence + order, 1.0)

    @staticmethod
    def _classify_phase(tc: dict[str, str]) -> str | None:
        """Classify a tool call into a CTF phase. Set-based, no regex."""
        name = tc.get("name", "")

        # Direct tool name classification
        if name in _TOOL_NAME_PHASES:
            return _TOOL_NAME_PHASES[name]

        # For shell wrappers, classify by first token (binary name)
        if name in _SHELL_WRAPPERS:
            cmd = Reward._extract_command(tc)
            if not cmd:
                return None
            first_token = cmd.split()[0].rsplit("/", 1)[-1].lower()
            if first_token in _RECON_BINARIES:
                return "recon"
            if first_token in _ENUM_BINARIES:
                return "enum"
            if first_token in _EXPLOIT_BINARIES:
                return "exploit"

        return None

    def _exploration_score(self, tool_calls: list[dict[str, str]]) -> float:
        """Exponentially-decayed novelty of known instruments.

        Uses gamma^t decay so earlier novel tool use carries exponentially
        more signal than late novelty. gamma=0.95 means step 0 gets weight
        1.0, step 10 gets 0.60, step 50 gets 0.08.

        Only known CTF tool names contribute. No regex.
        """
        if not tool_calls:
            return 0.0

        gamma = self.exploration_gamma
        seen: set = set()
        score = 0.0
        max_possible = 0.0

        for t, tc in enumerate(tool_calls):
            name = tc.get("name", "")
            if not name:
                continue

            decay = gamma**t

            # Only known instruments contribute to exploration signal.
            if self._is_known_tool(name):
                max_possible += decay
                if name not in seen:
                    seen.add(name)
                    score += decay

        return score / max_possible if max_possible > 0 else 0.0

    def _recovery_score(self, tool_calls: list[dict[str, str]]) -> float:
        """Resilience: reward pivots after stuck runs.

        A "stuck run" is 2+ consecutive calls with the same action
        fingerprint (tool name + binary name for shell wrappers).
        A "pivot" is transitioning out of a stuck run.

        Returns:
          - 0.5 (neutral) when < 3 tool calls or no stuck runs
          - pivots / stuck_runs for traces with stuck periods
          - 0.0 when stuck but never pivoting (worst case)

        No regex. Uses first-token extraction for shell command identity.
        """
        if len(tool_calls) < 3:
            return 0.5  # Too short to measure

        # Build action fingerprint sequence
        actions: list[str] = []
        for tc in tool_calls:
            actions.append(self._action_fingerprint(tc))

        if not actions:
            return 0.5

        # Count stuck runs and pivots
        stuck_runs = 0
        pivots = 0
        run_length = 1

        for i in range(1, len(actions)):
            if actions[i] == actions[i - 1]:
                run_length += 1
            else:
                if run_length >= 2:
                    stuck_runs += 1
                    pivots += 1  # Transitioned out = pivot
                run_length = 1

        # Check if trace ended in a stuck run (no pivot out)
        if run_length >= 2:
            stuck_runs += 1

        if stuck_runs == 0:
            return 0.5  # No stuck runs = neutral (no recovery needed)

        return pivots / stuck_runs

    @staticmethod
    def _action_fingerprint(tc: dict[str, str]) -> str:
        """Create a fingerprint for a tool call action.

        For shell wrappers, includes the binary name so that
        shell_command("nmap") and shell_command("curl") are distinct.
        For other tools, uses the tool name directly.
        No regex -- uses split + rsplit for binary extraction.
        """
        name = tc.get("name", "")
        if name in _SHELL_WRAPPERS:
            cmd = Reward._extract_command(tc)
            if cmd:
                binary = cmd.split()[0].rsplit("/", 1)[-1].lower()
                return f"{name}:{binary}"
        return name

    def _cognitive_score(self, text: str, tool_calls: list[dict[str, str]]) -> float:
        """Words-per-action scoring. Optimal reasoning density at ~42 WPA.

        Research finding (H12): success traces average 51 WPA, failures
        average 16 WPA. Optimal is ~42. Underthinking (< 10 WPA) strongly
        correlates with failure. Severe overthinking (> 80 WPA) wastes
        tokens without improving outcomes.

        Applies a repetition penalty: when >50% of tool calls share the
        same command, the score is capped at 0.3 (repetitive loops should
        not be rewarded for appearing to "reason" at the right density).

        Returns 0.5 (neutral) when text is too short to measure (< 5 words)
        or when there are no tool calls. No regex -- whitespace split only.
        """
        if not tool_calls:
            return 0.0

        words = text.split() if text else []
        word_count = len(words)

        if word_count < 5:
            return 0.5  # Insufficient text to measure (online mode, tests)

        wpa = word_count / len(tool_calls)

        # Piecewise linear scoring around optimal 42 WPA
        _OPTIMAL = 42.0
        if wpa < 10:
            score = 0.1  # Severe underthinking
        elif wpa <= _OPTIMAL:
            # Linear ramp from 0.1 at WPA=10 to 1.0 at WPA=42
            score = 0.1 + 0.9 * (wpa - 10) / (_OPTIMAL - 10)
        elif wpa <= 80:
            # Gentle decline from 1.0 at WPA=42 to 0.5 at WPA=80
            score = 1.0 - 0.5 * (wpa - _OPTIMAL) / (80 - _OPTIMAL)
        else:
            # Plateau at 0.3 for extreme overthinking
            score = 0.3

        # Repetition penalty: looping the same command wastes turns.
        if len(tool_calls) >= 4:
            commands = []
            for tc in tool_calls:
                cmd = self._extract_command(tc)
                commands.append(cmd)
            if commands:
                from collections import Counter

                most_common_count = Counter(commands).most_common(1)[0][1]
                if most_common_count > len(tool_calls) * 0.5:
                    score = min(score, 0.3)

        return score

    def _hallucination_score(
        self,
        tool_calls: list[dict[str, str]],
        flag_sc: float,
        ground_truth: str | None = None,
    ) -> float:
        """Energy loss for false flag submissions. Structural, no regex.

        When ``flag_found`` tool was called but flag_score < 1.0, the
        trajectory has moved backward. Returns a negative penalty.
        The caller also zeroes all process signals when this fires.
        """
        import difflib

        if flag_sc >= 1.0:
            return 0.0  # Correct flag -- no penalty

        for tc in tool_calls:
            if tc.get("name") in ("flag_found", "submit_flag"):
                penalty = -self.hallucination_penalty

                # Premature submission multiplier: penalize harder when the
                # model hallucinated a flag very early (few tool calls).
                # 1 tool call → penalty * 1.8, 2 → 1.6, 5+ → 1.0 (no extra).
                premature_mult = 1.0 + max(0.0, 1.0 - len(tool_calls) / 5.0)
                penalty *= premature_mult

                if ground_truth:
                    submitted = self._extract_command(tc).strip()
                    if submitted:
                        similarity = difflib.SequenceMatcher(
                            None, submitted, ground_truth
                        ).ratio()
                        if similarity >= 0.8:
                            # 80%+ similar (e.g., missing FLAG{} wrapper), decay the penalty by 90%
                            penalty *= 0.1

                return penalty

        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(completion: Any) -> tuple[str, list[dict[str, str]]]:
        """Extract flat text and structured tool calls from a completion.

        Returns:
            (text, tool_calls) where tool_calls is a list of
            {"name": str, "arguments": str} dicts.
        """
        if isinstance(completion, str):
            return completion, []
        if isinstance(completion, dict):
            # Single message dict (not wrapped in a list)
            content = completion.get("content") or ""
            tool_calls = []
            for tc in completion.get("tool_calls") or []:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append({"name": name, "arguments": args or ""})
            return str(content), tool_calls
        if isinstance(completion, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, str]] = []
            for msg in completion:
                if not isinstance(msg, dict):
                    text_parts.append(str(msg))
                    continue
                content = msg.get("content") or ""
                text_parts.append(str(content))
                for tc in msg.get("tool_calls") or []:
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append({"name": name, "arguments": args or ""})
            return "\n".join(text_parts), tool_calls
        return str(completion), []
