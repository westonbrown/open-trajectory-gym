"""Default StepAgent — extracts tool parsing + execution from TrajGymTextEnv.

This is the line-for-line equivalent of TrajGymTextEnv.step() logic, packaged
as a pluggable StepAgent. It uses parse_tool_calls() for model-agnostic parsing
and SubprocessExecutor for tool execution.

When ``resilient_mode=True`` (the default), a missing/invalid tool call does
NOT immediately terminate the episode. The agent returns explicit feedback and
keeps the rollout alive until ``max_steps`` (or flag submission). This replaces
the former ``ResilientStepAgent`` subclass.

Users who want custom tool handling can implement StepAgent and swap this out
via ``agent_class`` in the GRPO config or ``--agent`` on the CLI.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any

from trajgym.agent.protocol import StepResult
from trajgym.agent.rollout_status import RolloutStatus, normalize_rollout_status
from trajgym.agent.wire_protocol import (
    RuntimeProtocolError,
    build_runtime_request,
    normalize_runtime_response,
    parse_runtime_stdout,
)

logger = logging.getLogger(__name__)

# Severity ordering for rollout statuses: higher index = more severe.
# Used to prevent runtime_info from downgrading a status (e.g., MAX_TURN_ABORT -> OK).
_STATUS_SEVERITY: dict[str, int] = {
    RolloutStatus.OK.value: 0,
    RolloutStatus.NO_TOOL_CALL.value: 1,
    RolloutStatus.PARSER_ERROR.value: 2,
    RolloutStatus.EMPTY_ACTION_LOOP.value: 3,
    RolloutStatus.TOOL_ERROR.value: 4,
    RolloutStatus.TOOL_TIMEOUT.value: 5,
    RolloutStatus.INFRA_UNREACHABLE.value: 6,
    RolloutStatus.TARGET_MISMATCH.value: 7,
    RolloutStatus.RUNTIME_TIMEOUT.value: 8,
    RolloutStatus.RUNTIME_ERROR.value: 9,
    RolloutStatus.MAX_TURN_ABORT.value: 10,
    RolloutStatus.NON_TERMINAL_CLOSE.value: 11,
}


def _upgrade_status(current: str, candidate: str) -> str:
    """Return the more severe of two rollout status strings.

    Only allows upgrading severity (e.g., OK -> ERROR), never downgrading
    (e.g., MAX_TURN_ABORT -> OK).
    """
    cur_sev = _STATUS_SEVERITY.get(current, 0)
    cand_sev = _STATUS_SEVERITY.get(candidate, 0)
    return candidate if cand_sev > cur_sev else current


class DefaultStepAgent:
    """Default StepAgent using parse_tool_calls + SubprocessExecutor.

    Implements the StepAgent protocol via structural subtyping (no inheritance
    required). Logic is extracted from TrajGymTextEnv.step() lines 329-433.

    Attributes exposed for reward computation by the env:
        tool_calls_history: List of {name, arguments} dicts.
        tool_outputs: List of output strings.
        all_text: Concatenated LLM + tool output text.
        episode_done: Whether the flag was submitted successfully.
        turns: Number of steps taken.
    """

    # Rollout statuses that resilient mode treats as recoverable (non-fatal).
    _NON_FATAL_NO_ACTION = {
        RolloutStatus.NO_TOOL_CALL.value,
        RolloutStatus.PARSER_ERROR.value,
        RolloutStatus.EMPTY_ACTION_LOOP.value,
    }

    def __init__(self, **kwargs: Any):
        self._executor = None
        self._executor_type: str = kwargs.get("executor_type", "subprocess")
        self._executor_kwargs: dict[str, Any] = kwargs
        self._tool_call_format: str = (
            str(kwargs.get("tool_call_format", "hermes")).strip() or "hermes"
        )
        # When True, missing/invalid tool calls do NOT terminate the episode.
        # The agent returns feedback and keeps the rollout alive until
        # max_steps or flag submission.
        self.resilient_mode: bool = bool(kwargs.get("resilient_mode", True))
        self.max_consecutive_no_tool_calls: int = int(
            kwargs.get("max_consecutive_no_tool_calls", 3)
        )
        # Cap per-tool observation payload to keep long-horizon prompts within
        # vLLM context limits during multi-turn runs.
        self.max_tool_response_chars: int = int(
            kwargs.get("max_tool_response_chars", 2000)
        )
        # Optional BYO runtime hook. If set, this command receives a JSON
        # payload on stdin and should print a JSON response on stdout.
        self.runtime_cmd: str | None = kwargs.get("runtime_cmd")
        self.runtime_timeout_seconds: int = int(
            kwargs.get("runtime_timeout_seconds", 30)
        )
        self.runtime_workdir: str | None = kwargs.get("runtime_workdir")
        self.runtime_env: dict[str, str] = {
            str(k): str(v) for k, v in dict(kwargs.get("runtime_env") or {}).items()
        }
        # If True, the external runtime can fully own observations/done and
        # tool execution is skipped in this agent.
        self.runtime_passthrough: bool = bool(kwargs.get("runtime_passthrough", False))
        # If external runtime fails, fallback to native parse_tool_calls path.
        self.runtime_fallback_to_parser: bool = bool(
            kwargs.get("runtime_fallback_to_parser", True)
        )
        # Optional tokenizer used to decode token-id outputs when generator
        # paths return numeric sequences instead of plain text.
        self.tokenizer_name_or_path: str | None = (
            str(kwargs.get("tokenizer_name_or_path")).strip()
            if kwargs.get("tokenizer_name_or_path")
            else str(os.getenv("TRAJGYM_TOKENIZER_PATH", "")).strip() or None
        )
        self._tokenizer = None
        self._step_debug: bool = bool(kwargs.get("step_debug", False)) or (
            str(os.getenv("TRAJGYM_STEP_DEBUG", "0")).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._runtime_state: dict[str, Any] = {}
        self._target: str = ""
        self._ground_truth_flag: str = ""
        self._prompt_messages: list[dict[str, Any]] = []
        self._challenge_id: str = ""
        self._category: str = ""
        self._difficulty: str = ""
        self._infra_type: str = ""
        self._objective: str = ""
        self._challenge_workdir: str = (
            str(kwargs.get("challenge_workdir", "")).strip()
            or str(os.getenv("CHALLENGE_WORKDIR", "/root/challenge")).strip()
            or "/root/challenge"
        )

        # Episode state (exposed as properties for reward computation)
        self.tool_calls_history: list[dict[str, str]] = []
        self.tool_outputs: list[str] = []
        self.all_text: str = ""
        self.episode_done: bool = False
        self.turns: int = 0
        self.max_steps: int = 30
        self._consecutive_no_tool_calls: int = 0

    @staticmethod
    def _looks_like_tool_call(text: str) -> bool:
        snippet = (text or "").strip()
        if not snippet:
            return False
        lowered = snippet.lower()
        signals = (
            "<tool_call>",
            "<function=",
            "<command",
            "<flag>",
            '"name"',
            "flag_found(",
            "submit_flag(",
            "shell_command(",
            "exec_command(",
            "python_code(",
        )
        return any(sig in lowered for sig in signals)

    @staticmethod
    def _status_from_tool_output(output: str) -> str | None:
        lowered = output.lower()
        if "timed out" in lowered or "timeout" in lowered:
            return RolloutStatus.TOOL_TIMEOUT.value
        if (
            "connection refused" in lowered
            or "no route to host" in lowered
            or "name or service not known" in lowered
            or "temporary failure in name resolution" in lowered
            or "network is unreachable" in lowered
        ):
            return RolloutStatus.INFRA_UNREACHABLE.value
        if "target mismatch" in lowered:
            return RolloutStatus.TARGET_MISMATCH.value
        return None

    def _truncate_tool_output(self, output: str) -> str:
        if self.max_tool_response_chars <= 0:
            return output
        if len(output) <= self.max_tool_response_chars:
            return output
        return output[: self.max_tool_response_chars] + "\n...[tool output truncated]"

    def _rewrite_workspace_refs(self, payload: Any) -> Any:
        """Rewrite legacy /root/challenge paths to active challenge workdir."""
        workdir = str(self._challenge_workdir or "").strip()
        if not workdir:
            return payload
        normalized = workdir.rstrip("/")
        if not normalized or normalized == "/root/challenge":
            return payload
        file_target = f"file://{normalized}/"

        if isinstance(payload, str):
            out = payload
            out = out.replace("file:///root/challenge/", file_target)
            out = out.replace("/root/challenge/", f"{normalized}/")
            out = out.replace("/root/challenge", normalized)
            return out
        if isinstance(payload, list):
            return [self._rewrite_workspace_refs(item) for item in payload]
        if isinstance(payload, dict):
            return {
                key: self._rewrite_workspace_refs(value)
                for key, value in payload.items()
            }
        return payload

    def _decode_token_ids(self, token_ids: list[int]) -> str | None:
        """Decode token IDs to text using a lazily-loaded tokenizer."""
        if not token_ids or not self.tokenizer_name_or_path:
            return None
        try:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    trust_remote_code=True,
                )
            decoded = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            if isinstance(decoded, str) and decoded.strip():
                return decoded
        except Exception as exc:
            if self._step_debug:
                logger.info("Token-id decode failed: %s", exc)
        return None

    def _normalize_action_text(self, action: Any) -> str:
        """Normalize model output into parseable text."""
        if (
            isinstance(action, (list, tuple))
            and action
            and all(isinstance(x, int) for x in action)
        ):
            decoded = self._decode_token_ids(list(action))
            return decoded if decoded is not None else str(action)

        text = str(action or "")
        stripped = text.strip()
        if re.fullmatch(r"\[\s*\d+(?:\s*,\s*\d+){8,}\s*\]", stripped):
            try:
                token_ids = [int(x) for x in re.findall(r"\d+", stripped)]
            except Exception:
                token_ids = []
            decoded = self._decode_token_ids(token_ids)
            if decoded:
                return decoded
        return text

    def _extract_fallback_shell_command(self, action: str) -> str | None:
        """Recover one explicit command when model skipped tool-call wrappers.

        Intentionally conservative: only extract when the model emitted a
        direct invocation shape (function-call syntax or COMMAND XML tag).
        We avoid broad prose regex recovery to prevent accidental execution
        of narrative text.
        """
        text = str(action or "")
        if not text.strip():
            return None

        def _quote_balanced(cmd: str) -> bool:
            single = re.sub(r"\\'", "", cmd).count("'")
            double = re.sub(r'\\"', "", cmd).count('"')
            return (single % 2 == 0) and (double % 2 == 0)

        def _clean(cmd: str) -> str | None:
            value = " ".join(str(cmd or "").split()).strip().rstrip(".,;")
            if not value:
                return None
            if "<target_url>" in value or "<url>" in value:
                return None
            if not _quote_balanced(value):
                return None
            return value

        # 1) Strict function-style recovery.
        function_patterns = (
            r"shell_command\s*\(\s*command\s*=\s*([\"'])(.+?)\1\s*\)",
            r"exec_command\s*\(\s*cmd\s*=\s*([\"'])(.+?)\1\s*\)",
        )
        for pattern in function_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = _clean(match.group(2))
                if cleaned:
                    return cleaned

        # 2) COMMAND XML fallback (single explicit command).
        xml_match = re.search(
            r"<COMMAND(?:\s+maxtime=\d+)?>(.*?)</COMMAND>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if xml_match:
            return _clean(xml_match.group(1))

        return None

    def _format_hint_text(self, *, parser_error: bool, target: str) -> str:
        """Return a format-specific tool-call hint for parser misses."""
        fmt = str(self._tool_call_format or "").strip().lower()
        if fmt == "command_xml":
            if parser_error:
                return (
                    "Tool call parse failed. Reply with exactly one action tag and no prose:\n"
                    f"<COMMAND maxtime=30>curl -s {target}</COMMAND>"
                )
            return (
                "No tool call detected. Reply with exactly one action tag:\n"
                "<COMMAND maxtime=30>command</COMMAND> or <FLAG>flag_value</FLAG>."
            )
        if fmt == "qwen3_coder":
            if parser_error:
                return (
                    "Tool call parse failed. Reply with exactly one valid tool call and no prose:\n"
                    f"<tool_call><function=shell_command><parameter=command>curl -s {target}</parameter></function></tool_call>"
                )
            return (
                "No tool call detected. Reply with one valid tool call in Qwen3 format:\n"
                "<tool_call><function=tool_name><parameter=arg>value</parameter></function></tool_call>"
            )
        if fmt == "glm4":
            if parser_error:
                return (
                    "Tool call parse failed. Reply with exactly one valid tool call and no prose:\n"
                    f"<tool_call>shell_command<arg_key>command</arg_key><arg_value>curl -s {target}</arg_value></tool_call>"
                )
            return (
                "No tool call detected. Reply with one valid tool call in GLM4 format:\n"
                "<tool_call>tool_name<arg_key>param</arg_key><arg_value>value</arg_value></tool_call>"
            )
        if parser_error:
            return (
                "Tool call parse failed. Reply with exactly one valid tool call and no prose:\n"
                f'<tool_call>{{"name":"shell_command","arguments":{{"command":"curl -s {target}"}}}}</tool_call>'
            )
        return "No tool call detected. Reply with one valid tool call in <tool_call>{...}</tool_call> format."

    def _run_external_runtime(self, action: str) -> tuple[dict[str, Any] | None, float]:
        """Run optional external runtime hook and return (payload, seconds)."""
        if not self.runtime_cmd:
            return None, 0.0

        runtime_state = dict(self._runtime_state or {})
        if self._challenge_workdir:
            runtime_state.setdefault("agent_workdir", self._challenge_workdir)

        payload = build_runtime_request(
            action=action,
            turn=self.turns,
            max_steps=self.max_steps,
            target=self._target,
            ground_truth_flag=self._ground_truth_flag,
            tool_calls_history=self.tool_calls_history,
            tool_outputs=self.tool_outputs,
            all_text=self.all_text,
            runtime_state=runtime_state,
            prompt_messages=self._prompt_messages,
            challenge_id=self._challenge_id,
            category=self._category,
            difficulty=self._difficulty,
            infra_type=self._infra_type,
            objective=self._objective,
        )

        env = os.environ.copy()
        env.update(self.runtime_env)

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                ["bash", "-c", self.runtime_cmd],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.runtime_timeout_seconds,
                cwd=self.runtime_workdir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            seconds = time.perf_counter() - started
            msg = (
                f"External runtime timed out after {self.runtime_timeout_seconds}s: "
                f"{self.runtime_cmd}"
            )
            logger.warning(msg)
            if self.runtime_fallback_to_parser:
                return None, seconds
            return (
                {
                    "passthrough": True,
                    "done": True,
                    "observations": [],
                    "info": {
                        "rollout_status": RolloutStatus.RUNTIME_TIMEOUT.value,
                        "runtime_error": msg,
                    },
                },
                seconds,
            )

        seconds = time.perf_counter() - started
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            msg = f"External runtime exited {proc.returncode}: {self.runtime_cmd}" + (
                f" | stderr: {stderr[:500]}" if stderr else ""
            )
            logger.warning(msg)
            if self.runtime_fallback_to_parser:
                return None, seconds
            return (
                {
                    "passthrough": True,
                    "done": True,
                    "observations": [],
                    "info": {
                        "rollout_status": RolloutStatus.RUNTIME_ERROR.value,
                        "runtime_error": msg,
                    },
                },
                seconds,
            )

        try:
            parsed = parse_runtime_stdout(proc.stdout)
            response = normalize_runtime_response(parsed)
        except RuntimeProtocolError as exc:
            logger.warning("External runtime JSON parse failed: %s", exc)
            if self.runtime_fallback_to_parser:
                return None, seconds
            return (
                {
                    "passthrough": True,
                    "done": True,
                    "observations": [],
                    "info": {
                        "rollout_status": RolloutStatus.RUNTIME_ERROR.value,
                        "runtime_error": str(exc),
                    },
                },
                seconds,
            )

        state = response.get("state")
        if isinstance(state, dict):
            self._runtime_state = state
        return response, seconds

    def reset(
        self,
        target: str = "",
        ground_truth_flag: str = "",
        max_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Reset agent state and executor for a new episode."""
        # Close previous executor if any
        if self._executor is not None:
            self._executor.close()

        # Create executor
        from trajgym.envs.tool_executor import (
            BaseExecutor,
            RemoteBatchExecutor,
            SubprocessExecutor,
        )

        executor_type = kwargs.get("executor_type", self._executor_type)
        self._challenge_workdir = (
            str(
                kwargs.get("challenge_workdir")
                or os.getenv("CHALLENGE_WORKDIR")
                or "/root/challenge"
            ).strip()
            or "/root/challenge"
        )
        if executor_type == "remote":
            self._executor: BaseExecutor = RemoteBatchExecutor(
                target=target,
                ground_truth=ground_truth_flag,
                max_steps=max_steps * 5,
            )
        else:
            self._executor: BaseExecutor = SubprocessExecutor(
                target=target,
                ground_truth=ground_truth_flag,
                max_steps=max_steps * 5,
                default_workdir=self._challenge_workdir,
            )

        self._executor.reset()

        # Reset episode state
        self.tool_calls_history = []
        self.tool_outputs = []
        self.all_text = ""
        self.episode_done = False
        self.turns = 0
        self.max_steps = max_steps
        self._consecutive_no_tool_calls = 0
        # External runtime adapters need deterministic filesystem context.
        self._runtime_state = {"agent_workdir": self._challenge_workdir}
        self._target = target
        self._ground_truth_flag = ground_truth_flag
        prompt_messages = kwargs.get("prompt_messages")
        self._prompt_messages = (
            list(prompt_messages) if isinstance(prompt_messages, list) else []
        )
        self._challenge_id = str(kwargs.get("challenge_id", "") or "")
        self._category = str(kwargs.get("category", "") or "")
        self._difficulty = str(kwargs.get("difficulty", "") or "")
        self._infra_type = str(kwargs.get("infra_type", "") or "")
        self._objective = str(kwargs.get("objective", "") or "")
        if kwargs.get("tool_call_format") is not None:
            self._tool_call_format = (
                str(kwargs.get("tool_call_format", "")).strip()
                or self._tool_call_format
            )

    def step(self, action: str) -> StepResult:
        """Parse tool calls from LLM output and execute them.

        Logic is line-for-line identical to TrajGymTextEnv.step().
        """
        from trajgym.parsing import parse_tool_calls

        action = self._normalize_action_text(action)
        started = time.perf_counter()
        self.turns += 1
        self.all_text += "\n" + action

        # Parse tool calls from LLM output
        runtime_payload, runtime_seconds = self._run_external_runtime(action)
        runtime_info: dict[str, Any] = {}
        runtime_obs: list[dict[str, str]] = []
        tool_calls: list[dict[str, Any]] = []
        parse_seconds = 0.0

        if isinstance(runtime_payload, dict):
            runtime_info = dict(runtime_payload.get("info") or {})
            runtime_obs = runtime_payload.get("observations") or []
            if not isinstance(runtime_obs, list):
                runtime_obs = []
            runtime_tool_calls = runtime_payload.get("tool_calls") or []
            if not isinstance(runtime_tool_calls, list):
                runtime_tool_calls = []

            runtime_append = runtime_payload.get("all_text_append")
            if runtime_append:
                self.all_text += "\n" + str(runtime_append)

            if runtime_payload.get("episode_done"):
                self.episode_done = True

            # Passthrough mode: runtime fully controls observations and done.
            if self.runtime_passthrough or bool(runtime_payload.get("passthrough")):
                # Preserve tool/reward telemetry parity in passthrough mode.
                for tc in runtime_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_name = str(tc.get("name", "")).strip()
                    tc_args = tc.get("arguments", {})
                    if not tc_name:
                        continue
                    if not isinstance(tc_args, dict):
                        tc_args = {}
                    self.tool_calls_history.append(
                        {
                            "name": tc_name,
                            "arguments": json.dumps(tc_args, ensure_ascii=True),
                        }
                    )

                for obs in runtime_obs:
                    if not isinstance(obs, dict):
                        continue
                    content = str(obs.get("content", ""))
                    if not content:
                        continue
                    prompt_output = self._truncate_tool_output(content)
                    self.tool_outputs.append(prompt_output)
                    self.all_text += "\n" + prompt_output

                done = (
                    bool(runtime_payload.get("done", False))
                    or self.turns >= self.max_steps
                )
                status = normalize_rollout_status(
                    runtime_info.get(
                        "rollout_status",
                        (
                            RolloutStatus.OK.value
                            if (runtime_obs or runtime_tool_calls)
                            else RolloutStatus.NO_TOOL_CALL.value
                        ),
                    ),
                )
                if done and status == RolloutStatus.OK.value and not runtime_obs:
                    status = RolloutStatus.MAX_TURN_ABORT.value
                timing = {
                    "parse_s": runtime_seconds,
                    "execute_s": 0.0,
                    "total_s": time.perf_counter() - started,
                }
                info = {
                    **runtime_info,
                    "tool_calls": int(
                        runtime_info.get("tool_calls", len(runtime_tool_calls))
                    ),
                    "step": self.turns,
                    "episode_done": self.episode_done,
                    "rollout_status": status,
                    "timing": timing,
                }
                if done:
                    return StepResult(observations=[], done=True, info=info)
                return StepResult(observations=runtime_obs, done=False, info=info)

            tool_calls = runtime_payload.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                tool_calls = []
            parse_seconds = runtime_seconds
        else:
            parse_started = time.perf_counter()
            tool_calls = parse_tool_calls(action)
            parse_seconds = time.perf_counter() - parse_started

        # Fallback parse when runtime hook returns no tool calls and caller
        # explicitly allows parser fallback.
        if (
            not tool_calls
            and runtime_payload is not None
            and self.runtime_fallback_to_parser
        ):
            parse_started = time.perf_counter()
            tool_calls = parse_tool_calls(action)
            parse_seconds += time.perf_counter() - parse_started

        # Recovery path for weaker models that emit raw shell_command text
        # without valid tool-call wrappers.
        allow_text_fallback = runtime_payload is None or self.runtime_fallback_to_parser
        if not tool_calls and allow_text_fallback:
            fallback_command = self._extract_fallback_shell_command(action)
            if fallback_command:
                tool_calls = [
                    {
                        "name": "shell_command",
                        "arguments": {"command": fallback_command},
                    }
                ]
                runtime_info["fallback_tool_parser"] = "shell_command_from_text"
                if self._step_debug:
                    logger.info(
                        "Recovered fallback shell_command from text at step=%d: %s",
                        self.turns,
                        fallback_command[:240],
                    )

        execute_seconds = 0.0
        action_preview = (action or "").strip().replace("\n", " ")
        if len(action_preview) > 400:
            action_preview = action_preview[:400] + "..."
        has_tool_markers = self._looks_like_tool_call(action)

        if not tool_calls:
            self._consecutive_no_tool_calls += 1
            # No tool calls — model just generated text.
            done = (
                self.turns >= self.max_steps
                or self._consecutive_no_tool_calls >= self.max_consecutive_no_tool_calls
            )
            if (
                done
                and self._consecutive_no_tool_calls
                >= self.max_consecutive_no_tool_calls
            ):
                status = RolloutStatus.EMPTY_ACTION_LOOP.value
            elif has_tool_markers:
                status = RolloutStatus.PARSER_ERROR.value
            else:
                status = RolloutStatus.NO_TOOL_CALL.value
            if runtime_info.get("rollout_status"):
                candidate = normalize_rollout_status(
                    runtime_info.get("rollout_status"),
                    default=RolloutStatus.NO_TOOL_CALL,
                )
                status = _upgrade_status(status, candidate)
            logger.info(
                "StepAgent parse miss: step=%d status=%s no_tool_streak=%d chars=%d markers=%s preview=%s",
                self.turns,
                status,
                self._consecutive_no_tool_calls,
                len(action or ""),
                has_tool_markers,
                action_preview,
            )
            timing = {
                "parse_s": parse_seconds,
                "execute_s": execute_seconds,
                "total_s": time.perf_counter() - started,
            }
            info_payload = {
                "tool_calls": 0,
                "step": self.turns,
                "consecutive_no_tool_calls": self._consecutive_no_tool_calls,
                "rollout_status": status,
                "timing": timing,
                "action_chars": len(action or ""),
                "has_tool_markers": has_tool_markers,
                "action_preview": action_preview,
                **runtime_info,
            }
            if done:
                return self._apply_resilient_retry(
                    StepResult(
                        observations=[],
                        done=True,
                        info=info_payload,
                    )
                )
            if runtime_obs:
                return StepResult(
                    observations=runtime_obs,
                    done=False,
                    info=info_payload,
                )
            target = self._target or "<target_url>"
            hint = self._format_hint_text(
                parser_error=(status == RolloutStatus.PARSER_ERROR.value),
                target=target,
            )
            return StepResult(
                observations=[{"role": "user", "content": hint}],
                done=False,
                info=info_payload,
            )
        self._consecutive_no_tool_calls = 0
        if self._step_debug:
            logger.info(
                "StepAgent parsed tool calls: step=%d count=%d names=%s",
                self.turns,
                len(tool_calls),
                [str(tc.get("name", "")) for tc in tool_calls],
            )

        # Execute each tool call via executor
        obs_messages: list[dict[str, str]] = list(runtime_obs)
        status = RolloutStatus.OK.value
        for tc in tool_calls:
            tc_name = str(tc.get("name", ""))
            if self.episode_done:
                output = "[EPISODE COMPLETE] Flag already submitted."
                prompt_output = output
            else:
                tc_args_raw = tc.get("arguments", {})
                tc_args = self._rewrite_workspace_refs(tc_args_raw)
                if not isinstance(tc_args, dict):
                    tc_args = {}
                # Track for reward computation
                self.tool_calls_history.append(
                    {
                        "name": tc_name,
                        "arguments": (
                            json.dumps(tc_args)
                            if isinstance(tc_args, dict)
                            else str(tc_args)
                        ),
                    }
                )

                try:
                    execute_started = time.perf_counter()
                    resp = self._executor.step(tc_name, tc_args)
                    execute_seconds += time.perf_counter() - execute_started
                    stdout = resp.get("stdout", "")
                    stderr = resp.get("stderr", "")
                    env_done = resp.get("done", False)
                except Exception as exc:
                    logger.warning("Tool execution error: %s", exc)
                    stdout = f"[ERROR] Tool execution failed: {exc}"
                    stderr = ""
                    env_done = False
                    status = RolloutStatus.TOOL_ERROR.value

                output = stdout
                if stderr:
                    output += f"\n[stderr] {stderr}"
                derived_status = self._status_from_tool_output(output)
                if derived_status:
                    status = derived_status
                prompt_output = self._truncate_tool_output(output)
                self.tool_outputs.append(prompt_output)
                self.all_text += "\n" + prompt_output

                # Mark episode completion only from explicit tool signal.
                # String matching on stdout is unsafe (e.g. "Incorrect submission").
                if env_done:
                    self.episode_done = True
                    logger.info(
                        "Episode done at step %d (tool signaled completion)", self.turns
                    )

            # Wrap tool output in <tool_response> tags to match what ChatML
            # models expect (Nanbeige4.1-3B, Qwen3, Qwen3.5).  The tokenizer
            # chat template detects <tool_response> in user messages and treats
            # them as tool results rather than human queries, enabling correct
            # multi-turn tool-use handling and thinking-block management.
            obs_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<tool_response>\n"
                        f"[Tool: {tc_name}]\n"
                        f"{prompt_output}\n"
                        f"</tool_response>"
                    ),
                }
            )

        done = self.episode_done or self.turns >= self.max_steps
        if done:
            if self.episode_done:
                status = RolloutStatus.OK.value
            elif status == RolloutStatus.OK.value:
                status = RolloutStatus.MAX_TURN_ABORT.value
        if runtime_info.get("rollout_status"):
            candidate = normalize_rollout_status(
                runtime_info.get("rollout_status"), default=RolloutStatus.OK
            )
            status = _upgrade_status(status, candidate)
        timing = {
            "parse_s": parse_seconds,
            "execute_s": execute_seconds,
            "total_s": time.perf_counter() - started,
        }
        info_payload = {
            "tool_calls": len(tool_calls),
            "step": self.turns,
            "episode_done": self.episode_done,
            "rollout_status": status,
            "timing": timing,
            "action_chars": len(action or ""),
            "has_tool_markers": has_tool_markers,
            "action_preview": action_preview,
            "parsed_tool_names": [str(tc.get("name", "")) for tc in tool_calls],
            **runtime_info,
        }

        if done:
            result = StepResult(
                observations=[],
                done=True,
                info=info_payload,
            )
        else:
            result = StepResult(
                observations=obs_messages,
                done=False,
                info=info_payload,
            )

        return self._apply_resilient_retry(result)

    def _apply_resilient_retry(self, result: StepResult) -> StepResult:
        """Convert non-fatal done=True into a recoverable turn when resilient_mode is on.

        When resilient_mode is enabled, a missing/invalid tool call does NOT
        immediately terminate the episode. The agent returns explicit feedback
        and keeps the rollout alive until max_steps (or flag submission).
        """
        if not self.resilient_mode:
            return result
        if not result.done:
            return result
        # Preserve fatal exits (flag found, runtime failure, true max-turn end).
        if self.episode_done or self.turns >= self.max_steps:
            return result

        status = normalize_rollout_status((result.info or {}).get("rollout_status", ""))
        if status not in self._NON_FATAL_NO_ACTION:
            return result

        # Reset counter: resilient mode never terminates on missing tool calls.
        # This makes max_consecutive_no_tool_calls effectively dead when
        # resilient_mode is True, keeping rollouts alive until max_steps or
        # flag submission.
        self._consecutive_no_tool_calls = 0
        info = dict(result.info or {})
        info["rollout_status"] = RolloutStatus.NO_TOOL_CALL.value
        return StepResult(
            observations=[
                {
                    "role": "user",
                    "content": (
                        "No valid tool call was received; emit a concrete tool "
                        "call next (resilient retry behavior)."
                    ),
                }
            ],
            done=False,
            info=info,
        )

    @property
    def tools(self):
        """Use environment default tool schemas."""
        return None

    def close(self) -> None:
        """Release executor resources."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None
