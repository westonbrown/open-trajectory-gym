"""SkyRL-backed stage-2 online RL implementation.

This module is the online-RL pipeline orchestrator: dataset conversion,
SkyRL config build, env registration, and trainer launch.
Algorithm choice (GRPO/RLOO/etc) is
configured via ``advantage_estimator`` and related settings.
"""

import contextlib
import importlib.util
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import yaml

logger = logging.getLogger(__name__)

# Canonical difficulty ordering for curriculum filtering.
_DIFFICULTY_ORDER: list[str] = [
    "very_easy",
    "easy",
    "medium",
    "hard",
    "expert",
    "master",
]
_DIFFICULTY_RANK: dict[str, int] = {d: i for i, d in enumerate(_DIFFICULTY_ORDER)}

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS_DIR = _PROJECT_ROOT / "src" / "trajgym" / "training" / "grpo_templates"


def _as_positive_int(name: str, raw_value: Any, default: int) -> int:
    """Parse a positive int config value with warning-backed fallback."""
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; defaulting to %d.", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning(
            "Invalid %s=%r (must be >0); defaulting to %d.", name, raw_value, default
        )
        return default
    return value


def _as_float(name: str, raw_value: Any, default: float) -> float:
    """Parse a float config value with warning-backed fallback."""
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; defaulting to %s.", name, raw_value, default)
        return default


def _resolve_online_rl_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return canonical ``online_rl`` config section."""
    preferred = config.get("online_rl")
    if isinstance(preferred, dict):
        return preferred
    return {}


def _resolve_policy_loss_type(online_rl_cfg: dict[str, Any]) -> str:
    """Resolve SkyRL ``policy_loss_type`` with backward-compatible aliases.

    Historically configs used ``online_rl.loss_type`` (for example ``dapo``).
    SkyRL actually consumes ``trainer.algorithm.policy_loss_type``.
    """
    raw_policy = online_rl_cfg.get("policy_loss_type")
    if raw_policy is not None and str(raw_policy).strip():
        return str(raw_policy).strip()

    raw_legacy = online_rl_cfg.get("loss_type")
    if raw_legacy is None:
        return "regular"

    legacy = str(raw_legacy).strip().lower()
    alias_map = {
        "dapo": "regular",
        "grpo": "regular",
        "ppo": "regular",
    }
    mapped = alias_map.get(legacy, legacy)
    if mapped != legacy:
        logger.info(
            "Mapping online_rl.loss_type=%r -> policy_loss_type=%r for SkyRL compatibility.",
            raw_legacy,
            mapped,
        )
    return mapped


def _detect_visible_gpu_count() -> int | None:
    """Best-effort count of GPUs visible to the current process."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        stripped = visible.strip()
        if not stripped or stripped == "-1":
            return 0
        devices = [chunk.strip() for chunk in stripped.split(",") if chunk.strip()]
        if devices:
            return len(devices)

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return None


def _flash_attn_available() -> bool:
    """Return True if flash-attn exports the symbols Transformers expects."""
    try:
        import flash_attn  # type: ignore
    except Exception:
        return False
    return all(
        hasattr(flash_attn, attr)
        for attr in ("flash_attn_func", "flash_attn_varlen_func")
    )


def _is_qwen3_5_config(hf_cfg: Any) -> bool:
    """Return True if a HF config appears to be Qwen3.5."""
    model_type = str(getattr(hf_cfg, "model_type", "")).lower()
    cfg_cls_name = str(hf_cfg.__class__.__name__).lower()
    architectures = [
        str(arch).lower() for arch in (getattr(hf_cfg, "architectures", None) or [])
    ]
    return (
        "qwen3_5" in model_type
        or "qwen3_5" in cfg_cls_name
        or any("qwen3_5" in arch for arch in architectures)
    )


def _missing_qwen3_5_fast_path_deps() -> list[str]:
    """Return missing Qwen3.5 linear-attention fast-path dependencies.

    Qwen3.5 uses flash linear attention (`fla`) and causal-conv1d for
    optimized linear-attention blocks. If missing, Transformers falls back
    to a slow torch path (`torch_chunk_gated_delta_rule`) that is unstable
    for long-horizon online RL workloads.
    """
    missing = []
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        if not is_flash_linear_attention_available():
            missing.append("flash-linear-attention (module: fla)")
        if not is_causal_conv1d_available():
            missing.append("causal-conv1d")
        return missing
    except Exception:
        # Fallback for older Transformers that may not expose these helpers.
        if importlib.util.find_spec("fla") is None:
            missing.append("flash-linear-attention (module: fla)")
        if importlib.util.find_spec("causal_conv1d") is None:
            missing.append("causal-conv1d")
        return missing


def _validate_qwen3_5_runtime_dependencies(
    hf_cfg: Any,
    online_rl_cfg: dict[str, Any],
) -> None:
    """Fail fast when Qwen3.5 runtime dependencies are missing."""
    if not _is_qwen3_5_config(hf_cfg):
        return

    missing = _missing_qwen3_5_fast_path_deps()
    if not missing:
        return

    missing_str = ", ".join(missing)
    msg = (
        "Qwen3.5 detected but required linear-attention runtime deps are "
        f"missing: {missing_str}. Install with: "
        "`uv sync --extra grpo --frozen` (preferred) or "
        "`pip install --no-deps flash-linear-attention==0.4.1 causal-conv1d==1.6.0`. "
        "Without these libs, Transformers falls back to torch linear-attention "
        "kernels that are unstable for long-context online RL."
    )
    strict = bool(online_rl_cfg.get("require_fast_linear_attention", True))
    if strict:
        raise RuntimeError(
            msg
            + " To bypass temporarily, set online_rl.require_fast_linear_attention=false "
            "(not recommended for production)."
        )
    logger.warning(
        "%s Proceeding because online_rl.require_fast_linear_attention=false.",
        msg,
    )


def _has_step_wise_resp_index_guard(source: str) -> bool:
    """Return True when SkyRL step-wise reward writes are bounds-guarded.

    We specifically detect the historical crash pattern:
        per_token_reward[resp_end_idx] = float(reward)
    without a preceding bounds check, which can raise IndexError.
    """
    guarded_patterns = [
        r"if\s+0\s*<=\s*resp_end_idx\s*<\s*len\(per_token_reward\)\s*:\s*[\r\n]+\s*per_token_reward\[resp_end_idx\]\s*=\s*float\(reward\)",
        r"per_token_reward\[\s*max\(\s*0\s*,\s*min\(\s*resp_end_idx\s*,\s*len\(per_token_reward\)\s*-\s*1\s*\)\s*\)\s*\]\s*=",
    ]
    for pattern in guarded_patterns:
        if re.search(pattern, source, flags=re.MULTILINE):
            return True

    # If the bare write isn't present, upstream may have refactored safely.
    return (
        re.search(r"per_token_reward\[resp_end_idx\]\s*=\s*float\(reward\)", source)
        is None
    )


def _validate_step_wise_resp_index_guard(
    online_rl_cfg: dict[str, Any],
    step_wise_trajectories: bool,
) -> None:
    """Fail fast if SkyRL lacks the step-wise reward index guard patch."""
    if not step_wise_trajectories:
        return

    strict = bool(online_rl_cfg.get("require_step_wise_index_guard", True))
    spec = importlib.util.find_spec("skyrl_train.generators.skyrl_gym_generator")
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin:
        logger.warning(
            "Could not locate skyrl_train.generators.skyrl_gym_generator; "
            "skipping step-wise index guard validation."
        )
        return

    source_path = Path(origin)
    try:
        source = source_path.read_text()
    except Exception as exc:
        msg = (
            "Failed to read SkyRL generator source while validating step-wise "
            f"index guard ({source_path}): {exc}"
        )
        if strict:
            raise RuntimeError(msg) from exc
        logger.warning(
            "%s Proceeding because online_rl.require_step_wise_index_guard=false.",
            msg,
        )
        return

    if _has_step_wise_resp_index_guard(source):
        return

    msg = (
        "SkyRL step-wise reward index guard is missing. This can crash training "
        "with IndexError in skyrl_gym_generator.py when resp_end_idx exceeds "
        "per_token_reward bounds. Apply docker/patches/apply_all_patches.sh "
        "or set online_rl.require_step_wise_index_guard=false to bypass."
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(
        "%s Proceeding because online_rl.require_step_wise_index_guard=false.",
        msg,
    )


def _resolve_reward_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize reward config and enforce a dict payload."""
    reward_cfg = config.get("reward")
    if reward_cfg is None:
        logger.info("No reward config provided; using default Reward weights.")
        return {}
    if not isinstance(reward_cfg, dict):
        raise TypeError(
            f"config['reward'] must be a dict if provided, got {type(reward_cfg).__name__}."
        )
    return reward_cfg


def _should_force_legacy_inference(
    model_path: str,
    *,
    allow_qwen35_new_inference: bool = False,
) -> bool:
    """Return True when SkyRL new inference should be disabled for model config.

    SkyRL's new inference server path currently initializes vLLM renderers in a
    way that can mis-handle some HuggingFace text-wrapper configs (for example
    Qwen3_5TextConfig). When this happens, vLLM raises a config type mismatch in
    multimodal processor setup before training starts.
    """
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning(
            "Could not inspect model config for inference backend selection: %s",
            exc,
        )
        return False

    cfg_cls_name = hf_cfg.__class__.__name__
    if cfg_cls_name.endswith("TextConfig"):
        logger.warning(
            "Model config class %s detected at %s. Forcing SkyRL legacy "
            "inference path (new inference is incompatible with text-wrapper configs).",
            cfg_cls_name,
            model_path,
        )
        return True
    if _is_qwen3_5_config(hf_cfg) and not allow_qwen35_new_inference:
        logger.warning(
            "Qwen3.5 config detected at %s. Forcing SkyRL legacy inference path "
            "(new inference shows intermittent /inference/v1/generate 400s and "
            "engine-core exits on this stack). Set "
            "online_rl.allow_new_inference_for_qwen35=true to override.",
            model_path,
        )
        return True
    return False


def _resolve_vllm_ready_model_path(model_path: str) -> str:
    """Resolve a vLLM-compatible model handoff path for online RL.

    Qwen3.5 merged checkpoints used for HF training can expose a text-wrapper
    config (for example ``model_type=qwen3_5_text``). SkyRL+vLLM runtime paths
    expect the vLLM-ready variant (for example sibling ``*_vllm`` directory).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not inspect model config at %s: %s", model_path, exc)
        return model_path

    cfg_cls_name = cfg.__class__.__name__
    cfg_model_type = str(getattr(cfg, "model_type", ""))
    looks_text_wrapper = cfg_cls_name.endswith("TextConfig") or cfg_model_type.endswith(
        "_text"
    )
    if not looks_text_wrapper:
        return model_path

    candidate = f"{model_path.rstrip('/')}_vllm"
    if os.path.isdir(candidate):
        try:
            from transformers import AutoConfig

            cand_cfg = AutoConfig.from_pretrained(candidate, trust_remote_code=True)
            cand_cls_name = cand_cfg.__class__.__name__
            cand_model_type = str(getattr(cand_cfg, "model_type", ""))
            cand_is_text_wrapper = cand_cls_name.endswith(
                "TextConfig"
            ) or cand_model_type.endswith("_text")
            if not cand_is_text_wrapper:
                logger.warning(
                    "Model path %s uses text-wrapper config (%s/%s). "
                    "Auto-switching GRPO runtime model to sibling vLLM-ready path: %s",
                    model_path,
                    cfg_model_type or "<unknown_model_type>",
                    cfg_cls_name,
                    candidate,
                )
                return candidate
        except Exception as exc:
            logger.warning(
                "Failed to validate sibling vLLM model path %s: %s", candidate, exc
            )

    logger.warning(
        "Model path %s appears to use text-wrapper config (%s/%s) and no validated "
        "sibling '*_vllm' path was found. GRPO runtime may fail in vLLM.",
        model_path,
        cfg_model_type or "<unknown_model_type>",
        cfg_cls_name,
    )
    return model_path


def _canonical_system_prompt() -> str:
    """Return the canonical system prompt with tool docs."""
    try:
        from trajgym.prompts import get_canonical_system_prompt

        return str(get_canonical_system_prompt())
    except Exception:
        # Keep a minimal fallback if converter import fails in stripped envs.
        return (
            "You are an expert penetration tester participating in a CTF challenge.\n"
            "Your goal is to find and capture the flag by systematically exploring and exploiting the target.\n\n"
            "Flags are often hidden or obfuscated. Check for encoding (base64, hex), "
            "reversed strings, split fragments across locations, and embedded data. "
            "Always verify your flag matches the expected format before submitting.\n\n"
            "Available tools:\n"
            "- shell_command: Run a shell command and get output\n"
            "- exec_command: Start an interactive process in a PTY session\n"
            "- write_stdin: Send input to a running PTY session\n"
            "- python_code: Execute Python code\n"
            "- read_file: Read file contents\n"
            "- grep: Search for patterns in files\n"
            "- file_search: Find files by name\n"
            "- flag_found: Submit a discovered flag\n"
            "- web_search: Search the web for information\n"
            "- apply_patch: Apply patches to files"
        )


def _normalize_prompt_system_tools(
    prompt: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure the prompt has a system message with required tool docs.

    Some upstream traces occasionally include truncated system prompts
    (for example only two short sentences). This normalizes them so online
    GRPO always trains against the same tool-aware policy prompt.
    """
    required_markers = ("Available tools:", "- shell_command:", "- python_code:")

    for msg in prompt:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        text = content if isinstance(content, str) else str(content)
        if all(marker in text for marker in required_markers):
            return prompt

        canonical = _canonical_system_prompt()
        if "Available tools:" in text:
            # Keep existing content, append only missing canonical lines.
            canonical_lines = [ln for ln in canonical.splitlines() if ln.strip()]
            merged = text.rstrip()
            for line in canonical_lines:
                if line not in merged:
                    merged += f"\n{line}"
            msg["content"] = merged
        elif text.strip():
            # Preserve custom lead-in, append canonical tools section.
            tools_block = canonical.split("Available tools:", 1)
            if len(tools_block) == 2:
                msg["content"] = (
                    f"{text.rstrip()}\n\nAvailable tools:\n{tools_block[1].lstrip()}"
                )
            else:
                msg["content"] = canonical
        else:
            msg["content"] = canonical

        logger.warning(
            "Normalized system prompt to canonical version (%d chars). "
            "Runtime _inject_tool_schemas() will add tool definitions before tokenization.",
            len(msg.get("content", "")),
        )
        return prompt

    prompt.insert(0, {"role": "system", "content": _canonical_system_prompt()})
    logger.warning("Injected missing system prompt with canonical tool docs.")
    return prompt


def _parse_lora_rank(lora_cfg: dict[str, Any]) -> int:
    """Parse LoRA rank from config with a defensive fallback."""
    raw_rank = lora_cfg.get("r", 64)
    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        logger.warning("Invalid lora.r=%r; defaulting to rank 64.", raw_rank)
        return 64


def _normalize_module_filter(raw_modules: Any, *, default: str | None) -> Any:
    """Normalize LoRA target/exclude module filters for SkyRL + PEFT.

    Important: SkyRL's FSDP worker serializes PEFT target_modules via
    ``list(peft_config["target_modules"])`` before LoRA sync. If a plain
    string is used (for example "q_proj,k_proj"), that becomes a character list
    and breaks vLLM adapter loading. We therefore pass structured lists for
    explicit module selections, and reserve a string only for the special
    ``all-linear`` selector.
    """
    if raw_modules is None:
        return default

    if isinstance(raw_modules, str):
        value = raw_modules.strip()
        if not value:
            return default
        if value == "all-linear":
            return value
        if "," in value:
            modules = [part.strip() for part in value.split(",") if part.strip()]
            return modules or default
        return [value]

    if isinstance(raw_modules, (list, tuple, set)):
        modules = [str(module).strip() for module in raw_modules if str(module).strip()]
        return modules or default

    value = str(raw_modules).strip()
    if not value:
        return default
    if value == "all-linear":
        return value
    return [value]


def _parse_lora_modules(lora_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Parse LoRA target/exclude module filters from config."""
    target_modules = _normalize_module_filter(
        lora_cfg.get("target_modules"),
        default="all-linear",
    )
    exclude_modules = _normalize_module_filter(
        lora_cfg.get("exclude_modules"),
        default=None,
    )
    return target_modules, exclude_modules


def _normalize_remote_url(url: str) -> str:
    """Normalize remote vLLM URL to SkyRL host:port format."""
    return re.sub(r"^https?://", "", str(url).strip()).rstrip("/")


def _resolve_generator_topology(
    online_rl_cfg: dict[str, Any], lora_rank: int
) -> dict[str, Any]:
    """Resolve SkyRL generator topology from config.

    SkyRL currently supports LoRA weight sync only when engines are local
    (``run_engines_locally=true``). Remote engine mode does not support
    ``LoraLoadRequest`` in upstream SkyRL.
    """
    vllm_mode = str(online_rl_cfg.get("vllm_mode", "colocate")).strip().lower()
    requested_remote_url = online_rl_cfg.get("vllm_server_url")
    remote_requested = bool(requested_remote_url)

    local_disagg_modes = {
        "server",
        "disagg",
        "disaggregated",
        "non_colocate",
        "non-colocate",
    }
    valid_modes = {"colocate", "local"} | local_disagg_modes
    if vllm_mode not in valid_modes:
        logger.warning(
            "Unknown online_rl.vllm_mode=%r; defaulting to 'colocate'.", vllm_mode
        )
        vllm_mode = "colocate"

    # Upstream SkyRL remote inference mode does not support NCCL LoRA sync.
    # With our weight_sync patch (patch_skyrl_weight_sync.py), file-based
    # LoRA sync (save adapter → HTTP /v1/load_lora_adapter) works instead.
    # On constrained GPUs (e.g. GB10 unified memory), local non-colocated
    # engines crash due to vLLM V1 subprocess issues — remote is the only
    # working topology.  Set ``allow_remote_lora: true`` to keep remote mode.
    allow_remote_lora = bool(online_rl_cfg.get("allow_remote_lora", False))
    if remote_requested and lora_rank > 0 and not allow_remote_lora:
        logger.warning(
            "online_rl.vllm_server_url=%r requested with LoRA rank=%d. "
            "SkyRL remote engines do not support NCCL LoRA weight sync; "
            "falling back to local non-colocated vLLM engines. "
            "Set online_rl.allow_remote_lora=true to keep remote mode "
            "(requires patch_skyrl_weight_sync.py for file-based sync).",
            requested_remote_url,
            lora_rank,
        )
        remote_requested = False
        vllm_mode = "server"
    elif remote_requested and lora_rank > 0 and allow_remote_lora:
        logger.info(
            "Remote vLLM + LoRA: using file-based weight sync "
            "(online_rl.allow_remote_lora=true). Ensure patch_skyrl_weight_sync.py "
            "is applied and vLLM server supports /v1/load_lora_adapter."
        )

    run_engines_locally = not remote_requested
    colocate_all = run_engines_locally and vllm_mode in {"colocate", "local"}
    remote_urls = (
        [_normalize_remote_url(requested_remote_url)]
        if remote_requested
        else ["127.0.0.1:8001"]
    )

    # SkyRL's _SKYRL_USE_NEW_INFERENCE=1 path reads external_proxy_url /
    # external_server_urls instead of remote_inference_engine_urls.  When
    # remote mode is requested we must populate BOTH so that SkyRL does NOT
    # spawn an internal vLLM engine on the training GPU (which would OOM on
    # multi-GPU setups where vLLM is on a different device).
    external_proxy = str(requested_remote_url).strip() if remote_requested else None
    external_servers = [str(requested_remote_url).strip()] if remote_requested else None

    return {
        "remote_vllm": remote_requested,
        "run_engines_locally": run_engines_locally,
        "colocate_all": colocate_all,
        "weight_sync_backend": "broadcast" if remote_requested else "nccl",
        "remote_inference_engine_urls": remote_urls,
        "external_proxy_url": external_proxy,
        "external_server_urls": external_servers,
    }


def _convert_online_rl_data(
    data_path: str,
    output_dir: str,
    registry=None,
    drop_unresolved_registry_samples: bool = False,
    drop_static_challenges: bool = False,
    max_samples: int | None = None,
    max_samples_per_challenge: int | None = None,
    target_port_offset: int = 0,
    target_host_override: str | None = None,
    fail_on_target_collisions: bool = False,
    fail_on_flag_mismatch: bool = False,
    fail_on_missing_registry_flag: bool = False,
    require_all_registry_challenges: bool = False,
    prefer_registry_target: bool = False,
    difficulty_min: str | None = None,
    difficulty_max: str | None = None,
    exclude_challenge_ids: list[str] | None = None,
) -> str:
    """Convert our GRPO JSONL to SkyRL dataset format.

    SkyRL expects each sample to have:
      - prompt: list of message dicts (system + user)
      - Per-sample extras as flat top-level keys (ground_truth_flag, etc.)

    Our GRPO JSONL has:
      - messages: full trajectory (system, user, assistant, tool, ...)
      - ground_truth_flag: str
      - metadata: dict with optimal_steps, task_type, etc.

    We extract the prompt (system + user messages before first assistant)
    and flatten metadata as top-level keys for SkyRL extras.

    Args:
        data_path: Source GRPO JSONL path.
        output_dir: Output directory for converted JSONL.
        registry: Optional ChallengeRegistry for challenge ID normalization.
        drop_unresolved_registry_samples: If True and registry is provided,
            samples whose challenge ID cannot be resolved are dropped.
        drop_static_challenges: If True and registry is provided, samples
            whose resolved challenge has infra_type="static" are dropped.
            Static challenges have no running Docker service, so they waste
            compute during online GRPO training.
        max_samples: Optional cap on converted samples (after filtering).
        max_samples_per_challenge: Optional per-challenge cap for balancing.
        target_port_offset: Optional port offset applied to parsed target URLs.
            Useful for SSH-forwarded challenge ranges (e.g., 328xx -> 430xx).
        target_host_override: Optional host override for parsed target URLs.
        fail_on_target_collisions: If True, raise when multiple challenge IDs
            resolve to the same target URL.
        fail_on_flag_mismatch: If True, raise when dataset ground_truth_flag
            mismatches canonical registry ground_truth_flag.
        fail_on_missing_registry_flag: If True, raise when a resolved registry
            challenge has an empty ground_truth_flag.
        require_all_registry_challenges: If True, require converted data to
            include all registry challenge IDs after static/difficulty filtering.
        prefer_registry_target: If True, use registry-resolved target URL when
            available, even when a user message already contains a URL.
        difficulty_min: Optional minimum difficulty (inclusive). Requires registry.
            Samples below this difficulty are skipped. One of:
            very_easy, easy, medium, hard, expert, master.
        difficulty_max: Optional maximum difficulty (inclusive). Requires registry.
            Samples above this difficulty are skipped.

    Returns:
        Path to the converted JSONL file.
    """
    import jsonlines

    output_path = os.path.join(output_dir, "skyrl_online_rl_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Validate difficulty bounds.
    min_rank: int | None = None
    max_rank: int | None = None
    if difficulty_min is not None:
        if difficulty_min not in _DIFFICULTY_RANK:
            raise ValueError(
                f"Invalid difficulty_min={difficulty_min!r}. Must be one of: {_DIFFICULTY_ORDER}"
            )
        min_rank = _DIFFICULTY_RANK[difficulty_min]
    if difficulty_max is not None:
        if difficulty_max not in _DIFFICULTY_RANK:
            raise ValueError(
                f"Invalid difficulty_max={difficulty_max!r}. Must be one of: {_DIFFICULTY_ORDER}"
            )
        max_rank = _DIFFICULTY_RANK[difficulty_max]
    if min_rank is not None and max_rank is not None and min_rank > max_rank:
        raise ValueError(
            f"difficulty_min={difficulty_min!r} is harder than difficulty_max={difficulty_max!r}."
        )

    converted = 0
    skipped = 0
    skipped_static = 0
    skipped_difficulty = 0
    unresolved_counts: dict[str, int] = {}
    missing_challenge_id = 0
    per_challenge_counts: dict[str, int] = {}
    target_to_challenges: dict[str, set[str]] = {}
    target_to_infra_types: dict[str, set[str]] = {}
    flag_mismatch_counts: dict[str, int] = {}
    missing_registry_flag_ids: set[str] = set()
    converted_registry_ids: set[str] = set()
    prompt_target_mismatch_samples = 0
    prompt_target_rewrite_samples = 0

    def _rewrite_target(raw_url: str) -> str:
        """Apply host/port overrides to a target URL."""
        # Raw TCP targets (common for crypto/pwn) are stored as host:port.
        # urlparse() treats these as scheme/path and cannot rewrite host/port,
        # so handle this form explicitly first.
        raw = str(raw_url or "").strip()
        raw_host_port = re.fullmatch(r"(?P<host>[^:/\s]+):(?P<port>\d+)", raw)
        if raw_host_port:
            host = target_host_override or raw_host_port.group("host")
            port = int(raw_host_port.group("port"))
            if target_port_offset:
                port += int(target_port_offset)
            return f"{host}:{port}"

        try:
            parsed = urlparse(raw)
        except Exception:
            return raw_url
        if not parsed.scheme or not parsed.netloc:
            return raw_url

        host = target_host_override or parsed.hostname or ""
        port = parsed.port
        if port is not None and target_port_offset:
            port = port + int(target_port_offset)

        netloc = host
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth += f":{parsed.password}"
            netloc = f"{auth}@{netloc}"
        if port is not None:
            netloc = f"{netloc}:{port}"

        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    def _first_user_url(prompt_messages: list[dict[str, str]]) -> str | None:
        """Return first URL in user prompt text, if present."""
        for msg in prompt_messages:
            if msg.get("role") != "user":
                continue
            match = re.search(r"(?:https?|file)://[^\s)]+", str(msg.get("content", "")))
            if match:
                return match.group(0)
        return None

    def _rewrite_prompt_targets(
        prompt_messages: list[dict[str, str]],
        canonical_target: str,
    ) -> list[dict[str, str]]:
        """Rewrite stale connection URLs in user prompt text to canonical target.

        This helper is intentionally benchmark-neutral. It normalizes connection
        endpoints only, and strips known legacy prompt sections that leak
        challenge-specific shortcuts.
        """

        def _strip_legacy_non_neutral_sections(text: str) -> str:
            blocked_headers = {
                "# WEB RECON CHECKLIST",
                "# WEB EXPLOIT PLAYBOOK",
                "# CHALLENGE QUICKSTART (HIGH PRIORITY)",
            }
            lines = text.splitlines()
            cleaned: list[str] = []
            skipping = False
            for line in lines:
                stripped = line.strip()
                if stripped in blocked_headers:
                    skipping = True
                    continue
                if (
                    skipping
                    and stripped.startswith("# ")
                    and stripped not in blocked_headers
                ):
                    skipping = False
                if not skipping:
                    cleaned.append(line)
            return "\n".join(cleaned)

        def _rewrite_http_url_preserve_path(url: str) -> str:
            """Rewrite only scheme/host/port to canonical target, preserving path/query."""
            try:
                canonical = urlparse(canonical_target)
                parsed = urlparse(url)
                if not canonical.scheme or not canonical.netloc:
                    return canonical_target
                return urlunparse(
                    (
                        canonical.scheme,
                        canonical.netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )
            except Exception:
                return canonical_target

        rewritten: list[dict[str, str]] = []
        for msg in prompt_messages:
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role != "user":
                rewritten.append({"role": role or "", "content": content})
                continue

            updated = content
            updated = _strip_legacy_non_neutral_sections(updated)
            # Keep structured connection lines consistent with the effective target.
            updated = re.sub(
                r"(\*\*Connection\*\*:\s*)([^\n]+)",
                lambda m: f"{m.group(1)}{canonical_target}",
                updated,
            )
            updated = re.sub(
                r"(You can interact with the challenge service at:\s*)([^\n]+)",
                lambda m: f"{m.group(1)}{canonical_target}",
                updated,
                flags=re.IGNORECASE,
            )
            # Replace stale localhost URLs (for example 328xx / 8901) with canonical target.
            if canonical_target.startswith(("http://", "https://")):
                updated = re.sub(
                    r"https?://(?:localhost|127\.0\.0\.1):\d+(?:/[^\s)\"']*)?",
                    lambda m: _rewrite_http_url_preserve_path(m.group(0)),
                    updated,
                )
                # Also rewrite docker-compose service name URLs (e.g. http://web:8080)
                # that are unreachable from the training host.
                updated = re.sub(
                    r"https?://(?!localhost|127\.0\.0\.1)[a-z][a-z0-9_-]*:\d+(?:/[^\s)\"']*)?",
                    lambda m: _rewrite_http_url_preserve_path(m.group(0)),
                    updated,
                )
            elif canonical_target.startswith("file://"):
                updated = re.sub(r"file://[^\s)\"']+", canonical_target, updated)
            elif re.fullmatch(r"[^:/\s]+:\d+", canonical_target):
                updated = re.sub(
                    r"https?://(?:localhost|127\.0\.0\.1):\d+(?:/[^\s)\"']*)?",
                    lambda m: _rewrite_http_url_preserve_path(m.group(0)),
                    updated,
                )

            rewritten.append({"role": "user", "content": updated})
        return rewritten

    with (
        jsonlines.open(data_path) as reader,
        jsonlines.open(output_path, "w") as writer,
    ):
        for sample in reader:
            if max_samples and converted >= int(max_samples):
                break
            messages = sample.get("messages", [])

            # Extract prompt: system + user messages before first assistant/tool
            prompt = []
            for msg in messages:
                role = msg.get("role", "")
                if role in ("system", "user"):
                    prompt.append({"role": role, "content": msg.get("content", "")})
                else:
                    break

            # Normalize system prompt so tool docs are always present.
            prompt = _normalize_prompt_system_tools(prompt)

            # Ensure prompt ends with user message (SkyRL requirement)
            if not prompt or prompt[-1]["role"] != "user":
                challenge = sample.get("metadata", {}).get("challenge", "")
                prompt.append(
                    {
                        "role": "user",
                        "content": (
                            f"Solve the CTF challenge{f': {challenge}' if challenge else ''}. "
                            "Find and capture the flag."
                        ),
                    }
                )

            # Flatten extras as top-level keys (SkyRL reads them as extras).
            # env_class is required — SkyRL dataset pops it to find the registered env.
            metadata = sample.get("metadata", {})

            # Extract target URL from user messages (http(s) or file://).
            target = None
            for msg in messages:
                if msg.get("role") == "user":
                    urls = re.findall(
                        r"(?:https?|file)://[^\s)]+", msg.get("content", "")
                    )
                    if urls:
                        target = urls[0]
                        break
            if not target:
                target = metadata.get("target")

            # Resolve challenge ID against registry when available.
            challenge_id = metadata.get("challenge_id") or metadata.get("challenge")
            resolved_challenge_id = challenge_id
            if registry:
                if challenge_id:
                    resolved = registry.resolve_id(str(challenge_id))
                    if resolved is not None:
                        resolved_challenge_id = resolved
                    elif drop_unresolved_registry_samples:
                        skipped += 1
                        key = str(challenge_id)
                        unresolved_counts[key] = unresolved_counts.get(key, 0) + 1
                        continue
                elif drop_unresolved_registry_samples:
                    skipped += 1
                    missing_challenge_id += 1
                    continue

            # Drop static challenges (no Docker service to attack during online GRPO).
            if drop_static_challenges and registry and resolved_challenge_id:
                try:
                    _static_info = registry.get(str(resolved_challenge_id))
                    if _static_info.infra_type == "static":
                        skipped += 1
                        skipped_static += 1
                        continue
                except KeyError:
                    pass

            # Exclude specific challenge IDs (e.g. TCP-only challenges the model can't solve).
            if (
                exclude_challenge_ids
                and resolved_challenge_id
                and str(resolved_challenge_id) in exclude_challenge_ids
            ):
                skipped += 1
                logger.debug("Skipping excluded challenge: %s", resolved_challenge_id)
                continue

            # Difficulty curriculum filter: skip challenges outside the allowed range.
            if (
                (min_rank is not None or max_rank is not None)
                and registry
                and resolved_challenge_id
            ):
                try:
                    _diff_info = registry.get(str(resolved_challenge_id))
                    diff_rank = _DIFFICULTY_RANK.get(_diff_info.difficulty)
                    if diff_rank is not None:
                        if min_rank is not None and diff_rank < min_rank:
                            skipped += 1
                            skipped_difficulty += 1
                            continue
                        if max_rank is not None and diff_rank > max_rank:
                            skipped += 1
                            skipped_difficulty += 1
                            continue
                except KeyError:
                    pass

            registry_target = None
            registry_category = None
            registry_infra_type = None
            registry_path_hint = None
            registry_flag = None
            sample_flag = str(sample.get("ground_truth_flag") or "").strip() or None
            if registry and resolved_challenge_id:
                try:
                    info = registry.get(resolved_challenge_id)
                    registry_target = registry.get_target_url(resolved_challenge_id)
                    registry_category = info.category or None
                    registry_infra_type = info.infra_type or None
                    registry_path_hint = info.path_hint or None
                    registry_flag = info.ground_truth_flag or None
                    if not registry_flag:
                        missing_registry_flag_ids.add(str(resolved_challenge_id))
                    if (
                        registry_flag
                        and sample_flag
                        and sample_flag.strip() != str(registry_flag).strip()
                    ):
                        key = str(resolved_challenge_id)
                        flag_mismatch_counts[key] = flag_mismatch_counts.get(key, 0) + 1
                except KeyError:
                    registry_target = None

            # Prefer canonical registry target when configured (useful for
            # remote/tunneled runs where prompts may contain stale localhost URLs).
            if (
                prefer_registry_target
                and registry_target
                or not target
                and registry_target
            ):
                target = registry_target
            # Static challenges do not expose network targets in the registry.
            # Use a stable file:// target so envs avoid falling back to localhost.
            if not target and registry_infra_type == "static":
                target = "file:///root/challenge/"
            if target:
                target = _rewrite_target(str(target))

            # NOTE: Previously stripped http:// from crypto/pwn targets assuming
            # raw TCP, but many CyBench crypto/pwn challenges actually expose
            # HTTP services.  Keep the target URL as-is from the registry/target
            # map — the model and prompt already handle both protocols.

            # Category from registry (e.g. "crypto", "rev", "forensics", "web")
            # falls back to metadata.category if no registry match.
            category = registry_category or metadata.get("category")
            prompt_first_url_before = _first_user_url(prompt)
            if target:
                prompt = _rewrite_prompt_targets(
                    prompt,
                    str(target),
                )
                prompt_first_url_after = _first_user_url(prompt)
                if prompt_first_url_before and prompt_first_url_before != str(target):
                    prompt_target_mismatch_samples += 1
                if (
                    prompt_first_url_before
                    and prompt_first_url_after == str(target)
                    and prompt_first_url_before != prompt_first_url_after
                ):
                    prompt_target_rewrite_samples += 1

            if max_samples_per_challenge and resolved_challenge_id:
                key = str(resolved_challenge_id)
                current = per_challenge_counts.get(key, 0)
                if current >= int(max_samples_per_challenge):
                    skipped += 1
                    continue

            row = {
                "prompt": prompt,
                "env_class": "trajgym",
                # Registry is canonical whenever it provides a flag.
                "ground_truth_flag": registry_flag or sample_flag,
                "optimal_steps": sample.get("optimal_steps")
                or metadata.get("optimal_steps"),
                "challenge_id": resolved_challenge_id,
                "task_type": metadata.get("task_type", "challenge"),
                "target": target,
                "category": category,
                "infra_type": registry_infra_type or metadata.get("infra_type"),
                "path_hint": registry_path_hint or metadata.get("path_hint"),
            }

            writer.write(row)
            converted += 1
            if resolved_challenge_id:
                key = str(resolved_challenge_id)
                per_challenge_counts[key] = per_challenge_counts.get(key, 0) + 1
                converted_registry_ids.add(key)
                if target:
                    target_to_challenges.setdefault(str(target), set()).add(key)
                    infra_key = str(
                        registry_infra_type or metadata.get("infra_type") or ""
                    ).strip()
                    if infra_key:
                        target_to_infra_types.setdefault(str(target), set()).add(
                            infra_key
                        )

    if skipped:
        logger.warning(
            "Skipped %d/%d GRPO samples during conversion (registry filtering enabled=%s)",
            skipped,
            skipped + converted,
            bool(registry and drop_unresolved_registry_samples),
        )
    if unresolved_counts:
        top = sorted(unresolved_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.warning("Top unresolved challenge IDs (sample count): %s", top)
    if missing_challenge_id:
        logger.warning(
            "Skipped %d samples with missing challenge_id/challenge metadata.",
            missing_challenge_id,
        )
    if skipped_static:
        logger.info(
            "Dropped %d static challenge samples (infra_type='static', no Docker service).",
            skipped_static,
        )
    if skipped_difficulty:
        logger.info(
            "Dropped %d samples by difficulty filter (min=%s, max=%s).",
            skipped_difficulty,
            difficulty_min,
            difficulty_max,
        )
    if prompt_target_mismatch_samples:
        logger.info(
            "Detected %d prompt/target URL mismatches; rewrote %d prompts to canonical target URLs.",
            prompt_target_mismatch_samples,
            prompt_target_rewrite_samples,
        )
    if missing_registry_flag_ids:
        sorted_missing = sorted(missing_registry_flag_ids)
        msg = (
            "Resolved registry challenges with missing ground_truth_flag: "
            f"{sorted_missing[:20]} (total={len(sorted_missing)})."
        )
        if fail_on_missing_registry_flag:
            raise ValueError(msg)
        logger.warning("%s", msg)
    if flag_mismatch_counts:
        top = sorted(flag_mismatch_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        msg = (
            "Dataset ground_truth_flag mismatches registry for "
            f"{len(flag_mismatch_counts)} challenges. Top: {top}"
        )
        if fail_on_flag_mismatch:
            raise ValueError(msg)
        logger.warning("%s", msg)
    if converted == 0:
        raise ValueError(
            "No GRPO samples remained after conversion. "
            "Check challenge registry mappings or disable drop_unresolved_registry_samples."
        )
    if require_all_registry_challenges and registry:
        expected_ids: set[str] = set()
        for info in registry.list_all():
            if drop_static_challenges and info.infra_type == "static":
                continue
            diff_rank = _DIFFICULTY_RANK.get(info.difficulty)
            if diff_rank is not None:
                if min_rank is not None and diff_rank < min_rank:
                    continue
                if max_rank is not None and diff_rank > max_rank:
                    continue
            expected_ids.add(info.id)
        missing_expected = sorted(expected_ids - converted_registry_ids)
        if missing_expected:
            raise ValueError(
                "Converted online RL data is missing registry challenges after filtering: "
                f"{missing_expected[:20]} (missing={len(missing_expected)} total={len(expected_ids)})."
            )

    if max_samples_per_challenge:
        logger.info(
            "Per-challenge cap active: max_samples_per_challenge=%s (kept %d challenges)",
            max_samples_per_challenge,
            len(per_challenge_counts),
        )
    collisions = {}
    for tgt, ids in target_to_challenges.items():
        if len(ids) <= 1:
            continue
        # Static/file-based challenges intentionally share a local workspace
        # target (for example file:///root/challenge/) and should not fail
        # the docker tunnel collision gate.
        infra_types = target_to_infra_types.get(tgt, set())
        if tgt.startswith("file://") or infra_types == {"static"}:
            continue
        collisions[tgt] = sorted(ids)
    if collisions:
        top = sorted(collisions.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        logger.warning(
            "Detected %d target URL collisions (multiple challenge IDs share one target). "
            "This often indicates stale tunnel/port mapping. Top collisions: %s",
            len(collisions),
            top,
        )
        if fail_on_target_collisions:
            raise ValueError(
                "Target URL collisions detected during GRPO data conversion; "
                "provide a challenge target map (TRAJGYM_TARGET_MAP_PATH / "
                "online_rl.target_map_path) or disable fail_on_target_collisions."
            )

    logger.info("Converted %d online RL samples → %s", converted, output_path)
    return output_path


def _resolve_skyrl_logger(report_to: str, output_dir: str) -> str:
    """Map our ``report_to`` config value to a SkyRL logger backend name.

    SkyRL tracking backends: wandb, mlflow, swanlab, tensorboard, console.
    ``"none"`` is not supported and is mapped to ``"console"``.
    ``"tensorboard"`` is supported natively; we also set the env var
    ``TENSORBOARD_LOGDIR`` so SkyRL writes to the correct directory.
    """
    value = str(report_to).strip().lower()

    _VALID_SKYRL_LOGGERS = {"wandb", "mlflow", "swanlab", "tensorboard", "console"}

    if value in ("none", "", "null"):
        return "console"

    if value == "tensorboard":
        tb_dir = os.path.join(output_dir, "tensorboard")
        with contextlib.suppress(OSError):
            os.makedirs(tb_dir, exist_ok=True)
        # SkyRL's _TensorboardAdapter reads TENSORBOARD_DIR from env.
        # Our TrajectoryLogger and trajgym_env read TENSORBOARD_LOGDIR.
        # Set both so all metrics (SkyRL training + environment) land in
        # the same directory.
        os.environ["TENSORBOARD_DIR"] = tb_dir
        os.environ["TENSORBOARD_LOGDIR"] = tb_dir
        return "tensorboard"

    if value in _VALID_SKYRL_LOGGERS:
        return value

    logger.warning(
        "Unrecognized report_to=%r; falling back to 'console'. Valid options: %s",
        report_to,
        sorted(_VALID_SKYRL_LOGGERS),
    )
    return "console"


def _setup_persistent_logging(output_dir: str) -> None:
    """Configure file-based logging to ``{output_dir}/training.log``.

    Adds a FileHandler to the root logger so that all log output
    (including SkyRL, vLLM, and our own modules) is captured in a
    persistent file alongside the usual console output.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    logger.info("Persistent training log: %s", log_path)


def _load_skyrl_defaults() -> dict[str, Any]:
    """Load SkyRL's default config as a base dict.

    Tries the Python dataclass first (SkyRL < 0.3.1), then falls back to
    the Hydra YAML config (SkyRL 0.3.1+).
    """
    skyrl_defaults: dict[str, Any] = {}
    try:
        from dataclasses import asdict

        from skyrl_train.config.config import SkyRLConfig

        skyrl_defaults = asdict(SkyRLConfig())
    except (ImportError, ModuleNotFoundError):
        pass
    if not skyrl_defaults:
        try:
            import importlib.resources as pkg_resources

            from omegaconf import OmegaConf as _OC

            cfg_dir = Path(pkg_resources.files("skyrl_train") / "config")
            base_yaml = cfg_dir / "ppo_base_config.yaml"
            if base_yaml.exists():
                raw = _OC.load(base_yaml)
                skyrl_defaults = _OC.to_container(raw, resolve=False)
                logger.info("Loaded SkyRL base config from %s", base_yaml)
        except Exception as exc:
            logger.warning("Could not load SkyRL base config: %s", exc)
    return skyrl_defaults


def _resolve_vllm_params(
    online_rl_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Resolve vLLM context-window, sequence, and batching parameters.

    Returns a flat dict with all resolved integer params that downstream
    config assembly needs (max_prompt_length, max_completion_length,
    vllm_max_model_len, num_generations, max_num_seqs, etc.).
    """
    model_max_seq_length = _as_positive_int(
        "model.max_seq_length",
        model_cfg.get("max_seq_length"),
        8192,
    )
    max_completion_length = _as_positive_int(
        "online_rl.max_completion_length",
        online_rl_cfg.get("max_completion_length"),
        8192,
    )
    max_prompt_length = _as_positive_int(
        "online_rl.max_prompt_length",
        online_rl_cfg.get("max_prompt_length"),
        model_max_seq_length,
    )

    # Keep vLLM's max_model_len sized for actual rollout windows instead of
    # the model's full context (e.g. 262K Qwen max pos emb), which can
    # allocate excessive KV cache and OOM on otherwise-valid settings.
    vllm_headroom_tokens = _as_positive_int(
        "online_rl.vllm_context_headroom_tokens",
        online_rl_cfg.get("vllm_context_headroom_tokens"),
        1024,
    )
    min_required_vllm_len = max_prompt_length + max_completion_length
    default_vllm_max_model_len = min(
        model_max_seq_length,
        min_required_vllm_len + vllm_headroom_tokens,
    )
    if default_vllm_max_model_len < min_required_vllm_len:
        default_vllm_max_model_len = min_required_vllm_len
    vllm_max_model_len = _as_positive_int(
        "online_rl.vllm_max_model_len",
        online_rl_cfg.get("vllm_max_model_len"),
        default_vllm_max_model_len,
    )
    if vllm_max_model_len < min_required_vllm_len:
        logger.warning(
            "online_rl.vllm_max_model_len=%d is smaller than max_prompt_length + "
            "max_completion_length (%d + %d = %d); overriding to %d.",
            vllm_max_model_len,
            max_prompt_length,
            max_completion_length,
            min_required_vllm_len,
            min_required_vllm_len,
        )
        vllm_max_model_len = min_required_vllm_len

    num_generations = _as_positive_int(
        "online_rl.num_generations",
        online_rl_cfg.get("num_generations"),
        8,
    )
    max_num_seqs = _as_positive_int(
        "online_rl.max_num_seqs",
        online_rl_cfg.get("max_num_seqs"),
        max(8, num_generations * 2),
    )
    if max_num_seqs < num_generations:
        logger.warning(
            "online_rl.max_num_seqs=%d is smaller than num_generations=%d; "
            "overriding max_num_seqs to %d.",
            max_num_seqs,
            num_generations,
            num_generations,
        )
        max_num_seqs = num_generations

    default_batched_tokens = min(
        32768,
        max(max_prompt_length, num_generations * max(1024, max_completion_length // 2)),
    )
    max_num_batched_tokens = _as_positive_int(
        "online_rl.max_num_batched_tokens",
        online_rl_cfg.get("max_num_batched_tokens"),
        default_batched_tokens,
    )
    if max_num_batched_tokens < max_prompt_length:
        logger.warning(
            "online_rl.max_num_batched_tokens=%d is smaller than max_prompt_length=%d; "
            "overriding to %d.",
            max_num_batched_tokens,
            max_prompt_length,
            max_prompt_length,
        )
        max_num_batched_tokens = max_prompt_length
    max_prefill_capacity = max_prompt_length * max_num_seqs
    if max_num_batched_tokens > max_prefill_capacity:
        logger.warning(
            "online_rl.max_num_batched_tokens=%d exceeds max_prefill_capacity=%d "
            "(max_prompt_length * max_num_seqs); clamping.",
            max_num_batched_tokens,
            max_prefill_capacity,
        )
        max_num_batched_tokens = max_prefill_capacity

    vllm_attention_backend = online_rl_cfg.get("vllm_attention_backend")
    if vllm_attention_backend is not None:
        vllm_attention_backend = str(vllm_attention_backend).strip() or None
    vllm_mm_encoder_attn_backend = online_rl_cfg.get("vllm_mm_encoder_attn_backend")
    if vllm_mm_encoder_attn_backend is not None:
        vllm_mm_encoder_attn_backend = str(vllm_mm_encoder_attn_backend).strip() or None

    return {
        "model_max_seq_length": model_max_seq_length,
        "max_completion_length": max_completion_length,
        "max_prompt_length": max_prompt_length,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_language_model_only": bool(
            online_rl_cfg.get("vllm_language_model_only", False)
        ),
        "vllm_attention_backend": vllm_attention_backend,
        "vllm_mm_encoder_attn_backend": vllm_mm_encoder_attn_backend,
        "num_generations": num_generations,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
    }


def _resolve_generation_params(
    online_rl_cfg: dict[str, Any],
    remote_vllm: bool,
    chat_template_name: str | None,
) -> dict[str, Any]:
    """Resolve sampling / temperature / stop parameters for train and eval.

    Returns a dict with ``train_sampling`` and ``eval_sampling`` sub-dicts
    ready for insertion into the SkyRL generator config, plus ``logprobs``
    and ``eval_logprobs`` top-level keys.
    """
    default_logprobs = None if remote_vllm else 0
    train_logprobs = online_rl_cfg.get("logprobs", default_logprobs)
    eval_logprobs = online_rl_cfg.get("eval_logprobs", train_logprobs)

    # SkyRL's multi-turn generator does not support response-logprob
    # bookkeeping when a custom chat template is used.
    if chat_template_name and train_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with logprobs=%r; forcing "
            "generator.sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            train_logprobs,
        )
        train_logprobs = None
    if chat_template_name and eval_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with eval_logprobs=%r; forcing "
            "generator.eval_sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            eval_logprobs,
        )
        eval_logprobs = None

    generation_temperature = _as_float(
        "online_rl.generation_temperature",
        online_rl_cfg.get("generation_temperature"),
        1.0,
    )
    generation_top_p = _as_float(
        "online_rl.generation_top_p",
        online_rl_cfg.get("generation_top_p"),
        0.95,
    )
    generation_min_p = _as_float(
        "online_rl.generation_min_p",
        online_rl_cfg.get("generation_min_p"),
        0.0,
    )
    generation_top_k = int(online_rl_cfg.get("generation_top_k", -1))
    generation_stop = online_rl_cfg.get("generation_stop")
    if generation_stop is not None and not isinstance(generation_stop, list):
        generation_stop = [str(generation_stop)]

    eval_generation_temperature = _as_float(
        "online_rl.eval_generation_temperature",
        online_rl_cfg.get("eval_generation_temperature"),
        0.6,
    )
    eval_generation_top_p = _as_float(
        "online_rl.eval_generation_top_p",
        online_rl_cfg.get("eval_generation_top_p"),
        0.95,
    )
    eval_generation_min_p = _as_float(
        "online_rl.eval_generation_min_p",
        online_rl_cfg.get("eval_generation_min_p"),
        0.0,
    )
    eval_generation_top_k = int(online_rl_cfg.get("eval_generation_top_k", -1))
    eval_generation_stop = online_rl_cfg.get("eval_generation_stop")
    if eval_generation_stop is not None and not isinstance(eval_generation_stop, list):
        eval_generation_stop = [str(eval_generation_stop)]

    return {
        "train_logprobs": train_logprobs,
        "eval_logprobs": eval_logprobs,
        "train_sampling": {
            "temperature": generation_temperature,
            "top_p": generation_top_p,
            "min_p": generation_min_p,
            "top_k": generation_top_k,
            "stop": generation_stop,
        },
        "eval_sampling": {
            "temperature": eval_generation_temperature,
            "top_p": eval_generation_top_p,
            "min_p": eval_generation_min_p,
            "top_k": eval_generation_top_k,
            "stop": eval_generation_stop,
        },
    }


def _resolve_chat_template_and_tools(
    online_rl_cfg: dict[str, Any],
) -> tuple[str | None, dict[str, Any], bool, bool]:
    """Resolve chat template name, kwargs, native tool schema flag, and step-wise flag.

    Returns:
        (chat_template_name, chat_template_kwargs, native_tool_schemas, step_wise_trajectories)
    """
    chat_template_name = online_rl_cfg.get("chat_template")
    chat_template_kwargs = online_rl_cfg.get("chat_template_kwargs", {})

    # Native tool schema injection via chat_template_kwargs.
    # IMPORTANT: SkyRL custom templates (qwen3_without_thinking,
    # qwen3_with_thinking) do NOT have a {% if tools %} block — they
    # silently ignore the tools= kwarg.  Auto-downgrade when detected.
    native_tool_schemas = bool(online_rl_cfg.get("native_tool_schemas", True))
    if native_tool_schemas and chat_template_name:
        logger.warning(
            "native_tool_schemas=True but custom chat_template=%r is set. "
            "SkyRL custom templates do NOT have {%% if tools %%} — tools "
            "would be silently dropped.  Auto-downgrading to "
            "native_tool_schemas=False (text injection via "
            "_inject_tool_schemas).  [Issue #38]",
            chat_template_name,
        )
        native_tool_schemas = False
    if native_tool_schemas and "tools" not in chat_template_kwargs:
        from trajgym.formatters.tool_registry import get_runtime_tools

        chat_template_kwargs = dict(chat_template_kwargs)
        chat_template_kwargs["tools"] = get_runtime_tools()
        logger.info(
            "Native tool schemas enabled: injecting %d tools into "
            "chat_template_kwargs (tokenizer will format per model template).",
            len(chat_template_kwargs["tools"]),
        )

    step_wise_trajectories = bool(online_rl_cfg.get("step_wise_trajectories", False))
    allow_step_wise_with_custom_template = bool(
        online_rl_cfg.get("allow_step_wise_with_custom_chat_template", False)
    )
    if (
        chat_template_name
        and step_wise_trajectories
        and not allow_step_wise_with_custom_template
    ):
        if bool(online_rl_cfg.get("step_wise_strict_compat", False)):
            raise ValueError(
                "online_rl.step_wise_trajectories=true is incompatible with "
                f"online_rl.chat_template={chat_template_name!r} in current SkyRL. "
                "Set online_rl.step_wise_trajectories=false, remove online_rl.chat_template, "
                "or set online_rl.step_wise_strict_compat=false to auto-disable."
            )
        logger.warning(
            "online_rl.step_wise_trajectories=true is incompatible with custom "
            "chat_template=%r in current SkyRL; auto-disabling step-wise "
            "trajectories for this run.",
            chat_template_name,
        )
        step_wise_trajectories = False
    elif (
        chat_template_name
        and step_wise_trajectories
        and allow_step_wise_with_custom_template
    ):
        logger.warning(
            "Using online_rl.allow_step_wise_with_custom_chat_template=true with "
            "chat_template=%r. Ensure SkyRL includes the step-wise + custom "
            "chat-template compatibility fix.",
            chat_template_name,
        )

    _validate_step_wise_resp_index_guard(online_rl_cfg, step_wise_trajectories)

    return (
        chat_template_name,
        chat_template_kwargs,
        native_tool_schemas,
        step_wise_trajectories,
    )


def _detect_transformer_layer_cls(
    model_path: str,
    online_rl_cfg: dict[str, Any],
) -> tuple[str | None, Any]:
    """Auto-detect the transformer decoder layer class name for FSDP wrapping.

    Also validates Qwen3.5-specific runtime dependencies when applicable.

    Returns:
        (transformer_layer_cls_name, auto_config)
    """
    _ARCH_TO_LAYER_CLS = {
        "LlamaForCausalLM": "LlamaDecoderLayer",
        "Qwen2ForCausalLM": "Qwen2DecoderLayer",
        "Qwen3ForCausalLM": "Qwen3DecoderLayer",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5DecoderLayer",
        "Qwen3NextForCausalLM": "Qwen3_5DecoderLayer",
        "TransformersForCausalLM": None,  # resolved via model_type below
        "MistralForCausalLM": "MistralDecoderLayer",
        "GptOssForCausalLM": "GptOssDecoderLayer",
    }
    _MODEL_TYPE_TO_LAYER_CLS = {
        "qwen3_5_text": "Qwen3_5DecoderLayer",
        "qwen3_5": "Qwen3_5DecoderLayer",
    }
    auto_cfg = None
    transformer_layer_cls = None
    try:
        from transformers import AutoConfig

        auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = getattr(auto_cfg, "architectures", [None])[0]
        transformer_layer_cls = _ARCH_TO_LAYER_CLS.get(arch)
        # For generic backends (TransformersForCausalLM), resolve via model_type.
        if not transformer_layer_cls:
            model_type = getattr(auto_cfg, "model_type", None)
            transformer_layer_cls = _MODEL_TYPE_TO_LAYER_CLS.get(model_type)
        if not transformer_layer_cls and arch:
            base = arch.replace("ForCausalLM", "")
            transformer_layer_cls = f"{base}DecoderLayer"
    except Exception:
        pass
    if auto_cfg is not None:
        _validate_qwen3_5_runtime_dependencies(auto_cfg, online_rl_cfg)
    return transformer_layer_cls, auto_cfg


def _merge_skyrl_defaults(
    skyrl_config: dict[str, Any],
    skyrl_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge SkyRL defaults under our overrides, then sanitize.

    Strips Hydra/OmegaConf interpolation artefacts and removes vLLM 0.16
    incompatible ``additional_kwargs`` from sampling params.
    """
    if not skyrl_defaults:
        return skyrl_config

    _HYDRA_KEYS = {"defaults", "deepspeed_config", "megatron_config"}
    for k in _HYDRA_KEYS:
        skyrl_defaults.pop(k, None)

    def _strip_interpolations(d: Any) -> Any:
        """Remove dict values that are OmegaConf interpolation strings."""
        if not isinstance(d, dict):
            return d
        cleaned = {}
        for k, v in d.items():
            if isinstance(v, str) and "${" in v:
                continue
            elif isinstance(v, dict):
                cleaned[k] = _strip_interpolations(v)
            else:
                cleaned[k] = v
        return cleaned

    skyrl_defaults = _strip_interpolations(skyrl_defaults)

    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base dict."""
        result = dict(base)
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    merged = _deep_merge(skyrl_defaults, skyrl_config)

    # Remove any remaining Hydra/OmegaConf interpolation keys
    for k in ("defaults", "deepspeed_config", "megatron_config"):
        merged.pop(k, None)

    # vLLM 0.16 rejects SamplingParams.additional_kwargs; older SkyRL
    # defaults can reintroduce it during deep-merge.
    generator_cfg = merged.get("generator", {})
    for key in ("sampling_params", "eval_sampling_params"):
        sampling = generator_cfg.get(key)
        if isinstance(sampling, dict):
            sampling.pop("additional_kwargs", None)

    return merged


def _build_skyrl_config(
    model_path: str,
    output_dir: str,
    config: dict[str, Any],
    data_path: str,
) -> dict[str, Any]:
    """Build a SkyRL config dict matching SkyRLConfig dataclass schema.

    Delegates to focused helpers for each configuration concern, then
    assembles the final dict.
    """
    skyrl_defaults = _load_skyrl_defaults()

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    online_rl_cfg = _resolve_online_rl_cfg(config)
    output_cfg = config.get("output", {})
    lora_rank = _parse_lora_rank(lora_cfg)
    lora_target_modules, lora_exclude_modules = _parse_lora_modules(lora_cfg)
    topology = _resolve_generator_topology(online_rl_cfg, lora_rank=lora_rank)
    remote_vllm = topology["remote_vllm"]
    requested_flash_attn = bool(online_rl_cfg.get("flash_attn", False))
    enable_flash_attn = requested_flash_attn and _flash_attn_available()
    if requested_flash_attn and not enable_flash_attn:
        logger.warning(
            "flash_attn requested but unavailable in this environment; falling back to SDPA."
        )
    use_sample_packing = bool(online_rl_cfg.get("use_sample_packing", False))
    if use_sample_packing and not enable_flash_attn:
        logger.warning(
            "use_sample_packing requested without flash_attn support; disabling sample packing."
        )
        use_sample_packing = False
    # --- vLLM context / sequence / batching params ----------------------
    vllm = _resolve_vllm_params(online_rl_cfg, model_cfg)
    max_prompt_length = vllm["max_prompt_length"]
    max_completion_length = vllm["max_completion_length"]
    vllm_max_model_len = vllm["vllm_max_model_len"]
    num_generations = vllm["num_generations"]
    max_num_seqs = vllm["max_num_seqs"]
    max_num_batched_tokens = vllm["max_num_batched_tokens"]

    # --- Inference engine parallelism ----------------------------------
    num_inference_engines = _as_positive_int(
        "online_rl.num_inference_engines",
        online_rl_cfg.get("num_inference_engines"),
        1,
    )
    inference_engine_tensor_parallel_size = _as_positive_int(
        "online_rl.inference_engine_tensor_parallel_size",
        online_rl_cfg.get("inference_engine_tensor_parallel_size"),
        1,
    )
    inference_engine_pipeline_parallel_size = _as_positive_int(
        "online_rl.inference_engine_pipeline_parallel_size",
        online_rl_cfg.get("inference_engine_pipeline_parallel_size"),
        1,
    )
    inference_engine_data_parallel_size = _as_positive_int(
        "online_rl.inference_engine_data_parallel_size",
        online_rl_cfg.get("inference_engine_data_parallel_size"),
        1,
    )
    inference_engine_expert_parallel_size = _as_positive_int(
        "online_rl.inference_engine_expert_parallel_size",
        online_rl_cfg.get("inference_engine_expert_parallel_size"),
        1,
    )

    # --- Training loop params ------------------------------------------
    max_tool_calling_iterations = _as_positive_int(
        "online_rl.max_tool_calling_iterations",
        online_rl_cfg.get("max_tool_calling_iterations"),
        15,
    )
    eval_interval = int(online_rl_cfg.get("eval_interval", -1))
    eval_before_train = bool(online_rl_cfg.get("eval_before_train", False))
    eval_batch_size = _as_positive_int(
        "online_rl.eval_batch_size",
        online_rl_cfg.get("eval_batch_size"),
        1,
    )
    strategy = str(online_rl_cfg.get("strategy", "fsdp2"))
    policy_loss_type = _resolve_policy_loss_type(online_rl_cfg)
    loss_reduction = str(online_rl_cfg.get("loss_reduction", "token_mean"))
    use_tis = bool(online_rl_cfg.get("use_tis", False))
    off_policy_correction = online_rl_cfg.get("off_policy_correction", {}) or {}
    if not isinstance(off_policy_correction, dict):
        off_policy_correction = {}
    fully_async_cfg = online_rl_cfg.get("fully_async", {}) or {}
    if not isinstance(fully_async_cfg, dict):
        fully_async_cfg = {}
    fully_async_max_staleness_steps = _as_positive_int(
        "online_rl.fully_async.max_staleness_steps",
        fully_async_cfg.get("max_staleness_steps"),
        4,
    )
    fully_async_num_workers = _as_positive_int(
        "online_rl.fully_async.num_parallel_generation_workers",
        fully_async_cfg.get("num_parallel_generation_workers"),
        1,
    )
    async_engine = bool(online_rl_cfg.get("async_engine", True))
    batched = bool(online_rl_cfg.get("batched", False))
    apply_overlong_filtering = bool(
        online_rl_cfg.get("apply_overlong_filtering", False)
    )
    tool_call_format = (
        str(online_rl_cfg.get("tool_call_format", "qwen3_coder")).strip()
        or "qwen3_coder"
    )

    # --- Validation data -----------------------------------------------
    val_data_cfg = online_rl_cfg.get("val_data", [])
    if isinstance(val_data_cfg, str) and val_data_cfg.strip():
        val_data = [val_data_cfg]
    elif isinstance(val_data_cfg, list):
        val_data = [str(v) for v in val_data_cfg if str(v).strip()]
    else:
        val_data = []
    max_env_workers = _as_positive_int(
        "online_rl.max_env_workers",
        online_rl_cfg.get("max_env_workers"),
        32,
    )

    # --- GPU topology / placement --------------------------------------
    use_ref_model = bool(
        online_rl_cfg.get("beta", 0.0) > 0.0
        or online_rl_cfg.get("use_kl_in_reward", False)
    )
    policy_num_gpus_per_node = _as_positive_int(
        "online_rl.policy_num_gpus_per_node",
        online_rl_cfg.get("policy_num_gpus_per_node"),
        1,
    )
    policy_num_nodes = _as_positive_int(
        "online_rl.policy_num_nodes",
        online_rl_cfg.get("policy_num_nodes"),
        1,
    )
    critic_model_path = online_rl_cfg.get("critic_model_path")
    if critic_model_path:
        critic_num_gpus_per_node = _as_positive_int(
            "online_rl.critic_num_gpus_per_node",
            online_rl_cfg.get("critic_num_gpus_per_node"),
            1,
        )
    else:
        critic_num_gpus_per_node = 0
    critic_num_nodes = _as_positive_int(
        "online_rl.critic_num_nodes",
        online_rl_cfg.get("critic_num_nodes"),
        1,
    )
    ref_num_nodes = _as_positive_int(
        "online_rl.ref_num_nodes",
        online_rl_cfg.get("ref_num_nodes"),
        policy_num_nodes,
    )
    ref_num_gpus_per_node = _as_positive_int(
        "online_rl.ref_num_gpus_per_node",
        online_rl_cfg.get("ref_num_gpus_per_node"),
        policy_num_gpus_per_node,
    )
    if not use_ref_model:
        ref_num_gpus_per_node = 0
    colocate_policy_ref = (
        bool(online_rl_cfg.get("colocate_policy_ref", True)) and use_ref_model
    )

    # Auto-adjust inference engines if GPU count is insufficient.
    visible_gpu_count = _detect_visible_gpu_count()
    if (
        visible_gpu_count is not None
        and visible_gpu_count > 0
        and topology["run_engines_locally"]
        and not topology["colocate_all"]
    ):
        policy_gpus = policy_num_nodes * policy_num_gpus_per_node
        ref_gpus = ref_num_nodes * ref_num_gpus_per_node if use_ref_model else 0
        critic_gpus = (
            critic_num_nodes * critic_num_gpus_per_node if critic_model_path else 0
        )
        gpus_per_engine = (
            inference_engine_tensor_parallel_size
            * inference_engine_pipeline_parallel_size
            * inference_engine_data_parallel_size
        )
        required_gpus = (
            policy_gpus
            + ref_gpus
            + critic_gpus
            + (num_inference_engines * gpus_per_engine)
        )
        if required_gpus > visible_gpu_count:
            explicit_num_engines = "num_inference_engines" in online_rl_cfg
            available_for_inference = max(
                0, visible_gpu_count - policy_gpus - ref_gpus - critic_gpus
            )
            max_auto_engines = available_for_inference // max(1, gpus_per_engine)
            if not explicit_num_engines and max_auto_engines > 0:
                logger.warning(
                    "Local non-colocated topology requests %d GPUs but only %d are visible. "
                    "Auto-adjusting num_inference_engines from %d to %d.",
                    required_gpus,
                    visible_gpu_count,
                    num_inference_engines,
                    max_auto_engines,
                )
                num_inference_engines = max_auto_engines
            else:
                logger.warning(
                    "Local non-colocated topology requests %d GPUs (%d policy + %d ref + %d critic + "
                    "%d inference) but only %d are visible. Training may stall or OOM.",
                    required_gpus,
                    policy_gpus,
                    ref_gpus,
                    critic_gpus,
                    num_inference_engines * gpus_per_engine,
                    visible_gpu_count,
                )

    # --- Chat template, tool schemas, step-wise ------------------------
    (
        chat_template_name,
        chat_template_kwargs,
        native_tool_schemas,
        step_wise_trajectories,
    ) = _resolve_chat_template_and_tools(online_rl_cfg)

    # --- Generation sampling params ------------------------------------
    gen = _resolve_generation_params(online_rl_cfg, remote_vllm, chat_template_name)

    # --- FSDP transformer layer detection ------------------------------
    transformer_layer_cls, _ = _detect_transformer_layer_cls(model_path, online_rl_cfg)

    # --- Reference model & clipping ------------------------------------
    ref_model_path = config.get("ref_model_path", model_path)
    eps_clip_low = online_rl_cfg.get("epsilon_low", 0.2)
    eps_clip_high = online_rl_cfg.get("epsilon_high", eps_clip_low)

    # --- Warmup steps --------------------------------------------------
    warmup_ratio = float(online_rl_cfg.get("warmup_ratio", 0.0))
    if warmup_ratio > 0:
        total_episodes = int(online_rl_cfg.get("total_episodes", 100))
        num_warmup_steps = max(1, int(total_episodes * warmup_ratio))
        logger.info(
            "Warmup: ratio=%.3f, total_episodes=%d -> num_warmup_steps=%d",
            warmup_ratio,
            total_episodes,
            num_warmup_steps,
        )
    else:
        num_warmup_steps = int(online_rl_cfg.get("num_warmup_steps", 0))

    # --- FSDP wrap policy helper ---------------------------------------
    fsdp_wrap_policy = (
        {"transformer_layer_cls_to_wrap": [transformer_layer_cls]}
        if transformer_layer_cls
        else {}
    )

    # ===================================================================
    # Assemble the final SkyRL config dict
    # ===================================================================
    skyrl_config = {
        # Data
        "data": {
            "train_data": [data_path],
            "val_data": val_data,
        },
        # Trainer
        "trainer": {
            "strategy": strategy,
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "seed": 42,
            "sequence_parallel_backend": "ulysses",
            "epochs": online_rl_cfg.get("epochs", 1),
            "update_epochs_per_batch": _as_positive_int(
                "online_rl.update_epochs_per_batch",
                online_rl_cfg.get("update_epochs_per_batch"),
                1,
            ),
            "train_batch_size": online_rl_cfg.get("batch_size", 1),
            "policy_mini_batch_size": online_rl_cfg.get("batch_size", 1),
            "critic_mini_batch_size": online_rl_cfg.get("batch_size", 1),
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_completion_length,
            "use_sample_packing": use_sample_packing,
            "eval_batch_size": eval_batch_size,
            "eval_interval": eval_interval,
            "flash_attn": enable_flash_attn,
            "disable_fast_tokenizer": False,
            "update_ref_every_epoch": False,
            "resume_mode": None,
            "resume_path": None,
            "ckpt_path": output_dir,
            "max_ckpts_to_keep": -1,
            "hf_save_interval": -1,
            "ckpt_interval": output_cfg.get("save_steps", 50),
            "export_path": os.path.join(output_dir, "final"),
            "eval_before_train": eval_before_train,
            "project_name": "trajgym",
            "run_name": "online_rl",
            "logger": _resolve_skyrl_logger(
                output_cfg.get("report_to", "tensorboard"), output_dir
            ),
            "dump_data_batch": False,
            "dump_eval_results": False,
            "target_modules": None,
            "exclude_modules": None,
            "rope_scaling": None,
            "rope_theta": None,
            "placement": {
                "colocate_all": topology["colocate_all"],
                "colocate_policy_ref": colocate_policy_ref,
                "policy_num_nodes": policy_num_nodes,
                "policy_num_gpus_per_node": policy_num_gpus_per_node,
                "critic_num_nodes": critic_num_nodes,
                "critic_num_gpus_per_node": critic_num_gpus_per_node,
                "ref_num_nodes": ref_num_nodes,
                "ref_num_gpus_per_node": ref_num_gpus_per_node,
            },
            "fully_async": {
                "max_staleness_steps": fully_async_max_staleness_steps,
                "num_parallel_generation_workers": fully_async_num_workers,
            },
            "policy": {
                "model": {
                    "path": model_path,
                    "lora": {
                        "rank": lora_rank,
                        "alpha": lora_cfg.get("alpha", 128),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "lora_sync_path": os.path.join(output_dir, "lora_sync"),
                        "init_method": "kaiming",
                    },
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "fsdp_config": {
                    "cpu_offload": bool(online_rl_cfg.get("policy_cpu_offload", False)),
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                    "wrap_policy": fsdp_wrap_policy,
                },
                "sequence_parallel_size": 1,
                "use_torch_compile": False,
                "record_memory": False,
                "optimizer_config": {
                    "lr": online_rl_cfg.get("learning_rate", 5e-6),
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": online_rl_cfg.get("weight_decay", 0.0),
                    "max_grad_norm": online_rl_cfg.get("max_grad_norm", 5.0),
                    "offload_after_step": True,
                    "num_warmup_steps": num_warmup_steps,
                    "scheduler": "constant_with_warmup",
                },
            },
            "ref": {
                "model": {
                    "path": ref_model_path,
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    "cpu_offload": online_rl_cfg.get("beta", 0.0) == 0.0,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                    "wrap_policy": fsdp_wrap_policy,
                },
            },
            "critic": {
                "model": {
                    "path": critic_model_path,
                    "lora": {
                        "rank": 0,
                        "alpha": 16,
                        "dropout": 0,
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "init_method": "kaiming",
                    },
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
                "optimizer_config": {
                    "lr": 5e-6,
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "offload_after_step": True,
                    "num_warmup_steps": num_warmup_steps,
                    "scheduler": "constant_with_warmup",
                },
            },
            "algorithm": {
                "advantage_estimator": online_rl_cfg.get("advantage_estimator", "rloo"),
                "policy_loss_type": policy_loss_type,
                "kl_loss_coef": online_rl_cfg.get("beta", 0.0),
                "use_kl_loss": online_rl_cfg.get("beta", 0.0) > 0,
                "use_kl_in_reward": False,
                "kl_ctrl": {
                    "type": "fixed",
                    "kl_target": 0.1,
                    "horizon": 10000,
                },
                "kl_estimator_type": "k3",
                "use_kl_estimator_k3": False,
                "use_abs_kl": False,
                "use_entropy_loss": bool(online_rl_cfg.get("use_entropy_loss", False)),
                "entropy_loss_coef": float(
                    online_rl_cfg.get("entropy_loss_coef", 0.01)
                ),
                "advantage_batch_normalize": False,
                "value_head_prefix": "value_head",
                "loss_reduction": loss_reduction,
                "grpo_norm_by_std": True,
                "zero_variance_filter": bool(
                    online_rl_cfg.get("zero_variance_filter", False)
                ),
                "lambd": 1.0,
                "gamma": 1.0,
                "eps_clip_low": eps_clip_low,
                "eps_clip_high": eps_clip_high,
                "clip_ratio_c": 3.0,
                "tis_imp_ratio_cap": -1.0,
                "use_tis": use_tis,
                "off_policy_correction": {
                    "tis_ratio_type": off_policy_correction.get("tis_ratio_type"),
                    "token_tis_ratio_clip_high": float(
                        off_policy_correction.get("token_tis_ratio_clip_high", 2.0)
                    ),
                    "sequence_tis_ratio_clip_high": float(
                        off_policy_correction.get("sequence_tis_ratio_clip_high", 5.0)
                    ),
                    "sequence_mask_metric": off_policy_correction.get(
                        "sequence_mask_metric"
                    ),
                    "geo_mask_high": float(
                        off_policy_correction.get("geo_mask_high", 1.01)
                    ),
                    "geo_mask_low": float(
                        off_policy_correction.get("geo_mask_low", 0.99)
                    ),
                    "product_mask_high": float(
                        off_policy_correction.get("product_mask_high", 2.0)
                    ),
                    "product_mask_low": float(
                        off_policy_correction.get("product_mask_low", 0.5)
                    ),
                    "outlier_token_is_threshold_low": off_policy_correction.get(
                        "outlier_token_is_threshold_low"
                    ),
                    "outlier_token_is_threshold_high": off_policy_correction.get(
                        "outlier_token_is_threshold_high"
                    ),
                },
                "sapo": {"tau_pos": 1.0, "tau_neg": 1.05},
                "value_clip": 0.2,
                "dynamic_sampling": {
                    "type": online_rl_cfg.get("dynamic_sampling", {}).get("type", None),
                    "max_sample_batches": _as_positive_int(
                        "online_rl.dynamic_sampling.max_sample_batches",
                        online_rl_cfg.get("dynamic_sampling", {}).get(
                            "max_sample_batches"
                        ),
                        30,
                    ),
                    "min_replace_ratio": 0.3,
                },
                "clip_cov": {
                    "clip_ratio": 0.0002,
                    "clip_cov_lb": 1.0,
                    "clip_cov_ub": 5.0,
                },
                "kl_cov": {
                    "kl_cov_frac": 0.2,
                    "ppo_kl_coef": 1.0,
                },
                "cispo": {
                    "cispo_eps_clip_low": 0,
                    "cispo_eps_clip_high": 5,
                },
            },
        },
        # Generator (vLLM inference)
        "generator": {
            "model_name": model_path,
            "model_dtype": "bfloat16",
            "run_engines_locally": topology["run_engines_locally"],
            "num_inference_engines": num_inference_engines,
            "backend": "vllm",
            "weight_sync_backend": topology["weight_sync_backend"],
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "inference_engine_tensor_parallel_size": inference_engine_tensor_parallel_size,
            "inference_engine_pipeline_parallel_size": inference_engine_pipeline_parallel_size,
            "inference_engine_expert_parallel_size": inference_engine_expert_parallel_size,
            "inference_engine_data_parallel_size": inference_engine_data_parallel_size,
            "n_samples_per_prompt": num_generations,
            "async_engine": async_engine,
            "batched": batched,
            "max_input_length": vllm_max_model_len,
            "vllm_v1_disable_multiproc": True,
            "enable_prefix_caching": bool(
                online_rl_cfg.get("enable_prefix_caching", True)
            ),
            "enable_chunked_prefill": bool(
                online_rl_cfg.get("enable_chunked_prefill", True)
            ),
            "max_num_batched_tokens": max_num_batched_tokens,
            "enforce_eager": bool(online_rl_cfg.get("enforce_eager", True)),
            "fully_sharded_loras": False,
            "enable_ray_prometheus_stats": False,
            "gpu_memory_utilization": online_rl_cfg.get("gpu_memory_utilization", 0.4),
            "max_num_seqs": max_num_seqs,
            "remote_inference_engine_urls": topology["remote_inference_engine_urls"],
            "external_proxy_url": topology.get("external_proxy_url"),
            "external_server_urls": topology.get("external_server_urls"),
            "tool_call_format": tool_call_format,
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
            "served_model_name": None,
            "max_turns": max_tool_calling_iterations,
            "chat_template": {
                "source": "name",
                "name_or_path": chat_template_name,
            },
            "chat_template_kwargs": chat_template_kwargs,
            "engine_init_kwargs": {
                "max_model_len": vllm_max_model_len,
                **(
                    {"language_model_only": True}
                    if vllm["vllm_language_model_only"]
                    else {}
                ),
                **(
                    {"attention_backend": vllm["vllm_attention_backend"]}
                    if vllm["vllm_attention_backend"]
                    else {}
                ),
                **(
                    {"mm_encoder_attn_backend": vllm["vllm_mm_encoder_attn_backend"]}
                    if vllm["vllm_mm_encoder_attn_backend"]
                    else {}
                ),
            },
            "override_existing_update_group": "auto",
            "sampling_params": {
                "max_generate_length": max_completion_length,
                "repetition_penalty": online_rl_cfg.get("repetition_penalty", 1.0),
                **gen["train_sampling"],
                "logprobs": gen["train_logprobs"],
            },
            "use_conversation_multi_turn": True,
            "append_eos_token_after_stop_str_in_multi_turn": True,
            "eval_sampling_params": {
                "max_generate_length": max_completion_length,
                "repetition_penalty": online_rl_cfg.get("repetition_penalty", 1.0),
                **gen["eval_sampling"],
                "logprobs": gen["eval_logprobs"],
            },
            "eval_n_samples_per_prompt": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": apply_overlong_filtering,
            "rope_scaling": None,
            "rope_theta": None,
            "step_wise_trajectories": step_wise_trajectories,
            "native_tool_schemas": native_tool_schemas,
        },
        # Environment
        "environment": {
            "env_class": "trajgym",
            "skyrl_gym": {
                "max_env_workers": max_env_workers,
            },
        },
    }

    return _merge_skyrl_defaults(skyrl_config, skyrl_defaults)


def train_online_rl(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: dict[str, Any],
    resume_from: str | None = None,
    challenge_registry: str | None = None,
    agent_class: str | None = None,
) -> str:
    """Run stage-2 online RL training via SkyRL.

    Uses TrajGymTextEnv (SkyRL-Gym BaseTextEnv subclass) with ToolExecutor
    for direct tool execution (no HTTP server needed).

    The reward function is reconstructed inside each SkyRL env instance
    from ``config["reward"]`` (a serializable dict). This avoids passing
    non-serializable callables through Ray.

    Args:
        model_path: Path to the SFT model (merged directory).
        data_path: Path to online-RL JSONL data with ground_truth_flag.
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict from training.yaml.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    model_path = _resolve_vllm_ready_model_path(model_path)

    # Set up persistent file logging before any other output.
    os.makedirs(output_dir, exist_ok=True)
    _setup_persistent_logging(output_dir)

    logger.info("=" * 60)
    logger.info("ONLINE RL TRAINING (SkyRL)")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Convert data to SkyRL format
    online_rl_cfg = _resolve_online_rl_cfg(config)
    registry = None
    target_map_path = (
        os.getenv("TRAJGYM_TARGET_MAP_PATH")
        or online_rl_cfg.get("target_map_path")
        or None
    )
    if challenge_registry:
        from trajgym.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(challenge_registry)
        if target_map_path:
            strict_map = bool(int(os.getenv("TRAJGYM_TARGET_MAP_STRICT", "0"))) or bool(
                online_rl_cfg.get("target_map_strict", False)
            )
            registry.load_target_overrides(str(target_map_path), strict=strict_map)
    drop_unresolved = bool(online_rl_cfg.get("drop_unresolved_registry_samples", True))
    port_offset = int(
        os.getenv(
            "TRAJGYM_TARGET_PORT_OFFSET",
            str(online_rl_cfg.get("target_port_offset", 0)),
        )
    )
    host_override = (
        os.getenv("TRAJGYM_TARGET_HOST_OVERRIDE")
        or online_rl_cfg.get("target_host_override")
        or None
    )
    prefer_registry_target = bool(
        online_rl_cfg.get("prefer_registry_target", bool(target_map_path))
    )
    converted_data = _convert_online_rl_data(
        data_path,
        output_dir,
        registry=registry,
        drop_unresolved_registry_samples=drop_unresolved,
        drop_static_challenges=bool(online_rl_cfg.get("drop_static_challenges", True)),
        max_samples=online_rl_cfg.get("max_samples"),
        max_samples_per_challenge=online_rl_cfg.get("max_samples_per_challenge"),
        target_port_offset=port_offset,
        target_host_override=host_override,
        fail_on_target_collisions=bool(
            online_rl_cfg.get("fail_on_target_collisions", False)
        ),
        fail_on_flag_mismatch=bool(online_rl_cfg.get("fail_on_flag_mismatch", True)),
        fail_on_missing_registry_flag=bool(
            online_rl_cfg.get("fail_on_missing_registry_flag", True)
        ),
        require_all_registry_challenges=bool(
            online_rl_cfg.get("require_all_registry_challenges", False)
        ),
        prefer_registry_target=prefer_registry_target,
        difficulty_min=online_rl_cfg.get("difficulty_min"),
        difficulty_max=online_rl_cfg.get("difficulty_max"),
        exclude_challenge_ids=online_rl_cfg.get("exclude_challenge_ids"),
    )

    # 2. Build SkyRL config
    skyrl_config = _build_skyrl_config(model_path, output_dir, config, converted_data)

    if resume_from:
        skyrl_config["trainer"]["resume_path"] = resume_from
        skyrl_config["trainer"]["resume_mode"] = "from_path"

    # 3. Write config for reference
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "skyrl_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(skyrl_config, f, default_flow_style=False)
    logger.info("SkyRL config written to %s", config_path)

    # 4. Launch SkyRL training
    reward_config = _resolve_reward_config(config)
    use_new_inference = bool(online_rl_cfg.get("use_new_inference", False))
    allow_qwen35_new_inference = bool(
        online_rl_cfg.get("allow_new_inference_for_qwen35", False)
    )
    if use_new_inference and _should_force_legacy_inference(
        model_path,
        allow_qwen35_new_inference=allow_qwen35_new_inference,
    ):
        use_new_inference = False

    # Trajectory logging: pass output_dir through env kwargs (Ray-serializable string).
    logging_cfg = config.get("online_rl_logging", {})
    enable_trajectory_logging = bool(logging_cfg.get("enable_trajectory_logging", True))
    require_trajectory_files = bool(logging_cfg.get("require_trajectory_files", False))
    trajectory_output_dir = output_dir if enable_trajectory_logging else None

    # Agent class from CLI flag > config file > DefaultStepAgent fallback.
    resolved_agent_class = (
        agent_class
        or online_rl_cfg.get("agent_class")
        or "trajgym.agent.default_agent.DefaultStepAgent"
    )
    resolved_agent_kwargs = online_rl_cfg.get("agent_kwargs", {})
    if not isinstance(resolved_agent_kwargs, dict):
        resolved_agent_kwargs = {}
    else:
        resolved_agent_kwargs = dict(resolved_agent_kwargs)
    resolved_agent_kwargs.setdefault("tokenizer_name_or_path", str(model_path))
    if resolved_agent_class:
        logger.info("Using custom StepAgent class: %s", resolved_agent_class)
    if isinstance(resolved_agent_kwargs, dict) and resolved_agent_kwargs:
        logger.info(
            "Using StepAgent kwargs keys: %s",
            ", ".join(sorted(str(k) for k in resolved_agent_kwargs)),
        )
        runtime_cmd = resolved_agent_kwargs.get("runtime_cmd")
        if runtime_cmd:
            logger.info("Hybrid BYO runtime enabled via runtime_cmd: %s", runtime_cmd)
    pytorch_cuda_alloc_conf = online_rl_cfg.get("pytorch_cuda_alloc_conf")
    horizon_schedule = online_rl_cfg.get("horizon_schedule")
    hard_mask_statuses = online_rl_cfg.get("hard_mask_statuses")
    try:
        positive_only_until_step = int(
            online_rl_cfg.get("positive_only_until_step", 0) or 0
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid online_rl.positive_only_until_step=%r; defaulting to 0.",
            online_rl_cfg.get("positive_only_until_step"),
        )
        positive_only_until_step = 0
    try:
        positive_only_reward_floor = float(
            online_rl_cfg.get("positive_only_reward_floor", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        logger.warning(
            "Invalid online_rl.positive_only_reward_floor=%r; defaulting to 0.0.",
            online_rl_cfg.get("positive_only_reward_floor"),
        )
        positive_only_reward_floor = 0.0
    try:
        _run_skyrl_training(
            skyrl_config,
            reward_config,
            agent_class=resolved_agent_class,
            agent_kwargs=resolved_agent_kwargs,
            use_new_inference=use_new_inference,
            trajectory_output_dir=trajectory_output_dir,
            pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
            horizon_schedule=horizon_schedule,
            hard_mask_statuses=hard_mask_statuses,
            positive_only_until_step=positive_only_until_step,
            positive_only_reward_floor=positive_only_reward_floor,
        )
    except ImportError:
        logger.error(
            "SkyRL not installed. See docs/quickstart.md for fork installation instructions."
        )
        raise

    if trajectory_output_dir and require_trajectory_files:
        trajectories_dir = os.path.join(trajectory_output_dir, "trajectories")
        has_step_files = False
        has_summary = False
        if os.path.isdir(trajectories_dir):
            for name in os.listdir(trajectories_dir):
                if name.startswith("step_") and name.endswith(".jsonl"):
                    has_step_files = True
                elif name == "step_summaries.jsonl":
                    has_summary = True
        if not (has_step_files or has_summary):
            raise RuntimeError(
                "Trajectory logging enabled but no trajectory artifacts were produced "
                f"under {trajectories_dir}. This indicates an invalid training run."
            )

    # Save challenge scoreboard after training completes.
    if trajectory_output_dir:
        try:
            from .trajectory_logger import TrajectoryLogger

            tb_dir = os.environ.get("TENSORBOARD_LOGDIR")
            tl = TrajectoryLogger(trajectory_output_dir, tensorboard_dir=tb_dir)
            tl.save_scoreboard()
            latest_step = 0
            summary_path = os.path.join(
                trajectory_output_dir, "trajectories", "step_summaries.jsonl"
            )
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            latest_step = int(
                                json.loads(line).get("global_step", latest_step)
                            )
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue
            tl.flush_scoreboard_to_tensorboard(global_step=latest_step)
            tl.close()
        except Exception as exc:
            logger.warning("Failed to save final scoreboard: %s", exc)

    logger.info("Online RL training complete. Output: %s", output_dir)

    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        return final_dir
    return output_dir


def _run_skyrl_training(
    config: dict[str, Any],
    reward_config: dict[str, Any],
    agent_class: str | None = None,
    agent_kwargs: dict[str, Any] | None = None,
    use_new_inference: bool = False,
    trajectory_output_dir: str | None = None,
    pytorch_cuda_alloc_conf: str | None = None,
    horizon_schedule: dict[str, Any] | None = None,
    hard_mask_statuses: list[str] | None = None,
    positive_only_until_step: int = 0,
    positive_only_reward_floor: float = 0.0,
) -> None:
    """Launch SkyRL training with the given config.

    Uses SkyRL's Python API (BasePPOExp). The environment is registered
    inside a Ray remote task so Ray workers can access it.

    Args:
        config: SkyRL config dict.
        reward_config: Serializable reward weight dict (reconstructed
            into Reward inside each env instance).
        agent_class: Dotted path to a StepAgent class (Ray-safe string).
        agent_kwargs: Dict of primitives for the StepAgent constructor.
        trajectory_output_dir: Output directory for trajectory JSONL logs.
            Passed through as a string (Ray-serializable) to each env instance.

    Key: exp.run() already calls asyncio.run() internally -- do NOT wrap
    in another asyncio.run().
    """
    import ray
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf DictConfig (SkyRL expects this)
    cfg = OmegaConf.create(config)

    # Import SkyRL utilities
    from skyrl_train.entrypoints.main_base import validate_cfg
    from skyrl_train.utils import initialize_ray

    def _build_trajgym_env_kwargs(
        cfg_obj: Any,
        reward_cfg: dict[str, Any],
        *,
        agent_cls: str | None,
        agent_kw: dict[str, Any] | None,
        trajectory_dir: str | None,
        horizon: dict[str, Any] | None,
        hard_mask: list[str] | None,
        positive_until_step: int,
        positive_reward_floor: float,
    ) -> dict[str, Any]:
        """Build serializable kwargs passed to TrajGymTextEnv registration."""
        generator_max_turns = int(getattr(cfg_obj.generator, "max_turns", 15))
        env_kwargs: dict[str, Any] = {
            "reward_config": reward_cfg,
            "max_turns": generator_max_turns,
            # Pass the generator's max_turns as the authoritative iteration
            # limit so the env can clamp its own max_turns to match.  This
            # prevents done=True from never firing when SkyRL's agent_loop
            # terminates on sequence length before the env reaches its own
            # max_turns threshold.
            "max_tool_calling_iterations": generator_max_turns,
            # Context-budget safety net: env proactively triggers done=True
            # when estimated tokens reach 85% of this budget, ensuring
            # terminal Reward fires BEFORE SkyRL's agent_loop length-break.
            "max_input_length": int(getattr(cfg_obj.generator, "max_input_length", 0)),
            "step_wise_trajectories": bool(
                getattr(cfg_obj.generator, "step_wise_trajectories", False)
            ),
            "native_tool_schemas": bool(
                getattr(cfg_obj.generator, "native_tool_schemas", False)
            ),
        }
        tool_call_format = str(
            getattr(cfg_obj.generator, "tool_call_format", "") or ""
        ).strip()
        if tool_call_format:
            env_kwargs["tool_call_format"] = tool_call_format
        if horizon:
            env_kwargs["horizon_schedule"] = horizon
        if hard_mask:
            env_kwargs["hard_mask_statuses"] = hard_mask
        if int(positive_until_step) > 0:
            env_kwargs["positive_only_until_step"] = int(positive_until_step)
            env_kwargs["positive_only_reward_floor"] = float(positive_reward_floor)
        if agent_cls:
            env_kwargs["agent_class"] = agent_cls
        if agent_kw:
            env_kwargs["agent_kwargs"] = agent_kw
        if trajectory_dir:
            env_kwargs["trajectory_output_dir"] = trajectory_dir
        return env_kwargs

    # Validate config against SkyRLConfig schema
    validate_cfg(cfg)

    # SkyRL's Ray actors should run vLLM V1 with in-process worker handling.
    # Keep multiprocess disabled to avoid Ray actor engine bootstrap issues.
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if pytorch_cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_cuda_alloc_conf)
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=%s", pytorch_cuda_alloc_conf)
    # Use SkyRL's new HTTP inference layer when requested.
    # This avoids legacy Ray-wrapped vLLM LoRA startup issues on Qwen3.5.
    os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"
    if use_new_inference:
        logger.info("Enabled SkyRL new inference layer (_SKYRL_USE_NEW_INFERENCE=1)")

    # Initialize Ray cluster.
    # GPU isolation: when CUDA_VISIBLE_DEVICES is set (e.g. "1"), Ray only
    # discovers those GPUs.  The legacy inference path creates a local vLLM
    # Ray actor that claims GPU 0 (via Ray's placement group), naturally
    # reserving GPU 1 for the trainer.  The new inference path with
    # external_server_urls does NOT create a vLLM actor, so the trainer is
    # the only GPU consumer and gets placed on whichever GPU Ray picks.
    _cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    _external_urls = config.get("generator", {}).get("external_server_urls")
    print(
        f"[GPU_ISOLATION] CUDA_VISIBLE_DEVICES={_cuda_vis!r}, "
        f"external_urls={bool(_external_urls)}, "
        f"use_new_inference={use_new_inference}",
        flush=True,
    )
    initialize_ray(cfg)

    # Register env and run training inside a Ray remote task.
    # This ensures the env registration is visible to Ray workers.
    @ray.remote(num_cpus=1)
    def _skyrl_entrypoint(
        cfg_dict,
        reward_config,
        agent_class,
        agent_kwargs,
        use_new_inference,
        trajectory_output_dir,
        pytorch_cuda_alloc_conf,
        horizon_schedule,
        hard_mask_statuses,
        positive_only_until_step,
        positive_only_reward_floor,
    ):
        import os as _os

        _os.environ["VLLM_USE_V1"] = "1"
        _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if pytorch_cuda_alloc_conf:
            _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_cuda_alloc_conf)
        _os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"

        from omegaconf import OmegaConf
        from skyrl_gym.envs import register
        from skyrl_train.entrypoints.main_base import BasePPOExp

        from trajgym.envs.skyrl.trajgym_env import TrajGymTextEnv

        cfg = OmegaConf.create(cfg_dict)

        # Register TrajGymTextEnv with serializable kwargs only.
        # reward_config is a plain dict of floats -- JSON-safe for SkyRL's
        # EnvSpec._check_can_jsonify(). The env reconstructs Reward
        # from this config in __init__().
        # agent_class is a dotted path string (Ray-safe).
        # agent_kwargs is a dict of primitives (Ray-safe).
        env_kwargs = _build_trajgym_env_kwargs(
            cfg,
            reward_config,
            agent_cls=agent_class,
            agent_kw=agent_kwargs,
            trajectory_dir=trajectory_output_dir,
            horizon=horizon_schedule,
            hard_mask=hard_mask_statuses,
            positive_until_step=positive_only_until_step,
            positive_reward_floor=positive_only_reward_floor,
        )

        register(
            id="trajgym",
            entry_point=TrajGymTextEnv,
            kwargs=env_kwargs,
        )

        exp = BasePPOExp(cfg)
        exp.run()  # Already calls asyncio.run() internally

    # Convert back to dict for serialization through Ray
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ray.get(
        _skyrl_entrypoint.remote(
            cfg_dict,
            reward_config,
            agent_class,
            agent_kwargs or {},
            use_new_inference,
            trajectory_output_dir,
            pytorch_cuda_alloc_conf,
            horizon_schedule,
            hard_mask_statuses,
            positive_only_until_step,
            positive_only_reward_floor,
        )
    )
