# Bring Your Own

Open Trajectory Gym is a platform with three main extension points. Each subdirectory has a guide for integrating your own component.

## Extension Points

| What | Directory | Effort | Description |
|------|-----------|--------|-------------|
| **Model** | `model/` | Low | Add any HuggingFace model with a training.yaml config |
| **Agent** | `agent/` | Medium | Integrate an external agent framework (LangGraph, Autogen, ADK, etc.) |
| **Benchmark** | `benchmark/` | Low | Add custom CTF challenges beyond CyBench |

## How It Works

- **Model**: Create a `training.yaml` with model name, LoRA settings, SFT/GRPO parameters. Optionally add a formatter for custom tool-call formats. See `model/training_template.yaml` for an annotated blank config.

- **Agent**: Implement either the `StepAgent` protocol (for GRPO training) or the `Agent` protocol (for eval/GEPA). Or use the runtime bridge with `TRAJGYM_AGENT_MODE=native` to shell out to any external process.

- **Benchmark**: Create a YAML challenge registry following the CyBench format. Each challenge needs an ID, category, difficulty, infrastructure type (docker/static), and ground truth flag.
