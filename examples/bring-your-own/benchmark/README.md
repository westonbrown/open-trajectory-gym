# How to Add Custom Challenges

Add your own CTF challenges to the training pipeline by creating a challenge registry YAML file.

## Steps

### 1. Create a challenge registry

Copy the template and add your challenges:

```bash
cp examples/bring-your-own/benchmark/challenge_template.yaml \
   configs/challenges/my_challenges.yaml
```

Each challenge entry needs:
- `id` -- Unique identifier
- `name` -- Display name
- `category` -- Challenge category (crypto, web, pwn, forensics, misc, rev)
- `difficulty` -- One of: very_easy, easy, medium, hard, expert, master
- `infra_type` -- `docker` (networked service) or `static` (file-only)
- `ground_truth_flag` -- Exact flag string for reward computation
- `description` -- Challenge description given to the agent
- `port` -- (docker only) Host port the service is exposed on
- `path_hint` -- (optional) Path to challenge files relative to benchmark root

### 2. Set up challenge infrastructure

For docker challenges, ensure containers are running:

```bash
# Using the built-in challenge manager
trajgym-challenges setup --registry configs/challenges/my_challenges.yaml

# Or manage containers yourself and just provide the registry
```

For static challenges, place files at the `path_hint` paths relative to your benchmark root.

### 3. Use in training

```bash
# SFT -- generate training data referencing your challenges
trajgym-convert --registry configs/challenges/my_challenges.yaml \
  --traces /path/to/agent/traces \
  --output data/my_sft.jsonl

# GRPO -- train against live challenges
trajgym-train rl \
  --config examples/qwen35-27b/training.yaml \
  --model /path/to/sft-merged \
  --data data/my_grpo.jsonl \
  --output outputs/grpo_custom \
  --challenge-registry configs/challenges/my_challenges.yaml

# Eval
trajgym-eval \
  --model /path/to/model \
  --challenge-registry configs/challenges/my_challenges.yaml
```

## Challenge Registry Format

See `configs/challenges/cybench.yaml` for the full 40-challenge CyBench registry, or `challenge_template.yaml` in this directory for an annotated single-entry example.

## Tips

- Flags must be exact byte-for-byte matches. Verify by running the challenge manually.
- Docker challenges need a port mapping. The port in the registry is the host-side port.
- Use `trajgym-challenges status` to verify all containers are reachable before training.
- For remote GPU training, use `target_map_path` in your training config to map registry ports to actual endpoints (e.g., SSH tunnel ports).
