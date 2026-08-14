# Neural Combat AI

Reinforcement-learning combat AI for Unreal Engine encounters. The agent is trained in a fast Python/Gymnasium simulation, evaluated through deterministic seeded scenarios, distilled into multiple ONNX deployment tiers, and then consumed by the UE/C++ runtime.

The current pipeline is centred around the modular `training.main` CLI and a PPO implementation backed by shared training infrastructure.

## What This Project Contains

- **Python combat simulator** — lightweight 2D approximation of UE combat, including movement, obstacles, LOS, projectiles, weapons, cooldowns, reloads and reward computation.
- **Extended simulation systems** — allies, status effects, target acceleration, threat scoring, projectile tracking, sight cones and group summaries.
- **Structured policy network** — 249-float observations, 3-frame stacking, delta encoding, entity-slot encoders, cross-attention, GRU memory and independent one-pass action heads.
- **PPO training pipeline** — vectorised rollouts, GAE, action masks, observation/return normalisation, curriculum-aware hyperparameters, deterministic evaluation and checkpointing.
- **Distillation + ONNX export** — standard logit matching or amplified best-of-N distillation into Micro, Small, Medium and Large tiers.
- **Browser testing tool** — optional web-based debugging for ONNX inference, observation inspection and reward/action visualisation.

## Quick Start

```bash
# Train a single curriculum stage with PPO
python -m training.main --method ppo --stage 3 --tier large

# Run the full 7-stage curriculum
python -m training.main --method ppo --curriculum

# Train through curriculum, then distill/export ONNX tiers
python -m training.main --method ppo --curriculum --distill

# Distill from an existing PPO checkpoint
python -m training.main --distill_only --teacher checkpoints/ppo_stage7_best.pt

# Amplified distillation directly from the distillation module
python -m training.distillation --teacher checkpoints/ppo_stage7_best.pt \
    --mode amplified --rollouts 16 --top_k 0.25

# Visualise a trained checkpoint or ONNX model
python simulation/view_sim.py --stage 3 --model checkpoints/ppo_best.pt --render video
python simulation/view_sim.py --stage 3 --model models/v1/Combat_Large.onnx --render video
```

See [`UPDATED_quickstart.md`](docs/quickstart.md) for dependencies, CLI notes and common commands.

## Observation and Action Model

The runtime observation is **249 normalised floats per frame**. The policy receives a **3-frame stack**, giving a flat **747-float input**.

The policy outputs three independent action heads in one inference call:

- **Movement** — 9 actions: hold + 8 compass directions
- **Combat** — 9 actions: none, fire, reload, switch weapon 0/1, melee, block, dodge, reposition
- **Target** — 5 actions: 4 hostile slots + keep current target

Each head reads the same policy features and is masked and selected independently. This keeps deployment to one model inference per decision while allowing the movement direction to drive both ordinary movement and the explicit Reposition action.

## Architecture Summary

```text
747 flat observation
  -> reshape to 3 x 249 frames
  -> delta encoding: current, velocity, acceleration
  -> structured encoding per channel
       - unique features
       - hostile entity slots
       - ally entity slots
       - projectile threat slots
  -> cross-attention over entity groups
  -> MLP backbone
  -> GRU memory
  -> independent movement/combat/target heads
```

Observation normalisation can be baked into the exported ONNX graph, so the C++ runtime can feed raw flat observations directly.

## Training Pipeline

The training package is modular:

```text
training/main.py                 Unified CLI for training, curriculum and distillation
training/base_trainer.py         Shared trainer infrastructure
training/evaluation.py           Deterministic seeded evaluation
training/normalizers.py          Observation and return normalisation
training/distillation.py         Standard and amplified distillation
training/methods/ppo/config.py   Per-stage PPO hyperparameters
training/methods/ppo/trainer.py  PPO-specific training loop
training/methods/ppo/buffer.py   Vectorised PPO rollout buffer with GAE
training/methods/ppo/actor_critic.py
                                PPO actor-critic with GRU + independent heads
```

PPO currently supports:

- clipped surrogate objective
- Generalized Advantage Estimation
- KL early stopping
- entropy annealing
- per-head entropy weighting
- value-function clipping
- auxiliary target-movement prediction
- catastrophic regression reversion
- curriculum-aware stage configs

## Curriculum

Training progresses through seven stages:

1. **Melee basics** — close distance and attack
2. **Ranged fire/reload** — weapon range, shooting and reload cycle
3. **Moving targets/cover/flanking** — tracking, kiting and anti-degenerate behaviour
4. **Multi-weapon management** — switching, ammo management and arc fire
5. **Archetype behaviours** — role-specific 1v1 behaviour
6. **Basic coordination** — two allied enemies against one player
7. **Stratified squad combat** — equally sampled 1v1, 2v2, 3v3 and 4v4 encounters

See [`UPDATED_curriculum.md`](docs/curriculum.md) for per-stage details.

## ONNX Deployment Tiers

| Tier | Entity Dim | Unique Dim | Backbone | Layers | Attention Heads | GRU Hidden | Typical Use |
|---|---:|---:|---:|---:|---:|---:|---|
| Micro | 8 | 16 | 32 | 1 | 2 | 32 | Ultra-low latency |
| Small | 12 | 24 | 48 | 1 | 2 | 48 | Lightweight runtime |
| Medium | 16 | 32 | 48 | 2 | 4 | 48 | Mid-tier deployment |
| Large | 16 | 32 | 64 | 2 | 4 | 64 | Main training/deployment tier |

## Documentation

- [`UPDATED_quickstart.md`](docs/quickstart.md) — setup, commands and scripts
- [`UPDATED_architecture.md`](docs/architecture.md) — policy network, PPO model and ONNX export
- [`UPDATED_observations.md`](docs/observations.md) — 249-float observation layout
- [`UPDATED_curriculum.md`](docs/curriculum.md) — stage design and progression
- [`UPDATED_rewards.md`](docs/rewards.md) — reward philosophy and component activation
- [`UPDATED_weapons.md`](docs/weapons.md) — weapon presets and arc-fire mechanics
- [`UPDATED_web-tool.md`](docs/web-tool.md) — browser debugging/testing tool

## Generated Outputs

Runtime and training outputs are normally excluded from git:

```text
checkpoints/       PPO checkpoints
models/            ONNX exports and distillation reports
runs/              TensorBoard logs
sim_replay.mp4     Optional visualisation output
```

## Glossary

- **UU** — Unreal Units. 1 UU = 1 cm.
- **Decision tick** — simulation decision interval, commonly 0.2s / 5 Hz.
- **Frame stack** — 3 consecutive 249-float observations concatenated into 747 floats.
- **Delta encoding** — current, velocity and acceleration channels computed from the frame stack.
- **Cross-attention** — unique features query hostile, ally and threat entity slots.
- **GRU memory** — recurrent policy state used to remember encounter context.
- **Independent heads** — movement, combat and target logits projected separately from the same policy features in one inference call.
- **Distillation** — compression from a trained teacher into smaller ONNX deployment tiers.
