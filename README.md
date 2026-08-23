# Neural Combat AI

Reinforcement-learning combat AI for Unreal Engine encounters. The agent is trained in a fast Python/Gymnasium simulation, evaluated through deterministic seeded scenarios, distilled into multiple ONNX deployment tiers, and then consumed by the UE/C++ runtime.

## Command Runbook — Run in This Order

The commands below are the current end-to-end workflow. This machine is CPU-only: the smoke tests and evaluation commands are deliberately bounded, while full curriculum training can take a long time. Run all commands from the repository root.

### 1. Install and validate the Python environment

```bash
# Install the Python training and test dependencies.
python -m pip install -r requirements-dev.txt

# Required for CPU evaluation of the existing ONNX deployment models.
python -m pip install onnxruntime

# Verify mechanics, policy contracts, conditioning, manifests and evaluation.
python -m pytest \
  test/test_contracts.py \
  test/test_behavior_conditioning.py \
  test/test_policy_contract.py \
  test/test_manifest_evaluation.py
```

### 2. Generate the frozen scenario manifest

```bash
# Creates stratified train/validation/test scenario identities. Generate this
# once for an experiment; do not regenerate it while comparing candidates.
python -m training.scenario_manifest \
  --output scenario_manifest_v1.json \
  --scenarios_per_cell 20
```

The final-test split must remain untouched until the teacher and student choices are final. Use `validation` for the gates below.

### 3. Record the existing ONNX students as the baseline control

```bash
# CPU evaluation of all existing deployment tiers. Each manifest scenario is
# forced to use its recorded loadout and Stage-7 squad-size bucket.
python -m training.manifest_evaluation \
  --manifest scenario_manifest_v1.json \
  --split validation \
  --model large=models/Combat_Large.onnx \
  --model medium=models/Combat_Medium.onnx \
  --model small=models/Combat_Small.onnx \
  --model micro=models/Combat_Micro.onnx \
  --action_seeds 0 1 2 \
  --output_dir evaluation/baseline_students_v1
```

For a quick CPU plumbing check, append `--max_scenarios 2 --action_seeds 0`. This is not enough evidence for a behavior gate.

### 4. Smoke-test the two-profile conditioned teacher

```bash
# Short CPU-only plumbing run. It checks rollout storage, profile lifecycle,
# checkpointing and optimization; it does not establish behavior separation.
python -m training.main \
  --method ppo \
  --stage 4 \
  --tier large \
  --num_envs 2 \
  --timesteps 20000 \
  --behavior_profiles reactive tactical \
  --output_dir checkpoints/conditioned_rt_smoke
```

### 5. Train the real Reactive/Tactical teacher

```bash
# Fresh seven-stage conditioned teacher. Do not pass --bc_checkpoint or any
# distillation option. On CPU this is a long-running experiment.
python -m training.main \
  --method ppo \
  --curriculum \
  --tier large \
  --num_envs 4 \
  --behavior_profiles reactive tactical \
  --output_dir checkpoints/conditioned_rt_v1
```

`--timesteps` is ignored with `--curriculum`; use the single-stage command above for bounded smoke tests. Resume/warm-start behavior must not be used to seed this fresh conditioned teacher from an unconditioned checkpoint.

### 6. Evaluate and gate the two-profile teacher

```bash
# Greedy plus three reproducible stochastic action seeds on identical
# validation scenarios for both profiles.
python -m training.manifest_evaluation \
  --manifest scenario_manifest_v1.json \
  --split validation \
  --model rt_teacher=checkpoints/conditioned_rt_v1/ppo_stage7_best.pt \
  --profiles reactive tactical \
  --action_seeds 0 1 2 \
  --output_dir evaluation/conditioned_rt_v1
```

Inspect `results.json` and `episodes.csv`. Proceed only if Reactive and Tactical both retain objective competence and show consistent same-scenario separation in raw Heavy and Scout telemetry. Reward totals alone do not pass the gate.

### 7. Smoke-test all four profiles — Milestone 3 start

```bash
# Run only after the Reactive/Tactical gate passes. This is a bounded CPU
# plumbing test for the four categorical conditions, not an accepted teacher.
python -m training.main \
  --method ppo \
  --stage 4 \
  --tier large \
  --num_envs 2 \
  --timesteps 20000 \
  --behavior_profiles reactive competent tactical advanced \
  --output_dir checkpoints/conditioned_four_smoke
```

### 8. Train the four-profile teacher — Milestone 3

```bash
# Full fresh curriculum run. Add and gate one measurable profile distinction at
# a time; do not initialize from an unconditioned checkpoint.
python -m training.main \
  --method ppo \
  --curriculum \
  --tier large \
  --num_envs 4 \
  --behavior_profiles reactive competent tactical advanced \
  --output_dir checkpoints/conditioned_four_v1
```

### 9. Evaluate the four-profile teacher

```bash
# Produces episode telemetry, per-cell/overall/equal-cell summaries, paired
# profile deltas and 95% confidence intervals on CPU.
python -m training.manifest_evaluation \
  --manifest scenario_manifest_v1.json \
  --split validation \
  --model four_teacher=checkpoints/conditioned_four_v1/ppo_stage7_best.pt \
  --profiles reactive competent tactical advanced \
  --action_seeds 0 1 2 \
  --output_dir evaluation/conditioned_four_v1
```

Do not use the final-test split for iterative tuning. Milestone 3 is complete only when all four profiles pass objective guardrails, remain behaviorally distinct, and express those differences appropriately across required loadouts and archetypes.

### 10. Stop before Milestone 4 distillation

Conditioned reference-student distillation is not implemented or authorized yet. The existing commands below are valid only for the older unconditioned pipeline and must not be run on a behavior-conditioned checkpoint:

```bash
# Legacy/unconditioned training followed by standard distillation.
python -m training.main --method ppo --curriculum --distill

# Legacy/unconditioned distillation from an existing teacher.
python -m training.main \
  --distill_only \
  --teacher checkpoints/ppo_stage7_best.pt

# Legacy/unconditioned amplified distillation.
python -m training.distillation \
  --teacher checkpoints/ppo_stage7_best.pt \
  --mode amplified \
  --rollouts 16 \
  --top_k 0.25
```

Milestone 4 begins by implementing fixed-profile, target-contract-aware reference-student dataset generation and direct distillation after the four-profile teacher passes its gate.

The current pipeline is centred around the modular `training.main` CLI and a PPO implementation backed by shared training infrastructure.

## What This Project Contains

- **Python combat simulator** — lightweight 2D approximation of UE combat, including movement, obstacles, LOS, projectiles, weapons, cooldowns, reloads and reward computation.
- **Extended simulation systems** — allies, status effects, target acceleration, threat scoring, projectile tracking, sight cones and group summaries.
- **Structured policy network** — 249-float observations, 3-frame stacking, delta encoding, entity-slot encoders, cross-attention, GRU memory and independent one-pass action heads.
- **PPO training pipeline** — vectorised rollouts, GAE, action masks, observation/return normalisation, curriculum-aware hyperparameters, deterministic evaluation and checkpointing.
- **Distillation + ONNX export** — standard logit matching or amplified best-of-N distillation into Micro, Small, Medium and Large tiers.
- **Browser testing tool** — optional web-based debugging for ONNX inference, observation inspection and reward/action visualisation.

## Optional Inspection Commands

```bash
# Render a trained PyTorch checkpoint to video.
python simulation/view_sim.py \
  --stage 3 \
  --model checkpoints/ppo_stage3_best.pt \
  --render video

# Render an ONNX deployment model to video using CPU inference.
python simulation/view_sim.py \
  --stage 3 \
  --model models/Combat_Large.onnx \
  --render video

# Inspect training curves while a run is active.
python -m tensorboard.main --logdir runs
```

See [`UPDATED_quickstart.md`](docs/quickstart.md) for additional setup and CLI notes.

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
