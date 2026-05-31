# Neural Combat AI

Reinforcement learning agent for Unreal Engine combat encounters. Trained in a Python simulation via PPO curriculum learning, distilled to ONNX, and deployed in C++ at runtime.

## Quick Start

```bash
# Test the simulation:
python combat_sim.py --stage 3 --render human

# Train from scratch:
python 03_ppo_train.py --stage 1

# Full curriculum:
python 03_ppo_train.py --curriculum

# Distill and export to ONNX:
python 02_distill_and_export.py --teacher checkpoints/ppo_stage7_best.pt --output_dir models/v1
```

See [docs/quickstart.md](docs/quickstart.md) for full setup, dependencies, and CLI reference.

## How It Works

The agent observes **215 normalised floats** every 0.2s (frame-stacked ×3 = 645 input), and outputs actions across three heads: movement (9), combat (7), and target selection (5). Training progresses through 7 curriculum stages — from hitting a stationary target to fighting a full 4-player party with allies, cover, and multiple weapon loadouts.

The Python simulation mirrors the C++ UE5 environment field-for-field. After training, the policy is distilled into 5 size tiers (9K–85K params) and exported to ONNX for real-time inference.

## Documentation

| Document | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Dependencies, CLI reference, all runnable scripts |
| [Curriculum](docs/curriculum.md) | Stages 1–7: environments, targets, progression, DPS math |
| [Observations](docs/observations.md) | 215-float vector layout, encoding details, C++/Python parity |
| [Rewards](docs/rewards.md) | All reward components, activation table, design philosophy |
| [Weapons & Loadouts](docs/weapons.md) | 5 weapon presets, stats, arc mechanics |
| [Architecture](docs/architecture.md) | Policy network, delta encoding, tiers, ONNX export |
| [Web Testing Tool](docs/web-tool.md) | Browser-based debugging, batch evaluation, observation inspector |

## Glossary

| Term | Meaning |
|---|---|
| **UU** | Unreal Units — the spatial unit in both the sim and UE5. 1 UU ≈ 1 cm. |
| **Decision tick** | 0.2s interval at which the agent observes and acts. 5 Hz base rate. |
| **Frame stack** | 3 consecutive observation frames concatenated (215 × 3 = 645 floats). Gives the model temporal context without recurrence. |
| **Delta encoding** | The policy network reshapes the frame stack into current/velocity/acceleration channels before processing. Baked into ONNX. |
| **Entity encoder** | Shared-weight sub-network that processes variable-count slots (hostiles, allies, threats) with max-pooling for permutation invariance. |
| **Unique features** | The 127 observation floats that aren't entity slots — self state, weapons, archetype, primary target, spatial ring, cover, navmesh, group metrics, and weapon capabilities. |
| **Hostile slot** | One of 4 fixed-size slots (13 floats each) encoding a hostile target. Empty slots are zero-filled. |
| **Ally slot** | One of 3 fixed-size slots (12 floats each) encoding an allied robot. |
| **Cover height** | Continuous obstacle height per direction ([166-173]), normalised by 500 UU. Paired with arc clearance per weapon ([211-214]) so the model can reason about which weapons clear which cover. |
| **Arc clearance** | `MaxArcableObstacleHeight / 3000` per weapon slot. The maximum obstacle height a weapon's projectile can arc over. |
| **Curriculum stage** | One of 7 progressive training phases. Each adds complexity (moving targets → cover → weapons → allies → full squads). |
| **Shaping reward** | Small per-step incentive (flanking, positioning) that guides learning without dominating the objective rewards (kills, wins). Kept under 15% of total episode reward. |
| **Distillation** | Compressing a large trained policy into smaller tiers (Micro→XL) via knowledge distillation. Each tier trades inference cost for accuracy. |
| **ONNX tier** | One of 5 model sizes: Micro (9K params), Small (18K), Medium (38K), Large (48K), XL (85K). |

## File Overview

```
├── combat_sim.py               Python training environment (Gymnasium)
├── combat_extensions.py         Extended env with allies, debuffs, multi-agent
├── combat_policy.py             Structured policy network + ONNX export
├── reward.py                    Reward function and per-stage weights
├── frame_stack.py               Frame stacking and data pipeline
├── 03_ppo_train.py              PPO training loop with curriculum
├── 02_distill_and_export.py     Distillation pipeline (5 tiers → ONNX)
├── obs_vector_validator.py      C++/Python observation parity checker
│
├── NeuralCombatTypes.h          C++ observation layout and constants
├── NeuralCombatComponent.h/cpp  C++ observation assembler
├── EnemyPerceptionComponent.h/cpp  Spatial ring, cover, threat detection
├── CombatPipeline.h/cpp         C++ inference pipeline (ONNX Runtime)
│
└── docs/
    ├── quickstart.md
    ├── curriculum.md
    ├── observations.md
    ├── rewards.md
    ├── weapons.md
    ├── architecture.md
    └── web-tool.md
```
