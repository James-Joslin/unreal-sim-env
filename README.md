# Neural Combat AI

Reinforcement learning agent for Unreal Engine combat encounters. Trained in a Python simulation via curriculum learning, distilled to ONNX, and deployed in C++ at runtime.

## Quick Start

```bash
# Train a single stage with PPO:
python -m training.main --method ppo --stage 3 --tier large

# Full 7-stage curriculum:
python -m training.main --method ppo --curriculum

# Train then distill + export all ONNX tiers:
python -m training.main --method ppo --curriculum --distill

# Distill from an existing checkpoint (standard):
python -m training.main --distill_only --teacher checkpoints/ppo_stage7_best.pt

# Distill with AlphaGo-style amplification:
python -m training.distillation --teacher checkpoints/ppo_stage7_best.pt \
    --mode amplified --rollouts 16 --top_k 0.25

# Warm-start from a previous checkpoint:
python -m training.main --method ppo --stage 5 \
    --bc_checkpoint checkpoints/ppo_stage4_best.pt

# Test the simulation directly:
cd simulation && python combat_sim.py --stage 3 --render human
```

See [docs/quickstart.md](docs/quickstart.md) for full setup, dependencies, and CLI reference.

## How It Works

The agent observes **249 normalised floats** every 0.2s (frame-stacked x3 = 747 input), and outputs actions across three autoregressive heads: movement (9), combat (8, including dodge), and target selection (5). Training progresses through 7 curriculum stages, from hitting a stationary target to fighting a full 4-player party with allies, cover, and multiple weapon loadouts.

The structured encoder uses **4-head cross-attention** over entity slots (hostiles, allies, threats) instead of max-pooling, letting the model focus on the most tactically relevant entity given its current state. **Autoregressive action heads** condition each decision on the previous: the combat action sees which movement was chosen, and target selection sees both.

Player characters are spell-casters (Mage, Healer, Ranger) or melee fighters (Knight, Rogue) with mana pools, cast animations, and gap-closer abilities. The AI observes character type, mana fraction, commitment state, and gap-closer threat per target, enabling class-specific tactics: kiting melee characters, punishing casters mid-cast, pressuring low-mana targets.

The Python simulation mirrors the C++ UE5 environment field-for-field. After training, the policy is distilled into 5 size tiers (12K-95K params) and exported to ONNX for sub-millisecond inference.

## Training Methods

The training system is modular. Each RL algorithm is a self-contained method that plugs into shared infrastructure (environments, evaluation, curriculum progression, checkpointing, ONNX export).

| Method | Status | Description |
|---|---|---|
| `ppo` | Complete | Proximal Policy Optimization with GAE, KL early stopping, entropy annealing |
| `sac` | Skeleton | Discrete Soft Actor-Critic with twin Q-networks, auto-tuned entropy |

Adding a new method requires one file: subclass `BaseTrainer`, implement `build_model()`, `train()`, and `extract_policy()`, then register it. Every method produces a `CombatPolicy` that exports to ONNX with identical architecture.

## Documentation

| Document | What it covers |
|---|---|
| [Quickstart](docs/quickstart.md) | Dependencies, CLI reference, all runnable scripts |
| [Curriculum](docs/curriculum.md) | Stages 1-7: environments, targets, progression, DPS math |
| [Observations](docs/observations.md) | 249-float vector layout, encoding details, C++/Python parity |
| [Rewards](docs/rewards.md) | All reward components, activation table, design philosophy |
| [Weapons & Loadouts](docs/weapons.md) | 5 weapon presets, stats, arc mechanics |
| [Architecture](docs/architecture.md) | Policy network, delta encoding, cross-attention, autoregressive heads, tiers, ONNX export |
| [Web Testing Tool](docs/web-tool.md) | Browser-based debugging, batch evaluation, observation inspector |

## Glossary

| Term | Meaning |
|---|---|
| **UU** | Unreal Units. 1 UU = 1 cm. |
| **Decision tick** | 0.2s interval at which the agent observes and acts. 5 Hz standard, 10 Hz for bosses. |
| **Frame stack** | 3 consecutive observation frames concatenated (249 x 3 = 747 floats). |
| **Delta encoding** | Reshapes frame stack into current/velocity/acceleration channels. Baked into ONNX. |
| **Cross-attention** | 4-head attention from unique features (query) to entity slots (key/value). Replaces max-pooling. |
| **Autoregressive heads** | P(movement) x P(combat|movement) x P(target|movement, combat). |
| **Entity encoder** | Shared-weight sub-network for variable-count entity slots. |
| **Unique features** | 136 observation floats that are not entity slots. |
| **Hostile slot** | 17 floats per target: position, HP, LOS, class, mana, commitment, gap-closer. |
| **Ally slot** | 15 floats per ally: position, HP, velocity, target index, combat action, flanking angle. |
| **Player patterns** | 5 EMAs: aggression, evasion, predictability, range, mana burn. |
| **Cover height** | Obstacle height per direction [195-202], paired with arc clearance [240-243]. |
| **Curriculum stage** | One of 7 progressive training phases. |
| **Distillation** | Standard (logit matching) or amplified (best-of-N, AlphaGo-style). |
| **ONNX tier** | Micro (~12K), Small (~22K), Medium (~42K), Large (~56K), XL (~95K). |

## Project Structure

```
simulation/                          Python training environment
  combat_sim.py                        Core sim (Gymnasium), mirrors UE5
  combat_extensions.py                 Extended env: allies, debuffs, projectile tracking
  combat_policy.py                     Structured policy + ONNX export
  reward.py                            Reward function and per-stage weights
  frame_stack.py                       Frame stacking and vectorised env wrappers
  view_sim.py                          Visual debugger / replay viewer

training/                            Modular RL training package
  main.py                              Unified CLI (train, curriculum, distill)
  base_trainer.py                      Abstract base: env setup, eval, curriculum, checkpoints
  normalizers.py                       Observation + return normalisers
  evaluation.py                        Deterministic seeded evaluation
  distillation.py                      Standard + amplified distillation pipeline
  methods/
    ppo/
      config.py                          Hyperparameters
      actor_critic.py                    ActorCritic (autoregressive, cross-attention)
      buffer.py                          Vectorised rollout buffer with GAE
      trainer.py                         PPOTrainer
    sac/
      config.py, networks.py, buffer.py, trainer.py

docs/                                Design documentation
checkpoints/                         Training checkpoints (gitignored)
runs/                                TensorBoard logs (gitignored)
```