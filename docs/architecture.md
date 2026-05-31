# Policy Architecture

## Pipeline

```
C++ flat observation [645] → ONNX graph → 3 action head logits
```

Everything from normalisation to action output is baked into the ONNX graph. C++ feeds raw flat observations; no preprocessing needed at runtime.

## Stages

**1. Delta Encoding** (no learnable params): Reshapes `[batch, 645]` → `[batch, 3, 215]`, then computes three channels: current frame, velocity (1st derivative), acceleration (2nd derivative). This gives temporal awareness without recurrence.

**2. Structured Group Encoding** (shared across 3 channels): Splits each 215-float frame into four groups, encodes each with specialised sub-networks, and concatenates:

| Group | Indices | Size | Encoder |
|---|---|---|---|
| Unique | [0:70] + [158:215] | 127 | Linear → GELU |
| Hostile slots | [70:122] | 4 × 13 | Shared Linear → GELU → max-pool |
| Ally slots | [122:158] | 3 × 12 | Shared Linear → GELU → max-pool |
| Threat slots | extracted from 174,198,201 | 3 × 3 | Shared Linear → GELU → max-pool |

Entity encoders use weight-sharing across slots and max-pooling across the slot dimension for permutation invariance — the model doesn't care which slot a hostile is in.

**3. Policy Backbone**: Concatenates 3 channel embeddings → MLP with LayerNorm → 3 action heads (movement, combat, target). Logits bounded by `tanh × 3.0`.

## Tiers

| Tier | Entity dim | Unique dim | Backbone | Layers | Params |
|---|---|---|---|---|---|
| Micro | 8 | 16 | 32 | 1 | ~9K |
| Small | 12 | 24 | 48 | 1 | ~18K |
| Medium | 16 | 32 | 64 | 2 | ~38K |
| Large | 16 | 32 | 96 | 2 | ~48K |
| XL | 24 | 48 | 128 | 3 | ~85K |

Training uses Large. Distillation cascades Large → Medium → Small → Micro, and fine-tunes Large → XL.

## ONNX Export

```bash
python combat_policy.py --tier large --output_dir models/v1
```

The export bakes observation normalisation (running mean/std from training) into the graph via `NormalizedPolicyWrapper`. The model accepts raw observations and outputs logits — no pre/post processing needed in C++.

Input: `observation` [batch, 645]. Outputs: `movement_logits` [batch, 9], `combat_logits` [batch, 7], `target_logits` [batch, 5].

## PPO Actor-Critic

During training, `StructuredActorCritic` (in `03_ppo_train.py`) uses two separate encoder+backbone streams — one for the actor (policy) and one for the critic (value). At export, the critic is stripped and only the actor weights are saved as a `CombatPolicy`.

Key mapping: `actor_encoder.*` → `encoder.*`, `actor_backbone.*` → `backbone.*`. Critic keys are dropped.
