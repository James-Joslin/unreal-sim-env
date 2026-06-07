# Policy Architecture

## Pipeline

```
C++ flat observation [747] → ONNX graph → 3 autoregressive action head logits
```

Everything from normalisation to action output is baked into the ONNX graph. C++ feeds raw flat observations; no preprocessing needed at runtime.

## Stages

**1. Delta Encoding** (no learnable params): Reshapes `[batch, 747]` → `[batch, 3, 249]`, then computes three channels: current frame, velocity (1st derivative), acceleration (2nd derivative). This gives temporal awareness without recurrence.

**2. Structured Group Encoding** (shared across 3 channels): Splits each 249-float frame into four groups, encodes each with specialised sub-networks, then aggregates entity slots via cross-attention:

| Group | Indices | Size | Encoder | Aggregation |
|---|---|---|---|---|
| Unique | [0:74] + [187:249] | 136 | Linear → GELU | — (single vector) |
| Hostile slots | [74:142] | 4 × 17 | Shared Linear → GELU | 4-head cross-attention |
| Ally slots | [142:187] | 3 × 15 | Shared Linear → GELU | 4-head cross-attention |
| Threat slots | extracted from 203,227,230 | 3 × 3 | Shared Linear → GELU | 4-head cross-attention |

Entity encoders use weight-sharing across slots. Cross-attention replaces max-pooling: the unique embedding queries each entity group, so the model learns "given my current state, which entity matters most?" A reloading enemy at close range gets high attention; a full-health enemy behind cover gets low. This is particularly important with the enriched entity slots (mana, commitment, class type) — the model needs to attend to the right entity's state, not the extremum across all slots.

**3. Policy Backbone**: Concatenates 3 channel embeddings → MLP with LayerNorm → GELU → features.

**4. Autoregressive Action Heads**: Actions are sampled sequentially — each head conditions on the previous:

```
features → move_head → m_logits → sample m
features + embed(m) → combat_proj → GELU → combat_head → c_logits → sample c
features + embed(m) + embed(c) → target_proj → GELU → target_head → t_logits
```

This lets the policy learn action correlations: "I chose to strafe left → I should fire at the target I'm facing" becomes a learnable conditional. Projection layers before combat/target heads keep head output shapes identical to the non-autoregressive model, so old checkpoints partially load.

Logits bounded by `tanh × 3.0`.

Training uses Large tier. Distillation cascades Large → Medium → Small → Micro, and Large → XL.

## Tier Architecture

| Tier | entity_dim | unique_dim | backbone | layers | Approx Params |
|---|---|---|---|---|---|
| Micro | 8 | 16 | 32 | 1 | ~12K |
| Small | 12 | 24 | 48 | 1 | ~22K |
| Medium | 16 | 32 | 64 | 2 | ~42K |
| Large | 16 | 32 | 96 | 2 | ~56K |
| XL | 24 | 48 | 128 | 3 | ~95K |

All tiers use 4-head cross-attention and autoregressive heads. Inference remains sub-millisecond for all tiers on CPU.

## ONNX Export

```bash
# Export a single tier:
python combat_policy.py --tier large --output_dir models/v1

# Distill all 5 tiers from a checkpoint:
python -m training.distillation --teacher checkpoints/ppo_stage7_best.pt

# Amplified distillation (AlphaGo-style best-of-N):
python -m training.distillation --teacher checkpoints/ppo_stage7_best.pt \
    --mode amplified --rollouts 16 --top_k 0.25
```

The export bakes observation normalisation (running mean/std from training) into the graph via `NormalizedPolicyWrapper`. The model accepts raw observations and outputs logits — no pre/post processing needed in C++.

Input: `observation` [batch, 747]. Outputs: `movement_logits` [batch, 9], `combat_logits` [batch, 8], `target_logits` [batch, 5].

The autoregressive conditioning (argmax → embedding lookup → projection) runs inside the ONNX graph. C++ applies action masks to the output logits and takes argmax — the internal conditioning uses unmasked argmax, which is correct >95% of the time.

## PPO Actor-Critic

During training, `ActorCritic` (in `training/methods/ppo/actor_critic.py`) uses two separate encoder+backbone streams — one for the actor (policy) and one for the critic (value). Both use the same `StructuredEncoder` with 4-head cross-attention. At export, the critic is stripped and only the actor weights are saved as a `CombatPolicy`.

Key mapping: `actor_encoder.*` → `encoder.*`, `actor_backbone.*` → `backbone.*`. Critic keys (`critic_encoder`, `critic_backbone`, `value_head`) are dropped.

## Distillation

Two modes available:

**Standard** — teacher rollouts → student matches teacher logits via temperature-scaled KL divergence + hard label cross-entropy. Cascaded: Large → Medium → Small → Micro.

**Amplified** (AlphaGo-style) — runs N rollouts per scenario with stochastic sampling, keeps the top K% by episode reward, and trains the policy to reproduce the winning actions weighted by return. Multiple iterations ratchet quality upward: better policy → better rollouts → better training targets.

Both modes produce all 5 ONNX tiers with inference benchmarks and in-sim win rate evaluation.