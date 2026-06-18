# Policy Architecture

## Pipeline

```text
Raw observation [747]
  -> reshape [3, 249]
  -> delta encoder: current / velocity / acceleration
  -> structured encoder per channel
  -> cross-attention over hostiles, allies and threats
  -> MLP backbone
  -> GRU memory
  -> autoregressive action heads
  -> ONNX logits for movement, combat and target selection
```

The C++ runtime feeds raw frame-stacked observations. Observation normalisation can be embedded directly inside the ONNX graph during export.

## Observation Input

- Single frame: **249 floats**
- Frame stack: **3 frames**
- Policy input: **747 floats**

The frame stack is oldest-first: `[t-2, t-1, t]`.

## Delta Encoding

The policy computes three temporal channels:

```text
current      = frame[t]
velocity     = frame[t] - frame[t-1]
acceleration = frame[t] - 2 * frame[t-1] + frame[t-2]
```

This gives temporal awareness without requiring the observation builder itself to emit explicit derivatives for every field.

## Structured Encoder

Each 249-float frame is split into:

- **Unique features** — `[0:74] + [187:249]`, 136 floats
- **Hostile slots** — 4 × 17 floats
- **Ally slots** — 3 × 15 floats
- **Threat slots** — 3 × 3 floats extracted from projectile threat fields

Hostile, ally and threat slots use shared encoders followed by cross-attention. The unique feature embedding acts as the query, so the policy can learn which entity matters in the current tactical context.

## GRU Memory

Both the PPO actor and exported policy include a GRU on the actor path. This gives the agent encounter-level memory beyond the 3-frame input window.

The critic path in PPO does not use the GRU. It estimates value from the current encoded observation, simplifying hidden-state management during PPO updates.

## Autoregressive Action Heads

Action logits are produced sequentially:

```text
features -> movement logits -> movement action
features + movement embedding -> combat logits -> combat action
features + movement embedding + combat embedding -> target logits
```

This lets the model learn correlated decisions, such as pairing a strafe direction with a firing action and a matching target.

## Tier Configurations

| Tier | Entity Dim | Unique Dim | Backbone Hidden | Backbone Layers | Attention Heads | GRU Hidden |
|---|---:|---:|---:|---:|---:|---:|
| Micro | 8 | 16 | 32 | 1 | 2 | 32 |
| Small | 12 | 24 | 48 | 1 | 2 | 48 |
| Medium | 16 | 32 | 48 | 2 | 4 | 48 |
| Large | 16 | 32 | 64 | 2 | 4 | 64 |
| XL | 16 | 32 | 64 | 3 | 4 | 64 |

## PPO Actor-Critic

`training/methods/ppo/actor_critic.py` defines the PPO model:

- separate actor and critic encoders/backbones
- actor GRU memory
- autoregressive action heads
- value head
- auxiliary target-movement prediction head

During ONNX export, the critic-specific weights are dropped and only the policy path is retained.

## ONNX Export

Exports produce four outputs:

```text
movement_logits  [batch, 9]
combat_logits    [batch, 8]
target_logits    [batch, 5]
hidden_out       [1, batch, gru_hidden]
```

Inputs are:

```text
observation      [batch, 747]
hidden_in        [1, batch, gru_hidden]
```

The exported graph can include observation normalisation, delta encoding, structured encoding, GRU memory and policy heads.

## Distillation

Two modes are supported:

- **Standard** — teacher rollouts produce observations and logits; students match teacher logits with KL + hard-label losses.
- **Amplified** — multiple rollouts are run per scenario, the best trajectories are retained, and the policy is trained on reward-weighted winning actions.

The default distillation chain generates Micro, Small, Medium, Large and XL deployment tiers.
