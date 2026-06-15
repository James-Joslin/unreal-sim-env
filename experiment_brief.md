# Experiment Ideas — Combat AI

Two things to try with the neural combat system.

---

## 1. Mask empty entity slots in attention

Right now the cross-attention treats every entity slot equally — even the empty ones. If there's only 2 enemies alive, the other 2 hostile slots are zeros, but attention still computes weights for them. The model eventually figures out to ignore them (there's an `occupied` flag at index 0 of each slot), but it has to learn that from scratch and some attention weight always leaks through.

Each entity slot already has that occupied flag in the observation. Pull it out, turn it into a boolean mask, and pass it into the attention layer to zero out empty slots before softmax. No new model inputs, no C++ changes — it all happens inside the forward pass from data that's already there.

In `StructuredEncoder.forward()`, extract masks from the observation and pass them through:

```python
# Hostile presence: occupied flag is field 0 of each 17-float slot
hostile_mask = hostile_feats[:, :, 0] > 0.5   # [batch, 4]
hostile_attended = self.hostile_attn(unique_emb, hostile_embs, hostile_mask)

# Ally presence: occupied flag is field 0 of each 15-float slot
ally_mask = ally_feats[:, :, 0] > 0.5         # [batch, 3]
ally_attended = self.ally_attn(unique_emb, ally_embs, ally_mask)

# Threat presence: norm_dist < 1.0 means a threat exists
threat_mask = threats_feats[:, :, 0] < 0.99   # [batch, 3]
threats_attended = self.threat_attn(unique_emb, threats_embs, threat_mask)
```

In `EntityAttention`, add the mask parameter and apply before softmax:

```python
def forward(self, query, entities, mask=None):
    # ... existing q, k, v projections ...
    attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

    if mask is not None:
        # mask shape: [batch, num_slots], True = present
        # expand to [batch, heads, 1, num_slots]
        attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e8)

    attn = F.softmax(attn, dim=-1)
    # ... rest unchanged ...
```

One thing to watch: if ALL slots in a group are empty (no threats, no allies), softmax over all -inf produces uniform noise. Handle that by zeroing the output when the whole group is masked. Hostiles always have at least one target (otherwise you're not in combat), so this mainly matters for threats and allies.

The same changes go into both `ActorCritic`'s encoder paths (actor and critic) and `CombatPolicy`. Everything traces through ONNX cleanly since it's standard tensor ops.

**How to measure:** A/B training runs on the same stage and seed. Compare convergence speed and final win rate. Logging the raw attention weights periodically would also show whether empty-slot weights actually reach zero faster with the mask.

---

## 2. Pass confidence down the autoregressive chain

The autoregressive heads embed the chosen action as a discrete index — "I picked strafe-left" — and feed that into the next head. But it throws away how sure the model was. A movement picked with 90% confidence and one picked at 34% produce the exact same embedding.

The idea: concatenate the full softmax distribution alongside the discrete embedding so the combat head can see the movement uncertainty.

```python
m_logits = self._scaled(self.move_head(actor_feat))
if masks is not None:
    m_logits = m_logits.masked_fill(~masks[0], -1e8)
m_dist = torch.distributions.Categorical(logits=m_logits)
m_act = m_dist.sample()

m_emb = self.move_embed(m_act)
m_prob = F.softmax(m_logits, dim=-1).detach()  # stop gradient

c_features = F.gelu(self.combat_proj(
    torch.cat([features, m_emb, m_prob], dim=-1)))
```

Repeat for the target head — concatenate both `m_prob` (9 floats) and `c_prob` (8 floats) alongside the embeddings:

```python
c_prob = F.softmax(c_logits, dim=-1).detach()

t_features = F.gelu(self.target_proj(
    torch.cat([features, m_emb, c_emb, m_prob, c_prob], dim=-1)))
```

The `combat_proj` input dimension grows by 9, the `target_proj` input grows by 17. Small parameter increase.

**The `.detach()` is critical.** Without it, the combat head's loss backpropagates through the softmax into the movement head's logits. This creates a second gradient signal on the movement head that has nothing to do with movement quality — the movement logits get pushed around to make the combat head's job easier, which corrupts the movement policy. Detach makes the probability vector read-only: the combat head can observe the uncertainty but can't reshape it.

**The counterargument worth acknowledging:** the features tensor already encodes everything the model knows, including the uncertainty that produces those logits. The combat projection could theoretically reconstruct the distribution internally. The bet is that providing it explicitly is easier than expecting a 96-dim network to recompute it, especially at small model sizes.

**ONNX consideration:** during ONNX inference, `forward()` uses unmasked logits for autoregressive conditioning. The softmax distributions will differ slightly from training (where masks were applied). This gap already exists and is tolerated in the current architecture, but passing confidence makes the model more likely to rely on the distribution, which widens the gap. If this becomes an issue, the fix is to add mask tensors as ONNX inputs so the internal conditioning matches training.

**How to measure:** compare against baseline on the same stage. Track whether high movement uncertainty correlates with more conservative combat choices (more NONE/BLOCK when unsure about positioning). Stages 4-5 are the best test — that's where movement confidence genuinely matters for combat decisions.

---

## 3. Gated residual bypass around the GRU

The GRU sits between the backbone and the action heads. Every decision the model makes passes through it, which means the GRU can smooth out or overwrite immediate spatial features with its accumulated temporal state. Most of the time that's what you want — temporal context improves decisions. But there are moments where the immediate observation is more important than history (a new threat appearing, an ally suddenly dying), and the GRU has to learn when to let the current frame through unmodified. That's a lot to ask of a single recurrent layer.

The idea: add a learned gate that mixes the GRU output with a direct skip connection from the backbone, so the model can dynamically choose how much temporal memory vs immediate state to use.

```python
# In __init__:
self.gru_skip = layer_init(nn.Linear(backbone_hidden, gru_hidden))
self.mix_gate = layer_init(nn.Linear(backbone_hidden + gru_hidden, gru_hidden))

# In forward / _actor_features (after the GRU):
gru_in = backbone_out.unsqueeze(1)
gru_out, hidden_out = self.gru(gru_in, hidden)
gru_features = gru_out.squeeze(1)

skip_features = self.gru_skip(backbone_out)
gate = torch.sigmoid(self.mix_gate(
    torch.cat([backbone_out, gru_features], dim=-1)))
features = gate * gru_features + (1.0 - gate) * skip_features
```

Two extra linear layers, minimal parameter increase. ONNX-safe — it's just linear projections, sigmoid, and element-wise ops. No changes to C++ hidden state management since the GRU still produces `hidden_out` the same way; the gate mixing happens downstream.

The gradient flow benefit is real: during backprop, gradients can reach the backbone through the skip path even if the GRU's internal gradients weaken. This matters less at 100-step episodes (GRU handles that length fine) but becomes more relevant if episodes get longer in later curriculum stages.

Run this AFTER validating the plain GRU. If the fixed GRU already performs well, the gate may not add much. If it plateaus in S5/S6 despite the schedule fixes, the residual bypass is a clean next step.

**How to measure:** same A/B setup as the other experiments. Additionally, logging the gate values would be informative — if the gate learns to consistently favour one side (always ~0.9 or always ~0.1), the bypass isn't contributing and can be removed.

---

## Running the experiments properly

RL training has high variance. A single run that hits 65% win rate vs another at 60% tells you almost nothing — that difference could just be seed luck. To actually know whether a change helped, you need multiple runs and a way to quantify uncertainty.

### Seeds and number of runs

Run each variant (baseline, experiment) with **5 different random seeds**. Same stage, same total timesteps, same eval schedule — only the seed and the architectural change differ. 5 runs is the practical minimum where bootstrap confidence intervals start to tighten enough to be useful. 3 runs is tempting but the intervals are wide enough that you'll regularly see overlap even when there's a real difference.

Pick seeds deliberately rather than sequentially. Something like `[42, 137, 256, 512, 1024]` — spread out so you don't accidentally hit a correlated cluster. Use the same seed set for every variant so the comparison is paired.

Each run already evaluates on 50 seeded episodes (fixed eval seeds 42-91). That gives you 50 binary win/loss outcomes per eval checkpoint per run.

### What to log

For each run, record the full eval curve, not just the final number. The shape of the curve matters — an experiment that converges faster but plateaus at the same level is still useful (less compute for the same result). Log at every eval checkpoint:

- Win rate (primary metric)
- Mean episode reward
- Mean kills
- Entropy
- Approx KL

### Bootstrapping win rate

Win rate from 50 eval episodes is a proportion, and 50 is small enough that the raw number has wide uncertainty. Use bootstrap resampling to get confidence intervals at each eval checkpoint.

For a single eval (50 episodes, k wins):

```python
import numpy as np

def bootstrap_win_rate(wins, n_episodes, n_bootstrap=10000, ci=95):
    """Bootstrap CI for win rate from a single eval."""
    outcomes = np.array([1.0] * wins + [0.0] * (n_episodes - wins))
    boot_means = np.array([
        np.random.choice(outcomes, size=n_episodes, replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return boot_means.mean(), lower, upper
```

With 50 episodes and a true win rate of 60%, the 95% bootstrap CI is roughly ±13 percentage points. That's wide. Two things help: increasing eval episodes (100 instead of 50 tightens it to ±9pp), and aggregating across runs.

### Comparing variants across runs

Once you have 5 runs per variant, aggregate at each eval checkpoint. For each checkpoint step:

1. Collect the 5 win rates (one per seed) for variant A and variant B.
2. Bootstrap the difference in means.

```python
def bootstrap_difference(rates_a, rates_b, n_bootstrap=10000, ci=95):
    """Bootstrap CI for the difference between two variants.
    rates_a, rates_b: arrays of per-run win rates (e.g. 5 values each)."""
    diffs = []
    for _ in range(n_bootstrap):
        a = np.random.choice(rates_a, size=len(rates_a), replace=True)
        b = np.random.choice(rates_b, size=len(rates_b), replace=True)
        diffs.append(a.mean() - b.mean())
    diffs = np.array(diffs)
    lower = np.percentile(diffs, (100 - ci) / 2)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2)
    return np.mean(diffs), lower, upper
```

If the 95% CI for the difference excludes zero, you have reasonable evidence the change matters. If it includes zero, the difference might be noise.

### What counts as a meaningful result

For this project, the practical bar is:

- **Win rate improvement ≥ 5pp** with CI excluding zero → adopt the change
- **Win rate within ±3pp** but **faster convergence** (reaches the same level in fewer steps) → adopt if compute cost matters
- **Win rate within ±3pp** and same convergence speed → the change doesn't help, drop it
- **Win rate worse by ≥ 5pp** → the change hurts, drop it

Don't chase 1-2pp differences. With 5 seeds and 50 eval episodes, that's well within noise.

### Curve comparison over final-number comparison

Plot the eval win rate curves (mean ± 1 std across 5 seeds) for baseline and experiment on the same axes. This is more informative than comparing final numbers because:

- An experiment might help early training but hurt late training (or vice versa)
- Convergence speed differences are invisible in a final-number comparison
- You can see if the variance across seeds is tighter with the experiment (more stable training is valuable even at the same mean)

### Practical schedule

| Run | Seeds | Eval episodes | Eval frequency | Total runs |
|---|---|---|---|---|
| Baseline | 5 | 50 | Every 20K steps | 5 |
| + Entity masking | 5 | 50 | Every 20K steps | 5 |
| + Logit feedback | 5 | 50 | Every 20K steps | 5 |
| + GRU residual gate | 5 | 50 | Every 20K steps | 5 |

That's 20 training runs total. Run baseline and entity masking in parallel (independent changes), then logit feedback and GRU gate after deciding which earlier changes to keep.
