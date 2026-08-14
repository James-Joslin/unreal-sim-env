# Curriculum Stages

Training uses seven progressive stages. The best checkpoint from stage N is used to initialise stage N+1.

All stages include per-episode randomisation such as arena size, obstacle layout, target stats, target roles, spawn positions and target behaviours. Team composition itself is fixed or stratified as shown below; it is never sampled uniformly from `1..N`.

## Stage Overview

| Stage | Composition | Focus | Summary |
|---:|---|---|---|
| 1 | 1v1 | Melee basics | Close distance and melee a passive target. |
| 2 | 1v1 | Ranged fire/reload | Learn range positioning, firing and reload cycles. |
| 3 | 1v1 | Moving targets/cover/flanking | Track a moving target, use cover, kite, flank and avoid degenerate behaviour. |
| 4 | 1v1 | Multi-weapon management | Switch weapons, manage ammo and use arc fire over cover. |
| 5 | 1v1 | Archetype behaviours | Learn role-specific combat without team-size variance. |
| 6 | 2 enemies vs 1 player | Basic coordination | Introduce one allied robot, focus fire and ally protection. |
| 7 | Equal teams, 1v1 through 4v4 | Stratified squad combat | Sample the four team-size buckets equally and use all learned behaviours. |

## Training Budgets

Current PPO stage configs use these default timestep budgets:

| Stage | Timesteps | Eval Episodes | Rollout Steps | Notes |
|---:|---:|---:|---:|---|
| 1 | 50,000 | 30 | 256 | Fast basic behaviour acquisition. |
| 2 | 100,000 | 30 | 256 | Fire/reload loop. |
| 3 | 1,500,000 | 50 | 512 | Moving targets and shaping-heavy learning. |
| 4 | 6,000,000 | 80 | 512 | Weapon switching and arc/direct decisions. |
| 5 | 20,000,000 | 100 | 1024 | Archetype behaviour and ally awareness. |
| 6 | 20,000,000 | 120 | 2048 | Coordination and multi-target variance. |
| 7 | 30,000,000 | 150 | 2048 | Full squad combat. |

## Per-Stage PPO Strategy

- Early stages use higher entropy and larger policy updates to encourage exploration.
- Later stages reduce entropy pressure and tighten update constraints to preserve coordination strategies.
- Later stages use larger rollouts and more evaluation episodes because multi-target outcomes have higher variance.
- Catastrophic regression reversion is enabled to protect good checkpoints when a policy update destabilises behaviour.

## Stage Details

### Stage 1 — Melee Basics

Learns the basic loop: approach target, enter melee range, attack, receive reward.

### Stage 2 — Ranged Fire and Reload

Introduces ranged weapon use, ammo, reload timing and optimal range positioning.

### Stage 3 — Moving Targets, Cover and Flanking

Adds moving targets, obstacles, cover, flanking, aggression shaping and anti-degenerate penalties. Survival becomes costly, pushing the agent to end fights rather than farm shaping rewards.

### Stage 4 — Multi-Weapon Management

Adds weapon switching and arc-vs-direct fire decisions. The agent must choose between loaded weapons, reloads, range bands and arcing over cover.

### Stage 5 — Archetype Behaviours

Introduces role-specific shaping in a fixed 1v1 encounter. Healer behaviour is
reserved for a later stage because the current action contract has no heal or
buff action.

### Stage 6 — Multi-Target Coordination

Introduces one allied robot: two allied enemies fight one player. This isolates
basic focus fire and ally-protection behaviour before larger squads.

### Stage 7 — Stratified Squad Combat

Final deployment-like stage. Episodes are split equally across seeded 1v1,
2v2, 3v3 and 4v4 buckets instead of letting smaller random encounters dominate
the training distribution. Evaluation should report each bucket separately as
well as the aggregate.
