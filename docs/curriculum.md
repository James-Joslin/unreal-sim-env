# Curriculum Stages

Training uses seven progressive stages. The best checkpoint from stage N is used to initialise stage N+1.

All stages include per-episode randomisation such as arena size, obstacle layout, target stats, target roles, spawn positions and target behaviours.

## Stage Overview

| Stage | Focus | Summary |
|---:|---|---|
| 1 | Melee basics | Close distance and melee a passive target. |
| 2 | Ranged fire/reload | Learn range positioning, firing and reload cycles. |
| 3 | Moving targets/cover/flanking | Track moving targets, use cover, kite, flank and avoid degenerate behaviour. |
| 4 | Multi-weapon management | Switch weapons, manage ammo and use arc fire over cover. |
| 5 | Archetype behaviours/allies | Learn role-specific behaviour with one allied robot. |
| 6 | Multi-target coordination | Learn focus fire, target prioritisation and ally protection. |
| 7 | Full squad combat | Fight a full mixed party using all learned behaviours. |

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

### Stage 5 — Archetype Behaviours and Allies

Introduces role-specific shaping and allied robots. The agent observes allies but does not directly control them.

### Stage 6 — Multi-Target Coordination

Focuses on target prioritisation, ally protection, focus fire and fighting while outnumbered.

### Stage 7 — Full Squad Combat

Final deployment-like stage. The agent fights mixed groups with multiple targets, weapons, cover, allies and player-pattern observations.
