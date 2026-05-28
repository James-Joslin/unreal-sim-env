# Neural Combat AI — Curriculum Stage Breakdown

Each stage builds on the previous. The best-performing checkpoint from stage N is loaded as the starting point for stage N+1. All environments randomise arena size (±20%), obstacle count (±50%), obstacle shapes, target stats (±20%), and target behaviours per episode.

---

## Stage 1 — Melee Basics

**What the agent learns:** Close distance to a target and hit it with melee attacks.

### Environment
| Parameter | Value |
|---|---|
| Arena | 1500 UU (small, tight) |
| Obstacles | 0 (empty arena) |
| Agent weapon | `melee_bot` — weak sidearm (8 dmg, 800 range) + strong melee (35 dmg, 200 range, 0.6s CD) |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 (100 seconds) |
| Training budget | 200K timesteps |

### Targets
| Count | Roles | Behaviours | Speed | HP / Def | Attack |
|---|---|---|---|---|---|
| 1 | Ranged | Passive (stationary) | 0 | 100 / 10 | None (stationary targets don't fight in stage 1-2) |

### Active Rewards
- Damage dealt (+0.15 per 1% target HP)
- Kill target (+8.0)
- Take damage (-0.04 per 1% own HP)
- Die (-8.0)
- Alive per step (+0.001)
- Optimal range (+0.01/step)
- Episode win (+15.0 with speed bonus)
- Episode timeout (-3.0)
- Damage inactivity (-0.01/step after 10 idle steps)
- Invalid action (-0.1)

### What success looks like
Agent walks toward the target and melees it to death in ~3 seconds. Learns the concept of "close distance → attack → damage → reward."

---

## Stage 2 — Ranged Fire & Reload

**What the agent learns:** Shoot a ranged weapon, manage ammo, reload, and understand range.

### Environment
| Parameter | Value |
|---|---|
| Arena | 2000 UU |
| Obstacles | 0 |
| Agent weapon | `scout` — hitscan laser (8 dmg, 1200 range, 20 ammo, 0.2s CD) + melee (15 dmg) |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 |
| Training budget | 300K timesteps |

### Targets
| Count | Roles | Behaviours | Speed | HP / Def | Attack |
|---|---|---|---|---|---|
| 1 | Ranged | Passive (stationary) | 0 | 150 / 15 | Passive — no damage dealt to agent |

### Active Rewards (new at this stage)
All of Stage 1 plus:
- **Ammo management:** Reload behind cover (+0.3), reload in open (-0.1), switch to loaded weapon (+0.2), wasted shot (-0.05), all empty (-0.1), fire hit (+0.02)

### What success looks like
Agent moves into laser range (400-900 UU optimal), fires 20 shots, reloads, fires again. Learns fire→reload cycle and range positioning. Target takes ~11 shots to kill (10.8 dmg each vs 150 HP / 15 def).

---

## Stage 3 — Moving Targets, Cover, Flanking

**What the agent learns:** Track a moving target, use cover, kite, avoid incoming damage, flank.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~2500 UU (±20% per episode) |
| Obstacles | ~3 (±50%), mix of pillars/walls/L-shapes/cover/buildings |
| Agent weapon | `scout` — hitscan laser |
| Agent HP / Def | 120 / 20 |
| Max steps | 600 (120 seconds) |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 2 | 1-2 (randomised) | 60% ranged / 40% melee | Aggressive, kiting, passive | 300 UU/s (0.6×) | 120 / 20 |

### Target Attacks
- **Ranged targets:** 15-22 dmg, 1000-1500 range, 0.8-1.4s CD, projectiles (1500-2500 UU/s or hitscan). Can miss if agent is moving fast.
- **Melee targets:** 28-40 dmg, 180-250 range, 0.6-1.0s CD. Charges at agent at 520-600 UU/s. Spawns closer (40-80% of engagement distance).
- **Facing gating:** Targets can only shoot in their ~140° front arc. Agent behind target = target can't shoot back.

### Active Rewards (new at this stage)
All of Stages 1-2 plus:
- **Anti-degenerate:** Idle penalty (-0.02/step after 3+ idle steps), spinning penalty (-0.05), camping penalty (-0.01/step after 10+ stationary steps)
- **Flanking:** Behind target (+0.04/step), at target's side (+0.02/step), fire from behind (+0.08/shot), fire from side (+0.04/shot), used cover to flank (+0.10), lost flanking position (-0.03)

### What success looks like
Agent strafes to stay in laser range, uses obstacles to break target LOS, fires while moving. Against melee targets, agent kites backward while shooting. Starts discovering that circling behind obstacles and emerging behind the target gives free shots.

### DPS Math
```
Agent deals:   (8+5) × 100/(20+100) = 10.8 per hit, 1 hit/0.2s = 54 DPS max
Target takes:  120 HP / 54 DPS ≈ 2.2s to kill (if all hits land)
Agent receives: ~11.6 effective DPS per ranged target (with movement miss)
Agent survives: 120 HP / 11.6 ≈ 10.3s per target
Time budget:   120s — plenty of time, but timeout penalty pushes for speed
```

---

## Stage 4 — Multi-Weapon Management

**What the agent learns:** Switch between weapons based on range, manage ammo across 2 weapons, decide when to swap vs reload.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~2500 UU |
| Obstacles | ~4 |
| Agent weapon | `heavy` — Cannon (35 dmg, 2000 range, 6 ammo, 1.0s CD, 0.5s wind-up, slow projectile) + Missiles (25 dmg, 1800 range, 4 ammo, 1.5s CD, arc, slow projectile) + melee (40 dmg) |
| Agent HP / Def | 130 / 25 |
| Max steps | 700 (140 seconds) |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 2 | 1-2 | 60% ranged / 40% melee | Aggressive, kiting, passive | 350 UU/s (0.7×) | 150 / 25 |

### Active Rewards (new at this stage)
All of Stages 1-3 plus:
- **Weapon selection:** Fire in optimal band (+0.03/shot), fire outside optimal (-0.02/shot), swap to better range (+0.15), swap to worse range (-0.1), smart reload swap (+0.1), holding wrong weapon (-0.01/step)

### What success looks like
Agent uses cannon at long range (800-1600 UU optimal), switches to missiles when target is behind low cover, switches back when LOS clears. When cannon is empty, decides whether to reload cannon or switch to missiles based on current target distance. Learns the wind-up timing — starts charging cannon before the target reaches optimal range.

### Key Decision
```
Target at 2000 UU, cannon empty, missiles loaded:
  Option A: Reload cannon (3s), then fire at optimal range — good if target stays far
  Option B: Switch to missiles (0.3s delay), fire now — immediate but fewer shots
  Option C: Close distance, switch to melee — only if target is melee and closing

The reward function now gives the agent enough information (other weapon ranges
in the observation) and incentive (weapon selection rewards) to learn this decision.
```

---

## Stage 5 — Archetype-Specific Behaviours

**What the agent learns:** Role-appropriate combat style. A Ranged archetype kites and maintains distance. A Melee archetype rushes in. A Tank holds ground. A Healer stays back.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~3000 UU (bigger for ranged play) |
| Obstacles | ~5 |
| Agent weapon | `heavy` (same as stage 4) |
| Agent HP / Def | 150 / 25 |
| Max steps | 800 (160 seconds) |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 3 | 1-3 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours (aggressive, kiting, cover_user, passive) | 400 UU/s (0.8×) | 150 / 25 |

### Active Rewards (new at this stage)
All of Stages 1-4 plus:
- **Ranged archetype:** Too close penalty (-0.05/step), too far penalty (-0.02/step), kite success (+0.05), standing still under melee threat (-0.03/step), stagger ammo (+0.05)
- **Melee archetype:** Close distance (+0.02/step), in melee range (+0.01/step), retreat penalty (-0.05), gap close bonus (+0.1)
- **Tank archetype:** Absorb damage (+0.1), body block (+0.01/step), protect low-HP ally (+0.02/step), suppression (+0.01/step)
- **Healer archetype:** Heal ally (+0.15), ally died while heal available (-2.0), maintain distance (+0.01/step)

### What success looks like (Ranged archetype)
Agent maintains 800-1600 UU distance, backs away when melee targets rush in, strafes against ranged targets, uses cover when reloading. Against mixed parties, prioritises the melee target closing in (immediate threat) then handles ranged fighters from distance.

---

## Stage 6 — Multi-Target Coordination

**What the agent learns:** Target prioritisation, focus fire, threat assessment when outnumbered.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~5 |
| Agent weapon | `heavy` |
| Agent HP / Def | 180 / 30 (tankier to survive multiple attackers) |
| Max steps | 1000 (200 seconds) |
| Training budget | 1,000K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 3 | 1-3 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours | 450 UU/s (0.9×) | 150 / 25 |

### Active Rewards (new at this stage)
All of Stages 1-5 plus:
- **Group coordination:** Target diversity bonus (+0.1 for hitting multiple targets)

### What success looks like
Agent identifies the closest melee threat and focuses it down first. Uses an obstacle to break LOS from ranged targets while fighting the melee target. After the melee target is dead, repositions to engage ranged targets from cover. Switches targets mid-fight when a new melee threat closes in.

### Survival Math
```
Against 3 targets (worst case: all attacking simultaneously):
  Agent HP: 180, Defence: 30
  Effective DPS per target: ~9 (with movement miss + facing miss + cover)
  3 targets: ~27 effective DPS
  Agent survives: 180 / 27 ≈ 6.7 seconds under max pressure

  But agent using cover blocks 1-2 targets' LOS:
  1 target attacking: ~9 DPS → survives 20s
  Time to kill one target: 150 HP / 30 DPS (cannon) ≈ 5s

  Strategy: use cover to fight 1v1, sequentially eliminate targets.
```

---

## Stage 7 — Full Squad Combat

**What the agent learns:** Fight a full 4-member player party. Survive being heavily outnumbered. Use all learned skills together.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~4000 UU (large, lots of room) |
| Obstacles | ~7 (varied geometry) |
| Agent weapon | `heavy` |
| Agent HP / Def | 250 / 35 (very tanky) |
| Max steps | 1200 (240 seconds) |
| Training budget | 2,000K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 4 | 1-4 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours | 500 UU/s (full speed) | 150 / 25 |

### What success looks like
Agent fights the full party using everything learned in stages 1-6: melee kiting, ranged fire management, weapon switching, cover usage, flanking, target prioritisation. Against 4 targets, the agent can't face-tank — it must use terrain to isolate targets, focus one down while using cover against the others, then reposition for the next.

### Realistic Episode Flow
```
0-10s:   Agent identifies party composition. 2 melee rushing, 1 ranged kiting, 1 mixed.
         Agent retreats behind a building to break LOS from ranged targets.
10-25s:  First melee target arrives. Agent fights 1v1 behind cover.
         Agent uses cannon at point-blank wind-up → kill first melee.
25-40s:  Second melee arrives. Agent kites around the building.
         Switches to missiles (arc over cover) to chip the ranged target while kiting.
40-60s:  Second melee dead. Agent repositions to engage ranged target.
         Circles behind cover for a flanking approach.
60-80s:  Flanking successful. Agent emerges behind ranged target.
         3 free cannon shots from behind. Target dies.
80-100s: Mixed target left. Agent closes distance for melee exchange.
         Mixed target switches to ranged. Agent dodges, closes, melees to finish.
```

---

## Per-Episode Randomisation (All Stages)

Every single episode is unique. The agent never fights the same battle twice:

| Element | How It Varies |
|---|---|
| **Arena size** | ±20% of the stage's base value |
| **Obstacle count** | ±50% of the stage's base value |
| **Obstacle types** | Random mix of pillar, wall, L-shape, cover, building |
| **Obstacle positions** | Random placement within arena bounds |
| **Target count** | 1 to stage max (stages 3+) |
| **Target roles** | Random melee/ranged/mixed per the stage's distribution |
| **Target behaviours** | Random aggressive/kiting/cover_user/passive |
| **Target stats** | HP, defence, damage, range, cooldown ±20% |
| **Target projectile speed** | Random per target (0/1500/1800/2000/2500 UU/s) |
| **Target spawn positions** | Random angles and distances from agent |
| **Agent spawn position** | Random within central 60% of arena |

---

## Reward Activation Summary

| Reward Component | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Damage / Kill / Death / Alive | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Optimal Range | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Episode Win/Loss/Timeout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Damage Inactivity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Invalid Action | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ammo Management | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anti-Degenerate | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Flanking | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Weapon Selection | | | | ✓ | ✓ | ✓ | ✓ |
| Archetype-Specific | | | | | ✓ | ✓ | ✓ |
| Group Coordination | | | | | | ✓ | ✓ |
