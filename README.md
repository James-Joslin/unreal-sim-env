# Neural Combat AI — Curriculum Stage Breakdown

Each stage builds on the previous. The best-performing checkpoint from stage N is loaded as the starting point for stage N+1. All environments randomise arena size (±20%), obstacle count (±50%), obstacle shapes, target stats (±20%), and target behaviours per episode.

---

## How To Run

### Dependencies

```
numpy
gymnasium
torch
tensorboard
```

Optional (for visualisation):
```
pygame          # --render human
imageio[ffmpeg] # --render video
Pillow          # fallback frame export
```

### Quick Test — Visual Debugger

Run the sim with random actions to verify the environment:

```bash
# Stage 3 with pygame window:
python combat_sim.py --stage 3 --render human

# Stage 7 with specific weapon:
python combat_sim.py --stage 7 --weapon sniper --render human

# Record a video:
python combat_sim.py --stage 5 --render video --steps 800
```

CLI flags: `--stage 1-7`, `--archetype ranged|melee|tank|healer`, `--weapon scout|heavy|sniper|melee_bot|tank`, `--arena_size <UU>`, `--steps <N>`, `--render human|video`.

### Training — PPO

```bash
# Start from a behaviour cloning checkpoint (recommended):
python 03_ppo_train.py --bc_checkpoint checkpoints/bc_model.pt --stage 3

# Start from scratch:
python 03_ppo_train.py --stage 1

# Full curriculum run (auto-advances through stages):
python 03_ppo_train.py --curriculum

# Control parallelism (default 8 envs):
python 03_ppo_train.py --stage 3 --num_envs 4
```

Outputs: `checkpoints/ppo_stage{N}.pt` per stage, `checkpoints/ppo_final.pt` final model, `runs/ppo_{timestamp}/` TensorBoard logs.

### Distillation & Export

```bash
# Full pipeline: distill teacher → 5 tiers → ONNX → eval:
python 02_distill_and_export.py \
    --teacher checkpoints/ppo_stage7_best.pt \
    --output_dir models/v1

# Eval only (load existing ONNX models):
python 02_distill_and_export.py \
    --teacher checkpoints/ppo_stage7_best.pt \
    --eval_only
```

Outputs: `Combat_Micro.onnx`, `Combat_Small.onnx`, `Combat_Medium.onnx`, `Combat_Large.onnx`, `Combat_Xl.onnx`, plus `distillation_report.csv`.

### Observation Validator

```bash
# Structural check (no env needed):
python obs_vector_validator.py --check-structure

# Live range checks:
python obs_vector_validator.py --check-live --episodes 50

# Dump Python obs for diffing against C++ logs:
python obs_vector_validator.py --dump-py --episodes 20 --output validation/py_obs.csv
```

---

## Observation Vector (215 floats per frame)

The agent sees 215 normalised floats each decision tick (0.2s), frame-stacked 3 deep (645 total input). Both the Python training sim and C++ UE5 runtime produce identical vectors.

```
OBSERVATION LAYOUT (215 per frame)
    [  0.. 20]  Self State                        (21)
    [ 21.. 42]  Weapon State                      (22)      10 active + 3×4 other slots
    [ 43.. 49]  Archetype                         ( 7)
    [ 50.. 69]  Primary Target                    (20)
    [ 70..121]  Hostile Targets                   (52)  ←── 4 slots × 13 (shared hostile encoder)
    [122..157]  Allied Robots                     (36)  ←── 3 slots × 12 (shared ally encoder)
    [158..165]  Spatial Ring                      ( 8)      obstacle distance per direction
    [166..173]  Cover Height                      ( 8)      continuous obstacle height / 500
    [174..181]  Threat Sensing (Projectile 1)     ( 8)      nearest projectile + melee + dodge
    [182..190]  Navmesh Viability                 ( 9)
    [191..196]  Group Summary                     ( 6)
    [197..197]  Spawn/Leash                       ( 1)
    [198..200]  Threat Sensing (Projectile 2)     ( 3)      dist, dirX, dirY
    [201..203]  Threat Sensing (Projectile 3)     ( 3)      dist, dirX, dirY
    [204..204]  Incoming Threat Count             ( 1)      knowing when to dodge vs fight
    [205..208]  Can Hit Target Per Weapon         ( 4)      per weapon slot target availability
    [209..209]  Total Ammo Fraction               ( 1)      ammo conservation state
    [210..210]  Targets Killed Fraction           ( 1)      kill urgency tracker
    [211..214]  Arc Clearance Per Weapon          ( 4)      MaxArcableObstacleHeight / 3000
```

Cover Height [166-173] and Arc Clearance [211-214] work as a pair: the model compares obstacle height in each direction against each weapon's maximum arc clearance to decide which weapons can fire over which cover, and where to position for best engagement options.

---

## Weapon Presets

Five weapon loadouts are available. Stages 1-4 use a fixed preset. Stages 5-7 randomise from a pool each episode.

| Preset | Primary | Secondary | Melee |
|---|---|---|---|
| `melee_bot` | Sidearm (8 dmg, 800 range, 10 ammo, 0.4s CD, proj 3000) | — | 35 dmg, 200 range, 0.6s CD |
| `scout` | Laser (8 dmg, 1200 range, 20 ammo, 0.2s CD, proj 4500) | — | 15 dmg, 150 range, 0.8s CD |
| `heavy` | Cannon (35 dmg, 2000 range, 6 ammo, 1.0s CD, 0.5s wind-up, proj 2000) | Missiles (25 dmg, 1800 range, 4 ammo, 1.5s CD, arc 400 UU, proj 1200) | 40 dmg, 250 range, 1.5s CD |
| `sniper` | Railgun (80 dmg, 3000 range, 1 ammo, 2.0s CD, 1.0s wind-up, proj 6000) | Sidearm (10 dmg, 1000 range, 12 ammo, 0.3s CD, proj 3500) | 10 dmg, 150 range, 1.0s CD |
| `tank` | Gatling (5 dmg, 1500 range, 100 ammo, 0.08s CD, proj 4000) | — | 30 dmg, 250 range, 1.2s CD |

All ranged weapons fire projectiles (not hitscan). Faster projectile speed means harder to dodge but still possible at range.

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
| Engagement distance | 800 UU |
| Allies | None |
| Training budget | 200K timesteps |

### Targets
| Count | Roles | Behaviours | Speed | HP / Def | Attack |
|---|---|---|---|---|---|
| 1 | Ranged | Passive (stationary) | 0 | 100 / 10 (±20%) | None (stationary targets don't fight in stage 1-2) |

### Active Rewards
- Damage dealt (+0.15 per 1% target HP)
- Kill target (+35.0)
- Take damage (-0.015 per 1% own HP)
- Die (-10.0) + Episode loss (-5.0)
- Alive per step (+0.005) — positive in stage 1 to encourage early exploration
- Episode win (+50.0 with speed bonus up to ~75)
- Episode timeout (-8.0)
- Damage inactivity (-0.05/step after idle threshold)
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
| Agent weapon | `scout` — laser (8 dmg, 1200 range, 20 ammo, 0.2s CD, projectile 4500 UU/s) + melee (15 dmg) |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 |
| Engagement distance | 1200 UU |
| Allies | None |
| Training budget | 300K timesteps |

### Targets
| Count | Roles | Behaviours | Speed | HP / Def | Attack |
|---|---|---|---|---|---|
| 1 | Ranged | Passive (stationary) | 0 | 150 / 15 (±20%) | Passive — no damage dealt to agent |

### Active Rewards (new at this stage)
All of Stage 1, with adjustments:
- Alive per step set to 0.0 (neutral — not farming risk, not punishing)
- **Ammo management:** Reload behind cover (+0.1, lighter — no cover in stage 2 arena), switch to loaded weapon (+0.2), wasted shot (-0.0001), all empty (-0.1), fire hit (+0.15)

### What success looks like
Agent moves into laser range (400-900 UU optimal), fires 20 shots, reloads, fires again. Learns fire→reload cycle and range positioning. Target takes ~14 shots to kill (10.8 effective dmg vs 150 HP / 15 def).

---

## Stage 3 — Moving Targets, Cover, Flanking

**What the agent learns:** Track a moving target, use cover, kite, avoid incoming damage, flank.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~2500 UU (±20% per episode) |
| Obstacles | ~3 (±50%), mix of pillars/walls/L-shapes/cover/buildings |
| Agent weapon | `scout` — laser |
| Agent HP / Def | 120 / 20 |
| Max steps | 1000 (200 seconds) |
| Engagement distance | 1500 UU |
| Allies | None |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 2 | 1-2 (randomised) | ~60% ranged / ~40% melee | Aggressive, kiting, passive | 300 UU/s (0.6×) | 50 / 20 (±20%) |

Target HP is intentionally low (50) so the scout laser can kill before the agent dies — the learning objective is tracking and positioning, not sustained DPS.

### Target Attacks
- **Ranged targets:** 15-22 dmg, 1000-1500 range, 0.8-1.4s CD, projectiles (2500-4500 UU/s). Can miss if agent is moving fast.
- **Melee targets:** 28-40 dmg, 180-250 range, 0.6-1.0s CD. Charges at agent at 520-600 UU/s. Spawns closer (40-80% of engagement distance). Always aggressive.
- **Facing gating:** Targets can only shoot in their ~140° front arc. Agent behind target = target can't shoot back.

### Active Rewards (new at this stage)
All of Stages 1-2 plus:
- Alive per step now **negative** (-0.02/step) — survival costs, only dealing damage pays
- **Anti-degenerate:** Idle penalty (-0.01/step), spinning penalty (-0.03), camping penalty (-0.03/step), wall hugging (-0.02/step), corner penalty (-0.04)
- **Flanking:** Behind target (+0.008/step), at target's side (+0.003/step), fire from behind (+0.06/shot), fire from side (+0.03/shot), used cover to flank (+0.08), lost flanking position (-0.02)
- **Aggression:** Passive in range (-0.08/step) — per-step penalty for not firing when you could
- **Movement:** Mobile fire bonus (+0.02/shot), strafe fire bonus (+0.04/shot)
- **Multi-target:** Retarget urgency (-0.06/step when selected target dead but others alive), target low HP bonus (+3.0), surviving target penalty (-8.0 per living target at episode end)

### What success looks like
Agent strafes to stay in laser range, uses obstacles to break target LOS, fires while moving. Against melee targets, agent kites backward while shooting. Starts discovering that circling behind obstacles and emerging behind the target gives free shots.

### DPS Math
```
Agent deals:   (8+5) × 100/(20+100) = 10.8 per hit, 1 hit/0.2s = 54 DPS max
Target takes:  50 HP / 54 DPS ≈ 0.9s to kill (if all hits land)
Agent receives: ~7.5 effective DPS per ranged target (with movement miss + facing miss)
Agent survives: 120 HP / 7.5 ≈ 16s per target
Time budget:   200s — generous, but timeout penalty and alive_per_step push for speed
```

---

## Stage 4 — Multi-Weapon Management

**What the agent learns:** Switch between weapons based on range, manage ammo across 2 weapons, decide when to swap vs reload, use arc weapons over cover.

This is a bridging stage — same target difficulty as Stage 3 (low HP, moderate speed) but with the full heavy weapon kit. The agent learns weapon switching and arc fire against targets it can already kill, before Stage 5 increases the challenge.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~8 (dense cover for arc weapon practice) |
| Agent weapon | `heavy` — Cannon (35 dmg, 2000 range, 6 ammo, 1.0s CD, 0.5s wind-up, proj 2000) + Missiles (25 dmg, 1800 range, 4 ammo, 1.5s CD, arc 400 UU, proj 1200) + melee (40 dmg) |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 (100 seconds) |
| Engagement distance | 1500 UU |
| Allies | None |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 2 | 1-2 | ~60% ranged / ~40% melee | Aggressive, kiting, passive | 400 UU/s (0.8×) | 75 / 20 (±20%) |

### Active Rewards (new at this stage)
All of Stages 1-3 plus:
- **Weapon selection:** Fire in optimal band (+0.06/shot), fire outside optimal (-0.02/shot), swap to better range (+0.08), swap to worse range (-0.1), smart reload swap (+0.1), holding wrong weapon (-0.0025/step)
- **Arc weapons:** Arc over cover bonus (+0.03/shot when LOS blocked and arc clears), direct fire with LOS (+0.02/shot), holding arc with LOS (-0.005/step when a direct weapon has ammo)

### What success looks like
Agent uses cannon at long range (800-1600 UU optimal), switches to missiles when target is behind low cover, switches back when LOS clears. When cannon is empty, decides whether to reload cannon or switch to missiles based on current target distance. Learns the wind-up timing — starts charging cannon before the target reaches optimal range. The dense obstacle count (8) forces frequent arc vs direct decisions.

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

## Stage 5 — Archetype Behaviours & Allied Robots

**What the agent learns:** Role-appropriate combat style with an allied robot fighting alongside.

This is the first stage with allies. The agent has one AI-controlled allied robot (`num_enemies=2`, so allies = 1). The ally fights independently — the agent observes its position, health, velocity, archetype, and target but doesn't control it. The weapon preset is randomised each episode from a pool.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~3000 UU (bigger for ranged play) |
| Obstacles | ~8 |
| Agent weapon | Random from pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 200 / 25 |
| Max steps | 600 (120 seconds) |
| Engagement distance | 1500 UU |
| Allies | 1 allied robot |
| Training budget | 500K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 3 | 1-3 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours (aggressive, kiting, cover_user, passive) | 400 UU/s (0.8×) | 100 / 25 (±20%) |

### Allied Robot Observations
The ally appears in the Allied Robots section [122-157] with 12 floats per slot: occupied flag, relative position, distance, HP, ammo fraction, in-combat flag, dodging flag, archetype (scalar), velocity, and which hostile slot the ally is targeting. The agent uses this to avoid duplicating the ally's target, protect a low-HP ally, or coordinate focus fire.

### Active Rewards (new at this stage)
All of Stages 1-4 plus:
- **Ranged archetype:** Too close penalty (-0.02/step), too far penalty (-0.02/step), strafe fire (+0.03/shot in optimal range), standing still under melee threat (-0.02/step), stagger ammo (+0.05)
- **Melee archetype:** Close distance (+0.01/step), in melee range (+0.005/step), retreat penalty (-0.03), gap close bonus (+0.1)
- **Tank archetype:** Absorb damage (+0.1), body block (+0.005/step), protect low-HP ally (+0.01/step), suppression (+0.01/step), block while focused (+0.02)
- **Healer archetype:** Heal ally (+0.15), ally died while heal available (-2.0), apply buff (+0.1), maintain distance (+0.005/step), overheal (-0.05), reload heal (+0.03)

### What success looks like (Ranged archetype)
Agent adapts its behaviour to its randomly-assigned weapon loadout. With the sniper kit, it maintains maximum distance and uses the railgun. With the tank kit, it closes range and uses the gatling. The ally handles one target while the agent handles others — the agent starts learning not to waste time on targets the ally is already killing.

---

## Stage 6 — Multi-Target Coordination

**What the agent learns:** Target prioritisation, focus fire, threat assessment when outnumbered, ally awareness.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~12 (complex terrain) |
| Agent weapon | Random from pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 180 / 30 (tankier to survive multiple attackers) |
| Max steps | 700 (140 seconds) |
| Engagement distance | 1500 UU |
| Allies | 1 allied robot |
| Training budget | 1,000K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 3 | 1-3 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours | 450 UU/s (0.9×) | 150 / 25 (±20%) |

### Active Rewards (new at this stage)
All of Stages 1-5 plus:
- **Group coordination:** Target diversity bonus (+0.1 for hitting multiple targets)
- **Ally protection (all archetypes):** Protect low-HP ally (+0.015/step), fire at ally's threat (+0.04/shot), ally died nearby (-1.5)
- Ally collision penalty increased (-0.04)

### What success looks like
Agent identifies the closest melee threat and focuses it down first. Uses obstacles to break LOS from ranged targets while fighting the melee target. After the melee target is dead, repositions to engage ranged targets from cover. Switches targets mid-fight when a new melee threat closes in. Coordinates with ally — if the ally is engaging target A, the agent picks target B. If the ally is low HP and under attack, the agent fires at the ally's attacker.

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
  Ally draws aggro from 1 target, further reducing pressure.
```

---

## Stage 7 — Full Squad Combat

**What the agent learns:** Fight a full 4-member player party. Survive being heavily outnumbered. Use all learned skills together.

### Environment
| Parameter | Value |
|---|---|
| Arena | ~4000 UU (large, lots of room) |
| Obstacles | ~16 (dense, varied geometry) |
| Agent weapon | Random from pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 500 / 35 (boss-tier survivability) |
| Max steps | 800 (160 seconds) |
| Engagement distance | 2000 UU |
| Allies | 1 allied robot |
| Training budget | 2,000K timesteps |

### Targets
| Max Count | Actual per Episode | Roles | Behaviours | Speed | HP / Def |
|---|---|---|---|---|---|
| 4 | 1-4 | 40% ranged / 35% melee / 25% mixed | All 4 behaviours | 500 UU/s (full speed) | 150 / 25 (±20%) |

### What success looks like
Agent fights the full party using everything learned in stages 1-6: melee kiting, ranged fire management, weapon switching, arc fire over cover, flanking, target prioritisation, ally coordination. Against 4 targets, the agent can't face-tank all at once — it must use terrain to isolate targets, focus one down while using cover against the others, then reposition for the next.

### Realistic Episode Flow
```
0-10s:   Agent identifies party composition. 2 melee rushing, 1 ranged kiting, 1 mixed.
         Agent retreats behind cover to break LOS from ranged targets.
         Ally engages one of the melee targets.
10-25s:  First melee target arrives. Agent fights 1v1 behind cover.
         Agent uses cannon at point-blank wind-up → kill first melee.
25-40s:  Second melee arrives. Agent kites around the building.
         Switches to missiles (arc over cover) to chip the ranged target while kiting.
         Ally finishes its target and moves to the next.
40-60s:  Second melee dead. Agent repositions to engage ranged target.
         Circles behind cover for a flanking approach.
60-80s:  Flanking successful. Agent emerges behind ranged target.
         3 free cannon shots from behind. Target dies.
80-100s: Mixed target left. Agent and ally converge.
         Mixed target switches to ranged. Agent dodges, closes, finishes with melee.
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
| **Target projectile speed** | Per target: ranged 2500-4500, melee 2000-3000, mixed 1500-2000 UU/s |
| **Target spawn positions** | Random angles and distances from agent |
| **Agent spawn position** | Random within central 60% of arena |
| **Weapon preset** | Fixed in stages 1-4, random from pool in stages 5-7 |

---

## Reward Activation Summary

| Reward Component | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Damage / Kill / Death | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Episode Win/Loss/Timeout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Optimal Range | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Damage Inactivity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Invalid Action | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ammo Management | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anti-Degenerate | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Flanking / Positioning | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Aggression (passive-in-range) | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Movement (mobile/strafe fire) | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-Target Progression | | | ✓ | ✓ | ✓ | ✓ | ✓ |
| Weapon Selection | | | | ✓ | ✓ | ✓ | ✓ |
| Arc vs Direct Fire | | | | ✓ | ✓ | ✓ | ✓ |
| Archetype-Specific | | | | | ✓ | ✓ | ✓ |
| Ally Coordination | | | | | | ✓ | ✓ |

### Reward Design Philosophy

Objective rewards (kill +35, win +50-75, loss -15) dominate — they represent >80% of a winning episode's total. Shaping rewards (flanking, positioning, ammo) are kept under 15% to prevent reward farming. Per-step survival is negative (-0.02/step) so passive play is net-negative; only dealing damage overcomes the survival cost. The retarget urgency penalty (-0.06/step) ensures the agent switches targets after a kill instead of wandering. The surviving target penalty (-8.0 per living target at episode end) creates a gradient between partial and full clears.

# Combat AI — Web Testing Tool

A browser-based testing and debugging environment for the neural combat AI. It runs a full combat simulation with ONNX model inference, letting you play against the AI agent in real time while monitoring every aspect of its decision-making.

## What It Does

The tool replicates the Python training environment in the browser, matching the C++ UE5 runtime as closely as possible. You control a player character (WASD + mouse) while the AI agent runs its ONNX policy network to decide movement, combat actions, and target selection every decision tick.

Everything the AI sees, thinks, and does is visible — observations, action probabilities, reward accumulation, and budget projections. This lets you catch reward farming exploits, observation mismatches, and behavioural issues in minutes rather than waiting for multi-million-step training runs.

---

## File Structure

```
├── App.tsx                              Main application (simulation, rendering, inference)
├── main.tsx                             React entry point
├── components/
│   ├── RewardD3Chart.tsx                Reward timeline chart (D3)
│   ├── ActionProbabilityHeatmap.tsx      Per-head action probability bars
│   ├── RewardBudgetBar.tsx              WIN/DEATH/TIMEOUT budget gauge
│   ├── BatchEpisodeRunner.tsx           Headless batch evaluation
│   └── ObservationGroupInspector.tsx    198-feature observation debugger
```

---

## Prerequisites

- **Node.js** (18+) and a React project (Vite recommended)
- **Dependencies:**
  ```bash
  npm install react react-dom d3 onnxruntime-web
  npm install -D @types/d3 typescript
  ```
- **An ONNX model** exported from the training pipeline (`Combat_Medium.onnx`, etc.)

---

## Setup

1. Copy `App.tsx`, `main.tsx`, and the `components/` folder into your `src/` directory.
2. Ensure your `index.css` has a dark background (`background: #0d1117`).
3. Start the dev server:
   ```bash
   npm run dev
   ```
4. Open the browser. The simulation starts immediately with a scripted fallback AI (no model needed for basic testing).

---

## Layout

```
┌──────────────────────────────────────────────────────────┐
│  Header: Model upload, Reset, Pause, Speed controls      │
├──────────────┬───────────────────────────────────────────┤
│              │  ⚔ Combat  │  🔧 Tools                    │
│              ├──────────────────────────────────────────┤
│  Canvas      │  Config: Weapon, Arena, Targets, Stage,  │
│  460×460     │  Tier, Sampler                           │
│              ├──────────────────┬───────────────────────┤
│              │  Player HP       │ Action Probabilities  │
│              │  AI Agent HP     │ Reward Budget gauge   │
│              │  Targets HP      │                       │
├──────────────┤                  │                       │
│  Controls    │                  │                       │
├──────────────┤                  │                       │
│  📈 Reward   │                  │                       │
│  D3 Chart    │                  │                       │
└──────────────┴──────────────────┴───────────────────────┘
```

**Two tabs:**
- **⚔ Combat** — everything you need during play (config, HP bars, action probs, reward budget)
- **🔧 Tools** — data recorder, batch runner, observation inspector

---

## Controls

| Key / Action | Effect |
|---|---|
| **WASD** | Move your player character |
| **Click** (on canvas) | Shoot toward cursor position |
| **Space** | Shoot directly at the AI agent |
| **R** | Reload your weapon |
| **Enter** | Restart (when game over) |

---

## Loading an ONNX Model

1. Click **Upload ONNX** in the header bar.
2. Select a model file (e.g. `Combat_Medium.onnx`).
3. The tool auto-detects the tier from the input dimension and configures frame stacking and decision interval accordingly.
4. Once loaded, the AI switches from scripted fallback to neural network inference.

**Tier profiles:**

| Tier | Decision Rate | Frame Stack | Input Dim |
|---|---|---|---|
| Micro | 2.5 Hz | 3 | 633 |
| Small | 3.3 Hz | 3 | 633 |
| Medium | 5 Hz | 3 | 633 |
| Large | 6.6 Hz | 3 | 633 |
| XL | 10 Hz | 3 | 633 |

---

## Configuration (Combat Tab)

The config bar at the top of the Combat tab lets you adjust the environment without restarting:

| Setting | Options | Effect |
|---|---|---|
| **Weapon** | Heavy, Scout | AI agent's weapon loadout |
| **Arena** | 2000–5000 | Arena size in Unreal Units |
| **Targets** | 0–5 | Number of hostile targets |
| **Obstacles** | 0–16 | Number of cover obstacles |
| **Stage** | 1–7 | Curriculum stage (affects reward weights) |
| **Tier** | Micro–XL | Model tier profile |
| **Sampler** | Greedy / Stochastic | Action selection method |
| **Temp** | 0.1–2.0 | Temperature for stochastic sampling |

Changing any setting resets the simulation.

---

## Components

### 📈 Reward D3 Chart (`RewardD3Chart.tsx`)

A real-time line chart under the canvas showing reward over the episode.

**Two modes** (toggle with buttons):
- **Cumulative** (blue) — total reward accumulated so far. The final value is what PPO optimises.
- **Instant** (green) — per-step reward. Shows spikes (kills, damage) and dips (damage taken, penalties).

**Hover** over any point to see a tooltip with the exact step, action taken, reward value, and a breakdown of every contributing reward component (damage dealt, kill bonus, flanking, engagement, etc.).

**What to look for:**
- Cumulative climbing steadily = healthy episode
- Cumulative climbing without kills = possible farming
- Instant reward spikes without corresponding HP changes = phantom rewards

---

### 🎯 Action Probability Heatmap (`ActionProbabilityHeatmap.tsx`)

Shows the softmax probability distribution for all three action heads, updated every decision tick.

**Three sections:**
- **Movement** (9 actions) — Hold, Forward, Forward-Right, Right, etc.
- **Combat** (7 actions) — None, Fire, Reload, Switch Weapon 0/1, Melee, Block
- **Target** (5 actions) — Target 0–3, No Target

Each action shows a horizontal bar proportional to its probability. The chosen action is highlighted with a `◄` marker and glow. Masked (unavailable) actions are greyed out with dashes.

**Total entropy** is shown top-right — lower entropy = more confident decisions.

**What to look for:**
- Target head cycling rapidly between slots with high confidence while combat stays on "None" → target-switching exploit
- Movement head always selecting "Hold" → agent not engaging
- Combat head confident on "Fire" when no target in range → wasted shots
- Entropy near maximum → policy hasn't learned, still random

---

### 📊 Reward Budget Bar (`RewardBudgetBar.tsx`)

A live gauge comparing the episode's actual cumulative reward against theoretical budgets for three outcomes.

**The gauge:** A horizontal bar with three markers:
- 🔴 Red line = Timeout budget (worst outcome: no kills, full length penalty)
- 🟠 Orange line = Death budget (partial: 1 kill then die)
- 🟢 Green line = Win budget (best: all kills, speed bonus)
- 🔵 Blue marker = current cumulative reward (moves as the episode progresses)

**Trajectory prediction** (top-right): Based on the current reward rate projected to episode end — shows "→ Win track", "→ Fighting", "→ Struggling", or "→ Uncertain".

**Farming warning:** If the current reward exceeds 1.5× the Win budget without enough kills, a red warning appears: "⚠️ Reward exceeds WIN budget by X% with Y/Z kills — possible farming."

**Budget values** are computed from the current reward weights (`RW` in App.tsx) and need to stay synced with `reward.py`.

**What to look for:**
- Blue marker past the green line with 0 kills → reward farming (reward signal is broken)
- Blue marker between red and orange → agent is losing but fighting
- Blue marker near green → agent is on track to win

---

### 🏃 Batch Episode Runner (`BatchEpisodeRunner.tsx`)

Runs N episodes headlessly (no rendering) using the loaded ONNX model and displays aggregate statistics. Found in the **🔧 Tools** tab.

**Controls:**
- Select batch size: 10, 25, 50, or 100 episodes
- Click **▶ Run N Episodes**
- Results appear after all episodes complete

**Results shown:**
- **Win Rate** — percentage of episodes where all hostiles were eliminated
- **Avg Reward** — mean cumulative reward ± standard deviation
- **Avg Kills** — mean kills per episode
- **Avg Length** — mean episode length in steps
- **Reward by outcome** — separate averages for wins vs losses
- **Reward distribution histogram** — 20-bucket histogram showing the spread

**Misalignment warning:** If losses are more rewarding than wins on average, a red warning appears: "⚠️ Losses more rewarding than wins — reward signal is misaligned."

**When to use:**
- After changing reward weights — verify wins are still the highest-reward outcome
- After loading a new model — quick performance check before committing to training
- Comparing model versions — run the same batch size on each

---

### 🔬 Observation Group Inspector (`ObservationGroupInspector.tsx`)

An expandable accordion showing all 211 observation features grouped by semantic category. Found in the **🔧 Tools** tab.

**Groups:**

| Group | Indices | Features |
|---|---|---|
| Self State | 0–20 | HP, defence, speed, status effects, velocity |
| Weapon State | 21–42 | Active weapon ammo/range/cooldown, alt weapons |
| Archetype | 43–49 | Archetype encoding, optimal range, ammo flags |
| Primary Target | 50–69 | Position, HP, LOS, facing, velocity, cover |
| Hostile 0–3 | 70–121 | 4 hostile slots × 13 features each |
| Ally 0–2 | 122–157 | 3 ally slots × 12 features each |
| Spatial Ring | 158–165 | 8-direction obstacle distances |
| Cover Assessment | 166–173 | 8-direction low-cover detection |
| Threat Sensing | 174–181 | Nearest projectile, melee threat, dodge |
| Navmesh | 182–190 | 9-direction pathfinding viability |
| Group Summary | 191–196 | Alive counts, HP averages, outnumbered |
| Spawn/Leash | 197 | Distance to spawn point |
| Extended Threat | 198–204 | 2nd/3rd projectiles, threat count |
| Weapon Can-Hit | 205–208 | Per-slot can-hit-target flags |
| Ammo/Kills | 209–210 | Total ammo fraction, kills fraction |

**Features:**
- Click any group header to expand/collapse
- Each feature shows its current value with a semantic label
- **Change highlighting:** values that changed since last tick are coloured green (increased) or red (decreased), with a fade-out after 300ms
- Each group header shows a **change count badge** (e.g. "3Δ") so you can spot activity without expanding

**What to look for:**
- Primary Target HP dropping each tick = agent is hitting
- Hostile slot alive flags flipping to 0 = kills happening
- Weapon Can-Hit flags all at 0 = agent can't hit anything (wrong position or all weapons empty)
- Target values jumping when agent hasn't fired = target-switch phantom (observation bug)
- Ally target slot changing = ally switched targets (coordination signal)

---

## Typical Workflows

### 1. Testing a New Model

1. Load the ONNX model (Upload button)
2. Set config to match the training stage (weapon, targets, arena, stage number)
3. Play a few episodes manually — watch the action probabilities and reward chart
4. Switch to 🔧 Tools → run a 50-episode batch for aggregate stats
5. Check: Is win rate reasonable? Are losses more rewarding than wins?

### 2. Debugging Reward Farming

1. Play an episode and watch the Reward Budget gauge
2. If the blue marker passes the green WIN line with 0 kills → farming detected
3. Switch to Instant mode on the D3 chart — find the spikes
4. Hover over spikes to see which reward component is inflated
5. Cross-reference with the Observation Inspector — are target HP values changing without fire actions?

### 3. Validating a Reward Change

1. Update the `RW` weights in App.tsx to match your new `reward.py`
2. Run a 50-episode batch on the current model
3. Check: Win reward > Loss reward? Farming warning absent?
4. Play manually to verify the D3 chart shows the expected incentive structure

### 4. Checking Observation Parity

1. Load a model and start a game
2. Open the Observation Inspector (🔧 Tools)
3. Expand each group and verify values are sensible
4. Cross-reference with the Python `_build_observation` and C++ `GatherObservation`
5. Any group that's all zeros when it shouldn't be → missing feature implementation

---

## Keeping In Sync

The web tool must stay synchronised with three codebases:

| What | Web Tool | Python Training | C++ Runtime |
|---|---|---|---|
| Reward weights | `RW` in App.tsx | `RewardWeights` in reward.py | N/A (reward is training-only) |
| Observation layout | `buildObservation()` in App.tsx | `_build_observation()` in combat_sim.py | `GatherObservation()` in NeuralCombatComponent.cpp |
| Action space | `MOVE_LABELS`, `COMBAT_LABELS`, `TARGET_LABELS` | `MovementAction`, `CombatAction` | Action enums in NeuralCombatTypes.h |
| OBS_SIZE | `OBS_SIZE` constant | `OBS_SIZE` in combat_sim.py | `ObservationSize` in NeuralCombatTypes.h |

When any of these change in Python or C++, update the corresponding value in the web tool. The Observation Inspector and Reward Budget are only useful if they reflect the actual training configuration.