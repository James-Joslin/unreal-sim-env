# Curriculum Stages

Each stage builds on the previous. The best checkpoint from stage N starts stage N+1. All environments randomise arena size (±20%), obstacle count (±50%), obstacle shapes, target stats (±20%), and target behaviours per episode. See [rewards.md](rewards.md) for full reward component values.

---

## Stage 1 — Melee Basics

**Learns:** Close distance, melee attack.

| Parameter | Value |
|---|---|
| Arena | 1500 UU |
| Obstacles | 0 |
| Agent weapon | `melee_bot` |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 (100s) |
| Targets | 1 × passive, 100 HP / 10 Def |
| Training budget | 200K |

**Success:** Agent walks to target and melees it to death. Learns "close distance → attack → reward."

**Rewards activated:** Damage, kill, death, win/loss, optimal range, inactivity, invalid action.

---

## Stage 2 — Ranged Fire & Reload

**Learns:** Shoot, manage ammo, reload, range positioning.

| Parameter | Value |
|---|---|
| Arena | 2000 UU |
| Obstacles | 0 |
| Agent weapon | `scout` |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 |
| Targets | 1 × passive, 150 HP / 15 Def |
| Training budget | 300K |

**Success:** Agent positions in laser range, fires 20 shots, reloads, fires again. Learns the fire→reload cycle.

**New rewards:** Ammo management (reload cover, wasted shot, fire hit).

---

## Stage 3 — Moving Targets, Cover, Flanking

**Learns:** Track moving targets, use cover, kite melee, flank, fire while moving.

| Parameter | Value |
|---|---|
| Arena | ~2500 UU (±20%) |
| Obstacles | ~3 (±50%) |
| Agent weapon | `scout` |
| Agent HP / Def | 120 / 20 |
| Max steps | 1000 (200s) |
| Targets | 1-2, 60% ranged / 40% melee, 300 UU/s, 50 HP / 20 Def |
| Training budget | 500K |

Target HP is intentionally low (50) — the learning objective is tracking and positioning, not sustained DPS.

**Target attacks:** Ranged 15-22 dmg, 1000-1500 range, projectiles 2500-4500 UU/s. Melee 28-40 dmg, charges at 520-600 UU/s. Facing gated (~140° front arc).

**New rewards:** Anti-degenerate, flanking, aggression (passive-in-range), movement (strafe/mobile fire), multi-target progression. Alive per step goes negative (-0.02).

**DPS math:** Agent deals ~54 max DPS, kills a 50 HP target in ~0.9s. Agent survives ~16s per ranged target. 200s budget is generous — penalties push for speed.

---

## Stage 4 — Multi-Weapon Management

**Learns:** Weapon switching, ammo management across 2+ weapons, arc fire over cover.

Bridging stage — same target difficulty as stage 3 but with the full heavy weapon kit. Dense obstacles (8) force frequent arc vs direct decisions.

| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~8 |
| Agent weapon | `heavy` (cannon + missiles + melee) |
| Agent HP / Def | 100 / 20 |
| Max steps | 500 (100s) |
| Targets | 1-2, 60% ranged / 40% melee, 400 UU/s, 75 HP / 20 Def |
| Training budget | 500K |

**New rewards:** Weapon selection (optimal band, swap quality), arc vs direct fire (arc bonus when LOS blocked, direct preference with LOS).

**Key decision:** Target at 2000 UU, cannon empty, missiles loaded — reload cannon (3s delay, optimal if target stays far), switch to missiles (0.3s delay, immediate), or close for melee?

---

## Stage 5 — Archetype Behaviours & Allies

**Learns:** Role-appropriate combat with an allied robot.

First stage with allies and weapon pool randomisation. The ally fights independently — the agent observes its position, health, target, and archetype but doesn't control it.

| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~8 |
| Agent weapon | Pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 200 / 25 |
| Max steps | 600 (120s) |
| Allies | 1 |
| Targets | 1-3, mixed roles/behaviours, 400 UU/s, 100 HP / 25 Def |
| Training budget | 500K |

**Ally observations:** 12 floats per slot [122-157] — position, distance, HP, ammo, in-combat, dodging, archetype (scalar), velocity, target hostile index.

**New rewards:** Archetype-specific shaping (ranged kiting, melee gap-close, tank body-block, healer support).

**Success:** Agent adapts to its random weapon loadout. Sniper kit → maintain distance, railgun. Tank kit → close range, gatling. Ally handles one target while agent handles others.

---

## Stage 6 — Multi-Target Coordination

**Learns:** Target prioritisation, focus fire, ally awareness when outnumbered.

| Parameter | Value |
|---|---|
| Arena | ~3000 UU |
| Obstacles | ~12 |
| Agent weapon | Pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 180 / 30 |
| Max steps | 700 (140s) |
| Allies | 1 |
| Targets | 1-3, mixed, 450 UU/s, 150 HP / 25 Def |
| Training budget | 1,000K |

**New rewards:** Group coordination (target diversity), ally protection (protect low-HP ally, fire at ally's threat, ally death penalty).

**Survival math:** Under max pressure (3 targets, ~27 effective DPS), agent survives ~6.7s. Using cover to fight 1v1: survives 20s, kills in 5s. Ally draws one target's aggro.

---

## Stage 7 — Full Squad Combat

**Learns:** Fight a full 4-player party using everything from stages 1-6.

| Parameter | Value |
|---|---|
| Arena | ~4000 UU |
| Obstacles | ~16 |
| Agent weapon | Pool: `heavy`, `scout`, `sniper`, `tank` |
| Agent HP / Def | 500 / 35 (boss-tier) |
| Max steps | 800 (160s) |
| Allies | 1 |
| Targets | 1-4, mixed, 500 UU/s (full speed), 150 HP / 25 Def |
| Training budget | 2,000K |

**Success:** Agent isolates targets using terrain, focuses one down while using cover against others, repositions, coordinates with ally. Can't face-tank 4 attackers — must use all learned skills.

---

## Per-Episode Randomisation (All Stages)

| Element | Variation |
|---|---|
| Arena size | ±20% |
| Obstacle count | ±50% |
| Obstacle types | pillar, wall, L-shape, cover, building |
| Target count | 1 to stage max |
| Target roles | melee / ranged / mixed per stage distribution |
| Target behaviours | aggressive, kiting, cover_user, passive |
| Target stats | HP, defence, damage, range, cooldown ±20% |
| Target projectile speed | ranged 2500-4500, melee 2000-3000, mixed 1500-2000 UU/s |
| Spawn positions | random within arena |
| Weapon preset | fixed stages 1-4, random pool stages 5-7 |
