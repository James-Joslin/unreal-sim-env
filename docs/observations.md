# Observation Vector

215 normalised floats per frame, frame-stacked ×3 (645 total input). Both the Python training sim and C++ UE5 runtime produce identical vectors.

## Layout

```
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
[204..204]  Incoming Threat Count             ( 1)
[205..208]  Can Hit Target Per Weapon         ( 4)      per weapon slot
[209..209]  Total Ammo Fraction               ( 1)
[210..210]  Targets Killed Fraction           ( 1)
[211..214]  Arc Clearance Per Weapon          ( 4)      MaxArcableObstacleHeight / 3000
```

## Feature Groups

The policy network splits these into four encoder groups:

**Unique features (127 floats):** `[0:70] + [158:215]` — self state, weapons, archetype, primary target, spatial ring, cover height, threat sensing, navmesh, group summary, spawn/leash, weapon capabilities, ammo, kills, and arc clearance. Processed by a dedicated linear encoder.

**Hostile entity slots (4 × 13 = 52 floats):** `[70:122]` — each slot encodes one hostile target (occupied, position, distance, HP, LOS, player-controlled, facing, priority, threat, velocity, targeting-me). Processed by a shared-weight encoder with max-pooling for permutation invariance.

**Ally entity slots (3 × 12 = 36 floats):** `[122:158]` — each slot encodes one allied robot (occupied, position, distance, HP, ammo, in-combat, dodging, archetype, velocity, target-hostile-index). Same shared-weight + max-pool pattern.

**Threat entity slots (3 × 3 = 9 floats):** Extracted from indices 174/176/177, 198-200, 201-203 — three projectile threats, each with (distance, dirX, dirY). Shared encoder + max-pool.

## Cover Height + Arc Clearance

Cover Height [166-173] and Arc Clearance [211-214] work as a pair. The model compares obstacle height in each of 8 directions against each weapon's maximum arc clearance to decide which weapons can fire over which cover.

Cover Height values: 0.0 = open, 0.3 = 150 UU cover, 0.56 = 280 UU cover, 0.7 = full wall.

Arc Clearance values: 0.0 = weapon can't arc, 0.13 = clears up to 400 UU, 1.0 = unlimited clearance.

## Parity Validation

The observation vector must be identical between C++ (`NeuralCombatComponent::GatherObservation`) and Python (`combat_sim._build_observation`). Run `obs_vector_validator.py --check-structure` for structural checks, and `--diff-csv` for numerical comparison against C++ data logs.

The canonical source of truth for the layout is `NeuralCombatTypes.h` → `NeuralObsOffset` namespace.
