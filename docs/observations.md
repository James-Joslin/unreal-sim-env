# Observation Vector

249 normalised floats per frame, frame-stacked ×3 (747 total input). Both the Python training sim and C++ UE5 runtime produce identical vectors.

## Layout

```
[  0.. 20]  Self State                        (21)
[ 21.. 42]  Weapon State                      (22)      10 active + 3×4 other slots
[ 43.. 49]  Archetype                         ( 7)
[ 50.. 73]  Primary Target                    (24)      +class, mana, commitment, gap_closer
[ 74..141]  Hostile Targets                   (68)  ←── 4 slots × 17 (shared hostile encoder)
[142..186]  Allied Robots                     (45)  ←── 3 slots × 15 (shared ally encoder)
[187..194]  Spatial Ring                      ( 8)      sphere sweep distance per direction
[195..202]  Cover Height                      ( 8)      continuous obstacle height / 500
[203..210]  Threat Sensing (Projectile 1)     ( 8)      nearest projectile + melee + dodge
[211..219]  Navmesh Viability                 ( 9)
[220..225]  Group Summary                     ( 6)
[226..226]  Spawn/Leash                       ( 1)
[227..229]  Threat Sensing (Projectile 2)     ( 3)      dist, dirX, dirY
[230..232]  Threat Sensing (Projectile 3)     ( 3)      dist, dirX, dirY
[233..233]  Incoming Threat Count             ( 1)
[234..237]  Can Hit Target Per Weapon         ( 4)      per weapon slot
[238..238]  Total Ammo Fraction               ( 1)
[239..239]  Targets Killed Fraction           ( 1)
[240..243]  Arc Clearance Per Weapon          ( 4)      MaxArcableObstacleHeight / 3000
[244..248]  Player Patterns                   ( 5)      aggression, evasion, predictability, range, mana_burn
```

## Feature Groups

The policy network splits these into four encoder groups, each processed by a specialised sub-network. Entity groups use cross-attention (not max-pooling) for context-aware aggregation.

**Unique features (136 floats):** `[0:74] + [187:249]` — self state, weapons, archetype, primary target (including class, mana, commitment, gap-closer), spatial ring, cover height, threat sensing, navmesh, group summary, spawn/leash, weapon capabilities, ammo, kills, arc clearance, and player patterns. Processed by a dedicated linear encoder.

**Hostile entity slots (4 × 17 = 68 floats):** `[74:142]` — each slot encodes one hostile target:

| Offset | Field | Source |
|---|---|---|
| +0 | occupied | 1.0 if slot filled |
| +1 | rel_x | (target.x − self.x) / 5000 |
| +2 | rel_y | (target.y − self.y) / 5000 |
| +3 | distance | dist / 5000 |
| +4 | health_fraction | `UHealthComponent::GetHealthPercent()` |
| +5 | has_los | line-of-sight check |
| +6 | is_player | `APawn::IsPlayerControlled()` |
| +7 | facing_dot | target facing toward agent, [−1, 1] |
| +8 | priority_score | normalised priority / 120 |
| +9 | threat_level | accumulated damage / 200 |
| +10 | vel_x | velocity.x / 600 |
| +11 | vel_y | velocity.y / 600 |
| +12 | targeting_me | facing dot clamped [0, 1] |
| +13 | character_type | `NeuralCharacterTypeFloat(ECharacterType)` — Knight=0.0, Rogue=0.2, Ranger=0.4, Mage=0.6, Healer=0.8 |
| +14 | mana_fraction | `UManaComponent::GetManaPercent()`, 0 for melee classes |
| +15 | commitment | cast/attack animation progress 0.0–1.0 |
| +16 | gap_closer_threat | 1.0 if gap-closer ready AND in range, else 0.0 |

Processed by a shared-weight encoder, then aggregated via 4-head cross-attention where the unique embedding is the query. The model learns to focus on the most tactically relevant hostile given its current state.

**Ally entity slots (3 × 15 = 45 floats):** `[142:187]` — each slot encodes one allied robot:

| Offset | Field | Source |
|---|---|---|
| +0 | occupied | 1.0 if slot filled |
| +1 | rel_x | relative position X / 5000 |
| +2 | rel_y | relative position Y / 5000 |
| +3 | distance | dist / 5000 |
| +4 | health_fraction | `UHealthComponent::GetHealthPercent()` |
| +5 | has_los | line-of-sight to ally |
| +6 | vel_x | velocity.x / max_speed |
| +7 | vel_y | velocity.y / max_speed |
| +8 | facing_dot | ally facing toward agent, [−1, 1] |
| +9 | ammo_fraction | active weapon ammo fraction |
| +10 | is_reloading | 1.0 if currently reloading |
| +11 | fire_cooldown_frac | active weapon fire cooldown / 2.0 |
| +12 | target_index | which hostile slot this ally is engaging (normalised) |
| +13 | combat_action | what the ally is doing (normalised 0–1) |
| +14 | flanking_angle | cos(my→target vs ally→target), −1=stacked, +1=perfect flank |

The coordination fields (+12 through +14) enable emergent teamwork: the model sees which targets allies are focused on and how they're flanking, allowing it to choose complementary positions and targets without explicit coordination code.

**Threat entity slots (3 × 3 = 9 floats):** Extracted from indices 203/205/206, 227-229, 230-232 — three projectile threats, each with (distance, dirX, dirY). Shared encoder + 4-head cross-attention.

## Player Patterns (5 floats, [244-248])

Exponential moving averages (α=0.05, ~20 second window) tracking player party behavior:

| Index | Field | Description |
|---|---|---|
| 244 | aggression | Target fire rate EMA. High = aggressive players. |
| 245 | evasion | Target dodge frequency EMA. High = evasive players. |
| 246 | predictability | Movement direction entropy (8-bin histogram). 0 = predictable, 1 = random. |
| 247 | preferred_range | Normalised engagement distance EMA. Low = close-range, high = ranged. |
| 248 | mana_burn_rate | Mana spending rate EMA. High = burning mana fast (will run out). |

These let the model adapt mid-fight: kite aggressive players, rush passive ones, punish predictable movement, pressure targets burning mana.

## Action Space

Three autoregressive heads — each conditions on the previous:

| Head | Size | Actions |
|---|---|---|
| Movement | 9 | 8 compass directions + hold |
| Combat | 8 | None, Fire, Reload, Switch0, Switch1, Melee, Block, **Dodge** |
| Target | 5 | Primary + 4 hostile slots |

**Dodge** (combat action 7): model-controlled strategic dodging. The agent decides when to spend its dodge cooldown — conserving it for big threats vs burning it on weak attacks. Direction follows current movement or defaults to away-from-target. Shares cooldown with the emergency auto-dodge system (C++ `EnemyDodgeComponent`), which overrides for point-blank projectiles the 200ms decision tick can't react to.

## Spatial Ring

8 sphere sweeps at 45° intervals. Each sweep uses a sphere with radius = `AGENT_BODY_RADIUS` (30 UU), answering "can my body fit through there?" Detects obstacles and arena boundaries only — not characters. Character positions are encoded in the dedicated hostile/ally slots.

## Cover Height + Arc Clearance

Cover Height [195-202] and Arc Clearance [240-243] work as a pair. The model compares obstacle height in each of 8 directions against each weapon's maximum arc clearance to decide which weapons can fire over which cover.

Cover Height values: 0.0 = open, 0.3 = 150 UU cover, 0.56 = 280 UU cover, 0.7 = full wall.

Arc Clearance values: 0.0 = weapon can't arc, 0.13 = clears up to 400 UU, 1.0 = unlimited clearance.

## C++ Component Mapping

The new hostile/ally fields read directly from UE5 components:

| Field | C++ Source |
|---|---|
| character_type | `FCharacterStats::Identity.CharacterType` → `NeuralCharacterTypeFloat()` |
| mana_fraction | `UManaComponent::GetManaPercent()` |
| health_fraction | `UHealthComponent::GetHealthPercent()` |
| commitment | `UAnimInstance` montage progress, or custom combat state |
| gap_closer_threat | Ability system cooldown check + distance check |
| ally target_index | `UEnemyPerceptionComponent::GetDetectedTarget()` → match to hostile slot list |
| ally combat_action | `UNeuralCombatComponent::GetLastCombatAction()` normalised |
| ally flanking_angle | `FVector::DotProduct(MyToTarget, AllyToTarget)` |

## Parity Validation

The observation vector must be identical between C++ (`NeuralCombatComponent::GatherObservation`) and Python (`combat_sim._build_observation`). The canonical source of truth for the layout is `NeuralCombatTypes.h` → `NeuralObsOffset` namespace, with field-level offsets in `NeuralHostileField` and `NeuralAllyField`.

Run `obs_vector_validator.py --check-structure` for structural checks, and `--diff-csv` for numerical comparison against C++ data logs.