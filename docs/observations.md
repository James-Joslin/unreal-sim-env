# Observation Vector

The observation vector is **249 normalised floats per frame**. The deployed policy uses a **3-frame stack**, producing a **747-float** flat input.

Python and C++ must remain field-for-field compatible. Treat the C++ `NeuralCombatTypes.h` offsets and the Python observation builder as parity-critical code.

## Layout

```text
[  0.. 20] Self State                    (21)
[ 21.. 42] Weapon State                  (22)
[ 43.. 49] Archetype                     ( 7)
[ 50.. 73] Primary Target                (24)
[ 74..141] Hostile Targets               (68)  4 slots x 17
[142..186] Allied Robots                 (45)  3 slots x 15
[187..194] Spatial Ring                  ( 8)
[195..202] Cover Height                  ( 8)
[203..210] Threat Sensing / Projectile 1 ( 8)
[211..219] Navmesh Viability             ( 9)
[220..225] Group Summary                 ( 6)
[226..226] Spawn / Leash                 ( 1)
[227..229] Projectile 2 Threat           ( 3)
[230..232] Projectile 3 Threat           ( 3)
[233..233] Incoming Threat Count         ( 1)
[234..237] Can Hit Target Per Weapon     ( 4)
[238..238] Total Ammo Fraction           ( 1)
[239..239] Targets Killed Fraction       ( 1)
[240..243] Arc Clearance Per Weapon      ( 4)
[244..248] Player Patterns               ( 5)
```

Primary-target index `73`, previously reserved padding, is
`RepositionReady`. Reusing it preserves the 249-float observation and all four
hostile slots.

## Encoder Groups

The policy network splits each frame into these model-facing groups:

```text
Unique features: [0:74] + [187:249] = 136 floats
Hostiles:        [74:142]  = 4 x 17
Allies:          [142:187] = 3 x 15
Threats:         projectile threat triples from indices 203/205/206, 227-229, 230-232
```

## Action Space

The policy uses three independent one-pass action heads:

```text
Movement: 9 actions  = hold + 8 compass directions
Combat:   9 actions  = none, fire, reload, switch0, switch1, melee, block, dodge, reposition
Target:   5 actions  = hostile slots 0..3 + keep current target
```

Reposition uses the selected movement direction for 0.6 seconds at 1.75x
movement speed, has a 3.0-second cooldown, and grants no invulnerability.
Dodge is also explicit; neural agents never trigger it automatically.

## Player Patterns

Indices `[244..248]` contain exponential moving averages for party behaviour:

```text
244 aggression
245 evasion
246 predictability
247 preferred_range
248 mana_burn_rate
```

These let the policy adapt mid-fight to aggressive, evasive, predictable, ranged or mana-burning player behaviour.
