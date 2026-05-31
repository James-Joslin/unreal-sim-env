# Weapons & Loadouts

Five weapon presets are available. Stages 1-4 use a fixed preset. Stages 5-7 randomise from a pool each episode.

## Presets

| Preset | Primary | Secondary | Melee |
|---|---|---|---|
| `melee_bot` | Sidearm (8 dmg, 800 range, 10 ammo, 0.4s CD, proj 3000) | — | 35 dmg, 200 range, 0.6s CD |
| `scout` | Laser (8 dmg, 1200 range, 20 ammo, 0.2s CD, proj 4500) | — | 15 dmg, 150 range, 0.8s CD |
| `heavy` | Cannon (35 dmg, 2000 range, 6 ammo, 1.0s CD, 0.5s wind-up, proj 2000) | Missiles (25 dmg, 1800 range, 4 ammo, 1.5s CD, arc 400 UU, proj 1200) | 40 dmg, 250 range, 1.5s CD |
| `sniper` | Railgun (80 dmg, 3000 range, 1 ammo, 2.0s CD, 1.0s wind-up, proj 6000) | Sidearm (10 dmg, 1000 range, 12 ammo, 0.3s CD, proj 3500) | 10 dmg, 150 range, 1.0s CD |
| `tank` | Gatling (5 dmg, 1500 range, 100 ammo, 0.08s CD, proj 4000) | — | 30 dmg, 250 range, 1.2s CD |

All ranged weapons fire projectiles (not hitscan). Faster projectile speed means harder to dodge but still possible at range.

## Arc Mechanics

Arc weapons (e.g. Heavy missiles) can lob projectiles over cover. Whether a weapon can clear a specific obstacle depends on two values:

- **MaxArcableObstacleHeight** (on `FEnemyWeaponSlot`) — the tallest obstacle this weapon can clear. Exposed in the observation vector at [211-214] as `height / 3000`.
- **Cover Height** (per direction, [166-173]) — the actual height of the obstacle in each of 8 compass directions, normalised by 500 UU.

The model compares these to decide: "cover northeast is 200 UU, my missiles clear up to 400 UU → missiles can fire over it."

The C++ projectile's `ArcHeight` property controls the actual trajectory apex. `MaxArcableObstacleHeight` is the designer-set tactical constraint derived from it.

## Weapon State in Observations

The active weapon occupies 10 floats at [21-30]: slot index, ammo fraction, can-fire, reloading, reload progress, range, cooldown, wind-up, can-arc, is-ranged. Three other weapon slots get 4 floats each at [31-42]: ammo, range, reloading, can-arc.

Additional per-weapon observations at the end of the vector: Can Hit Target [205-208] (bool per slot — has ammo, in range, has path, not reloading) and Arc Clearance [211-214].
