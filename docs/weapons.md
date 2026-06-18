# Weapons and Loadouts

Stages 1-4 use fixed weapon presets. Stages 5-7 randomise from a weapon pool each episode.

## Presets

| Preset | Primary | Secondary | Melee |
|---|---|---|---|
| `melee_bot` | Sidearm: 8 damage, 800 range, 10 ammo, 0.4s cooldown, 3000 projectile speed | — | 35 damage, 200 range, 0.6s cooldown |
| `scout` | Laser: 8 damage, 1200 range, 20 ammo, 0.2s cooldown, 4500 projectile speed | — | 15 damage, 150 range, 0.8s cooldown |
| `heavy` | Cannon: 35 damage, 2000 range, 6 ammo, 1.0s cooldown, 0.5s wind-up, 2000 projectile speed | Missiles: 25 damage, 1800 range, 4 ammo, 1.5s cooldown, arc 400 UU, 1200 projectile speed | 40 damage, 250 range, 1.5s cooldown |
| `sniper` | Railgun: 80 damage, 3000 range, 1 ammo, 2.0s cooldown, 1.0s wind-up, 6000 projectile speed | Sidearm: 10 damage, 1000 range, 12 ammo, 0.3s cooldown, 3500 projectile speed | 10 damage, 150 range, 1.0s cooldown |
| `tank` | Gatling: 5 damage, 1500 range, 100 ammo, 0.08s cooldown, 4000 projectile speed | — | 30 damage, 250 range, 1.2s cooldown |

All ranged weapons fire projectiles rather than hitscan shots. Faster projectile speeds are harder to dodge, but still dodgeable at sufficient range.

## Arc Mechanics

Arc weapons can fire over cover if the weapon clearance is sufficient for the obstacle height.

Important observation fields:

```text
Cover height per direction:       [195..202]
Arc clearance per weapon slot:    [240..243]
Can-hit-target per weapon slot:   [234..237]
```

The model can learn decisions such as:

```text
cover northeast is 200 UU
missiles clear up to 400 UU
therefore missiles can fire over it
```

## Weapon State in Observations

Active weapon state occupies `[21..30]`:

```text
active slot index
ammo fraction
can fire
is reloading
reload progress
range
cooldown
wind-up
can arc
is ranged
```

Other weapon slots occupy `[31..42]`, with four floats each:

```text
ammo fraction
range
is reloading
can arc
```

Additional weapon capability fields appear near the end of the vector:

```text
[234..237] can hit target per weapon slot
[238]      total ammo fraction
[240..243] arc clearance per weapon slot
```
