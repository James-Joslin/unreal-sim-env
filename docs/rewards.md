# Rewards

## Activation by Stage

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

## Objective Rewards (all stages)

| Component | Value | Notes |
|---|---|---|
| Damage dealt | +0.15 per 1% target HP | Scales with target max HP |
| Kill target | +35.0 | |
| Take damage | -0.015 per 1% own HP | |
| Die | -10.0 | |
| Episode win | +50.0 | + speed bonus up to ~75 total |
| Episode loss | -5.0 | |
| Episode timeout | -8.0 | |
| Alive per step | +0.005 (stage 1), 0.0 (stage 2), -0.02 (stage 3+) | Negative = survival costs |

## Shaping Rewards

### Ammo Management (stage 2+)
Reload behind cover (+0.1), switch to loaded weapon (+0.2), wasted shot (-0.0001), all weapons empty (-0.1), fire hit (+0.15).

### Anti-Degenerate (stage 3+)
Idle penalty (-0.01/step), spinning (-0.03), camping (-0.03/step), wall hugging (-0.02/step), corner penalty (-0.04).

### Flanking (stage 3+)
Behind target (+0.008/step), at side (+0.003/step), fire from behind (+0.06/shot), fire from side (+0.03/shot), cover flank (+0.08), lost flanking (-0.02).

### Aggression (stage 3+)
Passive in range (-0.08/step) — per-step penalty for not firing when the agent could.

### Movement (stage 3+)
Mobile fire (+0.02/shot), strafe fire (+0.04/shot).

### Multi-Target (stage 3+)
Retarget urgency (-0.06/step when current target dead but others alive), target low HP bonus (+3.0), surviving target penalty (-8.0 per living target at episode end).

### Weapon Selection (stage 4+)
Fire in optimal band (+0.06/shot), fire outside optimal (-0.02/shot), swap to better range (+0.08), swap to worse (-0.1), smart reload swap (+0.1), holding wrong weapon (-0.0025/step).

### Arc vs Direct (stage 4+)
Arc over cover (+0.03/shot when LOS blocked), direct with LOS (+0.02/shot), holding arc with LOS (-0.005/step).

### Archetype-Specific (stage 5+)
Per-archetype shaping — ranged kiting, melee gap-closing, tank body-blocking, healer ally support. See [curriculum.md](curriculum.md) stages 5-7 for full values.

### Ally Coordination (stage 6+)
Protect low-HP ally (+0.015/step), fire at ally's threat (+0.04/shot), ally died nearby (-1.5), ally collision (-0.04).

## Design Philosophy

Objective rewards (kill, win, loss) represent >80% of a winning episode's total. Shaping rewards are kept under 15% to prevent reward farming. Per-step survival is negative (-0.02/step from stage 3) so passive play is net-negative — only dealing damage overcomes the survival cost. The surviving target penalty (-8.0 per living target) creates a gradient between partial and full clears.
