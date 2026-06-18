# Rewards

The reward function is designed to prioritise objective completion while using small shaping rewards to accelerate learning.

## Design Principles

1. **Objective rewards dominate** — kills and wins should be worth far more than per-step shaping.
2. **Shaping rewards stay small** — positioning and behaviour nudges should not be farmable.
3. **Timeouts should be unattractive** — passive survival should not outscore winning.
4. **Damage penalties should not prevent trading** — the agent must be willing to take reasonable damage to secure kills.
5. **Curriculum controls complexity** — later-stage rewards activate only after the basics are learned.

## Activation by Stage

| Reward Component | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Damage / Kill / Death | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Episode Win/Loss/Timeout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Optimal Range | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Damage Inactivity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Invalid Action | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ammo Management |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anti-Degenerate |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| Flanking / Positioning |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| Aggression |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| Movement / Strafe Fire |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-Target Progression |  |  | ✓ | ✓ | ✓ | ✓ | ✓ |
| Weapon Selection |  |  |  | ✓ | ✓ | ✓ | ✓ |
| Arc vs Direct Fire |  |  |  | ✓ | ✓ | ✓ | ✓ |
| Archetype-Specific |  |  |  |  | ✓ | ✓ | ✓ |
| Ally Coordination |  |  |  |  |  | ✓ | ✓ |

## Core Objective Rewards

- Damage dealt scales with target HP percentage.
- Kills and episode wins provide the main positive signal.
- Death, loss, timeout and surviving targets provide negative terminal pressure.
- From stage 3 onward, per-step survival is net-negative to discourage passive play.

## Shaping Categories

### Ammo Management

Rewards useful reload/switch behaviour and penalises wasteful fire or empty-weapon states.

### Anti-Degenerate Behaviour

Penalises idle behaviour, spinning, camping, wall hugging and corner abuse.

### Flanking and Positioning

Encourages tactically useful side/rear angles and firing from advantageous positions.

### Aggression

Penalises being passive when a valid shot is available.

### Movement

Rewards firing while moving and useful strafe-fire behaviour.

### Multi-Target Progression

Encourages retargeting dead targets quickly, finishing low-HP enemies and clearing all remaining targets.

### Weapon Selection

Rewards using the right weapon for the current range/ammo/LOS/cover situation.

### Arc vs Direct Fire

Rewards arcing over cover when direct LOS is blocked, while preferring direct weapons when LOS exists.

### Archetype-Specific Behaviour

Adds role-specific nudges for ranged, melee, healer and tank behaviours.

### Ally Coordination

Encourages protecting low-HP allies, attacking ally threats and avoiding ally collision/stacking.

## Practical Debugging

If reward rises but win rate stays flat, check for shaping farming. The intended pattern is:

```text
win rate improves
kills increase
episode length decreases or stabilises
reward increases for objective reasons, not passive shaping
```

The web tool reward budget view is useful for detecting episodes where reward exceeds a sensible win budget without corresponding kills.
