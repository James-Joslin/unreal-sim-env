"""
reward.py — Reward function for Neural Combat AI training.

This module defines the complete reward function used by the PPO training
pipeline. It's called every step inside the Gymnasium environment's step()
method.

DESIGN PRINCIPLES
    1. Reward what you want to see, penalise what you don't.
    2. Keep rewards SMALL and FREQUENT. Big sparse rewards (kill=+100)
       cause high variance in advantage estimation. Small dense rewards
       (+0.02 per tick for good positioning) train faster and more stably.
    3. Archetype conditioning: same base rewards for all, with additive
       archetype-specific bonuses that nudge each role toward its purpose.
    4. Anti-degenerate rewards prevent pathological learned behaviours
       (spinning, camping, stacking, ignoring targets).
    5. Curriculum stage can scale or disable certain rewards — early stages
       focus on basics (deal damage, don't die), later stages add
       complexity (ammo management, cover usage, coordination).

USAGE
    from reward import CombatRewardFunction

    reward_fn = CombatRewardFunction(
        archetype="ranged",
        curriculum_stage=3,
    )

    # Inside env.step():
    reward, info = reward_fn.compute(prev_state, action, next_state)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────
#  Archetype Enum (matches C++ EEnemyArchetype)
# ─────────────────────────────────────────────────────────────────

class Archetype(IntEnum):
    RANGED = 0
    MELEE = 1
    HEALER = 2
    TANK = 3


# ─────────────────────────────────────────────────────────────────
#  Combat Action Enum (matches C++ ENeuralCombatAction)
# ─────────────────────────────────────────────────────────────────

class CombatAction(IntEnum):
    NONE = 0
    FIRE = 1
    RELOAD = 2
    SWITCH_0 = 3
    SWITCH_1 = 4
    MELEE = 5
    BLOCK = 6


# ─────────────────────────────────────────────────────────────────
#  State Snapshot (what the reward function sees each step)
# ─────────────────────────────────────────────────────────────────

@dataclass
class CombatState:
    """Minimal state snapshot for reward computation.
    The sim environment builds this from its internal state."""

    # Self
    self_hp: float = 1.0            # [0, 1]
    self_alive: bool = True
    self_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    self_speed: float = 0.0         # current speed fraction [0, 1]

    # Active weapon
    active_weapon_index: int = 0
    active_ammo_fraction: float = 1.0
    active_weapon_range: float = 1000.0
    active_weapon_is_ranged: bool = True
    active_weapon_can_arc: bool = False  # True if weapon fires arcing projectiles over cover
    can_arc_over_target_cover: bool = False  # True if active arc weapon can clear the specific
                                             # obstacle blocking LOS (height-aware check)
    is_reloading: bool = False
    can_fire: bool = True

    # Other weapons
    other_ammo_fractions: list = field(default_factory=list)  # per-slot
    other_weapon_ranges: list = field(default_factory=list)   # per-slot effective range
    all_ranged_empty: bool = False
    has_direct_weapon_with_ammo: bool = False  # True if any non-arc weapon has ammo

    # Active weapon range context
    active_optimal_min: float = 400.0
    active_optimal_max: float = 1200.0

    # Did a weapon switch happen this step?
    weapon_switched: bool = False
    prev_weapon_index: int = 0

    # Target
    target_hp: float = 1.0
    target_alive: bool = True
    target_distance: float = 500.0
    has_los: bool = True
    target_in_range: bool = True
    target_behind_cover: bool = False
    target_cover_height: float = 0.0     # Height of obstacle blocking LOS (0 = clear)
    target_behind_low_cover: bool = False # Low blocked + high clear = arc opportunity

    # Flanking / positioning relative to target
    target_facing_agent: float = 1.0     # 1.0 = target facing us, 0.0 = facing away
    agent_behind_target: bool = False    # agent is outside target's vision cone (<0.17)
    agent_flanking: bool = False         # agent is to the side (0.17 < facing < 0.7)

    # Spatial / positioning
    behind_cover: bool = False       # is self behind cover relative to target
    in_optimal_range: bool = False   # within weapon's sweet spot
    height_above_ground: float = 0.0
    near_wall: bool = False          # within 150 UU of arena boundary
    in_corner: bool = False          # within 200 UU of TWO arena boundaries
    agent_speed: float = 0.0        # current speed in UU/s (raw, not normalised)
    moving_away_from_threat: bool = False  # velocity pointing away from nearest target

    # Group (for stages 6-7)
    alive_allies: int = 0
    alive_hostiles: int = 1
    nearest_ally_distance: float = 9999.0
    ally_overlap: bool = False       # overlapping another enemy

    # Allies needing help (healer/tank)
    lowest_ally_hp: float = 1.0
    ally_in_danger: bool = False     # ally < 30% HP
    ally_just_died: bool = False     # an ally died this step
    self_between_threat_and_ally: bool = False  # tank body-blocking

    # Step metadata
    step_count: int = 0
    episode_damage_dealt: float = 0.0
    targets_killed: int = 0
    # Multi-target: total HP fraction lost across ALL targets this step.
    # Used to reward damage even when the selected target isn't the one hit.
    total_damage_all_targets: float = 0.0
    targets_attacked: set = field(default_factory=set)  # unique target IDs hit


# ─────────────────────────────────────────────────────────────────
#  Reward Component Weights (tunable per iteration)
# ─────────────────────────────────────────────────────────────────

"""
reward_weights_v2.py — Corrected reward weights for combat AI.

PROBLEM WITH V1 WEIGHTS
    The agent farms per-step shaping rewards (flanking, kiting, positioning)
    instead of actually killing targets. Episode reward reaches +150 but
    win rate is only 25%. This happens because:

    1. Per-step shaping rewards (~0.05/step × 1200 steps = 60.0) EXCEED
       the kill reward (8.0 per kill × 4 targets = 32.0).
    2. The damage taken penalty (-0.15/%) makes trading HP for kills
       unprofitable. Agent takes -14 in damage but only deals -4.5.
    3. Die penalty (-12) + loss (-7) = -19 vs timeout (-5). Dying is
       3.8× worse than timing out, so the agent rationally avoids combat.

DESIGN PRINCIPLES FOR V2
    1. OBJECTIVE REWARDS MUST DOMINATE. Kill + win should be >80% of a
       winning episode's reward. Shaping should be <20%.
    2. PER-STEP SHAPING MUST BE TINY. A full episode of perfect positioning
       should earn ~5-10, not 60+.
    3. WINNING MUST BE DRAMATICALLY BETTER THAN TIMING OUT. The gap between
       "kill all targets" and "survive and timeout" must be so large that
       no amount of shaping-farming can close it.
    4. DAMAGE TAKEN PENALTY MUST BE MILD ENOUGH that the agent is willing
       to trade HP for kills. In a 4v1 fight, taking damage is INEVITABLE.

REWARD BUDGET ANALYSIS (Stage 7, 4 targets, 1200 steps)

    Winning episode (kill all 4, agent takes 60% HP damage):
        Damage dealt:     0.25 × 400 (4 targets × 100%)  = +100.0
        Kills:            20.0 × 4                        = +80.0
        Target low HP:    3.0 × 4                         = +12.0
        Episode win:      50.0 (with speed bonus ~65.0)   = +65.0
        Damage taken:     -0.05 × 60                      = -3.0
        Shaping (flanking, positioning, ammo):             ≈ +8.0
        TOTAL:                                            ≈ +262.0

    Timeout episode (kiting, chip damage, survive):
        Damage dealt:     0.25 × 40 (chip damage only)    = +10.0
        Target low HP:    3.0 × 1 (got one target low)    = +3.0
        Timeout:                                           = -8.0
        Surviving targets: -8.0 × 4                       = -32.0
        Retarget urgency: -0.06 × ~20 steps               = -1.2
        Damage taken:     -0.05 × 50                      = -2.5
        Shaping:                                           ≈ +5.0
        TOTAL:                                            ≈ -25.7

    Death episode (engaged but died, killed 1):
        Damage dealt:     0.25 × 150 (killed 1.5 targets) = +37.5
        Kills:            20.0 × 1                         = +20.0
        Target low HP:    3.0 × 1                          = +3.0
        Die:                                               = -10.0
        Episode loss:                                      = -5.0
        Surviving targets: -8.0 × 3                        = -24.0
        Damage taken:     -0.05 × 100                      = -5.0
        Shaping:                                           ≈ +3.0
        TOTAL:                                            ≈ +19.5

    KEY: Winning (+262) >> Dying-but-fighting (+19.5) >> Timeout (-25.7)
    The agent MUST learn that fighting and dying is BETTER than not fighting.
    And winning is MUCH better than either.
"""

from dataclasses import dataclass


@dataclass
class RewardWeights:

    # ═══════════════════════════════════════════════════════════════
    #  OBJECTIVE REWARDS — these must dominate everything else
    # ═══════════════════════════════════════════════════════════════

    # Damage dealt: the dense gradient signal.
    # 0.25 per 1% = 25.0 for depleting a full HP bar.
    # With 3 targets that's 75.0 — enough to guide learning but NOT
    # enough to dominate kill/win rewards.
    #
    # Was 1.0 (giving 100 per HP bar — 5× documented intent). At 1.0,
    # damage reward was 77% of a winning episode, drowning out win/kill
    # signals and causing 0.25 reward-winrate correlation.
    # Reduced from 0.25 → 0.15: damage alone is less valuable. Combined
    # with the focus fire multiplier, spread damage earns very little
    # while focused damage that leads to kills earns more.
    damage_dealt: float = 0.15

    # Kill: the key objective milestone. Increased from 20 → 35.
    # Must be large enough that focusing one target down is clearly better
    # than spreading chip damage across all targets.
    # 3 kills = 105.0 (was 60.0).
    kill_target: float = 35.0

    # Episode win: MUST dominate. With speed bonus up to 75.
    # This is what makes winning clearly the best outcome.
    # Old value (25) was only 8% of a winning episode's reward — invisible.
    episode_win: float = 50.0

    # Episode loss (death): dying while fighting must NOT be catastrophic.
    # The agent that engages and dies should still get net positive from
    # the damage it dealt. die + loss = -15, one kill = +45. Net = +30.
    episode_loss: float = -5.0

    # Episode timeout: timing out must be worse than dying while fighting.
    episode_timeout: float = -8.0

    # Surviving targets penalty: per-target penalty at episode end for
    # each target still alive. Creates a gradient between "killed 2 of 3"
    # (penalty = -8) and "killed 0 of 3" (penalty = -24).
    surviving_target_penalty: float = -8.0

    # Death penalty: combined with episode_loss = -15 total.
    die: float = -10.0

    # ═══════════════════════════════════════════════════════════════
    #  MULTI-TARGET PROGRESSION — gradient between kills
    # ═══════════════════════════════════════════════════════════════

    # Retarget urgency: per-step penalty when selected target is dead
    # but other hostiles are still alive. Without this, all positioning
    # signals (range_closing, optimal_range, flanking) go to zero after
    # a kill because they're gated by target_alive. The agent wanders
    # blind until it stumbles into range. This penalty creates urgency
    # to select a new target via the target action head.
    retarget_urgency: float = -0.06

    # Target low HP bonus: one-time reward when a target drops below
    # 30% HP. Creates a "finish them off" gradient — the agent learns
    # that getting a target low is a milestone worth pursuing, not just
    # the binary kill moment. Prevents the "spread damage evenly and
    # kill nobody" failure mode.
    target_low_hp_bonus: float = 3.0

    # ═══════════════════════════════════════════════════════════════
    #  DAMAGE ECONOMICS — the key ratio that determines aggression
    # ═══════════════════════════════════════════════════════════════

    # Damage taken: mild penalty so combat is profitable.
    # At -0.05, losing 100% HP costs -5.0. Dealing 100% HP gives +25.0.
    # The ratio is 5:1 in favour of dealing damage. Even taking 4× the
    # damage dealt, combat is still profitable.
    # Damage taken: -0.02 per 1% HP lost. Teaches cover usage but must
    # NOT make combat net-negative. At 2% HP/step in combat:
    #   -0.04/step (damage taken) vs -0.08/step (hiding far + wall hug)
    # Fighting is now CHEAPER than hiding, even when missing shots.
    # The old -0.05 made hiding (-0.08) cheaper than combat (-0.10),
    # teaching the agent to avoid fights entirely.
    take_damage: float = -0.015

    # [Fix 4] Alive per step: NEGATIVE. Every step of survival actively
    # costs the agent. The only way to overcome this cost is dealing damage
    # (damage_dealt = +0.25/% HP = ~0.75/step when hitting). This makes
    # passive survival (dodging for 1000 steps) net-negative:
    #   400 steps × -0.02 = -8.0 (comparable to a kill reward).
    # The old +0.000001 was effectively zero, making survival free and
    # enabling the "dodge until timeout" exploit.
    alive_per_step: float = -0.02

    # ═══════════════════════════════════════════════════════════════
    #  SHAPING REWARDS — guidance signals, NOT objectives
    #  Rule: total shaping per episode must be <15% of objective rewards
    # ═══════════════════════════════════════════════════════════════

    # Invalid action: keep as-is, one-time penalty.
    invalid_action: float = -0.1

    # Optimal range: REDUCED to near-zero. The agent should learn range
    # from damage success (hit vs miss), not a per-step drip.
    in_optimal_range: float = 0.01

    # Range closing: reward for approaching when out of range.
    # NOT gated by engagement — this IS the approach signal.
    range_closing: float = 0.04

    # Out of range penalty: continuous per-step cost for being beyond
    # weapon range. Scales with distance (no cap) so the agent always has
    # a gradient toward the target, even across a 4000 UU arena.
    # At 3000 UU from optimal_max: -0.06 × (0.3 + 0.7 × 3.75) = -0.18/step.
    # That's -18 over 100 steps — comparable to a kill reward. The agent
    # can't afford to sit far away.
    out_of_range_penalty: float = -0.06

    # Damage inactivity: INCREASED from -0.075 to -0.1.
    # This is the anti-kiting stick. After 10 steps without dealing
    # damage, the agent loses 0.1/step. Over 100 idle steps that's -10.
    # Combined with the -8 timeout, pure kiting earns ~-18 per episode.
    damage_per_step_min: float = -0.05  # Was -0.1 (escalation still reaches -0.15)

    # ── Ammo management ──────────────────────────────────────────
    # These are fine. One-time events, small magnitudes.

    reload_behind_cover: float = 0.3
    reload_in_open: float = -0.02    # Reduced — don't punish reloading
    switch_to_loaded: float = 0.2
    wasted_shot: float = -0.0001
    all_empty_penalty: float = -0.1
    fire_hit: float = 0.15

    # ── Aggression signal ────────────────────────────────────────
    # Per-step penalty for choosing None when you COULD fire.
    # Conditions: target alive, in range, has LOS, has ammo, can fire.
    # This is the missing "pull the trigger" signal. Without it,
    # approaching is rewarded but actually shooting is optional —
    # the agent learns that hesitating in range is nearly free while
    # avoiding return fire damage.
    # At -0.08/step, 10 steps of hesitation costs -0.8. That's
    # comparable to the reward from a single hit (+1.4), creating
    # real urgency to engage.
    passive_in_range: float = -0.08

    # ── Weapon selection ─────────────────────────────────────────
    # Fine as-is. Per-event, small, directional.

    fire_in_optimal_band: float = 0.06
    fire_outside_optimal: float = -0.02
    swap_to_better_range: float = 0.08
    swap_to_worse_range: float = -0.1
    swap_reload_smart: float = 0.1
    holding_wrong_weapon: float = -0.0025
    arc_over_cover_bonus: float = 0.03     # Was 0.06 — reduced to prevent arc weapon
                                           # becoming the universal default. Only triggers
                                           # when LOS IS blocked and arc CAN clear it.
    direct_fire_with_los: float = 0.02     # Bonus for firing a direct (non-arc) weapon
                                           # when LOS is clear. Teaches: use the faster
                                           # weapon when you have a clean shot.
    holding_arc_with_los: float = -0.005   # Per-step penalty for holding an arc weapon
                                           # when LOS to target is clear AND a direct weapon
                                           # has ammo. Arc weapons are slower — use them for
                                           # their purpose (over cover), not as a default.

    # ── Anti-degenerate ──────────────────────────────────────────
    # REDUCED across the board. These existed to prevent specific
    # degenerate behaviours, but at their V1 magnitudes they were
    # punishing NORMAL combat behaviour too.
    #
    # Rule: max total anti-degen per step should be ~0.03, not 0.10+

    idle_penalty: float = -0.01       # Was -0.015
    spinning_penalty: float = -0.03    # Was -0.5 (accidental 16× increase)
    camping_penalty: float = -0.03    # Was -0.03. HALVED. Standing still
                                      # to aim is valid combat behaviour!
    wall_hugging_penalty: float = -0.02  # Was -0.05
    corner_penalty: float = -0.04       # Was -0.2
    ally_collision: float = -0.02
    target_diversity: float = 0.1
    min_damage_penalty: float = -5.0  # INCREASED from -1.0. Dealing zero
                                      # damage in an entire episode is bad.

    # Mobile fire and strafe bonuses. The agent should move while shooting,
    # ideally LATERALLY (strafing) rather than running away then turning.
    # Strafing maintains engagement distance, makes the agent harder to hit,
    # and looks like competent combat behaviour.
    mobile_fire_bonus: float = 0.02    # Was 0.01. Any movement while firing.
    strafe_fire_bonus: float = 0.04    # NEW. Firing while moving laterally
                                       # (distance to target barely changes).
                                       # Replaces kite_damage_bonus which
                                       # rewarded running away specifically.

    # ── Flanking / Positioning ───────────────────────────────────
    # DRAMATICALLY REDUCED. These were the #1 source of reward farming.
    # At 0.05/step for 200 steps, flanking alone earned +10.0 per episode.
    # That's more than a kill reward! Now capped so a full episode of
    # flanking earns ~2-3, which is a nudge not a strategy.

    flank_behind_target: float = 0.008    # Was 0.05 — reduced by 70%
    flank_side_angle: float = 0.003       # Was 0.02 — reduced by 75%
    flank_fire_from_behind: float = 0.06  # Keep — per-event, rewards shooting
    flank_fire_from_side: float = 0.03    # Keep — per-event, rewards shooting
    flank_used_cover_to_flank: float = 0.08  # Keep — per-event, tactical
    flank_lost_position: float = -0.02    # Was -0.03

    # ── Melee archetype ──────────────────────────────────────────
    melee_close_distance: float = 0.01    # Was 0.02
    melee_in_range: float = 0.005         # Was 0.01
    melee_retreat_penalty: float = -0.03  # Was -0.05
    melee_gap_close_bonus: float = 0.1

    # ── Ranged archetype ─────────────────────────────────────────
    ranged_too_close: float = -0.02       # Was -0.05. Being close isn't
                                          # always wrong — melee finishers!
    ranged_too_far: float = -0.02         # Was -0.05; reduced since out_of_range_penalty
                                          # covers beyond-weapon-range; this covers the
                                          # "in range but not optimal" zone only
    ranged_strafe_fire: float = 0.03     # Bonus for strafing in optimal range
                                          # while dealing damage. Replaces the old
                                          # ranged_kite_success which rewarded fleeing.
    ranged_standing_still_threat: float = -0.02  # Was -0.03
    ranged_stagger_ammo: float = 0.05

    # ── Healer archetype ─────────────────────────────────────────
    healer_heal_ally: float = 0.15
    healer_ally_died: float = -2.0
    healer_apply_buff: float = 0.1
    healer_maintain_distance: float = 0.005  # Was 0.01. Halved.
    healer_overheal: float = -0.05
    healer_reload_heal: float = 0.03

    # ── Tank archetype ───────────────────────────────────────────
    tank_absorb_damage: float = 0.1
    tank_body_block: float = 0.005    # Was 0.01. Halved (per-step).
    tank_protect_low_hp: float = 0.01 # Was 0.02. Halved (per-step).
    tank_suppression: float = 0.01
    tank_block_while_focused: float = 0.02

    # ── Ally protection (all archetypes, stage 6+) ────────────────
    # These teach every archetype to be aware of allies, not just tanks.
    # A ranged DPS that ignores a dying ally is bad teamplay.
    protect_low_hp_ally: float = 0.015     # Per-step: between a threat and a low-HP ally
    fire_at_ally_threat: float = 0.04      # Per-shot: attacking the target that's
                                           # targeting a low-HP ally (draw aggro)
    ally_died_nearby: float = -1.5         # One-time: an ally died while agent was
                                           # in range to help. Teaches "don't ignore
                                           # allies being killed next to you."

# ─────────────────────────────────────────────────────────────────
#  Reward Function
# ─────────────────────────────────────────────────────────────────

class CombatRewardFunction:
    """Computes per-step rewards for PPO training.

    Args:
        archetype: Which enemy role (affects archetype-specific bonuses).
        curriculum_stage: 1-7. Controls which reward components are active.
        weights: Tunable reward values. Pass custom to override defaults.
    """

    def __init__(
        self,
        archetype: str | int = "ranged",
        curriculum_stage: int = 1,
        weights: Optional[RewardWeights] = None,
    ):
        if isinstance(archetype, str):
            self.archetype = Archetype[archetype.upper()]
        else:
            self.archetype = Archetype(archetype)

        self.stage = curriculum_stage
        self.w = weights or RewardWeights()

        # Reward ramping: new reward components for this stage start at 0
        # and linearly ramp to full strength over ramp_steps. This prevents
        # reward shock when new components activate (e.g. archetype rewards
        # appearing at stage 5 crashed the model before this fix).
        self._ramp_steps = 10_000  # decision steps to reach full new-reward strength
        self._global_step = 0

        # Tracking for multi-step rewards.
        self._consecutive_idle = 0
        self._consecutive_camping = 0
        self._recent_move_dirs: list = []  # last 5 movement directions
        self._last_position = np.zeros(2)
        self._episode_targets_hit: set = set()
        self._actual_move_streak = 0
        self._episode_damage_dealt = 0.0
        self._step_count = 0
        self._steps_since_damage = 0       # steps without dealing damage
        self._steps_since_kill = 999        # steps since last kill (for grace period)
        self._last_kill_count = 0           # targets killed as of last step
        self._targets_seen_low_hp: set = set()  # target IDs already rewarded for low HP

    def _ramp_factor(self) -> float:
        """Returns 0.0→1.0 over the first _ramp_steps of the stage.
        Applied to reward components that are new at this stage."""
        if self._ramp_steps <= 0:
            return 1.0
        return min(1.0, self._global_step / self._ramp_steps)

    def reset(self):
        """Call at episode start."""
        self._consecutive_idle = 0
        self._consecutive_camping = 0
        self._recent_move_dirs = []
        self._last_position = np.zeros(2)
        self._episode_targets_hit = set()
        self._episode_damage_dealt = 0.0
        self._step_count = 0
        self._steps_since_damage = 0
        self._steps_since_kill = 999
        self._last_kill_count = 0
        self._targets_seen_low_hp = set()
        self._actual_move_streak = 0
        # Note: _global_step does NOT reset — it tracks total steps across episodes.

    def compute(
        self,
        prev: CombatState,
        action: Tuple[int, int, int],  # (movement, combat, target)
        curr: CombatState,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute the reward for transitioning from prev to curr via action.

        Returns:
            reward: Total scalar reward for this step.
            info: Dict of individual reward components (for logging/debugging).
        """
        move_action, combat_action, target_action = action
        info: Dict[str, float] = {}
        self._step_count += 1
        self._global_step += 1
        ramp = self._ramp_factor()
        
        # Engagement gate: shaping rewards only pay out when actively fighting.
        # [Fix 3] Changed from gradual decay over 8 steps to binary gate
        # at 4 steps. The old gradual decay let the agent farm shaping by
        # dealing chip damage every 7 steps — engagement stayed at ~0.1-0.2,
        # enough to collect positioning/flanking/weapon rewards indefinitely.
        # Binary gate means: fight or get nothing. 4 steps (~0.8s) gives
        # enough time for weapon cooldowns but not for extended passive play.
        engagement = 1.0 if self._steps_since_damage <= 4 else 0.0

        # ── Shared Rewards ───────────────────────────────────────

        # Damage dealt — uses ONLY the env-tracked multi-target aggregate.
        # [CRITICAL] No single_target_damage fallback — that was exploited
        # via target switching to generate phantom damage rewards.
        total_damage = curr.total_damage_all_targets

        # Focus fire multiplier: concentrated damage on the selected target
        # is worth up to 1.5× base, while spread damage across targets is
        # worth 0.5× base. This prevents the "spray damage everywhere
        # without killing anything" exploit where the agent earns damage_dealt
        # reward without ever focusing a target to 0 HP.
        focus_mult = 1.0
        if total_damage > 1e-6 and curr.target_alive:
            # How much of this step's damage went to the selected target?
            # Use per-target HP tracking from the env.
            selected_dmg = max(0.0, prev.target_hp - curr.target_hp)
            # Only count selected_dmg if it's plausible (not from target switch)
            if selected_dmg <= total_damage + 1e-6:
                focus_ratio = selected_dmg / total_damage
            else:
                focus_ratio = 0.0  # target switch — don't credit
            focus_mult = 0.5 + focus_ratio  # 0.5 for spread, 1.5 for focused
            info["focus_ratio"] = focus_ratio

        r_damage = total_damage * self.w.damage_dealt * 100 * focus_mult
        info["damage_dealt"] = r_damage
        self._episode_damage_dealt += total_damage

        # Kill.
        r_kill = 0.0
        if prev.target_alive and not curr.target_alive:
            r_kill = self.w.kill_target
        info["kill"] = r_kill

        # ── Multi-Target Progression ──────────────────────────────

        # Retarget urgency: when selected target is dead but others live,
        # all positioning signals (range_closing, optimal_range, flanking)
        # go to zero because they're gated by target_alive. This penalty
        # fills the void — gives the agent a reason to select a new target.
        r_retarget = 0.0
        if not curr.target_alive and curr.alive_hostiles > 0:
            r_retarget = self.w.retarget_urgency
            info["retarget_urgency"] = r_retarget

        # Target low HP bonus: one-time reward when a target drops below
        # 30%. Creates a "finish them off" gradient. Gated on actual damage
        # to prevent phantom triggers when switching from high-HP to low-HP target.
        r_low_hp = 0.0
        if (curr.target_alive and curr.target_hp < 0.3
                and curr.total_damage_all_targets > 0
                and target_action not in self._targets_seen_low_hp):
            r_low_hp = self.w.target_low_hp_bonus
            self._targets_seen_low_hp.add(target_action)
            info["target_low_hp"] = r_low_hp

        # Damage taken.
        damage_taken_frac = max(0.0, prev.self_hp - curr.self_hp)
        r_taken = damage_taken_frac * self.w.take_damage * 100
        info["damage_taken"] = r_taken

        # Death.
        r_death = 0.0
        if prev.self_alive and not curr.self_alive:
            r_death = self.w.die
        info["death"] = r_death

        # Survival.
        r_alive = self.w.alive_per_step if curr.self_alive else 0.0
        info["alive"] = r_alive

        # Optimal range.
        r_range = self.w.in_optimal_range * engagement if curr.in_optimal_range else 0.0
        info["optimal_range"] = r_range
        
        # Range closing: reward approaching the optimal band when too far.
        # This gives a gradient toward engagement instead of relying on the
        # tiny in_optimal_range tick. Matches C++ BTTask_ScriptedCombat's
        # "distance > OptimalMax → MoveToActor" behaviour.
        #
        # NOT gated by engagement — this is how the agent GETS into engagement.
        # If gated, the agent can't hit → engagement drops → approach signal
        # dies → agent orbits aimlessly at max range.
        r_closing = 0.0
        r_out_of_range = 0.0
        if curr.target_alive and curr.target_distance > curr.active_optimal_max:
            # Reward closing distance (proportional to distance closed).
            # Stronger pull from far away — approaching from 3000 UU is more
            # valuable than fine-tuning at 1300 UU.
            if prev.target_distance > curr.active_optimal_max:
                dist_closed = prev.target_distance - curr.target_distance
                if dist_closed > 0:
                    overshoot = curr.target_distance - curr.active_optimal_max
                    far_bonus = 1.0 + min(overshoot / 800.0, 2.0)  # up to 3× at long range
                    r_closing = self.w.range_closing * min(dist_closed / 80.0, 1.0) * far_bonus
                    info["range_closing"] = r_closing

            # Continuous penalty for being out of range. Scales with distance
            # WITHOUT a cap — being 3000 UU away must feel much worse than 1500.
            # In a 4000 UU arena, the old 1000 UU cap made half the arena
            # feel identically "far", killing the gradient.
            overshoot = (curr.target_distance - curr.active_optimal_max)
            overshoot_frac = overshoot / 800.0  # no cap — scales indefinitely
            r_out_of_range = self.w.out_of_range_penalty * (0.3 + 0.7 * overshoot_frac)
            info["out_of_range"] = r_out_of_range

        # ── Ammo Management (stage 2+) ───────────────────────────

        r_ammo = 0.0
        if self.stage >= 2:
            r_ammo = self._compute_ammo_rewards(prev, curr, combat_action, info)

        # ── Invalid Action ───────────────────────────────────────

        r_invalid = 0.0
        if combat_action == CombatAction.FIRE and not prev.can_fire:
            r_invalid = self.w.invalid_action
        elif combat_action == CombatAction.RELOAD and prev.active_ammo_fraction >= 1.0:
            r_invalid = self.w.invalid_action
        info["invalid_action"] = r_invalid

        # ── Passive-in-Range Penalty ─────────────────────────────
        # The agent is in range, has LOS, has ammo, can fire — but
        # chose to do nothing. Every step of hesitation is wasted DPS.
        # Doesn't fire during reload (combat_action would be Reload)
        # or when action-locked (can_fire is False).
        r_passive = 0.0
        if (combat_action == CombatAction.NONE
                and curr.target_alive
                and prev.can_fire
                and prev.has_los
                and prev.target_in_range
                and prev.active_ammo_fraction > 0.0):
            r_passive = self.w.passive_in_range
            info["passive_in_range"] = r_passive

        # ── Anti-Degenerate (stage 3+) ───────────────────────────
        # Ramp applied so new anti-degenerate penalties don't shock the model.

        r_degen = 0.0
        if self.stage >= 3:
            r_degen = self._compute_anti_degenerate(prev, curr, move_action, combat_action, info)
            r_degen *= ramp

        # ── Weapon Selection (stage 4+) ──────────────────────────

        r_weapon = 0.0
        if self.stage >= 4:
            r_weapon = self._compute_weapon_selection(prev, curr, combat_action, info)
            r_weapon *= ramp * engagement  # [Fix] Was ramp only — ungated weapon
                                           # rewards let agent farm 32+/ep by
                                           # spam-firing at optimal range.

        # ── Flanking / Positioning (stage 3+) ────────────────────

        r_flank = 0.0
        if self.stage >= 3:
            r_flank = self._compute_flanking(prev, curr, combat_action, info)
            r_flank *= ramp * engagement

        # ── Archetype-Specific (stage 5+) ────────────────────────
        # THIS is the critical ramp. Archetype rewards appearing at full
        # strength crashed stage 5 training. Ramping over 10K steps lets
        # the value function gradually adapt to the new reward landscape.

        r_archetype = 0.0
        if self.stage >= 5:
            r_archetype = self._compute_archetype_rewards(prev, curr, action, info)
            r_archetype *= ramp * engagement  # [Fix] Was ramp only — ungated
                                              # archetype rewards (tank positioning,
                                              # ranged distance) farmable without
                                              # dealing damage.

        # ── Group Coordination (stage 6+) ────────────────────────

        r_group = 0.0
        if self.stage >= 6:
            r_group = self._compute_group_rewards(prev, curr, action, info)
            r_group *= ramp * engagement  # [Fix] Was ramp only — ally protection
                                          # and fire_at_ally_threat farmable by
                                          # positioning near allies without fighting.

        # ── Episode End Bonuses ──────────────────────────────────

        r_episode = 0.0
        if not curr.self_alive:
            r_episode += self.w.episode_loss
        if curr.alive_hostiles == 0:
            # Win bonus — scaled by how fast the agent won.
            # Faster kills = higher bonus (up to 1.5× base).
            speed_bonus = 1.0 + 0.5 * max(0.0, 1.0 - curr.step_count / 500.0)
            r_episode += self.w.episode_win * speed_bonus
        info["episode_end"] = r_episode

        # ── Damage Inactivity Penalty ────────────────────────────
        # Punish the agent for going too long without dealing damage.
        # This prevents the "run and hide forever" strategy.
        #
        # STAGE SCALING: Later stages have larger arenas with more obstacles.
        # The agent legitimately needs more time to reposition between kills.
        # Stage 1-4: 10 steps (2s), Stage 5: 15 steps (3s), Stage 6-7: 25 steps (5s).
        #
        # POST-KILL GRACE: After scoring a kill, the agent gets extra time
        # to acquire the next target without penalty. Navigating through
        # 16 obstacles to find the 3rd target is not "inactivity."

        r_inactivity = 0.0
        # Track all targets (matches the damage reward signal).
        if total_damage > 0:
            self._steps_since_damage = 0
        else:
            self._steps_since_damage += 1

        # Track kills for grace period.
        if curr.targets_killed > self._last_kill_count:
            self._steps_since_kill = 0
            self._last_kill_count = curr.targets_killed
        else:
            self._steps_since_kill += 1

        # Stage-scaled threshold. Base threshold is tight enough to
        # prevent episode-start idling but relaxed from the old 10 to
        # account for larger arenas. The post-kill grace (below) handles
        # the repositioning-between-kills case separately.
        if self.stage <= 4:
            inactivity_threshold = 10
        elif self.stage == 5:
            inactivity_threshold = 13
        else:
            inactivity_threshold = 10   # [Fix] Was 15 (3s). The larger-arena
                                        # argument doesn't hold — the agent was
                                        # using the slack to farm shaping between
                                        # chip-damage resets. Post-kill grace (35
                                        # steps) still covers legitimate reposition.

        # Post-kill grace: extend threshold generously after a kill.
        # Navigating through 16 obstacles to find target #3 takes time.
        if self._steps_since_kill < 30:
            inactivity_threshold = max(inactivity_threshold, 35)

        if self._steps_since_damage >= inactivity_threshold:
            # Escalating: gets worse the longer the drought lasts.
            idle_steps = self._steps_since_damage - inactivity_threshold
            escalation = 1.0 + min(idle_steps / 20.0, 2.0)
            r_inactivity = self.w.damage_per_step_min * escalation
            info["damage_inactivity"] = r_inactivity

        # ── Total ────────────────────────────────────────────────

        total = (
            r_damage + r_kill + r_taken + r_death + r_alive +
            r_range + r_ammo + r_invalid + r_degen + r_weapon +
            r_flank + r_archetype + r_group + r_episode + r_inactivity +
            r_closing + r_out_of_range + r_retarget + r_low_hp +
            r_passive
        )

        info["total"] = total
        return total, info

    # ═════════════════════════════════════════════════════════════
    #  Ammo Management
    # ═════════════════════════════════════════════════════════════

    def _compute_ammo_rewards(
        self, prev: CombatState, curr: CombatState,
        combat_action: int, info: Dict[str, float]
    ) -> float:
        r = 0.0

        # Reload behind cover vs in the open.
        if combat_action == CombatAction.RELOAD:
            if curr.behind_cover or not curr.has_los:
                r += self.w.reload_behind_cover
                info["reload_cover"] = self.w.reload_behind_cover
            else:
                r += self.w.reload_in_open
                info["reload_open"] = self.w.reload_in_open

        # Switched to a loaded weapon from an empty one.
        if combat_action in (CombatAction.SWITCH_0, CombatAction.SWITCH_1):
            if prev.active_ammo_fraction <= 0.0 and curr.active_ammo_fraction > 0.0:
                r += self.w.switch_to_loaded
                info["switch_to_loaded"] = self.w.switch_to_loaded

        # Wasted shot: fired but target was out of range or no LOS.
        # Exception: arc weapons that CAN clear the blocking cover are
        # exempt from the LOS penalty. Uses the height-aware check so
        # firing missiles at a target behind a 500 UU wall IS penalised,
        # but firing over 200 UU low cover is not.
        if combat_action == CombatAction.FIRE and prev.can_fire:
            if curr.total_damage_all_targets <= 0.0:
                los_blocked_for_weapon = (
                    not prev.has_los and not prev.can_arc_over_target_cover)
                if not prev.target_in_range or los_blocked_for_weapon:
                    r += self.w.wasted_shot
                    info["wasted_shot"] = self.w.wasted_shot
            else:
                r += self.w.fire_hit
                info["fire_hit"] = self.w.fire_hit

        # All ranged weapons empty simultaneously.
        if curr.all_ranged_empty and not prev.all_ranged_empty:
            r += self.w.all_empty_penalty
            info["all_empty"] = self.w.all_empty_penalty

        return r

    # ═════════════════════════════════════════════════════════════
    #  Weapon Selection (stage 4+)
    # ═════════════════════════════════════════════════════════════

    def _compute_weapon_selection(
        self, prev: CombatState, curr: CombatState,
        combat_action: int, info: Dict[str, float]
    ) -> float:
        r = 0.0
        dist = curr.target_distance

        # Is target in the active weapon's optimal band?
        in_band = (curr.active_optimal_min <= dist <= curr.active_optimal_max)

        # Fired while in optimal band — good weapon choice.
        # Requires actual damage dealt this step (not phantom target-switch).
        damage_this_step = curr.total_damage_all_targets
        if combat_action == CombatAction.FIRE and prev.can_fire:
            if in_band and damage_this_step > 0:
                r += self.w.fire_in_optimal_band
                info["fire_in_band"] = self.w.fire_in_optimal_band
            elif not in_band:
                # Fired outside optimal. Not as bad as wasted shot (that's
                # already penalised in ammo rewards), but suboptimal.
                r += self.w.fire_outside_optimal
                info["fire_out_of_band"] = self.w.fire_outside_optimal

        # Weapon switch happened — evaluate if it was a good swap.
        if curr.weapon_switched:
            new_in_band = (curr.active_optimal_min <= dist <= curr.active_optimal_max)
            old_in_band = (prev.active_optimal_min <= dist <= prev.active_optimal_max)

            if new_in_band and not old_in_band:
                # Swapped FROM out-of-band TO in-band. Smart.
                r += self.w.swap_to_better_range
                info["swap_better"] = self.w.swap_to_better_range
            elif not new_in_band and old_in_band:
                # Swapped FROM in-band TO out-of-band. Bad.
                r += self.w.swap_to_worse_range
                info["swap_worse"] = self.w.swap_to_worse_range

            # Smart reload swap: switched to an empty weapon whose range
            # covers the target, when the previous weapon didn't cover it.
            if (curr.active_ammo_fraction <= 0.0 and new_in_band
                    and not old_in_band):
                r += self.w.swap_reload_smart
                info["swap_reload_smart"] = self.w.swap_reload_smart

            # Tactical arc swap: switched TO an arc weapon when target is
            # behind low cover. This rewards the decision to equip the arc
            # launcher, not just the shot itself (which arc_over_cover_bonus
            # handles). Without this, the agent has to randomly discover
            # the switch→fire sequence with no reward on the switch step.
            if (curr.active_weapon_can_arc
                    and not prev.active_weapon_can_arc
                    and curr.target_behind_low_cover
                    and curr.can_arc_over_target_cover):
                r += self.w.arc_over_cover_bonus  # reuse same weight
                info["swap_to_arc_for_cover"] = self.w.arc_over_cover_bonus

        # Per-step penalty for holding the wrong weapon when a better
        # one is available. Checks if any other weapon's range is a
        # better fit for the current target distance.
        if not in_band and len(curr.other_weapon_ranges) > 0:
            for other_range in curr.other_weapon_ranges:
                other_mid = other_range * 0.7
                active_mid = (curr.active_optimal_min + curr.active_optimal_max) / 2
                if abs(other_mid - dist) < abs(active_mid - dist):
                    r += self.w.holding_wrong_weapon
                    info["holding_wrong_weapon"] = self.w.holding_wrong_weapon
                    break

        # Arc over cover: reward firing an arc weapon when LOS is blocked
        # by low cover that this weapon can clear.
        if (combat_action == CombatAction.FIRE and prev.can_fire
                and curr.can_arc_over_target_cover
                and curr.target_behind_low_cover):
            r += self.w.arc_over_cover_bonus
            info["arc_over_cover"] = self.w.arc_over_cover_bonus

        # Direct fire with LOS: reward for using a non-arc weapon when
        # there's a clear line of sight. Direct weapons are faster and
        # higher DPS — the agent should prefer them when LOS is clear.
        if (combat_action == CombatAction.FIRE and prev.can_fire
                and not curr.active_weapon_can_arc
                and curr.has_los and curr.target_in_range):
            r += self.w.direct_fire_with_los
            info["direct_fire_los"] = self.w.direct_fire_with_los

        # Penalty for holding an arc weapon when LOS is clear and a
        # direct weapon has ammo. Arc weapons are slow — use them for
        # their intended purpose (over cover), not as a fallback.
        if (curr.active_weapon_can_arc and curr.has_los
                and curr.has_direct_weapon_with_ammo
                and curr.target_in_range):
            r += self.w.holding_arc_with_los
            info["holding_arc_with_los"] = self.w.holding_arc_with_los

        # Mirror: penalty for holding a direct weapon when target is
        # behind low cover and LOS is blocked. Teaches either "switch
        # to arc weapon" (if one exists) or "reposition for LOS" (if not).
        # Doesn't check can_arc_over_target_cover because that field
        # only applies to the active weapon — which is direct here.
        if (not curr.active_weapon_can_arc
                and curr.target_behind_low_cover
                and not curr.has_los
                and curr.target_alive):
            r += self.w.holding_arc_with_los  # reuse same magnitude
            info["holding_direct_behind_cover"] = self.w.holding_arc_with_los

        return r

    # ═════════════════════════════════════════════════════════════
    #  Flanking / Positioning (stage 3+)
    # ═════════════════════════════════════════════════════════════

    def _compute_flanking(
        self, prev: CombatState, curr: CombatState,
        combat_action: int, info: Dict[str, float]
    ) -> float:
        r = 0.0

        if not curr.target_alive:
            return r

        # ── Per-step positional reward ───────────────────────────

        if curr.agent_behind_target:
            # Fully behind target (outside their vision cone).
            r += self.w.flank_behind_target
            info["flank_behind"] = self.w.flank_behind_target
        elif curr.agent_flanking:
            # At the target's side (partial advantage).
            r += self.w.flank_side_angle
            info["flank_side"] = self.w.flank_side_angle

        # ── Firing from a flanking position ──────────────────────
        # Bonus on top of damage reward for shots fired from advantageous angles.

        if combat_action == CombatAction.FIRE and prev.can_fire:
            if prev.agent_behind_target:
                r += self.w.flank_fire_from_behind
                info["flank_fire_behind"] = self.w.flank_fire_from_behind
            elif prev.agent_flanking:
                r += self.w.flank_fire_from_side
                info["flank_fire_side"] = self.w.flank_fire_from_side

        # ── Cover-based flanking ─────────────────────────────────
        # Reward for reaching a flanking position while using cover.
        # This teaches the agent to circle around obstacles to get
        # behind the target, rather than just walking straight at them.

        if (curr.agent_behind_target and not prev.agent_behind_target
                and curr.behind_cover):
            # Just achieved a flanking position while behind cover.
            r += self.w.flank_used_cover_to_flank
            info["flank_cover"] = self.w.flank_used_cover_to_flank

        # ── Lost flanking advantage ──────────────────────────────
        # Had a good position, target turned to face us.

        if (prev.agent_behind_target and not curr.agent_behind_target
                and not curr.agent_flanking):
            # Was behind, now target is facing us. Lost the advantage.
            r += self.w.flank_lost_position
            info["flank_lost"] = self.w.flank_lost_position

        return r

    # ═════════════════════════════════════════════════════════════
    #  Anti-Degenerate
    # ═════════════════════════════════════════════════════════════

    def _compute_anti_degenerate(
        self, prev: CombatState, curr: CombatState,
        move_action: int, combat_action: int,
        info: Dict[str, float]
    ) -> float:
        r = 0.0

        # Idle penalty: no combat action for 3+ consecutive steps.
        # Suppressed during post-kill grace — the agent can't fire at
        # a target it hasn't found yet.
        if combat_action == CombatAction.NONE:
            self._consecutive_idle += 1
        else:
            self._consecutive_idle = 0

        if self._consecutive_idle >= 3 and self._steps_since_kill > 15:
            r += self.w.idle_penalty
            info["idle_penalty"] = self.w.idle_penalty

        # Camping penalty: haven't moved significantly in 5+ steps.
        # Suppressed during post-kill grace — navigating through obstacles
        # to the next target involves pausing to adjust direction, which
        # looks like camping but is actually deliberate repositioning.
        dist_moved = np.linalg.norm(curr.self_position - self._last_position)
        if dist_moved < 50.0:
            self._consecutive_camping += 1
        else:
            self._consecutive_camping = 0
            self._last_position = curr.self_position.copy()

        if self._consecutive_camping >= 5 and self._steps_since_kill > 20:
            r += self.w.camping_penalty
            info["camping_penalty"] = self.w.camping_penalty

        # ── Wall hugging penalty ─────────────────────────────────
        # Penalise being near arena boundaries. This directly counters
        # corner camping since corners are where two walls meet.

        if curr.near_wall:
            r += self.w.wall_hugging_penalty
            info["wall_hugging"] = self.w.wall_hugging_penalty

        if curr.in_corner:
            # Corner = near two walls. Stacks with wall_hugging.
            r += self.w.corner_penalty
            info["corner_penalty"] = self.w.corner_penalty

        # ── Mobile combat bonuses ────────────────────────────────
        # Reward dealing damage while moving. Strafing (lateral movement)
        # gets a larger bonus than generic movement — this teaches the
        # agent to circle targets while shooting rather than running
        # away, turning, shooting, then running away again.

        if combat_action == CombatAction.FIRE and prev.can_fire:
            if curr.agent_speed > 150:
                r += self.w.mobile_fire_bonus
                info["mobile_fire"] = self.w.mobile_fire_bonus

            # Strafe-fire bonus: moving fast but distance to target barely
            # changed → lateral/perpendicular movement. This is the core
            # signal that teaches "circle the target while shooting."
            dist_change = abs(curr.target_distance - prev.target_distance)
            if curr.agent_speed > 100 and dist_change < 60:
                r += self.w.strafe_fire_bonus
                info["strafe_fire"] = self.w.strafe_fire_bonus

        # Spinning penalty: movement direction changed too often.
        self._recent_move_dirs.append(move_action)
        if len(self._recent_move_dirs) > 5:
            self._recent_move_dirs.pop(0)

        if len(self._recent_move_dirs) == 5:
            changes = sum(
                1 for i in range(1, 5)
                if self._recent_move_dirs[i] != self._recent_move_dirs[i - 1]
                and self._recent_move_dirs[i] != 0
            )
            if changes >= 3:
                r += self.w.spinning_penalty
                info["spinning_penalty"] = self.w.spinning_penalty
                
        # # Movement consistency: reward maintaining the same direction for 3+ steps.
        # # This breaks the flat reward landscape in open areas that causes jitter.
        # # The agent learns to commit to a direction rather than oscillating.
        # if len(self._recent_move_dirs) >= 3:
        #     last_three = self._recent_move_dirs[-3:]
        #     if (last_three[0] == last_three[1] == last_three[2]
        #             and last_three[0] != 0):  # Not "stop"
        #         r += 0.008
        #         info["move_consistency"] = 0.008
        
        # Movement consistency: reward ACTUAL sustained movement, not just
        # repeating the same action index (which rewards pressing into walls).
        dist_moved_this_step = np.linalg.norm(curr.self_position - prev.self_position)
        if dist_moved_this_step > 30.0:
            self._actual_move_streak += 1
        else:
            self._actual_move_streak = 0

        if self._actual_move_streak >= 3:
            # [Fix 3] Match the binary engagement gate from compute().
            engage_mult = 1.0 if self._steps_since_damage <= 4 else 0.0
            bonus = 0.008 * engage_mult
            r += bonus
            info["move_consistency"] = bonus

        # Ally collision.
        if curr.ally_overlap:
            r += self.w.ally_collision
            info["ally_collision"] = self.w.ally_collision

        return r

    # ═════════════════════════════════════════════════════════════
    #  Archetype-Specific
    # ═════════════════════════════════════════════════════════════

    def _compute_archetype_rewards(
        self, prev: CombatState, curr: CombatState,
        action: Tuple[int, int, int],
        info: Dict[str, float]
    ) -> float:
        # Ramp-in: archetype rewards start at 25% strength and reach
        # full strength after 50K steps (~250 episodes). This prevents
        # sudden policy collapse when archetype penalties are introduced
        # (the S5 drop to -7.5 was caused by full-strength "too close"
        # penalty instantly punishing the agent's existing strategy).
        ramp = min(1.0, self._global_step / 50_000) * 0.75 + 0.25

        if self.archetype == Archetype.MELEE:
            return self._melee_rewards(prev, curr, action, info) * ramp
        elif self.archetype == Archetype.RANGED:
            return self._ranged_rewards(prev, curr, action, info) * ramp
        elif self.archetype == Archetype.HEALER:
            return self._healer_rewards(prev, curr, action, info) * ramp
        elif self.archetype == Archetype.TANK:
            return self._tank_rewards(prev, curr, action, info) * ramp
        return 0.0

    def _melee_rewards(self, prev, curr, action, info) -> float:
        r = 0.0
        _, combat_action, _ = action
        melee_range = 250.0  # Approximate. Should come from env config.

        # Closing distance.
        if curr.target_distance < prev.target_distance and curr.target_distance > melee_range:
            r += self.w.melee_close_distance
            info["melee_closing"] = self.w.melee_close_distance

        # In melee range.
        if curr.target_distance <= melee_range:
            r += self.w.melee_in_range
            info["melee_in_range"] = self.w.melee_in_range

        # Retreating when close.
        if curr.target_distance > prev.target_distance and prev.target_distance <= melee_range * 1.5:
            r += self.w.melee_retreat_penalty
            info["melee_retreat"] = self.w.melee_retreat_penalty

        # Penalise hanging back with ranged weapon when ammo for melee approach.
        if curr.target_distance > melee_range * 3.0 and combat_action == CombatAction.NONE:
            r -= 0.01
            info["melee_too_passive"] = -0.01

        return r

    def _ranged_rewards(self, prev, curr, action, info) -> float:
        r = 0.0
        _, combat_action, _ = action
        danger_range = curr.active_weapon_range * 0.3

        # Too close.
        if curr.target_distance < danger_range:
            r += self.w.ranged_too_close
            info["ranged_too_close"] = self.w.ranged_too_close

        # Too far — but only in the "could hit but staying back" zone.
        # This covers the band between optimal_max and weapon_range.
        # Beyond weapon_range, the general out_of_range_penalty already
        # applies (and is not gated by archetype/stage), so we don't
        # stack them.
        if (curr.target_distance > curr.active_optimal_max
                and curr.target_distance <= curr.active_weapon_range):
            r += self.w.ranged_too_far
            info["ranged_too_far"] = self.w.ranged_too_far

        # Strafe success: dealt damage while moving laterally in the optimal
        # band. Uses total_damage_all_targets to avoid phantom damage.
        dist_change = abs(curr.target_distance - prev.target_distance)
        is_strafing = curr.agent_speed > 100 and dist_change < 60
        if is_strafing and curr.total_damage_all_targets > 0 and curr.in_optimal_range:
            r += self.w.ranged_strafe_fire
            info["ranged_strafe_fire"] = self.w.ranged_strafe_fire

        # Standing still when melee threat is closing.
        if (curr.target_distance < danger_range * 2.0
                and curr.self_speed < 0.1
                and curr.target_distance < prev.target_distance):
            r += self.w.ranged_standing_still_threat
            info["ranged_standing_still"] = self.w.ranged_standing_still_threat

        return r

    def _healer_rewards(self, prev, curr, action, info) -> float:
        r = 0.0
        _, combat_action, _ = action

        # Maintaining safe distance from threats.
        safe_distance = curr.active_weapon_range * 0.6
        if curr.target_distance > safe_distance:
            r += self.w.healer_maintain_distance
            info["healer_safe_distance"] = self.w.healer_maintain_distance

        # Ally in danger and healer not helping (placeholder — needs heal action tracking).
        if curr.ally_in_danger and combat_action == CombatAction.NONE:
            r -= 0.03
            info["healer_ignoring_ally"] = -0.03

        return r

    def _tank_rewards(self, prev, curr, action, info) -> float:
        r = 0.0
        _, combat_action, _ = action

        # Body-blocking: positioned between threat and ally.
        if curr.self_between_threat_and_ally:
            r += self.w.tank_body_block
            info["tank_body_block"] = self.w.tank_body_block

        # Protecting low-HP ally.
        if curr.ally_in_danger and curr.self_between_threat_and_ally:
            r += self.w.tank_protect_low_hp
            info["tank_protect_low_hp"] = self.w.tank_protect_low_hp

        # Suppression fire.
        if combat_action == CombatAction.FIRE and curr.can_fire:
            r += self.w.tank_suppression
            info["tank_suppression"] = self.w.tank_suppression

        # Absorbing damage (self HP decreased, which means threats are hitting us not allies).
        damage_taken = max(0.0, prev.self_hp - curr.self_hp)
        if damage_taken > 0 and curr.alive_allies > 0:
            r += damage_taken * self.w.tank_absorb_damage * 100
            info["tank_absorb"] = damage_taken * self.w.tank_absorb_damage * 100

        return r

    # ═════════════════════════════════════════════════════════════
    #  Group Coordination (stage 6+)
    # ═════════════════════════════════════════════════════════════

    def _compute_group_rewards(
        self, prev: CombatState, curr: CombatState,
        action: Tuple[int, int, int],
        info: Dict[str, float]
    ) -> float:
        r = 0.0
        _, combat_action, _ = action

        # Target diversity: track unique targets hit this episode.
        # The actual bonus is applied at episode end via compute_episode_end_bonus().

        # ── Ally Protection (all archetypes) ─────────────────────
        # These rewards teach every archetype to be team-aware, not
        # just tanks. A ranged DPS that ignores a dying ally is bad.

        # Protecting a low-HP ally: positioned between threat and ally.
        # Applies to all archetypes, not just tank. Any ally stepping
        # between a threat and a wounded teammate is good teamplay.
        if curr.ally_in_danger and curr.self_between_threat_and_ally:
            r += self.w.protect_low_hp_ally
            info["protect_ally"] = self.w.protect_low_hp_ally

        # Firing at a target that's threatening an ally.
        # "Threatening" = target is focusing on an ally (not us).
        # Drawing aggro by shooting them is good for ally survival.
        if (combat_action == CombatAction.FIRE and prev.can_fire
                and curr.ally_in_danger):
            # The agent is shooting while an ally is in danger.
            # Even if they're not shooting the exact threat, applying
            # pressure helps. But check: if we're hitting a target
            # and ally is in danger, that's helpful regardless.
            damage_dealt = curr.total_damage_all_targets
            if damage_dealt > 0:
                r += self.w.fire_at_ally_threat
                info["fire_at_ally_threat"] = self.w.fire_at_ally_threat

        # Ally died nearby — could we have helped?
        if curr.ally_just_died and curr.nearest_ally_distance < 1500:
            r += self.w.ally_died_nearby
            info["ally_died_nearby"] = self.w.ally_died_nearby

        # Ally collision is already handled in anti-degenerate.

        return r

    def compute_episode_end_bonus(self, final_state: CombatState,
                                   truncated: bool = False) -> Tuple[float, Dict[str, float]]:
        """Called once when the episode ends. Adds episode-level rewards."""
        info: Dict[str, float] = {}
        r = 0.0

        # Timeout penalty: episode ended without killing all targets.
        # This is the key fix for "survive forever" strategies.
        # [Defense-in-depth] Scale penalty with episode length so longer
        # timeouts are punished harder. At 200 steps, penalty = -8 (base).
        # At 400 steps, penalty = -16. At 1000 steps, penalty = -40.
        # This makes timeout unprofitable regardless of max_steps setting.
        if truncated and final_state.alive_hostiles > 0:
            length_scale = max(1.0, final_state.step_count / 200.0)
            timeout_penalty = self.w.episode_timeout * length_scale
            r += timeout_penalty
            info["timeout_penalty"] = timeout_penalty

        # Surviving targets penalty: per-target cost for each target left
        # alive at episode end. Creates a clear gradient:
        #   killed 3/3 → penalty 0   (win)
        #   killed 2/3 → penalty -8  (close)
        #   killed 1/3 → penalty -16 (poor)
        #   killed 0/3 → penalty -24 (terrible)
        # This makes reward correlate with kill count, which correlates
        # with win rate.
        if final_state.alive_hostiles > 0:
            surviving_penalty = self.w.surviving_target_penalty * final_state.alive_hostiles
            r += surviving_penalty
            info["surviving_targets"] = surviving_penalty

        # Target diversity bonus.
        if self.stage >= 6 and len(final_state.targets_attacked) > 1:
            r += self.w.target_diversity
            info["target_diversity"] = self.w.target_diversity

        # Minimum damage threshold penalty.
        min_damage_threshold = 0.05
        if self._episode_damage_dealt < min_damage_threshold and final_state.self_alive:
            r += self.w.min_damage_penalty
            info["min_damage_penalty"] = self.w.min_damage_penalty

        info["episode_total_damage"] = self._episode_damage_dealt
        info["episode_targets_hit"] = len(final_state.targets_attacked)

        return r, info


# ─────────────────────────────────────────────────────────────────
#  Reward Budget Analysis (dynamic — replaces stale inline comments)
# ─────────────────────────────────────────────────────────────────

def compute_reward_budget(weights: RewardWeights = None,
                          stage: int = 7) -> dict:
    """Compute expected reward breakdown for canonical episodes.
    
    Run this after any weight change to verify the reward landscape.
    Prints the actual budget with current weights so the inline
    comments don't go stale.
    """
    w = weights or RewardWeights()
    num_targets = 4 if stage >= 7 else 3
    max_steps = {1: 500, 2: 500, 3: 1000, 4: 400, 5: 1000, 6: 1000, 7: 1200}.get(stage, 1000)

    # Winning episode: kill all targets, take 60% HP damage.
    # Assume ~150 steps to win.
    win_steps = 150
    win = {
        "damage_dealt": w.damage_dealt * 100 * num_targets,
        "kills": w.kill_target * num_targets,
        "target_low_hp": w.target_low_hp_bonus * num_targets,
        "episode_win": w.episode_win * 1.3,               # speed bonus ~1.3×
        "surviving_targets": 0,                             # all dead
        "damage_taken": w.take_damage * 60,                # 60% HP lost
        "alive_cost": w.alive_per_step * win_steps,        # survival cost
        "shaping_estimate": 8.0,
    }
    win["total"] = sum(win.values())

    # Timeout episode: chip damage, survive, no kills.
    timeout = {
        "damage_dealt": w.damage_dealt * 100 * 0.4,       # 40% of one target
        "kills": 0,
        "target_low_hp": w.target_low_hp_bonus * 1,       # got one target low
        "timeout": w.episode_timeout * max(1.0, max_steps / 200.0),  # scaled
        "surviving_targets": w.surviving_target_penalty * num_targets,
        "retarget_penalty_est": w.retarget_urgency * 20,  # ~20 steps on dead target
        "damage_taken": w.take_damage * 50,                # 50% HP lost
        "alive_cost": w.alive_per_step * max_steps,        # survival cost for FULL episode
        "shaping_estimate": 5.0,
    }
    timeout["total"] = sum(timeout.values())

    # Death episode: killed 1 target, damaged another 50%, died.
    # Assume ~80 steps (engaged aggressively, died relatively fast).
    death_steps = 80
    death = {
        "damage_dealt": w.damage_dealt * 100 * 1.5,
        "kills": w.kill_target * 1,
        "target_low_hp": w.target_low_hp_bonus * 1,
        "die": w.die,
        "episode_loss": w.episode_loss,
        "surviving_targets": w.surviving_target_penalty * (num_targets - 1),
        "damage_taken": w.take_damage * 100,               # 100% HP lost (died)
        "alive_cost": w.alive_per_step * death_steps,
        "shaping_estimate": 3.0,
    }
    death["total"] = sum(death.values())

    # Kill most, timeout on last. Assume ~300 steps.
    partial_steps = 300
    partial = {
        "damage_dealt": w.damage_dealt * 100 * (num_targets - 0.7),
        "kills": w.kill_target * (num_targets - 1),
        "target_low_hp": w.target_low_hp_bonus * num_targets,
        "timeout": w.episode_timeout * max(1.0, partial_steps / 200.0),  # scaled
        "surviving_targets": w.surviving_target_penalty * 1,
        "damage_taken": w.take_damage * 70,                # 70% HP lost
        "alive_cost": w.alive_per_step * partial_steps,
        "shaping_estimate": 6.0,
    }
    partial["total"] = sum(partial.values())

    print(f"=== Reward Budget (current weights, {num_targets} targets, stage {stage}) ===")
    for name, scenario in [("WIN", win), (f"KILL-{num_targets-1}-TIMEOUT", partial),
                            ("DEATH-1-KILL", death), ("TIMEOUT-0-KILL", timeout)]:
        print(f"\n{name}: total = {scenario['total']:+.1f}")
        for k, v in scenario.items():
            if k != "total":
                print(f"  {k:25s} {v:+.1f}")

    print(f"\nKey ratio: WIN ({win['total']:+.1f}) >> "
          f"PARTIAL ({partial['total']:+.1f}) >> "
          f"DEATH ({death['total']:+.1f}) >> "
          f"TIMEOUT ({timeout['total']:+.1f})")

    return {"win": win, "partial": partial, "death": death, "timeout": timeout}


# ─────────────────────────────────────────────────────────────────
#  Curriculum Stage Presets
# ─────────────────────────────────────────────────────────────────

def get_curriculum_description(stage: int) -> str:
    """Human-readable description of what each stage trains."""
    descriptions = {
        1: "1v1 melee vs stationary target — close distance + attack",
        2: "1v1 single weapon vs stationary — fire + reload + range",
        3: "1v1 single weapon vs moving — track + kite + anti-degenerate",
        4: "1v1 multi-weapon vs moving — ammo management + switching",
        5: "1v1 per archetype vs moving — archetype-specific behaviours",
        6: "2v1 enemies vs player — coordination emerges, don't stack",
        7: "4v4 squad vs party — full group coordination",
    }
    return descriptions.get(stage, f"Unknown stage {stage}")


def get_reward_function_for_stage(
    stage: int,
    archetype: str = "ranged",
    weight_overrides: Optional[Dict[str, float]] = None,
) -> CombatRewardFunction:
    """Factory: create a reward function configured for a curriculum stage."""

    weights = RewardWeights()

    # Stage-specific weight adjustments.
    if stage <= 1:
        # Basics only: damage and survival. No ammo or positioning complexity.
        # Keep alive_per_step positive — agent must learn to approach before
        # we penalise survival. The -0.02 default would punish early exploration.
        weights.alive_per_step = 0.005
        weights.in_optimal_range = 0.0  # Don't worry about range yet.

    elif stage == 2:
        # Introduce ammo management. Zero survival reward — not positive
        # (farming risk) but not negative (still learning fire/reload).
        weights.alive_per_step = 0.0
        weights.reload_behind_cover = 0.1  # Lighter cover bonus (no cover in stage 2 arena).
        weights.reload_in_open = 0.0       # Don't penalise yet.

    elif stage <= 4:
        # Full ammo management + anti-degenerate.
        # Uses default alive_per_step = -0.02 (survival cost).
        pass

    elif stage == 5:
        # Archetype-specific rewards activate.
        pass  # Use defaults.

    elif stage >= 6:
        # Group rewards activate. Keep damage_dealt at full strength —
        # coordination signals are additive, they don't need "room."
        # The old 0.08 value cut the per-kill gradient by 68%, which
        # directly caused passive behaviour in stages 6-7.
        weights.ally_collision = -0.04  # Stronger stacking penalty in groups.

    # Apply any manual overrides.
    if weight_overrides:
        for key, value in weight_overrides.items():
            if hasattr(weights, key):
                setattr(weights, key, value)

    return CombatRewardFunction(
        archetype=archetype,
        curriculum_stage=stage,
        weights=weights,
    )


# ─────────────────────────────────────────────────────────────────
#  Example Integration with Gymnasium Environment
# ─────────────────────────────────────────────────────────────────

"""
USAGE IN YOUR GYM ENVIRONMENT:

class CombatEnv(gymnasium.Env):

    def __init__(self, archetype="ranged", curriculum_stage=1):
        super().__init__()
        self.reward_fn = get_reward_function_for_stage(
            stage=curriculum_stage,
            archetype=archetype,
        )
        # ... define observation_space, action_space ...

    def reset(self, **kwargs):
        self.reward_fn.reset()
        # ... reset sim state ...
        return obs, info

    def step(self, action):
        move, combat, target = self._decode_action(action)

        # Snapshot state before action.
        prev_state = self._build_combat_state()

        # Apply action to simulation.
        self._apply_action(move, combat, target)
        self._step_simulation()

        # Snapshot state after action.
        curr_state = self._build_combat_state()

        # Compute reward.
        reward, reward_info = self.reward_fn.compute(
            prev_state, (move, combat, target), curr_state)

        # Check episode end.
        done = not curr_state.self_alive or curr_state.alive_hostiles == 0
        truncated = self._step_count >= self.max_steps

        if done or truncated:
            end_bonus, end_info = self.reward_fn.compute_episode_end_bonus(curr_state)
            reward += end_bonus
            reward_info.update(end_info)

        obs = self._build_observation()
        return obs, reward, done, truncated, reward_info

    def _build_combat_state(self) -> CombatState:
        '''Build a CombatState snapshot from the sim's internal state.'''
        return CombatState(
            self_hp=self.agent.hp / self.agent.max_hp,
            self_alive=self.agent.hp > 0,
            self_position=np.array(self.agent.position),
            target_hp=self.target.hp / self.target.max_hp,
            target_alive=self.target.hp > 0,
            target_distance=self._distance(self.agent, self.target),
            has_los=self._check_los(self.agent, self.target),
            active_ammo_fraction=self.agent.weapons[self.agent.active_weapon].ammo_frac,
            # ... fill all fields from sim state ...
        )
"""