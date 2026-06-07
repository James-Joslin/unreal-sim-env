"""
combat_sim.py — Python combat simulation for PPO training.

Replicates the game's combat math in a lightweight 2D environment
that runs 100-1000x faster than headless UE. Trains are done here
first, then fine-tuned in headless UE to close the sim-to-real gap.

WHAT IT REPLICATES
    - 2D continuous movement with rectangular obstacle collision
    - Weapon system: ammo, cooldown, reload, wind-up, fire, switch, melee
    - Damage formula matching CombatPipeline.cpp:
        Outgoing = (BaseDamage + AttackStat) × Crit
        After defence = Outgoing × 100 / (Defence + 100)
        Barrier absorbs first. Minimum damage = 1.
    - Line of sight via 2D raycasting against obstacles
    - Cover detection (low vs full height obstacles)
    - Multiple agents and targets
    - The 215-float observation vector from NeuralCombatTypes.h

WHAT IT SIMPLIFIES
    - No navmesh — uses simple AABB collision
    - No status effects (Poison, Scold, etc.) — can add later
    - No charms/conditional damage — enemies don't use charms
    - Player targets use simple scripted behaviour, not real player input
    - No vertical dimension — pure 2D

USAGE
    env = CombatEnv(CombatEnvConfig(
        num_enemies=1,
        num_targets=1,
        arena_size=2000.0,
        curriculum_stage=3,
    ))
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward import CombatRewardFunction, CombatState, get_reward_function_for_stage

# ─────────────────────────────────────────────────────────────────
#  Constants (match NeuralCombatTypes.h)
# ─────────────────────────────────────────────────────────────────

OBS_SIZE = 249
                # [198-200] 2nd projectile (dist,dirX,dirY)
                # [201-203] 3rd projectile (dist,dirX,dirY)
                # [204]     incoming threat count
                # [205-208] can_hit_target per weapon slot (4)
                # [209]     total ammo fraction
                # [210]     targets killed fraction
                # [211-214] arc clearance per weapon slot (MaxArcableObstacleHeight / 3000)
MOVEMENT_ACTIONS = 9
COMBAT_ACTIONS = 8  # Added Dodge (action 7)
TARGET_ACTIONS = 5

DEFENCE_CONSTANT = 100.0
MIN_DAMAGE = 1.0
AGENT_BODY_RADIUS = 30.0  # UU — prevents agent from visually clipping into obstacles

# ── Character type normalised encoding (maps to ECharacterType) ──
CHARACTER_TYPE_MAP = {
    "knight": 0.0, "rogue": 0.2, "ranger": 0.4,
    "mage": 0.6, "healer": 0.8, "none": 0.5,
}

# ── Player pattern tracking ──────────────────────────────────────
class PlayerPatternTracker:
    """Tracks EMAs of player behavior for observation encoding."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.aggression = 0.0       # Fire rate EMA
        self.evasion = 0.0          # Dodge frequency EMA
        self.predictability = 0.5   # Movement entropy (0=predictable, 1=random)
        self.preferred_range = 0.5  # Normalised engagement distance
        self.mana_burn_rate = 0.0   # Mana spend rate EMA
        # Direction histogram for predictability (8 bins).
        self._dir_hist = np.zeros(8, dtype=np.float32)
        self._fire_count = 0
        self._dodge_count = 0
        self._tick_count = 0

    def update(self, targets, agent_pos, dt, arena_half):
        """Called once per sim tick with all target data."""
        self._tick_count += 1
        total_aggression = 0.0
        total_range = 0.0
        total_mana_rate = 0.0
        n_alive = 0

        for t in targets:
            if not t.alive:
                continue
            n_alive += 1

            # Aggression: was this target attacking this tick?
            if t.commitment > 0.01:
                total_aggression += 1.0

            # Range to agent.
            dist = np.linalg.norm(t.pos - agent_pos)
            total_range += dist / (arena_half * 2)

            # Mana burn: how fast are they spending?
            if t.max_mana > 0:
                total_mana_rate += (1.0 - t.mana_fraction())

            # Movement direction for predictability.
            speed = np.linalg.norm(t.velocity)
            if speed > 10.0:
                angle = math.atan2(t.velocity[1], t.velocity[0])
                bin_idx = int((angle + math.pi) / (math.pi / 4)) % 8
                self._dir_hist[bin_idx] += 1.0

        if n_alive > 0:
            a = self.alpha
            self.aggression = self.aggression * (1 - a) + (total_aggression / n_alive) * a
            self.preferred_range = self.preferred_range * (1 - a) + (total_range / n_alive) * a
            self.mana_burn_rate = self.mana_burn_rate * (1 - a) + (total_mana_rate / n_alive) * a

        # Decay direction histogram and compute entropy.
        self._dir_hist *= 0.99
        total = self._dir_hist.sum()
        if total > 1e-6:
            probs = self._dir_hist / total
            entropy = 0.0
            for p in probs:
                if p > 0.001:
                    entropy -= p * math.log(p)
            self.predictability = min(1.0, entropy / math.log(8))

    def as_array(self) -> np.ndarray:
        return np.array([
            self.aggression,
            self.evasion,
            self.predictability,
            self.preferred_range,
            self.mana_burn_rate,
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
#  Archetype / Action Enums
# ─────────────────────────────────────────────────────────────────

class Archetype(IntEnum):
    RANGED = 0; MELEE = 1; HEALER = 2; TANK = 3

class CombatAction(IntEnum):
    NONE = 0; FIRE = 1; RELOAD = 2
    SWITCH_0 = 3; SWITCH_1 = 4; MELEE = 5; BLOCK = 6; DODGE = 7


# ─────────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class WeaponSlot:
    """Mirrors FEnemyWeaponSlot from C++."""
    name: str = "Rifle"
    base_damage: float = 10.0
    weapon_range: float = 1500.0
    optimal_min: float = 600.0
    optimal_max: float = 1200.0
    max_ammo: int = 20
    current_ammo: int = 20
    reload_time: float = 2.0
    fire_cooldown: float = 0.3
    projectile_speed: float = 3000.0   # UU/s — all weapons fire projectiles
    wind_up_time: float = 0.0
    can_arc: bool = False
    max_arc_height: float = 400.0     # MaxArcableObstacleHeight — tallest obstacle we can arc over (0=unlimited)
    min_arc_clearance: float = 200.0  # MinArcClearance — vertical clearance needed above obstacle
    is_ranged: bool = True

    # Runtime state.
    reload_remaining: float = 0.0
    cooldown_remaining: float = 0.0
    is_reloading: bool = False

    def has_ammo(self) -> bool:
        return self.max_ammo == 0 or self.current_ammo > 0

    def is_ready(self) -> bool:
        return not self.is_reloading and self.cooldown_remaining <= 0

    def can_arc_over_height(self, obstacle_height: float) -> bool:
        """Matches C++ FEnemyWeaponSlot::CanArcOverHeight()."""
        if not self.can_arc:
            return False
        if self.max_arc_height > 0 and obstacle_height > self.max_arc_height:
            return False
        return True

    def ammo_fraction(self) -> float:
        return self.current_ammo / self.max_ammo if self.max_ammo > 0 else 1.0

    def tick(self, dt: float):
        if self.is_reloading:
            self.reload_remaining -= dt
            if self.reload_remaining <= 0:
                self.current_ammo = self.max_ammo
                self.is_reloading = False
                self.reload_remaining = 0.0
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= dt


@dataclass
class MeleeConfig:
    damage: float = 30.0
    range: float = 200.0
    cooldown: float = 1.0
    cooldown_remaining: float = 0.0

    def tick(self, dt: float):
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= dt


@dataclass
class Obstacle:
    """Axis-aligned rectangle in the arena."""
    x: float; y: float       # centre
    hw: float; hh: float     # half-width, half-height
    height: float = 300.0    # full = 300+, low cover = 100-200

    def contains(self, px, py) -> bool:
        return (abs(px - self.x) < self.hw and abs(py - self.y) < self.hh)

    def contains_circle(self, px, py, radius: float) -> bool:
        """True if a circle at (px,py) with given radius overlaps this AABB."""
        # Closest point on AABB to the circle centre.
        cx = max(self.x - self.hw, min(px, self.x + self.hw))
        cy = max(self.y - self.hh, min(py, self.y + self.hh))
        dx = px - cx
        dy = py - cy
        return (dx * dx + dy * dy) < (radius * radius)

    def push_out_circle(self, px, py, radius: float):
        """Push a circle at (px,py) out of this AABB. Returns new (px, py)."""
        if not self.contains_circle(px, py, radius):
            return px, py
        # Find the nearest edge and push along the shortest axis.
        # Expand AABB by radius for the effective collision boundary.
        ehw = self.hw + radius
        ehh = self.hh + radius
        dx = px - self.x
        dy = py - self.y
        # Penetration depth along each axis.
        pen_x = ehw - abs(dx)
        pen_y = ehh - abs(dy)
        if pen_x < pen_y:
            px = self.x + ehw * (1 if dx > 0 else -1)
        else:
            py = self.y + ehh * (1 if dy > 0 else -1)
        return px, py

    def aabb(self):
        return (self.x - self.hw, self.y - self.hh,
                self.x + self.hw, self.y + self.hh)

class ThreatTable:
    """Tracks accumulated damage from each target for priority scoring.
    
    [Audit §1.3] Mirrors C++ EnemyPerceptionComponent::GetThreatFor which
    tracks per-target damage dealt to the agent. Used by EvaluateTargetPriority
    for the DamageThreatWeight component.
    """

    def __init__(self, decay_rate: float = 5.0):
        self.threats: dict = {}  # target_id -> accumulated_damage
        self.decay_rate = decay_rate

    def record_damage(self, target_id: int, damage: float):
        self.threats[target_id] = self.threats.get(target_id, 0.0) + damage

    def get_threat(self, target_id: int) -> float:
        return self.threats.get(target_id, 0.0)

    def get_max_threat(self) -> float:
        return max(self.threats.values()) if self.threats else 0.0

    def get_normalised_threat(self, target_id: int) -> float:
        max_t = self.get_max_threat()
        if max_t <= 0:
            return 0.0
        return self.get_threat(target_id) / max_t

    def decay(self, dt: float):
        decay = self.decay_rate * dt
        to_remove = []
        for tid in self.threats:
            self.threats[tid] -= decay
            if self.threats[tid] <= 0:
                to_remove.append(tid)
        for tid in to_remove:
            del self.threats[tid]

    def reset(self):
        self.threats.clear()


@dataclass
class SimProjectile:
    """A projectile traveling through the arena.

    Matches the C++ projectile hierarchy:
    - Physics: straight line at constant speed (EnemyProjectile_Physics)
    - Beam: very fast straight line, effectively instant but still travels
            (EnemyProjectile_Beam — 8000+ UU/s)
    - Arc: quadratic bezier curve from start through apex to target
           (EnemyProjectile_Arc — missiles, mortars)

    All projectiles check collision each tick via distance to target.
    No hitscan — everything physically moves.
    """
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    speed: float = 3000.0
    damage: float = 10.0
    attack_stat: float = 5.0
    crit_chance: float = 0.0
    crit_multiplier: float = 1.5
    alive: bool = True
    age: float = 0.0
    max_lifetime: float = 5.0

    # Who fired this and who it's aimed at.
    is_agent_projectile: bool = True  # True = agent fired, False = target fired
    source_id: int = -1               # target_id of source (for target projectiles)
    target_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))  # aim point at fire time

    # Arc projectile fields.
    is_arc: bool = False
    max_arc_height: float = 400.0     # From weapon's MaxArcableObstacleHeight
    arc_start: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    arc_apex: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    arc_end: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    arc_flight_time: float = 1.0
    arc_elapsed: float = 0.0
    arc_impact_radius: float = 0.0  # 0 = point damage, >0 = AoE

    # Collision.
    hit_radius: float = 60.0         # Distance threshold for hit detection.
    hit_actor: object = None          # Set on hit for rendering flash.
    did_hit: bool = False

    def tick(self, dt: float, targets, agent, obstacles, arena_half: float):
        """Advance projectile one tick. Returns list of (actor, damage, crit) hits."""
        if not self.alive:
            return []

        self.age += dt

        if self.is_arc:
            return self._tick_arc(dt, targets, agent, obstacles, arena_half)
        else:
            return self._tick_straight(dt, targets, agent, obstacles, arena_half)

    def _tick_straight(self, dt, targets, agent, obstacles, arena_half):
        """Physics/beam projectile — straight line travel with swept collision."""
        hits = []

        old_pos = self.pos.copy()
        self.pos = self.pos + self.velocity * dt

        # Check out of bounds.
        if (abs(self.pos[0]) > arena_half * 1.1
                or abs(self.pos[1]) > arena_half * 1.1):
            self.alive = False
            return hits

        # Check lifetime.
        if self.age > self.max_lifetime:
            self.alive = False
            return hits

        # Swept obstacle collision (use existing ray-AABB, matches UE sweep).
        for obs in obstacles:
            if _ray_aabb_intersect(old_pos, self.pos, obs):
                # Arc weapons can clear obstacles up to their max_arc_height.
                # Matches C++ CanArcOverHeight: checks bCanArcOverCover AND
                # MaxArcableObstacleHeight. Straight weapons always blocked.
                if self.is_arc:
                    can_clear = (self.max_arc_height <= 0
                                 or obs.height <= self.max_arc_height)
                    if not can_clear:
                        self.alive = False
                        return hits
                    # Arc weapon clears this obstacle — continue flying.
                else:
                    self.alive = False
                    return hits

        # Swept target collision — closest point on segment to target centre.
        # This matches UE's UProjectileMovementComponent swept sphere check.
        seg = self.pos - old_pos
        seg_len_sq = float(np.dot(seg, seg))

        if self.is_agent_projectile:
            check_list = targets
        else:
            check_list = [agent]

        for actor in check_list:
            if hasattr(actor, 'alive') and not actor.alive:
                continue
            if hasattr(actor, 'is_dodging') and actor.is_dodging:
                # Dodge invulnerability (agent only).
                continue

            if seg_len_sq > 0.01:
                to_target = actor.pos - old_pos
                t_param = np.clip(
                    float(np.dot(to_target, seg)) / seg_len_sq, 0.0, 1.0)
                closest = old_pos + seg * t_param
            else:
                closest = self.pos

            dist = np.linalg.norm(closest - actor.pos)
            if dist < self.hit_radius:
                # Place projectile at hit point (for rendering).
                self.pos = closest

                # Apply damage.
                if self.is_agent_projectile:
                    dmg, actor.barrier, was_crit = compute_damage(
                        self.damage, self.attack_stat,
                        actor.defence, actor.barrier,
                        self.crit_chance, self.crit_multiplier)
                else:
                    dmg, actor.barrier, was_crit = compute_damage(
                        self.damage, self.attack_stat,
                        actor.defence, actor.barrier,
                        self.crit_chance, self.crit_multiplier)  # [Audit §5.4]

                actor.hp -= dmg
                if actor.hp <= 0:
                    actor.hp = 0
                    actor.alive = False

                hits.append((actor, dmg, was_crit))
                self.alive = False
                self.did_hit = True
                self.hit_actor = actor
                return hits

        return hits

    def _tick_arc(self, dt, targets, agent, obstacles, arena_half):
        """Arc projectile — follows bezier curve."""
        hits = []

        self.arc_elapsed += dt
        progress = min(1.0, self.arc_elapsed / max(self.arc_flight_time, 0.01))

        # Quadratic bezier: P = (1-t)²·Start + 2(1-t)t·Apex + t²·End
        t = progress
        omt = 1.0 - t
        self.pos = (omt * omt * self.arc_start
                    + 2.0 * omt * t * self.arc_apex
                    + t * t * self.arc_end)

        # Update velocity direction for rendering trail.
        if t < 0.99:
            next_t = min(1.0, t + 0.01)
            next_omt = 1.0 - next_t
            next_pos = (next_omt * next_omt * self.arc_start
                        + 2.0 * next_omt * next_t * self.arc_apex
                        + next_t * next_t * self.arc_end)
            delta = next_pos - self.pos
            d = np.linalg.norm(delta)
            if d > 0.1:
                self.velocity = delta / d * self.speed

        # Impact at end of flight.
        if progress >= 1.0:
            self.alive = False
            self.did_hit = True

            if self.arc_impact_radius > 0:
                # AoE damage — hit everything in radius.
                check_list = targets if self.is_agent_projectile else [agent]
                for actor in (check_list if isinstance(check_list, list) else [check_list]):
                    if hasattr(actor, 'alive') and not actor.alive:
                        continue
                    dist = np.linalg.norm(self.pos - actor.pos)
                    if dist < self.arc_impact_radius:
                        falloff = 1.0 - (dist / self.arc_impact_radius) * 0.5
                        dmg, actor.barrier, was_crit = compute_damage(
                            self.damage * falloff, self.attack_stat,
                            actor.defence, actor.barrier,
                            self.crit_chance, self.crit_multiplier)
                        actor.hp -= dmg
                        if actor.hp <= 0:
                            actor.hp = 0
                            actor.alive = False
                        hits.append((actor, dmg, was_crit))
            else:
                # Point damage — check nearest valid target.
                check_targets = targets if self.is_agent_projectile else [agent]
                for actor in (check_targets if isinstance(check_targets, list) else [check_targets]):
                    if hasattr(actor, 'alive') and not actor.alive:
                        continue
                    dist = np.linalg.norm(self.pos - actor.pos)
                    if dist < self.hit_radius * 1.5:  # Slightly generous for arc landing
                        dmg, actor.barrier, was_crit = compute_damage(
                            self.damage, self.attack_stat,
                            actor.defence, actor.barrier,
                            self.crit_chance, self.crit_multiplier)
                        actor.hp -= dmg
                        if actor.hp <= 0:
                            actor.hp = 0
                            actor.alive = False
                        hits.append((actor, dmg, was_crit))
                        self.hit_actor = actor
                        break

        return hits

# ─────────────────────────────────────────────────────────────────
#  Agent (enemy being trained) and Target (player party member)
# ─────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """The enemy robot being trained."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    spawn_pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    combat_leash_range: float = 2000.0  # [Audit §1.9] Matches C++ CombatLeashRange default
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    max_speed: float = 450.0
    max_acceleration: float = 2048.0        # UU/s², matches UE CharacterMovementComponent
    braking_deceleration: float = 2048.0    # UU/s², how fast we stop
    hp: float = 100.0
    max_hp: float = 100.0
    defence: float = 20.0
    barrier: float = 0.0
    attack_stat: float = 5.0
    crit_chance: float = 0.05
    crit_multiplier: float = 1.5
    alive: bool = True

    archetype: int = Archetype.RANGED
    weapons: List[WeaponSlot] = field(default_factory=list)
    active_weapon: int = 0
    melee: MeleeConfig = field(default_factory=MeleeConfig)

    # For observation: tracking.
    combat_time: float = 0.0
    targets_hit: set = field(default_factory=set)

    # Wind-up state (weapon charges before firing).
    is_winding_up: bool = False
    wind_up_remaining: float = 0.0
    _pending_fire: Optional[dict] = None  # [Audit §1.7] Stores target data during wind-up

    # Weapon switch delay.
    is_switching: bool = False
    switch_remaining: float = 0.0
    switch_target_idx: int = 0
    weapon_switch_time: float = 0.3  # matches EnemyWeaponTypes.h default

    # Dodge state.
    is_dodging: bool = False
    dodge_remaining: float = 0.0
    dodge_cooldown_remaining: float = 0.0
    dodge_duration: float = 0.4
    dodge_cooldown: float = 2.0
    dodge_speed: float = 800.0
    dodge_direction: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    facing: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))

    # Action lock system (matches C++ NeuralCombatComponent lock).
    # When locked, the agent cannot execute new combat actions.
    # Movement still works (hold last direction). The NN still runs
    # but combat actions are masked to NONE only.
    action_lock_remaining: float = 0.0
    action_lock_duration: float = 0.0   # Total duration of current lock (for progress calc)
    action_lock_reason: int = 0          # Matches C++ EActionLockReason:
                                         # 0=None, 1=Firing, 2=Reloading, 3=Dodging,
                                         # 4=Melee, 5=Switching, 6=WindUp  [Audit §1.1]

    # Auto-dodge config (matches C++ DodgeComponent autonomous threat detection).
    auto_dodge_enabled: bool = True
    auto_dodge_threat_range: float = 300.0   # React to projectiles this close
    auto_dodge_threat_tta: float = 0.5       # React when TTA < this (seconds)

    def hp_fraction(self) -> float:
        return max(0, self.hp / self.max_hp)

    def active_slot(self) -> Optional[WeaponSlot]:
        if 0 <= self.active_weapon < len(self.weapons):
            return self.weapons[self.active_weapon]
        return None

    def any_ammo(self) -> bool:
        return any(w.has_ammo() for w in self.weapons)

    def all_ranged_empty(self) -> bool:
        return all(not w.has_ammo() for w in self.weapons if w.is_ranged)

    def tick_weapons(self, dt: float):
        for w in self.weapons:
            w.tick(dt)
        self.melee.tick(dt)

        # Wind-up countdown.
        if self.is_winding_up:
            self.wind_up_remaining -= dt
            if self.wind_up_remaining <= 0:
                self.is_winding_up = False
                self.wind_up_remaining = 0
                # [Audit §1.7] _pending_fire remains set — env will
                # spawn the projectile after this tick returns.

        # Weapon switch countdown.
        if self.is_switching:
            self.switch_remaining -= dt
            if self.switch_remaining <= 0:
                self.active_weapon = self.switch_target_idx
                self.is_switching = False
                self.switch_remaining = 0

        # Dodge countdown.
        if self.is_dodging:
            self.dodge_remaining -= dt
            if self.dodge_remaining <= 0:
                self.is_dodging = False
                self.dodge_remaining = 0
        if self.dodge_cooldown_remaining > 0:
            self.dodge_cooldown_remaining -= dt

        # Action lock countdown.
        if self.action_lock_remaining > 0:
            self.action_lock_remaining -= dt
            if self.action_lock_remaining <= 0:
                self.action_lock_remaining = 0
                self.action_lock_reason = 0

    @property
    def is_action_locked(self) -> bool:
        return self.action_lock_remaining > 0

    @property
    def action_lock_progress(self) -> float:
        """0 = just started, 1 = about to finish."""
        if self.action_lock_duration <= 0:
            return 1.0
        return 1.0 - max(0.0, self.action_lock_remaining / self.action_lock_duration)

    def set_action_lock(self, duration: float, reason: int):
        """Set an action lock. Unconditionally overwrites current lock.
        
        Matches C++ NeuralCombatComponent::SetActionLock which always
        replaces duration and reason regardless of remaining time.
        [Audit §3.3]
        """
        self.action_lock_remaining = duration
        self.action_lock_duration = duration
        self.action_lock_reason = reason


@dataclass
class Target:
    """A player party member (scripted opponent)."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    max_speed: float = 500.0
    hp: float = 200.0
    max_hp: float = 200.0
    defence: float = 30.0
    barrier: float = 0.0
    alive: bool = True
    is_player_controlled: bool = True
    target_id: int = 0
    focus_target_id: int = 0  # ID of the actor this target is focusing on
                               # (0 = agent, -1 = ally, matches C++ AI focus)

    # Combat role: "ranged", "melee", "mixed" (has both).
    combat_role: str = "ranged"

    # ── Character type (maps to ECharacterType) ──────────────────
    # "knight"=0.0, "rogue"=0.2, "ranger"=0.4, "mage"=0.6, "healer"=0.8
    character_type: str = "mage"
    character_type_float: float = 0.6  # Pre-computed normalised value.

    # ── Mana system ──────────────────────────────────────────────
    mana: float = 50.0
    max_mana: float = 50.0
    mana_regen_per_second: float = 5.0
    mana_regen_delay: float = 2.0      # Seconds after spending before regen starts.
    mana_regen_delay_remaining: float = 0.0

    # ── Cast / attack commitment ─────────────────────────────────
    commitment: float = 0.0            # 0=idle, 0-1=animation progress.
    commitment_duration: float = 0.0   # Total duration of current action.
    commitment_timer: float = 0.0      # Time elapsed in current action.

    # ── Gap-closer ability ───────────────────────────────────────
    gap_closer_range: float = 600.0    # Max dash/charge range.
    gap_closer_cooldown: float = 8.0   # Cooldown duration.
    gap_closer_cooldown_remaining: float = 0.0
    gap_closer_speed: float = 3000.0   # Dash speed during gap-close.
    has_gap_closer: bool = True        # Whether this character has one.

    # Ranged attack stats.
    attack_damage: float = 18.0
    attack_range: float = 1200.0
    attack_cooldown: float = 1.0
    attack_cooldown_remaining: float = 0.0
    attack_projectile_speed: float = 1800.0
    attack_stat: float = 8.0
    attack_mana_cost: float = 8.0      # Mana cost per ranged attack/spell.
    attack_cast_time: float = 0.4      # Cast time before projectile fires.

    # Melee attack stats.
    melee_damage: float = 30.0
    melee_range: float = 200.0
    melee_cooldown: float = 0.8
    melee_cooldown_remaining: float = 0.0
    melee_stat: float = 12.0
    melee_mana_cost: float = 0.0       # Melee usually free.
    melee_commit_time: float = 0.3     # Swing animation duration.

    # [Audit §5.4] Targets should roll crits, matching C++ ProcessAttack.
    crit_chance: float = 0.05
    crit_multiplier: float = 1.5

    # AI state.
    move_timer: float = 0.0
    move_dir: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    facing: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0], dtype=np.float32))  # Unit vector.
    behaviour: str = "aggressive"
    strafe_dir: float = 1.0
    strafe_timer: float = 0.0

    def hp_fraction(self) -> float:
        return max(0, self.hp / self.max_hp)

    def mana_fraction(self) -> float:
        return max(0, self.mana / self.max_mana) if self.max_mana > 0 else 0.0

    def gap_closer_threat(self, agent_pos) -> float:
        """Returns 0-1: how threatened the agent is by a gap-closer."""
        if not self.has_gap_closer or not self.alive:
            return 0.0
        if self.gap_closer_cooldown_remaining > 0:
            return 0.0
        dist = np.linalg.norm(self.pos - agent_pos)
        if dist > self.gap_closer_range:
            return 0.0
        return 1.0

    def tick_mana(self, dt: float):
        """Regenerate mana after regen delay."""
        if self.mana_regen_delay_remaining > 0:
            self.mana_regen_delay_remaining -= dt
        elif self.mana < self.max_mana:
            self.mana = min(self.max_mana,
                            self.mana + self.mana_regen_per_second * dt)
        self.gap_closer_cooldown_remaining = max(
            0, self.gap_closer_cooldown_remaining - dt)

    def tick_commitment(self, dt: float):
        """Advance cast/attack animation progress."""
        if self.commitment_duration > 0 and self.commitment_timer < self.commitment_duration:
            self.commitment_timer += dt
            self.commitment = min(1.0, self.commitment_timer / self.commitment_duration)
        else:
            self.commitment = 0.0
            self.commitment_duration = 0.0
            self.commitment_timer = 0.0

    def start_commitment(self, duration: float):
        """Begin a cast/attack animation."""
        self.commitment_duration = duration
        self.commitment_timer = 0.0
        self.commitment = 0.01

    def spend_mana(self, cost: float) -> bool:
        """Attempt to spend mana. Returns False if insufficient."""
        if self.mana < cost:
            return False
        self.mana -= cost
        self.mana_regen_delay_remaining = self.mana_regen_delay
        return True

    def tick_ai(self, dt: float, arena_size: float, agent_pos=None,
                obstacles=None, rng=None):
        """Role-aware target AI. Melee targets rush in, ranged stay back."""
        _rng = rng or random
        self.attack_cooldown_remaining -= dt
        self.melee_cooldown_remaining -= dt

        if agent_pos is None:
            self._random_walk(dt, arena_size, _rng)
            return

        to_agent = agent_pos - self.pos
        dist = np.linalg.norm(to_agent)
        to_agent_dir = to_agent / max(dist, 1.0)
        perp = np.array([to_agent_dir[1], -to_agent_dir[0]], dtype=np.float32)

        self.strafe_timer -= dt
        if self.strafe_timer <= 0:
            self.strafe_dir *= -1
            self.strafe_timer = _rng.uniform(1.0, 3.0)

        # ── Role overrides behaviour for movement ────────────────
        # Melee targets always want to close distance.
        # Ranged targets use their assigned behaviour.

        if self.combat_role == "melee":
            # Melee: charge straight at the agent. Strafe when close.
            if dist > self.melee_range * 2.5:
                # Sprint directly toward agent.
                self.velocity = to_agent_dir * self.max_speed
            elif dist > self.melee_range:
                # Close but not in range — approach with slight strafe.
                approach = to_agent_dir * 0.8 + perp * self.strafe_dir * 0.2
                self.velocity = approach / max(np.linalg.norm(approach), 0.01) * self.max_speed
            else:
                # In melee range — stick to the target, circle strafe.
                self.velocity = perp * self.strafe_dir * self.max_speed * 0.4

        elif self.combat_role == "mixed":
            # Mixed: ranged at distance, closes to melee when ranged is on cooldown.
            if self.attack_cooldown_remaining > 0.5 and dist > self.melee_range * 3:
                # Ranged on cooldown — close in for melee.
                self.velocity = to_agent_dir * self.max_speed * 0.8
            elif dist > self.attack_range * 0.6:
                self.velocity = to_agent_dir * self.max_speed * 0.7
            elif dist < 300:
                self.velocity = -to_agent_dir * self.max_speed * 0.4
            else:
                self.velocity = perp * self.strafe_dir * self.max_speed * 0.5

        elif self.behaviour == "aggressive":
            if dist > self.attack_range * 0.7:
                self.velocity = to_agent_dir * self.max_speed
            elif dist < 300:
                self.velocity = -to_agent_dir * self.max_speed * 0.5
            else:
                self.velocity = perp * self.strafe_dir * self.max_speed * 0.6

        elif self.behaviour == "kiting":
            if dist < self.attack_range * 0.5:
                self.velocity = -to_agent_dir * self.max_speed
            elif dist < self.attack_range * 0.8:
                self.velocity = (-to_agent_dir * 0.3 + perp * self.strafe_dir * 0.7)
                self.velocity = self.velocity / max(np.linalg.norm(self.velocity), 0.01) * self.max_speed * 0.7
            else:
                self.velocity = perp * self.strafe_dir * self.max_speed * 0.4

        elif self.behaviour == "cover_user":
            best_cover = None
            best_score = -1
            if obstacles:
                for obs in obstacles:
                    cover_pos = np.array([obs.x, obs.y], dtype=np.float32)
                    d_to_cover = np.linalg.norm(self.pos - cover_pos)
                    d_cover_to_agent = np.linalg.norm(agent_pos - cover_pos)
                    if d_to_cover < 600 and d_cover_to_agent < dist:
                        score = 1.0 / max(d_to_cover, 50)
                        if score > best_score:
                            best_score = score
                            best_cover = cover_pos

            if best_cover is not None:
                to_cover = best_cover - self.pos
                d = np.linalg.norm(to_cover)
                if d > 80:
                    self.velocity = (to_cover / d) * self.max_speed * 0.8
                else:
                    self.velocity = perp * self.strafe_dir * self.max_speed * 0.3
            else:
                self.velocity = perp * self.strafe_dir * self.max_speed * 0.5

        else:  # passive
            self._random_walk(dt, arena_size)
            return

        self.pos += self.velocity * dt

        # ── Facing direction (turn toward agent with limited turn rate) ──
        # Targets don't instantly face the agent. A turn rate of ~360°/s
        # means flanking from behind gives a 0.25-0.5s window where the
        # target can't see or accurately shoot the agent.
        if agent_pos is not None:
            to_agent_now = agent_pos - self.pos
            d_now = np.linalg.norm(to_agent_now)
            if d_now > 1:
                desired_facing = to_agent_now / d_now
                # Turn rate: radians per second. 2π = full rotation in 1s.
                turn_rate = 2.0 * math.pi  # ~360°/s
                max_turn = turn_rate * dt

                # Signed angle between current facing and desired.
                cross = self.facing[0] * desired_facing[1] - self.facing[1] * desired_facing[0]
                dot = float(np.dot(self.facing, desired_facing))
                angle_diff = math.atan2(cross, dot)

                if abs(angle_diff) <= max_turn:
                    self.facing = desired_facing.copy()
                else:
                    # Rotate by max_turn toward desired.
                    sign = 1.0 if angle_diff > 0 else -1.0
                    cos_r = math.cos(sign * max_turn)
                    sin_r = math.sin(sign * max_turn)
                    fx, fy = self.facing[0], self.facing[1]
                    self.facing = np.array([
                        fx * cos_r - fy * sin_r,
                        fx * sin_r + fy * cos_r,
                    ], dtype=np.float32)

        # Obstacle avoidance.
        if obstacles:
            for obs in obstacles:
                if obs.contains(self.pos[0], self.pos[1]):
                    dx = self.pos[0] - obs.x
                    dy = self.pos[1] - obs.y
                    if abs(dx / max(obs.hw, 1)) > abs(dy / max(obs.hh, 1)):
                        self.pos[0] = obs.x + (obs.hw + 5) * (1 if dx > 0 else -1)
                    else:
                        self.pos[1] = obs.y + (obs.hh + 5) * (1 if dy > 0 else -1)

        half = arena_size * 0.45
        self.pos = np.clip(self.pos, -half, half)

    def _random_walk(self, dt, arena_size, rng=None):
        _rng = rng or random
        self.move_timer -= dt
        if self.move_timer <= 0:
            angle = _rng.uniform(0, 2 * math.pi)
            self.move_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            self.move_timer = _rng.uniform(0.5, 2.0)
        self.velocity = self.move_dir * self.max_speed * 0.6
        self.pos += self.velocity * dt
        half = arena_size * 0.45
        self.pos = np.clip(self.pos, -half, half)


# ─────────────────────────────────────────────────────────────────
#  Combat Math (matches CombatPipeline.cpp)
# ─────────────────────────────────────────────────────────────────

def compute_damage(
    base_damage: float,
    attack_stat: float,
    defence: float,
    barrier: float,
    crit_chance: float = 0.0,
    crit_multiplier: float = 1.5,
    rng=None,  # [Audit §4.1] Per-env RNG; falls back to global random
) -> Tuple[float, float, bool]:
    """Returns (damage_to_hp, barrier_remaining, was_crit)."""

    # Outgoing damage.
    outgoing = base_damage + attack_stat

    # Crit.
    _rng = rng or random
    was_crit = _rng.random() < crit_chance
    if was_crit:
        outgoing *= crit_multiplier

    # Barrier absorb.
    barrier_absorbed = min(barrier, outgoing)
    remaining_damage = outgoing - barrier_absorbed
    new_barrier = barrier - barrier_absorbed

    if remaining_damage <= 0:
        return 0.0, new_barrier, was_crit

    # Defence reduction: Damage * Constant / (Defence + Constant).
    after_defence = remaining_damage * DEFENCE_CONSTANT / (defence + DEFENCE_CONSTANT)
    after_defence = max(after_defence, MIN_DAMAGE)

    return after_defence, new_barrier, was_crit


# ─────────────────────────────────────────────────────────────────
#  2D Line-of-Sight
# ─────────────────────────────────────────────────────────────────

def check_los(p1: np.ndarray, p2: np.ndarray, obstacles: List[Obstacle]) -> bool:
    """True if no obstacle blocks the line between p1 and p2."""
    for obs in obstacles:
        if _ray_aabb_intersect(p1, p2, obs):
            return False
    return True


def _ray_aabb_intersect(p1: np.ndarray, p2: np.ndarray, obs: Obstacle) -> bool:
    """Simple 2D segment-AABB intersection."""
    return _ray_aabb_intersect_t(p1, p2, obs) is not None


def _ray_aabb_intersect_t(p1: np.ndarray, p2: np.ndarray, obs: Obstacle):
    """Returns t in [0, 1] of first hit along segment, or None if no hit.
    
    t represents hit_distance / segment_length, matching C++
    LowHit.Distance / TraceLength.  [Audit §1.6]
    """
    x1, y1 = p1; x2, y2 = p2
    bx1, by1, bx2, by2 = obs.aabb()

    dx = x2 - x1; dy = y2 - y1
    tmin = 0.0; tmax = 1.0

    for (p, d, lo, hi) in [(x1, dx, bx1, bx2), (y1, dy, by1, by2)]:
        if abs(d) < 1e-4:  # [Audit §5.1] Match UE KINDA_SMALL_NUMBER
            if p < lo or p > hi:
                return None
        else:
            t1 = (lo - p) / d; t2 = (hi - p) / d
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1); tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin


def _sphere_sweep_aabb_t(
    p1: np.ndarray, p2: np.ndarray, obs: Obstacle, sweep_radius: float,
):
    """Sphere sweep against an AABB. Returns t in [0, 1] or None.

    Equivalent to inflating the AABB by sweep_radius (Minkowski sum of
    the obstacle box with a circle of sweep_radius) and then doing a
    standard ray intersection. The returned t is the distance the sphere
    CENTER travels before the sphere surface first contacts the obstacle.
    Matches UE5 SweepSingleByChannel behaviour.
    """
    x1, y1 = p1; x2, y2 = p2
    # Expand AABB by sweep radius.
    bx1 = obs.x - obs.hw - sweep_radius
    by1 = obs.y - obs.hh - sweep_radius
    bx2 = obs.x + obs.hw + sweep_radius
    by2 = obs.y + obs.hh + sweep_radius

    dx = x2 - x1; dy = y2 - y1
    tmin = 0.0; tmax = 1.0

    for (p, d, lo, hi) in [(x1, dx, bx1, bx2), (y1, dy, by1, by2)]:
        if abs(d) < 1e-4:
            if p < lo or p > hi:
                return None
        else:
            t1 = (lo - p) / d; t2 = (hi - p) / d
            if t1 > t2: t1, t2 = t2, t1
            tmin = max(tmin, t1); tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    return max(tmin, 0.0)


def is_behind_cover(agent_pos, target_pos, obstacles) -> Tuple[bool, float]:
    """Check if LOS is blocked, and if so, estimate cover height."""
    for obs in obstacles:
        if _ray_aabb_intersect(agent_pos, target_pos, obs):
            return True, obs.height
    return False, 0.0


# Character body radius for spatial ring ray detection. Larger than
# AGENT_BODY_RADIUS (30 UU) to account for the full capsule collision
# of player and AI characters in UE5. Matches C++ CapsuleHalfRadius.
CHARACTER_DETECT_RADIUS = 45.0


def _ray_circle_intersect_t(
    origin: np.ndarray, direction: np.ndarray,
    center: np.ndarray, radius: float, trace_len: float,
):
    """Returns t in [0, 1] of a sphere sweep hitting a circle, or None.

    origin:    ray start (2D)
    direction: unit direction vector (2D)
    center:    circle center (2D)
    radius:    effective detection radius (character + sweep)
    trace_len: max ray length (t is normalised by this)

    Handles the case where the agent is already inside the detection
    circle (returns 0.0 = touching). This matches UE5's SweepSingle
    which reports bStartPenetrating when the sweep shape overlaps a
    pawn at the start position.
    """
    oc = center - origin
    dist_sq = float(np.dot(oc, oc))
    r_sq = radius * radius

    # If the agent is already inside the detection circle, report
    # immediate contact regardless of ray direction. Without this,
    # a character whose center is behind the ray (proj < 0) but
    # within detection range would be missed — a parity bug with
    # C++ SweepSingle which correctly reports initial overlaps.
    if dist_sq < r_sq:
        return 0.0

    proj = float(np.dot(oc, direction))
    if proj < 0:
        return None  # Circle is behind the ray AND we're outside it.

    d_sq = dist_sq - proj * proj
    if d_sq > r_sq:
        return None  # Ray misses the circle.

    t_hit = proj - math.sqrt(r_sq - d_sq)
    if t_hit < 0:
        t_hit = 0.0  # Shouldn't happen after the dist_sq check, but safe.
    t_norm = t_hit / trace_len
    return t_norm if t_norm <= 1.0 else None


# ─────────────────────────────────────────────────────────────────
#  Environment Config
# ─────────────────────────────────────────────────────────────────

@dataclass
class WeaponPreset:
    slots: List[dict] = field(default_factory=list)
    melee: dict = field(default_factory=dict)

# Standard weapon presets matching the design doc.
WEAPON_PRESETS = {
    "scout": WeaponPreset(
        slots=[dict(name="Laser", base_damage=8, weapon_range=1200, max_ammo=20,
                     fire_cooldown=0.2, optimal_min=400, optimal_max=900,
                     projectile_speed=4500)],  # Fast projectile (not hitscan).
        melee=dict(damage=15, range=150, cooldown=0.8)),
    "heavy": WeaponPreset(
        slots=[
            dict(name="Cannon", base_damage=35, weapon_range=2000, max_ammo=6,
                 fire_cooldown=1.0, reload_time=3.0, optimal_min=800, optimal_max=1600,
                 wind_up_time=0.5, projectile_speed=2000),  # Slow heavy round.
            dict(name="Missiles", base_damage=25, weapon_range=1800, max_ammo=4,
                 fire_cooldown=1.5, reload_time=4.0, can_arc=True, max_arc_height=400,
                 optimal_min=600, optimal_max=1400, projectile_speed=1200),  # Arcing over cover up to 400 UU.
        ],
        melee=dict(damage=40, range=250, cooldown=1.5)),
    "sniper": WeaponPreset(
        slots=[
            dict(name="Railgun", base_damage=80, weapon_range=3000, max_ammo=1,
                 fire_cooldown=2.0, reload_time=3.0, wind_up_time=1.0,
                 optimal_min=1500, optimal_max=2800,
                 projectile_speed=6000),  # Very fast rail projectile.
            dict(name="Sidearm", base_damage=10, weapon_range=1000, max_ammo=12,
                 fire_cooldown=0.3, optimal_min=300, optimal_max=800,
                 projectile_speed=3500),  # Standard sidearm projectile.
        ],
        melee=dict(damage=10, range=150, cooldown=1.0)),
    "melee_bot": WeaponPreset(
        slots=[dict(name="Sidearm", base_damage=8, weapon_range=800, max_ammo=10,
                     fire_cooldown=0.4, optimal_min=200, optimal_max=600,
                     projectile_speed=3000)],  # Standard projectile.
        melee=dict(damage=35, range=200, cooldown=0.6)),
    "tank": WeaponPreset(
        slots=[dict(name="Gatling", base_damage=5, weapon_range=1500, max_ammo=100,
                     fire_cooldown=0.08, reload_time=4.0, optimal_min=400, optimal_max=1200,
                     projectile_speed=4000)],  # Fast stream of projectiles.
        melee=dict(damage=30, range=250, cooldown=1.2)),
}


@dataclass
class CombatEnvConfig:
    num_enemies: int = 1
    num_targets: int = 1
    arena_size: float = 2000.0
    num_obstacles: int = 4
    max_steps: int = 1000              # at 0.2s = 200s of combat
    decision_interval: float = 0.2
    curriculum_stage: int = 1
    archetype: str = "ranged"
    weapon_preset: str = "scout"
    weapon_pool: list = None           # If set, randomly pick from this list each reset()
    enemy_hp: float = 100.0
    enemy_defence: float = 20.0
    enemy_attack: float = 5.0
    target_hp: float = 200.0
    target_defence: float = 30.0
    target_speed_fraction: float = 0.6  # how actively targets move (0=stationary, 1=full speed)
    engagement_distance: float = 1500.0


# ─────────────────────────────────────────────────────────────────
#  Gymnasium Environment
# ─────────────────────────────────────────────────────────────────

class CombatEnv(gym.Env):
    """Single-agent combat environment.

    The agent is one enemy robot. Targets are player party members
    with simple scripted movement. For multi-enemy training (stages 6-7),
    run multiple parallel envs or extend to multi-agent.

    render_mode="human" opens a pygame window showing the arena in real-time.
    render_mode="rgb_array" returns a numpy image array per frame.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: CombatEnvConfig = None, render_mode: str = None):
        super().__init__()
        self.cfg = config or CombatEnvConfig()
        self.render_mode = render_mode

        # Action space: MultiDiscrete [movement(9), combat(7), target(5)].
        self.action_space = spaces.MultiDiscrete([
            MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS])

        # Observation space: flat float vector.
        # Frame stacking is handled by the training wrapper, not here.
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self.reward_fn = get_reward_function_for_stage(
            self.cfg.curriculum_stage, self.cfg.archetype)

        self.agent: Optional[Agent] = None
        self.targets: List[Target] = []
        self.allies: List[Agent] = []   # Other enemies (for group obs).
        self.obstacles: List[Obstacle] = []
        self.current_target_idx: int = 0
        self.step_count: int = 0
        self._prev_target_hps: dict = {}  # target_id -> hp_fraction last step
        self._prev_weapon_index: int = 0  # Track weapon switches between steps.
        self._effective_arena_size: float = self.cfg.arena_size
        self._arena_half: float = self.cfg.arena_size * 0.45

        # Renderer state (lazy init on first render call).
        self._screen = None
        self._clock = None
        self._render_size = 600  # pixels
        self._projectiles: List[SimProjectile] = []
        self._projectile_snapshots: List[List[dict]] = []  # substep snapshots for render
        self._prev_alive_allies: int = 0  # Track ally deaths between steps
        self.threat_table = ThreatTable()  # [Audit §1.3] Damage-based target priority
        self.rng = random.Random()  # [Audit §4.1] Per-env RNG isolation
        self._player_patterns = PlayerPatternTracker()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)  # [Audit §4.2] Propagate seed to per-env RNG
        self.step_count = 0
        self._prev_weapon_index = 0
        self.reward_fn.reset()
        self._projectiles = []
        self.threat_table.reset()  # [Audit §1.3]
        self._player_patterns = PlayerPatternTracker()

        # Build arena.
        self._build_arena()

        # Randomize weapon each episode if weapon_pool is set.
        # This ensures all parallel envs experience all weapons,
        # reducing advantage estimate variance in PPO.
        if self.cfg.weapon_pool:
            self.cfg.weapon_preset = self.rng.choice(self.cfg.weapon_pool)
        
        self._spawn_agent()
        self._spawn_targets()          # MOVED BEFORE _prev_target_hps

        self._prev_target_hps = {
            t.target_id: t.hp_fraction() for t in self.targets
        }
        self._prev_alive_allies = sum(
            1 for ally in getattr(self, 'allies', []) if ally.alive)

        obs = self._build_observation()
        info = {"action_mask": self.build_action_mask()}
        return obs, info

    # ═════════════════════════════════════════════════════════════
    #  Action Masking (matches C++ FNeuralActionMask)
    # ═════════════════════════════════════════════════════════════

    def build_action_mask(self):
        """Build validity masks for each action head.

        Returns dict with m_mask, c_mask, t_mask — boolean arrays where
        True = valid action. Invalid actions get -inf logits in the model.
        Matches C++ NeuralCombatComponent::UpdateActionMask().
        """
        a = self.agent
        slot = a.active_slot() if a else None
        m_mask = np.ones(MOVEMENT_ACTIONS, dtype=bool)  # All movement always valid
        c_mask = np.zeros(COMBAT_ACTIONS, dtype=bool)
        t_mask = np.zeros(TARGET_ACTIONS, dtype=bool)

        # Combat action mask.
        c_mask[CombatAction.NONE] = True  # Always valid

        if a and a.alive and not a.is_action_locked:
            # [1] Fire — weapon ready + has ammo + not switching/dodging/winding
            if (slot and slot.is_ready() and slot.has_ammo()
                    and not a.is_switching and not a.is_dodging):
                c_mask[CombatAction.FIRE] = True

            # [2] Reload — ammo < max + not already reloading
            if (slot and slot.max_ammo > 0
                    and slot.current_ammo < slot.max_ammo
                    and not slot.is_reloading):
                c_mask[CombatAction.RELOAD] = True

            # [3] Switch to weapon 0
            if len(a.weapons) > 0 and a.active_weapon != 0 and not a.is_switching:
                c_mask[CombatAction.SWITCH_0] = True

            # [4] Switch to weapon 1
            if len(a.weapons) > 1 and a.active_weapon != 1 and not a.is_switching:
                c_mask[CombatAction.SWITCH_1] = True

            # [5] Melee — cooldown ready
            if a.melee.cooldown_remaining <= 0 and not a.is_dodging:
                c_mask[CombatAction.MELEE] = True

            # [6] Block — always valid (matches C++ CombatMask[6] = true).
            c_mask[CombatAction.BLOCK] = True

            # [7] Dodge — cooldown ready + not already dodging.
            if (not a.is_dodging
                    and a.dodge_cooldown_remaining <= 0):
                c_mask[CombatAction.DODGE] = True

        # Target mask — only valid if that slot has an alive target.
        # [Audit §1.3] Sort by priority score (not distance) to match
        # C++ ScoredTargets ordering.
        sorted_targets = self._get_sorted_targets()
        for i in range(TARGET_ACTIONS - 1):
            t_mask[i] = (i < len(sorted_targets))
        t_mask[TARGET_ACTIONS - 1] = True  # Keep current — always valid

        return {"m_mask": m_mask, "c_mask": c_mask, "t_mask": t_mask}

    # ═════════════════════════════════════════════════════════════
    #  Step
    # ═════════════════════════════════════════════════════════════

    def step(self, action):
        move_idx, combat_idx, target_idx = action
        dt = self.cfg.decision_interval
        self.step_count += 1

        # Snapshot pre-action state for reward.
        prev_state = self._build_combat_state()

        # 1. Execute agent actions.
        #    If action-locked (reload/dodge/wind-up/switch/melee), skip
        #    combat actions but still allow movement (hold direction).
        #    This matches C++ where the lock gates inference entirely,
        #    but in training we still run the NN with combat masked to NONE.
        if self.agent.is_dodging:
            # During dodge, move in dodge direction (overrides player movement).
            # Sub-step to prevent tunneling through thin walls.
            dodge_delta = self.agent.dodge_direction * self.agent.dodge_speed * dt
            dodge_dist = np.linalg.norm(dodge_delta)
            max_step = AGENT_BODY_RADIUS * 0.9
            num_sub = max(1, int(np.ceil(dodge_dist / max_step)))
            step_delta = dodge_delta / num_sub
            dodge_new_pos = self.agent.pos.copy()

            for _ in range(num_sub):
                dodge_new_pos = dodge_new_pos + step_delta
                for obs in self.obstacles:
                    if obs.contains_circle(dodge_new_pos[0], dodge_new_pos[1], AGENT_BODY_RADIUS):
                        dodge_new_pos[0], dodge_new_pos[1] = obs.push_out_circle(
                            dodge_new_pos[0], dodge_new_pos[1], AGENT_BODY_RADIUS)

            half = self._arena_half
            dodge_new_pos = np.clip(dodge_new_pos, -half + AGENT_BODY_RADIUS,
                                    half - AGENT_BODY_RADIUS)
            self.agent.pos = dodge_new_pos
        elif self.agent.is_action_locked:
            # Locked but not dodging — still allow movement and target selection.
            self._execute_movement(move_idx, dt)
            self._execute_target_selection(target_idx)
            # Combat action is skipped (mask should have forced NONE).
        else:
            # Normal — execute everything.
            self._execute_movement(move_idx, dt)
            self._execute_target_selection(target_idx)
            self._execute_combat(combat_idx, dt)

        # 2. Tick weapon cooldowns/reloads/wind-up/switch/dodge.
        self.agent.tick_weapons(dt)
        self._resolve_pending_fire()  # [Audit §1.7] Spawn projectile when wind-up completes
        self.agent.combat_time += dt
        self.threat_table.decay(dt)  # [Audit §1.3] Decay accumulated threat over time

        # 3. Tick target AI.
        for t in self.targets:
            if t.alive:
                speed_mult = self.cfg.target_speed_fraction
                if self.cfg.curriculum_stage <= 2:
                    speed_mult = 0.0
                t.max_speed = 500.0 * speed_mult
                t.tick_ai(dt, self._effective_arena_size,
                          agent_pos=self.agent.pos,
                          obstacles=self.obstacles,
                          rng=self.rng)
                t.tick_mana(dt)
                t.tick_commitment(dt)

        # 3b. Update player pattern tracker.
        self._player_patterns.update(
            self.targets, self.agent.pos, dt, self._arena_half)

        # 4. Targets fight back (simplified: periodic damage to agent).
        self._target_attacks_agent(dt)
        
        # 4b. Tick projectiles (they travel between decision ticks).
        # Sub-step projectiles at higher resolution for accurate collision.
        # [Audit §3.1] Match UE's ~60 Hz projectile tick rate (16.67ms).
        proj_substeps = max(1, int(round(dt / (1.0 / 60.0))))  # 12 substeps at dt=0.2
        proj_dt = dt / proj_substeps

        # Capture intermediate projectile positions for smooth rendering.
        # Snapshot BEFORE each tick so we see projectiles at their positions
        # before they get cleaned up on hit/expiry.
        self._projectile_snapshots = []
        for _ in range(proj_substeps):
            if self.render_mode is not None:
                self._projectile_snapshots.append(self._snapshot_projectiles())
            self._tick_projectiles(proj_dt)
        # Final snapshot after all substeps (surviving projectiles).
        if self.render_mode is not None:
            self._projectile_snapshots.append(self._snapshot_projectiles())

        # 4c. Auto-dodge: if an incoming projectile is about to hit and dodge
        #     is available, trigger an autonomous dodge. Matches C++ DodgeComponent
        #     which fires OnDodgeStarted → NeuralCombatComponent::SetActionLock.
        if (self.agent.alive and self.agent.auto_dodge_enabled
                and not self.agent.is_dodging
                and self.agent.dodge_cooldown_remaining <= 0
                and self.cfg.curriculum_stage >= 3):
            self._try_auto_dodge()
            
        # 5. Build post-action state.
        curr_state = self._build_combat_state()

        # 6. Update weapon tracking for next step.
        self._prev_weapon_index = self.agent.active_weapon

        # 7. Compute reward.
        reward, info = self.reward_fn.compute(
            prev_state, (move_idx, combat_idx, target_idx), curr_state)

        # 8. Check done conditions.
        done = False
        if not self.agent.alive:
            done = True
        if all(not t.alive for t in self.targets):
            done = True

        truncated = self.step_count >= self.cfg.max_steps

        if done or truncated:
            end_bonus, end_info = self.reward_fn.compute_episode_end_bonus(
                curr_state, truncated=truncated)
            reward += end_bonus
            info.update(end_info)
            # [Fix Bug 2] Include end_bonus in info["total"] so logged
            # reward/total matches actual episode reward. Previously,
            # info["total"] only captured compute() return, missing
            # timeout_penalty, surviving_targets, and min_damage_penalty.
            info["total"] = info.get("total", 0.0) + end_bonus

        # [Fix Bug 1] Set is_win flag in info dict so rollout/win is
        # tracked correctly. Previously infos[i].get("is_win", False)
        # always returned False because no code ever set this key.
        info["is_win"] = all(not t.alive for t in self.targets)

        # Track per-target HP for next step's multi-target damage reward.
        self._prev_target_hps = {
            t.target_id: t.hp_fraction() for t in self.targets if t.alive
        }
        # Track alive allies for next step's ally_just_died detection.
        self._prev_alive_allies = sum(
            1 for ally in getattr(self, 'allies', []) if ally.alive)
        
        obs = self._build_observation()

        # Include action mask for next step in info (used by PPO).
        info["action_mask"] = self.build_action_mask()

        return obs, reward, done, truncated, info

    # ═════════════════════════════════════════════════════════════
    #  Arena Setup
    # ═════════════════════════════════════════════════════════════

    def _build_arena(self):
        """Build a randomised arena with varied obstacle types.
        Called per episode, so every fight has different geometry."""
        self.obstacles = []

        # Randomise arena size ±20% around the configured value.
        # Store in a separate attribute so cfg.arena_size stays constant.
        self._effective_arena_size = self.cfg.arena_size * self.rng.uniform(0.8, 1.2)
        self._arena_half = self._effective_arena_size * 0.45
        half = self._effective_arena_size * 0.4

        # Randomise obstacle count ±50% around the configured value.
        base_count = self.cfg.num_obstacles
        actual_count = max(0, self.rng.randint(
            max(0, base_count - base_count // 2),
            base_count + base_count // 2 + 1))

        for _ in range(actual_count):
            obs_type = self.rng.choices(
                ["pillar", "wall", "l_shape", "cover", "building"],
                weights=[15, 25, 10, 30, 20],
                k=1)[0]

            ox = self.rng.uniform(-half, half)
            oy = self.rng.uniform(-half, half)

            if obs_type == "pillar":
                # Small square pillar.
                s = self.rng.uniform(30, 60)
                self.obstacles.append(Obstacle(ox, oy, s, s, 300.0))

            elif obs_type == "wall":
                # Long thin wall — building-like.
                if self.rng.random() < 0.5:
                    hw = self.rng.uniform(150, 350)
                    hh = self.rng.uniform(20, 40)
                else:
                    hw = self.rng.uniform(20, 40)
                    hh = self.rng.uniform(150, 350)
                self.obstacles.append(Obstacle(ox, oy, hw, hh, 300.0))

            elif obs_type == "l_shape":
                # L-shaped: two joined rectangles.
                hw1 = self.rng.uniform(80, 200)
                hh1 = self.rng.uniform(20, 40)
                self.obstacles.append(Obstacle(ox, oy, hw1, hh1, 300.0))
                # Second arm perpendicular.
                ox2 = ox + hw1 * self.rng.choice([-1, 1])
                oy2 = oy
                hw2 = self.rng.uniform(20, 40)
                hh2 = self.rng.uniform(80, 200)
                self.obstacles.append(Obstacle(ox2, oy2, hw2, hh2, 300.0))

            elif obs_type == "cover":
                # Low cover — can arc over.
                hw = self.rng.uniform(60, 180)
                hh = self.rng.uniform(20, 50)
                height = self.rng.uniform(100, 180)
                self.obstacles.append(Obstacle(ox, oy, hw, hh, height))

            elif obs_type == "building":
                # Large building-like obstacle.
                hw = self.rng.uniform(100, 250)
                hh = self.rng.uniform(80, 200)
                self.obstacles.append(Obstacle(ox, oy, hw, hh, 300.0))

    def _spawn_agent(self):
        half = self._effective_arena_size * 0.3
        preset = WEAPON_PRESETS.get(self.cfg.weapon_preset,
                                     WEAPON_PRESETS["scout"])

        weapons = []
        for slot_cfg in preset.slots:
            w = WeaponSlot(**slot_cfg)
            w.current_ammo = w.max_ammo
            weapons.append(w)
            
        # Randomise weapon parameters for range generalisation.
        # The model must learn to READ the range observation and adapt,
        # not memorise "stay at 1000 UU" for a specific preset.
        # Only apply in stages 4+ (once basic weapon mechanics are stable).
        if self.cfg.curriculum_stage >= 4:
            for w in weapons:
                range_scale = self.rng.uniform(0.6, 1.8)
                w.weapon_range *= range_scale
                w.optimal_min *= range_scale
                w.optimal_max *= range_scale
                w.fire_cooldown *= self.rng.uniform(0.7, 1.3)
                w.base_damage *= self.rng.uniform(0.7, 1.3)
                w.projectile_speed *= self.rng.uniform(0.8, 1.2)
                w.reload_time *= self.rng.uniform(0.8, 1.2)

        melee_cfg = MeleeConfig(**preset.melee)
        pos = np.array([self.rng.uniform(-half, half),
                        self.rng.uniform(-half, half)], dtype=np.float32)

        self.agent = Agent(
            pos=pos, spawn_pos=pos.copy(),
            max_speed=450.0,
            hp=self.cfg.enemy_hp, max_hp=self.cfg.enemy_hp,
            defence=self.cfg.enemy_defence,
            attack_stat=self.cfg.enemy_attack,
            archetype=Archetype[self.cfg.archetype.upper()],
            weapons=weapons, melee=melee_cfg,
        )

    def _spawn_targets(self):
        self.targets = []
        dist = self.cfg.engagement_distance

        # Randomise target count for stages 3+.
        num = self.cfg.num_targets
        if self.cfg.curriculum_stage >= 3 and num > 1:
            num = self.rng.randint(1, num)

        behaviours_ranged = ["aggressive", "kiting", "cover_user", "passive"]

        # Decide party composition: mix of melee, ranged, mixed.
        # Early stages: all ranged (simpler threat model).
        # Stage 3+: introduce melee hostiles.
        # Stage 5+: mixed role hostiles.
        roles = []
        for i in range(num):
            if self.cfg.curriculum_stage <= 2:
                roles.append("ranged")
            elif self.cfg.curriculum_stage <= 4:
                # 40% chance of melee, 60% ranged. At least 1 ranged.
                if i == 0:
                    roles.append(self.rng.choice(["ranged", "ranged", "melee"]))
                else:
                    roles.append(self.rng.choice(["ranged", "melee"]))
            else:
                # Full mix including mixed-role fighters.
                roles.append(self.rng.choices(
                    ["ranged", "melee", "mixed"],
                    weights=[40, 35, 25], k=1)[0])

        for i in range(num):
            role = roles[i]

            # Melee targets spawn closer (they need to close distance).
            if role == "melee":
                spawn_dist = dist * self.rng.uniform(0.4, 0.8)
            else:
                spawn_dist = dist * self.rng.uniform(0.7, 1.3)

            angle = self.rng.uniform(0, 2 * math.pi)
            offset = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32) * spawn_dist
            pos = self.agent.pos + offset
            half = self._arena_half
            pos = np.clip(pos, -half, half)

            # Behaviour: melee targets are always aggressive rushers.
            if role == "melee":
                behaviour = "aggressive"  # tick_ai reads combat_role for actual movement.
            elif self.cfg.curriculum_stage <= 2:
                behaviour = "passive"
            elif self.cfg.curriculum_stage <= 4:
                behaviour = self.rng.choice(["aggressive", "kiting", "passive"])
            else:
                behaviour = self.rng.choice(behaviours_ranged)

            # Role-appropriate stats.
            base_hp = self.cfg.target_hp * self.rng.uniform(0.8, 1.2)
            base_def = self.cfg.target_defence * self.rng.uniform(0.8, 1.2)

            if role == "melee":
                # Melee: faster, tankier, high melee damage, weak ranged.
                t = Target(
                    pos=pos, hp=base_hp * 1.2, max_hp=base_hp * 1.2,
                    defence=base_def * 1.2,
                    max_speed=self.rng.uniform(520, 600),  # Faster to close gap.
                    is_player_controlled=(i == 0),
                    target_id=i,
                    combat_role="melee",
                    melee_damage=self.rng.uniform(28, 40),
                    melee_range=self.rng.uniform(180, 250),
                    melee_cooldown=self.rng.uniform(0.6, 1.0),
                    melee_stat=self.rng.uniform(10, 16),
                    attack_damage=self.rng.uniform(8, 12),   # Weak ranged.
                    attack_range=self.rng.uniform(600, 900),
                    attack_cooldown=self.rng.uniform(1.2, 2.0),
                    attack_projectile_speed=self.rng.choice([2000, 2500, 3000]),
                    attack_stat=self.rng.uniform(3, 6),
                    behaviour=behaviour,
                )
            elif role == "mixed":
                # Mixed: decent at both, jack of all trades.
                t = Target(
                    pos=pos, hp=base_hp, max_hp=base_hp,
                    defence=base_def,
                    max_speed=self.rng.uniform(470, 530),
                    is_player_controlled=(i == 0),
                    target_id=i,
                    combat_role="mixed",
                    melee_damage=self.rng.uniform(22, 32),
                    melee_range=self.rng.uniform(170, 220),
                    melee_cooldown=self.rng.uniform(0.8, 1.2),
                    melee_stat=self.rng.uniform(8, 12),
                    attack_damage=self.rng.uniform(14, 20),
                    attack_range=self.rng.uniform(900, 1300),
                    attack_cooldown=self.rng.uniform(0.9, 1.3),
                    attack_projectile_speed=self.rng.choice([1500, 1800, 2000]),
                    attack_stat=self.rng.uniform(6, 10),
                    behaviour=behaviour,
                )
            else:
                # Ranged: standard ranged fighter.
                t = Target(
                    pos=pos, hp=base_hp, max_hp=base_hp,
                    defence=base_def,
                    max_speed=self.rng.uniform(440, 520),
                    is_player_controlled=(i == 0),
                    target_id=i,
                    combat_role="ranged",
                    melee_damage=self.rng.uniform(10, 18),   # Weak melee.
                    melee_range=self.rng.uniform(140, 180),
                    melee_cooldown=self.rng.uniform(1.0, 1.5),
                    melee_stat=self.rng.uniform(4, 8),
                    attack_damage=self.rng.uniform(15, 22),
                    attack_range=self.rng.uniform(1000, 1500),
                    attack_cooldown=self.rng.uniform(0.8, 1.4),
                    attack_projectile_speed=self.rng.choice([2500, 3000, 3500, 4000, 4500]),
                    attack_stat=self.rng.uniform(5, 12),
                    behaviour=behaviour,
                )

            self.targets.append(t)

            # Set character type and mana based on combat role.
            role_to_type = {
                "melee": "knight",
                "mixed": "rogue",
                "ranged": "mage",
            }
            t.character_type = role_to_type.get(role, "mage")
            t.character_type_float = CHARACTER_TYPE_MAP.get(t.character_type, 0.5)

            # Mana: melee characters have minimal/no mana, casters have full.
            if role == "melee":
                t.max_mana = 0.0
                t.mana = 0.0
                t.attack_mana_cost = 0.0
                t.melee_mana_cost = 0.0
                t.has_gap_closer = True
                t.gap_closer_range = self.rng.uniform(500, 800)
                t.gap_closer_cooldown = self.rng.uniform(6, 10)
            elif role == "mixed":
                t.max_mana = self.rng.uniform(30, 50)
                t.mana = t.max_mana
                t.attack_mana_cost = self.rng.uniform(5, 10)
                t.melee_mana_cost = 0.0
                t.has_gap_closer = self.rng.random() > 0.5
                t.gap_closer_range = self.rng.uniform(400, 600)
                t.gap_closer_cooldown = self.rng.uniform(8, 12)
            else:  # ranged / mage
                t.max_mana = self.rng.uniform(40, 80)
                t.mana = t.max_mana
                t.attack_mana_cost = self.rng.uniform(6, 12)
                t.melee_mana_cost = 0.0
                t.has_gap_closer = False
                t.attack_cast_time = self.rng.uniform(0.3, 0.6)

        self.current_target_idx = 0

    # ═════════════════════════════════════════════════════════════
    #  Action Execution
    # ═════════════════════════════════════════════════════════════

    def _execute_movement(self, move_idx: int, dt: float):
        a = self.agent
        if not a.alive:
            return

        # Compute desired velocity from movement action.
        if move_idx == 0:
            # Stop command — decelerate toward zero.
            desired = np.zeros(2, dtype=np.float32)
        else:
            # Build target-facing reference frame.
            target = self._current_target()
            if target:
                fwd = target.pos - a.pos
                d = np.linalg.norm(fwd)
                fwd = fwd / d if d > 1 else np.array([1.0, 0.0], dtype=np.float32)
            else:
                fwd = np.array([1.0, 0.0], dtype=np.float32)

            right = np.array([fwd[1], -fwd[0]], dtype=np.float32)

            # Map index 1-8 to angle: 0, 45, 90, ..., 315 degrees.
            angle_deg = (move_idx - 1) * 45.0
            angle_rad = math.radians(angle_deg)
            direction = fwd * math.cos(angle_rad) + right * math.sin(angle_rad)
            desired = direction * a.max_speed

        # ── Acceleration physics (matches UE CharacterMovementComponent) ──
        # Instead of snapping to desired velocity, accelerate toward it.
        # This makes direction changes take time, producing smooth purposeful
        # movement instead of jittery zigzagging.
        #
        # With max_accel=2048 and dt=0.2s:
        #   Max velocity change per tick = 409.6 UU/s
        #   Reach full speed from stop:   ~1.1 ticks (0.22s)
        #   Full reversal (N→S):          ~2.2 ticks (0.44s)
        #   45° course correction:         ~0.5 ticks (near-instant)

        diff = desired - a.velocity
        diff_mag = np.linalg.norm(diff)

        if diff_mag < 0.01:
            # Already at desired velocity.
            a.velocity = desired.copy()
        else:
            # Use acceleration for speeding up, braking for slowing down.
            current_speed = np.linalg.norm(a.velocity)
            desired_speed = np.linalg.norm(desired)

            if desired_speed < 1.0:
                # Braking to a stop.
                max_change = a.braking_deceleration * dt
            elif desired_speed < current_speed:
                # Slowing down (partially braking).
                max_change = a.braking_deceleration * dt
            else:
                # Accelerating or changing direction.
                max_change = a.max_acceleration * dt

            if diff_mag <= max_change:
                a.velocity = desired.copy()
            else:
                a.velocity = a.velocity + (diff / diff_mag) * max_change

        # Cap at max speed.
        speed = np.linalg.norm(a.velocity)
        if speed > a.max_speed:
            a.velocity = a.velocity / speed * a.max_speed

        # ── Position update (sub-stepped to prevent tunneling) ─────
        # With dt=0.2s and speed=400 UU/s, the agent moves 80 UU per
        # tick. A thin wall (40 UU) can be completely skipped. Sub-step
        # so no single step exceeds AGENT_BODY_RADIUS (30 UU), which
        # guarantees the overlap test catches every obstacle.
        total_delta = a.velocity * dt
        move_dist = np.linalg.norm(total_delta)

        if move_dist < 0.01:
            new_pos = a.pos.copy()
        else:
            max_step = AGENT_BODY_RADIUS * 0.9  # slightly under radius
            num_sub = max(1, int(np.ceil(move_dist / max_step)))
            step_delta = total_delta / num_sub
            new_pos = a.pos.copy()

            for _ in range(num_sub):
                new_pos = new_pos + step_delta

                for obs in self.obstacles:
                    if obs.contains_circle(new_pos[0], new_pos[1], AGENT_BODY_RADIUS):
                        new_pos[0], new_pos[1] = obs.push_out_circle(
                            new_pos[0], new_pos[1], AGENT_BODY_RADIUS)

            # [Audit §3.2] Wall sliding: project velocity onto wall surface,
            # matching UE CharacterMovementComponent slide-along-wall.
            intended_pos = a.pos + total_delta
            pushed_delta = new_pos - intended_pos
            push_mag = np.linalg.norm(pushed_delta)
            if push_mag > 0.1:
                wall_normal = pushed_delta / push_mag
                a.velocity = a.velocity - wall_normal * float(np.dot(a.velocity, wall_normal))

        # Bounds.
        half = self._arena_half
        new_pos = np.clip(new_pos, -half + AGENT_BODY_RADIUS,
                          half - AGENT_BODY_RADIUS)
        a.pos = new_pos
        # Lock body facing toward current target (simulates UE SetFocus).
        # This is the key difference from velocity: the agent faces the
        # target while moving in any direction (strafing).
        target = self._current_target()
        if target and target.alive:
            to_target = target.pos - a.pos
            d = np.linalg.norm(to_target)
            if d > 1:
                a.facing = to_target / d

    def _execute_target_selection(self, target_idx: int):
        if target_idx >= TARGET_ACTIONS - 1:
            return  # Keep current.
        if target_idx < len(self.targets) and self.targets[target_idx].alive:
            self.current_target_idx = target_idx

    def _execute_combat(self, combat_idx: int, dt: float):
        agent = self.agent
        if not agent.alive:
            return

        # Can't act during weapon switch.
        if agent.is_switching:
            return

        target = self._current_target()
        if not target or not target.alive:
            return

        action = CombatAction(combat_idx)
        slot = agent.active_slot()
        dist = np.linalg.norm(agent.pos - target.pos)
        has_los = check_los(agent.pos, target.pos, self.obstacles)

        if action == CombatAction.FIRE:
            # [Audit §1.7] Single-stage fire: consume ammo and start cooldown
            # IMMEDIATELY on the Fire action, matching C++ TryFire().
            # The weapon actor handles the visual wind-up delay; from the
            # agent's perspective ammo drops on the same tick as the action.
            if slot and slot.is_ready() and slot.has_ammo() and not agent.is_winding_up:
                # Consume ammo immediately (matches C++ UEWLC::TryFire line 174).
                slot.current_ammo -= 1
                slot.cooldown_remaining = slot.fire_cooldown

                if slot.wind_up_time > 0.0:
                    # Wind-up weapon: lock for wind_up + fire_cooldown.
                    # Projectile spawns when wind_up_remaining hits 0
                    # (handled by _resolve_pending_fire).
                    agent.is_winding_up = True
                    agent.wind_up_remaining = slot.wind_up_time
                    agent.set_action_lock(
                        slot.wind_up_time + slot.fire_cooldown,
                        6)  # reason=6 WindUp  [Audit §1.1]
                    # Store target data for deferred projectile spawn.
                    agent._pending_fire = {
                        'target_pos': target.pos.copy(),
                        'target_vel': target.velocity.copy(),
                        'slot_idx': agent.active_weapon,
                    }
                else:
                    # No wind-up — fire immediately.
                    agent.set_action_lock(
                        slot.fire_cooldown * 0.5,
                        1)  # reason=1 Firing  [Audit §1.1]
                    in_range = dist <= slot.weapon_range
                    if in_range and (has_los or slot.can_arc):
                        self._spawn_agent_projectile(slot, target, agent, dist)

        elif action == CombatAction.RELOAD:
            if slot and not slot.is_reloading and slot.current_ammo < slot.max_ammo:
                slot.is_reloading = True
                slot.reload_remaining = slot.reload_time
                agent.set_action_lock(slot.reload_time, 2)  # reason=2 Reloading  [Audit §1.1]
                # Cancel wind-up on reload.
                agent.is_winding_up = False
                agent._pending_fire = None

        elif action == CombatAction.SWITCH_0:
            if len(agent.weapons) > 0 and agent.active_weapon != 0:
                agent.is_switching = True
                agent.switch_remaining = agent.weapon_switch_time
                agent.switch_target_idx = 0
                agent.set_action_lock(
                    agent.weapon_switch_time,
                    5)  # reason=5 Switching  [Audit §1.1] was 3
                agent.is_winding_up = False
                agent._pending_fire = None

        elif action == CombatAction.SWITCH_1:
            if len(agent.weapons) > 1 and agent.active_weapon != 1:
                agent.is_switching = True
                agent.switch_remaining = agent.weapon_switch_time
                agent.switch_target_idx = 1
                agent.set_action_lock(
                    agent.weapon_switch_time,
                    5)  # reason=5 Switching  [Audit §1.1] was 3
                agent.is_winding_up = False
                agent._pending_fire = None

        elif action == CombatAction.MELEE:
            if dist <= agent.melee.range and agent.melee.cooldown_remaining <= 0:
                dmg, target.barrier, _ = compute_damage(
                    agent.melee.damage, agent.attack_stat,
                    target.defence, target.barrier,
                    rng=self.rng)
                target.hp -= dmg
                if target.hp <= 0:
                    target.hp = 0; target.alive = False
                agent.melee.cooldown_remaining = agent.melee.cooldown
                agent.set_action_lock(agent.melee.cooldown, 4)  # reason=4 Melee  [Audit §1.1]
                agent.targets_hit.add(target.target_id)

        elif action == CombatAction.BLOCK:
            # [Audit §1.2] No-op — matches C++ ENeuralCombatAction::Block.
            pass

        elif action == CombatAction.DODGE:
            # Model-controlled dodge. The agent chooses WHEN to dodge
            # (strategic: conserve cooldown for big threats). Direction
            # is the current movement direction, or away from target
            # if stationary. Matches C++ EnemyDodgeComponent.
            if (not agent.is_dodging
                    and agent.dodge_cooldown_remaining <= 0):
                agent.is_dodging = True
                agent.dodge_remaining = agent.dodge_duration

                # Dodge direction: current movement, or away from target.
                speed = np.linalg.norm(agent.velocity)
                if speed > 10:
                    dodge_dir = agent.velocity / speed
                else:
                    away = agent.pos - target.pos
                    away_d = np.linalg.norm(away)
                    dodge_dir = away / max(away_d, 1)

                agent.dodge_direction = dodge_dir
                agent.velocity = np.zeros(2, dtype=np.float32)
                agent.dodge_cooldown_remaining = agent.dodge_cooldown
                agent.set_action_lock(
                    agent.dodge_duration + 0.1, 3)  # reason=3 Dodging

    def _spawn_agent_projectile(self, slot: WeaponSlot, target, agent, dist: float):
        """Spawn a projectile from the agent toward the target.
        
        Extracted from _execute_combat so it can be reused by
        _resolve_pending_fire for deferred wind-up shots.
        """
        # Lead the target: aim at predicted position.
        flight_time = dist / max(slot.projectile_speed, 500.0)
        predicted_pos = target.pos + target.velocity * flight_time

        fire_dir = predicted_pos - agent.pos
        fire_dist = np.linalg.norm(fire_dir)
        if fire_dist > 1:
            fire_dir = fire_dir / fire_dist

        # Add slight inaccuracy (spread).
        spread = self.rng.uniform(-0.05, 0.05)
        cos_s, sin_s = math.cos(spread), math.sin(spread)
        fx, fy = fire_dir[0], fire_dir[1]
        fire_dir = np.array([fx * cos_s - fy * sin_s,
                             fx * sin_s + fy * cos_s], dtype=np.float32)

        if slot.can_arc:
            # Arc projectile — bezier curve over obstacles.
            arc_start = agent.pos.copy()
            arc_end = predicted_pos.copy()
            arc_height = max(200.0, dist * 0.3)
            midpoint = (arc_start + arc_end) / 2.0
            # [Audit §5.2] Removed dead perpendicular code — only forward offset.
            apex = midpoint + fire_dir * dist * 0.1

            arc_length = (np.linalg.norm(apex - arc_start)
                          + np.linalg.norm(arc_end - apex))
            flight_t = arc_length / max(slot.projectile_speed, 500.0)

            proj = SimProjectile(
                pos=arc_start.copy(),
                velocity=fire_dir * slot.projectile_speed,
                speed=slot.projectile_speed,
                damage=slot.base_damage,
                attack_stat=agent.attack_stat,
                crit_chance=agent.crit_chance,
                crit_multiplier=agent.crit_multiplier,
                is_agent_projectile=True,
                target_pos=predicted_pos.copy(),
                is_arc=True,
                max_arc_height=slot.max_arc_height,
                arc_start=arc_start.copy(),
                arc_apex=apex.copy(),
                arc_end=arc_end.copy(),
                arc_flight_time=flight_t,
                arc_impact_radius=150.0,
                hit_radius=35.0,
            )
        else:
            # Straight/beam projectile.
            proj = SimProjectile(
                pos=agent.pos.copy(),
                velocity=fire_dir * slot.projectile_speed,
                speed=slot.projectile_speed,
                damage=slot.base_damage,
                attack_stat=agent.attack_stat,
                crit_chance=agent.crit_chance,
                crit_multiplier=agent.crit_multiplier,
                is_agent_projectile=True,
                target_pos=predicted_pos.copy(),
                hit_radius=25.0,
            )

        self._projectiles.append(proj)

    def _resolve_pending_fire(self):
        """[Audit §1.7] Spawn the projectile once wind-up completes.
        
        Called each step after tick_weapons. In C++, TryFire dispatches
        to the weapon actor which handles the wind-up delay internally.
        This is the Python equivalent: when wind_up_remaining hits 0 and
        _pending_fire is set, the projectile spawns.
        """
        agent = self.agent
        if agent._pending_fire is None:
            return
        if agent.is_winding_up:
            return  # Still winding up

        pf = agent._pending_fire
        agent._pending_fire = None

        # Retrieve the weapon slot that was used (ammo already consumed).
        slot_idx = pf['slot_idx']
        if slot_idx < 0 or slot_idx >= len(agent.weapons):
            return
        slot = agent.weapons[slot_idx]

        # Build a lightweight target-like object from the stored data
        # so _spawn_agent_projectile can lead the shot.
        class _DeferredTarget:
            def __init__(self, pos, velocity):
                self.pos = pos
                self.velocity = velocity
        target = _DeferredTarget(pf['target_pos'], pf['target_vel'])

        dist = np.linalg.norm(agent.pos - target.pos)
        in_range = dist <= slot.weapon_range
        has_los = check_los(agent.pos, target.pos, self.obstacles)
        if in_range and (has_los or slot.can_arc):
            self._spawn_agent_projectile(slot, target, agent, dist)

    def _target_attacks_agent(self, dt: float):
        """Targets attack the agent with melee and/or ranged weapons.
        Melee targets deal high burst damage up close.
        Ranged targets fire projectiles that can miss.
        Mixed targets do both depending on distance."""
        if not self.agent.alive:
            return

        a = self.agent

        for t in self.targets:
            if not t.alive:
                continue

            dist = np.linalg.norm(a.pos - t.pos)

            # ── Melee attack (melee and mixed roles) ─────────────
            can_melee = t.combat_role in ("melee", "mixed")
            if can_melee and dist <= t.melee_range:
                if t.melee_cooldown_remaining <= 0 and t.commitment <= 0:
                    # Check mana (melee usually free, but configurable).
                    if t.melee_mana_cost > 0 and not t.spend_mana(t.melee_mana_cost):
                        continue
                    t.melee_cooldown_remaining = t.melee_cooldown
                    t.start_commitment(t.melee_commit_time)
                    t.focus_target_id = 0

                    # [Audit §5.3] Full invulnerability during dodge,
                    # matching C++ DodgeComponent.InvulnerabilityDuration.
                    if a.is_dodging:
                        continue

                    # Melee can hit even when not facing (swing is wide).
                    dmg, a.barrier, _ = compute_damage(
                        t.melee_damage, t.melee_stat,
                        a.defence, a.barrier,
                        t.crit_chance, t.crit_multiplier,
                        rng=self.rng)  # [Audit §5.4, §4.1]
                    a.hp -= dmg
                    self.threat_table.record_damage(t.target_id, dmg)  # [Audit §1.3]
                    if a.hp <= 0:
                        a.hp = 0; a.alive = False
                    continue

            # ── Facing check for ranged attacks ──────────────────
            # Targets need to be roughly facing the agent to shoot.
            # This creates the flanking window — attacking from behind
            # means the target can't return fire until it turns around.
            to_agent = (a.pos - t.pos)
            to_agent_d = np.linalg.norm(to_agent)
            if to_agent_d > 1:
                facing_dot = float(np.dot(t.facing, to_agent / to_agent_d))
            else:
                facing_dot = 1.0

            # Vision cone: ~140° front arc (dot > 0.17 ≈ cos(80°)).
            # Below this, the target can't see the agent and won't shoot.
            if facing_dot < 0.17:
                continue  # Agent is behind target — no ranged attack.

            # Accuracy penalty for targets not fully facing the agent.
            # Fully facing (dot=1.0): normal accuracy.
            # Partially facing (dot=0.5): 70% accuracy.
            # Barely facing (dot=0.2): 40% accuracy.
            facing_accuracy = 0.3 + 0.7 * max(0.0, facing_dot)

            # ── Ranged attack (ranged and mixed roles) ───────────
            can_ranged = t.combat_role in ("ranged", "mixed")
            if not can_ranged:
                continue

            if t.attack_cooldown_remaining > 0:
                continue
            if t.commitment > 0:
                continue  # Already casting/swinging.
            if dist > t.attack_range:
                continue
            if not check_los(t.pos, a.pos, self.obstacles):
                continue
            # Check mana cost.
            if t.attack_mana_cost > 0 and not t.spend_mana(t.attack_mana_cost):
                continue

            # Fire — spawn projectile aimed at agent's predicted position.
            t.attack_cooldown_remaining = t.attack_cooldown
            t.start_commitment(t.attack_cast_time)
            t.focus_target_id = 0

            # Lead the shot: predict where agent will be.
            flight_time = dist / max(t.attack_projectile_speed, 500.0)
            predicted_agent_pos = a.pos + a.velocity * flight_time

            fire_dir = predicted_agent_pos - t.pos
            fire_dist = np.linalg.norm(fire_dir)
            if fire_dist < 1:
                continue
            fire_dir = fire_dir / fire_dist

            # Accuracy based on facing (same as before).
            # Wider spread when not fully facing the agent.
            spread_scale = 0.15 * (1.0 - facing_accuracy)
            spread = self.rng.uniform(-spread_scale, spread_scale)
            cos_s, sin_s = math.cos(spread), math.sin(spread)
            fx, fy = fire_dir[0], fire_dir[1]
            fire_dir = np.array([fx * cos_s - fy * sin_s,
                                  fx * sin_s + fy * cos_s], dtype=np.float32)

            proj = SimProjectile(
                pos=t.pos.copy(),
                velocity=fire_dir * t.attack_projectile_speed,
                speed=t.attack_projectile_speed,
                damage=t.attack_damage,
                attack_stat=t.attack_stat,
                crit_chance=t.crit_chance,         # [Audit §5.4]
                crit_multiplier=t.crit_multiplier, # [Audit §5.4]
                is_agent_projectile=False,
                source_id=t.target_id,
                target_pos=predicted_agent_pos.copy(),
                hit_radius=30.0,  # C++ is 15 UU; slightly larger for 2D approximation
            )
            self._projectiles.append(proj)

            # # Hit.
            # dmg, a.barrier, _ = compute_damage(
            #     t.attack_damage, t.attack_stat,
            #     a.defence, a.barrier)
            # a.hp -= dmg
            # if a.hp <= 0:
            #     a.hp = 0; a.alive = False
            
    def _tick_projectiles(self, dt: float):
        """Advance all projectiles and apply damage on hit."""
        for proj in self._projectiles:
            hits = proj.tick(dt, self.targets, self.agent,
                             self.obstacles, self._arena_half)

            # Track agent hits for targets_hit set.
            if proj.is_agent_projectile:
                for (actor, dmg, was_crit) in hits:
                    if hasattr(actor, 'target_id'):
                        self.agent.targets_hit.add(actor.target_id)
            else:
                # [Audit §1.3] Record damage from enemy projectiles for
                # threat-based priority scoring.
                for (actor, dmg, was_crit) in hits:
                    if proj.source_id >= 0:
                        self.threat_table.record_damage(proj.source_id, dmg)

        # Remove dead projectiles.
        self._projectiles = [p for p in self._projectiles if p.alive]

    def _try_auto_dodge(self):
        """Autonomous dodge triggered by incoming projectile threat.
        
        Matches C++ EnemyDodgeComponent::EvaluateThreat() which scans for
        nearby projectiles and triggers a dodge if one is about to hit.
        The dodge direction is perpendicular to the incoming projectile,
        not directly away — this is more effective at avoiding linear shots.
        """
        a = self.agent
        best_threat = None
        best_tta = a.auto_dodge_threat_tta  # Only react if TTA < threshold

        for p in self._projectiles:
            if not p.alive or p.is_agent_projectile:
                continue  # Only dodge enemy projectiles

            # Time to arrival.
            rel = a.pos - p.pos
            dist = np.linalg.norm(rel)
            speed = np.linalg.norm(p.velocity)
            if speed < 10:
                continue
            
            # Check if projectile is heading roughly toward agent.
            proj_dir = p.velocity / speed
            to_agent = rel / max(dist, 1.0)
            heading_toward = float(np.dot(proj_dir, to_agent))
            if heading_toward < 0.3:
                continue  # Not heading our way

            tta = dist / speed
            if tta < best_tta and dist < a.auto_dodge_threat_range:
                best_tta = tta
                best_threat = p

        if best_threat is None:
            return

        # Trigger dodge perpendicular to the incoming projectile.
        proj_dir = best_threat.velocity / max(np.linalg.norm(best_threat.velocity), 1.0)
        perp = np.array([proj_dir[1], -proj_dir[0]], dtype=np.float32)
        # Randomly pick left or right perpendicular.
        dodge_dir = perp * self.rng.choice([-1.0, 1.0])
        # Slight backward component.
        backward = (a.pos - best_threat.pos)
        backward = backward / max(np.linalg.norm(backward), 1.0)
        dodge_dir = dodge_dir * 0.8 + backward * 0.2
        dodge_dir = dodge_dir / max(np.linalg.norm(dodge_dir), 0.01)

        a.is_dodging = True
        a.dodge_remaining = a.dodge_duration
        a.dodge_cooldown_remaining = a.dodge_cooldown
        a.dodge_direction = dodge_dir
        a.velocity = np.zeros(2, dtype=np.float32)  # [Audit §3.6] Matches C++ StopMovement()
        a.set_action_lock(a.dodge_duration + 0.1, 3)  # reason=3 Dodging  [Audit §1.1]

    # ═════════════════════════════════════════════════════════════
    #  Target Priority Scoring  [Audit §1.3]
    # ═════════════════════════════════════════════════════════════

    def _score_target(self, t) -> float:
        """Composite priority score matching C++ EvaluateTargetPriority.
        
        Weights mirror FTargetPriorityConfig defaults:
          PlayerControlledWeight = 10
          LowHealthWeight        = 20
          DamageThreatWeight     = 25
          DistanceWeight         = 30
          LOSWeight              = 15
          CurrentTargetBonus     = 5
          RandomJitter           = 2
        
        NOTE: If C++ config is changed via blueprint/data asset, these
        values must be updated to match. See EnemyPerceptionComponent.cpp
        EvaluateTargetPriority() and FTargetPriorityConfig.
        """
        a = self.agent
        dist = np.linalg.norm(a.pos - t.pos)
        max_range = 3000.0
        norm_dist = min(dist / max_range, 1.0)

        pc       = 10.0 if t.is_player_controlled else 0.0
        low_hp   = (1.0 - t.hp_fraction()) * 20.0
        threat   = self.threat_table.get_normalised_threat(t.target_id) * 25.0
        distance = (1.0 - norm_dist) * 30.0
        los      = 15.0 if check_los(a.pos, t.pos, self.obstacles) else 0.0
        sticky   = 5.0 if t.target_id == self.current_target_idx else 0.0
        jitter   = self.rng.uniform(0.0, 2.0)

        return pc + low_hp + threat + distance + los + sticky + jitter

    def _get_sorted_targets(self) -> list:
        """Return alive targets sorted by priority (highest first).
        
        Replaces the old distance-only sort, matching C++ ScoredTargets
        ordering from EvaluateTargetPriority.
        """
        alive = [t for t in self.targets if t.alive]
        return sorted(alive, key=lambda t: -self._score_target(t))

    # ═════════════════════════════════════════════════════════════
    #  Observation Builder (198 floats, matches C++ exactly)
    # ═════════════════════════════════════════════════════════════

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        a = self.agent
        target = self._current_target()
        idx = 0

        # ── Self State (21) ──────────────────────────────────────
        obs[idx] = a.hp_fraction(); idx += 1                   # 0
        obs[idx] = min(a.defence / 200.0, 1.0); idx += 1      # 1
        speed = np.linalg.norm(a.velocity)
        obs[idx] = min(speed / a.max_speed, 1.0) if a.max_speed > 0 else 0; idx += 1  # 2
        obs[idx] = 0.0; idx += 1  # stunned                   # 3
        obs[idx] = 0.0; idx += 1  # slowed                    # 4
        for _ in range(6): obs[idx] = 0.0; idx += 1           # 5-10 debuffs
        vel_dir = a.velocity / max(speed, 1.0)
        obs[idx] = vel_dir[0]; idx += 1                        # 11
        obs[idx] = vel_dir[1]; idx += 1                        # 12
        obs[idx] = min(a.combat_time / 120.0, 1.0); idx += 1  # 13
        # [Audit §1.8] C++ TraceHeightAboveGround returns ~88 UU (capsule
        # half-height) for a grounded character. 88/500 ≈ 0.176.
        obs[idx] = 0.176; idx += 1  # height above ground   # 14

        # Action lock state (matches C++ IsActionLocked / ActionLockProgress).
        obs[idx] = 1.0 if a.is_action_locked else 0.0; idx += 1  # 15
        obs[idx] = a.action_lock_progress; idx += 1               # 16
        # [Audit §1.1] Lock reason: use C++ formula directly:
        #   static_cast<float>(LockReason) / static_cast<float>(EActionLockReason::WindUp)
        # With corrected enum: None=0, Firing=1, Reloading=2, Dodging=3,
        #                       Melee=4, Switching=5, WindUp=6
        obs[idx] = (a.action_lock_reason / 6.0) if a.is_action_locked else 0.0
        idx += 1                                               # 17

        obs[idx] = 1.0 if a.is_dodging else 0.0; idx += 1     # 18 is dodging
        obs[idx] = 1.0 if a.dodge_cooldown_remaining <= 0 else 0.0; idx += 1  # 19 dodge ready
        obs[idx] = 1.0 if a.is_dodging else 0.0; idx += 1     # 20 invulnerable

        # ── Weapon State (22) ────────────────────────────────────
        slot = a.active_slot()
        n_slots = len(a.weapons)
        obs[idx] = a.active_weapon / max(n_slots - 1, 1) if n_slots > 0 else 0; idx += 1
        obs[idx] = slot.ammo_fraction() if slot else 0; idx += 1
        obs[idx] = 1.0 if (slot and slot.is_ready() and slot.has_ammo()) else 0; idx += 1
        obs[idx] = 1.0 if (slot and slot.is_reloading) else 0; idx += 1
        obs[idx] = (1.0 - slot.reload_remaining / slot.reload_time
                    if slot and slot.is_reloading and slot.reload_time > 0 else 1.0); idx += 1
        obs[idx] = min((slot.weapon_range if slot else 0) / 5000, 1.0); idx += 1
        obs[idx] = (slot.cooldown_remaining / slot.fire_cooldown
                    if slot and slot.fire_cooldown > 0 else 0); idx += 1
        obs[idx] = min((slot.wind_up_time if slot else 0) / 3.0, 1.0); idx += 1
        obs[idx] = 1.0 if (slot and slot.can_arc) else 0; idx += 1
        obs[idx] = 1.0 if (slot and slot.is_ranged) else 0; idx += 1
        # Other weapon slots (12 floats = 3 slots × 4: ammo, range, reloading, can_arc).
        for si in range(3):
            actual = si if si < a.active_weapon else si + 1
            if actual < n_slots:
                w = a.weapons[actual]
                obs[idx] = w.ammo_fraction(); idx += 1
                obs[idx] = min(w.weapon_range / 5000, 1.0); idx += 1
                obs[idx] = 1.0 if w.is_reloading else 0; idx += 1
                obs[idx] = 1.0 if w.can_arc else 0; idx += 1
            else:
                idx += 4

        # ── Archetype (7) ────────────────────────────────────────
        for ai in range(4):
            obs[idx] = 1.0 if a.archetype == ai else 0; idx += 1
        weapon_range = slot.weapon_range if slot else 1000
        obs[idx] = min(weapon_range * 0.6 / 5000, 1.0); idx += 1
        obs[idx] = 1.0 if a.any_ammo() else 0; idx += 1
        obs[idx] = 1.0 if a.melee.cooldown_remaining <= 0 else 0; idx += 1

        # ── Primary Target (24) ──────────────────────────────────
        if target and target.alive:
            rel = target.pos - a.pos
            dist = np.linalg.norm(rel)
            obs[idx] = np.clip(rel[0] / 5000, -1, 1); idx += 1
            obs[idx] = np.clip(rel[1] / 5000, -1, 1); idx += 1
            obs[idx] = min(dist / 5000, 1.0); idx += 1
            obs[idx] = target.hp_fraction(); idx += 1
            obs[idx] = 1.0 if dist <= weapon_range else 0; idx += 1
            has_los = check_los(a.pos, target.pos, self.obstacles)
            obs[idx] = 1.0 if has_los else 0; idx += 1
            obs[idx] = 1.0; idx += 1  # in sight cone (always true in sim)
            obs[idx] = float(np.dot(a.facing, rel / max(dist, 1)))
            idx += 1
            to_agent_dir = -rel / max(dist, 1)  # unit vector from target toward agent
            target_facing_dot = float(np.dot(target.facing, to_agent_dir))
            obs[idx] = target_facing_dot; idx += 1
            t_vel = target.velocity
            obs[idx] = np.clip(t_vel[0] / 600, -1, 1); idx += 1
            obs[idx] = np.clip(t_vel[1] / 600, -1, 1); idx += 1
            idx += 2  # acceleration (skip)
            obs[idx] = min(70.0 / max(dist, 1), 1.0); idx += 1  # angular size
            obs[idx] = 1.0 if target.is_player_controlled else 0; idx += 1
            blocked, cover_h = is_behind_cover(a.pos, target.pos, self.obstacles)
            obs[idx] = 1.0 if (blocked and cover_h < 200) else 0; idx += 1
            obs[idx] = min(cover_h / 500, 1.0) if blocked else 0; idx += 1
            obs[idx] = 1.0 if dist <= a.melee.range else 0; idx += 1
            closing = 0
            if dist > 1:
                to_target = rel / dist
                closing = float(np.dot(a.velocity - t_vel, to_target))
            obs[idx] = np.clip(closing / 1000, -1, 1); idx += 1
            # ── New fields (Phase 1) ─────────────────────────
            obs[idx] = target.character_type_float; idx += 1
            obs[idx] = target.mana_fraction(); idx += 1
            obs[idx] = target.commitment; idx += 1
            obs[idx] = target.gap_closer_threat(a.pos); idx += 1
        else:
            idx += 24

        # ── Hostile Targets (68 = 4 × 17) ───────────────────────
        # [Audit §1.3] Sort by priority score, not distance, matching
        # C++ ScoredTargets from EvaluateTargetPriority.
        sorted_targets = self._get_sorted_targets()
        for si in range(4):
            if si < len(sorted_targets):
                t = sorted_targets[si]
                rel = t.pos - a.pos; dist = np.linalg.norm(rel)
                obs[idx] = 1.0; idx += 1  # occupied
                obs[idx] = np.clip(rel[0] / 5000, -1, 1); idx += 1
                obs[idx] = np.clip(rel[1] / 5000, -1, 1); idx += 1
                obs[idx] = min(dist / 5000, 1.0); idx += 1
                obs[idx] = t.hp_fraction(); idx += 1
                obs[idx] = 1.0 if check_los(a.pos, t.pos, self.obstacles) else 0; idx += 1
                obs[idx] = 1.0 if t.is_player_controlled else 0; idx += 1
                # Facing: C++ ComputeRelativeFacing(Target, Owner), raw dot [-1,1].
                to_agent = (a.pos - t.pos)
                to_agent_d = np.linalg.norm(to_agent)
                if to_agent_d > 1:
                    facing_dot = float(np.dot(t.facing, to_agent / to_agent_d))
                    obs[idx] = facing_dot
                else:
                    obs[idx] = 1.0
                idx += 1
                # [Audit §1.3] Priority score: normalised against max possible ~120.
                obs[idx] = min(self._score_target(t) / 120.0, 1.0); idx += 1
                # [Audit §1.3] Threat level from accumulated damage.
                threat = self.threat_table.get_threat(t.target_id)
                obs[idx] = min(threat / 200.0, 1.0); idx += 1
                # Velocity (2D normalised).
                obs[idx] = np.clip(t.velocity[0] / 600, -1, 1); idx += 1
                obs[idx] = np.clip(t.velocity[1] / 600, -1, 1); idx += 1
                # [Audit §1.5] Is targeting me: continuous facing signal.
                if to_agent_d > 1:
                    targeting_dot = float(np.dot(t.facing, to_agent / to_agent_d))
                    obs[idx] = max(0.0, min(1.0, targeting_dot))
                else:
                    obs[idx] = 1.0
                idx += 1
                # ── New fields (Phase 1) ─────────────────────────
                obs[idx] = t.character_type_float; idx += 1
                obs[idx] = t.mana_fraction(); idx += 1
                obs[idx] = t.commitment; idx += 1
                obs[idx] = t.gap_closer_threat(a.pos); idx += 1
            else:
                idx += 17

        # ── Allied Robots (45 = 3 × 15) ─────────────────────────
        allies = getattr(self, 'allies', [])
        for si in range(3):
            if si < len(allies) and allies[si].alive:
                ally = allies[si]
                rel = ally.pos - a.pos
                dist = np.linalg.norm(rel)
                obs[idx] = 1.0; idx += 1  # occupied
                obs[idx] = np.clip(rel[0] / 5000, -1, 1); idx += 1
                obs[idx] = np.clip(rel[1] / 5000, -1, 1); idx += 1
                obs[idx] = min(dist / 5000, 1.0); idx += 1
                obs[idx] = ally.hp_fraction(); idx += 1
                obs[idx] = 1.0 if check_los(a.pos, ally.pos, self.obstacles) else 0; idx += 1
                # Velocity (normalised).
                obs[idx] = np.clip(ally.velocity[0] / 600, -1, 1); idx += 1
                obs[idx] = np.clip(ally.velocity[1] / 600, -1, 1); idx += 1
                # Ally facing dot toward agent (computed from velocity).
                ally_speed = np.linalg.norm(ally.velocity)
                if ally_speed > 10:
                    ally_facing = ally.velocity / ally_speed
                else:
                    # Stationary: face toward current target.
                    if (hasattr(ally, 'target_idx')
                            and ally.target_idx < len(sorted_targets)
                            and sorted_targets[ally.target_idx].alive):
                        to_tgt = sorted_targets[ally.target_idx].pos - ally.pos
                        to_tgt_d = np.linalg.norm(to_tgt)
                        ally_facing = to_tgt / max(to_tgt_d, 1.0)
                    else:
                        ally_facing = np.array([1.0, 0.0], dtype=np.float32)
                to_me = a.pos - ally.pos
                to_me_d = np.linalg.norm(to_me)
                if to_me_d > 1:
                    obs[idx] = float(np.dot(ally_facing, to_me / to_me_d))
                else:
                    obs[idx] = 1.0
                idx += 1
                # Weapon state (normalised).
                obs[idx] = ally.ammo_fraction if hasattr(ally, 'ammo_fraction') else 1.0; idx += 1
                obs[idx] = 1.0 if getattr(ally, 'is_reloading', False) else 0.0; idx += 1
                obs[idx] = min(getattr(ally, 'active_weapon_fire_cooldown', 0) / 2.0, 1.0); idx += 1
                # ── New coordination fields (Phase 1) ────────────
                # Target index: which hostile is this ally engaging?
                ally_tgt = getattr(ally, 'target_idx', -1)
                obs[idx] = (ally_tgt + 1) / 5.0; idx += 1  # -1→0, 0→0.2, 3→0.8
                # Combat action: what is this ally doing?
                ally_action = getattr(ally, 'current_combat_action', 0)
                obs[idx] = ally_action / 7.0; idx += 1  # normalise to [0,1]
                # Flanking angle: cos between my→target and ally→target.
                flanking = 0.0
                if (ally_tgt >= 0
                        and ally_tgt < len(sorted_targets)
                        and sorted_targets[ally_tgt].alive):
                    shared_tgt = sorted_targets[ally_tgt]
                    my_to_tgt = shared_tgt.pos - a.pos
                    ally_to_tgt = shared_tgt.pos - ally.pos
                    my_d = np.linalg.norm(my_to_tgt)
                    ally_d = np.linalg.norm(ally_to_tgt)
                    if my_d > 1 and ally_d > 1:
                        flanking = float(np.dot(
                            my_to_tgt / my_d, ally_to_tgt / ally_d))
                obs[idx] = flanking; idx += 1
            else:
                idx += 15

        # ── Spatial Ring (8) ─────────────────────────────────────
        # 8 sphere sweeps at 45° spacing. Each sweep uses a sphere
        # with radius = AGENT_BODY_RADIUS, answering "can my body fit
        # through there?" Detects obstacles and arena boundaries only.
        TRACE_LEN = 1500.0
        SWEEP_RADIUS = AGENT_BODY_RADIUS  # 30 UU
        half = self._arena_half
        NUM_SPATIAL_RAYS = 8
        angles = [i * (360.0 / NUM_SPATIAL_RAYS) for i in range(NUM_SPATIAL_RAYS)]

        # Effective arena boundary: the sphere center can't get closer
        # than SWEEP_RADIUS to the wall.
        effective_half = half - SWEEP_RADIUS

        for ang in angles:
            rad = math.radians(ang)
            direction = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            probe = a.pos + direction * TRACE_LEN
            blocked_dist = 1.0

            # Sweep against obstacles (inflated AABB).
            for o in self.obstacles:
                t = _sphere_sweep_aabb_t(a.pos, probe, o, SWEEP_RADIUS)
                if t is not None and t < blocked_dist:
                    blocked_dist = t

            # Sweep against arena boundaries (pulled inward by sweep radius).
            for axis in range(2):
                d = direction[axis] * TRACE_LEN
                if abs(d) > 1e-6:
                    wall = effective_half if d > 0 else -effective_half
                    t_wall = (wall - a.pos[axis]) / d
                    if 0 < t_wall < blocked_dist:
                        blocked_dist = t_wall

            obs[idx] = blocked_dist; idx += 1

        # ── Cover Height (8) ──────────────────────────────────────
        # Continuous obstacle height per direction, normalised by 500.
        # 8 rays matching the spatial ring.
        HIGH_TRACE_HEIGHT = 350.0
        for ang in angles:
            rad = math.radians(ang)
            direction = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            probe = a.pos + direction * 1500
            obstacle_height = 0.0
            for o in self.obstacles:
                if _ray_aabb_intersect(a.pos, probe, o):
                    if o.height < HIGH_TRACE_HEIGHT:
                        obstacle_height = max(obstacle_height, o.height)
                    else:
                        obstacle_height = HIGH_TRACE_HEIGHT
                    break
            obs[idx] = min(obstacle_height / 500.0, 1.0); idx += 1

        # ── Threat Sensing (8) ───────────────────────────────────
        # [Audit §1.4] Matches C++ GatherThreatSensing.
        # Now tracks top-3 nearest incoming projectiles for multi-threat evasion.
        scan_radius = 600.0
        melee_threat_dist = 350.0

        # Collect ALL incoming threats, then sort by distance.
        incoming_threats = []  # list of (norm_dist, tta_n, dir_x, dir_y)

        for p in self._projectiles:
            if not p.alive or p.is_agent_projectile:
                continue
            rel = a.pos - p.pos
            dist = np.linalg.norm(rel)
            if dist > scan_radius:
                continue
            speed = np.linalg.norm(p.velocity)
            if speed < 1e-3:
                continue
            vel_dir = p.velocity / speed
            to_me = rel / max(dist, 1.0)
            if float(np.dot(vel_dir, to_me)) <= 0.5:
                continue
            norm_dist = min(dist / scan_radius, 1.0)
            tta_n = min(dist / speed, 2.0) / 2.0
            incoming_threats.append((norm_dist, tta_n, float(vel_dir[0]), float(vel_dir[1])))

        # Sort by distance (nearest first).
        incoming_threats.sort(key=lambda t: t[0])
        threat_count = len(incoming_threats)

        # Extract top-3 (pad with safe defaults if fewer).
        def _get_threat(idx):
            if idx < len(incoming_threats):
                return incoming_threats[idx]
            return (1.0, 1.0, 0.0, 0.0)  # safe defaults

        t1 = _get_threat(0)  # nearest (goes to existing indices 174-177)
        t2 = _get_threat(1)  # 2nd nearest (goes to new indices 198-200)
        t3 = _get_threat(2)  # 3rd nearest (goes to new indices 201-203)

        # Nearest melee threat.
        nearest_melee_dist_n = 1.0
        nearest_melee_dir = np.zeros(2, dtype=np.float32)
        for t in self.targets:
            if not t.alive:
                continue
            dist = np.linalg.norm(a.pos - t.pos)
            if dist > melee_threat_dist:
                continue
            norm_dist = min(dist / melee_threat_dist, 1.0)
            if norm_dist < nearest_melee_dist_n:
                nearest_melee_dist_n = norm_dist
                melee_rel = t.pos - a.pos
                melee_d = np.linalg.norm(melee_rel)
                nearest_melee_dir = melee_rel / max(melee_d, 1.0)

        # Existing threat features (indices 174-181, unchanged).
        obs[idx] = t1[0]; idx += 1              # 174 proj 1 dist
        obs[idx] = t1[1]; idx += 1              # 175 proj 1 TTA
        obs[idx] = t1[2]; idx += 1              # 176 proj 1 dir X
        obs[idx] = t1[3]; idx += 1              # 177 proj 1 dir Y
        obs[idx] = nearest_melee_dist_n; idx += 1  # 178 melee dist
        obs[idx] = nearest_melee_dir[0]; idx += 1  # 179 melee dir X
        obs[idx] = nearest_melee_dir[1]; idx += 1  # 180 melee dir Y
        obs[idx] = 1.0 if a.dodge_cooldown_remaining <= 0 else 0; idx += 1  # 181 dodge avail

        # ── Navmesh Viability (9) ────────────────────────────────
        # [Audit §2.2] C++ FVector(1,1,0).GetSafeNormal() = 1/√2, not 0.7.
        _S = 0.7071067811865476  # 1/√2
        nav_dirs = [(1, 0), (_S, _S), (0, 1), (-_S, _S),
                    (-1, 0), (-_S, -_S), (0, -1), (_S, -_S), (0, 0)]
        for dx, dy in nav_dirs:
            probe = a.pos + np.array([dx, dy], dtype=np.float32) * 400
            navigable = True
            for o in self.obstacles:
                if o.contains(probe[0], probe[1]):
                    navigable = False; break
            half = self._arena_half
            if abs(probe[0]) > half or abs(probe[1]) > half:
                navigable = False
            obs[idx] = 1.0 if navigable else 0; idx += 1

        # ── Group Summary (6) ────────────────────────────────────
        alive_hostiles = sum(1 for t in self.targets if t.alive)
        alive_allies = 0  # Base env is single-agent; no allies
        avg_hostile_hp = (sum(t.hp_fraction() for t in self.targets if t.alive)
                         / max(alive_hostiles, 1))
        obs[idx] = min(alive_allies / 10.0, 1.0); idx += 1  # alive allies (normalised by 10)
        obs[idx] = min(alive_hostiles / 4, 1.0); idx += 1
        obs[idx] = 0.0; idx += 1  # avg ally HP
        obs[idx] = avg_hostile_hp; idx += 1
        # [Audit §1.11] C++ computes AliveAllies / (AliveAllies + AliveHostiles).
        # Does NOT count the agent itself. In single-agent env with 0 allies,
        # this is 0 / (0 + N) = 0.0 — NOT 0.5.
        total = alive_allies + alive_hostiles
        obs[idx] = (alive_allies / total) if total > 0 else 0.5; idx += 1
        obs[idx] = 1.0 if alive_hostiles > alive_allies else 0; idx += 1  # outnumbered

        # ── Spawn Leash (1) ──────────────────────────────────────
        spawn_dist = np.linalg.norm(a.pos - a.spawn_pos)
        leash_norm = a.combat_leash_range if a.combat_leash_range > 0 else 5000.0
        obs[idx] = float(np.clip(spawn_dist / leash_norm, 0.0, 1.0)); idx += 1

        # ══════════════════════════════════════════════════════════
        #  NEW FEATURES (indices 198-210, appended after SpawnLeash)
        # ══════════════════════════════════════════════════════════

        # ── Extended Threat (7) ─────────────────────────────────
        obs[idx] = t2[0]; idx += 1                             # 198 proj 2 dist
        obs[idx] = t2[2]; idx += 1                             # 199 proj 2 dir X
        obs[idx] = t2[3]; idx += 1                             # 200 proj 2 dir Y
        obs[idx] = t3[0]; idx += 1                             # 201 proj 3 dist
        obs[idx] = t3[2]; idx += 1                             # 202 proj 3 dir X
        obs[idx] = t3[3]; idx += 1                             # 203 proj 3 dir Y
        obs[idx] = min(threat_count / 5.0, 1.0); idx += 1     # 204 threat count

        # ── Weapon Can-Hit-Target (4) ───────────────────────────
        # Per weapon slot: 1.0 if weapon has ammo, target in range,
        # and path to target exists (LOS for direct, or arc for arc weapons
        # when target is behind cover). Answers "should I switch?" directly.
        target = self._current_target()
        target_dist = np.linalg.norm(a.pos - target.pos) if target and target.alive else 9999
        target_has_los = check_los(a.pos, target.pos, self.obstacles) if target and target.alive else False
        target_behind_cover = not target_has_los and target is not None and target.alive

        for wi in range(4):
            if wi < len(a.weapons):
                w = a.weapons[wi]
                in_range = target_dist <= w.weapon_range
                has_path = target_has_los if not w.can_arc else (target_has_los or target_behind_cover)
                can_hit = w.has_ammo() and in_range and has_path and not w.is_reloading
                obs[idx] = 1.0 if can_hit else 0.0
            idx += 1                                           # 205-208

        # ── Total Ammo Fraction (1) ─────────────────────────────
        if len(a.weapons) > 0:
            total_ammo = sum(w.ammo_fraction() for w in a.weapons) / len(a.weapons)
        else:
            total_ammo = 0.0
        obs[idx] = total_ammo; idx += 1                        # 209

        # ── Targets Killed Fraction (1) ─────────────────────────
        # Fraction of hostile targets killed in this encounter.
        # Matches C++ UpdateTargetDefeatTracking() + GatherTargetsKilled().
        #
        # Reset behaviour:
        #   - Python: implicitly reset on env.reset() — self.targets is
        #     rebuilt with all targets alive. Dead targets stay in the list
        #     (alive=False) so this fraction grows during the episode.
        #   - C++: EncounterKilledHostiles and EncounterTotalHostiles are
        #     reset in StartObserving() when combat begins. Tracked per
        #     encounter via weak pointers. Targets that leave without dying
        #     (actor destroyed) reduce the total; targets that die increase
        #     the killed count. StopObserving() ends the encounter.
        total_hostiles = len([t for t in self.targets if not t.is_player_controlled])
        killed_hostiles = len([t for t in self.targets if not t.is_player_controlled and not t.alive])
        obs[idx] = (killed_hostiles / max(total_hostiles, 1)); idx += 1  # 210

        # ── Arc Clearance per Weapon (4) ─────────────────────────
        # MaxArcableObstacleHeight per weapon slot, normalised by 3000 UU.
        # The model compares these against CoverHeight per direction [166-173]
        # to decide which weapons can fire over which cover.
        #   0.0 = weapon cannot arc (or slot empty)
        #   1.0 = unlimited clearance (max_arc_height <= 0 means no limit)
        for wi in range(4):
            if wi < len(a.weapons):
                w = a.weapons[wi]
                if w.can_arc:
                    if w.max_arc_height <= 0:
                        obs[idx] = 1.0  # unlimited clearance
                    else:
                        obs[idx] = min(w.max_arc_height / 3000.0, 1.0)
            idx += 1                                               # 240-243

        # ── Player Patterns (5) ──────────────────────────────────
        # EMAs of player behavior: aggression, evasion, predictability,
        # preferred range, mana burn rate. Updated once per tick in
        # the environment step. Gives the model awareness of player
        # tendencies for adaptive strategy.
        if hasattr(self, '_player_patterns'):
            patterns = self._player_patterns.as_array()
            for p in patterns:
                obs[idx] = p; idx += 1
        else:
            idx += 5                                               # 244-248

        return obs

    # ═════════════════════════════════════════════════════════════
    #  CombatState Builder (for reward function)
    # ═════════════════════════════════════════════════════════════

    def _build_combat_state(self) -> CombatState:
        a = self.agent
        target = self._current_target()
        slot = a.active_slot()
        dist = np.linalg.norm(a.pos - target.pos) if target else 9999

        has_los = check_los(a.pos, target.pos, self.obstacles) if target else False
        blocked, cover_h = is_behind_cover(a.pos, target.pos, self.obstacles) if target else (False, 0)

        in_optimal = False
        if slot and target:
            in_optimal = slot.optimal_min <= dist <= slot.optimal_max

        # Check if agent is behind cover relative to target.
        behind_cover = False
        if target:
            behind_cover, _ = is_behind_cover(target.pos, a.pos, self.obstacles)

        # ── Boundary / corner detection ──────────────────────────
        half = self._arena_half
        wall_threshold = 150.0
        corner_threshold = 200.0
        near_left = a.pos[0] < (-half + wall_threshold)
        near_right = a.pos[0] > (half - wall_threshold)
        near_bottom = a.pos[1] < (-half + wall_threshold)
        near_top = a.pos[1] > (half - wall_threshold)
        near_wall = near_left or near_right or near_bottom or near_top
        walls_near = sum([near_left, near_right, near_bottom, near_top])
        in_corner = walls_near >= 2 and (
            a.pos[0] < (-half + corner_threshold) or a.pos[0] > (half - corner_threshold)
        ) and (
            a.pos[1] < (-half + corner_threshold) or a.pos[1] > (half - corner_threshold)
        )

        # [Fix] Continuous wall proximity: 0.0 = touching wall, 1.0 = at
        # threshold, >1.0 = safely away. Gives the reward function a
        # gradient to push the agent away from walls rather than a flat
        # penalty that provides no directional signal.
        dist_to_nearest_wall = min(
            a.pos[0] - (-half),   # left wall
            half - a.pos[0],      # right wall
            a.pos[1] - (-half),   # bottom wall
            half - a.pos[1],      # top wall
        )
        wall_proximity = max(0.0, dist_to_nearest_wall / wall_threshold)

        # ── Movement state ───────────────────────────────────────
        agent_speed_raw = np.linalg.norm(a.velocity)
        moving_away = False
        if target and target.alive and dist > 1:
            to_target = (target.pos - a.pos) / dist
            vel_dot = float(np.dot(a.velocity, to_target))
            moving_away = vel_dot < -50  # Moving away from target at >50 UU/s

        # ── Flanking calculation ─────────────────────────────────
        # How directly is the target facing the agent?
        target_facing_agent = 1.0
        agent_behind_target = False
        agent_flanking = False

        if target and target.alive and dist > 1:
            to_agent = (a.pos - target.pos) / dist
            target_facing_agent = max(0.0, float(np.dot(target.facing, to_agent)))

            # Behind: target can't see us (outside ~140° front arc).
            agent_behind_target = target_facing_agent < 0.17

            # Flanking: at the target's side (partially outside focused view).
            agent_flanking = (0.17 <= target_facing_agent < 0.7)

        # Can fire: weapon ready AND not blocked by wind-up/switch/dodge.
        can_fire_now = False
        if slot and slot.is_ready() and slot.has_ammo():
            can_fire_now = (not a.is_switching
                           and not a.is_dodging
                           and not a.is_winding_up)

        # Total damage dealt across ALL targets this step (for multi-target reward).
        total_damage_all_targets = 0.0
        for t in self.targets:
            prev_hp = self._prev_target_hps.get(t.target_id, t.hp_fraction())
            delta = max(0.0, prev_hp - t.hp_fraction())
            total_damage_all_targets += delta

        return CombatState(
            self_hp=a.hp_fraction(),
            self_alive=a.alive,
            self_position=a.pos.copy(),
            self_speed=np.linalg.norm(a.velocity) / a.max_speed if a.max_speed > 0 else 0,
            active_weapon_index=a.active_weapon,
            active_ammo_fraction=slot.ammo_fraction() if slot else 0,
            active_weapon_range=slot.weapon_range if slot else 0,
            active_weapon_is_ranged=slot.is_ranged if slot else False,
            active_weapon_can_arc=slot.can_arc if slot else False,
            can_arc_over_target_cover=(
                slot.can_arc_over_height(cover_h) if slot and blocked else False),
            is_reloading=slot.is_reloading if slot else False,
            can_fire=can_fire_now,
            other_ammo_fractions=[w.ammo_fraction() for i, w in enumerate(a.weapons)
                                  if i != a.active_weapon],
            other_weapon_ranges=[w.weapon_range for i, w in enumerate(a.weapons)
                                  if i != a.active_weapon],
            all_ranged_empty=a.all_ranged_empty(),
            has_direct_weapon_with_ammo=any(
                w.has_ammo() and not w.can_arc for w in a.weapons),
            active_optimal_min=slot.optimal_min if slot else 0,
            active_optimal_max=slot.optimal_max if slot else 0,
            weapon_switched=(a.active_weapon != self._prev_weapon_index),
            prev_weapon_index=self._prev_weapon_index,
            target_hp=target.hp_fraction() if target else 0,
            target_alive=target.alive if target else False,
            target_distance=dist,
            has_los=has_los,
            target_in_range=(dist <= slot.weapon_range) if slot and target else False,
            target_behind_cover=blocked,
            target_cover_height=cover_h if blocked else 0.0,
            target_behind_low_cover=(blocked and cover_h < 300),
            target_facing_agent=target_facing_agent,
            agent_behind_target=agent_behind_target,
            agent_flanking=agent_flanking,
            behind_cover=behind_cover,
            in_optimal_range=in_optimal,
            near_wall=near_wall,
            in_corner=in_corner,
            wall_proximity=wall_proximity,
            agent_speed=agent_speed_raw,
            moving_away_from_threat=moving_away,
            active_weapon_wind_up_time=slot.wind_up_time if slot else 0.0,
            active_weapon_fire_cooldown=slot.fire_cooldown if slot else 0.2,
            alive_allies=sum(1 for ally in getattr(self, 'allies', []) if ally.alive),
            alive_hostiles=sum(1 for t in self.targets if t.alive),
            nearest_ally_distance=min(
                (np.linalg.norm(ally.pos - a.pos)
                 for ally in getattr(self, 'allies', []) if ally.alive),
                default=9999.0),
            ally_in_danger=any(
                ally.hp_fraction() < 0.3
                for ally in getattr(self, 'allies', []) if ally.alive),
            ally_just_died=(
                sum(1 for ally in getattr(self, 'allies', []) if ally.alive)
                < self._prev_alive_allies),
            self_between_threat_and_ally=self._check_between_threat_and_ally(),
            lowest_ally_hp=min(
                (ally.hp_fraction()
                 for ally in getattr(self, 'allies', []) if ally.alive),
                default=1.0),
            step_count=self.step_count,
            episode_damage_dealt=self.reward_fn._episode_damage_dealt,
            # Add to the CombatState constructor call:
            total_damage_all_targets=total_damage_all_targets,
            targets_attacked=a.targets_hit,
        )

    # ═════════════════════════════════════════════════════════════
    #  Helpers
    # ═════════════════════════════════════════════════════════════

    def _current_target(self) -> Optional[Target]:
        if 0 <= self.current_target_idx < len(self.targets):
            return self.targets[self.current_target_idx]
        return None

    def _check_between_threat_and_ally(self) -> bool:
        """Check if the agent is positioned between any threat and any low-HP ally.
        Matches C++ body-blocking detection for tank archetype."""
        allies = getattr(self, 'allies', [])
        a = self.agent
        if not allies or not a or not a.alive:
            return False

        for ally in allies:
            if not ally.alive or ally.hp_fraction() >= 0.5:
                continue  # Only check for allies that need help
            for t in self.targets:
                if not t.alive:
                    continue
                # Is agent roughly between threat and ally?
                threat_to_ally = ally.pos - t.pos
                threat_to_agent = a.pos - t.pos
                d_ta = np.linalg.norm(threat_to_ally)
                d_tg = np.linalg.norm(threat_to_agent)
                if d_ta < 1 or d_tg < 1:
                    continue
                # Agent is "between" if: closer to threat than ally is,
                # and roughly on the line between them.
                if d_tg < d_ta:
                    dot = float(np.dot(threat_to_agent / d_tg,
                                       threat_to_ally / d_ta))
                    if dot > 0.7:  # Within ~45° cone
                        return True
        return False

    # ═════════════════════════════════════════════════════════════
    #  Renderer
    # ═════════════════════════════════════════════════════════════

    def _snapshot_projectiles(self) -> List[dict]:
        """Capture lightweight rendering data for all live projectiles."""
        snaps = []
        for p in self._projectiles:
            snaps.append({
                'pos': p.pos.copy(),
                'velocity': p.velocity.copy(),
                'alive': p.alive,
                'is_agent_projectile': p.is_agent_projectile,
                'is_arc': p.is_arc,
                'arc_flight_time': p.arc_flight_time,
                'arc_elapsed': p.arc_elapsed,
                'arc_start': p.arc_start.copy() if p.is_arc else None,
                'arc_apex': p.arc_apex.copy() if p.is_arc else None,
                'arc_end': p.arc_end.copy() if p.is_arc else None,
                'did_hit': p.did_hit,
                'arc_impact_radius': p.arc_impact_radius,
                'hit_radius': p.hit_radius,
            })
        return snaps

    def render_subframes(self, n_frames: int = 4):
        """Generate multiple render frames per decision step for smooth projectile motion.

        Uses the snapshots captured during the projectile substep loop.
        Each call to render_subframes replays the substep snapshots and
        generates n_frames evenly-spaced render frames, so projectiles
        visibly travel across the arena instead of appearing as static dots.

        Returns:
            list of numpy arrays (rgb) if render_mode='rgb_array',
            None if render_mode='human' (draws to screen directly).
        """
        if self.render_mode is None or not self._projectile_snapshots:
            return [self.render()]

        snapshots = self._projectile_snapshots
        num_snaps = len(snapshots)
        frames = []

        # Map n_frames evenly across the available snapshots.
        for fi in range(n_frames):
            # Which snapshot index (float) corresponds to this frame?
            t = fi / max(n_frames - 1, 1) * max(num_snaps - 1, 0)
            snap_idx = int(t)
            frac = t - snap_idx

            # Clamp to valid range.
            snap_idx = min(snap_idx, num_snaps - 1)
            snap = snapshots[snap_idx]

            # If we can interpolate to the next snapshot, do so for smoothness.
            if frac > 0.01 and snap_idx + 1 < num_snaps:
                snap_next = snapshots[snap_idx + 1]
                interp_snap = self._interpolate_snapshot(snap, snap_next, frac)
            else:
                interp_snap = snap

            # Temporarily replace projectiles with snapshot data for rendering.
            saved_projectiles = self._projectiles
            self._projectiles = self._snap_to_render_projectiles(interp_snap)

            frame = self.render()
            frames.append(frame)

            # Restore real projectiles.
            self._projectiles = saved_projectiles

        return frames

    def _interpolate_snapshot(self, snap_a: List[dict], snap_b: List[dict],
                               frac: float) -> List[dict]:
        """Linearly interpolate between two projectile snapshots."""
        result = []
        # Match projectiles by index (they share the same order within a step).
        for i in range(max(len(snap_a), len(snap_b))):
            if i < len(snap_a) and i < len(snap_b):
                a, b = snap_a[i], snap_b[i]
                interp = dict(a)  # shallow copy
                interp['pos'] = a['pos'] * (1 - frac) + b['pos'] * frac
                # Show projectile as alive if it was alive in either snapshot.
                interp['alive'] = a['alive'] or b['alive']
                interp['did_hit'] = b['did_hit']
                result.append(interp)
            elif i < len(snap_a):
                # Projectile was removed in snap_b (hit/expired) — still show it.
                entry = dict(snap_a[i])
                entry['alive'] = True  # Keep visible during interpolation.
                result.append(entry)
            else:
                # New projectile spawned mid-step — show from snap_b.
                result.append(dict(snap_b[i]))
        return result

    def _snap_to_render_projectiles(self, snap: List[dict]) -> List[SimProjectile]:
        """Convert a snapshot back to SimProjectile objects for the render method."""
        projs = []
        for s in snap:
            if not s.get('alive', False):
                continue
            p = SimProjectile(
                pos=s['pos'].copy(),
                velocity=s['velocity'].copy(),
                alive=True,
                is_agent_projectile=s['is_agent_projectile'],
                is_arc=s['is_arc'],
                arc_flight_time=s.get('arc_flight_time', 1.0),
                arc_elapsed=s.get('arc_elapsed', 0.0),
                arc_start=s['arc_start'].copy() if s.get('arc_start') is not None else np.zeros(2, dtype=np.float32),
                arc_apex=s['arc_apex'].copy() if s.get('arc_apex') is not None else np.zeros(2, dtype=np.float32),
                arc_end=s['arc_end'].copy() if s.get('arc_end') is not None else np.zeros(2, dtype=np.float32),
                did_hit=s.get('did_hit', False),
                arc_impact_radius=s.get('arc_impact_radius', 0.0),
                hit_radius=s.get('hit_radius', 25.0),
            )
            projs.append(p)
        return projs

    def render(self):
        """Render the arena. Requires pygame (`pip install pygame`)."""
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            raise ImportError("Rendering requires pygame: pip install pygame")

        sz = self._render_size
        half = self._arena_half

        # World→pixel conversion.
        def w2p(wx, wy):
            px = int((wx + half) / (2 * half) * sz)
            py = int((half - wy) / (2 * half) * sz)  # flip Y
            return (px, py)

        def w2r(whw, whh):
            return (max(2, int(whw / (2 * half) * sz)),
                    max(2, int(whh / (2 * half) * sz)))

        # Lazy init.
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((sz, sz))
                pygame.display.set_caption(
                    f"Combat Sim — Stage {self.cfg.curriculum_stage}")
            else:
                # rgb_array mode — offscreen surface, works headless.
                self._screen = pygame.Surface((sz, sz))
            self._clock = pygame.time.Clock()
            try:
                self._font = pygame.font.SysFont("monospace", 12)
            except Exception:
                self._font = pygame.font.Font(None, 14)

        # Handle events (so the window doesn't freeze).
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        # Clear.
        self._screen.fill((20, 20, 30))

        # Pre-compute shared state for rendering.
        a = self.agent
        slot = a.active_slot() if a and a.alive else None
        target = self._current_target()

        # Arena border.
        border = [w2p(-half, -half), w2p(-half, half),
                  w2p(half, half), w2p(half, -half)]
        pygame.draw.polygon(self._screen, (60, 60, 70), border, 1)

        # Obstacles.
        for obs in self.obstacles:
            ox, oy = w2p(obs.x - obs.hw, obs.y + obs.hh)
            ow, oh = w2r(obs.hw * 2, obs.hh * 2)
            color = (80, 70, 50) if obs.height >= 300 else (60, 80, 50)
            pygame.draw.rect(self._screen, color, (ox, oy, ow, oh))
            if obs.height < 200:
                # Low cover indicator.
                pygame.draw.rect(self._screen, (90, 110, 70), (ox, oy, ow, oh), 1)

        # Agent (enemy robot).
        if a and a.alive:
            ax, ay = w2p(a.pos[0], a.pos[1])
            # Body.
            pygame.draw.circle(self._screen, (70, 130, 220), (ax, ay), 8)
            # Direction indicator.
            if np.linalg.norm(a.velocity) > 10:
                vdir = a.velocity / np.linalg.norm(a.velocity)
                ex, ey = int(ax + vdir[0] * 14), int(ay - vdir[1] * 14)
                pygame.draw.line(self._screen, (120, 180, 255), (ax, ay), (ex, ey), 2)

            # Weapon range circle (faint).
            if slot:
                range_px = int(slot.weapon_range / (2 * half) * sz)
                pygame.draw.circle(self._screen, (40, 60, 100),
                                   (ax, ay), range_px, 1)
                # Optimal band.
                opt_min_px = int(slot.optimal_min / (2 * half) * sz)
                opt_max_px = int(slot.optimal_max / (2 * half) * sz)
                pygame.draw.circle(self._screen, (50, 80, 50),
                                   (ax, ay), opt_min_px, 1)
                pygame.draw.circle(self._screen, (50, 80, 50),
                                   (ax, ay), opt_max_px, 1)

            # HP bar.
            bar_w = 20
            hp_w = int(bar_w * a.hp_fraction())
            pygame.draw.rect(self._screen, (40, 40, 40),
                             (ax - bar_w // 2, ay - 14, bar_w, 3))
            pygame.draw.rect(self._screen, (70, 180, 70),
                             (ax - bar_w // 2, ay - 14, hp_w, 3))

            # Ammo bar.
            if slot:
                ammo_w = int(bar_w * slot.ammo_fraction())
                pygame.draw.rect(self._screen, (40, 40, 40),
                                 (ax - bar_w // 2, ay - 18, bar_w, 2))
                color = (200, 200, 60) if not slot.is_reloading else (200, 100, 60)
                pygame.draw.rect(self._screen, color,
                                 (ax - bar_w // 2, ay - 18, ammo_w, 2))

        # Targets (player party members).
        for i, t in enumerate(self.targets):
            if not t.alive:
                continue
            tx, ty = w2p(t.pos[0], t.pos[1])
            color = (220, 60, 60) if t.is_player_controlled else (200, 120, 60)
            # Highlight current target.
            if i == self.current_target_idx:
                pygame.draw.circle(self._screen, (255, 255, 100),
                                   (tx, ty), 12, 1)
            pygame.draw.circle(self._screen, color, (tx, ty), 6)

            # HP bar.
            bar_w = 16
            hp_w = int(bar_w * t.hp_fraction())
            pygame.draw.rect(self._screen, (40, 40, 40),
                             (tx - bar_w // 2, ty - 12, bar_w, 3))
            pygame.draw.rect(self._screen, (220, 60, 60),
                             (tx - bar_w // 2, ty - 12, hp_w, 3))

            # Velocity trail.
            if np.linalg.norm(t.velocity) > 10:
                vdir = t.velocity / np.linalg.norm(t.velocity)
                ex = int(tx + vdir[0] * 10)
                ey = int(ty - vdir[1] * 10)
                pygame.draw.line(self._screen, (180, 80, 80),
                                 (tx, ty), (ex, ey), 1)

        # Allied robots (from combat_extensions).
        for ally in getattr(self, 'allies', []):
            if not ally.alive:
                continue
            alx, aly = w2p(ally.pos[0], ally.pos[1])

            # Archetype color: 0=ranged(cyan), 1=melee(green), 2=healer(pink), 3=tank(yellow)
            ally_colors = {
                0: (60, 200, 200), 1: (60, 200, 80),
                2: (200, 100, 200), 3: (200, 200, 60),
            }
            color = ally_colors.get(ally.archetype, (100, 160, 200))
            pygame.draw.circle(self._screen, color, (alx, aly), 7)
            # Diamond shape to distinguish from agent's circle.
            pygame.draw.polygon(self._screen, color, [
                (alx, aly - 9), (alx + 6, aly),
                (alx, aly + 9), (alx - 6, aly)], 1)

            # HP bar.
            bar_w = 16
            hp_w = int(bar_w * ally.hp_fraction())
            pygame.draw.rect(self._screen, (40, 40, 40),
                             (alx - bar_w // 2, aly - 14, bar_w, 3))
            pygame.draw.rect(self._screen, (60, 180, 180),
                             (alx - bar_w // 2, aly - 14, hp_w, 3))

        # LOS line from agent to current target.
        if a and a.alive and target and target.alive:
            ax, ay = w2p(a.pos[0], a.pos[1])
            tx, ty = w2p(target.pos[0], target.pos[1])
            has_los = check_los(a.pos, target.pos, self.obstacles)
            los_color = (60, 120, 60) if has_los else (120, 40, 40)
            pygame.draw.line(self._screen, los_color, (ax, ay), (tx, ty), 1)

        # Projectiles.
        for proj in self._projectiles:
            if not proj.alive:
                continue
            px, py = w2p(proj.pos[0], proj.pos[1])

            if proj.is_agent_projectile:
                if proj.is_arc:
                    # Agent arc: cyan with larger dot.
                    color = (80, 220, 255)
                    pygame.draw.circle(self._screen, color, (px, py), 4)
                    # Draw remaining arc path.
                    if proj.arc_flight_time > 0:
                        t_now = proj.arc_elapsed / proj.arc_flight_time
                        prev_pt = (px, py)
                        for step in range(5):
                            t_f = t_now + (1.0 - t_now) * (step + 1) / 5
                            omt = 1.0 - t_f
                            fp = (omt*omt*proj.arc_start
                                  + 2*omt*t_f*proj.arc_apex
                                  + t_f*t_f*proj.arc_end)
                            fpx, fpy = w2p(fp[0], fp[1])
                            pygame.draw.line(self._screen, (40, 100, 130),
                                             prev_pt, (fpx, fpy), 1)
                            prev_pt = (fpx, fpy)
                else:
                    # Agent straight: bright cyan dot with trail.
                    color = (100, 220, 255)
                    pygame.draw.circle(self._screen, color, (px, py), 3)
            else:
                if proj.is_arc:
                    color = (255, 160, 50)
                    pygame.draw.circle(self._screen, color, (px, py), 4)
                else:
                    # Target straight: orange-red dot.
                    color = (255, 140, 50)
                    pygame.draw.circle(self._screen, color, (px, py), 3)

            # Velocity trail.
            speed = np.linalg.norm(proj.velocity)
            if speed > 10:
                trail_dir = proj.velocity / speed
                tx = int(px - trail_dir[0] * 8)
                ty = int(py + trail_dir[1] * 8)  # flip Y for pygame
                trail_c = (70, 150, 180) if proj.is_agent_projectile else (180, 90, 30)
                pygame.draw.line(self._screen, trail_c, (px, py), (tx, ty), 1)

        # AoE impact rings (brief flash on arc projectile impact).
        for proj in self._projectiles:
            if proj.did_hit and proj.is_arc and proj.arc_impact_radius > 0:
                ipx, ipy = w2p(proj.pos[0], proj.pos[1])
                radius_px = int(proj.arc_impact_radius / (2 * half) * sz)
                c = (120, 220, 255) if proj.is_agent_projectile else (255, 120, 50)
                pygame.draw.circle(self._screen, c, (ipx, ipy), radius_px, 1)

        # HUD text.
        hud_lines = [
            f"Step: {self.step_count}  Time: {a.combat_time:.1f}s" if a else "",
            f"Agent HP: {a.hp_fraction()*100:.0f}%" if a else "",
            f"Weapon: {slot.name if slot else 'none'}  "
            f"Ammo: {slot.current_ammo}/{slot.max_ammo if slot else 0}"
            if a and slot else "",
            f"Target dist: {np.linalg.norm(a.pos - target.pos):.0f} UU"
            if a and target and target.alive else "",
            f"Arena: {self._effective_arena_size:.0f}  "
            f"Stage: {self.cfg.curriculum_stage}",
            f"Allies: {sum(1 for a in getattr(self, 'allies', []) if a.alive)}"
            f"/{len(getattr(self, 'allies', []))}"
            if getattr(self, 'allies', []) else "",
            f"Projectiles: {len(self._projectiles)}"
            if self._projectiles else "",
        ]
        for i, line in enumerate(hud_lines):
            if line:
                surf = self._font.render(line, True, (180, 180, 180))
                self._screen.blit(surf, (5, 5 + i * 14))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(30)  # Cap at 30 FPS for visibility.
            return None
        else:
            # rgb_array mode.
            return np.transpose(
                pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))

    def close(self):
        """Clean up renderer."""
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self._screen = None


# ─────────────────────────────────────────────────────────────────
#  Curriculum Environment Factory
# ─────────────────────────────────────────────────────────────────

def make_curriculum_env(stage: int, archetype: str = "ranged",
                       render_mode: str = None) -> CombatEnv:
    """Create an environment configured for a specific curriculum stage."""

    configs = {
        # Stage 1: Melee basics. Easy target, no obstacles.
        1: CombatEnvConfig(
            num_enemies=1, num_targets=1, num_obstacles=0,
            arena_size=1500.0,
            curriculum_stage=1, archetype="melee",
            weapon_preset="melee_bot", target_speed_fraction=0.0,
            engagement_distance=800, max_steps=500,
            enemy_hp=100, enemy_defence=20,
            target_hp=100, target_defence=10),

        # Stage 2: Ranged basics. Stationary target, learn fire/reload.
        2: CombatEnvConfig(
            num_enemies=1, num_targets=1, num_obstacles=0,
            arena_size=2000.0,
            curriculum_stage=2, archetype=archetype,
            weapon_preset="scout", target_speed_fraction=0.0,
            engagement_distance=1200, max_steps=500,
            enemy_hp=100, enemy_defence=20,
            target_hp=150, target_defence=15),

        # Stage 3: Moving targets. Learn tracking, kiting, cover.
        # Reduced target HP so scout laser can kill before agent dies.
        3: CombatEnvConfig(
            num_enemies=1, num_targets=2, num_obstacles=3,
            arena_size=2500.0,
            curriculum_stage=3, archetype=archetype,
            weapon_preset="scout", target_speed_fraction=0.6,
            engagement_distance=1500, max_steps=1000,
            enemy_hp=120, enemy_defence=20,
            target_hp=50, target_defence=20),

        # Stage 3.5: Heavy weapons, easy targets. Isolates weapon
        # management learning from survivability. Same targets as S3
        # (50 HP, defence 20, speed 0.6) but heavy weapon kit (cannon +
        # missiles + melee). The agent learns switching, ammo management,
        # arc-vs-direct fire, and reload timing against targets it can
        # already kill — before S4 doubles the target HP.
        4: CombatEnvConfig(
            num_enemies=1, num_targets=2, num_obstacles=8,
            arena_size=3000.0,
            curriculum_stage=4, archetype=archetype,
            weapon_preset="heavy", target_speed_fraction=0.8,
            engagement_distance=1500, max_steps=500,
            enemy_hp=100, enemy_defence=20,
            target_hp=75, target_defence=20),

        # # Stage 4: Multi-weapon. Learn ammo management, switching.
        # # [Fix 1] max_steps reduced from 1000 to 400. At 1000 steps, the
        # # 1.4% of episodes that reached timeout accumulated ~1506 avg reward
        # # from per-step shaping — dwarfing kill (20) and win (50-75) bonuses.
        # # This taught the agent that surviving > killing. At 400 steps, max
        # # per-step accumulation is ~380, well below objective rewards.
        # 4: CombatEnvConfig(
        #     num_enemies=1, num_targets=2, num_obstacles=4,
        #     arena_size=2500.0,
        #     curriculum_stage=4, archetype=archetype,
        #     weapon_preset="heavy", target_speed_fraction=0.7,
        #     engagement_distance=1500, max_steps=400,
        #     enemy_hp=130, enemy_defence=25,
        #     target_hp=100, target_defence=25),

        # Stage 5: Archetype-specific. Full weapon kit, varied targets. Introducing alliess
        5: CombatEnvConfig(
            num_enemies=2, num_targets=3, num_obstacles=8,
            arena_size=3000.0,
            curriculum_stage=5, archetype=archetype,
            weapon_preset="heavy",
            weapon_pool=["heavy", "scout", "sniper", "tank"],
            target_speed_fraction=0.8,
            engagement_distance=1500, max_steps=600,   # [Fix] Was 1000. Defense-in-depth
            enemy_hp=200, enemy_defence=25,             # against per-step shaping accumulation.
            target_hp=100, target_defence=25),          # 3 targets × ~100 steps each = ~300 steps
                                                        # to kill. 600 = 2× expected completion.

        # Stage 6: Multi-target coordination. More agent HP to survive.
        6: CombatEnvConfig(
            num_enemies=2, num_targets=3, num_obstacles=12,
            arena_size=3000.0,
            curriculum_stage=6, archetype=archetype,
            weapon_preset="heavy",
            weapon_pool=["heavy", "scout", "sniper", "tank"],
            target_speed_fraction=0.9,
            engagement_distance=1500, max_steps=700,   # [Fix] Was 1000. Group coord needs more
            enemy_hp=180, enemy_defence=30,             # time than 1v3 but ally helps with DPS.
            target_hp=150, target_defence=25),          # 3 targets × 150 HP + navigation ≈ 350 steps.

        # Stage 7: Full squad. Agent is tanky to survive 4 targets.
        7: CombatEnvConfig(
            num_enemies=2, num_targets=4, num_obstacles=16,
            arena_size=4000.0,
            curriculum_stage=7, archetype=archetype,
            weapon_preset="heavy",
            weapon_pool=["heavy", "scout", "sniper", "tank"],
            target_speed_fraction=1.0,
            engagement_distance=2000, max_steps=800,   # [Fix] Was 1200. Biggest arena (4000) but
            enemy_hp=500, enemy_defence=35,             # 4 targets × ~120 steps each ≈ 480 steps.
            target_hp=150, target_defence=25),          # 800 = ~1.7× expected completion.
    }

    cfg = configs.get(stage, configs[3])
    cfg.curriculum_stage = stage
    cfg.archetype = archetype
    return CombatEnv(cfg, render_mode=render_mode)


# ─────────────────────────────────────────────────────────────────
#  Quick Test / Visual Debugger
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combat sim test / visualiser")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage 1-7")
    parser.add_argument("--archetype", type=str, default="ranged")
    parser.add_argument("--render", type=str, default=None,
                        choices=["human", "video"],
                        help="'human' = pygame window, 'video' = save replay.mp4")
    parser.add_argument("--arena_size", type=float, default=None,
                        help="Override arena size (UU)")
    parser.add_argument("--steps", type=int, default=500, help="Max steps to run")
    parser.add_argument("--weapon", type=str, default=None,
                        help="Override weapon preset (scout/heavy/sniper/melee_bot/tank)")
    args = parser.parse_args()

    # Set up render mode.
    if args.render == "human":
        render_mode = "human"
    elif args.render == "video":
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = make_curriculum_env(args.stage, args.archetype, render_mode=render_mode)

    # Apply overrides.
    if args.arena_size is not None:
        env.cfg.arena_size = args.arena_size
    if args.weapon is not None:
        env.cfg.weapon_preset = args.weapon

    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"Arena: {env.cfg.arena_size:.0f} UU, Stage: {args.stage}, "
          f"Targets: {env.cfg.num_targets}, Obstacles: {env.cfg.num_obstacles}")

    total_reward = 0
    video_frames = []
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if render_mode == "human":
            env.render()
        elif render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                video_frames.append(frame)

        if done or truncated:
            print(f"Episode ended at step {step}: reward={total_reward:.2f}, "
                  f"agent_hp={env.agent.hp_fraction()*100:.0f}%, "
                  f"target_hp={env.targets[0].hp_fraction()*100:.0f}%")
            break

    if not done and not truncated:
        print(f"Episode running after {args.steps} steps: reward={total_reward:.2f}")

    env.close()

    # Save video if frames were collected.
    if video_frames:
        output = "sim_test.mp4"
        print(f"Saving {len(video_frames)} frames to {output}...")
        try:
            import imageio.v3 as iio
            iio.imwrite(output, video_frames, fps=15)
            print(f"Saved: {output}")
        except ImportError:
            print("Install imageio for video export: pip install imageio[ffmpeg]")
            print("Saving PNGs instead...")
            os.makedirs("sim_frames", exist_ok=True)
            from PIL import Image
            for i, frame in enumerate(video_frames):
                Image.fromarray(frame).save(f"sim_frames/frame_{i:05d}.png")
            print(f"Saved {len(video_frames)} PNGs to sim_frames/")