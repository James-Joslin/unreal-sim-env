"""
combat_extensions.py — Fills the sim-to-real observation gap.

Implements the game systems that exist in UE but were missing from the
Python combat sim. Adds these as a CombatEnvExtended class that inherits
from CombatEnv and overrides the observation builder + step logic.

WHAT THIS ADDS (matching NeuralCombatComponent.cpp)
    1. Status effects — stun (locks actions), slow (reduces speed),
       debuff slots with duration tracking.
    2. Target acceleration — velocity delta per tick.
    3. Threat scoring — accumulated damage per target, decaying over time.
       Matches EnemyPerceptionComponent::EvaluateTargetPriority().
    4. Priority scoring — composite of distance, HP, threat, LOS.
    5. Allied robots — simple scripted allies that fight alongside the agent.
    6. Projectile tracking — incoming projectiles with position, velocity,
       time-to-arrival, and direction (for dodge training).
    7. Sight cone — actual vision cone check instead of always-true.
    8. Group summary — derived from ally data.

USAGE
    # Drop-in replacement for make_curriculum_env:
    from combat_extensions import make_extended_curriculum_env
    env = make_extended_curriculum_env(stage=3, archetype="ranged")

    # Or wrap an existing env:
    from combat_extensions import CombatEnvExtended
    env = CombatEnvExtended(config)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from combat_sim import (
    CombatEnv, CombatEnvConfig, Agent, Target, Obstacle, WeaponSlot,
    OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS,
    compute_damage, check_los, is_behind_cover, _ray_aabb_intersect,
    make_curriculum_env, WEAPON_PRESETS,
)


# ─────────────────────────────────────────────────────────────────
#  Status Effect System
# ─────────────────────────────────────────────────────────────────

@dataclass
class StatusEffect:
    """A single active status effect on the agent."""
    name: str = "none"
    duration_remaining: float = 0.0
    strength: float = 1.0  # 0-1, affects severity

    def tick(self, dt: float) -> bool:
        """Tick and return True if still active."""
        self.duration_remaining -= dt
        return self.duration_remaining > 0

    @property
    def active(self) -> bool:
        return self.duration_remaining > 0


@dataclass
class StatusEffectState:
    """All status effects on an agent. Matches UStatusEffectComponent."""
    stunned: StatusEffect = field(default_factory=StatusEffect)
    slowed: StatusEffect = field(default_factory=StatusEffect)
    debuff_slots: List[StatusEffect] = field(
        default_factory=lambda: [StatusEffect() for _ in range(6)])

    def tick(self, dt: float):
        self.stunned.tick(dt)
        self.slowed.tick(dt)
        for slot in self.debuff_slots:
            slot.tick(dt)

    def apply_stun(self, duration: float):
        self.stunned = StatusEffect("stun", duration, 1.0)

    def apply_slow(self, duration: float, strength: float = 0.5):
        self.slowed = StatusEffect("slow", duration, strength)

    def apply_debuff(self, name: str, duration: float, strength: float = 0.5):
        # Find empty slot or replace weakest.
        for slot in self.debuff_slots:
            if not slot.active:
                slot.name = name
                slot.duration_remaining = duration
                slot.strength = strength
                return
        # All full — replace shortest remaining.
        weakest = min(self.debuff_slots, key=lambda s: s.duration_remaining)
        weakest.name = name
        weakest.duration_remaining = duration
        weakest.strength = strength

    @property
    def is_stunned(self) -> bool:
        return self.stunned.active

    @property
    def speed_multiplier(self) -> float:
        if self.slowed.active:
            return 1.0 - self.slowed.strength  # 0.5 strength = 50% speed
        return 1.0


# ─────────────────────────────────────────────────────────────────
#  Projectile Tracking
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrackedProjectile:
    """An incoming projectile the agent should be aware of."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    damage: float = 10.0
    alive: bool = True
    age: float = 0.0

    def tick(self, dt: float, agent_pos: np.ndarray, hit_radius: float = 80.0):
        """Move projectile. Returns damage if it hits the agent."""
        if not self.alive:
            return 0.0
        self.age += dt
        if self.age > 5.0:  # match SimProjectile.max_lifetime
            self.alive = False
            return 0.0
        self.pos += self.velocity * dt
        dist = np.linalg.norm(self.pos - agent_pos)
        if dist < hit_radius:
            self.alive = False
            return self.damage
        return 0.0

    def time_to_arrival(self, agent_pos: np.ndarray) -> float:
        """Estimated seconds until this projectile reaches the agent."""
        dist = np.linalg.norm(self.pos - agent_pos)
        speed = np.linalg.norm(self.velocity)
        if speed < 1.0:
            return 999.0
        return dist / speed

    def direction_to_agent(self, agent_pos: np.ndarray) -> np.ndarray:
        """Unit vector from projectile toward agent."""
        d = agent_pos - self.pos
        n = np.linalg.norm(d)
        return d / max(n, 1.0)


# ─────────────────────────────────────────────────────────────────
#  Allied Robot (simple scripted ally)
# ─────────────────────────────────────────────────────────────────

@dataclass
class AlliedRobot:
    """A scripted ally that fights alongside the agent.
    Simple AI: picks nearest target, moves toward it, attacks periodically.
    The agent doesn't control allies — it just observes them."""
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    hp: float = 100.0
    max_hp: float = 100.0
    defence: float = 15.0
    alive: bool = True
    archetype: int = 0  # 0=ranged, 1=melee, 2=healer, 3=tank
    max_speed: float = 400.0
    attack_range: float = 1000.0
    attack_damage: float = 12.0
    attack_cooldown: float = 1.0
    attack_cooldown_remaining: float = 0.0
    target_idx: int = 0  # Which hostile this ally is targeting.

    def hp_fraction(self) -> float:
        return max(0, self.hp / self.max_hp)

    def tick(self, dt: float, targets: List[Target], obstacles: List[Obstacle],
             arena_half: float):
        """Simple ally AI: approach nearest alive target and attack."""
        if not self.alive:
            return

        self.attack_cooldown_remaining -= dt

        # Pick nearest alive target.
        best_dist = 99999
        best_idx = 0
        for i, t in enumerate(targets):
            if t.alive:
                d = np.linalg.norm(self.pos - t.pos)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        self.target_idx = best_idx
        target = targets[best_idx] if best_idx < len(targets) and targets[best_idx].alive else None

        if not target:
            self.velocity = np.zeros(2, dtype=np.float32)
            return

        to_target = target.pos - self.pos
        dist = np.linalg.norm(to_target)
        direction = to_target / max(dist, 1.0)

        # Movement based on archetype.
        if self.archetype == 1:  # Melee — charge in.
            if dist > 200:
                self.velocity = direction * self.max_speed
            else:
                self.velocity = direction * self.max_speed * 0.3
        elif self.archetype == 3:  # Tank — move toward but slower.
            if dist > self.attack_range * 0.7:
                self.velocity = direction * self.max_speed * 0.6
            else:
                perp = np.array([direction[1], -direction[0]], dtype=np.float32)
                self.velocity = perp * self.max_speed * 0.3
        else:  # Ranged/Healer — maintain distance.
            if dist > self.attack_range * 0.8:
                self.velocity = direction * self.max_speed * 0.7
            elif dist < self.attack_range * 0.3:
                self.velocity = -direction * self.max_speed * 0.5
            else:
                perp = np.array([direction[1], -direction[0]], dtype=np.float32)
                self.velocity = perp * self.max_speed * 0.4

        self.pos += self.velocity * dt
        self.pos = np.clip(self.pos, -arena_half, arena_half)

        # Attack if in range and off cooldown.
        if dist <= self.attack_range and self.attack_cooldown_remaining <= 0:
            has_los = check_los(self.pos, target.pos, obstacles)
            if has_los:
                self.attack_cooldown_remaining = self.attack_cooldown
                dmg, target.barrier, _ = compute_damage(
                    self.attack_damage, 5.0, target.defence, target.barrier)
                target.hp -= dmg
                if target.hp <= 0:
                    target.hp = 0
                    target.alive = False


# ─────────────────────────────────────────────────────────────────
#  Threat Table (matches EnemyPerceptionComponent)
# ─────────────────────────────────────────────────────────────────

class ThreatTable:
    """Tracks accumulated damage from each target for priority scoring."""

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


# ─────────────────────────────────────────────────────────────────
#  Extended Combat Environment
# ─────────────────────────────────────────────────────────────────

class CombatEnvExtended(CombatEnv):
    """CombatEnv with all observation features properly implemented.

    Adds:
    - Status effects (stun/slow/debuffs) applied randomly by targets
    - Target acceleration tracking
    - Threat/priority scoring per hostile
    - Allied robots (stages 6+)
    - Projectile tracking with TTA
    - Sight cone checks
    - Group summary from ally data
    """

    def __init__(self, config: CombatEnvConfig = None, render_mode: str = None):
        super().__init__(config, render_mode)

        # New systems.
        self.status_effects = StatusEffectState()
        self.threat_table = ThreatTable()
        self.projectiles: List[TrackedProjectile] = []
        self.allies: List[AlliedRobot] = []
        self.prev_target_velocities: dict = {}  # target_id -> prev velocity

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Reset extended systems.
        self.status_effects = StatusEffectState()
        self.threat_table.reset()
        self.projectiles = []
        self.prev_target_velocities = {}

        # Spawn allies for stages 6+.
        self._spawn_allies()

        # Rebuild observation with extended features.
        obs = self._build_observation()
        return obs, info

    def step(self, action):
        dt = self.cfg.decision_interval

        # Check stun — if stunned, override action to do nothing.
        if self.status_effects.is_stunned:
            action = np.array([0, 0, 4])  # Stay, no combat, keep target.

        # Apply slow to agent speed.
        original_speed = self.agent.max_speed
        self.agent.max_speed *= self.status_effects.speed_multiplier

        # Record target velocities BEFORE step for acceleration calc.
        for t in self.targets:
            if t.alive:
                self.prev_target_velocities[t.target_id] = t.velocity.copy()

        # Run the base step.
        obs, reward, done, truncated, info = super().step(action)

        # Restore agent speed.
        self.agent.max_speed = original_speed

        # Tick status effects.
        self.status_effects.tick(dt)

        # NOTE: threat_table.decay and damage recording are now handled
        # directly in the base env's step (melee hits + projectile hits).
        # [Audit §1.3]

        # Targets may apply status effects.
        self._maybe_apply_status_effects(dt)

        # Tick allies.
        for ally in self.allies:
            ally.tick(dt, self.targets, self.obstacles, self._arena_half)

        # [Fix Bug 3] Update _prev_target_hps AFTER allies act. Previously,
        # the base step set _prev_target_hps before allies ticked, so ally
        # damage between steps was attributed to the agent's damage_dealt
        # reward on the next step. This inflated damage_dealt in the 20% of
        # stage 4 episodes with allies, producing impossible episode_total_damage
        # values (e.g. 225.70 on a 2-target/200% HP-pool arena).
        self._prev_target_hps = {
            t.target_id: t.hp_fraction() for t in self.targets if t.alive
        }

        # Targets may attack allies too.
        self._targets_attack_allies(dt)

        # Tick tracked projectiles (observation-only, separate from base SimProjectiles).
        self._tick_tracked_projectiles(dt)

        # Rebuild observation with extended features filled in.
        obs = self._build_observation()

        return obs, reward, done, truncated, info

    # ═════════════════════════════════════════════════════════════
    #  Ally Spawning
    # ═════════════════════════════════════════════════════════════

    def _spawn_allies(self):
        """Spawn allied robots for group stages."""
        self.allies = []
        stage = self.cfg.curriculum_stage

        # num_enemies includes the agent, so allies = num_enemies - 1.
        num_allies = max(0, self.cfg.num_enemies - 1)

        if stage < 6 and num_allies == 0:
            # [Fix 5] Removed the 20% random ally spawn for stages 1-5.
            # Allies created reward noise via the damage attribution bug
            # (Bug 3), and even after that fix, ally damage in stages
            # without group rewards (stages < 6) confuses the agent's
            # credit assignment — it can't distinguish its own damage
            # from ally damage in the reward signal. Allies are only
            # meaningful once group coordination rewards activate at stage 6.
            return

        for _ in range(num_allies):
            arch = self.rng.choices([0, 1, 2, 3], weights=[40, 30, 15, 15], k=1)[0]
            self.allies.append(self._make_ally(arch))

    def _make_ally(self, archetype: int) -> AlliedRobot:
        """Create an allied robot near the agent."""
        offset = np.array([
            self.rng.uniform(-400, 400),
            self.rng.uniform(-400, 400),
        ], dtype=np.float32)
        pos = self.agent.pos + offset
        pos = np.clip(pos, -self._arena_half, self._arena_half)

        configs = {
            0: dict(max_speed=380, attack_range=1000, attack_damage=10,
                    attack_cooldown=0.8, hp=80, defence=15),  # Ranged
            1: dict(max_speed=480, attack_range=200, attack_damage=25,
                    attack_cooldown=0.7, hp=120, defence=20),  # Melee
            2: dict(max_speed=350, attack_range=800, attack_damage=6,
                    attack_cooldown=1.2, hp=70, defence=10),  # Healer
            3: dict(max_speed=320, attack_range=600, attack_damage=8,
                    attack_cooldown=1.0, hp=150, defence=30),  # Tank
        }
        cfg = configs.get(archetype, configs[0])

        return AlliedRobot(
            pos=pos, archetype=archetype,
            max_hp=cfg["hp"], hp=cfg["hp"],
            defence=cfg["defence"],
            **{k: v for k, v in cfg.items() if k not in ("hp", "defence")},
        )

    # ═════════════════════════════════════════════════════════════
    #  Status Effects
    # ═════════════════════════════════════════════════════════════

    def _maybe_apply_status_effects(self, dt: float):
        """Targets have a small chance of applying status effects per attack.
        This teaches the agent to handle being stunned/slowed."""
        stage = self.cfg.curriculum_stage
        if stage < 3:
            return  # No status effects in early stages.

        for t in self.targets:
            if not t.alive:
                continue

            dist = np.linalg.norm(self.agent.pos - t.pos)

            # Melee targets can stun on close hits.
            if t.combat_role == "melee" and dist < t.melee_range * 1.5:
                if t.melee_cooldown_remaining <= 0 and self.rng.random() < 0.08:
                    self.status_effects.apply_stun(0.4 + self.rng.uniform(0, 0.3))

            # Ranged targets can slow.
            if t.combat_role in ("ranged", "mixed"):
                if t.attack_cooldown_remaining <= 0 and self.rng.random() < 0.05:
                    self.status_effects.apply_slow(
                        1.0 + self.rng.uniform(0, 1.0),
                        strength=self.rng.uniform(0.2, 0.5))

            # Any target can apply a generic debuff (simulates poison, scold, etc.)
            if self.rng.random() < 0.02 * dt:
                debuff_names = ["poison", "scold", "weaken", "blind", "mark", "burn"]
                self.status_effects.apply_debuff(
                    self.rng.choice(debuff_names),
                    duration=self.rng.uniform(2.0, 5.0),
                    strength=self.rng.uniform(0.1, 0.4))

    # ═════════════════════════════════════════════════════════════
    #  Threat Tracking
    # ═════════════════════════════════════════════════════════════

    def _update_threat_from_targets(self, dt: float):
        """Track which targets have dealt damage to the agent."""
        # Simplified: attribute damage proportionally to targets in range with LOS.
        damage_this_tick = max(0, self.agent.max_hp * (
            self.reward_fn._episode_damage_dealt -
            getattr(self, '_prev_episode_damage', 0)))

        in_range_targets = []
        for t in self.targets:
            if t.alive:
                dist = np.linalg.norm(self.agent.pos - t.pos)
                if dist < t.attack_range and check_los(t.pos, self.agent.pos, self.obstacles):
                    in_range_targets.append(t)

        if in_range_targets and damage_this_tick > 0:
            per_target = damage_this_tick / len(in_range_targets)
            for t in in_range_targets:
                self.threat_table.record_damage(t.target_id, per_target)

        self._prev_episode_damage = self.reward_fn._episode_damage_dealt

    # ═════════════════════════════════════════════════════════════
    #  Priority Scoring (matches EnemyPerceptionComponent)
    # ═════════════════════════════════════════════════════════════

    def _compute_priority_score(self, target: Target) -> float:
        """Composite priority score matching EvaluateTargetPriority()."""
        dist = np.linalg.norm(self.agent.pos - target.pos)
        max_range = 3000.0

        # Distance score: closer = higher.
        dist_score = (1.0 - min(dist / max_range, 1.0)) * 30.0

        # Low HP score: lower HP = higher priority.
        hp_score = (1.0 - target.hp_fraction()) * 20.0

        # Threat score: targets that damaged us get priority.
        threat_score = self.threat_table.get_normalised_threat(target.target_id) * 25.0

        # LOS bonus.
        has_los = check_los(self.agent.pos, target.pos, self.obstacles)
        los_score = 15.0 if has_los else 0.0

        # Player-controlled bonus.
        pc_score = 10.0 if target.is_player_controlled else 0.0

        total = dist_score + hp_score + threat_score + los_score + pc_score
        return total / 100.0  # Normalise to [0, 1].

    def _compute_threat_level(self, target: Target) -> float:
        """How threatening this target is (DPS × proximity)."""
        dist = np.linalg.norm(self.agent.pos - target.pos)
        if dist > target.attack_range * 1.5:
            return 0.0

        # Approximate DPS.
        if target.combat_role == "melee":
            dps = target.melee_damage / max(target.melee_cooldown, 0.1)
        else:
            dps = target.attack_damage / max(target.attack_cooldown, 0.1)

        proximity = 1.0 - min(dist / (target.attack_range * 1.5), 1.0)
        return min((dps * proximity) / 50.0, 1.0)  # Normalise to [0, 1].

    # ═════════════════════════════════════════════════════════════
    #  Projectile Tracking
    # ═════════════════════════════════════════════════════════════

    def _tick_tracked_projectiles(self, dt: float):
        """Advance TrackedProjectiles (observation-only) and remove expired ones.

        NOTE: This is intentionally NOT named _tick_projectiles to avoid
        shadowing the base CombatEnv._tick_projectiles, which ticks the
        SimProjectile list (self._projectiles) — the ones that actually
        do damage and are rendered. If this method shadowed the base,
        SimProjectiles would never advance and would appear static.
        """
        self.projectiles = [p for p in self.projectiles if p.alive]
        for p in self.projectiles:
            p.tick(dt, self.agent.pos)

    def _spawn_projectile_from_target(self, target: Target):
        """When a target fires, create a tracked projectile."""
        direction = self.agent.pos - target.pos
        dist = np.linalg.norm(direction)
        if dist < 1:
            return
        direction = direction / dist

        # Add some inaccuracy.
        spread = self.rng.uniform(-0.1, 0.1)
        cos_s, sin_s = math.cos(spread), math.sin(spread)
        dx, dy = direction[0], direction[1]
        direction = np.array([dx * cos_s - dy * sin_s, dx * sin_s + dy * cos_s], dtype=np.float32)

        proj = TrackedProjectile(
            pos=target.pos.copy(),
            velocity=direction * target.attack_projectile_speed,
            damage=target.attack_damage,
        )
        self.projectiles.append(proj)

    # ═════════════════════════════════════════════════════════════
    #  Ally Combat (targets attack allies)
    # ═════════════════════════════════════════════════════════════

    def _targets_attack_allies(self, dt: float):
        """Targets split attention between agent and allies."""
        for t in self.targets:
            if not t.alive:
                continue
            # 30% chance of targeting an ally instead of the agent each tick.
            if self.rng.random() < 0.3 and self.allies:
                alive_allies = [a for a in self.allies if a.alive]
                if alive_allies:
                    target_ally = self.rng.choice(alive_allies)
                    dist = np.linalg.norm(t.pos - target_ally.pos)
                    if dist < t.attack_range and t.attack_cooldown_remaining <= 0:
                        if check_los(t.pos, target_ally.pos, self.obstacles):
                            dmg, target_ally.barrier, _ = compute_damage(
                                t.attack_damage, t.attack_stat,
                                target_ally.defence, 0.0)
                            target_ally.hp -= dmg
                            if target_ally.hp <= 0:
                                target_ally.hp = 0
                                target_ally.alive = False

    # ═════════════════════════════════════════════════════════════
    #  Observation Builder (fills ALL 215 features)
    # ═════════════════════════════════════════════════════════════

    def _build_observation(self) -> np.ndarray:
        """Override: builds the full 215-float observation with all features."""
        obs = super()._build_observation()

        a = self.agent
        target = self._current_target()
        se = self.status_effects

        # ── Self State: fill status effect slots ─────────────────
        obs[3] = 1.0 if se.is_stunned else 0.0             # stunned
        obs[4] = 1.0 if se.slowed.active else 0.0          # slowed
        for i in range(6):                                   # debuff slots
            obs[5 + i] = se.debuff_slots[i].strength if se.debuff_slots[i].active else 0.0
        # [Audit §1.8] C++ TraceHeightAboveGround returns ~88 UU for a
        # grounded ACharacter. 88/500 ≈ 0.176.
        obs[14] = 0.176                                      # height above ground

        # ── Primary Target: sight cone and acceleration ──────────
        if target and target.alive:
            # Sight cone: check if target is in our forward arc.
            rel = target.pos - a.pos
            dist = np.linalg.norm(rel)
            if dist > 1:
                # Use body facing (not velocity) for sight cone check.
                # Matches C++ where GetActorForwardVector() is target-locked.
                target_dir = rel / dist
                facing_dot = float(np.dot(a.facing, target_dir))
                obs[56] = 1.0 if facing_dot > -0.17 else 0.0  # ~100° half-cone

            # Target acceleration (velocity delta / dt).
            # C++ GatherPrimaryTarget writes accel at indices 61-62 (after vel at 59-60).
            # Base env skips these with `idx += 2`. We fill them here.
            prev_vel = self.prev_target_velocities.get(target.target_id, np.zeros(2, dtype=np.float32))
            accel = (target.velocity - prev_vel) / max(self.cfg.decision_interval, 0.01)
            obs[61] = np.clip(accel[0] / 2000, -1, 1)  # normalised
            obs[62] = np.clip(accel[1] / 2000, -1, 1)

        # ── Hostile Targets: priority and threat scores ──────────
        # [Audit §1.3] Sort by priority score (not distance), matching
        # C++ ScoredTargets from EvaluateTargetPriority.
        sorted_targets = self._get_sorted_targets()

        for si in range(4):
            base = 70 + si * 13
            if si < len(sorted_targets):
                t = sorted_targets[si]
                obs[base + 8] = self._compute_priority_score(t)  # priority
                obs[base + 9] = self._compute_threat_level(t)    # threat

        # ── Allied Robots ────────────────────────────────────────
        # Matches C++ GatherAlliedRobots per-slot layout (12 floats each):
        #   +0  occupied
        #   +1  rel_x / 5000
        #   +2  rel_y / 5000
        #   +3  distance / 5000
        #   +4  HP fraction
        #   +5  ammo fraction (GetActiveAmmoFraction)
        #   +6  is in combat (IsInCombat)
        #   +7  is dodging (IsDodging)
        #   +8  archetype scalar ((enum+1)/4.0)
        #   +9  velocity X / ally max_speed
        #   +10 velocity Y / ally max_speed
        #   +11 target hostile index (normalised slot, -1 if none)
        alive_allies_sorted = sorted(
            [ally for ally in self.allies if ally.alive],
            key=lambda ally: np.linalg.norm(ally.pos - a.pos))

        for si in range(3):
            base = 122 + si * 12
            if si < len(alive_allies_sorted):
                ally = alive_allies_sorted[si]
                rel = ally.pos - a.pos
                dist = np.linalg.norm(rel)

                obs[base + 0] = 1.0                                    # occupied
                obs[base + 1] = np.clip(rel[0] / 5000, -1, 1)        # rel_x
                obs[base + 2] = np.clip(rel[1] / 5000, -1, 1)        # rel_y
                obs[base + 3] = min(dist / 5000, 1.0)                  # distance
                obs[base + 4] = ally.hp_fraction()                      # HP

                # [+5] Ammo fraction. C++ reads GetActiveAmmoFraction().
                # Python AlliedRobot has no weapon system — uses cooldown-based
                # attacks. 1.0 = always ready (infinite "ammo"). If a weapon
                # system is added to AlliedRobot later, read it here instead.
                obs[base + 5] = 1.0

                # [+6] Is in combat. C++ reads AllyPerception->IsInCombat().
                # Python allies only exist during combat encounters, so always 1.0.
                obs[base + 6] = 1.0

                # [+7] Is dodging. C++ reads DodgeComp->IsDodging().
                # Python AlliedRobot doesn't implement dodging.
                obs[base + 7] = 0.0

                # [+8] Archetype as single scalar. C++ encodes as (enum+1)/4:
                #   Ranged(0)=0.25, Melee(1)=0.5, Healer(2)=0.75, Tank(3)=1.0
                obs[base + 8] = (ally.archetype + 1) / 4.0

                # [+9,10] Velocity normalised by ally's own max speed.
                # C++ uses AMC->MaxWalkSpeed per ally.
                ally_max_spd = ally.max_speed if ally.max_speed > 0 else 450.0
                obs[base + 9]  = np.clip(ally.velocity[0] / ally_max_spd, -1, 1)
                obs[base + 10] = np.clip(ally.velocity[1] / ally_max_spd, -1, 1)

                # [+11] Which hostile slot this ally is targeting.
                # C++ maps the ally's DetectedTarget to our ScoredTargets list,
                # then normalises: slot_index / (MaxHostileSlots - 1).
                # -1.0 if no target or target not in our hostile list.
                target_slot_norm = -1.0
                if ally.target_idx < len(self.targets):
                    ally_target = self.targets[ally.target_idx]
                    for h in range(min(len(sorted_targets), 4)):
                        if sorted_targets[h] is ally_target:
                            target_slot_norm = h / 3.0  # MaxHostileSlots - 1 = 3
                            break
                obs[base + 11] = target_slot_norm

        # ── Threat Sensing ────────────────────────────────────────
        # [Audit §1.4] Now handled correctly by base env's _build_observation
        # which scans self._projectiles (the SimProjectile list that's actually
        # populated). The old TrackedProjectile-based override here was reading
        # from self.projectiles which was never populated, and had wrong
        # direction (toward-agent not velocity) and TTA scale (/3.0 not /2.0).

        # ── Group Summary ────────────────────────────────────────
        alive_ally_count = sum(1 for ally in self.allies if ally.alive)
        alive_hostiles = sum(1 for t in self.targets if t.alive)

        obs[191] = min(alive_ally_count / 10.0, 1.0)            # alive allies (normalised by 10)
        if alive_ally_count > 0:
            obs[193] = sum(ally.hp_fraction() for ally in self.allies if ally.alive) / alive_ally_count
        else:
            obs[193] = 0.0                                    # avg ally HP

        # [Audit §1.11] C++ computes AliveAllies / (AliveAllies + AliveHostiles).
        # Does NOT count the agent itself — only other allied robots.
        total = alive_ally_count + alive_hostiles
        obs[195] = (alive_ally_count / total) if total > 0 else 0.5

        return obs


# ─────────────────────────────────────────────────────────────────
#  Extended Environment Factory
# ─────────────────────────────────────────────────────────────────

def make_extended_curriculum_env(stage: int, archetype: str = "ranged",
                                 render_mode: str = None) -> CombatEnvExtended:
    """Drop-in replacement for make_curriculum_env that uses the extended env."""
    # Use the base factory to get the config, then wrap in extended.
    base_env = make_curriculum_env(stage, archetype, render_mode)
    cfg = base_env.cfg
    base_env.close()

    return CombatEnvExtended(cfg, render_mode=render_mode)