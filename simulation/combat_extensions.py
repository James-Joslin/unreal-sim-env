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
       AlliedRobot exposes ammo_fraction, is_reloading, fire_cooldown,
       and current_combat_action for the base class's 15-field ally slots.
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

DEBUFF_NAMES = (
    "poison", "scold", "shock", "frostbite", "weakness", "curse",
)

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
        default_factory=lambda: [
            StatusEffect(name=name) for name in DEBUFF_NAMES
        ])

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
        normalized = str(name).lower()
        if normalized not in DEBUFF_NAMES:
            raise ValueError(f"Unsupported production debuff: {name}")
        slot = self.debuff_slots[DEBUFF_NAMES.index(normalized)]
        slot.duration_remaining = max(slot.duration_remaining, duration)
        slot.strength = strength

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
    target_id: int = -1  # Stable ID of the hostile this ally is targeting.
    ally_id: int = -1
    current_combat_action: int = 0  # What the ally is doing (for obs encoding).

    # ── Properties read by base CombatEnv._build_observation() ──
    # AlliedRobot uses cooldown-based attacks, not a weapon system,
    # so these map the cooldown state to weapon-like observations.

    @property
    def ammo_fraction(self) -> float:
        """Effective ammo: 1.0 when off cooldown, 0.0 when on cooldown."""
        if self.attack_cooldown <= 0:
            return 1.0
        return max(0.0, 1.0 - self.attack_cooldown_remaining / self.attack_cooldown)

    @property
    def is_reloading(self) -> bool:
        """No reload mechanic — always False."""
        return False

    @property
    def active_weapon_fire_cooldown(self) -> float:
        """Maps attack_cooldown_remaining for the obs fire_cd field."""
        return self.attack_cooldown_remaining

    def hp_fraction(self) -> float:
        return max(0, self.hp / self.max_hp)

    def tick(self, dt: float, targets: List[Target], obstacles: List[Obstacle],
             arena_half: float, rng=None):
        """Simple ally AI: approach nearest alive target and attack."""
        hits = []
        if not self.alive:
            return hits

        self.attack_cooldown_remaining -= dt

        # Pick nearest alive target.
        best_dist = 99999
        best_target = None
        for t in targets:
            if t.alive:
                d = np.linalg.norm(self.pos - t.pos)
                if d < best_dist:
                    best_dist = d
                    best_target = t

        self.target_id = best_target.target_id if best_target else -1
        target = best_target

        if not target:
            self.velocity = np.zeros(2, dtype=np.float32)
            return hits

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
                # Track combat action: melee archetype = 5 (Melee), others = 1 (Fire).
                self.current_combat_action = 5 if self.archetype == 1 else 1
                hp_before = max(0.0, float(target.hp))
                dmg, target.barrier, was_crit = compute_damage(
                    self.attack_damage, 5.0, target.defence, target.barrier,
                    rng=rng)
                target.hp = max(0.0, target.hp - dmg)
                hp_lost = max(0.0, hp_before - target.hp)
                killed = hp_before > 0.0 and target.hp <= 0.0
                if killed:
                    target.alive = False
                hits.append((target, hp_lost, was_crit, killed))
            else:
                self.current_combat_action = 0  # No LOS — can't attack.
        else:
            self.current_combat_action = 0  # Not attacking this tick.

        return hits


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

    The base CombatEnv produces the full 249-float observation vector.
    This class adds simulation of systems that exist in UE but are
    simplified in the base sim:
    - Status effects (stun/slow/debuffs) applied randomly by targets
    - Target acceleration tracking (fills base's skipped accel slots)
    - Threat/priority scoring per hostile (fills base's placeholder scores)
    - Allied robots (stages 6+) with coordination attributes
    - Projectile tracking with TTA
    - Sight cone checks (replaces base's always-true)
    - Group summary with real ally count (base hardcodes allies=0)
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
        self._prev_alive_allies = sum(
            1 for ally in self.allies if ally.alive)

        # Rebuild observation with extended features.
        obs = self._build_observation()
        action_mask = self.build_action_mask()
        info["action_mask"] = action_mask
        info["skip_inference"] = action_mask["skip_inference"]
        return obs, info

    def build_action_mask(self):
        """Expose only the action that will actually execute while stunned."""
        masks = super().build_action_mask()
        if hasattr(self, "status_effects") and self.status_effects.is_stunned:
            # Runtime stun masks execution but does not pause neural inference
            # or recurrent-state advancement unless an action lock also exists.
            masks["m_mask"][:] = False
            masks["m_mask"][0] = True
            masks["c_mask"][:] = False
            masks["c_mask"][0] = True
            masks["t_mask"][:] = False
            masks["t_mask"][4] = True
        return masks

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

        try:
            return super().step(action)
        finally:
            # The returned observation is already materialized; restore the
            # persistent stat even if transition finalization raises.
            self.agent.max_speed = original_speed

    def _before_transition_finalization(self, dt: float):
        """Apply extended systems before reward/done/obs/mask are finalized."""
        # Tick status effects.
        self.status_effects.tick(dt)

        # NOTE: threat_table.decay and damage recording are now handled
        # directly in the base env's step (melee hits + projectile hits).
        # [Audit §1.3]

        # Targets may apply status effects.
        self._maybe_apply_status_effects(dt)

        # Tick allies.
        for ally in self.allies:
            hits = ally.tick(
                dt, self.targets, self.obstacles, self._arena_half,
                rng=self.rng)
            for target, hp_lost, was_crit, killed in hits:
                self._record_damage_event(
                    "ally", ally.ally_id, target, hp_lost,
                    source_name="AllyAttack",
                    delivery="scripted_ally",
                    was_crit=was_crit,
                    killed=killed,
                    intended_target_id=target.target_id,
                )

        # Targets may attack allies too.
        self._targets_attack_allies(dt)

        # Tick tracked projectiles (observation-only, separate from base SimProjectiles).
        self._tick_tracked_projectiles(dt)

    # ═════════════════════════════════════════════════════════════
    #  Ally Spawning
    # ═════════════════════════════════════════════════════════════

    def _spawn_allies(self):
        """Spawn allied robots for group stages."""
        self.allies = []
        stage = self.cfg.curriculum_stage

        # num_enemies includes the agent, so allies = num_enemies - 1.
        num_allies = max(0, self.cfg.num_enemies - 1)

        if stage < 6:
            return

        for _ in range(num_allies):
            # Healer remains a compatible enum value but is not part of the
            # current production spawn pool.
            arch = self.rng.choices(
                [0, 1, 3], weights=[40, 30, 15], k=1)[0]
            ally = self._make_ally(arch)
            ally.ally_id = len(self.allies)
            self.allies.append(ally)

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
        """Roll status effects only for accepted target-to-agent hits."""
        stage = self.cfg.curriculum_stage
        if stage < 3:
            return  # No status effects in early stages.

        accepted_hits = [
            event for event in self._step_damage_events
            if (event.attacker_kind == "target"
                and event.victim_kind == "agent"
                and event.damage > 0.0)
        ]
        for event in accepted_hits:
            target = next(
                (
                    candidate for candidate in self.targets
                    if candidate.target_id == event.attacker_id
                ),
                None,
            )

            if event.delivery == "melee" and self.rng.random() < 0.08:
                self.status_effects.apply_stun(
                    0.4 + self.rng.uniform(0.0, 0.3))
                self._record_status_event(
                    "target", event.attacker_id, self.agent, "stun",
                    source_name=event.source_name,
                    intended_target_id=event.intended_target_id)

            if (event.delivery in ("projectile", "arc_projectile")
                    and target is not None
                    and target.combat_role in ("ranged", "mixed")
                    and self.rng.random() < 0.05):
                self.status_effects.apply_slow(
                    1.0 + self.rng.uniform(0.0, 1.0),
                    strength=self.rng.uniform(0.2, 0.5))
                self._record_status_event(
                    "target", event.attacker_id, self.agent, "slow",
                    source_name=event.source_name,
                    intended_target_id=event.intended_target_id)

            if self.rng.random() < 0.02:
                debuff_name = self.rng.choice(DEBUFF_NAMES)
                self.status_effects.apply_debuff(
                    debuff_name,
                    duration=self.rng.uniform(2.0, 5.0),
                    strength=self.rng.uniform(0.1, 0.4))
                self._record_status_event(
                    "target", event.attacker_id, self.agent, debuff_name,
                    source_name=event.source_name,
                    intended_target_id=event.intended_target_id)

    # ═════════════════════════════════════════════════════════════
    #  Threat Tracking
    # ═════════════════════════════════════════════════════════════

    def _update_threat_from_targets(self, dt: float):
        """Compatibility no-op: threat now consumes the damage-event ledger.

        Base ``CombatEnv._record_damage_event`` attributes each real
        target→agent hit to its actual source ID and updates threat from the
        same event ledger. The previous implementation guessed attackers from
        HP/reward deltas and nearby targets, which was ambiguous and wrong for
        asynchronous projectiles.
        """
        return None

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
                            hp_before = max(0.0, float(target_ally.hp))
                            dmg, _, was_crit = compute_damage(
                                t.attack_damage, t.attack_stat,
                                target_ally.defence, 0.0,
                                t.crit_chance, t.crit_multiplier,
                                rng=self.rng)
                            target_ally.hp = max(0.0, target_ally.hp - dmg)
                            hp_lost = max(0.0, hp_before - target_ally.hp)
                            killed = hp_before > 0.0 and target_ally.hp <= 0.0
                            if killed:
                                target_ally.alive = False
                            self._record_damage_event(
                                "target", t.target_id, target_ally, hp_lost,
                                source_name="TargetAllyAttack",
                                delivery="direct",
                                was_crit=was_crit,
                                killed=killed,
                                intended_target_id=target_ally.ally_id,
                            )

    # ═════════════════════════════════════════════════════════════
    #  Observation Builder (fills ALL 249 features)
    # ═════════════════════════════════════════════════════════════

    def _build_observation(self) -> np.ndarray:
        """Override: enhances the base 249-float observation with features
        that only CombatEnvExtended simulates (status effects, sight cone,
        acceleration, threat/priority scoring, group summary with allies).

        The base class already handles:
          - Allied robot slots [142..186] via getattr(self, 'allies', [])
          - Player patterns [244..248] via self._player_patterns
          - All other sections
        So we only override the fields we add value to.
        """
        obs = super()._build_observation()

        a = self.agent
        target = self._current_target()
        se = self.status_effects

        # ── Self State: fill status effect slots ─────────────────
        obs[3] = 1.0 if se.is_stunned else 0.0             # stunned
        obs[4] = 1.0 if se.slowed.active else 0.0          # slowed
        for i in range(6):                                   # debuff slots
            obs[5 + i] = 1.0 if se.debuff_slots[i].active else 0.0
        # [Audit §1.8] C++ TraceHeightAboveGround returns ~88 UU for a
        # grounded ACharacter. 88/500 ≈ 0.176.
        obs[14] = 0.176                                      # height above ground

        # ── Primary Target: sight cone and acceleration ──────────
        # Primary target starts at offset 50 (unchanged).
        # Sight cone = +6 = index 56. Accel = +11,+12 = indices 61,62.
        if target and target.alive:
            # Sight cone: check if target is in our forward arc.
            rel = target.pos - a.pos
            dist = np.linalg.norm(rel)
            if dist > 1:
                target_dir = rel / dist
                facing_dot = float(np.dot(a.facing, target_dir))
                obs[56] = 1.0 if facing_dot > -0.17 else 0.0  # ~100° half-cone

            # Target acceleration (velocity delta / dt).
            # Base env skips these with `idx += 2`. We fill them here.
            prev_vel = self.prev_target_velocities.get(target.target_id, np.zeros(2, dtype=np.float32))
            accel = (target.velocity - prev_vel) / max(self.cfg.decision_interval, 0.01)
            obs[61] = np.clip(accel[0] / 2000, -1, 1)
            obs[62] = np.clip(accel[1] / 2000, -1, 1)

        # Hostile priority/threat values are already written by the base
        # builder from the same published slots used by the action mask.

        # ── Allied Robots ────────────────────────────────────────
        # The base class handles ally observation slots [142..186]
        # via getattr(self, 'allies', []) and reads attributes that
        # AlliedRobot now provides (ammo_fraction, is_reloading,
        # active_weapon_fire_cooldown, current_combat_action).
        # No override needed here.

        # ── Threat Sensing ────────────────────────────────────────
        # Handled correctly by base env's _build_observation which
        # scans self._projectiles (the SimProjectile list).

        # ── Group Summary ────────────────────────────────────────
        # Group summary starts at offset 220.
        # Base class hardcodes alive_allies = 0, so we override
        # with the real ally count from CombatEnvExtended.
        alive_ally_count = sum(1 for ally in self.allies if ally.alive)
        alive_hostiles = sum(1 for t in self.targets if t.alive)

        obs[220] = min(alive_ally_count / 10.0, 1.0)          # [+0] alive allies
        # obs[221] = alive_hostiles — already set correctly by base.
        if alive_ally_count > 0:
            obs[222] = sum(ally.hp_fraction() for ally in self.allies if ally.alive) / alive_ally_count
        else:
            obs[222] = 0.0                                     # [+2] avg ally HP
        # obs[223] = avg hostile HP — already set correctly by base.

        # [Audit §1.11] C++ computes AliveAllies / (AliveAllies + AliveHostiles).
        # Does NOT count the agent itself — only other allied robots.
        total = alive_ally_count + alive_hostiles
        obs[224] = (alive_ally_count / total) if total > 0 else 0.5  # [+4] numerical advantage
        obs[225] = 1.0 if alive_hostiles > alive_ally_count else 0.0  # [+5] outnumbered

        return obs


# ─────────────────────────────────────────────────────────────────
#  Extended Environment Factory
# ─────────────────────────────────────────────────────────────────

def make_extended_curriculum_env(stage: int, archetype: str = "ranged",
                                 render_mode: str = None,
                                 behavior_profiles=None,
                                 behavior_profile_offset: int = 0) -> CombatEnvExtended:
    """Drop-in replacement for make_curriculum_env that uses the extended env."""
    # Use the base factory to get the config, then wrap in extended.
    base_env = make_curriculum_env(
        stage, archetype, render_mode,
        behavior_profiles=behavior_profiles,
        behavior_profile_offset=behavior_profile_offset)
    cfg = base_env.cfg
    base_env.close()

    return CombatEnvExtended(cfg, render_mode=render_mode)
