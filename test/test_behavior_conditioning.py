"""Mechanics, telemetry, and categorical teacher-conditioning regressions."""

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from behavior_profiles import (  # noqa: E402
    BEHAVIOR_CONDITION_DIM,
    BehaviorProfile,
    behavior_condition,
)
from combat_sim import (  # noqa: E402
    CombatEnv,
    CombatEnvConfig,
    HEAVY_MISSILE_MECHANICS,
    SimProjectile,
    Target,
    WeaponSlot,
    estimate_projectile_flight_time,
    predict_firing_solution,
)
from scenario_manifest import (  # noqa: E402
    generate_scenario_manifest,
    validate_scenario_manifest,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from frame_stack import VecFrameStackEnv
    from methods.ppo.actor_critic import ActorCritic
    from methods.ppo.buffer import VecRolloutBuffer
else:
    VecFrameStackEnv = None


class ProjectileParityTests(unittest.TestCase):
    def test_firing_solution_includes_windup_and_iterated_flight(self):
        start = np.array([0.0, 0.0], dtype=np.float32)
        target = np.array([1000.0, 0.0], dtype=np.float32)
        velocity = np.array([0.0, 100.0], dtype=np.float32)
        flight = estimate_projectile_flight_time(
            start, target, velocity, 0.5, 1000.0)
        aim = predict_firing_solution(
            start, target, velocity, 0.5, 1000.0)
        self.assertGreater(flight, 1.0)
        self.assertAlmostEqual(flight, 1.01135647, places=6)
        np.testing.assert_allclose(
            aim, target + velocity * (0.5 + flight), rtol=0, atol=1e-4)
        self.assertAlmostEqual(float(aim[1]), 151.13565, places=4)

    def test_heavy_missile_uses_measured_production_asset_mechanics(self):
        self.assertAlmostEqual(
            HEAVY_MISSILE_MECHANICS["projectile_speed"], 5053.8667, places=4)
        self.assertAlmostEqual(
            HEAVY_MISSILE_MECHANICS["projectile_lifetime"], 10.0)
        self.assertAlmostEqual(
            HEAVY_MISSILE_MECHANICS["arc_height"], 2247.2354, places=4)
        self.assertAlmostEqual(
            HEAVY_MISSILE_MECHANICS["impact_radius"], 509.8804, places=4)
        self.assertAlmostEqual(
            HEAVY_MISSILE_MECHANICS["damage_falloff"], 0.5)
        flight = estimate_projectile_flight_time(
            np.zeros(2, dtype=np.float32),
            np.array([3000.0, 0.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32), 0.0,
            HEAVY_MISSILE_MECHANICS["projectile_speed"], True,
            HEAVY_MISSILE_MECHANICS["arc_height"])
        self.assertAlmostEqual(flight, 1.0692263, places=6)

    def test_nominal_range_does_not_suppress_projectile_spawn(self):
        env = CombatEnv(CombatEnvConfig(
            num_targets=1, num_obstacles=0, weapon_preset="scout",
            engagement_distance=2000.0, target_speed_fraction=0.0))
        env.reset(seed=7)
        env.agent.pos[:] = 0.0
        env.targets[0].pos[:] = (2000.0, 0.0)
        slot = env.agent.active_slot()
        self.assertGreater(2000.0, slot.weapon_range)
        ammo_before = slot.current_ammo
        env._execute_combat(1, 0, env.cfg.decision_interval)
        self.assertEqual(slot.current_ammo, ammo_before - 1)
        self.assertEqual(len(env._projectiles), 1)
        env.close()

    def test_physically_unreachable_shot_still_spawns_but_is_not_credible(self):
        env = CombatEnv(CombatEnvConfig(
            num_targets=1, num_obstacles=0, weapon_preset="scout",
            engagement_distance=1000.0, target_speed_fraction=0.0))
        env.reset(seed=11)
        env.agent.pos[:] = 0.0
        env.targets[0].pos[:] = (1000.0, 0.0)
        slot = env.agent.active_slot()
        slot.projectile_lifetime = 0.01
        state = env._build_combat_state()
        self.assertFalse(state.target_physically_reachable)
        self.assertFalse(state.credible_fire_opportunity)
        env._execute_combat(1, 0, env.cfg.decision_interval)
        self.assertEqual(len(env._projectiles), 1)
        env.close()

    def test_configured_aoe_radius_and_falloff(self):
        center = Target(
            pos=np.array([0.0, 0.0], dtype=np.float32), hp=100.0,
            max_hp=100.0, defence=0.0)
        edge = Target(
            pos=np.array([100.0, 0.0], dtype=np.float32), hp=100.0,
            max_hp=100.0, defence=0.0)
        projectile = SimProjectile(
            pos=np.zeros(2, dtype=np.float32), damage=10.0,
            attack_stat=0.0, is_arc=True, arc_impact_radius=100.0,
            arc_damage_falloff=0.2)
        hits = projectile._apply_arc_impact([center, edge], None)
        dealt = [hit[1] for hit in hits]
        self.assertEqual(len(dealt), 2)
        self.assertAlmostEqual(dealt[0], 10.0)
        self.assertAlmostEqual(dealt[1], 2.0)


class ProfileLifecycleTests(unittest.TestCase):
    def test_profile_is_one_hot(self):
        for profile in BehaviorProfile:
            condition = behavior_condition(profile)
            self.assertEqual(condition.shape, (BEHAVIOR_CONDITION_DIM,))
            self.assertEqual(float(condition.sum()), 1.0)
            self.assertEqual(int(condition.argmax()), int(profile))

    @unittest.skipIf(
        VecFrameStackEnv is None,
        "PyTorch training dependencies are required for vector lifecycle test")
    def test_autoreset_preserves_terminal_profile_then_assigns_next(self):
        def make_env():
            return CombatEnv(CombatEnvConfig(
                max_steps=1, num_targets=1, num_obstacles=0,
                behavior_profiles=("reactive", "tactical")))

        vec = VecFrameStackEnv([make_env], frame_stack=1)
        _, initial = vec.reset()
        self.assertEqual(
            initial[0]["behavior_profile_id"], int(BehaviorProfile.REACTIVE))
        _, _, _, truncated, infos = vec.step(np.array([[0, 0, 4]]))
        self.assertTrue(truncated[0])
        self.assertEqual(
            infos[0]["terminal_behavior_profile_id"],
            int(BehaviorProfile.REACTIVE))
        self.assertEqual(
            infos[0]["behavior_profile_id"], int(BehaviorProfile.TACTICAL))
        self.assertNotEqual(
            infos[0]["terminal_scenario_id"], infos[0]["scenario_id"])
        vec.close()

    def test_manifest_splits_scenario_before_profile_comparison(self):
        manifest = generate_scenario_manifest(
            stages=(4,), archetypes=("ranged",), loadouts=("heavy",),
            scenarios_per_cell=10)
        self.assertTrue(validate_scenario_manifest(manifest))
        all_rows = sum((manifest[key] for key in ("train", "validation", "test")), [])
        self.assertTrue(all("profile" not in row for row in all_rows))


@unittest.skipIf(torch is None, "PyTorch is required for conditioning tests")
class ConditionedModelTests(unittest.TestCase):
    def test_single_frame_conditioned_model_uses_zero_temporal_deltas(self):
        model = ActorCritic(
            obs_size=249, tier="large", behavior_conditioned=True).eval()
        obs = torch.randn(2, 249)

        with torch.no_grad():
            deltas = model.delta(obs)

        self.assertTrue(torch.equal(deltas[:, 0], obs))
        self.assertTrue(torch.equal(deltas[:, 1], torch.zeros_like(obs)))
        self.assertTrue(torch.equal(deltas[:, 2], torch.zeros_like(obs)))

    def test_actor_and_critic_receive_direct_four_value_condition(self):
        model = ActorCritic(
            obs_size=249, tier="large", behavior_conditioned=True).eval()
        self.assertEqual(model.behavior_condition_dim, 4)
        self.assertEqual(
            model.actor_backbone[0].in_features,
            3 * model.actor_encoder.channel_dim + 4)
        self.assertEqual(
            model.critic_backbone[0].in_features,
            3 * model.critic_encoder.channel_dim + 4)

        obs = torch.zeros(1, 249)
        hidden = model.init_hidden(1)
        reactive = torch.from_numpy(
            behavior_condition("reactive")).unsqueeze(0)
        tactical = torch.from_numpy(
            behavior_condition("tactical")).unsqueeze(0)
        with torch.no_grad():
            reactive_features, _ = model._actor_features(
                obs, hidden, reactive)
            tactical_features, _ = model._actor_features(
                obs, hidden, tactical)
        self.assertFalse(torch.equal(reactive_features, tactical_features))

    def test_rollout_buffer_preserves_profile_id_and_condition(self):
        buffer = VecRolloutBuffer(
            2, 1, 249, gru_hidden=4, behavior_condition_dim=4)
        buffer.behavior_conditions[0, 0] = behavior_condition("reactive")
        buffer.behavior_conditions[1, 0] = behavior_condition("tactical")
        buffer.behavior_profile_ids[:, 0] = (
            int(BehaviorProfile.REACTIVE), int(BehaviorProfile.TACTICAL))
        flat = buffer.flatten()
        self.assertEqual(flat["behavior_conditions"].shape, (2, 4))
        self.assertEqual(
            flat["behavior_profile_ids"].tolist(),
            [int(BehaviorProfile.REACTIVE), int(BehaviorProfile.TACTICAL)])


if __name__ == "__main__":
    unittest.main()
