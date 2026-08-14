"""Regression tests for production/training environment contracts."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from combat_sim import (  # noqa: E402
    COMBAT_ACTIONS,
    TARGET_SNAPSHOT_INTERVAL,
    CombatAction,
    Obstacle,
    SimProjectile,
    Target,
    make_curriculum_env,
)
from combat_extensions import (  # noqa: E402
    AlliedRobot,
    make_extended_curriculum_env,
)
from normalizers import ReturnNormalizer  # noqa: E402


class TargetActionSlotTests(unittest.TestCase):
    def setUp(self):
        self.env = make_curriculum_env(3, "ranged")
        self.env.reset(seed=123)

    def tearDown(self):
        self.env.close()

    def _install_targets(self):
        dead = Target(
            pos=np.array([100.0, 0.0], dtype=np.float32),
            hp=0.0,
            alive=False,
            is_player_controlled=True,
            target_id=10,
        )
        survivor = Target(
            pos=np.array([400.0, 125.0], dtype=np.float32),
            hp=150.0,
            alive=True,
            is_player_controlled=False,
            target_id=20,
        )
        self.env.targets = [dead, survivor]
        self.env.current_target_idx = 0
        self.env._publish_target_action_slots()
        return survivor

    def test_dead_raw_slot_compacts_consistently(self):
        survivor = self._install_targets()

        obs = self.env._build_observation()
        mask = self.env.build_action_mask()["t_mask"]

        self.assertEqual(tuple(self.env._target_action_slots), (survivor.target_id,))
        self.assertEqual(obs[74], 1.0)
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])
        self.assertTrue(mask[4])

        self.env._execute_target_selection(0)
        self.assertEqual(self.env.current_target_idx, 1)
        self.assertEqual(self.env._current_target().target_id, survivor.target_id)

    def test_observation_and_mask_are_pure_slot_consumers(self):
        self._install_targets()
        slot_ids = self.env._target_action_slots
        rng_state = self.env.rng.getstate()

        obs_a = self.env._build_observation()
        mask_a = self.env.build_action_mask()
        obs_b = self.env._build_observation()
        mask_b = self.env.build_action_mask()

        self.assertEqual(self.env._target_action_slots, slot_ids)
        self.assertEqual(self.env.rng.getstate(), rng_state)
        np.testing.assert_array_equal(obs_a, obs_b)
        for head in ("m_mask", "c_mask", "t_mask"):
            np.testing.assert_array_equal(mask_a[head], mask_b[head])

    def test_action_uses_preceding_published_order(self):
        first = Target(target_id=10, alive=True)
        second = Target(target_id=20, alive=True)
        self.env.targets = [first, second]
        self.env.current_target_idx = 0
        self.env._target_action_slots = (20, 10)
        self.env._target_slot_scores = {20: 100.0, 10: 50.0}

        self.env._execute_target_selection(0)

        self.assertEqual(self.env.current_target_idx, 1)
        self.assertEqual(self.env._current_target().target_id, 20)

    def test_step_consumes_the_slots_returned_by_the_previous_state(self):
        first = Target(target_id=10, alive=True)
        second = Target(target_id=20, alive=True)
        self.env.targets = [first, second]
        self.env.current_target_idx = 0
        self.env._target_action_slots = (20, 10)
        self.env._target_slot_scores = {20: 100.0, 10: 50.0}

        self.env.step(np.array([0, 0, 0]))

        self.assertEqual(self.env._current_target().target_id, 20)

    def test_hp_los_and_score_remain_snapshotted_for_two_seconds(self):
        first = Target(
            pos=np.array([1000.0, 0.0], dtype=np.float32),
            hp=100.0, max_hp=100.0, target_id=10,
        )
        second = Target(
            pos=np.array([1200.0, 400.0], dtype=np.float32),
            hp=100.0, max_hp=100.0, target_id=20,
        )
        self.env.agent.pos = np.zeros(2, dtype=np.float32)
        self.env.targets = [first, second]
        self.env.obstacles = []
        self.env._publish_target_action_slots()

        original_slots = self.env._target_action_slots
        original_scores = self.env._target_slot_scores.copy()
        first.hp = 50.0
        self.env.obstacles = [Obstacle(500.0, 0.0, 50.0, 100.0)]

        self.env._advance_target_action_snapshot(
            TARGET_SNAPSHOT_INTERVAL - self.env.cfg.decision_interval)
        obs = self.env._build_observation()
        first_slot = self.env._target_action_slots.index(first.target_id)
        first_base = 74 + first_slot * 17

        self.assertEqual(self.env._target_action_slots, original_slots)
        self.assertEqual(self.env._target_slot_scores, original_scores)
        self.assertEqual(obs[first_base + 4], 1.0)
        self.assertEqual(obs[first_base + 5], 1.0)

        self.env._advance_target_action_snapshot(
            self.env.cfg.decision_interval)
        obs = self.env._build_observation()
        first_slot = self.env._target_action_slots.index(first.target_id)
        first_base = 74 + first_slot * 17

        self.assertEqual(obs[first_base + 4], 0.5)
        self.assertEqual(obs[first_base + 5], 0.0)

    def test_dead_published_slot_forces_immediate_safe_refresh(self):
        targets = [
            Target(target_id=10, alive=True),
            Target(target_id=20, alive=True),
            Target(target_id=30, alive=True),
        ]
        self.env.targets = targets
        self.env._publish_target_action_slots()
        victim_id = self.env._target_action_slots[0]
        victim = next(t for t in targets if t.target_id == victim_id)
        victim.hp = 0.0
        victim.alive = False

        mask = self.env.build_action_mask()["t_mask"]

        self.assertNotIn(victim_id, self.env._target_action_slots)
        self.assertEqual(self.env._target_snapshot_elapsed, 0.0)
        self.assertEqual(
            np.flatnonzero(mask[:-1]).size,
            sum(1 for target in targets if target.alive),
        )


class CombatActionContractTests(unittest.TestCase):
    def setUp(self):
        self.env = make_curriculum_env(2, "ranged")
        self.env.reset(seed=456)
        self.env.obstacles = []
        self.env.agent.pos = np.zeros(2, dtype=np.float32)
        self.env.agent.hp = 10_000.0
        self.env.agent.max_hp = 10_000.0
        self.env.targets[0].pos = np.array([1000.0, 0.0], dtype=np.float32)

    def tearDown(self):
        self.env.close()

    def test_combat_head_contract_and_reposition_readiness(self):
        obs = self.env._build_observation()
        masks = self.env.build_action_mask()

        self.assertEqual(COMBAT_ACTIONS, 9)
        self.assertEqual(int(CombatAction.REPOSITION), 8)
        self.assertEqual(int(self.env.action_space.nvec[1]), 9)
        self.assertEqual(obs[73], 1.0)
        self.assertTrue(masks["c_mask"][CombatAction.REPOSITION])
        self.assertFalse(masks["skip_inference"])

    def test_reposition_uses_movement_head_and_holds_it_during_lock(self):
        self.env.agent.velocity = np.array(
            [self.env.agent.max_speed, 0.0], dtype=np.float32)

        obs, _, _, _, info = self.env.step(np.array([1, 8, 4]))

        self.assertEqual(info["executed_action"], (1, 8, 4))
        self.assertTrue(self.env.agent.is_repositioning)
        self.assertFalse(self.env.agent.is_dodging)
        self.assertEqual(self.env.agent.action_lock_reason, 7)
        self.assertAlmostEqual(
            np.linalg.norm(self.env.agent.velocity),
            self.env.agent.max_speed
            * self.env.agent.reposition_speed_multiplier,
            places=4,
        )
        self.assertEqual(obs[20], 0.0)
        self.assertEqual(obs[73], 0.0)
        self.assertAlmostEqual(
            self.env.agent.reposition_remaining, 0.4, places=5)
        self.assertAlmostEqual(
            self.env.agent.reposition_cooldown_remaining, 2.8, places=5)

        masks = info["action_mask"]
        np.testing.assert_array_equal(np.flatnonzero(masks["m_mask"]), [1])
        np.testing.assert_array_equal(np.flatnonzero(masks["c_mask"]), [0])
        np.testing.assert_array_equal(np.flatnonzero(masks["t_mask"]), [4])
        self.assertTrue(masks["skip_inference"])
        self.assertTrue(info["skip_inference"])

        _, _, _, _, locked_info = self.env.step(np.array([8, 7, 0]))
        self.assertEqual(locked_info["executed_action"], (1, 0, 4))
        self.assertFalse(self.env.agent.is_dodging)

    def test_reposition_with_stay_is_noop_without_cooldown(self):
        _, _, _, _, info = self.env.step(np.array([0, 8, 4]))

        self.assertEqual(info["executed_action"], (0, 0, 4))
        self.assertFalse(self.env.agent.is_repositioning)
        self.assertFalse(self.env.agent.is_action_locked)
        self.assertEqual(self.env.agent.reposition_cooldown_remaining, 0.0)

    def test_masked_combat_action_logs_effective_none(self):
        self.assertFalse(
            self.env.build_action_mask()["c_mask"][CombatAction.RELOAD])

        _, _, _, _, info = self.env.step(
            np.array([0, CombatAction.RELOAD, 4]))

        self.assertEqual(
            info["executed_combat_action"], CombatAction.NONE)
        self.assertFalse(self.env.agent.active_slot().is_reloading)

    def test_explicit_dodge_is_the_only_dodge_entrypoint(self):
        self.assertFalse(hasattr(self.env, "_try_auto_dodge"))
        self.assertFalse(hasattr(self.env.agent, "auto_dodge_enabled"))

        self.env.step(np.array([1, 0, 4]))
        self.assertFalse(self.env.agent.is_dodging)

        self.env.step(np.array([1, 7, 4]))
        self.assertTrue(self.env.agent.is_dodging)

    def test_block_applies_once_scales_movement_and_clears(self):
        base_defence = self.env.agent.defence
        self.env.agent.velocity = np.array(
            [self.env.agent.max_speed, 0.0], dtype=np.float32)

        self.env.step(np.array([1, 6, 4]))
        self.assertTrue(self.env.agent.is_blocking)
        self.assertEqual(
            self.env.agent.defence,
            base_defence + self.env.agent.block_defence_bonus,
        )
        self.assertAlmostEqual(
            np.linalg.norm(self.env.agent.velocity),
            self.env.agent.max_speed
            * self.env.agent.block_movement_multiplier,
            places=4,
        )

        self.env.step(np.array([1, 6, 4]))
        self.assertEqual(
            self.env.agent.defence,
            base_defence + self.env.agent.block_defence_bonus,
        )

        self.env.step(np.array([1, 0, 4]))
        self.assertFalse(self.env.agent.is_blocking)
        self.assertEqual(self.env.agent.defence, base_defence)


class CurriculumContractTests(unittest.TestCase):
    def test_stages_one_through_five_are_exactly_one_v_one(self):
        for stage in range(1, 6):
            with self.subTest(stage=stage):
                env = make_extended_curriculum_env(stage, "ranged")
                try:
                    env.reset(seed=100 + stage)
                    self.assertEqual(env.cfg.num_enemies, 1)
                    self.assertEqual(env.cfg.num_targets, 1)
                    self.assertEqual(len(env.allies), 0)
                    self.assertEqual(len(env.targets), 1)
                finally:
                    env.close()

    def test_stage_six_is_exactly_two_enemies_v_one_player(self):
        env = make_extended_curriculum_env(6, "ranged")
        try:
            env.reset(seed=606)
            self.assertEqual(env.cfg.num_enemies, 2)
            self.assertEqual(env.cfg.num_targets, 1)
            self.assertEqual(len(env.allies), 1)
            self.assertEqual(len(env.targets), 1)
            self.assertNotEqual(env.allies[0].archetype, 2)
        finally:
            env.close()

    def test_stage_seven_uses_seeded_equal_team_buckets(self):
        first = make_extended_curriculum_env(7, "ranged")
        second = make_extended_curriculum_env(7, "ranged")
        try:
            self.assertEqual(first.cfg.squad_size_buckets, (1, 2, 3, 4))
            sampled_sizes = []
            for seed in range(12):
                first.reset(seed=seed)
                second.reset(seed=seed)
                size = first.episode_squad_size
                sampled_sizes.append(size)

                self.assertEqual(size, second.episode_squad_size)
                self.assertIn(size, (1, 2, 3, 4))
                self.assertEqual(first.cfg.num_enemies, size)
                self.assertEqual(first.cfg.num_targets, size)
                self.assertEqual(len(first.allies), size - 1)
                self.assertEqual(len(first.targets), size)
                self.assertTrue(
                    all(ally.archetype != 2 for ally in first.allies))
            self.assertEqual(
                Counter(sampled_sizes),
                Counter({1: 3, 2: 3, 3: 3, 4: 3}),
            )
        finally:
            first.close()
            second.close()


class ExtendedTransitionTests(unittest.TestCase):
    def setUp(self):
        self.env = make_extended_curriculum_env(6, "ranged")
        self.env.reset(seed=321)

    def tearDown(self):
        self.env.close()

    def test_stun_mask_matches_executed_noop(self):
        self.env.status_effects.apply_stun(1.0)

        masks = self.env.build_action_mask()

        np.testing.assert_array_equal(
            np.flatnonzero(masks["m_mask"]), [0])
        np.testing.assert_array_equal(
            np.flatnonzero(masks["c_mask"]), [0])
        np.testing.assert_array_equal(
            np.flatnonzero(masks["t_mask"]), [4])
        self.assertFalse(masks["skip_inference"])

    def test_reset_reinitializes_previous_ally_count(self):
        self.env.allies[0].alive = False

        self.env.reset(seed=321)

        self.assertEqual(
            self.env._prev_alive_allies,
            sum(1 for ally in self.env.allies if ally.alive),
        )

    def test_ally_final_kill_is_rewarded_and_terminated_same_step(self):
        target = Target(
            pos=np.zeros(2, dtype=np.float32),
            hp=1.0,
            max_hp=100.0,
            defence=0.0,
            alive=True,
            target_id=77,
        )
        ally = AlliedRobot(
            pos=np.zeros(2, dtype=np.float32),
            attack_range=10_000.0,
            attack_damage=100.0,
            attack_cooldown_remaining=0.0,
        )
        self.env.targets = [target]
        self.env.allies = [ally]
        self.env.obstacles = []
        self.env.current_target_idx = 0
        self.env.agent.hp = 10_000.0
        self.env.agent.max_hp = 10_000.0
        self.env._prev_target_hps = {target.target_id: target.hp_fraction()}
        self.env._publish_target_action_slots()

        _, reward, done, truncated, info = self.env.step(
            np.array([0, 0, 4]))

        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertTrue(info["is_win"])
        self.assertGreater(info["episode_end"], 0.0)
        self.assertGreater(reward, 0.0)
        self.assertEqual(info["kill"], 0.0)
        self.assertEqual(info["damage_dealt"], 0.0)
        self.assertEqual(info["episode_total_damage"], 0.0)

    def test_partial_ally_damage_cannot_trigger_agent_shaping(self):
        selected = Target(
            pos=np.array([1000.0, 0.0], dtype=np.float32),
            hp=31.0,
            max_hp=100.0,
            defence=0.0,
            alive=True,
            target_id=78,
        )
        other = Target(
            pos=np.zeros(2, dtype=np.float32),
            hp=100.0,
            max_hp=100.0,
            defence=0.0,
            alive=True,
            is_player_controlled=False,
            target_id=79,
        )
        ally = AlliedRobot(
            pos=selected.pos.copy(),
            attack_range=10_000.0,
            attack_damage=10.0,
            attack_cooldown_remaining=0.0,
        )
        projectile = SimProjectile(
            pos=np.array([-10.0, 0.0], dtype=np.float32),
            velocity=np.array([1200.0, 0.0], dtype=np.float32),
            damage=10.0,
            attack_stat=5.0,
            is_agent_projectile=True,
        )
        self.env.targets = [selected, other]
        self.env.allies = [ally]
        self.env._projectiles = [projectile]
        self.env.obstacles = []
        self.env.current_target_idx = 0
        self.env.cfg.target_speed_fraction = 0.0
        self.env.agent.hp = 10_000.0
        self.env.agent.max_hp = 10_000.0
        self.env._prev_target_hps = {
            target.target_id: target.hp_fraction()
            for target in self.env.targets
        }
        self.env._publish_target_action_slots()

        _, _, _, _, info = self.env.step(np.array([0, 0, 4]))

        self.assertLess(selected.hp_fraction(), 0.3)
        self.assertLess(other.hp_fraction(), 1.0)
        self.assertEqual(info["kill"], 0.0)
        self.assertGreater(info["damage_dealt"], 0.0)
        self.assertEqual(info["focus_ratio"], 0.0)
        self.assertNotIn("target_low_hp", info)

    def test_ally_target_id_maps_to_published_observation_slot(self):
        first = Target(target_id=10, alive=True)
        second = Target(target_id=20, alive=True)
        ally = AlliedRobot(target_id=10)
        self.env.targets = [first, second]
        self.env.allies = [ally]
        self.env._target_action_slots = (20, 10)
        self.env._target_slot_scores = {20: 100.0, 10: 50.0}

        obs = self.env._build_observation()

        self.assertAlmostEqual(obs[154], 0.4)


class ReturnNormalizerTests(unittest.TestCase):
    def test_vector_env_returns_do_not_contaminate_each_other(self):
        normalizer = ReturnNormalizer(gamma=0.5)

        normalizer.update(
            np.array([1.0, 10.0]),
            np.array([False, False]),
        )
        np.testing.assert_allclose(normalizer.running_return, [1.0, 10.0])

        normalizer.update(
            np.array([1.0, 10.0]),
            np.array([False, False]),
        )
        np.testing.assert_allclose(normalizer.running_return, [1.5, 15.0])
        self.assertAlmostEqual(
            normalizer.running_mean,
            np.mean([1.0, 10.0, 1.5, 15.0]),
            places=3,
        )
        self.assertGreater(normalizer.running_var, 0.0)
        self.assertTrue(np.isfinite(
            normalizer.normalize(np.array([1.0, 10.0]))).all())

        normalizer.update(
            np.array([1.0, 10.0]),
            np.array([True, False]),
        )
        np.testing.assert_allclose(normalizer.running_return, [0.0, 17.5])


if __name__ == "__main__":
    unittest.main()
