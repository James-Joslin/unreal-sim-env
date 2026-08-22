"""Focused regressions for the deployed policy/training contract."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from combat_policy import (
        ACTIVE_TIERS,
        BEHAVIOR_TIER_DEFINITIONS,
        CombatPolicy,
        TIER_CONFIGS,
        build_feature_visibility,
    )
    from distillation import _masked_kd_per_step
    from methods.ppo.trainer import freeze_skipped_hidden


@unittest.skipIf(torch is None, "PyTorch is required for policy tests")
class PolicyContractTests(unittest.TestCase):
    def _make_policy(self, tier):
        return CombatPolicy(tier=tier, **TIER_CONFIGS[tier]).eval()

    def test_active_tiers_and_nested_feature_budgets(self):
        self.assertEqual(
            ACTIVE_TIERS, ("micro", "small", "medium", "large"))

        masks = [build_feature_visibility(tier) for tier in ACTIVE_TIERS]
        self.assertEqual(
            [int(mask.sum()) for mask in masks],
            [89, 145, 193, 249],
        )
        for lower, higher in zip(masks, masks[1:]):
            self.assertTrue(torch.all(~lower | higher))

        micro, small = masks[:2]
        self.assertFalse(micro[65])
        self.assertFalse(micro[66])
        for slot in range(4):
            base = 74 + slot * 17
            self.assertTrue(small[base])
            self.assertTrue(small[base + 10])
            self.assertTrue(small[base + 11])

    def test_tier_action_budgets_are_deliberate(self):
        counts = []
        for tier in ACTIVE_TIERS:
            definition = BEHAVIOR_TIER_DEFINITIONS[tier]
            counts.append((
                len(definition["movement_actions"]),
                len(definition["combat_actions"]),
                len(definition["target_actions"]),
            ))
        self.assertEqual(
            counts,
            [(9, 4, 2), (9, 8, 5), (9, 9, 5), (9, 9, 5)],
        )

    def test_independent_heads_have_exact_output_contract(self):
        for tier in ACTIVE_TIERS:
            with self.subTest(tier=tier):
                policy = self._make_policy(tier)
                obs = torch.zeros(2, 3 * 249)
                movement, combat, target, hidden = policy(obs)

                self.assertEqual(tuple(movement.shape), (2, 9))
                self.assertEqual(tuple(combat.shape), (2, 9))
                self.assertEqual(tuple(target.shape), (2, 5))
                self.assertEqual(
                    tuple(hidden.shape),
                    (1, 2, TIER_CONFIGS[tier]["gru_hidden"]),
                )
                parameter_names = tuple(
                    name for name, _ in policy.named_parameters())
                for legacy_name in (
                    "move_embed", "combat_proj",
                    "combat_embed", "target_proj",
                ):
                    self.assertFalse(any(
                        legacy_name in name for name in parameter_names))

    def test_forward_sequence_matches_stepwise_hidden_chaining(self):
        torch.manual_seed(123)
        policy = self._make_policy("micro")
        obs = torch.randn(2, 4, 3 * 249)

        sequence = policy.forward_sequence(obs)
        hidden = policy.init_hidden(batch_size=2)
        step_logits = [[], [], []]
        for step in range(obs.shape[1]):
            outputs = policy(obs[:, step], hidden)
            for head in range(3):
                step_logits[head].append(outputs[head].unsqueeze(1))
            hidden = outputs[3]

        for sequence_head, pieces in zip(sequence[:3], step_logits):
            torch.testing.assert_close(
                sequence_head, torch.cat(pieces, dim=1))
        torch.testing.assert_close(sequence[3], hidden)

    def test_masked_distillation_is_finite_and_backpropagates(self):
        micro = self._make_policy("micro")
        shapes = (9, 9, 5)
        student_logits = tuple(
            torch.randn(2, 3, size, requires_grad=True)
            for size in shapes
        )
        teacher_logits = tuple(
            torch.randn(2, 3, size) for size in shapes)
        availability = (
            micro.movement_availability,
            micro.combat_availability,
            micro.target_availability,
        )
        masks = tuple(
            mask.view(1, 1, -1).expand(2, 3, -1)
            for mask in availability
        )

        loss = _masked_kd_per_step(
            student_logits, teacher_logits, masks,
            alpha=0.7, temperature=3.0,
        ).mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for logits in student_logits:
            self.assertIsNotNone(logits.grad)
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_mixed_batch_hidden_freeze_is_per_environment(self):
        hidden_in = torch.tensor([[
            [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],
        ]])
        hidden_out = torch.tensor([[
            [10.0, 10.0], [20.0, 20.0], [30.0, 30.0],
        ]])
        frozen = freeze_skipped_hidden(
            hidden_in, hidden_out, [False, True, False])

        torch.testing.assert_close(frozen[:, 0], hidden_out[:, 0])
        torch.testing.assert_close(frozen[:, 1], hidden_in[:, 1])
        torch.testing.assert_close(frozen[:, 2], hidden_out[:, 2])


if __name__ == "__main__":
    unittest.main()
