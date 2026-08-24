"""Regression tests for manifest-driven evaluation aggregation."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "simulation"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from training.manifest_evaluation import summarize
from training.scenario_manifest import generate_scenario_manifest


class ManifestEvaluationTests(unittest.TestCase):
    def test_non_stage7_manifest_identity_uses_executed_single_actor_bucket(self):
        manifest = generate_scenario_manifest(
            stages=(1,), archetypes=("ranged",), loadouts=("scout",),
            scenarios_per_cell=3)
        rows = manifest["train"] + manifest["validation"] + manifest["test"]
        self.assertTrue(all(row["squad_size_bucket"] == 1 for row in rows))
        self.assertTrue(all(row["scenario_id"].endswith("squad1") for row in rows))

    def test_summary_reports_cells_and_same_scenario_paired_deltas(self):
        common = {
            "model": "teacher",
            "scenario_id": "scenario-1", "mode": "greedy", "action_seed": 0,
            "stage": 4, "archetype": "ranged", "weapon_preset": "heavy",
            "squad_size_bucket": 1, "kills": 1.0, "episode_length": 20,
            "credible_fire_conversion": 0.5,
        }
        episodes = [
            {**common, "profile": "reactive", "reward": 2.0, "win": 0.0},
            {**common, "profile": "tactical", "reward": 5.0, "win": 1.0},
        ]
        result = summarize(episodes)
        self.assertEqual(len(result["cells"]), 2)
        delta = result["paired_profile_deltas"][0]
        self.assertEqual((delta["left"], delta["right"]),
                         ("reactive", "tactical"))
        self.assertEqual(delta["metrics"]["reward"]["mean"], 3.0)
        self.assertEqual(delta["metrics"]["win"]["mean"], 1.0)
        self.assertEqual(delta["metrics"]["reward"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
