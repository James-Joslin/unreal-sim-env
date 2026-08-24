"""Deterministic scenario manifests and capability-observability audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from combat_sim import WEAPON_PRESETS


# Keep manifest generation independent of PyTorch/model imports so mechanics
# and split audits can run in lightweight simulation environments.
DEFAULT_ARCHETYPES = ("ranged", "melee", "tank")


@dataclass(frozen=True)
class ScenarioIdentity:
    stage: int
    archetype: str
    reset_seed: int
    weapon_preset: str
    squad_size_bucket: int

    @property
    def scenario_id(self):
        return (
            f"s{self.stage}:a{self.archetype}:seed{self.reset_seed}:"
            f"loadout{self.weapon_preset}:squad{self.squad_size_bucket}")


def generate_scenario_manifest(
        stages=range(1, 8),
        archetypes=DEFAULT_ARCHETYPES,
        loadouts=tuple(WEAPON_PRESETS),
        scenarios_per_cell=20,
        validation_fraction=0.15,
        test_fraction=0.15):
    """Split within each Stage/Loadout/Archetype/Squad cell.

    Behavior profile is deliberately absent from ScenarioIdentity so every
    profile reuses exactly the same comparison scenarios.
    """
    if scenarios_per_cell < 3:
        raise ValueError("scenarios_per_cell must be at least 3")
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("validation and test fractions must be positive")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation + test fraction must be below one")

    manifest = {"schema": "behavior_scenarios_v1",
                "train": [], "validation": [], "test": []}
    for stage in stages:
        squads = (1, 2, 3, 4) if int(stage) == 7 else (1,)
        for archetype in archetypes:
            for loadout in loadouts:
                for squad in squads:
                    val_count = max(1, round(
                        scenarios_per_cell * validation_fraction))
                    test_count = max(1, round(
                        scenarios_per_cell * test_fraction))
                    if val_count + test_count >= scenarios_per_cell:
                        raise ValueError("Not enough scenarios for three splits")
                    for cell_index in range(scenarios_per_cell):
                        seed = (int(stage) * 10_000_000
                                + list(archetypes).index(archetype) * 1_000_000
                                + list(loadouts).index(loadout) * 100_000
                                + int(squad) * 10_000 + cell_index)
                        scenario = ScenarioIdentity(
                            int(stage), str(archetype), seed,
                            str(loadout), int(squad))
                        if cell_index < test_count:
                            split = "test"
                        elif cell_index < test_count + val_count:
                            split = "validation"
                        else:
                            split = "train"
                        row = asdict(scenario)
                        row["scenario_id"] = scenario.scenario_id
                        manifest[split].append(row)
    validate_scenario_manifest(manifest)
    return manifest


def validate_scenario_manifest(manifest):
    seen = {}
    for split in ("train", "validation", "test"):
        for row in manifest[split]:
            scenario_id = row["scenario_id"]
            if scenario_id in seen:
                raise ValueError(
                    f"Scenario {scenario_id} crosses {seen[scenario_id]}/{split}")
            seen[scenario_id] = split
    return True


def capability_observability_collisions():
    """Find ready-state loadouts that look identical but need different fire logic."""
    visible_groups = {}
    for preset_name, preset in WEAPON_PRESETS.items():
        for slot_index, slot in enumerate(preset.slots):
            visible = (
                slot_index,
                slot.get("weapon_range", 1500.0),
                slot.get("fire_cooldown", 0.3),
                slot.get("wind_up_time", 0.0),
                bool(slot.get("can_arc", False)),
            )
            hidden = (
                slot.get("projectile_speed", 3000.0),
                slot.get("projectile_lifetime", 10.0),
                slot.get("impact_radius", 0.0),
                slot.get("damage_falloff", 0.5),
                slot.get("optimal_min", 600.0),
                slot.get("optimal_max", 1200.0),
                slot.get("base_damage", 10.0),
                slot.get("reload_time", 2.0),
            )
            visible_groups.setdefault(visible, []).append(
                (preset_name, slot_index, hidden))
    return [
        group for group in visible_groups.values()
        if len({item[2] for item in group}) > 1
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scenario_manifest_v1.json")
    parser.add_argument("--scenarios_per_cell", type=int, default=20)
    args = parser.parse_args()
    manifest = generate_scenario_manifest(
        scenarios_per_cell=args.scenarios_per_cell)
    Path(args.output).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {sum(len(manifest[s]) for s in ('train', 'validation', 'test'))} "
          f"scenarios to {args.output}")
    collisions = capability_observability_collisions()
    print(f"Hidden-capability observation collisions: {len(collisions)}")


if __name__ == "__main__":
    main()
