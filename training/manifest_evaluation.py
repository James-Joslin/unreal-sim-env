"""CPU-friendly, manifest-driven combat policy evaluation."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from behavior_profiles import PROFILE_NAMES, behavior_condition
from combat_extensions import make_extended_curriculum_env
from combat_sim import OBS_SIZE
from frame_stack import FrameStackEnvWrapper
from training.methods.ppo.actor_critic import ActorCritic
from training.normalizers import RunningNormalizer
from training.scenario_manifest import validate_scenario_manifest

CORE_METRICS = ("reward", "win", "kills", "episode_length")
CELL_KEYS = ("stage", "archetype", "weapon_preset", "squad_size_bucket")

class PolicyRunner:
    """Common recurrent inference interface for PyTorch and production ONNX."""
    def __init__(self, path: str):
        self.path = path
        self.conditioned = False
        self.normalizer = None
        if path.endswith(".onnx"):
            import onnxruntime as ort
            self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inputs = {item.name: item for item in self.session.get_inputs()}
            self.frame_stack = int(inputs["observation"].shape[-1]) // OBS_SIZE
            self.hidden_size = int(inputs["hidden_in"].shape[-1])
            self.model = None
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.frame_stack = int(checkpoint.get("frame_stack", 3))
            self.conditioned = (
                checkpoint.get("model_type") == "behavior_conditioned_actor_critic"
                or bool(checkpoint.get("supported_profiles")))
            self.model = ActorCritic(
                obs_size=self.frame_stack * OBS_SIZE,
                tier=checkpoint.get("tier", "large"),
                behavior_conditioned=self.conditioned).eval()
            state = checkpoint.get("full_state_dict", checkpoint.get("model_state_dict"))
            if state is None:
                raise ValueError(f"Checkpoint has no model state: {path}")
            self.model.load_state_dict(state)
            self.hidden_size = self.model.gru_hidden
            if "obs_normalizer" in checkpoint:
                self.normalizer = RunningNormalizer(self.frame_stack * OBS_SIZE)
                self.normalizer.load_state_dict(checkpoint["obs_normalizer"])
            self.session = None

    def initial_hidden(self):
        if self.model is not None:
            return self.model.init_hidden(1, torch.device("cpu"))
        return np.zeros((1, 1, self.hidden_size), dtype=np.float32)

    def logits(self, obs, hidden, profile=None):
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.session is not None:
            outputs = self.session.run(None, {"observation": obs, "hidden_in": hidden})
            return outputs[0][0], outputs[1][0], outputs[2][0], outputs[3]
        condition = None
        if self.conditioned:
            condition = torch.from_numpy(behavior_condition(profile)).reshape(1, -1)
        with torch.no_grad():
            features, hidden_out = self.model._actor_features(
                torch.from_numpy(obs), hidden, condition)
            outputs = self.model._policy_logits(features)
        logits = tuple(output.detach().cpu().numpy()[0] for output in outputs)
        return logits + (hidden_out,)

def _choose(logits, mask, rng, greedy):
    masked = np.where(mask, logits, -1e8)
    if greedy:
        return int(np.argmax(masked))
    shifted = masked - np.max(masked)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))

def run_episode(policy, model_label, scenario, profile, action_seed, greedy):
    raw_env = make_extended_curriculum_env(
        int(scenario["stage"]), str(scenario["archetype"]),
        behavior_profiles=(profile,) if policy.conditioned else None)
    env = FrameStackEnvWrapper(raw_env, frame_stack=policy.frame_stack)
    options = {"weapon_preset": scenario["weapon_preset"]}
    if int(scenario["stage"]) == 7:
        options["squad_size"] = int(scenario["squad_size_bucket"])
    if policy.conditioned:
        options["behavior_profile"] = profile
    try:
        obs, info = env.reset(seed=int(scenario["reset_seed"]), options=options)
        if info["scenario_id"] != scenario["scenario_id"]:
            raise RuntimeError(
                f"Manifest/environment scenario mismatch: {scenario['scenario_id']} "
                f"!= {info['scenario_id']}")
        hidden = policy.initial_hidden()
        rng = np.random.default_rng(action_seed)
        reward_total = 0.0
        terminal_info = info
        done = truncated = False
        while not (done or truncated):
            masks = raw_env.build_action_mask()
            if masks.get("skip_inference", False):
                action = [int(np.flatnonzero(masks[key])[0])
                          for key in ("m_mask", "c_mask", "t_mask")]
            else:
                m_l, c_l, t_l, hidden = policy.logits(obs, hidden, profile)
                action = [
                    _choose(m_l, masks["m_mask"], rng, greedy),
                    _choose(c_l, masks["c_mask"], rng, greedy),
                    _choose(t_l, masks["t_mask"], rng, greedy)]
            obs, reward, done, truncated, terminal_info = env.step(
                np.asarray(action, dtype=np.int64))
            reward_total += float(reward)
        metrics = dict(terminal_info.get("behavior_metrics", {}))
        metrics.update({
            "scenario_id": scenario["scenario_id"],
            "stage": int(scenario["stage"]),
            "archetype": scenario["archetype"],
            "weapon_preset": scenario["weapon_preset"],
            "squad_size_bucket": int(scenario["squad_size_bucket"]),
            "model": model_label,
            "profile": profile,
            "mode": "greedy" if greedy else "stochastic",
            "action_seed": int(action_seed),
            "reward": reward_total,
            "win": float(terminal_info.get("is_win", False)),
            "kills": float(metrics.get("kills", 0.0)),
            "episode_length": int(metrics.get("episode_length", raw_env.step_count)),
        })
        return metrics
    finally:
        env.close()

def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    ci95 = 0.0 if len(values) < 2 else float(
        1.96 * values.std(ddof=1) / math.sqrt(len(values)))
    return {"mean": mean, "ci95": ci95, "count": int(len(values))}

def summarize(episodes):
    grouped = defaultdict(list)
    for row in episodes:
        key = tuple(row[name] for name in ("model", "profile", "mode", *CELL_KEYS))
        grouped[key].append(row)
    cells = []
    for key, rows in sorted(grouped.items()):
        summary = dict(zip(("model", "profile", "mode", *CELL_KEYS), key))
        for metric in CORE_METRICS:
            summary[metric] = _mean_ci([row[metric] for row in rows])
        numeric_sets = [
            {k for k, v in row.items()
             if isinstance(v, (int, float, np.integer, np.floating))}
            for row in rows]
        numeric_behavior = sorted(
            set.intersection(*numeric_sets) - set(CORE_METRICS)
            - {"action_seed", "stage", "squad_size_bucket"})
        summary["behavior_metrics"] = {
            metric: _mean_ci([row[metric] for row in rows])
            for metric in numeric_behavior}
        cells.append(summary)
    paired = []
    profiles = sorted({row["profile"] for row in episodes})
    by_pair_key = defaultdict(dict)
    for row in episodes:
        key = (row["model"], row["scenario_id"], row["mode"], row["action_seed"])
        by_pair_key[key][row["profile"]] = row
    models = sorted({row["model"] for row in episodes})
    for model in models:
        for left_index, left in enumerate(profiles):
            for right in profiles[left_index + 1:]:
                matches = [
                    pair for pair in by_pair_key.values()
                    if left in pair and right in pair
                    and pair[left]["model"] == model
                ]
                if not matches:
                    continue
                entry = {
                    "model": model, "left": left, "right": right,
                    "metrics": {},
                }
                for metric in CORE_METRICS:
                    entry["metrics"][metric] = _mean_ci([
                        pair[right][metric] - pair[left][metric]
                        for pair in matches
                    ])
                paired.append(entry)
    aggregate_groups = defaultdict(list)
    for row in episodes:
        aggregate_groups[(row["model"], row["profile"], row["mode"])].append(row)
    overall = []
    equal_cell = []
    for key, rows in sorted(aggregate_groups.items()):
        identity = dict(zip(("model", "profile", "mode"), key))
        overall.append({
            **identity,
            **{metric: _mean_ci([row[metric] for row in rows])
               for metric in CORE_METRICS},
        })
        matching_cells = [cell for cell in cells
                          if all(cell[name] == identity[name] for name in identity)]
        equal_cell.append({
            **identity,
            **{metric: _mean_ci([cell[metric]["mean"] for cell in matching_cells])
               for metric in CORE_METRICS},
        })
    return {
        "cells": cells, "overall": overall, "equal_cell": equal_cell,
        "paired_profile_deltas": paired,
    }

def evaluate_manifest(manifest, split, model_specs, profiles, action_seeds,
                      include_greedy=True, max_scenarios=None):
    validate_scenario_manifest(manifest)
    scenarios = manifest[split]
    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]
    episodes = []
    for label, path in model_specs:
        policy = PolicyRunner(path)
        policy_profiles = profiles if policy.conditioned else (label,)
        modes = [(True, 0)] if include_greedy else []
        modes += [(False, seed) for seed in action_seeds]
        for profile in policy_profiles:
            for scenario in scenarios:
                for greedy, seed in modes:
                    episodes.append(run_episode(policy, scenario, profile, seed, greedy))
    return episodes, summarize(episodes)

def _parse_model(value):
    if "=" not in value:
        path = Path(value)
        return path.stem, str(path)
    label, path = value.split("=", 1)
    return label, path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="validation")
    parser.add_argument("--model", action="append", required=True,
                        help="Repeatable PATH or LABEL=PATH")
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_NAMES,
                        default=("reactive", "tactical"))
    parser.add_argument("--action_seeds", nargs="*", type=int, default=(0, 1, 2))
    parser.add_argument("--no_greedy", action="store_true")
    parser.add_argument("--max_scenarios", type=int)
    parser.add_argument("--output_dir", default="evaluation/manifest_eval")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    episodes, summary = evaluate_manifest(
        manifest, args.split, [_parse_model(value) for value in args.model],
        tuple(args.profiles), tuple(args.action_seeds), not args.no_greedy,
        args.max_scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps({
        "manifest": str(Path(args.manifest).resolve()), "split": args.split,
        "device": "cpu", "episodes": episodes, "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in episodes for key in row})
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episodes)
    print(f"Wrote {len(episodes)} CPU evaluation episodes to {output_dir}")

if __name__ == "__main__":
    main()
