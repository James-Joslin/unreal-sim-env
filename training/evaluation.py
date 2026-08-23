"""
evaluation.py — Shared evaluation logic for all training methods.

Every method needs to periodically evaluate its policy against deterministic
scenarios. This module provides that capability without any method-specific
dependencies. It only needs a forward-callable model that produces logits.

DETERMINISTIC EVALUATION
    Each episode gets a fixed seed: base_seed + episode_index.
    Eval at step 10K sees the EXACT same 50 arenas as eval at step 20K.
    The only thing that changes between evals is the model weights,
    so reward/win rate differences reflect genuine improvement, not luck.
"""

from typing import Dict, Optional, Callable

import numpy as np
import torch

from combat_sim import MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_extensions import make_extended_curriculum_env
from frame_stack import FrameStackEnvWrapper


def evaluate(
    model: torch.nn.Module,
    stage: int,
    archetype: str,
    num_episodes: int,
    device: torch.device,
    frame_stack: int = 3,
    obs_normalizer=None,
    base_seed: int = 42,
    is_actor_critic: bool = True,
    behavior_profile=None,
) -> Dict[str, float]:
    """Run evaluation with deterministic scenarios.

    Works with any model that returns logits from forward().
    Set is_actor_critic=True for models returning (m, c, t, value),
    or False for policy-only models returning (m, c, t).
    """
    raw_env = make_extended_curriculum_env(
        stage, archetype,
        behavior_profiles=(behavior_profile,) if behavior_profile is not None else None)
    env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)
    model.eval()

    rewards = []
    lengths = []
    wins = []
    kills = []
    behavior_metrics = []

    for ep_idx in range(num_episodes):
        # Seed the environment's per-instance RNG, not only module globals.
        # This keeps every evaluation checkpoint on the exact same arenas.
        obs, reset_info = env.reset(seed=base_seed + ep_idx)
        condition = None
        if getattr(model, "behavior_conditioned", False):
            condition = torch.from_numpy(
                reset_info["behavior_condition"]).float().unsqueeze(0).to(device)
        ep_reward = 0.0
        ep_length = 0
        done = False
        num_targets = len(raw_env.targets)

        # GRU hidden state (reset per episode).
        hidden = None
        if hasattr(model, 'init_hidden'):
            hidden = model.init_hidden(1, device)

        while not done:
            if obs_normalizer:
                obs_normed = obs_normalizer.normalize(obs)
            else:
                obs_normed = obs

            # Build action mask (same as training — prevents invalid actions).
            mask_dict = raw_env.build_action_mask()

            if mask_dict.get("skip_inference", False):
                # Match production: the frame stack still advances, but ONNX
                # and its GRU are not called while the action lock owns input.
                m = int(np.flatnonzero(mask_dict["m_mask"])[0])
                c = int(np.flatnonzero(mask_dict["c_mask"])[0])
                t = int(np.flatnonzero(mask_dict["t_mask"])[0])
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(
                        obs_normed).float().unsqueeze(0).to(device)

                    if hasattr(model, 'select_actions'):
                        m_mask_t = torch.from_numpy(
                            mask_dict["m_mask"]).unsqueeze(0).to(device)
                        c_mask_t = torch.from_numpy(
                            mask_dict["c_mask"]).unsqueeze(0).to(device)
                        t_mask_t = torch.from_numpy(
                            mask_dict["t_mask"]).unsqueeze(0).to(device)
                        select_kwargs = {}
                        if getattr(model, "behavior_conditioned", False):
                            select_kwargs["behavior_condition"] = condition
                        result = model.select_actions(
                            obs_t, (m_mask_t, c_mask_t, t_mask_t), hidden,
                            **select_kwargs)
                        if len(result) == 4:
                            m_a, c_a, t_a, hidden = result
                        else:
                            m_a, c_a, t_a = result
                        m = m_a.item()
                        c = c_a.item()
                        t = t_a.item()
                    else:
                        outputs = model(obs_t, hidden)
                        m_l, c_l, t_l = outputs[0], outputs[1], outputs[2]
                        if is_actor_critic and len(outputs) > 4:
                            hidden = outputs[4]
                        elif not is_actor_critic and len(outputs) > 3:
                            hidden = outputs[3]
                        m_mask_t = torch.from_numpy(
                            mask_dict["m_mask"]).unsqueeze(0).to(device)
                        c_mask_t = torch.from_numpy(
                            mask_dict["c_mask"]).unsqueeze(0).to(device)
                        t_mask_t = torch.from_numpy(
                            mask_dict["t_mask"]).unsqueeze(0).to(device)
                        m_l = m_l.masked_fill(~m_mask_t, -1e8)
                        c_l = c_l.masked_fill(~c_mask_t, -1e8)
                        t_l = t_l.masked_fill(~t_mask_t, -1e8)
                        m = m_l.argmax(1).item()
                        c = c_l.argmax(1).item()
                        t = t_l.argmax(1).item()

            obs, reward, done, truncated, step_info = env.step(
                np.array([m, c, t]))
            ep_reward += reward
            ep_length += 1
            if truncated:
                break
        if "behavior_metrics" in step_info:
            behavior_metrics.append(step_info["behavior_metrics"])

        targets_killed = sum(1 for t in raw_env.targets if not t.alive)
        is_win = targets_killed == num_targets

        rewards.append(ep_reward)
        lengths.append(ep_length)
        wins.append(float(is_win))
        kills.append(targets_killed)

    env.close()
    stats = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "win_rate": np.mean(wins),
        "mean_kills": np.mean(kills),
        "reward_ci95": 1.96 * np.std(rewards) / max(np.sqrt(len(rewards)), 1),
    }
    if behavior_metrics:
        metric_keys = behavior_metrics[0].keys()
        stats["behavior_metrics"] = {
            key: float(np.mean([metrics[key] for metrics in behavior_metrics]))
            for key in metric_keys
        }

    return stats
