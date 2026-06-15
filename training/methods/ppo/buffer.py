"""
buffer.py — Vectorized rollout buffer for PPO.

Stores rollout data from N parallel environments in [num_steps, num_envs, ...]
layout. For minibatch sampling, they're flattened to [total, ...].

This is inherently PPO-specific because it stores log_probs, advantages,
and returns computed via GAE — structures that SAC (replay buffer) or
APPO (async queue) wouldn't use.
"""

import numpy as np
import torch

from combat_sim import MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS


class VecRolloutBuffer:
    """Stores rollout data from N parallel environments.

    Layout: all arrays are [num_steps, num_envs, ...].
    For minibatch sampling, they're flattened to [num_steps * num_envs, ...].
    """

    def __init__(self, num_steps: int, num_envs: int, obs_size: int,
                 gru_hidden: int = 0):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.gru_hidden = gru_hidden
        self.total = num_steps * num_envs

        self.obs = np.zeros(
            (num_steps, num_envs, obs_size), dtype=np.float32)
        self.m_acts = np.zeros(
            (num_steps, num_envs), dtype=np.int64)
        self.c_acts = np.zeros(
            (num_steps, num_envs), dtype=np.int64)
        self.t_acts = np.zeros(
            (num_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros(
            (num_steps, num_envs), dtype=np.float32)
        # Action masks (True = valid).
        self.m_masks = np.ones(
            (num_steps, num_envs, MOVEMENT_ACTIONS), dtype=bool)
        self.c_masks = np.ones(
            (num_steps, num_envs, COMBAT_ACTIONS), dtype=bool)
        self.t_masks = np.ones(
            (num_steps, num_envs, TARGET_ACTIONS), dtype=bool)
        # GRU hidden states (stored for PPO update).
        if gru_hidden > 0:
            self.hiddens = np.zeros(
                (num_steps, num_envs, gru_hidden), dtype=np.float32)

        # Auxiliary prediction labels (target movement direction, 9 classes).
        self.target_move_labels = np.zeros(
            (num_steps, num_envs), dtype=np.int64)

    def compute_gae(self, last_values: np.ndarray, gamma: float,
                    lam: float):
        """Compute GAE for all envs. last_values: (num_envs,)."""
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def flatten(self):
        """Flatten [num_steps, num_envs] → [total] for minibatch sampling."""
        flat = {
            "obs": self.obs.reshape(self.total, self.obs_size),
            "m_acts": self.m_acts.reshape(self.total),
            "c_acts": self.c_acts.reshape(self.total),
            "t_acts": self.t_acts.reshape(self.total),
            "log_probs": self.log_probs.reshape(self.total),
            "advantages": self.advantages.reshape(self.total),
            "returns": self.returns.reshape(self.total),
            "values": self.values.reshape(self.total),
            "m_masks": self.m_masks.reshape(self.total, MOVEMENT_ACTIONS),
            "c_masks": self.c_masks.reshape(self.total, COMBAT_ACTIONS),
            "t_masks": self.t_masks.reshape(self.total, TARGET_ACTIONS),
            "target_move_labels": self.target_move_labels.reshape(self.total),
        }
        if self.gru_hidden > 0:
            flat["hiddens"] = self.hiddens.reshape(self.total, self.gru_hidden)
        return flat

    def sample_minibatches(self, batch_size: int):
        flat = self.flatten()
        indices = np.random.permutation(self.total)

        for start in range(0, self.total, batch_size):
            end = start + batch_size
            if end > self.total:
                break
            idx = indices[start:end]
            batch = {
                "obs": torch.from_numpy(flat["obs"][idx]),
                "m_acts": torch.from_numpy(flat["m_acts"][idx]),
                "c_acts": torch.from_numpy(flat["c_acts"][idx]),
                "t_acts": torch.from_numpy(flat["t_acts"][idx]),
                "old_log_probs": torch.from_numpy(flat["log_probs"][idx]),
                "advantages": torch.from_numpy(flat["advantages"][idx]),
                "returns": torch.from_numpy(flat["returns"][idx]),
                "old_values": torch.from_numpy(flat["values"][idx]),
                "m_masks": torch.from_numpy(flat["m_masks"][idx]),
                "c_masks": torch.from_numpy(flat["c_masks"][idx]),
                "t_masks": torch.from_numpy(flat["t_masks"][idx]),
                "target_move_labels": torch.from_numpy(
                    flat["target_move_labels"][idx]),
            }
            # Pass stored GRU hidden states so evaluate_actions uses
            # the same hidden context as the rollout collection phase.
            if "hiddens" in flat:
                batch["hiddens"] = torch.from_numpy(flat["hiddens"][idx])
            yield batch