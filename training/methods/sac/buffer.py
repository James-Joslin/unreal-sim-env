"""
buffer.py — Replay buffer for off-policy SAC training.

Unlike PPO's rollout buffer (which stores full trajectories and computes
GAE), the replay buffer stores individual transitions and samples them
independently. This decorrelation is key to SAC's stability.

Supports batch adds from vectorised environments.
"""

import numpy as np
import torch

from combat_sim import MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS


class ReplayBuffer:
    """Fixed-size replay buffer with batch add support."""

    def __init__(self, capacity: int, obs_size: int):
        self.capacity = capacity
        self.obs_size = obs_size
        self.pos = 0
        self.size = 0

        # Transitions: (s, a, r, s', done, masks).
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.m_acts = np.zeros(capacity, dtype=np.int64)
        self.c_acts = np.zeros(capacity, dtype=np.int64)
        self.t_acts = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # Store action masks for the NEXT state (used in target computation).
        self.next_m_masks = np.ones((capacity, MOVEMENT_ACTIONS), dtype=bool)
        self.next_c_masks = np.ones((capacity, COMBAT_ACTIONS), dtype=bool)
        self.next_t_masks = np.ones((capacity, TARGET_ACTIONS), dtype=bool)

    def add(self, obs, next_obs, m_act, c_act, t_act, reward, done,
            next_m_mask=None, next_c_mask=None, next_t_mask=None):
        """Add a single transition."""
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.m_acts[self.pos] = m_act
        self.c_acts[self.pos] = c_act
        self.t_acts[self.pos] = t_act
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        if next_m_mask is not None:
            self.next_m_masks[self.pos] = next_m_mask
        if next_c_mask is not None:
            self.next_c_masks[self.pos] = next_c_mask
        if next_t_mask is not None:
            self.next_t_masks[self.pos] = next_t_mask

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, next_obs, m_acts, c_acts, t_acts,
                  rewards, dones,
                  next_m_masks=None, next_c_masks=None, next_t_masks=None):
        """Add a batch of transitions from vectorised environments.

        All inputs are [num_envs, ...].
        """
        batch = obs.shape[0]
        for i in range(batch):
            self.add(
                obs[i], next_obs[i],
                m_acts[i], c_acts[i], t_acts[i],
                rewards[i], dones[i],
                next_m_masks[i] if next_m_masks is not None else None,
                next_c_masks[i] if next_c_masks is not None else None,
                next_t_masks[i] if next_t_masks is not None else None,
            )

    def sample(self, batch_size: int) -> dict:
        """Sample a random batch of transitions."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "m_acts": torch.from_numpy(self.m_acts[idx]),
            "c_acts": torch.from_numpy(self.c_acts[idx]),
            "t_acts": torch.from_numpy(self.t_acts[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
            "next_m_masks": torch.from_numpy(self.next_m_masks[idx]),
            "next_c_masks": torch.from_numpy(self.next_c_masks[idx]),
            "next_t_masks": torch.from_numpy(self.next_t_masks[idx]),
        }

    def __len__(self):
        return self.size