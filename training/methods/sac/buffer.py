"""
buffer.py — Replay buffer for SAC.

Stores transitions collected from environment interaction in a fixed-size
circular buffer for off-policy learning.

Each transition contains:

    obs, next_obs, m_act, c_act, t_act, reward, done

The buffer samples random minibatches of past transitions, which is the
standard replay-buffer pattern used by SAC. Unlike PPO rollout buffers, this
does not store log probabilities, advantages, returns, or GAE-related values.
"""

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────
#  Replay Buffer (SAC / off-policy)
# ─────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-size replay buffer for SAC-style off-policy training.

    Transitions are stored in NumPy arrays and sampled uniformly at random.
    When the buffer reaches capacity, new transitions overwrite the oldest
    entries using a circular write pointer.
    """

    def __init__(self, capacity: int, obs_size: int):
        self.capacity = capacity
        self.obs_size = obs_size
        self.pos = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.m_acts = np.zeros(capacity, dtype=np.int64)
        self.c_acts = np.zeros(capacity, dtype=np.int64)
        self.t_acts = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, next_obs, m_act, c_act, t_act, reward, done):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.m_acts[self.pos] = m_act
        self.c_acts[self.pos] = c_act
        self.t_acts[self.pos] = t_act
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "m_acts": torch.from_numpy(self.m_acts[idx]),
            "c_acts": torch.from_numpy(self.c_acts[idx]),
            "t_acts": torch.from_numpy(self.t_acts[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "dones": torch.from_numpy(self.dones[idx]),
        }