"""
normalizers.py — Observation and return normalizers shared across all training methods.

Both normalizers are method-agnostic. Any on-policy or off-policy algorithm
benefits from normalised inputs (prevents feature distribution shift at
curriculum transitions) and normalised returns (adapts to reward scale changes).

These are serialised into checkpoints so training can resume with consistent
statistics, and the obs normalizer is baked into the ONNX graph at export time.
"""

import numpy as np


class RunningNormalizer:
    """Welford's online algorithm for running mean and variance.

    WHY THIS MATTERS
        In stages 1-2, spatial features (obstacle ring, cover ring, navmesh)
        are all constant (no obstacles → all 1.0). When stage 3 adds
        obstacles, these 25 features suddenly vary. The first linear layer
        has calibrated its weights for those inputs being constant. Without
        normalization, the activation distribution shifts violently and
        destabilises the whole network.

        Running normalization absorbs these shifts — the normalizer
        adapts its statistics over ~1000 steps, so the network sees
        gradually changing normalised inputs rather than a sudden cliff.

    USAGE
        normalizer = RunningNormalizer(obs_size)
        normalizer.update(obs_batch)                  # (N, obs_size)
        normed = normalizer.normalize(obs_batch)      # zero-mean, unit-variance
    """

    def __init__(self, shape: int, clip: float = 5.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Small initial count to avoid division by zero.
        self.clip = clip
        self.epsilon = epsilon

    def update(self, batch: np.ndarray):
        """Update running statistics with a batch of observations."""
        batch = batch.reshape(-1, self.mean.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        # Welford's parallel algorithm.
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to roughly zero mean, unit variance."""
        return np.clip(
            (obs - self.mean.astype(np.float32)) /
            np.sqrt(self.var.astype(np.float32) + self.epsilon),
            -self.clip, self.clip
        )

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


class ReturnNormalizer:
    """Running estimate of return variance for reward scaling.

    Divides rewards by sqrt(running_var(returns)) so the value function
    always sees roughly unit-variance targets. This prevents the value
    head from being overwhelmed when reward scale jumps between stages.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_return = 0.0
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4

    def update(self, rewards: np.ndarray, dones: np.ndarray):
        """Update with a batch of (rewards, dones) from one timestep."""
        for r, d in zip(rewards, dones):
            self.running_return = r + self.gamma * self.running_return * (1 - d)
            self.count += 1
            # Welford's online algorithm for mean and variance.
            delta = self.running_return - self.running_mean
            self.running_mean += delta / self.count
            delta2 = self.running_return - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / self.count

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return rewards / (np.sqrt(max(self.running_var, 1e-6)) + self.epsilon)
