"""
config.py — Discrete SAC hyperparameters.

Key differences from PPO:
    - Off-policy: replay buffer instead of rollout buffer
    - No GAE/advantages: uses soft Bellman backup with twin Q
    - Entropy is a constraint (auto-tuned α), not a loss coefficient
    - More sample-efficient but less stable at scale
"""

from dataclasses import dataclass


@dataclass
class SACConfig:
    # ── Learning rates ───────────────────────────────────────────
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 1e-4           # Entropy temperature LR. Lower than
                                      # actor/critic — α should change slowly.

    # ── Core SAC ─────────────────────────────────────────────────
    gamma: float = 0.99
    tau: float = 0.005                # Polyak averaging for target network.
                                      # θ_target = τ*θ + (1-τ)*θ_target.
    initial_alpha: float = 0.2        # Starting entropy temperature. Auto-tuned
                                      # from here via the α-loss.
    target_entropy_ratio: float = 0.5 # Target entropy as fraction of max.
                                      # Max entropy for a uniform categorical
                                      # over N actions = log(N). We target
                                      # 50% of that — enough exploration without
                                      # being overly random.

    # ── Replay buffer ────────────────────────────────────────────
    buffer_size: int = 1_500_000        # Transitions. At 12 envs × 5 Hz = 60/s,
                                      # 500K = ~2.3 hours of experience.
    batch_size: int = 256
    learning_starts: int = 25_000     # Random actions before training begins.
                                      # Fills the buffer with diverse experience.

    # ── Update frequency ─────────────────────────────────────────
    train_freq: int = 12               # Train every N env steps (across all envs).
                                      # With 12 envs, this means 48 transitions
                                      # between gradient updates.
    gradient_steps: int = 12           # Gradient updates per train call. Higher
                                      # replay ratio = more sample-efficient.

    # ── Evaluation & saving ──────────────────────────────────────
    total_timesteps: int = 6_000_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42

    # ── Normalisation ────────────────────────────────────────────
    normalize_obs: bool = True

    # ── Safety ───────────────────────────────────────────────────
    max_grad_norm: float = 1.0        # Wider than PPO (0.5) — SAC gradients
                                      # are naturally more stable from the
                                      # replay buffer's decorrelated batches.
    revert_on_regression: bool = True
    revert_patience: int = 80
    revert_min_drop: float = 0.15