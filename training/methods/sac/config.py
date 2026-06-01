"""
config.py — SAC hyperparameters.

All SAC-specific tuning lives here.
"""

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────
#  SAC Config
# ─────────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4        # Entropy temperature learning rate.
    gamma: float = 0.99
    tau: float = 0.005             # Soft target update rate.
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 10_000  # Random actions before training.
    train_freq: int = 1            # Train every N env steps.
    gradient_steps: int = 1        # Gradient updates per train call.
    target_entropy_ratio: float = 0.5  # Target entropy as ratio of max.
    eval_interval: int = 10_000
    save_interval: int = 50_000
    num_eval_episodes: int = 50
    normalize_obs: bool = True
    total_timesteps: int = 6_000_000

