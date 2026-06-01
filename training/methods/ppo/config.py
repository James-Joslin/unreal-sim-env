"""
config.py — PPO hyperparameters.

All PPO-specific tuning lives here.
"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.06
    vf_clip_range: float = 10.0     # Value function clipping — prevents
                                     # huge value updates at stage transitions.
    entropy_coef: float = 0.01       # Start of entropy anneal range.
    entropy_coef_final: float = 0.002 # Entropy decays to this. Near-zero late
                                      # in training lets the policy converge.
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 196             # Steps per env per rollout.
                                     # Total transitions = num_steps × num_envs.
    mini_batch_size: int = 256
    update_epochs: int = 4           # Reduced — action masking makes each step
                                     # more informative, fewer passes needed.
    target_kl: float = 0.015         # KL early stopping. If approx KL exceeds
                                     # this, stop the epoch loop early.
    total_timesteps: int = 6_000_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42
    normalize_obs: bool = True
    normalize_returns: bool = True
    revert_on_regression: bool = True  # Revert on CATASTROPHIC regression only.
    revert_patience: int = 80          # ~80 evals without improvement before reverting.
    revert_min_drop: float = 0.15      # Only revert if win rate dropped >15%.
