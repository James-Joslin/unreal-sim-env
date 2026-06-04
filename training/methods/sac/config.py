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
    lr_actor: float = 1e-4            # Lower than critic (3e-4). The critic
                                      # must settle before the actor chases its
                                      # Q-estimates. At equal LR, the actor
                                      # outruns the critic and locks into a
                                      # random policy that the critic validates.
    lr_critic: float = 1.5e-4
    lr_alpha: float = 3e-5           # Very slow. α should change glacially —
                                      # fast α updates cause oscillation and
                                      # the runaway α problem seen at 1e-4.

    # ── Core SAC ─────────────────────────────────────────────────
    gamma: float = 0.99              # Safe at 0.99 now that the α loss sign is
                                      # correct. The Q-value explosion was caused
                                      # by the inverted α, not by γ itself.
    tau: float = 0.005                # Polyak averaging for target network.
    initial_alpha: float = 0.1        # Lower starting α. 0.2 was too much
                                      # entropy for a combat policy that needs
                                      # to be mostly deterministic.
    alpha_max: float = 0.5            # Hard cap on α.
    target_entropy_ratio: float = 0.4 # Raised from 0.3 to give more exploration
                                      # in early curriculum stages. The corrected
                                      # α auto-tuner will pull entropy down as
                                      # the policy converges.

    # ── Reward normalisation ─────────────────────────────────────
    normalize_rewards: bool = True    # Divide rewards by running stdev. Keeps
                                      # Q-targets bounded regardless of raw scale.
    reward_scale: float = 1.0         # No extra scaling beyond normalisation.
                                      # 0.1 compressed the signal too much — kill
                                      # rewards became 0.35, invisible at γ=0.97.

    # ── Replay buffer ────────────────────────────────────────────
    buffer_size: int = 1_000_000
    batch_size: int = 512             # Larger batches stabilise Q-estimates.
    learning_starts: int = 5_000      # Was 25,000 — but stage 1 only has 25K
                                      # total timesteps, so zero learning happened.
                                      # 5K fills enough buffer for diversity while
                                      # leaving room for actual gradient updates.

    # ── Update frequency ─────────────────────────────────────────
    train_freq: int = 8
    gradient_steps: int = 4
    critic_warmup_steps: int = 10_000 # Critic-only training steps before the
                                      # actor starts updating. Lets Q-values
                                      # differentiate between actions before
                                      # the actor chases gradients. Without
                                      # this, the actor immediately pushes
                                      # toward entropy maximisation because
                                      # the flat initial Q-values provide no
                                      # actionable preference signal.

    # ── Evaluation & saving ──────────────────────────────────────
    total_timesteps: int = 6_000_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42

    # ── Normalisation ────────────────────────────────────────────
    normalize_obs: bool = True

    # ── Safety ───────────────────────────────────────────────────
    max_grad_norm: float = 1.0
    q_target_clip: float = 100.0     # Clamp Q-value targets to prevent
                                      # extreme values from propagating
                                      # through the Bellman backup.
    revert_on_regression: bool = True
    revert_patience: int = 80
    revert_min_drop: float = 0.15