from dataclasses import dataclass
@dataclass
class PPOConfig:
    lr: float = 2.5e-4              # ↑ from 1.2e-4 — headroom for larger updates
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.18        # ↑ from 0.08 — let the policy actually move
    vf_clip_range: float = 10.0

    entropy_coef: float = 0.01      # Slight reduction — exploration is fine (64% of max)
    entropy_coef_final: float = 0.002
    value_coef: float = 0.35

    max_grad_norm: float = 0.5      # ↑ from 0.3 — standard safe ceiling

    num_steps: int = 1024            # ↑ from 196 — ~56 episodes/rollout (8 envs)
    mini_batch_size: int = 512
    update_epochs: int = 5          # ↑ from 3 — extract more from each rollout
    target_kl: float = 0.02         # ↑ from 0.012 — allow larger policy shifts

    total_timesteps: int = 9_000_000
    eval_interval: int = 20_000
    save_interval: int = 100_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42
    normalize_obs: bool = True
    normalize_returns: bool = True

    revert_on_regression: bool = True
    revert_patience: int = 20
    revert_min_drop: float = 0.15
    
    
"""
ppo_stage_configs.py — Per-stage PPO hyperparameters.

Each stage has fundamentally different signal-to-noise characteristics.
The key insight from S4/S5 training runs:

    S4 (1-2 targets): entropy=0.016, policy_loss=0.036 → ratio 1.00 → climbed to 84%
    S5 (1-3 targets): entropy=0.016, policy_loss=0.038 → ratio 1.37 → stuck at 56%

The entropy bonus scales with action-space uncertainty, which grows with
target count. Without per-stage tuning, later stages have entropy dominating
the policy gradient, preventing the model from committing to strategies.

PARAMETER DESIGN RATIONALE

    entropy_coef    Decreases with stage. Early stages need broad exploration;
                    later stages need the policy gradient to dominate so the
                    model commits to coordination tactics. Target: entropy
                    term / policy loss magnitude ≈ 0.7-1.0.

    num_steps       Increases with stage. More targets = noisier episodes =
                    need more complete episodes per rollout for clean GAE
                    advantage estimates. At 8 envs:
                      256 steps → ~28 episodes/rollout (1 target, short)
                      1024 steps → ~70 episodes/rollout (multi-target)
                      2048 steps → ~100 episodes/rollout (full squad)

    clip_range      Decreases with stage. Early stages benefit from large
                    policy jumps (fast learning). Later stages need tighter
                    bounds to prevent catastrophic updates that destroy
                    hard-won coordination strategies.

    target_kl       Decreases with stage. Tighter early stopping on later
                    stages prevents oversized updates that destabilise
                    multi-entity strategies.

    eval_episodes   Increases with stage. More targets = higher outcome
                    variance = need more episodes for reliable win rate
                    estimates. At 50 episodes, SE is ±12pp at 65% WR.
                    At 150 episodes, SE is ±7pp.

    lr              Decreases with stage. Later stages fine-tune an already
                    partially-trained policy; smaller LR prevents
                    overshooting.

    update_epochs   Increases with num_steps. Larger rollouts contain more
                    information; more epochs extract it without re-collecting.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PPOStageConfig:
    """PPO hyperparameters for a single curriculum stage."""

    # ── Learning ─────────────────────────────────────────────────
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.18
    vf_clip_range: float = 10.0

    # ── Exploration / Exploitation Balance ────────────────────────
    entropy_coef: float = 0.020
    entropy_coef_final: float = 0.002
    value_coef: float = 0.35

    # ── Gradient Safety ──────────────────────────────────────────
    max_grad_norm: float = 0.5

    # ── Rollout & Update ─────────────────────────────────────────
    num_steps: int = 512
    mini_batch_size: int = 512
    update_epochs: int = 5
    target_kl: float = 0.02

    # ── Horizons ─────────────────────────────────────────────────
    total_timesteps: int = 500_000
    eval_interval: int = 20_000
    save_interval: int = 100_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42

    # ── Normalisation ────────────────────────────────────────────
    normalize_obs: bool = True
    normalize_returns: bool = True

    # ── Reversion ────────────────────────────────────────────────
    revert_on_regression: bool = True
    revert_patience: int = 4
    revert_min_drop: float = 0.15


# ═════════════════════════════════════════════════════════════════
#  Stage Definitions
# ═════════════════════════════════════════════════════════════════

STAGE_CONFIGS: Dict[int, PPOStageConfig] = {

    # ─────────────────────────────────────────────────────────────
    #  Stage 1 — Melee Basics
    #  1 passive target, no obstacles. Learn: approach → attack.
    #  Broad exploration, fast updates, short episodes.
    # ─────────────────────────────────────────────────────────────
    1: PPOStageConfig(
        lr=3.0e-4,
        clip_range=0.20,
        entropy_coef=0.025,
        entropy_coef_final=0.005,
        num_steps=256,
        update_epochs=4,
        target_kl=0.025,
        total_timesteps=200_000,
        num_eval_episodes=30,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 2 — Ranged Fire & Reload
    #  1 stationary target. Learn: fire/reload cycle, range.
    #  Still exploratory, slightly longer rollouts for reload timing.
    # ─────────────────────────────────────────────────────────────
    2: PPOStageConfig(
        lr=3.0e-4,
        clip_range=0.20,
        entropy_coef=0.025,
        entropy_coef_final=0.004,
        num_steps=256,
        update_epochs=4,
        target_kl=0.025,
        total_timesteps=300_000,
        num_eval_episodes=30,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 3 — Moving Targets, Cover, Flanking
    #  1-2 targets, 3 obstacles, 1000 max steps (200s).
    #  First real combat stage. Longer episodes need bigger rollouts.
    # ─────────────────────────────────────────────────────────────
    3: PPOStageConfig(
        lr=2.5e-4,
        clip_range=0.18,
        entropy_coef=0.020,
        entropy_coef_final=0.003,
        num_steps=512,
        update_epochs=5,
        target_kl=0.020,
        total_timesteps=500_000,
        num_eval_episodes=50,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 4 — Multi-Weapon Management
    #  1-2 targets, 8 obstacles, heavy weapon kit.
    #  Weapon switching adds discrete decision complexity.
    #  S4 data showed: healthy KL=0.010, clip=15%, climbed to 84%.
    #  These params are validated by that run.
    # ─────────────────────────────────────────────────────────────
    4: PPOStageConfig(
        lr=2.5e-4,
        clip_range=0.18,
        entropy_coef=0.018,
        entropy_coef_final=0.003,
        num_steps=512,
        update_epochs=5,
        target_kl=0.018,
        total_timesteps=500_000,
        num_eval_episodes=80,
        revert_patience=5,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 5 — Archetype Behaviours & Allies
    #  1-3 targets, 1 ally, weapon pool randomisation.
    #  CRITICAL: entropy_coef halved from S4.
    #
    #  S5 data showed entropy/policy ratio of 1.37 (entropy wins),
    #  causing the policy to drift toward randomness instead of
    #  committing to multi-target strategies. Cutting entropy_coef
    #  to 0.010 brings the ratio to ~0.8.
    #
    #  num_steps doubled (512→1024): multi-target episodes have
    #  higher outcome variance; more episodes per rollout clean up
    #  the advantage estimates.
    # ─────────────────────────────────────────────────────────────
    5: PPOStageConfig(
        lr=2.0e-4,
        clip_range=0.16,
        entropy_coef=0.010,
        entropy_coef_final=0.0005,
        num_steps=1024,
        update_epochs=5,
        target_kl=0.015,
        total_timesteps=500_000,
        num_eval_episodes=100,
        revert_patience=5,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 6 — Multi-Target Coordination
    #  1-3 targets, 1 ally, 12 obstacles. Coordination rewards.
    #  Policy must commit to focus-fire and target-switching rules.
    #  Entropy further reduced; rollout stays large.
    # ─────────────────────────────────────────────────────────────
    6: PPOStageConfig(
        lr=2.0e-4,
        clip_range=0.15,
        entropy_coef=0.008,
        entropy_coef_final=0.0005,
        num_steps=1024,
        update_epochs=6,
        target_kl=0.012,
        total_timesteps=1_000_000,
        num_eval_episodes=120,
        revert_patience=6,
        revert_min_drop=0.12,
    ),

    # ─────────────────────────────────────────────────────────────
    #  Stage 7 — Full Squad Combat
    #  1-4 targets (full player party), 1 ally, 16 obstacles.
    #  Boss-tier HP (500). Maximum complexity.
    #
    #  Largest rollouts (2048) for cleanest advantage signal.
    #  Tightest clip/KL to protect the complex strategy the model
    #  has built across 6 prior stages. Lowest entropy to force
    #  full exploitation of learned tactics.
    #
    #  Longer training budget (2M) with more frequent eval and
    #  tighter reversion to catch regressions early.
    # ─────────────────────────────────────────────────────────────
    7: PPOStageConfig(
        lr=1.5e-4,
        clip_range=0.14,
        entropy_coef=0.006,
        entropy_coef_final=0.0003,
        num_steps=2048,
        mini_batch_size=1024,
        update_epochs=6,
        target_kl=0.010,
        total_timesteps=2_000_000,
        eval_interval=15_000,
        num_eval_episodes=150,
        revert_patience=6,
        revert_min_drop=0.10,
    ),
}


# ═════════════════════════════════════════════════════════════════
#  Accessor
# ═════════════════════════════════════════════════════════════════

def get_stage_config(stage: int) -> PPOStageConfig:
    """Get hyperparameters for a curriculum stage.

    Stages 1-3 use broad exploration with small rollouts.
    Stages 4-5 balance exploration/exploitation with medium rollouts.
    Stages 6-7 prioritise exploitation with large rollouts.

    Falls back to stage 4 config for unknown stages.
    """
    if stage in STAGE_CONFIGS:
        return STAGE_CONFIGS[stage]

    # Fallback: use stage 4 as a sensible default.
    print(f"[PPOStageConfig] No config for stage {stage}, using stage 4 defaults")
    return STAGE_CONFIGS[4]


def get_stage_summary(stage: int) -> str:
    """One-line summary for logging."""
    cfg = get_stage_config(stage)
    return (
        f"S{stage}: lr={cfg.lr:.1e} clip={cfg.clip_range:.2f} "
        f"ent={cfg.entropy_coef:.3f}→{cfg.entropy_coef_final:.3f} "
        f"steps={cfg.num_steps} epochs={cfg.update_epochs} "
        f"kl={cfg.target_kl:.3f} eval_eps={cfg.num_eval_episodes}"
    )


# ═════════════════════════════════════════════════════════════════
#  Quick Reference (printed when run directly)
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("PPO Stage Configs — Quick Reference\n")
    print(f"{'Stage':<8s} {'LR':>8s} {'Clip':>6s} {'Ent':>7s} {'→Final':>7s} "
          f"{'Steps':>6s} {'Epochs':>7s} {'KL':>6s} {'Eval':>5s} {'Budget':>8s}")
    print("─" * 78)
    for stage in sorted(STAGE_CONFIGS.keys()):
        c = STAGE_CONFIGS[stage]
        print(f"  {stage:<6d} {c.lr:>8.1e} {c.clip_range:>6.2f} {c.entropy_coef:>7.3f} "
              f"{c.entropy_coef_final:>7.3f} {c.num_steps:>6d} {c.update_epochs:>7d} "
              f"{c.target_kl:>6.3f} {c.num_eval_episodes:>5d} {c.total_timesteps:>8,d}")