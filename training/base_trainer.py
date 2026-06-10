"""
base_trainer.py — Abstract base class for all RL training methods.

Every method (PPO, SAC, APPO, etc.) subclasses BaseTrainer and implements
the method-specific pieces. The shared infrastructure — env creation,
curriculum progression, evaluation scheduling, checkpointing format,
TensorBoard logging, and policy extraction for ONNX — lives here.

WHAT A METHOD MUST IMPLEMENT
    build_model()           → returns the method-specific nn.Module
    train()                 → runs the full training loop
    extract_policy()        → returns a CombatPolicy for ONNX export
    default_curriculum_timesteps()  → per-stage training budgets

WHAT A METHOD GETS FOR FREE
    - Vectorized environment with frame stacking (VecFrameStackEnv)
    - Observation normalization (RunningNormalizer)
    - Deterministic evaluation with seeded scenarios
    - Curriculum runner (calls train() per stage, chains checkpoints)
    - TensorBoard writer (pre-configured)
    - Checkpoint save/load with consistent format
"""

import os
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_extensions import make_extended_curriculum_env
from combat_policy import (
    CombatPolicy, make_policy, save_checkpoint,
    load_teacher_from_checkpoint, TIER_CONFIGS,
)
from frame_stack import (
    FrameStackEnvWrapper, VecFrameStackEnv,
    stacked_obs_size, SINGLE_OBS_SIZE,
)

from .normalizers import RunningNormalizer
from .evaluation import evaluate


DEFAULT_FRAME_STACK = 3
DEFAULT_NUM_ENVS = 12


class BaseTrainer(ABC):
    """Abstract base for all RL training methods.

    Provides shared infrastructure. Methods override the abstract methods
    to plug in their specific model, update rule, and buffer.
    """

    # ─── Class-level identifier (override in subclass) ───────────
    method_name: str = "base"

    def __init__(
        self,
        stage: int = 3,
        archetype: str = "ranged",
        tier: str = "large",
        frame_stack: int = DEFAULT_FRAME_STACK,
        num_envs: int = DEFAULT_NUM_ENVS,
        bc_checkpoint: Optional[str] = None,
        output_dir: str = "checkpoints",
        total_timesteps: int = 6_000_000,
    ):
        self.stage = stage
        self.archetype = archetype
        self.tier = tier
        self.frame_stack = frame_stack
        self.num_envs = num_envs
        self.bc_checkpoint = bc_checkpoint
        self.output_dir = output_dir
        self.total_timesteps = total_timesteps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = stacked_obs_size(frame_stack)

        os.makedirs(output_dir, exist_ok=True)

        # TensorBoard.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = (f"runs/{self.method_name}_s{stage}_{archetype}_{timestamp}")
        self.writer = SummaryWriter(self.log_dir)

        # Model and optimizer (populated by subclass in build_model).
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Observation normalizer (shared across methods).
        self.obs_normalizer: Optional[RunningNormalizer] = None

        # Evaluation tracking.
        self.eval_history = []  # (step, reward, win_rate)
        self.best_eval_win_rate = -1.0
        self.best_eval_reward = float("-inf")

    # ═════════════════════════════════════════════════════════════
    #  ABSTRACT — must be implemented by each method
    # ═════════════════════════════════════════════════════════════

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Create the method-specific model (e.g. ActorCritic for PPO).

        Must call self.model = ... and self.optimizer = ... before returning.
        Called once at the start of train().
        """
        ...

    @abstractmethod
    def train(self):
        """Run the full training loop for self.total_timesteps.

        Should use self.create_vec_env() for environment creation,
        self.run_eval() for periodic evaluation, and
        self.save_checkpoint() for periodic saves.
        """
        ...

    @abstractmethod
    def extract_policy(self) -> CombatPolicy:
        """Extract a CombatPolicy (inference-only, no critic) from the trained model.

        This is the model that gets distilled and exported to ONNX.
        Must map training-model weights → CombatPolicy weights:
            actor_encoder → encoder
            actor_backbone → backbone
            move_head, combat_head, target_head → same names
        """
        ...

    def default_curriculum_timesteps(self) -> Dict[int, int]:
        """Per-stage training budgets. Override for method-specific schedules."""
        return {
            1: 50_000,        # 1v1 no obstacles — learn to approach and shoot
            2: 100_000,       # 1v1 with obstacles — learn navigation
            3: 1_500_000,       # 1v2 — learn target switching
            4: 2_000_000,     # 1v2 with obstacles, 2x HP — cover + weapon mgmt
            5: 10_000_000,    # 2v3 — multi-target, ally coordination
            6: 20_000_000,    # 2v3 full arena — complex navigation
            7: 30_000_000,    # 2v4 full arena — everything together
        }

    # ═════════════════════════════════════════════════════════════
    #  SHARED INFRASTRUCTURE — used by all methods
    # ═════════════════════════════════════════════════════════════

    def create_vec_env(self) -> VecFrameStackEnv:
        """Create a vectorized environment with frame stacking."""
        env_fns = [
            lambda s=self.stage, a=self.archetype: make_extended_curriculum_env(s, a)
            for _ in range(self.num_envs)
        ]
        return VecFrameStackEnv(env_fns, frame_stack=self.frame_stack)

    def extract_masks(self, infos_list):
        """Extract action masks from info dicts into batched tensors."""
        m = np.stack([
            info.get("action_mask", {}).get(
                "m_mask", np.ones(MOVEMENT_ACTIONS, dtype=bool))
            for info in infos_list
        ])
        c = np.stack([
            info.get("action_mask", {}).get(
                "c_mask", np.ones(COMBAT_ACTIONS, dtype=bool))
            for info in infos_list
        ])
        t = np.stack([
            info.get("action_mask", {}).get(
                "t_mask", np.ones(TARGET_ACTIONS, dtype=bool))
            for info in infos_list
        ])
        return (
            torch.from_numpy(m).to(self.device),
            torch.from_numpy(c).to(self.device),
            torch.from_numpy(t).to(self.device),
        )

    def run_eval(self, global_step: int, eval_episodes: int = 50,
                 eval_base_seed: int = 42, batch_total: int = 0) -> Dict:
        """Run evaluation and log results. Returns eval stats dict."""
        eval_stats = evaluate(
            self.model, self.stage, self.archetype,
            eval_episodes, self.device,
            self.frame_stack, self.obs_normalizer,
            base_seed=eval_base_seed,
            is_actor_critic=True,
        )

        self.writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], global_step)
        self.writer.add_scalar("eval/std_reward", eval_stats["std_reward"], global_step)
        self.writer.add_scalar("eval/win_rate", eval_stats["win_rate"], global_step)
        self.writer.add_scalar("eval/mean_kills", eval_stats["mean_kills"], global_step)
        self.writer.add_scalar("eval/mean_length", eval_stats["mean_length"], global_step)

        # Live correlation plot.
        self.eval_history.append(
            (global_step, eval_stats["mean_reward"], eval_stats["win_rate"]))

        if len(self.eval_history) >= 3:
            self._plot_eval_correlation(global_step)

        self.writer.flush()

        print(f"  Eval ({eval_episodes} ep, seeded): "
              f"reward={eval_stats['mean_reward']:.1f} "
              f"±{eval_stats['reward_ci95']:.1f}, "
              f"win={eval_stats['win_rate']:.0%}, "
              f"kills={eval_stats['mean_kills']:.1f}, "
              f"len={eval_stats['mean_length']:.0f}")

        return eval_stats

    def check_best_model(self, eval_stats: Dict, global_step: int) -> bool:
        """Check if current eval is best. Saves checkpoint if so.

        Returns True if a new best was saved.

        Selection rules:
          1. Win rate improved by >1% → always save (reward irrelevant)
          2. Win rate roughly equal (±1%) but reward improved by >5
             → save (same wins, better play quality)
          3. Otherwise → don't save
        """
        wr = eval_stats["win_rate"]
        mr = eval_stats["mean_reward"]

        improved = False
        reason = ""

        if wr > self.best_eval_win_rate + 0.01:
            improved = True
            reason = f"win rate: {self.best_eval_win_rate:.0%} → {wr:.0%}"
        elif (wr >= self.best_eval_win_rate - 0.01
                and mr > self.best_eval_reward + 5.0):
            improved = True
            reason = (f"same win rate ({wr:.0%}), "
                      f"reward: {self.best_eval_reward:.1f} → {mr:.1f}")

        if improved:
            self.best_eval_win_rate = wr
            self.best_eval_reward = mr
            path = os.path.join(
                self.output_dir,
                f"{self.method_name}_stage{self.stage}_best.pt")
            self.save_checkpoint(path, global_step)
            print(f"  → New best model saved ({reason})")
            return True

        return False

    def save_checkpoint(self, path: str, global_step: int):
        """Save checkpoint in the shared format all methods use.

        The checkpoint contains:
            - full_state_dict: complete model (method-specific, for resume)
            - policy_state_dict: actor-only mapped to CombatPolicy keys (for distill/export)
            - optimizer_state_dict: for training resume
            - metadata: tier, stage, archetype, step, frame_stack, method
            - obs_normalizer: running stats (if enabled)
        """
        save_checkpoint(
            self.model, self.optimizer, path,
            stage=self.stage,
            archetype=self.archetype,
            step=global_step,
            frame_stack=self.frame_stack,
            tier=self.tier,
            obs_normalizer=self.obs_normalizer,
        )

    def load_checkpoint(self, path: str) -> dict:
        """Load a checkpoint. Returns the raw checkpoint dict.

        Subclasses should call this and then handle method-specific loading
        (e.g. critic reinitialization for stage transitions).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Restore normalizer state if saved.
        if self.obs_normalizer and "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
            print(f"Restored observation normalizer state")

        return ckpt

    def print_setup(self):
        """Print training configuration summary."""
        tier_cfg = TIER_CONFIGS[self.tier]
        print(f"\n{'='*60}")
        print(f"{self.method_name.upper()} Training — Stage {self.stage}, "
              f"Archetype {self.archetype}")
        print(f"{'='*60}")
        print(f"Device: {self.device}, Timesteps: {self.total_timesteps:,}")
        print(f"Tier: {self.tier} "
              f"(entity={tier_cfg['entity_dim']}, "
              f"unique={tier_cfg['unique_dim']}, "
              f"backbone={tier_cfg['backbone_hidden']}"
              f"×{tier_cfg['backbone_layers']})")
        print(f"Frame stack: {self.frame_stack}, "
              f"input size: {self.input_size}")
        print(f"Envs: {self.num_envs}")
        print(f"Logs: {self.log_dir}")

    # ─── Curriculum Runner ───────────────────────────────────────

    def run_curriculum(self):
        """Run all 7 curriculum stages sequentially.

        Each stage trains from the previous stage's best checkpoint.
        The curriculum timesteps come from default_curriculum_timesteps()
        which methods can override.
        """
        stage_timesteps = self.default_curriculum_timesteps()
        current_checkpoint = self.bc_checkpoint

        for stage in range(1, 8):
            print(f"\n{'='*60}")
            print(f"CURRICULUM STAGE {stage}/7 ({self.method_name.upper()})")
            print(f"{'='*60}")

            # Reconfigure for this stage.
            self.stage = stage
            self.total_timesteps = stage_timesteps[stage]
            self.bc_checkpoint = current_checkpoint

            # Reset eval tracking for new stage.
            self.eval_history = []
            self.best_eval_win_rate = -1.0
            self.best_eval_reward = float("-inf")

            # Reset TensorBoard writer for new stage.
            self.writer.close()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = (f"runs/{self.method_name}_s{stage}_"
                            f"{self.archetype}_{timestamp}")
            self.writer = SummaryWriter(self.log_dir)

            self.train()

            # Chain to next stage.
            best_path = os.path.join(
                self.output_dir,
                f"{self.method_name}_stage{stage}_best.pt")
            final_path = os.path.join(
                self.output_dir,
                f"{self.method_name}_stage{stage}_final.pt")

            if os.path.exists(best_path):
                current_checkpoint = best_path
                print(f"Stage {stage} complete. "
                      f"Next stage loads BEST: {current_checkpoint}")
            else:
                current_checkpoint = final_path
                print(f"Stage {stage} complete. "
                      f"No best found, using final: {current_checkpoint}")

    def cleanup(self):
        """Close writer and any open resources."""
        self.writer.close()

    # ─── Private Helpers ─────────────────────────────────────────

    def _plot_eval_correlation(self, global_step: int):
        """Plot reward vs win-rate correlation to TensorBoard."""
        hist = self.eval_history
        rewards = [h[1] for h in hist]
        winrates = [h[2] for h in hist]
        progress = np.linspace(0, 1, len(hist))

        fig, ax = plt.subplots(1, 1, figsize=(6, 1.75), dpi=250)
        ax.scatter(rewards, winrates, c=progress, cmap="viridis",
                   s=20, edgecolors="white", linewidths=0.5, zorder=3)

        # Regression line.
        r = np.array(rewards)
        w = np.array(winrates)
        coeffs = np.polyfit(r, w, 1)
        r_sorted = np.sort(r)
        ax.plot(r_sorted, np.poly1d(coeffs)(r_sorted),
                "--", color="red", alpha=0.6, linewidth=1.5)
        corr = np.corrcoef(r, w)[0, 1]

        ax.set_xlabel("Reward", fontsize=10, fontweight="bold")
        ax.set_ylabel("Win Rate", fontsize=10, fontweight="bold")
        ax.set_title(f"r = {corr:.3f}", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(pad=0.5)

        self.writer.add_figure(
            "eval/reward_vs_winrate", fig, global_step, close=True)
        self.writer.add_scalar("eval/reward_winrate_corr", corr, global_step)
