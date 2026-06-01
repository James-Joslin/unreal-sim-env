"""
sac.py — Soft Actor-Critic skeleton for discrete action spaces.

This is a TEMPLATE showing how to add a new training method.
Implement the TODO sections to get a working SAC trainer.

SAC KEY DIFFERENCES FROM PPO
    - Off-policy: uses a replay buffer, not on-policy rollouts
    - No GAE: uses soft Bellman backup with twin Q-networks
    - Entropy is a constraint (auto-tuned α), not a loss coefficient
    - Much more sample-efficient, but less stable at scale
    - Needs discrete action variant (Christodoulou 2019) for our
      multi-head categorical action space

REGISTER THIS METHOD
    In training/methods/__init__.py, uncomment:
        from .sac import SACTrainer
        register_method("sac", SACTrainer)

    Then use: python -m training.main --method sac --stage 3
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    CombatPolicy, make_policy, TIER_CONFIGS,
    StructuredEncoder, DeltaEncoder, layer_init, LOGIT_SCALE,
)
from frame_stack import stacked_obs_size

from training.base_trainer import BaseTrainer
from training.normalizers import RunningNormalizer

from .config import SACConfig
from .buffer import ReplayBuffer
from .actor import TwinQNetwork, SACPolicyNetwork

# ─────────────────────────────────────────────────────────────────
#  SAC Trainer
# ─────────────────────────────────────────────────────────────────

class SACTrainer(BaseTrainer):
    """Soft Actor-Critic trainer for discrete action spaces.

    TODO: Implement the training loop. The architecture, buffer, and
    Q-networks are already set up. You need to implement:
    1. The soft Bellman backup for discrete actions
    2. Auto-tuned entropy temperature (α)
    3. Soft target network updates
    """

    method_name = "sac"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = SACConfig(total_timesteps=self.total_timesteps)

    def build_model(self) -> nn.Module:
        """Create SAC actor, twin critics, and target networks."""
        self.actor = SACPolicyNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)

        self.critic = TwinQNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)

        self.critic_target = TwinQNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic)

        # Auto-tuned entropy temperature.
        self.target_entropy = -self.cfg.target_entropy_ratio * np.log(
            1.0 / MOVEMENT_ACTIONS)  # Per-head target, simplified.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.cfg.lr_alpha)

        # The "model" for BaseTrainer's save_checkpoint is the actor.
        self.model = self.actor
        self.optimizer = self.actor_optimizer
        return self.actor

    def extract_policy(self) -> CombatPolicy:
        """Extract CombatPolicy from SAC actor.

        SAC actor keys map directly to CombatPolicy (no actor_ prefix):
            encoder.* → encoder.*
            backbone.* → backbone.*
            move_head.* → move_head.*
        """
        policy = make_policy(self.tier, frame_stack=self.frame_stack)
        policy.load_state_dict(self.actor.state_dict(), strict=False)
        policy.eval()
        return policy

    def default_curriculum_timesteps(self) -> Dict[int, int]:
        """SAC is more sample-efficient — can use fewer timesteps."""
        return {
            1: 25_000,
            2: 50_000,
            3: 250_000,
            4: 3_000_000,
            5: 5_000_000,
            6: 10_000_000,
            7: 15_000_000,
        }

    def train(self):
        """SAC training loop — TODO: implement."""
        raise NotImplementedError(
            "SAC training loop not yet implemented. "
            "See the docstring and TODO comments in this file for guidance. "
            "The architecture, buffer, and Q-networks are ready — "
            "you need to implement the soft Bellman backup, "
            "entropy temperature tuning, and target network updates."
        )

        # SKETCH of what the training loop would look like:
        #
        # self.build_model()
        # self.print_setup()
        # vec_env = self.create_vec_env()
        # replay_buffer = ReplayBuffer(self.cfg.buffer_size, self.input_size)
        # obs, infos = vec_env.reset()
        # global_step = 0
        #
        # while global_step < self.cfg.total_timesteps:
        #     # 1. Select action (random during warmup, then from policy)
        #     # 2. Step environment
        #     # 3. Add transition to replay buffer
        #     # 4. If enough samples, train:
        #     #    a. Sample minibatch from replay buffer
        #     #    b. Compute target Q: r + γ * (min(Q1_target, Q2_target) - α * log_prob)
        #     #    c. Update critics (MSE loss)
        #     #    d. Update actor (maximise Q - α * entropy)
        #     #    e. Update α (match target entropy)
        #     #    f. Soft-update target networks: θ_target = τ*θ + (1-τ)*θ_target
        #     # 5. Periodic eval + checkpointing (use self.run_eval, self.check_best_model)
        #
        # vec_env.close()
        # self.writer.close()
