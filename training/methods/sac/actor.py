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

# ─────────────────────────────────────────────────────────────────
#  Discrete SAC Actor (same architecture as PPO actor, minus critic)
# ─────────────────────────────────────────────────────────────────

class SACPolicyNetwork(nn.Module):
    """
    Discrete SAC policy network.

    Produces one set of action logits for each discrete action branch:

        movement actions
        combat actions
        target actions

    The network uses the shared combat observation encoding pipeline:

        frame delta encoding → structured entity encoding → MLP backbone

    The forward pass returns raw logits for each action branch. During SAC
    training these logits are converted to probabilities/log-probabilities
    with softmax/log_softmax so entropy-regularised discrete action objectives
    can be computed.

    This actor contains no value function or critic head. Q-values are learned
    separately by TwinQNetwork.
    """


    def __init__(self, obs_size=OBS_SIZE, tier="large"):
        super().__init__()
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["large"])
        self.obs_size = obs_size
        self.frame_stack = max(1, obs_size // OBS_SIZE) if obs_size > OBS_SIZE else 1
        self.tier = tier

        self.delta = DeltaEncoder(self.frame_stack)
        self.encoder = StructuredEncoder(cfg["entity_dim"], cfg["unique_dim"])

        channel_dim = self.encoder.channel_dim
        concat_dim = 3 * channel_dim

        # Build backbone.
        layers = []
        in_dim = concat_dim
        for i in range(cfg["backbone_layers"]):
            layers.append(layer_init(nn.Linear(in_dim, cfg["backbone_hidden"])))
            if i == 0:
                layers.append(nn.LayerNorm(cfg["backbone_hidden"]))
            layers.append(nn.GELU())
            in_dim = cfg["backbone_hidden"]
        self.backbone = nn.Sequential(*layers)

        # Policy heads (output log-probs for discrete SAC).
        self.move_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], TARGET_ACTIONS), std=0.01)

    
    def forward(self, obs):
        """
        Return action logits for each discrete action branch.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                m_logits: movement action logits, shape [batch, MOVEMENT_ACTIONS]
                c_logits: combat action logits, shape [batch, COMBAT_ACTIONS]
                t_logits: target action logits, shape [batch, TARGET_ACTIONS]
        """

        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = self.encoder(channels_flat)
        embeddings = emb_flat.view(batch, 3 * self.encoder.channel_dim)
        features = self.backbone(embeddings)

        m = torch.tanh(self.move_head(features)) * LOGIT_SCALE
        c = torch.tanh(self.combat_head(features)) * LOGIT_SCALE
        t = torch.tanh(self.target_head(features)) * LOGIT_SCALE
        return m, c, t


# ─────────────────────────────────────────────────────────────────
#  Twin Q-Network (discrete: outputs Q(s,a) for every action)
# ─────────────────────────────────────────────────────────────────

class TwinQNetwork(nn.Module):
    """
    Twin Q-network for discrete SAC.

    Maintains two independent Q-functions to reduce positive bias during
    target estimation. Each Q-network outputs a Q-value for every possible
    action in each discrete action branch:

        Q_movement(s, a)
        Q_combat(s, a)
        Q_target(s, a)

    The actor selects actions from policy logits, while this network estimates
    the value of those actions. SAC training typically uses the minimum of the
    two Q-functions when computing target values and policy losses.
    """

    def __init__(self, obs_size=OBS_SIZE, tier="large"):
        super().__init__()
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["large"])
        self.frame_stack = max(1, obs_size // OBS_SIZE) if obs_size > OBS_SIZE else 1

        self.delta = DeltaEncoder(self.frame_stack)

        # Two separate encoders + backbones (twin Q).
        self.encoder_1 = StructuredEncoder(cfg["entity_dim"], cfg["unique_dim"])
        self.encoder_2 = StructuredEncoder(cfg["entity_dim"], cfg["unique_dim"])

        channel_dim = self.encoder_1.channel_dim
        concat_dim = 3 * channel_dim
        h = cfg["backbone_hidden"]

        def _build_q_head(in_dim, hidden, layers):
            parts = []
            d = in_dim
            for i in range(layers):
                parts.append(layer_init(nn.Linear(d, hidden)))
                if i == 0:
                    parts.append(nn.LayerNorm(hidden))
                parts.append(nn.GELU())
                d = hidden
            return nn.Sequential(*parts)

        self.backbone_1 = _build_q_head(concat_dim, h, cfg["backbone_layers"])
        self.backbone_2 = _build_q_head(concat_dim, h, cfg["backbone_layers"])

        # Q-value heads: output Q(s,a) for each possible action.
        self.q1_m = nn.Linear(h, MOVEMENT_ACTIONS)
        self.q1_c = nn.Linear(h, COMBAT_ACTIONS)
        self.q1_t = nn.Linear(h, TARGET_ACTIONS)

        self.q2_m = nn.Linear(h, MOVEMENT_ACTIONS)
        self.q2_c = nn.Linear(h, COMBAT_ACTIONS)
        self.q2_t = nn.Linear(h, TARGET_ACTIONS)

    def forward(self, obs):
        """
        Return Q-values from both critics for each discrete action branch.

        Returns:
            tuple:
                q1: tuple of Q-value tensors from critic 1:
                    (q1_m, q1_c, q1_t)

                q2: tuple of Q-value tensors from critic 2:
                    (q2_m, q2_c, q2_t)

            Each tensor has shape [batch, num_actions_for_branch].
        """
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)

        emb1 = self.encoder_1(channels_flat).view(batch, -1)
        emb2 = self.encoder_2(channels_flat).view(batch, -1)

        feat1 = self.backbone_1(emb1)
        feat2 = self.backbone_2(emb2)

        q1 = (self.q1_m(feat1), self.q1_c(feat1), self.q1_t(feat1))
        q2 = (self.q2_m(feat2), self.q2_c(feat2), self.q2_t(feat2))
        return q1, q2
