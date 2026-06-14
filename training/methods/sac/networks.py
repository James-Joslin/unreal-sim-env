"""
networks.py — Discrete SAC actor and twin Q-network.

Architecture mirrors CombatPolicy exactly: delta encode → group encode →
backbone → heads. Actor keys map 1:1 to CombatPolicy for ONNX export:
    encoder.* → encoder.*
    backbone.* → backbone.*
    move_head.* → move_head.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    TIER_CONFIGS, layer_init,
    StructuredEncoder, DeltaEncoder,
)


def _build_backbone(input_size: int, hidden: int, num_layers: int) -> nn.Sequential:
    layers = []
    in_dim = input_size
    for i in range(num_layers):
        layers.append(layer_init(nn.Linear(in_dim, hidden)))
        if i == 0:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.GELU())
        in_dim = hidden
    return nn.Sequential(*layers)


class SACPolicyNetwork(nn.Module):
    """Discrete SAC actor: outputs action logits for 3 categorical heads.

    Same structured encoding as CombatPolicy. The forward pass returns
    raw logits; the trainer converts to probabilities with softmax and
    applies action masking.
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

        self.backbone = _build_backbone(
            concat_dim, cfg["backbone_hidden"], cfg["backbone_layers"])

        self.move_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(
            nn.Linear(cfg["backbone_hidden"], TARGET_ACTIONS), std=0.01)

    def forward(self, obs):
        """Returns (m_logits, c_logits, t_logits)."""
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = self.encoder(channels_flat)
        embeddings = emb_flat.view(batch, 3 * self.encoder.channel_dim)
        features = self.backbone(embeddings)

        m = torch.tanh(self.move_head(features))
        c = torch.tanh(self.combat_head(features))
        t = torch.tanh(self.target_head(features))
        return m, c, t

    def get_action_probs(self, obs, masks=None, epsilon=1e-8):
        """Compute action probabilities and log-probabilities with masking.

        Returns:
            probs: tuple of (m_probs, c_probs, t_probs), each [batch, N]
            log_probs: tuple of (m_lp, c_lp, t_lp), each [batch, N]
            entropies: tuple of (m_ent, c_ent, t_ent), each [batch]
        """
        m_logits, c_logits, t_logits = self.forward(obs)

        if masks is not None:
            m_mask, c_mask, t_mask = masks
            m_logits = m_logits.masked_fill(~m_mask, -1e8)
            c_logits = c_logits.masked_fill(~c_mask, -1e8)
            t_logits = t_logits.masked_fill(~t_mask, -1e8)

        m_probs = F.softmax(m_logits, dim=-1)
        c_probs = F.softmax(c_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)

        # Clamp for numerical stability in log.
        m_lp = torch.log(m_probs.clamp(min=epsilon))
        c_lp = torch.log(c_probs.clamp(min=epsilon))
        t_lp = torch.log(t_probs.clamp(min=epsilon))

        # Per-head entropy: -sum(p * log p).
        m_ent = -(m_probs * m_lp).sum(dim=-1)
        c_ent = -(c_probs * c_lp).sum(dim=-1)
        t_ent = -(t_probs * t_lp).sum(dim=-1)

        return (m_probs, c_probs, t_probs), (m_lp, c_lp, t_lp), (m_ent, c_ent, t_ent)

    def sample_actions(self, obs, masks=None):
        """Sample actions from the policy. Returns actions and log-probs.

        Returns:
            actions: (m_act, c_act, t_act), each [batch]
            log_prob: sum of per-head log-probs, [batch]
        """
        (m_p, c_p, t_p), (m_lp, c_lp, t_lp), _ = self.get_action_probs(obs, masks)

        m_act = torch.multinomial(m_p, 1).squeeze(-1)
        c_act = torch.multinomial(c_p, 1).squeeze(-1)
        t_act = torch.multinomial(t_p, 1).squeeze(-1)

        log_prob = (
            m_lp.gather(1, m_act.unsqueeze(-1)).squeeze(-1) +
            c_lp.gather(1, c_act.unsqueeze(-1)).squeeze(-1) +
            t_lp.gather(1, t_act.unsqueeze(-1)).squeeze(-1)
        )

        return (m_act, c_act, t_act), log_prob


class TwinQNetwork(nn.Module):
    """Twin Q-networks for discrete SAC.

    Two independent Q-functions, each outputting Q(s,a) for every
    possible action per head. The minimum of the two is used for
    target computation to reduce positive bias.
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

        self.backbone_1 = _build_backbone(
            concat_dim, h, cfg["backbone_layers"])
        self.backbone_2 = _build_backbone(
            concat_dim, h, cfg["backbone_layers"])

        # Q-value heads: Q(s,a) for each possible action.
        self.q1_m = nn.Linear(h, MOVEMENT_ACTIONS)
        self.q1_c = nn.Linear(h, COMBAT_ACTIONS)
        self.q1_t = nn.Linear(h, TARGET_ACTIONS)

        self.q2_m = nn.Linear(h, MOVEMENT_ACTIONS)
        self.q2_c = nn.Linear(h, COMBAT_ACTIONS)
        self.q2_t = nn.Linear(h, TARGET_ACTIONS)

    def _encode(self, obs, encoder, backbone):
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb = encoder(channels_flat).view(batch, -1)
        return backbone(emb)

    def forward(self, obs):
        """Returns (q1_m, q1_c, q1_t), (q2_m, q2_c, q2_t)."""
        feat1 = self._encode(obs, self.encoder_1, self.backbone_1)
        feat2 = self._encode(obs, self.encoder_2, self.backbone_2)

        q1 = (self.q1_m(feat1), self.q1_c(feat1), self.q1_t(feat1))
        q2 = (self.q2_m(feat2), self.q2_c(feat2), self.q2_t(feat2))
        return q1, q2