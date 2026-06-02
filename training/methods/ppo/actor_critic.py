"""
actor_critic.py — PPO-specific ActorCritic model.

The actor side mirrors CombatPolicy from combat_policy.py exactly:
same hidden sizes, same layer count, same activations, same logit
scaling. The critic side is a separate encoder+backbone that feeds a
value head. Separate paths prevent gradient contamination.

KEY NAMING CONVENTION
    ActorCritic keys          CombatPolicy keys
    ──────────────────        ──────────────────
    actor_encoder.*       →   encoder.*
    actor_backbone.*      →   backbone.*
    move_head.*           →   move_head.*
    combat_head.*         →   combat_head.*
    target_head.*         →   target_head.*
    critic_encoder.*      →   (dropped at export)
    critic_backbone.*     →   (dropped at export)
    value_head.*          →   (dropped at export)
"""

import torch
import torch.nn as nn
import numpy as np

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    TIER_CONFIGS, LOGIT_SCALE, layer_init,
    StructuredEncoder, DeltaEncoder,
)


def _build_backbone(input_size: int, hidden: int, num_layers: int) -> nn.Sequential:
    """Build a backbone using GELU activations and LayerNorm."""
    layers = []
    in_dim = input_size
    for i in range(num_layers):
        layers.append(layer_init(nn.Linear(in_dim, hidden)))
        if i == 0:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.GELU())
        in_dim = hidden
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """PPO actor-critic with structured encoding.

    Architecture: delta encode → group encode → backbone → heads.
    Actor and critic have separate encoder+backbone paths to prevent
    gradient contamination from the value function leaking into policy
    features.
    """

    def __init__(self, obs_size=OBS_SIZE, hidden=128, tier="large"):
        super().__init__()
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["large"])
        entity_dim = cfg["entity_dim"]
        unique_dim = cfg["unique_dim"]
        backbone_hidden = cfg["backbone_hidden"]
        backbone_layers = cfg["backbone_layers"]

        self.obs_size = obs_size
        self.frame_stack = max(1, obs_size // OBS_SIZE) if obs_size > OBS_SIZE else 1
        self.tier = tier

        # Stage 1: Delta encoding (no learnable params, shared).
        self.delta = DeltaEncoder(self.frame_stack)

        # Stage 2: Group encoders (separate for actor/critic).
        self.actor_encoder = StructuredEncoder(entity_dim, unique_dim)
        self.critic_encoder = StructuredEncoder(entity_dim, unique_dim)

        channel_dim = self.actor_encoder.channel_dim
        concat_dim = 3 * channel_dim  # 3 delta channels concatenated

        # Stage 3: Backbones (separate).
        self.actor_backbone = _build_backbone(
            concat_dim, backbone_hidden, backbone_layers)
        self.critic_backbone = _build_backbone(
            concat_dim, backbone_hidden, backbone_layers)

        # Policy heads.
        self.move_head = layer_init(
            nn.Linear(backbone_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(
            nn.Linear(backbone_hidden, COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(
            nn.Linear(backbone_hidden, TARGET_ACTIONS), std=0.01)

        # Value head.
        self.value_head = layer_init(
            nn.Linear(backbone_hidden, 1), std=1.0)

    def _encode(self, obs: torch.Tensor,
                encoder: StructuredEncoder) -> torch.Tensor:
        """Delta encode → group encode → concat channels."""
        deltas = self.delta(obs)  # [batch, 3, OBS_SIZE]
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = encoder(channels_flat)  # [batch*3, channel_dim]
        return emb_flat.view(batch, 3 * encoder.channel_dim)

    def _scaled_logits(self, raw):
        """Bound logits to [-LOGIT_SCALE, +LOGIT_SCALE] via tanh."""
        return torch.tanh(raw) * LOGIT_SCALE

    def forward(self, obs):
        actor_feat = self.actor_backbone(
            self._encode(obs, self.actor_encoder))
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))

        m_logits = self._scaled_logits(self.move_head(actor_feat))
        c_logits = self._scaled_logits(self.combat_head(actor_feat))
        t_logits = self._scaled_logits(self.target_head(actor_feat))
        value = self.value_head(critic_feat)

        return m_logits, c_logits, t_logits, value

    def forward_inference(self, obs):
        """Policy only — no value head."""
        actor_feat = self.actor_backbone(
            self._encode(obs, self.actor_encoder))
        m_logits = self._scaled_logits(self.move_head(actor_feat))
        c_logits = self._scaled_logits(self.combat_head(actor_feat))
        t_logits = self._scaled_logits(self.target_head(actor_feat))
        return m_logits, c_logits, t_logits

    def get_action_and_value(self, obs, masks=None):
        m_logits, c_logits, t_logits, value = self.forward(obs)

        if masks is not None:
            m_mask, c_mask, t_mask = masks
            m_logits = m_logits.masked_fill(~m_mask, -1e8)
            c_logits = c_logits.masked_fill(~c_mask, -1e8)
            t_logits = t_logits.masked_fill(~t_mask, -1e8)

        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)

        m_act = m_dist.sample()
        c_act = c_dist.sample()
        t_act = t_dist.sample()

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = m_dist.entropy() + c_dist.entropy() + t_dist.entropy()

        return (m_act, c_act, t_act), log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(self, obs, m_act, c_act, t_act, masks=None):
        m_logits, c_logits, t_logits, value = self.forward(obs)

        if masks is not None:
            m_mask, c_mask, t_mask = masks
            m_logits = m_logits.masked_fill(~m_mask, -1e8)
            c_logits = c_logits.masked_fill(~c_mask, -1e8)
            t_logits = t_logits.masked_fill(~t_mask, -1e8)

        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = m_dist.entropy() + c_dist.entropy() + t_dist.entropy()

        return log_prob, entropy, value.squeeze(-1)

    def get_value(self, obs):
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))
        return self.value_head(critic_feat).squeeze(-1)

    # ═════════════════════════════════════════════════════════════
    #  Checkpoint Loading
    # ═════════════════════════════════════════════════════════════

    def load_from_ppo_checkpoint(self, ckpt_path: str,
                                 reinit_critic: bool = False):
        """Load from a PPO checkpoint.

        Args:
            ckpt_path: Path to checkpoint.
            reinit_critic: If True, skip loading critic weights (keep fresh
                random init). Use this for curriculum stage transitions.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "full_state_dict" in ckpt:
            state = ckpt["full_state_dict"]

            # Filter out critic weights for fresh init on stage transitions.
            critic_keys = [
                "critic_encoder", "critic_backbone", "value_head"]
            if reinit_critic:
                before = len(state)
                state = {k: v for k, v in state.items()
                         if not any(c in k for c in critic_keys)}
                print(f"Fresh critic: dropped {before - len(state)} "
                      f"critic tensors (reinit_critic=True)")

            own_state = self.state_dict()

            # Partial load — match by key name and shape.
            loaded = 0
            skipped = 0
            for key in state:
                dst_key = key
                if (dst_key in own_state
                        and state[key].shape == own_state[dst_key].shape):
                    own_state[dst_key] = state[key]
                    loaded += 1
                else:
                    skipped += 1
            self.load_state_dict(own_state, strict=False)

            actor_loaded = sum(
                1 for k in state
                if any(a in k for a in [
                    "actor_encoder", "actor_backbone",
                    "move_head", "combat_head", "target_head"])
                and k in own_state
            )
            print(f"Loaded {loaded}/{loaded + skipped} tensors "
                  f"(actor: {actor_loaded}, "
                  f"critic: {'fresh' if reinit_critic else 'restored'})")
            return True

        print(f"No full_state_dict in checkpoint")
        return False
