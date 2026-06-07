"""
actor_critic.py — PPO ActorCritic with autoregressive action heads.

The actor samples actions SEQUENTIALLY:
    1. movement  = P(m | obs)
    2. combat    = P(c | obs, m)       ← conditioned on chosen movement
    3. target    = P(t | obs, m, c)    ← conditioned on movement + combat

This lets the policy learn action correlations: "if I chose FIRE, I
should select the target I'm facing" becomes a learnable conditional
rather than a coincidence of independent distributions.

Projection layers before combat/target heads keep the head OUTPUT shapes
identical to the non-autoregressive model, so old checkpoints partially
load (heads transfer, new embedding/projection layers init fresh).

KEY NAMING CONVENTION
    ActorCritic keys          CombatPolicy keys
    ──────────────────        ──────────────────
    actor_encoder.*       →   encoder.*
    actor_backbone.*      →   backbone.*
    move_head.*           →   move_head.*
    move_embed.*          →   move_embed.*          (NEW)
    combat_proj.*         →   combat_proj.*         (NEW)
    combat_head.*         →   combat_head.*
    combat_embed.*        →   combat_embed.*        (NEW)
    target_proj.*         →   target_proj.*         (NEW)
    target_head.*         →   target_head.*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    TIER_CONFIGS, LOGIT_SCALE, layer_init,
    StructuredEncoder, DeltaEncoder,
)

ACTION_EMBED_DIM = 16


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


class ActorCritic(nn.Module):
    """PPO actor-critic with autoregressive action heads."""

    def __init__(self, obs_size=OBS_SIZE, hidden=128, tier="large"):
        super().__init__()
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["large"])
        entity_dim = cfg["entity_dim"]
        unique_dim = cfg["unique_dim"]
        backbone_hidden = cfg["backbone_hidden"]
        backbone_layers = cfg["backbone_layers"]
        attention_heads = cfg["attention_heads"]

        self.obs_size = obs_size
        self.frame_stack = max(1, obs_size // OBS_SIZE) if obs_size > OBS_SIZE else 1
        self.tier = tier
        self.backbone_hidden = backbone_hidden

        # Delta encoding (no learnable params, shared).
        self.delta = DeltaEncoder(self.frame_stack)

        # Group encoders (separate for actor/critic).
        self.actor_encoder = StructuredEncoder(entity_dim, unique_dim, attention_heads)
        self.critic_encoder = StructuredEncoder(entity_dim, unique_dim, attention_heads)

        channel_dim = self.actor_encoder.channel_dim
        concat_dim = 3 * channel_dim

        # Backbones (separate).
        self.actor_backbone = _build_backbone(
            concat_dim, backbone_hidden, backbone_layers)
        self.critic_backbone = _build_backbone(
            concat_dim, backbone_hidden, backbone_layers)

        # ── Autoregressive policy heads ──────────────────────────
        # Head 1: movement (unconditioned — same as before).
        self.move_head = layer_init(
            nn.Linear(backbone_hidden, MOVEMENT_ACTIONS), std=0.01)

        # Embedding for conditioning subsequent heads.
        self.move_embed = nn.Embedding(MOVEMENT_ACTIONS, ACTION_EMBED_DIM)

        # Head 2: combat (conditioned on movement).
        # Projection fuses action embedding with backbone features.
        self.combat_proj = layer_init(
            nn.Linear(backbone_hidden + ACTION_EMBED_DIM, backbone_hidden))
        self.combat_head = layer_init(
            nn.Linear(backbone_hidden, COMBAT_ACTIONS), std=0.01)

        self.combat_embed = nn.Embedding(COMBAT_ACTIONS, ACTION_EMBED_DIM)

        # Head 3: target (conditioned on movement + combat).
        self.target_proj = layer_init(
            nn.Linear(backbone_hidden + 2 * ACTION_EMBED_DIM, backbone_hidden))
        self.target_head = layer_init(
            nn.Linear(backbone_hidden, TARGET_ACTIONS), std=0.01)

        # Value head (critic).
        self.value_head = layer_init(
            nn.Linear(backbone_hidden, 1), std=1.0)

    def _encode(self, obs, encoder):
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = encoder(channels_flat)
        return emb_flat.view(batch, 3 * encoder.channel_dim)

    def _scaled(self, raw):
        return torch.tanh(raw) * LOGIT_SCALE

    def _get_actor_features(self, obs):
        return self.actor_backbone(self._encode(obs, self.actor_encoder))

    def _get_critic_features(self, obs):
        return self.critic_backbone(self._encode(obs, self.critic_encoder))

    def _autoregressive_logits(self, features, m_action, c_action):
        """Compute all three logit tensors with autoregressive conditioning.

        Args:
            features: backbone output [batch, backbone_hidden]
            m_action: movement actions for conditioning [batch] (ints)
            c_action: combat actions for conditioning [batch] (ints)

        Returns:
            m_logits, c_logits, t_logits
        """
        m_logits = self._scaled(self.move_head(features))

        m_emb = self.move_embed(m_action)
        c_features = F.gelu(self.combat_proj(
            torch.cat([features, m_emb], dim=-1)))
        c_logits = self._scaled(self.combat_head(c_features))

        c_emb = self.combat_embed(c_action)
        t_features = F.gelu(self.target_proj(
            torch.cat([features, m_emb, c_emb], dim=-1)))
        t_logits = self._scaled(self.target_head(t_features))

        return m_logits, c_logits, t_logits

    # ─── forward (for basic eval — uses unmasked argmax for conditioning) ───

    def forward(self, obs):
        actor_feat = self._get_actor_features(obs)
        critic_feat = self._get_critic_features(obs)

        # Autoregressive: sample m, condition c, condition t.
        m_logits = self._scaled(self.move_head(actor_feat))
        m_action = m_logits.argmax(dim=-1)

        m_emb = self.move_embed(m_action)
        c_features = F.gelu(self.combat_proj(
            torch.cat([actor_feat, m_emb], dim=-1)))
        c_logits = self._scaled(self.combat_head(c_features))
        c_action = c_logits.argmax(dim=-1)

        c_emb = self.combat_embed(c_action)
        t_features = F.gelu(self.target_proj(
            torch.cat([actor_feat, m_emb, c_emb], dim=-1)))
        t_logits = self._scaled(self.target_head(t_features))

        value = self.value_head(critic_feat)
        return m_logits, c_logits, t_logits, value

    # ─── get_action_and_value (training rollout — masked sequential sampling) ───

    def get_action_and_value(self, obs, masks=None):
        actor_feat = self._get_actor_features(obs)
        critic_feat = self._get_critic_features(obs)

        # Head 1: movement.
        m_logits = self._scaled(self.move_head(actor_feat))
        if masks is not None:
            m_logits = m_logits.masked_fill(~masks[0], -1e8)
        m_dist = torch.distributions.Categorical(logits=m_logits)
        m_act = m_dist.sample()

        # Head 2: combat conditioned on movement.
        m_emb = self.move_embed(m_act)
        c_features = F.gelu(self.combat_proj(
            torch.cat([actor_feat, m_emb], dim=-1)))
        c_logits = self._scaled(self.combat_head(c_features))
        if masks is not None:
            c_logits = c_logits.masked_fill(~masks[1], -1e8)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        c_act = c_dist.sample()

        # Head 3: target conditioned on movement + combat.
        c_emb = self.combat_embed(c_act)
        t_features = F.gelu(self.target_proj(
            torch.cat([actor_feat, m_emb, c_emb], dim=-1)))
        t_logits = self._scaled(self.target_head(t_features))
        if masks is not None:
            t_logits = t_logits.masked_fill(~masks[2], -1e8)
        t_dist = torch.distributions.Categorical(logits=t_logits)
        t_act = t_dist.sample()

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = m_dist.entropy() + c_dist.entropy() + t_dist.entropy()
        value = self.value_head(critic_feat).squeeze(-1)

        return (m_act, c_act, t_act), log_prob, entropy, value

    # ─── evaluate_actions (PPO update — uses given actions for conditioning) ───

    def evaluate_actions(self, obs, m_act, c_act, t_act, masks=None):
        actor_feat = self._get_actor_features(obs)
        critic_feat = self._get_critic_features(obs)

        # Use the PROVIDED actions for conditioning (teacher forcing).
        m_logits, c_logits, t_logits = self._autoregressive_logits(
            actor_feat, m_act, c_act)

        if masks is not None:
            m_logits = m_logits.masked_fill(~masks[0], -1e8)
            c_logits = c_logits.masked_fill(~masks[1], -1e8)
            t_logits = t_logits.masked_fill(~masks[2], -1e8)

        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = m_dist.entropy() + c_dist.entropy() + t_dist.entropy()
        value = self.value_head(critic_feat).squeeze(-1)

        return log_prob, entropy, value

    # ─── select_actions (eval — masked autoregressive argmax) ───

    def select_actions(self, obs, masks):
        """Deterministic action selection with proper autoregressive masking."""
        actor_feat = self._get_actor_features(obs)
        m_mask, c_mask, t_mask = masks

        m_logits = self._scaled(self.move_head(actor_feat))
        m_logits = m_logits.masked_fill(~m_mask, -1e8)
        m = m_logits.argmax(dim=-1)

        m_emb = self.move_embed(m)
        c_features = F.gelu(self.combat_proj(
            torch.cat([actor_feat, m_emb], dim=-1)))
        c_logits = self._scaled(self.combat_head(c_features))
        c_logits = c_logits.masked_fill(~c_mask, -1e8)
        c = c_logits.argmax(dim=-1)

        c_emb = self.combat_embed(c)
        t_features = F.gelu(self.target_proj(
            torch.cat([actor_feat, m_emb, c_emb], dim=-1)))
        t_logits = self._scaled(self.target_head(t_features))
        t_logits = t_logits.masked_fill(~t_mask, -1e8)
        t = t_logits.argmax(dim=-1)

        return m, c, t

    def get_value(self, obs):
        critic_feat = self._get_critic_features(obs)
        return self.value_head(critic_feat).squeeze(-1)

    # ─── Checkpoint loading ───

    def load_from_ppo_checkpoint(self, ckpt_path, reinit_critic=False):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "full_state_dict" in ckpt:
            state = ckpt["full_state_dict"]

            critic_keys = [
                "critic_encoder", "critic_backbone", "value_head"]
            if reinit_critic:
                before = len(state)
                state = {k: v for k, v in state.items()
                         if not any(c in k for c in critic_keys)}
                print(f"Fresh critic: dropped {before - len(state)} "
                      f"critic tensors")

            own_state = self.state_dict()
            loaded = 0
            skipped = 0
            for key in state:
                if (key in own_state
                        and state[key].shape == own_state[key].shape):
                    own_state[key] = state[key]
                    loaded += 1
                else:
                    skipped += 1
            self.load_state_dict(own_state, strict=False)

            print(f"Loaded {loaded}/{loaded + skipped} tensors "
                  f"(skipped {skipped} — shape mismatch or new params)")
            return True

        print(f"No full_state_dict in checkpoint")
        return False