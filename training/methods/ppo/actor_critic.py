"""
actor_critic.py — PPO ActorCritic with GRU memory + autoregressive heads.

Architecture:
    obs → DeltaEncoder → StructuredEncoder → backbone → GRU → heads
    
The GRU gives episodic memory — the agent remembers the entire encounter.
The critic does NOT use the GRU (it estimates V(s) from the current 
observation only). This simplifies hidden state management: only the 
actor's hidden state needs to be tracked per agent.
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
    """PPO actor-critic with GRU memory and autoregressive action heads.
    
    The actor path: encoder → backbone → GRU → heads.
    The critic path: encoder → backbone → value_head (no GRU).
    """

    def __init__(self, obs_size=OBS_SIZE, hidden=128, tier="large"):
        super().__init__()
        cfg = TIER_CONFIGS.get(tier, TIER_CONFIGS["large"])
        entity_dim = cfg["entity_dim"]
        unique_dim = cfg["unique_dim"]
        backbone_hidden = cfg["backbone_hidden"]
        backbone_layers = cfg["backbone_layers"]
        attention_heads = cfg["attention_heads"]
        gru_hidden = cfg["gru_hidden"]

        self.obs_size = obs_size
        self.frame_stack = max(1, obs_size // OBS_SIZE) if obs_size > OBS_SIZE else 1
        self.tier = tier
        self.backbone_hidden = backbone_hidden
        self.gru_hidden = gru_hidden

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

        # GRU on actor only.
        self.gru = nn.GRU(backbone_hidden, gru_hidden, num_layers=1,
                          batch_first=True)

        # ── Autoregressive policy heads (on GRU output) ─────────
        self.move_head = layer_init(
            nn.Linear(gru_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.move_embed = nn.Embedding(MOVEMENT_ACTIONS, ACTION_EMBED_DIM)

        self.combat_proj = layer_init(
            nn.Linear(gru_hidden + ACTION_EMBED_DIM, gru_hidden))
        self.combat_head = layer_init(
            nn.Linear(gru_hidden, COMBAT_ACTIONS), std=0.01)
        self.combat_embed = nn.Embedding(COMBAT_ACTIONS, ACTION_EMBED_DIM)

        self.target_proj = layer_init(
            nn.Linear(gru_hidden + 2 * ACTION_EMBED_DIM, gru_hidden))
        self.target_head = layer_init(
            nn.Linear(gru_hidden, TARGET_ACTIONS), std=0.01)

        # Value head (critic — no GRU, uses backbone directly).
        self.value_head = layer_init(
            nn.Linear(backbone_hidden, 1), std=1.0)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _encode(self, obs, encoder):
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = encoder(channels_flat)
        return emb_flat.view(batch, 3 * encoder.channel_dim)

    def _scaled(self, raw):
        return torch.tanh(raw) * LOGIT_SCALE

    def _actor_features(self, obs, hidden):
        """Encode through actor path + GRU. Returns (gru_output, new_hidden)."""
        backbone_out = self.actor_backbone(
            self._encode(obs, self.actor_encoder))
        gru_in = backbone_out.unsqueeze(1)
        gru_out, hidden_out = self.gru(gru_in, hidden)
        return gru_out.squeeze(1), hidden_out

    def _autoregressive_logits(self, features, m_action, c_action):
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

    # ─── get_action_and_value (training rollout) ─────────────────

    def get_action_and_value(self, obs, masks=None, hidden=None):
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        actor_feat, hidden_out = self._actor_features(obs, hidden)
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))

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

        return (m_act, c_act, t_act), log_prob, entropy, value, hidden_out

    # ─── evaluate_actions (PPO update) ───────────────────────────

    def evaluate_actions(self, obs, m_act, c_act, t_act,
                         masks=None, hidden=None):
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        actor_feat, _ = self._actor_features(obs, hidden)
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))

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

    # ─── select_actions (eval — masked autoregressive argmax) ────

    def select_actions(self, obs, masks, hidden=None):
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        actor_feat, hidden_out = self._actor_features(obs, hidden)
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

        return m, c, t, hidden_out

    def get_value(self, obs):
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))
        return self.value_head(critic_feat).squeeze(-1)

    # ─── Checkpoint loading ──────────────────────────────────────

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