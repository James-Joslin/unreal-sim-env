"""
actor_critic.py — PPO ActorCritic with GRU memory + independent heads.

Architecture:
    obs → DeltaEncoder → StructuredEncoder → backbone → GRU → heads
    
The GRU gives episodic memory — the agent remembers the entire encounter.
The critic does NOT use the GRU (it estimates V(s) from the current 
observation only). This simplifies hidden state management: only the 
actor's hidden state needs to be tracked per agent.
"""

import torch
import torch.nn as nn
import warnings

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    TIER_CONFIGS, BEHAVIOR_TIER_DEFINITIONS, layer_init, resolve_tier,
    build_feature_visibility, StructuredEncoder, DeltaEncoder,
)

TARGET_MOVE_CLASSES = 9  # stationary + 8 compass directions


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
    """PPO actor-critic with GRU memory and independent action heads.
    
    The actor path: encoder → backbone → GRU → heads.
    The critic path: encoder → backbone → value_head (no GRU).
    """

    def __init__(self, obs_size=OBS_SIZE, hidden=128, tier="large"):
        super().__init__()
        tier = resolve_tier(tier)
        cfg = TIER_CONFIGS[tier]
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
        self.register_buffer(
            "feature_visibility", build_feature_visibility(tier))
        behavior = BEHAVIOR_TIER_DEFINITIONS[tier]
        for name, size in (
            ("movement", MOVEMENT_ACTIONS),
            ("combat", COMBAT_ACTIONS),
            ("target", TARGET_ACTIONS),
        ):
            available = torch.zeros(size, dtype=torch.bool)
            available[list(behavior[f"{name}_actions"])] = True
            self.register_buffer(f"{name}_availability", available)

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

        # Independent one-pass heads on the shared recurrent features.
        self.move_head = layer_init(
            nn.Linear(gru_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(
            nn.Linear(gru_hidden, COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(
            nn.Linear(gru_hidden, TARGET_ACTIONS), std=0.01)

        # Value head (critic — no GRU, uses backbone directly).
        self.value_head = layer_init(
            nn.Linear(backbone_hidden, 1), std=1.0)

        # Auxiliary prediction head: opponent movement direction (9 classes).
        # Uses critic features — predicts target's current movement from the
        # observation to regularise the encoder.  Disabled when aux_pred_coef=0.
        self.predict_head = layer_init(
            nn.Linear(backbone_hidden, TARGET_MOVE_CLASSES), std=0.01)

    def init_hidden(self, batch_size=1, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _encode(self, obs, encoder):
        frames = obs.reshape(-1, self.frame_stack, OBS_SIZE)
        frames = frames * self.feature_visibility.to(obs.dtype).view(1, 1, -1)
        deltas = self.delta(frames.reshape_as(obs))
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = encoder(channels_flat)
        return emb_flat.view(batch, 3 * encoder.channel_dim)

    def _actor_features(self, obs, hidden):
        """Encode through actor path + GRU. Returns (gru_output, new_hidden)."""
        backbone_out = self.actor_backbone(
            self._encode(obs, self.actor_encoder))
        gru_in = backbone_out.unsqueeze(1)
        gru_out, hidden_out = self.gru(gru_in, hidden)
        return gru_out.squeeze(1), hidden_out

    def _policy_logits(self, features):
        movement = self.move_head(features).masked_fill(
            ~self.movement_availability, -1e8)
        combat = self.combat_head(features).masked_fill(
            ~self.combat_availability, -1e8)
        target = self.target_head(features).masked_fill(
            ~self.target_availability, -1e8)
        return movement, combat, target

    # ─── get_action_and_value (training rollout) ─────────────────

    def get_action_and_value(self, obs, masks=None, hidden=None):
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        actor_feat, hidden_out = self._actor_features(obs, hidden)
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))

        m_logits, c_logits, t_logits = self._policy_logits(actor_feat)
        if masks is not None:
            m_logits = m_logits.masked_fill(~masks[0], -1e8)
            c_logits = c_logits.masked_fill(~masks[1], -1e8)
            t_logits = t_logits.masked_fill(~masks[2], -1e8)
        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)
        m_act, c_act, t_act = (
            m_dist.sample(), c_dist.sample(), t_dist.sample())

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = (m_dist.entropy(), c_dist.entropy(), t_dist.entropy())
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

        m_logits, c_logits, t_logits = self._policy_logits(actor_feat)

        if masks is not None:
            m_logits = m_logits.masked_fill(~masks[0], -1e8)
            c_logits = c_logits.masked_fill(~masks[1], -1e8)
            t_logits = t_logits.masked_fill(~masks[2], -1e8)

        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)

        log_prob = (m_dist.log_prob(m_act) + c_dist.log_prob(c_act)
                    + t_dist.log_prob(t_act))
        entropy = (m_dist.entropy(), c_dist.entropy(), t_dist.entropy())
        value = self.value_head(critic_feat).squeeze(-1)
        pred_logits = self.predict_head(critic_feat)

        return log_prob, entropy, value, pred_logits

    # ─── select_actions (eval — masked independent argmax) ───────

    def select_actions(self, obs, masks, hidden=None):
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        actor_feat, hidden_out = self._actor_features(obs, hidden)
        m_mask, c_mask, t_mask = masks

        m_logits, c_logits, t_logits = self._policy_logits(actor_feat)
        m_logits = m_logits.masked_fill(~m_mask, -1e8)
        c_logits = c_logits.masked_fill(~c_mask, -1e8)
        t_logits = t_logits.masked_fill(~t_mask, -1e8)
        return (
            m_logits.argmax(dim=-1),
            c_logits.argmax(dim=-1),
            t_logits.argmax(dim=-1),
            hidden_out,
        )

    def get_value(self, obs):
        critic_feat = self.critic_backbone(
            self._encode(obs, self.critic_encoder))
        return self.value_head(critic_feat).squeeze(-1)

    # ─── Checkpoint loading ──────────────────────────────────────

    def load_from_ppo_checkpoint(self, ckpt_path, reinit_critic=False):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "full_state_dict" in ckpt:
            state = ckpt["full_state_dict"]
            legacy_autoregressive = (
                ckpt.get("policy_contract") != "independent_heads_v1"
                and any(
                    key.startswith(("move_embed.", "combat_proj.",
                                    "combat_embed.", "target_proj."))
                    for key in state
                )
            )
            if legacy_autoregressive:
                warnings.warn(
                    "Legacy autoregressive PPO checkpoint detected. Shared "
                    "representations and movement head are retained; combat "
                    "and target heads are reinitialised for independent heads.",
                    RuntimeWarning,
                    stacklevel=2,
                )

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
                if key in {
                        "feature_visibility", "movement_availability",
                        "combat_availability", "target_availability"}:
                    # Keep the destination tier's current deploy contract.
                    skipped += 1
                    continue
                if (legacy_autoregressive
                        and key.startswith((
                            "move_embed.", "combat_proj.", "combat_head.",
                            "combat_embed.", "target_proj.", "target_head.",
                        ))):
                    skipped += 1
                    continue
                if (key in own_state
                        and state[key].shape == own_state[key].shape):
                    own_state[key] = state[key]
                    loaded += 1
                elif (not legacy_autoregressive
                        and key in {
                            "combat_head.weight", "combat_head.bias"}
                        and state[key].shape[0] == COMBAT_ACTIONS - 1
                        and state[key].shape[1:] == own_state[key].shape[1:]):
                    expanded = own_state[key].clone()
                    expanded[:COMBAT_ACTIONS - 1] = state[key]
                    own_state[key] = expanded
                    loaded += 1
                    warnings.warn(
                        "Expanded legacy 8-action combat head to 9 actions; "
                        "Reposition starts from fresh initial weights.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    skipped += 1
            self.load_state_dict(own_state, strict=False)

            print(f"Loaded {loaded}/{loaded + skipped} tensors "
                  f"(skipped {skipped} — shape mismatch or new params)")
            return True

        print(f"No full_state_dict in checkpoint")
        return False
