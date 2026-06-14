"""
combat_policy.py — Structured policy network with delta encoding and entity-aware processing.

ARCHITECTURE
    Input: flat [batch, 645] (3 frames x 215 features, from C++ frame stacking)

    Stage 1 — Reshape + Delta Encode (baked into ONNX, zero params):
        Reshape [batch, 645] → [batch, 3, 215]
        current      = frames[:, 2]                                    (latest)
        velocity     = frames[:, 2] - frames[:, 1]                    (1st derivative)
        acceleration = frames[:, 2] - 2·frames[:, 1] + frames[:, 0]   (2nd derivative)

    Stage 2 — Per-Channel Group Encoding (shared weights across 3 channels):
        For each delta channel:
            Split 215 features → unique (127) + hostile slots (4x13) + ally slots (3x12) + dynamic threat slots (3x3)
            unique_emb  = Linear(127, unique_dim) → GELU
            hostile_emb = shared Linear(13, entity_dim) → GELU, x4 slots, max-pool
            ally_emb    = shared Linear(12, entity_dim) → GELU, x3 slots, max-pool
            threat_emb  = shared Linear(3, entity_dim) → GELU, x3 slots, max-pool
            channel_emb = [unique_emb ‖ hostile_pool ‖ ally_pool ‖ threat_pool]

    Stage 3 — Policy Backbone:
        Concat 3 channels → MLP backbone → 3 policy heads

OBSERVATION LAYOUT (215 per frame)
    [  0.. 20]  Self State                        (21)     ─┐
    [ 21.. 42]  Weapon State                      (22)      │
    [ 43.. 49]  Archetype                         ( 7)      ├─ Unique features (127 total)
    [ 50.. 69]  Primary Target                    (20)     ─┘ (+ spatial/threat/nav/metrics below)
    [ 70..121]  Hostile Targets                   (52)  ←── 4 slots x 13 (shared hostile encoder)
    [122..157]  Allied Robots                     (36)  ←── 3 slots x 12 (shared ally encoder)
    [158..165]  Spatial Ring                      ( 8)     ─┐
    [166..173]  Cover Height                      ( 8)      │
    [174..181]  Threat Sensing (Projectile 1)     ( 8)      ├─ Unique features (continued)
                (174: dist, 175: speed, 176: dirX, 177: dirY, etc.) // nearest threat
    [182..190]  Navmesh Viability                 ( 9)      │
    [191..196]  Group Summary                     ( 6)      │
    [197..197]  Spawn/Leash                       ( 1)     ─┘
    [198..200]  Threat Sensing (Projectile 2)     ( 3)      (dist, dirX, dirY) [shared threat encoder]
    [201..203]  Threat Sensing (Projectile 3)     ( 3)      (dist, dirX, dirY) [shared threat encoder]
    [204..204]  Incoming Threat Count             ( 1)      Knowing when to dodge vs fight
    [205..208]  Can Hit Target Per Weapon         ( 4)      (weapon 1, 2, 3, 4) target availability
    [209..209]  Total Ammo Fraction               ( 1)      Ammo conservation state
    [210..210]  Targets Killed Fraction           ( 1)      Kill urgency tracker
    [211..214]  Arc Clearance Per Weapon          ( 4)      MaxArcableObstacleHeight / 3000

C++ SIDE: Only change FrameStackCount from 8 to 3. Input stays flat TArray<float>.

TIER ARCHITECTURE (structured)
    | Tier   | entity | unique | backbone | layers | Approx Params |
    |--------|--------|--------|----------|--------|---------------|
    | Micro  | 8      | 16     | 32       | 1      | ~9K           |
    | Small  | 12     | 24     | 48       | 1      | ~18K          |
    | Medium | 16     | 32     | 64       | 2      | ~38K          |
    | Large  | 16     | 32     | 96       | 2      | ~48K          |
    | XL     | 24     | 48     | 128      | 3      | ~85K          |
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
#  Constants (must match C++ NeuralCombatTypes.h)
# ─────────────────────────────────────────────────────────────────

OBS_SIZE = 249
MOVEMENT_ACTIONS = 9
COMBAT_ACTIONS = 8   # Added Dodge (action 7)
TARGET_ACTIONS = 5

DEFAULT_FRAME_STACK = 3

# Observation layout: feature group boundaries within one 249-float frame.
_HOSTILE_START = 74                        # 4 slots x 17 features
_HOSTILE_SLOTS = 4
_HOSTILE_SLOT_SIZE = 17                    # was 13: +class, +mana, +commitment, +gap_closer
_ALLY_START = 142                          # 3 slots x 15 features
_ALLY_SLOTS = 3
_ALLY_SLOT_SIZE = 15                       # was 12: +target_idx, +combat_action, +flanking
_UNIQUE_SIZE = 136                         # 74 (self+weapon+arch+primary24) + 62 (spatial..patterns)

# Logits are unbounded — standard for PPO. Clip range, grad norm
# clipping, and KL early stopping provide sufficient stability.
# The tanh × 3.0 bounding was removed because it saturated gradients
# once raw logits exceeded ±2, creating a ceiling on policy confidence
# and contributing to entropy stagnation in later curriculum stages.


# ─────────────────────────────────────────────────────────────────
#  Tier Configurations
# ─────────────────────────────────────────────────────────────────
TIER_CONFIGS = {
    # d_k = 4. Ultra-low latency deployment tier.
    "micro": dict(
        entity_dim=8, 
        unique_dim=16, 
        backbone_hidden=32, 
        backbone_layers=1, 
        attention_heads=2,
        gru_hidden=32
    ),
    
    # d_k = 6. Light deployment tier.
    "small": dict(
        entity_dim=12, 
        unique_dim=24, 
        backbone_hidden=48, 
        backbone_layers=1, 
        attention_heads=2,
        gru_hidden=48
    ),
    
    # d_k = 4. Mid-capacity tier — matches old large dims.
    # Suitable for S1-S4 training or mid-tier deployment.
    "medium": dict(
        entity_dim=16, 
        unique_dim=32, 
        backbone_hidden=48, 
        backbone_layers=2, 
        attention_heads=4,
        gru_hidden=48
    ),
    
    # d_k = 6. Primary training tier for S5-S6.
    # Expanded from 16/32/96 to give the encoder richer entity
    # representations for multi-target coordination and the
    # backbone more capacity for ally-aware decision making.
    "large": dict(
        entity_dim=16,
        unique_dim=32,
        backbone_hidden=64,
        backbone_layers=2,
        attention_heads=4,
        gru_hidden=64
    ),
    
    # d_k = 8. High-capacity tier for S7 training.
    # Widest representations for full 4-target squad combat.
    "xl": dict(
        entity_dim=16,
        unique_dim=32,
        backbone_hidden=64,
        backbone_layers=3,
        attention_heads=4,
        gru_hidden=64
    )
}


# ─────────────────────────────────────────────────────────────────
#  Initialisation Helpers
# ─────────────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ─────────────────────────────────────────────────────────────────
#  Cross-Attention for Entity Slots
# ─────────────────────────────────────────────────────────────────

class EntityAttention(nn.Module):
    """Cross-attention from context (unique features) to entity slots.

    Replaces max-pooling. Instead of "what's the most extreme entity?"
    the model learns "given my current state, which entity matters most?"

    A reloading enemy at close range gets high attention weight; a
    full-health enemy behind cover gets low. This is particularly
    important now that entity slots carry rich information (mana,
    commitment, class type) — the model needs to attend to the RIGHT
    entity's mana state, not the max across all slots.

    Uses manual multi-head attention (matmul + softmax) for clean
    ONNX export — no nn.MultiheadAttention internals to trace.
    """

    def __init__(self, query_dim: int, entity_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = entity_dim // num_heads
        self.entity_dim = entity_dim
        assert entity_dim % num_heads == 0, (
            f"entity_dim ({entity_dim}) must be divisible by "
            f"num_heads ({num_heads})")

        self.q_proj = layer_init(nn.Linear(query_dim, entity_dim))
        self.k_proj = layer_init(nn.Linear(entity_dim, entity_dim))
        self.v_proj = layer_init(nn.Linear(entity_dim, entity_dim))
        self.out_proj = layer_init(nn.Linear(entity_dim, entity_dim))
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor,
                entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query:    [batch, query_dim] — context from unique features.
            entities: [batch, num_slots, entity_dim] — per-slot embeddings.
        Returns:
            [batch, entity_dim] — attended entity summary.
        """
        batch = query.shape[0]
        num_slots = entities.shape[1]
        h, d = self.num_heads, self.head_dim

        # Project to multi-head space.
        q = self.q_proj(query).view(batch, 1, h, d).transpose(1, 2)
        k = self.k_proj(entities).view(batch, num_slots, h, d).transpose(1, 2)
        v = self.v_proj(entities).view(batch, num_slots, h, d).transpose(1, 2)

        # Scaled dot-product attention.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [batch, heads, 1, head_dim]

        # Reshape and project.
        out = out.transpose(1, 2).reshape(batch, self.entity_dim)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────
#  Structured Encoder (shared building block)
# ─────────────────────────────────────────────────────────────────

class StructuredEncoder(nn.Module):
    def __init__(self, entity_dim: int, unique_dim: int, attention_heads: int):
        super().__init__()
        self.entity_dim = entity_dim
        self.unique_dim = unique_dim
        self.threat_dim = entity_dim
        self.channel_dim = unique_dim + entity_dim + entity_dim + self.threat_dim

        # Unique features (136 inputs -> 64)
        self.unique_encoder = nn.Sequential(
            layer_init(nn.Linear(136, unique_dim)),
            nn.GELU(),
        )

        # Hostile slot encoder (17 -> 32)
        self.hostile_encoder = nn.Sequential(
            layer_init(nn.Linear(_HOSTILE_SLOT_SIZE, entity_dim)),
            nn.GELU(),
        )

        # Ally slot encoder (15 -> 32)
        self.ally_encoder = nn.Sequential(
            layer_init(nn.Linear(_ALLY_SLOT_SIZE, entity_dim)),
            nn.GELU(),
        )

        # Threat slot encoder (3 -> 32)
        self.threat_slot_encoder = nn.Sequential(
            layer_init(nn.Linear(3, self.threat_dim)), # CHECK THIS
            nn.GELU(),
        )

        # --- Cross-Attention Layer Gating ---
        # Dynamically scale hostile and ally tracking using the tier configuration
        self.hostile_attn = EntityAttention(unique_dim, entity_dim, num_heads=attention_heads)
        self.ally_attn = EntityAttention(unique_dim, entity_dim, num_heads=attention_heads)
        
        threat_heads = 2 if attention_heads >= 2 else 1
        self.threat_attn = EntityAttention(unique_dim, self.threat_dim, num_heads=threat_heads)
        
    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        batch = frame.shape[0]

        # 1. Unique features.
        unique_feats = torch.cat([
            frame[:, 0:74],
            frame[:, 187:OBS_SIZE],
        ], dim=-1)  # [batch, 136]
        unique_emb = self.unique_encoder(unique_feats)  # [batch, unique_dim]

        # 2. Hostile entities → shared encoder → cross-attention.
        hostile_feats = frame[:, _HOSTILE_START:_HOSTILE_START + _HOSTILE_SLOTS * _HOSTILE_SLOT_SIZE]
        hostile_feats = hostile_feats.view(batch, _HOSTILE_SLOTS, _HOSTILE_SLOT_SIZE)
        hostile_flat = hostile_feats.reshape(batch * _HOSTILE_SLOTS, _HOSTILE_SLOT_SIZE)
        hostile_embs = self.hostile_encoder(hostile_flat).view(batch, _HOSTILE_SLOTS, self.entity_dim)
        hostile_attended = self.hostile_attn(unique_emb, hostile_embs)

        # 3. Allied entities → shared encoder → cross-attention.
        ally_feats = frame[:, _ALLY_START:_ALLY_START + _ALLY_SLOTS * _ALLY_SLOT_SIZE]
        ally_feats = ally_feats.view(batch, _ALLY_SLOTS, _ALLY_SLOT_SIZE)
        ally_flat = ally_feats.reshape(batch * _ALLY_SLOTS, _ALLY_SLOT_SIZE)
        ally_embs = self.ally_encoder(ally_flat).view(batch, _ALLY_SLOTS, self.entity_dim)
        ally_attended = self.ally_attn(unique_emb, ally_embs)

        # 4. Threat entities → shared encoder → cross-attention.
        t1 = torch.stack([frame[:, 203], frame[:, 205], frame[:, 206]], dim=-1)
        t2 = torch.stack([frame[:, 227], frame[:, 228], frame[:, 229]], dim=-1)
        t3 = torch.stack([frame[:, 230], frame[:, 231], frame[:, 232]], dim=-1)
        threats_feats = torch.stack([t1, t2, t3], dim=1)
        threats_flat = threats_feats.reshape(batch * 3, 3)
        threats_embs = self.threat_slot_encoder(threats_flat).view(batch, 3, self.threat_dim)
        threats_attended = self.threat_attn(unique_emb, threats_embs)

        # 5. Concatenate.
        return torch.cat([unique_emb, hostile_attended, ally_attended, threats_attended], dim=-1)

# ─────────────────────────────────────────────────────────────────
#  Delta Encoding Module (no learnable params)
# ─────────────────────────────────────────────────────────────────

class DeltaEncoder(nn.Module):
    """Reshapes flat frame-stacked input and computes temporal deltas.

    Input:  [batch, frame_stack * 215]  (flat, from C++ frame stacking)
    Output: [batch, 3, 215]  (current, velocity, acceleration)

    Baked into ONNX — C++ feeds raw flat observations unchanged.
    """

    def __init__(self, frame_stack: int = 3):
        super().__init__()
        self.frame_stack = frame_stack
        self.obs_size = OBS_SIZE

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        batch = flat_obs.shape[0]

        # Reshape: [batch, N*215] → [batch, N, 215]
        frames = flat_obs.view(batch, self.frame_stack, self.obs_size)

        # Delta encoding. Frames are oldest-first: [t-2, t-1, t].
        current = frames[:, -1]                                         # t
        velocity = frames[:, -1] - frames[:, -2]                       # t - (t-1)
        acceleration = frames[:, -1] - 2.0 * frames[:, -2] + frames[:, -3]  # t - 2(t-1) + (t-2)

        # Stack as [batch, 3, 215]: channels = (current, velocity, acceleration).
        return torch.stack([current, velocity, acceleration], dim=1)


# ─────────────────────────────────────────────────────────────────
#  Structured Combat Policy (inference-only, no value head)
# ─────────────────────────────────────────────────────────────────

class CombatPolicy(nn.Module):
    """Structured policy for distillation and ONNX export.

    Architecture: delta encode → group encode → backbone → GRU → heads.
    The GRU gives episodic memory — the agent remembers the entire
    encounter, not just the last 3 frames. Everything bakes into the
    ONNX graph, including hidden state as an additional input/output.
    """

    def __init__(self, frame_stack: int = DEFAULT_FRAME_STACK,
                 entity_dim: int = 16, unique_dim: int = 32,
                 backbone_hidden: int = 96, backbone_layers: int = 2,
                 attention_heads = 4, gru_hidden = 32):
        super().__init__()
        self.frame_stack = frame_stack
        self.gru_hidden = gru_hidden

        # Stage 1: Delta encoding (no params).
        self.delta = DeltaEncoder(frame_stack)

        # Stage 2: Structured group encoder (shared across 3 delta channels).
        self.encoder = StructuredEncoder(entity_dim, unique_dim, attention_heads)
        channel_dim = self.encoder.channel_dim
        concat_dim = 3 * channel_dim

        # Stage 3: Policy backbone.
        backbone_layers_list = []
        in_dim = concat_dim
        for i in range(backbone_layers):
            backbone_layers_list.append(layer_init(nn.Linear(in_dim, backbone_hidden)))
            if i == 0:
                backbone_layers_list.append(nn.LayerNorm(backbone_hidden))
            backbone_layers_list.append(nn.GELU())
            in_dim = backbone_hidden
        self.backbone = nn.Sequential(*backbone_layers_list)

        # Stage 4: GRU memory (episodic memory across the encounter).
        self.gru = nn.GRU(backbone_hidden, gru_hidden, num_layers=1,
                          batch_first=True)

        # Stage 5: Autoregressive policy heads (on GRU output).
        ACTION_EMBED_DIM = 16
        self.move_head = layer_init(nn.Linear(gru_hidden, MOVEMENT_ACTIONS), std=0.01)

        self.move_embed = nn.Embedding(MOVEMENT_ACTIONS, ACTION_EMBED_DIM)
        self.combat_proj = layer_init(nn.Linear(gru_hidden + ACTION_EMBED_DIM, gru_hidden))
        self.combat_head = layer_init(nn.Linear(gru_hidden, COMBAT_ACTIONS), std=0.01)

        self.combat_embed = nn.Embedding(COMBAT_ACTIONS, ACTION_EMBED_DIM)
        self.target_proj = layer_init(nn.Linear(gru_hidden + 2 * ACTION_EMBED_DIM, gru_hidden))
        self.target_head = layer_init(nn.Linear(gru_hidden, TARGET_ACTIONS), std=0.01)

    def init_hidden(self, batch_size: int = 1, device=None):
        """Create zero-initialised hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _encode_features(self, obs):
        """Encode obs through delta → structured encoder → backbone."""
        deltas = self.delta(obs)
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        embeddings_flat = self.encoder(channels_flat)
        embeddings = embeddings_flat.view(batch, 3 * self.encoder.channel_dim)
        return self.backbone(embeddings)  # [batch, backbone_hidden]

    def _heads(self, features):
        """Autoregressive heads with unmasked argmax conditioning."""
        m = self.move_head(features)
        m_action = m.argmax(dim=-1)

        m_emb = self.move_embed(m_action)
        c_feat = F.gelu(self.combat_proj(
            torch.cat([features, m_emb], dim=-1)))
        c = self.combat_head(c_feat)
        c_action = c.argmax(dim=-1)

        c_emb = self.combat_embed(c_action)
        t_feat = F.gelu(self.target_proj(
            torch.cat([features, m_emb, c_emb], dim=-1)))
        t = self.target_head(t_feat)

        return m, c, t

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """Forward pass for ONNX export.

        Args:
            obs:    [batch, frame_stack * OBS_SIZE]
            hidden: [1, batch, gru_hidden] or None (zeros)
        Returns:
            m_logits, c_logits, t_logits, hidden_out
        """
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        backbone_out = self._encode_features(obs)  # [batch, backbone_hidden]
        gru_in = backbone_out.unsqueeze(1)          # [batch, 1, backbone_hidden]
        gru_out, hidden_out = self.gru(gru_in, hidden)
        features = gru_out.squeeze(1)               # [batch, gru_hidden]

        m, c, t = self._heads(features)
        return m, c, t, hidden_out

    def select_actions(self, obs, masks, hidden=None):
        """Autoregressive masked action selection with GRU."""
        batch = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)

        backbone_out = self._encode_features(obs)
        gru_in = backbone_out.unsqueeze(1)
        gru_out, hidden_out = self.gru(gru_in, hidden)
        features = gru_out.squeeze(1)

        m_mask, c_mask, t_mask = masks

        m_logits = self.move_head(features)
        m_logits = m_logits.masked_fill(~m_mask, -1e8)
        m = m_logits.argmax(dim=-1)

        m_emb = self.move_embed(m)
        c_feat = F.gelu(self.combat_proj(
            torch.cat([features, m_emb], dim=-1)))
        c_logits = self.combat_head(c_feat)
        c_logits = c_logits.masked_fill(~c_mask, -1e8)
        c = c_logits.argmax(dim=-1)

        c_emb = self.combat_embed(c)
        t_feat = F.gelu(self.target_proj(
            torch.cat([features, m_emb, c_emb], dim=-1)))
        t_logits = self.target_head(t_feat)
        t_logits = t_logits.masked_fill(~t_mask, -1e8)
        t = t_logits.argmax(dim=-1)

        return m, c, t, hidden_out


# ─────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────

def make_policy(tier: str, frame_stack: int = DEFAULT_FRAME_STACK) -> CombatPolicy:
    """Create a CombatPolicy for a specific tier."""
    cfg = TIER_CONFIGS[tier]
    return CombatPolicy(
        frame_stack=frame_stack,
        entity_dim=cfg["entity_dim"],
        unique_dim=cfg["unique_dim"],
        backbone_hidden=cfg["backbone_hidden"],
        backbone_layers=cfg["backbone_layers"],
        attention_heads=cfg["attention_heads"],
        gru_hidden=cfg["gru_hidden"],
    )


# ─────────────────────────────────────────────────────────────────
#  Load Teacher from PPO Checkpoint
# ─────────────────────────────────────────────────────────────────

def load_teacher_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device("cpu"),
) -> CombatPolicy:
    """Extract the policy from a PPO StructuredActorCritic checkpoint.

    Key mapping:
        StructuredActorCritic        CombatPolicy
        ─────────────────────        ─────────────
        actor_encoder.*          →   encoder.*
        actor_backbone.*         →   backbone.*
        delta.*                  →   delta.*  (no params, but present)
        move_head.*              →   move_head.*
        combat_head.*            →   combat_head.*
        target_head.*            →   target_head.*
        critic_*                 →   (dropped)
        value_head.*             →   (dropped)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    frame_stack = ckpt.get("frame_stack", DEFAULT_FRAME_STACK)
    tier = ckpt.get("tier", "large")

    print(f"Loading teacher from: {checkpoint_path}")
    print(f"  Tier: {tier}, Frame stack: {frame_stack}, "
          f"Stage: {ckpt.get('stage', '?')}, Step: {ckpt.get('step', '?')}")

    teacher = make_policy(tier, frame_stack=frame_stack).to(device)
    source_dict = ckpt.get("full_state_dict", ckpt.get("model_state_dict", {}))

    policy_state = teacher.state_dict()
    loaded = 0
    skipped = 0

    for src_key, src_val in source_dict.items():
        # Skip critic keys.
        if any(skip in src_key for skip in
               ["critic_encoder", "critic_backbone", "value_head"]):
            skipped += 1
            continue

        # Map actor_encoder → encoder, actor_backbone → backbone.
        dst_key = src_key.replace("actor_encoder.", "encoder.")
        dst_key = dst_key.replace("actor_backbone.", "backbone.")

        if dst_key in policy_state:
            if src_val.shape == policy_state[dst_key].shape:
                policy_state[dst_key] = src_val
                loaded += 1
            else:
                print(f"  Shape mismatch: {src_key} {src_val.shape} "
                      f"vs {dst_key} {policy_state[dst_key].shape}")

    teacher.load_state_dict(policy_state, strict=False)
    teacher.eval()

    total_policy_keys = len(policy_state)
    print(f"  Loaded {loaded}/{total_policy_keys} policy tensors "
          f"(skipped {skipped} critic tensors)")

    return teacher


# ─────────────────────────────────────────────────────────────────
#  Save PPO Checkpoint
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, path, stage, archetype, step,
                        frame_stack=DEFAULT_FRAME_STACK, tier="large",
                        obs_normalizer=None):
    """Save a PPO checkpoint with clean policy extraction."""
    policy_state = {}
    for key, val in model.state_dict().items():
        if any(skip in key for skip in
               ["critic_encoder", "critic_backbone", "value_head"]):
            continue
        new_key = key.replace("actor_encoder.", "encoder.")
        new_key = new_key.replace("actor_backbone.", "backbone.")
        policy_state[new_key] = val

    save_dict = {
        "policy_state_dict": policy_state,
        "full_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tier": tier,
        "stage": stage,
        "archetype": archetype,
        "step": step,
        "frame_stack": frame_stack,
        "architecture": "structured",
    }

    if obs_normalizer:
        save_dict["obs_normalizer"] = obs_normalizer.state_dict()

    save_dict["model_state_dict"] = policy_state
    torch.save(save_dict, path)


# ─────────────────────────────────────────────────────────────────
#  Normalized Wrapper (bakes normalizer into ONNX graph)
# ─────────────────────────────────────────────────────────────────

class NormalizedPolicyWrapper(nn.Module):
    """Wraps a CombatPolicy with input normalization baked in.

    The ONNX model accepts raw observations — normalization happens
    inside the graph. The C++ side feeds raw flat observations.

    Normalization is applied to the FLAT input (all frames concatenated)
    before the model's delta encoding reshapes it. This matches the
    training pipeline where RunningNormalizer operates on flat observations.
    """

    def __init__(self, policy: CombatPolicy, mean: np.ndarray,
                 var: np.ndarray, clip: float = 5.0, epsilon: float = 1e-8):
        super().__init__()
        self.policy = policy
        self.clip = clip
        self.register_buffer("obs_mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("obs_std",
            torch.from_numpy(np.sqrt(var.astype(np.float32) + epsilon)))

    def forward(self, obs, hidden=None):
        normed = torch.clamp((obs - self.obs_mean) / self.obs_std,
                             -self.clip, self.clip)
        return self.policy(normed, hidden)


# ─────────────────────────────────────────────────────────────────
#  ONNX Export
# ─────────────────────────────────────────────────────────────────

def export_onnx(model: CombatPolicy, tier: str, output_dir: str,
                frame_stack: int = DEFAULT_FRAME_STACK,
                obs_normalizer=None) -> str:
    """Export a CombatPolicy to ONNX.

    The exported graph includes: normalisation → reshape → delta encode
    → structured group encode → backbone → policy heads.

    C++ feeds raw flat observations. Everything else is in the graph.
    """
    model.eval().cpu()
    input_size = OBS_SIZE * frame_stack

    if obs_normalizer is not None:
        export_model = NormalizedPolicyWrapper(
            model,
            mean=obs_normalizer.mean,
            var=obs_normalizer.var,
            clip=getattr(obs_normalizer, "clip", 5.0),
            epsilon=getattr(obs_normalizer, "epsilon", 1e-8),
        ).eval().cpu()
        print(f"  Baking observation normalizer into ONNX graph")
    else:
        export_model = model

    dummy = torch.randn(1, input_size)
    dummy_hidden = torch.zeros(1, 1, model.gru_hidden)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"Combat_{tier.capitalize()}.onnx")

    torch.onnx.export(
        export_model, (dummy, dummy_hidden), path,
        input_names=["observation", "hidden_in"],
        output_names=["movement_logits", "combat_logits",
                       "target_logits", "hidden_out"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "hidden_in": {1: "batch_size"},
            "hidden_out": {1: "batch_size"},
        },
        opset_version=17,
    )

    # Consolidate to single file.
    try:
        import onnx
        onnx_model = onnx.load(path, load_external_data=True)
        onnx.save(onnx_model, path, save_as_external_data=False)
        data_path = path + ".data"
        if os.path.exists(data_path):
            os.remove(data_path)
    except ImportError:
        print("  Warning: onnx package not installed — cannot consolidate.")

    size_kb = os.path.getsize(path) / 1024
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Exported: {path} ({size_kb:.1f} KB, {param_count:,} params)")
    return path


# ─────────────────────────────────────────────────────────────────
#  Verify Round-Trip
# ─────────────────────────────────────────────────────────────────

def verify_export(model: CombatPolicy, onnx_path: str,
                  frame_stack: int = DEFAULT_FRAME_STACK,
                  obs_normalizer=None):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Skipping verification (onnxruntime not installed)")
        return

    model.eval().cpu()
    input_size = OBS_SIZE * frame_stack
    dummy = torch.randn(1, input_size)
    dummy_hidden = torch.zeros(1, 1, model.gru_hidden)

    if obs_normalizer is not None:
        pt_model = NormalizedPolicyWrapper(
            model,
            mean=obs_normalizer.mean,
            var=obs_normalizer.var,
            clip=getattr(obs_normalizer, "clip", 5.0),
            epsilon=getattr(obs_normalizer, "epsilon", 1e-8),
        ).eval().cpu()
    else:
        pt_model = model

    with torch.no_grad():
        pt_m, pt_c, pt_t, pt_h = pt_model(dummy, dummy_hidden)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "observation": dummy.numpy(),
        "hidden_in": dummy_hidden.numpy(),
    })

    m_diff = abs(pt_m.numpy() - ort_out[0]).max()
    c_diff = abs(pt_c.numpy() - ort_out[1]).max()
    t_diff = abs(pt_t.numpy() - ort_out[2]).max()
    max_diff = max(m_diff, c_diff, t_diff)

    status = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"  Verify {status}: max diff = {max_diff:.6f} "
          f"(m={m_diff:.6f}, c={c_diff:.6f}, t={t_diff:.6f})")


# ─────────────────────────────────────────────────────────────────
#  Quick CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test structured policy and ONNX export")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tier", type=str, default="large",
                        choices=TIER_CONFIGS.keys())
    parser.add_argument("--frame_stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--output_dir", type=str, default="models/test")
    args = parser.parse_args()

    print("=" * 60)
    print("Structured Policy Network Test")
    print("=" * 60)

    print(f"\nArchitecture: delta encode → entity-shared group encode → backbone")
    print(f"Frame stack: {args.frame_stack} (input size: {OBS_SIZE * args.frame_stack})")

    print("\nTier parameter counts:")
    for tier_name in TIER_CONFIGS:
        m = make_policy(tier_name, frame_stack=args.frame_stack)
        params = sum(p.numel() for p in m.parameters())
        enc_params = sum(p.numel() for p in m.encoder.parameters())
        bb_params = sum(p.numel() for p in m.backbone.parameters())
        head_params = sum(p.numel() for n, p in m.named_parameters() if "head" in n)
        print(f"  {tier_name:8s}: {params:>8,} total "
              f"(encoder: {enc_params:,}, backbone: {bb_params:,}, heads: {head_params:,})")

    print(f"\nExporting {args.tier} tier...")
    model = make_policy(args.tier, frame_stack=args.frame_stack)
    onnx_path = export_onnx(model, args.tier, args.output_dir,
                             frame_stack=args.frame_stack)
    verify_export(model, onnx_path, frame_stack=args.frame_stack)

    if args.checkpoint:
        print(f"\nExtracting teacher from checkpoint...")
        teacher = load_teacher_from_checkpoint(args.checkpoint)
        teacher_path = export_onnx(teacher, "large_teacher", args.output_dir,
                                    frame_stack=args.frame_stack)
        verify_export(teacher, teacher_path, frame_stack=args.frame_stack)

    print("\nDone.")