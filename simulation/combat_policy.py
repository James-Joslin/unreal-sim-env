"""
combat_policy.py — Structured policy network with delta encoding and entity-aware processing.

ARCHITECTURE
    Input: flat [batch, 747] (3 frames x 249 features, from C++ frame stacking)

    Stage 1 — Reshape + Delta Encode (baked into ONNX, zero params):
        Reshape [batch, 747] → [batch, 3, 249]
        current      = frames[:, 2]                                    (latest)
        velocity     = frames[:, 2] - frames[:, 1]                    (1st derivative)
        acceleration = frames[:, 2] - 2·frames[:, 1] + frames[:, 0]   (2nd derivative)

    Stage 2 — Per-Channel Group Encoding (shared weights across 3 channels):
        For each delta channel:
            Split 249 → unique (136) + hostile (4x17) + ally (3x15) + threat (3x3)
            unique_emb  = Linear(136, unique_dim) → GELU
            hostile_emb = shared Linear(17, entity_dim) → GELU, x4 → cross-attention
            ally_emb    = shared Linear(15, entity_dim) → GELU, x3 → cross-attention
            threat_emb  = shared Linear(3, entity_dim)  → GELU, x3 → cross-attention
            channel_emb = [unique_emb ‖ hostile_attn ‖ ally_attn ‖ threat_attn]

    Stage 3 — Policy Backbone:
        Concat 3 channels → MLP backbone → GRU → 3 independent heads

OBSERVATION LAYOUT (249 per frame — must match NeuralCombatTypes.h)

    Unique features: frame[:, 0:74] + frame[:, 187:249] = 136 total

    Self State (21)
    [  0]       hp_fraction
    [  1]       defence / 200
    [  2]       speed / max_speed
    [  3]       stunned                           (CombatEnvExtended fills)
    [  4]       slowed                            (CombatEnvExtended fills)
    [  5.. 10]  debuff slots x6                   (CombatEnvExtended fills)
    [ 11]       velocity_dir X
    [ 12]       velocity_dir Y
    [ 13]       combat_time / 120
    [ 14]       height_above_ground (≈0.176)
    [ 15]       is_action_locked
    [ 16]       action_lock_progress
    [ 17]       action_lock_reason / 7
    [ 18]       is_dodging
    [ 19]       dodge_ready
    [ 20]       invulnerable

    Weapon State (22)
    [ 21]       active_weapon / (n_slots - 1)
    [ 22]       active ammo_fraction
    [ 23]       active is_ready && has_ammo
    [ 24]       active is_reloading
    [ 25]       active reload_progress
    [ 26]       active range / 5000
    [ 27]       active cooldown_frac
    [ 28]       active wind_up_time / 3
    [ 29]       active can_arc
    [ 30]       active is_ranged
    [ 31.. 42]  other weapon slots x3 (ammo, range/5000, reloading, can_arc)

    Archetype (7)
    [ 43.. 46]  one-hot (ranged/melee/healer/tank)
    [ 47]       min_engagement_range / 5000
    [ 48]       has_any_ammo
    [ 49]       melee_ready

    Primary Target (24)
    [ 50]       rel X / 5000
    [ 51]       rel Y / 5000
    [ 52]       dist / 5000
    [ 53]       hp_fraction
    [ 54]       in_weapon_range
    [ 55]       has_LOS
    [ 56]       in_sight_cone                     (CombatEnvExtended overrides)
    [ 57]       agent_facing_dot
    [ 58]       target_facing_toward_agent
    [ 59]       velocity X / 600
    [ 60]       velocity Y / 600
    [ 61]       acceleration X / 2000             (CombatEnvExtended fills)
    [ 62]       acceleration Y / 2000             (CombatEnvExtended fills)
    [ 63]       angular_size (70 / dist)
    [ 64]       is_player_controlled
    [ 65]       behind_low_cover
    [ 66]       cover_height / 500
    [ 67]       in_melee_range
    [ 68]       closing_speed / 1000
    [ 69]       character_type  (Phase 1)
    [ 70]       mana_fraction   (Phase 1)
    [ 71]       commitment      (Phase 1)
    [ 72]       gap_closer_threat (Phase 1)
    [ 73]       reposition_ready

    Hostile Targets (68 = 4 x 17)    _HOSTILE_START = 74
    [ 74.. 90]  slot 0   [ 91..107]  slot 1
    [108..124]  slot 2   [125..141]  slot 3
    Per slot:
        [+0]  occupied (1=present, 0=empty)   ← mask key
        [+1]  rel X / 5000
        [+2]  rel Y / 5000
        [+3]  dist / 5000
        [+4]  hp_fraction
        [+5]  has_LOS
        [+6]  is_player_controlled
        [+7]  facing_dot (target facing → agent)
        [+8]  priority_score / 120            (CombatEnvExtended overrides)
        [+9]  threat_level / 200              (CombatEnvExtended overrides)
        [+10] velocity X / 600
        [+11] velocity Y / 600
        [+12] is_targeting_me (facing dot, 0–1)
        [+13] character_type  (Phase 1)
        [+14] mana_fraction   (Phase 1)
        [+15] commitment      (Phase 1)
        [+16] gap_closer_threat (Phase 1)
    Sorted by priority score, not distance.

    Allied Robots (45 = 3 x 15)      _ALLY_START = 142
    [142..156]  slot 0   [157..171]  slot 1   [172..186]  slot 2
    Per slot:
        [+0]  occupied (1=present, 0=empty)   ← mask key
        [+1]  rel X / 5000
        [+2]  rel Y / 5000
        [+3]  dist / 5000
        [+4]  hp_fraction
        [+5]  has_LOS
        [+6]  velocity X / 600
        [+7]  velocity Y / 600
        [+8]  facing_dot (ally facing → agent)
        [+9]  ammo_fraction
        [+10] is_reloading
        [+11] fire_cooldown / 2
        [+12] target_idx (Phase 1): (idx+1)/5
        [+13] combat_action (Phase 1): action/8
        [+14] flanking_angle (Phase 1): cos(my→tgt, ally→tgt)

    ─── Unique features continued (frame[:, 187:249]) ───

    Spatial Ring (8)                  [187..194]
    Cover Height (8)                 [195..202]

    Threat Sensing — Proj 1 (8)      [203..210]
        [203] proj1 dist             ← threat slot t1[0]
        [204] proj1 TTA
        [205] proj1 dir X            ← threat slot t1[1]
        [206] proj1 dir Y            ← threat slot t1[2]
        [207] nearest melee dist
        [208] nearest melee dir X
        [209] nearest melee dir Y
        [210] dodge_available

    Navmesh Viability (9)            [211..219]
    Group Summary (6)                [220..225]
        [220] alive_allies / 10      (CombatEnvExtended overrides)
        [221] alive_hostiles / 4
        [222] avg_ally_hp            (CombatEnvExtended overrides)
        [223] avg_hostile_hp
        [224] numerical_advantage    (CombatEnvExtended overrides)
        [225] outnumbered
    Spawn / Leash (1)                [226]

    Extended Threat (7)              [227..233]
        [227] proj2 dist             ← threat slot t2[0]
        [228] proj2 dir X            ← threat slot t2[1]
        [229] proj2 dir Y            ← threat slot t2[2]
        [230] proj3 dist             ← threat slot t3[0]
        [231] proj3 dir X            ← threat slot t3[1]
        [232] proj3 dir Y            ← threat slot t3[2]
        [233] threat_count / 5

    Weapon Can-Hit (4)               [234..237]
    Total Ammo Fraction (1)          [238]
    Targets Killed Fraction (1)      [239]
    Arc Clearance / Weapon (4)       [240..243]
    Player Patterns (5)              [244..248]
        aggression, evasion, predictability, preferred_range, mana_burn_rate

TIER ARCHITECTURE (structured)
    | Tier   | entity | unique | backbone | layers | attn_heads | gru  |
    |--------|--------|--------|----------|--------|------------|------|
    | Micro  | 8      | 16     | 32       | 1      | 2          | 32   |
    | Small  | 12     | 24     | 48       | 1      | 2          | 48   |
    | Medium | 16     | 32     | 48       | 2      | 4          | 48   |
    | Large  | 16     | 32     | 64       | 2      | 4          | 64   |
"""

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
#  Constants (must match C++ NeuralCombatTypes.h)
# ─────────────────────────────────────────────────────────────────

OBS_SIZE = 249
MOVEMENT_ACTIONS = 9
COMBAT_ACTIONS = 9   # 0-7 existing actions; 8 = Reposition
TARGET_ACTIONS = 5
REPOSITION_ACTION = 8

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
# The tanh x 3.0 bounding was removed because it saturated gradients
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
    )
}

ACTIVE_TIERS = tuple(TIER_CONFIGS)
TRAINABLE_ARCHETYPES = ("ranged", "melee", "tank")

# Product-facing difficulty definitions. Architecture stays in TIER_CONFIGS;
# these labels define what each deployed tier is expected to feel like.
BEHAVIOR_TIER_DEFINITIONS = {
    "micro": {
        "label": "reactive",
        "description": "Immediate valid responses with limited tactical depth.",
        "movement_actions": tuple(range(MOVEMENT_ACTIONS)),
        "combat_actions": (0, 1, 2, 5),
        "target_actions": (0, 4),
        "curriculum_stages": tuple(range(1, 8)),
        "training_focus": "Reactive survival and direct engagement.",
    },
    "small": {
        "label": "competent",
        "description": "Consistent basic combat, movement, and target selection.",
        "movement_actions": tuple(range(MOVEMENT_ACTIONS)),
        "combat_actions": tuple(range(REPOSITION_ACTION)),
        "target_actions": tuple(range(TARGET_ACTIONS)),
        "curriculum_stages": tuple(range(1, 8)),
        "training_focus": "Reliable core combat and basic target switching.",
    },
    "medium": {
        "label": "tactical",
        "description": "Purposeful positioning, cooldown use, and target switching.",
        "movement_actions": tuple(range(MOVEMENT_ACTIONS)),
        "combat_actions": tuple(range(COMBAT_ACTIONS)),
        "target_actions": tuple(range(TARGET_ACTIONS)),
        "curriculum_stages": tuple(range(1, 8)),
        "training_focus": "Threat scoring, cover, flanking, and repositioning.",
    },
    "large": {
        "label": "advanced",
        "description": "Strong context use and multi-actor tactical decisions.",
        "movement_actions": tuple(range(MOVEMENT_ACTIONS)),
        "combat_actions": tuple(range(COMBAT_ACTIONS)),
        "target_actions": tuple(range(TARGET_ACTIONS)),
        "curriculum_stages": tuple(range(1, 8)),
        "training_focus": "Full ally coordination and opponent adaptation.",
    },
}

# Per-frame observation visibility is a deployment contract, not a training
# hint. The mask is applied after optional normalisation and before temporal
# delta encoding, so PPO, distillation, PyTorch inference, and ONNX all see the
# same progressively richer information while retaining the 249-float ABI.
# Ranges are half-open [start, stop) indices into one observation frame.
_BASIC_HOSTILE_RANGES = tuple(
    item
    for slot in range(_HOSTILE_SLOTS)
    for item in (
        (_HOSTILE_START + slot * _HOSTILE_SLOT_SIZE,
         _HOSTILE_START + slot * _HOSTILE_SLOT_SIZE + 7),
        (_HOSTILE_START + slot * _HOSTILE_SLOT_SIZE + 10,
         _HOSTILE_START + slot * _HOSTILE_SLOT_SIZE + 12),
    )
)

FEATURE_VISIBILITY_RANGES = {
    # Agent/weapon/archetype and primary-target state, with primary cover
    # fields [65:67) deliberately withheld. Spatial/nav viability keeps this
    # reactive tier functional around collision in every scenario stage.
    "micro": ((0, 65), (67, 74), (187, 195), (211, 220)),
    # Add basic hostile occupancy/position/HP/LOS/identity/velocity so target
    # choices 0-3 have observable meaning, plus projectile/can-hit readiness.
    "small": (
        ((0, 65), (67, 74))
        + _BASIC_HOSTILE_RANGES
        + ((187, 195), (203, 220), (227, 239))
    ),
    # Add hostile roster, cover, spawn/leash, and full tactical readiness;
    # ally slots, group summary, and player patterns remain unavailable.
    "medium": ((0, 142), (187, 220), (226, 244)),
    # Full coordination and opponent-pattern context.
    "large": ((0, OBS_SIZE),),
}

# Metrics already produced by the shared evaluator. Tier comparisons should be
# reported in ACTIVE_TIERS order; no speculative numeric thresholds are baked in.
BEHAVIOR_METRICS = {
    "primary": "win_rate",
    "secondary": ("mean_reward", "mean_kills"),
    "guardrail": "mean_length",
    "expected_order": ACTIVE_TIERS,
}


def resolve_tier(tier: str) -> str:
    """Return an active tier name, mapping the removed legacy XL tier."""
    normalized = str(tier).lower()
    if normalized == "xl":
        warnings.warn(
            "Legacy XL tier is no longer active; mapping it to Large. "
            "XL-shaped checkpoint tensors that do not fit Large are skipped.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "large"
    if normalized not in TIER_CONFIGS:
        raise ValueError(
            f"Unknown policy tier '{tier}'. Active tiers: "
            f"{', '.join(ACTIVE_TIERS)}")
    return normalized


def build_feature_visibility(tier: str) -> torch.Tensor:
    """Build the immutable 249-feature visibility mask for a policy tier."""
    tier = resolve_tier(tier)
    visible = torch.zeros(OBS_SIZE, dtype=torch.bool)
    for start, stop in FEATURE_VISIBILITY_RANGES[tier]:
        visible[start:stop] = True
    return visible


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

    Input:  [batch, frame_stack * 249]  (flat, from C++ frame stacking)
    Output: [batch, 3, 249]  (current, velocity, acceleration)

    Baked into ONNX — C++ feeds raw flat observations unchanged.
    """

    def __init__(self, frame_stack: int = 3):
        super().__init__()
        self.frame_stack = frame_stack
        self.obs_size = OBS_SIZE

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        batch = flat_obs.shape[0]

        # Reshape: [batch, N*249] → [batch, N, 249]
        frames = flat_obs.view(batch, self.frame_stack, self.obs_size)

        # Delta encoding. Frames are oldest-first: [t-2, t-1, t].
        current = frames[:, -1]                                         # t
        velocity = frames[:, -1] - frames[:, -2]                       # t - (t-1)
        acceleration = frames[:, -1] - 2.0 * frames[:, -2] + frames[:, -3]  # t - 2(t-1) + (t-2)

        # Stack as [batch, 3, 249]: channels = (current, velocity, acceleration).
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
                 attention_heads=4, gru_hidden=32, tier: str = "large"):
        super().__init__()
        self.tier = resolve_tier(tier)
        self.frame_stack = frame_stack
        self.backbone_hidden = backbone_hidden
        self.gru_hidden = gru_hidden
        self.register_buffer(
            "feature_visibility", build_feature_visibility(self.tier))
        behavior = BEHAVIOR_TIER_DEFINITIONS[self.tier]
        for name, size in (
            ("movement", MOVEMENT_ACTIONS),
            ("combat", COMBAT_ACTIONS),
            ("target", TARGET_ACTIONS),
        ):
            available = torch.zeros(size, dtype=torch.bool)
            available[list(behavior[f"{name}_actions"])] = True
            self.register_buffer(f"{name}_availability", available)

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

        # Stage 5: Independent one-pass policy heads on shared GRU features.
        self.move_head = layer_init(
            nn.Linear(gru_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(
            nn.Linear(gru_hidden, COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(
            nn.Linear(gru_hidden, TARGET_ACTIONS), std=0.01)

    def init_hidden(self, batch_size: int = 1, device=None):
        """Create zero-initialised hidden state."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(1, batch_size, self.gru_hidden, device=device)

    def _encode_features(self, obs):
        """Encode obs through delta → structured encoder → backbone."""
        # Normalization (when present) happens outside this module. Masking
        # here therefore prevents hidden features' normalized offsets from
        # leaking into a lower tier and is baked into the exported ONNX graph.
        frames = obs.reshape(-1, self.frame_stack, OBS_SIZE)
        frames = frames * self.feature_visibility.to(obs.dtype).view(1, 1, -1)
        deltas = self.delta(frames.reshape_as(obs))
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        embeddings_flat = self.encoder(channels_flat)
        embeddings = embeddings_flat.view(batch, 3 * self.encoder.channel_dim)
        return self.backbone(embeddings)  # [batch, backbone_hidden]

    def _heads(self, features):
        """Return independent logits from the same recurrent features."""
        movement = self.move_head(features).masked_fill(
            ~self.movement_availability, -1e8)
        combat = self.combat_head(features).masked_fill(
            ~self.combat_availability, -1e8)
        target = self.target_head(features).masked_fill(
            ~self.target_availability, -1e8)
        return movement, combat, target

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

    def forward_sequence(self, obs: torch.Tensor,
                         hidden: torch.Tensor = None):
        """Run complete episode sequences for recurrent distillation.

        Args:
            obs: [batch, time, frame_stack * OBS_SIZE]
            hidden: [1, batch, gru_hidden] or None
        """
        batch, steps, input_size = obs.shape
        if hidden is None:
            hidden = self.init_hidden(batch, obs.device)
        encoded = self._encode_features(
            obs.reshape(batch * steps, input_size))
        encoded = encoded.reshape(batch, steps, self.backbone_hidden)
        features, hidden_out = self.gru(encoded, hidden)
        m, c, t = self._heads(features)
        return m, c, t, hidden_out

    def select_actions(self, obs, masks, hidden=None):
        """Select masked greedy actions from independent heads."""
        m_logits, c_logits, t_logits, hidden_out = self(obs, hidden)
        m_logits = m_logits.masked_fill(~masks[0], -1e8)
        c_logits = c_logits.masked_fill(~masks[1], -1e8)
        t_logits = t_logits.masked_fill(~masks[2], -1e8)
        return (
            m_logits.argmax(dim=-1),
            c_logits.argmax(dim=-1),
            t_logits.argmax(dim=-1),
            hidden_out,
        )

    def sample_actions(self, obs, masks=None, hidden=None):
        """Sample masked actions independently from one recurrent pass."""
        m_logits, c_logits, t_logits, hidden_out = self(obs, hidden)
        if masks is not None:
            m_logits = m_logits.masked_fill(~masks[0], -1e8)
            c_logits = c_logits.masked_fill(~masks[1], -1e8)
            t_logits = t_logits.masked_fill(~masks[2], -1e8)
        m_dist = torch.distributions.Categorical(logits=m_logits)
        c_dist = torch.distributions.Categorical(logits=c_logits)
        t_dist = torch.distributions.Categorical(logits=t_logits)
        m, c, t = m_dist.sample(), c_dist.sample(), t_dist.sample()
        log_prob = (
            m_dist.log_prob(m) + c_dist.log_prob(c) + t_dist.log_prob(t)
        )
        return (m, c, t), log_prob, hidden_out


# ─────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────

def make_policy(tier: str, frame_stack: int = DEFAULT_FRAME_STACK) -> CombatPolicy:
    """Create a CombatPolicy for a specific tier."""
    tier = resolve_tier(tier)
    cfg = TIER_CONFIGS[tier]
    policy = CombatPolicy(
        frame_stack=frame_stack,
        entity_dim=cfg["entity_dim"],
        unique_dim=cfg["unique_dim"],
        backbone_hidden=cfg["backbone_hidden"],
        backbone_layers=cfg["backbone_layers"],
        attention_heads=cfg["attention_heads"],
        gru_hidden=cfg["gru_hidden"],
        tier=tier,
    )
    return policy


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
    raw_tier = ckpt.get("tier", "large")
    tier = resolve_tier(raw_tier)

    print(f"Loading teacher from: {checkpoint_path}")
    print(f"  Tier: {raw_tier} -> {tier}, Frame stack: {frame_stack}, "
          f"Stage: {ckpt.get('stage', '?')}, Step: {ckpt.get('step', '?')}")

    teacher = make_policy(tier, frame_stack=frame_stack).to(device)
    source_dict = ckpt.get("full_state_dict", ckpt.get("model_state_dict", {}))
    legacy_autoregressive = (
        ckpt.get("policy_contract") != "independent_heads_v1"
        and any(
            key.startswith(("move_embed.", "combat_proj.",
                            "combat_embed.", "target_proj."))
            for key in source_dict
        )
    )
    if legacy_autoregressive:
        raise RuntimeError(
            "Legacy autoregressive checkpoint cannot be used as a teacher "
            "or exported under the independent_heads_v1 action contract. "
            "Retrain an independent-head policy; legacy checkpoints may only "
            "be used as an explicit encoder warm start during PPO training.")

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

        if dst_key in {
                "feature_visibility", "movement_availability",
                "combat_availability", "target_availability"}:
            # Tier contracts are derived from current code, never checkpoint
            # payloads (which may come from another or legacy tier).
            skipped += 1
            continue

        if (legacy_autoregressive
                and dst_key.startswith((
                    "move_embed.", "combat_proj.", "combat_head.",
                    "combat_embed.", "target_proj.", "target_head.",
                ))):
            skipped += 1
            continue

        if dst_key in policy_state:
            if src_val.shape == policy_state[dst_key].shape:
                policy_state[dst_key] = src_val
                loaded += 1
            elif (not legacy_autoregressive
                    and dst_key in {
                        "combat_head.weight", "combat_head.bias"}
                    and src_val.shape[0] == COMBAT_ACTIONS - 1
                    and src_val.shape[1:] == policy_state[dst_key].shape[1:]):
                expanded = policy_state[dst_key].clone()
                expanded[:COMBAT_ACTIONS - 1] = src_val
                policy_state[dst_key] = expanded
                loaded += 1
                warnings.warn(
                    "Expanded legacy 8-action combat head to 9 actions; "
                    "Reposition starts from fresh initial weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                print(f"  Shape mismatch: {src_key} {src_val.shape} "
                      f"vs {dst_key} {policy_state[dst_key].shape}")
                skipped += 1

    teacher.load_state_dict(policy_state, strict=False)
    teacher.checkpoint_obs_normalizer_state = ckpt.get("obs_normalizer")
    teacher.eval()

    total_policy_keys = len(policy_state)
    print(f"  Loaded {loaded}/{total_policy_keys} policy tensors "
          f"(skipped {skipped} incompatible/non-policy tensors)")

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
        "policy_contract": "independent_heads_v1",
        "action_dims": (MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS),
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

def export_onnx(
    model: CombatPolicy,
    tier: str,
    output_dir: str,
    frame_stack: int = DEFAULT_FRAME_STACK,
    obs_normalizer=None,
) -> str:
    """Export a CombatPolicy to ONNX.

    The exported graph includes: normalisation → reshape → delta encode
    → structured group encode → backbone → policy heads.

    C++ feeds raw flat observations. Everything else is in the graph.
    """
    tier = resolve_tier(tier)
    model.eval().cpu()

    input_size = OBS_SIZE * frame_stack

    checkpoint_normalizer = getattr(
        model,
        "checkpoint_obs_normalizer_state",
        None,
    )

    if obs_normalizer is not None:
        normalizer_mean = obs_normalizer.mean
        normalizer_var = obs_normalizer.var
        normalizer_clip = getattr(
            obs_normalizer,
            "clip",
            5.0,
        )
        normalizer_epsilon = getattr(
            obs_normalizer,
            "epsilon",
            1e-8,
        )

    elif checkpoint_normalizer is not None:
        normalizer_mean = checkpoint_normalizer["mean"]
        normalizer_var = checkpoint_normalizer["var"]
        normalizer_clip = checkpoint_normalizer.get(
            "clip",
            5.0,
        )
        normalizer_epsilon = checkpoint_normalizer.get(
            "epsilon",
            1e-8,
        )

    else:
        normalizer_mean = None

    if normalizer_mean is not None:
        export_model = NormalizedPolicyWrapper(
            model,
            mean=normalizer_mean,
            var=normalizer_var,
            clip=normalizer_clip,
            epsilon=normalizer_epsilon,
        ).eval().cpu()

        print("  Baking observation normalizer into ONNX graph")

    else:
        export_model = model

    dummy = torch.randn(
        1,
        input_size,
        dtype=torch.float32,
    )

    dummy_hidden = torch.zeros(
        1,
        1,
        model.gru_hidden,
        dtype=torch.float32,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        f"Combat_{tier.capitalize()}.onnx",
    )

    torch.onnx.export(
        export_model,
        (dummy, dummy_hidden),
        path,

        input_names=[
            "observation",
            "hidden_in",
        ],

        output_names=[
            "movement_logits",
            "combat_logits",
            "target_logits",
            "hidden_out",
        ],

        dynamic_axes={
            "observation": {
                0: "batch_size",
            },
            "hidden_in": {
                1: "batch_size",
            },
            "movement_logits": {
                0: "batch_size",
            },
            "combat_logits": {
                0: "batch_size",
            },
            "target_logits": {
                0: "batch_size",
            },
            "hidden_out": {
                1: "batch_size",
            },
        },

        opset_version=17,

        # Use TorchScript exporter.
        # Important for GRU compatibility/parity test.
        dynamo=False,
    )

    # Consolidate to single file.
    try:
        import onnx

        onnx_model = onnx.load(
            path,
            load_external_data=True,
        )

        onnx.save(
            onnx_model,
            path,
            save_as_external_data=False,
        )

        data_path = path + ".data"

        if os.path.exists(data_path):
            os.remove(data_path)

    except ImportError:
        print(
            "  Warning: onnx package not installed "
            "— cannot consolidate."
        )

    size_kb = os.path.getsize(path) / 1024

    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"  Exported: {path} "
        f"({size_kb:.1f} KB, "
        f"{param_count:,} params)"
    )

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
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to verify production exports") from exc

    model.eval().cpu()
    input_size = OBS_SIZE * frame_stack

    checkpoint_normalizer = getattr(
        model, "checkpoint_obs_normalizer_state", None)
    if obs_normalizer is not None:
        normalizer_mean = obs_normalizer.mean
        normalizer_var = obs_normalizer.var
        normalizer_clip = getattr(obs_normalizer, "clip", 5.0)
        normalizer_epsilon = getattr(obs_normalizer, "epsilon", 1e-8)
    elif checkpoint_normalizer is not None:
        normalizer_mean = checkpoint_normalizer["mean"]
        normalizer_var = checkpoint_normalizer["var"]
        normalizer_clip = checkpoint_normalizer.get("clip", 5.0)
        normalizer_epsilon = checkpoint_normalizer.get("epsilon", 1e-8)
    else:
        normalizer_mean = None

    if normalizer_mean is not None:
        pt_model = NormalizedPolicyWrapper(
            model,
            mean=normalizer_mean,
            var=normalizer_var,
            clip=normalizer_clip,
            epsilon=normalizer_epsilon,
        ).eval().cpu()
    else:
        pt_model = model

    sess = ort.InferenceSession(onnx_path)
    generator = torch.Generator().manual_seed(12345)
    pt_hidden = torch.randn(
        1, 1, model.gru_hidden, generator=generator)
    ort_hidden = pt_hidden.numpy().copy()
    max_diffs = [0.0, 0.0, 0.0, 0.0]
    expected_shapes = (
        (1, MOVEMENT_ACTIONS),
        (1, COMBAT_ACTIONS),
        (1, TARGET_ACTIONS),
        (1, 1, model.gru_hidden),
    )

    # Exercise non-zero hidden input and recurrent chaining, not only one
    # stateless zero-hidden call.
    for _ in range(4):
        observation = torch.randn(
            1, input_size, generator=generator)
        with torch.no_grad():
            pt_out = pt_model(observation, pt_hidden)
        ort_out = sess.run(None, {
            "observation": observation.numpy(),
            "hidden_in": ort_hidden,
        })

        for index, expected_shape in enumerate(expected_shapes):
            pt_array = pt_out[index].numpy()
            ort_array = np.asarray(ort_out[index])
            if (pt_array.shape != expected_shape
                    or ort_array.shape != expected_shape):
                raise RuntimeError(
                    "ONNX output shape mismatch for output "
                    f"{index}: PyTorch={pt_array.shape}, "
                    f"ONNX={ort_array.shape}, expected={expected_shape}")
            if (not np.isfinite(pt_array).all()
                    or not np.isfinite(ort_array).all()):
                raise RuntimeError(
                    f"Non-finite values in export output {index}")
            diff = float(np.max(np.abs(pt_array - ort_array)))
            max_diffs[index] = max(max_diffs[index], diff)
        pt_hidden = pt_out[3]
        ort_hidden = ort_out[3]

    m_diff, c_diff, t_diff, h_diff = max_diffs
    max_diff = max(max_diffs)

    status = "PASS" if max_diff < 1e-4 else "FAIL"
    print(f"  Verify {status}: max diff = {max_diff:.6f} "
          f"(m={m_diff:.6f}, c={c_diff:.6f}, t={t_diff:.6f}, "
          f"h={h_diff:.6f})")
    if status == "FAIL":
        raise RuntimeError(
            f"ONNX round-trip verification failed (max diff {max_diff:.6f})")
    return True


# ─────────────────────────────────────────────────────────────────
#  Quick CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test structured policy and ONNX export")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--tier", type=str, default="large",
                        choices=ACTIVE_TIERS)
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
        teacher_path = export_onnx(
            teacher, teacher.tier, args.output_dir,
            frame_stack=teacher.frame_stack)
        verify_export(
            teacher, teacher_path, frame_stack=teacher.frame_stack)

    print("\nDone.")
