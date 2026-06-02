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

OBS_SIZE = 231
MOVEMENT_ACTIONS = 9
COMBAT_ACTIONS = 7
TARGET_ACTIONS = 5

DEFAULT_FRAME_STACK = 3

# Observation layout: feature group boundaries within one 215-float frame.
# Unique features = everything EXCEPT hostile/ally entity slots.
_HOSTILE_START = 70                        # 4 slots x 13 features
_HOSTILE_SLOTS = 4
_HOSTILE_SLOT_SIZE = 13
_ALLY_START = 122                          # 3 slots x 12 features
_ALLY_SLOTS = 3
_ALLY_SLOT_SIZE = 12
_UNIQUE_SIZE = 143                         # 70 (self+weapon+arch+target) + 73 (spatial16/cover16/threat/nav/metrics/arc)

# Logit bounding (same as before).
LOGIT_SCALE = 1.0


# ─────────────────────────────────────────────────────────────────
#  Tier Configurations
# ─────────────────────────────────────────────────────────────────

TIER_CONFIGS = {
    "micro":  dict(entity_dim=8,  unique_dim=16, backbone_hidden=32,  backbone_layers=1),
    "small":  dict(entity_dim=12, unique_dim=24, backbone_hidden=48,  backbone_layers=1),
    "medium": dict(entity_dim=16, unique_dim=32, backbone_hidden=64,  backbone_layers=2),
    "large":  dict(entity_dim=16, unique_dim=32, backbone_hidden=96,  backbone_layers=2),
    "xl":     dict(entity_dim=24, unique_dim=48, backbone_hidden=128, backbone_layers=3),
}


# ─────────────────────────────────────────────────────────────────
#  Initialisation Helpers
# ─────────────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ─────────────────────────────────────────────────────────────────
#  Structured Encoder (shared building block)
# ─────────────────────────────────────────────────────────────────

class StructuredEncoder(nn.Module):
    """Encodes one 231-float frame into a compact embedding.

    Splits features into unique/hostile/ally/threat groups, encodes each
    with weight-shared layers, and concatenates the results.
    """

    def __init__(self, entity_dim: int = 16, unique_dim: int = 32):
        super().__init__()
        self.entity_dim = entity_dim
        self.unique_dim = unique_dim
        
        # New permutation-invariant threat dimension
        self.threat_dim = entity_dim 
        
        # Total output dimensions automatically scales (backbones adapt dynamically)
        self.channel_dim = unique_dim + entity_dim + entity_dim + self.threat_dim

        # Unique features tracking self and global metrics (127 inputs total)
        self.unique_encoder = nn.Sequential(
            layer_init(nn.Linear(_UNIQUE_SIZE, unique_dim)),
            nn.GELU(),
        )

        # Shared hostile slot encoder (13 → entity_dim)
        self.hostile_encoder = nn.Sequential(
            layer_init(nn.Linear(_HOSTILE_SLOT_SIZE, entity_dim)),
            nn.GELU(),
        )

        # Shared ally slot encoder (12 → entity_dim)
        self.ally_encoder = nn.Sequential(
            layer_init(nn.Linear(_ALLY_SLOT_SIZE, entity_dim)),
            nn.GELU(),
        )

        # ─── Shared Dynamic Threat (Projectile) Encoder ───
        # Each projectile threat slot is parameterized by: (distance, heading_x, heading_y)
        self.threat_slot_encoder = nn.Sequential(
            layer_init(nn.Linear(3, self.threat_dim)),
            nn.GELU(),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        batch = frame.shape[0]

        # 1. Unique features (retains absolute ranking and system metrics)
        # Includes Self, Weapon, Archetype, Primary Target, plus Spatial, Cover Height,
        # Threat 1, Navmesh, Group, Spawn, Threat 2 & 3, threat counts, weapon
        # capabilities, ammo, targets killed, and arc clearance.
        unique_feats = torch.cat([
            frame[:, 0:70],
            frame[:, 158:OBS_SIZE],
        ], dim=-1)  # [batch, 143] (was 127 with 8 rays)
        unique_emb = self.unique_encoder(unique_feats)  # [batch, unique_dim]

        # 2. Shared Hostile Encoder
        hostile_feats = frame[:, _HOSTILE_START:_HOSTILE_START + _HOSTILE_SLOTS * _HOSTILE_SLOT_SIZE]
        hostile_feats = hostile_feats.view(batch, _HOSTILE_SLOTS, _HOSTILE_SLOT_SIZE)
        hostile_flat = hostile_feats.reshape(batch * _HOSTILE_SLOTS, _HOSTILE_SLOT_SIZE)
        hostile_embs = self.hostile_encoder(hostile_flat).view(batch, _HOSTILE_SLOTS, self.entity_dim)
        hostile_pooled = hostile_embs.max(dim=1).values  # [batch, entity_dim]

        # 3. Shared Ally Encoder
        ally_feats = frame[:, _ALLY_START:_ALLY_START + _ALLY_SLOTS * _ALLY_SLOT_SIZE]
        ally_feats = ally_feats.view(batch, _ALLY_SLOTS, _ALLY_SLOT_SIZE)
        ally_flat = ally_feats.reshape(batch * _ALLY_SLOTS, _ALLY_SLOT_SIZE)
        ally_embs = self.ally_encoder(ally_flat).view(batch, _ALLY_SLOTS, self.entity_dim)
        ally_pooled = ally_embs.max(dim=1).values  # [batch, entity_dim]

        # 4. ─── Shared Projectile Threat Encoder with Symmetric Max-Pooling ───
        # Extract features for Threat 1, 2, and 3: (distance, heading_x, heading_y)
        # Indices shifted +16 from 8→16 spatial ring expansion.
        # Nearest projectile Threat 1 is at index 190 (dist), 192 (dirX), 193 (dirY) of the Threat Sensing block
        t1 = torch.stack([frame[:, 190], frame[:, 192], frame[:, 193]], dim=-1) # nearest
        # Second-nearest Threat 2 is at index 214 (dist), 215 (dirX), 216 (dirY)
        t2 = torch.stack([frame[:, 214], frame[:, 215], frame[:, 216]], dim=-1) # second-nearest
        # Third-nearest Threat 3 is at index 217 (dist), 218 (dirX), 219 (dirY)
        t3 = torch.stack([frame[:, 217], frame[:, 218], frame[:, 219]], dim=-1) # third-nearest
        
        # Reshape to slot tensor: [batch, 3_slots, 3_features]
        threats_feats = torch.stack([t1, t2, t3], dim=1)
        threats_flat = threats_feats.reshape(batch * 3, 3)
        threats_embs = self.threat_slot_encoder(threats_flat).view(batch, 3, self.threat_dim)
        
        # Apply max pooling along slot dimension for true translation & swap invariance
        threats_pooled = threats_embs.max(dim=1).values  # [batch, threat_dim]

        # 5. Concatenate everything adaptively (channels are packed cleanly)
        return torch.cat([unique_emb, hostile_pooled, ally_pooled, threats_pooled], dim=-1)

# ─────────────────────────────────────────────────────────────────
#  Delta Encoding Module (no learnable params)
# ─────────────────────────────────────────────────────────────────

class DeltaEncoder(nn.Module):
    """Reshapes flat frame-stacked input and computes temporal deltas.

    Input:  [batch, frame_stack * 231]  (flat, from C++ frame stacking)
    Output: [batch, 3, 231]  (current, velocity, acceleration)

    Baked into ONNX — C++ feeds raw flat observations unchanged.
    """

    def __init__(self, frame_stack: int = 3):
        super().__init__()
        self.frame_stack = frame_stack
        self.obs_size = OBS_SIZE

    def forward(self, flat_obs: torch.Tensor) -> torch.Tensor:
        batch = flat_obs.shape[0]

        # Reshape: [batch, N*231] → [batch, N, 231]
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

    Replaces the flat MLP CombatPolicy. Same external interface:
    forward(obs) → (movement_logits, combat_logits, target_logits).

    Internally: delta encode → group encode → backbone → heads.
    Everything bakes into the ONNX graph.
    """

    def __init__(self, frame_stack: int = DEFAULT_FRAME_STACK,
                 entity_dim: int = 16, unique_dim: int = 32,
                 backbone_hidden: int = 96, backbone_layers: int = 2):
        super().__init__()
        self.frame_stack = frame_stack

        # Stage 1: Delta encoding (no params).
        self.delta = DeltaEncoder(frame_stack)

        # Stage 2: Structured group encoder (shared across 3 delta channels).
        self.encoder = StructuredEncoder(entity_dim, unique_dim)
        channel_dim = self.encoder.channel_dim
        concat_dim = 3 * channel_dim  # 3 delta channels concatenated

        # Stage 3: Policy backbone.
        backbone_layers_list = []
        in_dim = concat_dim
        for i in range(backbone_layers):
            out_dim = backbone_hidden if i < backbone_layers - 1 else backbone_hidden
            backbone_layers_list.append(layer_init(nn.Linear(in_dim, out_dim)))
            if i == 0:
                backbone_layers_list.append(nn.LayerNorm(out_dim))
            backbone_layers_list.append(nn.GELU())
            in_dim = out_dim
        self.backbone = nn.Sequential(*backbone_layers_list)

        # Policy heads.
        self.move_head = layer_init(nn.Linear(backbone_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(nn.Linear(backbone_hidden, COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(nn.Linear(backbone_hidden, TARGET_ACTIONS), std=0.01)

    def forward(self, obs: torch.Tensor):
        """Forward pass.

        Args:
            obs: [batch, frame_stack * 215] flat observations from C++.
        Returns:
            (movement_logits, combat_logits, target_logits) — each [batch, N].
        """
        # Stage 1: Reshape + delta encode → [batch, 3, 215]
        deltas = self.delta(obs)  # [batch, 3, 215]

        # Stage 2: Encode each delta channel through the shared group encoder.
        batch = deltas.shape[0]
        # Flatten channels: [batch*3, 215] so encoder processes all at once.
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        embeddings_flat = self.encoder(channels_flat)  # [batch*3, channel_dim]
        embeddings = embeddings_flat.view(batch, 3 * self.encoder.channel_dim)

        # Stage 3: Backbone → heads.
        features = self.backbone(embeddings)
        m = torch.tanh(self.move_head(features)) * LOGIT_SCALE
        c = torch.tanh(self.combat_head(features)) * LOGIT_SCALE
        t = torch.tanh(self.target_head(features)) * LOGIT_SCALE
        return m, c, t


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

    def forward(self, obs):
        normed = torch.clamp((obs - self.obs_mean) / self.obs_std,
                             -self.clip, self.clip)
        return self.policy(normed)


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

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"Combat_{tier.capitalize()}.onnx")

    torch.onnx.export(
        export_model, dummy, path,
        input_names=["observation"],
        output_names=["movement_logits", "combat_logits", "target_logits"],
        dynamic_axes={"observation": {0: "batch_size"}},
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
        pt_m, pt_c, pt_t = pt_model(dummy)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"observation": dummy.numpy()})

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