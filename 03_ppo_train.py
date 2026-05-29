"""
03_ppo_train.py — PPO fine-tuning using the Python combat simulation.

CHANGES FROM PREVIOUS VERSION
    1. Vectorized environments (8 parallel envs by default). This is the
       single biggest variance-reduction technique — with 8 envs you get
       ~8x more episodes per rollout, stabilising advantage estimates.
    2. Observation normalization (running mean/std). Prevents feature
       distribution shift across curriculum stages from shocking the
       network.
    3. Return normalization. Adapts to reward scale changes between stages.
    4. Proper truncation handling with terminal_observation from vec env.

USAGE
    # Start from BC checkpoint (recommended):
    python 03_ppo_train.py --bc_checkpoint checkpoints/bc_model.pt --stage 3

    # Start from scratch (slower but works):
    python 03_ppo_train.py --stage 1

    # Full curriculum run:
    python 03_ppo_train.py --curriculum

    # Control parallelism:
    python 03_ppo_train.py --stage 3 --num_envs 4

OUTPUTS
    checkpoints/ppo_stage{N}.pt   — checkpoint per curriculum stage
    checkpoints/ppo_final.pt      — final model
    runs/ppo_{timestamp}/         — TensorBoard logs
"""

import argparse
import os
import time
from collections import deque
from datetime import datetime

import random as _random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from combat_sim import CombatEnv, OBS_SIZE
from combat_extensions import make_extended_curriculum_env as make_curriculum_env

from combat_sim import MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from frame_stack import (
    FrameStackEnvWrapper, VecFrameStackEnv,
    stacked_obs_size, SINGLE_OBS_SIZE,
)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from dataclasses import dataclass

DEFAULT_FRAME_STACK = 3
DEFAULT_NUM_ENVS = 12


# ─────────────────────────────────────────────────────────────────
#  Observation Normalizer (running mean/std)
# ─────────────────────────────────────────────────────────────────

class RunningNormalizer:
    """Welford's online algorithm for running mean and variance.

    WHY THIS MATTERS
        In stages 1-2, spatial features (obstacle ring, cover ring, navmesh)
        are all constant (no obstacles → all 1.0). When stage 3 adds
        obstacles, these 25 features suddenly vary. The first linear layer
        has calibrated its weights for those inputs being constant. Without
        normalization, the activation distribution shifts violently and
        destabilises the whole network.

        Running normalization absorbs these shifts — the normalizer
        adapts its statistics over ~1000 steps, so the network sees
        gradually changing normalised inputs rather than a sudden cliff.

    USAGE
        normalizer = RunningNormalizer(obs_size)
        normalizer.update(obs_batch)                  # (N, obs_size)
        normed = normalizer.normalize(obs_batch)      # zero-mean, unit-variance
    """

    def __init__(self, shape: int, clip: float = 5.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Small initial count to avoid division by zero.
        self.clip = clip
        self.epsilon = epsilon

    def update(self, batch: np.ndarray):
        """Update running statistics with a batch of observations."""
        batch = batch.reshape(-1, self.mean.shape[0])
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        # Welford's parallel algorithm.
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to roughly zero mean, unit variance."""
        return np.clip(
            (obs - self.mean.astype(np.float32)) /
            np.sqrt(self.var.astype(np.float32) + self.epsilon),
            -self.clip, self.clip
        )

    def state_dict(self):
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state):
        self.mean = state["mean"].copy()
        self.var = state["var"].copy()
        self.count = state["count"]


# ─────────────────────────────────────────────────────────────────
#  Flat MLP Actor-Critic — architecture derived from combat_policy.py
# ─────────────────────────────────────────────────────────────────

"""
ActorCritic for PPO training.

SINGLE SOURCE OF TRUTH
    The actor side mirrors CombatPolicy from combat_policy.py exactly:
    same hidden sizes, same layer count, same activations, same logit
    scaling. This is enforced by importing TIER_CONFIGS and LOGIT_SCALE
    from combat_policy.py rather than redefining them here.

    When you change CombatPolicy's architecture, ActorCritic automatically
    matches — no manual sync needed.

KEY NAMING CONVENTION
    ActorCritic keys          CombatPolicy keys
    ──────────────────        ──────────────────
    actor_backbone.*      →   backbone.*
    move_head.*           →   move_head.*
    combat_head.*         →   combat_head.*
    target_head.*         →   target_head.*
    critic_backbone.*     →   (no equivalent)
    value_head.*          →   (no equivalent)

    save_checkpoint() stores both full_state_dict (ActorCritic keys)
    and policy_state_dict (CombatPolicy keys, actor_backbone→backbone).
    load_teacher_from_checkpoint() reads policy_state_dict.

OBSERVATION LAYOUT (198 floats per frame)
    [  0.. 20]  Self State          (21)
    [ 21.. 42]  Weapon State        (22)
    [ 43.. 49]  Archetype           ( 7)
    [ 50.. 69]  Primary Target      (20)
    [ 70..121]  Hostile Targets     (52) — 4 slots × 13
    [122..157]  Allied Robots       (36) — 3 slots × 12
    [158..165]  Spatial Ring        ( 8)
    [166..173]  Cover Assessment    ( 8)
    [174..181]  Threat Sensing      ( 8)
    [182..190]  Navmesh Viability   ( 9)
    [191..196]  Group Summary       ( 6)
    [197..197]  Spawn/Leash         ( 1)
"""

from combat_policy import (
    TIER_CONFIGS, LOGIT_SCALE, layer_init,
    CombatPolicy, make_policy, save_ppo_checkpoint,
    StructuredEncoder, DeltaEncoder,
    DEFAULT_FRAME_STACK as POLICY_FRAME_STACK,
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


# ─────────────────────────────────────────────────────────────────
#  ActorCritic (structured: delta encode → group encode → backbone)
# ─────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
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

        # Stage 2: Group encoders (separate for actor/critic — no gradient contamination).
        self.actor_encoder = StructuredEncoder(entity_dim, unique_dim)
        self.critic_encoder = StructuredEncoder(entity_dim, unique_dim)

        channel_dim = self.actor_encoder.channel_dim
        concat_dim = 3 * channel_dim  # 3 delta channels concatenated

        # Stage 3: Backbones (separate).
        self.actor_backbone = _build_backbone(concat_dim, backbone_hidden, backbone_layers)
        self.critic_backbone = _build_backbone(concat_dim, backbone_hidden, backbone_layers)

        # Policy heads.
        self.move_head = layer_init(nn.Linear(backbone_hidden, MOVEMENT_ACTIONS), std=0.01)
        self.combat_head = layer_init(nn.Linear(backbone_hidden, COMBAT_ACTIONS), std=0.01)
        self.target_head = layer_init(nn.Linear(backbone_hidden, TARGET_ACTIONS), std=0.01)

        # Value head.
        self.value_head = layer_init(nn.Linear(backbone_hidden, 1), std=1.0)

    def _encode(self, obs: torch.Tensor, encoder: StructuredEncoder) -> torch.Tensor:
        """Delta encode → group encode → concat channels."""
        deltas = self.delta(obs)  # [batch, 3, 198]
        batch = deltas.shape[0]
        channels_flat = deltas.view(batch * 3, OBS_SIZE)
        emb_flat = encoder(channels_flat)  # [batch*3, channel_dim]
        return emb_flat.view(batch, 3 * encoder.channel_dim)

    def _scaled_logits(self, raw):
        """Bound logits to [-LOGIT_SCALE, +LOGIT_SCALE] via tanh."""
        return torch.tanh(raw) * LOGIT_SCALE

    def forward(self, obs):
        actor_feat = self.actor_backbone(self._encode(obs, self.actor_encoder))
        critic_feat = self.critic_backbone(self._encode(obs, self.critic_encoder))

        m_logits = self._scaled_logits(self.move_head(actor_feat))
        c_logits = self._scaled_logits(self.combat_head(actor_feat))
        t_logits = self._scaled_logits(self.target_head(actor_feat))
        value = self.value_head(critic_feat)

        return m_logits, c_logits, t_logits, value

    def forward_inference(self, obs):
        """Policy only — no value head."""
        actor_feat = self.actor_backbone(self._encode(obs, self.actor_encoder))
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
        critic_feat = self.critic_backbone(self._encode(obs, self.critic_encoder))
        return self.value_head(critic_feat).squeeze(-1)

    # ═════════════════════════════════════════════════════════════
    #  Checkpoint Loading
    # ═════════════════════════════════════════════════════════════

    def load_from_ppo_checkpoint(self, ckpt_path: str, reinit_critic: bool = False):
        """Load from a PPO checkpoint.
        
        Args:
            ckpt_path: Path to checkpoint.
            reinit_critic: If True, skip loading critic weights (keep fresh
                random init). Use this for curriculum stage transitions — the
                stage N-1 critic's value estimates are systematically wrong
                for stage N, causing biased advantage estimates that degrade
                the policy. A fresh critic has high variance but zero bias.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if "full_state_dict" in ckpt:
            state = ckpt["full_state_dict"]
            
            # Filter out critic weights for fresh init on stage transitions.
            critic_keys = ["critic_encoder", "critic_backbone", "value_head"]
            if reinit_critic:
                before = len(state)
                state = {k: v for k, v in state.items()
                         if not any(c in k for c in critic_keys)}
                print(f"Fresh critic: dropped {before - len(state)} critic tensors "
                      f"(reinit_critic=True)")

            own_state = self.state_dict()

            # Partial load — match by key name and shape.
            loaded = 0
            skipped = 0
            for key in state:
                dst_key = key
                if dst_key in own_state and state[key].shape == own_state[dst_key].shape:
                    own_state[dst_key] = state[key]
                    loaded += 1
                else:
                    skipped += 1
            self.load_state_dict(own_state, strict=False)
            
            actor_loaded = sum(1 for k in state if any(a in k for a in 
                              ["actor_encoder", "actor_backbone", "move_head", "combat_head", "target_head"])
                              and k in own_state)
            print(f"Loaded {loaded}/{loaded + skipped} tensors "
                  f"(actor: {actor_loaded}, critic: {'fresh' if reinit_critic else 'restored'})")
            return True

        print(f"No full_state_dict in checkpoint")
        return False
    
# ─────────────────────────────────────────────────────────────────
#  PPO Hyperparameters
# ─────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.06
    vf_clip_range: float = 10.0     # Value function clipping — prevents
                                     # huge value updates at stage transitions.
    entropy_coef: float = 0.01       # Was 0.05. Start of entropy anneal range.
    entropy_coef_final: float = 0.002 # Entropy decays to this. Near-zero late
                                      # in training lets the policy converge.
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_steps: int = 196             # Steps per env per rollout.
                                     # Total transitions = num_steps × num_envs.
    mini_batch_size: int = 256
    update_epochs: int = 4           # Was 8. Reduced — action masking makes
                                     # each step more informative, so fewer
                                     # passes are needed. 8 with masking was
                                     # causing overfitting to rollout data.
    target_kl: float = 0.015         # KL early stopping. If approx KL exceeds
                                     # this, stop the epoch loop early. Prevents
                                     # catastrophic policy updates from bad batches.
    total_timesteps: int = 6_000_000
    eval_interval: int = 10_000
    save_interval: int = 50_000
    num_eval_episodes: int = 50
    eval_base_seed: int = 42
    normalize_obs: bool = True
    normalize_returns: bool = True
    revert_on_regression: bool = True  # Only revert on CATASTROPHIC regression —
                                       # normal fluctuations are healthy exploration.
    revert_patience: int = 80          # ~80 evals without improvement before reverting.
                                       # At eval every 100K steps, this is 8M steps of
                                       # exploration — enough for strategic shifts.
    revert_min_drop: float = 0.15      # Only revert if current win rate is more than
                                       # 15% BELOW the best. A drop from 40% to 35% is
                                       # normal exploration. A drop to 20% is collapse.


# ─────────────────────────────────────────────────────────────────
#  Vectorized Rollout Buffer
# ─────────────────────────────────────────────────────────────────

class VecRolloutBuffer:
    """Stores rollout data from N parallel environments.

    Layout: all arrays are [num_steps, num_envs, ...].
    For minibatch sampling, they're flattened to [num_steps * num_envs, ...].
    """

    def __init__(self, num_steps: int, num_envs: int, obs_size: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_size = obs_size
        self.total = num_steps * num_envs

        self.obs = np.zeros((num_steps, num_envs, obs_size), dtype=np.float32)
        self.m_acts = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.c_acts = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.t_acts = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)
        # Action masks (True = valid).
        self.m_masks = np.ones((num_steps, num_envs, MOVEMENT_ACTIONS), dtype=bool)
        self.c_masks = np.ones((num_steps, num_envs, COMBAT_ACTIONS), dtype=bool)
        self.t_masks = np.ones((num_steps, num_envs, TARGET_ACTIONS), dtype=bool)

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float):
        """Compute GAE for all envs. last_values: (num_envs,)."""
        gae = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def flatten(self):
        """Flatten [num_steps, num_envs] → [total] for minibatch sampling."""
        return {
            "obs": self.obs.reshape(self.total, self.obs_size),
            "m_acts": self.m_acts.reshape(self.total),
            "c_acts": self.c_acts.reshape(self.total),
            "t_acts": self.t_acts.reshape(self.total),
            "log_probs": self.log_probs.reshape(self.total),
            "advantages": self.advantages.reshape(self.total),
            "returns": self.returns.reshape(self.total),
            "values": self.values.reshape(self.total),
            "m_masks": self.m_masks.reshape(self.total, MOVEMENT_ACTIONS),
            "c_masks": self.c_masks.reshape(self.total, COMBAT_ACTIONS),
            "t_masks": self.t_masks.reshape(self.total, TARGET_ACTIONS),
        }

    def sample_minibatches(self, batch_size: int):
        flat = self.flatten()
        indices = np.random.permutation(self.total)

        for start in range(0, self.total, batch_size):
            end = start + batch_size
            if end > self.total:
                break
            idx = indices[start:end]
            yield {
                "obs": torch.from_numpy(flat["obs"][idx]),
                "m_acts": torch.from_numpy(flat["m_acts"][idx]),
                "c_acts": torch.from_numpy(flat["c_acts"][idx]),
                "t_acts": torch.from_numpy(flat["t_acts"][idx]),
                "old_log_probs": torch.from_numpy(flat["log_probs"][idx]),
                "advantages": torch.from_numpy(flat["advantages"][idx]),
                "returns": torch.from_numpy(flat["returns"][idx]),
                "old_values": torch.from_numpy(flat["values"][idx]),
                "m_masks": torch.from_numpy(flat["m_masks"][idx]),
                "c_masks": torch.from_numpy(flat["c_masks"][idx]),
                "t_masks": torch.from_numpy(flat["t_masks"][idx]),
            }


# ─────────────────────────────────────────────────────────────────
#  Return Normalizer
# ─────────────────────────────────────────────────────────────────

class ReturnNormalizer:
    """Running estimate of return variance for reward scaling.

    Divides rewards by sqrt(running_var(returns)) so the value function
    always sees roughly unit-variance targets. This prevents the value
    head from being overwhelmed when reward scale jumps between stages.
    """

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_return = 0.0
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 1e-4

    def update(self, rewards: np.ndarray, dones: np.ndarray):
        """Update with a batch of (rewards, dones) from one timestep."""
        for r, d in zip(rewards, dones):
            self.running_return = r + self.gamma * self.running_return * (1 - d)
            self.count += 1
            # Welford's online algorithm for mean and variance.
            delta = self.running_return - self.running_mean
            self.running_mean += delta / self.count
            delta2 = self.running_return - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / self.count

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        return rewards / (np.sqrt(max(self.running_var, 1e-6)) + self.epsilon)


# ─────────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────────

def train_ppo(
    stage: int = 3,
    archetype: str = "ranged",
    bc_checkpoint: str = None,
    output_dir: str = "checkpoints",
    total_timesteps: int = 6_000_000,
    frame_stack: int = DEFAULT_FRAME_STACK,
    num_envs: int = DEFAULT_NUM_ENVS,
    tier: str = "large",
):
    eval_history = []  # (step, reward, win_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PPOConfig(total_timesteps=total_timesteps)

    input_size = stacked_obs_size(frame_stack)
    batch_total = cfg.num_steps * num_envs  # Transitions per rollout.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/ppo_s{stage}_{archetype}_{timestamp}"
    writer = SummaryWriter(log_dir)
    os.makedirs(output_dir, exist_ok=True)

    tier_cfg = TIER_CONFIGS[tier]
    print(f"PPO Training — Stage {stage}, Archetype {archetype}")
    print(f"Device: {device}, Timesteps: {cfg.total_timesteps:,}")
    print(f"Tier: {tier} (entity={tier_cfg['entity_dim']}, unique={tier_cfg['unique_dim']}, "
          f"backbone={tier_cfg['backbone_hidden']}×{tier_cfg['backbone_layers']})")
    print(f"Frame stack: {frame_stack}, input size: {input_size}")
    print(f"Envs: {num_envs}, steps/env: {cfg.num_steps}, "
          f"batch: {batch_total} transitions/rollout")
    print(f"Logs: {log_dir}")

    # ── Create vectorized environment ────────────────────────────
    env_fns = [lambda s=stage, a=archetype: make_curriculum_env(s, a)
               for _ in range(num_envs)]
    vec_env = VecFrameStackEnv(env_fns, frame_stack=frame_stack)

    # ── Create model ─────────────────────────────────────────────
    model = ActorCritic(obs_size=input_size, tier=tier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    # ── Load checkpoint ──────────────────────────────────────────
    loaded_from_ppo = False
    obs_normalizer = RunningNormalizer(input_size) if cfg.normalize_obs else None
    return_normalizer = ReturnNormalizer(cfg.gamma) if cfg.normalize_returns else None

    if bc_checkpoint and os.path.exists(bc_checkpoint):
        ckpt = torch.load(bc_checkpoint, map_location=device, weights_only=False)
        is_ppo_checkpoint = "full_state_dict" in ckpt

        if is_ppo_checkpoint:
            # Detect stage transition: if checkpoint is from a different stage,
            # reinitialise the critic to avoid biased value estimates.
            ckpt_stage = ckpt.get("stage", stage)
            is_stage_transition = (ckpt_stage != stage)
            
            if is_stage_transition:
                print(f"Stage transition detected: checkpoint stage {ckpt_stage} → training stage {stage}")
                print(f"Reinitialising critic (fresh value function for new stage)")
            
            model.load_from_ppo_checkpoint(bc_checkpoint, reinit_critic=is_stage_transition)
            loaded_from_ppo = True

            # Restore normalizer state if saved.
            if obs_normalizer and "obs_normalizer" in ckpt:
                obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
                print(f"Restored observation normalizer state")

            print(f"Loaded PPO checkpoint: {bc_checkpoint} "
                  f"(stage {ckpt_stage}, step {ckpt.get('step', '?')})")
        else:
            model.load_policy_from_bc(bc_checkpoint)
            print(f"Warm-started from BC checkpoint: {bc_checkpoint}")

    # ── LR annealing (linear decay to 0) ─────────────────────────
    # Standard PPO practice. Without this, the agent oscillates
    # around good policies without converging.
    warmup_steps = 50_000 if loaded_from_ppo else 0
    total_rollouts = cfg.total_timesteps // batch_total

    def lr_lambda(rollout_step):
        # Linear anneal from 1.0 → 0.0 over total training.
        # If loading from checkpoint, brief warmup first.
        if warmup_steps > 0:
            # Convert warmup from timesteps to rollout steps.
            warmup_rollouts = max(1, warmup_steps // batch_total)
            if rollout_step < warmup_rollouts:
                return 0.2 + 0.8 * rollout_step / warmup_rollouts
        
        # Linear anneal for the rest of training.
        progress = min(1.0, rollout_step / max(total_rollouts, 1))
        return max(0.01, 1.0 - progress)  # Floor at 1% (was 5% — too high for
                                           # long runs, caused slow drift).

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    print(f"LR schedule: {cfg.lr:.1e} → {cfg.lr * 0.01:.1e} over "
          f"{total_rollouts} rollouts ({cfg.total_timesteps:,} steps)"
          + (f" with warmup over {warmup_steps:,} steps" if warmup_steps > 0 else ""))
    
    # ── Buffer ───────────────────────────────────────────────────
    buffer = VecRolloutBuffer(cfg.num_steps, num_envs, input_size)

    # ── Helper: extract batched masks from per-env info dicts ────
    def extract_masks(infos_list, device_=device):
        """Extract action masks from info dicts into batched tensors."""
        m = np.stack([info.get("action_mask", {}).get("m_mask",
                      np.ones(MOVEMENT_ACTIONS, dtype=bool))
                      for info in infos_list])
        c = np.stack([info.get("action_mask", {}).get("c_mask",
                      np.ones(COMBAT_ACTIONS, dtype=bool))
                      for info in infos_list])
        t = np.stack([info.get("action_mask", {}).get("t_mask",
                      np.ones(TARGET_ACTIONS, dtype=bool))
                      for info in infos_list])
        return (torch.from_numpy(m).to(device_),
                torch.from_numpy(c).to(device_),
                torch.from_numpy(t).to(device_))

    # ── Training state ───────────────────────────────────────────
    obs, initial_infos = vec_env.reset()  # (num_envs, input_size)
    current_masks = extract_masks(initial_infos)  # Initial masks
    global_step = 0
    episode_count = 0
    best_eval_win_rate = -1.0
    best_eval_reward = float('-inf')
    scheduler_step = 0
    consecutive_regressions = 0  # Counter for best-model reversion

    # Per-env episode tracking.
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    ep_components = [{} for _ in range(num_envs)]

    # Rolling window for reporting.
    recent_rewards = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)
    recent_wins = deque(maxlen=50)

    start_time = time.time()

    while global_step < cfg.total_timesteps:

        # ── Collect rollout ──────────────────────────────────────
        model.eval()
        rollout_episodes = 0

        for step in range(cfg.num_steps):
            # Normalise observations.
            if obs_normalizer:
                obs_normalizer.update(obs)
                obs_normed = obs_normalizer.normalize(obs)
            else:
                obs_normed = obs

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_normed).float().to(device)
                actions, log_probs, _, values = model.get_action_and_value(
                    obs_t, masks=current_masks)
                # actions: tuple of (num_envs,) tensors
                m_acts = actions[0].cpu().numpy()
                c_acts = actions[1].cpu().numpy()
                t_acts = actions[2].cpu().numpy()

            # Build action array for vec env: (num_envs, 3).
            actions_np = np.stack([m_acts, c_acts, t_acts], axis=1)

            next_obs, rewards, dones, truncateds, infos = vec_env.step(actions_np)

            # Update return normalizer and scale rewards.
            if return_normalizer:
                terminals = np.logical_or(dones, truncateds).astype(np.float32)
                return_normalizer.update(rewards, terminals)
                rewards_normed = return_normalizer.normalize(rewards)
            else:
                rewards_normed = rewards

            # ── Handle truncations: bootstrap value into reward ──
            for i in range(num_envs):
                if truncateds[i] and not dones[i]:
                    # Episode hit time limit but agent is still alive.
                    # Bootstrap V(terminal_obs) into the reward.
                    term_obs = infos[i]["terminal_observation"]
                    if obs_normalizer:
                        term_obs = obs_normalizer.normalize(term_obs)
                    with torch.no_grad():
                        term_t = torch.from_numpy(term_obs).float().unsqueeze(0).to(device)
                        term_val = model.get_value(term_t).cpu().item()
                    rewards_normed[i] += cfg.gamma * term_val

            # Store in buffer.
            buffer.obs[step] = obs_normed
            buffer.m_acts[step] = m_acts
            buffer.c_acts[step] = c_acts
            buffer.t_acts[step] = t_acts
            buffer.log_probs[step] = log_probs.cpu().numpy()
            buffer.rewards[step] = rewards_normed
            buffer.dones[step] = np.logical_or(dones, truncateds).astype(np.float32)
            buffer.values[step] = values.cpu().numpy()
            # Store action masks used for this step's action selection.
            buffer.m_masks[step] = current_masks[0].cpu().numpy()
            buffer.c_masks[step] = current_masks[1].cpu().numpy()
            buffer.t_masks[step] = current_masks[2].cpu().numpy()

            # Update masks for NEXT step from env infos.
            current_masks = extract_masks(infos)

            # ── Per-env episode accounting ───────────────────────
            ep_rewards += rewards  # Track raw rewards (not normalised).
            ep_lengths += 1

            for i in range(num_envs):
                for key, val in infos[i].items():
                    if isinstance(val, (int, float)) and key not in ("terminal_observation",):
                        ep_components[i][key] = ep_components[i].get(key, 0.0) + val

            for i in range(num_envs):
                if dones[i] or truncateds[i]:
                    # Log this episode.
                    writer.add_scalar("rollout/episode_reward", ep_rewards[i], global_step)
                    writer.add_scalar("rollout/episode_length", ep_lengths[i], global_step)

                    # [Fix] Read win flag from info (set during step, before
                    # auto-reset clears the targets). The old code checked
                    # vec_env.envs[i].targets which are already reset to a
                    # new episode by VecFrameStackEnv.step().
                    is_win = bool(infos[i].get("is_win", False))
                    writer.add_scalar("rollout/win", float(is_win), global_step)

                    for key, val in ep_components[i].items():
                        if abs(val) > 1e-6:
                            writer.add_scalar(f"reward/{key}", val, global_step)

                    recent_rewards.append(ep_rewards[i])
                    recent_lengths.append(ep_lengths[i])
                    recent_wins.append(float(is_win))

                    episode_count += 1
                    rollout_episodes += 1
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0
                    ep_components[i] = {}

            obs = next_obs
            global_step += num_envs

        # ── Compute last values for GAE ──────────────────────────
        if obs_normalizer:
            obs_normed = obs_normalizer.normalize(obs)
        else:
            obs_normed = obs

        with torch.no_grad():
            obs_t = torch.from_numpy(obs_normed).float().to(device)
            last_values = model.get_value(obs_t).cpu().numpy()

        buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

        # ── PPO Update ───────────────────────────────────────────
        model.train()

        # Normalise advantages (across all envs and steps).
        flat_adv = buffer.advantages.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        buffer.advantages = flat_adv.reshape(cfg.num_steps, num_envs)

        # Anneal entropy coefficient: linear decay from initial to final.
        ent_progress = min(1.0, global_step / max(cfg.total_timesteps, 1))
        current_ent_coef = (cfg.entropy_coef
                            + (cfg.entropy_coef_final - cfg.entropy_coef) * ent_progress)

        total_pg_loss = 0; total_v_loss = 0; total_ent = 0
        total_clip_frac = 0; total_approx_kl = 0; n_updates = 0
        kl_early_stopped = False

        for epoch in range(cfg.update_epochs):
            if kl_early_stopped:
                break
            for batch in buffer.sample_minibatches(cfg.mini_batch_size):
                b_obs = batch["obs"].to(device)
                b_m = batch["m_acts"].to(device)
                b_c = batch["c_acts"].to(device)
                b_t = batch["t_acts"].to(device)
                b_old_lp = batch["old_log_probs"].to(device)
                b_adv = batch["advantages"].to(device)
                b_ret = batch["returns"].to(device)
                b_old_val = batch["old_values"].to(device)
                b_masks = (batch["m_masks"].to(device),
                           batch["c_masks"].to(device),
                           batch["t_masks"].to(device))

                new_lp, entropy, new_val = model.evaluate_actions(
                    b_obs, b_m, b_c, b_t, masks=b_masks)

                # Policy loss (clipped).
                ratio = (new_lp - b_old_lp).exp()
                pg_loss1 = -b_adv * ratio
                pg_loss2 = -b_adv * ratio.clamp(
                    1 - cfg.clip_range, 1 + cfg.clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with clipping.
                v_clipped = b_old_val + (new_val - b_old_val).clamp(
                    -cfg.vf_clip_range, cfg.vf_clip_range)
                v_loss1 = (new_val - b_ret) ** 2
                v_loss2 = (v_clipped - b_ret) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Entropy bonus (annealed).
                ent_loss = -entropy.mean()

                loss = (pg_loss
                        + cfg.value_coef * v_loss
                        + current_ent_coef * ent_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                # Track diagnostics.
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean()
                    # Approximate KL divergence (Schulman's approximation).
                    approx_kl = (0.5 * (new_lp - b_old_lp).pow(2)).mean()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += entropy.mean().item()
                total_clip_frac += clip_frac.item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

                # KL early stopping: if policy has moved too far, stop
                # updating to prevent catastrophic drift.
                if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl * 1.5:
                    kl_early_stopped = True
                    break

        # Step scheduler.
        scheduler_step += 1
        scheduler.step()

        # ── Logging ──────────────────────────────────────────────
        nu = max(n_updates, 1)
        writer.add_scalar("train/policy_loss", total_pg_loss / nu, global_step)
        writer.add_scalar("train/value_loss", total_v_loss / nu, global_step)
        writer.add_scalar("train/entropy", total_ent / nu, global_step)
        writer.add_scalar("train/clip_fraction", total_clip_frac / nu, global_step)
        writer.add_scalar("train/approx_kl", total_approx_kl / nu, global_step)
        writer.add_scalar("train/entropy_coef", current_ent_coef, global_step)
        writer.add_scalar("train/kl_early_stopped", float(kl_early_stopped), global_step)
        writer.add_scalar("train/episodes_total", episode_count, global_step)
        writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)

        if len(recent_rewards) > 0:
            writer.add_scalar("rollout/mean_reward_50ep", np.mean(recent_rewards), global_step)
            writer.add_scalar("rollout/mean_length_50ep", np.mean(recent_lengths), global_step)
            writer.add_scalar("rollout/win_rate_50ep", np.mean(recent_wins), global_step)

        writer.flush()

        sps = global_step / max(time.time() - start_time, 1)
        mean_r = np.mean(recent_rewards) if recent_rewards else 0
        win_r = np.mean(recent_wins) if recent_wins else 0
        print(f"Step {global_step:>8,}/{cfg.total_timesteps:,} | "
              f"Ep: {episode_count} ({rollout_episodes}/rollout) | "
              f"R(50): {mean_r:+.1f} | "
              f"Win: {win_r:.0%} | "
              f"PG: {total_pg_loss/nu:.4f} | "
              f"VL: {total_v_loss/nu:.4f} | "
              f"Ent: {total_ent/nu:.2f} | "
              f"Clip: {total_clip_frac/nu:.2f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
              f"SPS: {sps:.0f}")

        # ── Periodic evaluation ──────────────────────────────────
        if global_step % cfg.eval_interval < batch_total:
            eval_stats = evaluate(model, stage, archetype,
                                  cfg.num_eval_episodes, device,
                                  frame_stack, obs_normalizer,
                                  base_seed=cfg.eval_base_seed)
            writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], global_step)
            writer.add_scalar("eval/std_reward", eval_stats["std_reward"], global_step)
            writer.add_scalar("eval/win_rate", eval_stats["win_rate"], global_step)
            writer.add_scalar("eval/mean_kills", eval_stats["mean_kills"], global_step)
            writer.add_scalar("eval/mean_length", eval_stats["mean_length"], global_step)

            # ── Live correlation plot ────────────────────────────
            eval_history.append((global_step, eval_stats["mean_reward"],
                                 eval_stats["win_rate"]))

            if len(eval_history) >= 3:
                hist = eval_history
                rewards = [h[1] for h in hist]
                winrates = [h[2] for h in hist]
                progress = np.linspace(0, 1, len(hist))

                fig, ax = plt.subplots(1, 1, figsize=(6, 1.75), dpi = 250)
                sc = ax.scatter(rewards, winrates, c=progress,
                                cmap="viridis", s=20, edgecolors="white",
                                linewidths=0.5, zorder=3)

                # Regression line.
                r = np.array(rewards)
                w = np.array(winrates)
                coeffs = np.polyfit(r, w, 1)
                r_sorted = np.sort(r)
                ax.plot(r_sorted, np.poly1d(coeffs)(r_sorted),
                        "--", color="red", alpha=0.6, linewidth=1.5)
                corr = np.corrcoef(r, w)[0, 1]

                ax.set_xlabel("Reward", fontsize=10, fontweight="bold")
                ax.set_ylabel("Win Rate", fontsize=10, fontweight="bold")
                ax.set_title(f"r = {corr:.3f}", fontsize=12, fontweight="bold")
                ax.set_ylim(-0.05, 1.05)
                ax.tick_params(labelsize=11)
                ax.grid(True, alpha=0.3)
                fig.tight_layout(pad=0.5)

                writer.add_figure("eval/reward_vs_winrate", fig, global_step,
                    close=True)

                # Also log the correlation coefficient as a scalar.
                writer.add_scalar("eval/reward_winrate_corr", corr, global_step)

            writer.flush()

            print(f"  Eval ({cfg.num_eval_episodes} ep, seeded): "
                  f"reward={eval_stats['mean_reward']:.1f} ±{eval_stats['reward_ci95']:.1f}, "
                  f"win={eval_stats['win_rate']:.0%}, "
                  f"kills={eval_stats['mean_kills']:.1f}, "
                  f"len={eval_stats['mean_length']:.0f}")

            # ── Best checkpoint selection ─────────────────────────
            # Win rate is primary (did the agent actually win?).
            # Reward is secondary (HOW did it win?).
            #
            # A 70% win rate at reward=150 means efficient, aggressive
            # play. A 70% win rate at reward=80 means camping and
            # barely scraping wins. Both win the same, but the first
            # policy generalises better and is more fun to play against.
            #
            # Rules:
            #   1. Win rate improved by >1% → always save (reward irrelevant)
            #   2. Win rate roughly equal (±1%) but reward improved by >5
            #      → save (same wins, better play quality)
            #   3. Otherwise → don't save

            wr = eval_stats["win_rate"]
            mr = eval_stats["mean_reward"]

            improved = False
            reason = ""

            if wr > best_eval_win_rate + 0.01:
                # Rule 1: win rate meaningfully improved.
                improved = True
                reason = f"win rate: {best_eval_win_rate:.0%} → {wr:.0%}"
            elif (wr >= best_eval_win_rate - 0.01
                    and mr > best_eval_reward + 5.0):
                # Rule 2: same win rate, better play quality.
                improved = True
                reason = (f"same win rate ({wr:.0%}), "
                          f"reward: {best_eval_reward:.1f} → {mr:.1f}")

            if improved:
                best_eval_win_rate = wr
                best_eval_reward = mr
                path = os.path.join(output_dir, f"ppo_stage{stage}_best.pt")
                save_checkpoint(model, optimizer, path, stage, archetype,
                                global_step, obs_normalizer)
                print(f"  → New best model saved ({reason})")
                consecutive_regressions = 0
            else:
                consecutive_regressions += 1

            # Revert to best model ONLY on catastrophic regression.
            # Normal fluctuations (40% → 35% → 42%) are healthy exploration.
            # Only revert when the policy has truly collapsed (e.g. 40% → 20%).
            current_wr = eval_stats["win_rate"]
            has_collapsed = (best_eval_win_rate - current_wr) > cfg.revert_min_drop
            if (cfg.revert_on_regression
                    and consecutive_regressions >= cfg.revert_patience
                    and has_collapsed
                    and best_eval_win_rate > 0.05):
                best_path = os.path.join(output_dir, f"ppo_stage{stage}_best.pt")
                if os.path.exists(best_path):
                    print(f"  ⚠ Win rate collapsed: current={current_wr:.0%} "
                          f"vs best={best_eval_win_rate:.0%} "
                          f"(>{cfg.revert_min_drop:.0%} drop, "
                          f"{consecutive_regressions} evals). "
                          f"Reverting model weights only.")
                    model.load_from_ppo_checkpoint(best_path)
                    # DON'T revert optimizer — keep the exploration momentum.
                    # Reverting optimizer + model sends training down the same
                    # path that led to the current best, creating a loop.
                    # Fresh optimizer momentum from the reverted weights lets
                    # training take a different trajectory.
                    if obs_normalizer:
                        ckpt = torch.load(best_path, map_location=device, weights_only=False)
                        if "obs_normalizer" in ckpt:
                            obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
                    consecutive_regressions = 0

        # ── Periodic save ────────────────────────────────────────
        if global_step % cfg.save_interval < batch_total:
            path = os.path.join(output_dir, f"ppo_stage{stage}.pt")
            save_checkpoint(model, optimizer, path, stage, archetype,
                            global_step, obs_normalizer)

    # Final save.
    path = os.path.join(output_dir, f"ppo_stage{stage}_final.pt")
    save_checkpoint(model, optimizer, path, stage, archetype,
                    global_step, obs_normalizer)
    print(f"\nTraining complete. Best eval win rate: {best_eval_win_rate:.0%}")

    vec_env.close()
    writer.close()


def evaluate(model, stage, archetype, num_episodes, device,
             frame_stack=DEFAULT_FRAME_STACK, obs_normalizer=None,
             base_seed=42):
    """Run evaluation with deterministic scenarios.

    Each episode gets a fixed seed: base_seed + episode_index.
    Eval at step 10K sees the EXACT same 50 arenas as eval at step 20K.
    The only thing that changes between evals is the model weights,
    so reward/win rate differences reflect genuine improvement, not luck.
    """
    raw_env = make_curriculum_env(stage, archetype)
    env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)
    model.eval()

    rewards = []
    lengths = []
    wins = []
    kills = []

    for ep_idx in range(num_episodes):
        # Seed BEFORE reset — controls arena, spawns, target composition.
        _random.seed(base_seed + ep_idx)
        np.random.seed(base_seed + ep_idx)
        torch.manual_seed(base_seed + ep_idx)

        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0
        done = False
        num_targets = len(raw_env.targets)

        while not done:
            if obs_normalizer:
                obs_normed = obs_normalizer.normalize(obs)
            else:
                obs_normed = obs

            # Build action mask (same as training — prevents invalid actions).
            mask_dict = raw_env.build_action_mask()
            m_mask = torch.from_numpy(mask_dict["m_mask"]).unsqueeze(0).to(device)
            c_mask = torch.from_numpy(mask_dict["c_mask"]).unsqueeze(0).to(device)
            t_mask = torch.from_numpy(mask_dict["t_mask"]).unsqueeze(0).to(device)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_normed).float().unsqueeze(0).to(device)
                m_l, c_l, t_l, _ = model(obs_t)
                # Apply masks before argmax — matches training pipeline.
                m_l = m_l.masked_fill(~m_mask, -1e8)
                c_l = c_l.masked_fill(~c_mask, -1e8)
                t_l = t_l.masked_fill(~t_mask, -1e8)
                m = m_l.argmax(1).item()
                c = c_l.argmax(1).item()
                t = t_l.argmax(1).item()

            obs, reward, done, truncated, _ = env.step(np.array([m, c, t]))
            ep_reward += reward
            ep_length += 1
            if truncated:
                break

        targets_killed = sum(1 for t in raw_env.targets if not t.alive)
        is_win = targets_killed == num_targets

        rewards.append(ep_reward)
        lengths.append(ep_length)
        wins.append(float(is_win))
        kills.append(targets_killed)

    env.close()
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "win_rate": np.mean(wins),
        "mean_kills": np.mean(kills),
        "reward_ci95": 1.96 * np.std(rewards) / max(np.sqrt(len(rewards)), 1),
    }


def save_checkpoint(model, optimizer, path, stage, archetype, step,
                    obs_normalizer=None):
    save_ppo_checkpoint(
        model, optimizer, path,
        stage=stage, archetype=archetype, step=step,
        frame_stack=DEFAULT_FRAME_STACK,
        tier=getattr(model, "tier", "large"),
        obs_normalizer=obs_normalizer,
    )


# ─────────────────────────────────────────────────────────────────
#  Curriculum Runner
# ─────────────────────────────────────────────────────────────────

def run_curriculum(archetype="ranged", bc_checkpoint=None, output_dir="checkpoints",
                   frame_stack=DEFAULT_FRAME_STACK, num_envs=DEFAULT_NUM_ENVS,
                   tier="large"):
    """Run all 7 curriculum stages sequentially."""

    stage_timesteps = {
        1: 50_000,       # 1v1 no obstacles — learn to approach and shoot
        2: 100_000,       # 1v1 with obstacles — learn navigation
        3: 2_000_000,       # 1v2 — learn target switching
        4: 4_000_000,     # 1v2 with obstacles, 2x HP — cover + weapon management
        5: 10_000_000,     # 2v3 — multi-target, ally coordination
        6: 20_000_000,    # 2v3 full arena — complex navigation
        7: 30_000_000,    # 2v4 full arena — everything together
    }

    current_checkpoint = bc_checkpoint

    for stage in range(1, 8):
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage}/7")
        print(f"{'='*60}")

        train_ppo(
            stage=stage, archetype=archetype,
            bc_checkpoint=current_checkpoint,
            output_dir=output_dir,
            total_timesteps=stage_timesteps[stage],
            frame_stack=frame_stack,
            num_envs=num_envs,
            tier=tier,
        )

        best_path = os.path.join(output_dir, f"ppo_stage{stage}_best.pt")
        final_path = os.path.join(output_dir, f"ppo_stage{stage}_final.pt")

        if os.path.exists(best_path):
            current_checkpoint = best_path
            print(f"Stage {stage} complete. Next stage loads BEST: {current_checkpoint}")
        else:
            current_checkpoint = final_path
            print(f"Stage {stage} complete. No best found, using final: {current_checkpoint}")


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training for combat AI")
    parser.add_argument("--stage", type=int, default=3,
                        help="Curriculum stage (1-7)")
    parser.add_argument("--archetype", type=str, default="ranged")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC model checkpoint for warm-start")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--timesteps", type=int, default=6_000_000)
    parser.add_argument("--frame_stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--num_envs", type=int, default=DEFAULT_NUM_ENVS,
                        help=f"Number of parallel environments (default {DEFAULT_NUM_ENVS})")
    parser.add_argument("--tier", type=str, default="large",
                        choices=list(TIER_CONFIGS.keys()),
                        help="Model tier (architecture size)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Run all 7 stages sequentially")
    args = parser.parse_args()

    if args.curriculum:
        run_curriculum(args.archetype, args.bc_checkpoint, args.output_dir,
                       args.frame_stack, args.num_envs, args.tier)
    else:
        train_ppo(args.stage, args.archetype, args.bc_checkpoint,
                  args.output_dir, args.timesteps, args.frame_stack, args.num_envs,
                  args.tier)