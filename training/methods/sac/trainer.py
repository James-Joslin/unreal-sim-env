"""
trainer.py — Discrete Soft Actor-Critic for multi-head categorical actions.

Implements Christodoulou (2019) discrete SAC with:
    - Twin Q-networks (reduces positive bias in target estimation)
    - Per-head auto-tuned entropy temperature (α_m, α_c, α_t)
    - Polyak-averaged target networks
    - Action masking in policy and Q-value computation
    - Additive Q-value decomposition across the 3 action heads

KEY DIFFERENCE FROM PPO:
    Off-policy. The replay buffer decorrelates samples, so each gradient
    step uses independent transitions rather than correlated rollout
    segments. This makes SAC more sample-efficient but requires careful
    tuning of the replay ratio (gradient_steps / train_freq).
"""

import os
import copy
import time
from collections import deque
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    CombatPolicy, make_policy, TIER_CONFIGS,
)
from frame_stack import stacked_obs_size

from training.base_trainer import BaseTrainer
from training.normalizers import RunningNormalizer

from .config import SACConfig
from .networks import SACPolicyNetwork, TwinQNetwork
from .buffer import ReplayBuffer


class SACTrainer(BaseTrainer):
    """Discrete Soft Actor-Critic trainer for multi-head combat AI."""

    method_name = "sac"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = SACConfig(total_timesteps=self.total_timesteps)

    # ═════════════════════════════════════════════════════════════
    #  BaseTrainer Interface
    # ═════════════════════════════════════════════════════════════

    def build_model(self) -> nn.Module:
        """Create actor, twin critics, target critics, and optimizers."""

        # ── Actor ────────────────────────────────────────────────
        self.actor = SACPolicyNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)

        # ── Twin Q-networks ──────────────────────────────────────
        self.critic = TwinQNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)

        # ── Target Q-networks (Polyak-averaged copy) ─────────────
        self.critic_target = TwinQNetwork(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Target never receives gradients directly.
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimizers ───────────────────────────────────────────
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.lr_actor, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic, eps=1e-5)

        # ── Per-head entropy temperatures (auto-tuned) ───────────
        # Each action head gets its own α because the action spaces
        # have different sizes (9, 7, 5) and optimal entropy levels
        # differ. log(α) is the learnable parameter for numerical
        # stability.
        init_log_alpha = np.log(self.cfg.initial_alpha)
        self.log_alpha_m = torch.tensor(
            init_log_alpha, dtype=torch.float32,
            device=self.device, requires_grad=True)
        self.log_alpha_c = torch.tensor(
            init_log_alpha, dtype=torch.float32,
            device=self.device, requires_grad=True)
        self.log_alpha_t = torch.tensor(
            init_log_alpha, dtype=torch.float32,
            device=self.device, requires_grad=True)

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha_m, self.log_alpha_c, self.log_alpha_t],
            lr=self.cfg.lr_alpha)

        # Target entropies: ratio × max_entropy = ratio × log(N).
        ratio = self.cfg.target_entropy_ratio
        self.target_entropy_m = ratio * np.log(MOVEMENT_ACTIONS)
        self.target_entropy_c = ratio * np.log(COMBAT_ACTIONS)
        self.target_entropy_t = ratio * np.log(TARGET_ACTIONS)

        # BaseTrainer expects self.model and self.optimizer for
        # checkpointing. Point them at the actor.
        self.model = self.actor
        self.optimizer = self.actor_optimizer

        return self.actor

    def extract_policy(self) -> CombatPolicy:
        """Extract CombatPolicy from SAC actor.

        SAC actor keys map directly (no actor_ prefix):
            encoder.* → encoder.*
            backbone.* → backbone.*
            move_head.* → move_head.*
        """
        policy = make_policy(self.tier, frame_stack=self.frame_stack)
        policy.load_state_dict(self.actor.state_dict(), strict=False)
        policy.eval()
        return policy

    def default_curriculum_timesteps(self) -> Dict[int, int]:
        """SAC is more sample-efficient — reduced timestep budgets."""
        return {
            1: 25_000,
            2: 50_000,
            3: 300_000,
            4: 3_000_000,
            5: 6_000_000,
            6: 12_000_000,
            7: 20_000_000,
        }

    def run_eval(self, global_step: int, **kwargs) -> Dict:
        """Override to set is_actor_critic=False (SAC has no value head)."""
        from training.evaluation import evaluate
        eval_stats = evaluate(
            self.actor, self.stage, self.archetype,
            kwargs.get("eval_episodes", 50), self.device,
            self.frame_stack, self.obs_normalizer,
            base_seed=kwargs.get("eval_base_seed", 42),
            is_actor_critic=False,
        )

        self.writer.add_scalar("eval/mean_reward", eval_stats["mean_reward"], global_step)
        self.writer.add_scalar("eval/std_reward", eval_stats["std_reward"], global_step)
        self.writer.add_scalar("eval/win_rate", eval_stats["win_rate"], global_step)
        self.writer.add_scalar("eval/mean_kills", eval_stats["mean_kills"], global_step)
        self.writer.add_scalar("eval/mean_length", eval_stats["mean_length"], global_step)

        self.eval_history.append(
            (global_step, eval_stats["mean_reward"], eval_stats["win_rate"]))
        if len(self.eval_history) >= 3:
            self._plot_eval_correlation(global_step)
        self.writer.flush()

        print(f"  Eval: reward={eval_stats['mean_reward']:.1f} "
              f"±{eval_stats['reward_ci95']:.1f}, "
              f"win={eval_stats['win_rate']:.0%}, "
              f"kills={eval_stats['mean_kills']:.1f}")
        return eval_stats

    # ═════════════════════════════════════════════════════════════
    #  SAC Training Loop
    # ═════════════════════════════════════════════════════════════

    def train(self):
        """Full discrete SAC training loop."""
        cfg = self.cfg
        cfg.total_timesteps = self.total_timesteps

        self.build_model()
        self.print_setup()

        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"Actor: {actor_params:,} params, "
              f"Twin critics: {critic_params:,} params (×2 with target)")

        # ── Environment ──────────────────────────────────────────
        vec_env = self.create_vec_env()

        # ── Observation normalizer ───────────────────────────────
        self.obs_normalizer = (
            RunningNormalizer(self.input_size)
            if cfg.normalize_obs else None
        )

        # ── Replay buffer ────────────────────────────────────────
        replay = ReplayBuffer(cfg.buffer_size, self.input_size)

        # ── Checkpoint loading ───────────────────────────────────
        if self.bc_checkpoint and os.path.exists(self.bc_checkpoint):
            self._load_sac_checkpoint(self.bc_checkpoint)

        # ── Training state ───────────────────────────────────────
        obs, initial_infos = vec_env.reset()
        current_masks = self.extract_masks(initial_infos)
        global_step = 0
        episode_count = 0
        updates_done = 0
        consecutive_regressions = 0

        ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=np.int32)

        recent_rewards = deque(maxlen=50)
        recent_lengths = deque(maxlen=50)
        recent_wins = deque(maxlen=50)

        start_time = time.time()

        while global_step < cfg.total_timesteps:
            # ── Normalise observations ───────────────────────────
            if self.obs_normalizer:
                self.obs_normalizer.update(obs)
                obs_normed = self.obs_normalizer.normalize(obs)
            else:
                obs_normed = obs

            # ── Select actions ───────────────────────────────────
            if global_step < cfg.learning_starts:
                # Random actions during warmup.
                m_acts = np.random.randint(0, MOVEMENT_ACTIONS, self.num_envs)
                c_acts = np.random.randint(0, COMBAT_ACTIONS, self.num_envs)
                t_acts = np.random.randint(0, TARGET_ACTIONS, self.num_envs)
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_normed).float().to(self.device)
                    (m_a, c_a, t_a), _ = self.actor.sample_actions(
                        obs_t, masks=current_masks)
                    m_acts = m_a.cpu().numpy()
                    c_acts = c_a.cpu().numpy()
                    t_acts = t_a.cpu().numpy()

            # ── Step environment ─────────────────────────────────
            actions_np = np.stack([m_acts, c_acts, t_acts], axis=1)
            next_obs, rewards, dones, truncateds, infos = vec_env.step(actions_np)

            # Normalise next obs for storage.
            if self.obs_normalizer:
                next_obs_normed = self.obs_normalizer.normalize(next_obs)
            else:
                next_obs_normed = next_obs

            # Extract next-state masks for target computation.
            next_masks = self.extract_masks(infos)
            next_m_masks = next_masks[0].cpu().numpy()
            next_c_masks = next_masks[1].cpu().numpy()
            next_t_masks = next_masks[2].cpu().numpy()

            # ── Store transitions ────────────────────────────────
            terminals = np.logical_or(dones, truncateds).astype(np.float32)
            replay.add_batch(
                obs_normed, next_obs_normed,
                m_acts, c_acts, t_acts,
                rewards, terminals,
                next_m_masks, next_c_masks, next_t_masks,
            )

            # ── Episode tracking ─────────────────────────────────
            ep_rewards += rewards
            ep_lengths += 1

            for i in range(self.num_envs):
                if dones[i] or truncateds[i]:
                    is_win = bool(infos[i].get("is_win", False))
                    self.writer.add_scalar(
                        "rollout/episode_reward", ep_rewards[i], global_step)
                    self.writer.add_scalar(
                        "rollout/episode_length", ep_lengths[i], global_step)
                    self.writer.add_scalar(
                        "rollout/win", float(is_win), global_step)

                    recent_rewards.append(ep_rewards[i])
                    recent_lengths.append(ep_lengths[i])
                    recent_wins.append(float(is_win))

                    episode_count += 1
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

            obs = next_obs
            current_masks = next_masks
            global_step += self.num_envs

            # ── Gradient updates ─────────────────────────────────
            if (global_step >= cfg.learning_starts
                    and global_step % cfg.train_freq < self.num_envs
                    and len(replay) >= cfg.batch_size):

                for _ in range(cfg.gradient_steps):
                    metrics = self._update(replay)
                    updates_done += 1

                # Log training metrics.
                if updates_done % 100 < cfg.gradient_steps:
                    alpha_m = self.log_alpha_m.exp().item()
                    alpha_c = self.log_alpha_c.exp().item()
                    alpha_t = self.log_alpha_t.exp().item()

                    self.writer.add_scalar(
                        "train/critic_loss", metrics["critic_loss"], global_step)
                    self.writer.add_scalar(
                        "train/actor_loss", metrics["actor_loss"], global_step)
                    self.writer.add_scalar(
                        "train/alpha_loss", metrics["alpha_loss"], global_step)
                    self.writer.add_scalar(
                        "train/alpha_m", alpha_m, global_step)
                    self.writer.add_scalar(
                        "train/alpha_c", alpha_c, global_step)
                    self.writer.add_scalar(
                        "train/alpha_t", alpha_t, global_step)
                    self.writer.add_scalar(
                        "train/entropy_m", metrics["entropy_m"], global_step)
                    self.writer.add_scalar(
                        "train/entropy_c", metrics["entropy_c"], global_step)
                    self.writer.add_scalar(
                        "train/entropy_t", metrics["entropy_t"], global_step)
                    self.writer.add_scalar(
                        "train/q_mean", metrics["q_mean"], global_step)
                    self.writer.add_scalar(
                        "train/updates", updates_done, global_step)

            # ── Periodic logging ─────────────────────────────────
            batch_total = cfg.train_freq * self.num_envs
            if global_step % 5000 < self.num_envs and recent_rewards:
                sps = global_step / max(time.time() - start_time, 1)
                alpha_m = self.log_alpha_m.exp().item()
                print(
                    f"Step {global_step:>8,}/{cfg.total_timesteps:,} | "
                    f"Ep: {episode_count} | "
                    f"R(50): {np.mean(recent_rewards):+.1f} | "
                    f"Win: {np.mean(recent_wins):.0%} | "
                    f"Buf: {len(replay):,} | "
                    f"α_m: {alpha_m:.3f} | "
                    f"Updates: {updates_done:,} | "
                    f"SPS: {sps:.0f}"
                )

                self.writer.add_scalar(
                    "rollout/mean_reward_50ep",
                    np.mean(recent_rewards), global_step)
                self.writer.add_scalar(
                    "rollout/win_rate_50ep",
                    np.mean(recent_wins), global_step)
                self.writer.flush()

            # ── Periodic evaluation ──────────────────────────────
            if global_step % cfg.eval_interval < self.num_envs:
                eval_stats = self.run_eval(
                    global_step,
                    eval_episodes=cfg.num_eval_episodes,
                    eval_base_seed=cfg.eval_base_seed,
                )

                if self.check_best_model(eval_stats, global_step):
                    consecutive_regressions = 0
                else:
                    consecutive_regressions += 1

                # Catastrophic regression reversion.
                current_wr = eval_stats["win_rate"]
                has_collapsed = (
                    (self.best_eval_win_rate - current_wr)
                    > cfg.revert_min_drop)
                if (cfg.revert_on_regression
                        and consecutive_regressions >= cfg.revert_patience
                        and has_collapsed
                        and self.best_eval_win_rate > 0.05):
                    best_path = os.path.join(
                        self.output_dir,
                        f"{self.method_name}_stage{self.stage}_best.pt")
                    if os.path.exists(best_path):
                        print(f"  ⚠ Reverting to best checkpoint")
                        self._load_sac_checkpoint(best_path)
                        consecutive_regressions = 0

            # ── Periodic save ────────────────────────────────────
            if global_step % cfg.save_interval < self.num_envs:
                path = os.path.join(
                    self.output_dir,
                    f"{self.method_name}_stage{self.stage}.pt")
                self._save_sac_checkpoint(path, global_step)

        # ── Final save ───────────────────────────────────────────
        path = os.path.join(
            self.output_dir,
            f"{self.method_name}_stage{self.stage}_final.pt")
        self._save_sac_checkpoint(path, global_step)
        print(f"\nSAC training complete. "
              f"Best win rate: {self.best_eval_win_rate:.0%}")

        vec_env.close()
        self.writer.close()

    # ═════════════════════════════════════════════════════════════
    #  SAC Update Step
    # ═════════════════════════════════════════════════════════════

    def _update(self, replay: ReplayBuffer) -> dict:
        """Single SAC gradient step: critic → actor → α → target."""
        cfg = self.cfg
        batch = replay.sample(cfg.batch_size)

        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        m_acts = batch["m_acts"].to(self.device)
        c_acts = batch["c_acts"].to(self.device)
        t_acts = batch["t_acts"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        dones = batch["dones"].to(self.device)
        next_masks = (
            batch["next_m_masks"].to(self.device),
            batch["next_c_masks"].to(self.device),
            batch["next_t_masks"].to(self.device),
        )

        alpha_m = self.log_alpha_m.exp()
        alpha_c = self.log_alpha_c.exp()
        alpha_t = self.log_alpha_t.exp()

        # ── 1. Critic update ─────────────────────────────────────
        with torch.no_grad():
            # Next-state action probs and log-probs from actor.
            (n_m_p, n_c_p, n_t_p), (n_m_lp, n_c_lp, n_t_lp), _ = \
                self.actor.get_action_probs(next_obs, masks=next_masks)

            # Next-state Q-values from TARGET critics.
            (q1_m, q1_c, q1_t), (q2_m, q2_c, q2_t) = \
                self.critic_target(next_obs)

            # Min of twin Q for each head.
            min_q_m = torch.min(q1_m, q2_m)
            min_q_c = torch.min(q1_c, q2_c)
            min_q_t = torch.min(q1_t, q2_t)

            # Soft V(s') per head: E_a[Q(s',a) - α * log π(a|s')]
            v_m = (n_m_p * (min_q_m - alpha_m * n_m_lp)).sum(dim=-1)
            v_c = (n_c_p * (min_q_c - alpha_c * n_c_lp)).sum(dim=-1)
            v_t = (n_t_p * (min_q_t - alpha_t * n_t_lp)).sum(dim=-1)

            # Bellman target (shared across heads).
            target = rewards + cfg.gamma * (1 - dones) * (v_m + v_c + v_t)

        # Current Q-values for taken actions (additive decomposition).
        (cq1_m, cq1_c, cq1_t), (cq2_m, cq2_c, cq2_t) = self.critic(obs)

        q1_taken = (
            cq1_m.gather(1, m_acts.unsqueeze(-1)).squeeze(-1) +
            cq1_c.gather(1, c_acts.unsqueeze(-1)).squeeze(-1) +
            cq1_t.gather(1, t_acts.unsqueeze(-1)).squeeze(-1)
        )
        q2_taken = (
            cq2_m.gather(1, m_acts.unsqueeze(-1)).squeeze(-1) +
            cq2_c.gather(1, c_acts.unsqueeze(-1)).squeeze(-1) +
            cq2_t.gather(1, t_acts.unsqueeze(-1)).squeeze(-1)
        )

        critic_loss = F.mse_loss(q1_taken, target) + F.mse_loss(q2_taken, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        # ── 2. Actor update ──────────────────────────────────────
        # Freeze critic to save compute (actor loss doesn't backprop
        # through Q-networks, only through the policy).
        for p in self.critic.parameters():
            p.requires_grad = False

        (m_p, c_p, t_p), (m_lp, c_lp, t_lp), (ent_m, ent_c, ent_t) = \
            self.actor.get_action_probs(obs)

        # Use Q1 only for actor update (standard SAC practice).
        (aq1_m, aq1_c, aq1_t), _ = self.critic(obs)

        # Policy loss: E_s[sum_a π(a|s) * (α * log π(a|s) - Q(s,a))]
        actor_loss_m = (m_p * (alpha_m.detach() * m_lp - aq1_m)).sum(dim=-1)
        actor_loss_c = (c_p * (alpha_c.detach() * c_lp - aq1_c)).sum(dim=-1)
        actor_loss_t = (t_p * (alpha_t.detach() * t_lp - aq1_t)).sum(dim=-1)
        actor_loss = (actor_loss_m + actor_loss_c + actor_loss_t).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # Unfreeze critic.
        for p in self.critic.parameters():
            p.requires_grad = True

        # ── 3. Entropy temperature (α) update ────────────────────
        # α adjusts to maintain target entropy per head.
        alpha_loss_m = -self.log_alpha_m * (ent_m.mean() - self.target_entropy_m).detach()
        alpha_loss_c = -self.log_alpha_c * (ent_c.mean() - self.target_entropy_c).detach()
        alpha_loss_t = -self.log_alpha_t * (ent_t.mean() - self.target_entropy_t).detach()
        alpha_loss = alpha_loss_m + alpha_loss_c + alpha_loss_t

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ── 4. Soft target update (Polyak averaging) ─────────────
        with torch.no_grad():
            for p, p_target in zip(
                    self.critic.parameters(),
                    self.critic_target.parameters()):
                p_target.data.mul_(1 - cfg.tau)
                p_target.data.add_(cfg.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "entropy_m": ent_m.mean().item(),
            "entropy_c": ent_c.mean().item(),
            "entropy_t": ent_t.mean().item(),
            "q_mean": q1_taken.mean().item(),
        }

    # ═════════════════════════════════════════════════════════════
    #  Checkpointing (SAC-specific — more state than PPO)
    # ═════════════════════════════════════════════════════════════

    def _save_sac_checkpoint(self, path: str, global_step: int):
        """Save full SAC state for resume + policy_state_dict for export."""
        from combat_policy import save_checkpoint

        # Use the base checkpoint format for the actor (policy compat).
        save_checkpoint(
            self.actor, self.actor_optimizer, path,
            stage=self.stage, archetype=self.archetype,
            step=global_step, frame_stack=self.frame_stack,
            tier=self.tier, obs_normalizer=self.obs_normalizer,
        )

        # Append SAC-specific state to the saved checkpoint.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["sac_critic"] = self.critic.state_dict()
        ckpt["sac_critic_target"] = self.critic_target.state_dict()
        ckpt["sac_critic_optimizer"] = self.critic_optimizer.state_dict()
        ckpt["sac_log_alpha_m"] = self.log_alpha_m.detach().cpu()
        ckpt["sac_log_alpha_c"] = self.log_alpha_c.detach().cpu()
        ckpt["sac_log_alpha_t"] = self.log_alpha_t.detach().cpu()
        ckpt["sac_alpha_optimizer"] = self.alpha_optimizer.state_dict()
        ckpt["method"] = "sac"
        torch.save(ckpt, path)

    def _load_sac_checkpoint(self, path: str):
        """Load SAC checkpoint. Falls back to actor-only for PPO ckpts."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Actor weights (compatible with both PPO and SAC checkpoints).
        if "full_state_dict" in ckpt:
            # PPO-style checkpoint — full_state_dict has actor_encoder etc.
            # Map to SAC actor keys (no actor_ prefix).
            state = ckpt["full_state_dict"]
            own_state = self.actor.state_dict()
            loaded = 0
            for key, val in state.items():
                # Try direct match first (SAC checkpoint).
                if key in own_state and val.shape == own_state[key].shape:
                    own_state[key] = val
                    loaded += 1
                    continue
                # Try PPO actor → SAC mapping.
                mapped = key.replace("actor_encoder.", "encoder.")
                mapped = mapped.replace("actor_backbone.", "backbone.")
                if mapped in own_state and val.shape == own_state[mapped].shape:
                    own_state[mapped] = val
                    loaded += 1
            self.actor.load_state_dict(own_state, strict=False)
            print(f"Loaded actor: {loaded} tensors from {path}")

        elif "policy_state_dict" in ckpt:
            self.actor.load_state_dict(
                ckpt["policy_state_dict"], strict=False)
            print(f"Loaded actor from policy_state_dict")

        # SAC-specific state (only present in SAC checkpoints).
        if "sac_critic" in ckpt:
            self.critic.load_state_dict(ckpt["sac_critic"])
            self.critic_target.load_state_dict(ckpt["sac_critic_target"])
            self.critic_optimizer.load_state_dict(
                ckpt["sac_critic_optimizer"])
            self.log_alpha_m.data.copy_(ckpt["sac_log_alpha_m"])
            self.log_alpha_c.data.copy_(ckpt["sac_log_alpha_c"])
            self.log_alpha_t.data.copy_(ckpt["sac_log_alpha_t"])
            self.alpha_optimizer.load_state_dict(
                ckpt["sac_alpha_optimizer"])
            print(f"Restored SAC critic, target, and α state")

        # Normalizer.
        if self.obs_normalizer and "obs_normalizer" in ckpt:
            self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
            print(f"Restored observation normalizer")