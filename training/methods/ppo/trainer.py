"""
trainer.py — PPO training method.

Implements BaseTrainer for Proximal Policy Optimization with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - KL early stopping
    - Entropy annealing
    - Value function clipping
    - Catastrophic regression reversion

This is the ONLY file that needs to change for PPO-specific tuning.
The base trainer handles env creation, evaluation, curriculum, and checkpointing.
"""

import os
import time
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_policy import (
    CombatPolicy, make_policy, TIER_CONFIGS,
    load_teacher_from_checkpoint,
)
from frame_stack import stacked_obs_size

from training.base_trainer import BaseTrainer
from training.normalizers import RunningNormalizer, ReturnNormalizer

from .config import PPOConfig
from .actor_critic import ActorCritic
from .buffer import VecRolloutBuffer


class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization trainer."""

    method_name = "ppo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg = PPOConfig(total_timesteps=self.total_timesteps)

    # ═════════════════════════════════════════════════════════════
    #  BaseTrainer Interface
    # ═════════════════════════════════════════════════════════════

    def build_model(self) -> nn.Module:
        """Create ActorCritic and optimizer."""
        model = ActorCritic(
            obs_size=self.input_size, tier=self.tier
        ).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.cfg.lr, eps=1e-5)

        self.model = model
        self.optimizer = optimizer
        return model

    def extract_policy(self) -> CombatPolicy:
        """Extract CombatPolicy from the trained ActorCritic.

        Maps: actor_encoder → encoder, actor_backbone → backbone.
        Drops: critic_encoder, critic_backbone, value_head.
        """
        policy = make_policy(self.tier, frame_stack=self.frame_stack)
        policy_state = policy.state_dict()

        model_state = self.model.state_dict()
        loaded = 0

        for src_key, src_val in model_state.items():
            # Skip critic keys.
            if any(skip in src_key for skip in
                   ["critic_encoder", "critic_backbone", "value_head"]):
                continue

            # Map actor keys to policy keys.
            dst_key = src_key.replace("actor_encoder.", "encoder.")
            dst_key = dst_key.replace("actor_backbone.", "backbone.")

            if dst_key in policy_state:
                if src_val.shape == policy_state[dst_key].shape:
                    policy_state[dst_key] = src_val
                    loaded += 1

        policy.load_state_dict(policy_state, strict=False)
        policy.eval()
        print(f"Extracted CombatPolicy: {loaded} tensors loaded")
        return policy

    # ═════════════════════════════════════════════════════════════
    #  Training Loop
    # ═════════════════════════════════════════════════════════════

    def train(self):
        """Full PPO training loop."""
        cfg = self.cfg
        cfg.total_timesteps = self.total_timesteps

        self.build_model()
        self.print_setup()

        batch_total = cfg.num_steps * self.num_envs
        print(f"Steps/env: {cfg.num_steps}, "
              f"batch: {batch_total} transitions/rollout")

        # ── Create vectorized environment ────────────────────────
        vec_env = self.create_vec_env()

        # ── Observation / return normalizers ──────────────────────
        self.obs_normalizer = (
            RunningNormalizer(self.input_size)
            if cfg.normalize_obs else None
        )
        return_normalizer = (
            ReturnNormalizer(cfg.gamma)
            if cfg.normalize_returns else None
        )

        # ── Load checkpoint ──────────────────────────────────────
        loaded_from_ppo = False

        if self.bc_checkpoint and os.path.exists(self.bc_checkpoint):
            ckpt = torch.load(
                self.bc_checkpoint, map_location=self.device,
                weights_only=False)
            is_ppo_checkpoint = "full_state_dict" in ckpt

            if is_ppo_checkpoint:
                ckpt_stage = ckpt.get("stage", self.stage)
                is_stage_transition = (ckpt_stage != self.stage)

                if is_stage_transition:
                    print(f"Stage transition detected: checkpoint stage "
                          f"{ckpt_stage} → training stage {self.stage}")
                    print(f"Reinitialising critic (fresh value function "
                          f"for new stage)")

                self.model.load_from_ppo_checkpoint(
                    self.bc_checkpoint,
                    reinit_critic=is_stage_transition)
                loaded_from_ppo = True

                # Restore normalizer state if saved.
                if self.obs_normalizer and "obs_normalizer" in ckpt:
                    self.obs_normalizer.load_state_dict(
                        ckpt["obs_normalizer"])
                    print(f"Restored observation normalizer state")

                print(f"Loaded PPO checkpoint: {self.bc_checkpoint} "
                      f"(stage {ckpt_stage}, "
                      f"step {ckpt.get('step', '?')})")
            else:
                # BC checkpoint — actor weights only, no critic.
                self.model.load_from_ppo_checkpoint(self.bc_checkpoint)
                print(f"Warm-started from BC checkpoint: "
                      f"{self.bc_checkpoint}")

        # ── LR annealing ─────────────────────────────────────────
        warmup_steps = 50_000 if loaded_from_ppo else 0
        total_rollouts = cfg.total_timesteps // batch_total

        def lr_lambda(rollout_step):
            if warmup_steps > 0:
                warmup_rollouts = max(1, warmup_steps // batch_total)
                if rollout_step < warmup_rollouts:
                    return 0.2 + 0.8 * rollout_step / warmup_rollouts
            progress = min(1.0, rollout_step / max(total_rollouts, 1))
            return max(0.01, 1.0 - progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda)

        print(f"LR schedule: {cfg.lr:.1e} → {cfg.lr * 0.01:.1e} "
              f"over {total_rollouts} rollouts "
              f"({cfg.total_timesteps:,} steps)"
              + (f" with warmup over {warmup_steps:,} steps"
                 if warmup_steps > 0 else ""))

        # ── Buffer ───────────────────────────────────────────────
        buffer = VecRolloutBuffer(
            cfg.num_steps, self.num_envs, self.input_size)

        # ── Training state ───────────────────────────────────────
        obs, initial_infos = vec_env.reset()
        current_masks = self.extract_masks(initial_infos)
        global_step = 0
        episode_count = 0
        scheduler_step = 0
        consecutive_regressions = 0

        # Per-env episode tracking.
        ep_rewards = np.zeros(self.num_envs, dtype=np.float32)
        ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        ep_components = [{} for _ in range(self.num_envs)]

        # Rolling window for reporting.
        recent_rewards = deque(maxlen=50)
        recent_lengths = deque(maxlen=50)
        recent_wins = deque(maxlen=50)

        start_time = time.time()

        while global_step < cfg.total_timesteps:

            # ── Collect rollout ──────────────────────────────────
            self.model.eval()
            rollout_episodes = 0

            for step in range(cfg.num_steps):
                # Normalise observations.
                if self.obs_normalizer:
                    self.obs_normalizer.update(obs)
                    obs_normed = self.obs_normalizer.normalize(obs)
                else:
                    obs_normed = obs

                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_normed).float().to(
                        self.device)
                    actions, log_probs, _, values = \
                        self.model.get_action_and_value(
                            obs_t, masks=current_masks)
                    m_acts = actions[0].cpu().numpy()
                    c_acts = actions[1].cpu().numpy()
                    t_acts = actions[2].cpu().numpy()

                actions_np = np.stack(
                    [m_acts, c_acts, t_acts], axis=1)
                next_obs, rewards, dones, truncateds, infos = \
                    vec_env.step(actions_np)

                # Update return normalizer and scale rewards.
                if return_normalizer:
                    terminals = np.logical_or(
                        dones, truncateds).astype(np.float32)
                    return_normalizer.update(rewards, terminals)
                    rewards_normed = return_normalizer.normalize(rewards)
                else:
                    rewards_normed = rewards

                # Handle truncations: bootstrap value into reward.
                for i in range(self.num_envs):
                    if truncateds[i] and not dones[i]:
                        term_obs = infos[i]["terminal_observation"]
                        if self.obs_normalizer:
                            term_obs = self.obs_normalizer.normalize(
                                term_obs)
                        with torch.no_grad():
                            term_t = torch.from_numpy(
                                term_obs).float().unsqueeze(0).to(
                                    self.device)
                            term_val = self.model.get_value(
                                term_t).cpu().item()
                        rewards_normed[i] += cfg.gamma * term_val

                # Store in buffer.
                buffer.obs[step] = obs_normed
                buffer.m_acts[step] = m_acts
                buffer.c_acts[step] = c_acts
                buffer.t_acts[step] = t_acts
                buffer.log_probs[step] = log_probs.cpu().numpy()
                buffer.rewards[step] = rewards_normed
                buffer.dones[step] = np.logical_or(
                    dones, truncateds).astype(np.float32)
                buffer.values[step] = values.cpu().numpy()
                buffer.m_masks[step] = current_masks[0].cpu().numpy()
                buffer.c_masks[step] = current_masks[1].cpu().numpy()
                buffer.t_masks[step] = current_masks[2].cpu().numpy()

                # Update masks for NEXT step from env infos.
                current_masks = self.extract_masks(infos)

                # Per-env episode accounting.
                ep_rewards += rewards  # Track raw rewards.
                ep_lengths += 1

                for i in range(self.num_envs):
                    for key, val in infos[i].items():
                        if isinstance(val, (int, float)) and \
                                key != "terminal_observation":
                            ep_components[i][key] = \
                                ep_components[i].get(key, 0.0) + val

                for i in range(self.num_envs):
                    if dones[i] or truncateds[i]:
                        self.writer.add_scalar(
                            "rollout/episode_reward",
                            ep_rewards[i], global_step)
                        self.writer.add_scalar(
                            "rollout/episode_length",
                            ep_lengths[i], global_step)

                        is_win = bool(infos[i].get("is_win", False))
                        self.writer.add_scalar(
                            "rollout/win", float(is_win), global_step)

                        for key, val in ep_components[i].items():
                            if abs(val) > 1e-6:
                                self.writer.add_scalar(
                                    f"reward/{key}", val, global_step)

                        recent_rewards.append(ep_rewards[i])
                        recent_lengths.append(ep_lengths[i])
                        recent_wins.append(float(is_win))

                        episode_count += 1
                        rollout_episodes += 1
                        ep_rewards[i] = 0.0
                        ep_lengths[i] = 0
                        ep_components[i] = {}

                obs = next_obs
                global_step += self.num_envs

            # ── Compute last values for GAE ──────────────────────
            if self.obs_normalizer:
                obs_normed = self.obs_normalizer.normalize(obs)
            else:
                obs_normed = obs

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_normed).float().to(
                    self.device)
                last_values = self.model.get_value(obs_t).cpu().numpy()

            buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

            # ── PPO Update ───────────────────────────────────────
            self.model.train()

            # Normalise advantages.
            flat_adv = buffer.advantages.reshape(-1)
            flat_adv = ((flat_adv - flat_adv.mean())
                        / (flat_adv.std() + 1e-8))
            buffer.advantages = flat_adv.reshape(
                cfg.num_steps, self.num_envs)

            # Anneal entropy coefficient.
            ent_progress = min(
                1.0, global_step / max(cfg.total_timesteps, 1))
            current_ent_coef = (
                cfg.entropy_coef
                + (cfg.entropy_coef_final - cfg.entropy_coef)
                * ent_progress
            )

            total_pg_loss = 0
            total_v_loss = 0
            total_ent = 0
            total_clip_frac = 0
            total_approx_kl = 0
            n_updates = 0
            kl_early_stopped = False

            for epoch in range(cfg.update_epochs):
                if kl_early_stopped:
                    break
                for batch in buffer.sample_minibatches(
                        cfg.mini_batch_size):
                    b_obs = batch["obs"].to(self.device)
                    b_m = batch["m_acts"].to(self.device)
                    b_c = batch["c_acts"].to(self.device)
                    b_t = batch["t_acts"].to(self.device)
                    b_old_lp = batch["old_log_probs"].to(self.device)
                    b_adv = batch["advantages"].to(self.device)
                    b_ret = batch["returns"].to(self.device)
                    b_old_val = batch["old_values"].to(self.device)
                    b_masks = (
                        batch["m_masks"].to(self.device),
                        batch["c_masks"].to(self.device),
                        batch["t_masks"].to(self.device),
                    )

                    new_lp, entropy, new_val = \
                        self.model.evaluate_actions(
                            b_obs, b_m, b_c, b_t, masks=b_masks)

                    # Policy loss (clipped).
                    ratio = (new_lp - b_old_lp).exp()
                    pg_loss1 = -b_adv * ratio
                    pg_loss2 = -b_adv * ratio.clamp(
                        1 - cfg.clip_range, 1 + cfg.clip_range)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping.
                    v_clipped = b_old_val + (
                        new_val - b_old_val).clamp(
                            -cfg.vf_clip_range, cfg.vf_clip_range)
                    v_loss1 = (new_val - b_ret) ** 2
                    v_loss2 = (v_clipped - b_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                    # Entropy bonus (annealed).
                    ent_loss = -entropy.mean()

                    loss = (pg_loss
                            + cfg.value_coef * v_loss
                            + current_ent_coef * ent_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    # Track diagnostics.
                    with torch.no_grad():
                        clip_frac = (
                            (ratio - 1.0).abs() > cfg.clip_range
                        ).float().mean()
                        approx_kl = (
                            0.5 * (new_lp - b_old_lp).pow(2)
                        ).mean()

                    total_pg_loss += pg_loss.item()
                    total_v_loss += v_loss.item()
                    total_ent += entropy.mean().item()
                    total_clip_frac += clip_frac.item()
                    total_approx_kl += approx_kl.item()
                    n_updates += 1

                    # KL early stopping.
                    if (cfg.target_kl > 0
                            and approx_kl.item()
                            > cfg.target_kl * 1.5):
                        kl_early_stopped = True
                        break

            # Step scheduler.
            scheduler_step += 1
            scheduler.step()

            # ── Logging ──────────────────────────────────────────
            nu = max(n_updates, 1)
            self.writer.add_scalar(
                "train/policy_loss", total_pg_loss / nu, global_step)
            self.writer.add_scalar(
                "train/value_loss", total_v_loss / nu, global_step)
            self.writer.add_scalar(
                "train/entropy", total_ent / nu, global_step)
            self.writer.add_scalar(
                "train/clip_fraction",
                total_clip_frac / nu, global_step)
            self.writer.add_scalar(
                "train/approx_kl",
                total_approx_kl / nu, global_step)
            self.writer.add_scalar(
                "train/entropy_coef",
                current_ent_coef, global_step)
            self.writer.add_scalar(
                "train/kl_early_stopped",
                float(kl_early_stopped), global_step)
            self.writer.add_scalar(
                "train/episodes_total", episode_count, global_step)
            self.writer.add_scalar(
                "train/learning_rate",
                self.optimizer.param_groups[0]["lr"], global_step)

            if len(recent_rewards) > 0:
                self.writer.add_scalar(
                    "rollout/mean_reward_50ep",
                    np.mean(recent_rewards), global_step)
                self.writer.add_scalar(
                    "rollout/mean_length_50ep",
                    np.mean(recent_lengths), global_step)
                self.writer.add_scalar(
                    "rollout/win_rate_50ep",
                    np.mean(recent_wins), global_step)

            self.writer.flush()

            sps = global_step / max(time.time() - start_time, 1)
            mean_r = (np.mean(recent_rewards)
                      if recent_rewards else 0)
            win_r = (np.mean(recent_wins)
                     if recent_wins else 0)
            print(
                f"Step {global_step:>8,}/{cfg.total_timesteps:,} | "
                f"Ep: {episode_count} "
                f"({rollout_episodes}/rollout) | "
                f"R(50): {mean_r:+.1f} | "
                f"Win: {win_r:.0%} | "
                f"PG: {total_pg_loss/nu:.4f} | "
                f"VL: {total_v_loss/nu:.4f} | "
                f"Ent: {total_ent/nu:.2f} | "
                f"Clip: {total_clip_frac/nu:.2f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
                f"SPS: {sps:.0f}"
            )

            # ── Periodic evaluation ──────────────────────────────
            if global_step % cfg.eval_interval < batch_total:
                eval_stats = self.run_eval(
                    global_step,
                    eval_episodes=cfg.num_eval_episodes,
                    eval_base_seed=cfg.eval_base_seed,
                    batch_total=batch_total,
                )

                if self.check_best_model(eval_stats, global_step):
                    consecutive_regressions = 0
                else:
                    consecutive_regressions += 1

                # Revert on catastrophic regression.
                current_wr = eval_stats["win_rate"]
                has_collapsed = (
                    (self.best_eval_win_rate - current_wr)
                    > cfg.revert_min_drop
                )
                if (cfg.revert_on_regression
                        and consecutive_regressions
                        >= cfg.revert_patience
                        and has_collapsed
                        and self.best_eval_win_rate > 0.05):
                    best_path = os.path.join(
                        self.output_dir,
                        f"{self.method_name}_stage"
                        f"{self.stage}_best.pt")
                    if os.path.exists(best_path):
                        print(
                            f"  ⚠ Win rate collapsed: "
                            f"current={current_wr:.0%} "
                            f"vs best="
                            f"{self.best_eval_win_rate:.0%} "
                            f"(>{cfg.revert_min_drop:.0%} drop, "
                            f"{consecutive_regressions} evals). "
                            f"Reverting model weights only.")
                        self.model.load_from_ppo_checkpoint(
                            best_path)
                        if self.obs_normalizer:
                            ckpt = torch.load(
                                best_path,
                                map_location=self.device,
                                weights_only=False)
                            if "obs_normalizer" in ckpt:
                                self.obs_normalizer.load_state_dict(
                                    ckpt["obs_normalizer"])
                        consecutive_regressions = 0

            # ── Periodic save ────────────────────────────────────
            if global_step % cfg.save_interval < batch_total:
                path = os.path.join(
                    self.output_dir,
                    f"{self.method_name}_stage{self.stage}.pt")
                self.save_checkpoint(path, global_step)

        # Final save.
        path = os.path.join(
            self.output_dir,
            f"{self.method_name}_stage{self.stage}_final.pt")
        self.save_checkpoint(path, global_step)
        print(f"\nTraining complete. "
              f"Best eval win rate: {self.best_eval_win_rate:.0%}")

        vec_env.close()
        self.writer.close()
