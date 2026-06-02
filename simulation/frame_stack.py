"""
frame_stack.py — Frame stacking for both BC (offline) and PPO (online),
                 plus vectorized environment for multi-env PPO rollouts.

THE PROBLEM THIS SOLVES
    The C++ NeuralCombatComponent stacks N consecutive observation frames
    into one flat tensor before feeding it to the model. A model with
    FrameStackCount=3 receives 198×3 = 594 floats, not 198. The model
    needs temporal context to learn timing-dependent behaviour like
    "the target is accelerating left" or "I'm 80% through a reload."

    Training must use the same stacking. This module provides:

    1. FrameStackedDataset — for BC. Takes consecutive CSV rows from
       the same episode and builds rolling windows. The action label
       is from the LAST frame in the window (the decision being made).

    2. FrameStackEnvWrapper — for PPO (single env). Wraps a Gymnasium
       env and maintains a ring buffer of recent observations.

    3. VecFrameStackEnv — for PPO (multiple parallel envs). Runs N
       independent environments in lockstep, returning batched
       observations. This is the single biggest variance-reduction
       technique for PPO — with 8 envs you get ~8x more episodes per
       rollout, dramatically stabilising advantage estimates.

    All produce the same flat tensor layout as the C++ BuildFlatInput():
       [frame_oldest, ..., frame_newest] where each frame is 198 floats.

USAGE (BC):
    dataset = FrameStackedDataset("Saved/NeuralData/Default", frame_stack=3)

USAGE (PPO, single env):
    env = FrameStackEnvWrapper(CombatEnv(config), frame_stack=3)
    obs, info = env.reset()  # obs.shape == (594,)

USAGE (PPO, vectorized):
    vec_env = VecFrameStackEnv(
        env_fns=[lambda: make_curriculum_env(3, "ranged") for _ in range(8)],
        frame_stack=3,
    )
    obs = vec_env.reset()  # obs.shape == (8, 594)
    obs, rewards, dones, truncs, infos = vec_env.step(actions)  # actions.shape == (8, 3)
"""

from __future__ import annotations

import glob
import os
from collections import deque
from typing import Optional, List, Callable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from torch.utils.data import Dataset

from combat_sim import MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS

# Must match NeuralCombatTypes.h
SINGLE_OBS_SIZE = 231
METADATA_COLS = 5  # EncounterID, EnemyName, Archetype, Frame, CombatTime


# ─────────────────────────────────────────────────────────────────
#  Stacked observation size helper
# ─────────────────────────────────────────────────────────────────

def stacked_obs_size(frame_stack: int = 3) -> int:
    """Total floats in a stacked observation tensor."""
    return SINGLE_OBS_SIZE * frame_stack


# ─────────────────────────────────────────────────────────────────
#  BC Dataset with Frame Stacking
# ─────────────────────────────────────────────────────────────────

class FrameStackedDataset(Dataset):
    """Loads CSV recordings and builds frame-stacked training samples.

    Each sample is a rolling window of `frame_stack` consecutive rows
    from the same episode (same enemy in the same encounter). The action
    label comes from the LAST row in the window — the decision the
    scripted brain made given the full temporal context.

    If an episode has fewer than `frame_stack` rows, the first frame
    is repeated to fill the missing slots (same as C++ BuildFlatInput).
    """

    def __init__(
        self,
        data_dir: str,
        frame_stack: int = 3,
        archetype_filter: Optional[str] = None,
    ):
        self.frame_stack = frame_stack
        self.stacked_obs: list[np.ndarray] = []
        self.actions_m: list[int] = []
        self.actions_c: list[int] = []
        self.actions_t: list[int] = []

        csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
        print(f"Found {len(csv_files)} CSV files in {data_dir}")

        total_rows = 0
        total_samples = 0

        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)
            except Exception as e:
                print(f"  Skipping {filepath}: {e}")
                continue

            if archetype_filter and "Archetype" in df.columns:
                df = df[df["Archetype"].str.lower() == archetype_filter.lower()]
                if len(df) == 0:
                    continue

            obs_cols = df.columns[METADATA_COLS:METADATA_COLS + SINGLE_OBS_SIZE]
            if len(obs_cols) != SINGLE_OBS_SIZE:
                print(f"  Skipping {filepath}: expected {SINGLE_OBS_SIZE} obs cols, "
                      f"got {len(obs_cols)}")
                continue

            obs_data = df[obs_cols].values.astype(np.float32)
            act_col_start = METADATA_COLS + SINGLE_OBS_SIZE
            act_m = df.iloc[:, act_col_start].values.astype(np.int64)
            act_c = df.iloc[:, act_col_start + 1].values.astype(np.int64)
            act_t = df.iloc[:, act_col_start + 2].values.astype(np.int64)

            total_rows += len(obs_data)
            n_rows = len(obs_data)

            for i in range(n_rows):
                frames = []
                for f in range(frame_stack):
                    row_idx = i - (frame_stack - 1 - f)
                    if row_idx < 0:
                        frames.append(obs_data[0])
                    else:
                        frames.append(obs_data[row_idx])

                stacked = np.concatenate(frames)
                self.stacked_obs.append(stacked)

                self.actions_m.append(int(np.clip(act_m[i], 0, MOVEMENT_ACTIONS - 1)))
                self.actions_c.append(int(np.clip(act_c[i], 0, COMBAT_ACTIONS - 1)))
                self.actions_t.append(int(np.clip(act_t[i], 0, TARGET_ACTIONS - 1)))
                total_samples += 1

        self.stacked_obs = np.array(self.stacked_obs, dtype=np.float32)
        self.actions_m = np.array(self.actions_m, dtype=np.int64)
        self.actions_c = np.array(self.actions_c, dtype=np.int64)
        self.actions_t = np.array(self.actions_t, dtype=np.int64)

        print(f"Built {total_samples} frame-stacked samples "
              f"(stack={frame_stack}, input_size={SINGLE_OBS_SIZE * frame_stack}) "
              f"from {total_rows} raw rows across {len(csv_files)} files")

    def __len__(self):
        return len(self.stacked_obs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.stacked_obs[idx]),
            self.actions_m[idx],
            self.actions_c[idx],
            self.actions_t[idx],
        )


# ─────────────────────────────────────────────────────────────────
#  PPO Environment Wrapper with Frame Stacking (single env)
# ─────────────────────────────────────────────────────────────────

class FrameStackEnvWrapper(gym.Wrapper):
    """Wraps a CombatEnv to stack N consecutive observations.

    Maintains a ring buffer of the last N observations. On each step,
    pushes the new obs into the buffer and returns the full stack
    concatenated into one flat array.

    On reset, fills the entire buffer with the initial observation
    (same behaviour as C++ BuildFlatInput when FramesFilled < FrameStackCount).
    """

    def __init__(self, env: gym.Env, frame_stack: int = 3):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)

        single_shape = env.observation_space.shape[0]
        stacked_shape = single_shape * frame_stack
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(stacked_shape,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(obs.copy())
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.frames.append(obs.copy())
        return self._get_stacked_obs(), reward, done, truncated, info

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate frames oldest-first (matches C++ BuildFlatInput)."""
        return np.concatenate(list(self.frames))


# ─────────────────────────────────────────────────────────────────
#  Vectorized Environment with Frame Stacking
# ─────────────────────────────────────────────────────────────────

class VecFrameStackEnv:
    """Runs N independent environments in lockstep with frame stacking.

    WHY THIS MATTERS
        With a single env and 2048-step rollouts, you get ~3-5 episodes
        per rollout in stages 3+. That means GAE is estimated from 3-5
        episodes — extremely noisy. With 8 envs you get 25-40 episodes
        per rollout, which dramatically stabilises advantage estimates
        and reduces the variance that causes catastrophic forgetting.

    USAGE
        vec_env = VecFrameStackEnv(
            env_fns=[lambda: make_curriculum_env(3, "ranged") for _ in range(8)],
            frame_stack=3,
        )
        obs = vec_env.reset()                    # (8, 594)
        obs, rew, done, trunc, info = vec_env.step(actions)  # actions: (8, 3)

    AUTORESET
        When an env finishes (done or truncated), it auto-resets and
        returns the FIRST observation of the new episode. The info dict
        for that env contains 'terminal_observation' with the actual
        final observation (needed for value bootstrapping).
    """

    def __init__(
        self,
        env_fns: List[Callable],
        frame_stack: int = 3,
    ):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.frame_stack = frame_stack

        # Determine obs size from first env.
        single_obs_size = self.envs[0].observation_space.shape[0]
        self.stacked_obs_size = single_obs_size * frame_stack

        # Frame buffers per env.
        self._frame_buffers: List[deque] = [
            deque(maxlen=frame_stack) for _ in range(self.num_envs)
        ]

        # Expose action space from first env (all envs are identical).
        self.action_space = self.envs[0].action_space
        self.single_obs_size = single_obs_size

    def reset(self):
        """Reset all environments. Returns stacked obs and initial info with action masks."""
        all_obs = np.zeros((self.num_envs, self.stacked_obs_size), dtype=np.float32)
        all_infos = [{} for _ in range(self.num_envs)]

        for i, env in enumerate(self.envs):
            obs, info = env.reset()
            self._frame_buffers[i].clear()
            for _ in range(self.frame_stack):
                self._frame_buffers[i].append(obs.copy())
            all_obs[i] = np.concatenate(list(self._frame_buffers[i]))
            all_infos[i] = info

        return all_obs, all_infos

    def step(self, actions: np.ndarray):
        """Step all environments. Auto-resets finished envs.

        Args:
            actions: (num_envs, 3) array — [movement, combat, target] per env.

        Returns:
            obs:       (num_envs, stacked_obs_size) — stacked observations.
                       For auto-reset envs, this is the FIRST obs of the new episode.
            rewards:   (num_envs,) — rewards from the step.
            dones:     (num_envs,) — True if episode ended naturally (death/win).
            truncated: (num_envs,) — True if episode hit time limit.
            infos:     list of info dicts. Finished envs get:
                       'terminal_observation': the stacked obs at the moment of
                       termination (before auto-reset). Use this for value
                       bootstrapping on truncation.
        """
        all_obs = np.zeros((self.num_envs, self.stacked_obs_size), dtype=np.float32)
        all_rewards = np.zeros(self.num_envs, dtype=np.float32)
        all_dones = np.zeros(self.num_envs, dtype=bool)
        all_truncated = np.zeros(self.num_envs, dtype=bool)
        all_infos = [{} for _ in range(self.num_envs)]

        for i, env in enumerate(self.envs):
            obs, reward, done, truncated, info = env.step(actions[i])
            self._frame_buffers[i].append(obs.copy())

            all_rewards[i] = reward
            all_dones[i] = done
            all_truncated[i] = truncated
            all_infos[i] = info

            if done or truncated:
                # Save the terminal stacked observation BEFORE resetting.
                terminal_obs = np.concatenate(list(self._frame_buffers[i]))
                all_infos[i]["terminal_observation"] = terminal_obs

                # Auto-reset. Capture the new info (contains action mask).
                new_obs, new_info = env.reset()
                self._frame_buffers[i].clear()
                for _ in range(self.frame_stack):
                    self._frame_buffers[i].append(new_obs.copy())
                # Replace action_mask with the new episode's mask.
                if "action_mask" in new_info:
                    all_infos[i]["action_mask"] = new_info["action_mask"]

            all_obs[i] = np.concatenate(list(self._frame_buffers[i]))

        return all_obs, all_rewards, all_dones, all_truncated, all_infos

    def get_raw_envs(self):
        """Access underlying CombatEnv instances (for win checking etc)."""
        return self.envs

    def close(self):
        for env in self.envs:
            env.close()