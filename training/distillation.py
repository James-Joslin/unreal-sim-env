"""
distillation.py — Knowledge distillation pipeline for combat AI.

TWO MODES:

  STANDARD — Match teacher logits (KL divergence + hard labels).
    Teacher rollouts → collect (obs, logits) → train student to match.
    Used for compressing a trained model into smaller ONNX tiers.

  AMPLIFIED — AlphaGo Zero-style iterated self-improvement.
    Run N rollouts per scenario → keep the best by reward →
    train policy to match the winning actions (reward-weighted).
    Each iteration produces a stronger policy because the search
    (best-of-N) finds better action sequences than the raw policy,
    and distilling them back sharpens the policy for the next round.

    The amplified dataset is "the teacher teaching itself to be better"
    — the policy proposes many strategies, natural selection keeps the
    winners, and the policy learns to produce winners directly.

USAGE:
    # Standard distillation (compress to ONNX tiers):
    python -m training.distillation --teacher checkpoint.pt

    # Amplified distillation (self-improvement):
    python -m training.distillation --teacher checkpoint.pt \\
        --mode amplified --rollouts_per_scenario 16 --top_k 0.25

    # Iterated amplification (multiple rounds):
    python -m training.distillation --teacher checkpoint.pt \\
        --mode amplified --iterations 3
"""

import os
import time as _time
import argparse
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
#  Standard Dataset Generation (teacher rollouts)
# ─────────────────────────────────────────────────────────────────

def generate_teacher_dataset(
    teacher: nn.Module,
    archetype: str,
    stages: list,
    frame_stack: int,
    device: torch.device,
    num_episodes: int = 500,
    obs_normalizer=None,
) -> dict:
    """Roll out the teacher and preserve masks plus episode sequences."""
    from combat_extensions import make_extended_curriculum_env
    from frame_stack import FrameStackEnvWrapper

    teacher.eval()
    all_obs, all_m, all_c, all_t = [], [], [], []
    all_m_masks, all_c_masks, all_t_masks = [], [], []
    all_episode_starts, all_stages = [], []

    eps_per_stage = max(1, num_episodes // len(stages))

    for stage in stages:
        raw_env = make_extended_curriculum_env(stage, archetype)
        env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)

        for ep in range(eps_per_stage):
            obs, _ = env.reset(seed=stage * 100_000 + ep)
            done = False
            episode_start = True

            # GRU hidden state — reset per episode.
            hidden = None
            if hasattr(teacher, 'init_hidden'):
                hidden = teacher.init_hidden(1, device)

            while not done:
                obs_n = obs_normalizer.normalize(obs) if obs_normalizer else obs
                mask_dict = raw_env.build_action_mask()
                if mask_dict.get("skip_inference", False):
                    # Production does not call ONNX on locked ticks. Omit the
                    # transition and keep recurrent state frozen; the next
                    # recorded stacked observation still contains this frame.
                    m = int(np.flatnonzero(mask_dict["m_mask"])[0])
                    c = int(np.flatnonzero(mask_dict["c_mask"])[0])
                    t = int(np.flatnonzero(mask_dict["t_mask"])[0])
                else:
                    with torch.no_grad():
                        obs_t = torch.from_numpy(
                            obs_n).float().unsqueeze(0).to(device)
                        m_l, c_l, t_l, hidden = teacher(obs_t, hidden)

                    all_obs.append(obs_n.copy())
                    all_m.append(m_l.cpu().squeeze(0).numpy())
                    all_c.append(c_l.cpu().squeeze(0).numpy())
                    all_t.append(t_l.cpu().squeeze(0).numpy())
                    all_m_masks.append(mask_dict["m_mask"].copy())
                    all_c_masks.append(mask_dict["c_mask"].copy())
                    all_t_masks.append(mask_dict["t_mask"].copy())
                    all_episode_starts.append(episode_start)
                    all_stages.append(stage)
                    episode_start = False

                    m_mask = torch.from_numpy(
                        mask_dict["m_mask"]).unsqueeze(0).to(device)
                    c_mask = torch.from_numpy(
                        mask_dict["c_mask"]).unsqueeze(0).to(device)
                    t_mask = torch.from_numpy(
                        mask_dict["t_mask"]).unsqueeze(0).to(device)
                    m = m_l.masked_fill(~m_mask, -1e8).argmax(1).item()
                    c = c_l.masked_fill(~c_mask, -1e8).argmax(1).item()
                    t = t_l.masked_fill(~t_mask, -1e8).argmax(1).item()
                obs, _, done, trunc, _ = env.step(np.array([m, c, t]))
                if trunc:
                    break
        env.close()
        print(f"  Stage {stage}: {eps_per_stage} ep, "
              f"{len(all_obs)} transitions total")

    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "m_logits": np.array(all_m, dtype=np.float32),
        "c_logits": np.array(all_c, dtype=np.float32),
        "t_logits": np.array(all_t, dtype=np.float32),
        "m_masks": np.array(all_m_masks, dtype=bool),
        "c_masks": np.array(all_c_masks, dtype=bool),
        "t_masks": np.array(all_t_masks, dtype=bool),
        "episode_starts": np.array(all_episode_starts, dtype=bool),
        "stages": np.array(all_stages, dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────────
#  Amplified Dataset Generation (best-of-N selection)
# ─────────────────────────────────────────────────────────────────

def generate_amplified_dataset(
    policy: nn.Module,
    archetype: str,
    stages: list,
    frame_stack: int,
    device: torch.device,
    num_scenarios: int = 200,
    rollouts_per_scenario: int = 16,
    top_k_fraction: float = 0.25,
    obs_normalizer=None,
) -> dict:
    """AlphaGo-style amplified dataset: best-of-N rollout selection.

    For each scenario (seed), runs N rollouts with the policy using
    stochastic sampling (exploration). Keeps the top K% by episode
    reward. The winning episodes' (state, action, weight) triples
    become training targets.

    This is the "search" phase — equivalent to MCTS in AlphaGo Zero.
    Running multiple rollouts from the same initial state and keeping
    the best effectively searches the action space for strategies the
    raw policy wouldn't consistently find.
    """
    from combat_extensions import make_extended_curriculum_env
    from frame_stack import FrameStackEnvWrapper
    import random as _random

    policy.eval()
    scenarios_per_stage = max(1, num_scenarios // len(stages))
    top_k = max(1, int(rollouts_per_scenario * top_k_fraction))

    all_obs = []
    all_m_acts = []
    all_c_acts = []
    all_t_acts = []
    all_weights = []
    all_m_masks = []
    all_c_masks = []
    all_t_masks = []
    all_episode_starts = []
    all_stages = []

    total_episodes = 0
    total_kept = 0

    for stage in stages:
        for scenario_idx in range(scenarios_per_stage):
            base_seed = stage * 10000 + scenario_idx

            # Run N rollouts from the same seed.
            episode_data = []  # list of (reward, [(obs, m, c, t), ...])

            for rollout_idx in range(rollouts_per_scenario):
                _random.seed(base_seed)
                np.random.seed(base_seed)
                torch.manual_seed(base_seed + rollout_idx * 1000)

                raw_env = make_extended_curriculum_env(stage, archetype)
                env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)
                # Every candidate rollout must start from the same arena.
                # Policy sampling still differs through the rollout torch seed.
                obs, _ = env.reset(seed=base_seed)

                transitions = []
                ep_reward = 0.0
                done = False

                # GRU hidden state — reset per rollout.
                hidden = None
                if hasattr(policy, 'init_hidden'):
                    hidden = policy.init_hidden(1, device)

                while not done:
                    obs_n = (obs_normalizer.normalize(obs)
                             if obs_normalizer else obs)
                    mask_dict = raw_env.build_action_mask()
                    if mask_dict.get("skip_inference", False):
                        m = int(np.flatnonzero(mask_dict["m_mask"])[0])
                        c = int(np.flatnonzero(mask_dict["c_mask"])[0])
                        t = int(np.flatnonzero(mask_dict["t_mask"])[0])
                    else:
                        with torch.no_grad():
                            obs_t = torch.from_numpy(
                                obs_n).float().unsqueeze(0).to(device)
                            masks_t = (
                                torch.from_numpy(mask_dict["m_mask"])
                                .unsqueeze(0).to(device),
                                torch.from_numpy(mask_dict["c_mask"])
                                .unsqueeze(0).to(device),
                                torch.from_numpy(mask_dict["t_mask"])
                                .unsqueeze(0).to(device),
                            )

                            if hasattr(policy, 'sample_actions'):
                                result = policy.sample_actions(
                                    obs_t, masks=masks_t, hidden=hidden)
                                (m_a, c_a, t_a), _ = result[0], result[1]
                                if len(result) > 2:
                                    hidden = result[2]
                                m, c, t = m_a.item(), c_a.item(), t_a.item()
                            elif hasattr(policy, 'get_action_and_value'):
                                result = policy.get_action_and_value(
                                    obs_t, masks=masks_t, hidden=hidden)
                                (m_a, c_a, t_a) = result[0]
                                if len(result) > 4:
                                    hidden = result[4]
                                m, c, t = m_a.item(), c_a.item(), t_a.item()
                            else:
                                m_l, c_l, t_l, hidden = policy(obs_t, hidden)
                                m_l = m_l.masked_fill(~masks_t[0], -1e8)
                                c_l = c_l.masked_fill(~masks_t[1], -1e8)
                                t_l = t_l.masked_fill(~masks_t[2], -1e8)
                                m_dist = torch.distributions.Categorical(logits=m_l)
                                c_dist = torch.distributions.Categorical(logits=c_l)
                                t_dist = torch.distributions.Categorical(logits=t_l)
                                m = m_dist.sample().item()
                                c = c_dist.sample().item()
                                t = t_dist.sample().item()

                        transitions.append((
                            obs_n.copy(), m, c, t,
                            mask_dict["m_mask"].copy(),
                            mask_dict["c_mask"].copy(),
                            mask_dict["t_mask"].copy(),
                        ))
                    obs, reward, done, trunc, _ = env.step(np.array([m, c, t]))
                    ep_reward += reward
                    if trunc:
                        break

                env.close()
                episode_data.append((ep_reward, transitions))
                total_episodes += 1

            # Sort by reward, keep top K.
            episode_data.sort(key=lambda x: x[0], reverse=True)
            kept = episode_data[:top_k]
            total_kept += top_k

            # Compute reward weights (normalised across kept episodes).
            rewards = np.array([r for r, _ in kept])
            if rewards.std() > 1e-6:
                weights = (rewards - rewards.mean()) / rewards.std()
                weights = np.exp(weights)  # softmax-like weighting
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(kept)) / len(kept)

            for i, (ep_r, transitions) in enumerate(kept):
                w = weights[i]
                for step_idx, (obs_n, m, c, t,
                               m_mask, c_mask, t_mask) in enumerate(transitions):
                    all_obs.append(obs_n)
                    all_m_acts.append(m)
                    all_c_acts.append(c)
                    all_t_acts.append(t)
                    all_weights.append(w)
                    all_m_masks.append(m_mask)
                    all_c_masks.append(c_mask)
                    all_t_masks.append(t_mask)
                    all_episode_starts.append(step_idx == 0)
                    all_stages.append(stage)

        print(f"  Stage {stage}: {scenarios_per_stage} scenarios × "
              f"{rollouts_per_scenario} rollouts, "
              f"kept top {top_k} each")

    print(f"  Total: {total_episodes} episodes → "
          f"{total_kept} kept → "
          f"{len(all_obs):,} transitions")

    return {
        "obs": np.array(all_obs, dtype=np.float32),
        "m_acts": np.array(all_m_acts, dtype=np.int64),
        "c_acts": np.array(all_c_acts, dtype=np.int64),
        "t_acts": np.array(all_t_acts, dtype=np.int64),
        "weights": np.array(all_weights, dtype=np.float32),
        "m_masks": np.array(all_m_masks, dtype=bool),
        "c_masks": np.array(all_c_masks, dtype=bool),
        "t_masks": np.array(all_t_masks, dtype=bool),
        "episode_starts": np.array(all_episode_starts, dtype=bool),
        "stages": np.array(all_stages, dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────────
#  Standard KD Training
# ─────────────────────────────────────────────────────────────────

def _episode_ranges(dataset):
    """Return half-open episode ranges, validating dataset alignment."""
    n = len(dataset["obs"])
    if n == 0:
        raise ValueError("Distillation dataset is empty")
    for key, values in dataset.items():
        if len(values) != n:
            raise ValueError(
                f"Dataset field '{key}' has {len(values)} rows; expected {n}")
    starts = np.asarray(dataset["episode_starts"], dtype=bool)
    if not starts[0]:
        raise ValueError("First distillation transition must start an episode")
    begin = np.flatnonzero(starts)
    end = np.concatenate((begin[1:], np.array([n], dtype=np.int64)))
    return [(int(a), int(b)) for a, b in zip(begin, end)]


def _split_episode_ranges(ranges, validation_fraction=0.1):
    """Split on episode boundaries so recurrent context is never severed."""
    if len(ranges) < 2:
        return list(ranges), list(ranges)
    val_n = max(1, int(round(len(ranges) * validation_fraction)))
    val_n = min(val_n, len(ranges) - 1)
    return list(ranges[val_n:]), list(ranges[:val_n])


def _pad_episode_batch(dataset, ranges):
    batch = len(ranges)
    steps = max(end - start for start, end in ranges)
    valid = np.zeros((batch, steps), dtype=bool)
    padded = {}
    for key, values in dataset.items():
        if key == "episode_starts":
            continue
        values = np.asarray(values)
        padded[key] = np.zeros(
            (batch, steps) + values.shape[1:], dtype=values.dtype)
    for row, (start, end) in enumerate(ranges):
        length = end - start
        valid[row, :length] = True
        for key in padded:
            padded[key][row, :length] = dataset[key][start:end]
    padded["valid"] = valid
    return padded


def _iter_sequence_batches(dataset, ranges, batch_size, shuffle):
    """Yield padded batches of complete episodes near a transition budget."""
    ordered = list(ranges)
    if shuffle:
        np.random.shuffle(ordered)
    pending, transitions = [], 0
    for episode in ordered:
        length = episode[1] - episode[0]
        if pending and transitions + length > batch_size:
            yield _pad_episode_batch(dataset, pending)
            pending, transitions = [], 0
        pending.append(episode)
        transitions += length
    if pending:
        yield _pad_episode_batch(dataset, pending)


def _effective_action_masks(student, batch, device):
    """Intersect recorded environment masks with the student tier contract."""
    masks = []
    for key, availability in (
        ("m_masks", student.movement_availability),
        ("c_masks", student.combat_availability),
        ("t_masks", student.target_availability),
    ):
        env_mask = torch.from_numpy(batch[key]).bool().to(device)
        masks.append(env_mask & availability.view(1, 1, -1))
    return tuple(masks)


def _masked_kd_per_step(student_logits, teacher_logits, masks,
                        alpha, temperature):
    """Independent-head KD after projecting teacher mass onto valid actions."""
    total = torch.zeros(student_logits[0].shape[:2],
                        device=student_logits[0].device)
    for student_logit, teacher_logit, mask in zip(
            student_logits, teacher_logits, masks):
        student_logit = student_logit.masked_fill(~mask, -1e8)
        teacher_logit = teacher_logit.masked_fill(~mask, -1e8)
        teacher_soft = F.softmax(teacher_logit / temperature, dim=-1)
        teacher_log = F.log_softmax(teacher_logit / temperature, dim=-1)
        student_log = F.log_softmax(student_logit / temperature, dim=-1)
        kd = (teacher_soft * (teacher_log - student_log)).sum(dim=-1)
        kd = kd * (temperature ** 2)
        hard = teacher_logit.argmax(dim=-1)
        ce = F.cross_entropy(
            student_logit.reshape(-1, student_logit.shape[-1]),
            hard.reshape(-1), reduction="none").reshape_as(hard)
        total = total + alpha * kd + (1.0 - alpha) * ce
    return total

def distill_student(student, dataset, alpha=0.7, temperature=3.0,
                    epochs=50, batch_size=256, lr=3e-4,
                    device=torch.device("cpu"), tier_name=""):
    """Train independent heads on complete recurrent episode sequences."""
    student = student.to(device).train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    train_ranges, val_ranges = _split_episode_ranges(
        _episode_ranges(dataset))
    print(f"  {tier_name or student.tier}: {len(train_ranges)} train / "
          f"{len(val_ranges)} validation episodes; recurrent context kept")

    for epoch in range(epochs):
        total_loss, n_b = 0.0, 0
        for batch in _iter_sequence_batches(
                dataset, train_ranges, batch_size, shuffle=True):
            b_obs = torch.from_numpy(batch["obs"]).float().to(device)
            teacher_logits = tuple(
                torch.from_numpy(batch[key]).float().to(device)
                for key in ("m_logits", "c_logits", "t_logits"))
            valid = torch.from_numpy(batch["valid"]).bool().to(device)
            masks = _effective_action_masks(student, batch, device)
            s_m, s_c, s_t, _ = student.forward_sequence(b_obs)
            per_step = _masked_kd_per_step(
                (s_m, s_c, s_t), teacher_logits, masks,
                alpha, temperature)
            loss = per_step[valid].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            student.eval()
            v_loss, v_b = 0.0, 0
            with torch.no_grad():
                for batch in _iter_sequence_batches(
                        dataset, val_ranges, batch_size, shuffle=False):
                    b_obs = torch.from_numpy(
                        batch["obs"]).float().to(device)
                    teacher_logits = tuple(
                        torch.from_numpy(batch[key]).float().to(device)
                        for key in ("m_logits", "c_logits", "t_logits"))
                    valid = torch.from_numpy(
                        batch["valid"]).bool().to(device)
                    masks = _effective_action_masks(student, batch, device)
                    logits = student.forward_sequence(b_obs)[:3]
                    per_step = _masked_kd_per_step(
                        logits, teacher_logits, masks, 0.0, 1.0)
                    v_loss += per_step[valid].mean().item()
                    v_b += 1
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"train={total_loss/max(n_b,1):.4f}, "
                  f"val_ce={v_loss/max(v_b,1):.4f}")
            student.train()

    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────
#  Amplified Training (reward-weighted behavioral cloning)
# ─────────────────────────────────────────────────────────────────

def distill_amplified(student, dataset, epochs=50, batch_size=256,
                      lr=3e-4, device=torch.device("cpu"),
                      tier_name=""):
    """Train student on reward-weighted best-of-N actions.

    Unlike standard KD (match teacher logits), amplified distillation
    trains the policy to reproduce the ACTIONS from the highest-scoring
    rollouts, weighted by their episode return. This is reward-weighted
    regression — the AlphaGo equivalent of training the policy network
    to match MCTS-improved move probabilities.
    """
    student = student.to(device).train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    train_ranges, val_ranges = _split_episode_ranges(
        _episode_ranges(dataset))
    print(f"  {tier_name or student.tier}: {len(train_ranges)} train / "
          f"{len(val_ranges)} validation episodes; recurrent context kept")

    for epoch in range(epochs):
        total_loss, n_b = 0.0, 0
        for batch in _iter_sequence_batches(
                dataset, train_ranges, batch_size, shuffle=True):
            b_obs = torch.from_numpy(batch["obs"]).float().to(device)
            labels = tuple(
                torch.from_numpy(batch[key]).long().to(device)
                for key in ("m_acts", "c_acts", "t_acts"))
            weights = torch.from_numpy(batch["weights"]).float().to(device)
            valid = torch.from_numpy(batch["valid"]).bool().to(device)
            masks = _effective_action_masks(student, batch, device)
            logits = student.forward_sequence(b_obs)[:3]

            # Heads are independent. Keep supervision for each available
            # action and deterministically drop only that head's unavailable
            # label when the source tier has a broader action contract.
            numerator = torch.tensor(0.0, device=device)
            denominator = torch.tensor(0.0, device=device)
            for logit, label, mask in zip(logits, labels, masks):
                logit = logit.masked_fill(~mask, -1e8)
                allowed = mask.gather(-1, label.unsqueeze(-1)).squeeze(-1)
                supervised = valid & allowed
                ce = F.cross_entropy(
                    logit.reshape(-1, logit.shape[-1]), label.reshape(-1),
                    reduction="none").reshape_as(label)
                numerator = numerator + (ce * weights * supervised).sum()
                denominator = denominator + (weights * supervised).sum()
            if denominator.item() <= 0:
                continue
            loss = numerator / denominator

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            student.eval()
            v_loss, v_b = 0.0, 0
            with torch.no_grad():
                for batch in _iter_sequence_batches(
                        dataset, val_ranges, batch_size, shuffle=False):
                    b_obs = torch.from_numpy(
                        batch["obs"]).float().to(device)
                    labels = tuple(
                        torch.from_numpy(batch[key]).long().to(device)
                        for key in ("m_acts", "c_acts", "t_acts"))
                    valid = torch.from_numpy(
                        batch["valid"]).bool().to(device)
                    masks = _effective_action_masks(student, batch, device)
                    logits = student.forward_sequence(b_obs)[:3]
                    head_losses = []
                    for logit, label, mask in zip(logits, labels, masks):
                        logit = logit.masked_fill(~mask, -1e8)
                        allowed = mask.gather(
                            -1, label.unsqueeze(-1)).squeeze(-1)
                        supervised = valid & allowed
                        if supervised.any():
                            ce = F.cross_entropy(
                                logit.reshape(-1, logit.shape[-1]),
                                label.reshape(-1), reduction="none"
                            ).reshape_as(label)
                            head_losses.append(ce[supervised].mean())
                    if head_losses:
                        v_loss += torch.stack(head_losses).mean().item()
                        v_b += 1
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"train={total_loss/max(n_b,1):.4f}, "
                  f"val_ce={v_loss/max(v_b,1):.4f}")
            student.train()

    student.eval()
    return student


# ─────────────────────────────────────────────────────────────────
#  ONNX Benchmark
# ─────────────────────────────────────────────────────────────────

def benchmark_onnx(onnx_path, input_size, gru_hidden=0,
                   n_iters=500):
    """Returns mean ms per forward pass, or None."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Benchmark skipped (onnxruntime not installed)")
        return None

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, input_size).astype(np.float32)

    # Build feed dict — include hidden_in if model has GRU.
    feed = {"observation": dummy}
    input_names = [inp.name for inp in sess.get_inputs()]
    if "hidden_in" in input_names:
        h_size = gru_hidden if gru_hidden > 0 else 96  # fallback (Large tier default)
        # Try to get exact shape from the model's input spec.
        for inp in sess.get_inputs():
            if inp.name == "hidden_in":
                shape = inp.shape
                # shape is [1, 'batch_size', gru_hidden] or [1, 1, N]
                if len(shape) == 3 and isinstance(shape[2], int):
                    h_size = shape[2]
                break
        feed["hidden_in"] = np.zeros((1, 1, h_size), dtype=np.float32)

    for _ in range(50):
        sess.run(None, feed)

    start = _time.perf_counter()
    for _ in range(n_iters):
        sess.run(None, feed)
    elapsed = _time.perf_counter() - start

    ms = (elapsed / n_iters) * 1000
    print(f"  Inference: {ms:.3f} ms/forward "
          f"({n_iters/elapsed:.0f} FPS)")
    return ms


# ─────────────────────────────────────────────────────────────────
#  Full Pipeline
# ─────────────────────────────────────────────────────────────────

DEFAULT_DISTILL_CHAIN = [
    ("large",  None,     0.7, 3.0, 0),
    ("medium", "large",  0.7, 3.0, 1),
    ("small",  "medium", 0.7, 3.0, 1),
    ("micro",  "small",  0.5, 4.0, 1),
]


def run_distillation(
    teacher_path: str,
    output_dir: str,
    frame_stack: int = 3,
    archetype: str = "ranged",
    num_episodes: int = 500,
    epochs: int = 50,
    mode: str = "amplified",
    rollouts_per_scenario: int = 16,
    top_k_fraction: float = 0.25,
    iterations: int = 1,
    distill_chain: list = None,
):
    """Full distillation pipeline.

    Modes:
        standard:  Match teacher logits → active ONNX tiers.
        amplified: Best-of-N self-improvement → distill → ONNX tiers.
                   With iterations > 1, repeats the amplify/distill
                   loop (iterated amplification à la AlphaGo Zero).
    """
    from combat_policy import (
        load_teacher_from_checkpoint, export_onnx, verify_export,
        make_policy, resolve_tier, ACTIVE_TIERS, TRAINABLE_ARCHETYPES,
        BEHAVIOR_TIER_DEFINITIONS,
    )
    from frame_stack import stacked_obs_size

    use_default_chain = distill_chain is None
    if archetype not in TRAINABLE_ARCHETYPES:
        raise ValueError(
            f"Archetype '{archetype}' is not trainable. Active archetypes: "
            f"{', '.join(TRAINABLE_ARCHETYPES)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
    fs = ckpt.get("frame_stack", frame_stack)
    input_size = stacked_obs_size(fs)
    raw_teacher_tier = ckpt.get("tier", "large")
    teacher_tier = resolve_tier(raw_teacher_tier)
    teacher_stage = max(1, min(int(ckpt.get("stage", 3)), 7))
    stages = list(range(1, teacher_stage + 1))

    if use_default_chain:
        start = [tier for tier, *_ in DEFAULT_DISTILL_CHAIN].index(
            teacher_tier)
        selected = DEFAULT_DISTILL_CHAIN[start:]
        distill_chain = []
        for index, (tier, _, alpha, temp, epoch_mult) in enumerate(selected):
            source = None if index == 0 else selected[index - 1][0]
            distill_chain.append((tier, source, alpha, temp, epoch_mult))
    for tier, source, *_ in distill_chain:
        if tier not in ACTIVE_TIERS or (
                source is not None and source not in ACTIVE_TIERS):
            raise ValueError(
                "Distillation chains may use only active tiers: "
                f"{', '.join(ACTIVE_TIERS)}")

    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        from training.normalizers import RunningNormalizer
        obs_normalizer = RunningNormalizer(input_size)
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"Loaded observation normalizer")

    teacher = load_teacher_from_checkpoint(teacher_path, device)
    print(f"Teacher: tier={raw_teacher_tier} -> {teacher_tier}, "
          f"stage={teacher_stage}, "
          f"params={sum(p.numel() for p in teacher.parameters()):,}")

    current_policy = teacher

    # ── Iterated amplification loop ──────────────────────────────
    for iteration in range(iterations):
        if iterations > 1:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration+1}/{iterations}")
            print(f"{'='*60}")

        # ── Generate dataset ─────────────────────────────────────
        if mode == "amplified":
            print(f"\nAmplified dataset (best-of-{rollouts_per_scenario}, "
                  f"top {top_k_fraction:.0%})...")
            dataset = generate_amplified_dataset(
                current_policy, archetype=archetype, stages=stages,
                frame_stack=fs, device=device,
                num_scenarios=num_episodes,
                rollouts_per_scenario=rollouts_per_scenario,
                top_k_fraction=top_k_fraction,
                obs_normalizer=obs_normalizer,
            )
            train_fn = distill_amplified
        else:
            print(f"\nStandard dataset ({num_episodes} episodes)...")
            dataset = generate_teacher_dataset(
                current_policy, archetype=archetype, stages=stages,
                frame_stack=fs, device=device,
                num_episodes=num_episodes,
                obs_normalizer=obs_normalizer,
            )
            train_fn = distill_student

        # ── Distill chain ────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"DISTILLATION — {mode.upper()} mode")
        print(f"{'='*60}")

        models = {teacher_tier: current_policy}
        results = {}

        for tier, src_tier, alpha, temp, epoch_mult in distill_chain:
            print(f"\n── {tier.upper()} {'─' * (55 - len(tier))}──")

            if src_tier is None:
                models[tier] = current_policy
                print(f"  Current policy "
                      f"({sum(p.numel() for p in current_policy.parameters()):,} params)")
            else:
                if src_tier not in models:
                    print(f"  ⚠ Skipping — {src_tier} not available")
                    continue

                student = make_policy(tier, frame_stack=fs)
                params = sum(p.numel() for p in student.parameters())
                tier_epochs = epochs * epoch_mult
                print(f"  Student: {params:,} params, "
                      f"{tier_epochs} epochs")

                if mode == "amplified":
                    student = train_fn(
                        student, dataset, epochs=tier_epochs,
                        device=device, tier_name=tier)
                else:
                    # For cascaded standard KD, regenerate logits
                    # from the source tier model.
                    if src_tier != teacher_tier:
                        src_dataset = _regenerate_logits(
                            models[src_tier], dataset, device)
                    else:
                        src_dataset = dataset
                    student = train_fn(
                        student, src_dataset, alpha=alpha,
                        temperature=temp, epochs=tier_epochs,
                        device=device, tier_name=tier)

                models[tier] = student

            # Export + verify + benchmark + evaluate.
            onnx_path = export_onnx(
                models[tier], tier, output_dir,
                frame_stack=fs, obs_normalizer=obs_normalizer)
            verify_export(
                models[tier], onnx_path,
                frame_stack=fs, obs_normalizer=obs_normalizer)
            ms = benchmark_onnx(
                onnx_path, input_size, models[tier].gru_hidden)

            # In-sim evaluation — the real test of distillation quality.
            eval_stats = _evaluate_tier(
                models[tier], stages[-1], archetype, fs,
                obs_normalizer, device, num_eval_episodes=50)

            params = sum(p.numel() for p in models[tier].parameters())
            results[tier] = {
                "params": params, "ms": ms, "path": onnx_path,
                "win_rate": eval_stats["win_rate"],
                "mean_reward": eval_stats["mean_reward"],
                "mean_kills": eval_stats["mean_kills"],
                "behavior": BEHAVIOR_TIER_DEFINITIONS[tier]["label"],
            }

        # For iterated amplification, the improved teacher-tier model
        # becomes the policy for the next iteration.
        if teacher_tier in models and iterations > 1:
            current_policy = models[teacher_tier]

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Tier':>8} | {'Params':>10} | {'Size':>8} | {'Latency':>10} | {'Win':>5} | {'Kills':>5} | {'Reward':>7}")
    print(f"{'-'*70}")
    for tier in [t for t, *_ in distill_chain if t in results]:
        r = results[tier]
        size_kb = os.path.getsize(r["path"]) / 1024
        ms_str = f"{r['ms']:.3f} ms" if r["ms"] else "N/A"
        print(f"{tier:>8} | {r['params']:>10,} | {size_kb:>6.1f} KB | {ms_str:>10} "
              f"| {r['win_rate']:>4.0%} | {r['mean_kills']:>5.1f} | {r['mean_reward']:>+7.1f}")
    print(f"{'='*70}")


def _evaluate_tier(model, stage, archetype, frame_stack,
                   obs_normalizer, device, num_eval_episodes=50):
    """Run in-sim evaluation for a distilled model tier.

    Uses the same seeded evaluation as training — deterministic
    scenarios with independent, masked action selection.
    """
    from training.evaluation import evaluate

    model.eval().to(device)
    stats = evaluate(
        model, stage, archetype, num_eval_episodes,
        device, frame_stack=frame_stack,
        obs_normalizer=obs_normalizer,
        base_seed=42,
        is_actor_critic=False,  # CombatPolicy has no value head.
    )
    print(f"  Eval: win={stats['win_rate']:.0%}, "
          f"kills={stats['mean_kills']:.1f}, "
          f"reward={stats['mean_reward']:+.1f}")
    return stats


def _regenerate_logits(model, dataset, device):
    """Regenerate cascaded teacher logits with intact recurrent context."""
    model.eval().to(device)
    regenerated = {
        key: np.array(values, copy=True)
        for key, values in dataset.items()
    }
    n = len(dataset["obs"])
    regenerated["m_logits"] = np.empty(
        (n, len(model.movement_availability)), dtype=np.float32)
    regenerated["c_logits"] = np.empty(
        (n, len(model.combat_availability)), dtype=np.float32)
    regenerated["t_logits"] = np.empty(
        (n, len(model.target_availability)), dtype=np.float32)
    for start, end in _episode_ranges(dataset):
        obs = torch.from_numpy(
            dataset["obs"][start:end]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            movement, combat, target, _ = model.forward_sequence(obs)
        regenerated["m_logits"][start:end] = movement[0].cpu().numpy()
        regenerated["c_logits"][start:end] = combat[0].cpu().numpy()
        regenerated["t_logits"][start:end] = target[0].cpu().numpy()
    return regenerated


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def main():
    from combat_policy import TRAINABLE_ARCHETYPES
    parser = argparse.ArgumentParser(
        description="Distill combat AI into ONNX tiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard distillation:
  %(prog)s --teacher checkpoints/ppo_stage4_best.pt

  # Amplified (AlphaGo-style best-of-N):
  %(prog)s --teacher checkpoints/ppo_stage4_best.pt \\
      --mode amplified --rollouts 16 --top_k 0.25

  # Iterated amplification (3 rounds):
  %(prog)s --teacher checkpoints/ppo_stage4_best.pt \\
      --mode amplified --iterations 3
        """)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--archetype", default="ranged",
                        choices=TRAINABLE_ARCHETYPES)
    parser.add_argument("--mode", choices=["standard", "amplified"],
                        default="standard")
    parser.add_argument("--rollouts", type=int, default=16,
                        help="Rollouts per scenario (amplified mode)")
    parser.add_argument("--top_k", type=float, default=0.25,
                        help="Fraction of rollouts to keep (amplified mode)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Amplification iterations (amplified mode)")
    args = parser.parse_args()

    run_distillation(
        teacher_path=args.teacher,
        output_dir=args.output_dir,
        frame_stack=args.frame_stack,
        archetype=args.archetype,
        num_episodes=args.episodes,
        epochs=args.epochs,
        mode=args.mode,
        rollouts_per_scenario=args.rollouts,
        top_k_fraction=args.top_k,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
