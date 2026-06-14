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
    """Roll out the teacher, collect (obs, logits) pairs."""
    from combat_extensions import make_extended_curriculum_env
    from frame_stack import FrameStackEnvWrapper

    teacher.eval()
    all_obs, all_m, all_c, all_t = [], [], [], []

    eps_per_stage = max(1, num_episodes // len(stages))

    for stage in stages:
        raw_env = make_extended_curriculum_env(stage, archetype)
        env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)

        for ep in range(eps_per_stage):
            obs, _ = env.reset()
            done = False

            # GRU hidden state — reset per episode.
            hidden = None
            if hasattr(teacher, 'init_hidden'):
                hidden = teacher.init_hidden(1, device)

            while not done:
                obs_n = obs_normalizer.normalize(obs) if obs_normalizer else obs
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_n).float().unsqueeze(0).to(device)
                    m_l, c_l, t_l, hidden = teacher(obs_t, hidden)

                all_obs.append(obs_n.copy())
                all_m.append(m_l.cpu().squeeze(0).numpy())
                all_c.append(c_l.cpu().squeeze(0).numpy())
                all_t.append(t_l.cpu().squeeze(0).numpy())

                m = m_l.argmax(1).item()
                c = c_l.argmax(1).item()
                t = t_l.argmax(1).item()
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
                obs, _ = env.reset()

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
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_n).float().unsqueeze(0).to(device)

                        if hasattr(policy, 'sample_actions'):
                            result = policy.sample_actions(obs_t, hidden=hidden)
                            (m_a, c_a, t_a), _ = result[0], result[1]
                            if len(result) > 2:
                                hidden = result[2]
                            m, c, t = m_a.item(), c_a.item(), t_a.item()
                        elif hasattr(policy, 'get_action_and_value'):
                            result = policy.get_action_and_value(
                                obs_t, hidden=hidden)
                            (m_a, c_a, t_a) = result[0]
                            if len(result) > 4:
                                hidden = result[4]
                            m, c, t = m_a.item(), c_a.item(), t_a.item()
                        else:
                            m_l, c_l, t_l, hidden = policy(obs_t, hidden)
                            m_dist = torch.distributions.Categorical(logits=m_l)
                            c_dist = torch.distributions.Categorical(logits=c_l)
                            t_dist = torch.distributions.Categorical(logits=t_l)
                            m = m_dist.sample().item()
                            c = c_dist.sample().item()
                            t = t_dist.sample().item()

                    transitions.append((obs_n.copy(), m, c, t))
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
                for obs_n, m, c, t in transitions:
                    all_obs.append(obs_n)
                    all_m_acts.append(m)
                    all_c_acts.append(c)
                    all_t_acts.append(t)
                    all_weights.append(w)

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
    }


# ─────────────────────────────────────────────────────────────────
#  Standard KD Training
# ─────────────────────────────────────────────────────────────────

def distill_student(student, dataset, alpha=0.7, temperature=3.0,
                    epochs=50, batch_size=256, lr=3e-4,
                    device=torch.device("cpu"), tier_name=""):
    """Train student to match teacher logits (standard KD)."""
    student = student.to(device).train()
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    obs_arr = dataset["obs"]
    m_arr, c_arr, t_arr = dataset["m_logits"], dataset["c_logits"], dataset["t_logits"]

    n = len(obs_arr)
    val_n = max(1, int(n * 0.1))
    indices = np.arange(n)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        total_loss, n_b = 0.0, 0

        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[i:i + batch_size]
            b_obs = torch.from_numpy(obs_arr[idx]).to(device)
            t_m = torch.from_numpy(m_arr[idx]).to(device)
            t_c = torch.from_numpy(c_arr[idx]).to(device)
            t_t = torch.from_numpy(t_arr[idx]).to(device)

            s_m, s_c, s_t, _ = student(b_obs)
            loss = torch.tensor(0.0, device=device)
            for s_l, t_l in [(s_m, t_m), (s_c, t_c), (s_t, t_t)]:
                t_soft = F.softmax(t_l / temperature, dim=-1)
                s_log = F.log_softmax(s_l / temperature, dim=-1)
                kd = F.kl_div(s_log, t_soft, reduction="batchmean") * (temperature ** 2)
                ce = F.cross_entropy(s_l, t_l.argmax(dim=-1))
                loss = loss + alpha * kd + (1 - alpha) * ce

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            student.eval()
            v_loss, v_b = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_idx), batch_size):
                    idx = val_idx[i:i + batch_size]
                    b = torch.from_numpy(obs_arr[idx]).to(device)
                    s_m, s_c, s_t, _ = student(b)
                    for s_l, t_l_np in [(s_m, m_arr[idx]), (s_c, c_arr[idx]), (s_t, t_arr[idx])]:
                        t_l = torch.from_numpy(t_l_np).to(device)
                        v_loss += F.cross_entropy(s_l, t_l.argmax(dim=-1)).item()
                    v_b += 1
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"train={total_loss/max(n_b,1):.4f}, "
                  f"val_ce={v_loss/max(v_b*3,1):.4f}")
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

    obs_arr = dataset["obs"]
    m_arr = dataset["m_acts"]
    c_arr = dataset["c_acts"]
    t_arr = dataset["t_acts"]
    w_arr = dataset["weights"]

    n = len(obs_arr)
    val_n = max(1, int(n * 0.1))
    indices = np.arange(n)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        total_loss, n_b = 0.0, 0

        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[i:i + batch_size]
            b_obs = torch.from_numpy(obs_arr[idx]).to(device)
            b_m = torch.from_numpy(m_arr[idx]).long().to(device)
            b_c = torch.from_numpy(c_arr[idx]).long().to(device)
            b_t = torch.from_numpy(t_arr[idx]).long().to(device)
            b_w = torch.from_numpy(w_arr[idx]).to(device)

            s_m, s_c, s_t, _ = student(b_obs)

            # Weighted cross-entropy: high-reward episodes count more.
            ce_m = F.cross_entropy(s_m, b_m, reduction="none")
            ce_c = F.cross_entropy(s_c, b_c, reduction="none")
            ce_t = F.cross_entropy(s_t, b_t, reduction="none")
            loss = (b_w * (ce_m + ce_c + ce_t)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_b += 1

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            student.eval()
            v_loss, v_b = 0.0, 0
            with torch.no_grad():
                for i in range(0, len(val_idx), batch_size):
                    idx = val_idx[i:i + batch_size]
                    b = torch.from_numpy(obs_arr[idx]).to(device)
                    s_m, s_c, s_t, _ = student(b)
                    v_loss += F.cross_entropy(s_m, torch.from_numpy(m_arr[idx]).long().to(device)).item()
                    v_loss += F.cross_entropy(s_c, torch.from_numpy(c_arr[idx]).long().to(device)).item()
                    v_loss += F.cross_entropy(s_t, torch.from_numpy(t_arr[idx]).long().to(device)).item()
                    v_b += 1
            print(f"    Epoch {epoch+1}/{epochs}: "
                  f"train={total_loss/max(n_b,1):.4f}, "
                  f"val_ce={v_loss/max(v_b*3,1):.4f}")
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
    ("xl",     "large",  0.7, 3.0, 1),
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
        standard:  Match teacher logits → 5 ONNX tiers.
        amplified: Best-of-N self-improvement → distill → ONNX tiers.
                   With iterations > 1, repeats the amplify/distill
                   loop (iterated amplification à la AlphaGo Zero).
    """
    from combat_policy import (
        load_teacher_from_checkpoint, export_onnx, verify_export,
        make_policy, OBS_SIZE,
    )
    from frame_stack import stacked_obs_size

    if distill_chain is None:
        distill_chain = DEFAULT_DISTILL_CHAIN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load checkpoint ──────────────────────────────────────────
    ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
    fs = ckpt.get("frame_stack", frame_stack)
    input_size = stacked_obs_size(fs)
    teacher_tier = ckpt.get("tier", "large")
    teacher_stage = ckpt.get("stage", 3)
    stages = list(range(1, min(teacher_stage + 1, 8)))

    obs_normalizer = None
    if "obs_normalizer" in ckpt:
        from training.normalizers import RunningNormalizer
        obs_normalizer = RunningNormalizer(input_size)
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"Loaded observation normalizer")

    teacher = load_teacher_from_checkpoint(teacher_path, device)
    print(f"Teacher: tier={teacher_tier}, stage={teacher_stage}, "
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
                            models[src_tier], dataset["obs"], device)
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
            ms = benchmark_onnx(onnx_path, input_size)

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
    scenarios with autoregressive masked action selection.
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


def _regenerate_logits(model, obs_array, device, batch_size=512):
    """Regenerate logits from a source-tier model for cascaded KD.

    NOTE: GRU hidden state is zero-initialised for every batch because
    the dataset has no episode structure (observations are shuffled).
    The regenerated logits therefore don't reflect temporal context.
    This is acceptable for cascaded distillation — the student learns
    a per-observation logit mapping, and the GRU builds context at
    sequential inference time.
    """
    model.eval()
    all_m, all_c, all_t = [], [], []
    for i in range(0, len(obs_array), batch_size):
        batch = torch.from_numpy(obs_array[i:i+batch_size]).to(device)
        with torch.no_grad():
            m, c, t, _ = model(batch)
        all_m.append(m.cpu().numpy())
        all_c.append(c.cpu().numpy())
        all_t.append(t.cpu().numpy())
    return {
        "obs": obs_array,
        "m_logits": np.concatenate(all_m),
        "c_logits": np.concatenate(all_c),
        "t_logits": np.concatenate(all_t),
    }


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def main():
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
    parser.add_argument("--archetype", default="ranged")
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