"""
02_distill_and_export.py — Full distillation pipeline for all model tiers.

WHAT THIS DOES
    1. Loads a trained teacher from a PPO checkpoint (strips critic).
    2. Generates training data by rolling the teacher out in the Python sim
       (covers the full state distribution, not just scripted brain recordings).
    3. Distills Large → Medium → Small → Micro (cascading).
    4. Fine-tunes Large → XL (more capacity, same data).
    5. Exports every tier to ONNX.
    6. Evaluates every tier's actual combat performance (win rate, kills,
       reward) across multiple curriculum stages.
    7. Benchmarks inference time per tier.
    8. Prints a comparison report.

USAGE
    python 02_distill_and_export.py \
        --teacher checkpoints/ppo_stage7_best.pt \
        --output_dir models/v1

    # More rollout data for better distillation:
    python 02_distill_and_export.py \
        --teacher checkpoints/ppo_stage7_best.pt \
        --num_episodes 

    # Eval only (skip distillation, load existing ONNX):
    python 02_distill_and_export.py \
        --teacher checkpoints/ppo_stage7_best.pt \
        --eval_only

OUTPUTS
    models/v1/Combat_Micro.onnx
    models/v1/Combat_Small.onnx
    models/v1/Combat_Medium.onnx
    models/v1/Combat_Large.onnx
    models/v1/Combat_Xl.onnx
    models/v1/distillation_report.csv
"""

import argparse
import os
import time
import random as _random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from combat_sim import OBS_SIZE, MOVEMENT_ACTIONS, COMBAT_ACTIONS, TARGET_ACTIONS
from combat_extensions import make_extended_curriculum_env
from frame_stack import FrameStackEnvWrapper, stacked_obs_size
from combat_policy import (
    CombatPolicy, make_policy, load_teacher_from_checkpoint,
    export_onnx, verify_export, TIER_CONFIGS,
)

DEFAULT_FRAME_STACK = 1


# ─────────────────────────────────────────────────────────────────
#  Distillation Loss
# ─────────────────────────────────────────────────────────────────

def distillation_loss(student_logits, teacher_logits, target_actions,
                      alpha=0.7, temperature=3.0):
    """Combined loss: hard labels (CE) + soft labels (KL from teacher)."""
    total = 0.0
    for s_log, t_log, actions in zip(student_logits, teacher_logits, target_actions):
        ce = F.cross_entropy(s_log, actions)
        s_soft = F.log_softmax(s_log / temperature, dim=-1)
        t_soft = F.softmax(t_log / temperature, dim=-1)
        kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (temperature ** 2)
        total += (1 - alpha) * ce + alpha * kl
    return total


# ─────────────────────────────────────────────────────────────────
#  Teacher Rollout Dataset
# ─────────────────────────────────────────────────────────────────

def generate_teacher_dataset(
    teacher: CombatPolicy,
    num_episodes: int = 5000,
    stages: List[int] = None,
    archetype: str = "ranged",
    frame_stack: int = DEFAULT_FRAME_STACK,
    device: torch.device = torch.device("cpu"),
    obs_normalizer=None,
) -> TensorDataset:
    """Roll out the teacher in the sim and record obs + soft logits."""
    if stages is None:
        stages = [3, 4, 5, 6, 7]

    teacher.eval()

    all_obs = []
    all_m_logits = []
    all_c_logits = []
    all_t_logits = []
    all_m_acts = []
    all_c_acts = []
    all_t_acts = []

    start_time = time.time()

    for ep in range(num_episodes):
        stage = _random.choice(stages)
        raw_env = make_extended_curriculum_env(stage, archetype)
        env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)

        _random.seed(ep)
        np.random.seed(ep)

        obs, _ = env.reset()
        done = False

        while not done:
            if obs_normalizer:
                obs_input = obs_normalizer.normalize(obs)
            else:
                obs_input = obs

            # Build action mask (matches PPO training — prevents invalid actions).
            mask_dict = raw_env.build_action_mask()
            m_mask = torch.from_numpy(mask_dict["m_mask"]).unsqueeze(0)
            c_mask = torch.from_numpy(mask_dict["c_mask"]).unsqueeze(0)
            t_mask = torch.from_numpy(mask_dict["t_mask"]).unsqueeze(0)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_input).float().unsqueeze(0).to(device)
                m_l, c_l, t_l = teacher(obs_t)

                # Mask invalid actions before sampling and storing logits.
                m_l = m_l.masked_fill(~m_mask.to(device), -1e8)
                c_l = c_l.masked_fill(~c_mask.to(device), -1e8)
                t_l = t_l.masked_fill(~t_mask.to(device), -1e8)

                m = torch.distributions.Categorical(logits=m_l).sample().item()
                c = torch.distributions.Categorical(logits=c_l).sample().item()
                t = torch.distributions.Categorical(logits=t_l).sample().item()

            all_obs.append(obs_input.copy())
            all_m_logits.append(m_l.squeeze(0).cpu().numpy())
            all_c_logits.append(c_l.squeeze(0).cpu().numpy())
            all_t_logits.append(t_l.squeeze(0).cpu().numpy())
            all_m_acts.append(m)
            all_c_acts.append(c)
            all_t_acts.append(t)

            obs, _, done, truncated, _ = env.step(np.array([m, c, t]))
            if truncated:
                done = True

        env.close()

        if (ep + 1) % 500 == 0:
            elapsed = time.time() - start_time
            fps = len(all_obs) / elapsed
            print(f"  {ep+1}/{num_episodes} episodes, "
                  f"{len(all_obs):,} frames ({fps:.0f} frames/s)")

    print(f"Teacher dataset: {len(all_obs):,} frames from "
          f"{num_episodes} episodes ({time.time()-start_time:.0f}s)")

    return TensorDataset(
        torch.from_numpy(np.array(all_obs, dtype=np.float32)),
        torch.from_numpy(np.array(all_m_logits, dtype=np.float32)),
        torch.from_numpy(np.array(all_c_logits, dtype=np.float32)),
        torch.from_numpy(np.array(all_t_logits, dtype=np.float32)),
        torch.tensor(all_m_acts, dtype=torch.long),
        torch.tensor(all_c_acts, dtype=torch.long),
        torch.tensor(all_t_acts, dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────
#  Distillation Training
# ─────────────────────────────────────────────────────────────────

def distill_from_teacher_data(
    student: CombatPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    alpha: float = 0.7,
    temperature: float = 3.0,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    tier_name: str = "student",
) -> CombatPolicy:
    """Train student on pre-recorded teacher logits."""
    student = student.to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"\n  Distilling {tier_name} ({s_params:,} params), "
          f"α={alpha}, T={temperature}, epochs={epochs}")

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        train_correct = {"m": 0, "c": 0, "t": 0}
        train_total = 0

        for batch in train_loader:
            obs, t_m, t_c, t_t, a_m, a_c, a_t = [b.to(device) for b in batch]

            s_m, s_c, s_t = student(obs)
            loss = distillation_loss(
                (s_m, s_c, s_t), (t_m, t_c, t_t), (a_m, a_c, a_t),
                alpha=alpha, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            train_total += obs.size(0)
            train_correct["m"] += (s_m.argmax(1) == a_m).sum().item()
            train_correct["c"] += (s_c.argmax(1) == a_c).sum().item()
            train_correct["t"] += (s_t.argmax(1) == a_t).sum().item()

        scheduler.step()
        train_acc = {k: v / train_total * 100 for k, v in train_correct.items()}

        # Validation.
        val_loss = 0.0
        val_acc = {"m": 0, "c": 0, "t": 0}
        val_total = 0
        if val_loader:
            student.eval()
            with torch.no_grad():
                for batch in val_loader:
                    obs, t_m, t_c, t_t, a_m, a_c, a_t = [b.to(device) for b in batch]
                    s_m, s_c, s_t = student(obs)
                    loss = distillation_loss(
                        (s_m, s_c, s_t), (t_m, t_c, t_t), (a_m, a_c, a_t),
                        alpha=alpha, temperature=temperature)
                    val_loss += loss.item() * obs.size(0)
                    val_total += obs.size(0)
                    val_acc["m"] += (s_m.argmax(1) == a_m).sum().item()
                    val_acc["c"] += (s_c.argmax(1) == a_c).sum().item()
                    val_acc["t"] += (s_t.argmax(1) == a_t).sum().item()

            val_acc = {k: v / max(val_total, 1) * 100 for k, v in val_acc.items()}
            avg_val_loss = val_loss / max(val_total, 1)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in student.state_dict().items()}

        # Logging.
        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == epochs - 1:
            line = (f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"Train: loss={train_loss/train_total:.4f} "
                    f"M={train_acc['m']:.1f}% C={train_acc['c']:.1f}% T={train_acc['t']:.1f}%")
            if val_loader:
                line += (f" | Val: loss={avg_val_loss:.4f} "
                         f"M={val_acc['m']:.1f}% C={val_acc['c']:.1f}% T={val_acc['t']:.1f}%")
            print(line)

    # Restore best validation model.
    if best_state is not None:
        student.load_state_dict(best_state)
        print(f"  Restored best validation model (loss={best_val_loss:.4f})")

    return student


# ─────────────────────────────────────────────────────────────────
#  Combat Evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_combat(
    model: nn.Module,
    stages: List[int] = None,
    archetype: str = "ranged",
    episodes_per_stage: int = 50,
    frame_stack: int = DEFAULT_FRAME_STACK,
    device: torch.device = torch.device("cpu"),
    obs_normalizer=None,
    base_seed: int = 42,
    is_policy_only: bool = True,
) -> Dict:
    """Evaluate a model's actual combat performance across stages."""
    if stages is None:
        stages = [5, 7]

    model.eval()
    all_results = {}

    for stage in stages:
        raw_env = make_extended_curriculum_env(stage, archetype)
        env = FrameStackEnvWrapper(raw_env, frame_stack=frame_stack)

        rewards, wins, kills, lengths = [], [], [], []
        hp_remaining = []

        for ep in range(episodes_per_stage):
            _random.seed(base_seed + stage * 1000 + ep)
            np.random.seed(base_seed + stage * 1000 + ep)

            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            steps = 0
            num_targets = len(raw_env.targets)

            while not done:
                if obs_normalizer:
                    obs_input = obs_normalizer.normalize(obs)
                else:
                    obs_input = obs

                # Build action mask (must match PPO eval for fair comparison).
                mask_dict = raw_env.build_action_mask()
                m_mask = torch.from_numpy(mask_dict["m_mask"]).unsqueeze(0).to(device)
                c_mask = torch.from_numpy(mask_dict["c_mask"]).unsqueeze(0).to(device)
                t_mask = torch.from_numpy(mask_dict["t_mask"]).unsqueeze(0).to(device)

                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_input).float().unsqueeze(0).to(device)
                    outputs = model(obs_t)
                    if is_policy_only:
                        m_l, c_l, t_l = outputs
                    else:
                        m_l, c_l, t_l, _ = outputs
                    # Apply masks before argmax — prevents invalid actions.
                    m_l = m_l.masked_fill(~m_mask, -1e8)
                    c_l = c_l.masked_fill(~c_mask, -1e8)
                    t_l = t_l.masked_fill(~t_mask, -1e8)
                    m = m_l.argmax(1).item()
                    c = c_l.argmax(1).item()
                    t = t_l.argmax(1).item()

                obs, reward, done, truncated, _ = env.step(np.array([m, c, t]))
                ep_reward += reward
                steps += 1
                if truncated:
                    break

            killed = sum(1 for t in raw_env.targets if not t.alive)
            rewards.append(ep_reward)
            wins.append(float(killed == num_targets))
            kills.append(killed)
            lengths.append(steps)
            hp_remaining.append(raw_env.agent.hp_fraction())

        env.close()

        all_results[stage] = {
            "win_rate": np.mean(wins),
            "mean_reward": np.mean(rewards),
            "mean_kills": np.mean(kills),
            "mean_length": np.mean(lengths),
            "mean_hp_remaining": np.mean(hp_remaining),
            "std_reward": np.std(rewards),
        }

    # Aggregate across stages.
    agg = {}
    for key in ["win_rate", "mean_reward", "mean_kills", "mean_length", "mean_hp_remaining"]:
        agg[key] = np.mean([all_results[s][key] for s in stages])
    agg["per_stage"] = all_results

    return agg


# ─────────────────────────────────────────────────────────────────
#  Inference Benchmark
# ─────────────────────────────────────────────────────────────────

def benchmark_inference(model: nn.Module, frame_stack: int = DEFAULT_FRAME_STACK,
                        warmup: int = 50, trials: int = 500) -> float:
    """Returns mean inference time in milliseconds."""
    model.eval().cpu()
    input_size = OBS_SIZE * frame_stack
    dummy = torch.randn(1, input_size)

    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)

    start = time.perf_counter()
    for _ in range(trials):
        with torch.no_grad():
            model(dummy)
    elapsed = time.perf_counter() - start

    return (elapsed / trials) * 1000  # ms


# ─────────────────────────────────────────────────────────────────
#  Per-Tier Results
# ─────────────────────────────────────────────────────────────────

@dataclass
class TierResult:
    tier: str = ""
    params: int = 0
    onnx_size_kb: float = 0.0
    inference_ms: float = 0.0
    train_acc_m: float = 0.0
    train_acc_c: float = 0.0
    train_acc_t: float = 0.0
    win_rate: float = 0.0
    mean_reward: float = 0.0
    mean_kills: float = 0.0
    mean_length: float = 0.0
    mean_hp_remaining: float = 0.0
    per_stage: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full distillation pipeline")
    parser.add_argument("--teacher", type=str, required=True,
                        help="Path to PPO checkpoint")
    parser.add_argument("--output_dir", type=str, default="models/v1")
    parser.add_argument("--frame_stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--num_episodes", type=int, default=5000,
                        help="Teacher rollout episodes for dataset")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Distillation epochs per tier")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="Combat eval episodes per stage per tier")
    parser.add_argument("--eval_stages", type=int, nargs="+", default=[5, 7],
                        help="Stages to evaluate on")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip distillation, only evaluate existing models")
    parser.add_argument("--archetype", type=str, default="ranged")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("DISTILLATION PIPELINE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Teacher: {args.teacher}")
    print(f"Output: {args.output_dir}")
    print(f"Frame stack: {args.frame_stack}")

    # ── Load obs normalizer if saved in checkpoint ───────────────
    obs_normalizer = None
    ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    frame_stack = ckpt.get("frame_stack", args.frame_stack)
    if "obs_normalizer" in ckpt:
        class RunningNormalizer:
            def __init__(self, shape, clip=5.0, epsilon=1e-8):
                self.mean = np.zeros(shape, dtype=np.float64)
                self.var = np.ones(shape, dtype=np.float64)
                self.count = 1e-4
                self.clip = clip
                self.epsilon = epsilon

            def normalize(self, obs):
                return np.clip(
                    (obs - self.mean.astype(np.float32)) /
                    np.sqrt(self.var.astype(np.float32) + self.epsilon),
                    -self.clip, self.clip)

            def load_state_dict(self, state):
                self.mean = state["mean"].copy()
                self.var = state["var"].copy()
                self.count = state["count"]
                
        input_size = stacked_obs_size(frame_stack)
        obs_normalizer = RunningNormalizer(input_size)
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"Loaded observation normalizer from checkpoint")

    # ── Load teacher ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("LOADING TEACHER")
    print(f"{'─'*70}")
    teacher = load_teacher_from_checkpoint(args.teacher, device)

    # Sync frame_stack with what the teacher was actually built with.
    teacher_fs = teacher.frame_stack
    if teacher_fs != frame_stack:
        print(f"  WARNING: checkpoint header said frame_stack={frame_stack}, "
              f"teacher model built with {teacher_fs} — using {teacher_fs}")
        frame_stack = teacher_fs

    # ── Distillation order ───────────────────────────────────────
    # Large = teacher (direct export)
    # Medium = distilled from Large
    # Small = distilled from Medium (cascading)
    # Micro = distilled from Small (cascading)
    # XL = distilled from Large (scale up, not down)

    distill_chain = [
        # (tier, teacher_tier, alpha, temperature, epochs)
        ("large",  None,     0.0, 0.0, 0),       # Direct export.
        ("medium", "large",  0.7, 3.0, args.epochs),
        ("small",  "medium", 0.7, 3.0, args.epochs),
        ("micro",  "small",  0.5, 4.0, args.epochs),  # Higher T, lower α for tiny.
        ("xl",     "large",  0.7, 3.0, args.epochs),
    ]

    results: Dict[str, TierResult] = {}
    models: Dict[str, CombatPolicy] = {"large": teacher}

    if not args.eval_only:

        # ── Generate teacher dataset ─────────────────────────────
        print(f"\n{'─'*70}")
        print("GENERATING TEACHER ROLLOUT DATASET")
        print(f"{'─'*70}")
        dataset = generate_teacher_dataset(
            teacher, num_episodes=args.num_episodes,
            stages=[3, 4, 5, 6, 7], archetype=args.archetype,
            frame_stack=frame_stack, device=device,
            obs_normalizer=obs_normalizer)

        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        split_generator = torch.Generator().manual_seed(42)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=split_generator)
        # Save indices for reuse in cascading distillation.
        train_indices = train_set.indices
        val_indices = val_set.indices
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                 shuffle=False, num_workers=0)
        print(f"Train: {train_size:,}, Val: {val_size:,}")

        # ── Distill each tier ────────────────────────────────────
        for tier, teacher_tier, alpha, temperature, epochs in distill_chain:
            print(f"\n{'─'*70}")
            print(f"TIER: {tier.upper()}")
            print(f"{'─'*70}")

            if teacher_tier is None:
                # Direct export (teacher = Large).
                print(f"  Direct export (no distillation needed)")
                models[tier] = teacher
            else:
                teacher_model = models[teacher_tier]
                student = make_policy(tier, frame_stack=frame_stack)

                # For cascading distillation, we need to regenerate logits
                # from the intermediate teacher (not the original Large).
                # But the obs+actions in the dataset are still valid — we
                # just need updated teacher logits.
                if teacher_tier != "large":
                    print(f"  Regenerating soft labels from {teacher_tier} teacher...")
                    teacher_model.eval().to(device)
                    new_logits_m, new_logits_c, new_logits_t = [], [], []
                    with torch.no_grad():
                        # Use train_set's obs to get new teacher logits.
                        obs_all = dataset.tensors[0]  # Full dataset obs.
                        for start in range(0, len(obs_all), args.batch_size):
                            batch_obs = obs_all[start:start+args.batch_size].to(device)
                            t_m, t_c, t_t = teacher_model(batch_obs)
                            new_logits_m.append(t_m.cpu())
                            new_logits_c.append(t_c.cpu())
                            new_logits_t.append(t_t.cpu())

                    cascade_dataset = TensorDataset(
                        dataset.tensors[0],                    # obs
                        torch.cat(new_logits_m),               # teacher m logits
                        torch.cat(new_logits_c),               # teacher c logits
                        torch.cat(new_logits_t),               # teacher t logits
                        dataset.tensors[4],                    # hard m actions
                        dataset.tensors[5],                    # hard c actions
                        dataset.tensors[6],                    # hard t actions
                    )
                    c_train = torch.utils.data.Subset(cascade_dataset, train_indices)
                    c_val = torch.utils.data.Subset(cascade_dataset, val_indices)
                    c_train_loader = DataLoader(c_train, batch_size=args.batch_size,
                                                 shuffle=True)
                    c_val_loader = DataLoader(c_val, batch_size=args.batch_size,
                                               shuffle=False)
                else:
                    c_train_loader = train_loader
                    c_val_loader = val_loader

                student = distill_from_teacher_data(
                    student, c_train_loader, c_val_loader,
                    alpha=alpha, temperature=temperature, epochs=epochs,
                    device=device, tier_name=tier)

                models[tier] = student

            # Export ONNX (bake normalizer into graph if available).
            onnx_path = export_onnx(models[tier], tier, args.output_dir,
                                     frame_stack=frame_stack,
                                     obs_normalizer=obs_normalizer)
            verify_export(models[tier], onnx_path, frame_stack=frame_stack,
                          obs_normalizer=obs_normalizer)

    else:
        # Eval-only: load existing models.
        print("\n  Eval-only mode — loading existing ONNX is not supported.")
        print("  Will evaluate the teacher and any tiers that can be created.")
        models["large"] = teacher

    # ── Evaluate all tiers ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMBAT EVALUATION")
    print(f"{'='*70}")
    print(f"Stages: {args.eval_stages}, Episodes/stage: {args.eval_episodes}")

    for tier in ["micro", "small", "medium", "large", "xl"]:
        if tier not in models:
            continue

        model = models[tier]
        params = sum(p.numel() for p in model.parameters())

        print(f"\n── {tier.upper()} ({params:,} params) ──")

        # Combat eval.
        eval_results = evaluate_combat(
            model, stages=args.eval_stages, archetype=args.archetype,
            episodes_per_stage=args.eval_episodes,
            frame_stack=frame_stack, device=device,
            obs_normalizer=obs_normalizer,
            is_policy_only=True)

        for stage, sr in eval_results["per_stage"].items():
            print(f"  Stage {stage}: win={sr['win_rate']:.0%}, "
                  f"kills={sr['mean_kills']:.1f}, "
                  f"reward={sr['mean_reward']:.1f}, "
                  f"hp_left={sr['mean_hp_remaining']:.0%}, "
                  f"len={sr['mean_length']:.0f}")

        # Inference benchmark.
        infer_ms = benchmark_inference(model, frame_stack=frame_stack)
        print(f"  Inference: {infer_ms:.3f} ms")

        # ONNX size.
        onnx_path = os.path.join(args.output_dir, f"Combat_{tier.capitalize()}.onnx")
        onnx_kb = os.path.getsize(onnx_path) / 1024 if os.path.exists(onnx_path) else 0

        # Store results.
        results[tier] = TierResult(
            tier=tier, params=params,
            onnx_size_kb=onnx_kb, inference_ms=infer_ms,
            win_rate=eval_results["win_rate"],
            mean_reward=eval_results["mean_reward"],
            mean_kills=eval_results["mean_kills"],
            mean_length=eval_results["mean_length"],
            mean_hp_remaining=eval_results["mean_hp_remaining"],
            per_stage=eval_results["per_stage"],
        )

    # ── Summary Report ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY REPORT")
    print(f"{'='*70}")

    # Header.
    print(f"\n{'Tier':>8s} | {'Params':>9s} | {'ONNX':>7s} | {'Infer':>7s} | "
          f"{'Win%':>6s} | {'Kills':>5s} | {'Reward':>7s} | {'HP Left':>7s} | {'Length':>6s}")
    print("-" * 85)

    tier_order = ["micro", "small", "medium", "large", "xl"]
    for tier in tier_order:
        if tier not in results:
            continue
        r = results[tier]
        print(f"{r.tier:>8s} | {r.params:>9,} | {r.onnx_size_kb:>5.1f}KB | "
              f"{r.inference_ms:>5.3f}ms | {r.win_rate:>5.0%} | "
              f"{r.mean_kills:>5.1f} | {r.mean_reward:>+7.1f} | "
              f"{r.mean_hp_remaining:>6.0%} | {r.mean_length:>6.0f}")

    # Delta from Large.
    if "large" in results:
        print(f"\nDelta vs Large (teacher):")
        large = results["large"]
        for tier in tier_order:
            if tier not in results or tier == "large":
                continue
            r = results[tier]
            dw = r.win_rate - large.win_rate
            dk = r.mean_kills - large.mean_kills
            di = r.inference_ms / max(large.inference_ms, 0.001)
            print(f"  {tier:>8s}: win {dw:+.0%}, kills {dk:+.1f}, "
                  f"speed {di:.2f}x, size {r.onnx_size_kb/max(large.onnx_size_kb,1):.2f}x")

    # Per-stage breakdown.
    print(f"\nPer-Stage Win Rates:")
    header = f"{'Tier':>8s}"
    for stage in args.eval_stages:
        header += f" | S{stage:>2d}"
    print(header)
    print("-" * (10 + 7 * len(args.eval_stages)))
    for tier in tier_order:
        if tier not in results:
            continue
        line = f"{tier:>8s}"
        for stage in args.eval_stages:
            if stage in results[tier].per_stage:
                wr = results[tier].per_stage[stage]["win_rate"]
                line += f" | {wr:>4.0%}"
            else:
                line += f" |   --"
        print(line)

    # Save CSV.
    try:
        import pandas as pd
        rows = []
        for tier in tier_order:
            if tier not in results:
                continue
            r = results[tier]
            row = {
                "tier": r.tier, "params": r.params,
                "onnx_kb": r.onnx_size_kb, "inference_ms": r.inference_ms,
                "win_rate": r.win_rate, "mean_reward": r.mean_reward,
                "mean_kills": r.mean_kills, "mean_length": r.mean_length,
                "mean_hp_remaining": r.mean_hp_remaining,
            }
            for stage in args.eval_stages:
                if stage in r.per_stage:
                    row[f"s{stage}_win"] = r.per_stage[stage]["win_rate"]
                    row[f"s{stage}_kills"] = r.per_stage[stage]["mean_kills"]
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, "distillation_report.csv")
        df.to_csv(csv_path, index=False, float_format="%.4f")
        print(f"\nSaved: {csv_path}")
    except ImportError:
        print("\n(Install pandas to save CSV report)")

    print(f"\nONNX models in: {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()