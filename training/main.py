"""
main.py — Unified entry point for training and distillation.

USAGE
    # Single stage with PPO:
    python -m training.main --method ppo --stage 3 --archetype ranged

    # Full curriculum:
    python -m training.main --method ppo --curriculum

    # Train then distill + export ONNX:
    python -m training.main --method ppo --curriculum --distill

    # Warm-start from a checkpoint:
    python -m training.main --method ppo --stage 5 \
        --bc_checkpoint checkpoints/ppo_stage4_best.pt

    # Distill only (skip training, use existing teacher checkpoint):
    python -m training.main --distill_only \
        --teacher checkpoints/ppo_stage7_best.pt

    # List available methods:
    python -m training.main --list_methods

ADDING A NEW METHOD
    1. Create training/methods/your_method.py (or subpackage)
    2. Subclass BaseTrainer
    3. Register in training/methods/__init__.py
    4. Run: python -m training.main --method your_method --stage 3
"""

import argparse
import os
import sys

from combat_policy import TIER_CONFIGS

from training.methods import METHOD_REGISTRY, get_trainer_class


def main():
    parser = argparse.ArgumentParser(
        description="Combat AI — Modular RL Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --method ppo --stage 3
  %(prog)s --method ppo --curriculum --distill
  %(prog)s --distill_only --teacher checkpoints/ppo_stage7_best.pt
  %(prog)s --list_methods
        """)

    # ── Method selection ─────────────────────────────────────────
    parser.add_argument(
        "--method", type=str, default="ppo",
        choices=list(METHOD_REGISTRY.keys()),
        help=f"RL method to use ({', '.join(METHOD_REGISTRY.keys())})")
    parser.add_argument(
        "--list_methods", action="store_true",
        help="List available training methods and exit")

    # ── Training config ──────────────────────────────────────────
    parser.add_argument(
        "--stage", type=int, default=3,
        help="Curriculum stage (1-7)")
    parser.add_argument(
        "--archetype", type=str, default="ranged")
    parser.add_argument(
        "--tier", type=str, default="large",
        choices=list(TIER_CONFIGS.keys()),
        help="Model tier (architecture size)")
    parser.add_argument(
        "--bc_checkpoint", type=str, default=None,
        help="Path to checkpoint for warm-start")
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--timesteps", type=int, default=6_000_000,
        help="Total training timesteps (ignored with --curriculum)")
    parser.add_argument(
        "--frame_stack", type=int, default=3)
    parser.add_argument(
        "--num_envs", type=int, default=16,
        help="Number of parallel environments")

    # ── Curriculum mode ──────────────────────────────────────────
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Run all 7 curriculum stages sequentially")

    # ── Distillation ─────────────────────────────────────────────
    parser.add_argument(
        "--distill", action="store_true",
        help="Distill + export ONNX after training")
    parser.add_argument(
        "--distill_only", action="store_true",
        help="Skip training, only distill from --teacher checkpoint")
    parser.add_argument(
        "--teacher", type=str, default=None,
        help="Teacher checkpoint for --distill_only mode")
    parser.add_argument(
        "--distill_episodes", type=int, default=5000,
        help="Teacher rollout episodes for distillation dataset")
    parser.add_argument(
        "--distill_epochs", type=int, default=200,
        help="Distillation training epochs per tier")

    args = parser.parse_args()

    # ── List methods ─────────────────────────────────────────────
    if args.list_methods:
        print("Available training methods:")
        for name, cls in sorted(METHOD_REGISTRY.items()):
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:12s}  {doc}")
        return

    # ── Distill-only mode ────────────────────────────────────────
    if args.distill_only:
        if not args.teacher:
            parser.error(
                "--distill_only requires --teacher checkpoint path")
        _run_distillation(
            teacher_path=args.teacher,
            output_dir=args.output_dir,
            frame_stack=args.frame_stack,
            archetype=args.archetype,
            num_episodes=args.distill_episodes,
            epochs=args.distill_epochs,
        )
        return

    # ── Training ─────────────────────────────────────────────────
    TrainerClass = get_trainer_class(args.method)
    trainer = TrainerClass(
        stage=args.stage,
        archetype=args.archetype,
        tier=args.tier,
        frame_stack=args.frame_stack,
        num_envs=args.num_envs,
        bc_checkpoint=args.bc_checkpoint,
        output_dir=args.output_dir,
        total_timesteps=args.timesteps,
    )

    if args.curriculum:
        trainer.run_curriculum()
    else:
        trainer.train()

    # ── Post-training distillation ───────────────────────────────
    if args.distill:
        # Find the best checkpoint from training.
        if args.curriculum:
            teacher_path = os.path.join(
                args.output_dir,
                f"{args.method}_stage7_best.pt")
            if not os.path.exists(teacher_path):
                teacher_path = os.path.join(
                    args.output_dir,
                    f"{args.method}_stage7_final.pt")
        else:
            teacher_path = os.path.join(
                args.output_dir,
                f"{args.method}_stage{args.stage}_best.pt")
            if not os.path.exists(teacher_path):
                teacher_path = os.path.join(
                    args.output_dir,
                    f"{args.method}_stage{args.stage}_final.pt")

        if os.path.exists(teacher_path):
            print(f"\n{'='*60}")
            print(f"POST-TRAINING DISTILLATION")
            print(f"{'='*60}")
            print(f"Teacher: {teacher_path}")
            _run_distillation(
                teacher_path=teacher_path,
                output_dir=os.path.join(args.output_dir, "models"),
                frame_stack=args.frame_stack,
                archetype=args.archetype,
                num_episodes=args.distill_episodes,
                epochs=args.distill_epochs,
            )
        else:
            print(f"Warning: no checkpoint found at {teacher_path} "
                  f"— skipping distillation")

    trainer.cleanup()


def _run_distillation(teacher_path, output_dir, frame_stack,
                      archetype, num_episodes, epochs):
    """Run the distillation pipeline.

    Delegates to 02_distill_and_export.py's infrastructure.
    This keeps distillation code in one place while making it
    callable from the unified entry point.
    """
    # Import lazily to avoid circular deps and keep startup fast.
    from combat_policy import load_teacher_from_checkpoint, export_onnx, verify_export, make_policy
    from frame_stack import stacked_obs_size
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Load normalizer if present.
    obs_normalizer = None
    ckpt = torch.load(teacher_path, map_location="cpu", weights_only=False)
    fs = ckpt.get("frame_stack", frame_stack)

    if "obs_normalizer" in ckpt:
        from training.normalizers import RunningNormalizer
        input_size = stacked_obs_size(fs)
        obs_normalizer = RunningNormalizer(input_size)
        obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        print(f"Loaded observation normalizer from checkpoint")

    # Load teacher.
    teacher = load_teacher_from_checkpoint(teacher_path, device)

    # Delegate to 02_distill_and_export if available, otherwise
    # do a minimal export of just the teacher tier.
    try:
        from importlib import import_module
        distill_mod = import_module("02_distill_and_export")

        # Generate dataset and run full pipeline.
        dataset = distill_mod.generate_teacher_dataset(
            teacher, num_episodes=num_episodes,
            stages=[3, 4, 5, 6, 7], archetype=archetype,
            frame_stack=fs, device=device,
            obs_normalizer=obs_normalizer,
        )

        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        split_gen = torch.Generator().manual_seed(42)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=split_gen)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=256, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=256, shuffle=False)

        # Distill chain.
        distill_chain = [
            ("large",  None,     0.0, 0.0, 0),
            ("medium", "large",  0.7, 3.0, epochs),
            ("small",  "medium", 0.7, 3.0, epochs),
            ("micro",  "small",  0.5, 4.0, epochs),
            ("xl",     "large",  0.7, 3.0, epochs),
        ]

        models = {"large": teacher}

        for tier, teacher_tier, alpha, temp, ep in distill_chain:
            print(f"\n── {tier.upper()} ──")
            if teacher_tier is None:
                models[tier] = teacher
            else:
                student = make_policy(tier, frame_stack=fs)
                student = distill_mod.distill_from_teacher_data(
                    student, train_loader, val_loader,
                    alpha=alpha, temperature=temp, epochs=ep,
                    device=device, tier_name=tier)
                models[tier] = student

            onnx_path = export_onnx(
                models[tier], tier, output_dir,
                frame_stack=fs, obs_normalizer=obs_normalizer)
            verify_export(
                models[tier], onnx_path,
                frame_stack=fs, obs_normalizer=obs_normalizer)

        print(f"\nAll ONNX models saved to: {output_dir}/")

    except (ImportError, ModuleNotFoundError):
        # Fallback: just export the teacher directly.
        print("02_distill_and_export not found — exporting teacher only")
        teacher_tier = ckpt.get("tier", "large")
        onnx_path = export_onnx(
            teacher, teacher_tier, output_dir,
            frame_stack=fs, obs_normalizer=obs_normalizer)
        verify_export(
            teacher, onnx_path,
            frame_stack=fs, obs_normalizer=obs_normalizer)
        print(f"Exported: {onnx_path}")


if __name__ == "__main__":
    main()
