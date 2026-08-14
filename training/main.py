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

from combat_policy import (
    ACTIVE_TIERS, BEHAVIOR_TIER_DEFINITIONS, TRAINABLE_ARCHETYPES,
)

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
        "--archetype", type=str, default="ranged",
        choices=TRAINABLE_ARCHETYPES,
        help="Trainable combat archetype")
    parser.add_argument(
        "--tier", type=str, default="large",
        choices=ACTIVE_TIERS,
        help="Behavior/model tier")
    parser.add_argument(
        "--bc_checkpoint", type=str, default=None,
        help="Path to checkpoint for warm-start")
    parser.add_argument(
        "--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override the stage's configured training budget "
             "(ignored with --curriculum)")
    parser.add_argument(
        "--frame_stack", type=int, default=3)
    parser.add_argument(
        "--num_envs", type=int, default=6,
        help="Number of parallel environments")

    # ── Curriculum mode ──────────────────────────────────────────
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Run the selected tier's supported curriculum stages")

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
            final_stage = BEHAVIOR_TIER_DEFINITIONS[
                args.tier]["curriculum_stages"][-1]
            teacher_path = os.path.join(
                args.output_dir,
                f"{args.method}_stage{final_stage}_best.pt")
            if not os.path.exists(teacher_path):
                teacher_path = os.path.join(
                    args.output_dir,
                    f"{args.method}_stage{final_stage}_final.pt")
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
    """Delegate to the distillation module."""
    from training.distillation import run_distillation
    run_distillation(
        teacher_path=teacher_path,
        output_dir=output_dir,
        frame_stack=frame_stack,
        archetype=archetype,
        num_episodes=num_episodes,
        epochs=epochs,
    )


if __name__ == "__main__":
    main()
