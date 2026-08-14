# Quickstart

## Dependencies

Core development/training environment (Python 3.13):

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Optional visualisation/export:

```bash
pip install pygame imageio[ffmpeg] Pillow onnx onnxruntime
```

Optional browser test tool:

```bash
npm install react react-dom d3 onnxruntime-web
npm install -D @types/d3 typescript
```

## Primary Training CLI

The current entry point is `training.main`.

```bash
# List registered training methods
python -m training.main --list_methods

# Train one PPO stage
python -m training.main --method ppo --stage 3 --archetype ranged --tier large

# Full curriculum
python -m training.main --method ppo --curriculum

# Curriculum followed by distillation/export
python -m training.main --method ppo --curriculum --distill

# Continue/warm-start from a checkpoint
python -m training.main --method ppo --stage 5 \
    --bc_checkpoint checkpoints/ppo_stage4_best.pt

# Distill only from an existing teacher checkpoint
python -m training.main --distill_only --teacher checkpoints/ppo_stage7_best.pt
```

## Distillation CLI

```bash
# Standard distillation
python -m training.distillation \
    --teacher checkpoints/ppo_stage7_best.pt \
    --output_dir models/v1 \
    --mode standard

# Amplified best-of-N distillation
python -m training.distillation \
    --teacher checkpoints/ppo_stage7_best.pt \
    --output_dir models/v1 \
    --mode amplified \
    --rollouts 16 \
    --top_k 0.25 \
    --iterations 1
```

Expected outputs:

```text
models/v1/Combat_Micro.onnx
models/v1/Combat_Small.onnx
models/v1/Combat_Medium.onnx
models/v1/Combat_Large.onnx
```

## Visual Debugging

```bash
# Random/actions-only sim check
python simulation/combat_sim.py --stage 3 --render human
python simulation/combat_sim.py --stage 7 --weapon sniper --render human
python simulation/combat_sim.py --stage 5 --render video --steps 800

# View a checkpoint or ONNX model
python simulation/view_sim.py --stage 3 --model checkpoints/ppo_best.pt --render video
python simulation/view_sim.py --stage 3 --model models/v1/Combat_Large.onnx --render video
python simulation/view_sim.py --stage 3 --arena_size 4000 --render video
```

Useful `view_sim.py` flags:

```text
--stage            Curriculum stage, 1-7
--archetype        ranged, melee, tank
--model            .pt checkpoint or .onnx model path
--render           human, video, none
--arena_size       Override arena size
--weapon           scout, heavy, sniper, melee_bot, tank
--fps              Video FPS
--episodes         Number of episodes
--deterministic    Use deterministic action choice where supported
--output           Output video filename
```

## Policy/ONNX Checks

```bash
# Export/test a blank tier model
python simulation/combat_policy.py --tier large --frame_stack 3 --output_dir models/test

# Extract/export from checkpoint
python simulation/combat_policy.py \
    --checkpoint checkpoints/ppo_stage7_best.pt \
    --output_dir models/test
```

## TensorBoard

```bash
tensorboard --logdir runs
```

## Notes

Older commands such as `03_ppo_train.py` and `02_distill_and_export.py` have been superseded by the modular `training.main` and `training.distillation` entry points.
