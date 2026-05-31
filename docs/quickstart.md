# Quickstart

## Dependencies

```
numpy
gymnasium
torch
tensorboard
```

Optional (for visualisation):
```
pygame          # --render human
imageio[ffmpeg] # --render video
Pillow          # fallback frame export
onnx            # ONNX model consolidation
onnxruntime     # ONNX verification
```

## Scripts

### Visual Debugger

Run the sim with random actions to verify the environment:

```bash
python combat_sim.py --stage 3 --render human
python combat_sim.py --stage 7 --weapon sniper --render human
python combat_sim.py --stage 5 --render video --steps 800
```

| Flag | Options | Default |
|---|---|---|
| `--stage` | 1–7 | 3 |
| `--archetype` | ranged, melee, tank, healer | ranged |
| `--weapon` | scout, heavy, sniper, melee_bot, tank | per stage |
| `--arena_size` | any float (UU) | per stage |
| `--steps` | max steps | 500 |
| `--render` | human, video | none |

### PPO Training

```bash
python 03_ppo_train.py --bc_checkpoint checkpoints/bc_model.pt --stage 3
python 03_ppo_train.py --stage 1
python 03_ppo_train.py --curriculum
python 03_ppo_train.py --stage 3 --num_envs 4
```

Outputs: `checkpoints/ppo_stage{N}.pt`, `checkpoints/ppo_final.pt`, `runs/ppo_{timestamp}/` (TensorBoard).

### Distillation & ONNX Export

```bash
python 02_distill_and_export.py \
    --teacher checkpoints/ppo_stage7_best.pt \
    --output_dir models/v1

python 02_distill_and_export.py \
    --teacher checkpoints/ppo_stage7_best.pt \
    --eval_only
```

Outputs: `Combat_{Micro,Small,Medium,Large,Xl}.onnx`, `distillation_report.csv`.

### Observation Validator

```bash
python obs_vector_validator.py --check-structure
python obs_vector_validator.py --check-live --episodes 50
python obs_vector_validator.py --dump-py --episodes 20 --output validation/py_obs.csv
python obs_vector_validator.py --diff-csv \
    --cpp_csv Saved/NeuralData/Default/Enemy_001.csv \
    --py_csv validation/py_obs.csv
```

### Policy Network Test

```bash
python combat_policy.py --tier large --frame_stack 3
python combat_policy.py --checkpoint checkpoints/ppo_stage7_best.pt --output_dir models/test
```
