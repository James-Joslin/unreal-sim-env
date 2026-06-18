# Web Testing Tool - OUTDATED NEEDS UPDATING

The web testing tool is a browser-based debugging environment for ONNX combat policies. It can run a combat simulation, load a policy model, inspect observations, visualise rewards and monitor action probabilities.

## Setup

```bash
npm install react react-dom d3 onnxruntime-web
npm install -D @types/d3 typescript
npm run dev:local
```

The tool can run with a scripted fallback AI, so a model is not required for basic UI testing.

## Controls

| Key/Input | Effect |
|---|---|
| WASD | Move player |
| Click canvas | Shoot toward cursor |
| Space | Shoot at AI agent |
| R | Reload |
| Enter | Restart after game over |

## ONNX Model Loading

Upload an ONNX model through the header bar. Current policy exports use:

```text
observation input: [batch, 747]
hidden_in input:   [1, batch, gru_hidden]
outputs: movement_logits, combat_logits, target_logits, hidden_out
```

The web tool should stay aligned with the Python/C++ model constants:

```text
OBS_SIZE = 249
FRAME_STACK = 3
INPUT_DIM = 747
MOVEMENT_ACTIONS = 9
COMBAT_ACTIONS = 8
TARGET_ACTIONS = 5
```

## Components

### Reward Chart

Shows instant or cumulative reward. Useful for spotting reward farming or reward spikes that do not correspond to meaningful combat progress.

### Action Probability Heatmap

Displays movement, combat and target-head probabilities. Useful for checking entropy, invalid-action masking and head collapse.

### Reward Budget Bar

Compares actual cumulative reward against expected win/loss/timeout budgets. Useful for detecting episodes where shaping rewards outscore objectives.

### Batch Episode Runner

Runs multiple headless episodes and reports aggregate win rate, reward, kills and warning flags.

### Observation Inspector

Displays the 249 observation features grouped by category, with change highlighting. Useful for finding all-zero groups, wrong offsets or unexpected jumps.

## Keeping In Sync

When these change in Python or C++, update the web tool as well:

- observation layout and `OBS_SIZE`
- frame stack/input dimension
- action labels and action counts
- reward weights used for visualisation
- weapon presets and projectile/arc rules
- ONNX input/output names, especially `hidden_in` and `hidden_out`
