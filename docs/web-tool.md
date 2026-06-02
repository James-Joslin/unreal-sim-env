# Web Testing Tool

Browser-based testing and debugging environment for the neural combat AI. Runs a full combat simulation with ONNX model inference, letting you play against the AI agent in real time while monitoring every aspect of its decision-making.

## Setup

```bash
npm install react react-dom d3 onnxruntime-web
npm install -D @types/d3 typescript
npm run dev:local
```

Copy `App.tsx`, `main.tsx`, and `components/` into `src/`. The simulation starts immediately with a scripted fallback AI (no model needed for basic testing).

## Controls

| Key | Effect |
|---|---|
| WASD | Move player |
| Click (canvas) | Shoot toward cursor |
| Space | Shoot at AI agent |
| R | Reload |
| Enter | Restart (when game over) |

## Loading an ONNX Model

Upload via the header bar. The tool auto-detects the tier from input dimension and configures frame stacking accordingly.

| Tier | Decision Rate | Frame Stack | Input Dim |
|---|---|---|---|
| Micro | 2.5 Hz | 3 | 645 |
| Small | 3.3 Hz | 3 | 645 |
| Medium | 5 Hz | 3 | 645 |
| Large | 6.6 Hz | 3 | 645 |
| XL | 10 Hz | 3 | 645 |

## Components

### Reward D3 Chart

Real-time line chart showing cumulative (blue) or instant (green) reward. Hover for per-step breakdown. Look for: cumulative climbing without kills (farming), instant spikes without HP changes (phantom rewards).

### Action Probability Heatmap

Softmax probabilities for all three action heads. Shows chosen action, masked actions, and total entropy. Look for: target head cycling without firing (exploit), movement stuck on hold (not engaging), high entropy (hasn't learned).

### Reward Budget Bar

Live gauge comparing actual cumulative reward against win/death/timeout budgets. Shows trajectory prediction and farming warnings (reward exceeds win budget without kills).

### Batch Episode Runner (Tools tab)

Runs 10-100 episodes headlessly, shows win rate, avg reward, avg kills, reward distribution histogram, and misalignment warnings (losses more rewarding than wins).

### Observation Inspector (Tools tab)

Expandable accordion showing all 215 observation features grouped by category. Change highlighting (green = increased, red = decreased) with per-group change count badges. Look for: all-zero groups (missing features), jumping values without agent action (observation bugs).

## Keeping In Sync

| What | Web Tool | Python | C++ |
|---|---|---|---|
| Reward weights | `RW` in App.tsx | `RewardWeights` in reward.py | N/A |
| Observation layout | `buildObservation()` | `_build_observation()` | `GatherObservation()` |
| Action space | label constants | action enums | NeuralCombatTypes.h |
| OBS_SIZE | `OBS_SIZE` constant | `OBS_SIZE` in combat_sim.py | `ObservationSize` in NeuralCombatTypes.h |

When any value changes in Python or C++, update the web tool to match.
