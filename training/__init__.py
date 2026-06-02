"""
training/ — Modular RL training package for combat AI.

STRUCTURE
    project/
    ├── simulation/              ← Python sim, policy arch, reward, frame stacking
    │   ├── combat_sim.py
    │   ├── combat_extensions.py
    │   ├── combat_policy.py
    │   ├── frame_stack.py
    │   ├── reward.py
    │   └── view_sim.py
    └── training/                ← this package
        ├── __init__.py          ← you are here
        ├── main.py              ← unified CLI entry point
        ├── normalizers.py       ← RunningNormalizer, ReturnNormalizer
        ├── evaluation.py        ← shared evaluate() across all methods
        ├── base_trainer.py      ← BaseTrainer ABC (shared infra every method inherits)
        └── methods/
            ├── __init__.py      ← METHOD_REGISTRY (add new methods here by referring to registered trainer in submethod)
            └── ppo/
                ├── __init__.py  ← Register the trainer at the submodule level
                ├── config.py    ← PPOConfig dataclass
                ├── actor_critic.py  ← ActorCritic(nn.Module)
                ├── buffer.py    ← VecRolloutBuffer
                └── trainer.py   ← PPOTrainer(BaseTrainer)

ADDING A NEW METHOD (e.g. SAC)
    1. Create training/methods/sac.py (or sac/ subpackage)
    2. Subclass BaseTrainer → implement build_model(), train_step(), extract_policy()
    3. Register in training/methods/__init__.py:
           from .sac import SACTrainer
           METHOD_REGISTRY["sac"] = SACTrainer
    4. Done. Use: python -m training.main --method sac --stage 3

CONTRACT
    Every method must:
    - Accept the same env interface (CombatEnv / VecFrameStackEnv)
    - Produce the same CombatPolicy for ONNX export (delta encode → group encode → backbone → 3 heads)
    - Save checkpoints with 'full_state_dict' and 'policy_state_dict' keys
    - Support curriculum stage transitions
"""

# ─────────────────────────────────────────────────────────────────
#  Path bootstrap — simulation/ is a sibling directory containing
#  combat_sim, combat_policy, frame_stack, etc. as flat modules.
#  Adding it to sys.path lets every file in training/ import them
#  with their original names (e.g. `from combat_sim import ...`).
#  This MUST run before any other imports in this package.
# ─────────────────────────────────────────────────────────────────
import os as _os
import sys as _sys

_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sim_dir = _os.path.join(_project_root, "simulation")

if _os.path.isdir(_sim_dir) and _sim_dir not in _sys.path:
    _sys.path.insert(0, _sim_dir)

# ─────────────────────────────────────────────────────────────────

from .normalizers import RunningNormalizer, ReturnNormalizer
from .evaluation import evaluate
from .base_trainer import BaseTrainer
from .methods import METHOD_REGISTRY, get_trainer_class