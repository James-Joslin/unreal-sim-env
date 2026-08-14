"""
methods/ — Registry of available RL training methods.

TO ADD A NEW METHOD:
    1. Create a new file or subpackage under methods/ (e.g. methods/sac.py)
    2. Subclass BaseTrainer and implement build_model(), train(), extract_policy()
    3. Import and register it here:
           from .sac import SACTrainer
           METHOD_REGISTRY["sac"] = SACTrainer

The registry is a simple dict. main.py looks up --method by name.
"""

from typing import Dict, Type
from training.base_trainer import BaseTrainer

# ─────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────

METHOD_REGISTRY: Dict[str, Type[BaseTrainer]] = {}


def register_method(name: str, cls: Type[BaseTrainer]):
    """Register a training method by name."""
    METHOD_REGISTRY[name] = cls


def get_trainer_class(name: str) -> Type[BaseTrainer]:
    """Look up a trainer class by method name."""
    if name not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise ValueError(
            f"Unknown method '{name}'. Available: {available}")
    return METHOD_REGISTRY[name]


# ─────────────────────────────────────────────────────────────────
#  Auto-register built-in methods
# ─────────────────────────────────────────────────────────────────

from .ppo import PPOTrainer
register_method("ppo", PPOTrainer)

# SAC remains in-tree for development, but is not advertised by the production
# CLI until it satisfies the BaseTrainer/evaluation/export contract.
# from .sac import SACTrainer
# register_method("sac", SACTrainer)

# Future methods
# from .impala import IMPALATrainer
# register_method("impala", IMPALATrainer)
