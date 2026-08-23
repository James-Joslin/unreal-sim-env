"""Categorical behavior-profile contract for conditioned teacher training.

The four-value condition is training-only. Deployment students bake one
profile into their weights and retain the existing observation/hidden ONNX ABI.
"""

from enum import IntEnum
from typing import Iterable, Tuple

import numpy as np


PROFILE_SCHEMA = "categorical_v1"
BEHAVIOR_CONDITIONING_VERSION = 1
BEHAVIOR_CONDITION_DIM = 4


class BehaviorProfile(IntEnum):
    REACTIVE = 0
    COMPETENT = 1
    TACTICAL = 2
    ADVANCED = 3


PROFILE_NAMES = tuple(profile.name.lower() for profile in BehaviorProfile)


def resolve_behavior_profile(profile) -> BehaviorProfile:
    if isinstance(profile, BehaviorProfile):
        return profile
    if isinstance(profile, (int, np.integer)):
        return BehaviorProfile(int(profile))
    normalized = str(profile).strip().lower()
    try:
        return BehaviorProfile[normalized.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown behavior profile '{profile}'. Expected one of: "
            f"{', '.join(PROFILE_NAMES)}") from exc


def normalize_profile_set(profiles: Iterable) -> Tuple[BehaviorProfile, ...]:
    resolved = tuple(resolve_behavior_profile(profile) for profile in profiles)
    if not resolved:
        raise ValueError("At least one behavior profile is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Behavior profiles must be unique")
    return resolved


def behavior_condition(profile, dtype=np.float32) -> np.ndarray:
    condition = np.zeros(BEHAVIOR_CONDITION_DIM, dtype=dtype)
    condition[int(resolve_behavior_profile(profile))] = 1.0
    return condition


def profile_objective_reward(profile, prev, action, curr):
    """Small spike objectives with measurable opposing preferences.

    Shared objective/pressure rewards remain dominant in reward.py. These
    bounded terms only make Reactive and Tactical prefer different actions in
    identical opportunity states for the first conditioning experiment.
    """
    profile = resolve_behavior_profile(profile)
    _, combat_action, target_action = (int(value) for value in action)
    components = {}

    if profile == BehaviorProfile.REACTIVE:
        if combat_action == 1 and prev.credible_fire_opportunity:
            components["profile_reactive_direct_response"] = 0.01
        if combat_action == 8 and prev.credible_fire_opportunity:
            components["profile_reactive_unnecessary_reposition"] = -0.01
        if combat_action in (3, 4) and prev.credible_fire_opportunity:
            components["profile_reactive_unnecessary_switch"] = -0.008
        if target_action != 4 and prev.target_alive:
            components["profile_reactive_unnecessary_target_churn"] = -0.004

    elif profile == BehaviorProfile.COMPETENT:
        if curr.weapon_switched and curr.credible_fire_opportunity:
            components["profile_competent_useful_switch"] = 0.006
        if combat_action in (6, 7) and curr.damage_taken_this_step > 0.0:
            components["profile_competent_defence"] = 0.006

    elif profile == BehaviorProfile.TACTICAL:
        if combat_action == 8 and (
                not prev.has_los or not prev.in_optimal_range):
            components["profile_tactical_reposition"] = 0.012
        if curr.weapon_switched and (
                curr.arc_fire_opportunity or curr.credible_fire_opportunity):
            components["profile_tactical_useful_switch"] = 0.01
        if target_action != 4 and prev.alive_hostiles > 1:
            components["profile_tactical_target_choice"] = 0.006

    else:  # Advanced: add gates only after the two-profile spike succeeds.
        if (curr.ally_target_id >= 0
                and curr.target_id == curr.ally_target_id
                and combat_action == 1
                and prev.credible_fire_opportunity):
            components["profile_advanced_focus_fire"] = 0.008

    return float(sum(components.values())), components
