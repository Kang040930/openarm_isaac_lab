# source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/events.py

```python
# Copyright 2025 Enactic, Inc.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def staged_reset_object_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
    velocity_range: dict,
    asset_cfg: SceneEntityCfg,
):
    """Reset object position with curriculum stages.

    Stage 1+2 (common_step < 48000): cube stays at default pos (no randomization).
    Stage 3+ (common_step >= 48000): uniform randomization within pose_range.
    """
    sc = env.common_step_counter
    avg_step = sc.float().mean().item() if hasattr(sc, "float") else 0.0
    if avg_step < 1e9:  # Stage 3 disabled — keep fixed position
        return  # Stage 1+2: fixed position

    # Stage 2+: randomize
    from isaaclab.envs.mdp.events import reset_root_state_uniform
    reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)

```
