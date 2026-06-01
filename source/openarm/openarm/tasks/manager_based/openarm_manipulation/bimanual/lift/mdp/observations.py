# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

VISION_DIM = 64

_SAFE = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return _SAFE(object_pos_b)


def hand_side_label(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hand side assignment with curriculum stages.

    Stage 1 (common_step < 24000): force left hand only.
    Stage 2 (common_step >= 24000): random 0=left, 1=right.
    Stage 3 (future): random with full randomization.
    Applies at episode boundaries.
    """
    import os
    num_envs = env.num_envs
    if not hasattr(env, "_hand_side_buf") or env._hand_side_buf.shape[0] != num_envs:
        env._hand_side_buf = torch.zeros(num_envs, device=env.device)
    forced = os.environ.get("OPENARM_HAND_SIDE")
    if forced is not None:
        env._hand_side_buf[:] = float(forced)
    else:
        reset_mask = env.episode_length_buf == 0
        if reset_mask.any():
            env._hand_side_buf[reset_mask] = torch.randint(
                0, 2, (reset_mask.sum().item(),), device=env.device
            ).float()
    return env._hand_side_buf.unsqueeze(-1)


def left_hand_object_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    object: RigidObject = env.scene["object"]
    left_idx = robot.find_bodies("openarm_left_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, left_idx, :] - object.data.root_pos_w[:, :3])


def right_hand_object_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    object: RigidObject = env.scene["object"]
    right_idx = robot.find_bodies("openarm_right_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, right_idx, :] - object.data.root_pos_w[:, :3])


def vision_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, VISION_DIM, device=env.device)


def vision_validity_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, VISION_DIM, device=env.device)


def object_type_label(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, 4, device=env.device)
