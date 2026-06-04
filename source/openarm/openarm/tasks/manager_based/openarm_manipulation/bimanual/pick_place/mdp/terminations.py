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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_block_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )

    if not hasattr(env, "_target_block_buf"):
        env._target_block_buf = torch.zeros(env.num_envs, device=env.device)
    idx = env._target_block_buf.long()
    obj0: RigidObject = env.scene["object_1"]
    obj1: RigidObject = env.scene["object_2"]
    obj2: RigidObject = env.scene["object_3"]

    pos0 = obj0.data.root_pos_w[:, :3]
    pos1 = obj1.data.root_pos_w[:, :3]
    pos2 = obj2.data.root_pos_w[:, :3]

    dist0 = torch.norm(des_pos_w - pos0, dim=1)
    dist1 = torch.norm(des_pos_w - pos1, dim=1)
    dist2 = torch.norm(des_pos_w - pos2, dim=1)

    dist = torch.where(idx == 0, dist0, torch.where(idx == 1, dist1, dist2))
    return dist < threshold
