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

_SAFE = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _object_pos_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
    )
    return _SAFE(obj_pos_b)


def block1_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _object_pos_in_robot_frame(env, object_cfg=SceneEntityCfg("object_1"))


def block2_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _object_pos_in_robot_frame(env, object_cfg=SceneEntityCfg("object_2"))


def block3_position_in_robot_root_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _object_pos_in_robot_frame(env, object_cfg=SceneEntityCfg("object_3"))


def hand_side_label(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_envs = env.num_envs
    if not hasattr(env, "_hand_side_buf") or env._hand_side_buf.shape[0] != num_envs:
        env._hand_side_buf = torch.zeros(num_envs, device=env.device)
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._hand_side_buf[reset_mask] = torch.randint(
            0, 2, (reset_mask.sum().item(),), device=env.device
        ).float()
    return env._hand_side_buf.unsqueeze(-1)


def target_block_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    num_envs = env.num_envs
    if not hasattr(env, "_target_block_buf") or env._target_block_buf.shape[0] != num_envs:
        env._target_block_buf = torch.zeros(num_envs, device=env.device)
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._target_block_buf[reset_mask] = torch.randint(
            0, 3, (reset_mask.sum().item(),), device=env.device
        ).float()
    return env._target_block_buf.unsqueeze(-1)


def left_hand_block1_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_1"]
    left_idx = robot.find_bodies("openarm_left_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, left_idx, :] - obj.data.root_pos_w[:, :3])


def left_hand_block2_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_2"]
    left_idx = robot.find_bodies("openarm_left_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, left_idx, :] - obj.data.root_pos_w[:, :3])


def left_hand_block3_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_3"]
    left_idx = robot.find_bodies("openarm_left_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, left_idx, :] - obj.data.root_pos_w[:, :3])


def right_hand_block1_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_1"]
    right_idx = robot.find_bodies("openarm_right_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, right_idx, :] - obj.data.root_pos_w[:, :3])


def right_hand_block2_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_2"]
    right_idx = robot.find_bodies("openarm_right_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, right_idx, :] - obj.data.root_pos_w[:, :3])


def right_hand_block3_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    obj: RigidObject = env.scene["object_3"]
    right_idx = robot.find_bodies("openarm_right_hand")[0][0]
    return _SAFE(robot.data.body_pos_w[:, right_idx, :] - obj.data.root_pos_w[:, :3])
