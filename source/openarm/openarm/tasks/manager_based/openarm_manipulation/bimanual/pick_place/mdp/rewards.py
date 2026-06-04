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
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_hand_side(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_hand_side_buf") or env._hand_side_buf.shape[0] != env.num_envs:
        env._hand_side_buf = torch.zeros(env.num_envs, device=env.device)
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._hand_side_buf[reset_mask] = torch.randint(
            0, 2, (reset_mask.sum().item(),), device=env.device
        ).float()
    return env._hand_side_buf


def _get_target_block_idx(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_target_block_buf") or env._target_block_buf.shape[0] != env.num_envs:
        env._target_block_buf = torch.zeros(env.num_envs, device=env.device)
    reset_mask = env.episode_length_buf == 0
    if reset_mask.any():
        env._target_block_buf[reset_mask] = torch.randint(
            0, 3, (reset_mask.sum().item(),), device=env.device
        ).float()
    return env._target_block_buf


def _get_target_block_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    idx = _get_target_block_idx(env).long()
    mask_0 = (idx == 0).unsqueeze(-1).float()
    mask_1 = (idx == 1).unsqueeze(-1).float()
    mask_2 = (idx == 2).unsqueeze(-1).float()
    obj0: RigidObject = env.scene["object_1"]
    obj1: RigidObject = env.scene["object_2"]
    obj2: RigidObject = env.scene["object_3"]
    return (
        mask_0 * obj0.data.root_pos_w
        + mask_1 * obj1.data.root_pos_w
        + mask_2 * obj2.data.root_pos_w
    )


def left_reaching_target_block(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
) -> torch.Tensor:
    hand_side = _get_hand_side(env)
    target_pos = _get_target_block_position(env)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(target_pos[:, :3] - ee_w, dim=1)
    return (1 - torch.tanh(dist / std)) * (hand_side < 0.5).float()


def right_reaching_target_block(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
) -> torch.Tensor:
    hand_side = _get_hand_side(env)
    target_pos = _get_target_block_position(env)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(target_pos[:, :3] - ee_w, dim=1)
    return (1 - torch.tanh(dist / std)) * (hand_side > 0.5).float()


def target_block_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
) -> torch.Tensor:
    target_pos = _get_target_block_position(env)
    return torch.where(target_pos[:, 2] > minimal_height, 1.0, 0.0)


def wrong_block_lifted_penalty(
    env: ManagerBasedRLEnv,
    minimal_height: float,
) -> torch.Tensor:
    idx = _get_target_block_idx(env).long()
    obj0: RigidObject = env.scene["object_1"]
    obj1: RigidObject = env.scene["object_2"]
    obj2: RigidObject = env.scene["object_3"]

    lifted_0 = (obj0.data.root_pos_w[:, 2] > minimal_height).float()
    lifted_1 = (obj1.data.root_pos_w[:, 2] > minimal_height).float()
    lifted_2 = (obj2.data.root_pos_w[:, 2] > minimal_height).float()

    return (idx != 0).float() * lifted_0 + (idx != 1).float() * lifted_1 + (idx != 2).float() * lifted_2


def target_block_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    target_pos = _get_target_block_position(env)
    distance = torch.norm(des_pos_w - target_pos[:, :3], dim=1)
    return (target_pos[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def target_block_goal_distance_fine(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    target_pos = _get_target_block_position(env)
    distance = torch.norm(des_pos_w - target_pos[:, :3], dim=1)
    return (target_pos[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
