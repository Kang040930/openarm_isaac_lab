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


def left_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
) -> torch.Tensor:
    """Reaching reward for left hand (only active when hand_side=0)."""
    hand_side = _get_hand_side(env)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(object.data.root_pos_w[:, :3] - ee_w, dim=1)
    return (1 - torch.tanh(dist / std)) * (hand_side < 0.5).float()


def right_reaching_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
) -> torch.Tensor:
    """Reaching reward for right hand (only active when hand_side=1)."""
    hand_side = _get_hand_side(env)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(object.data.root_pos_w[:, :3] - ee_w, dim=1)
    return (1 - torch.tanh(dist / std)) * (hand_side > 0.5).float()


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def continuous_lifting_reward(
    env: ManagerBasedRLEnv,
    initial_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous lifting reward: height above initial position.
    
    Reward = z - initial_height, clipped at 0.
    This makes every millimeter of lift a tiny step toward the goal.
    """
    object: RigidObject = env.scene[object_cfg.name]
    lift = object.data.root_pos_w[:, 2] - initial_height
    return torch.clamp(lift, min=0.0)


def left_grasp_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_left"),
) -> torch.Tensor:
    """Reward closing left gripper when hand is near the cube (hand_side=0 only)."""
    hand_side = _get_hand_side(env)
    robot: Articulation = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    dist = torch.norm(object.data.root_pos_w[:, :3] - ee_w, dim=1)
    near_object = dist < distance_threshold

    finger_idx = robot.find_joints("openarm_left_finger_joint.*")[0]
    finger_pos = robot.data.joint_pos[:, finger_idx]
    gripper_closed = finger_pos.mean(dim=1) < 0.02

    return (near_object & gripper_closed).float() * (hand_side < 0.5).float()


def right_grasp_reward(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_right"),
) -> torch.Tensor:
    """Reward closing right gripper when hand is near the cube (hand_side=1 only)."""
    hand_side = _get_hand_side(env)
    robot: Articulation = env.scene["robot"]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    dist = torch.norm(object.data.root_pos_w[:, :3] - ee_w, dim=1)
    near_object = dist < distance_threshold

    finger_idx = robot.find_joints("openarm_right_finger_joint.*")[0]
    finger_pos = robot.data.joint_pos[:, finger_idx]
    gripper_closed = finger_pos.mean(dim=1) < 0.02

    return (near_object & gripper_closed).float() * (hand_side > 0.5).float()
