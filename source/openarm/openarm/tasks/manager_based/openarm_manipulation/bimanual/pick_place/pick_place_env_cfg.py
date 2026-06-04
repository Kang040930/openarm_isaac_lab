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


from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the bimanual pick-and-place scene with three blocks."""

    robot: ArticulationCfg = MISSING
    ee_frame_left: FrameTransformerCfg = MISSING
    ee_frame_right: FrameTransformerCfg = MISSING
    object_1: RigidObjectCfg | DeformableObjectCfg = MISSING
    object_2: RigidObjectCfg | DeformableObjectCfg = MISSING
    object_3: RigidObjectCfg | DeformableObjectCfg = MISSING
    platform: RigidObjectCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.05, 0.05, 0.05)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(1e6, 1e6),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.45),
            pos_y=(0.0, 0.0),
            pos_z=(0.55, 0.55),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: (
        mdp.JointPositionActionCfg
        | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    right_arm_action: (
        mdp.JointPositionActionCfg
        | mdp.DifferentialInverseKinematicsActionCfg
    ) = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint1",
                        "openarm_left_joint2",
                        "openarm_left_joint3",
                        "openarm_left_joint4",
                        "openarm_left_joint5",
                        "openarm_left_joint6",
                        "openarm_left_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_left_joint1",
                        "openarm_left_joint2",
                        "openarm_left_joint3",
                        "openarm_left_joint4",
                        "openarm_left_joint5",
                        "openarm_left_joint6",
                        "openarm_left_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "openarm_right_joint1",
                        "openarm_right_joint2",
                        "openarm_right_joint3",
                        "openarm_right_joint4",
                        "openarm_right_joint5",
                        "openarm_right_joint6",
                        "openarm_right_joint7",
                    ],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_left_finger_joint.*"]
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["openarm_right_finger_joint.*"]
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        block1_position = ObsTerm(func=mdp.block1_position_in_robot_root_frame)
        block2_position = ObsTerm(func=mdp.block2_position_in_robot_root_frame)
        block3_position = ObsTerm(func=mdp.block3_position_in_robot_root_frame)

        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )

        hand_side_label = ObsTerm(func=mdp.hand_side_label)
        target_block_index = ObsTerm(func=mdp.target_block_index)

        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object1_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.25, -0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_1"),
        },
    )

    reset_object2_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_2"),
        },
    )

    reset_object3_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (0.05, 0.25),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object_3"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    left_reaching = RewTerm(
        func=mdp.left_reaching_target_block, params={"std": 0.1}, weight=1.1
    )
    right_reaching = RewTerm(
        func=mdp.right_reaching_target_block, params={"std": 0.1}, weight=1.1
    )

    lifting_target_block = RewTerm(
        func=mdp.target_block_is_lifted, params={"minimal_height": 0.38}, weight=15.0
    )

    wrong_block_penalty = RewTerm(
        func=mdp.wrong_block_lifted_penalty,
        params={"minimal_height": 0.38},
        weight=-5.0,
    )

    block_goal_tracking = RewTerm(
        func=mdp.target_block_goal_distance,
        params={"std": 0.3, "minimal_height": 0.38, "command_name": "object_pose"},
        weight=16.0,
    )

    block_goal_tracking_fine = RewTerm(
        func=mdp.target_block_goal_distance_fine,
        params={"std": 0.05, "minimal_height": 0.38, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
                    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
                    "openarm_left_joint7",
                ],
            )
        },
    )

    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
                    "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6",
                    "openarm_right_joint7",
                ],
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object_1")},
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 50.0},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual multi-block pick-and-place environment."""

    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
