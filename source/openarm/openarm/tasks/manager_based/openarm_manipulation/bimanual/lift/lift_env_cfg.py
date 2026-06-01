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
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

import math

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the bimanual lift scene with a robot and an object."""

    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    platform: RigidObjectCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
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
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.3),
            pos_y=(-0.1, 0.1),
            pos_z=(0.5, 0.7),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg | mdp.MaskedJointPositionActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg | mdp.MaskedJointPositionActionCfg = MISSING
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
        lef_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_left_finger_joint.*"],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_gripper_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["openarm_right_finger_joint.*"],
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        hand_side_label = ObsTerm(func=mdp.hand_side_label)
        left_hand_obj_rel = ObsTerm(func=mdp.left_hand_object_rel_pos)
        right_hand_obj_rel = ObsTerm(func=mdp.right_hand_object_rel_pos)
        vision_features = ObsTerm(func=mdp.vision_features)
        vision_validity = ObsTerm(func=mdp.vision_validity_mask)
        object_type_label = ObsTerm(func=mdp.object_type_label)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.staged_reset_object_position,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (aligned with OpenArm unimanual lift demo)."""

    left_reaching = RewTerm(func=mdp.left_reaching_reward, params={"std": 0.5}, weight=2.0)
    right_reaching = RewTerm(func=mdp.right_reaching_reward, params={"std": 0.5}, weight=2.0)

    lifting_object = RewTerm(func=mdp.continuous_lifting_reward, params={"initial_height": 0.34}, weight=50.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.37, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.37, "command_name": "object_pose"},
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
    )

    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
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
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    joint_vel_out_of_limit = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_velocity": 50.0},
    )


@configclass
class CurriculumCfg:
    """No curriculum — hand_side always random, cube always fixed.
    Action masking handles hand selection.
    """
    pass


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the bimanual lifting environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
