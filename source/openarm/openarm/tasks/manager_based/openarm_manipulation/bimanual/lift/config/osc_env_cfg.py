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


from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import (
    OPEN_ARM_HIGH_PD_CFG,
)


@configclass
class OpenArmCubeLiftOSCEnvCfg(joint_pos_env_cfg.OpenArmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # disable joint stiffness/damping for torque-based OSC
        self.scene.robot = OPEN_ARM_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=self.scene.robot.init_state,
        )
        self.scene.robot.actuators["openarm_arm"].stiffness = 0.0
        self.scene.robot.actuators["openarm_arm"].damping = 0.0

        # left arm OSC
        self.actions.left_arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            body_name="openarm_left_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                motion_stiffness_task=150.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 300.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=0.05,
            orientation_scale=0.1,
            stiffness_scale=150.0,
        )

        # right arm OSC
        self.actions.right_arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ],
            body_name="openarm_right_hand",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                motion_stiffness_task=150.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 300.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=0.05,
            orientation_scale=0.1,
            stiffness_scale=150.0,
        )

        # keep binary gripper actions unchanged
        # self.actions.left_gripper_action is already set from parent
        # self.actions.right_gripper_action is already set from parent

        # OSC works in task space → remove joint-level observations to shrink obs dim
        self.observations.policy.left_joint_pos = None
        self.observations.policy.right_joint_pos = None
        self.observations.policy.left_joint_vel = None
        self.observations.policy.right_joint_vel = None

        self.scene.num_envs = 1024
        self.scene.env_spacing = 3.0


@configclass
class OpenArmCubeLiftOSCEnvCfg_PLAY(OpenArmCubeLiftOSCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
