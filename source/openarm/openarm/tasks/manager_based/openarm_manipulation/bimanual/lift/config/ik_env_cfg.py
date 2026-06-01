# Copyright 2025 Enactic, Inc.

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg


@configclass
class OpenArmCubeLiftIKEnvCfg(joint_pos_env_cfg.OpenArmCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # left arm: EE pose → IK → joint positions
        self.actions.left_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_left_joint1", "openarm_left_joint2",
                "openarm_left_joint3", "openarm_left_joint4",
                "openarm_left_joint5", "openarm_left_joint6",
                "openarm_left_joint7",
            ],
            body_name="openarm_left_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="pinv",
                ik_params={"k_val": 0.5},
            ),
            scale=0.02,
        )

        # right arm: EE pose → IK → joint positions
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint1", "openarm_right_joint2",
                "openarm_right_joint3", "openarm_right_joint4",
                "openarm_right_joint5", "openarm_right_joint6",
                "openarm_right_joint7",
            ],
            body_name="openarm_right_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="pinv",
                ik_params={"k_val": 0.5},
            ),
            scale=0.02,
        )

        # IK works in task space → drop joint-level obs (not needed anymore)
        self.observations.policy.left_joint_pos = None
        self.observations.policy.right_joint_pos = None
        self.observations.policy.left_joint_vel = None
        self.observations.policy.right_joint_vel = None


@configclass
class OpenArmCubeLiftIKEnvCfg_PLAY(OpenArmCubeLiftIKEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
