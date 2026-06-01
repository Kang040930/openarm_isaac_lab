# Copyright 2025 Enactic, Inc.

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class MaskedJointPositionAction(JointPositionAction):
    """Joint position action that masks output when hand is non-active.

    When side="left" and env says hand_side=1 (right active): output zeroed.
    When side="right" and env says hand_side=0 (left active): output zeroed.
    """

    def process_actions(self, actions: torch.Tensor):
        hand_side_buf = getattr(self._env, "_hand_side_buf", None)
        if hand_side_buf is not None:
            if self.cfg.side == "left":
                mask = (hand_side_buf < 0.5).float()
            else:
                mask = (hand_side_buf > 0.5).float()
            actions = actions * mask.unsqueeze(-1)
        super().process_actions(actions)


from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg


from isaaclab.utils import configclass

@configclass
class MaskedJointPositionActionCfg(JointPositionActionCfg):
    """Config for MaskedJointPositionAction with hand_side awareness.

    side: "left" or "right" — which hand this action controls.
    """
    class_type: type = MaskedJointPositionAction
    side: str = "left"
