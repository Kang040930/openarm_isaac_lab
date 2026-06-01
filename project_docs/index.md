# OpenArm Bimanual Lift — Project Files (Updated 2026-05-30)

## Config Layer
- [lift_env_cfg.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/lift_env_cfg.py.md) — 主环境 (场景/观测/奖励/终止/课程)
- [joint_pos_env_cfg.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/joint_pos_env_cfg.py.md) — 关节控制 (机器人/平台/ActionMasking)
- [ik_env_cfg.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/ik_env_cfg.py.md) — IK控制 (差分逆运动学)
- [rsl_rl_ppo_cfg.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/config/agents/rsl_rl_ppo_cfg.py.md) — PPO超参数

## MDP Layer
- [rewards.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/rewards.py.md) — 奖励函数 (per-hand + lifting + goal_tracking)
- [observations.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/observations.py.md) — 观测函数 (hand_side + vision预留)
- [actions.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/actions.py.md) — **Action Masking** (非活动手物理屏蔽)
- [events.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/events.py.md) — 事件函数 (方块位置固定)
- [terminations.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/lift/mdp/terminations.py.md) — 终止条件

## Asset Layer
- [openarm_bimanual.py](source/openarm/openarm/tasks/manager_based/openarm_manipulation/assets/openarm_bimanual.py.md) — 机器人配置

## Documentation
- [roadmap.md](docs/roadmap.md.md)
- [experiment_design.md](docs/experiment_design.md.md)
- [changelog.md](docs/changelog.md.md)
- [progress_report.md](docs/progress_report.md.md)

## Key Features (v10)
- **Action Masking**: 非活动手物理屏蔽 (mdp/actions.py)
- **奖励对齐单臂demo**: 8项固定权重, 无grasp, 无idle惩罚
- **方块固定**: 简化搜索空间
- **hand_side随机**: 正向激励选臂
