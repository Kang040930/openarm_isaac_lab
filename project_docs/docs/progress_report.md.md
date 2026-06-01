# docs/progress_report.md

```markdown
# OpenArm 双臂抓取 — 进展报告

**日期**：2026-05-30  
**版本**：v10（Action Masking + 无阶段课程）

---

## 1. 项目概述

在 Isaac Lab 上训练 OpenArm 双臂机器人，根据指令条件（`hand_side=0`=左手, `=1`=右手）用指定手臂抓取平台上的方块，举升并移动到目标位置。

**约束**：纯 RL，无人类遥操作演示数据。

---

## 2. 环境布局

| 对象 | 位置 (x, y, z) | 说明 |
|---|---|---|
| 机器人底座 | (0, 0, 0) | 臂基座在 z=0.698 |
| 平台 | (0.25, 0, 0.28) | 运动学刚体 50×40×4cm |
| 方块 | (0.25, 0, 0.34) | DexCube scale=0.8 (8cm) |
| 地面 | (0, 0, -1.05) | 视觉地面 |
| 目标 x | 0.15~0.30 | |
| 目标 y | -0.10~0.10 | |
| 目标 z | 0.50~0.70 | 举升阈值 z>0.37 上方 |

---

## 3. 控制方式

**关节位置控制**（Isaac-Lift-Cube-OpenArm-Bi-v0）

| 动作 | 维度 |
|---|---|
| left_arm_action | 7 (关节角度) |
| right_arm_action | 7 (关节角度) |
| left_gripper_action | 1 (二值) |
| right_gripper_action | 1 (二值) |
| **总计** | **16** |

---

## 4. Action Masking（核心创新）

非活动手在物理层被屏蔽——解决了"两手互抢+非活动手乱动"问题：

```python
# mdp/actions.py — MaskedJointPositionAction
hand_side=0 (左手模式): right_arm_action = [0,0,0,0,0,0,0]
hand_side=1 (右手模式): left_arm_action  = [0,0,0,0,0,0,0]
```

效果：
- 非活动手物理上不可移动 — 不需要惩罚
- 探索空间从 14 维降至 7 维（每个 episode）
- 策略从第一天就学会"看标签选手"

---

## 5. 奖励函数

**完全对齐 OpenArm 单臂 Lift Demo**

| 奖励 | 权重 | 公式 |
|---|---|---|
| left_reaching | 1.1 | `(1-tanh(左手距/0.1)) × (hand_side=0)` |
| right_reaching | 1.1 | `(1-tanh(右手距/0.1)) × (hand_side=1)` |
| lifting_object | 8.0 | `(方块z > 0.37) — 二值` |
| object_goal_tracking | 16.0 | `(1-tanh(距目标/0.3)) × (z>0.37)` |
| object_goal_tracking_fine | 5.0 | 同上，精细版 std=0.05 |
| action_rate | -1e-4 | 动作平滑 |
| left_joint_vel | -1e-4 | 左关节速度 |
| right_joint_vel | -1e-4 | 右关节速度 |

**关键设计**：
- lifting=8.0 + 阈值 0.37：方块起始 0.34，举 3cm 即触发 → 底薪角色
- goal_tracking=16.0：主力驱动方块到目标 (z=0.5~0.7)
- 无 grasp 奖励：单臂 demo 证明 reaching→lifting 链已足够
- 无 idle 惩罚：Action Masking 替代

---

## 6. 观测量 (195 维)

| 类别 | 维度 | 内容 |
|---|---|---|
| 关节位 | 7×2=14 | 左右臂 7 关节 |
| 关节速 | 7×2=14 | 左右臂速度 |
| 夹爪 | 2×2=4 | 左右手指位置 |
| 物体 | 3 | 方块机器人坐标系位置 |
| 目标 | 7 | 目标位姿(四元数) |
| 动作历史 | 7×2=14 | 左右臂上一次 action |
| hand_side | 1 | 0=左手, 1=右手 |
| 手-物相对 | 3×2=6 | 左右手到方块的位移向量 |
| 视觉预留 | 64+64=128 | vision_features + validity_mask |
| 物体分类 | 4 | object_type_label |

---

## 7. 训练超参数

| 参数 | 值 |
|---|---|
| 网络 | MLP [256,128,64] ELU |
| 环境数 | 4096 |
| episodes | 10s = 500 控制步 |
| 总迭数 | 40000 |
| PPO学习率 | 3e-5 自适应 |
| γ/λ | 0.98/0.95 |
| 梯度裁剪 | 0.5 |
| 初始噪声 | 1.0 |
| 值损失系数 | 0.5 |
| 熵正则系数 | 0.006 |
| 物理步长 | 0.01 (100Hz) |
| 控制步长 | 0.02 (50Hz, decimation=2) |

---

## 8. 训练历史

| 版本 | 日期 | 关键改动 | 结果 |
|---|---|---|---|
| v1-v8 | 5/26-5/28 | 多次 reward 结构迭代、IK 控制尝试 | NaN 多次爆炸、grasp 未激活 |
| v9 | 5/29 | 奖励对齐单臂 demo、网络扩容[256,128,64] | entropy 持续飙升到 40+ |
| **v10** | **5/29** | **Action Masking、方块固定、无阶段** | **首次稳定收敛** |

### v10 关键数据 (iter 154)

```
noise_std = 0.42    (vs 旧版 3+)
entropy   = 8.1     (vs 旧版 40+)
dropping  = 0%      (vs 旧版 20%)
ep时长     = 500    (全部跑满)
```

---

## 9. 当前状态

- 训练进行中（09:19 启动，Stage-2 配置：手随机、方块固定）
- Action Masking 效果验证：右手 reaching=0, joint_vel=0
- 系统内存不足（24GB/30GB），存在崩溃风险

---

## 10. 文档结构

```
docs/
├── roadmap.md              ← 4 阶段开发路线
├── experiment_design.md    ← 实验设计参数
├── changelog.md            ← 完整历史记录
└── progress_report.md      ← 本报告

project_docs/               ← 所有源文件转为 Markdown

trainings/                  ← 训练模型存档
├── 20260529_134803_network256/  (model_3000, 旧v9版本)
├── 20260529_154309_actor_resume/ (model_500)
├── 20260529_172407_pre_curriculum/ (课程学习前)
└── run.sh / save_run.sh    ← 训练自动归档脚本
```

---

## 11. 已知问题

| 问题 | 状态 |
|---|---|
| FrameTransformer 无法用于双手（USD 缺少 link0）| 降级为 body_pos_w |
| ContactSensor 不可用（手指碰撞几何缺失）| 降级为距离代理 |
| 系统内存耗尽 | 训练时需降 envs 关视频 |
| 奖励变化时模型续接失败 | Actor 重置 Critic 方案被否定 |
| IK 控制多次 NaN | 已放弃，关节位置控制稳定 |

---

## 12. 下一步

1. 观察 v10 训练首次举升时间（预计 500-800 步）
2. 降 envs 到 2048 缓解内存
3. 首次举升成功后添加课程阶段 3（目标追踪全随机）
4. 搭建 hand_side 控制的数据集记录 pipeline
5. 长期：域态随机化 → sim-to-real 基础

```
