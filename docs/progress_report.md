# OpenArm 双臂抓取 — 进展报告

**日期**：2026-06-01  
**版本**：v15（三重 NaN 防线 + empirical_normalization）

---

## 1. 项目概述

在 Isaac Lab 上训练 OpenArm 双臂机器人，根据指令条件（`hand_side=0`=左手, `=1`=右手）用指定手臂抓取平台上的方块，举升并移动到目标位置。

**约束**：纯 RL，无人类遥操作演示数据。

---

## 2. 环境布局

| 对象 | 位置 (x, y, z) | 说明 |
|---|---|---|
| 机器人底座 | (0, 0, 0) | 臂基座在 z=0.698 |
| 平台 | (0.25, 0, 0.18) | 运动学刚体 50×40×4cm |
| 方块 | (0.25, 0, 0.24) | DexCube scale=0.6 (4.8cm) |
| 地面 | (0, 0, -1.05) | 视觉地面 |
| 目标 x | 0.15~0.30 | |
| 目标 y | -0.10~0.10 | |
| 目标 z | 0.50~0.70 | 举升阈值 z>0.37 上方 |
| 目标坐标系 | world 基座 | **body_name="openarm_body_link"**（v11 修复）|

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

## 5. 奖励函数 (对齐 Isaac Lab 官方)

| 奖励 | 权重 | 公式 |
|---|---|---|
| left_reaching | 1.0 | `(1-tanh(左手距/0.5)) × (hand_side=0)` |
| right_reaching | 1.0 | `(1-tanh(右手距/0.5)) × (hand_side=1)` |
| left_grasp | 2.0 | `(距<6cm & 夹爪闭合) × (hand_side=0)` |
| right_grasp | 2.0 | `(距<6cm & 夹爪闭合) × (hand_side=1)` |
| lifting_object | 15.0 | `(z > 0.24) — 二元` |
| goal_tracking | 16.0 | `(z>0.24) × (1-tanh(距目标/0.3))` |
| goal_tracking_fine | 5.0 | 同上，std=0.05 |
| action_rate | -1e-4 | 动作平滑 |
| joint_vel | -1e-4 | 关节速度惩罚 |

**链条**: reach(1) → grasp(2) → lift(15) → track(16+5)

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
| 观测归一化 | empirical_normalization=True（±5σ 截断） |
| 观测截断 | clip_obs=10.0 |
| 动作截断 | clip_actions=1.0 |
| 视频间隔 | 200 iter（约 10 分钟） |
| 视频时长 | 500 步（完整 1 episode） |

---

## 8. 训练历史

| 版本 | 日期 | 关键改动 | 结果 |
|---|---|---|---|
| v1-v8 | 5/26-5/28 | 多次 reward 结构迭代、IK 控制尝试 | NaN 多次爆炸、grasp 未激活 |
| v9 | 5/29 | 奖励对齐单臂 demo、网络扩容[256,128,64] | entropy 持续飙升到 40+ |
| **v10** | **5/29** | **Action Masking、方块固定、无阶段** | **首次稳定收敛** |
| **v15** | **6/01** | **三重 NaN 防线: empirical_normalization=True + 夹爪 effort 40 + 刚度 500** | 待训练 |
| **v13** | **6/01** | **初始姿态: pos=-0.2, joint2=±1.2, joint4=1.5, 手指张开** | NaN 爆炸 |

### v10 关键数据 (iter 154)

```
noise_std = 0.42    (vs 旧版 3+)
entropy   = 8.1     (vs 旧版 40+)
dropping  = 0%      (vs 旧版 20%)
ep时长     = 500    (全部跑满)
```

---

## 9. 当前状态

- v15 待训练：三重 NaN 防线已部署，对标 Isaac Lab 官方
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

| 问题 | 状态 | 版本 |
|---|---|---|
| body_name="openarm_left_hand" 导致目标漂移 | ✅ 已修复 | v11 |
| 夹爪刚度过高 (2000 N/m) 弹飞方块 | ✅ 已修复 | v11 |
| 立方体尺寸过大 (8cm) 夹不住 | ✅ 已修复 | v10 |
| FrameTransformer 无法用于双手（USD 缺少 link0）| ❌ 降级 | — |
| ContactSensor 不可用（手指碰撞几何缺失）| ❌ 降级 | — |
| 系统内存耗尽 | ⚠️ 训练时需降 envs | — |
| 奖励变化时模型续接失败 | ⚠️ Actor 重置 | — |

---

## 12. 下一步

1. `rm -rf logs/*` 清空旧日志，重新起跑 v11
2. 观察 v11 训练首次举升时间
3. 降 envs 到 2048 缓解内存
4. 首次举升成功后考虑连续举升奖励（continuous_lifting_reward, weight=500）
5. 搭建 hand_side 控制的数据集记录 pipeline
6. 长期：域态随机化 → sim-to-real 基础
