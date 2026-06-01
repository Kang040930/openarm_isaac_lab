# OpenArm 双臂抓取 — 开发路线图

## 时间线

```
Phase 1          Phase 2           Phase 3           Phase 4 (future)
仿真RL训练     → 联合视觉RL训练  → 传感器失活过渡  → 真实机器人部署
```

---

## Phase 1: 仿真 RL 训练（进行中）

**目标**：在 Isaac Lab 中训练出一个能按指令用左手或右手抓取方块的策略。

**当前状态**：切换到 ContactSensor 物理接触检测（对齐 robosuite/ManiSkill），消去距离代理假阳性问题。

### 1.1 环境搭建 ✅

- 双臂 OpenArm 机器人 (7-DoF × 2 臂)
- 方块物体 (DexCube, scale=0.8)
- 运动学平台作为支撑面
- 400 并行环境

### 1.2 基础训练（迭代中）

| 尝试 | 控制方式 | 问题 | 状态 |
|---|---|---|---|
| v1 | 关节位置 | 奖励太稀疏，无法抓取 | 🔴 |
| v2 | 关节位置 + best_hand 奖励 | 两手互推，掉落率高 | 🔴 |
| v3 | 关节位置 + side 标签 + per-hand 奖励 | 策略未学会读标签 | 🔴 |
| v4 | 关节位置 + Robosuite 风格奖励 | 接近物体但无法举起 | 🟡 |
| v5 | **IK 控制** + Robosuite 风格奖励 | NaN 数值爆炸 | 🟡 |
| v6 | **IK 控制** + 减 padding + 小步长 | NaN 多次爆炸 | 🔴 |
| v7 | **关节位置控制** + NaN 守卫 + idle 惩罚 | 稳定运行，grasp 未触发 | 🟡 |
| v8 | **IK 控制** + ContactSensor 物理接触 | ContactSensor 不支持 | 🔴 |
| v9 | **IK 控制** + grasp 阈值修复 + 双重 idle 约束 | 过长训练后卡死 | 🔴 |
| v10 | **关节位置** + 奖励对齐单臂 demo + 正向选手 | **训练中** | 🟢 |

### 1.3 关键技术决策

| 决策 | 原因 |
|---|---|
| IK 控制而非关节位置 | 探索效率提升 100×，可部署到真实机器人 |
| per-hand 奖励而非 best_hand | 支持 hand_side 标签控制左右手 |
| hand_side 遮挡奖励 | 策略学会按标签选择动手 |
| 无 action_rate 惩罚 | Robosuite 风格 — IK 天然平滑 |
| 无 object_dropping 终止 | 确保完整梯度信号 |

---

## Phase 2: 联合视觉 RL 训练（后续）

**目标**：向观测中添加真实视觉编码特征，训练多模态策略。

### 2.1 规划

- 摄像头传感器接入 Isaac Lab 仿真
- ResNet/ViT 编码器提取视觉特征
- 填充到当前预留的 vision_features (64维)
- 联合训练（本体感觉 + 视觉）

### 2.2 依赖

- Phase 1 中的策略必须学会基本抓取行为
- 视觉编码器管道就绪

---

## Phase 3: 传感器失活过渡（后续）

**目标**：模拟真实部署时的传感器缺失情况。

### 3.1 规划

- 随机 dropout 方块位置观测（`object_position`）
- 策略从 proprioception + vision 推断物体位置
- 最终实现纯视觉 + proprioception 控制

---

## Phase 4: 真实机器人部署（未来）

**目标**：将训练好的策略部署到真实 OpenArm 双臂机器人上。

### 4.1 部署流水线

```
摄像头 → ViT 编码器 → vision_features (64维)
编码器 → joint_pos (14维)
         joint_vel (14维)
         gripper_pos (4维)
         → 拼接 → MLP 策略 → EE 位姿
                              → pinocchio IK → 关节角度
                                               → ETHERCAT → 电机
```

### 4.2 前提条件

- 域态随机化训练（仿真中随机化物理参数、视觉条件）
- 策略能在纯视觉 + proprioception 条件下控制
- IK 求解器和仿真中使用的 URDF 模型一致
