# docs/changelog.md

```markdown
## 2026-05-29
### 15:40 — 奖励完全对齐 OpenArm 单臂 demo

**问题**：之前训练 40000 步完全失败 (entropy=-8, lifting=0, 两手只在初期学习后卡死)。
**诊断**：双臂奖励太复杂 (grasp×2 + active_lift + success*25 + idle×4)，和单臂 demo 简洁路线相反。

**研究**：
- OpenArm 单臂 lift demo 只用 4 项奖励 (reaching + lifting + goal×2) + 2 项惩罚 (action_rate + joint_vel)
- Robosuite TwoArmHandover 用阶段式正向激励，不设 idle 惩罚
- ALOHA 模仿学习直接学习双手机器人行为，不需要"选臂"

**修改**（完全对齐单臂 demo）：
- 移除：grasp×2, active_lift, success, holding, idle_velocity×2, idle_ee×2
- `left_reaching` weight: 2.0→**1.1** (与单臂一致)
- `right_reaching` weight: 2.0→**1.1** (与单臂一致)
- `lifting` 改为二值 (z>0.04→**0.5**, weight=15.0) (单臂 demo style)
- `goal_tracking` weight=16.0, `goal_tracking_fine` weight=5.0 (与单臂一致)
- `action_rate` -1e-4 (单臂 style)
- `left_joint_vel`/`right_joint_vel` -1e-4 (分臂，单臂 style)
- 恢复 `object_dropping` 终止 (z<-0.05)
- 移除 CurriculumCfg (不再需要课程惩罚)

**选择手臂机制**：hand_side 只控制 reaching→正向激励，举升/目标共享

---

### 14:35 — 举升阈值修复

**问题**：z>0.04 在方块起始z=0.34时永远为1，策略白拿分不学习。
**修改**：minimal_height 0.04→**0.5** (方块需举16cm)

---

### 12:30 — 长序列训练设置

**修改**：
- max_iterations: 3000→40000
- save_interval: 50→500
- num_envs: 4096→8192
- init_noise_std: 1.0 (保持)
- lr: 3e-5, max_grad_norm: 0.5

---

# 修改日志

## 2026-05-28
### 14:22 — grasp 门槛修复 + 非活动手双重约束

**问题**：
- grasp 手距阈值 2cm 几何不可达（手掌中心+方块中心间距 > 6cm）
- 非活动手慢漂移无法被纯速度惩罚捕获

**研究**：
- ManiSkill `is_grasping()` 依赖 SAPIEN 物理接触 → Isaac Lab 不可用
- Robosuite `_check_grasp()` 依赖 MuJoCo → Isaac Lab 不可用
- ContactSensor 在 OpenArm USD 模型上无法激活（关节矩阵不符合一对一约束）

**修改**：
- `lift_env_cfg.py`: grasp threshold 0.02→0.12 (12cm, 手进入抓取范围)
- `lift_env_cfg.py`: close_threshold 0.005→0.01 (夹爪开始闭合, 不需锁死)
- `rewards.py`: 抓取检测从绝对速度改为手-物相对速度
- `rewards.py`: 新增 `idle_hand_ee_deviation_penalty`（末端位偏惩罚, 权重 -0.005）
- `lift_env_cfg.py`: 新增 `idle_left_ee_penalty` / `idle_right_ee_penalty` 奖励项
- 清理所有 ContactSensor 代码（确认 Isaac Lab 下不可用）

### 13:58 — ContactSensor 物理接触抓取检测

**问题**：grasp reward 用距离代理（手距<2cm + 夹爪<0.5cm + 相对速度<0.1）门槛太紧，
策略从未触发 → grasp 始终为 0。

**研究**：
- Robosuite: `_check_grasp()` 物理接触 → 0.25 小权重
- ManiSkill: `is_grasping()` 物理接触 → 1.0 完整权重
- Isaac Lab 官方 Franka Lift: 无抓取奖励

**修改**：
- `joint_pos_env_cfg.py`: 去掉 `filter_prim_paths_expr`，ContactSensor 只挂 prim_path
- `joint_pos_env_cfg.py`: `activate_contact_sensors=True` 在 replace 后设置
- `lift_env_cfg.py`: 场景类添加 `contact_sensor: ContactSensorCfg`
- 保留 `left_grasp` / `right_grasp` 权重=1.0，待奖励函数适配 ContactSensor

---

### 11:20 — 相对速度抓住检测

**问题**：cube 被抓住移动时绝对速度 > 0.1m/s，被误判为未抓稳。
**修改**：`rewards.py` grasp 检测从绝对速度改为手-物相对速度。

---

### 11:08 — 去除抓取奖励（对齐 Isaac Lab）

**研究**：Isaac Lab 官方 Franka Lift 不使用 grasp 奖励，reaching→lifting 链已足够。
**修改**：`lift_env_cfg.py` grasp 权重 1.0→0（后回退）。

---

### 10:45 — NaN 防护 #4：观测层 NaN 守卫 + pinv IK

**错误**：4 次训练全部在 200~500 步 NaN 爆炸（value_function → 4e22~5e29）。
**根因**：IK 产出 NaN 关节值 → 观测带入 NaN → 网络输出 NaN。
**修改**：
- `observations.py`: 所有观测加 `torch.nan_to_num(nan=0.0)` 
- `ik_env_cfg.py`: dls → pinv(k_val=0.5), scale 0.03→0.02
- `rsl_rl_ppo_cfg.py`: empirical_normalization→False, lr 1e-4→3e-5, max_grad_norm 1.0→0.5
- `rewards.py`: 添加 idle_hand_joint_vel_penalty，非活动手禁止乱动


### 01:35 — NaN 爆炸修复 #3：观测层 NaN 防护 + IK 换 pinv

**错误**：
```
RuntimeError: normal expects all elements of std >= 0.0
at iter=216, value_function_loss=5.3e29, noise_std=1.42
```
三次训练全部在 200~500 次迭代同一故障：value_function → NaN → policy std < 0 → crash。

**根因分析**：IK 控制器的 `dls`/`pinv` 在探索期随机动作下产出 NaN 关节值 → 物理引擎状态变 NaN → body_pos_w 变 NaN → 观测函数（hand_object_rel_pos）携带 NaN → 神经网络输入 NaN → 输出 NaN → 分布采样崩溃。

**修复**：
- `observations.py`: 所有观测函数输出加 `torch.nan_to_num(nan=0.0)` 守卫
- `ik_env_cfg.py`: `dls(λ=0.2)` → `pinv(k_val=0.5)`，scale 0.03→0.02
- `rsl_rl_ppo_cfg.py`: empirical_normalization True→False（NaN 不进统计）

---

### 01:17 — NaN 爆炸修复 #2：降学习率 + 紧梯度裁剪

**错误**：
```
RuntimeError: normal expects all elements of std >= 0.0
at iter=467, value_function_loss=2.5e25, noise_std=2.52↑, entropy=32↑
```
noise_std 和 entropy 反常上升，说明梯度更新幅度过大。

**修复**：learning_rate 1e-4→3e-5，max_grad_norm 1.0→0.5，IK λ 0.1→0.2

---

### 00:55 — NaN 爆炸修复 #1：网络缩容 + value_loss_coef

**错误**：
```
RuntimeError: normal expects all elements of std >= 0.0
at iter=246, value_function_loss=4e22
```
critic 值函数发散到天文数字。

**修复**：[256,128,64]→[128,64,32]，value_loss_coef 1.0→0.5，empirical_normalization→True

## 2026-05-27
### 17:35 — IK 阻尼提升至 0.1（10x）+ 平台位置回退

**问题**：手臂被平台挡住（IK 硬碰硬），NaN 爆炸（DSL 阻尼太小）。
**研究**：
- robosuite 不做碰撞过滤，OSC 柔顺接触自动处理（[完整报告]）
- Isaac Lab `differential_ik.py` 源码零 NaN 保护，默认 DLS λ=0.01 太小
- 回退平台移至 0.25（用户指出布局不是根本原因）

**修改**：
- `ik_env_cfg.py`: 左右臂 IK 均添加 `ik_params={"lambda_val": 0.1}`
- `joint_pos_env_cfg.py`: 回退平台→(0.25,0,0.28)、方块→(0.25,0,0.34)
- `docs/changelog.md` / `docs/experiment_design.md`: 更新


### 17:30 — 回退平台位置修改

**原因**：手臂从 z=0.698 到方块 z=0.34 可以越过平台，布局不是根本原因。
**回退**：平台→(0.25,0,0.28)，方块→(0.25,0,0.34)，宽度→50cm。
**方向**：检查碰撞过滤或 IK 约束。

**问题**：手臂被挡在平台底下，无法下行至方块。
**原因**：平台 x=0.25、宽 50cm，后端延伸至 x=0（机器人原点），无下行间隙。
**修改**：
- `joint_pos_env_cfg.py`: 平台 pos x: 0.25→0.35, 平台 x-scale: 5.0→3.0
- `joint_pos_env_cfg.py`: 方块 pos x: 0.25→0.35
- `experiment_design.md`: 更新场景布局表格

---

### 17:10 — 创建文档目录

**新增文件**：
- `docs/roadmap.md` — 四阶段开发路线图
- `docs/experiment_design.md` — 实验设计文档
- `docs/changelog.md` — 本文件

---

### 16:58 — NaN 数值爆炸修复

**问题**：训练至 ~570 次迭代时 `action_rate` 惩罚从 -0.001 爆炸至 -1.5e8，污染 mean_reward 至 -3500 万。
**根因**：IK 控制器在关节奇异点输出 NaN → action_rate_l2 计算 Infinity → 神经网络数值不稳定。

**修改**：
- `lift_env_cfg.py`: `action_rate` 权重 -1e-5 → **0**（完全关闭）
- `lift_env_cfg.py`: `left_joint_vel` 权重 -5e-5 → -1e-6
- `lift_env_cfg.py`: `right_joint_vel` 权重 -5e-5 → -1e-6
- `lift_env_cfg.py`: 课程 `left_joint_vel` 目标 -5e-4 → -1e-5
- `lift_env_cfg.py`: 课程 `right_joint_vel` 目标 -5e-4 → -1e-5

---

### 16:48 — 缩减视觉 padding + IK 步长保护

**数值稳定性优化**：
- `observations.py`: `VISION_DIM` 256→64（释放 75% 网络容量）
- `ik_env_cfg.py`: IK `scale` 0.05→0.03（更小步长，更少奇异点）

**观测维度更新**：
| 版本 | 总观测维度 | 零填充占比 |
|---|---|---|
| v1（256 维填充）| ~600 | 86% |
| v2（64 维填充）| ~216 | 30% |

---

### 16:42 — IK 端末控制方案

**新增**：差分 IK 控制（`Isaac-Lift-Cube-OpenArm-Bi-IK-v0`）

**新增文件**：
- `config/ik_env_cfg.py`：IK 版环境配置

**关键参数**：
- 控制方法：`DifferentialIKControllerCfg(dls, pose, relative=True)`
- 步长：scale=0.03
- 关节位置/速度观测已移除（IK 不需要）

**对比**：
| 属性 | 关节位置 v0 | IK v0 |
|---|---|---|
| 动作维度 | 14 + 2 = 16 | 12 + 2 = 14 |
| 观测维度 | ~572 | ~544 |
| 探索效率 | 低（14 维搜索）| 高（6 维末端位移）|

---

### 16:40 — Robosuite 风格 action_rate 移除

**理由**：参考 robosuite，其 `reward()` 函数不接收 `action` 参数。

**修改**：`lift_env_cfg.py`: `action_rate` 权重保持 -1e-5，但课程目标从 -1e-1 降至 -5e-4。

---

### 16:35 — 参考 Robosuite 优化

**修改汇总**：
- 奖励函数：改二值 contact→二值 grasp 检测（手<8cm 且夹爪闭合）
- 举升奖励：从二值改为**平滑线性**（z 从 0.34→0.50 线性递增）
- 成功奖励：大方块 z>0.38 且距目标<5cm
- 保持奖励：成功后每步额外 2.0
- 每手独立奖励（`left_reaching` / `right_reaching` / `left_grasp` / `right_grasp`）
- 移除 `object_dropping` 终止
- episode 从 24s 缩至 **10s**（500 步）
- 新增**手→方块相对位置观测**：`left_hand_object_rel_pos` / `right_hand_object_rel_pos`（各 3 维）

**新增函数**（`observations.py`）：
- `left_hand_object_rel_pos()`
- `right_hand_object_rel_pos()`

---

### 16:30 — Hand-Side 标签控制

**功能**：策略根据观测中的 `hand_side_label` 决定用哪只手。

| hand_side | 奖励左手 | 奖励右手 |
|---|---|---|
| 0 | ✅ reach + grasp | ❌ |
| 1 | ❌ | ✅ reach + grasp |

**推理控制**：设置环境变量 `OPENARM_HAND_SIDE=0` 或 `=1`。

**文件更改**：
- `observations.py`：`hand_side_label()` 支持推理时固定值
- `rewards.py`：各手奖励由 `hand_side` 掩码控制

---

### 16:25 — Vision Padding 准备

**新增观测**：
- `vision_features`（64维，全 0）— 未来视觉编码器填充
- `vision_validity_mask`（64维，全 0）— 1=有效 token，0=padding
- `object_type_label`（4维，全 0）— 未来区分方块类型

**设计模式**：Transformer 风格的 padding + attention mask。

---

### 16:22 — IK + IK 控制器方案

**新增文件**：
- `config/ik_env_cfg.py`：差分 IK 控制器配置
- `config/osc_env_cfg.py`：参考 OSC 控制器配置

**任务注册**（`config/__init__.py`）：
- `Isaac-Lift-Cube-OpenArm-Bi-IK-v0`
- `Isaac-Lift-Cube-OpenArm-Bi-IK-Play-v0`
- `Isaac-Lift-Cube-OpenArm-Bi-OSC-v0`

---

### 16:18 — Robosuite 风格奖励函数

**完全重写 `rewards.py`**（从 robosuite TwoArmLift 参考）：

新增函数：
- `left_reaching_reward()` / `right_reaching_reward()`
- `left_grasp_reward()` / `right_grasp_reward()`
- `smooth_lift_reward()`
- `success_reward()`
- `holding_reward()`
- `_hand_body_idx()`, `_hand_pos()`, `_gripper_pos()` 辅助函数

**奖励结构变化**：

| 旧版 | 新版 |
|---|---|
| `reaching_object`（最佳手）8.0 | `left_reaching` 2.0 + `right_reaching` 2.0 |
| `grasp_object`（最佳手）12.0 | `left_grasp` 1.0 + `right_grasp` 2.0 |
| `lifting_object`（二值）20.0 | `lifting`（平滑）5.0 |
| 无 | `success`（大额成功）10.0 |
| `holding_bonus` 8.0 | `holding` 2.0 |

---

### 15:30 — 平台大小调整

**修改**：平台 scale 从 (2.0, 1.5, 0.3) → **(3.5, 2.0, 0.3)** → **(5.0, 4.0, 0.4)**

**原因**：方块被从过小的平台上推落（dropping 率达 80%+）。

---

### 15:00 — 多种探索策略实验

按 robosuite 参考实施：
- 取消 object_dropping 终止（完整梯度信号）
- episode 从 1200 步缩至 500 步（更多次 reset）
- 移除 action_rate 极大惩罚（curriculum 从 -1e-1 降至 -5e-4）
- 延迟课程至 2500 次迭代后才生效

---

### 14:00 — Hands Side 随机化

**新增文件**：
- `mdp/observations.py`: `hand_side_label()` 函数

**机制**：每 episode 随机分配 hand_side（0/1），训练时 50/50。

---

### 首次训练
- 基于 robosuite 参考的**逐手奖励**
- 举升门槛：0.45→**0.38**（更容易触发首次成功）
- 抓取奖励权重：1.0/2.0
- 从关节位置控制改为**差分 IK** 控制

## 2026-05-26

### 21:51 — 初始版本

**新建任务**：`Isaac-Lift-Cube-OpenArm-Bi-v0`

**首次训练配置**：
- 控制方式：关节位置（JointPositionActionCfg）
- 方块位置：(0.25, 0, 0.34)，平台支撑
- 奖励：双手独立 reaching + grasping
- 400 环境
- 1200 步 episode

**首次问题**：
- 方块抓不住（夹爪不会闭合）
- 方块掉落率 80%+
- 举升信号太稀疏

---

## 符号说明

- ✅ 已完成 / 已解决
- ⚠️ 已知问题
- ❌ 待修复

```
