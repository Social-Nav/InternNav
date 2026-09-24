# InternVLA-N1 数据格式与预处理流程

本文说明 InternVLA-N1 训练所需的原始数据格式，以及 Stage1、Stage2 真正使用的数据和预处理流程。

## 1. 总体流程

```text
原始导航数据
  -> 转换为 LeRobot episode 格式
  -> 补充 pose、pixel goal 和 depth 数据
  -> NavPixelGoalDataset 按时间点生成训练样本
  -> Stage1：训练 System2 的视觉语言输出
  -> Stage2：冻结 System2，训练 System1 的连续轨迹预测
```

Stage1 和 Stage2 使用同一个数据集类：

```text
internnav/dataset/internvla_n1_lerobot_dataset.py
```

两阶段的主要区别由 `pixel_goal_only` 控制：

| 阶段 | `pixel_goal_only` | 使用的样本 |
|---|---:|---|
| Stage1 | `False` | pixel-goal、转向、STOP |
| Stage2 | `True` | 仅 pixel-goal，并生成连续轨迹监督 |

Stage1 还可以通过 `turn_sample_repeat` 控制每条转向样本重复进入训练集的次数。

## 2. 磁盘上的原始数据

以当前 SocialGen 配置为例：

```text
traj_data/social_gen/grscenes/
├── scene_001/
│   ├── meta/
│   │   └── episodes.jsonl
│   ├── data/
│   │   └── chunk-000/
│   │       └── episode_000000.parquet
│   └── videos/
│       └── chunk-000/
│           ├── observation.images.rgb.132cm_30deg/
│           │   ├── episode_000000_0.jpg
│           │   └── ...
│           └── observation.images.depth.132cm_30deg/
│               ├── episode_000000_0.png
│               └── ...
└── scene_002/
    └── ...
```

数据集注册配置为：

```python
SOCIALNAV_132CM_30_30 = {
    "data_path": "traj_data/social_gen/grscenes",
    "height": 132,
    "pitch_1": 30,
    "pitch_2": 30,
}
```

- `height + pitch_2` 决定 parquet 标签列后缀，本例为 `132cm_30deg`。
- `pitch_1` 决定主 RGB 观测目录。
- `pitch_2` 决定俯视 RGB、depth 目录及轨迹标签使用的相机角度。

### 2.1 `episodes.jsonl`

每行表示一个 episode：

```json
{
  "episode_index": 0,
  "tasks": ["Walk forward and turn left at the table."],
  "length": 120
}
```

训练实际读取：

| 字段 | 用途 |
|---|---|
| `episode_index` | 定位 parquet 和图像文件 |
| `tasks[0]` | 导航指令 |
| `length` | 校验 episode 帧数 |

一条 episode 可以包含多条指令，使用 `<INSTRUCTION_SEP>` 分隔。加载时每条指令会展开为一个独立样本来源。

### 2.2 Parquet

Parquet 每行对应一帧，训练至少需要以下列：

| 列名 | 内容 | 主要用途 |
|---|---|---|
| `action` | 离散导航动作 | 构造转向和 STOP 样本 |
| `pose.132cm_30deg` | 每帧相机/机器人位姿 | Stage2 生成连续轨迹 |
| `goal.132cm_30deg` | pixel-goal 坐标 | 生成 waypoint 文本标签 |
| `relative_goal_frame_id.132cm_30deg` | 当前帧到目标帧的相对长度；无目标时为 `-1` | 筛选 pixel-goal 样本并确定轨迹区间 |

离散动作的文本映射为：

| 动作值 | 文本 |
|---:|---|
| `0` | `STOP` |
| `1` | `↑` |
| `2` | `←` |
| `3` | `→` |
| `5` | `↓` |

### 2.3 图像与深度

图像文件名必须与 episode 和帧号对应：

```text
episode_{episode_index:06d}_{frame_index}.jpg
episode_{episode_index:06d}_{frame_index}.png
```

例如：

```text
episode_000000_37.jpg
episode_000000_37.png
```

基础转换器 `scripts/dataset_converters/vlnce2lerobot.py` 主要生成 RGB、`action` 和 LeRobot 元数据，不会自动补齐以下内容：

```text
pose.<setting>
goal.<setting>
relative_goal_frame_id.<setting>
depth 图像
```

因此，基础格式转换后还必须运行对应的位姿、pixel-goal 和 depth 数据生成流程，才能用于完整的 Stage1/Stage2 训练。

## 3. 通用样本构造

### 3.1 动作对齐

加载后先对动作序列做一帧偏移，并在结尾补 STOP：

```python
actions = original_actions[1:] + [0]
```

### 3.2 时间采样

数据集初始化时，代码按 episode 的原始顺序处理数据，并每隔 `sample_step` 帧创建一个候选起点。例如 `sample_step=10` 时，候选起点为：

```text
0, 10, 20, 30, ...
```

这里的顺序只用于**构造候选样本列表**，不代表最终训练顺序。所有 episode 都处理完后，列表会拼成：

```text
pixel_goal_list
+ turn_list x turn_sample_repeat
+ stop_list x 5
```

训练时 Hugging Face `Trainer` 使用 `RandomSampler` 对这个完整列表的索引做全局随机排列，再分发到多张 GPU。因此：

- `sample_step` 决定哪些帧可以成为训练样本；
- `RandomSampler` 决定样本进入模型的先后顺序；
- 没有被 `sample_step` 选中的帧不会因为打乱而进入训练；
- `turn_sample_repeat` 只是增加 turn 样本在整个 epoch 中的出现次数，不保证每个 batch 有固定的 turn 比例。

### 3.3 三类样本

**Pixel-goal 样本**

当 `relative_goal_frame_id != -1` 且目标长度至少为 3 帧时创建。样本包含导航指令、历史图像、当前图像、目标坐标及从当前帧到目标帧的 pose 区间。

**Turn 样本**

当没有有效 pixel goal，且当前动作为左转、右转等非前进动作时创建。最多向后收集 `num_future_steps` 个连续转向动作，遇到前进动作停止。

**STOP 样本**

每条 episode 在最后一帧创建一个 STOP 样本。Stage1 中该样本会复制 5 次，以提高 STOP 的训练占比。

Stage1 的最终样本数为：

```text
pixel-goal 数 + turn 数 x turn_sample_repeat + STOP 数 x 5
```

`turn_sample_repeat=1` 保持原始配比，设为 `0` 会移除 turn 样本，设为大于 1 的整数会重复使用 turn 样本。该参数只在 `pixel_goal_only=False` 时生效。

### 3.4 历史观测

对于当前帧之前的全部历史帧，代码通过确定性的 `linspace` 均匀选取最多 `num_history` 帧：

```python
history_id = np.unique(
    np.linspace(0, start_frame_id - 1, num_history, dtype=np.int32)
).tolist()
```

例如：

```text
start_frame_id=300, num_history=8
历史范围：0..299
大致抽取：[0, 42, 85, 128, 170, 213, 256, 299]
```

该抽样是均匀、确定性的，不是随机抽样。`num_history` 表示历史帧数量上限，不包含当前帧；历史长度小于 `num_history` 时，实际数量会少于该值。当前帧会额外读取一次。

Stage1 和 Stage2 都会将这些稀疏历史主视角 RGB 输入 System2。数据加载按需进行：未被选中的历史帧不会打开；Stage1 不读取未来轨迹帧，Stage2 只读取最终抽样进入 `traj_images` / `traj_depths` 的轨迹时刻。

## 4. Stage1：System2 数据

当前训练脚本：

```text
scripts/train/qwenvl_train/train_system2_socialgen.sh
```

关键配置：

```bash
--pixel_goal_only False
--system1 none
--sample_step 30
--num_history 8
--num_future_steps 10
--turn_sample_repeat 1
```

可在脚本顶部调整：

```bash
# 每条 turn 样本使用 3 次
turn_sample_repeat=3
```

### 4.1 真正使用的数据

| 数据 | 是否使用 | 用途 |
|---|---:|---|
| 导航指令 `tasks` | 是 | 构造用户 prompt |
| 主 RGB 和历史 RGB | 是 | System2 视觉输入 |
| 俯视 RGB | 是 | pixel-goal 样本的 waypoint 图像 |
| `action` | 是 | 构造转向、STOP 文本标签 |
| `goal.*` | 是 | 构造坐标文本标签 `x y` |
| `relative_goal_frame_id.*` | 是 | 判断目标是否有效并确定目标区间 |
| `pose.*` | 间接使用 | 截取 pixel-goal 对应区间，但不生成轨迹 tensor |
| depth | 否 | Stage1 不读取 depth 文件 |
| 连续轨迹 `traj_poses` | 否 | Stage1 不训练 System1 |

### 4.2 RGB 预处理

Stage1 的单个样本读取：

```text
最多 num_history 张历史主视角 RGB
+ 当前帧主视角 RGB
+ pixel-goal 样本的当前帧俯视 RGB
```

Stage1 不读取未来轨迹帧、未来俯视图或 depth。`num_history=8` 时，普通 turn/STOP 样本最多有 8 张历史图加 1 张当前图；pixel-goal 样本再额外加入 1 张当前俯视图。

主 RGB 图在开启 `data_augmentation` 时依次经过：

```text
ColorJitter
RandomPosterize
RandomAdjustSharpness
RandomAutocontrast
Resize(resize_h, resize_w)
Qwen image processor
```

当前脚本的 resize 尺寸为 `384 x 384`。俯视图会进入 Qwen image processor，但不经过上述主 RGB 数据增强链。

### 4.3 监督格式

Stage1 将导航目标转换成对话文本，用语言模型交叉熵训练。

Pixel-goal 样本：

```text
User:      导航指令 + 历史图像 + 当前图像
Assistant: ↓
User:      俯视图
Assistant: x y
```

Turn 样本：

```text
Assistant: ←←→
```

STOP 样本：

```text
Assistant: STOP
```

因此，Stage1 学习的是“根据指令和视觉观测输出 waypoint、转向或 STOP”，不使用 depth，也不计算连续轨迹 loss。

## 5. Stage2：System1 数据

当前训练脚本：

```text
scripts/train/qwenvl_train/train_dual_system_socialgen.sh
```

关键配置：

```bash
--pixel_goal_only True
--system1 nextdit_async
--sample_step 10
--predict_step_num 32
--num_history 8
```

Stage2 冻结 System2，训练 System1 及 `latent_queries`。由于 `pixel_goal_only=True`，Turn 和 STOP 样本不会进入最终训练集。

### 5.1 真正使用的数据

| 数据 | 是否使用 | 用途 |
|---|---:|---|
| 导航指令和历史 RGB | 是 | 由冻结的 System2 提取语言视觉条件 |
| 当前 RGB 和俯视 RGB | 是 | System2/System1 视觉输入 |
| `goal.*` | 是 | 保留 System2 waypoint 对话结构 |
| `relative_goal_frame_id.*` | 是 | 确定有效目标和轨迹区间 |
| `pose.*` | 是 | 生成固定长度连续轨迹标签 |
| depth | 是 | 生成 `traj_depths`；NavDP 会直接消费 |
| Turn、STOP 样本 | 否 | `pixel_goal_only=True` 时过滤 |

### 5.2 轨迹 RGB 和 depth 预处理

Stage2 仍然需要历史主视角 RGB，因为冻结的 System2 需要用历史观测和当前观测生成语言视觉条件。Stage2 的按需读取范围是：

```text
最多 num_history 张历史主视角 RGB
+ 当前帧主视角 RGB
+ 当前帧俯视 RGB
+ 最终轨迹抽样时刻的俯视 RGB
+ 最终轨迹抽样时刻的 depth
```

未被 `history_id` 选中的历史帧、历史俯视图、历史 depth、未来主视角 RGB 都不会读取。轨迹时刻默认每隔 2 帧抽取；超过 12 个时刻时增大间隔，因此实际进入 `traj_images` 和 `traj_depths` 的是最终抽样后的时刻，而不是整个目标区间的所有帧。

目标区间内的俯视 RGB 图：

```text
Resize 到 224 x 224
转换为 tensor
除以 255，归一化到 [0, 1]
```

Depth 图：

```text
使用最近邻插值 Resize 到 224 x 224
除以固定 depth_scale=1000
将大于 5.0 的值裁剪到 5.0
转换为 float32 tensor
```

轨迹时刻默认每 2 帧取一次；如果超过 12 个时刻，则自动增大间隔，将每个样本控制在约 12 个轨迹时刻以内。

### 5.3 连续轨迹标签预处理

对每个选中的轨迹时刻执行：

1. 将 camera pose 转换到 robot 坐标系。
2. 以当前帧为原点，得到相对 `[x, y, yaw]`。
3. 按 XY 平面累计弧长，每约 `0.1 m` 选择一个真实记录点。
4. 取 33 个位置点，转换成 32 个增量。
5. 输出 `[dx, dy, dyaw]`，其中 `dx`、`dy` 乘以 4。
6. 不足 32 步时补零，超过时截断。

最终单个时刻的轨迹标签形状为：

```text
[32, 3]
```

其中每一步为：

```text
[dx, dy, dyaw]
```

32 步、每步约 `0.1 m`，对应约 `3.2 m` 的监督范围。

### 5.4 Batch 格式

Stage2 的 collator 额外输出：

```python
{
    "traj_images":     [B, T, 224, 224, 3],
    "traj_depths":     [B, T, 224, 224],
    "traj_poses":      [B, T, 32, 3],
    "video_frame_num": [B],
    "t_s_pos":         list[int],
}
```

不同样本的 `T` 不同时，通过重复最后一帧图像、深度和轨迹补齐到 batch 内最大长度；`video_frame_num` 保存真实长度，计算 loss 时会屏蔽 padding。

### 5.5 不同 System1 的实际输入

| System1 | 使用的数据 |
|---|---|
| `nextdit_async` | System2 latent queries、轨迹 RGB、`traj_poses` |
| `navdp_async` | System2 latent queries、轨迹 RGB、轨迹 depth、`traj_poses` |

两者都使用 diffusion 类 MSE 监督连续轨迹，而不是 Stage1 的语言 token 交叉熵。

## 6. 两阶段对照

| 项目 | Stage1 | Stage2 |
|---|---|---|
| 训练对象 | System2 | System1 + latent queries |
| 样本类型 | pixel-goal + turn x `turn_sample_repeat` + STOP x5 | 仅 pixel-goal |
| 主 RGB/历史 RGB | 使用 | 使用 |
| 俯视 RGB | waypoint 图像 | System1 轨迹图像 |
| depth | 不读取 | 读取并预处理 |
| `action` | 生成 turn/STOP 文本 | 主要用于样本对齐和筛选 |
| `goal.*` | 生成 `x y` 文本 | 保留 waypoint 条件 |
| `relative_goal_frame_id.*` | 确定目标区间 | 确定轨迹区间 |
| `pose.*` | 间接用于样本构造 | 生成 `[32, 3]` 连续轨迹 |
| 监督目标 | waypoint、转向、STOP token | `[dx, dy, dyaw]` 轨迹 |
| 主要 loss | 语言模型交叉熵 | diffusion MSE |

两阶段的样本构造顺序都是：

```text
按 episode 顺序读取标签
  -> 按 sample_step 产生候选起点
  -> 分类为 pixel-goal / turn / STOP 或丢弃
  -> 按 turn_sample_repeat 和 STOP 重复规则组成固定 dataset
  -> Trainer 在训练时用 RandomSampler 全局打乱
```

## 7. 训练前最低检查项

1. `episodes.jsonl` 中的 `length` 必须等于对应 parquet 行数。
2. 每个 episode 必须包含 `action`、`pose.<setting>`、`goal.<setting>` 和 `relative_goal_frame_id.<setting>`。
3. RGB/depth 的帧号必须与 parquet 行号一一对应。
4. Stage1 的 `pixel_goal_list` 不能为 0，否则模型基本只会看到 turn/STOP。
5. Stage2 必须存在 depth 文件，并确认 `traj_poses.abs().sum() > 0`。
6. 数据生成时使用的深度单位必须与训练阶段固定的 `depth_scale=1000` 一致。
7. Stage1 和 Stage2 的 `predict_step_num` 应保持一致，当前为 32。

## 8. 一句话总结

```text
Stage1 把 instruction、RGB、goal 和离散 action 转成视觉语言对话，训练模型输出 waypoint、转向和 STOP；
Stage2 只保留有效 pixel-goal 区间，再用 RGB、depth 和 pose 生成固定 32 步连续轨迹，训练 System1。
```
