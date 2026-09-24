# InternVLA-N1 训练可调参数

对应脚本：
- Stage1 System2：`scripts/train/qwenvl_train/train_system2.sh` → `internnav/trainer/internvla_n1_trainer.py`
- Stage2 双系统联合：`scripts/train/qwenvl_train/train_dual_system.sh` → 同一个 trainer
- IIGN/对话变体：`scripts/train/qwenvl_train/train_system2_vlln.sh` → `internnav/trainer/internvla_vlln_trainer.py`

参数定义见 `internnav/trainer/internvla_n1_argument.py`。

## 1. 数据与任务构造

| 参数 | 默认 | 作用 |
|---|---|---|
| `vln_dataset_use` | — | `internvla_n1_lerobot_dataset.py` 中 `data_dict` 的 key 列表，逗号分隔。后缀 `%30` = 随机采样该组 30% 样本（`:941`） |
| `iign_dataset_use` | — | 同上，但走 `VLLNDataset`（对话式 IIGN）。与 `vln_dataset_use` 可同时给，会 concat |
| `sample_step` | 4 | 按 episode 顺序每隔几帧产生一个候选起点；只影响候选帧集合，不决定最终训练顺序。训练时 Trainer 再用 `RandomSampler` 全局打乱 |
| `num_history` | 8 | 从当前帧之前的 `[0, start_frame-1]` 中用 `linspace` 确定性均匀抽取的历史帧上限（不含当前帧）。直接影响每样本图像数和显存 |
| `num_future_steps` | 4 | 仅用于无 pixel-goal 的 turn 候选：向前最多查看几帧收集非前进动作，遇到前进动作停止 |
| `turn_sample_repeat` | 1 | Stage1 中每条 turn 样本的使用次数；0=移除，1=原始配比，>1=过采样。仅在 `pixel_goal_only=False` 时生效 |
| `predict_step_num` | 32 | 轨迹重采样到的固定长度，System1 的输出维度（`:1124`）。Stage1/Stage2 必须一致 |
| `pixel_goal_only` | False | False 时额外加入 turn 样本和 5 倍复制的 stop 样本（`:938-940`）；True 只留 pixel-goal 样本 |
| `data_augmentation` | True | 开启 ColorJitter/Posterize/Sharpness/Autocontrast（trainer `:134-147`），仅作用于 RGB 主观测 |
| `resize_h` / `resize_w` | 384 | 上述 transform 末端的 Resize 尺寸。lookdown 图与深度图不走此变换 |
| `max_pixels` / `min_pixels` | 313600 / 3136 | 覆写 Qwen image processor 的动态分辨率上下界（`:324-327`），决定每图 visual token 数 |
| `max_dialog_turns` | 6 | 仅 vlln：单样本最多保留几轮对话 |
| `data_flatten` | False | 开启后换 packed attention + `FlattenedCollator` |
| `data_packing` | False | 不要开：`make_supervised_data_module_packed` 在仓库里没有定义（trainer `:214` 带 `noqa: F821`），开了必 NameError |

`height` / `pitch_1` / `pitch_2` 不是命令行参数，只能在 `data_dict` 里改：
`height` + `pitch_2` 拼出 parquet 列名 `setting = f'{height}cm_{pitch_2}deg'`，
`pitch_1` 决定 RGB 目录名，`pitch_2` 决定 lookdown / depth 目录名。

深度缩放在 dataloader 里硬编码为 1000（`:1025`），`gen_pixel_goal_labels.py --depth-scale` 只影响离线标签生成。

### 1.1 实际抽样顺序

Stage1/Stage2 都不是在训练 step 中按概率临时选择 episode 或帧。数据集初始化时会：

1. 按 episode 原始顺序读取 `action`、`goal`、`relative_goal_frame_id` 和 `pose`。
2. 按 `sample_step` 检查候选起点，例如 `sample_step=30` 时检查 `0, 30, 60, ...`。
3. 将候选起点分类为 pixel-goal、turn，或直接丢弃；每条 episode/instruction 另外加入一个 STOP 样本。
4. 组成最终列表：

```text
pixel_goal_list
+ turn_list x turn_sample_repeat
+ stop_list x 5       # 仅 pixel_goal_only=False
```

随后 Hugging Face `Trainer` 使用 `RandomSampler` 对整个列表的索引做随机排列。`CombinedDataset(shuffle=False)` 只表示数据集自身不预先打乱，不会关闭 Trainer 的随机 sampler。多卡训练时，Accelerate 再将随机 batch 分片给各进程。

因此，`sample_step` 控制“哪些帧进入候选集合”，`RandomSampler` 控制“进入模型的顺序”。重复 turn 样本通常会被打散，但不保证固定 batch 配比，也不保证同一原始 turn 的副本一定落在不同 batch。

### 1.2 历史帧的精确读取

历史帧使用：

```python
np.unique(np.linspace(0, start_frame_id - 1, num_history, dtype=np.int32))
```

这是确定性的均匀抽样，不是随机抽样，也不是遍历读取。实际读取数量是：

```text
min(num_history, 当前帧之前的可用帧数)
```

短历史可能因为帧数不足或整数索引去重而少于 `num_history`。当前帧始终额外读取，不计入 `num_history`。

当前按需读取逻辑：

| 阶段 | 主视角历史帧 | 当前主视角 | 未来主视角 | 俯视图/depth |
|---|---|---|---|---|
| Stage1 | 只读均匀选中的历史帧 | 读取 | 不读 | pixel-goal 只读当前俯视图；不读 depth |
| Stage2 | 只读均匀选中的历史帧 | 读取 | 不读 | 读取当前俯视图，以及最终轨迹抽样时刻的俯视图和 depth |

Stage2 不能删除历史主视角，因为冻结的 System2 仍使用历史观测生成 latent condition；可以删除的是未选中的历史帧和未进入最终轨迹抽样的未来帧。

## 2. 模型结构

| 参数 | Stage1 | Stage2 | 作用 |
|---|---|---|---|
| `model_name_or_path` | `Qwen/Qwen2.5-VL-7B-Instruct` | `checkpoints/InternVLA-N1-System2` | 路径小写含 `internvla-n1-system2` 才会走 `InternVLAN1ForCausalLM` 并建 System1（trainer `:149`），否则静默退化成纯 VLM |
| `system1` | `none` | `nextdit_async` | `nextdit`=DiT 轨迹头；`+async` 额外建 DepthAnythingV2 + MemoryEncoder + QFormer；`navdp_async`=NavDP 头。填 `navdp`（无 async）会缺属性崩（arch `:162-164`） |
| `n_query` | 4 | 4 | latent query 数，System2→System1 的接口宽度。两阶段必须一致，否则 ckpt 形状不匹配 |
| `tune_mm_vision` | True | False | ViT 是否解冻 |
| `tune_mm_mlp` | True | False | visual merger 是否解冻 |
| `tune_mm_llm` | True | False | LLM backbone + lm_head 是否解冻 |

三个 `tune_*` 均为 False 时，`set_model`（trainer `:78-122`）只解冻 System1 那组模块：
`action_encoder` / `action_decoder` / `traj_dit` / `cond_projector` / `memory_encoder` / `rgb_resampler` / `rgb_model` / `latent_queries`。这就是 Stage2 的“冻 VLM 训 System1”。

## 3. 优化与运行时

| 参数 | Stage1 | Stage2 | 作用 |
|---|---|---|---|
| `learning_rate` | 2e-5 | 1e-4 | 主 lr。Stage2 只训随机初始化的新模块，所以更大 |
| `vision_tower_lr` | 5e-6 | 不设 | ViT 单独 lr，非 0 时才拆参数组（`qwenvl_base.py:186-243`）。Stage2 冻了 ViT 所以无意义 |
| `mm_projector_lr` | 不设 | 不设 | 同理，projector 单独 lr |
| `per_device_train_batch_size` | 2 | 2 | 单卡 micro batch |
| `gradient_accumulation_steps` | 1 | 1 | 等效 batch = 卡数 × bs × ga |
| `num_train_epochs` | 2.0 | 3.0 | vlln 变体是 6.0 |
| `lr_scheduler_type` | `cosine` | `cosine_with_min_lr` | Stage2 配 `--lr_scheduler_kwargs '{"min_lr":1e-05}'` 防止后期 lr 归零 |
| `warmup_ratio` | 0.003 | 0.003 | warmup 占总步数比例 |
| `weight_decay` / `max_grad_norm` | 0 / 1 | 同 | 不做权重衰减，梯度裁剪到 1 |
| `model_max_length` | 8192 | 8192 | token 上限，需容纳 `num_history` 张图的 visual token |
| `gradient_checkpointing` | True | True | 省显存换约 30% 速度 |
| `dataloader_num_workers` | 8 | 8 | 数据集构造阶段要逐个读 parquet，首次启动较慢 |
| `save_steps` / `save_total_limit` | 5000 / 5 | 同 | `output_dir` 存在 `checkpoint-*` 会自动续训（trainer `:225`） |
| `deepspeed` | `zero2.json` | `zero2.json` | zero2=分片优化器；zero3=再分片参数；zero3_offload=offload 到 CPU |
| `report_to` | `wandb` | `wandb` | 离线环境改 `none`，否则卡在等 API key |

其余 `TrainingArguments` 字段（`bf16`、`eval_strategy`、`logging_steps`、`run_name`、`output_dir`）为 HF Trainer 标准语义。
