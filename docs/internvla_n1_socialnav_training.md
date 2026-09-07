# InternVLA-N1 System2 微调流程（8×A100 80G / SocialNav）

面向已经对齐 InternNav LeRobot 格式的自有数据集。数据转换与标签生成（`gen_pixel_goal_labels.py` 等）不在本文范围内。

- 数据集 setting：`height=132`, `pitch_1=30`, `pitch_2=30` → 列名后缀 `132cm_30deg`
- 起点权重：官方 `InternRobotics/InternVLA-N1-System2`
- 训练脚本：`scripts/train/qwenvl_train/train_system2_socialnav.sh`

---

## 0. 目录布局

所有路径都是**相对仓库根目录**的（`data_dict` 里的 `data_path`、脚本里的 `checkpoints/...`、
以及 `internvla_n1_arch.py:7` 硬编码的 `MODEL_PATH_TO = "checkpoints"`）。
训练必须 `cd` 到仓库根目录再启动。

```
InternNav/                                    ← 所有命令的 CWD
├── traj_data/
│   └── socialnav/                            ← 你的数据集
│       ├── scene_001/
│       │   ├── meta/episodes.jsonl
│       │   ├── data/chunk-000/episode_000000.parquet
│       │   └── videos/chunk-000/
│       │       ├── observation.images.rgb.132cm_30deg/episode_000000_0.jpg
│       │       └── observation.images.depth.132cm_30deg/episode_000000_0.png
│       └── scene_002/ ...
├── checkpoints/
│   ├── qwen2.5-vl-n1s2-base/                 ← 官方 System2 权重（微调起点）
│   └── InternVLA-N1-System2-SocialNav/       ← 训练输出（自动创建）
└── scripts/train/qwenvl_train/train_system2_socialnav.sh
```

数据或权重在别的挂载点时用软链接，不要改代码里的相对路径：

```bash
ln -s /mnt/data/my_socialnav        traj_data/socialnav
ln -s /mnt/models/internvla_ckpts   checkpoints
```

### 目录命名的硬约束

`internvla_n1_trainer.py:149-181` 用**路径字符串**分派模型类，三个分支互斥：

| 路径小写后 | 命中 | 结果 |
|---|---|---|
| 含 `internvla-n1-system2` | `InternVLAN1ForCausalLM` | Stage1（`system1="none"`）会 `raise NotImplementedError` |
| 含 `qwen2.5`，不含上者 | `Qwen2_5_VLForConditionalGeneration` | ✅ Stage1 正确路径 |
| 两者都不含 | `Qwen2VLForConditionalGeneration` | ❌ 按 Qwen2-VL 加载，rope 用错 |

所以起点权重目录必须叫 `qwen2.5-vl-n1s2-base` 这类名字，**不能**直接传 HF hub id
`InternRobotics/InternVLA-N1-System2`（它本身含 `internvla-n1-system2`，必崩）。

反过来，Stage2 的 `system2_ckpt` 要求路径**含** `internvla-n1-system2`。
本文的输出目录 `InternVLA-N1-System2-SocialNav` 已满足，接 Stage2 时无需改名。

评测配置 `scripts/eval/configs/habitat_s2_cfg.py:9` 写死了 `checkpoints/InternVLA-N1-System2`，
需要时做个软链接即可，磁盘上只有一份权重。

---

## 1. 环境与权重

```bash
cd /path/to/InternNav

pip install -r requirements/model_requirements.txt
pip install -e .

python -c "import torch, transformers, deepspeed; \
print(torch.__version__, torch.cuda.device_count(), transformers.__version__)"
# 期望: 2.x  8  4.5x.x
```

下载官方 System2 权重：

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com          # 国内镜像，可选

hf download InternRobotics/InternVLA-N1-System2 \
  --local-dir checkpoints/qwen2.5-vl-n1s2-base

python -c "
import json; c=json.load(open('checkpoints/qwen2.5-vl-n1s2-base/config.json'))
print('arch:', c.get('architectures'))
print('hidden:', c.get('hidden_size') or c.get('text_config',{}).get('hidden_size'))"
```

`architectures` 应为 `["Qwen2_5_VLForConditionalGeneration"]`，hidden_size 应为 **3584**（7B）。
若 hidden_size 是 2048（3B），后续接 Stage2 会与 `cond_projector` 的硬编码 3584 维度冲突。

不需要 wandb 账号：脚本用 `--report_to tensorboard`，`tensorboard` 已在
`requirements/model_requirements.txt` 中。

---

## 2. 注册数据集

已在 `internnav/dataset/internvla_n1_lerobot_dataset.py` 的 `data_dict` 顶部加入：

```python
SOCIALNAV_132CM_30_30 = {
    "data_path": "traj_data/socialnav",
    "height": 132,
    "pitch_1": 30,
    "pitch_2": 30,
}
```

三个字段的用途（不是命令行参数，只能在这里改）：

- `height` + `pitch_2` → `setting = "132cm_30deg"`，决定读哪几列 parquet（`:850`）
- `pitch_1` → RGB 目录名 `observation.images.rgb.132cm_30deg`（`:1015`）
- `pitch_2` → lookdown 图与 depth 目录名（`:1018-1021`）

本例 `pitch_1 == pitch_2`，`replace` 是恒等操作，lookdown 与主观测是同一张图。
这与官方 `r2r_60cm_30_30` 的用法一致，是受支持的配置。

---

## 3. 启动前校验（关键）


### 3.1 为什么必须校验

`get_annotations_from_lerobot_data`（`:793-799`）在列名缺失时**只打印一行 warning**，
不抛异常也不填默认值，随后第 802 行引用未赋值的 `ep_poses`：

- 首个 episode 就缺列 → `UnboundLocalError` → 被 `:816` 的 `except Exception` 吞掉 → **整个 scene 静默丢弃**
- 部分 episode 缺列 → `ep_poses` 残留上一 episode 的值 → **数据静默错配，训练照跑**

两种情况都不会让训练崩，只会让你训出垃圾。所以先跑校验。

### 3.2 校验脚本

```bash
cd /path/to/InternNav

python - <<'EOF'
import os, json, glob
import pyarrow.parquet as pq

ROOT, H, P1, P2 = "traj_data/socialnav", 132, 30, 30
setting = f"{H}cm_{P2}deg"
need = [f"pose.{setting}", f"goal.{setting}", f"relative_goal_frame_id.{setting}", "action"]

scenes = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
print(f"setting = {setting} | {len(scenes)} scenes\n")

n_ep = n_bad = n_frames = n_goals = 0
for s in scenes:
    sp = os.path.join(ROOT, s)
    mp = os.path.join(sp, "meta", "episodes.jsonl")
    if not os.path.exists(mp):
        print(f"  [{s}] 缺 meta/episodes.jsonl"); n_bad += 1; continue

    eps = [json.loads(l) for l in open(mp)]
    for ep in eps:
        i = ep["episode_index"]
        pf = os.path.join(sp, "data", f"chunk-{i//1000:03d}", f"episode_{i:06d}.parquet")
        if not os.path.exists(pf):
            print(f"  [{s}] ep{i}: parquet 缺失"); n_bad += 1; continue

        df = pq.read_table(pf).to_pandas()
        missing = [c for c in need if c not in df.columns]
        if missing:
            print(f"  [{s}] ep{i}: 缺列 {missing}"); n_bad += 1; continue
        if len(df) != ep["length"]:
            print(f"  [{s}] ep{i}: length={ep['length']} 但 parquet {len(df)} 行"); n_bad += 1; continue

        goals = df[f"relative_goal_frame_id.{setting}"]
        n_frames += len(df); n_goals += int((goals >= 0).sum()); n_ep += 1

        # RGB / depth 目录
        vd = os.path.join(sp, "videos", f"chunk-{i//1000:03d}")
        for kind, p in (("rgb", P1), ("depth", P2)):
            d = os.path.join(vd, f"observation.images.{kind}.{H}cm_{p}deg")
            if not os.path.isdir(d):
                print(f"  [{s}] ep{i}: 缺目录 {d}"); n_bad += 1

        n_instr = len(eps[0]["tasks"][0].split("<INSTRUCTION_SEP>")) if eps else 0

print(f"\n可用 episode : {n_ep}   问题 episode: {n_bad}")
print(f"总帧数       : {n_frames}")
print(f"pixel goal   : {n_goals} ({n_goals/max(n_frames,1)*100:.1f}%)")
print(f"每条指令数   : {n_instr}")
print(f"\n预估样本数   ≈ {n_ep * n_frames // max(n_ep,1) // 4 * n_instr}  (sample_step=4)")
EOF
```

判据：

| 指标 | 期望 | 不达标怎么办 |
|---|---|---|
| 问题 episode | **0** | 缺列 → 重新生成标签；缺目录 → 检查 `pitch_1`/`pitch_2` 是否与目录名一致 |
| pixel goal 占比 | **>60%** | <50% 说明标签生成时 hfov/depth_scale/flip-v 有误，重训也学不到东西 |
| 每条指令数 | ≥1 | 为 1 正常；R2R 官方是 3（`<INSTRUCTION_SEP>` 分隔），语言泛化更好 |

### 3.3 确认注册生效

```bash
python -c "
from internnav.dataset.internvla_n1_lerobot_dataset import data_dict
print(data_dict['socialnav_132cm_30_30'])"
```

---

## 4. 训练参数

脚本：`scripts/train/qwenvl_train/train_system2_socialnav.sh`

| 参数 | 值 | 相对官方脚本的改动理由 |
|---|---|---|
| 启动方式 | `torchrun --standalone --nnodes=1 --nproc_per_node=8` | 官方是 `srun torchrun` + `$SLURM_*`，单机无 SLURM 时变量为空起不来 |
| `model_name_or_path` | `checkpoints/qwen2.5-vl-n1s2-base` | 从官方权重微调，而非从 Qwen 从零训 |
| `gradient_accumulation_steps` | 2 | 8×2×2 = 等效 batch **32**。官方 128（64卡×2×1）；小数据集需要更多 optimizer step |
| `learning_rate` | 1e-5 | 官方 2e-5 是从 Qwen 起训；微调已收敛权重要减半 |
| `vision_tower_lr` | 2e-6 | 同比例下调 |
| `warmup_ratio` | 0.03 | 官方 0.003 是按几万 step 算的；小数据集总步数少，比例要提高 |
| `num_train_epochs` | 4.0 | 官方 2.0 对应大数据集 |
| `save_steps` | 500 | 官方 5000，小数据集根本到不了 |
| `report_to` | `tensorboard` | 免 wandb 账号与外网 |
| `dataloader_num_workers` | 12 | 8×12 = 96 进程，128 核够用 |

不建议改的：`per_device_train_batch_size=2`（80G 显存刚好）、`num_history=8`、
`predict_step_num=32`（接 Stage2 时必须与之一致）、`deepspeed=zero2.json`。

**显存预估（单卡 80G）**：参数 bf16 14G + 梯度 14G + ZeRO-2 分片优化器 ~10.5G
+ 激活（9 图/样本 × bs2）~25-35G ≈ **60-75G**。OOM 时优先降 `num_history` 到 6，
再考虑换 `zero3.json`。

---

## 5. 启动

```bash
cd /path/to/InternNav          # 必须；trainer 里 `import qwenvl_base` 是裸 import

tmux new -s train              # 防 SSH 断开
bash scripts/train/qwenvl_train/train_system2_socialnav.sh 2>&1 | tee train.log
# Ctrl+B D 脱离； tmux attach -t train 回来
```

---

## 6. 怎么确认"正常开始训练了"

按时间顺序会出现这 6 组输出。**任何一组缺失或数值异常都说明有问题**。

### ① 数据集配置回显（约 10 秒）

```
Loading datasets: [{'data_path': 'traj_data/socialnav', 'height': 132,
                    'pitch_1': 30, 'pitch_2': 30, 'sampling_rate': 1.0}]
```

来源 `:827`。`data_path` 不对就是软链接错了。

### ② 样本统计 —— **最关键的一行**

```
1523 8734 412
```

来源 `:937`，三个数依次是 `len(turn_list) len(pixel_goal_list) len(stop_list)`。

- **中间那个（pixel_goal）为 0 → 立刻停**。说明列名不匹配或 goal 全是 -1，
  继续跑只是在浪费卡时。回到第 3 节校验。
- 第三个（stop）应约等于 `episode 数 × 指令数`
- 最终训练样本数 = `pixel_goal + turn + stop×5`（`pixel_goal_only=False` 时，`:938-940`）

这一步要遍历全部 parquet，数据多时可能几分钟无输出，属正常。

### ③ 可训练模块清单

```
Vision Module - Attention Blocks:
Trainable Block Indices: [0, 1, 2, ..., 31]
Merger Module Trainable: True
LLM Module - Embed Tokens Trainable: True
LLM Module - Trainable Layer Indices: [0, 1, ..., 27]
LLM Module - Non-Trainable Layer Indices: None
```

来源 `qwenvl_base.py:128-177`（monkey-patch 上去的）。
Stage1 三个 `tune_*` 全 True，所以应该**全部可训练、Non-Trainable 为 None**。
若出现大量 Non-Trainable，说明参数传错了。

### ④ 参数表

`tabulate` 打出的 idx/name/shape/trainable 全表（trainer `:220-224`），很长，扫一眼 trainable 列即可。

### ⑤ DeepSpeed 初始化

```
[INFO] DeepSpeed info: version=0.x.x
[INFO] Using /usr/bin/ld ... Loading extension module fused_adam...
```

首次运行编译 CUDA kernel 需 2-3 分钟，之后走缓存。这一步卡住不动是正常的。

### ⑥ 训练步开始

```
  0%|          | 0/124 [00:00<?, ?it/s]
{'loss': 1.8234, 'grad_norm': 1.2456, 'learning_rate': 1e-06, 'epoch': 0.03}
{'loss': 1.7891, 'grad_norm': 1.1023, 'learning_rate': 2e-06, 'epoch': 0.06}
```

`logging_steps=1`，每步一行。看到这个就是真的在训了。

**健康区间**：

| 指标 | 正常 | 异常信号 |
|---|---|---|
| `loss` 初值 | 1.5-3.0 | >5 说明权重没加载对；<0.5 说明数据太简单或有泄漏 |
| `loss` 趋势 | 缓降到 0.5-1.5 | 完全不降 → lr 太小或数据有问题；断崖式降到 ~0 → 过拟合 |
| `grad_norm` | 0.5-2.0 | 持续 >10 → 不稳定，降 lr；恒为 0 → 参数没解冻 |
| `learning_rate` | warmup 升，之后 cosine 降 | 恒为 0 → warmup_ratio 配错 |

### 用 nvidia-smi 交叉验证

```bash
watch -n 2 nvidia-smi
```

8 张卡都应在 60-75GB 显存、90%+ 利用率。若利用率长期在 20-30% 反复波动，
是数据加载瓶颈，调大 `dataloader_num_workers`。

---

## 7. 监控曲线

```bash
tensorboard --logdir checkpoints/InternVLA-N1-System2-SocialNav/runs \
            --port 6006 --bind_all
```

浏览器开 `http://<服务器IP>:6006`；端口不通就本地建隧道：

```bash
ssh -L 6006:localhost:6006 user@server     # 之后开 http://localhost:6006
```

SCALARS 页有 `train/loss`、`train/grad_norm`、`train/learning_rate`、`train/epoch`。

**没有验证集曲线**：脚本 `--eval_strategy "no"`，且 `make_supervised_data_module`
返回 `eval_dataset=None`（`:1381`）。判断是否过拟合只能靠：

1. 延长 `num_train_epochs` 续训，看 loss 是否还降（`output_dir` 有 `checkpoint-*` 会自动续训，trainer `:225`）
2. 拿 ckpt 跑 `scripts/eval/bash/eval_system2.sh`（需装 habitat-sim）
3. `scripts/notebooks/inference_only_demo.ipynb` 改 model_path 看几个样例输出

---

## 8. 产出与后续

```
checkpoints/InternVLA-N1-System2-SocialNav/
├── checkpoint-500/     ← 完整 HF 模型，可直接 from_pretrained
├── checkpoint-1000/
└── ...                 ← save_total_limit=5，只留最新 5 个
```

接 Stage2 双系统联合训练时：

- `system2_ckpt=checkpoints/InternVLA-N1-System2-SocialNav`（名字已含 `internvla-n1-system2`，直接可用）
- `system1=nextdit_async`，需下载 `depth_anything_v2_metric_hypersim_vits.pth` 放到 `checkpoints/`
- `tune_mm_vision/mlp/llm` 全设 `False`，`pixel_goal_only True`，`lr` 提到 1e-4
- `predict_step_num` 必须与 Stage1 一致（32）

---

## 9. 排错速查

| 症状 | 原因 | 处理 |
|---|---|---|
| `ImportError: qwenvl_base` | CWD 不在仓库根，或用了 `python -m` | `cd` 到根目录，按文件路径调用 |
| `KeyError: 'socialnav_132cm_30_30'` | `data_dict` 未注册 | 见第 2 节 |
| ② 中 pixel_goal 为 0 | 列名不匹配 / goal 全 -1 | 第 3 节校验 |
| `Error processing scene X` | 该 scene 被静默丢弃（多为缺列） | 第 3 节校验，别忽略这行 |
| `NotImplementedError` in `initialize_vision_modules` | 权重目录名含 `internvla-n1-system2` 但 `system1="none"` | 改用 `qwen2.5-vl-n1s2-base` 这类目录名 |
| `CUDA OOM` | 激活过大 | `--num_history 6`，或换 `zero3.json` |
| 卡在 `wandb: Logging into wandb.ai` | `--report_to wandb` 且无 API key | 用 `tensorboard`；或 `export WANDB_MODE=offline` |
| GPU 利用率忽高忽低 | 数据加载瓶颈 | 调大 `dataloader_num_workers` |
| loss 断崖降到近 0 | 数据量太小，过拟合 | 减 epoch；或 `tune_mm_llm False` 只训 merger |
