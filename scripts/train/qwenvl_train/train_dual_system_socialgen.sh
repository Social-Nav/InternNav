#!/bin/bash
# Stage2 dual-system finetune on a custom SocialNav dataset.
# Single node, 8x A100 80G. Launched directly with torchrun (no SLURM).
#
# Trains System 1 (trajectory diffusion) + latent_queries on top of a frozen
# System 2. See trainer:78-122: tune_mm_* False freezes the whole VLM, then the
# 'nextdit' branch re-enables only the System 1 modules plus latent_queries
# (prompt tuning, paper sec 3.2).
#
# Prereqs:
#   1. Stage1 must be finished. --model_name_or_path points at its output dir,
#      whose name must contain "internvla-n1-system2" so the trainer's string
#      dispatch (trainer:149) loads InternVLAN1ForCausalLM -- the only class
#      that carries the System 1 modules. The stage1 socialnav run already
#      produces checkpoints/InternVLA-N1-System2-SocialNav, which matches.
#   2. Depth-Anything-V2 ViT-S weights on disk for rgb_model:
#        depth_anything_v2_metric_hypersim_vits.pth  (see internvla_n1_arch.py:37)
#   3. socialnav_132cm_30_30 registered in internvla_n1_lerobot_dataset.py data_dict
#
# --pixel_goal_only True turns on trajectory supervision. This fork subsamples
# logged poses every TRAJ_ARC_INTERVAL (0.1 m) of arc instead of interpolating;
# upstream's >0.2236 m step filter dropped every frame of a 30 FPS capture and
# silently trained "stand still". On a new capture, dump one batch and confirm
# traj_poses.abs().sum() > 0.
#
# Must be run from the repo root; the trainer does a bare `import qwenvl_base`.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

deepspeed=scripts/train/qwenvl_train/zero2.json

# 8 GPUs x bs2 x ga2 = 32.
batch_size=2
grad_accum_steps=2
lr=1e-4

max_pixels=313600
min_pixels=3136

vln_datasets=socialnav_132cm_30_30

## Stage 2a
# run_name=InternVLA-N1-DualVLN-SocialGen
## Stage 2b
run_name=InternVLA-N1-DualVLN-SocialGen
output_dir=checkpoints/${run_name}

# Stage1 checkpoint. Must contain "internvla-n1-system2".
system2_ckpt=checkpoints/InternVLA-N1-System2-SocialGen

# Optional: dual-system checkpoint to seed System 1 from. Unset to start random.
## Stage 2a
# system1_ckpt=checkpoints/InternVLA-N1-DualVLN

## Stage 2b
system1_ckpt=checkpoints/InternVLA-N1-w-NavDP

# system1 options: nextdit_async, navdp_async, nextdit
# system1=nextdit_async
system1=nextdit_async

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 10 \
    --num_future_steps 10 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --system1 ${system1} \
    --system1_ckpt ${system1_ckpt} \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 10 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --run_name ${run_name} \
    --logging_dir ${output_dir}/tensorboard_logs \
    --report_to tensorboard
