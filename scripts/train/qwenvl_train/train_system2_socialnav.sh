#!/bin/bash
# Stage1 System2 finetune on a custom SocialNav dataset.
# Single node, 8x A100 80G. Launched directly with torchrun (no SLURM).
#
# Prereqs:
#   1. Official System2 weights, in a dir whose name contains "qwen2.5" and
#      NOT "internvla-n1-system2" (see trainer:149 string dispatch):
#        huggingface-cli download InternRobotics/InternVLA-N1-System2 \
#          --local-dir checkpoints/qwen2.5-vl-n1s2-base
#   2. traj_data/socialnav/<scene>/{meta,data,videos} with labels for 132cm_30deg:
#        python scripts/data/gen_pixel_goal_labels.py \
#          --data-path traj_data/socialnav --height 132 --pitch-1 30 --pitch-2 30 \
#          --hfov 90 --dry-run        # drop --dry-run once goal hit-rate looks sane
#   3. socialnav_132cm_30_30 registered in internvla_n1_lerobot_dataset.py data_dict
#
# Must be run from the repo root; the trainer does a bare `import qwenvl_base`.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

deepspeed=scripts/train/qwenvl_train/zero2.json
llm=checkpoints/qwen2.5-vl-n1s2-base

# 8 GPUs x bs2 x ga2 = effective batch 32.
# Official is 128 (64 GPUs x 2 x 1); 32 is deliberate here -- a small dataset
# needs more optimizer steps per epoch, and lr is scaled down to match.
batch_size=2
grad_accum_steps=2
lr=1e-5
vision_tower_lr=2e-6

max_pixels=313600
min_pixels=3136

vln_datasets=socialnav_132cm_30_30

run_name=InternVLA-N1-System2-SocialNav
output_dir=checkpoints/${run_name}

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only False \
    --system1 "none" \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 2.0 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --vision_tower_lr ${vision_tower_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 12 \
    --run_name ${run_name} \
    --report_to tensorboard
