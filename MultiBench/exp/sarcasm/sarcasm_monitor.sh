#!/bin/bash
# source /home/zz/miniconda3/etc/profile.d/conda.sh
source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1
dataset="sarcasm"
modality_list=("xy" "x" "y")
lr="1e-4"
zdim_list=(40)
num_epochs=30
step_k_list=(-1)
n_seeds=3
pos_embd_list=(true)
pos_learnable_list=(true)

cd /home/zheng/zheng/Gradient-Unpaired-Multimodal-Learning/MultiBench || exit 1

export CUDA_VISIBLE_DEVICES=$GPU_ID
for modality in "${modality_list[@]}"; do
  for zdim in "${zdim_list[@]}"; do
    for step_k in "${step_k_list[@]}"; do
      for pos_embd in "${pos_embd_list[@]}"; do
        for pos_learnable in "${pos_learnable_list[@]}"; do
          cmd=(
            python main.py -d
            --ds_name "$dataset"
            --modality "$modality"
            --lr "$lr"
            --zdim "$zdim"
            --num_epochs "$num_epochs"
            --step_k "$step_k"
            --n_seeds "$n_seeds"
            --train_jsonl
            --gpop_monitor
            --gpop_monitor_beta 0.9
            --gpop_monitor_enable_common_block
          )

          if [ "$pos_embd" = "true" ]; then
            cmd+=(--pos_embd)
          fi
          if [ "$pos_learnable" = "true" ]; then
            cmd+=(--pos_learnable)
          fi

          "${cmd[@]}"
        done
      done
    done
  done
done
