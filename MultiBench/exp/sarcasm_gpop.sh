#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1
dataset="sarcasm"
modality="xy"
lr="1e-4"
zdim_list=(40 300)
num_epochs=30
step_k_list=(-1 15 20)
n_seeds=3
pos_embd_list=(true false)
pos_learnable_list=(true false)
beta_list=(0.9)
gpop_weight=(
  "loss_x=0.0,loss_y=1.0"
  "loss_x=0.1,loss_y=0.9"
  "loss_x=0.2,loss_y=0.8"
  "loss_x=0.3,loss_y=0.7"
  "loss_x=0.4,loss_y=0.6"
  "loss_x=0.5,loss_y=0.5"
  "loss_x=0.6,loss_y=0.4"
  "loss_x=0.7,loss_y=0.3"
  "loss_x=0.8,loss_y=0.2"
  "loss_x=0.9,loss_y=0.1"
)

cd /home/zz/zheng/Unpaired-Multimodal-Learning/MultiBench || exit 1


export CUDA_VISIBLE_DEVICES=$GPU_ID
for zdim in "${zdim_list[@]}"; do
  for step_k in "${step_k_list[@]}"; do
    for pos_embd in "${pos_embd_list[@]}"; do
      for pos_learnable in "${pos_learnable_list[@]}"; do
        for beta in "${beta_list[@]}"; do
          for gpop_weight in "${gpop_weight[@]}"; do
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
              --results_dir "./gpop_results"
              --gpop_monitor
              --gpop_monitor_beta "$beta"
              --gpop_monitor_enable_common_block
              --gpop
              --gpop_ref_build_kind "weighted_mean"
              --gpop_ema_beta "$beta"
              --gpop_edit_kind "project"
              --gpop_weights "$gpop_weight"
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
done
