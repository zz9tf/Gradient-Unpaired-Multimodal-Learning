#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1
dataset="mimic"
modality="xy"
lr="1e-4"
zdim_list=(40 300)
num_epochs=30
step_k_list=(-1 15 20)
n_seeds=3
pos_embd_list=(true false)
pos_learnable_list=(true false)

cd /home/zz/zheng/Unpaired-Multimodal-Learning/MultiBench || exit 1

export CUDA_VISIBLE_DEVICES=$GPU_ID
for zdim in "${zdim_list[@]}"; do
  for step_k in "${step_k_list[@]}"; do
    for pos_embd in "${pos_embd_list[@]}"; do
      for pos_learnable in "${pos_learnable_list[@]}"; do
        cmd=(
          python main.py -d
          --dataset1 "$dataset"
          --dataset2 "$dataset"
          --modality "$modality"
          --lr "$lr"
          --zdim "$zdim"
          --num_epochs "$num_epochs"
          --step_k "$step_k"
          --n_seeds "$n_seeds"
          --results_dir "./results/mimic"
          --train_jsonl
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