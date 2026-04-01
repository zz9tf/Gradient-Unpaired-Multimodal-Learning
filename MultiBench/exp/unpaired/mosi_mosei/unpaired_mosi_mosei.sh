#!/bin/bash
source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1
dataset1_list=("mosi")
dataset2_list=("mosei")
modality_list=("xy" "x" "y")
lr="1e-4"
zdim_list=(40 300)
num_epochs=30
step_k_list=(-1 10 20)
n_seeds=3
pos_embd_list=(true false)
pos_learnable_list=(true false)

cd /home/zheng/zheng/Gradient-Unpaired-Multimodal-Learning/MultiBench || exit 1

export CUDA_VISIBLE_DEVICES=$GPU_ID
for dataset1 in "${dataset1_list[@]}"; do
  for dataset2 in "${dataset2_list[@]}"; do
    for modality in "${modality_list[@]}"; do
      for zdim in "${zdim_list[@]}"; do
        for step_k in "${step_k_list[@]}"; do
          for pos_embd in "${pos_embd_list[@]}"; do
            for pos_learnable in "${pos_learnable_list[@]}"; do
              if [ "$dataset1" = "$dataset2" ]; then
                continue;
              fi
              cmd=(
                python main.py -d
                --dataset1 "$dataset1"
                --dataset2 "$dataset2"
                --modality "$modality"
                --lr "$lr"
                --zdim "$zdim"
                --num_epochs "$num_epochs"
                --step_k "$step_k"
                --n_seeds "$n_seeds"
                --train_jsonl
                --results_dir "./results/unpaired_$dataset1_$dataset2"
              )

              if [ "$pos_embd" = "true" ]; then
                cmd+=(--pos_embd)
              fi
              if [ "$pos_learnable" = "true" ]; then
                cmd+=(--pos_learnable)
              fi

              "${cmd[@]}"
              # done
            done
          done
        done
      done
    done
  done
done
