#!/bin/bash
source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1
dataset1_list=("mosi")
dataset2_list=("mosei")
modality_list=("xy" "y")
lr="1e-4"
zdim_list=(40)
num_epochs=30
step_k_list=(-1)
n_seeds=1
pos_embd_list=(false)
pos_learnable_list=(false)

cd /home/zheng/zheng/Gradient-Unpaired-Multimodal-Learning/MultiBench || exit 1

export CUDA_VISIBLE_DEVICES=$GPU_ID
python main.py -d --dataset1 "mosi" --dataset2 "mosei" --modality "xy" --lr "1e-4" --zdim "40" --num_epochs "30" --step_k "-1" --n_seeds "1" --train_jsonl --results_dir "./results/unpaired_mosi_mosei"