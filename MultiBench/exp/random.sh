#!/bin/bash
source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate uml

GPU_ID=$1

cd /home/zheng/zheng/Gradient-Unpaired-Multimodal-Learning/MultiBench || exit 1

export CUDA_VISIBLE_DEVICES=$GPU_ID
# python main.py \
#     --outer_debug \
#     --dataset1 sarcasm \
#     --dataset2 sarcasm \
#     --lr 1e-5 \
#     --modality y \
#     --zdim 40 \
#     --num_epochs 100 \
#     --step_k -1 \
#     --n_seeds 1

# python main.py \
#     --outer_debug \
#     --dataset1 sarcasm \
#     --dataset2 sarcasm \
#     --lr 1e-5 \
#     --modality xy \
#     --zdim 40 \
#     --num_epochs 100 \
#     --step_k -1 \
#     --n_seeds 1

python main.py \
    --outer_debug \
    --dataset1 sarcasm \
    --dataset2 sarcasm \
    --lr 1e-5 \
    --modality xy \
    --zdim 40 \
    --num_epochs 100 \
    --step_k -1 \
    --x_random_noise \
    --n_seeds 1