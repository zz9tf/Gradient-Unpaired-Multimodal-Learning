#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd /home/zz/zheng/Unpaired-Multimodal-Learning/MultiBench

echo "[job_10] command 1/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --pos_learnable --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.8,loss_y=0.2"

echo "[job_10] command 2/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --pos_learnable --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.9,loss_y=0.1"

echo "[job_10] command 3/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --pos_learnable --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=1.0,loss_y=0.0"

echo "[job_10] command 4/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20

echo "[job_10] command 5/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.0,loss_y=1.0"

echo "[job_10] command 6/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.1,loss_y=0.9"

echo "[job_10] command 7/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.2,loss_y=0.8"

echo "[job_10] command 8/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.3,loss_y=0.7"

echo "[job_10] command 9/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.4,loss_y=0.6"

echo "[job_10] command 10/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.5,loss_y=0.5"

echo "[job_10] command 11/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.6,loss_y=0.4"

echo "[job_10] command 12/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.7,loss_y=0.3"

echo "[job_10] command 13/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.8,loss_y=0.2"

echo "[job_10] command 14/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=0.9,loss_y=0.1"

echo "[job_10] command 15/15"
python main.py -d --dataset1 mosei --dataset2 sarcasm --modality xy --lr 1e-4 --num_epochs 30 --n_seeds 3 --train_jsonl --results_dir ./results/unpaired_mosei_sarcasm_gpop --zdim 40 --step_k 20 --gpop --gpop_ref_build_kind weighted_mean --gpop_ema_beta 0.9 --gpop_edit_kind project --gpop_weights "loss_x=1.0,loss_y=0.0"
