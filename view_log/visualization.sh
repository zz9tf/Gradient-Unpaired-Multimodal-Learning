#!/bin/bash

# source /home/zz/miniconda3/etc/profile.d/conda.sh
source /home/zheng/zheng/miniconda3/etc/profile.d/conda.sh
conda activate uml
# cd /home/zz/zheng/Unpaired-Multimodal-Learning/view_log
cd /home/zheng/zheng/Gradient-Unpaired-Multimodal-Learning/view_log
input_folder="${1:-0}"

# for seed in {0..2}; do
#     python plot_all.py \
#         --input ${input_folder}/seed_${seed}/stats.jsonl
# done

# python plot_repr.py \
#     --input ${input_path}

python summarize_res.py --results-dir ${input_folder} --output ${input_folder}/summary.md
python multibanch_heatmap.py --input ${input_folder}/summary.md --outdir ${input_folder}/heatmap
