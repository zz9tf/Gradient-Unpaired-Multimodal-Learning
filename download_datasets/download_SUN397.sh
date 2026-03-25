#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/SUN397"
python download_dataset.py --url https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/SUN397"