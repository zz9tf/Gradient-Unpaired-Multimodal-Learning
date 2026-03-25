#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/DTD"
