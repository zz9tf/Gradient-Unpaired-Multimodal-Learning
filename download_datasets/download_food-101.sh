#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/Food101"