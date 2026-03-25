#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url http://madm.dfki.de/files/sentinel/EuroSAT.zip --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/EuroSAT"