#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url "https://drive.google.com/file/d/150nwNkg7qXh28yabiqNAOfBMMll7ALlu/view?usp=drive_link" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/StanfordCars"