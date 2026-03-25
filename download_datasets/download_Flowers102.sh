#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url "https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/Flowers102"
python download_dataset.py --url "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/Flowers102"