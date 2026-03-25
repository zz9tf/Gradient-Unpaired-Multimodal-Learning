#!/bin/bash
source /home/zz/miniconda3/etc/profile.d/conda.sh
conda activate uml

cd /home/zz/zheng/Unpaired-Multimodal-Learning/download_datasets
python download_dataset.py --url "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/ImageNet"
python download_dataset.py --url "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/ImageNet"
python download_dataset.py --url "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/ImageNet"
python download_dataset.py --url "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar" --output_dir "/home/zz/zheng/Unpaired-Multimodal-Learning/data/image/ImageNet"