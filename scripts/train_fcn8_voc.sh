#!/bin/bash
python3 examples/voc/train_fcn8.py --config examples/voc/train_fcn8.json --dataroot /mnt/datasets/voc \
    --run_dir runs --name voc_fcn8 \
    --gpu_ids -1 \
    --resume_ckp_num 0