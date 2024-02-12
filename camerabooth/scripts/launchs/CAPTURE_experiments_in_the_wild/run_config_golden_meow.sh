#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${GPU_ID} python \
    main.py \
    -t \
    --base configs/CAPTURE_experiments_in_the_wild/config_golden_meow.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from zero123_sm.ckpt \
    --project_name experiments_debug/config_golden_meow \
    --logdir logs_DEBUG
