#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${GPU_ID} python \
    main.py \
    -t \
    --base configs/all_demo/config_Sperry_TopSider_tNB9t6YBUf3.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from zero123_sm.ckpt \
    --project_name all_demo/config_Sperry_TopSider_tNB9t6YBUf3 \
    --logdir logs_all_demo
