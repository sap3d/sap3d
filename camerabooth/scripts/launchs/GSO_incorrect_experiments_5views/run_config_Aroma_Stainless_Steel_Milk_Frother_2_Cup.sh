#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${GPU_ID} python \
    main.py \
    -t \
    --base configs/GSO_incorrect_experiments_5views/config_Aroma_Stainless_Steel_Milk_Frother_2_Cup.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from zero123_sm.ckpt \
    --project_name GSO_incorrect_experiments_5views/config_Aroma_Stainless_Steel_Milk_Frother_2_Cup \
    --logdir logs_GSO_add
