#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${GPU_ID} python \
    main.py \
    -t \
    --base configs/GSO_incorrect_experiments_noreg/config_Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from zero123_sm.ckpt \
    --project_name GSO_incorrect_experiments_noreg/config_Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker \
    --logdir logs_GSO_noema_relpose_zero123sm
