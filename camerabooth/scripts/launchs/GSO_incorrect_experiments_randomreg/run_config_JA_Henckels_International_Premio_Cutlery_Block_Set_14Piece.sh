#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${GPU_ID} python \
    main.py \
    -t \
    --base configs/GSO_incorrect_experiments_randomreg/config_JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --check_val_every_n_epoch 10 \
    --finetune_from zero123_sm.ckpt \
    --project_name GSO_incorrect_experiments_randomreg/config_JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece \
    --logdir logs_GSO_noema_relpose_zero123sm
