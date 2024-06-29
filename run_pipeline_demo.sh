#!/usr/bin/bash

# how to run
# cd $ROOT_DIR
# sh run_pipeline.sh GSO_demo MINI_ROLLER 3 0
# sh run_pipeline_demo.sh DEMO test1 3 5

# Check if the ROOT_DIR environment variable is set and not empty
if [ -z "$ROOT_DIR" ]; then
    # If ROOT_DIR is not set, use the default value
    ROOT_DIR="/home/xinyang/sap3d" # Change to your ROOT_DIR
else
    # Explicitly set ROOT_DIR to its current value from the environment variable
    ROOT_DIR=$ROOT_DIR
    echo "Using ROOT_DIR from environment: $ROOT_DIR"
fi

OBJECT_TYPE=$1
OBJECT_NAME=$2
OBJECT_VIEW=$3
GPU_ID=$4

ZERO123_PYTHON_PATH=$(conda run -n zero123 which python)
SAP3D_PYTHON_PATH=$(conda run -n sap3d which python)
echo "Zero123 env python path: " $ZERO123_PYTHON_PATH 
echo "SAP3D env python path: " $SAP3D_PYTHON_PATH

if [ "${OBJECT_VIEW}" -gt "1" ]
then
    # * pass test
    # ! run camerabooth
    # note 1. generate config for camerabooth
    echo '######### Stage1: generate config for camerabooth #########'
    cd ${ROOT_DIR}/camerabooth
    ${SAP3D_PYTHON_PATH} scripts/generate_config/edit_config_mission_two.py --object_type ${OBJECT_TYPE} \
                                                --object_name ${OBJECT_NAME} \
                                                --train_view ${OBJECT_VIEW}

    # note 2. inference relpose and estimate camera pose into config.yaml
    echo '######### Stage2: inference relpose and estimate camera pose into config.yaml #########'
    cd ${ROOT_DIR}/relposepp
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${SAP3D_PYTHON_PATH} relpose/go_relpose.py --mode infer \
                                                                                                        --object_type ${OBJECT_TYPE} \
                                                                                                        --object_name ${OBJECT_NAME} \
                                                                                                        --num_to_eval ${OBJECT_VIEW} \
                                                                                                        --use_org no \
                                                                                                        --backbone no \
                                                                                                        --inference_time 50
    
    # note 3. go camerabooth
    echo '######### Stage3: go camerabooth #########'
    cd ${ROOT_DIR}/camerabooth
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${ZERO123_PYTHON_PATH} \
        main.py \
        -t \
        --base configs/${OBJECT_TYPE}/config_${OBJECT_NAME}_view_${OBJECT_VIEW}.yaml \
        --gpus 0, \
        --scale_lr False \
        --num_nodes 1 \
        --seed 42 \
        --check_val_every_n_epoch 10 \
        --finetune_from zero123_sm.ckpt \
        --project_name ${OBJECT_NAME} \
        --logdir experiments_${OBJECT_TYPE}_view_${OBJECT_VIEW}
fi

# note 4. go nvs from diffusion model
echo '######### Stage4: go nvs from diffusion model #########'
cd ${ROOT_DIR}/camerabooth
CUDA_VISIBLE_DEVICES=${GPU_ID} ${ZERO123_PYTHON_PATH} go_nvs.py --object_type ${OBJECT_TYPE} \
                                                                                    --object_name ${OBJECT_NAME} \
                                                                                    --train_view ${OBJECT_VIEW}

# ! run reconstrcution
# note 1. generate launch for nerf
echo '######### Stage5: generate launch for nerf #########'
cd ${ROOT_DIR}/3D_Recon/threestudio
${SAP3D_PYTHON_PATH} scripts/generate_launch/edit_launch_nerf_mission_two.py --object_type ${OBJECT_TYPE} \
                                                        --object_name ${OBJECT_NAME} \
                                                        --train_view ${OBJECT_VIEW} \
                                                        --debug_pose
# note 2. go nerf
echo '######### Stage6: go nerf #########'
sh scripts/launchs/${OBJECT_TYPE}/run_nerf_${OBJECT_NAME}_view_${OBJECT_VIEW}.sh ${GPU_ID}

