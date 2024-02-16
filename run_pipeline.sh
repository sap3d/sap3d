#!/usr/bin/bash

# how to run
# cd $ROOT_DIR
# sh run_pipeline.sh GSO_demo MINI_FIRE_ENGINE 5 0

ROOT_DIR=ROOT_DIR # Change to Your ROOT_DIR

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

# 3. generate launch for mesh
${SAP3D_PYTHON_PATH} scripts/generate_launch/edit_launch_mesh_em11.py --object_type ${OBJECT_TYPE} \
                                                        --object_name ${OBJECT_NAME} \
                                                        --train_view ${OBJECT_VIEW}

# 4. go mesh
sh scripts/launchs/${OBJECT_TYPE}/run_mesh_${OBJECT_NAME}_view_${OBJECT_VIEW}.sh ${GPU_ID}

# 5. Calculate 3D Metrics
echo '######### Reconstruction Eval #########'
cd ${ROOT_DIR}/SyncDreamer
${SAP3D_PYTHON_PATH} eval_mesh.py --target_res experiments_GSO_demo_mesh_view_${OBJECT_VIEW} --OBJECT_TYPE ${OBJECT_TYPE} --OBJECT_NAME ${OBJECT_NAME} --OBJECT_VIEW ${OBJECT_VIEW} --ROOT_DIR ${ROOT_DIR}

# 6. Calculate Pose Error
echo '######### Pose Eval #########'
cd ${ROOT_DIR}/camerabooth
${SAP3D_PYTHON_PATH} scripts/generate_summary/summary_experiments.py --OBJECT_TYPE ${OBJECT_TYPE} --OBJECT_NAME ${OBJECT_NAME} --OBJECT_VIEW ${OBJECT_VIEW} --ROOT_DIR ${ROOT_DIR}

# 7. Calculate 2D Metrics
echo '######### 2D Metrics -- NVS #########'
cd ${ROOT_DIR}/camerabooth
${SAP3D_PYTHON_PATH} scripts/generate_summary/summary_experiments_nvs.py --OBJECT_TYPE ${OBJECT_TYPE} --OBJECT_NAME ${OBJECT_NAME} --OBJECT_VIEW ${OBJECT_VIEW} --ROOT_DIR ${ROOT_DIR}

# 7. Calculate 2D Metrics
echo '######### 2D Metrics -- 3D Rendering #########'
cd ${ROOT_DIR}/camerabooth
${SAP3D_PYTHON_PATH} scripts/generate_summary/summary_experiments_nvs_NeRF.py --target_res experiments_GSO_demo_view_${OBJECT_VIEW}_nerf --OBJECT_TYPE ${OBJECT_TYPE} --OBJECT_NAME ${OBJECT_NAME} --OBJECT_VIEW ${OBJECT_VIEW} --ROOT_DIR ${ROOT_DIR}

