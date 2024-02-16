#!/usr/bin/bash
    GPU_ID=$1
    CUDA_VISIBLE_DEVICES=${GPU_ID} /home/xinyang/miniconda3/envs/sap3d/bin/python launch.py --export \
                                                    --config configs/zero123.yaml \
                                                    --gpu 0 \
                                                    resume="experiments_GSO_demo_view_5_nerf/Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj_ours/[64, 128, 256]_000.png_prog0@20240215-220924/ckpts/last.ckpt" \
                                                    system.guidance.pretrained_model_name_or_path="experiments_GSO_demo_view_5_nerf/Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj_ours/[64, 128, 256]_000.png_prog0@20240215-220924/ckpts/last.ckpt" \
                                                    system.guidance.cond_image_path=../../dataset/data/train/GSO_demo/Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj/images/000.png \
                                                    system.exporter_type=mesh-exporter \
                                                    system.geometry.isosurface_method=mc-cpu \
                                                    system.geometry.isosurface_resolution=256 \
                                                    data.image_path=../../dataset/data/train/GSO_demo/Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj/images/000.png \
                                                    exp_root_dir=experiments_GSO_demo_mesh_view_5 \
                                                    name=Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj  
    