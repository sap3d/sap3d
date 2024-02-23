#!/usr/bin/bash
    GPU_ID=$1
    CUDA_VISIBLE_DEVICES=${GPU_ID} /home/xinyang/miniconda3/envs/sap3d/bin/python launch.py --export \
                                                    --config configs/zero123.yaml \
                                                    --gpu 0 \
                                                    resume="experiments_GSO_add_view_3_nerf/TOP_TEN_HI_ours/[64, 128, 256]_000.png_prog0@20240223-121113/ckpts/last.ckpt" \
                                                    system.guidance.pretrained_model_name_or_path="experiments_GSO_add_view_3_nerf/TOP_TEN_HI_ours/[64, 128, 256]_000.png_prog0@20240223-121113/ckpts/last.ckpt" \
                                                    system.guidance.cond_image_path=../../dataset/data/train/GSO_add/TOP_TEN_HI/images/000.png \
                                                    system.exporter_type=mesh-exporter \
                                                    system.geometry.isosurface_method=mc-cpu \
                                                    system.geometry.isosurface_resolution=256 \
                                                    data.image_path=../../dataset/data/train/GSO_add/TOP_TEN_HI/images/000.png \
                                                    exp_root_dir=experiments_GSO_add_mesh_view_3 \
                                                    name=TOP_TEN_HI  
    