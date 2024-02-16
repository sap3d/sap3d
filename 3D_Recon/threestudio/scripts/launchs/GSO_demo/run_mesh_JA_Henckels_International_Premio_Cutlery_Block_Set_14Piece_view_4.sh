#!/usr/bin/bash
    GPU_ID=$1
    CUDA_VISIBLE_DEVICES=${GPU_ID} /home/xinyang/miniconda3/envs/sap3d/bin/python launch.py --export \
                                                    --config configs/zero123.yaml \
                                                    --gpu 0 \
                                                    resume="experiments_GSO_demo_view_4_nerf/JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece_ours/[64, 128, 256]_000.png_prog0@20240216-001457/ckpts/last.ckpt" \
                                                    system.guidance.pretrained_model_name_or_path="experiments_GSO_demo_view_4_nerf/JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece_ours/[64, 128, 256]_000.png_prog0@20240216-001457/ckpts/last.ckpt" \
                                                    system.guidance.cond_image_path=../../dataset/data/train/GSO_demo/JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece/images/000.png \
                                                    system.exporter_type=mesh-exporter \
                                                    system.geometry.isosurface_method=mc-cpu \
                                                    system.geometry.isosurface_resolution=256 \
                                                    data.image_path=../../dataset/data/train/GSO_demo/JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece/images/000.png \
                                                    exp_root_dir=experiments_GSO_demo_mesh_view_4 \
                                                    name=JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece  
    