#!/usr/bin/bash
GPU_ID=$1

CUDA_VISIBLE_DEVICES=${GPU_ID} /home/xinyang/miniconda3/envs/sap3d/bin/python launch.py --train \
                                                                                    --config configs/zero123.yaml \
                                                                                    --gpu 0 \
                                                                                    data_type=multiview-image-datamodule \
                                                                                    data.batch_size=1 \
                                                                                    data.height=[128,256,512] \
                                                                                    data.width=[128,256,512] \
                                                                                    data.image_path=../../dataset/data/train/GSO_add/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure/images/000.png \
                                                                                    data.default_elevation_deg=32.373992919921875 \
                                                                                    data.default_azimuth_deg=0.0 \
                                                                                    data.default_camera_distance=3.8 \
                                                                                    data.multiple_image_folder=../../dataset/data/train/GSO_add/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure \
                                                                                    data.estimate_pose_path=../../camerabooth/experiments_GSO_add_view_3/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure/2024-02-23T00-50-56_config_Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure_view_3-1-1000-1e-06-0.1-1e-07/evaluation/epoch_99/results.tar \
                                                                                    data.random_camera.elevation_range=[-60,60] \
                                                                                    data.random_camera.batch_size=[12,4,4] \
                                                                                    data.random_camera.height=[64,128,256] \
                                                                                    data.random_camera.width=[64,128,256] \
                                                                                    data.random_camera.resolution_milestones=[200,300] \
                                                                                    exp_root_dir=experiments_GSO_add_view_3_nerf \
                                                                                    name=Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure_ours \
                                                                                    system.is_gso=True \
                                                                                    system.guidance.guidance_scale=5.0 \
                                                                                    system.guidance.pretrained_model_name_or_path=../../camerabooth/experiments_GSO_add_view_3/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure/2024-02-23T00-50-56_config_Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure_view_3-1-1000-1e-06-0.1-1e-07/checkpoints/last.ckpt \
                                                                                    system.guidance.use_sc_sds=True \
                                                                                    system.guidance.cond_view=1 \
                                                                                    system.guidance.cond_image_folder=../../dataset/data/train/GSO_add/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure \
                                                                                    system.guidance.estimate_pose_path=../../camerabooth/experiments_GSO_add_view_3/Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure/2024-02-23T00-50-56_config_Guardians_of_the_Galaxy_Galactic_Battlers_Rocket_Raccoon_Figure_view_3-1-1000-1e-06-0.1-1e-07/evaluation/epoch_99/results.tar \
                                                                                    system.loss.lambda_sparsity=1.0 \
                                                                                    trainer.max_steps=4500 \
                                                                                    trainer.val_check_interval=4500 \
                                                                                    checkpoint.every_n_train_steps=10000 \
                                                                                    checkpoint.save_last=True
