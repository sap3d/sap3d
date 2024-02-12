import os
import glob
import tqdm
import torch
import numpy as np
import subprocess

def get_conda_env_python_path(env_name):
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "which", "python"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def create_run_script(object_type: str, object_name: str, ckpt_path: str, train_view: int, output_script: str):
    if train_view > 1:
        import torch
        import numpy as np
        env_name = "sap3d"
        python_path = get_conda_env_python_path(env_name)
        finetune_path   = f'{ckpt_path}/evaluation/epoch_99/results.tar'
        finetune_result = torch.load(finetune_path)
        elevation_deg_pred = torch.FloatTensor(finetune_result['elevation_pred']) / np.pi * 180.
        azimuth_deg_pred   = torch.FloatTensor(finetune_result['azimuth_pred']) / np.pi * 180.

        cond_elevation_deg = (90.0 - elevation_deg_pred) - min(90.0 - elevation_deg_pred)
        # cond_azimuth_deg   = (azimuth_deg_pred - azimuth_deg_pred[0] + 180) % 360 - 180 # [-180, 180]
        cond_azimuth_deg   = azimuth_deg_pred - azimuth_deg_pred[0]
        print(cond_elevation_deg, cond_azimuth_deg)
        template = f"""#!/usr/bin/bash
GPU_ID=$1

CUDA_VISIBLE_DEVICES=${{GPU_ID}} {python_path} launch.py --train \\
                                                                                    --config configs/zero123.yaml \\
                                                                                    --gpu 0 \\
                                                                                    data_type=multiview-image-datamodule \\
                                                                                    data.batch_size=1 \\
                                                                                    data.height=[128,256,512] \\
                                                                                    data.width=[128,256,512] \\
                                                                                    data.image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                                                    data.default_elevation_deg={cond_elevation_deg[0]} \\
                                                                                    data.default_azimuth_deg=0.0 \\
                                                                                    data.default_camera_distance=3.8 \\
                                                                                    data.multiple_image_folder=../../dataset/data/train/{object_type}/{object_name} \\
                                                                                    data.estimate_pose_path={ckpt_path}/evaluation/epoch_99/results.tar \\
                                                                                    data.random_camera.elevation_range=[-60,60] \\
                                                                                    data.random_camera.batch_size=[12,4,4] \\
                                                                                    data.random_camera.height=[64,128,256] \\
                                                                                    data.random_camera.width=[64,128,256] \\
                                                                                    data.random_camera.resolution_milestones=[200,300] \\
                                                                                    exp_root_dir=experiments_{object_type}_view_{train_view}_nerf \\
                                                                                    name={object_name}_ours \\
                                                                                    system.is_gso=True \\
                                                                                    system.guidance.guidance_scale=5.0 \\
                                                                                    system.guidance.pretrained_model_name_or_path={ckpt_path}/checkpoints/last.ckpt \\
                                                                                    system.guidance.use_sc_sds=True \\
                                                                                    system.guidance.cond_view=1 \\
                                                                                    system.guidance.cond_image_folder=../../dataset/data/train/{object_type}/{object_name} \\
                                                                                    system.guidance.estimate_pose_path={ckpt_path}/evaluation/epoch_99/results.tar \\
                                                                                    system.loss.lambda_sparsity=1.0 \\
                                                                                    trainer.max_steps=4500 \\
                                                                                    trainer.val_check_interval=4500 \\
                                                                                    checkpoint.every_n_train_steps=10000 \\
                                                                                    checkpoint.save_last=True
"""
    else:
        import torch
        import numpy as np
        ckpt_path = find_last_log(f'../../camerabooth/experiments_{opt.object_type}_view_3/{opt.object_name}')
        finetune_path   = f'{ckpt_path}/evaluation/epoch_99/results.tar'
        finetune_result = torch.load(finetune_path)
        elevation_deg_pred = torch.FloatTensor(finetune_result['elevation_pred']) / np.pi * 180.
        azimuth_deg_pred   = torch.FloatTensor(finetune_result['azimuth_pred']) / np.pi * 180.

        cond_elevation_deg = 90.0 - (elevation_deg_pred - min(elevation_deg_pred))
        # cond_azimuth_deg   = (azimuth_deg_pred - azimuth_deg_pred[0] + 180) % 360 - 180 # [-180, 180]
        cond_azimuth_deg   = azimuth_deg_pred - azimuth_deg_pred[0]
        print(cond_elevation_deg, cond_azimuth_deg)

        template = f"""#!/usr/bin/bash
GPU_ID=$1

CUDA_VISIBLE_DEVICES=${{GPU_ID}} {python_path} launch.py --train \\
                                                                                    --config configs/zero123.yaml \\
                                                                                    --gpu 0 \\
                                                                                    data_type=single-image-datamodule \\
                                                                                    data.batch_size=1 \\
                                                                                    data.height=[128,256,512] \\
                                                                                    data.width=[128,256,512] \\
                                                                                    data.image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                                                    data.default_elevation_deg={cond_elevation_deg[0]} \\
                                                                                    data.default_azimuth_deg=0.0 \\
                                                                                    data.default_camera_distance=3.8 \\
                                                                                    data.random_camera.elevation_range=[-60,60] \\
                                                                                    data.random_camera.batch_size=[12,4,2] \\
                                                                                    data.random_camera.height=[64,128,256] \\
                                                                                    data.random_camera.width=[64,128,256] \\
                                                                                    data.random_camera.resolution_milestones=[200,300] \\
                                                                                    exp_root_dir=experiments_{object_type}_view_{train_view}_nerf \\
                                                                                    name={object_name}_zero123 \\
                                                                                    system.is_gso=True \\
                                                                                    system.guidance.guidance_scale=5.0 \\
                                                                                    system.guidance.pretrained_model_name_or_path=../../camerabooth/zero123_sm.ckpt \\
                                                                                    system.loss.lambda_sparsity=1.0 \\
                                                                                    trainer.max_steps=3000 \\
                                                                                    trainer.val_check_interval=3000 \\
                                                                                    checkpoint.every_n_train_steps=10000 \\
                                                                                    checkpoint.save_last=True
"""

    with open(output_script, 'w') as file:
        file.write(template)

def find_last_log(dir_path):
    timestamp_dirs = glob.glob(os.path.join(dir_path, "*"))
    timestamp_dirs.sort(key=os.path.getmtime, reverse=True)
    if timestamp_dirs:
        latest_dir = timestamp_dirs[0]
        return latest_dir
    return None

def get_configures():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_type', type=str, default='XINYANG')
    parser.add_argument('--object_name', type=str, default='meow1')
    parser.add_argument('--train_view', type=int, default=5)
    parser.add_argument('--debug_pose', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    opt = get_configures()
    ckpt_path = find_last_log(f'../../camerabooth/experiments_{opt.object_type}_view_{opt.train_view}/{opt.object_name}')
    if ckpt_path or opt.train_view==1:
        os.makedirs(f'scripts/launchs/{opt.object_type}', exist_ok=True)
        output_path = f'scripts/launchs/{opt.object_type}/run_nerf_{opt.object_name}_view_{opt.train_view}.sh'
        create_run_script(opt.object_type, opt.object_name, ckpt_path, opt.train_view, output_path)
        print(f'Done! Save at {output_path}')

    else:
        print(f'No Find {opt.object_name}!!!')