import os
import glob
import tqdm
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

def create_run_script(object_type: str, object_name: str, train_view: int, ckpt_path: str, output_script: str):
    env_name = "sap3d"
    python_path = get_conda_env_python_path(env_name)
    if train_view == 1:
        template = f"""#!/usr/bin/bash
    GPU_ID=$1
    CUDA_VISIBLE_DEVICES=${{GPU_ID}} {python_path} launch.py --export \\
                                                    --config configs/zero123.yaml \\
                                                    --gpu 0 \\
                                                    resume="{ckpt_path}/ckpts/last.ckpt" \\
                                                    system.guidance.pretrained_model_name_or_path=../../camerabooth/zero123_sm.ckpt \\
                                                    system.guidance.cond_image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                    system.exporter_type=mesh-exporter \\
                                                    system.geometry.isosurface_method=mc-cpu \\
                                                    system.geometry.isosurface_resolution=256 \\
                                                    data.image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                    exp_root_dir=experiments_{object_type}_mesh_view_{train_view} \\
                                                    name={object_name}  
    """
    else:
        template = f"""#!/usr/bin/bash
    GPU_ID=$1
    CUDA_VISIBLE_DEVICES=${{GPU_ID}} {python_path} launch.py --export \\
                                                    --config configs/zero123.yaml \\
                                                    --gpu 0 \\
                                                    resume="{ckpt_path}/ckpts/last.ckpt" \\
                                                    system.guidance.pretrained_model_name_or_path="{ckpt_path}/ckpts/last.ckpt" \\
                                                    system.guidance.cond_image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                    system.exporter_type=mesh-exporter \\
                                                    system.geometry.isosurface_method=mc-cpu \\
                                                    system.geometry.isosurface_resolution=256 \\
                                                    data.image_path=../../dataset/data/train/{object_type}/{object_name}/images/000.png \\
                                                    exp_root_dir=experiments_{object_type}_mesh_view_{train_view} \\
                                                    name={object_name}  
    """

    with open(output_script, 'w') as file:
        file.write(template)

def find_last_log(dir_path):
    # 查找所有日期和时间戳文件夹
    timestamp_dirs = glob.glob(os.path.join(dir_path, "*"))
    # 排序以确保最新的文件夹排在最前面
    timestamp_dirs.sort(key=os.path.getmtime, reverse=True)
    if timestamp_dirs:
        # 从最新的文件夹中查找last.ckpt文件
        latest_dir = timestamp_dirs[0]
        return latest_dir
    # 如果没有找到，返回None
    return None

def get_configures():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--object_type', type=str, default='XINYANG')
    parser.add_argument('--object_name', type=str, default='meow1')
    parser.add_argument('--train_view', type=int, default=3)
    parser.add_argument('--debug_pose', action='store_true', default=False)

    return parser.parse_args()

if __name__ == '__main__':
    opt = get_configures()
    if opt.train_view == 1:
        ckpt_path = find_last_log(f'experiments_{opt.object_type}_view_{opt.train_view}_nerf/{opt.object_name}_zero123')
    else:
        ckpt_path = find_last_log(f'experiments_{opt.object_type}_view_{opt.train_view}_nerf/{opt.object_name}_ours')

    if ckpt_path:
        os.makedirs(f'scripts/launchs/{opt.object_type}', exist_ok=True)
        output_path = f'scripts/launchs/{opt.object_type}/run_mesh_{opt.object_name}_view_{opt.train_view}.sh'
        create_run_script(opt.object_type, opt.object_name, opt.train_view, ckpt_path, output_path)
        print(f'Done! Save at {output_path}')
    else:
        print(f'No Find {opt.object_name}!!!')
