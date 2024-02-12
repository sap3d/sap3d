import os
import tqdm

def create_run_script(config_file: str, port_id:int, gpu_index: int, project_name: str, output_script: str):
    template = f"""#!/bin/bash
GPU_ID=$1 
CUDA_VISIBLE_DEVICES=${{GPU_ID}} python \\
    main.py \\
    -t \\
    --base {config_file} \\
    --gpus 0, \\
    --scale_lr False \\
    --num_nodes 1 \\
    --seed 42 \\
    --check_val_every_n_epoch 10 \\
    --finetune_from zero123_sm.ckpt \\
    --project_name {project_name} \\
    --logdir logs_GSO_noema_debug
"""

    with open(output_script, 'w') as file:
        file.write(template)

if __name__ == '__main__':
    
    # * debug param name
    debug_param_names = [
        'max_epochs',
        # 'lr_maxs',
        # 'lr_ratios',
    ]

    for i in range(len(debug_param_names)):
        debug_param_name = debug_param_names[i]
        class_name = 'Racoon'

        config_folder = f"configs/GSO_incorrect_debug_{debug_param_name}"
        output_folder = f'scripts/launchs/GSO_incorrect_debug_{debug_param_name}'
        os.makedirs(output_folder, exist_ok=True)

        max_gpu = 8
        gpu_indices = [i for i in range(max_gpu)]
        config_files = [file for file in sorted(os.listdir(config_folder)) if file.endswith('.yaml')]

        for index in tqdm.tqdm(range(len(config_files))):
            config_file = config_files[index]
            gpu_index = gpu_indices[index % len(gpu_indices)]
            project_name = os.path.join(f'exp_{debug_param_name}', f'{class_name}')
            output_script = os.path.join(output_folder, f"run_{os.path.splitext(config_file)[0]}.sh")
            config_file_path = os.path.join(config_folder, config_file)
            print(config_file_path)
            create_run_script(config_file_path, int(30000+100*index), gpu_index, project_name, output_script)
