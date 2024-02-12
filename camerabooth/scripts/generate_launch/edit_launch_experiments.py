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
    --logdir logs_GSO_add
"""

    with open(output_script, 'w') as file:
        file.write(template)

if __name__ == '__main__':
    
    # todo 👇 debug param name，add new name (refer to edit_config_param.py)
    debug_param_names = [
        # '!!!! new_name_here !!!!',
        # 'experiments_max_epoch_300_lr_max_0_1_lr_ratio_1000',
        # 'experiments_max_epoch_300_lr_max_0_1_lr_ratio_1000_nearest_view',
        # 'experiments_max_epoch_400_lr_max_0_1_lr_ratio_1000_nearest_view',
        "GSO_incorrect_experiments_1views",
        # 'GSO_incorrect_experiments_2views',
        # 'GSO_incorrect_experiments_3views',
        # 'GSO_incorrect_experiments_4views',
        # 'GSO_incorrect_experiments_5views',
        # 'GSO_incorrect_experiments_6views',
        # 'GSO_incorrect_experiments_noreg',
        # 'GSO_incorrect_experiments_randomreg',
        # 'GSO_incorrect_experiments_relpose_org',
        # 'GSO_incorrect_experiments_gsoadd',
    ]
    # todo 👆 debug param name

    for i in range(len(debug_param_names)):
        debug_param_name = debug_param_names[i]
        # class_names = sorted(os.listdir('configs/GSO_incorrect_experiments_gsoadd'))
        class_names = sorted(os.listdir('configs/GSO_incorrect_experiments_3views'))

        config_folder = f"configs/{debug_param_name}"
        output_folder = f'scripts/launchs/{debug_param_name}'
        os.makedirs(output_folder, exist_ok=True)

        max_gpu = 8
        gpu_indices = [i for i in range(max_gpu)]
        config_files = [file for file in sorted(os.listdir(config_folder)) if file.endswith('.yaml')]

        for index in tqdm.tqdm(range(len(class_names))):
            class_name = class_names[index].split('.')[0]
            
            # if class_name.endswith('Black_Decker_CM2035B_12Cup_Thermal_Coffeemaker') or + \
            #    class_name.endswith('JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece') or + \
            #    class_name.endswith('Racoon') or + \
            #    class_name.endswith('Sonny_School_Bus') or + \
            #    class_name.endswith('Schleich_Hereford_Bull') or + \
            #    class_name.endswith('Crosley_Alarm_Clock_Vintage_Metal') or + \
            #    class_name.endswith('Schleich_Bald_Eagle'):

            config_file = config_files[index]
            gpu_index = gpu_indices[index % len(gpu_indices)]
            project_name = os.path.join(f'{debug_param_name}', f'{class_name}')
            output_script = os.path.join(output_folder, f"run_{os.path.splitext(config_file)[0]}.sh")
            config_file_path = os.path.join(config_folder, config_file)
            print(config_file_path)
            create_run_script(config_file_path, int(30000+100*index), gpu_index, project_name, output_script)
