import os
import tqdm
import yaml

if __name__ == "__main__":
    data_folder = "../dataset_ok/data/GSO/train"
    config_folder = "configs/GSO_incorrect_param_setting_max_epoch_120_lr_max_0.14_lr_ratio_2000_relpose_zero123sm"

    # * debug param name, comment means done
    debug_param_name = 'max_epochs'
    # debug_param_name = 'lr_maxs'
    # debug_param_name = 'lr_ratios'

    output_folder = f"configs/GSO_incorrect_debug_{debug_param_name}"
    os.makedirs(output_folder, exist_ok=True)

    # * debug type
    class_name = 'Racoon'

    # * param to debug
    max_epochs = [
        100, 200, 300, 400, 
        500, 600, 700, 800, 
        900, 1000,
    ]
    # lr_maxs = [
    #     0.05, 0.10, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25
    # ]
    # lr_ratios = [
    #     500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000,
    # ]

    for max_epoch in tqdm.tqdm(max_epochs, desc='writing config files'):
        config_path = os.path.join(config_folder, f'config_{class_name}.yaml')
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            raise RuntimeError('File not exist!')

        # * param okay
        
        # * edit param here
        config_data["data"]["params"]["root_dir"]           = f"../dataset/data/GSO/train/{class_name}"
        config_data["data"]["params"]["reg_objaverse"]      = True
        config_data["data"]["params"]["nearest_cond_views"] = True

        # config_data["lightning"]["callbacks"]["image_logger"]["params"]["a_gt"] = [0.5,0.5,0.5]
        # config_data["lightning"]["callbacks"]["image_logger"]["params"]["b_gt"] = [0.0,0.125,0.875]
        # config_data["lightning"]["callbacks"]["image_logger"]["params"]["c_gt"] = [0.5,0.5,0.5]
        
        config_data["lightning"]["trainer"]["max_epochs"] = max_epoch

        # config_data["model"]["params"]["a_init"] = [0.5,0.5,0.5]
        # config_data["model"]["params"]["b_init"] = [0.0,0.175,0.825] # assume angle error = 0.05*2*pi rad, 20 degree
        # config_data["model"]["params"]["c_init"] = [0.5,0.5,0.5]

        config_data["model"]["params"]["lr_ratio"]                                      = 1000
        config_data["model"]["params"]["scheduler_config"]["params"]["lr_max"]          = 0.1
        config_data["model"]["params"]["scheduler_config"]["params"]["lr_min"]          = 1e-6
        config_data["model"]["params"]["scheduler_config"]["params"]["lr_start"]        = 1e-7
        config_data["model"]["params"]["scheduler_config"]["params"]["warm_up_steps"]   = 20
        config_data["model"]["params"]["scheduler_config"]["params"]["max_decay_steps"] = max_epoch

        config_data["lightning"]["callbacks"]["image_logger"]["params"]["batch_frequency"]                    = 40000
        config_data["lightning"]["callbacks"]["image_logger"]["params"]["log_images_kwargs"]["use_ema_scope"] = False

        config_data["lightning"]["callbacks"]["ckpt_callback"]                        = {}
        config_data["lightning"]["callbacks"]["ckpt_callback"]["params"]              = {}
        config_data["lightning"]["callbacks"]["ckpt_callback"]["params"]["save_flag"] = False

        output_path = os.path.join(output_folder, f'config_{class_name}_{debug_param_name}_{max_epoch}.yaml')
        with open(output_path, "w") as f:
            yaml.safe_dump(config_data, f)