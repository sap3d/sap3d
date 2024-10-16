import os
import tqdm
import yaml
import numpy as np

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])


def get_T(RT_mtx):
    R, T = RT_mtx[:3, :3], RT_mtx[:, -1]
    RT_mtx_w2c = -R.T @ T
    elevation, azimuth, z = cartesian_to_spherical(RT_mtx_w2c[None, :])
    return elevation, azimuth, z


if __name__ == "__main__":
    data_folder = "../dataset/data/CAPTURE"
    config_folder = "configs/GSO_incorrect_param_setting_max_epoch_120_lr_max_0.14_lr_ratio_2000_relpose_zero123sm"

    # todo 👇 debug param name, change debug param and its name
    debug_param_names = [
        "experiments_in_the_wild",
    ]
    debug_params = [
        {'max_epoch': 100, 'lr_max': 0.1, 'lr_ratio': 1000, 'warm_up_steps': 20, 'batch_frequency': 40000},
    ]
    # todo 👆 debug param name, change debug param and its name
    
    for i in range(len(debug_param_names)):
        debug_param_name = debug_param_names[i]
        debug_param = debug_params[i]
        class_names = sorted(os.listdir(data_folder))
        
        # * output foldoer
        output_folder = f"configs/CAPTURE_{debug_param_name}"
        os.makedirs(output_folder, exist_ok=True)
        
        for class_name in tqdm.tqdm(class_names, desc='writing config files'):
            config_path = os.path.join(config_folder, f'config_Racoon.yaml')
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                print(config_path)
                raise RuntimeError('File not exist!')

            # * edit param here
            config_data["data"]["params"]["root_dir"]           = f"../dataset/data/CAPTURE/{class_name}"
            config_data["data"]["params"]["reg_objaverse"]      = True
            config_data["data"]["params"]["nearest_cond_views"] = True
            config_data["data"]["params"]["in_the_wild"]        = True

            config_data["lightning"]["callbacks"]["image_logger"]["params"]["in_the_wild"] = True
            
            config_data["lightning"]["trainer"]["max_epochs"]                               = debug_param['max_epoch']

            config_data["model"]["params"]["lr_ratio"]                                      = debug_param['lr_ratio']
            config_data["model"]["params"]["scheduler_config"]["params"]["lr_max"]          = debug_param['lr_max']
            config_data["model"]["params"]["scheduler_config"]["params"]["lr_min"]          = 1e-6
            config_data["model"]["params"]["scheduler_config"]["params"]["lr_start"]        = 1e-7
            config_data["model"]["params"]["scheduler_config"]["params"]["warm_up_steps"]   = debug_param['warm_up_steps']
            config_data["model"]["params"]["scheduler_config"]["params"]["max_decay_steps"] = 400

            config_data["lightning"]["callbacks"]["ckpt_callback"] = {}
            config_data["lightning"]["callbacks"]["ckpt_callback"]["params"] = {}
            config_data["lightning"]["callbacks"]["ckpt_callback"]["params"]["save_flag"]   = True

            output_path = os.path.join(output_folder, f'config_{class_name}.yaml')
            with open(output_path, "w") as f:
                yaml.safe_dump(config_data, f)
                
    