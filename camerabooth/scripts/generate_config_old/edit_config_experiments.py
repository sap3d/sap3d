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
    data_folder = "../dataset/data/GSO/train"
    config_folder = "configs/GSO_incorrect_experiments_3views"
    # config_folder = "configs/GSO_incorrect_experiments_relpose_org"

    # todo 👇 debug param name, change debug param and its name
    debug_param_names = [
        "experiments_1views",
        # "experiments_2views",
        # "experiments_3views",
        # "experiments_4views",
        # "experiments_5views",
        # "experiments_6views",
        # "experiments_noreg",
        # "experiments_randomreg",
        # "experiments_relpose_org"
        # "experiments_gsoadd"
    ]
    debug_params = [
        {'max_epoch': 100, 'lr_max': 0.1, 'lr_ratio': 1000, 'warm_up_steps': 20, 'batch_frequency': 40000},
    ]
    # todo 👆 debug param name, change debug param and its name
    
    for i in range(len(debug_param_names)):
        
        debug_param_name = debug_param_names[i]
        debug_param = debug_params[0]
        class_names = sorted(os.listdir(data_folder))
        view_num = 1

        # * output foldoer
        output_folder = f"configs/GSO_incorrect_{debug_param_name}"
        os.makedirs(output_folder, exist_ok=True)
        
        for class_name in tqdm.tqdm(class_names, desc='writing config files'):
            config_path = os.path.join(config_folder, f'config_{class_name}.yaml')
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                print(config_path)
                raise RuntimeError('File not exist!')

            # * edit param here
            config_data["data"]["params"]["root_dir"]           = f"../dataset/data/GSO/train/{class_name}"
            config_data["data"]["params"]["reg_objaverse"]      = True
            config_data["data"]["params"]["nearest_cond_views"] = True
            config_data["data"]["params"]["train_view"]         = 1
            config_data["data"]["params"]["random_objaverse"]   = False
            config_data["model"]["params"]["reg_objaverse"]     = True
            config_data["data"]["params"]["in_the_wild"]        = True

            # load ground truth camera poses 
            poses = []
            for i in range(view_num):
                poses.append(
                    get_T(np.load(f'{data_folder}/{class_name}/poses/{i:03d}.npy'))
                )
            poses = np.stack(poses)[..., 0]

            elevations = []
            azimuths = []
            radius = []
            for i in range(view_num):
                elevations.append(
                    poses[i][0].item() / np.pi,
                )
                azimuths.append(
                    0.5 * ((poses[i][1].item() - poses[0][1].item()) % (2 * np.pi)) / np.pi
                )
                radius.append(
                    (poses[i][2].item() - 1.5)/ 0.7,
                )

            config_data["lightning"]["callbacks"]["image_logger"]["params"]["a_gt"] = elevations
            config_data["lightning"]["callbacks"]["image_logger"]["params"]["b_gt"] = azimuths
            config_data["lightning"]["callbacks"]["image_logger"]["params"]["c_gt"] = radius
            
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