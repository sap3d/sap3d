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

def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * np.pi)
    d_z = z_target - z_cond

    return d_theta.item(), d_azimuth.item(), d_z.item()

def get_T_my(RT_mtx):
    R, T = RT_mtx[:3, :3], RT_mtx[:, -1]
    RT_mtx_w2c = -R.T @ T
    elevation, azimuth, z = cartesian_to_spherical(RT_mtx_w2c[None, :])
    return elevation, azimuth, z

def get_configures():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--object_type', type=str, default='XINYANG')
    parser.add_argument('--object_name', type=str, default='meow1')
    parser.add_argument('--train_view', type=int, default=3)
    parser.add_argument('--em7', type=bool, default=False)

    return parser.parse_args()

if __name__ == "__main__":
    train_params = {
        'max_epoch'      : 1,
        'lr_max'         : 0.1,
        'lr_ratio'       : 1000,
        'warm_up_steps'  : 20,
        'batch_frequency': 40000,
    }
    opt = get_configures()

    # * output foldoer
    output_folder = f"configs/{opt.object_type}"
    os.makedirs(output_folder, exist_ok=True)

    with open("configs/config_base.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    # * edit param here
    config_data["data"]["params"]["root_dir"]      = f"../dataset/data/train/{opt.object_type}/{opt.object_name}"
    config_data["data"]["params"]["reg_objaverse"] = True
    config_data["data"]["params"]["train_view"]    = opt.train_view

    if opt.em7:
        config_data["data"]["params"]["objaverse_path"] = "../dataset/data/objaverse"
    else:
        config_data["data"]["params"]["objaverse_path"] = "../dataset/data/objaverse"

    
    config_data["model"]["params"]["lr_ratio"]                                      = train_params['lr_ratio']
    config_data["model"]["params"]["scheduler_config"]["params"]["lr_max"]          = train_params['lr_max']
    config_data["model"]["params"]["scheduler_config"]["params"]["lr_min"]          = 1e-6
    config_data["model"]["params"]["scheduler_config"]["params"]["lr_start"]        = 1e-7
    config_data["model"]["params"]["scheduler_config"]["params"]["warm_up_steps"]   = train_params['warm_up_steps']
    config_data["model"]["params"]["scheduler_config"]["params"]["max_decay_steps"] = 400
    
    config_data["lightning"]["trainer"]["max_epochs"]                             = train_params['max_epoch']
    config_data["lightning"]["callbacks"]["ckpt_callback"]                        = {}
    config_data["lightning"]["callbacks"]["ckpt_callback"]["params"]              = {}
    config_data["lightning"]["callbacks"]["ckpt_callback"]["params"]["save_flag"] = True

    if os.path.exists(f'../dataset/data/train/{opt.object_type}/{opt.object_name}/poses'):
        RT_gt = []
        for i in range(opt.train_view):
            RT_gt.append(
                get_T_my(np.load(f'../dataset/data/train/{opt.object_type}/{opt.object_name}/poses/{i:03d}.npy'))
            )
        RT_gt = np.stack(RT_gt)[..., 0]

        elevations = []
        azimuths = []
        radius = []
        for i in range(opt.train_view):
            elevations.append(
                RT_gt[i][0].item() / np.pi,
            )
            azimuths.append(
                0.5 * ((RT_gt[i][1].item() - RT_gt[0][1].item()) % (2 * np.pi)) / np.pi
            )
            radius.append(
                (RT_gt[i][2].item() - 1.5)/ 0.7,
            )
        print(f'{opt.object_type} {opt.object_name} GT Pose', elevations, azimuths, radius)
        config_data["lightning"]["callbacks"]["image_logger"]["params"]["a_gt"] = elevations
        config_data["lightning"]["callbacks"]["image_logger"]["params"]["b_gt"] = azimuths
        config_data["lightning"]["callbacks"]["image_logger"]["params"]["c_gt"] = radius
            
    output_path = os.path.join(output_folder, f'config_{opt.object_name}_view_{opt.train_view}.yaml')
    with open(output_path, "w") as f:
        yaml.safe_dump(config_data, f)
    
    print(f'Done! Save at {output_path}')