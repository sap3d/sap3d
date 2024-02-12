import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tqdm
import numpy as np
import torch
import yaml
from dataset import CustomDataset
from eval import evaluate_coordinate_ascent, evaluate_mst
from models import get_model


from pytorch_lightning import seed_everything


def normalize_rotation_matrices(rotations):
    normalized_rotations = []

    for R in rotations:
        # 使用奇异值分解（SVD）分解矩阵 R
        U, S, Vt = np.linalg.svd(R)

        # 修正行列式为+1（确保正向旋转）
        det = np.linalg.det(U @ Vt)
        if det < 0:
            Vt[-1, :] *= -1  # 反转最后一行以修正行列式为+1

        # 重新构建归一化的旋转矩阵并添加到列表中
        normalized_R = U @ Vt
        normalized_rotations.append(normalized_R)

    return np.array(normalized_rotations)


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


def estimate_elevation_azimuth_r(
    model      : torch.nn.Module,
    image_dir  : str,
    mask_dir   : str,
    num_to_eval: int = 3,
    device     : str = 'cuda',
):

    # * load in the wild images and crop parameters
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        num_to_eval=num_to_eval,
    )
    num_frames = dataset.n
    batch = dataset.get_data(ids=np.arange(num_frames))
    images = batch["image"].to(device)
    crop_params = batch["crop_params"].to(device)

    # * quickly initialize a coarse set of poses using MST reasoning
    batched_images, batched_crop_params = images.unsqueeze(0), crop_params.unsqueeze(0)
    _, hypothesis = evaluate_mst(
        model=model,
        images=batched_images,
        crop_params=batched_crop_params,
    )
    R_pred = np.stack(hypothesis)

    # * regress to optimal translation
    with torch.no_grad():
        _, _, T_pred = model(
            images=batched_images,
            crop_params=batched_crop_params,
        )

    # * search for optimal rotation via coordinate ascent.
    R_pred_rel, hypothesis = evaluate_coordinate_ascent(
        model=model,
        images=batched_images,
        crop_params=batched_crop_params,
    )
    R_final = torch.from_numpy(np.stack(hypothesis))

    R_final_pred = []
    T_final_pred = []
    for i in range(num_to_eval):
        R_final_i = R_final[i].T
        T_pred_i = T_pred[i].cpu()
        pose_i = torch.cat([R_final_i, T_pred_i[...,None]], dim=-1).cpu().numpy()
        pose_i = torch.from_numpy((pose_i))

        # * ++++++++++++++ important ++++++++++++++
        asd_i = pose_i[:3, :3]
        R_asd = torch.stack([asd_i[:, 0], asd_i[:, 2], -asd_i[:, 1]], dim=0)
        qwe_i = pose_i[:3, 3]
        T_qwe = torch.cat([qwe_i[0:1], qwe_i[1:2], qwe_i[2:3]], dim=-1)
        # * ++++++++++++++ important ++++++++++++++

        R_final_pred.append(R_asd)
        T_final_pred.append(T_qwe)

    R_final_pred = torch.stack(R_final_pred)
    T_final_pred = torch.stack(T_final_pred)
    R_final_pred = R_final_pred

    RT_final_pred = []
    for i in range(num_to_eval):
        RT_final_pred.append(
            np.concatenate([R_final_pred[i], T_final_pred[i][..., None]], axis=-1)
        )
    RT_final_pred = np.stack(RT_final_pred)

    elevation_ests, azimuth_ests, z_ests = [], [], []
    for i in range(num_to_eval):
        elevation_est, azimuth_est, z_est = get_T(RT_final_pred[i])
        
        elevation_ests.append(elevation_est)
        azimuth_ests.append(azimuth_est)
        z_ests.append(z_est)

    return elevation_ests, azimuth_ests, z_ests


if __name__ == '__main__':
    seed_everything(0)
    # * load pretrained weights
    model, _ = get_model(
        model_dir='/shared/xinyang/threetothreed/relposepp/ckpts_finetune/1026_0506_LR1e-05_N8_RandomNTrue_B36_Pretrainedckpt_back_AMP_TROURS_DDP',
        # model_dir='ckpts_refine_objaverse/finetune_relpose/0918_0658_LR1e-05_N8_RandomNTrue_B64_Pretrained0917_1455_AMP_TROURS_DDP',
        # model_dir='/home/xinyang/scratch/zelin_dev/threetothreed/relposepp/ckpts_refine_objaverse/relposepp_masked',
        num_images=3,
        device='cuda',
    )
    # * debug param name
    debug_param_names = [
        # 'GSO_incorrect_experiments_2views',
        # 'GSO_incorrect_experiments_3views',
        # 'GSO_incorrect_experiments_4views',
        # 'GSO_incorrect_experiments_5views',
        # 'GSO_incorrect_experiments_6views',
        # 'CAPTURE_experiments_in_the_wild',
        # 'GSO_incorrect_experiments_relpose_org',
        'demo',
    ]

    for i in range(len(debug_param_names)):
        debug_param_name = debug_param_names[i]
        class_names = sorted(os.listdir(f'../camerabooth/configs/{debug_param_name}'))
        config_folder = f"../camerabooth/configs/{debug_param_name}"
        config_files = [file for file in sorted(os.listdir(config_folder)) if file.endswith('.yaml')]
        view_num = 3

        for i in tqdm.tqdm(range(len(config_files))):

            config_file = config_files[i]
            class_name  = class_names[i]
            config_file_path = os.path.join(config_folder, config_file)

            if os.path.exists(config_file_path):
                with open(config_file_path, "r") as f:
                    config_data = yaml.safe_load(f)
            else:
                print(config_file_path)
                raise RuntimeError('File not exist!')

            # if config_file_path.endswith('Aroma_Stainless_Steel_Milk_Frother_2_Cup.yaml'):

            image_dir = f'../camerabooth/{config_data["data"]["params"]["root_dir"]}/images'
            mask_dir  = f'../camerabooth/{config_data["data"]["params"]["root_dir"]}/masks'

            all_elevations = []
            all_azimuths   = []
            all_zs         = []
            for i in tqdm.tqdm(range(25), desc='searching camera pose', leave=False):
                elevations, azimuths, zs = estimate_elevation_azimuth_r(
                    model,
                    image_dir,
                    mask_dir,
                    num_to_eval=view_num,
                )
                
                # compute 
                elevations = [
                    elevations[i].item() for i in range(view_num)
                ]
                azimuths = [
                    (azimuths[i].item() - azimuths[0].item()) % (2 * np.pi) for i in range(view_num)
                ]
                zs = [
                    zs[i].item() for i in range(view_num)
                ]
                    
                all_elevations.append(np.stack(elevations))
                all_azimuths.append(np.stack(azimuths))
                all_zs.append(np.stack(zs))
            
            all_elevations = np.stack(all_elevations, axis=0)
            all_azimuths   = np.stack(all_azimuths, axis=0)
            all_zs         = np.stack(all_zs, axis=0)

            best_elevations = np.median(all_elevations, axis=0) / np.pi
            best_azimuths   = 0.5 * np.median(all_azimuths, axis=0) / np.pi
            best_zs         = (np.median(all_zs, axis=0) - 1.5) / 0.7
            
            tqdm.tqdm.write(f'=== Best Estimation Results for {class_name} ===')
            tqdm.tqdm.write(f'best_d_elevations: {best_elevations}')
            tqdm.tqdm.write(f'best_d_azimuths: {best_azimuths}')
            tqdm.tqdm.write(f'best_d_rs: {best_zs}')

            config_data["model"]["params"]["a_init"] = best_elevations.tolist()
            config_data["model"]["params"]["b_init"] = best_azimuths.tolist()
            config_data["model"]["params"]["c_init"] = best_zs.tolist()

            with open(config_file_path, "w") as f:
                yaml.safe_dump(config_data, f)