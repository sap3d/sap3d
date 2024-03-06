import os
import yaml
import tqdm
import numpy as np
import torch
from dataset import CustomDataset
from eval import evaluate_coordinate_ascent, evaluate_mst
from models import get_model
from pytorch3d.renderer import FoVPerspectiveCameras
from gradio_client import Client
import torch.nn.functional as F

def ear2rotation(elevation, azimuth, camera_distances):
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )[None, :]
    # print(camera_positions.shape)
    center = torch.zeros_like(camera_positions)
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :]
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1,
    )
    c2w[:, 3, 3] = 1.0
    return c2w

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

def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi

def get_permutations(num_images, eval_time=False):
    if not eval_time:
        permutations = []
        for i in range(1, num_images):
            for j in range(num_images - 1):
                if i > j:
                    permutations.append((j, i))
    else:
        permutations = []
        for i in range(0, num_images):
            for j in range(0, num_images):
                if i != j:
                    permutations.append((j, i))

    return permutations

def get_n_consistent_cameras(R_pred, num_frames):
    R_pred_n = torch.zeros(num_frames, 3, 3)
    R_pred_n[0] = torch.eye(3)
    for k, (i, j) in enumerate(get_permutations(num_frames, eval_time=True)):
        if i == 0:
            R_pred_n[j] = R_pred[k]

    return R_pred_n

def full_scene_scale(R, T):
    def blender2torch3d_objaverse(blw2blc):
        homo = torch.tensor([0,0,0,1]).float()
        R_blc2blw = blw2blc[:3, :3].T.float()
        T_blc2blw = -R_blc2blw @ blw2blc[:3, 3].float()
        blc2blw = torch.cat([R_blc2blw, T_blc2blw[...,None]], axis=-1)
        blc2blw = torch.cat([blc2blw, homo[None,...]], axis=0)
        # * blender to opengl
        blw2glw = torch.tensor([[1,0,0,0], [0,0,1,0], [0,-1,0,0], [0,0,0,1]]).float()
        glw2blc = torch.linalg.inv(blw2glw @ blc2blw)
        # * opengl to pytorch3d
        blc2pyc = torch.tensor([[-1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]]).float()
        glw2pyc = blc2pyc @ glw2blc
        return glw2pyc
    
    R_mtx_ok = torch.zeros_like(R)
    T_vec_ok = torch.zeros_like(T)
    for i in range(len(R)):
        RT_mtx = torch.cat([R[i], T[i][..., None]], dim=-1)
        RT_mtx_ok = blender2torch3d_objaverse(RT_mtx)
        R_mtx_ok[i] = RT_mtx_ok[:3, :3]
        T_vec_ok[i] = RT_mtx_ok[:3, 3]
    # Calculate centroid of cameras in batch
    cameras = FoVPerspectiveCameras(R=R_mtx_ok, T=T_vec_ok)
    cc = cameras.get_camera_center()
    centroid = torch.mean(cc, dim=0)

    # Determine distance from centroid to each camera
    diffs = cc - centroid
    norms = torch.linalg.norm(diffs, dim=1)

    # Scene scale is the distance from the centroid to the furthest camera
    furthest_index = torch.argmax(norms).item()
    scale = norms[furthest_index].item()
    return scale

def compute_optimal_translation_alignment(T_A, T_B, R_B):
    """
    Assuming right-multiplied rotation matrices.

    E.g., for world2cam R and T, a world coordinate is transformed to camera coordinate
    system using X_cam = X_world.T @ R + T = R.T @ X_world + T

    Finds s, t that minimizes || T_A - (s * T_B + R_B.T @ t) ||^2

    Args:
        T_A (torch.Tensor): Target translation (N, 3).
        T_B (torch.Tensor): Initial translation (N, 3).
        R_B (torch.Tensor): Initial rotation (N, 3, 3).

    Returns:
        T_A_hat (torch.Tensor): s * T_B + t @ R_B (N, 3).
        scale s (torch.Tensor): (1,).
        translation t (torch.Tensor): (1, 3).
    """
    n = len(T_A)

    T_A = T_A.unsqueeze(2)
    T_B = T_B.unsqueeze(2)

    A = torch.sum(T_B * T_A)
    B = (T_B.transpose(1, 2) @ R_B.transpose(1, 2)).sum(0) @ (R_B @ T_A).sum(0) / n
    C = torch.sum(T_B * T_B)
    D = (T_B.transpose(1, 2) @ R_B.transpose(1, 2)).sum(0)
    E = (D * D).sum() / n

    s = (A - B.sum()) / (C - E.sum())

    t = (R_B @ (T_A - s * T_B)).sum(0) / n

    T_A_hat = s * T_B + R_B.transpose(1, 2) @ t

    return T_A_hat.squeeze(2), s, t.transpose(1, 0)

def get_error(R_pred, T_pred, R_gt, T_gt, gt_scene_scale):
    T_A_hat, _, _ = compute_optimal_translation_alignment(T_gt, T_pred, R_pred)
    norm = torch.linalg.norm(T_gt - T_A_hat, dim=1) / gt_scene_scale
    norms = np.ndarray.tolist(norm.detach().cpu().numpy())
    norms = np.array(norms)
    return norms, T_A_hat

def rotation_distance(R1, R2, eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2, -1)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    angle = (0.5 * (trace - 1.)).clamp(-1.+eps, 1.-eps).acos_() # numerical stability near -1/+1
    return angle * 180 / np.pi

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

    return d_theta, d_azimuth, d_z

def get_T_my(RT_mtx):
    R, T = RT_mtx[:3, :3], RT_mtx[:, -1]
    RT_mtx_w2c = -R.T @ T
    elevation, azimuth, z = cartesian_to_spherical(RT_mtx_w2c[None, :])
    elevation = elevation # [0, 180]
    azimuth = azimuth # [0, 360]
    return elevation, azimuth, z

# def evaluation_relpose(
#     model_dir  : str,
#     image_dirs : str,
#     mask_dirs  : str,
#     pose_dirs  : str,
#     num_to_eval: int,
#     device     : str = 'cuda',
# ):
#     # * load pretrained weights
#     model, args = get_model(
#         model_dir=model_dir,
#         num_images=num_to_eval,
#         device=device,
#     )

#     rotation_error_all    = []
#     translation_error_all = []
#     elevation_error_all   = []
#     azimuth_error_all     = []
#     r_error_all           = []

#     for image_dir, mask_dir, pose_dir in zip(image_dirs, mask_dirs, pose_dirs):
#         error_Rs = []
#         error_ts = []
#         RT_pred = []
#         elevation_errors, azimuth_errors, r_errors = [], [], []
#         for i in range(1):
#             class_name = image_dir.split('/')[-2]
#             # * load in the wild images and crop parameters
#             dataset = CustomDataset(
#                 image_dir=image_dir,
#                 mask_dir=mask_dir,
#             )
#             num_frames = num_to_eval
#             batch = dataset.get_data(ids=np.arange(num_frames))
#             images = batch["image"].to(device)
#             crop_params = batch["crop_params"].to(device)

#             # * quickly initialize a coarse set of poses using MST reasoning
#             batched_images, batched_crop_params = images.unsqueeze(0), crop_params.unsqueeze(0)
#             _, hypothesis = evaluate_mst(
#                 model=model,
#                 images=batched_images,
#                 crop_params=batched_crop_params,
#             )
#             R_pred = np.stack(hypothesis)

#             # * regress to optimal translation
#             with torch.no_grad():
#                 _, _, T_pred = model(
#                     images=batched_images,
#                     crop_params=batched_crop_params,
#                 )

#             # * search for optimal rotation via coordinate ascent.
#             R_pred_rel, hypothesis = evaluate_coordinate_ascent(
#                 model=model,
#                 images=batched_images,
#                 crop_params=batched_crop_params,
#             )
#             R_final = torch.from_numpy(np.stack(hypothesis))

#             R_final_pred = []
#             T_final_pred = []
#             for i in range(num_to_eval):
#                 R_final_i = R_final[i].T
#                 T_pred_i = T_pred[i].cpu()
#                 pose_i = torch.cat([R_final_i, T_pred_i[...,None]], dim=-1).cpu().numpy()
#                 pose_i = torch.from_numpy((pose_i))

#                 # * ++++++++++++++ important ++++++++++++++
#                 asd_i = pose_i[:3, :3]
#                 R_asd = torch.stack([asd_i[:, 0], asd_i[:, 2], -asd_i[:, 1]], dim=0)
#                 qwe_i = pose_i[:3, 3]
#                 # T_qwe = torch.cat([-qwe_i[1:2], -qwe_i[0:1], qwe_i[2:3]], dim=-1)
#                 T_qwe = torch.cat([qwe_i[0:1], qwe_i[1:2], qwe_i[2:3]], dim=-1)
#                 # * ++++++++++++++ important ++++++++++++++

#                 R_final_pred.append(R_asd)
#                 T_final_pred.append(T_qwe)

#             R_final_pred = torch.stack(R_final_pred)
#             T_final_pred = torch.stack(T_final_pred)

#             R_final_gt = []
#             T_final_gt = []
                
#             for i in range(num_to_eval):
#                 pose = np.load(f'{pose_dir}/{i:03d}.npy')

#                 R_final_gt.append(pose[:3, :3])
#                 T_final_gt.append(torch.tensor(pose[:3, 3]))

#             R_final_gt = np.stack(R_final_gt)
#             T_final_gt = torch.stack(T_final_gt)
            
#             # # ! compute metrics
#             # permutations = get_permutations(num_frames, eval_time=True)
#             # n_p = len(permutations)
#             # relative_rotation = np.zeros((n_p, 3, 3))
#             # for k, t in enumerate(permutations):
#             #     i, j = t
#             #     relative_rotation[k] = R_final_gt[i].T @ R_final_gt[j]
#             # R_gt_rel = relative_rotation

#             # permutations = get_permutations(num_frames, eval_time=True)
#             # n_p = len(permutations)
#             # relative_rotation = np.zeros((n_p, 3, 3))
#             # for k, t in enumerate(permutations):
#             #     i, j = t
#             #     relative_rotation[k] = R_final_pred[i].T @ R_final_pred[j]
#             # R_final_pred_rel = relative_rotation

#             # error_R = compute_angular_error_batch(R_final_pred_rel, R_gt_rel)

#             # R_pred_n = get_n_consistent_cameras(torch.from_numpy(R_pred_rel), num_frames)
#             # gt_scene_scale = full_scene_scale(torch.from_numpy(R_final_gt), T_final_gt)
#             # error_t, A_hat = get_error(R_pred_n.float(), T_pred.cpu().float(), R_final_gt, T_final_gt.float(), gt_scene_scale)

#             # error_R = np.mean(error_R)
#             # error_t = np.mean(error_t)
#             # error_Rs.append(error_R)
#             # error_ts.append(error_t)
            
#             RT_final_gt = []
#             RT_final_pred = []
#             for i in range(num_to_eval):
#                 RT_final_pred.append(
#                     np.concatenate([R_final_pred[i], T_final_pred[i][..., None]], axis=-1)
#                 )
#                 RT_final_gt.append(
#                     np.concatenate([R_final_gt[i], T_final_gt[i][..., None]], axis=-1)
#                 )

#             RT_final_gt = np.stack(RT_final_gt)
#             RT_final_pred = np.stack(RT_final_pred)
#             RT_pred.append(RT_final_pred)

#             d_elevation_preds, d_azimuth_preds, d_r_preds = [], [], []
#             d_elevation_gts, d_azimuth_gts, d_r_gts = [], [], []
            
#             # elevation_error, azimuth_error, r_error = [], [], []
#             for i in range(num_to_eval):
#                 d_elevation_pred, d_azimuth_pred, d_r_pred = get_T(RT_final_pred[i], RT_final_pred[0])
#                 d_elevation_gt, d_azimuth_gt, d_r_gt = get_T(RT_final_gt[i], RT_final_gt[0])

#                 d_elevation_preds.append(d_elevation_pred)                
#                 d_azimuth_preds.append(d_azimuth_pred)                
#                 d_r_preds.append(d_r_pred)                
#                 d_elevation_gts.append(d_elevation_gt)                
#                 d_azimuth_gts.append(d_azimuth_gt)                
#                 d_r_gts.append(d_r_gt)                

#         #     e_errors = []
#         #     a_errors = []
#         #     r_errors = []

#         #     for i in range(num_to_eval):
#         #         for j in range(num_to_eval):
#         #             mm = np.abs(d_elevation_preds[i] - d_elevation_preds[j])
#         #             nn = np.abs(d_azimuth_preds[i] - d_azimuth_preds[j])
#         #             kk = np.abs(d_r_preds[i] - d_r_preds[j])
                    
#         #             qq = np.abs(d_elevation_gts[i] - d_elevation_gts[j])
#         #             ww = np.abs(d_azimuth_gts[i] - d_azimuth_gts[j])
#         #             ee = np.abs(d_r_gts[i] - d_r_gts[j])
                    
#         #             e_error = np.abs(mm - qq)
#         #             a_error = np.abs(nn - ww)
#         #             r_error = np.abs(kk - ee)
                    
#         #             e_errors.append(e_error)
#         #             a_errors.append(a_error)
#         #             r_errors.append(r_error)
            
#         #     elevation_error = np.mean(e_errors)
#         #     azimuth_error = np.mean(a_errors)
#         #     radius_error = np.mean(r_errors)

#         #     elevation_errors.append(elevation_error)
#         #     azimuth_errors.append(azimuth_error)
#         #     r_errors.append(r_error)

#         # elevation_error = np.mean(elevation_errors)
#         # azimuth_error = np.mean(azimuth_errors)
           
#         # error_R = np.mean(error_Rs)
#         # error_t = np.mean(error_ts)

#         # r_error = np.mean(r_errors)
#         # print(f'class: {class_name} rotation error: {error_R:.5f} translation error: {error_t:.3f} azimuth_error: {np.rad2deg(azimuth_error):.3f} elevation_error: {np.rad2deg(elevation_error):.3f}')

#         # sort_index = np.array(error_Rs).argsort()[::-1][17]
#         # RT_pred_select = RT_pred[sort_index]
#         # os.makedirs(f'pose', exist_ok=True)
#         # np.save(f'pose/{class_name}.npy',RT_pred_select)

#         rotation_error_all.append(error_R)
#         translation_error_all.append(error_t)
#         elevation_error_all.append(elevation_error)
#         azimuth_error_all.append(azimuth_error)
#         r_error_all.append(r_error)

#     rotation_error_all    = np.mean(rotation_error_all)
#     translation_error_all = np.mean(translation_error_all)
#     elevation_error_all   = np.mean(elevation_error_all)
#     azimuth_error_all     = np.mean(azimuth_error_all)
#     r_error_all           = np.mean(r_error_all)

#     return rotation_error_all, translation_error_all, np.rad2deg(elevation_error_all), np.rad2deg(azimuth_error_all), r_error_all

def evaluation_relpose(
    model_dir  : str,
    image_dirs : str,
    mask_dirs  : str,
    pose_dirs  : str,
    num_to_eval: int,
    device     : str = 'cuda',
):
    # * load pretrained weights
    model, args = get_model(
        model_dir=model_dir,
        num_images=num_to_eval,
        device=device,
    )

    rotation_error_all    = []
    translation_error_all = []
    elevation_error_all   = []
    azimuth_error_all     = []
    r_error_all           = []

    for image_dir, mask_dir, pose_dir in zip(image_dirs, mask_dirs, pose_dirs):
        error_Rs = []
        error_ts = []
        RT_pred = []
        elevation_final, azimuth_final, r_final = [], [], []
        elevation_gt, azimuth_gt, r_gt = [], [], []
        for _ in range(2):
            class_name = image_dir.split('/')[-2]
            # * load in the wild images and crop parameters
            dataset = CustomDataset(
                image_dir=image_dir,
                mask_dir=mask_dir,
            )
            num_frames = num_to_eval
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
                # T_qwe = torch.cat([-qwe_i[1:2], -qwe_i[0:1], qwe_i[2:3]], dim=-1)
                T_qwe = torch.cat([qwe_i[0:1], qwe_i[1:2], qwe_i[2:3]], dim=-1)
                # * ++++++++++++++ important ++++++++++++++

                R_final_pred.append(R_asd)
                T_final_pred.append(T_qwe)

            R_final_pred = torch.stack(R_final_pred)
            T_final_pred = torch.stack(T_final_pred)

            R_final_gt = []
            T_final_gt = []
                
            for i in range(num_to_eval):
                pose = np.load(f'{pose_dir}/{i:03d}.npy')

                R_final_gt.append(pose[:3, :3])
                T_final_gt.append(torch.tensor(pose[:3, 3]))

            R_final_gt = np.stack(R_final_gt)
            T_final_gt = torch.stack(T_final_gt)
            
            RT_final_gt = []
            RT_final_pred = []
            for i in range(num_to_eval):
                RT_final_pred.append(
                    np.concatenate([R_final_pred[i], T_final_pred[i][..., None]], axis=-1)
                )
                RT_final_gt.append(
                    np.concatenate([R_final_gt[i], T_final_gt[i][..., None]], axis=-1)
                )

            RT_final_gt = np.stack(RT_final_gt)
            RT_final_pred = np.stack(RT_final_pred)
            RT_pred.append(RT_final_pred)

            d_elevation_preds, d_azimuth_preds, d_r_preds = [], [], []
            d_elevation_gts, d_azimuth_gts, d_r_gts = [], [], []
            
            for i in range(num_to_eval):
                d_elevation_pred, d_azimuth_pred, d_r_pred = get_T(RT_final_pred[i], RT_final_pred[0])
                d_elevation_gt, d_azimuth_gt, d_r_gt = get_T(RT_final_gt[i], RT_final_gt[0])

                d_elevation_preds.append(d_elevation_pred)                
                d_azimuth_preds.append(d_azimuth_pred)                
                d_r_preds.append(d_r_pred)
                                
                d_elevation_gts.append(d_elevation_gt)                
                d_azimuth_gts.append(d_azimuth_gt)                
                d_r_gts.append(d_r_gt)
            
            print(d_azimuth_preds)
            print(d_azimuth_gts)
            
            elevation_final.append(d_elevation_preds)
            azimuth_final.append(d_azimuth_preds)
            r_final.append(d_r_preds)

            elevation_gt.append(d_elevation_gts)
            azimuth_gt.append(d_azimuth_gts)
            r_gt.append(d_r_gts)
            
        elevation_final = np.median(np.stack(elevation_final, axis=0), axis=0)[:,0]
        azimuth_final = np.median(np.stack(azimuth_final, axis=0), axis=0)[:,0]
        r_final = np.median(np.stack(r_final, axis=0), axis=0)[:,0] * 0.7 + 1.5

        elevation_gt = np.median(np.stack(elevation_gt, axis=0), axis=0)[:,0]
        azimuth_gt = np.median(np.stack(azimuth_gt, axis=0), axis=0)[:,0]
        r_gt = np.median(np.stack(r_gt, axis=0), axis=0)[:,0] * 0.7 + 1.5
        
        
        e_errors = []
        a_errors = []
        r_errors = []

        for i in range(num_to_eval):
            for j in range(num_to_eval):
                if i != j:
                    mm = np.abs(elevation_final[i] - elevation_final[j])
                    nn = np.abs(azimuth_final[i] - azimuth_final[j])
                    kk = np.abs(r_final[i] - r_final[j])
                    
                    qq = np.abs(elevation_gt[i] - elevation_gt[j])
                    ww = np.abs(azimuth_gt[i] - azimuth_gt[j])
                    ee = np.abs(r_gt[i] - r_gt[j])
                    
                    e_error = np.abs(mm - qq)
                    a_error = np.abs(nn - ww)
                    r_error = np.abs(kk - ee)
                    
                    e_errors.append(e_error)
                    a_errors.append(a_error)
                    r_errors.append(r_error)
        
        elevation_error = np.mean(e_errors)
        azimuth_error = np.mean(a_errors)
        radius_error = np.mean(r_errors)
        
        elevation_error_all.append(elevation_error)
        azimuth_error_all.append(azimuth_error)
        r_error_all.append(radius_error)
        
        # ! compute metrics   
        train_view = num_to_eval     
        c2w_gt = torch.cat([
            ear2rotation(
                torch.tensor(elevation_final[i]).float(), 
                torch.tensor(azimuth_final[i]).float(), 
                torch.tensor(r_final[i]).float(), 
            ) 
        for i in range(train_view)])
        
        c2w_pred = torch.cat([
            ear2rotation(
                torch.tensor(elevation_gt[i]).float(), 
                torch.tensor(azimuth_gt[i]).float(), 
                torch.tensor(r_gt[i]).float(), 
            ) 
        for i in range(train_view)])

        permutations = get_permutations(train_view, eval_time=True)
        n_p = len(permutations)
        relative_rotation = np.zeros((n_p, 3, 3))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] = c2w_gt[i,:3,:3].T @ c2w_gt[j,:3,:3]
        R_gt_rel = relative_rotation

        n_p = len(permutations)
        relative_rotation = np.zeros((n_p, 3, 3))
        for k, t in enumerate(permutations):
            i, j = t
            relative_rotation[k] = c2w_pred[i,:3,:3].T @ c2w_pred[j,:3,:3]
        R_pred_rel = relative_rotation
        error_R = compute_angular_error_batch(R_pred_rel, R_gt_rel)
        
        rotation_error_all.append(np.mean(error_R))

    rotation_error_all = np.mean(rotation_error_all)
    translation_error_all = 0.0
    elevation_error_all = np.mean(elevation_error_all)
    azimuth_error_all = np.mean(azimuth_error_all)
    r_error_all = np.mean(r_error_all)

    return rotation_error_all, translation_error_all, np.rad2deg(elevation_error_all), np.rad2deg(azimuth_error_all), r_error_all

def inference_evaluation(
    data_dir   : str,
    model_dir  : str,
    num_to_eval: int,
    device     : str = 'cuda',
):
    image_dirs, mask_dirs, pose_dirs = [], [], []
    for i in sorted(os.listdir(data_dir)):
        image_dirs.append(f'{data_dir}/{i}/images')
        mask_dirs.append(f'{data_dir}/{i}/masks')
        pose_dirs.append(f'{data_dir}/{i}/poses')

    rotation_error_all, translation_error_all, elevation_deg_error, azimuth_deg_error, radius_error = evaluation_relpose(
        model_dir,
        image_dirs,
        mask_dirs,
        pose_dirs,
        num_to_eval,
        device,
    )
    ret = {
        "rotation_error_all"   : rotation_error_all,
        "translation_error_all": translation_error_all,
        "elevation_deg_error": elevation_deg_error,
        "azimuth_deg_error": azimuth_deg_error,
        "radius_error": radius_error,
    }
    return ret

def inference_relpose(
    model,
    image_dir  : str,
    mask_dir   : str,
    num_to_eval: int,
    device     : str = 'cuda',
):

    # * load in the wild images and crop parameters
    dataset = CustomDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        num_to_eval=num_to_eval,
    )
    num_frames = num_to_eval
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
    # print("Iteratively finetuning the initial MST solution.")
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
        # T_qwe = torch.cat([-qwe_i[1:2], -qwe_i[0:1], qwe_i[2:3]], dim=-1)
        T_qwe = torch.cat([qwe_i[0:1], qwe_i[1:2], qwe_i[2:3]], dim=-1)
        # * ++++++++++++++ important ++++++++++++++

        R_final_pred.append(R_asd)
        T_final_pred.append(T_qwe)

    R_final_pred = torch.stack(R_final_pred)
    T_final_pred = torch.stack(T_final_pred)
    R_final_pred = (R_final_pred)

    RT_final_pred = []
    for i in range(num_to_eval):
        RT_final_pred.append(
            np.concatenate([R_final_pred[i], T_final_pred[i][..., None]], axis=-1)
        )
    RT_final_pred = np.stack(RT_final_pred)

    pose_pred = []
    for i in range(num_to_eval):
        pose_pred.append(
            get_T_my(RT_final_pred[i])
        )
    pose_pred = np.stack(pose_pred)[..., 0]
    elevations = []
    azimuths = []
    radius = []
    for i in range(num_to_eval):
        elevations.append(
            pose_pred[i][0].item() / np.pi,
        )
        azimuths.append(
            (pose_pred[i][1].item() - pose_pred[0][1].item()) % (2*np.pi) / (2*np.pi)
        )
        radius.append(
            (pose_pred[i][2].item() - 1.5)/ 0.7,
        )
    return elevations, azimuths, radius

def best_inference(
    model_dir,
    image_dir,
    mask_dir,
    num_to_eval=3,
    inference_time=25,
    device=0,
):
    # * load pretrained weights
    model, _ = get_model(
        model_dir=model_dir,
        num_images=num_to_eval,
        device=device,
    )
    # * inference model
    d_elevations, d_azimuths, d_rs = [], [], []

    for _ in tqdm.tqdm(range(inference_time), desc='searching camera poses'):
        d_elevation, d_azimuth, d_r = inference_relpose(
            model=model,
            image_dir=image_dir,
            mask_dir=mask_dir,
            num_to_eval=num_to_eval,
        )
        d_elevations.append(d_elevation)
        d_azimuths.append(d_azimuth)
        d_rs.append(d_r)

    d_elevations = np.stack(d_elevations, 0)
    d_azimuths = np.stack(d_azimuths, 0)
    d_rs = np.stack(d_rs, 0)
    d_elevation_pred = np.median(d_elevations, axis=0)
    d_azimuth_pred = np.median(d_azimuths, axis=0)
    d_r_pred = np.median(d_rs, axis=0)
    return d_elevation_pred, d_azimuth_pred, d_r_pred

def best_evaluation(
    model_dir,
    image_dirs,
    mask_dirs,
    num_to_eval=3,
    inference_time=25,
    device=0,
):

    # * inference model
    d_elevations, d_azimuths, d_rs = [], [], []

    for _ in tqdm.tqdm(range(inference_time), desc='searching camera poses'):
        d_elevation, d_azimuth, d_r = inference_relpose(
            model=model,
            image_dir=image_dir,
            mask_dir=mask_dir,
            num_to_eval=num_to_eval,
        )
        d_elevations.append(d_elevation)
        d_azimuths.append(d_azimuth)
        d_rs.append(d_r)

    d_elevations = np.stack(d_elevations, 0)
    d_azimuths = np.stack(d_azimuths, 0)
    d_rs = np.stack(d_rs, 0)
    d_elevation_pred = np.median(d_elevations, axis=0)
    d_azimuth_pred = np.median(d_azimuths, axis=0)
    d_r_pred = np.median(d_rs, axis=0)
    return d_elevation_pred, d_azimuth_pred, d_r_pred

def get_configures():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--object_type', type=str, default='GSO')
    parser.add_argument('--object_name', type=str, default='BUNNY_RACER')
    parser.add_argument('--num_to_eval', type=int, default=3)
    parser.add_argument('--inference_time', type=int, default=25)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--backbone', type=str, default=False)
    parser.add_argument('--use_org', type=str, default=False)

    return parser.parse_args()

if __name__ == '__main__':
    opt = get_configures()
    if opt.mode == 'infer':
        object_path = f'../dataset/data/train/{opt.object_type}/{opt.object_name}'
        config_file_path = f'../camerabooth/configs/{opt.object_type}/config_{opt.object_name}_view_{opt.num_to_eval}.yaml'
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            print(config_file_path)
            raise RuntimeError('File not exist!')
        
        num_to_eval = opt.num_to_eval
        if opt.use_org == 'yes':
            mp = '/shared/xinyang/threetothreed/relposepp/ckpt_wild/relposepp_masked'
        else:
            mp = "relpose/ckpts_finetune/best"   
            
        if opt.backbone == 'yes':
            mp = 'relpose/ckpt_backbone'

        d_elevation_pred, d_azimuth_pred, d_r_pred = best_inference(
            mp,
            f'{object_path}/images',
            f'{object_path}/masks',
            num_to_eval=num_to_eval,
            inference_time=opt.inference_time,
            device=opt.device_id,
        )
        # * one2345 elevation adjust
        client = Client("https://one-2-3-45-one-2-3-45.hf.space/")
        input_img_path = f'{object_path}/images/002.png'
        best_elevation_list = []
        # for i in range(5):
            # elevation_angle_deg = client.predict(input_img_path, True, api_name="/estimate_elevation")
            # best_elevation_list.append(elevation_angle_deg)
        best_elevation_list = client.predict(input_img_path, True, api_name="/estimate_elevation")
        d_elevation_pred = d_elevation_pred - d_elevation_pred[0] + (90.0-best_elevation_list) / 180.

        print(f'{opt.object_type} {opt.object_name} Estimate Pose', d_elevation_pred, d_azimuth_pred, d_r_pred)
        
        config_data["model"]["params"]["a_init"] = d_elevation_pred.tolist()
        config_data["model"]["params"]["b_init"] = d_azimuth_pred.tolist()
        config_data["model"]["params"]["c_init"] = d_r_pred.tolist()

        with open(config_file_path, "w") as f:
            yaml.safe_dump(config_data, f)
        
    elif opt.mode == 'eval':
        # * load pretrained weights
        model, _ = get_model(
            model_dir=model_dir,
            num_images=num_to_eval,
            device=device,
        )
        
        image_dirs, mask_dirs, pose_dirs = [], [], []
        for i in sorted(os.listdir(data_dir)):
            image_dirs.append(f'{data_dir}/{i}/images')
            mask_dirs.append(f'{data_dir}/{i}/masks')
            pose_dirs.append(f'{data_dir}/{i}/poses')
            
        for image_dir, mask_dir, pose_dir in image_dirs, mask_dirs, pose_dirs:
            d_elevation_pred, d_azimuth_pred, d_r_pred = best_evaluation(
                model,
                f'{object_path}/images',
                f'{object_path}/masks',
                num_to_eval=num_to_eval,
                inference_time=opt.inference_time,
                device=opt.device_id,
            )
                
            for i in range(num_to_eval):
                pose = np.load(f'{pose_dir}/{i:03d}.npy')
                R_final_gt.append(pose[:3, :3])
                T_final_gt.append(pose[:3, 3])
            
            RT_final_gt = np.stack(RT_final_gt)
            RT_final_pred = np.stack(RT_final_pred)
            RT_pred.append(RT_final_pred)

            d_elevation_preds, d_azimuth_preds, d_r_preds = [], [], []
            d_elevation_gts, d_azimuth_gts, d_r_gts = [], [], []
            
            for i in range(num_to_eval):
                d_elevation_pred, d_azimuth_pred, d_r_pred = get_T(RT_final_pred[i], RT_final_pred[0])
                d_elevation_gt, d_azimuth_gt, d_r_gt = get_T(RT_final_gt[i], RT_final_gt[0])
            
            delta_elevation_gt = [
                elevation_gt[i] - elevation_gt[0] for i in range(0, num_to_eval)
            ]
            delta_azimuth_gt = [
                azimuth_gt[i] - azimuth_gt[0] for i in range(0, num_to_eval) 
            ]
            delta_radius_gt = [
                radius_gt[i] for i in range(0, num_to_eval) 
            ]
            
            delta_elevation_pred = [
                d_elevation_pred[i] - d_elevation_pred[0] for i in range(0, num_to_eval)
            ]
            delta_azimuth_pred = [
                d_azimuth_pred[i] - d_azimuth_pred[0] for i in range(0, num_to_eval)
            ]
            delta_radius_pred = [
                d_r_pred[i]  for i in range(0, num_to_eval)
            ]
            
            
        
        # rot_error = []
        # trans_error = []
        # elevation_error = []
        # azimuth_error = []
        # radius_error = []
        # # for i in range(1):
        # ret = inference_evaluation(
        #     # model_dir="ckpts_finetune/best",
        #     # model_dir="ckpt_backbone",
        #     model_dir='/shared/xinyang/threetothreed/relposepp/ckpt_wild/relposepp_masked',
        #     data_dir="../dataset/data/train/GSO_demo",
        #     # data_dir="../dataset/data/train/GSO_demo",
        #     # data_dir='/shared/xinyang/threetothreed/relposepp/data/objaverse_eval',
        #     num_to_eval=3,
        # )
        #     rot_error.append(ret['rotation_error_all'])
        #     trans_error.append(ret['translation_error_all'])
        #     elevation_error.append(ret['elevation_deg_error'])
        #     azimuth_error.append(ret['azimuth_deg_error'])
        #     radius_error.append(ret['radius_error'])
        # print(np.mean(rot_error))
        # print(np.mean(trans_error))
        # print(np.mean(elevation_error))
        # print(np.mean(azimuth_error))
        # print(np.mean(radius_error))
        # print(f"rot_error: {ret['rotation_error_all']:.3f}, trans_error: {ret['translation_error_all'] * 100:.3f}")
        # print(f"elevation_error: {ret['elevation_deg_error']:.3f}, azimuth_error: {ret['azimuth_deg_error']:.3f}, radius_error: {ret['radius_error']:.3f}")
        
        
    # # * load pretrained weights
    # model_dir = 'ckpts_finetune/1026_0506_LR1e-05_N8_RandomNTrue_B36_Pretrainedckpt_back_AMP_TROURS_DDP'
    # model, _ = get_model(
    #     model_dir=model_dir,
    #     num_images=3,
    #     device=0,
    # )
    # # * inference model
    # d_elevations, d_azimuths, d_rs = [], [], []

    # for i in range(5):
    #     d_elevation, d_azimuth, d_r = inference_relpose(
    #         model=model,
    #         image_dir="../dataset/data/train/GSO/BUNNY_RACER/images",
    #         mask_dir="../dataset/data/train/GSO/BUNNY_RACER/masks",
    #         num_to_eval=3,
    #     )
    #     d_elevations.append(d_elevation)
    #     d_azimuths.append(d_azimuth)
    #     d_rs.append(d_r)

    # d_elevations = np.stack(d_elevations, 0)[..., 0]
    # d_azimuths = np.stack(d_azimuths, 0)[..., 0]
    # d_rs = np.stack(d_rs, 0)[..., 0]
    # d_elevation_pred = np.median(d_elevations, axis=0)
    # d_azimuth_pred = np.median(d_azimuths, axis=0)
    # d_r_pred = np.median(d_rs, axis=0)
    # print(d_elevation_pred)
    # print(d_azimuth_pred)
    # print(d_r_pred)
    
    # # * see ground truth pose
    # pose_paths = [f"../dataset/data/train/GSO/BUNNY_RACER/poses/{i:03d}.npy" for i in range(3)]
    # RT_gt = [np.load(pose_path) for pose_path in pose_paths]
    # d_elevations, d_azimuths, d_rs = [], [], []
    # for i in range(3):
    #     d_elevation, d_azimuth, d_r = get_T(RT_gt[i], RT_gt[0])

    #     d_elevations.append(d_elevation / np.pi)
    #     d_azimuths.append(d_azimuth / (2.0 * np.pi))
    #     d_rs.append(d_r)
    # print(d_elevations)
    # print(d_azimuths)
    # print(d_rs)
    # rot_error = []
    # trans_error = []
 
    # rot_error = np.mean(rot_error)
    # trans_error = np.mean(trans_error) * 100.
    # print(f"rot_error: {rot_error:.3f}, trans_error: {trans_error:.3f}")