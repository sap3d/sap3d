import os
import re
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pdb
import argparse
import json

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

def find_evaluation_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            if dir_name == "evaluation":
                evaluation_folder_path = os.path.join(root, dir_name)
                return evaluation_folder_path

def ear2rotation(elevation, azimuth, camera_distances):
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )[None, :]
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

def update_results_in_json(new_data, path):
    """Update specific dictionary within a list in a JSON file with new key-value pairs."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing_data = json.load(f)
        # Check if the data structure is a list of dictionaries
        if isinstance(existing_data, list) and all(isinstance(item, dict) for item in existing_data):
            # Try to find the dictionary with the matching NAME
            for item in existing_data:
                item.update(new_data)  # Update the dictionary with new key-value pairs
                break
            else:
                # If no matching NAME is found, append new data as a new dictionary
                existing_data.append(new_data)
        else:
            print("Error: JSON structure is not a list of dictionaries")
            return
    else:
        # If the file does not exist, initialize it with the new data in a list
        existing_data = [new_data]
    
    with open(path, 'w') as f:
        json.dump(existing_data, f, indent=4)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="Evaluate 3D meshes.")
    parser.add_argument('--OBJECT_TYPE', type=str, required=True, help='The type of the object')
    parser.add_argument('--OBJECT_NAME', type=str, required=True, help='The name of the object')
    parser.add_argument('--OBJECT_VIEW', type=int, required=True, help='The view number of the object')
    parser.add_argument('--ROOT_DIR', type=str, required=True, help='The root directory for saving results')
    args = parser.parse_args()              
    object_type = args.OBJECT_TYPE
    object_name = args.OBJECT_NAME
    object_view = args.OBJECT_VIEW
    root_dir = args.ROOT_DIR
    train_view = object_view

    results_directory = os.path.join(root_dir, "results", object_type, object_name, str(object_view))
    os.makedirs(results_directory, exist_ok=True)
    results_file_path = os.path.join(results_directory, "results.json")

    epoch_folder_pattern = re.compile(r'epoch_(\d+)')

    root_folder   = f"experiments_{object_type}_view_{train_view}"
    exp_log_paths = [f'{root_folder}/{i}' for i in sorted(os.listdir(root_folder))]
    # class_names = [f'{i}' for i in sorted(os.listdir(root_folder))]

    mean_psnr_before , mean_psnr_after  = [], []
    mean_lpips_before, mean_lpips_after = [], []
    mean_e_beofre, mean_e_after = [], []
    mean_a_beofre, mean_a_after = [], []
    mean_r_beofre, mean_r_after = [], []
    
    elevation_preds, elevation_gts = [], []
    azimuth_preds, azimuth_gts = [], [] 
    radius_preds, radius_gts = [], []
    
    all_error_R = []

    pdb.set_trace()
    class_name = object_name
    # for ind, class_name in enumerate(class_names):
    exp_log_path = f'{root_folder}/{class_name}'
    evaluation_folder_path = find_evaluation_folder(exp_log_path)
    if evaluation_folder_path:
        zero123_results = None
        ours_results = None
        for i in os.listdir(evaluation_folder_path):
            match = epoch_folder_pattern.match(i)
            epoch_number = int(match.group(1))
            if epoch_number == 0:
                zero123_results = torch.load(f'{evaluation_folder_path}/epoch_{epoch_number}/results.tar')
            else:
                ours_results = torch.load(f'{evaluation_folder_path}/epoch_{epoch_number}/results.tar')

            print(f'find log: {class_name}')
            config_path = f'{root_dir}/camerabooth/configs/{object_type}/config_{class_name}_view_{train_view}.yaml'
            if not os.path.exists(config_path):
                continue
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            elevation_pred = ours_results['elevation_pred'] 
            azimuth_pred = ours_results['azimuth_pred'] 
            radius_pred = ours_results['radius_pred'] 

            elevation_gt = np.array(config_data["lightning"]["callbacks"]["image_logger"]["params"]["a_gt"]) * np.pi
            azimuth_gt = np.array(config_data["lightning"]["callbacks"]["image_logger"]["params"]["b_gt"]) * 2.0 * np.pi
            radius_gt = np.array(config_data["lightning"]["callbacks"]["image_logger"]["params"]["c_gt"]) * 0.7 + 1.5
            
            delta_elevation_gt = [
                elevation_gt[i] - elevation_gt[0] for i in range(0, train_view)
            ]
            delta_azimuth_gt = [
                azimuth_gt[i] - azimuth_gt[0] for i in range(0, train_view) 
            ]
            delta_radius_gt = [
                radius_gt[i] for i in range(0, train_view) 
            ]
            
            delta_elevation_pred = [
                elevation_pred[i] - elevation_pred[0] for i in range(0, train_view)
            ]
            delta_azimuth_pred = [
                azimuth_pred[i] - azimuth_pred[0] for i in range(0, train_view)
            ]
            delta_radius_pred = [
                radius_pred[i]  for i in range(0, train_view)
            ]

            num_v = train_view
            e_errors = []
            a_errors = []
            r_errors = []

            for i in range(num_v):
                for j in range(num_v):
                    if i != j:
                        mm = np.abs(delta_elevation_gt[i] - delta_elevation_gt[j])
                        nn = np.abs(delta_azimuth_gt[i] - delta_azimuth_gt[j])
                        kk = np.abs(delta_radius_gt[i] - delta_radius_gt[j])
                        
                        qq = np.abs(delta_elevation_pred[i] - delta_elevation_pred[j])
                        ww = np.abs(delta_azimuth_pred[i] - delta_azimuth_pred[j])
                        ee = np.abs(delta_radius_pred[i] - delta_radius_pred[j])
                        
                        e_error = np.abs(mm - qq)
                        a_error = np.abs(nn - ww)
                        r_error = np.abs(kk - ee)
                        
                        e_errors.append(e_error)
                        a_errors.append(a_error)
                        r_errors.append(r_error)
                
            elevation_error = np.mean(e_errors)
            azimuth_error = np.mean(a_errors)
            radius_error = np.mean(r_errors)

            mean_e_after.append(elevation_error)
            mean_a_after.append(azimuth_error)
            mean_r_after.append(radius_error)
            
            c2w_gt = torch.cat([
                ear2rotation(
                    torch.tensor(delta_elevation_gt[i]).float(), 
                    torch.tensor(delta_azimuth_gt[i]).float(), 
                    torch.tensor(delta_radius_gt[i]).float(), 
                ) 
            for i in range(train_view)])
            
            c2w_pred = torch.cat([
                ear2rotation(
                    torch.tensor(delta_elevation_pred[i]).float(), 
                    torch.tensor(delta_azimuth_pred[i]).float(), 
                    torch.tensor(delta_radius_pred[i]).float(), 
                ) 
            for i in range(train_view)])
            
            # ! compute metrics
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
            
            all_error_R.append(np.mean(error_R))

    new_results = {
        "Rotation Error": np.mean(all_error_R),
        "Elevation": np.mean(mean_e_after) * 180 / np.pi,
        "Azimuth": np.mean(mean_a_after) * 180 / np.pi,
        "Radius": np.mean(mean_r_after)
    }
    update_results_in_json(new_results, results_file_path)

    print('View')
    print(train_view)
    print('Rotation Error')
    print(np.mean(all_error_R))
    print('ELEVATION')
    print(np.mean(mean_e_after) * 180 / np.pi)
    print('AZIMUTH')
    print(np.mean(mean_a_after) * 180 / np.pi)
    print('RADIUS')
    print(np.mean(mean_r_after))