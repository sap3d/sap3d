import os
import gzip
import json
import tqdm
import imageio
import numpy as np
from matplotlib import pyplot as plt
import pdb
from concurrent.futures import ThreadPoolExecutor, as_completed

def normalize_rotation_matrices(rotation):

    # 使用奇异值分解（SVD）分解矩阵 R
    U, S, Vt = np.linalg.svd(rotation)

    # 修正行列式为+1（确保正向旋转）
    det = np.linalg.det(U @ Vt)
    if det < 0:
        Vt[-1, :] *= -1  # 反转最后一行以修正行列式为+1

    # 重新构建归一化的旋转矩阵并添加到列表中
    normalized_R = U @ Vt

    return normalized_R

def normalize_translation(translation):

    normalized_tran = translation / np.linalg.norm(translation)

    return normalized_tran

def mask_to_bbox(mask):
    """
    xyxy format
    """
    mask[mask > 0.1] = 255
    if not np.any(mask):
        return []
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1], mask

def get_configures():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/shared/xinyang/views_release')
    parser.add_argument('--target_path', type=str, default='/shared/xinyang/zelin_dev/threetothreed/data/objaverse')
    parser.add_argument('--output_path', type=str, default='data/retrain_data')

    return parser.parse_args()

def process_file(image_png_path, pose_npy_path, focal_x, focal_y, principal_point_x, principal_point_y):
    mask = imageio.imread(image_png_path)[..., -1]
    result = mask_to_bbox(mask)

    if (result is not None) and (len(result) != 0):
        bbox, mask = result
        pose = np.load(pose_npy_path)
        r_mtx = pose[:3,:3].tolist()
        t_vec = pose[:3,3].tolist()
        focal = [focal_x, focal_y]
        principal_pt = [principal_point_x, principal_point_y]

        return {
            "filepath"       : image_png_path,
            "bbox"           : bbox,
            "R"              : r_mtx,
            "T"              : t_vec,
            "focal_length"   : focal,
            "principal_point": principal_pt,
        }

if __name__ == '__main__':
    args = get_configures()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    image_hight       = 512
    image_width       = 512
    camera_angle_x    = 0.857 # 49.1, default setted by zero123
    focal_x           = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
    focal_y           = 0.5 * image_hight / np.tan(0.5 * camera_angle_x)
    principal_point_x = image_width//2 * 1.0 - 0.5
    principal_point_y = image_hight//2 * 1.0 - 0.5
    # convert to co3d annotation format
    scale             = (min(image_width,image_hight) - 1.0) * 0.5
    focal_x           = 2.0 * focal_x / (image_width-1.0)
    focal_y           = 2.0 * focal_y / (image_hight-1.0)
    principal_point_x = (0.5 * (image_width-1.0) - principal_point_x) / scale
    principal_point_y = (0.5 * (image_hight-1.0) - principal_point_y) / scale

    view_release_dict  = {}

    # * for objaverse we select using clip
    all_class_set = set()
    data_folders = [
        '../dataset/data/train/GSO',
        '../dataset/data/train/XINYANG_NEW',
        '../dataset/data/train/ANGJOO',
        '../dataset/data/train/NAVI',
    ]
    for data_folder in data_folders:
        data_paths  = [f'{data_folder}/{i}' for i in os.listdir(data_folder)]
        for data_path in data_paths:
            # * top 20 from low to high
            query_match_paths = np.load(f'{data_path}/query_class/query_match_paths.npy')
            for query_match_path in query_match_paths:
                sub_class = query_match_path.split('/')[-1]
                all_class_set.add(sub_class)
    
    gso_paths = [f'../dataset/data/objaverse/{i}' for i in all_class_set]
    
    entire_objaverse_root = "/shared/xinyang/objaverse_rendering_whole/views_release"
    objaverse_paths = os.listdir(entire_objaverse_root)
    num_of_imgs = 12
    
    for name in tqdm.tqdm(objaverse_paths):
        gso_path = os.path.join(entire_objaverse_root, name)
        if len(os.listdir(gso_path)) == 0:
            continue
        view_cls = gso_path.split('/')[-1]
        view_data = []

        image_png_paths = [f'{gso_path}/{i:03d}.png' for i in range(num_of_imgs)]
        pose_npy_paths = [f'{gso_path}/{i:03d}.npy' for i in range(num_of_imgs)]

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(process_file, image_png_path, pose_npy_path, focal_x, focal_y, principal_point_x, principal_point_y)
                for image_png_path, pose_npy_path in zip(image_png_paths, pose_npy_paths)
            ]
            
            for future in as_completed(futures):
                data = future.result()
                if data:
                    view_data.append(data)

        if len(view_data) == num_of_imgs:
            view_release_dict[f'{view_cls}'] = view_data

    output_file = f'{args.output_path}/objaverse_train.jgz'
    with gzip.open(output_file, "w") as f:
        print(f'Saving objaverse_train.jgz at {args.output_path}')
        print(len(view_release_dict))
        f.write(json.dumps(view_release_dict).encode("utf-8"))