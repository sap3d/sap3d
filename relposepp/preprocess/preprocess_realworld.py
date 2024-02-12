import os
import gzip
import json
import tqdm
import imageio
import numpy as np
from matplotlib import pyplot as plt

def mask_to_bbox(mask):
    """
    xyxy format
    """
    mask[mask > 0.1]  = 255
    mask[mask == 0.1] = 255
    mask[mask < 0.1]  = 0
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
    parser.add_argument('--input_path', type=str, default='../dataset/data/CAPTURE')
    parser.add_argument('--output_path', type=str, default='data/finetune_data')

    return parser.parse_args()

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

    view_release_paths = [f'{args.input_path}/{i}' for i in tqdm.tqdm(sorted(os.listdir(args.input_path)))]
    view_release_dict  = {}
    test_dict          = {}
    for view_release_path in tqdm.tqdm(view_release_paths):
        view_data = []
        view_cls  = view_release_path.split('/')[-1]
        image_png_paths = [
            f'{view_release_path}/images/000.png',
            f'{view_release_path}/images/001.png',
            f'{view_release_path}/images/002.png',
        ]

        mask_png_paths = [
            f'{view_release_path}/masks/000.png',
            f'{view_release_path}/masks/001.png',
            f'{view_release_path}/masks/002.png',
        ]
        os.makedirs(f'{view_release_path}/masks', exist_ok=True)

        for index in range(3):
            image_png_path = image_png_paths[index]
            # compute mask and bbox
            mask   = imageio.imread(image_png_path)[..., -1]
            result = mask_to_bbox(mask)

            if result is not None:
                bbox, mask = result

                # save mask for evlauation
                imageio.imsave(mask_png_paths[index], mask)