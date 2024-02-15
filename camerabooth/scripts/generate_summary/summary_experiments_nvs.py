import os
import re
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import pdb
import argparse
import json
import lpips
from PIL import Image
import imageio
from math import exp
from torch.autograd import Variable

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

def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def _ssim(img1, img2, window, window_size, channel, mask=None, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if not (mask is None):
        b = mask.size(0)
        ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
        ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(b, -1).sum(dim=1).clamp(
            min=1
        )
        return ssim_map

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=None):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1,
            img2,
            window,
            self.window_size,
            channel,
            mask,
            self.size_average,
        )


def ssim(img1, img2, window_size=11, mask=None, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, mask, size_average)

def read_and_convert_image(image_path, size=(256, 256), background_color=(255, 255, 255)):
    image = imageio.imread(image_path)

    if image.shape[-1] == 4:
        image = Image.fromarray(image)
        rgb_image = Image.new('RGB', image.size, background_color)
        rgb_image.paste(image, mask=image.split()[3])
        image = rgb_image
        image = np.array(image)

    image = Image.fromarray(image)
    image = image.resize(size)
    image = np.array(image)

    return image

def calculate_metrics(predict_path, groundtruth_path):
    if not os.path.exists(predict_path):
        print(f"Predicted images folder not found: {predict_path}")
        return
    if not os.path.exists(groundtruth_path):
        print(f"Ground truth folder not found: {groundtruth_path}")
        return

    psnr_metric  = lambda x, y : -10. * np.log10(np.mean((x-y)**2))
    lpips_metric = lpips.LPIPS(net='alex')
    ssim_metric  = ssim

    psnr_all = []
    ssim_all = []
    lpips_all = []

    image_preds = []
    image_gts = []

    for i in range(84):
        pred_image = read_and_convert_image(f'{predict_path}/preds/{i:03}.png')
        image_preds.append(torch.from_numpy(pred_image / 255.).float().permute([2, 0, 1]))
        # os.makedirs(f'{predict_path}/converted_preds', exist_ok=True)
        # Image.fromarray(pred_image).save(f'{predict_path}/converted_preds/{i:03}.png')
        # print(f'{predict_path}/converted_preds/{i:03}.png')

        gt_image = read_and_convert_image(f'{groundtruth_path}/images/{i:03}.png')
        image_gts.append(torch.from_numpy(gt_image / 255.).float().permute([2, 0, 1]))
        # os.makedirs(f'{groundtruth_path}/converted_images', exist_ok=True)
        # Image.fromarray(gt_image).save(f'{groundtruth_path}/converted_images/{i:03}.png')
        # print(f'{groundtruth_path}/converted_images/{i:03}.png')
        # pdb.set_trace()
        
    image_preds = torch.stack(image_preds, dim=0)
    image_gts = torch.stack(image_gts, dim=0)

    psnr_all.append(psnr_metric(image_preds.numpy(), image_gts.numpy()))
    ssim_all.append(ssim_metric(image_preds, image_gts).detach().clone().cpu().numpy())
    lpips_all.append(lpips_metric(image_preds, image_gts).detach().clone().cpu().numpy())
        
    psnr_mean = np.mean(psnr_all)
    ssim_mean = np.mean(ssim_all)
    lpips_mean = np.mean(lpips_all)

    return psnr_mean, ssim_mean, lpips_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate 3D meshes and 2D NVS metrics.")
    parser.add_argument('--target_res', type=str, default='experiments_GSO_demo_mesh_view_5', help='The target resolution (default: experiments_GSO_demo_mesh_view_5)')
    parser.add_argument('--OBJECT_TYPE', type=str, required=True, help='The type of the object')
    parser.add_argument('--OBJECT_NAME', type=str, required=True, help='The name of the object')
    parser.add_argument('--OBJECT_VIEW', type=int, required=True, help='The view number of the object')
    parser.add_argument('--ROOT_DIR', type=str, required=True, help='The root directory for saving results')
    args = parser.parse_args()

    object_type = args.OBJECT_TYPE
    object_name = args.OBJECT_NAME
    train_view = args.OBJECT_VIEW  # Assuming train_view is equivalent to OBJECT_VIEW
    root_dir = args.ROOT_DIR

    gt_root = f'{root_dir}/dataset/data/test'
    predict_path = f"experiments_nvs/{object_type}/{object_name}_view_{train_view}"
    groundtruth_path = f"{gt_root}/{object_name}"
    psnr_mean, ssim_mean, lpips_mean = calculate_metrics(predict_path, groundtruth_path)

    results_directory = os.path.join(root_dir, "results", object_type, object_name, str(train_view))
    os.makedirs(results_directory, exist_ok=True)
    results_file_path = os.path.join(results_directory, "results.json")

    new_results = {
        "2d_psnr_mean": float(psnr_mean), 
        "2d_ssim_mean": float(ssim_mean),
        "2d_lpips_mean": float(lpips_mean)
    }

    update_results_in_json(new_results, results_file_path)
