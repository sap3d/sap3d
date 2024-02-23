import glob
import math
import multiprocessing
import os
import pdb
import subprocess
import time
from contextlib import nullcontext
from functools import partial
from multiprocessing import Manager, Process

import cv2
import fire
import gradio as gr
import imageio
import numpy as np
import torch
import tqdm
from einops import rearrange
from ldm.models.diffusion.ddim_sc import DDIMSampler
from ldm.util import instantiate_from_config, load_and_preprocess
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms

setting1 = [
    '3D_Dollhouse_Happy_Brother',
    'CHICKEN_RACER',
    'Crosley_Alarm_Clock_Vintage_Metal',
    'Marvel_Avengers_Titan_Hero_Series_Doctor_Doom',
    'MINI_FIRE_ENGINE',
    'My_Little_Pony_Princess_Celestia',
    'Schleich_African_Black_Rhino',
    'Schleich_Lion_Action_Figure',
    'Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj',
    'Transformers_Age_of_Extinction_Stomp_and_Chomp_Grimlock_Figure',
    'Vtech_Roll_Learn_Turtle'
]

def diffto8b(img): return (np.clip(0.5*(img+1.0),0.0,1.0)*255.).astype(np.uint8)

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def load_image(path, alpha=[1., 1., 1., 1.], device='cuda'):
    img_rgba = plt.imread(path)
    if img_rgba.shape[-1] == 4:
        img_rgba[img_rgba[:, :, -1] == 0.] = alpha
    img_rgba = Image.fromarray(np.uint8(img_rgba[:, :, :3] * 255.))
    img_rgb  = img_rgba.convert("RGB")
    img_rgb  = transforms.ToTensor()(img_rgb)[None, ...].to(device) * 2.0 - 1.0
    img_rgb  = transforms.functional.resize(img_rgb, [256, 256])
    return img_rgb

def update_cond(
    save_cond_dir, 
    model,  
    elevation, 
    azimuth, 
    elevation_conds,
    azimuth_conds,
    view_num,
    device='cuda',
):
    update_conds = []
    for i in range(view_num):
        T = torch.tensor([
            elevation - elevation_conds[i], 
            math.sin(azimuth - azimuth_conds[i]), 
            math.cos(azimuth - azimuth_conds[i]), 
            0.0,
        ]).float()[None, None, :].to(device)
        
        input_im   = load_image(f"{save_cond_dir}/{i:03d}.png", alpha=[1., 1., 1., 1.], device=device)
        clip_embed = model.get_learned_conditioning(input_im)
        
        c_crossattn_fea = model.cc_projection(torch.cat([clip_embed, T], dim=-1))
        c_concat_fea    = model.encode_first_stage((input_im)).mode().detach()
        
        cond                = {}
        cond['c_crossattn'] = [c_crossattn_fea]
        cond['c_concat']    = [c_concat_fea]
        
        update_conds.append(cond)
    uc = {}
    uc['c_concat']    = [torch.zeros_like(cond['c_concat'][0]).to(device)]
    uc['c_crossattn'] = [torch.zeros_like(cond['c_crossattn'][0]).to(device)]
    return update_conds, uc

def go_nvs(
    config_path     : str  = None,
    ckpt_path       : str  = None,
    cond_img_dir    : str  = None,
    cond_view_num   : int  = 3,
    elevation_conds : list = None,
    azimuth_conds   : list = None,
    radius_conds    : list = None,
    saving_dir_path : str  = None,
    object_name     : str  = "",
    use_sc          : bool = True,
    device_id       : int  = 0,
):
    syn_video = []
    obj_name  = saving_dir_path.split('/')[-1]
    # * init model
    if cond_view_num > 1:
        device = f"cuda:{device_id}"
        config = OmegaConf.load(config_path)
        model  = load_model_from_config(config, ckpt_path, device=device)
    else:
        device = f"cuda:{device_id}"
        config = OmegaConf.load('/shared/xinyang/threetothreed/camerabooth/configs/config_base.yaml')
        model  = load_model_from_config(config, ckpt_path, device=device)
    
    # * pre save cond image of 3 views
    save_cond_dir = f'{saving_dir_path}/conds'
    save_syn_dir  = f'{saving_dir_path}/preds'
    os.makedirs(save_cond_dir, exist_ok=True)
    os.makedirs(save_syn_dir, exist_ok=True)
    for i in range(cond_view_num):
        image_cond = load_image(f'{cond_img_dir}/{i:03d}.png', alpha=[1., 1., 1., 1.], device=device).permute([0,2,3,1])[0].cpu().numpy()
        image_cond = Image.fromarray(diffto8b(image_cond))
        image_cond.save(f'{save_cond_dir}/{i:03d}.png')
    
    # * pre-compute 7x12 all camera poses
    if object_name in setting1:
        delta_elevations = [-45. * np.pi / 180.] * 12 + \
                        [-30. * np.pi / 180.] * 12 + \
                        [-15. * np.pi / 180.] * 12 + \
                        [  0. * np.pi / 180.] * 12 + \
                        [ 15. * np.pi / 180.] * 12 + \
                        [ 30. * np.pi / 180.] * 12 + \
                        [ 45. * np.pi / 180.] * 12
    else:
        delta_elevations = [-15. * np.pi / 180.] * 12 + \
                        [  0. * np.pi / 180.] * 12 + \
                        [ 15. * np.pi / 180.] * 12 + \
                        [ 30. * np.pi / 180.] * 12 + \
                        [ 45. * np.pi / 180.] * 12 + \
                        [ 60. * np.pi / 180.] * 12 + \
                        [ 75. * np.pi / 180.] * 12
    delta_azimuths = [
        (0 + i * 30. * np.pi / 180.) for i in range(12)
    ] * 7
    # note reference view is 000.png
    elevations_nvs, azimuths_nvs, radius_nvs = [], [], []
    for i in range(84):
        elevations_nvs.append(delta_elevations[i] + elevation_conds[0])
        azimuths_nvs.append(delta_azimuths[i] + azimuth_conds[0]) 
        radius_nvs.append(0.0)
    
    # * diffusion images
    view_num     = cond_view_num
    ddim_sampler = DDIMSampler(model)
    for i in tqdm.tqdm(range(84), desc=f'Novel View Synthesis for {obj_name}'):
        elevation = elevations_nvs[i]
        azimuth   = azimuths_nvs[i]
        sc_conds, uncond = update_cond(
            save_cond_dir, 
            model,  
            elevation, 
            azimuth, 
            elevation_conds,
            azimuth_conds,
            view_num,
            device,
        )
        
        samples_ddim, _ = ddim_sampler.sample(
            S=75,
            conditionings=sc_conds,
            batch_size=1,
            shape=[4,32,32],
            verbose=False,
            unconditional_guidance_scale=3.0,
            unconditional_conditioning=uncond,
            eta=1.0,
            cond_id=-1 if use_sc else 0,
        )
        
        # * save and update condition images/poses and video 
        x_samples_ddim = model.decode_first_stage(samples_ddim).permute([0,2,3,1])[0].cpu().numpy()
        x_samples_ddim = Image.fromarray(diffto8b(x_samples_ddim))
        x_samples_ddim.save(f'{save_syn_dir}/{(i):03d}.png')
        
        # * save video
        syn_video.append(np.array(x_samples_ddim))
    imageio.mimsave(f'{save_syn_dir}/{obj_name}_sc.mp4', syn_video, fps=15)
        
def find_last_log(dir_path):
    timestamp_dirs = glob.glob(os.path.join(dir_path, "*"))
    timestamp_dirs.sort(key=os.path.getmtime, reverse=True)
    if timestamp_dirs:
        latest_dir = timestamp_dirs[0]
        return latest_dir
    return None

def get_configures():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--object_type', type=str, default='XINYANG')
    parser.add_argument('--object_name', type=str, default='meow1')
    parser.add_argument('--train_view', type=int, default=5)

    return parser.parse_args()

if __name__ == '__main__':
    opt = get_configures()
    log_path = find_last_log(f'experiments_{opt.object_type}_view_{opt.train_view}/{opt.object_name}')

    if log_path:
        config_path = f'configs/{opt.object_type}/config_{opt.object_name}_view_{opt.train_view}.yaml'
        cond_img_dir_path = f"../dataset/data/train/{opt.object_type}/{opt.object_name}/images"
        ckpt_path        = f'{log_path}/checkpoints/last.ckpt'
        finetune_results = torch.load(f'{log_path}/evaluation/epoch_99/results.tar')
        elevation_pred   = finetune_results['elevation_pred'].tolist()
        azimuth_pred     = finetune_results['azimuth_pred'].tolist()
        radius_pred      = finetune_results['radius_pred'].tolist()
        saving_dir_path  = f'experiments_nvs/{opt.object_type}/{opt.object_name}_view_{opt.train_view}'

        cond_view_num = len(radius_pred)

        go_nvs(
            config_path, 
            ckpt_path, 
            cond_img_dir_path, 
            cond_view_num,
            elevation_pred, 
            azimuth_pred, 
            radius_pred, 
            saving_dir_path,
            object_name=opt.object_name,
            use_sc=True,
        )

    if opt.train_view == 1:
        config_path = f'configs/{opt.object_type}/config_{opt.object_name}_view_{opt.train_view}.yaml'
        cond_img_dir_path = f"../dataset/data/train/{opt.object_type}/{opt.object_name}/images"
        ckpt_path        = 'zero123_sm.ckpt'
        saving_dir_path  = f'experiments_nvs/{opt.object_type}/{opt.object_name}_view_{opt.train_view}'
        cond_view_num = 1
        elevation_pred = [30.0 / 180.0 * np.pi]
        azimuth_pred = [0.0]
        radius_pred = 0.0
        go_nvs(
            config_path, 
            ckpt_path, 
            cond_img_dir_path, 
            cond_view_num,
            elevation_pred, 
            azimuth_pred, 
            radius_pred, 
            saving_dir_path,
            use_sc=False,
        )