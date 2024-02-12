import copy
import csv
import json
import math
import os
import cv2
import pdb
import random
import sys
from pathlib import Path
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import webdataset as wds
from datasets import load_dataset
from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet
    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)


class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(
            self, 
            root_dir, 
            batch_size, 
            total_view, 
            train=None, 
            validation=None,
            test=None, 
            num_workers=4, 
            reg_objaverse=True, 
            nearest_cond_views=True,
            in_the_wild=False,
            train_view=3,
            random_objaverse=False,
            objaverse_path="/shared/xinyang/threetothreed/dataset/data/objaverse",
            **kwargs,
    ):
        super().__init__(self)
        self.root_dir           = root_dir
        self.batch_size         = batch_size
        self.num_workers        = num_workers
        self.total_view         = total_view
        self.reg_objaverse      = reg_objaverse
        self.nearest_cond_views = nearest_cond_views
        self.in_the_wild        = in_the_wild
        self.train_view         = train_view
        self.random_objaverse   = random_objaverse
        self.objaverse_path     = objaverse_path

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation

        if 'image_transforms' in dataset_config:
            image_transforms = [torchvision.transforms.Resize(dataset_config.image_transforms.size)]
        else:
            image_transforms = []
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c')),
            ]
        )
        self.image_transforms = torchvision.transforms.Compose(image_transforms)

    def train_dataloader(self):
        dataset = ObjaverseData(
            root_dir=self.root_dir, 
            total_view=self.total_view, 
            validation=False, 
            image_transforms=self.image_transforms, 
            reg_objaverse=self.reg_objaverse,
            nearest_cond_views=self.nearest_cond_views,
            in_the_wild=self.in_the_wild,
            train_view=self.train_view,
            random_objaverse=self.random_objaverse,
            objaverse_path=self.objaverse_path,
        )
        sampler = DistributedSampler(dataset)
        print("train@@@")
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    # def val_dataloader(self):
    #     dataset = ObjaverseData(
    #         root_dir=self.root_dir, 
    #         total_view=self.total_view, 
    #         validation=True, 
    #         reg_objaverse=self.reg_objaverse,
    #         image_transforms=self.image_transforms,
    #         nearest_cond_views=self.nearest_cond_views, 
    #         in_the_wild=self.in_the_wild,
    #         train_view=self.train_view,
    #     )
    #     sampler = DistributedSampler(dataset)
    #     print("val@@@")
    #     return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    # def test_dataloader(self):
    #     dataset = ObjaverseData(
    #         root_dir=self.root_dir, 
    #         total_view=self.total_view, 
    #         validation=self.validation,
    #         reg_objaverse=self.reg_objaverse, 
    #         image_transforms=self.image_transforms,
    #         nearest_cond_views=self.nearest_cond_views,
    #         in_the_wild=self.in_the_wild,
    #         train_view=self.train_view,
    #     )
    #     print("test@@@")
    #     return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class POSE_MATCHER():
    def __init__(self, train_pose_folder=None, train_view=3):
        self.train_poses = np.stack([
            np.load(
                os.path.join(
                    train_pose_folder, f'{i:03d}.npy',
                )
            ) for i in range(train_view)
        ], axis=0)
        if train_view == 2:
            self.hard_code_index = [
                0,           #   0,
                1,1,1,1,1,   #  30,  60,  90, 120, 150
                1,           # 180,
                1,1,1,1,1,   # 210, 240, 270, 200, 300, 330
            ]
        elif train_view == 3:
            self.hard_code_index = [
                0,           #   0,
                1,1,1,1,1,   #  30,  60,  90, 120, 150
                1,           # 180,
                1,1,1,1,2,   # 210, 240, 270, 200, 300, 330
            ]
        elif train_view == 4:
            self.hard_code_index = [
                0,           #   0,
                1,1,1,1,1,   #  30,  60,  90, 120, 150
                3,           # 180,
                3,3,3,3,2,   # 210, 240, 270, 200, 300, 330
            ]
        elif train_view == 5:
            self.hard_code_index = [
                0,           #   0,
                1,1,1,1,4,   #  30,  60,  90, 120, 150
                3,           # 180,
                3,3,3,3,2,   # 210, 240, 270, 200, 300, 330
            ]
        elif train_view == 6:
            self.hard_code_index = [
                0,           #   0,
                1,1,1,1,4,   #  30,  60,  90, 120, 150
                3,           # 180,
                3,5,5,5,2,   # 210, 240, 270, 200, 300, 330
            ]
        
            
    def cartesian_to_spherical(self, xyz):
        xy      = xyz[:,0]**2 + xyz[:,1]**2
        z       = np.sqrt(xy + xyz[:,2]**2)
        theta   = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_delta(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta   = theta_target - theta_cond
        d_azimuth = (np.rad2deg(azimuth_target)%360.) - (np.rad2deg(azimuth_cond)%360.)
        d_z       = z_target - z_cond
        return d_theta, d_azimuth, d_z

    def find_closest_cond_pose(self, target_index, target_pose):
        closest_index = self.hard_code_index[int(target_index%12)]
        closest_train_pose = self.train_poses[closest_index]
        # min_azimuth = float('inf')
        # closest_index = None
        # closest_train_pose = None
        # for index in range(len(self.train_poses)):
        #     closest_index = self.hard_code_index[index]
        #     closest_train_pose = self.train_poses[self.hard_code_index[index]]
        #     d_theta, d_azimuth, d_z = self.get_delta(self.train_poses[index], target_pose) 
        #     d_azimuth = np.abs(d_azimuth)
        #     if d_azimuth < min_azimuth:
        #         # Smaller azimuth angle found
        #         min_azimuth = d_azimuth
        #         min_theta = d_theta
        #         closest_index = index
        #         closest_train_pose = self.train_poses[index]
        return closest_index, closest_train_pose

class ObjaverseData(Dataset):
    def __init__(
        self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_transforms=[],
        ext="png",
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        total_view=4,
        validation=False,
        reg_objaverse=True,
        nearest_cond_views=True,
        in_the_wild=False,
        train_view=3,
        random_objaverse=False,
        objaverse_path="/shared/xinyang/threetothreed/dataset/data/objaverse",
    ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.reg_root = Path(objaverse_path)
        self.root_dir = Path(root_dir)
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        self.total_view = total_view
        self.reg_objaverse = reg_objaverse
        self.random_objaverse = random_objaverse

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]
        
        self.tform = image_transforms
        self.in_the_wild = in_the_wild
        # * pair list
        self.train_view = train_view
        self.pair_list = []
        for i in range(self.train_view):
            for j in range(self.train_view):
                if i == j:
                    pass
                else:
                    self.pair_list.append([i, j])

        # * select matched object in objaverse
        self.reg_list = []
        reg_class = 6
        for i in range(reg_class):
            self.reg_list.append(i)
        if not self.random_objaverse:
            self.query_match_paths = np.load(f'{self.root_dir}/query_class/query_match_paths.npy').tolist()
        else:
            self.query_match_paths = np.load(f'/shared/xinyang/threetothreed/dataset/CLIPFEA_cache/reg_features_paths.npy').tolist()

        # * preload data
        self.train_images = self.preload_train_data()
        # if not self.in_the_wild:
        #     self.total_eval_view = 84 # ! 42 for debug running 
        #     if nearest_cond_views:
        #         # * nearest views
        #         print('============= use nearest view =============')
        #         self.pose_matcher = POSE_MATCHER(os.path.join(str(self.root_dir), 'poses'), self.train_view)
        #         self.eval_images_target, self.eval_images_cond, self.eval_delta_poses = self.preload_evaluation_data_nearview()
        #         self.vis_images_target, self.vis_images_cond, self.vis_delta_poses = self.preload_visualization_data_nearview() 
        #         # * for init comparision
        #         self.init_eval_images_cond = np.stack([
        #                                                 self.load_image_eval(
        #                                                     os.path.join(self.root_dir, 'images', f'001.png')
        #                                                 ) for i in range(self.total_eval_view)
        #                                             ])
        #         self.init_eval_delta_poses = np.stack([
        #                                                 self.get_T(
        #                                                     np.load(os.path.join(str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy')), 
        #                                                     np.load(os.path.join(str(self.root_dir), 'poses', f'001.npy',)),
        #                                                 ) for i in range(self.total_eval_view)
        #                                             ])
        #         self.init_vis_images_cond = np.stack([
        #                                                 self.load_image_visualization(
        #                                                     os.path.join(
        #                                                         self.root_dir, 'images', f'001.png',
        #                                                     )
        #                                                 ) for i in range(self.total_eval_view)
        #                                             ])
        #         self.init_vis_delta_poses = np.stack([
        #                                                 self.get_T(
        #                                                     np.load(os.path.join(str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy')), 
        #                                                     np.load(os.path.join(str(self.root_dir), 'poses', f'001.npy',)),
        #                                                 ) for i in range(self.total_eval_view)
        #                                             ])
        #     else:
        #         # * random selected
        #         self.eval_images_target, self.eval_images_cond, self.eval_delta_poses = self.perload_evaluation_data()
        #         self.vis_images_target, self.vis_images_cond, self.vis_delta_poses = self.preload_visualization_data() 
            
    # * preload train data
    def preload_train_data(self):
        train_images = np.stack([
            self.load_image_train(
                os.path.join(
                    str(self.root_dir), 'images', f'{i:03d}.png',
                )
            ) for i in range(self.train_view)
        ])
        return train_images

    # * perload evaluation data
    def perload_evaluation_data(self):
        eval_images_cond = np.stack([
            self.load_image_eval(
                os.path.join(
                    self.root_dir, 'images', f'001.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        eval_pose_cond = np.load(
            os.path.join(
                str(self.root_dir), 'poses', f'001.npy',
            )
        )
        eval_images_target = np.stack([
            self.load_image_eval(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'images', f'{i:03d}.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        eval_delta_poses   = np.stack([
            self.get_T(
                np.load(
                    os.path.join(
                        str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy',
                    )
                ), 
                eval_pose_cond,
            ) for i in range(self.total_eval_view)
        ])
        return eval_images_target, eval_images_cond, eval_delta_poses
    
    # * perload visualization data
    def preload_visualization_data(self):
        vis_images_cond = np.stack([
            self.load_image_visualization(
                os.path.join(
                    self.root_dir, 'images', f'001.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        vis_pose_cond = np.load(
            os.path.join(
                str(self.root_dir), 'poses', f'001.npy',
            )
        )
        vis_images_target = np.stack([
            self.load_image_visualization(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'images', f'{i:03d}.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        vis_delta_poses   = np.stack([
            self.get_T(
                np.load(
                    os.path.join(
                        str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy',
                    )
                ), 
                vis_pose_cond,
            ) for i in range(self.total_eval_view)
        ])
        return vis_images_target, vis_images_cond, vis_delta_poses

    # * adaptive nearest views for evaluation
    def preload_evaluation_data_nearview(self):
        eval_images_target = np.stack([
            self.load_image_eval(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'images', f'{i:03d}.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        eval_poses_target = np.stack([
            np.load(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy',
                )
            ) for i in range(self.total_eval_view)
        ])
        
        eval_images_cond = []
        eval_delta_poses = []
        for index, eval_pose_target in enumerate(eval_poses_target):
            cond_index, cond_pose = self.pose_matcher.find_closest_cond_pose(index, eval_pose_target)
            eval_images_cond.append(
                self.load_image_eval(
                    os.path.join(
                        self.root_dir, 'images', f'{cond_index:03d}.png',
                    )
                )
            )

            eval_delta_poses.append(
                self.get_T(
                    eval_pose_target,
                    cond_pose,
                )
            )
        eval_images_cond = np.stack(eval_images_cond)
        eval_delta_poses = np.stack(eval_delta_poses)
        return eval_images_target, eval_images_cond, eval_delta_poses

    # * adaptive nearest views for visualization
    def preload_visualization_data_nearview(self):
        vis_images_target = np.stack([
            self.load_image_visualization(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'images', f'{i:03d}.png',
                )
            ) for i in range(self.total_eval_view)
        ])
        vis_poses_target = np.stack([
            np.load(
                os.path.join(
                    str(self.root_dir).replace('train', 'test'), 'poses', f'{i:03d}.npy',
                )
            ) for i in range(self.total_eval_view)
        ])
        
        vis_images_cond = []
        vis_delta_poses = []
        for index, vis_pose_target in enumerate(vis_poses_target):
            cond_index, cond_pose = self.pose_matcher.find_closest_cond_pose(index, vis_pose_target)
            vis_images_cond.append(
                self.load_image_visualization(
                    os.path.join(
                        self.root_dir, 'images', f'{cond_index:03d}.png',
                    )
                )
            )
            vis_delta_poses.append(
                self.get_T(
                    vis_pose_target,
                    cond_pose,
                )
            )
        vis_images_cond = np.stack(vis_images_cond)
        vis_delta_poses = np.stack(vis_delta_poses)
        return vis_images_target, vis_images_cond, vis_delta_poses

    def __len__(self):
        if self.train_view > 1:
            return int(self.train_view * (self.train_view-1))
        else:
            return 1
        
    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:,0]**2 + xyz[:,1]**2
        z = np.sqrt(xy + xyz[:,2]**2)
        theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
        #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:,1], xyz[:,0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T
    
    def get_T_my(self, theta_target, theta_cond, azimuth_target, azimuth_cond, z_target, z_cond):
        
        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond
        
        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def add_margin(self, pil_img, color, size=256):
        width, height = pil_img.size
        result = Image.new(pil_img.mode, (size, size), color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result

    def load_image_train(self, path, alpha=[1., 1., 1., 1.]):
        img_rgba = plt.imread(path)
        img_rgba[img_rgba[:, :, -1] == 0.] = alpha
        img_rgba = Image.fromarray(np.uint8(img_rgba[:, :, :3] * 255.))
        img_rgb = img_rgba.convert("RGB")
        if self.tform is not None:
            img_rgb = self.tform(img_rgb)
        return img_rgb
    
    def load_image_eval(self, path, alpha=[1., 1., 1., 1.]):
        img_rgba = plt.imread(path)
        img_rgba[img_rgba[:, :, -1] == 0.] = alpha
        img_rgba = Image.fromarray(np.uint8(img_rgba[:, :, :3] * 255.))
        img_rgb = img_rgba.convert("RGB")
        if self.tform is not None:
            img_rgb = self.tform(img_rgb)
        return img_rgb

    def load_image_visualization(self, path, alpha=[1., 1., 1., 1.]):
        img_rgba   = plt.imread(path)
        img_a      = img_rgba[..., 3:]
        x, y, w, h = cv2.boundingRect(img_a.astype(np.uint8))

        img_rgba = img_rgba[y:y+h, x:x+w, :]
        img_rgba[img_rgba[:, :, -1] == 0.] = alpha
        img_rgba = Image.fromarray(np.uint8(img_rgba[:, :, :3] * 255.))
        img_rgb  = img_rgba.convert("RGB")

        img_rgb.thumbnail([200, 200], Image.LANCZOS)
        img_rgb = self.add_margin(img_rgb, (255, 255, 255), size=256)
        
        if self.tform is not None:
            img_rgb = self.tform(img_rgb)
        return img_rgb

    def check_pair_list_and_reg_list(self):
        if len(self.pair_list) == 0:
            self.pair_list = []
            for i in range(self.train_view):
                for j in range(self.train_view):
                    if i == j:
                        pass
                    else:
                        self.pair_list.append([i, j])
                    
        if len(self.reg_list) == 0:
                reg_class = 6
                self.reg_list = []
                for i in range(reg_class):
                    self.reg_list.append(i)

    def __getitem__(self, index):
        data = {}
        if self.train_view > 1:
            # * random select one pair without replacement
            total_pair = len(self.pair_list)
            random_ind = random.sample(range(total_pair), 1)[0]
            index_target, index_cond = self.pair_list.pop(random_ind)
            
            # * load train images
            data["image_target"] = self.train_images[index_target]
            data["image_cond"]   = self.train_images[index_cond]
            data["index_target"] = index_target
            data["index_cond"]   = index_cond
        else:
            # * load train images
            data["image_target"] = self.train_images[0]
            data["image_cond"]   = self.train_images[0]
            data["index_target"] = 0
            data["index_cond"]   = 0

        data["train_view"] = self.train_view

        # if not self.in_the_wild:
        #     # * load evaluation images
        #     data["eval_image_target"] = self.eval_images_target
        #     data["eval_images_cond"]  = self.eval_images_cond
        #     data["eval_delta_poses"]  = self.eval_delta_poses

        #     # * load visualization image
        #     data["vis_images_target"] = self.vis_images_target
        #     data["vis_images_cond"]   = self.vis_images_cond
        #     data["vis_delta_poses"]   = self.vis_delta_poses
            
        #     # * load init image
        #     data["init_eval_images_cond"] = self.init_eval_images_cond
        #     data["init_eval_delta_poses"] = self.init_eval_delta_poses
        #     data["init_vis_images_cond"]  = self.init_vis_images_cond
        #     data["init_vis_delta_poses"]  = self.init_vis_delta_poses
            
        if self.reg_objaverse:
            if not self.random_objaverse:
                # * random select one class withou replacement
                total_class    = len(self.reg_list)
                random_ind     = random.sample(range(total_class), 1)[0]
                random_index   = self.reg_list.pop(random_ind)
                reg_class_name = self.query_match_paths[random_index].split('/')[-1]
                
                # total_view = 24
                # index_target, index_cond = random.sample(range(total_view), 2) # without replacement
                # 0 45 90 135 180 225 270 315
                # 0  1  2   3   4   5   6   7
                all_reg_views = [0, 1, 2, 3, 4, 5, 6, 7]
                reg_views = random.sample(all_reg_views, 2)
                for i in range(2):
                    elevation_status = np.random.randint(0, 3, 1).item()
                    if elevation_status == 0:
                        reg_views[i] = reg_views[i] + 0
                    elif elevation_status == 1:
                        reg_views[i] = reg_views[i] + 8
                    else:
                        reg_views[i] = reg_views[i] + 16
                index_target, index_cond = reg_views

                target_im = self.load_image_train(os.path.join(self.reg_root, reg_class_name, 'images', '%03d.png' % index_target))
                cond_im   = self.load_image_train(os.path.join(self.reg_root, reg_class_name, 'images', '%03d.png' % index_cond))
                target_RT = np.load(os.path.join(self.reg_root, reg_class_name, 'poses', '%03d.npy' % index_target))
                cond_RT   = np.load(os.path.join(self.reg_root, reg_class_name, 'poses', '%03d.npy' % index_cond))
            else:
                reg_class_name = random.sample(self.query_match_paths, 1)[0]
                total_view = 12
                index_target, index_cond = random.sample(range(total_view), 2) # without replacement
                reg_root = '/shared/xinyang/views_release'
                target_im = self.load_image_train(os.path.join(reg_root, reg_class_name, '%03d.png' % index_target))
                cond_im   = self.load_image_train(os.path.join(reg_root, reg_class_name, '%03d.png' % index_cond))
                target_RT = np.load(os.path.join(reg_root, reg_class_name, '%03d.npy' % index_target))
                cond_RT   = np.load(os.path.join(reg_root, reg_class_name, '%03d.npy' % index_cond))
                
            data["image_target_reg"] = target_im
            data["image_cond_reg"]   = cond_im
            data["T_reg"]            = self.get_T(target_RT, cond_RT)

        self.check_pair_list_and_reg_list()
        return data

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = Path(root_dir)
        self.default_caption = default_caption
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        if caption_file is not None:
            with open(caption_file, "rt") as f:
                ext = Path(caption_file).suffix.lower()
                if ext == ".json":
                    captions = json.load(f)
                elif ext == ".jsonl":
                    lines = f.readlines()
                    lines = [json.loads(x) for x in lines]
                    captions = {x["file_name"]: x["text"].strip("\n") for x in lines}
                else:
                    raise ValueError(f"Unrecognised format: {ext}")
            self.captions = captions
        else:
            self.captions = None

        if not isinstance(ext, (tuple, list, ListConfig)):
            ext = [ext]

        # Only used if there is no caption file
        self.paths = []
        for e in ext:
            self.paths.extend(sorted(list(self.root_dir.rglob(f"*.{e}"))))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        if self.captions is not None:
            return len(self.captions.keys())
        else:
            return len(self.paths)

    def __getitem__(self, index):
        data = {}
        if self.captions is not None:
            chosen = list(self.captions.keys())[index]
            caption = self.captions.get(chosen, None)
            if caption is None:
                caption = self.default_caption
            filename = self.root_dir/chosen
        else:
            filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)

        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        data["image"] = im

        if self.captions is not None:
            data["txt"] = caption
        else:
            data["txt"] = self.default_caption

        if self.postprocess is not None:
            data = self.postprocess(data)

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)
import random


class TransformDataset():
    def __init__(self, ds, extra_label="sksbspic"):
        self.ds = ds
        self.extra_label = extra_label
        self.transforms = {
            "align": transforms.Resize(768),
            "centerzoom": transforms.CenterCrop(768),
            "randzoom": transforms.RandomCrop(768),
        }


    def __getitem__(self, index):
        data = self.ds[index]

        im = data['image']
        im = im.permute(2,0,1)
        # In case data is smaller than expected
        im = transforms.Resize(1024)(im)

        tform_name = random.choice(list(self.transforms.keys()))
        im = self.transforms[tform_name](im)

        im = im.permute(1,2,0)

        data['image'] = im
        data['txt'] = data['txt'] + f" {self.extra_label} {tform_name}"

        return data

    def __len__(self):
        return len(self.ds)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    tform = make_tranforms(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds

class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]



import json
import random


class IdRetreivalDataset(FolderData):
    def __init__(self, ret_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(ret_file, "rt") as f:
            self.ret = json.load(f)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        key = self.paths[index].name
        matches = self.ret[key]
        if len(matches) > 0:
            retreived = random.choice(matches)
        else:
            retreived = key
        filename = self.root_dir/retreived
        im = Image.open(filename).convert("RGB")
        im = self.process_im(im)
        # data["match"] = im
        data["match"] = torch.cat((data["image"], im), dim=-1)
        return data