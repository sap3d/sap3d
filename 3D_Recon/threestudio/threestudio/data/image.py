import bisect
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False


class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
        up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()
        self.load_images()
        self.prev_height = self.height

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"
        rgba = cv2.cvtColor(
            cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        print(
            f"[INFO] single image dataset: load image {self.cfg.image_path} {self.rgb.shape}"
        )

        # load depth
        if self.cfg.requires_depth:
            depth_path = self.cfg.image_path.replace("_rgba.png", "_depth.png")
            assert os.path.exists(depth_path)
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            self.depth: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(depth.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            print(
                f"[INFO] single image dataset: load depth {depth_path} {self.depth.shape}"
            )
        else:
            self.depth = None

        # load normal
        if self.cfg.requires_normal:
            normal_path = self.cfg.image_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(
                normal, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            self.normal: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(normal.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            print(
                f"[INFO] single image dataset: load normal {normal_path} {self.normal.shape}"
            )
        else:
            self.normal = None

    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "ref_depth": self.depth,
            "ref_normal": self.normal,
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]
        # if index == 0:
        #     return {
        #         'rays_o': self.rays_o[0],
        #         'rays_d': self.rays_d[0],
        #         'mvp_mtx': self.mvp_mtx[0],
        #         'camera_positions': self.camera_position[0],
        #         'light_positions': self.light_position[0],
        #         'elevation': self.elevation_deg[0],
        #         'azimuth': self.azimuth_deg[0],
        #         'camera_distances': self.camera_distance[0],
        #         'rgb': self.rgb[0],
        #         'depth': self.depth[0],
        #         'mask': self.mask[0]
        #     }
        # else:
        #     return self.random_pose_generator[index - 1]


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = SingleImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

@dataclass
class MultiviewImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height                 : Any       = 96
    width                  : Any       = 96
    resolution_milestones  : List[int] = field(default_factory=lambda: [])
    default_elevation_deg  : float     = 0.0
    default_azimuth_deg    : float     = -180.0
    default_camera_distance: float     = 1.2
    default_fovy_deg       : float     = 60.0
    image_path             : str       = ""
    use_random_camera      : bool      = True
    random_camera          : dict      = field(default_factory=dict)
    rays_noise_scale       : float     = 2e-3
    batch_size             : int       = 1
    requires_depth         : bool      = False
    requires_normal        : bool      = False

    # * load multiple images
    multiple_image_folder: str = "../dataset/data/train/GSO/JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece"
    estimate_pose_path   : str = "../camerabooth/GSO_incorrect_experiments_3views/config_JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece/2023-09-27T09-33-44_config_JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece-1-1000-1e-06-0.1-1e-07/evaluation/epoch_99/results.tar"


class MultiViewImageDataBase:
    def setup_multiview(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: MultiviewImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )
        
        finetune_result    = torch.load(self.cfg.estimate_pose_path)
        elevation_deg_pred = torch.FloatTensor(finetune_result['elevation_pred']) / math.pi * 180.
        azimuth_deg_pred   = torch.FloatTensor(finetune_result['azimuth_pred']) / math.pi * 180.
        # adjust polar from zero123 to threestudio
        self.n_views    = len(azimuth_deg_pred)
        elevation_deg = (90.0 - elevation_deg_pred) - min(90.0 - elevation_deg_pred)
        # azimuth_deg = (azimuth_deg_pred - azimuth_deg_pred[0] + 180) % 360 - 180 # [-180, 180]
        azimuth_deg = azimuth_deg_pred - azimuth_deg_pred[0]
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance] * self.n_views)

        elevation = elevation_deg * math.pi / 180
        azimuth   = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None, :].repeat(self.n_views, 1)

        light_position: Float[Tensor, "B 3"] = camera_position
        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
        up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
        self.c2w: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        # threestudio.info(lookat[2:3])
        # threestudio.info(right[2:3])
        # threestudio.info(up[2:3])
        # threestudio.info(camera_position[2:3])
        # threestudio.info(self.c2w[2:3])

        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_position = camera_position
        self.light_position  = light_position
        self.camera_distance = camera_distance
        self.fovy            = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height               : int = self.heights[0]
        self.width                : int = self.widths[0]
        self.directions_unit_focal: int = self.directions_unit_focals[0]
        self.focal_length         : int = self.focal_lengths[0]
        self.set_rays()
        self.load_images()
        self.prev_height = self.height

    def set_rays(self):
        # * get directions by dividing directions_unit_focal by focal length
        focal_length = self.focal_length.repeat(self.n_views)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[None, ...].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = directions[:, :, :, :2] / focal_length[:, None, None, None]

        rays_o, rays_d = get_rays(directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale)

        fovy = self.fovy.repeat(self.n_views)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(fovy, self.width / self.height, 0.1, 100.0)  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        image_paths = [f'{self.cfg.multiple_image_folder}/images/{i:03d}.png' for i in range(self.n_views)]
        rgb, mask = [], []
        for image_path in image_paths:
            assert os.path.exists(image_path), f"Could not find image {image_path}!"
            rgbai = cv2.cvtColor(
                cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            rgbai = cv2.resize(
                    rgbai, (self.width, self.height), interpolation=cv2.INTER_AREA
                ).astype(np.float32) / 255.0
            maski = rgbai[..., 3:] > 0.5
            rgbi  = rgbai[..., :3]
            rgb.append(torch.from_numpy(rgbi))
            mask.append(torch.from_numpy(maski))
        self.rgb : Float[Tensor, "B H W 3"] = torch.stack(rgb, dim=0).contiguous().to(self.rank)
        self.mask: Float[Tensor, "B H W 1"] = torch.stack(mask, dim=0).contiguous().to(self.rank)
        print(f"[INFO] single image dataset: load image {self.cfg.multiple_image_folder} {self.rgb.shape}")
        # load depth
        self.depth = None
        # load normal
        self.normal = None

    def get_all_images(self):
        return self.rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()

import random    

class MultiViewImageIterableDataset(IterableDataset, MultiViewImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup_multiview(cfg, split)
        self.index_list = list(range(self.n_views))
            
    def collate(self, batch) -> Dict[str, Any]:
        if len(self.index_list) == 0:
            self.index_list = list(range(self.n_views))
        total_view    = len(self.index_list)
        random_index  = random.sample(list(range(total_view)), self.cfg.batch_size)
        # collate_index = random_index
        collate_index = [self.index_list.pop(i) for i in random_index]
        
        batch = {
            "rays_o"          : self.rays_o[collate_index],
            "rays_d"          : self.rays_d[collate_index],
            "mvp_mtx"         : self.mvp_mtx[collate_index],
            "camera_positions": self.camera_position[collate_index],
            "light_positions" : self.light_position[collate_index],
            "elevation"       : self.elevation_deg[collate_index],
            "azimuth"         : self.azimuth_deg[collate_index],
            "camera_distances": self.camera_distance[collate_index],
            "rgb"             : self.rgb[collate_index],
            "mask"            : self.mask[collate_index],
            "ref_depth"       : self.depth,
            "ref_normal"      : self.normal,
            "height"          : self.cfg.height,
            "width"           : self.cfg.width,
        }
        
        # threestudio.info(f'rays_o: {batch["rays_o"].shape }')
        # threestudio.info(f'rays_d: {batch["rays_d"].shape }')
        # threestudio.info(f'mvp_mtx: {batch["mvp_mtx"].shape }')
        # threestudio.info(f'camera_positions: {batch["camera_positions"].shape }')
        # threestudio.info(f'light_positions: {batch["light_positions"].shape }')
        # threestudio.info(f'elevation: {batch["elevation"].shape }')
        # threestudio.info(f'azimuth: {batch["azimuth"].shape }')
        # threestudio.info(f'camera_distances: {batch["camera_distances"].shape }')
        # threestudio.info(f'rgb: {batch["rgb"].shape }')
        # threestudio.info(f'mask: {batch["mask"].shape }')
        # threestudio.info(self.c2w[2:3])
        # threestudio.info(batch['mvp_mtx'])
        
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}
            
            
class MultiViewImageDataset(Dataset, MultiViewImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup_multiview(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]
    
    
@register("multiview-image-datamodule")
class MultiViewImageDataModule(pl.LightningDataModule):
    cfg: MultiviewImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiviewImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiViewImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiViewImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiViewImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(self.train_dataset, batch_size=self.cfg.batch_size, collate_fn=self.train_dataset.collate)

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
    