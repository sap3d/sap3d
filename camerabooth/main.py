import argparse
import copy
import csv
import datetime
import glob
import importlib
import os
import pdb
import tqdm
import re
import sys
import time
from functools import partial
from typing import Any, Optional

import lpips
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytz
import termcolor
import torch
import torch.nn.functional as F
import torchvision
import wandb
from easydict import EasyDict as edict
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from packaging import version
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import ToTensor

from external.pohsun_ssim import pytorch_ssim
MULTINODE_HACKS = False

def to8b(img): return (np.clip(img, 0., 1.) * 255).astype(np.uint8)

# convert to colored strings
def red(message,**kwargs): return termcolor.colored(str(message),color="red",attrs=[k for k,v in kwargs.items() if v is True])
def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
def blue(message,**kwargs): return termcolor.colored(str(message),color="blue",attrs=[k for k,v in kwargs.items() if v is True])
def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])
def magenta(message,**kwargs): return termcolor.colored(str(message),color="magenta",attrs=[k for k,v in kwargs.items() if v is True])
def grey(message,**kwargs): return termcolor.colored(str(message),color="grey",attrs=[k for k,v in kwargs.items() if v is True])

class Log:
    def __init__(self): pass
    def process(self,pid):
        print(grey("Process ID: {}".format(pid),bold=True))
    def title(self,message):
        print(yellow(message,bold=True,underline=True))
    def info(self,message):
        print(magenta(message,bold=True))
    def options(self,opt,level=0):
        for key,value in sorted(opt.items()):
            if isinstance(value,(dict,edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value,level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":",yellow(value))
log = Log()


def stitch_multiview_images(multiview_images, labels):
    
    images = [multiview_images[key] for key in multiview_images.keys()]

    # Convert tensors to PIL images
    pil_images = []
    for img in images:
        # just use batch one
        img = img[0].squeeze(axis=0)
        
        grid = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        grid = np.squeeze(grid)
        if grid.ndim == 2:  # Handle grayscale images
            grid = np.stack((grid,) * 3, axis=-1)  # Convert grayscale to RGB
        pil_images.append(Image.fromarray(grid))

    label_height = 30
    stitched_image = Image.new("RGB", (pil_images[0].width * len(pil_images), pil_images[0].height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(stitched_image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf", size=18)

    for i, (img, label) in enumerate(zip(pil_images, labels)):
        draw.text((i * img.width + 10, 5), label, font=font, fill=(0, 0, 0))
        stitched_image.paste(img, (i * img.width, label_height))

    return stitched_image




@rank_zero_only
def rank_zero_print(*args):
    print(*args)

def modify_weights(w, scale = 1e-6, n=2):
    """Modify weights to accomodate concatenation to unet"""
    extra_w = scale*torch.randn_like(w)
    new_w = w.clone()
    for i in range(n):
        new_w = torch.cat((new_w, extra_w.clone()), dim=1)
    return new_w


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        const=True,
        default="0514_180epochs",
        nargs="?",
        help="project name",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of image",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, num_val_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if num_val_workers is None:
            self.num_val_workers = self.num_workers
        else:
            self.num_val_workers = num_val_workers
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_val_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            rank_zero_print("Summoning checkpoint.")
            # ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            # trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            
            log.info("#### Project Configure ####")
            log.options(OmegaConf.to_container(self.config))

            if MULTINODE_HACKS:
                import time
                time.sleep(5)
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            log.info("#### Lightning Configure ####")
            log.options(OmegaConf.to_container(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self, 
        batch_frequency, 
        max_images, clamp=True, 
        increase_log_steps=True,
        rescale=True, 
        disabled=False, 
        log_on_batch_idx=False, 
        log_first_step=False,
        log_images_kwargs=None, 
        log_all_val=False, 
        only_camera=False, 
        a_gt=None, 
        b_gt=None, 
        c_gt=None,
        in_the_wild=False,
    ):
        super().__init__()
        self.only_camera       = only_camera
        self.rescale           = rescale
        self.batch_freq        = batch_frequency
        self.max_images        = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp             = clamp
        self.disabled          = disabled
        self.log_on_batch_idx  = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step    = log_first_step
        self.log_all_val       = log_all_val
        self.lpips_metric      = lpips.LPIPS(net='alex').to('cuda')
        self.ssim_metric       = pytorch_ssim.ssim
        self.in_the_wild       = in_the_wild
        
        if a_gt and b_gt and c_gt:
            self.a_gt = torch.tensor(a_gt, dtype=torch.float32) * np.pi # [0,pi]
            self.b_gt = torch.tensor(b_gt, dtype=torch.float32) * (2.0*np.pi) # [0,2pi]
            self.c_gt = torch.tensor(c_gt, dtype=torch.float32) * (0.7) + 1.5 # [1.5,2.2]
            self.gt_abc_tensor = torch.stack((self.a_gt, self.b_gt, self.c_gt), dim=0)
        else:
            self.gt_abc_tensor = torch.randn([3, 3], dtype=torch.float32)
        self.gt_abc_tensor = torch.randn([3, 3], dtype=torch.float32)
            
        self.gt_diff_0_2 = self.gt_abc_tensor[:,0] - self.gt_abc_tensor[:,2]
        self.gt_diff_0_1 = self.gt_abc_tensor[:,0] - self.gt_abc_tensor[:,1]


        # * define metrics
        self.nvs_psnr  = 0.0
        self.nvs_ssim  = 0.0
        self.nvs_lpips = 0.0

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        log_dict = {}
        
        for k in images:
            if k == 'multiview_images':
                pil_image = images[k]  # This is already a PIL image
                tag = f"{split}/multiview"
                # pl_module.logger.experiment.add_image(tag, pil_image, global_step=pl_module.global_step)
                tensor_image = ToTensor()(pil_image)
                pl_module.logger.experiment.add_image(tag, tensor_image, global_step=pl_module.global_step)
            else:
                # don't log these
                continue
                grid = torchvision.utils.make_grid(images[k])
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                pil_image = Image.fromarray(grid)
                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(tag, pil_image, global_step=pl_module.global_step)

        # print(pl_module.camera_poses.weight)
        # print(pl_module.camera_poses.weight.grad)

        for i in range(pl_module.camera_poses.weight.shape[0]):
            if pl_module.camera_poses.weight.grad is not None:
                for j in range(pl_module.camera_poses.weight.shape[1]):
                    tag_grad = f"gradient_camera_pose_{i}_{j}"
                    tag_value = f"camera_pose_{i}_{j}"

                    pl_module.logger.experiment.add_scalar(tag_grad, pl_module.camera_poses.weight.grad[i][j].item(), global_step=pl_module.global_step)
                    pl_module.logger.experiment.add_scalar(tag_value, pl_module.camera_poses.weight[i][j].item(), global_step=pl_module.global_step)
                    
            else:
                for j in range(pl_module.camera_poses.weight.shape[1]):
                    tag_grad = f"gradient_camera_pose_{i}_{j}"
                    tag_value = f"camera_pose_{i}_{j}"
                    
                    pl_module.logger.experiment.add_scalar(tag_grad, 0.0, global_step=pl_module.global_step)  # or some other default value
                    pl_module.logger.experiment.add_scalar(tag_value, pl_module.camera_poses.weight[i][j].item(), global_step=pl_module.global_step)

        # # pdb.set_trace()
        # # Calculate the difference between the 0th and 2th camera poses
        # diff_0_2 = pl_module.camera_poses.weight[0] - pl_module.camera_poses.weight[2]

        # # Calculate the difference between the 0th and 1th camera poses
        # diff_0_1 = pl_module.camera_poses.weight[0] - pl_module.camera_poses.weight[1]

        # # log the differences between the current camera poses
        # for j in range(diff_0_2.shape[0]):
        #     tag_diff_0_2 = f"diff_0_2_{j}"
        #     tag_diff_0_1 = f"diff_0_1_{j}"
            
        #     pl_module.logger.experiment.add_scalars(
        #         f"camera_pose_difference_{j}",
        #         {
        #             "current_diff_0_2": diff_0_2[j].item(),
        #             "current_diff_0_1": diff_0_1[j].item()
        #         },
        #         global_step=pl_module.global_step,
        #     )

        # # log the differences between the ground truth camera poses
        # if self.gt_diff_0_2 is not None and self.gt_diff_0_1 is not None:
        #     # pdb.set_trace()
        #     for j in range(self.gt_diff_0_2.shape[0]):
        #         tag_gt_diff_0_2 = f"gt_diff_0_2_{j}"
        #         tag_gt_diff_0_1 = f"gt_diff_0_1_{j}"
                
        #         pl_module.logger.experiment.add_scalars(
        #             f"camera_pose_difference_{j}",
        #             {
        #                 "gt_diff_0_2": self.gt_diff_0_2[j].item(), 
        #                 "gt_diff_0_1": self.gt_diff_0_1[j].item(),
        #             },
        #             global_step=pl_module.global_step,
        #         )

        # # log lpips psnr
        # pl_module.logger.experiment.add_scalars(
        #     f"Evaluate_Metrics",
        #     {
        #         "NVS_PSNR": self.nvs_psnr,
        #         "NVS_SSIM": self.nvs_ssim,
        #         "NVS_LPIPS": self.nvs_lpips, 
        #     },
        #     global_step=pl_module.global_step,
        # )

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        
        log_dict = {}

        for k in images:
            if k == 'multiview_images':
                pil_image = images[k]  # This is already a PIL image
                log_dict[f"{split}/multiview"] = wandb.Image(pil_image)

            else:
                # don't log these
                continue
                grid = torchvision.utils.make_grid(images[k])
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                pil_image = Image.fromarray(grid)
                tag = f"{split}/{k}"
                log_dict[tag] = wandb.Image(pil_image)

        # if pl_module.camera_poses is not None:
        #     for i in range(pl_module.camera_poses.weight.shape[0]):
        #         for j in range(pl_module.camera_poses.weight.shape[1]):
        #             pose_key = f"camera_poses_{i}_{j}"
        #             pose_value = pl_module.camera_poses.weight[i, j].item()
        #             log_dict[pose_key] = pose_value
                    
        # Calculate the difference between the 0th and 2th camera poses
        diff_0_2 = pl_module.camera_poses.weight[0] - pl_module.camera_poses.weight[2]
        
        # Calculate the difference between the 0th and 1th camera poses
        diff_0_1 = pl_module.camera_poses.weight[0] - pl_module.camera_poses.weight[1]

        # Log the differences
        for j in range(diff_0_2.shape[0]):
            log_dict[f"diff_0_2_{j}"] = diff_0_2[j].item()
            log_dict[f"diff_0_1_{j}"] = diff_0_1[j].item()
        
        print(pl_module.global_step)
        print("$$$")
        
        wandb.log(log_dict, step=pl_module.global_step)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isinstance(images[k], Image.Image):
                img_to_save = images[k]
            else:
                # don't log these
                continue
            
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            img_to_save.save(path)
            # tag = f"{split}/{k}"
            # wandb.log({tag: wandb.Image(img_to_save)})            

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
                pass
            return True
        return False

    @torch.no_grad()
    def log_img(self, trainer: Trainer, pl_module: pl.LightningModule, batch, batch_idx, split="train", init_flag=False):
        log.info(f"Log images at Epcoch {pl_module.current_epoch}")
        logger = type(pl_module.logger)

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        if not self.only_camera:
            images = {}
            with torch.no_grad():
                # * log evaluation
                # ret_val = pl_module.log_evaluation_images(
                #     batch,
                #     init_flag,
                #     **self.log_images_kwargs,
                # )
                # n_views = ret_val['n_views']

                # rgb_preds         = []
                # rgb_gts           = []
                # rgb_conds         = []
                # psnr_metric_mean  = []
                # # ssim_metric_mean  = []
                # lpips_metric_mean = []
                
                # for index in tqdm.tqdm(range(n_views), desc='computing evaluation results', leave=False):
                #     rgb_pred = ret_val[f'validation_synthesis_{index}'].cpu()
                #     rgb_gt   = ret_val[f'validation_target_{index}'].cpu()
                #     rgb_cond = ret_val[f'validation_cond_{index}'].cpu()

                #     # ensure both tensors are in the range [-1, 1]
                #     rgb_pred = (torch.clamp(rgb_pred, -1., 1.) + 1.0) / 2.0  # -1, 1 -> 0, 1; c, h, w
                #     rgb_gt   = (torch.clamp(rgb_gt, -1., 1.) + 1.0) / 2.0    # -1,   1 -> 0, 1; c, h, w
                #     rgb_cond = (torch.clamp(rgb_cond, -1., 1.) + 1.0) / 2.0

                #     # calculate metrics
                #     psnr_metric  = (-10. * torch.log10(torch.mean((rgb_pred-rgb_gt)**2))).item()
                #     # ssim_metric  = self.ssim_metric(rgb_pred, rgb_gt).item()
                #     lpips_metric = self.lpips_metric(rgb_pred.to('cuda'), rgb_gt.to('cuda')).mean().item()

                #     rgb_pred = to8b(rgb_pred[0].cpu().permute(1, 2, 0).numpy())
                #     rgb_gt   = to8b(rgb_gt[0].cpu().permute(1, 2, 0).numpy())
                #     rgb_cond = to8b(rgb_cond[0].cpu().permute(1, 2, 0).numpy())

                #     rgb_preds.append(rgb_pred)
                #     rgb_gts.append(rgb_gt)
                #     rgb_conds.append(rgb_cond)
                #     psnr_metric_mean.append(psnr_metric)
                #     # ssim_metric_mean.append(ssim_metric)
                #     lpips_metric_mean.append(lpips_metric)

                # # * update metrics
                # # for view synthesis
                # self.nvs_psnr  = np.mean(psnr_metric_mean)
                # # self.nvs_ssim  = np.mean(ssim_metric_mean)
                # self.nvs_lpips = np.mean(lpips_metric_mean)

                # for camera pose
                pose_pred  = pl_module.camera_poses.weight.detach().clone().cpu().numpy()
                pose_gt    = self.gt_abc_tensor.detach().clone().cpu().numpy().T
                train_view = len(pose_pred)

                elevation_pred = pose_pred[:, 0]
                elevation_gt   = pose_gt[:, 0]
                azimuth_pred   = pose_pred[:, 1]
                azimuth_gt     = pose_gt[:, 1]
                radius_pred    = pose_pred[:, 2]
                radius_gt      = pose_gt[:, 2]

                # delta_elevation_gt = [
                #     elevation_gt[i] - elevation_gt[0] for i in range(1, train_view)
                # ]
                # delta_azimuth_gt = [
                #     azimuth_gt[i] - azimuth_gt[0] for i in range(1, train_view) 
                # ]
                # delta_radius_gt = [
                #     radius_gt[i] - radius_gt[0] for i in range(1, train_view) 
                # ]
                
                # delta_elevation_pred = [
                #     elevation_pred[i] - elevation_pred[0] for i in range(1, train_view)
                # ]
                # delta_azimuth_pred = [
                #     azimuth_pred[i] - azimuth_pred[0] for i in range(1, train_view)
                # ]
                # delta_radius_pred = [
                #     radius_pred[i] - radius_pred[0] for i in range(1, train_view)
                # ]
                
                # elevation_error = np.mean(np.abs(np.array(delta_elevation_gt) - np.array(delta_elevation_pred)))
                # azimuth_error   = np.mean(np.abs(np.array(delta_azimuth_gt) - np.array(delta_azimuth_pred)))
                # radius_error    = np.mean(np.abs(np.array(delta_radius_gt) - np.array(delta_radius_pred)))

                # # print performance
                # log.info("========== Test Performance ==========")
                # log.info(f"ELEVATION (rad): {elevation_error:.3f} AZIMUTH (rad): {azimuth_error:.3f} RADIUS: {radius_error:.3f}")
                # log.info(f"PSNR: {self.nvs_psnr:.3f} SSIM: {self.nvs_ssim:.3f} LPIPS: {self.nvs_lpips:.3f}")
                
                # pl_module.log("TEST ELEVATION", elevation_error, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                # pl_module.log("TEST AZIMUTH", azimuth_error, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                # pl_module.log("TEST RADIUS", radius_error, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                # pl_module.log("TEST PSNR", self.nvs_psnr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                # pl_module.log("TEST SSIM", self.nvs_ssim, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                # pl_module.log("TEST LPIPS", self.nvs_lpips, on_step=True, on_epoch=False, prog_bar=False, logger=True)

                ret_save = {
                    'elevation_pred' : elevation_pred,
                    'elevation_gt'   : elevation_gt,
                    'azimuth_pred'   : azimuth_pred,
                    'azimuth_gt'     : azimuth_gt,
                    'radius_pred'    : radius_pred,
                    'radius_gt'      : radius_gt,
                    # 'elevation_error': elevation_error,
                    # 'azimuth_error'  : azimuth_error,
                    # 'radius_error'   : radius_error,
                    # 'nvs_psnr'       : self.nvs_psnr,
                    # 'nvs_ssim'       : self.nvs_ssim,
                    # 'nvs_lpips'      : self.nvs_lpips,
                    # 'rgb_preds'      : np.stack(rgb_preds),
                    # 'rgb_gts'        : np.stack(rgb_gts),
                    # 'rgb_conds'      : np.stack(rgb_conds),
                }

                save_folder = f'{pl_module.logger.save_dir}/evaluation/epoch_99'
                os.makedirs(f'{save_folder}', exist_ok=True)
                # os.makedirs(f'{save_folder}/images/gts', exist_ok=True)
                # os.makedirs(f'{save_folder}/images/preds', exist_ok=True)
                # os.makedirs(f'{save_folder}/images/conds', exist_ok=True)
                
                # for index in tqdm.tqdm(range(len(rgb_preds)), desc='saving evaluation results', leave=False):
                #     img_pred = Image.fromarray(rgb_preds[index])
                #     img_pred.save(f'{save_folder}/images/preds/{index:03d}.png')
                #     img_gt = Image.fromarray(rgb_gts[index])
                #     img_gt.save(f'{save_folder}/images/gts/{index:03d}.png')
                #     img_cond = Image.fromarray(rgb_conds[index])
                #     img_cond.save(f'{save_folder}/images/conds/{index:03d}.png')
                torch.save(ret_save, f'{save_folder}/results.tar')

                # # * log visualization
                # ret_val = pl_module.log_visualization_images(
                #     batch,
                #     init_flag,
                #     **self.log_images_kwargs,
                # )
                # n_views = ret_val['n_views']

                # rgb_preds = []
                # rgb_conds = []
                
                # for index in tqdm.tqdm(range(n_views), desc='computing visualization results', leave=False):
                #     rgb_pred = ret_val[f'validation_synthesis_{index}'].cpu()
                #     rgb_cond = ret_val[f'validation_cond_{index}'].cpu()
                #     # ensure both tensors are in the range [-1, 1]
                #     rgb_pred = (torch.clamp(rgb_pred, -1., 1.) + 1.0) / 2.0  # -1, 1 -> 0, 1; c, h, w
                #     rgb_cond = (torch.clamp(rgb_cond, -1., 1.) + 1.0) / 2.0
                #     rgb_pred = to8b(rgb_pred[0].cpu().permute(1, 2, 0).numpy())
                #     rgb_cond = to8b(rgb_cond[0].cpu().permute(1, 2, 0).numpy())
                #     rgb_preds.append(rgb_pred)
                #     rgb_conds.append(rgb_cond)

                # ret_save = {
                #     'rgb_preds' : np.stack(rgb_preds),
                #     'rgb_conds' : np.stack(rgb_conds),
                # }

                # save_folder = f'{pl_module.logger.save_dir}/visualization/epoch_{pl_module.current_epoch}'
                # os.makedirs(f'{save_folder}/images/preds', exist_ok=True)
                # os.makedirs(f'{save_folder}/images/conds', exist_ok=True)

                # for index in tqdm.tqdm(range(len(rgb_preds)), desc='saving visualization results', leave=False):
                #     img_pred = Image.fromarray(rgb_preds[index])
                #     img_pred.save(f'{save_folder}/images/preds/{index:03d}.png')
                #     img_cond = Image.fromarray(rgb_conds[index])
                #     img_cond.save(f'{save_folder}/images/conds/{index:03d}.png')
                # torch.save(ret_save, f'{save_folder}/results.tar')

        logger_log_images = self.logger_log_images.get(
            logger, 
            lambda *args, 
            **kwargs: None
        )
        logger_log_images(
            pl_module, 
            images, 
            pl_module.global_step, 
            split
        )

        if is_train:
            pl_module.train()

    def on_train_batch_start(
        self, 
        trainer: Trainer, 
        pl_module: pl.LightningModule, 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if pl_module.current_epoch==0 and batch_idx==0:
            pass
            # self.log_img(trainer, pl_module, batch, batch_idx, split="train", init_flag=True)

    def on_train_batch_end(      
        self, 
        trainer: Trainer, 
        pl_module: pl.LightningModule, 
        outputs, 
        batch, 
        batch_idx, 
        dataloader_idx,
    ) -> None:
        if pl_module.current_epoch==(trainer.max_epochs-1) and (batch_idx+1)==int(batch["train_view"]*(batch["train_view"]-1)):
            # pass
            self.log_img(trainer, pl_module, batch, batch_idx, split="train", init_flag=False)

    def on_validation_batch_end(      
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx, 
        dataloader_idx,
    ) -> None:
        pass
        # if not self.disabled and pl_module.global_step > 0:
        #     self.log_img(trainer, pl_module, batch, batch_idx, split="val")
            
        # if hasattr(pl_module, 'calibrate_grad_norm'):
        #     if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
        #         self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        # torch.cuda.synchronize(trainer.root_gpu)
        # self.start_time = time.time()
        pass

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pass
        # torch.cuda.synchronize(trainer.root_gpu)
        # max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        # epoch_time = time.time() - self.start_time

        # try:
        #     max_memory = trainer.training_type_plugin.reduce(max_memory)
        #     epoch_time = trainer.training_type_plugin.reduce(epoch_time)

        #     rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        #     rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        # except AttributeError:
        #     pass

# todo add into config
class CheckPointCallback(Callback):
    def __init__(self, save_flag, ckpt_path, save_epoch_interval) -> None:
        super().__init__()
        self.save_flag = save_flag
        self.ckpt_path = ckpt_path
        self.save_epoch_interval = save_epoch_interval

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     if pl_module.current_epoch%self.save_epoch_interval == 0 and self.save_flag:
    #         log.info(f"=====save checkpoint at Epoch {pl_module.current_epoch}=====")
    #         # trainer.save_checkpoint(f"{self.ckpt_path}/best.ckpt")
    #         print("Done.")

    def on_train_end(self, trainer, pl_module: pl.LightningModule) -> None:
            if self.save_flag:
                log.info(f"=====save checkpoint at Epoch {pl_module.current_epoch}=====")
                # trainer.save_checkpoint(f"{self.ckpt_path}/last.ckpt")
                save_ret = {
                    'state_dict': pl_module.state_dict(),
                    'global_step': 101,
                }
                torch.save(save_ret, f"{self.ckpt_path}/last.ckpt")
                print("Done.")


if __name__ == "__main__":
    # * Date for Recording
    current_time = datetime.datetime.utcnow().replace(tzinfo=pytz.UTC)
    pacific_tz = pytz.timezone('US/Pacific')
    current_time = current_time.astimezone(pacific_tz)
    current_time_str = current_time.strftime("%Y-%m-%dT%H-%M-%S")
    now = current_time_str

    # * add cwd for making classes (in particular `main.DataModuleFromConfig`) available 
    # * when running as `python main.py`
    sys.path.append(os.getcwd())

    # * get training configures
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    log.options(vars(opt))

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)


    # * init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "ddp"

    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        rank_zero_print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    lr_ratio = config['model']['params']['lr_ratio']
    lr_min = config['model']['params']['scheduler_config']['params']['lr_min']
    lr_max = config['model']['params']['scheduler_config']['params']['lr_max']
    lr_start = config['model']['params']['scheduler_config']['params']['lr_start']

    # * path to save ckpt and log results
    nowname = nowname + '-' + str(config['data']['params']['batch_size']) + '-' + str(lr_ratio) + '-' + str(lr_min) + '-' + str(lr_max) + '-' + str(lr_start)
    project_name = opt.project_name
    logdir = os.path.join(opt.logdir, project_name, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    log.title(f"#### {logdir} ####")
    seed_everything(opt.seed)

    # * initialize model
    model = instantiate_from_config(config.model)

    if not opt.finetune_from == "":
        rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
        old_state = torch.load(opt.finetune_from, map_location="cpu")

        if "state_dict" in old_state:
            rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]

        # * check if we need to port weights from 4ch input to 8ch
        in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
        new_state = model.state_dict()
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        in_shape = in_filters_current.shape
        if in_shape != in_filters_load.shape:
            input_keys = [
                "model.diffusion_model.input_blocks.0.0.weight",
                "model_ema.diffusion_modelinput_blocks00weight",
            ]
            
            for input_key in input_keys:
                if input_key not in old_state or input_key not in new_state:
                    continue
                input_weight = new_state[input_key]
                if input_weight.size() != old_state[input_key].size():
                    print(f"Manual init: {input_key}")
                    input_weight.zero_()
                    input_weight[:, :4, :, :].copy_(old_state[input_key])
                    old_state[input_key] = torch.nn.parameter.Parameter(input_weight)

        m, u = model.load_state_dict(old_state, strict=False)

        if len(m) > 0:
            rank_zero_print("missing keys:")
            rank_zero_print(m)
        if len(u) > 0:
            rank_zero_print("unexpected keys:")
            rank_zero_print(u)

    # * trainer callbacks
    trainer_kwargs = dict()

    # * default logger configs
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": project_name,
                "version": nowname,
            }
        },
    }
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
    # specify which metric is used to determine best models
    # default_modelckpt_cfg = {
    #     "target": "pytorch_lightning.callbacks.ModelCheckpoint",
    #     "params": {
    #         "dirpath": ckptdir,
    #         "filename": "{epoch:06}",
    #         "verbose": True,
    #         "save_last": False,
    #     }
    # }

    # if hasattr(model, "monitor"):
    #     rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
    #     default_modelckpt_cfg["params"]["monitor"] = model.monitor
    #     default_modelckpt_cfg["params"]["save_top_k"] = 3

    # if "modelcheckpoint" in lightning_config:
    #     modelckpt_cfg = lightning_config.modelcheckpoint
    # else:
    #     modelckpt_cfg =  OmegaConf.create()
    # modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    # rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
    # if version.parse(pl.__version__) < version.parse('1.4.0'):
    #     trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume"          : opt.resume,
                "now"             : now,
                "logdir"          : logdir,
                "ckptdir"         : ckptdir,
                "cfgdir"          : cfgdir,
                "config"          : config,
                "lightning_config": lightning_config,
                "debug"           : opt.debug,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {
                "batch_frequency": 5,
                "max_images"     : 6400,
                "clamp"          : True
            }
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            }
        },
        "cuda_callback": {
            "target": "main.CUDACallback"
        },
        "ckpt_callback": {
            "target": "main.CheckPointCallback",
            "params": {
                "save_flag"          : False,
                "ckpt_path"          : ckptdir,
                "save_epoch_interval": 500,
            }
        },
    }
    # if version.parse(pl.__version__) >= version.parse('1.4.0'):
    #     default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()

    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
    elif 'ignore_keys_callback' in callbacks_cfg:
        del callbacks_cfg['ignore_keys_callback']

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    if not "plugins" in trainer_kwargs:
        trainer_kwargs["plugins"] = list()
    if not lightning_config.get("find_unused_parameters", True):
        from pytorch_lightning.plugins import DDPPlugin
        trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))

    # * PL TRAINER
    trainer_opt.val_check_interval = 1e100
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir
    log.info("#### Trainer Configure ####")
    log.options(vars(trainer_opt))

    # * PL DATA
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    log.info("#### Data Configure ####")
    log.options(OmegaConf.to_container(config.data))
    try:
        for k in data.datasets:
            log.info(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    except:
        log.info("datasets not yet initialized.")

    # * PL LR
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    log.info(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        log.info(
            f"setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate_grad_batches) * {ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)"
        )
    else:
        model.learning_rate = base_lr
        log.info("++++ NOT USING LR SCALING ++++")
        log.info(f"Setting learning rate to {model.learning_rate:.2e}")


    # # allow checkpointing via USR1
    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         log.info("Summoning checkpoint.")
    #         ckpt_path = os.path.join(ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)


    # def divein(*args, **kwargs):
    #     if trainer.global_rank == 0:
    #         import pudb
    #         pudb.set_trace()


    # import signal

    # signal.signal(signal.SIGUSR1, melk)
    # signal.signal(signal.SIGUSR2, divein)

    # * PL run
    if opt.train:
        trainer.fit(model, data)
    # if not opt.no_test and not trainer.interrupted:
    #     trainer.test(model, data)

    # move newly created debug project to debug_runs
    if opt.debug and not opt.resume and trainer.global_rank == 0:
        dst, name = os.path.split(logdir)
        dst = os.path.join(dst, "debug_runs", name)
        os.makedirs(os.path.split(dst)[0], exist_ok=True)
        os.rename(logdir, dst)
    # if trainer.global_rank == 0:
    #     rank_zero_print(trainer.profiler.summary())
