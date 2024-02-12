import numpy as np

from dataset.co3d_v2 import Co3dDataset
from dataset.finetune import FineTune_Dataset

def compute_angular_error(rotation1, rotation2):
    R_rel = rotation1.T @ rotation2
    tr = (np.trace(R_rel) - 1) / 2
    theta = np.arccos(tr.clip(-1, 1))
    return theta * 180 / np.pi


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def get_dataset(
    dataset_type='finetune',
    category=None,
    num_images=None,
    split="test",
    eval_time=True,
    normalize_cameras=False,
    first_camera_transform=False,
):
    if dataset_type == 'co3d':
        dataset = Co3dDataset(
            split=split,
            category=[category],
            num_images=num_images,
            eval_time=eval_time,
            normalize_cameras=normalize_cameras,
            first_camera_transform=first_camera_transform,
        )
    elif dataset_type == 'finetune':
        dataset = FineTune_Dataset(
            split=split,
            category=[category],
            num_images=num_images,
            eval_time=eval_time,
            normalize_cameras=normalize_cameras,
            first_camera_transform=first_camera_transform,
        )
    return dataset
