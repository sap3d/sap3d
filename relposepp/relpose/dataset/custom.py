import os
import os.path as osp

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from utils.bbox import mask_to_bbox, square_bbox


class CustomDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        bboxes=None,
        mask_images=False,
        num_to_eval=3,
    ):
        """
        Dataset for custom images. If mask_dir is provided, bounding boxes are extracted
        from the masks. Otherwise, bboxes must be provided.
        """
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.mask_images = mask_images
        self.bboxes      = []
        self.images      = []
        self.num_to_eval = num_to_eval

        if mask_images:
            for image_name, mask_name in tqdm(zip(sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir)))):
                img_cache = imageio.imread(osp.join(image_dir, image_path))
                if img_cache.shape[-1] == 4:
                    img_rgba = Image.open(osp.join(image_dir, image_path))
                    image = Image.new('RGB', size=img_rgba.size, color=(255, 255, 255))
                    image.paste(img_rgba, (0, 0), mask=img_rgba)
                else:
                    image = Image.open(osp.join(image_dir, image_path))
                mask = Image.open(osp.join(mask_dir, mask_name)).convert("L")
                white_image = Image.new("RGB", image.size, (255, 255, 255))
                if mask.size != image.size:
                    mask = mask.resize(image.size)
                mask = Image.fromarray(np.array(mask) > 125)
                image = Image.composite(image, white_image, mask)
                self.images.append(image)
        else:
            for image_path in sorted(os.listdir(image_dir)):
                img_cache = imageio.imread(osp.join(image_dir, image_path))
                if img_cache.shape[-1] == 4:
                    img_rgba = Image.open(osp.join(image_dir, image_path))
                    image = Image.new('RGB', size=img_rgba.size, color=(255, 255, 255))
                    image.paste(img_rgba, (0, 0), mask=img_rgba)
                else:
                    image = Image.open(osp.join(image_dir, image_path))
                self.images.append(image)
        self.n = len(self.images)
        if bboxes is None:
            for mask_path in sorted(os.listdir(mask_dir))[: self.n]:
                mask = plt.imread(osp.join(mask_dir, mask_path))
                if len(mask.shape) == 3:
                    mask = mask[:, :, :3]
                else:
                    mask = np.dstack([mask, mask, mask])
                self.bboxes.append(mask_to_bbox(mask))
        else:
            self.bboxes = bboxes
        self.jitter_scale = [1.15, 1.15]
        self.jitter_trans = [0, 0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return 1

    def _jitter_bbox(self, bbox):
        bbox = square_bbox(bbox.astype(np.float32))
        s = np.random.uniform(self.jitter_scale[0], self.jitter_scale[1])
        tx, ty = np.random.uniform(self.jitter_trans[0], self.jitter_trans[1], size=2)

        side_length = bbox[2] - bbox[0]
        center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
        extent = side_length / 2 * s

        # Final coordinates need to be integer for cropping.
        ul = (center - extent).round().astype(int)
        lr = ul + np.round(2 * extent).astype(int)
        return np.concatenate((ul, lr))

    def _crop_image(self, image, bbox, white_bg=False):
        if white_bg:
            # Only support PIL Images
            image_crop = Image.new(
                "RGB", (bbox[2] - bbox[0], bbox[3] - bbox[1]), (255, 255, 255)
            )
            image_crop.paste(image, (-bbox[0], -bbox[1]))
        else:
            image_crop = transforms.functional.crop(
                image,
                top=bbox[1],
                left=bbox[0],
                height=bbox[3] - bbox[1],
                width=bbox[2] - bbox[0],
            )
        return image_crop

    def __getitem__(self, index):
        return self.get_data()

    def get_data(self, ids=(0, 1, 2)):
        ids = range(self.num_to_eval)
        images = [self.images[i] for i in ids]
        bboxes = [self.bboxes[i] for i in ids]
        images_transformed = []
        crop_parameters = []
        for _, (bbox, image) in enumerate(zip(bboxes, images)):
            w, h = image.width, image.height
            bbox = np.array(bbox)
            bbox_jitter = self._jitter_bbox(bbox)
            image = self._crop_image(image, bbox_jitter, white_bg=self.mask_images)
            images_transformed.append(self.transform(image))
            crop_center = (bbox_jitter[:2] + bbox_jitter[2:]) / 2
            cc = (2 * crop_center / min(h, w)) - 1
            crop_width = 2 * (bbox_jitter[2] - bbox_jitter[0]) / min(h, w)

            crop_parameters.append(torch.tensor([-cc[0], -cc[1], crop_width]).float())
        images = images_transformed

        batch = {}
        batch["image"] = torch.stack(images)
        batch["n"] = len(images)
        batch["crop_params"] = torch.stack(crop_parameters)

        return batch
