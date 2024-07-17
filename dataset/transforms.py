
# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as nF
import numpy as np
import cv2
from PIL import Image, ImageChops
from utils.box_ops import box_xyxy_to_cxcywh
from typing import Tuple, List
from utils.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target["masks"] = target["masks"][:, i : i + h, j : j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            # keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            keep = target["area"] > 1.
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]
        ) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(
        float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size)
    )
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height]
        )
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = (
            interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0]
            > 0.5
        )

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(
            target["masks"], (0, padding[0], 0, padding[1])
        )
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std, reparam=False):
        self.mean = mean
        self.std = std
        self.reparam = reparam

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            if self.reparam:
                boxes = boxes.to(torch.float32)
            else:
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string



class RandomShadows(object):
    def __init__(self, p=0.5, high_ratio=(1,2), low_ratio=(0.01, 0.5), left_low_ratio=(0.4,0.6), \
    left_high_ratio=(0,0.2), right_low_ratio=(0.4,0.6), right_high_ratio = (0,0.2)):
        self.p = p
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.left_low_ratio = left_low_ratio
        self.left_high_ratio = left_high_ratio
        self.right_low_ratio = right_low_ratio
        self.right_high_ratio = right_high_ratio

    @staticmethod
    def process(img, high_ratio, low_ratio, left_low_ratio, left_high_ratio, \
            right_low_ratio, right_high_ratio):

        w, h = img.size
        high_bright_factor = random.uniform(high_ratio[0], high_ratio[1])
        low_bright_factor = random.uniform(low_ratio[0], low_ratio[1])

        left_low_factor = random.uniform(left_low_ratio[0]*h, left_low_ratio[1]*h)
        left_high_factor = random.uniform(left_high_ratio[0]*h, left_high_ratio[1]*h)
        right_low_factor = random.uniform(right_low_ratio[0]*h, right_low_ratio[1]*h)
        right_high_factor = random.uniform(right_high_ratio[0]*h, right_high_ratio[1]*h)

        tl = (0, left_high_factor)
        bl = (0, left_high_factor+left_low_factor)

        tr = (w, right_high_factor)
        br = (w, right_high_factor+right_low_factor)

        contour = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros([h, w, 3],np.uint8)
        cv2.fillPoly(mask,[contour],(255,255,255))
        inverted_mask = cv2.bitwise_not(mask)
        # we need to convert this cv2 masks to PIL images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # we skip the above convertion because our mask is just black and white
        mask_pil = Image.fromarray(mask)
        inverted_mask_pil = Image.fromarray(inverted_mask)

        low_brightness = F.adjust_brightness(img, low_bright_factor)
        low_brightness_masked = ImageChops.multiply(low_brightness, mask_pil)
        high_brightness = F.adjust_brightness(img, high_bright_factor)
        high_brightness_masked = ImageChops.multiply(high_brightness, inverted_mask_pil)

        return ImageChops.add(low_brightness_masked, high_brightness_masked)

    def __call__(self, img, target=None):
        if random.uniform(0, 1) < self.p:
            img = self.process(img, self.high_ratio, self.low_ratio, \
            self.left_low_ratio, self.left_high_ratio, self.right_low_ratio, \
            self.right_high_ratio)
            return img, target
        else:
            return img, target




class Normalize2(object):
    def __init__(self, mean, std, enc_img_size=1024):
        self.pixel_mean = torch.Tensor(mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(std).view(-1, 1, 1)
        self.enc_img_size = enc_img_size

    def __call__(self, image, target=None):
        image = torch.as_tensor(image)
        image = image.permute(2, 0, 1).contiguous()
        image = (image - self.pixel_mean) / self.pixel_std
        h, w = image.shape[-2:]
        padh = self.enc_img_size - h
        padw = self.enc_img_size - w
        image = nF.pad(image, (0, padw, 0, padh))


        return image, target
    
class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def __call__(self, image: PIL.Image, target = None) -> np.ndarray:

        target_size = self.get_preprocess_shape(image.size[1], image.size[0], self.target_length)
        rescaled_image = np.array(F.resize(image, target_size))
        if target is None:
            return rescaled_image, None

        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(target_size[::-1], image.size)
        )
        ratio_width, ratio_height = ratios

        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height]
            )
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = target_size
        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target["masks"] = (
                interpolate(target["masks"][:, None].float(), target_size, mode="nearest")[:, 0]
                > 0.5
            )


        return rescaled_image, target

    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)