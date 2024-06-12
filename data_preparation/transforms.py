# data/transforms.py

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class Scale:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, image, mask = None):
        width, height = image.size
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        image = image.resize((new_width, new_height), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_width, new_height), Image.NEAREST)
            return image, mask
        return image

class LabelMaskToTensor:
    def __call__(self, image, mask = None):
        image = transforms.ToTensor()(image)
        if mask is not None:
            mask = torch.tensor(np.array(mask), dtype=torch.long)
            return image, mask
        return image

class LabelingToAssignment:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, image, mask = None):
        if mask is not None:
            mask = torch.nn.functional.one_hot(mask, num_classes=self.num_classes).permute(2, 0, 1)
            return image, mask
        return image

class SmoothSimplexCorners:
    def __init__(self, smoothing):
        self.smoothing = smoothing

    def __call__(self, image, mask = None):
        if mask is not None:
            mask = mask.float() * self.smoothing
            return image, mask
        return image

class ToTensor:
    def __call__(self, image, mask = None):
        image = transforms.ToTensor()(image)
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.long)
            return image, mask
        return image

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        if mask is not None:
            image = transforms.Normalize(mean=self.mean, std=self.std)(image)
            return image, mask
        return image
