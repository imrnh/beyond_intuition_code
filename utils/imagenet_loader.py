import torch
import torch.utils.data as data

import os
import h5py
import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class ImagenetLoader(data.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform

        self.dataset = datasets.ImageFolder(root=path, transform=self.transform)

    def __getitem__(self, index):
        img, target = self.dataset[index]
                
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)