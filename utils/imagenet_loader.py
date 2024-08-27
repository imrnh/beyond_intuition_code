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
    def __init__(self, path, class_limit=100, length_limit=None, transform=None):
        self.path = path
        self.transform = transform
        self.length_limit = length_limit
        self.class_limit = list(range(class_limit))

        self.sub_dir = os.listdir(path)
        self.label_and_category_tracker = dict()

        self.read_items()

    def read_items(self):
        imagenet_dataset = datasets.ImageFolder(root=self.path, transform=self.transform)

        # Filter the dataset to include only the top k classes
        class_to_idx = imagenet_dataset.class_to_idx
        k_classes = [k for k, v in class_to_idx.items() if v in self.class_limit]
        k_idx = [i for i, (path, _) in enumerate(imagenet_dataset.samples) if imagenet_dataset.classes[imagenet_dataset.targets[i]] in k_classes]

        # Create a subset of the top 100 classes
        self.dataset = Subset(imagenet_dataset, k_idx)

    def __getitem__(self, index):
        img_path, target = self.dataset[index]
        
        # Load and process the image
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.dataset)