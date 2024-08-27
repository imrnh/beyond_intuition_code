"""
    Train the models for given saliency map.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.model_loaders import vit_base_patch16_224
from utils.imagenet_loader import ImagenetLoader

np.random.seed(0)
torch.manual_seed(0)


class ModelTrainer:
    def __init__(self, data_path, epochs, batch_size, num_workers, lr, weight_decay) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        self.save_dir = "lib/benchmark__trained_on_noisy_data/"
        self.data_path = data_path

        self.model = vit_base_patch16_224(pretrained=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs

        self.lr = lr
        self.weight_decay = weight_decay

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = CrossEntropyLoss()

        # Data loading and transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.label_transform = transforms.Compose([transforms.Resize((224, 224), Image.NEAREST), ])

        self.dataset = ImagenetLoader(self.data_path + "/train", self.data_length, transform=self.image_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.dataloader = tqdm(self.dataloader)  # Would help tracking loop iteration along with setting some verbose text.

        self.validation_dataset = ImagenetLoader(self.data_path + "/validation", self.data_length, transform=self.image_transform)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.validation_dataloader = tqdm(self.validation_dataloader)



    """
        Train the model and finally save it.
    """
    def train(self):
        for epoch in range(self.epochs):
            train_loss = 0.0
            for batch in tqdm(self.dataloader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                train_loss += loss.detach().cpu().item() / len(self.dataloader)

                self.optimizer.zero_grad()
                loss.backward()  # backprop calculation
                self.optimizer.step()  # Updating the weight based on these calculation.

            print(f"Epoch {epoch + 1}/{self.epochs} loss: {train_loss:.2f}")

        self.save_model()

    def save_model(self, model_name):
        os.makedirs(self.save_dir, exist_ok=True)
        
        model_path = os.path.join(self.save_dir, f"jx_vit_base_p16_224_{model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    trainer = ModelTrainer(data_path="", epochs=10, batch_size=32, num_workers=4, lr=0.0004, weight_decay=0.001)