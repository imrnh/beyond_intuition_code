"""
    Train the models for given saliency map.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.model_loaders import vit_base_patch16_224

np.random.seed(0)
torch.manual_seed(0)


class ModelTrainer:
    def __init__(self, data_path, epochs, batch_size, num_workers, lr, weight_decay) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        self.save_dir = "/kaggle/working/custom_trained_models/"
        self.data_path = data_path
        self.train_logs = []


        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Model, Optimizer and Loss function setup
        self.model = vit_base_patch16_224(pretrained=True).cuda()

        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = CrossEntropyLoss()

        # Data loading and transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.dataset = datasets.ImageFolder(root=self.data_path + "/train", transform=self.image_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)

        self.validation_dataset = datasets.ImageFolder(root=self.data_path + "/test", transform=self.image_transform)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
    
        
        
    """
        Train the model and finally save it.
    """
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = 0.0
            val_loss = 0.0
            val_accuracy = 0.0
            
            # Training
#             for batch in self.dataloader:
#                 x, y = batch
#                 x, y = x.cuda(), y.cuda()
#                 y_hat = self.model(x)
#                 loss = self.criterion(y_hat, y)
#                 train_loss += loss.detach().cpu().item() / len(self.dataloader)

#                 loss.backward()  # backprop calculation
#                 self.optimizer.step()  # Updating the weight based on these calculation.
#                 self.optimizer.zero_grad()

            # Validation
            rr = 0
            for batch in self.validation_dataloader:
                xv, yv = batch
                xv, yv = xv.cuda(), yv.cuda()
                yv_hat = self.model(xv)
                vloss = self.criterion(yv_hat, yv)
                
                val_loss += vloss.detach().cpu().item() / len(self.validation_dataloader)

                val_accuracy += self.measure_accuracy(yv_hat, yv) / len(self.validation_dataloader)
                
                rr += 1
                if rr > 50:
                    break

                
            self.callbacks(train_loss, val_loss, val_accuracy, epoch)

        
    def save_model(self, model_name):
        os.makedirs(self.save_dir, exist_ok=True)
        
        model_path = os.path.join(self.save_dir, f"jx_vit_base_p16_224_{model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def embd_to_class(self, batched_embed):
        class_indices = []
        for embd in batched_embed:
            cls_idx = torch.argmax(embd)
            class_indices.append(cls_idx)
        return torch.Tensor(class_indices)
    
    def measure_accuracy(self, yhat, y):
        yhat_idx = self.embd_to_class(yhat)

        accuracy = 0.0
        for yh, yval in zip(yhat_idx, y):
            accuracy += 1 if (yh == yval) else 0
            
        return accuracy
    

    def callbacks(self, t_loss, v_loss, vacc, epoch, verbose=True):
        self.train_logs.append({'train_loss': t_loss, 'val_loss': v_loss, 'val_accuracy': vacc})
        
        if verbose:
            print(f"""tr_loss: {t_loss:.5f} \t val_loss: {v_loss:.5f} \t val_acc: {vacc:.5f}""")
            
        # Save best model per epoch
        if epoch > 1:
            prev_vl = self.train_logs[-2]['val_loss']
            prev_vacc = self.train_logs[-2]['val_accuracy']
            
            print(f"prev_vl = {prev_vl} \t  prev_acc: {prev_vacc}")
            print(f"curr_vl = {v_loss} \t curr_vacc: {vacc}")

            # Save 2 state. 1 for low val loss. 1 for highest val acc.
            if vacc >= prev_vacc :
                self.save_model(f"raw_images_{str(epoch)}_max_validation_accuracy")  # ___________________________ CHANGE IT BEFORE TRAINING PLEASE....
                print(f"Saved for acc:{vacc}")

            if v_loss <= prev_vl :
                self.save_model(f"raw_images_{str(epoch)}_min_validation_loss")  # ___________________________ CHANGE IT BEFORE TRAINING PLEASE....
                print(f"Saved for loss: {v_loss}")