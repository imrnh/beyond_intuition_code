"""
    Run this file to generate output
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import imageio, warnings, os

from interpretation_methods import *
from utils.imagenet_seg_loader import ImagenetSegLoader
from utils.model_loaders import vit_base_patch16_224_dino, vit_base_patch16_224, vit_small_patch16_224
from utils.input_arguments import get_arg_parser
from utils.saver import Saver
from utils.sideplot import side_plot
from utils.image_denorm import image_vizformat

warnings.filterwarnings("ignore")
plt.switch_backend("agg")


class Main:
    def __init__(self, batch_size=1, num_workers=1, alpha=2, tracker_file="heatmaps"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.alpha = alpha
        self.tracker_file = tracker_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.argument_parser = get_arg_parser()
        self.args = self.argument_parser.parse_args()
        self.args.checkname = self.args.method + '_' + self.args.arc

        self.data_path = self.args.imagenet_seg_path
        self.data_length = self.args.len_lim

        self.saver = Saver(self.args)
        self.saver.create_directories()

        # Data loading and transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.label_transform = transforms.Compose([transforms.Resize((224, 224), Image.NEAREST), ])

        self.dataset = ImagenetSegLoader(self.data_path, self.data_length, transform=self.image_transform, target_transform=self.label_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.dataloader = tqdm(self.dataloader)  # Would help tracking loop iteration along with setting some verbose text.

        # Model loading
        self.model = vit_small_patch16_224(pretrained=True).to(self.device)
        self.lrp_model = vit_base_patch16_224(pretrained=True, w_rel=True).to(self.device)
        self.model.eval()  # Set to eval mode. Helps tarcking gradient.

        # Track input and heatmap
        self.resulting_heatmaps = []
        self.input_images = []

    def write_tracker_file(self, tracker_dict, file_name):
        torch.save(tracker_dict, file_name)
        

    def generate_heatmap(self, image, int_method):        
        self.model.zero_grad()
        self.lrp_model.zero_grad()

        image.requires_grad = True
        image = image.requires_grad_()
        
        self.model(image)  # Forward pass to calculate gradients.

        if int_method == 'bi_t':
            heatmap, _ = beyond_intuition_tokenwise(self.model, image, self.device, dino=True, start_layer=self.args.start_layer, taken_head_idx=None)

        elif int_method == 'bi_h':
            heatmap = beyond_intuition_headwise(self.model, image, self.device, dino=True, start_layer=self.args.start_layer)

        elif int_method == 'rollout':
            heatmap = rollout_attention(self.model, image, self.device, start_layer=1)

        elif int_method == 'transformer_attribution':
            heatmap = layerwise_relevance_propagation(self.lrp_model, image, self.device, start_layer=1, method="transformer_attribution")

        elif int_method == 'attn_gradcam':
            heatmap = cam_attn(self.model, image, self.device)

        elif int_method == 'attn_last_layer':
            heatmap = raw_attention_map(self.model, image, self.device)

        elif int_method == 'generic_attribution':
            heatmap = generic_attribution(self.model, image, self.device)
        else:
            heatmap = torch.zeros_like(image)  # If no method specified, return zeroed heatmap.

        heatmap = heatmap.reshape(self.batch_size, 1, 14, 14)
        if int_method != 'full_lrp':  # Interpolate to full image size (224,224)
            heatmap = torch.nn.functional.interpolate(heatmap, scale_factor=16, mode='bilinear').to(self.device)

        return heatmap

    def generate_explanation(self, limit=None):
        heatmaps = dict()
        for batch_idx, (image, labels) in enumerate(self.dataloader):
            image = image.to(self.device)
            labels = labels.to(self.device)
            heatmaps[batch_idx] = self.generate_heatmap(image, "bi_t")

            if limit is not None and limit <= batch_idx:
                break
                        
        self.write_tracker_file(heatmaps, "draft/heatmaps/heatmap_default_model.pth")


if __name__ == '__main__':
    explainer = Main()
    explainer.generate_explanation(limit=5)