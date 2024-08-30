"""
For any given model & interpretation method, generate the saliency map. 
Saliency map data is stored in a pytorch file inside benchmark folder.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from interpretation_methods import beyond_intuition_tokenwise, beyond_intuition_headwise, integrated_gradient
from utils.model_loaders import vit_base_patch16_224_dino, vit_base_patch16_224
from utils.imagenet_seg_loader import ImagenetSegLoader
from utils.image_denorm import image_vizformat
from torchvision import datasets
from hyp.helpers import mismatched_image_corruption, find_mismatched_tokens
from hyp.corrupt_data import get_corrupted_image
from tqdm.auto import tqdm
from PIL import Image


class SaliencyGenerator(nn.Module):

    """
        @param corrupt_percent: used for hypothesis 1.
        @param corrupt_count: used for hypothesis 2 to get more control.
        @param hyp2_pat : patience i.e. the abs difference to be accredated for hypothesis 2.
    """

    def __init__(self, corrupt_percent, corrupt_count, hyp2_pat, verbose=False, data_lim=2000) -> None:
        super().__init__()
        self.corrupt_percent = corrupt_percent
        self.corrupt_count = corrupt_count
        self.hyp2_pat = hyp2_pat
        self.data_path = "content/lib/dataset/gtsegs_ijcv.mat"
        self.data_length = data_lim
        self.batch_size = 1
        self.num_workers = 1
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.saliency_information = dict()

        # Data loading and transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.dataset = datasets.ImageFolder(self.data_path, transform=self.image_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.dataloader = tqdm(self.dataloader)  # Would help tracking loop iteration along with setting some verbose text.

        # Model loading
        self.model = vit_base_patch16_224_dino(pretrained=True).to(self.device)
        self.model.eval()  # Set to eval mode. Helps tarcking gradient.

        self.saliency_writer = lambda output_path: torch.save(self.saliency_information, output_path)
        self.accuracies = {'default': 0}


        print("""Please select a valid saliency method while calling the forward pass. Valid saliency methods are: 
                ig -> Integrated Gradient \n
                bi_t -> Tokenwise Beyond Intuition
                bi_h -> Headwise Beyond Intuition
              """)

    def find_accuracy(self, img, y):
        out = self.model(img)
        index = torch.argmax(out)
        return 1 if index == y else 0


    def make_heatmap(self, image, saliency):
        if saliency == "ig": heatmap = integrated_gradient(self.model, image, self.device);
        elif saliency == "bi_t": heatmap, _ = beyond_intuition_tokenwise(self.model, image, self.device, dino=True, start_layer=0);
        elif saliency == "bi_h": heatmap = beyond_intuition_headwise(self.model, image, self.device, dino=True, start_layer=self.args.start_layer);

        return heatmap.reshape((14, 14)).detach().cpu()
        

    def forward(self, saliency="bi_t"):
        for batch_idx, (image, target) in enumerate(self.dataloader):
            # Heatmap without any perturbation
            rig_hmap, rbit_hmap = self.make_heatmap(image, saliency="ig"), self.make_heatmap(image, saliency="bit")

            # Track accuracies
            self.accuracies['default'] += self.find_accuracy(image, target) # Update default accuracy   

            corrupted_stats = dict()
            for per in self.corrupt_percent:
                corrupted_image = mismatched_image_corruption(image, rig_hmap, rbit_hmap, corrupt_percentage=per)
                ig_hmp, bit_hmp = self.make_heatmap(image, saliency="ig"), self.make_heatmap(image, saliency="bit")
                
                corrupted_stats[f'per_{str(int(per * 100))}'] = {'heatmaps': [ig_hmp.tolist(), bit_hmp.tolist()], 
                                'image': corrupted_image.tolist(), 'logit': self.model(image, register_hook=False).tolist()} # key = per_10 for 10%


            self.saliency_information[batch_idx] = {
                'default': {'heatmap': [rig_hmap.tolist(), rbit_hmap.tolist()], 'target': target.tolist(), 
                            'logit': self.model(image, register_hook=False).tolist(), 'image': image},
                'corrupted': corrupted_stats,
            }

        self.saliency_writer(f'/content/benchmark/saliency_records/{saliency}_saliency.pt')
        print(f"Finished generating saliency for {self.data_length} images")



if __name__ == "__main__":
    os.makedirs("saliency_records", exist_ok=True)

    slgen = SaliencyGenerator(data_lim=5)
    slgen("bi_t")