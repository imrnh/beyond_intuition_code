"""
For any given model & interpretation method, generate the saliency map. 
Saliency map data is stored in a pytorch file inside benchmark folder.
"""
import torch
import saliency
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils.model_loaders import vit_base_patch16_224_dino, vit_base_patch16_224
from interpretation_methods import beyond_intuition_tokenwise, beyond_intuition_headwise, integrated_gradient
from utils.imagenet_seg_loader import ImagenetSegLoader
from utils.image_denorm import image_vizformat
from tqdm import tqdm
from PIL import Image


class SaliencyGenerator(nn.Module):
    def __init__(self, verbose=False, data_lim=2000) -> None:
        self.data_path = "lib\dataset\gtsegs_ijcv.mat"
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
        self.label_transform = transforms.Compose([transforms.Resize((224, 224), Image.NEAREST), ])

        self.dataset = ImagenetSegLoader(self.data_path, self.data_length, transform=self.image_transform, target_transform=self.label_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False)
        self.dataloader = tqdm(self.dataloader)  # Would help tracking loop iteration along with setting some verbose text.

        # Model loading
        self.model = vit_base_patch16_224_dino(pretrained=True).to(self.device)
        self.model.eval()  # Set to eval mode. Helps tarcking gradient.

        self.saliency_writer = lambda output_path: torch.save(self.saliency_information, output_path)

        print("""Please select a valid saliency method while calling the forward pass. Valid saliency methods are: 
                ig -> Integrated Gradient \n
                bi_t -> Tokenwise Beyond Intuition
                bi_h -> Headwise Beyond Intuition
              """)
        


    def forward(self, saliency):
        for batch_idx, (image, labels) in enumerate(self.dataloader):
            if saliency == "ig":
                heatmap = integrated_gradient(self.model, image, self.device)

            elif saliency == "bi_t":
                heatmap, _ = beyond_intuition_tokenwise(self.model, image, self.device, dino=True, start_layer=0)
                
            elif saliency == "bi_h":
                heatmap = beyond_intuition_headwise(self.model, image, self.device, dino=True, start_layer=self.args.start_layer)
            
            heatmap = heatmap.reshape((14, 14)).detach().cpu()

            self.saliency_information[batch_idx] = {
                'image': image_vizformat(image.tolist()),
                'heatmap': heatmap.tolist(),
                'label': labels.tolist()
            }

            self.saliency_writer('benchmark/saliency.pt')

        print(f"Finished generating saliency for {self.data_length} images")



if __name__ == "__main__":
    slgen = SaliencyGenerator(data_lim=5)
    slgen("bi_t")