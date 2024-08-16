import torch
from torchvision import transforms

def image_vizformat(image):
    inv_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )

    img = inv_normalize(image[0])
    img = torch.permute(img, (1, 2, 0))
    img = img.detach().cpu().numpy()
    return img
