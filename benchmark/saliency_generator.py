"""
For any given model & interpretation method, generate the saliency map. 
Saliency map data is stored in a pytorch file inside benchmark folder.
"""

from utils.model_loaders import vit_base_patch16_224_dino, vit_base_patch16_224