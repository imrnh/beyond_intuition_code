"""
    Hypothesis: Beyond Intuition method misses some important tokens which shoundn't be missed. 
"""

import os
import numpy as np
from scipy import stats
from hypothesis_eval.corrupt_data import get_corrupted_image


"""
    Find tokens that were important on IG but said to be unimportant in BI_T or BI_H

    @param ig_hmap: numpy array of shape 196,
    @param bi_hmap: numpy array of shape 196


"""
def find_mismatched_tokens(ig_hmap, bi_hmap, tolerate_limit=0.2):
    nig = stats.zscore(ig_hmap)  # Starndarizing
    nbi = stats.zscore(bi_hmap)  # Standarizing

    match_array = np.zeros_like(ig_hmap) * True

    cond1 = lambda x, y: abs(x-y) < tolerate_limit
    stt_diff = lambda arr, idx : arr[idx] > arr[idx-1]  # Find the relation between current value and previous one.

    for idx in range(len(nig)):
        x = nig[idx]
        y = nbi[idx]

        if cond1(x, y) and (stt_diff(nig, idx) == stt_diff(nbi, idx)):
            match_array[idx] = True
    
    return match_array



"""
    Select few top mis-matched tokens. 15% i.e ~29 tokens for now. 

    Non-common important tokens will be corrupted via corruption function.
    
    Clever trick:
        To corrupt only non-common pixels and not the important common pixels, 
        in mismatched array, common pixels will be set to minimum of the bi_saliency.
    
"""

def mismatched_image_corruption(dataloader, ig_state_dict, bi_state_dict):
    corrupted_dataset = []
    for index, (image, label) in enumerate(dataloader):
        ig_hmap = ig_state_dict[index]['heatmap']
        bi_hmap = bi_state_dict[index]['heatmap']

        mismatch = find_mismatched_tokens(ig_hmap, bi_hmap)
        modified_hmap =[(hv if mismatch[idx] != 1 else bi_hmap.min()) for idx, hv in enumerate(ig_hmap)]

        corrupted_image = get_corrupted_image(image, modified_hmap, corrupt_percentage=0.15)

        corrupted_dataset.append((corrupted_image, label))

    print("Corrupted All the data")