
import torch
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
from DeDoDe_main.DeDoDe.utils import dual_softmax_matcher, to_pixel_coords, to_normalized_coords
from utils.utils import get_unique_idxs
from config import device


class DualSoftMaxMatcherV2(nn.Module):        
    @torch.inference_mode()
    def match(self, keypoints_A, descriptions_A, 
              keypoints_B, descriptions_B, P_A = None, P_B = None, 
              normalize = False, inv_temp = 1, threshold = 0.0):
        if isinstance(descriptions_A, list):
            matches = [self.match(k_A[None], d_A[None], k_B[None], d_B[None], normalize = normalize,
                               inv_temp = inv_temp, threshold = threshold) 
                    for k_A,d_A,k_B,d_B in
                    zip(keypoints_A, descriptions_A, keypoints_B, descriptions_B)]
            matches_A = torch.cat([m[0] for m in matches])
            matches_B = torch.cat([m[1] for m in matches])
            inds = torch.cat([m[2] + b for b, m in enumerate(matches)])
            return matches_A, matches_B, inds
        
        P = dual_softmax_matcher(descriptions_A, descriptions_B, 
                                 normalize = normalize, inv_temperature=inv_temp,
                                 )
        inds = torch.nonzero((P == P.max(dim=-1, keepdim = True).values) 
                        * (P == P.max(dim=-2, keepdim = True).values) * (P > threshold))

        batch_inds = inds[:,0]
        matches_A = keypoints_A[batch_inds, inds[:,1]]
        matches_B = keypoints_B[batch_inds, inds[:,2]]
        return matches_A, matches_B, inds[:,1:]

    def to_pixel_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_pixel_coords(x_A, H_A, W_A), to_pixel_coords(x_B, H_B, W_B)
    
    def to_normalized_coords(self, x_A, x_B, H_A, W_A, H_B, W_B):
        return to_normalized_coords(x_A, H_A, W_A), to_normalized_coords(x_B, H_B, W_B)



class DeDoDeMatcherV2:
    def __init__(self, matcher, min_matches=15, device="cuda"):
        self.min_matches = min_matches
        self.matcher = matcher

    def match(self, img_fnames, f_kpts, f_conf, f_descs):
        index_pairs = dict()
        num_imgs = len(img_fnames)
        print("Matching to get index pairs")
        pair_count = 0
        f_matches = defaultdict(dict)
        for idx1 in tqdm(range(num_imgs - 1)):
            index_pairs[idx1] = []
            for idx2 in range(idx1 + 1, num_imgs):
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split("/")[-1], fname2.split("/")[-1]
                keypoints_A = torch.from_numpy(f_kpts[key1]).unsqueeze(0).to(device)
                keypoints_B = torch.from_numpy(f_kpts[key2]).unsqueeze(0).to(device)
                P_A = f_conf[key1]
                P_B = f_conf[key2]
                description_A = f_descs[key1]
                description_B = f_descs[key2]
                
                matches_A, matches_B, idxs = self.matcher.match(keypoints_A, description_A,
                    keypoints_B, description_B,
                    P_A = P_A, P_B = P_B,
                    normalize = True, inv_temp=20, threshold = 0.5)#Increasing threshold -> fewer matches, fewer outliers

                if len(idxs) > self.min_matches:
                    first_indices = get_unique_idxs(idxs[:, 1])
                    idxs = idxs[first_indices]
                    #dists = dists[first_indices]
                    n_matches = len(idxs)
                    if n_matches >= self.min_matches:
                        pair_count += 1
                        f_matches[key1][key2] = (
                            idxs.detach().cpu().numpy().reshape(-1, 2)
                        )

        print(f" Get {pair_count} from {int(num_imgs * (num_imgs-1)/2)} possible pairs")
        torch.cuda.empty_cache()
        gc.collect()
        return f_kpts, f_matches