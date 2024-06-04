
import numpy as np
from tqdm import tqdm
import gc
from collections import defaultdict
import torch
import kornia.feature as KF

from utils.utils import get_unique_idxs


##laf matcher
class LafMatcher:
    def __init__(self, min_matches=15, device="cuda", matcher="adalam"):
        self.adalam_config = KF.adalam.get_adalam_default_config()
        self.adalam_config["force_seed_mnn"] = True
        self.adalam_config["search_expansion"] = 16
        self.adalam_config["ransac_iters"] = 256
        self.adalam_config["device"] = device
        # self.adalam_config["orientation_difference_threshold"] = None
        # self.adalam_config['scale_rate_threshold'] = None
        self.min_matches = min_matches
        self.matcher = matcher

    def match(self, img_fnames, f_lafs, f_kpts, f_descs, f_raw_size, get_roi = False):
        index_pairs = dict()
        num_imgs = len(img_fnames)
        print("Matching to get index pairs")
        pair_count = 0
        f_matches = defaultdict(dict)
        f_rois = defaultdict(dict)
        for idx1 in tqdm(range(num_imgs - 1)):
            index_pairs[idx1] = []
            for idx2 in range(idx1 + 1, num_imgs):
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split("/")[-1], fname2.split("/")[-1]
                lafs1 = f_lafs[key1]
                lafs2 = f_lafs[key2]
                desc1 = f_descs[key1]
                desc2 = f_descs[key2]
                if self.matcher == "adalam":
                    hw1, hw2 = f_raw_size[key1], f_raw_size[key2]

                    dists, idxs = KF.match_adalam(
                        desc1,
                        desc2,
                        lafs1,
                        lafs2,  # Adalam takes into account also geometric information
                        hw1=hw1,
                        hw2=hw2,
                        config=self.adalam_config,
                    )  # Adalam also benefits from knowing image size
                else:
                    dists, idxs = KF.match_smnn(desc1, desc2, 0.98)

                if dists.mean().cpu().numpy() < 0.5:
                    first_indices = get_unique_idxs(idxs[:, 1])
                    idxs = idxs[first_indices]
                    dists = dists[first_indices]
                    n_matches = len(idxs)
                    if n_matches >= self.min_matches:
                        pair_count += 1
                        index_pairs[idx1].append(
                            [idx2, dists.mean().cpu().numpy().item(), n_matches]
                        )
                        f_matches[key1][key2] = (
                            idxs.detach().cpu().numpy().reshape(-1, 2)
                        )

                        # Compute ROI
                        if get_roi:
                            mkpts1 = f_kpts[key1][idxs.cpu().numpy()[:, 0]]
                            mkpts2 = f_kpts[key2][idxs.cpu().numpy()[:, 1]]
                            roi_min_w_1, roi_max_w_1 = np.percentile(mkpts1[:, 0], [5, 95])
                            roi_min_h_1, roi_max_h_1 = np.percentile(mkpts1[:, 1], [5, 95])
                            roi_area_1 = (roi_max_w_1 - roi_min_w_1) * (
                                roi_max_h_1 - roi_min_h_1
                            )
                            roi1 = {
                                "roi_min_w": roi_min_w_1,
                                "roi_min_h": roi_min_h_1,
                                "roi_max_w": roi_max_w_1,
                                "roi_max_h": roi_max_h_1,
                                "area": roi_area_1,
                            }
                            roi_min_w_2, roi_max_w_2 = np.percentile(mkpts2[:, 0], [5, 95])
                            roi_min_h_2, roi_max_h_2 = np.percentile(mkpts2[:, 1], [5, 95])
                            roi_area_2 = (roi_max_w_2 - roi_min_w_2) * (
                                roi_max_h_2 - roi_min_h_2
                            )
                            roi2 = {
                                "roi_min_w": roi_min_w_2,
                                "roi_min_h": roi_min_h_2,
                                "roi_max_w": roi_max_w_2,
                                "roi_max_h": roi_max_h_2,
                                "area": roi_area_2,
                            }
                            f_rois[key1][key2] = [roi1, roi2]

        print(f" Get {pair_count} from {int(num_imgs * (num_imgs-1)/2)} possible pairs")
        torch.cuda.empty_cache()
        gc.collect()
        return index_pairs, f_kpts, f_matches, f_rois