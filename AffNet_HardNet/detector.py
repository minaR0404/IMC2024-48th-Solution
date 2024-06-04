
from tqdm import tqdm
import gc
import torch
import kornia as K
import kornia.feature as KF

from config import device
from utils.image import load_torch_image, resize_torch_image


##scene feature detector
class AffNetHardNetDetector:
    def __init__(
        self,
        model,
        device=torch.device("cuda"),
        resize_long_edge_to=600,
        matcher="adalam",
        min_matches=15,
        rgb_input = False
    ):
        self.rgb_input = rgb_input
        print("Init AffNetHardNetDetector")
        self.model = model
        self.device = device
        self.resize_long_edge_to = resize_long_edge_to
        print("Longer edge will be resized to", self.resize_long_edge_to)

    def detect_features(self, img_fnames):
        f_lafs = dict()
        f_descs = dict()
        f_kpts = dict()
        f_raw_size = dict()
        f_matches = dict()
        # Get features
        print("Detecting AffNetHardNet features")
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split("/")[-1]
            key = img_fname
            f_matches[key] = dict()
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                raw_size = torch.tensor(timg.shape[2:])
                timg_resized, h_scale, w_scale = resize_torch_image(
                    timg, self.resize_long_edge_to, disable_enlarge=True
                )   
                if self.rgb_input:
                    lafs, resps, descs = self.model(timg_resized)
                else:
                    lafs, resps, descs = self.model(K.color.rgb_to_grayscale(timg_resized))
                
                # Recover scale?
                lafs[:, :, 0, :] *= 1 / w_scale
                lafs[:, :, 1, :] *= 1 / h_scale
                desc_dim = descs.shape[-1]
                # Move keypoints to cpu for later colmap operations
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach()
                f_lafs[key] = lafs.detach()
                f_kpts[key] = kpts
                f_descs[key] = descs
                f_raw_size[key] = raw_size
        gc.collect()
        torch.cuda.empty_cache()
        return f_lafs, f_kpts, f_descs, f_raw_size