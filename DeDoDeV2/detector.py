
from tqdm import tqdm
from PIL import Image
import gc
import torch

from DeDoDe_main.DeDoDe.utils import to_pixel_coords


class DeDoDeDetectorV2:
    def __init__(
        self,
        detector,
        descriptor,
        device=torch.device("cuda"),
        resize_long_edge_to=1024,
        min_matches=15,
    ):
        print("Init DeDoDeDetector")
        self.detector = detector
        self.descriptor = descriptor
        self.device = device
        self.resize_long_edge_to = resize_long_edge_to
        print("Longer edge will be resized to", self.resize_long_edge_to)

    def detect_features(self, img_fnames):
        f_descs = dict()
        f_kpts = dict()
        f_conf = dict()
        # Get features
        print("Detecting DeDoDe features")
        for img_path in tqdm(img_fnames):
            img_fname = img_path.split("/")[-1]
            key = img_fname
            with torch.inference_mode():
                img = Image.open(img_path)
                W_A, H_A = img.size
                
                detections_A = self.detector.detect_from_path(img_path, num_keypoints = 10_000)
                keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
                description_A = self.descriptor.describe_keypoints_from_path(img_path, keypoints_A)["descriptions"]
                
                keypoints_A = to_pixel_coords(keypoints_A, H_A, W_A)

                f_kpts[key] = keypoints_A.squeeze().detach().cpu().numpy()
                f_conf[key] = P_A.detach()
                f_descs[key] = description_A.detach()
        gc.collect()
        torch.cuda.empty_cache()
        return f_kpts, f_conf, f_descs