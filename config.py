import torch

# Mode can only be train or test. This will be used to find the image directory.
# Use "test" for submission 
MODE = "train"
MODE = "test"

# Option to change path for local testing
is_local = True
is_local = False

if is_local:
    NUM_CORES = 2
    SRC = "./kaggle/input/image-matching-challenge-2023"
    MODEL_DIR = "./kaggle/input/kornia-local-feature-weights/"
    DISK_PATH = "./loftr_disk.ckpt"
    HARDNET_PT = "./kaggle/input/kornia-local-feature-weights/hardnet8v2.pt"
else:
    NUM_CORES = 2
    SRC = "/kaggle/input/image-matching-challenge-2024"
    SAVE = "/kaggle/temp/images-out"
    MODEL_DIR = "/kaggle/input/kornia-local-feature-weights/"
    DISK_PATH = "/kaggle/input/disk/pytorch/depth-supervision/1/loftr_outdoor.ckpt"
    HARDNET_PT = "/kaggle/input/hardnet8v2/hardnet8v2.pt"

LOG_MESSAGE = "Final submission"
MATCHES_CAP = None

DEBUG = True
DEBUG = False

# DEBUG_SCENE = ["cyprus", "kyiv-puppet-theater"]
# DEBUG_SCENE = ["cyprus"]
# DEBUG_SCENE = ["kyiv-puppet-theater"]
# DEBUG_SCENE = ["kyiv-puppet-theater", "cyprus", "wall", "chairs"]
DEBUG_SCENE = ["chairs"]
# DEBUG_SCENE = ["wall"]

# Longer edge limit of the input image
hardnet_res = 1024

MODEL_DICT = {
    "Keynet": {"enable": False, "resize_long_edge_to": hardnet_res, "pair_only": False},
    "GFTT": {"enable": False, "resize_long_edge_to": hardnet_res},
    "DoG": {"enable": False, "resize_long_edge_to": hardnet_res},
    "Harris": {"enable": False, "resize_long_edge_to": hardnet_res},
    "DeDoDe": {"enable": True},
}

# Find fundamental matrix parameters
FM_PARAMS = {"ransacReprojThreshold": 5, "confidence": 0.9999, "maxIters": 50000, "removeOutliers": True}

# Remove a "match" if the number of matches is lower than MATCH_FILTER_RATIO*max_num_matches
# e.g. img1 and img2 have max 10000 matches with some other images, img2 and img1 only have 99 matches. The matches btw img1 and img2 won't be selected.
MATCH_FILTER_RATIO = 0.01

# for logging
LOG_DICT = dict()
LOG_DICT["mode"] = MODE
LOG_DICT["log_message"] = LOG_MESSAGE
LOG_DICT["matches_cap"] = MATCHES_CAP
LOG_DICT["debug"] = DEBUG
LOG_DICT["debug_scene"] = DEBUG_SCENE

if MODE == "test":
    DEBUG = False
device = torch.device("cuda")
print(torch.cuda.is_available())

from pathlib import Path
SKIP = len([p for p in Path("/kaggle/input/image-matching-challenge-2024/test/").iterdir() if p.is_dir()]) == 1