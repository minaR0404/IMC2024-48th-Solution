# General utilities
import os
from tqdm import tqdm
from time import time
from collections import defaultdict
from copy import deepcopy
import concurrent.futures

# CV/ML
import cv2
import torch
import torch.nn.functional as F
import kornia as K
import kornia.feature as KF
from PIL import Image

# 3D reconstruction
import pycolmap

print("Kornia version", K.__version__)
print("Pycolmap version", pycolmap.__version__)


# others
from config import MODE, SRC, DEBUG, DEBUG_SCENE, SKIP, NUM_CORES

from generate_db import generate_scene_db
from reconstruction import reconstruct_from_db

from utils.data_dict import get_data_dict
from utils.submission import create_submission



# Get datadict from csv.
data_dict, all_scenes = get_data_dict(MODE, SRC, DEBUG, DEBUG_SCENE)


##main loop
# Main loop to add kpts and matches, and reconstruction.
datasets = []
time_dict = dict()

for dataset in data_dict:
    datasets.append(dataset)

if DEBUG:
    matching_start = time()
    for dataset, scene in all_scenes:
        print(dataset, scene)
        time_dict["matching-" + scene] = generate_scene_db(dataset, scene)

    matching_end = time()
    time_dict["matching-TOTAL"] = matching_end - matching_start
elif SKIP:
    out_results = defaultdict(dict)
else:
    # Run db generation and reconstuction with multiprocessing if not DEBUG
    out_results = defaultdict(dict)
    total_start = time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CORES) as executors:
        futures = defaultdict(dict)

        for dataset, scene in all_scenes:
            print(dataset, scene)
            time_dict["matching-" + scene] = generate_scene_db(dataset, scene)
            futures[dataset][scene] = executors.submit(reconstruct_from_db, dataset, scene)

        for dataset, scene in all_scenes:
            result = futures[dataset][scene].result()
            if result is not None:
                out_results[dataset][scene], time_dict["reconst-" + scene] = result
    total_end = time()
    time_dict["TOTAL"] = total_end - total_start


create_submission(out_results, data_dict, MODE)
