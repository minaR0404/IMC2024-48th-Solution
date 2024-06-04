# Modified from https://github.com/cvlab-epfl/disk/blob/37f1f7e971cea3055bb5ccfc4cf28bfd643fa339/colmap/h5_to_db.py

#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os, argparse, h5py, warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ExifTags

from database import image_ids_to_pair_id


def get_focal(image_path, kpts, matches, err_on_default=False):
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    
    #
    # Modified to add exif_ifd to exif dict
    #
    exif_ifd = exif.get_ifd(0x8769)
    exif.update(exif_ifd)

    focal = None
    is_from_exif = False
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLength": #"FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                is_from_exif = True
                #raise RuntimeError("Exist exif information")
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35.0 * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        ##calculate best focal_prior
        key1 = image_path.split('/')[-1]
        best_matches = 0
        for key2 in matches[key1]:
            best_matches = max(len(matches[key1][key2]), best_matches)
        ratio_match = best_matches / len(kpts[key1])
        FOCAL_PRIOR = 0.7 + ratio_match * 2.0
            
        # failed to find it in exif, use prior
        #FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size
    
    #
    # Modified to return a bool indicating if the focal length is from exif 
    #
    return focal, is_from_exif


def create_camera(db, image_path, camera_model, kpts, matches):
    image = Image.open(image_path)
    width, height = image.size

    focal, is_from_exif = get_focal(image_path, kpts, matches)

    if camera_model == "simple-pinhole":
        model = 0  # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == "pinhole":
        model = 1  # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2  # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "opencv":
        model = 4  # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])
        
        
    #
    # Modified to set prior_focal_length if the focal length is from exif
    #
    return db.add_camera(
        model, width, height, param_arr, prior_focal_length=is_from_exif
    )


def add_kpts_matches(db, img_dir, kpts, matches, fms = None):
    fname_to_id = {}

    # Add keypoints
    for filename in tqdm(kpts):
        path = os.path.join(img_dir, filename)
        camera_model = "simple-radial"
        camera_id = create_camera(db, path, camera_model, kpts, matches)
        image_id = db.add_image(filename, camera_id)
        fname_to_id[filename] = image_id
        db.add_keypoints(image_id, kpts[filename])

        n_keys = len(matches)
        n_total = (n_keys * (n_keys - 1)) // 2
    # Add matches
    added = set()
    with tqdm(total=n_total) as pbar:
        for key1 in matches:
            for key2 in matches[key1]:
                id_1 = fname_to_id[key1]
                id_2 = fname_to_id[key2]
                pair_id = image_ids_to_pair_id(id_1, id_2)
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                    continue
                db.add_matches(id_1, id_2, matches[key1][key2])
                added.add(pair_id)
                pbar.update(1)
                if fms is not None:
                    db.add_two_view_geometry(id_1, id_2, matches[key1][key2], fms[key1][key2], np.eye(3), np.eye(3))

    db.commit()
