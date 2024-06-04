
import os
from time import time
import gc
import torch

from main import data_dict
from config import SRC, DEBUG, MODEL_DICT, MATCHES_CAP, MATCH_FILTER_RATIO
from setup import keynet_detector, gftt_detector, DoG_detector, harris_detector, laf_matcher, dedode_detector, dedode_matcher
from utils.matches import merge_kpts_matches, select_matches
from utils.fms import get_fms

from colmap_db.database import COLMAPDatabase
from colmap_db.h5_to_db import add_kpts_matches


##generate db
def generate_scene_db(dataset, scene):
    feature_det_start = time()
    # Process a scene and write matches and keypoints to the database
    img_dir = SRC + '/' + '/'.join(data_dict[dataset][scene][0].split("/")[:-1])  #f"{SRC}/{MODE}/{dataset}/{scene}/images"
    if not os.path.exists(img_dir):
        print("Image dir does not exist:", img_dir)
        return

    img_fnames = [f"{SRC}/{x}" for x in data_dict[dataset][scene]] #[f"{SRC}/{MODE}/{x}" for x in data_dict[dataset][scene]]
    print(f"Got {len(img_fnames)} images")

    matches = dict()
    kpts = dict()

    if MODEL_DICT["Keynet"]["enable"]:
        f_lafs, f_kpts, f_descs, f_raw_size = keynet_detector.detect_features(
            img_fnames
        )
        keynet_pairs, keynet_kpts, keynet_matches, keynet_rois = laf_matcher.match(
            img_fnames, f_lafs, f_kpts, f_descs, f_raw_size
        )
        if not MODEL_DICT["Keynet"]["pair_only"]:
            kpts, matches = merge_kpts_matches(kpts, matches, keynet_kpts, keynet_matches, MATCHES_CAP)

    if MODEL_DICT["GFTT"]["enable"]:
        gftt_lafs, gftt_kpts, gftt_descs, gftt_raw_size = gftt_detector.detect_features(
            img_fnames
        )
        index_pairs, gftt_kpts, gftt_matches, gftt_rois = laf_matcher.match(
            img_fnames, gftt_lafs, gftt_kpts, gftt_descs, gftt_raw_size
        )
        kpts, matches = merge_kpts_matches(kpts, matches, gftt_kpts, gftt_matches, MATCHES_CAP)

    if MODEL_DICT["DoG"]["enable"]:
        DoG_lafs, DoG_kpts, DoG_descs, DoG_raw_size = DoG_detector.detect_features(
            img_fnames
        )
        index_pairs, DoG_kpts, DoG_matches, DoG_rois = laf_matcher.match(
            img_fnames, DoG_lafs, DoG_kpts, DoG_descs, DoG_raw_size
        )
        kpts, matches = merge_kpts_matches(kpts, matches, DoG_kpts, DoG_matches, MATCHES_CAP)
        
    if MODEL_DICT["Harris"]["enable"]:
        harris_lafs, harris_kpts, harris_descs, harris_raw_size = harris_detector.detect_features(
            img_fnames
        )
        harris_pairs, harris_kpts, harris_matches, harris_rois = laf_matcher.match(
            img_fnames, harris_lafs, harris_kpts, harris_descs, harris_raw_size
        )
        kpts, matches = merge_kpts_matches(kpts, matches, harris_kpts, harris_matches, MATCHES_CAP)
        # compare_pairs(keynet_pairs, harris_pairs)
        
    if MODEL_DICT["DeDoDe"]["enable"]:
        dedode_kpts, dedode_conf, dedode_descs = dedode_detector.detect_features(
            img_fnames
        )
        dedode_kpts, dedode_matches = dedode_matcher.match(
            img_fnames, dedode_kpts, dedode_conf, dedode_descs
        )
        kpts, matches = merge_kpts_matches(kpts, matches, dedode_kpts, dedode_matches, MATCHES_CAP)

    ##LG+ALIKED
#     alike_kpts, alike_descs = detect_aliked(img_fnames)
#     alike_kpts, alike_matches = match_lightglue(img_fnames, alike_kpts, alike_descs)
#     kpts, matches = merge_kpts_matches(kpts, matches, alike_kpts, alike_matches, MATCHES_CAP)
    
    # Get fundamental matrices
    kpts, matches, fms = get_fms(kpts, matches)
    
    matches = select_matches(matches, MATCH_FILTER_RATIO)

    if DEBUG:
        import random
        random.seed(0)
        for i in range(5):
            print(matches.keys())
           
            key1 = random.choice(list(matches.keys()))
            key2 = random.choice(list(matches[key1].keys()))
            print(key1, key2)
            fname1, fname2 = os.path.join(img_dir, key1), os.path.join(img_dir, key2)

            print("Plot Combined matches")
            # plot_images_with_keypoints(
            #     fname1, fname2, kpts[key1], kpts[key2], matches[key1][key2]
            # )
    # Write to database
    feature_dir = f"featureout/{dataset}_{scene}"
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir, exist_ok=True)
    database_path = f"{feature_dir}/colmap.db"
    if os.path.isfile(database_path):
        os.remove(database_path)

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    print("Add kpts and matches to database")
    add_kpts_matches(db, img_dir, kpts, matches, fms)
    feature_det_end = time()
    matching_time = feature_det_end - feature_det_start
    torch.cuda.empty_cache()
    gc.collect()

    return matching_time