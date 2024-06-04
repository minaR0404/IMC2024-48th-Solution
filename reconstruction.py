import numpy as np
from time import time
import os
import gc
from copy import deepcopy
from IPython.display import clear_output
import pycolmap

from main import data_dict
from config import SRC
from colmap_db.database import COLMAPDatabase


##function of reconstruction
def reconstruct_from_db(dataset, scene):
    scene_result = {}
    reconst_start = time()

    img_dir = SRC + '/' + '/'.join(data_dict[dataset][scene][0].split("/")[:-1])  #f"{SRC}/{MODE}/{dataset}/{scene}/images"
    if not os.path.exists(img_dir):
        print("Image dir does not exist:", img_dir)
        return

    feature_dir = f"featureout/{dataset}_{scene}"
    database_path = f"{feature_dir}/colmap.db"
    db = COLMAPDatabase.connect(database_path)
    output_path = f"{feature_dir}/colmap_rec"
    t = time()
    gc.collect()

#    # Skip match_exhaustive
#     pycolmap.match_exhaustive(database_path, match_options)

    t = time() - t
    print(f"RANSAC in  {t:.4f} sec")
    t = time()
    
    # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
    mapper_options = pycolmap.IncrementalMapperOptions()
    
    triangular_options = pycolmap.IncrementalTriangulatorOptions()
    #triangular_options.re_max_trials = 3
    #triangular_options.re_min_ratio = 0.8
    
    incremental_options = pycolmap.IncrementalPipelineOptions()
    #incremental_options.ba_local_max_refinement_change = 0.05
    incremental_options.min_model_size = 3
    incremental_options.mapper = mapper_options
    incremental_options.triangulation = triangular_options

    for attribute_name in dir(mapper_options):
        if not attribute_name.startswith("__"):
            attribute_value = getattr(mapper_options, attribute_name)
            print(f"{attribute_name}: {attribute_value}")
    os.makedirs(output_path, exist_ok=True)
    
    ##multiple maps
    num_regs = len(data_dict[dataset][scene])
    multiple = 3 if num_regs < 200 else 1
    imgs_registered = 0
    best_3d = 0
    best_idx = None
    best_maps = None
    
    for mul in range(multiple):
        print(f"Start Reconstruction {mul}-Mapping")
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=img_dir,
            output_path=output_path,
            options=incremental_options,
        )
        print(maps)
        clear_output(wait=False)
        #t = time() - t
        #print(f"Reconstruction done in  {t:.4f} sec")
        print("Looking for the best reconstruction")
        if isinstance(maps, dict):
            for idx1, rec in maps.items():
                print(idx1, rec.summary())
                num_3d = int(rec.summary().split('\n\t')[3].split('=')[-1])
                if len(rec.images) > imgs_registered:
                    imgs_registered = len(rec.images)
                    best_3d = num_3d
                    best_idx = idx1
                    best_maps = maps
                elif (len(rec.images) == imgs_registered) and (num_3d > best_3d):
                    best_3d = num_3d
                    best_idx = idx1
                    best_maps = maps
    
    if best_idx is not None:
        print(best_maps[best_idx].summary())
        for k, im in best_maps[best_idx].images.items():
            key1 = f"test/{scene}/images/{im.name}"  #f"{dataset}/{scene}/images/{im.name}"
            scene_result[key1] = {}
            scene_result[key1]["R"] = deepcopy(im.cam_from_world.rotation.matrix())
            scene_result[key1]["t"] = deepcopy(np.array(im.cam_from_world.translation))

    gc.collect()
    reconst_end = time()
    reconst_time = reconst_end - reconst_start
    return scene_result, reconst_time