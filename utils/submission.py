import numpy as np


##submission utils
def arr_to_str(a):
    return ";".join([str(x) for x in a.reshape(-1)])


# Function to create a submission file.
def create_submission(out_results, data_dict, mode="test"):
    if mode == "train":
        file_name = "submission_train.csv"
    else:
        file_name = "submission.csv"

    with open(file_name, "w") as f:
        f.write("image_path,dataset,scene,rotation_matrix,translation_vector\n")
        for dataset in data_dict:
            if dataset in out_results:
                res = out_results[dataset]
            else:
                res = {}
            for scene in data_dict[dataset]:
                if scene in res:
                    scene_res = res[scene]
                else:
                    scene_res = {"R": {}, "t": {}}
                for image in data_dict[dataset][scene]:
                    if image in scene_res:
                        print(image)
                        R = scene_res[image]["R"].reshape(-1)
                        T = scene_res[image]["t"].reshape(-1)
                    else:
                        R = np.eye(3).reshape(-1)
                        T = np.zeros((3))
                    f.write(
                        f"{image},{dataset},{scene},{arr_to_str(R)},{arr_to_str(T)}\n"
                    )