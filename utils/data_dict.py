# Get datadict from csv.
def get_data_dict(MODE="test", SRC="/kaggle/input/image-matching-challenge-2024", DEBUG=False, DEBUG_SCENE=None):
    if MODE == "train":
        sample_path = f"{SRC}/train/train_labels.csv"
    else:
        sample_path = f"{SRC}/sample_submission.csv"

    data_dict = {}
    with open(sample_path, "r") as f:
        for i, l in enumerate(f):
            # Skip header.
            if l and i > 0:
                if MODE == "train":
                    dataset, scene, image, _, _ = l.strip().split(",")
                else:
                    image, dataset, scene, _, _ = l.strip().split(",")
                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(image)
                
    all_scenes = []
    scene_len = []
    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")
            if DEBUG and (scene not in DEBUG_SCENE):
                continue
            all_scenes.append((dataset, scene))
            scene_len.append(len(data_dict[dataset][scene]))

    # sort all scenes by length, lowest first
    all_scenes = [x for _, x in sorted(zip(scene_len, all_scenes), reverse=True)]

    # Print reconst order
    print("Reconstruction order: ")
    for scene in all_scenes:
        print(f" --{scene[0]} / {scene[1]}")

    return data_dict, all_scenes