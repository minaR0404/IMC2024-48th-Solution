
import cv2
from tqdm import tqdm
from collections import defaultdict
from config import FM_PARAMS


##get fundamental matrices
def get_fms(kpts, matches):
    prev_len = dict()
    fms = defaultdict(dict)
    print("Get Fundamental Matrix")
    for key1 in tqdm(matches):
        for key2 in matches[key1]:
            match = matches[key1][key2]
            mkpts1 = kpts[key1][match[:, 0]]
            mkpts2 = kpts[key2][match[:, 1]]
            Fm, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, FM_PARAMS["ransacReprojThreshold"], FM_PARAMS["confidence"], FM_PARAMS["maxIters"])
            #keep inliers matches
            #print how many matches are inliers
            # print(f"key1: {key1}, key2: {key2}, inliers: {len(new_match)}/{len(match)}")
            if FM_PARAMS["removeOutliers"] == True:
                new_match = match[inliers.ravel() == 1]
                matches[key1][key2] = new_match
            fms[key1][key2] = Fm
    # print(Fm.shape)
    return kpts, matches, fms