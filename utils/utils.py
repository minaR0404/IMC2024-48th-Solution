import numpy as np
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import torch


def get_unique_idxs(A, dim=0):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(
        A, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
    first_indices = ind_sorted[cum_sum]
    return first_indices



def get_unique_matches(f_match_kpts):
    kpts = defaultdict(list)
    match_indexes = defaultdict(dict)
    total_kpts = defaultdict(int)
    for key1 in f_match_kpts:
        for key2 in f_match_kpts[key1]:
            matches = f_match_kpts[key1][key2]
            kpts[key1].append(matches[:, :2])
            kpts[key2].append(matches[:, 2:])
            current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
            current_match[:, 0] += total_kpts[key1]
            current_match[:, 1] += total_kpts[key2]
            total_kpts[key1] += len(matches)
            total_kpts[key2] += len(matches)
            match_indexes[key1][key2] = current_match

    for key in kpts:
        kpts[key] = np.round(np.concatenate(kpts[key], axis=0))

    unique_kpts = {}
    unique_match_idxs = {}
    out_match = defaultdict(dict)

    for key in kpts.keys():
        uniq_kps, uniq_reverse_idxs = torch.unique(
            torch.from_numpy(kpts[key]), dim=0, return_inverse=True
        )
        unique_match_idxs[key] = uniq_reverse_idxs
        unique_kpts[key] = uniq_kps.numpy()

    for key1 in match_indexes:
        for key2 in match_indexes[key1]:
            m2 = deepcopy(match_indexes[key1][key2])
            m2[:, 0] = unique_match_idxs[key1][m2[:, 0]]
            m2[:, 1] = unique_match_idxs[key2][m2[:, 1]]
            mkpts = np.concatenate(
                [
                    unique_kpts[key1][m2[:, 0]],
                    unique_kpts[key2][m2[:, 1]],
                ],
                axis=1,
            )
            unique_idxs_current = get_unique_idxs(torch.from_numpy(mkpts), dim=0)
            m2_semiclean = m2[unique_idxs_current]
            unique_idxs_current1 = get_unique_idxs(m2_semiclean[:, 0], dim=0)
            m2_semiclean = m2_semiclean[unique_idxs_current1]
            unique_idxs_current2 = get_unique_idxs(m2_semiclean[:, 1], dim=0)
            m2_semiclean2 = m2_semiclean[unique_idxs_current2]
            out_match[key1][key2] = m2_semiclean2.numpy()
    return unique_kpts, out_match



# Util to check if the pairs are identital
def compare_pairs(pairs1, pairs2):
    pair1_dict = dict()
    pair2_dict = dict()
    for idx1 in tqdm(range(len(pairs1) - 1)):
        for pair in pairs1[idx1]:
            pair1_dict[(idx1, pair[0])] = 0
    for idx2 in tqdm(range(len(pairs2) - 1)):
        for pair in pairs2[idx2]:
            pair2_dict[(idx2, pair[0])] = 0

    for key in tqdm(pair1_dict):
        if key not in pair2_dict:
            print(f"Key{key} not in pair2_dict")
    for key in tqdm(pair2_dict):
        if key not in pair1_dict:
            print(f"Key{key} not in pair1_dict")
