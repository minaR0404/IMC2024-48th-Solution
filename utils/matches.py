
import numpy as np
from collections import defaultdict
from config import DEBUG


##matcher and pair operation
def merge_kpts_matches(kpts, matches, new_kpts, new_matches, cap = None):
    # merge kpts
    prev_len = dict()
    for new_key in new_kpts:
        if new_key in kpts:
            old_len = len(kpts[new_key])
            kpts[new_key] = np.concatenate([kpts[new_key], new_kpts[new_key]], axis=0)
        else:
            old_len = 0
            kpts[new_key] = new_kpts[new_key]
        prev_len[new_key] = old_len

    for new_key1 in new_matches:
        for new_key2 in new_matches[new_key1]:
            old_len1 = prev_len[new_key1]
            old_len2 = prev_len[new_key2]
            new_match = new_matches[new_key1][new_key2] + [old_len1, old_len2]
            if cap is not None and len(new_match) > cap:
                keep = np.random.choice(len(new_match), cap, replace=False)
                new_match = new_match[keep, :]
            if new_key1 in matches and new_key2 in matches[new_key1]:

                matches[new_key1][new_key2] = np.concatenate(
                    [
                        matches[new_key1][new_key2],
                        new_match,
                    ],
                    axis=0,
                )
            else:
                if new_key1 not in matches:
                    matches[new_key1] = dict()
                matches[new_key1][new_key2] = new_match
    return kpts, matches


def keep_matches(matches, max_num=None):
    if max_num is None:
        return matches
    if len(matches) > max_num:
        # radnomly select max_num matches
        matches = np.random.choice(matches, max_num, replace=False)
    return matches


def keep_pairs(index_pairs, max_num_pairs=20):
    new_count = 0
    old_count = 0
    new_idx_count = defaultdict(int)
    new_pairs = defaultdict(list)
    for key1 in index_pairs:
        # sort pairs by number of pairs
        index_pairs[key1] = sorted(index_pairs[key1], key=lambda x: x[2], reverse=True)
        for pair in index_pairs[key1]:
            old_count += 1
            idx1 = key1
            idx2 = pair[0]

            if new_idx_count[key1] < max_num_pairs:
                new_pairs[idx1].append(pair)
                new_count += 1
                new_idx_count[idx1] += 1
                new_idx_count[idx2] += 1
            else:
                continue

    if DEBUG:
        print(f"origin pairs: {old_count}, kept pairs: {new_count}")
    return index_pairs


def select_matches(matches, keep_ratio = 0.01):
    max_matches = defaultdict(int)
    old_matches_count = 0
    for key1 in matches:
        for key2 in matches[key1]:
            max_matches[key1] = max(max_matches[key1], len(matches[key1][key2]))
            max_matches[key2] = max(max_matches[key2], len(matches[key1][key2]))
            old_matches_count +=1

    new_matches_count = 0
    new_matches = defaultdict(dict)
    for key1 in matches:
        for key2 in matches[key1]:
            n_matches = len(matches[key1][key2])
            if n_matches > max_matches[key1] * keep_ratio or n_matches > max_matches[key2] * keep_ratio:
                new_matches[key1][key2] = matches[key1][key2]
                new_matches_count+=1
    if DEBUG:
        print(f"origin matches: {old_matches_count}, kept matches: {new_matches_count}")
    return new_matches
