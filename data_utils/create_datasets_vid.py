import numpy as np
import argparse
import random
import pickle
import os
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="gtea")
parser.add_argument("--ratio", default=1.0)
args = parser.parse_args()


features_path = f"datasets/{args.dataset}/vidc_features/"
gt_path = f"datasets/{args.dataset}/groundTruth/"
mapping_file = f"datasets/{args.dataset}/mapping.txt"

with open(mapping_file, "r") as f:
    lines = f.readlines()

actions = {line.split()[-1]: int(line.split()[0]) for line in lines}
num_actions = len(actions)


def find_segments(indices):
    if not indices:
        return []

    continuous_sets = []
    current_set = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_set.append(indices[i])
        else:
            continuous_sets.append(current_set)
            current_set = [indices[i]]

    continuous_sets.append(current_set)
    return continuous_sets


def get_progression(indices):
    sets = find_segments(indices)

    durations = list(map(len, sets))
    pos = []
    for d in durations:
        rel_pos = np.linspace(0, 1, d)
        pos.extend(rel_pos)

    return np.array(pos, dtype=np.float32)


for split in [1]:
    vid_list_path = f"datasets/{args.dataset}/splits/"
    with open(os.path.join(vid_list_path, f"train.split{split}.bundle"), "r") as f:
        videos = f.read().splitlines()

    # num_videos = int(len(videos) * args.ratio)
    # sub_videos = random.sample(videos, k=num_videos)
    task_feats = []
    task_feat_path = vid_list_path + f"train.split{split}_vidc.npy"
    for video in tqdm(videos):
        video_feat = features_path + video.replace("txt", "npy")
        video_gt = gt_path + video
        feats = np.load(video_feat, allow_pickle=True)
        with open(video_gt, "r") as f:
            labels = f.read().splitlines()
            uniq_labels = set(labels)
            for uniq_label in uniq_labels:
                idx = np.where(np.array(labels) == uniq_label)
                act_idx = actions[uniq_label]
                # features
                cur_feats = feats[idx]
                # positions
                cur_pos = get_progression(idx[0].tolist())
                # labels
                cur_labels = np.zeros([idx[0].shape[0], num_actions], dtype=np.float32)
                cur_labels[:, act_idx] = 1

                task_feats += list(zip(cur_feats, cur_pos, cur_labels))

    with open(task_feat_path, "wb") as f:
        pickle.dump(task_feats, f)
