"""
Build Top-N lists for your saved CF and SVD models on MovieLens 100k,
then compute Coverage, Novelty (avg popularity rank), and optional ILD.
"""

import os
import pickle
from collections import defaultdict
import os.path as osp

import numpy as np
import pandas as pd
from surprise import Dataset

MODELS_DIR = "models"
CF_PATH  = osp.join(MODELS_DIR, "cf_knn_user_cosine.pkl")
SVD_PATH = osp.join(MODELS_DIR, "svd.pkl")

TOP_N = 10
COMPUTE_ILD = True  # set to False if you don't want genre-based ILD


DEFAULT_CACHE_DIR = osp.expanduser("~/.surprise_data/ml-100k/ml-100k")
U_ITEM_PATH = osp.join(DEFAULT_CACHE_DIR, "u.item")


def make_top_n(predictions, n=10):
    per_user = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        per_user[uid].append((iid, est))
    for uid, lst in per_user.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        per_user[uid] = lst[:n]
    return per_user


def compute_coverage(topn_dict, all_items_raw):
    """Fraction of catalog items that appear at least once in any user's Top-N."""
    recommended = set()
    for recs in topn_dict.values():
        for iid, _ in recs:
            recommended.add(iid)
    return len(recommended) / len(all_items_raw)


def popularity_rank_from_trainset(trainset):
    """Return dict raw_iid -> popularity_rank (1 = most popular)."""
    counts = defaultdict(int)
    for _, iid_inner, _ in trainset.all_ratings():
        raw_iid = trainset.to_raw_iid(iid_inner)
        counts[raw_iid] += 1
    pop_df = pd.DataFrame([{"item": k, "cnt": v} for k, v in counts.items()])
    pop_df["pop_rank"] = pop_df["cnt"].rank(ascending=False, method="min")
    return dict(zip(pop_df["item"], pop_df["pop_rank"]))


def average_popularity_rank(topn_dict, pop_rank_dict):
    ranks = []
    for recs in topn_dict.values():
        for iid, _ in recs:
            if iid in pop_rank_dict:
                ranks.append(pop_rank_dict[iid])
    return (sum(ranks) / len(ranks)) if ranks else float("nan")


def load_genre_map(u_item_path):
    genre_map = {}
    if not osp.exists(u_item_path):
        return genre_map  # empty triggers ILD skip gracefully

    with open(u_item_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = parts[0]
            flags = np.array(list(map(int, parts[-19:])))
            genre_map[iid] = flags
    return genre_map


def intra_list_diversity(topn_dict, genre_map):
    if not genre_map:
        return float("nan")

    def jaccard(a, b):
        inter = np.minimum(a, b).sum()
        union = np.maximum(a, b).sum()
        return 0.0 if union == 0 else inter / union

    diversities = []
    for recs in topn_dict.values():
        items = [iid for iid, _ in recs if iid in genre_map]
        if len(items) < 2:
            continue
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sims.append(jaccard(genre_map[items[i]], genre_map[items[j]]))
        if sims:
            diversities.append(1 - (sum(sims) / len(sims)))
    return (sum(diversities) / len(diversities)) if diversities else float("nan")


if not osp.exists(CF_PATH) or not osp.exists(SVD_PATH):
    raise FileNotFoundError(
        "Missing saved models. Make sure you've run 01_cf_train.py and 03_svd_train.py.\n"
        f"Expected: {CF_PATH} and {SVD_PATH}"
    )

with open(CF_PATH, "rb") as f:
    cf_algo = pickle.load(f)

with open(SVD_PATH, "rb") as f:
    svd_algo = pickle.load(f)


data = Dataset.load_builtin("ml-100k")
full_train = data.build_full_trainset()


print("[INFO] Fitting CF model on full data ...")
cf_algo.fit(full_train)
print("[INFO] Fitting SVD model on full data ...")
svd_algo.fit(full_train)

# Anti-testset = all unseen (user, item) pairs
anti = full_train.build_anti_testset()

# Predictions for all unknown pairs
print("[INFO] Scoring anti-testset ... this may take a minute.")
preds_cf  = cf_algo.test(anti)
preds_svd = svd_algo.test(anti)

# Top-N dicts
topn_cf  = make_top_n(preds_cf,  n=TOP_N)
topn_svd = make_top_n(preds_svd, n=TOP_N)

# Catalog (raw item ids)
all_items_raw = {full_train.to_raw_iid(i) for i in full_train.all_items()}

# Popularity rank dict for Novelty
pop_rank = popularity_rank_from_trainset(full_train)

# Coverage & Novelty
cov_cf  = compute_coverage(topn_cf,  all_items_raw)
cov_svd = compute_coverage(topn_svd, all_items_raw)

nov_cf  = average_popularity_rank(topn_cf,  pop_rank)
nov_svd = average_popularity_rank(topn_svd, pop_rank)

# Optional ILD
ild_cf = ild_svd = float("nan")
if COMPUTE_ILD:
    genre_map = load_genre_map(U_ITEM_PATH)
    ild_cf  = intra_list_diversity(topn_cf,  genre_map)
    ild_svd = intra_list_diversity(topn_svd, genre_map)


print("\n=== Beyond-Accuracy Metrics on Top-{} (higher is better for all below) ===".format(TOP_N))
print(f"Coverage         | CF:  {cov_cf:.3f}    SVD: {cov_svd:.3f}")
print(f"Novelty (avg pop rank) | CF:  {nov_cf:.1f}   SVD: {nov_svd:.1f}  (higher = more novel)")
if COMPUTE_ILD:
    print(f"ILD (genre)      | CF:  {ild_cf:.3f}    SVD: {ild_svd:.3f}  (higher = more varied lists)")
print("-------------------------------------------------------------------------------\n")
