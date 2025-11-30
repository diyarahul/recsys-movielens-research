"""
09_svd_novelty_diversity.py
---------------------------
Compare default SVD vs tuned SVD on *beyond-accuracy* metrics:

  - Coverage: fraction of catalog that appears in any user's Top-N list
  - Novelty: average popularity rank of recommended items
             (higher rank = less popular = more novel)
  - ILD (Intra-List Diversity): average pairwise dissimilarity inside Top-N
             (higher ILD = more varied recommendations)

Supports: ml-100k, ml-1m
Note:
  - ml-100k: novelty + coverage + ILD (using genres from u.item)
  - ml-1m:   novelty + coverage only (ILD not computed due to missing genre metadata)
"""

import argparse
import math
import os
import random
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, SVD, accuracy, get_dataset_dir
from surprise.model_selection import train_test_split


def build_popularity_rank(trainset):
    """
    Count how often each item is rated in the train set
    and convert that into a 1-based popularity rank (1 = most popular).
    Returns: (rank_dict, count_dict)
    """
    counts = Counter()
    for u_inner in trainset.all_users():
        for j_inner, _ in trainset.ur[u_inner]:
            raw_iid = trainset.to_raw_iid(j_inner)
            counts[raw_iid] += 1

    ranked_items = [iid for iid, _ in counts.most_common()]
    rank = {iid: r + 1 for r, iid in enumerate(ranked_items)}
    return rank, counts


def get_topn_from_predictions(predictions, n=10):
    """Convert (uid, iid, true_r, est, details) predictions to Top-N dict."""
    by_user = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        by_user[uid].append((iid, est))
    topn = {}
    for uid, pairs in by_user.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        topn[uid] = [iid for iid, _ in pairs[:n]]
    return topn


def compute_novelty(topn, pop_rank):
    vals = []
    for uid, items in topn.items():
        ranks = [pop_rank.get(iid) for iid in items if iid in pop_rank]
        if ranks:
            vals.append(np.mean(ranks))
    return np.mean(vals) if vals else 0.0


def compute_coverage(topn, all_iids):
    """
    Coverage = number of distinct items recommended / total items in catalog.
    """
    recommended_items = set()
    for items in topn.values():
        recommended_items.update(items)
    return len(recommended_items) / len(all_iids) if all_iids else 0.0


def load_genres_ml100k():
    #Load genres for ml-100k from u.item.
    base = get_dataset_dir()
    path = os.path.join(base, "ml-100k", "u.item")
    genres_by_iid = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if not parts or len(parts) < 5:
                continue
            iid = parts[0]  
            flags = parts[-19:]  
            gset = {f"g{i}" for i, v in enumerate(flags) if v == "1"}
            genres_by_iid[iid] = gset
    return genres_by_iid


def jaccard_dissimilarity(g1, g2):
    """
    Jaccard-based dissimilarity between two genre sets.
    0  = identical genre set
    1  = completely disjoint
    """
    if not g1 and not g2:
        return 0.0
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return 1.0 - (inter / union) if union > 0 else 0.0


def compute_ild(topn, genres_by_iid):
    #ILD per user = average pairwise genre dissimilarity among Top-N items.
    ild_vals = []

    for uid, items in topn.items():
        if len(items) < 2:
            continue

        gsets = [genres_by_iid.get(iid) for iid in items if iid in genres_by_iid]
        gsets = [g for g in gsets if g is not None]

        if len(gsets) < 2:
            continue

        pair_dists = []
        for i in range(len(gsets)):
            for j in range(i + 1, len(gsets)):
                pair_dists.append(jaccard_dissimilarity(gsets[i], gsets[j]))

        if pair_dists:
            ild_vals.append(np.mean(pair_dists))

    return np.mean(ild_vals) if ild_vals else 0.0



def run(dataset_name="ml-1m", k=10):
    print(f"\n[INFO] Loading dataset: {dataset_name}")
    data = Dataset.load_builtin(dataset_name)

    # 80/20 split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Popularity and catalog info
    pop_rank, pop_counts = build_popularity_rank(trainset)
    all_iids = [trainset.to_raw_iid(i) for i in trainset.all_items()]

    print(f"[INFO] Users={trainset.n_users}, Items={trainset.n_items}")

    anti = trainset.build_anti_testset()
    print(f"[INFO] Anti-testset size: {len(anti):,}")
    if dataset_name == "ml-1m":
        random.shuffle(anti)
        ANTI_LIMIT = 300_000
        anti = anti[:ANTI_LIMIT]
        print(f"[INFO] Subsampled anti-testset to {len(anti):,} pairs.")

    if dataset_name == "ml-100k":
        genres_by_iid = load_genres_ml100k()
        ild_supported = True
        print(f"[INFO] Loaded genre metadata for {len(genres_by_iid)} items (ILD enabled).")
    else:
        genres_by_iid = None
        ild_supported = False
        print("[INFO] Genre metadata not available for ml-1m in this script. ILD will be reported as N/A.")

    svd_default = SVD(
        n_factors=80,
        n_epochs=30,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )

    svd_tuned = SVD(
        n_factors=80,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
    )

    models = [("SVD_default", svd_default), ("SVD_tuned", svd_tuned)]

    results = []

    for name, algo in models:
        print(f"\n[INFO] Training {name} ...")
        algo.fit(trainset)

     
        preds_test = algo.test(testset)
        rmse = accuracy.rmse(preds_test, verbose=False)
        mae = accuracy.mae(preds_test, verbose=False)


        print(f"[INFO] Building Top-{k} recommendations for {name} ...")
        preds_anti = algo.test(anti)
        topn = get_topn_from_predictions(preds_anti, n=k)

        cov = compute_coverage(topn, all_iids)
        nov = compute_novelty(topn, pop_rank)
        if ild_supported:
            ild = compute_ild(topn, genres_by_iid)
        else:
            ild = float("nan")  # not available

        results.append((name, rmse, mae, cov, nov, ild))

    print("\n=== Beyond-Accuracy Metrics on Top-{} ({}) ===".format(k, dataset_name))
    print("Model         RMSE     MAE     Coverage  Novelty(avg rank)   ILD")
    for name, rmse, mae, cov, nov, ild in results:
        if math.isnan(ild):
            ild_str = "  N/A"
        else:
            ild_str = f"{ild:6.3f}"
        print(f"{name:<12} {rmse:6.4f}  {mae:6.4f}   {cov:6.3f}    {nov:10.1f}      {ild_str}")

    labels = [r[0] for r in results]
    ild_vals = [0 if math.isnan(r[5]) else r[5] for r in results]
    nov_vals = [r[4] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, ild_vals, width, label="ILD (diversity)")
    plt.bar(x + width / 2, nov_vals, width, label="Novelty (avg rank)")
    plt.xticks(x, labels)
    plt.ylabel("Score (ILD, Novelty)")
    plt.title(f"Novelty & Diversity: Default vs Tuned SVD ({dataset_name})")
    plt.legend()
    plt.tight_layout()
    out_name = f"svd_novelty_diversity_{dataset_name}.png"
    plt.savefig(out_name)
    print(f"[SAVED] {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-1m")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    run(dataset_name=args.dataset, k=args.k)
