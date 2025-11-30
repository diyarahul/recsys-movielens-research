"""
08_svd_tuning.py
----------------
Hyperparameter tuning script for the SVD model in the Surprise library.

For each SVD configuration, this script:
  - Trains on the training split of MovieLens (ml-100k or ml-1m)
  - Evaluates rating accuracy (RMSE, MAE) on the held-out test split
  - Builds an (optionally subsampled) anti-testset
  - Computes Precision@K and nDCG@K on the Top-K recommendations

Results are saved to:
  svd_tuning_<dataset>.csv
"""

import argparse
import csv
import math
import random
from collections import defaultdict
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, SVD, accuracy
from surprise.model_selection import train_test_split

# ----------------------------
# Metric helpers
# ----------------------------

def precision_recall_at_k(topn, relevant, k):
    """Compute mean Precision@K and Recall@K over users."""
    precs, recs = [], []
    for uid, recs_k in topn.items():
        if uid not in relevant:
            continue
        rel_u = relevant[uid]
        if not rel_u:
            continue
        hit = len(set(recs_k[:k]) & rel_u)
        precs.append(hit / max(1, min(k, len(recs_k))))
        recs.append(hit / len(rel_u))
    return (np.mean(precs) if precs else 0.0,
            np.mean(recs) if recs else 0.0)


def dcg_at_k(rels, k):
    return sum((rel / math.log2(i + 2) for i, rel in enumerate(rels[:k])))


def ndcg_at_k(topn, relevant, k):
    """Compute mean nDCG@K"""
    vals = []
    for uid, recs_k in topn.items():
        actual = relevant.get(uid, set())
        gains = [1 if iid in actual else 0 for iid in recs_k[:k]]
        dcg = dcg_at_k(gains, k)
        idcg = dcg_at_k(sorted(gains, reverse=True), k)
        if idcg > 0:
            vals.append(dcg / idcg)
    return np.mean(vals) if vals else 0.0


def get_relevant_by_user(testset, thresh=4.0):
    """Get relevant items per user from testset based on rating threshold."""
    rel = defaultdict(set)
    for uid, iid, r in testset:
        if r >= thresh:
            rel[uid].add(iid)
    return rel


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


def run_tuning(dataset_name="ml-100k", k=10):
    print(f"\n[INFO] Loading dataset: {dataset_name}")
    data = Dataset.load_builtin(dataset_name)

    # Standard 80/20 split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    relevant = get_relevant_by_user(testset, thresh=4.0)

    print(f"[INFO] Users={trainset.n_users}, Items={trainset.n_items}, "
          f"Relevant test users={len(relevant)}")

    print("[INFO] Building anti-testset...")
    anti_testset = trainset.build_anti_testset()
    print(f"[INFO] Full anti-testset size: {len(anti_testset):,}")

    if dataset_name == "ml-1m":
        random.shuffle(anti_testset)
        ANTI_LIMIT = 300_000  
        anti_testset = anti_testset[:ANTI_LIMIT]
        print(f"[INFO] Subsampled anti-testset to {len(anti_testset):,} pairs.")

    if dataset_name == "ml-100k":
        n_factors_list = [50, 80, 120]
        n_epochs_list = [20, 40]
        lr_list = [0.005, 0.007]
        reg_list = [0.02, 0.05]
    else: 
        n_factors_list = [80, 120]
        n_epochs_list = [20, 30]
        lr_list = [0.005]
        reg_list = [0.02, 0.05]

    grid: List[Dict] = []
    for nf in n_factors_list:
        for ne in n_epochs_list:
            for lr in lr_list:
                for reg in reg_list:
                    grid.append({
                        "n_factors": nf,
                        "n_epochs": ne,
                        "lr_all": lr,
                        "reg_all": reg
                    })

    print(f"[INFO] Number of SVD configurations to evaluate: {len(grid)}")

    results = []

    for cfg in grid:
        print("\n[INFO] Training SVD with params:",
              f"factors={cfg['n_factors']}, epochs={cfg['n_epochs']}, "
              f"lr={cfg['lr_all']}, reg={cfg['reg_all']}")

        algo = SVD(
            n_factors=cfg["n_factors"],
            n_epochs=cfg["n_epochs"],
            lr_all=cfg["lr_all"],
            reg_all=cfg["reg_all"],
            random_state=42,
        )

       
        algo.fit(trainset)

        
        preds_test = algo.test(testset)
        rmse = accuracy.rmse(preds_test, verbose=False)
        mae = accuracy.mae(preds_test, verbose=False)

        
        preds_anti = algo.test(anti_testset)
        topn = get_topn_from_predictions(preds_anti, n=k)
        prec, rec = precision_recall_at_k(topn, relevant, k)
        ndcg = ndcg_at_k(topn, relevant, k)

        print(f"    → RMSE={rmse:.4f}, MAE={mae:.4f}, "
              f"P@{k}={prec:.4f}, nDCG@{k}={ndcg:.4f}")

        row = {
            "n_factors": cfg["n_factors"],
            "n_epochs": cfg["n_epochs"],
            "lr_all": cfg["lr_all"],
            "reg_all": cfg["reg_all"],
            "RMSE": rmse,
            "MAE": mae,
            f"P@{k}": prec,
            f"nDCG@{k}": ndcg,
        }
        results.append(row)

    csv_name = f"svd_tuning_{dataset_name}.csv"
    fieldnames = ["n_factors", "n_epochs", "lr_all", "reg_all",
                  "RMSE", "MAE", f"P@{k}", f"nDCG@{k}"]
    with open(csv_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[SAVED] {csv_name}")

    plt.figure(figsize=(6, 4))
    colors = {50: "tab:blue", 80: "tab:orange", 120: "tab:green", 200: "tab:red"}
    for row in results:
        nf = row["n_factors"]
        plt.scatter(
            row["RMSE"],
            row[f"P@{k}"],
            color=colors.get(nf, "gray"),
            label=f"factors={nf}",
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), title="n_factors", loc="best")

    plt.xlabel("RMSE (Test Split)")
    plt.ylabel(f"Precision@{k} (Top-{k})")
    plt.title(f"SVD Tuning: RMSE vs Precision@{k} ({dataset_name})")
    plt.tight_layout()
    plot_name = f"svd_tuning_rmse_vs_p@{k}_{dataset_name}.png"
    plt.savefig(plot_name)
    print(f"[SAVED] {plot_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    run_tuning(dataset_name=args.dataset, k=args.k)
