""" 
Full evaluation script for recommender system research
------------------------------------------------------
Evaluates:
- RMSE, MAE (rating accuracy)
- Precision@K, Recall@K, MAP@K, HitRate@K, nDCG@K (ranking)
Models:
- SVD (matrix factorization)
- KNNBaseline (item-based Pearson)
- KNNWithMeans (user-based cosine)
- Popularity baseline
- Random baseline
Supports leave-one-out (LOO) ranking mode and CSV/plot outputs.
"""

import argparse, math, random, csv
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from surprise import Dataset, SVD, KNNBaseline, KNNWithMeans, accuracy
from surprise.model_selection import train_test_split

# ----------------------------
# Metric functions
# ----------------------------
def precision_recall_at_k(topn, relevant, k):
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

def hitrate_at_k(topn, relevant, k):
    hits = []
    for uid, recs_k in topn.items():
        if set(recs_k[:k]) & relevant.get(uid, set()):
            hits.append(1.0)
        else:
            hits.append(0.0)
    return np.mean(hits) if hits else 0.0

def apk(actual_set, predicted_list, k):
    if not actual_set:
        return 0.0
    score, hit = 0.0, 0
    for i, iid in enumerate(predicted_list[:k], start=1):
        if iid in actual_set:
            hit += 1
            score += hit / i
    return score / min(len(actual_set), k)

def map_at_k(topn, relevant, k):
    if not topn:
        return 0.0
    vals = []
    for uid, recs_k in topn.items():
        vals.append(apk(relevant.get(uid, set()), recs_k, k))
    return np.mean(vals)

def dcg_at_k(rels, k):
    return sum((rel / math.log2(i + 2) for i, rel in enumerate(rels[:k])))

def ndcg_at_k(topn, relevant, k):
    vals = []
    for uid, recs_k in topn.items():
        actual = relevant.get(uid, set())
        gains = [1 if iid in actual else 0 for iid in recs_k[:k]]
        dcg = dcg_at_k(gains, k)
        idcg = dcg_at_k(sorted(gains, reverse=True), k)
        if idcg > 0:
            vals.append(dcg / idcg)
    return np.mean(vals) if vals else 0.0

# ----------------------------
# Helpers for test/anti-testset
# ----------------------------
def get_relevant_by_user(testset, thresh=4.0):
    rel = defaultdict(set)
    for uid, iid, r in testset:
        if r >= thresh:
            rel[uid].add(iid)
    return rel

def all_raw_iids(trainset):
    return [trainset.to_raw_iid(i) for i in trainset.all_items()]

def collect_user_seen_in_train(trainset):
    seen = defaultdict(set)
    for u_inner in trainset.all_users():
        uid = trainset.to_raw_uid(u_inner)
        for j_inner, _ in trainset.ur[u_inner]:
            seen[uid].add(trainset.to_raw_iid(j_inner))
    return seen

def get_topn_from_predictions(predictions, n=10):
    by_user = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        by_user[uid].append((iid, est))
    topn = {}
    for uid, pairs in by_user.items():
        pairs.sort(key=lambda x: x[1], reverse=True)
        topn[uid] = [iid for iid, _ in pairs[:n]]
    return topn

def popularity_topn(trainset, all_iids, seen_by_user, n=10):
    counts = Counter()
    for u_inner in trainset.all_users():
        for j_inner, _ in trainset.ur[u_inner]:
            counts[trainset.to_raw_iid(j_inner)] += 1
    ranked = [iid for iid, _ in counts.most_common()]
    out = {}
    for uid in seen_by_user:
        unseen = [iid for iid in ranked if iid not in seen_by_user[uid]]
        out[uid] = unseen[:n]
    return out

def random_topn(all_iids, seen_by_user, n=10):
    out = {}
    for uid in seen_by_user:
        unseen = [iid for iid in all_iids if iid not in seen_by_user[uid]]
        random.shuffle(unseen)
        out[uid] = unseen[:n]
    return out

# ----------------------------
# Leave-One-Out (LOO) setup
# ----------------------------
def build_leave_one_out(data, thresh=4.0):
    raw_ratings = data.raw_ratings
    by_user = defaultdict(list)
    for (uid, iid, r, _) in raw_ratings:
        by_user[uid].append((iid, r))
    train, test = [], []
    for uid, ratings in by_user.items():
        pos = [x for x in ratings if x[1] >= thresh]
        if not pos:
            # no positive ratings: keep all in train, nothing in test
            for iid, r in ratings:
                train.append((uid, iid, r, None))
            continue
        holdout = random.choice(pos)
        test.append((uid, holdout[0], holdout[1]))
        for (iid, r) in ratings:
            if (iid, r) != holdout:
                train.append((uid, iid, r, None))
    data.raw_ratings = train
    trainset = data.build_trainset()
    return trainset, test

# ----------------------------
# Main evaluation
# ----------------------------
def run(dataset_name='ml-100k', k=10, loo=False, csv_out=True):
    print(f"\n[INFO] Loading dataset: {dataset_name}")
    data = Dataset.load_builtin(dataset_name)

    # Choose split
    if loo:
        print("[INFO] Using Leave-One-Out (LOO) evaluation.")
        trainset, testset = build_leave_one_out(data)
    else:
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    relevant = get_relevant_by_user(testset, thresh=4.0)
    seen_by_user = collect_user_seen_in_train(trainset)
    all_iids = all_raw_iids(trainset)
    print(f"[INFO] Users={trainset.n_users}, Items={trainset.n_items}, Relevant test users={len(relevant)}")

    # Train models
    algo_svd = SVD(n_factors=80, n_epochs=30, random_state=42)
    algo_knn_item = KNNBaseline(sim_options={'name': 'pearson_baseline', 'user_based': False})
    algo_knn_user = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})

    print("[INFO] Training models...")
    algo_svd.fit(trainset)
    algo_knn_item.fit(trainset)
    algo_knn_user.fit(trainset)

    # Evaluate RMSE/MAE on test split
    preds_svd = algo_svd.test(testset)
    preds_knn_item = algo_knn_item.test(testset)
    preds_knn_user = algo_knn_user.test(testset)

    metrics = []
    for name, preds in [
        ("SVD", preds_svd),
        ("KNNBaseline_Item", preds_knn_item),
        ("KNNWithMeans_User", preds_knn_user),
    ]:
        rmse = accuracy.rmse(preds, verbose=False)
        mae = accuracy.mae(preds, verbose=False)
        metrics.append((name, rmse, mae))

    # ----------------------------
    # Build & (optionally) subsample anti-testset for Top-K
    # ----------------------------
    print("[INFO] Building anti-testset...")
    anti_testset = trainset.build_anti_testset()
    print(f"[INFO] Full anti-testset size: {len(anti_testset):,}")

    # For ml-1m this is huge; sub-sample for speed
    if dataset_name == "ml-1m" and not loo:
        random.shuffle(anti_testset)
        ANTI_LIMIT = 500_000   # you can adjust this (e.g., 200k, 1M)
        anti_testset = anti_testset[:ANTI_LIMIT]
        print(f"[INFO] Subsampled anti-testset to {len(anti_testset):,} pairs for speed.")

    print("[INFO] Predicting on anti-testset (may still take a few mins)...")
    preds_svd_anti = algo_svd.test(anti_testset)
    preds_knn_item_anti = algo_knn_item.test(anti_testset)
    preds_knn_user_anti = algo_knn_user.test(anti_testset)

    topn_svd = get_topn_from_predictions(preds_svd_anti, n=k)
    topn_knn_item = get_topn_from_predictions(preds_knn_item_anti, n=k)
    topn_knn_user = get_topn_from_predictions(preds_knn_user_anti, n=k)
    topn_pop = popularity_topn(trainset, all_iids, seen_by_user, n=k)
    topn_rand = random_topn(all_iids, seen_by_user, n=k)

    # Ranking metrics
    def eval_block(name, topn):
        p, r = precision_recall_at_k(topn, relevant, k)
        hr = hitrate_at_k(topn, relevant, k)
        mapv = map_at_k(topn, relevant, k)
        ndcg = ndcg_at_k(topn, relevant, k)
        return {
            "Model": name,
            "P@K": p,
            "R@K": r,
            "Hit@K": hr,
            "MAP@K": mapv,
            "nDCG@K": ndcg,
        }

    rank_results = [
        eval_block("SVD", topn_svd),
        eval_block("KNNBaseline_Item", topn_knn_item),
        eval_block("KNNWithMeans_User", topn_knn_user),
        eval_block("Popularity", topn_pop),
        eval_block("Random", topn_rand),
    ]

    # Print summary
    print(f"\n=== DATASET: {dataset_name} | K={k} | LOO={loo} ===")
    print("-- Rating Error --")
    for name, rmse, mae in metrics:
        print(f"{name:<20} RMSE={rmse:.4f}  MAE={mae:.4f}")

    print("\n-- Ranking Metrics (relevance: test ratings ≥ 4) --")
    for r in rank_results:
        print(
            f"{r['Model']:<20} "
            f"P@K={r['P@K']:.3f}  R@K={r['R@K']:.3f}  "
            f"Hit@K={r['Hit@K']:.3f}  MAP@K={r['MAP@K']:.3f}  "
            f"nDCG@K={r['nDCG@K']:.3f}"
        )

    # CSV output
    if csv_out:
        fname = f"metrics_{dataset_name}.csv"
        with open(fname, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "RMSE", "MAE", "P@K", "R@K", "Hit@K", "MAP@K", "nDCG@K"])
            for (name, rmse, mae), r in zip(metrics, rank_results):
                writer.writerow([
                    name, rmse, mae,
                    r["P@K"], r["R@K"], r["Hit@K"], r["MAP@K"], r["nDCG@K"]
                ])
        print(f"[SAVED] {fname}")

    # Plot summary
    models = [r["Model"] for r in rank_results]
    p_values = [r["P@K"] for r in rank_results]
    ndcg_values = [r["nDCG@K"] for r in rank_results]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(models))
    plt.bar(x - 0.2, p_values, width=0.4, label="Precision@K")
    plt.bar(x + 0.2, ndcg_values, width=0.4, label="nDCG@K")
    plt.xticks(x, models, rotation=30)
    plt.ylabel("Score")
    plt.title(f"Precision@{k} and nDCG@{k} Comparison ({dataset_name})")
    plt.legend()
    plt.tight_layout()
    out_plot = f"ranking_metrics_{dataset_name}.png"
    plt.savefig(out_plot)
    print(f"[SAVED] {out_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-100k", "ml-1m"], default="ml-100k")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--loo", action="store_true", help="Use leave-one-out evaluation")
    args = parser.parse_args()
    run(dataset_name=args.dataset, k=args.k, loo=args.loo)
