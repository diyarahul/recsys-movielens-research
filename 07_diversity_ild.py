
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Dataset, KNNBasic, SVD

TOP_N = 10
RUN_CF = True      
RUN_SVD = True     

U_ITEM_PATH = os.path.expanduser("~/.surprise_data/ml-100k/ml-100k/u.item")
OUT_DIR = os.getcwd()


def load_genre_map(u_item_path=U_ITEM_PATH):
    """raw_iid -> 19-dim binary genre vector"""
    if not os.path.exists(u_item_path):
        raise FileNotFoundError(
            f"Could not find u.item at {u_item_path}. "
            "Adjust U_ITEM_PATH if your cache location differs."
        )
    genre_map = {}
    with open(u_item_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            iid = parts[0] 
            flags = np.array(list(map(int, parts[-19:])))
            genre_map[iid] = flags
    return genre_map


def make_top_n(predictions, n=10):
    """Surprise predictions -> {uid: [(iid, est), ...top n]}"""
    per_user = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        per_user[uid].append((iid, est))
    for uid, lst in per_user.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        per_user[uid] = lst[:n]
    return per_user


def jaccard(a, b):
    inter = np.minimum(a, b).sum()
    union = np.maximum(a, b).sum()
    return 0.0 if union == 0 else inter / union


def ild_per_user(topn_dict, genre_map):
    """ILD = 1 - avg Jaccard similarity across all pairs in a user's Top-N."""
    ild = {}
    for uid, recs in topn_dict.items():
        items = [iid for iid, _ in recs if iid in genre_map]
        if len(items) < 2:
            continue
        sims = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                sims.append(jaccard(genre_map[items[i]], genre_map[items[j]]))
        if sims:
            ild[uid] = 1.0 - float(np.mean(sims))
    return ild


def main():
    # Load data and build full trainset 
    data = Dataset.load_builtin("ml-100k")
    full_train = data.build_full_trainset()
    anti = full_train.build_anti_testset()

    genre_map = load_genre_map()


    results_rows = []
    mean_ild = {}

   
    if RUN_CF:
        print("[INFO] Training CF on full data ...")
        cf = KNNBasic(sim_options={"name": "cosine", "user_based": True})
        cf.fit(full_train)

        print("[INFO] Scoring anti-testset with CF (this may take several minutes) ...")
        preds_cf = cf.test(anti)

        print("[INFO] Building CF Top-N ...")
        topn_cf = make_top_n(preds_cf, n=TOP_N)

        print("[INFO] Computing CF ILD ...")
        ild_cf = ild_per_user(topn_cf, genre_map)

        df_cf = pd.DataFrame(
            [(u, v, "CF (KNNBasic)") for u, v in ild_cf.items()],
            columns=["user", "ILD", "model"]
        )
        mean_ild["CF (KNNBasic)"] = df_cf["ILD"].mean()
        results_rows.append(df_cf)


    if RUN_SVD:
        print("[INFO] Training SVD on full data ...")
        svd = SVD(random_state=42)
        svd.fit(full_train)

        print("[INFO] Scoring anti-testset with SVD ...")
        preds_svd = svd.test(anti)

        print("[INFO] Building SVD Top-N ...")
        topn_svd = make_top_n(preds_svd, n=TOP_N)

        print("[INFO] Computing SVD ILD ...")
        ild_svd = ild_per_user(topn_svd, genre_map)

        df_svd = pd.DataFrame(
            [(u, v, "SVD") for u, v in ild_svd.items()],
            columns=["user", "ILD", "model"]
        )
        mean_ild["SVD"] = df_svd["ILD"].mean()
        results_rows.append(df_svd)


    if not results_rows:
        print("[WARN] No models were run. Set RUN_CF/RUN_SVD to True.")
        return

    df_all = pd.concat(results_rows, ignore_index=True)
    csv_path = os.path.join(OUT_DIR, "anti_ild_by_user.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")


    print("\n=== Mean ILD on Anti-Testset Top-{} ===".format(TOP_N))
    for k, v in mean_ild.items():
        print(f"{k:15s} : {v:.3f}")

    models = list(mean_ild.keys())
    vals = [mean_ild[m] for m in models]
    plt.figure(figsize=(6, 5))
    plt.bar(models, vals)
    plt.ylabel("Mean ILD (higher = more varied Top-N)")
    plt.title(f"Mean ILD by Model (Anti-Testset, Top-{TOP_N})")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    out_bar = os.path.join(OUT_DIR, "anti_mean_ild_by_model.png")
    plt.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {out_bar}")


if __name__ == "__main__":
    main()
