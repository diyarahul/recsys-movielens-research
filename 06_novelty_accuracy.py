import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from surprise import Dataset, SVD, KNNBasic
from surprise.model_selection import train_test_split

data = Dataset.load_builtin("ml-100k")
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


cf = KNNBasic(sim_options={"name": "cosine", "user_based": True})
cf.fit(trainset)
preds_cf = cf.test(testset)

svd = SVD(random_state=42)
svd.fit(trainset)
preds_svd = svd.test(testset)


def get_top_n(predictions, n=10):
    per_user = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        per_user[uid].append((iid, est))
    for uid, lst in per_user.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        per_user[uid] = lst[:n]
    return per_user

topn_cf = get_top_n(preds_cf, n=10)
topn_svd = get_top_n(preds_svd, n=10)


default_path = os.path.expanduser("~/.surprise_data/ml-100k/ml-100k/u.data")
ratings = pd.read_csv(
    default_path, sep="\t",
    names=["user", "item", "rating", "timestamp"]
)
pop_counts = ratings.groupby("item").size().sort_values(ascending=False)
pop_rank = {str(iid): rank for rank, iid in enumerate(pop_counts.index.tolist(), start=1)}

# --- Helper to compute per-user RMSE and Novelty ---
def get_rmse_novelty(preds, topn, pop_rank_dict):
    rows = []
    by_user = defaultdict(list)
    for p in preds:
        by_user[p.uid].append(p)

    for uid in topn.keys():
        user_preds = by_user.get(uid, [])
        if not user_preds:
            continue
        true_ratings = np.array([p.r_ui for p in user_preds]) 
        est_ratings  = np.array([p.est  for p in user_preds])
        rmse_user = float(np.sqrt(np.mean((true_ratings - est_ratings) ** 2)))

        
        nov_vals = [pop_rank_dict.get(str(iid)) for iid, _ in topn[uid]]
        nov_vals = [v for v in nov_vals if v is not None]
        if not nov_vals:
            continue
        novelty_user = float(np.mean(nov_vals))

        rows.append((uid, rmse_user, novelty_user))

    df = pd.DataFrame(rows, columns=["user", "RMSE", "Novelty"])
    corr = df["RMSE"].corr(df["Novelty"]) if len(df) > 1 else np.nan
    return df, corr

df_cf, corr_cf   = get_rmse_novelty(preds_cf,  topn_cf,  pop_rank)
df_svd, corr_svd = get_rmse_novelty(preds_svd, topn_svd, pop_rank)

print(f"CF   correlation (RMSE vs Novelty): {corr_cf:.4f}")
print(f"SVD  correlation (RMSE vs Novelty): {corr_svd:.4f}")


fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

axes[0].scatter(df_cf["RMSE"], df_cf["Novelty"], alpha=0.6, edgecolor="k")
axes[0].set_title(f"CF (KNNBasic) — Corr={corr_cf:.3f}")
axes[0].set_xlabel("RMSE (Prediction Error)")
axes[0].set_ylabel("Novelty (Avg Popularity Rank)")
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].scatter(df_svd["RMSE"], df_svd["Novelty"], alpha=0.6, edgecolor="k")
axes[1].set_title(f"SVD — Corr={corr_svd:.3f}")
axes[1].set_xlabel("RMSE (Prediction Error)")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.suptitle("Accuracy–Novelty per User (Test Split)", y=1.02, fontsize=14)
plt.tight_layout()

out_path = os.path.join(os.getcwd(), "cf_svd_rmse_novelty_comparison.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"[SAVED] {out_path}")
