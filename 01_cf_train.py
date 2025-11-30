"""
01_cf_train.py
Train a Collaborative Filtering model (KNNBasic, user-based) on MovieLens 100k
and save the trained model + the exact testset for later evaluation.
"""

import os
import pickle
from collections import defaultdict

from surprise import Dataset, KNNBasic
from surprise.model_selection import train_test_split


# 1) Load MovieLens 100k
data = Dataset.load_builtin('ml-100k')


# 2) Train/test split (80/20)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)


# 3) Define CF model (KNNBasic)
#    - cosine similarity
#    - user-based (set to False for item-based)
sim_options = {"name": "cosine", "user_based": True}
algo = KNNBasic(sim_options=sim_options)


print("[INFO] Training KNNBasic (user-based, cosine) ...")
algo.fit(trainset)
print("[INFO] Training complete.")

os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "cf_knn_user_cosine.pkl")
testset_path = os.path.join("models", "cf_testset.pkl")

with open(model_path, "wb") as f:
    pickle.dump(algo, f)

with open(testset_path, "wb") as f:
    pickle.dump(testset, f)

print(f"[INFO] Saved model to:   {model_path}")
print(f"[INFO] Saved testset to: {testset_path}")
print("[DONE] You can now run 02_cf_evaluate.py to compute RMSE/MAE.")
