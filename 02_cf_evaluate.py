"""
02_cf_evaluate.py
Load the trained CF model + saved testset, run predictions,
and report RMSE and MAE.
"""

import os
import pickle

from surprise import accuracy

MODEL_PATH = os.path.join("models", "cf_knn_user_cosine.pkl")
TESTSET_PATH = os.path.join("models", "cf_testset.pkl")


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run `python 01_cf_train.py` first."
    )

if not os.path.exists(TESTSET_PATH):
    raise FileNotFoundError(
        f"Testset not found at {TESTSET_PATH}. "
        "Run `python 01_cf_train.py` first."
    )

with open(MODEL_PATH, "rb") as f:
    algo = pickle.load(f)

with open(TESTSET_PATH, "rb") as f:
    testset = pickle.load(f)


print("[INFO] Generating predictions on the held-out test set ...")
predictions = algo.test(testset)


rmse = accuracy.rmse(predictions, verbose=False)
mae  = accuracy.mae(predictions,  verbose=False)

print("\n=== Collaborative Filtering (KNNBasic, user-based, cosine) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print("==============================================================\n")
