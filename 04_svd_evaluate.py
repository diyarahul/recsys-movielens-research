"""
04_svd_evaluate.py
Load the trained SVD model + saved testset, run predictions,
and report RMSE and MAE.
"""
import os, pickle
from surprise import accuracy

MODEL_PATH = os.path.join("models", "svd.pkl")
TESTSET_PATH = os.path.join("models", "svd_testset.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run `python 03_svd_train.py` first.")
if not os.path.exists(TESTSET_PATH):
    raise FileNotFoundError("Testset not found. Run `python 03_svd_train.py` first.")

with open(MODEL_PATH, "rb") as f:
    algo = pickle.load(f)
with open(TESTSET_PATH, "rb") as f:
    testset = pickle.load(f)

print("[INFO] Generating predictions on the held-out test set ...")
preds = algo.test(testset)

rmse = accuracy.rmse(preds, verbose=False)
mae  = accuracy.mae(preds,  verbose=False)

print("\n=== SVD (matrix factorization) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print("==================================\n")
