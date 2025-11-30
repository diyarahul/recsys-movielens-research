"""
03_svd_train.py
Train an SVD (matrix factorization) model on MovieLens 100k
and save the trained model + exact testset for evaluation.
"""
import os, pickle
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split

# 1) Load data
data = Dataset.load_builtin('ml-100k')

# 2) Split (80/20) for accuracy eval
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3) Define SVD
algo = SVD(random_state=42)

# 4) Fit
print("[INFO] Training SVD ...")
algo.fit(trainset)
print("[INFO] Training complete.")

# 5) Save model + testset
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "svd.pkl")
testset_path = os.path.join("models", "svd_testset.pkl")

with open(model_path, "wb") as f:
    pickle.dump(algo, f)

with open(testset_path, "wb") as f:
    pickle.dump(testset, f)

print(f"[INFO] Saved model to:   {model_path}")
print(f"[INFO] Saved testset to: {testset_path}")
print("[DONE] You can now run 04_svd_evaluate.py to compute RMSE/MAE.")
