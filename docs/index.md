---
layout: default
title: Beyond Accuracy: Evaluating User Satisfaction in Movie Recommendation Systems
---

<div class="hero">
  <div class="hero-text">
    <p class="hero-kicker">Recommender Systems · MovieLens · User Satisfaction</p>
    <h1>Recommendations For You That Know <em>You</em></h1>
    <p class="hero-subtitle">
      Exploring how different recommender system models balance accuracy, ranking quality,
      novelty, and diversity using the MovieLens datasets.
    </p>
    <p class="hero-meta"><strong>Student Researcher:</strong> Diya Rahul</p>
  </div>
</div>

---

## Motivation

Recommender systems quietly decide what we watch, listen to, and buy — but they are usually judged
by **how closely they predict a rating** (metrics like RMSE or MAE).  
Silveira et&nbsp;al. (2019) argue that rating error is a **weak proxy** for user satisfaction.  
A good system should also recommend items that are:

- **Novel** – not just the same popular hits everyone already knows  
- **Diverse** – a mix of different genres or types  
- **Well-ranked in a Top-N list** – the best items appear near the top  

This project puts those ideas into practice by comparing multiple recommender models on MovieLens
data and evaluating **both accuracy and user-oriented metrics**.

---

## Data

<div class="pill pill-section">MovieLens datasets</div>

- **MovieLens 100K**  
  - 100,000 ratings from 943 users on 1,682 movies  
  - Used for early experiments and debugging

- **MovieLens 1M**  
  - 1,000,209 ratings from 6,040 users on 3,900 movies  
  - Used as the main dataset to test scalability and more realistic behavior

For each dataset I:

1. Used the Surprise library to **load the built-in MovieLens splits**.  
2. Performed an **80/20 train–test split**.  
   - Train set: what each model learns from (user–movie–rating triples).  
   - Test set: ratings the model never sees during training; used to evaluate prediction error.  
3. Built an **anti-testset**: all user–movie pairs not in the train set, approximating
   “movies this user hasn’t rated yet.”  
   - For MovieLens-1M, the full anti-testset would have >21M pairs, so I **subsampled** to
     a manageable 300–500k pairs for ranking experiments.

---

## Models Compared

<div class="pill pill-section">Recommender Models</div>

I focused on widely used, classical models so that the results would be easy to interpret:

- **SVD (Matrix Factorization)**  
  - Learns a low-dimensional vector (latent factors) for each user and item.  
  - Captures hidden patterns like “action-movie preference” or “rom-com vs. drama.”  

- **KNNBaseline (Item-based Collaborative Filtering with Pearson Baseline)**  
  - Computes similarity between movies based on rating patterns.  
  - Uses baseline bias terms (user & item averages) to adjust for harsh / generous raters.  

- **KNNWithMeans (User-based Collaborative Filtering with cosine similarity)**  
  - Finds similar users and recommends what they liked, after subtracting each user’s mean rating.  

- **Popularity Baseline**  
  - Always recommends the globally most-rated movies, ignoring personalization.  

- **Random Baseline**  
  - Recommends random unseen items; useful as a very weak lower bound.

---

## Evaluation Metrics

<div class="pill pill-section">Accuracy metrics</div>

**On the held-out test set:**

- **RMSE (Root Mean Squared Error)**  
  Measures how close predicted ratings are to the true ratings (penalizes large errors).

- **MAE (Mean Absolute Error)**  
  Measures average absolute difference between predicted and true ratings.

<div class="pill pill-section">Ranking metrics (Top-10)</div>

On the anti-testset, for each user I built a **personalized Top-10 list** and asked:
“How well do these 10 items line up with the movies this user actually rated highly in the test set?”

A movie is treated as **relevant** if its true rating in the test set is **≥ 4** stars.

- **Precision@10** – Of the 10 recommended movies, what fraction are truly relevant?  
- **Recall@10** – Of all the relevant movies for this user in the test set, how many appear in the Top-10?  
- **HitRate@10** – For how many users does the Top-10 contain at least one relevant movie?  
- **MAP@10** – Averages precision at each position where a relevant item appears  
  (rewards putting good items earlier in the list).  
- **nDCG@10** – Measures how much “useful gain” the user gets from the ranking, taking position into account.

<div class="pill pill-section">Beyond accuracy: Novelty & Diversity</div>

To better approximate **user satisfaction**, I also computed:

- **Novelty (Average Popularity Rank)**  
  - Every movie gets a popularity rank based on how many ratings it has in the train set  
    (1 = most popular).  
  - For each Top-10 list, I compute the average rank; **higher average rank = more novel**.

- **Coverage**  
  - Fraction of the entire catalog that appears in at least one user’s Top-10 list.  
  - Higher coverage ⇒ the system uses more of the catalog instead of repeatedly recommending
    the same blockbusters.

- **Intra-List Diversity (ILD)**  
  - Uses genre metadata from MovieLens (e.g., Action, Comedy, Drama).  
  - For each Top-10 list, I compute the average pairwise **genre dissimilarity** between movies  
    (Jaccard dissimilarity over genre sets).  
  - Higher ILD ⇒ the list mixes different types of movies instead of ten nearly identical ones.

---

## Key Results (MovieLens-100K)

<div class="pill pill-section">Accuracy vs. Ranking</div>

- **SVD** and **item-based KNNBaseline** had similar or slightly better **RMSE/MAE** than user-based KNN.  
- But when I looked at **Top-10 ranking metrics**, SVD clearly outperformed the KNN models:
  higher **Precision@10**, **MAP@10**, and **nDCG@10**.  
- The **Popularity** baseline surprisingly scored very high on ranking metrics  
  (especially HitRate@10), illustrating that popularity alone can look “good” if we only
  track accuracy-like metrics—yet it offers **no personalization**.

<div class="pill pill-section">Novelty & Diversity</div>

- SVD’s Top-10 lists had **higher novelty** and **slightly higher diversity (ILD)** than the KNN models.  
- Popularity achieved strong ranking metrics but much **lower novelty**  
  (it keeps recommending the same famous movies to everyone).

---

## Scaling Up: MovieLens-1M

When moving to **MovieLens-1M**, the same patterns largely held:

- **Item-based KNNBaseline** achieved the best raw **RMSE/MAE**,  
  but **SVD continued to lead on nDCG@10** (ranking quality).  
- **Popularity** remained a very strong baseline for ranking,  
  but again with poor novelty and limited diversity.  
- **User-based KNNWithMeans** struggled on both accuracy and ranking metrics at this scale.

These results reinforce the main idea from the literature: **accuracy alone is not enough** to
judge how satisfying recommendations are.

---

## Hyperparameter Tuning for SVD (MovieLens-1M)

<div class="pill pill-section">Why tune?</div>

SVD has several key hyperparameters:

- `n_factors` – how many latent dimensions to learn  
- `n_epochs` – how many passes over the data  
- `reg_all` – how strongly we regularize to avoid overfitting  

Instead of fixing these values arbitrarily, I ran a small **grid search** to see how they affect
both **RMSE** and **Precision@10**:

- `n_factors ∈ {80, 120}`  
- `n_epochs ∈ {20, 30}`  
- `reg_all ∈ {0.02, 0.05}`  

For each configuration, I:

1. Trained an SVD model on the 80% training split.  
2. Evaluated **RMSE/MAE** on the 20% test split.  
3. Generated Top-10 recommendations from a **subsampled anti-testset**  
   (300,000 user–movie pairs for speed).  
4. Computed **Precision@10** and **nDCG@10**.

<div class="pill pill-section">What changed?</div>

- All configurations produced **similar RMSE** (around 0.86–0.88),  
  but **ranking quality varied slightly**.  
- Using **80 latent factors**, **20 epochs**, and **reg\_all = 0.02**    
  gave one of the best trade-offs: strong **Precision@10** and competitive **nDCG@10**  
  without extra training cost.  
- Increasing to **120 factors** did not consistently improve ranking, suggesting
  diminishing returns for added model complexity on this dataset.

These findings support the idea that **hyperparameter tuning should be guided by ranking metrics,
not just RMSE**, especially when the goal is user satisfaction.

---

## Takeaways

<div class="pill pill-section">What did we learn?</div>

- **SVD is the most promising personalized model** in this study:
  it balances good rating accuracy with better ranking, novelty, and diversity than the KNN variants.  
- **Popularity baselines can look very strong** on ranking metrics, but they lack personalization
  and novelty — a reminder that evaluation must align with human goals, not just numbers.  
- **Hyperparameter tuning matters**: small changes in SVD’s configuration can slightly
  improve ranking quality without hurting accuracy.

---

## Next Steps

- Explore **larger MovieLens datasets** (e.g., 10M) or **different domains** (music, e-commerce).  
- Incorporate **additional user-centric metrics**, such as serendipity or coverage by time.  
- Compare classical models with **modern deep learning recommenders** to see whether they
  offer better trade-offs between accuracy, novelty, and diversity.

---

## Code & Website

- 📂 GitHub repository:  
  <https://github.com/diyarahul/recsys-movielens-research>  
- 🌐 Project page (this site):  
  <https://diyarahul.github.io/recsys-movielens-research/>
