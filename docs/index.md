---
title: Beyond Accuracy: Evaluating User Satisfaction in Movie Recommendation Systems
---

<nav class="top-nav">
  <div class="nav-title">Beyond Accuracy in Recommender Systems</div>
  <div class="nav-links">
    <a href="#motivation">Motivation</a>
    <a href="#methods">Methods</a>
    <a href="#results">Results</a>
    <a href="#future-work">Future Work</a>
    <a href="https://github.com/diyarahul/recsys-movielens-research" target="_blank">Code</a>
  </div>
</nav>

<section class="hero">
  <div class="hero-content">
    <h1 class="hero-title">Recommendations For You <span class="italic">That Know <em>You</em></span></h1>
    <p class="hero-subtitle">
      Exploring how different recommender models balance accuracy, ranking quality,
      novelty, and diversity using the MovieLens datasets.
    </p>
    <p class="hero-meta">
      <strong>Student Researcher:</strong> Diya Rahul<br/>
      <strong>Data:</strong> MovieLens 100K &amp; 1M<br/>
      <strong>Models:</strong> Collaborative Filtering (KNN), SVD (matrix factorization), baselines
    </p>
  </div>
</section>

<div class="page-wrapper">

<section id="motivation" class="card">
  <div class="pill-label">Motivation</div>
  <h2>Why this project?</h2>
  <p>
    Recommender systems quietly decide what we watch, listen to, and buy — but they are still
    mostly judged by how close they get to our original ratings (metrics like RMSE or MAE).
    Prior work (for example, Silveira et&nbsp;al., 2019) argues that rating error is a weak proxy
    for user satisfaction.
  </p>
  <p>
    In real life, a “good” recommendation is not just <em>accurate</em>. It should also feel
    <strong>novel</strong> (not something we already know), <strong>diverse</strong> (not ten
    nearly identical items), and still <strong>highly ranked</strong> in a Top-N list.
  </p>
  <p>
    This project puts those ideas into practice by comparing models on MovieLens data and by
    asking: <strong>Which models give recommendations that are not only accurate, but also more
    varied and surprising?</strong>
  </p>
</section>

<section id="methods" class="card">
  <div class="pill-label">Methods</div>
  <h2>How did I study this?</h2>

  <h3>1. Data</h3>
  <ul>
    <li><strong>MovieLens 100K &amp; 1M</strong> (user–movie ratings on a 1–5 scale).</li>
    <li>Each dataset is randomly split into <strong>80% train</strong> and <strong>20% test</strong>.</li>
    <li>The <em>anti-testset</em> is all user–movie pairs that the user has <strong>never rated</strong>, which simulates the “unknown” catalog a recommender suggests from.</li>
  </ul>

  <h3>2. Models compared</h3>
  <ul>
    <li><strong>SVD (matrix factorization)</strong>: learns hidden “taste” factors from the ratings.</li>
    <li><strong>KNNBaseline (item-based CF)</strong>: recommends items similar to what the user liked, with bias correction.</li>
    <li><strong>KNNWithMeans (user-based CF)</strong>: finds similar users first, then recommends what they liked.</li>
    <li><strong>Popularity baseline</strong>: always recommends the most popular movies.</li>
    <li><strong>Random baseline</strong>: recommends random unseen movies (sanity check).</li>
  </ul>

  <h3>3. Evaluation</h3>
  <p><strong>Accuracy (ratings):</strong></p>
  <ul>
    <li><strong>RMSE, MAE</strong> on the 20% held-out test set.</li>
  </ul>

  <p><strong>Ranking quality (Top-10 recommendations):</strong></p>
  <ul>
    <li><strong>Precision@10</strong>: Of the 10 suggested movies, how many are truly relevant (test rating ≥ 4)?</li>
    <li><strong>nDCG@10</strong>: Gives higher credit when relevant movies appear near the top of the list.</li>
    <li><strong>Recall@10, MAP@10, HitRate@10</strong> were also computed in the full experiments.</li>
  </ul>

  <p><strong>Beyond accuracy (user satisfaction proxies):</strong></p>
  <ul>
    <li><strong>Novelty</strong>: average popularity rank of recommended movies
      (higher rank = less popular = more novel).</li>
    <li><strong>Intra-List Diversity (ILD)</strong>: how different the genres are within each user’s Top-10 list.</li>
    <li><strong>Coverage</strong>: fraction of the catalog that appears in at least one user’s Top-10 list.</li>
  </ul>
</section>

<section id="results" class="card">
  <div class="pill-label">Results</div>
  <h2>What did I find?</h2>

  <h3>1. Rating accuracy alone is misleading</h3>
  <p>
    On both datasets, SVD and item-based KNN achieve lower RMSE/MAE than user-based KNN,
    meaning they predict ratings more accurately. However, when I look at what users
    actually see in their Top-10 lists, accuracy is only part of the story.
  </p>

  <div class="figure-row">
    <figure>
      <img src="assets/img/ranking_metrics_ml-100k.png" alt="Ranking metrics on MovieLens 100K" />
      <figcaption>Precision@10 and nDCG@10 on MovieLens 100K.</figcaption>
    </figure>
    <figure>
      <img src="assets/img/ranking_metrics_ml-1m.png" alt="Ranking metrics on MovieLens 1M" />
      <figcaption>Precision@10 and nDCG@10 on MovieLens 1M.</figcaption>
    </figure>
  </div>

  <p>
    The plots show that:
  </p>
  <ul>
    <li><strong>Popularity</strong> has the highest Precision@10 and nDCG@10, because popular movies are often relevant to many users.</li>
    <li><strong>SVD</strong> consistently beats the KNN models on ranking metrics while also having strong rating accuracy.</li>
    <li><strong>Random</strong> behaves as expected: very low precision and ranking quality.</li>
  </ul>

  <h3>2. Looking beyond accuracy</h3>
  <p>
    When I compute novelty and diversity on Top-10 lists, SVD tends to recommend
    <strong>slightly more varied and less popular</strong> movies than the baselines,
    without collapsing into purely niche content.
  </p>

  <div class="figure-row">
    <figure>
      <img src="assets/img/anti_mean_ild_by_model.png" alt="Mean ILD comparison" />
      <figcaption>Mean ILD (genre diversity) across models.</figcaption>
    </figure>
    <figure>
      <img src="assets/img/cf_svd_rmse_novelty_comparison.png" alt="Accuracy vs Novelty" />
      <figcaption>Per-user trade-off between RMSE and novelty.</figcaption>
    </figure>
  </div>

  <p>
    Overall, the results suggest that
    <strong>SVD offers a better balance</strong>: it is competitive in accuracy,
    gives reasonable ranking quality, and produces recommendations that are
    both somewhat novel and diverse.
  </p>
</section>

<section id="future-work" class="card">
  <div class="pill-label">Future Work</div>
  <h2>Where can this go next?</h2>
  <ul>
    <li><strong>Include more modern models</strong> such as neural recommenders or hybrid models that mix content and collaborative features.</li>
    <li><strong>Use implicit feedback</strong> (clicks, watch time) instead of relying only on explicit ratings.</li>
    <li><strong>Collect real user feedback</strong> to connect these metrics (accuracy, novelty, ILD) to perceived satisfaction.</li>
    <li><strong>Study fairness and exposure</strong>: who benefits from “popular” recommendations, and who is ignored?</li>
  </ul>
</section>

<section class="card">
  <h2>Repository</h2>
  <p>
    All code for data loading, model training, evaluation, and plotting is available on GitHub:
    <a href="https://github.com/diyarahul/recsys-movielens-research" target="_blank">
      diyarahul/recsys-movielens-research
    </a>.
  </p>
</section>

</div>
