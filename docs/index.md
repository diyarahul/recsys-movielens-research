---
layout: default
title: Beyond Accuracy: Evaluating User Satisfaction in Movie Recommendations
---

<div class="hero-card">
  <p class="hero-kicker">Recommender Systems · MovieLens · User Satisfaction</p>
  <h1>Recommendations For You That Know <em>You</em></h1>
  <p class="hero-subtitle">
    How well do different recommender system models balance accuracy, ranking quality, novelty,
    and diversity when generating movie suggestions?
  </p>
  <p class="hero-meta"><strong>Student Researcher:</strong> Diya Rahul</p>
</div>

---

<div class="section-block">
  <div class="pill">Motivation</div>
  <h2>Why this project?</h2>
  <p>
    Recommender systems quietly decide what we watch, listen to, and buy — but they are usually
    judged by <strong>how close their predicted ratings are</strong> to the true ratings
    (metrics like RMSE or MAE). Prior work, including Silveira et&nbsp;al. (2019), argues that
    rating error is a <strong>weak proxy for user satisfaction</strong>.
  </p>
  <p>
    In reality, a satisfying recommender should also:
  </p>
  <ul>
    <li>Suggest items that are <strong>novel</strong> (not just the same blockbusters).</li>
    <li>Offer <strong>diverse</strong> lists (a mix of genres and styles).</li>
    <li>Rank the best items near the top of a <strong>Top-N list</strong>.</li>
  </ul>
  <p>
    This project puts these ideas into practice using the MovieLens datasets and compares
    several models not only on accuracy, but also on <strong>ranking quality, novelty, and diversity</strong>.
  </p>
</div>

<div class="section-block">
  <div class="pill">Data</div>
  <h2>MovieLens datasets</h2>
  <ul>
    <li>
      <strong>MovieLens 100K</strong> – 100,000 ratings from 943 users on 1,682 movies.
      Used for early experiments and debugging.
    </li>
    <li>
      <strong>MovieLens 1M</strong> – 1,000,209 ratings from 6,040 users on 3,900 movies.
      Used as the main dataset to test scalability and more realistic behavior.
    </li>
  </ul>
  <p>
    For each dataset I perform an <strong>80/20 train–test split</strong> using the Surprise
    library. The train set is what each model learns from; the test set contains ratings that
    the model never sees.
  </p>
  <p>
    To simulate “real” recommendation scenarios, I also build an <strong>anti-testset</strong>:
    all user–movie pairs that are not in the training data (unseen items). For MovieLens-1M this
    would be over 21 million pairs, so I <strong>subsample</strong> 300–500k pairs for ranking
    experiments to keep computations tractable.
  </p>
</div>

<div class="section-block">
  <div class="pill">Models</div>
  <h2>Recommender models compared</h2>
  <ul>
    <li>
      <strong>SVD (Matrix Factorization)</strong> – learns latent factors for each user and item,
      capturing hidden preferences such as “likes action movies” or “prefers slow dramas.”
    </li>
    <li>
      <strong>KNNBaseline (Item-based CF, Pearson baseline)</strong> – finds similar movies based
      on rating patterns, correcting for generous or harsh raters with baseline biases.
    </li>
    <li>
      <strong>KNNWithMeans (User-based CF, cosine)</strong> – finds users with similar taste and
      recommends the movies they liked, after subtracting each user’s mean rating.
    </li>
    <li>
      <strong>Popularity baseline</strong> – always recommends the most frequently rated movies.
      Very strong if we only care about accuracy; offers no personalization.
    </li>
    <li>
      <strong>Random baseline</strong> – recommends random unseen movies; a lower bound.
    </li>
  </ul>
</div>

<div class="section-block">
  <div class="pill">Evaluation</div>
  <h2>Accuracy metrics</h2>
  <p>
    On the held-out test split, I compute:
  </p>
  <ul>
    <li><strong>RMSE</strong> (Root Mean Squared Error) – penalizes large rating errors.</li>
    <li><strong>MAE</strong> (Mean Absolute Error) – average absolute difference between
    predicted and true ratings.</li>
  </ul>

  <h2>Ranking metrics (Top-10)</h2>
  <p>
    On the anti-testset, each model generates a <strong>Top-10 recommendation list</strong> per
    user. A movie is considered <strong>relevant</strong> if its true rating in the test set is
    at least 4 stars. I then compute:
  </p>
  <ul>
    <li><strong>Precision@10</strong> – fraction of the Top-10 that is truly relevant.</li>
    <li><strong>Recall@10</strong> – fraction of a user’s relevant movies that appear in the Top-10.</li>
    <li><strong>HitRate@10</strong> – how many users get at least one relevant movie.</li>
    <li><strong>MAP@10</strong> – rewards putting relevant items earlier in the list.</li>
    <li><strong>nDCG@10</strong> – overall ranking quality, taking position into account.</li>
  </ul>
</div>

<div class="section-block">
  <div class="pill">Beyond Accuracy</div>
  <h2>Novelty &amp; diversity metrics</h2>
  <ul>
    <li>
      <strong>Novelty</strong> – for each movie I compute a popularity rank based on how many
      ratings it has. Each Top-10 list gets an average rank; higher values mean more novel
      (less popular) recommendations.
    </li>
    <li>
      <strong>Coverage</strong> – fraction of the entire catalog that appears in at least one
      Top-10 across all users.
    </li>
    <li>
      <strong>Intra-List Diversity (ILD)</strong> – uses MovieLens genre metadata; for each
      Top-10 list I compute the average genre dissimilarity between pairs of movies.
      Higher ILD means a more varied list.
    </li>
  </ul>
</div>

<div class="section-block">
  <div class="pill">Results · ml-100k</div>
  <h2>Accuracy vs. ranking on MovieLens-100K</h2>
  <p>
    On the smaller ml-100k dataset, SVD and item-based KNNBaseline achieved similar (and strong)
    RMSE / MAE, while user-based KNNWithMeans lagged behind. When switching to ranking metrics,
    <strong>SVD clearly outperformed the KNN models</strong> on Precision@10, MAP@10, and nDCG@10.
  </p>

  <figure class="result-figure">
    <img src="{{ site.baseurl }}/assets/img/ranking_metrics_ml-100k.png"
         alt="Bar chart comparing Precision@10 and nDCG@10 for SVD, KNN models, Popularity, and Random on ml-100k">
    <figcaption>
      Precision@10 and nDCG@10 on MovieLens-100K. SVD provides better ranked lists than the
      KNN-based models, while Popularity is strong but non-personalized.
    </figcaption>
  </figure>
</div>

<div class="section-block">
  <div class="pill">Results · ml-1M</div>
  <h2>Scaling to MovieLens-1M</h2>
  <p>
    On the larger ml-1m dataset, item-based KNNBaseline achieved the best raw
    <strong>RMSE/MAE</strong>, but <strong>SVD continued to lead on nDCG@10</strong>.
    Popularity again performed surprisingly well in ranking metrics, but with low novelty and
    limited diversity, confirming that popularity alone is not enough for satisfying
    recommendations.
  </p>

  <figure class="result-figure">
    <img src="{{ site.baseurl }}/assets/img/ranking_metrics_ml-1m.png"
         alt="Bar chart comparing Precision@10 and nDCG@10 for SVD, KNN models, Popularity, and Random on ml-1m">
    <figcaption>
      Precision@10 and nDCG@10 on MovieLens-1M. SVD remains a strong personalized model even as
      the data scale increases.
    </figcaption>
  </figure>
</div>

<div class="section-block">
  <div class="pill">Hyperparameter Tuning</div>
  <h2>Fine-tuning SVD on MovieLens-1M</h2>
  <p>
    To understand how model choices affect both accuracy and ranking, I ran a small grid search
    over SVD hyperparameters:
  </p>
  <ul>
    <li><code>n_factors ∈ {80, 120}</code></li>
    <li><code>n_epochs ∈ {20, 30}</code></li>
    <li><code>reg_all ∈ {0.02, 0.05}</code></li>
  </ul>
  <p>
    For each configuration I measured both <strong>RMSE</strong> on the test split and
    <strong>Precision@10</strong> on a subsampled anti-testset (300k user–movie pairs).
  </p>

  <figure class="result-figure">
    <img src="{{ site.baseurl }}/assets/img/svd_tuning_rmse_vs_p@10_ml-1m.png"
         alt="Scatter plot of RMSE vs Precision@10 for different SVD hyperparameter settings on ml-1m">
    <figcaption>
      SVD tuning on MovieLens-1M: each point is a configuration. Changes in latent factors and
      regularization slightly shift the trade-off between test RMSE and Precision@10.
    </figcaption>
  </figure>

  <p>
    All configurations had similar RMSE (≈0.86–0.88), but their ranking quality varied.
    A configuration with <code>n_factors = 80</code>, <code>n_epochs = 20</code>, and
    <code>reg_all = 0.02</code> offered one of the best trade-offs: good test RMSE with
    competitive Precision@10 and nDCG@10, without extra training cost.
  </p>
</div>

<div class="section-block">
  <div class="pill">Takeaways</div>
  <h2>What does this say about user satisfaction?</h2>
  <ul>
    <li>
      <strong>SVD emerges as the best personalized model</strong> in this study, balancing
      decent accuracy with stronger ranking quality, novelty, and diversity than the KNN variants.
    </li>
    <li>
      <strong>Popularity baselines look strong on paper</strong> (high hit rate, high nDCG),
      but they repeatedly recommend the same movies and ignore personalization.
    </li>
    <li>
      <strong>Hyperparameter tuning needs to consider ranking metrics</strong> and
      beyond-accuracy measures, not just RMSE, if we care about real user satisfaction.
    </li>
  </ul>
</div>

<div class="section-block">
  <div class="pill">Future Work</div>
  <h2>Next steps</h2>
  <ul>
    <li>Extend the analysis to even larger datasets (e.g., MovieLens-10M) or new domains.</li>
    <li>Incorporate serendipity and temporal dynamics (e.g., recency) as additional metrics.</li>
    <li>Compare classical models with modern deep-learning recommenders using the same evaluation pipeline.</li>
  </ul>
</div>

<div class="section-block">
  <div class="pill">Code</div>
  <h2>Repository</h2>
  <p>
    Full code for the experiments, including evaluation scripts and plotting, is available on GitHub:
  </p>
  <p>
    <a href="https://github.com/diyarahul/recsys-movielens-research" target="_blank">
      github.com/diyarahul/recsys-movielens-research
    </a>
  </p>
</div>
