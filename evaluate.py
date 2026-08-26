"""
CineMatch AI - Fast, leakage-aware offline evaluation.

Protocol:
1. Select up to 100 users with >= 5 positive ratings.
2. Hold out one positive rating per user.
3. Remove those ratings from the training data.
4. Rebuild the collaborative model using training data only.
5. Generate one top-10 recommendation list per user.
6. Evaluate Precision@10, Recall@10 and NDCG@10.
7. Reuse the generated lists to calculate coverage, diversity and latency.

This is an assignment-scale offline benchmark, not a production A/B test.
"""

import random
import time

import numpy as np

from src.recommender import (
    CineMatchEngine,
    precision_recall_ndcg_at_k,
)


SEED = 42
MAX_USERS = 100
TOP_K = 10


random.seed(SEED)
np.random.seed(SEED)


print("Loading CineMatch AI...")
engine = CineMatchEngine()


# ---------------------------------------------------------
# 1. Select evaluation users and hold out one positive item
# ---------------------------------------------------------

original_ratings = engine.ratings.copy()

held_out = {}

eligible_users = []

for user_id in engine.user_ids:

    user_ratings = original_ratings[
        original_ratings["userId"] == user_id
    ]

    positive_ratings = user_ratings[
        user_ratings["rating"] >= 4.0
    ]

    if len(positive_ratings) >= 5:
        eligible_users.append(user_id)


# Deterministic sample so every run is comparable.
eligible_users = sorted(eligible_users)

random.Random(SEED).shuffle(
    eligible_users
)

eligible_users = eligible_users[
    :MAX_USERS
]


for user_id in eligible_users:

    positive_ratings = original_ratings[
        (original_ratings["userId"] == user_id)
        &
        (original_ratings["rating"] >= 4.0)
    ]

    held_out_row = positive_ratings.sample(
        n=1,
        random_state=SEED
    ).iloc[0]

    held_out[user_id] = int(
        held_out_row["movieId"]
    )


# ---------------------------------------------------------
# 2. Build training ratings
# ---------------------------------------------------------

train_ratings = original_ratings.copy()

holdout_pairs = set(
    held_out.items()
)

remove_mask = [
    (user_id, movie_id)
    in holdout_pairs
    for user_id, movie_id
    in zip(
        train_ratings["userId"],
        train_ratings["movieId"]
    )
]

train_ratings = train_ratings.loc[
    ~np.array(remove_mask)
].copy()


# ---------------------------------------------------------
# 3. Rebuild collaborative model with TRAINING only
# ---------------------------------------------------------

engine.ratings = train_ratings

engine._build_collaborative_model()


# ---------------------------------------------------------
# 4. Generate each user's recommendations ONCE
# ---------------------------------------------------------

metrics = []

all_recommended_movies = set()

latencies = []

recommendation_lists = []


print(
    f"Evaluating {len(held_out)} users..."
)


for counter, (user_id, hidden_movie) in enumerate(
    held_out.items(),
    start=1
):

    start = time.perf_counter()

    recommendations = (
        engine.personalized_recommendations(
            user_id=user_id,
            n=TOP_K,
            content_weight=0.60,
            diversity_strength=0.20
        )
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    recommended_movies = (
        recommendations["movieId"]
        .tolist()
    )

    precision, recall, ndcg = (
        precision_recall_ndcg_at_k(
            recommended_movies,
            [hidden_movie],
            k=TOP_K
        )
    )

    metrics.append(
        (precision, recall, ndcg)
    )

    all_recommended_movies.update(
        recommended_movies
    )

    latencies.append(
        latency_ms
    )

    recommendation_lists.append(
        recommendations
    )

    # Progress indicator
    if counter % 10 == 0:
        print(
            f"  Evaluated {counter}/{len(held_out)} users..."
        )


# ---------------------------------------------------------
# 5. Ranking metrics
# ---------------------------------------------------------

metrics = np.asarray(metrics)

precision_at_10 = metrics[:, 0].mean()
recall_at_10 = metrics[:, 1].mean()
ndcg_at_10 = metrics[:, 2].mean()


# ---------------------------------------------------------
# 6. Catalog coverage
# ---------------------------------------------------------

catalog_size = len(
    engine.movies
)

coverage_at_10 = (
    len(all_recommended_movies)
    / catalog_size
)


# ---------------------------------------------------------
# 7. Diversity
# ---------------------------------------------------------

# Reuse the recommendation lists generated above.
# No second recommendation pass is needed.

diversity_scores = []

for recommendations in recommendation_lists:

    indices = [
        engine.movie_id_to_idx[
            movie_id
        ]
        for movie_id
        in recommendations["movieId"]
    ]

    if len(indices) < 2:
        continue

    similarities = []

    for i in range(len(indices)):

        for j in range(i + 1, len(indices)):

            similarity = (
                engine._genre_similarity(
                    indices[i],
                    indices[j]
                )
            )

            similarities.append(
                similarity
            )

    if similarities:

        diversity_scores.append(
            1 - np.mean(similarities)
        )


diversity_at_10 = (
    np.mean(diversity_scores)
    if diversity_scores
    else 0.0
)


# ---------------------------------------------------------
# 8. Latency
# ---------------------------------------------------------

mean_latency = np.mean(
    latencies
)

p95_latency = np.percentile(
    latencies,
    95
)


# ---------------------------------------------------------
# 9. Print final report
# ---------------------------------------------------------

print()
print("=" * 50)
print("          CineMatch AI Evaluation")
print("=" * 50)

print(
    f"Users evaluated : {len(metrics)}"
)

print(
    f"Precision@10    : {precision_at_10:.4f}"
)

print(
    f"Recall@10       : {recall_at_10:.4f}"
)

print(
    f"NDCG@10         : {ndcg_at_10:.4f}"
)

print(
    f"Coverage@10     : {coverage_at_10:.4f}"
)

print(
    f"Diversity@10    : {diversity_at_10:.4f}"
)

print(
    f"Mean latency ms : {mean_latency:.2f}"
)

print(
    f"P95 latency ms  : {p95_latency:.2f}"
)

print("=" * 50)

print()
print(
    "Evaluation complete."
)
