import re
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import load_data


def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else ""


class CineMatchEngine:

    def __init__(self):

        self.movies, self.ratings = load_data()

        self.movies = self.movies.reset_index(drop=True)

        self.movie_ids = self.movies["movieId"].tolist()

        self.movie_id_to_idx = {
            movie_id: index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.movie_titles = self.movies["title"].tolist()

        self.default_movie_index = (
            self.movie_titles.index("Toy Story (1995)")
            if "Toy Story (1995)" in self.movie_titles
            else 0
        )

        self._build_content_model()
        self._build_collaborative_model()

    # -----------------------------------------------------
    # Content model
    # -----------------------------------------------------

    def _build_content_model(self):

        text = (
            self.movies["title"]
            .str.replace(r"[^\w\s]", " ", regex=True)
            + " "
            + self.movies["genres"]
            .str.replace("|", " ", regex=False)
            + " "
            + self.movies["genres"]
            .str.replace("|", " ", regex=False)
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000
        )

        self.content_matrix = (
            self.vectorizer.fit_transform(text)
        )

    # -----------------------------------------------------
    # Collaborative model
    # -----------------------------------------------------

    def _build_collaborative_model(self):

        users = pd.Index(
            self.ratings["userId"].unique()
        )

        user_to_idx = {
            user_id: index
            for index, user_id in enumerate(users)
        }

        movie_columns = (
            self.ratings["movieId"]
            .map(self.movie_id_to_idx)
        )

        valid = movie_columns.notna().to_numpy()

        rows = (
            self.ratings["userId"]
            .map(user_to_idx)
            .to_numpy()[valid]
        )

        columns = (
            movie_columns[valid]
            .astype(int)
            .to_numpy()
        )

        values = (
            self.ratings["rating"]
            .to_numpy(dtype=float)[valid]
        )

        self.user_ids = users.tolist()

        self.user_to_idx = user_to_idx

        self.user_item_matrix = csr_matrix(
            (
                values,
                (rows, columns)
            ),
            shape=(
                len(users),
                len(self.movie_ids)
            ),
            dtype=np.float32
        )

        self.item_similarity = cosine_similarity(
            self.user_item_matrix.T
        )

        np.fill_diagonal(
            self.item_similarity,
            0.0
        )

        popularity = (
            self.ratings
            .groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                mean_rating=("rating", "mean")
            )
            .reindex(self.movie_ids)
            .fillna(0)
        )

        global_mean = (
            popularity.loc[
                popularity["mean_rating"] > 0,
                "mean_rating"
            ].mean()
        )

        minimum_votes = popularity[
            "rating_count"
        ].quantile(0.60)

        popularity["bayesian_score"] = (
            (
                popularity["rating_count"]
                / (
                    popularity["rating_count"]
                    + minimum_votes
                )
            )
            * popularity["mean_rating"]
            +
            (
                minimum_votes
                / (
                    popularity["rating_count"]
                    + minimum_votes
                )
            )
            * global_mean
        ).fillna(global_mean)

        self.popularity = popularity

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------

    @staticmethod
    def normalize(scores):

        scores = np.nan_to_num(
            np.asarray(scores, dtype=float)
        )

        minimum = scores.min()
        maximum = scores.max()

        if maximum - minimum < 1e-12:
            return np.zeros_like(scores)

        return (
            (scores - minimum)
            / (maximum - minimum)
        )

    def _genre_similarity(self, first_index, second_index):

        first = set(
            self.movies.iloc[first_index]["genres"].split("|")
        )

        second = set(
            self.movies.iloc[second_index]["genres"].split("|")
        )

        first.discard("(no genres listed)")
        second.discard("(no genres listed)")

        if not first and not second:
            return 0.0

        union = first | second

        if not union:
            return 0.0

        return len(first & second) / len(union)

    def _diversity_rerank(
        self,
        candidates,
        scores,
        n,
        diversity_strength
    ):
        """
        Efficient MMR-style reranking.

        Redundancy uses genre Jaccard similarity rather than
        repeatedly multiplying sparse TF-IDF vectors. This is
        much faster during repeated offline evaluation while
        remaining easy to explain.
        """

        selected = []
        remaining = list(candidates)

        while remaining and len(selected) < n:

            best_candidate = None
            best_value = -float("inf")

            for candidate in remaining:

                relevance = scores[candidate]

                if selected:

                    redundancy = max(
                        self._genre_similarity(
                            candidate,
                            previous
                        )
                        for previous in selected
                    )

                else:
                    redundancy = 0.0

                value = (
                    (1 - diversity_strength)
                    * relevance
                    -
                    diversity_strength
                    * redundancy
                )

                if value > best_value:
                    best_value = value
                    best_candidate = candidate

            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected

    def _result_frame(
        self,
        indices,
        scores,
        reasons
    ):

        rows = []

        for index in indices:

            movie = self.movies.iloc[index]

            rows.append(
                {
                    "movieId": int(movie["movieId"]),
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "year": extract_year(movie["title"]),
                    "score": float(scores[index]),
                    "reason": reasons[index]
                }
            )

        return pd.DataFrame(rows)

    # -----------------------------------------------------
    # Similar movies
    # -----------------------------------------------------

    def similar_movies(
        self,
        title,
        n=10,
        content_weight=0.60,
        diversity_strength=0.20
    ):

        movie_index = self.movie_titles.index(title)

        content_scores = cosine_similarity(
            self.content_matrix[movie_index],
            self.content_matrix
        ).ravel()

        collaborative_scores = (
            self.item_similarity[movie_index]
        )

        content_scores = self.normalize(
            content_scores
        )

        collaborative_scores = self.normalize(
            collaborative_scores
        )

        final_scores = (
            content_weight * content_scores
            +
            (1 - content_weight)
            * collaborative_scores
        )

        final_scores[movie_index] = -1

        candidate_count = min(
            80,
            len(final_scores)
        )

        candidates = np.argsort(
            -final_scores
        )[:candidate_count]

        selected = self._diversity_rerank(
            candidates,
            final_scores,
            n,
            diversity_strength
        )

        base_genres = set(
            self.movies.iloc[movie_index]
            ["genres"]
            .split("|")
        )

        reasons = {}

        for index in selected:

            recommendation_genres = set(
                self.movies.iloc[index]
                ["genres"]
                .split("|")
            )

            overlap = sorted(
                (
                    base_genres
                    & recommendation_genres
                )
                - {"(no genres listed)"}
            )

            reason = (
                f"{content_weight:.0%} content + "
                f"{(1-content_weight):.0%} "
                "collaborative score."
            )

            if overlap:

                reason += (
                    " Shared genres: "
                    + ", ".join(overlap[:3])
                    + "."
                )

            reasons[index] = reason

        return self._result_frame(
            selected,
            final_scores,
            reasons
        )

    # -----------------------------------------------------
    # User history
    # -----------------------------------------------------

    def user_history(self, user_id):

        history = self.ratings[
            self.ratings["userId"] == user_id
        ].copy()

        history = history.merge(
            self.movies[
                ["movieId", "title"]
            ],
            on="movieId",
            how="left"
        )

        return history.sort_values(
            ["rating", "timestamp"],
            ascending=[False, False]
        )

    # -----------------------------------------------------
    # Personalized recommendations
    # -----------------------------------------------------

    def personalized_recommendations(
        self,
        user_id,
        n=10,
        content_weight=0.60,
        diversity_strength=0.20
    ):

        user_index = self.user_to_idx[user_id]

        ratings_vector = (
            self.user_item_matrix[
                user_index
            ]
            .toarray()
            .ravel()
        )

        rated_mask = ratings_vector > 0

        liked_mask = ratings_vector >= 4.0

        if liked_mask.sum() == 0:
            liked_mask = rated_mask

        # Cold-start fallback
        if liked_mask.sum() == 0:

            final_scores = self.normalize(
                self.popularity[
                    "bayesian_score"
                ].to_numpy()
            )

            final_scores[rated_mask] = -1

            candidates = np.argsort(
                -final_scores
            )[:n]

            reasons = {
                index:
                    "Popularity fallback because "
                    "this user has insufficient "
                    "rating history."
                for index in candidates
            }

            return self._result_frame(
                candidates,
                final_scores,
                reasons
            )

        liked_indices = np.flatnonzero(
            liked_mask
        )

        weights = np.maximum(
            ratings_vector[liked_indices] - 2.5,
            0.1
        )

        user_content_profile = np.asarray(
            self.content_matrix[
                liked_mask
            ]
            .multiply(
                weights[:, None]
            )
            .sum(axis=0)
        )

        content_scores = cosine_similarity(
            user_content_profile,
            self.content_matrix
        ).ravel()

        collaborative_scores = np.average(
            self.item_similarity[
                liked_indices
            ],
            axis=0,
            weights=weights
        )

        content_scores = self.normalize(
            content_scores
        )

        collaborative_scores = self.normalize(
            collaborative_scores
        )

        final_scores = (
            content_weight * content_scores
            +
            (1 - content_weight)
            * collaborative_scores
        )

        # Never recommend something already rated.
        final_scores[rated_mask] = -1

        candidate_count = min(
            80,
            len(final_scores)
        )

        candidates = np.argsort(
            -final_scores
        )[:candidate_count]

        selected = self._diversity_rerank(
            candidates,
            final_scores,
            n,
            diversity_strength
        )

        preferred_genres = set()

        for genres in self.movies.iloc[
            liked_indices
        ]["genres"]:

            preferred_genres.update(
                genres.split("|")
            )

        reasons = {}

        for index in selected:

            recommendation_genres = set(
                self.movies.iloc[index]
                ["genres"]
                .split("|")
            )

            overlap = sorted(
                preferred_genres
                & recommendation_genres
            )

            reason = (
                "Based on highly rated movies; "
                f"{content_weight:.0%} content + "
                f"{(1-content_weight):.0%} "
                "collaborative signals."
            )

            if overlap:

                reason += (
                    " Genre affinity: "
                    + ", ".join(overlap[:3])
                    + "."
                )

            reasons[index] = reason

        return self._result_frame(
            selected,
            final_scores,
            reasons
        )


def precision_recall_ndcg_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = list(recommended)[:k]
    relevant = set(relevant)

    hits = [
        1 if item in relevant else 0
        for item in recommended
    ]

    precision = sum(hits) / k

    recall = (
        sum(hits) / len(relevant)
        if relevant
        else 0.0
    )

    dcg = sum(
        hit / np.log2(position + 2)
        for position, hit
        in enumerate(hits)
    )

    ideal_hits = min(
        len(relevant),
        k
    )

    idcg = sum(
        1 / np.log2(position + 2)
        for position in range(ideal_hits)
    )

    ndcg = (
        dcg / idcg
        if idcg
        else 0.0
    )

    return precision, recall, ndcg

