# 🎬 CineMatch AI

## Orbo.ai AI/ML Engineer Technical Assignment

CineMatch AI is an explainable hybrid movie recommendation system combining:

- Content-based filtering
- Collaborative filtering
- Hybrid ranking
- Diversity-aware reranking
- Personalized recommendations
- Similar-movie recommendations

## Problem Statement

A movie platform must rank a large catalog so users can quickly discover relevant movies. A practical recommender should combine movie characteristics with evidence from user behavior and avoid returning repetitive results.

## Use Case and Motivation

The project is inspired by streaming recommendation products such as Netflix. It is not intended to reproduce a proprietary production system; instead it demonstrates the engineering principles behind retrieval, ranking, personalization, explainability and evaluation.

## Recommendation Approach

### Content model

Movie title + genres are converted into TF-IDF vectors.

Cosine similarity measures content similarity.

### Collaborative model

Ratings form a sparse user × movie matrix.

Item-item cosine similarity captures movies that receive similar rating patterns from users.

### Hybrid model

The final score is:

`Final Score = α × Content Score + (1 − α) × Collaborative Score`

Default:

`α = 0.60`

### Diversity reranking

The top candidate pool is reranked using an MMR-style objective:

`MMR = (1 − λ) × relevance − λ × redundancy`

Redundancy uses genre Jaccard similarity. This keeps evaluation and production recommendation latency lightweight.

## Architecture

```text
                    Streamlit UI
                         |
                  User / Movie Input
                         |
                  CineMatch Engine
                         |
          +--------------+--------------+
          |                             |
     Content Model                Collaborative Model
     TF-IDF vectors               User-Movie matrix
          |                             |
     Content similarity            Item similarity
          +--------------+--------------+
                         |
                    Hybrid Score
                         |
                  Rated-item filter
                         |
                Diversity reranking
                         |
                    Top-K output
                         |
                    Explanation
```

## Dataset

Dataset: **MovieLens latest-small** from GroupLens Research.

The application automatically downloads:

https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Files used:

- `movies.csv`
- `ratings.csv`

The dataset is not committed to GitHub. It is downloaded when needed.

## Assumptions

1. Ratings >= 4.0 represent positive preference.
2. Historical ratings are a useful proxy for user preference.
3. Title and genre metadata are sufficient for a strong baseline.
4. Cosine similarity is suitable for sparse representations.
5. The assignment favors reproducibility and explainability.
6. MovieLens latest-small is suitable for development and demonstration, not production-scale claims.

## Key Design Decisions

### Why hybrid?

Content-based filtering helps with item similarity and cold-start content. Collaborative filtering captures collective user behavior. Combining them gives complementary signals.

### Why TF-IDF?

TF-IDF is fast, interpretable and reproducible.

### Why item-item collaborative filtering?

It directly supports a "movies similar to this" product experience and can also support personalization.

### Why diversity reranking?

A recommendation list containing ten nearly identical movies can feel poor. Reranking reduces repetitive genre patterns.

### Why Streamlit?

The assignment requires a deployed interactive interface. Streamlit provides a simple deployment path and makes the ML pipeline easy for evaluators to test.

## Technologies

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- TF-IDF
- Cosine similarity
- Streamlit
- Pytest
- Git/GitHub

## Evaluation

`evaluate.py` uses a leakage-aware leave-one-out protocol.

### Protocol

1. Select users with at least five positive ratings.
2. Select up to 100 users for a fast, reproducible benchmark.
3. Hold out one positive rating per selected user.
4. Remove those ratings from training data.
5. Rebuild the collaborative model using training data only.
6. Generate one top-10 recommendation list per user.
7. Compare recommendations with the hidden test item.
8. Reuse those same lists for coverage, diversity and latency metrics.

### Metrics

**Precision@10**

Fraction of the top-10 recommendations that are relevant.

**Recall@10**

Fraction of relevant held-out items recovered.

**NDCG@10**

Ranking-sensitive metric that rewards relevant items appearing earlier.

**Coverage@10**

Fraction of the movie catalog that appears across recommendation lists.

**Diversity@10**

One minus average pairwise genre similarity within recommendation lists.

**Latency**

Mean and P95 recommendation latency.

Run:

```bash
python evaluate.py
```

### Important evaluation limitation

This is an assignment-scale offline benchmark. A production benchmark should use a larger temporal train/test split and compare against strong baselines. Offline metrics should not be interpreted as real-world user satisfaction.

## Test Cases

### Successful scenarios

**TC-01 — Similar movie**

Input: `Toy Story (1995)`

Expected: related animation/family/children movies should rank strongly.

**TC-02 — Personalized recommendations**

Input: a MovieLens user with sufficient rating history.

Expected: recommendations reflect the user's high-rated movies.

**TC-03 — Rated-item filtering**

Expected: movies already rated by the user are excluded.

**TC-04 — Diversity**

Expected: increasing diversity strength reduces repetitive genre patterns.

### Failure scenarios

**TC-05 — Sparse user**

Very little rating history provides weak preference evidence.

**TC-06 — Cold-start movie**

A movie with limited collaborative information relies more heavily on content signals.

**TC-07 — Genre ambiguity**

Movies can share a genre while differing significantly in plot, tone and audience.

## Known Limitations

- MovieLens small has limited metadata.
- The content model does not understand plot semantics.
- There is no live feedback loop.
- There is no online retraining.
- Collaborative filtering is affected by sparse interactions.
- The benchmark is not a production A/B test.
- No images/posters are used by the ML model.

## Future Improvements

### Data

- MovieLens tags
- Plot descriptions
- Cast/director
- Keywords
- Release metadata

### ML

- Sentence-transformer embeddings
- Neural collaborative filtering
- Learning-to-rank
- Approximate nearest-neighbor search

### Product

- User accounts
- Session-based recommendations
- "Not interested" feedback
- Novelty controls
- Better recommendation explanations

### Production

- FastAPI model service
- Vector search
- Model registry
- Monitoring
- A/B testing
- Real-time feature pipelines

## Netflix-inspired Product Benchmark

### Similarities

- Personalized recommendations
- Similar-title recommendations
- Multi-signal ranking
- Preference modeling

### Differences

- Public small dataset
- No proprietary Netflix data
- Lightweight architecture
- No production-scale infrastructure

### Current limitations

- Limited metadata
- No session-level personalization
- No online experimentation

### What I would build next

A two-stage production system:

1. Candidate generation using content and collaborative retrieval.
2. Learning-to-rank model using user, movie and contextual features.

## Deployment

Recommended platform: Streamlit Community Cloud.

### Deployment steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect GitHub.
4. Select the repository and branch.
5. Select `app.py`.
6. Deploy.
7. Test the public URL.
8. Add the live URL to this README.

Before submission replace:

`LIVE DEMO: <ADD-YOUR-STREAMLIT-URL-HERE>`

with the actual URL.

## Local Setup — Windows

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python download_data.py

python -m pytest -q

python evaluate.py

streamlit run app.py
```

Open:

`http://localhost:8501`

## GitHub Submission Checklist

- [ ] Push source code
- [ ] Push requirements/configuration
- [ ] Keep dataset out of repository
- [ ] Verify README
- [ ] Run tests
- [ ] Run evaluation
- [ ] Record evaluation results
- [ ] Deploy Streamlit application
- [ ] Test public deployment
- [ ] Add live URL
- [ ] Submit GitHub URL + deployment URL + documentation

## Interview Explanation

> I built CineMatch AI as an explainable hybrid movie recommender. The content model uses TF-IDF over titles and genres, while the collaborative model uses item-item similarity from a sparse user-rating matrix. I normalize and combine both signals, filter already-rated movies, and apply diversity-aware reranking. I evaluated the system using a leakage-aware leave-one-out protocol with Precision@10, Recall@10 and NDCG@10, plus coverage, diversity and latency.

## Dataset Reference

MovieLens:
https://grouplens.org/datasets/movielens/

MovieLens download:
https://files.grouplens.org/datasets/movielens/ml-latest-small.zip


## UI Highlights

The Streamlit interface includes a product-style hero section, recommendation controls, hybrid-weight display, diversity control, latency metrics, ranked movie cards, genre badges, hybrid scores, explanations, personalized and similar-movie modes, and architecture/evaluation explainers.
