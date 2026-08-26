# CineMatch AI — Final Submission Checklist

## Local verification

- [ ] `pip install -r requirements.txt`
- [ ] `python download_data.py`
- [ ] `python -m pytest -q`
- [ ] `python evaluate.py`
- [ ] `streamlit run app.py`
- [ ] Test Similar Movies
- [ ] Test Personalized For User
- [ ] Test content weight
- [ ] Test diversity strength
- [ ] Test rated-item filtering

## Evaluation

- [ ] Record Precision@10
- [ ] Record Recall@10
- [ ] Record NDCG@10
- [ ] Record Coverage@10
- [ ] Record Diversity@10
- [ ] Record Mean latency
- [ ] Record P95 latency

## GitHub

- [ ] Create public/accesssible repository
- [ ] Push source code
- [ ] Push README
- [ ] Push requirements.txt
- [ ] Do not push `data/ml-latest-small/`
- [ ] Verify README renders

## Deployment

- [ ] Deploy `app.py`
- [ ] Open public URL
- [ ] Test in an incognito browser
- [ ] Verify recommendations work
- [ ] Add live URL to README

## Final submission

- [ ] GitHub repository URL
- [ ] Deployment URL
- [ ] Documentation
