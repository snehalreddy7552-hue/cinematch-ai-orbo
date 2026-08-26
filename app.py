import html
import time
import streamlit as st
from src.recommender import CineMatchEngine

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

@st.cache_resource(show_spinner="Loading CineMatch AI models…")
def load_engine():
    return CineMatchEngine()

engine = load_engine()

st.markdown("""
<style>
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1400px;}
.hero {padding:2rem 2.2rem; border-radius:22px; color:white; margin-bottom:1.5rem; background:radial-gradient(circle at 85% 20%,rgba(108,92,231,.28),transparent 32%), linear-gradient(135deg,#111827 0%,#24144d 52%,#4c1d95 100%);}
.hero h1 {font-size:3rem;margin:0;letter-spacing:-1px;}
.hero p {color:#e9e7ff;font-size:1.08rem;margin:.55rem 0 0;max-width:820px;}
.pill {display:inline-block;padding:.3rem .7rem;border-radius:999px;background:rgba(255,255,255,.14);color:white;margin:.8rem .35rem 0 0;font-size:.82rem;}
.section-title {font-size:1.65rem;font-weight:700;margin:.6rem 0 .8rem;}
.movie-card {border:1px solid #e5e7eb;border-radius:18px;padding:1.05rem;margin-bottom:.9rem;background:white;box-shadow:0 3px 14px rgba(15,23,42,.06);min-height:170px;}
.movie-rank {font-size:.78rem;font-weight:700;color:#6c5ce7;text-transform:uppercase;letter-spacing:.08em;}
.movie-title {font-size:1.22rem;font-weight:750;margin:.18rem 0 .3rem;}
.movie-meta {color:#667085;font-size:.9rem;margin-bottom:.7rem;}
.genre-badge {display:inline-block;padding:.22rem .5rem;margin:.1rem .2rem .15rem 0;border-radius:999px;background:#f1efff;color:#5142b8;font-size:.74rem;}
.score-box {display:inline-block;padding:.35rem .55rem;border-radius:8px;background:#f4f4f5;font-weight:700;font-size:.85rem;}
.reason {margin-top:.7rem;color:#344054;line-height:1.45;}
.metric {border:1px solid #e5e7eb;border-radius:14px;padding:1rem;background:#fafafa;}
.metric-label {color:#667085;font-size:.8rem;}
.metric-value {font-size:1.35rem;font-weight:750;margin-top:.15rem;}
.footer {color:#98a2b3;text-align:center;font-size:.82rem;margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

def genre_badges(genres):
    vals=[g for g in str(genres).split("|") if g and g!="(no genres listed)"]
    if not vals: return '<span class="genre-badge">Genre unavailable</span>'
    return "".join(f'<span class="genre-badge">{html.escape(g)}</span>' for g in vals[:6])

def render_movie_card(rank,row):
    st.markdown(f"""
    <div class="movie-card">
      <div class="movie-rank">#{rank} Recommendation</div>
      <div class="movie-title">🎬 {html.escape(str(row["title"]))}</div>
      <div class="movie-meta">{genre_badges(row["genres"])}</div>
      <span class="score-box">Hybrid score: {float(row["score"]):.3f}</span>
      <div class="reason">🧠 {html.escape(str(row["reason"]))}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🎬 CineMatch AI</h1>
<p>Discover movies using a hybrid recommendation engine that combines movie content, collaborative behavior and diversity-aware ranking.</p>
<span class="pill">TF-IDF Content</span><span class="pill">Collaborative Filtering</span><span class="pill">Hybrid Ranking</span><span class="pill">Explainable AI</span><span class="pill">Diversity Reranking</span>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Recommendation Controls")
    mode=st.radio("Recommendation mode",["Similar Movies","Personalized For User"])
    content_weight=st.slider("Content weight",0.0,1.0,0.60,0.05,help="Higher values give more importance to movie content.")
    diversity_strength=st.slider("Diversity strength",0.0,1.0,0.20,0.05,help="Higher values penalize repetitive genres.")
    n=st.slider("Number of recommendations",5,15,10)
    st.divider()
    st.caption(f"Content signal: **{content_weight:.0%}**")
    st.caption(f"Collaborative signal: **{1-content_weight:.0%}**")
    st.info("Increase diversity for a broader mix of genres.")

if mode=="Similar Movies":
    st.markdown('<div class="section-title">🔎 Find movies similar to a movie you like</div>',unsafe_allow_html=True)
    title=st.selectbox("Choose a movie",engine.movie_titles,index=engine.default_movie_index)
    c1,c2=st.columns([4,1])
    with c1: st.caption("The model compares movie content and collaborative similarity.")
    with c2: generate=st.button("✨ Recommend",type="primary",use_container_width=True)
    if generate:
        start=time.perf_counter()
        recs=engine.similar_movies(title=title,n=n,content_weight=content_weight,diversity_strength=diversity_strength)
        latency=(time.perf_counter()-start)*1000
        st.success(f"Recommendations inspired by **{title}**")
        m1,m2,m3=st.columns(3)
        with m1: st.markdown(f'<div class="metric"><div class="metric-label">CONTENT</div><div class="metric-value">{content_weight:.0%}</div></div>',unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric"><div class="metric-label">COLLABORATIVE</div><div class="metric-value">{1-content_weight:.0%}</div></div>',unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric"><div class="metric-label">LATENCY</div><div class="metric-value">{latency:.0f} ms</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">🍿 Recommended Movies</div>',unsafe_allow_html=True)
        for rank,(_,row) in enumerate(recs.iterrows(),1): render_movie_card(rank,row)
else:
    st.markdown('<div class="section-title">🎯 Personalized recommendations</div>',unsafe_allow_html=True)
    user_id=st.selectbox("Choose a MovieLens user",engine.user_ids,format_func=lambda x:f"User {x}")
    history=engine.user_history(user_id)
    left,right=st.columns([2,1])
    with left:
        st.caption("Highly rated movies from this user")
        h=history[["title","rating"]].head(8).copy()
        h["rating"]=h["rating"].map(lambda x:f"⭐ {x:.1f}")
        st.dataframe(h,use_container_width=True,hide_index=True)
    with right:
        st.markdown('<div class="metric"><div class="metric-label">USER PROFILE</div><div class="metric-value">Personalized</div><p style="color:#667085">Recommendations are inferred from historical ratings.</p></div>',unsafe_allow_html=True)
    generate=st.button("🎯 Recommend For This User",type="primary",use_container_width=True)
    if generate:
        start=time.perf_counter()
        recs=engine.personalized_recommendations(user_id=user_id,n=n,content_weight=content_weight,diversity_strength=diversity_strength)
        latency=(time.perf_counter()-start)*1000
        st.success(f"Personalized recommendations for **User {user_id}**")
        m1,m2,m3=st.columns(3)
        with m1: st.markdown(f'<div class="metric"><div class="metric-label">CONTENT</div><div class="metric-value">{content_weight:.0%}</div></div>',unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric"><div class="metric-label">COLLABORATIVE</div><div class="metric-value">{1-content_weight:.0%}</div></div>',unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric"><div class="metric-label">LATENCY</div><div class="metric-value">{latency:.0f} ms</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="section-title">🍿 Your Recommendations</div>',unsafe_allow_html=True)
        for rank,(_,row) in enumerate(recs.iterrows(),1): render_movie_card(rank,row)

st.divider()
st.markdown('<div class="section-title">🧠 How CineMatch AI works</div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
with a: st.markdown("### 1. Content"); st.write("TF-IDF represents movie titles and genres.")
with b: st.markdown("### 2. Collaborative"); st.write("User ratings create item-item similarity.")
with c: st.markdown("### 3. Hybrid"); st.write("Normalized signals are combined using the selected weights.")
with d: st.markdown("### 4. Diversity"); st.write("MMR-style reranking reduces repetitive recommendations.")
with st.expander("🔍 Why was this recommended?"): st.write("Each result includes its hybrid score and genre affinity so the recommendation is interpretable.")
with st.expander("📊 Model evaluation"): st.write("Run evaluate.py for Precision@10, Recall@10, NDCG@10, Coverage@10, Diversity@10 and latency.")
st.markdown('<div class="footer">CineMatch AI • Hybrid Recommendation System • Orbo.ai Technical Assignment</div>',unsafe_allow_html=True)
