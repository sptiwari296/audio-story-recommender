"""Streamlit demo: AI-powered audio story recommendation assistant.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import llm
from recommender import Recommender, load_data

st.set_page_config(page_title="StoryMatch - audio story recommender", page_icon="🎧", layout="wide")
ROOT = Path(__file__).parent


@st.cache_resource
def get_engine():
    stories, users, events = load_data()
    return Recommender(stories, events), stories, users, events


@st.cache_data(show_spinner=False)
def cached_parse(text):
    return llm.parse_interests(text)


@st.cache_data(show_spinner=False)
def cached_copy(summary, facts_json):
    return llm.explain_and_intro(summary, json.loads(facts_json))


rec, stories, users, events = get_engine()

# ------------------------------------------------------------------ sidebar: who is listening?
st.sidebar.title("🎧 StoryMatch")
st.sidebar.caption("Groq LLM: " + (f"**on** ({llm.MODEL})" if llm.available() else "**off**: template fallback (set GROQ_API_KEY)"))
mode = st.sidebar.radio("Listener", ["Existing listener", "New listener (cold start)"],
                        index=1 if st.query_params.get("mode") == "new" else 0)  # ?mode=new opens cold start

if "sim" not in st.session_state:
    st.session_state.sim = {}  # user_id -> list of simulated listening events

if mode == "Existing listener":
    counts = events.groupby("user_id").size()
    labels = {u: f"{u} · {counts.get(u, 0)} listens" for u in users["user_id"]}
    uid = st.sidebar.selectbox("Pick a listener", users["user_id"], format_func=labels.get, index=3)
    user = users.set_index("user_id").loc[uid].to_dict() | {"user_id": uid}
    bio = st.sidebar.text_area("What they told us at sign-up", user["bio"], height=100)
    base_hist = events[events["user_id"] == uid]
else:
    uid = "NEW"
    picked = st.sidebar.multiselect("Pick genres you like", llm.GENRES, default=["Mystery"])
    bio = st.sidebar.text_area("Describe what you like, in your own words",
                               "I like clever detective stories and a bit of suspense, nothing too scary. Short episodes for my commute.",
                               height=110)
    user = {"user_id": "NEW", "stated_genres": "|".join(picked), "language": "English", "bio": bio}
    base_hist = events.iloc[0:0]

k = st.sidebar.slider("Number of recommendations", 3, 10, 5)
diversity = st.sidebar.slider("Diversity (MMR)", 0.0, 0.8, 0.3, 0.05,
                              help="0 = most similar first; higher = more variety in the list")
if st.sidebar.button("Reset simulated listens"):
    st.session_state.sim[uid] = []

sim = pd.DataFrame(st.session_state.sim.get(uid, []))
history = pd.concat([base_hist, sim], ignore_index=True) if len(sim) else base_hist

# ------------------------------------------------------------------ 1. understand preferences
parsed, parse_src = cached_parse(bio)
profile = rec.build_profile(user, history, parsed=parsed)
summary = profile.summary(stories)

tab_rec, tab_hist, tab_eval = st.tabs(["✨ Recommendations", "🕘 Listening history", "📊 Evaluation"])

with tab_rec:
    st.subheader("What we understand about you")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stories listened", profile.n_listens)
    c2.metric("Top genres", ", ".join((profile.top_genres or profile.stated_genres)[:2]) or "-")
    c3.metric("Episode length", profile.duration_pref or "unknown")
    c4.metric("Stage", "Cold start" if profile.is_cold_start else "Warm")
    st.info(f"**Profile:** {summary}")
    with st.expander(f"Interests parsed from text (by {parse_src})"):
        st.json(parsed)

    if profile.in_progress:
        st.markdown("#### ⏯ Pick up where you left off")
        for sid in profile.in_progress[:3]:
            s = stories.set_index("story_id").loc[sid]
            h = history[history["story_id"] == sid].iloc[-1]
            st.write(f"**{s['title']}** · episode {int(h['episodes_listened']) + 1} of {int(s['episodes'])} · {s['genre']}")

    # ------------------------------------------------------------------ 2. recommend
    recs = rec.recommend(profile, history, k=k, diversity=diversity)
    facts = [rec.reasons(profile, r) for _, r in recs.iterrows()]
    with st.spinner("Writing personalised explanations..."):
        copy, copy_src = cached_copy(summary, json.dumps(facts))

    st.markdown(f"#### Recommended for you  <span style='font-size:0.8em;color:gray'>(text by {copy_src})</span>",
                unsafe_allow_html=True)
    for (_, r), f in zip(recs.iterrows(), facts):
        with st.container(border=True):
            left, right = st.columns([4, 1.3])
            with left:
                st.markdown(f"### {r['title']}")
                st.caption(f"{r['genre']} · {r['subgenre']} · {r['moods'].replace('|', ', ')} · "
                           f"{int(r['episodes'])} ep × {int(r['episode_minutes'])} min · {r['language']} · {r['narrator']}")
                st.markdown(f"**Why this:** {copy[r['story_id']]['why']}")
                st.markdown(f"🎙️ *{copy[r['story_id']]['intro']}*")
            with right:
                st.metric("Match score", f"{r['score']:.2f}")
                b1, b2 = st.columns(2)
                if b1.button("✅ Finish", key=f"fin{r['story_id']}", help="Simulate: listened to the end and kept going"):
                    st.session_state.sim.setdefault(uid, []).append({
                        "user_id": uid, "story_id": r["story_id"], "timestamp": pd.Timestamp.now(),
                        "episodes_listened": min(3, int(r["episodes"])), "minutes_listened": float(r["episode_minutes"]),
                        "completion_rate": 1.0, "liked": True})
                    st.rerun()
                if b2.button("⏭ Skip", key=f"skip{r['story_id']}", help="Simulate: stopped after a minute"):
                    st.session_state.sim.setdefault(uid, []).append({
                        "user_id": uid, "story_id": r["story_id"], "timestamp": pd.Timestamp.now(),
                        "episodes_listened": 1, "minutes_listened": 1.0, "completion_rate": 0.05, "liked": False})
                    st.rerun()
            with st.expander("Score breakdown and grounded facts"):
                st.write({"content similarity": round(r["content_sim"], 3), "co-listening": round(r["collab_sim"], 3),
                          "hook rate": r["hook_rate"], "duration fit": r["duration_fit"]})
                st.json(f["signals"])

with tab_hist:
    if len(history):
        h = history.merge(stories[["story_id", "title", "genre", "moods", "episodes"]], on="story_id")
        st.dataframe(h[["timestamp", "title", "genre", "moods", "episodes_listened", "episodes", "completion_rate", "liked"]]
                     .sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.write("No listening history yet: recommendations come from stated interests plus stories with a high hook rate.")

with tab_eval:
    res = ROOT / "eval" / "results.csv"
    if res.exists():
        st.markdown("Temporal hold-out evaluation (`python evaluate.py`). **hybrid** is this system.")
        st.dataframe(pd.read_csv(res), use_container_width=True, hide_index=True)
        if (ROOT / "eval" / "metrics.png").exists():
            st.image(str(ROOT / "eval" / "metrics.png"))
    else:
        st.write("Run `python evaluate.py` to generate evaluation results.")
