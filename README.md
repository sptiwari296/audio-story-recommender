# StoryMatch: AI-powered audio story recommendation assistant

A prototype that recommends audio stories from a listener's stated interests and listening behaviour, explains **why** each story was picked, writes a short **personalised intro** for each one, and **evaluates** recommendation quality against baselines.

Stack: Python · pandas · scikit-learn · Groq LLM API (`openai/gpt-oss-120b`, with automatic fallback to other Groq models) · Streamlit

---

## 1. The problem, reframed

The platform has two linked problems:

1. **Discovery.** Thousands of stories, and users can't find ones that match their taste.
2. **Early churn.** Many users listen to one or two episodes and leave.

So the system optimises for "**will this person keep listening?**", not only "is this similar to what they played?". Concretely:

- **Hook rate** is a ranking signal: the share of listeners who continue past episode 1. It carries more weight for new users, because their next listen decides whether they stay.
- **Early abandonment counts as a negative signal**, not just "played".
- A **"Pick up where you left off"** row nudges users back into series they were enjoying.

## 2. Architecture

```
 stated interests (free text) ──► LLM (Groq) ──► structured prefs ┐
                                                                   ├─► UserProfile ─► hybrid scorer ─► MMR re-rank ─► top-K
 listening events ──► implicit signals (completion, binge, skip) ──┘                       │
                                                                                           ▼
                                                     grounded facts per recommendation (code)
                                                                                           │
                                                                                           ▼
                                                LLM (Groq) writes "why" + personalised intro
```

The LLM never ranks the catalogue. Ranking is fast, cheap and measurable in pandas and scikit-learn. The LLM does the language work: understanding free text, and writing explanations and intros.

| Objective | How it's done | Where |
|---|---|---|
| 1. Understand preferences | LLM parses free text into genres, moods, themes, avoid list and episode length. Listening events become weighted signals: <20% completion = −0.6, ≥80% = +1, binging further episodes and likes add more. Genres the user keeps abandoning become an implicit "avoid". | `llm.parse_interests`, `Recommender.build_profile` |
| 2. Recommend | **Taste vector** = TF-IDF blend of stories the user engaged with + stated interests (stated interests carry most weight at cold start, behaviour takes over as history grows). **Score** = 0.45 content similarity + 0.18 co-listening ("people who finished X also finished Y") + smoothed hook rate (0.25 cold / 0.12 warm) + 0.14 duration fit + 0.10 genre match + 0.05 popularity prior, × language match. Avoided genres are down-weighted and already-heard stories removed. **MMR** re-ranking adds variety. | `Recommender.score`, `Recommender.recommend` |
| 3. Explain | Code builds **verifiable facts** for each pick (most similar loved story and shared tags, genre or mood match, length fit, hook rate, co-listening). The LLM is told to phrase *only* these facts, so it can't hallucinate reasons. | `Recommender.reasons`, `llm.explain_and_intro` |
| 4. Personalised intro | The same batched LLM call writes a spoiler-free, 2–3 sentence hook in the story's tone, addressed to the listener. | `llm.explain_and_intro` |
| 5. Evaluate | Temporal hold-out vs random, popularity and content-only baselines; accuracy, beyond-accuracy and ground-truth persona metrics; optional LLM-as-judge for copy quality. | `evaluate.py` |

**Graceful degradation:** Groq rate limits and model availability are per model, so if the configured model is retired or rate-limited, `llm.py` automatically tries the next one (`gpt-oss-120b` → `qwen3.8-27b` → `gpt-oss-20b`). If there's no key or every model fails, each LLM step falls back to deterministic rules or templates, so the prototype always runs. The UI shows which mode produced each piece of text. Explanations and intros for a whole list come from **one batched call** (about 5 s) and are cached.

## 3. Sample dataset (synthetic, seeded, reproducible)

`python generate_data.py` writes to `data/`:

| File | Rows | Contents |
|---|---|---|
| `stories.csv` | 240 | title, genre, subgenre, moods, tags, narrator, language, episodes, episode length, synopsis, release date, plus behaviour-derived `plays`, `avg_completion`, `hook_rate` |
| `users.csv` | 60 | stated genres, free-text bio, language, plus **hidden persona** columns (`_fav_genres`, `_fav_moods`, `_avoid`, `_duration`, `_activity`) |
| `listening_events.csv` | 806 | user, story, timestamp, episodes listened, minutes, completion rate, liked |

**Why synthetic data:** each user has a hidden "true taste", and their listening is *simulated from it*. Stated interests are deliberately a noisy version of that persona. This gives ground truth for evaluation, which real logs don't have. About 30% of users are low-activity (2–4 listens) to mirror the churn problem. The recommender never reads the `_`-prefixed columns.

## 4. Evaluation results

`python evaluate.py` → `eval/results.csv`, `eval/metrics.png`

Protocol: each user's first 70% of listens are history, and later listens with ≥60% completion are the relevant items. Catalogue stats and co-listening are rebuilt from training events only (no leakage). The **cold** segment uses only stated interests plus the first listen. K = 10.

| Segment | Method | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Persona match | Diversity | Coverage |
|---|---|---|---|---|---|---|---|---|
| warm | random | 0.012 | 0.024 | 0.021 | 0.118 | 0.125 | 0.943 | 0.892 |
| warm | popularity | 0.027 | 0.064 | 0.046 | 0.255 | 0.139 | 0.921 | 0.092 |
| warm | content-only | 0.153 | 0.406 | 0.273 | **0.843** | **0.978** | 0.748 | 0.825 |
| warm | **hybrid (ours)** | **0.188** | **0.474** | **0.377** | **0.843** | 0.849 | **0.850** | **0.871** |
| cold | random | 0.039 | 0.025 | 0.042 | 0.333 | 0.149 | 0.946 | 0.888 |
| cold | popularity | 0.047 | 0.034 | 0.047 | 0.392 | 0.139 | 0.940 | 0.079 |
| cold | content-only | 0.333 | 0.302 | 0.373 | 0.922 | **0.925** | 0.736 | 0.704 |
| cold | **hybrid (ours)** | **0.373** | **0.339** | **0.460** | **0.941** | 0.786 | **0.820** | **0.850** |

(Diversity and coverage are bolded only among the personalised methods. Random scores high on them by definition.)

What the results show:

- **Hybrid beats every baseline on ranking quality.** NDCG@10 is +38% over content-only for warm users and +23% for cold-start users, and about 8–10× better than popularity.
- **Better and more varied lists.** Diversity rises from 0.75 to 0.85 and catalogue coverage reaches 87%. Popularity covers only 9% of the catalogue.
- **The trade-off is stated, not hidden.** Persona match is lower than content-only (0.85 vs 0.98) because MMR, the hook-rate signal and length fit deliberately bring in a "stretch" pick from outside the user's top genre. The diversity slider in the app controls this trade-off.
- **Tuning note:** hook rate is Bayesian-smoothed (a story with 1 play and 100% hook rate shouldn't outrank one with 40 plays at 80%). The duration-fit weight was chosen by comparing 0.10, 0.14 and 0.18 on this hold-out. With more data this should be a proper validation split or a learned ranker.
- **Copy quality (LLM-as-judge):** `python evaluate.py --judge 6` asks the Groq LLM to score each explanation and intro on a 1–5 scale. Results over 18 recommendations (6 users × 3), in `eval/judge_scores.csv`:

  | Faithfulness to facts | Relevance to listener | Intro appeal | Spoiler-free |
  |---|---|---|---|
  | **5.00** | **4.83** | **4.00** | **5.00** |

  Perfect faithfulness is what grounding is for: the LLM only phrases facts that code computed, so it can't invent reasons. Intro appeal is the weakest score and the obvious next thing to iterate on. Caveat: the judge is the same model family as the writer, which tends to inflate scores. A human-labelled sample or a different judge model would be more rigorous.

Online metrics to track in production: recommendation click-through, episode-1 completion, % continuing to episode 2, sessions per week, and D7/D30 retention of new users (the churn metric).

## 5. How to run

```bash
pip install -r requirements.txt
python generate_data.py          # optional: data/ is already included
python evaluate.py               # offline metrics + chart
python evaluate.py --judge 5     # + LLM-as-judge (needs GROQ_API_KEY)
streamlit run app.py             # demo UI at http://localhost:8501
```

Enable the LLM (free key from https://console.groq.com). The easiest way is to copy `.env.example` to `.env` and paste your key there (`.env` is git-ignored and loaded automatically). Or set an environment variable:

```bash
# macOS/Linux
export GROQ_API_KEY=gsk_...
# Windows PowerShell
$env:GROQ_API_KEY="gsk_..."
# optional: choose another Groq model
$env:GROQ_MODEL="openai/gpt-oss-120b"
```

## 6. Demo walkthrough

1. **Existing listener:** pick a user (e.g. `U003`). You see their inferred profile, a "Pick up where you left off" row, and 5 recommendations, each with a match score, a "Why this" line, a personalised intro, and an expandable score breakdown with the grounded facts.
2. **Live learning:** click **✅ Finish** or **⏭ Skip** on a card. The listen is added to history and the profile and recommendations update immediately (skipping a genre twice makes it an implicit "avoid").
3. **Cold start:** choose *New listener*, type something like "clever detective stories, nothing too scary, short episodes". The text is parsed into Mystery, avoid Horror and short episodes, and recommendations lean on high hook-rate safe bets.
4. **Evaluation tab:** the metrics table and chart.

Screenshots are in `screenshots/`: `01_recommendations.png` (existing listener), `02_cold_start.png` (new listener) and `03_evaluation_chart.png`. To regenerate them while the app is running, use `powershell -File take_screenshots.ps1` on Windows with Edge. Opening `http://localhost:8501/?mode=new` goes straight to the cold-start view.

## 7. Project structure

```
generate_data.py   synthetic catalogue, users (with hidden personas), listening events
recommender.py     profile building, hybrid scoring, MMR, grounded reasons
llm.py             Groq calls: interest parsing, explanations + intros, LLM-as-judge (+ fallbacks)
app.py             Streamlit UI
evaluate.py        offline evaluation vs baselines, chart, optional LLM-as-judge
data/              generated CSVs
eval/              results.csv, metrics.png
```

## 8. Limitations

- **Synthetic data.** Behaviour is simulated from simple personas, so absolute numbers are optimistic. What carries over is the *relative* comparison and the method.
- **TF-IDF content features** match on words, not meaning ("spooky" ≠ "eerie"). Synopses are template-generated.
- **Small co-listening matrix** (60 users), so collaborative filtering adds little here. It matters at scale.
- **Hand-tuned score weights**, not learned.
- **LLM output** is checked by prompt rules and an optional judge, not by human review. Latency and cost grow with K, which is why one batched call is used per list.
- **No position or exposure bias correction.** Users can only complete what they were shown.

## 9. Future improvements

- **Semantic embeddings** (e.g. sentence-transformers) for stories and free-text interests, with a vector index for thousands of stories.
- **Learning-to-rank** (LightGBM or a two-tower model) trained on real engagement, replacing hand-set weights.
- **Matrix factorisation or sequence models** (e.g. SASRec) for collaborative signal and "what to play next".
- **Contextual bandits** to balance exploration and exploitation online, tuned for D7 retention rather than clicks.
- **Churn-risk model:** detect at-risk users after episode 1 and trigger the "safe bet" strategy, push notifications or continue-listening nudges.
- **Audio intros:** convert the personalised intro to speech in the narrator's style, a natural fit for an audio platform.
- **Online A/B testing**, an LLM-response cache, and a human-reviewed golden set for explanation quality.
- **Signals not used yet:** time of day, device, session length, multilingual preference and narrator affinity.
