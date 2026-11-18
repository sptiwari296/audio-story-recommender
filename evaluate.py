"""Offline evaluation of the recommender against baselines.

Protocol
  * Temporal hold-out: for every user, the first 70% of their listens are "history"; later listens with
    completion >= 0.6 are the relevant items we hope to recommend.
  * Catalogue stats (plays, hook rate) and the co-listening matrix are rebuilt from TRAIN events only (no leakage).
  * Two segments: "warm" (70% history) and "cold start" (only stated interests + first listen).
  * Ground-truth persona check: share of recommendations that fall in the user's HIDDEN favourite genres.
  * Optional: --judge N runs a Groq-hosted LLM as an LLM-as-judge on explanation/intro quality for N users.

Run:  python evaluate.py            (offline metrics, no API key needed)
      python evaluate.py --judge 5  (adds LLM-as-judge scores; needs GROQ_API_KEY)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import llm
from recommender import Recommender, load_data

K = 10
OUT = Path(__file__).parent / "eval"
METHODS = ["random", "popularity", "content", "hybrid"]


def ndcg(recs, relevant, k=K):
    dcg = sum(1 / np.log2(i + 2) for i, s in enumerate(recs[:k]) if s in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def rebuild_stats(stories, train):
    g = train.groupby("story_id")
    st = pd.DataFrame({"plays": g.size(),
                       "hook_rate": g.apply(lambda d: ((d["episodes_listened"] > 1) | (d["completion_rate"] > 0.85)).mean())})
    s = stories.drop(columns=["plays", "hook_rate"]).merge(st, left_on="story_id", right_index=True, how="left")
    s["plays"] = s["plays"].fillna(0)
    s["hook_rate"] = s["hook_rate"].fillna(s["hook_rate"].median())
    return s


def get_recs(rec, method, profile, history, rng):
    if method == "random":
        heard = set(history["story_id"])
        pool = [s for s in rec.stories["story_id"] if s not in heard]
        return list(rng.choice(pool, size=K, replace=False))
    mode = "hybrid" if method == "hybrid" else method
    return rec.recommend(profile, history, k=K, mode=mode, diversity=0.25)["story_id"].tolist()


def evaluate(segment: str, seed=0):
    stories, users, events = load_data()
    rng = np.random.default_rng(seed)
    splits = {}
    for uid, h in events.groupby("user_id"):
        h = h.sort_values("timestamp")
        if len(h) < 3:
            continue
        cut = 1 if segment == "cold" else max(1, int(len(h) * 0.7))
        test = h.iloc[cut:]
        relevant = set(test[test["completion_rate"] >= 0.6]["story_id"])
        if relevant:
            splits[uid] = (h.iloc[:cut], relevant)
    train_all = pd.concat([v[0] for v in splits.values()])
    rec = Recommender(rebuild_stats(stories, train_all), train_all)
    genre = rec.stories.set_index("story_id")["genre"]
    pop = rec.stories.set_index("story_id")["plays"]
    p_pop = (pop + 1) / (pop + 1).sum()

    rows, all_recs = [], {m: set() for m in METHODS}
    for uid, (hist, relevant) in splits.items():
        u = users.set_index("user_id").loc[uid].to_dict() | {"user_id": uid}
        profile = rec.build_profile(u, hist, parsed=llm._parse_rules(u["bio"]))
        fav = set(u["_fav_genres"].split("|"))
        for m in METHODS:
            r = get_recs(rec, m, profile, hist, rng)
            all_recs[m].update(r)
            idx = [rec.idx[s] for s in r]
            sims = rec.item_sim[np.ix_(idx, idx)]
            ild = 1 - (sims.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
            hits = len(set(r) & relevant)
            rows.append({"segment": segment, "method": m, "user_id": uid,
                         "precision@10": hits / K, "recall@10": hits / len(relevant),
                         "ndcg@10": ndcg(r, relevant), "hit_rate@10": float(hits > 0),
                         "persona_match": np.mean([genre[s] in fav for s in r]),
                         "diversity": ild, "novelty": float(np.mean([-np.log2(p_pop[s]) for s in r]))})
    df = pd.DataFrame(rows)
    summary = df.groupby("method").mean(numeric_only=True).reindex(METHODS)
    summary["coverage"] = [len(all_recs[m]) / len(rec.stories) for m in METHODS]
    summary.insert(0, "users", len(splits))
    summary.insert(0, "segment", segment)
    return summary.round(3), rec, splits


def run_judge(rec, splits, n):
    if not llm.available():
        print("\n[judge] skipped: set GROQ_API_KEY to enable LLM-as-judge.")
        return None
    _, users, _ = load_data()
    users = users.set_index("user_id")
    scores = []
    for uid in list(splits)[:n]:
        hist = splits[uid][0]
        u = users.loc[uid].to_dict() | {"user_id": uid}
        p = rec.build_profile(u, hist, parsed=llm._parse_rules(u["bio"]))
        recs = rec.recommend(p, hist, k=3)
        facts = [rec.reasons(p, r) for _, r in recs.iterrows()]
        copy, src = llm.explain_and_intro(p.summary(rec.stories), facts)
        for f in facts:
            j = llm.judge(p.summary(rec.stories), f, copy[f["story_id"]]["why"], copy[f["story_id"]]["intro"])
            if j:
                scores.append({**{k: j.get(k) for k in ["faithfulness", "relevance", "intro_appeal", "spoiler_free"]},
                               "copy_source": src})
    if not scores:
        print("[judge] no scores returned", llm.LAST_ERROR)
        return None
    js = pd.DataFrame(scores)
    print("\nLLM-as-judge (1-5, mean over", len(js), "recommendations):")
    print(js.groupby("copy_source").mean(numeric_only=True).round(2).to_string())
    js.to_csv(OUT / "judge_scores.csv", index=False)
    return js


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["ndcg@10", "hit_rate@10", "persona_match", "coverage"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = ["#b0b7c3", "#7aa6d6", "#e0a458", "#2e7d5b"]
    for ax, seg in zip(axes, ["warm", "cold"]):
        r = results[results["segment"] == seg]
        x = np.arange(len(metrics))
        for i, m in enumerate(METHODS):
            ax.bar(x + i * 0.2 - 0.3, r.loc[m, metrics], 0.2, label=m, color=colors[i])
        ax.set_xticks(x, metrics)
        ax.set_title(f"{seg} users (n={int(r['users'].iloc[0])})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Recommender vs baselines (higher is better)")
    fig.tight_layout()
    fig.savefig(OUT / "metrics.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=int, default=0, help="number of users to score with the Groq LLM-as-judge")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    warm, rec, splits = evaluate("warm")
    cold, _, _ = evaluate("cold")
    results = pd.concat([warm, cold])
    results.to_csv(OUT / "results.csv")
    pd.set_option("display.width", 200)
    print(results.to_string())
    plot(results)
    print(f"\nSaved {OUT / 'results.csv'} and {OUT / 'metrics.png'}")
    if args.judge:
        run_judge(rec, splits, args.judge)


if __name__ == "__main__":
    main()
