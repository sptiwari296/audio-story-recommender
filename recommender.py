"""Hybrid, churn-aware audio story recommender.

Pipeline
  1. Preference understanding  -> UserProfile (stated interests + implicit listening signals)
  2. Candidate scoring         -> content similarity + co-listening + hook rate + duration fit + popularity
  3. Diversification           -> MMR re-ranking so the list is not five near-identical stories
  4. Grounded reasons          -> structured facts per recommendation (fed to the LLM for explanations)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data"

DURATION_RANGES = {"short": (0, 15), "medium": (15, 30), "long": (30, 999)}


def load_data():
    stories = pd.read_csv(DATA / "stories.csv")
    users = pd.read_csv(DATA / "users.csv")
    events = pd.read_csv(DATA / "listening_events.csv", parse_dates=["timestamp"])
    return stories, users, events


def engagement_weight(row) -> float:
    """Turn one listening event into an implicit preference signal."""
    c = row["completion_rate"]
    if c < 0.2:
        w = -0.6                      # bailed early: strong negative
    elif c < 0.5:
        w = 0.0                       # sampled, not hooked
    elif c < 0.8:
        w = 0.5
    else:
        w = 1.0
    if row.get("episodes_listened", 1) > 1:
        w += min(1.0, 0.25 * (row["episodes_listened"] - 1))   # binged on: very strong positive
    if row.get("liked", False):
        w += 0.5
    return w


@dataclass
class UserProfile:
    user_id: str
    stated_genres: list[str] = field(default_factory=list)
    stated_moods: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    avoid_genres: list[str] = field(default_factory=list)
    duration_pref: str | None = None          # short | medium | long
    language: str | None = None
    bio: str = ""
    # derived from behaviour
    n_listens: int = 0
    top_genres: list[str] = field(default_factory=list)
    top_moods: list[str] = field(default_factory=list)
    loved: list[str] = field(default_factory=list)        # story_ids with strong positive signal
    skipped: list[str] = field(default_factory=list)      # story_ids abandoned early
    in_progress: list[str] = field(default_factory=list)  # multi-episode stories not finished

    @property
    def is_cold_start(self) -> bool:
        return self.n_listens < 3

    def summary(self, stories: pd.DataFrame) -> str:
        title = stories.set_index("story_id")["title"]
        parts = []
        g = self.top_genres or self.stated_genres
        if g:
            parts.append(f"enjoys {', '.join(g[:3])}")
        m = self.top_moods or self.stated_moods
        if m:
            parts.append(f"likes {', '.join(m[:3])} moods")
        if self.duration_pref:
            parts.append(f"prefers {self.duration_pref} episodes")
        if self.avoid_genres:
            parts.append(f"avoids {', '.join(self.avoid_genres)}")
        if self.loved:
            parts.append("loved " + ", ".join(f"'{title[s]}'" for s in self.loved[:3]))
        stage = "new listener (cold start)" if self.is_cold_start else f"{self.n_listens} stories listened"
        return f"{stage}; " + "; ".join(parts)


class Recommender:
    def __init__(self, stories: pd.DataFrame, events: pd.DataFrame):
        self.stories = stories.reset_index(drop=True).copy()
        # Bayesian smoothing: a story with 1 play and 100% hook rate shouldn't beat one with 40 plays at 80%
        prior, m = self.stories["hook_rate"].mean(), 8
        plays = self.stories["plays"].fillna(0) if "plays" in self.stories else 0
        self.stories["hook_rate"] = ((self.stories["hook_rate"] * plays + prior * m) / (plays + m)).round(3)
        self.idx = {sid: i for i, sid in enumerate(self.stories["story_id"])}
        self.events = events
        text = (self.stories["genre"] + " " + self.stories["subgenre"] + " "
                + self.stories["moods"].str.replace("|", " ") + " "
                + self.stories["tags"].str.replace("|", " ") + " " + self.stories["synopsis"])
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vectorizer.fit_transform(text.str.lower())
        self.item_sim = cosine_similarity(self.X)
        self._build_collab(events)
        pop = self.stories["plays"].astype(float) if "plays" in self.stories else pd.Series(1.0, index=self.stories.index)
        self.pop_norm = (np.log1p(pop) / np.log1p(pop).max()).to_numpy()
        self.hook = self.stories["hook_rate"].to_numpy()

    # ------------------------------------------------------------------ collaborative signal
    def _build_collab(self, events: pd.DataFrame):
        """Item-item cosine similarity over users' positive listens ("people who finished X also finished Y")."""
        pos = events[events["completion_rate"] >= 0.6]
        users = {u: i for i, u in enumerate(pos["user_id"].unique())}
        M = np.zeros((len(self.stories), max(1, len(users))))
        for u, s in zip(pos["user_id"], pos["story_id"]):
            if s in self.idx:
                M[self.idx[s], users[u]] = 1.0
        self.co_sim = cosine_similarity(M) if len(users) else np.zeros((len(self.stories),) * 2)
        np.fill_diagonal(self.co_sim, 0)

    # ------------------------------------------------------------------ 1. preferences
    def build_profile(self, user: dict, history: pd.DataFrame, parsed: dict | None = None) -> UserProfile:
        """Combine explicit interests (optionally LLM-parsed from free text) with implicit behaviour."""
        stated = [g for g in str(user.get("stated_genres", "")).split("|") if g and g != "nan"]
        p = UserProfile(user_id=user.get("user_id", "NEW"), stated_genres=stated,
                        language=user.get("language"), bio=str(user.get("bio", "")))
        if parsed:
            p.stated_genres = list(dict.fromkeys(p.stated_genres + parsed.get("genres", [])))
            p.stated_moods = parsed.get("moods", [])
            p.themes = parsed.get("themes", [])
            p.avoid_genres = parsed.get("avoid_genres", [])
            p.duration_pref = parsed.get("duration_pref") or None
        if history is not None and len(history):
            h = history.merge(self.stories, on="story_id", how="left")
            h["w"] = h.apply(engagement_weight, axis=1)
            p.n_listens = len(h)
            pos = h[h["w"] > 0.4]
            p.loved = pos.sort_values("w", ascending=False)["story_id"].tolist()
            p.skipped = h[h["w"] < 0]["story_id"].tolist()
            p.in_progress = h[(h["episodes"] > h["episodes_listened"]) & (h["w"] >= 0.5)]["story_id"].tolist()
            if len(pos):
                p.top_genres = pos.groupby("genre")["w"].sum().sort_values(ascending=False).index[:3].tolist()
                moods = pos.assign(m=pos["moods"].str.split("|")).explode("m")
                p.top_moods = moods.groupby("m")["w"].sum().sort_values(ascending=False).index[:3].tolist()
                if not p.duration_pref:
                    med = np.average(pos["episode_minutes"], weights=pos["w"])
                    p.duration_pref = next(k for k, (lo, hi) in DURATION_RANGES.items() if lo <= med < hi)
            # genres the user repeatedly abandons become implicit "avoid"
            neg = h[h["w"] < 0].groupby("genre").size()
            for g, n in neg.items():
                if n >= 2 and g not in p.top_genres and g not in p.avoid_genres:
                    p.avoid_genres.append(g)
        return p

    def _taste_vector(self, p: UserProfile, history: pd.DataFrame | None):
        """Weighted blend of: stories they engaged with  +  their stated interests (text)."""
        stated_text = " ".join(p.stated_genres + p.stated_moods + p.themes) + " " + p.bio
        v_stated = self.vectorizer.transform([stated_text.lower()]).toarray()[0]
        v_hist = np.zeros(self.X.shape[1])
        if history is not None and len(history):
            for _, r in history.iterrows():
                if r["story_id"] in self.idx:
                    v_hist += engagement_weight(r) * self.X[self.idx[r["story_id"]]].toarray()[0]
        # trust stated interests at cold start; trust behaviour once we have evidence
        alpha = 0.8 if p.n_listens == 0 else max(0.2, 0.7 - 0.08 * p.n_listens)
        def unit(v):
            n = np.linalg.norm(v)
            return v / n if n else v
        return alpha * unit(v_stated) + (1 - alpha) * unit(v_hist)

    # ------------------------------------------------------------------ 2+3. score & rank
    def score(self, p: UserProfile, history: pd.DataFrame | None, mode: str = "hybrid") -> pd.DataFrame:
        taste = self._taste_vector(p, history)
        content = np.asarray(self.X @ taste).ravel()
        content = content / content.max() if content.max() > 0 else content

        collab = np.zeros(len(self.stories))
        if p.loved:
            ids = [self.idx[s] for s in p.loved if s in self.idx]
            collab = self.co_sim[ids].mean(axis=0)
            collab = collab / collab.max() if collab.max() > 0 else collab

        dur = np.full(len(self.stories), 0.5)
        if p.duration_pref in DURATION_RANGES:
            lo, hi = DURATION_RANGES[p.duration_pref]
            m = self.stories["episode_minutes"].to_numpy()
            dur = np.where((m >= lo) & (m < hi), 1.0, 0.2)

        lang = np.where(self.stories["language"] == p.language, 1.0, 0.6) if p.language else np.ones(len(self.stories))
        genre_bonus = self.stories["genre"].isin(p.top_genres or p.stated_genres).to_numpy().astype(float)

        if mode == "content":
            total = content
        elif mode == "popularity":
            total = self.pop_norm.copy()
        else:
            # cold-start users lean on "safe bets": stories with a strong hook rate.
            w_hook = 0.25 if p.is_cold_start else 0.12
            w_col = 0.0 if not p.loved else 0.18
            total = (0.45 * content + w_col * collab + w_hook * self.hook + 0.14 * dur
                     + 0.10 * genre_bonus + 0.05 * self.pop_norm) * lang

        avoid = self.stories["genre"].isin(p.avoid_genres).to_numpy()
        total = np.where(avoid, total * 0.2, total)
        df = self.stories.assign(score=total, content_sim=content, collab_sim=collab, duration_fit=dur)
        heard = set(history["story_id"]) if history is not None and len(history) else set()
        return df[~df["story_id"].isin(heard)]

    def recommend(self, p: UserProfile, history: pd.DataFrame | None, k: int = 5,
                  mode: str = "hybrid", diversity: float = 0.3) -> pd.DataFrame:
        cands = self.score(p, history, mode).sort_values("score", ascending=False).head(60)
        if mode != "hybrid" or diversity <= 0:
            return cands.head(k)
        # MMR: balance relevance with novelty relative to what's already picked
        chosen, pool = [], cands.copy()
        rel = (pool["score"] / pool["score"].max()).to_dict()
        while len(chosen) < k and len(pool):
            best, best_val = None, -1e9
            for i, sid in zip(pool.index, pool["story_id"]):
                sim = max((self.item_sim[self.idx[sid], self.idx[c]] for c in chosen), default=0.0)
                val = (1 - diversity) * rel[i] - diversity * sim
                if val > best_val:
                    best, best_val = i, val
            chosen.append(pool.loc[best, "story_id"])
            pool = pool.drop(best)
        return cands.set_index("story_id").loc[chosen].reset_index()

    # ------------------------------------------------------------------ 4. grounded reasons
    def reasons(self, p: UserProfile, rec: pd.Series) -> dict:
        """Structured, verifiable facts explaining a recommendation. The LLM only phrases these."""
        facts = {"story_id": rec["story_id"], "title": rec["title"], "genre": rec["genre"],
                 "subgenre": rec["subgenre"], "moods": rec["moods"].split("|"), "tags": rec["tags"].split("|"),
                 "episodes": int(rec["episodes"]), "episode_minutes": int(rec["episode_minutes"]),
                 "synopsis": rec["synopsis"], "signals": []}
        sid = self.idx[rec["story_id"]]
        if p.loved:
            sims = [(s, self.item_sim[sid, self.idx[s]]) for s in p.loved if s in self.idx]
            s, v = max(sims, key=lambda t: t[1])
            if v > 0.08:
                src = self.stories.iloc[self.idx[s]]
                shared = sorted(set(rec["tags"].split("|") + rec["moods"].split("|"))
                                & set(src["tags"].split("|") + src["moods"].split("|")))
                facts["signals"].append({"type": "similar_to_loved", "story": src["title"], "shared": shared})
        if rec.get("collab_sim", 0) > 0.3 and p.loved:
            facts["signals"].append({"type": "co_listened", "detail": "listeners who finished stories you loved also finished this"})
        g = p.top_genres or p.stated_genres
        if rec["genre"] in g:
            facts["signals"].append({"type": "genre_match", "genre": rec["genre"],
                                     "source": "listening history" if p.top_genres else "stated interests"})
        mood_match = sorted(set(rec["moods"].split("|")) & set(p.top_moods or p.stated_moods))
        if mood_match:
            facts["signals"].append({"type": "mood_match", "moods": mood_match})
        if rec.get("duration_fit", 0) == 1.0 and p.duration_pref:
            facts["signals"].append({"type": "length_fit", "detail": f"{int(rec['episode_minutes'])}-min episodes suit your {p.duration_pref} preference"})
        if rec["hook_rate"] >= 0.55 and rec.get("plays", 0) >= 3:
            facts["signals"].append({"type": "high_hook", "detail": f"about {int(round(rec['hook_rate'] * 100))}% of listeners get hooked and keep listening"})
        if not facts["signals"]:
            facts["signals"].append({"type": "exploration", "detail": "a well-rated pick slightly outside your usual taste"})
        return facts
