"""Generate a synthetic, reproducible dataset for the audio story recommender.

Outputs (in ./data):
  stories.csv           - catalogue of audio stories
  users.csv             - users with stated interests + a HIDDEN persona (ground truth for evaluation)
  listening_events.csv  - simulated listening behaviour driven by each user's hidden persona

Run:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_STORIES = 240
N_USERS = 60
OUT = Path(__file__).parent / "data"

rng = np.random.default_rng(SEED)

# --------------------------------------------------------------------------- catalogue vocab
GENRES = {
    "Thriller": ["psychological thriller", "crime thriller", "spy thriller"],
    "Horror": ["supernatural horror", "folk horror", "haunted house"],
    "Romance": ["slow-burn romance", "romantic comedy", "second-chance romance"],
    "Sci-Fi": ["space opera", "dystopian", "time travel"],
    "Fantasy": ["epic fantasy", "urban fantasy", "mythic retelling"],
    "Mystery": ["detective mystery", "cozy mystery", "locked-room mystery"],
    "Comedy": ["sitcom-style comedy", "satire", "dark comedy"],
    "Drama": ["family drama", "coming-of-age", "workplace drama"],
    "History": ["historical fiction", "war story", "royal intrigue"],
    "Motivation": ["self-improvement", "true-life inspiration", "founder story"],
}
MOODS = ["dark", "suspenseful", "heartwarming", "funny", "emotional", "uplifting",
         "eerie", "romantic", "thought-provoking", "light", "intense", "nostalgic"]
GENRE_MOODS = {
    "Thriller": ["suspenseful", "dark", "intense"],
    "Horror": ["eerie", "dark", "intense"],
    "Romance": ["romantic", "heartwarming", "emotional"],
    "Sci-Fi": ["thought-provoking", "intense", "suspenseful"],
    "Fantasy": ["thought-provoking", "uplifting", "intense"],
    "Mystery": ["suspenseful", "light", "thought-provoking"],
    "Comedy": ["funny", "light", "heartwarming"],
    "Drama": ["emotional", "heartwarming", "nostalgic"],
    "History": ["nostalgic", "intense", "thought-provoking"],
    "Motivation": ["uplifting", "emotional", "thought-provoking"],
}
TAGS = {
    "Thriller": ["twist ending", "unreliable narrator", "heist", "revenge", "cat and mouse"],
    "Horror": ["ghosts", "small town", "cursed object", "isolation", "creature"],
    "Romance": ["enemies to lovers", "small town", "letters", "rivals", "wedding"],
    "Sci-Fi": ["AI", "space station", "first contact", "clones", "rebellion"],
    "Fantasy": ["magic school", "dragons", "chosen one", "gods", "quest"],
    "Mystery": ["detective", "small town", "poison", "missing person", "twist ending"],
    "Comedy": ["office", "family chaos", "road trip", "misunderstanding", "wedding"],
    "Drama": ["siblings", "letters", "first job", "illness", "reunion"],
    "History": ["empire", "rebellion", "letters", "spies", "revenge"],
    "Motivation": ["startup", "sports", "comeback", "habits", "first job"],
}
SETTINGS = ["a fog-bound coastal town", "a crowded Mumbai high-rise", "a dying space colony",
            "a 1920s railway line", "a remote mountain monastery", "a buzzing tech startup",
            "a forgotten royal palace", "a sleepy hill station", "an underground city",
            "a college hostel", "a floating market", "a snowed-in hotel"]
LEADS = ["a retired detective", "a night-shift nurse", "a young cartographer", "an exiled prince",
         "a radio host", "two estranged sisters", "a failed magician", "a rookie journalist",
         "a widowed baker", "a teenage hacker", "a forest ranger", "a wedding planner"]
HOOKS = ["uncovers a secret that someone will kill to keep", "receives letters from a stranger who knows too much",
         "must survive one impossible night", "is given a second chance at a life they walked away from",
         "discovers the town remembers a past that never happened", "is drawn into a rivalry that turns personal",
         "has thirty days to win back everything they lost", "starts hearing voices on an old radio",
         "accidentally becomes the most wanted person in the city", "learns their mentor was never who they seemed"]
TITLE_A = ["The Last", "Whispers of", "Midnight", "The Silent", "Letters from", "The Hollow",
           "Beneath the", "City of", "The Glass", "Echoes of", "The Ninth", "A Song for",
           "The Paper", "Shadows over", "The Lantern", "Salt and"]
TITLE_B = ["Lighthouse", "Monsoon", "Archive", "Orchard", "Station", "Crown", "Harbour",
           "Signal", "Garden", "Mirror", "Express", "Ember", "Compass", "Tide", "Ledger", "Wolves"]
NARRATORS = [f"Narrator {c}" for c in "ABCDEFGHIJKL"]
LANGUAGES = ["English", "Hindi"]


def make_stories() -> pd.DataFrame:
    rows, used_titles = [], set()
    genre_names = list(GENRES)
    for i in range(N_STORIES):
        genre = genre_names[i % len(genre_names)]
        sub = rng.choice(GENRES[genre])
        moods = list(rng.choice(GENRE_MOODS[genre], size=2, replace=False))
        if rng.random() < 0.25:  # occasional off-genre mood for variety
            moods[1] = rng.choice(MOODS)
        tags = list(rng.choice(TAGS[genre], size=3, replace=False))
        while True:
            title = f"{rng.choice(TITLE_A)} {rng.choice(TITLE_B)}"
            if title not in used_titles:
                break
            title = f"{title} {rng.integers(2, 9)}"
            if title not in used_titles:
                break
        used_titles.add(title)
        lead, setting, hook = rng.choice(LEADS), rng.choice(SETTINGS), rng.choice(HOOKS)
        article = "An" if moods[0][0] in "aeiou" else "A"
        synopsis = (f"{article} {moods[0]}, {moods[1]} {sub} set in {setting}, where {lead} {hook}. "
                    f"Themes: {', '.join(tags)}.")
        episodes = int(rng.choice([1, 3, 5, 8, 12, 20], p=[.15, .2, .2, .2, .15, .1]))
        ep_min = int(rng.choice([8, 12, 15, 20, 25, 35, 45]))
        quality = float(np.clip(rng.beta(5, 3), 0.15, 0.98))  # latent "how gripping is it"
        rows.append({
            "story_id": f"S{i:03d}", "title": title, "genre": genre, "subgenre": sub,
            "moods": "|".join(moods), "tags": "|".join(tags), "narrator": rng.choice(NARRATORS),
            "language": rng.choice(LANGUAGES, p=[.7, .3]), "episodes": episodes,
            "episode_minutes": ep_min, "synopsis": synopsis,
            "release_date": (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 680)))).date(),
            "_quality": quality,
        })
    return pd.DataFrame(rows)


def make_users() -> pd.DataFrame:
    rows = []
    genre_names = list(GENRES)
    for u in range(N_USERS):
        fav = list(rng.choice(genre_names, size=int(rng.choice([1, 2], p=[.4, .6])), replace=False))
        fav_moods = list(dict.fromkeys(sum((GENRE_MOODS[g][:2] for g in fav), [])))[:3]
        avoid = [g for g in rng.choice(genre_names, size=2, replace=False) if g not in fav][:1]
        dur = str(rng.choice(["short", "medium", "long"]))
        # Stated interests are a noisy view of the hidden persona (real users are imperfect at self-description)
        stated = fav[:1] if rng.random() < 0.35 else fav
        if rng.random() < 0.2:
            stated = stated + [str(rng.choice(genre_names))]
        dur_txt = {"short": "quick episodes under 15 minutes", "medium": "episodes around 20 minutes",
                   "long": "long immersive episodes"}[dur]
        bio = (f"I enjoy {', '.join(s.lower() for s in stated)} stories that feel "
               f"{' and '.join(fav_moods[:2])}. I prefer {dur_txt}."
               + (f" Not into {avoid[0].lower()}." if avoid and rng.random() < 0.6 else ""))
        # activity level: ~30% of users are "at-risk" and only listen to 1-3 stories
        activity = int(rng.choice([2, 4, 10, 18, 28], p=[.15, .15, .3, .25, .15]))
        rows.append({
            "user_id": f"U{u:03d}", "name": f"Listener {u:03d}", "language": rng.choice(LANGUAGES, p=[.75, .25]),
            "stated_genres": "|".join(stated), "bio": bio,
            "signup_date": (pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 200)))).date(),
            # hidden ground truth (prefixed with _ ; never used by the recommender)
            "_fav_genres": "|".join(fav), "_fav_moods": "|".join(fav_moods),
            "_avoid": "|".join(avoid), "_duration": dur, "_activity": activity,
        })
    return pd.DataFrame(rows)


def affinity(user, story) -> float:
    fav = user["_fav_genres"].split("|")
    moods = set(user["_fav_moods"].split("|"))
    a = 0.0
    a += 2.2 if story["genre"] in fav else 0.0
    a -= 2.5 if story["genre"] in user["_avoid"].split("|") else 0.0
    a += 0.6 * len(moods & set(story["moods"].split("|")))
    m = story["episode_minutes"]
    want = {"short": (0, 15), "medium": (15, 30), "long": (30, 99)}[user["_duration"]]
    a += 0.5 if want[0] <= m < want[1] else -0.3
    a += 0.4 if story["language"] == user["language"] else -0.2
    a += 1.5 * (story["_quality"] - 0.6)
    return a


def make_events(stories: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    events = []
    pop_prior = rng.gamma(1.2, 1.0, len(stories))  # some stories are just more visible on the home page
    for _, u in users.iterrows():
        aff = np.array([affinity(u, s) for _, s in stories.iterrows()])
        logits = aff * 1.3 + 0.35 * np.log(pop_prior + 0.1)
        p = np.exp(logits - logits.max()); p /= p.sum()
        n = u["_activity"]
        chosen = rng.choice(len(stories), size=n, replace=False, p=p)
        t = pd.Timestamp(u["signup_date"]) + pd.Timedelta(hours=int(rng.integers(1, 48)))
        for idx in chosen:
            s = stories.iloc[idx]
            # completion of episode 1 depends on true affinity + story quality
            sig = 1 / (1 + np.exp(-(aff[idx] - 1.0)))
            completion = float(np.clip(rng.normal(0.15 + 0.8 * sig, 0.15), 0.02, 1.0))
            eps_listened = 1
            if completion > 0.7 and s["episodes"] > 1:
                cont = sig * s["_quality"]
                eps_listened = 1 + int(rng.binomial(s["episodes"] - 1, min(0.95, cont)))
            liked = bool(completion > 0.8 and rng.random() < 0.45)
            events.append({
                "user_id": u["user_id"], "story_id": s["story_id"], "timestamp": t,
                "episodes_listened": eps_listened,
                "minutes_listened": round(completion * s["episode_minutes"] + (eps_listened - 1) * s["episode_minutes"] * 0.9, 1),
                "completion_rate": round(completion, 3), "liked": liked,
            })
            t += pd.Timedelta(hours=float(rng.exponential(30)))
    return pd.DataFrame(events).sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def main():
    OUT.mkdir(exist_ok=True)
    stories, users = make_stories(), make_users()
    events = make_events(stories, users)
    # Observable catalogue stats computed from behaviour (what a real platform would know)
    g = events.groupby("story_id")
    stats = pd.DataFrame({
        "plays": g.size(),
        "avg_completion": g["completion_rate"].mean().round(3),
        "hook_rate": g.apply(lambda d: ((d["episodes_listened"] > 1) | (d["completion_rate"] > 0.85)).mean()).round(3),
    })
    stories = stories.merge(stats, left_on="story_id", right_index=True, how="left")
    stories[["plays"]] = stories[["plays"]].fillna(0).astype(int)
    stories["avg_completion"] = stories["avg_completion"].fillna(0.5)
    stories["hook_rate"] = stories["hook_rate"].fillna(0.5)
    stories.drop(columns=["_quality"]).to_csv(OUT / "stories.csv", index=False)
    users.to_csv(OUT / "users.csv", index=False)
    events.to_csv(OUT / "listening_events.csv", index=False)
    print(f"stories={len(stories)} users={len(users)} events={len(events)} -> {OUT}")


if __name__ == "__main__":
    main()
