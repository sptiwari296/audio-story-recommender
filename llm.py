"""Groq LLM layer: preference parsing, grounded explanations, personalised intros, LLM-as-judge.

Every function has a deterministic template fallback so the prototype runs without an API key.
Set GROQ_API_KEY (and optionally GROQ_MODEL) to enable the LLM.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _load_dotenv():
    """Minimal .env loader (KEY=value lines) so the key never has to be hard-coded."""
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# tried in order if the configured model is retired / unavailable on the account
FALLBACK_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]
GENRES = ["Thriller", "Horror", "Romance", "Sci-Fi", "Fantasy", "Mystery", "Comedy", "Drama", "History", "Motivation"]
MOODS = ["dark", "suspenseful", "heartwarming", "funny", "emotional", "uplifting", "eerie", "romantic",
         "thought-provoking", "light", "intense", "nostalgic"]

_client = None
LAST_ERROR = None


def available() -> bool:
    if os.getenv("DISABLE_LLM"):
        return False
    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return bool(os.getenv("GROQ_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq()  # reads GROQ_API_KEY
    return _client


def _call(system: str, prompt: str, max_tokens: int = 4000) -> str | None:
    """One Groq chat-completion call in JSON mode. Returns text or None (caller then uses the fallback)."""
    global LAST_ERROR, MODEL
    if not available():
        return None
    import groq
    candidates = [MODEL] + [m for m in FALLBACK_MODELS if m != MODEL]
    for model in candidates:
        try:
            resp = _get_client().chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.4,
                response_format={"type": "json_object"},  # Groq JSON mode: response is guaranteed valid JSON
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            )
        except groq.NotFoundError as e:  # model retired or not on this account -> try the next one
            LAST_ERROR = f"model {model} unavailable: {e.message}"; continue
        except groq.RateLimitError as e:  # limits are per model on Groq -> try the next one
            LAST_ERROR = f"rate limited on {model}: {e.message}"; continue
        except groq.APIStatusError as e:
            LAST_ERROR = f"API error {e.status_code}: {e.message}"; return None
        except groq.APIConnectionError as e:
            LAST_ERROR = f"connection error: {e}"; return None
        MODEL, LAST_ERROR = model, None  # remember the working model
        return resp.choices[0].message.content
    return None


def _json(text: str | None):
    if not text:
        return None
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- 1. preference parsing
def parse_interests(free_text: str) -> tuple[dict, str]:
    """Free text ("slow-burn mysteries, nothing gory, short episodes") -> structured profile."""
    system = "You convert a listener's description of their taste into a structured profile for an audio story app. Reply with JSON only."
    prompt = (f"Allowed genres: {GENRES}\nAllowed moods: {MOODS}\n"
              "Return JSON with keys: genres (list, from allowed), moods (list, from allowed), "
              "themes (list of short phrases), avoid_genres (list, from allowed), "
              "duration_pref (one of short|medium|long or null).\n\n"
              f"Listener said: \"{free_text}\"")
    out = _json(_call(system, prompt, 1500))
    if isinstance(out, dict):
        out["genres"] = [g for g in out.get("genres", []) if g in GENRES]
        out["avoid_genres"] = [g for g in out.get("avoid_genres", []) if g in GENRES]
        out["moods"] = [m for m in out.get("moods", []) if m in MOODS]
        return out, "groq"
    return _parse_rules(free_text), "rules"


def _parse_rules(text: str) -> dict:
    t = text.lower()
    neg = re.findall(r"(?:not into|no|nothing|hate|avoid|don't like)\s+([a-z\- ]+)", t)
    neg_txt = " ".join(neg)
    syn = {"Sci-Fi": ["sci-fi", "science fiction", "space"], "Mystery": ["mystery", "mysteries", "detective"],
           "Thriller": ["thriller"], "Horror": ["horror", "scary", "gory", "ghost"], "Romance": ["romance", "love"],
           "Fantasy": ["fantasy", "magic"], "Comedy": ["comedy", "funny"], "Drama": ["drama"],
           "History": ["history", "historical"], "Motivation": ["motivation", "inspiring", "self-improvement"]}
    genres, avoid = [], []
    for g, words in syn.items():
        if any(w in neg_txt for w in words):
            avoid.append(g)
        elif any(w in t for w in words):
            genres.append(g)
    dur = "short" if re.search(r"short|quick|under 15", t) else "long" if re.search(r"long|immersive", t) else \
        "medium" if re.search(r"20 min|medium", t) else None
    return {"genres": genres, "moods": [m for m in MOODS if m in t], "themes": [],
            "avoid_genres": avoid, "duration_pref": dur}


# --------------------------------------------------------------------------- 2+3. explanations and intros
def explain_and_intro(profile_summary: str, facts_list: list[dict]) -> tuple[dict, str]:
    """One batched call: for each recommendation, a grounded 'why' and a spoiler-free personalised intro."""
    system = ("You write copy for an audio storytelling app. You are given verified facts about why each story "
              "was recommended. Use ONLY those facts for the explanation - never invent listening history, stats "
              "or plot details. Intros must be spoiler-free, 2-3 sentences, in the story's tone, addressed to the "
              "listener as 'you'. Reply with JSON only.")
    prompt = (f"Listener profile: {profile_summary}\n\nRecommendations (facts):\n{json.dumps(facts_list, indent=1)}\n\n"
              "Rules for 'why': if there is a similar_to_loved signal, start with 'Because you loved <that story title>'; "
              "phrase hook rates naturally (e.g. 'most listeners keep going after episode 1'); use only the signals.\n"
              "Return a JSON object mapping story_id -> {\"why\": one sentence (max 30 words) grounded in the signals, "
              "\"intro\": 2-3 sentence personalised hook (max 60 words)}.")
    out = _json(_call(system, prompt, 6000))
    if isinstance(out, dict) and all(f["story_id"] in out for f in facts_list):
        return out, "groq"
    return {f["story_id"]: {"why": template_why(f), "intro": template_intro(f)} for f in facts_list}, "template"


def template_why(f: dict) -> str:
    bits = []
    for s in f["signals"]:
        t = s["type"]
        if t == "similar_to_loved":
            bits.append(f"it's close to '{s['story']}', which you loved" + (f" (both {', '.join(s['shared'][:2])})" if s["shared"] else ""))
        elif t == "genre_match":
            bits.append(f"it's {f['genre']}, a genre from your {s['source']}")
        elif t == "mood_match":
            bits.append(f"it has the {' and '.join(s['moods'])} feel you enjoy")
        elif t == "length_fit":
            bits.append(s["detail"])
        elif t == "high_hook":
            bits.append(s["detail"])
        elif t == "co_listened":
            bits.append(s["detail"])
        else:
            bits.append(s["detail"])
    return "Recommended because " + "; ".join(bits[:3]) + "."


def template_intro(f: dict) -> str:
    syn = f["synopsis"].split(". Themes")[0]
    ep = "a single episode" if f["episodes"] == 1 else f"{f['episodes']} episodes of about {f['episode_minutes']} minutes"
    return f"{syn}. Press play on '{f['title']}': {ep} of {f['moods'][0]} {f['subgenre']} for you."


# --------------------------------------------------------------------------- 5. LLM-as-judge
def judge(profile_summary: str, facts: dict, why: str, intro: str) -> dict | None:
    system = "You are a strict evaluator of recommendation copy. Reply with JSON only."
    prompt = (f"Listener: {profile_summary}\nFacts: {json.dumps(facts)}\nExplanation: {why}\nIntro: {intro}\n\n"
              "Score 1-5 each: faithfulness (explanation only states things supported by facts), "
              "relevance (does the explanation connect to the listener), intro_appeal (would it make them press play), "
              "spoiler_free (5 = no spoilers). Return JSON {faithfulness, relevance, intro_appeal, spoiler_free, note}.")
    return _json(_call(system, prompt, 1500))
