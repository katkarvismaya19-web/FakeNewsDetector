"""Source retrieval + validation.
1. Google Fact Check Tools API  -> published fact-checks (BOOM, Vishvas News, Alt News, PIB Fact Check, etc.)
2. Google News RSS (Hindi or English, matched to input)      -> whether mainstream outlets carry the same story
3. Multilingual embeddings       -> keep only sources that are actually about the same claim
"""
import importlib.util
import os
import urllib.parse

import feedparser
import requests

from data import detect_lang
from config import EMBED_MODEL, FACTCHECK_API_KEY, SIM_THRESHOLD, TFIDF_THRESHOLD

FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
FAKE_WORDS = ["false", "fake", "misleading", "incorrect", "not true", "untrue", "no evidence", "manipulated",
              "altered", "fabricated", "satire", "scam", "hoax", "फर्जी", "फ़र्ज़ी", "गलत", "ग़लत", "भ्रामक",
              "झूठ", "झूठा", "अफवाह", "फेक", "फ़ेक", "भ्रम"]
TRUE_WORDS = ["true", "correct", "accurate", "सही", "सच", "सत्य"]
_embedder = None
# "embed" = multilingual sentence embeddings (better, also matches Hindi claim to English fact-check);
# "tfidf" = character n-grams (no extra model, fits free hosting). Auto-picks based on what's installed.
SIM_BACKEND = os.getenv("SIM_BACKEND") or ("embed" if importlib.util.find_spec("sentence_transformers") else "tfidf")


def _emb():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def similarities(text, candidates):
    if SIM_BACKEND == "embed":
        from sentence_transformers import util
        return [float(s) for s in util.cos_sim(_emb().encode(text), _emb().encode(candidates))[0]], SIM_THRESHOLD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    query = make_query(text, 25)
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform([query] + candidates)
    return [float(s) for s in cosine_similarity(X[0], X[1:])[0]], TFIDF_THRESHOLD


def stance(text):
    t = text.lower()
    if any(w in t for w in FAKE_WORDS):
        return "FAKE"
    if any(w in t for w in TRUE_WORDS):
        return "REAL"
    return "UNCLEAR"


def make_query(text, n_words=12):
    return " ".join(text.split()[:n_words])


def factcheck_search(query, lang=None):
    if not FACTCHECK_API_KEY:
        return []
    params = {"query": query, "pageSize": 10, "key": FACTCHECK_API_KEY}
    if lang:
        params["languageCode"] = lang
    try:
        r = requests.get(FACTCHECK_URL, params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException:
        return []
    out = []
    for c in r.json().get("claims", []):
        for rev in c.get("claimReview", []):
            out.append({"type": "fact-check", "text": c.get("text", ""), "rating": rev.get("textualRating", ""),
                        "publisher": rev.get("publisher", {}).get("name", ""), "url": rev.get("url", ""),
                        "date": rev.get("reviewDate", "")[:10]})
    return out


def news_search(query, lang="hi", n=10):
    loc = "hl=hi&gl=IN&ceid=IN:hi" if lang == "hi" else "hl=en-IN&gl=IN&ceid=IN:en"
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&" + loc
    try:
        feed = feedparser.parse(url)
    except Exception:
        return []
    return [{"type": "news", "text": e.get("title", ""), "rating": "", "url": e.get("link", ""),
             "publisher": (e.get("source") or {}).get("title", ""), "date": e.get("published", "")[:16]}
            for e in feed.entries[:n]]


def verify(text):
    """Returns evidence verdict, evidence P(fake), and matched sources with similarity scores."""
    q, lang = make_query(text), detect_lang(text)
    items, seen = [], set()
    for it in factcheck_search(q, lang) + factcheck_search(q) + news_search(q, lang):
        if it["url"] and it["url"] not in seen:
            seen.add(it["url"])
            items.append(it)
    if not items:
        return {"verdict": "UNVERIFIED", "prob_fake": None, "strength": None, "sources": []}

    sims, threshold = similarities(text, [i["text"] for i in items])
    for it, s in zip(items, sims):
        it["similarity"] = round(float(s), 3)
        it["stance"] = stance(it["rating"] or it["text"])
    items.sort(key=lambda i: (i["type"] != "fact-check", -i["similarity"]))
    rel = [i for i in items if i["similarity"] >= threshold]

    fc = [i for i in rel if i["type"] == "fact-check" and i["stance"] != "UNCLEAR"]
    if fc:
        top = max(fc, key=lambda i: i["similarity"])
        s = min(top["similarity"] / threshold * 0.6, 1.0)  # normalise so both backends scale alike
        p = 0.5 + 0.45 * s if top["stance"] == "FAKE" else 0.5 - 0.45 * s
        return {"verdict": top["stance"], "prob_fake": p, "strength": "fact-check", "sources": rel}

    news = [i for i in rel if i["type"] == "news"]
    debunks = [i for i in news if i["stance"] == "FAKE"]
    if debunks:
        return {"verdict": "FAKE", "prob_fake": 0.75, "strength": "news-debunk", "sources": rel}
    if len({i["publisher"] for i in news}) >= 2:
        return {"verdict": "REAL", "prob_fake": 0.30, "strength": "news-corroborated", "sources": rel}
    return {"verdict": "UNVERIFIED", "prob_fake": None, "strength": None, "sources": rel or items[:5]}


def combine(model_p, ev):
    """Final P(fake): published fact-checks dominate; news coverage is a weaker signal."""
    if ev["prob_fake"] is None:
        return model_p
    w = {"fact-check": 0.7, "news-debunk": 0.5, "news-corroborated": 0.3}[ev["strength"]]
    return w * ev["prob_fake"] + (1 - w) * model_p
