#!/usr/bin/env python3
"""Phase 3 job: daily media-skeptic digest.

Fetches recent headlines from a spread of mainstream outlets, runs each
through a skeptical media-literacy persona via the local inference API
(read-only, low-stakes per the brief's Phase 3 guidance), and posts the
result into Open WebUI as a Note. Scheduled by media-digest.timer.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
import trafilatura

sys.path.insert(0, os.path.dirname(__file__))
import jobqueue

JOBS_DB = "/var/lib/agent-forge/jobs.db"
LLAMA_API = "http://127.0.0.1:8090/v1/chat/completions"
OPENWEBUI_API = "http://127.0.0.1:3000/api/v1/notes/create"
MODEL = "qwen2.5-7b"
CLASSIFY_MODEL = "qwen2.5-3b"
ARTICLES_PER_FEED = 4
MAX_AGE_HOURS = 24
MIN_FULLTEXT_CHARS = 400
FULLTEXT_CHAR_LIMIT = 4000

# Fixed taxonomy classified by the fast model before the expensive analysis
# step runs -- cheap classification gates expensive work, not the other way
# round. Order here is the section order in the rendered digest.
CATEGORY_ORDER = ["Politics", "War & Conflict", "Economy", "World"]
EXCLUDED_CATEGORIES = {"Sports", "Celebrity"}
ALL_CATEGORIES = CATEGORY_ORDER + list(EXCLUDED_CATEGORIES)

CLASSIFY_PROMPT = (
    "Classify this news headline into exactly one category: "
    + ", ".join(ALL_CATEGORIES)
    + ". Respond with only the category name, nothing else."
)

FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "NYT": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Fox News": "https://moxie.foxnews.com/google-publisher/latest.xml",
}

PERSONA_PROMPT = """You are a media literacy analyst. You do not trust mainstream framing by default -- you treat every headline as a claim to interrogate, not a fact to accept. For the article given, respond with three parts:

**What happened** -- a strictly neutral, factual summary, stripped of the original's loaded language.

**The intended takeaway** -- what conclusion or emotional reaction the framing/word choice/omissions seem designed to produce in the reader.

**Worth asking** -- 2-3 specific, concrete questions about why this framing, timing, or emphasis was chosen, and what's absent from the piece.

IMPORTANT: Use ONLY the names, titles, and facts given in the article text below. Do not substitute any other name or role, even if a different name feels more familiar or commonly associated with this type of story or role. If the text does not specify something, say the text does not specify it -- do not guess or fill in from outside knowledge.

Ground everything in specifics from the article -- no vague suspicion, no reflexive contrarianism for its own sake. The goal is sharper thinking, not a conspiracy theory."""


def load_api_key(env_path, var_name):
    with open(env_path) as f:
        for line in f:
            if line.startswith(f"{var_name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{var_name} not found in {env_path}")


def fetch_recent_articles():
    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    articles = []
    feed_errors = {}
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
            if parsed.bozo and not parsed.entries:
                raise RuntimeError(str(parsed.bozo_exception))
            entries = []
            for entry in parsed.entries:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    published_dt = datetime(*published[:6], tzinfo=timezone.utc)
                    if published_dt < cutoff:
                        continue
                entries.append(entry)
            entries = entries[:ARTICLES_PER_FEED]
            for entry in entries:
                articles.append({
                    "source": source,
                    "title": entry.get("title", "(no title)"),
                    "url": entry.get("link", ""),
                    "summary": entry.get("summary", entry.get("description", "")),
                })
        except Exception as e:
            feed_errors[source] = str(e)
    return articles, feed_errors


def classify_article(client, api_key, article):
    """Cheap fast-model classification, used to gate the expensive analysis
    step -- filtered-out articles skip both full-text fetch and analysis
    entirely, so this makes the whole run faster, not slower."""
    user_msg = f"Title: {article['title']}\nSummary: {article['summary']}"
    payload = {
        "model": CLASSIFY_MODEL,
        "messages": [
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 10,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = client.post(LLAMA_API, json=payload, headers=headers, timeout=60.0)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        for cat in ALL_CATEGORIES:
            if cat.lower() in raw.lower():
                return cat
        return "World"
    except Exception as e:
        print(f"classify_article failed for {article['title']!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return "World"


def fetch_full_text(url):
    """Best-effort full article text via trafilatura. Falls back to the RSS
    summary (handled by the caller) if extraction fails or comes back thin --
    some sites paywall or block scraping, and a one-sentence RSS blurb is
    still better than nothing."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded)
        if not text or len(text) < MIN_FULLTEXT_CHARS:
            return None
        return text[:FULLTEXT_CHAR_LIMIT]
    except Exception as e:
        print(f"fetch_full_text failed for {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def analyze_article(client, api_key, article):
    body = article.get("full_text") or article["summary"]
    body_label = "Full article text" if article.get("full_text") else "Summary"
    user_msg = (
        f"Source: {article['source']}\n"
        f"Title: {article['title']}\n"
        f"{body_label}: {body}\n"
        f"URL: {article['url']}\n\n"
        "Analyze this."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PERSONA_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 500,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    last_error = None
    for attempt in range(2):
        try:
            start = time.monotonic()
            resp = client.post(LLAMA_API, json=payload, headers=headers, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            latency_ms = int((time.monotonic() - start) * 1000)
            return content, latency_ms, None
        except Exception as e:
            last_error = str(e)
    return None, None, last_error


def render_digest(results, feed_errors, excluded_count):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"# Media Skeptic Digest -- {today}\n"]
    if feed_errors:
        lines.append("_Feeds that failed to fetch: " + ", ".join(feed_errors) + "_\n")
    if excluded_count:
        lines.append(f"_{excluded_count} sports/celebrity article(s) filtered out before analysis._\n")

    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    for category in CATEGORY_ORDER:
        articles = by_category.get(category)
        if not articles:
            continue
        lines.append(f"## {category}\n")
        for a in articles:
            lines.append(f"### [{a['title']}]({a['url']}) -- _{a['source']}_\n")
            if a.get("analysis"):
                lines.append(a["analysis"] + "\n")
            else:
                lines.append(f"_Analysis failed: {a.get('error', 'unknown error')}_\n")
    return "\n".join(lines)


def post_to_openwebui(api_key, title, markdown):
    payload = {"title": title, "data": {"content": {"md": markdown}}}
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = httpx.post(OPENWEBUI_API, json=payload, headers=headers, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def main():
    jobqueue.init_db(JOBS_DB)
    llama_key = load_api_key("/etc/agent-forge/llama-swap.env", "LLAMA_SWAP_API_KEY")
    openwebui_key = load_api_key("/etc/agent-forge/openwebui.env", "OPENWEBUI_API_KEY")

    job_id = jobqueue.create_job(JOBS_DB, "media_digest", {"feeds": list(FEEDS.keys())})
    start = jobqueue.start_job(JOBS_DB, job_id)

    articles, feed_errors = fetch_recent_articles()
    results = []
    excluded = []

    try:
        with httpx.Client() as client:
            for article in articles:
                article["category"] = classify_article(client, llama_key, article)

            survivors = [a for a in articles if a["category"] not in EXCLUDED_CATEGORIES]
            excluded = [a for a in articles if a["category"] in EXCLUDED_CATEGORIES]

            for article in survivors:
                article["full_text"] = fetch_full_text(article["url"])
                analysis, latency_ms, error = analyze_article(client, llama_key, article)
                results.append({
                    **article,
                    "analysis": analysis,
                    "error": error,
                    "latency_ms": latency_ms,
                })

        digest_md = render_digest(results, feed_errors, len(excluded))
        title = f"Media Skeptic Digest -- {datetime.now().strftime('%Y-%m-%d')}"
        note = post_to_openwebui(openwebui_key, title, digest_md)

        jobqueue.finish_job(JOBS_DB, job_id, start, {
            "articles_processed": len(results),
            "articles_excluded": [{"title": a["title"], "category": a["category"]} for a in excluded],
            "feed_errors": feed_errors,
            "note_id": note.get("id"),
            "results": results,
        })
        print(f"Job {job_id} done: {len(results)} articles ({len(excluded)} filtered out), note id {note.get('id')}")
    except Exception as e:
        jobqueue.fail_job(JOBS_DB, job_id, start, e, {"results": results, "feed_errors": feed_errors})
        print(f"Job {job_id} failed: {e}")
        raise


if __name__ == "__main__":
    main()
