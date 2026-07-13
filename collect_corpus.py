#!/usr/bin/env python
"""Collect a public-web educational corpus on rope bondage (shibari/kinbaku).

Sources: Wikipedia (CC BY-SA via API) + open sites whose robots.txt permits
crawling (checked 2026-07-10; theduchy.com signals ai-train=no and is excluded).
Output: data/raw/*.json  (one doc per file: {source, url, title, text})
"""
import json, re, time, hashlib, sys
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

OUT = Path("/home/catid/specialist/data/raw")
OUT.mkdir(parents=True, exist_ok=True)
UA = "specialist-research-collector/0.1 (contact: kuang2@kuang2.ai; respectful crawl, 1 req/s)"
S = requests.Session()
S.headers["User-Agent"] = UA

WIKI_TITLES = [
    "Japanese bondage", "Bondage (BDSM)", "Suspension bondage", "Hojōjutsu",
    "Bondage positions and methods", "Self-bondage", "Breast bondage",
    "Crotch rope", "Frog tie", "Hogtie (BDSM)", "Over-arm tie", "Strappado",
    "Bondage rigger", "Rope", "Jute", "Hemp", "Sisal", "Bondage rope",
    "Risk-aware consensual kink", "Safeword", "Safe, sane and consensual",
    "Positional asphyxia", "Radial nerve", "Ulnar nerve", "Median nerve",
    "Brachial plexus", "Radial neuropathy", "BDSM", "Kink (sexuality)",
    "Rope splicing", "Whipping knot", "Seizing", "Knot", "Lark's head knot",
    "Bowline", "Somerville Bowline", "Sadomasochism", "Erotic asphyxiation",
    "Munter hitch", "Carrick bend", "Sheet bend", "Square knot", "Half hitch",
    "Timber hitch", "Kinbaku artists", "Seiu Ito", "Dan Oniroku",
]

def save(source, url, title, text):
    if not text or len(text) < 400:
        return False
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    (OUT / f"{source}_{h}.json").write_text(json.dumps(
        {"source": source, "url": url, "title": title, "text": text.strip()},
        ensure_ascii=False))
    return True

def wikipedia():
    n = 0
    for title in WIKI_TITLES:
        try:
            r = S.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "prop": "extracts", "explaintext": 1,
                "redirects": 1, "format": "json", "titles": title}, timeout=30)
            pages = r.json()["query"]["pages"]
            for p in pages.values():
                if "extract" in p and save("wikipedia", f"https://en.wikipedia.org/wiki/{p['title'].replace(' ', '_')}", p["title"], p["extract"]):
                    n += 1
        except Exception as e:
            print(f"  wiki fail {title}: {e}", file=sys.stderr)
        time.sleep(0.5)
    print(f"wikipedia: {n} docs")

def sitemap_urls(url, seen=None):
    seen = seen if seen is not None else set()
    if url in seen or len(seen) > 4000:
        return []
    seen.add(url)
    try:
        xml = S.get(url, timeout=30).text
    except Exception:
        return []
    subs = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml)
    urls = []
    for u in subs:
        u = u.strip()
        if u.endswith(".xml") and "sitemap" in u:
            urls += sitemap_urls(u, seen)
        else:
            urls.append(u)
    return urls

SKIP_PAT = re.compile(
    r"/(tag|category|author|cart|checkout|shop|product|wp-content|feed|comments|"
    r"privacy|terms|contact|about-me|account|login|register)[/-]|"
    r"\.(jpg|jpeg|png|gif|webp|mp4|pdf)$", re.I)

def crawl_site(name, base, url_filter=None, cap=400):
    urls = sitemap_urls(base)
    urls = [u for u in dict.fromkeys(urls) if not SKIP_PAT.search(u)]
    if url_filter:
        urls = [u for u in urls if url_filter(u)]
    urls = urls[:cap]
    print(f"{name}: {len(urls)} candidate urls")
    n = 0
    for u in urls:
        try:
            html = S.get(u, timeout=30).text
            text = trafilatura.extract(html, include_comments=False,
                                       include_tables=True, favor_precision=True)
            title = (re.search(r"<title[^>]*>(.*?)</title>", html, re.S) or [None, ""])[1]
            title = re.sub(r"\s+", " ", title).strip()[:200]
            if save(name, u, title, text or ""):
                n += 1
        except Exception as e:
            print(f"  {name} fail {u}: {e}", file=sys.stderr)
        time.sleep(0.8)
    print(f"{name}: saved {n} docs")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "wiki"):
        wikipedia()
    if which in ("all", "rope365"):
        crawl_site("rope365", "https://rope365.com/sitemap_index.xml")
    if which in ("all", "esinem"):
        crawl_site("esinem", "https://esinem.com/sitemap.xml")
    if which in ("all", "kinbakutoday"):
        crawl_site("kinbakutoday", "https://kinbakutoday.com/sitemap.xml")
    if which in ("all", "theropegeek"):
        crawl_site("theropegeek", "https://www.theropegeek.com/sitemap.xml")
    print("done")
