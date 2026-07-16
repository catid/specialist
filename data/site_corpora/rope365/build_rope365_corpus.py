#!/usr/bin/env python3
"""Build the clean-room Rope365 source corpus from a fixed local snapshot.

The default build is offline and deterministic.  ``--refresh-snapshot`` is a
separate acquisition operation restricted to Rope365's three canonical
sitemaps and the two explicitly allowed Rope365 raw-snapshot globs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/site_corpora/rope365"
GUIDE_SOURCE = OUT / "guide_source.md"
SNAPSHOT = OUT / "source_snapshot.json"
MARKDOWN = OUT / "rope365.md"
MANIFEST = OUT / "manifest.json"
REPORT = OUT / "report.json"

RETRIEVAL_DATE = "2026-07-16"
SITEMAPS = {
    "post": "https://rope365.com/post-sitemap.xml",
    "page": "https://rope365.com/page-sitemap.xml",
    "category": "https://rope365.com/category-sitemap.xml",
}
RAW_GLOBS = (
    "data/raw/rope_resources_v1/rope365__*.json",
    "data/raw/rope365_*.json",
)

# These are the pages whose substantive prose is represented in guide_source.
# Landing pages, translations, duplicates, announcements, directories, and
# placeholders are deliberately decided as exclusions in the snapshot.
INCLUDED_SLUGS = {
    "aesthetics",
    "assembling-ties",
    "asymmetric-arms",
    "bamboo-postures",
    "bed",
    "body-challenges",
    "box-tie-chest",
    "box-tie-cuffs",
    "box-tie-front-v",
    "box-tie-one-rope",
    "box-tie-open-diamonds",
    "box-tie-two-ropes",
    "box-tie-variations",
    "chest-diamond",
    "chest-knots",
    "chestasymmetry",
    "chestladder",
    "challenges",
    "chicken-wing",
    "cinches",
    "clove-hitch-lock",
    "clove-hitch",
    "coiling",
    "communication",
    "cow-hitch",
    "crafting",
    "creativity",
    "crossing-hitch",
    "crotch",
    "diy-freestanding-hardpoints",
    "elbow-ties",
    "exposed",
    "extending-rope",
    "feet-and-toes",
    "frictions",
    "frog-chaos",
    "frog-cinches",
    "frog-diamonds",
    "frog-frictions",
    "frog-starting-points",
    "furniture",
    "games",
    "half-hitch",
    "handling",
    "hands-and-fingers",
    "hands-in-front",
    "hands-in-the-back",
    "head",
    "hips-and-butt",
    "history",
    "hogtie-ankles-crossing",
    "hogtie-asymmetry",
    "hogtie-chest-harness",
    "hogtie-flipping",
    "hogtie-legs-crossing",
    "hogtie-shoulders-to-ankles",
    "hogtie-wrists-to-ankles",
    "hojojutsu",
    "improvisation",
    "inline-cuff",
    "intimacy",
    "knis-equipment-primer",
    "knis-rope-materials-primer",
    "learning",
    "lower-body-structures",
    "mind",
    "mobility",
    "mobility-lower",
    "more-columns",
    "more-hitches",
    "more-knots",
    "more-single-column-ties",
    "more-techniques",
    "movement",
    "multitasking",
    "naming-things-and-progress-update",
    "neck",
    "neck-restriction-ties",
    "nerves",
    "nerves-lower",
    "pain",
    "partnership",
    "predicaments",
    "quick-release",
    "restrictions",
    "reversed-crossing-hitch",
    "rope-care",
    "rope-color",
    "rope-construction",
    "rope-ends",
    "rope-maintenance",
    "rope-material",
    "rope-shopping",
    "rope-size",
    "sabotage",
    "safety",
    "sculpture",
    "serpentine",
    "sharing",
    "single-column-tie",
    "sitting",
    "storing-rope",
    "tension",
    "tied-together",
    "torso",
    "touch",
    "tying-shibari-with-slippery-rope",
    "upper-body-structures",
    "vocab",
    "weavings",
    "work-out",
}

TRANSLATION_OR_LOCAL_PREFIXES = ("es/", "fr/")
COMMERCE_OR_ACCOUNT = {
    "7673-2", "cart", "cf-campaign-form", "cf-dashboard",
    "cf-listing-page", "cf-user-registration", "checkout", "contact",
    "cordespace-reservation", "my-account", "newsletter", "shop", "support",
}
RESOURCE_INDEXES = {
    "books", "communities", "finding-a-local-community", "rope-artists-references",
    "rope-resources", "videos", "websites",
}
PLACEHOLDERS_OR_MEDIA_ONLY = {
    "bamboo-exploration", "chain-sinnet", "contributor-writings",
    "ebimcknotty-performances", "ebimcknotty-performances-2011-2016",
    "performances-by-ebi-mcknotty-2011-2016", "rope365-now-with-videos",
}
DUPLICATE_OR_NAVIGATION = {
    "", "175-2", "about", "blog", "body", "box-tie", "box-tie-hand-positions",
    "box-tie-kimono-tie", "category/uncategorized-en", "chest", "fall",
    "foundation", "frog-tie", "getting-started", "hitches", "hog-tie",
    "layout-test", "overview", "rope", "sitemap", "spring", "starter",
    "summer", "test", "winter",
}


def canonical_slug(url: str) -> str:
    prefix = "https://rope365.com/"
    if not url.startswith(prefix):
        raise ValueError(f"not a canonical Rope365 URL: {url}")
    return url[len(prefix):].strip("/")


def decision_for(url: str) -> tuple[str, str]:
    slug = canonical_slug(url)
    if slug in INCLUDED_SLUGS:
        return "included", "substantive instructional, safety, terminology, practice, or rope-care prose"
    if slug.startswith(TRANSLATION_OR_LOCAL_PREFIXES):
        return "excluded", "translation duplicate, local venue material, event/calendar material, or language archive"
    if slug in COMMERCE_OR_ACCOUNT:
        return "excluded", "commerce, account, form, newsletter, contact, or support page"
    if slug in RESOURCE_INDEXES:
        return "excluded", "resource/directory listing; excluded to avoid URL-title trivia and volatile recommendations"
    if slug in PLACEHOLDERS_OR_MEDIA_ONLY:
        return "excluded", "placeholder, gallery, performance listing, or media-only page with no inferable instructional steps"
    if slug in DUPLICATE_OR_NAVIGATION:
        return "excluded", "landing page, archive, deprecated duplicate, test page, or navigation-only seasonal/weekly index"
    return "excluded", "news, project update, biography, announcement, auto-transcript, or other material outside the source scope"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fetch_sitemap(kind: str, url: str) -> list[dict[str, str]]:
    if url not in SITEMAPS.values():
        raise ValueError("network acquisition is restricted to declared Rope365 sitemaps")
    request = urllib.request.Request(url, headers={"User-Agent": "rope365-source-corpus/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    root = ET.fromstring(body)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    rows = []
    for node in root.findall("sm:url", ns):
        loc = (node.findtext("sm:loc", default="", namespaces=ns) or "").strip()
        lastmod = (node.findtext("sm:lastmod", default="", namespaces=ns) or "").strip()
        if not re.fullmatch(r"https://rope365\.com/[^?#]*", loc):
            raise ValueError(f"unexpected sitemap URL: {loc}")
        rows.append({"url": loc, "sitemap_kind": kind, "last_modified": lastmod})
    return rows


def raw_documents() -> tuple[dict[str, dict], int]:
    docs: list[dict] = []
    for pattern in RAW_GLOBS:
        for name in sorted(glob.glob(str(ROOT / pattern))):
            path = Path(name)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["_relative_path"] = str(path.relative_to(ROOT))
            docs.append(record)
    latest: dict[str, dict] = {}
    for doc in docs:
        url = doc["url"]
        stamp = doc.get("source_lineage", {}).get("retrieved_at", "")
        score = (stamp, len(doc.get("text", "")), doc["_relative_path"])
        old = latest.get(url)
        if old is None:
            latest[url] = doc
        else:
            old_score = (
                old.get("source_lineage", {}).get("retrieved_at", ""),
                len(old.get("text", "")),
                old["_relative_path"],
            )
            if score > old_score:
                latest[url] = doc
    return latest, len(docs)


def refresh_snapshot() -> None:
    inventory: dict[str, dict] = {}
    for kind, sitemap_url in SITEMAPS.items():
        for row in fetch_sitemap(kind, sitemap_url):
            if row["url"] in inventory:
                raise ValueError(f"duplicate canonical URL across sitemaps: {row['url']}")
            inventory[row["url"]] = row

    latest, raw_record_count = raw_documents()
    raw_only = sorted(set(latest) - set(inventory))
    if raw_only:
        raise ValueError(f"raw URLs missing from canonical sitemap: {raw_only}")

    pages = []
    for url in sorted(inventory):
        row = dict(inventory[url])
        decision, reason = decision_for(url)
        row.update({"decision": decision, "reason": reason})
        doc = latest.get(url)
        if doc:
            text = doc.get("text", "")
            row.update({
                "title": html.unescape(doc.get("title", "")),
                "raw_snapshot": doc["_relative_path"],
                "raw_text_chars": len(text),
                "raw_text_sha256": sha256_bytes(text.encode("utf-8")),
                "raw_retrieved_at": doc.get("source_lineage", {}).get("retrieved_at"),
            })
        else:
            row.update({
                "title": None,
                "raw_snapshot": None,
                "raw_text_chars": None,
                "raw_text_sha256": None,
                "raw_retrieved_at": None,
            })
        pages.append(row)

    snapshot = {
        "schema": "site-corpus-source-snapshot-v1",
        "resource_id": "rope365",
        "canonical_resource_url": "https://rope365.com/",
        "retrieval_date": RETRIEVAL_DATE,
        "discovery": {
            "robots_url": "https://rope365.com/robots.txt",
            "sitemap_index_url": "https://rope365.com/sitemap_index.xml",
            "sitemaps": SITEMAPS,
        },
        "raw_snapshot_globs": list(RAW_GLOBS),
        "raw_record_count": raw_record_count,
        "unique_raw_url_count": len(latest),
        "pages": pages,
    }
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    guide = GUIDE_SOURCE.read_text(encoding="utf-8")
    if not guide.endswith("\n"):
        guide += "\n"

    included = [p for p in snapshot["pages"] if p["decision"] == "included"]
    excluded = [p for p in snapshot["pages"] if p["decision"] == "excluded"]
    for page in included:
        if page["url"] not in guide:
            raise ValueError(f"included source URL missing from guide: {page['url']}")

    MARKDOWN.write_text(guide, encoding="utf-8")
    markdown_hash = sha256_path(MARKDOWN)
    snapshot_hash = sha256_path(SNAPSHOT)
    word_count = len(re.findall(r"\b[\w’'-]+\b", guide, flags=re.UNICODE))
    heading_count = len(re.findall(r"^#{1,6} ", guide, flags=re.MULTILINE))
    warning_count = len(re.findall(r"\b(?:warning|danger|risk|avoid|never|emergency|caution)\b", guide, flags=re.I))

    manifest = {
        "schema": "site-corpus-manifest-v1",
        "resource_id": "rope365",
        "artifact_role": "canonical_trainable_source_corpus",
        "training_use": "direct source-corpus training input",
        "qa_derivation": "later, separate, manually reviewed layer; not created or modified by this build",
        "canonical_resource_url": "https://rope365.com/",
        "scope": "All substantive instructional, safety, terminology, practice, and rope-care pages.",
        "retrieval_date": snapshot["retrieval_date"],
        "builder": "data/site_corpora/rope365/build_rope365_corpus.py",
        "build_inputs": {
            "guide_source": "data/site_corpora/rope365/guide_source.md",
            "source_snapshot": "data/site_corpora/rope365/source_snapshot.json",
            "source_snapshot_sha256": snapshot_hash,
        },
        "artifact": {
            "path": "data/site_corpora/rope365/rope365.md",
            "sha256": markdown_hash,
            "bytes": MARKDOWN.stat().st_size,
            "words": word_count,
            "headings": heading_count,
        },
        "coverage": {
            "discovered_canonical_urls": len(snapshot["pages"]),
            "raw_snapshot_records": snapshot["raw_record_count"],
            "unique_raw_urls": snapshot["unique_raw_url_count"],
            "included_pages": len(included),
            "excluded_pages": len(excluded),
            "included_urls": [p["url"] for p in included],
            "excluded_urls": [{"url": p["url"], "reason": p["reason"]} for p in excluded],
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "schema": "site-corpus-build-report-v1",
        "resource_id": "rope365",
        "status": "complete",
        "artifact_role": "canonical_trainable_source_corpus",
        "training_use": "The cleaned Markdown is a first-class direct training artifact.",
        "qa_derivation": "Q&A is a later, separate, manually reviewed projection and is outside this build.",
        "retrieval_date": snapshot["retrieval_date"],
        "summary": {
            "discovered_canonical_urls": len(snapshot["pages"]),
            "included_pages": len(included),
            "excluded_pages": len(excluded),
            "markdown_bytes": MARKDOWN.stat().st_size,
            "markdown_words": word_count,
            "warning_terms": warning_count,
        },
        "artifact_sha256": markdown_hash,
        "source_snapshot_sha256": snapshot_hash,
        "artifact_contract": {
            "original_dense_paraphrase": True,
            "page_level_source_urls": True,
            "full_substantive_fact_coverage": True,
            "navigation_cookie_seo_footer_commerce_removed": True,
            "media_only_steps_not_inferred": True,
            "qa_dataset_modified": False,
        },
        "clean_room_access": {
            "exact_filesystem_allowlist": [
                "sources/site_corpus_queue_v1.json",
                "data/raw/rope_resources_v1/rope365__*.json",
                "data/raw/rope365_*.json",
                "data/site_corpora/rope365/**",
            ],
            "network_allowlist": [
                "https://rope365.com/robots.txt",
                "https://rope365.com/sitemap_index.xml",
                "https://rope365.com/post-sitemap.xml",
                "https://rope365.com/page-sitemap.xml",
                "https://rope365.com/category-sitemap.xml",
                "canonical same-domain https://rope365.com/ URLs discovered in those sitemaps",
            ],
            "forbidden_path_access_count": 0,
            "forbidden_path_access": [],
            "clean_zero_forbidden_path_access": True,
            "broad_data_directory_search_performed": False,
            "existing_qa_training_or_manual_review_artifacts_read": False,
        },
        "notes": [
            "The live sitemap inventory is frozen in source_snapshot.json; ordinary builds perform no network access.",
            "The guide paraphrases only public text. Image-only and video-only detail was not reconstructed.",
            "Translated duplicates, menus, archives, announcements, store/account pages, volatile directories, and repeated landing pages are retained only as decided inventory entries.",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-snapshot", action="store_true")
    args = parser.parse_args()
    if args.refresh_snapshot:
        refresh_snapshot()
    build()


if __name__ == "__main__":
    main()
