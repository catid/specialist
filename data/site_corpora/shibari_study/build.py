#!/usr/bin/env python3
"""Build the policy-excluded Shibari Study artifact deterministically."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
REASON = "terms_of_service_prohibits_content_collection_and_storage"
ROBOTS_URL = "https://shibaristudy.com/robots.txt"
INDEX_URL = "https://shibaristudy.com/sitemap-index.xml"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def locs(path: Path) -> list[str]:
    root = ET.fromstring(path.read_bytes())
    return [node.text.strip() for node in root.findall(f".//{{{NS}}}loc") if node.text]


def build() -> None:
    provenance = json.loads((SNAPSHOT / "provenance.json").read_text(encoding="utf-8"))
    policy = json.loads((ROOT / "policy_decision.json").read_text(encoding="utf-8"))
    if provenance["capture_scope"] != "robots_and_sitemap_metadata_only":
        raise RuntimeError("snapshot scope is not metadata-only")
    if provenance["canonical_page_body_requests"] or provenance["canonical_page_body_snapshots_retained"]:
        raise RuntimeError("metadata capture touched a canonical page body")
    if policy["decision_reason"] != REASON or policy["canonical_instructional_body_snapshots_retained"]:
        raise RuntimeError("policy decision does not prove zero retained instructional bodies")
    resources = provenance["resources"]
    by_url = {item["url"]: item for item in resources}
    for item in resources:
        path = ROOT / item["path"]
        data = path.read_bytes()
        if len(data) != item["byte_length"] or digest(data) != item["sha256"]:
            raise RuntimeError(f"snapshot mismatch: {path}")
    robots = (SNAPSHOT / "robots.txt").read_text(encoding="utf-8")
    for line in ("Allow: /programs/", "Allow: /pages/", "Sitemap: https://shibaristudy.com/sitemap-index.xml"):
        if line not in robots:
            raise RuntimeError(f"robots audit signal missing: {line}")
    child_urls = set(locs(SNAPSHOT / "sitemap-index.xml"))
    captured_children = {item["url"] for item in resources if item["role"] == "child_sitemap"}
    if child_urls != captured_children:
        raise RuntimeError("sitemap index and captured children differ")

    memberships: dict[str, set[str]] = defaultdict(set)
    sitemap_counts: dict[str, int] = {}
    for sitemap_url in sorted(child_urls):
        urls = locs(ROOT / by_url[sitemap_url]["path"])
        sitemap_counts[sitemap_url] = len(urls)
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "shibaristudy.com" or parsed.query or parsed.fragment:
                raise RuntimeError(f"noncanonical sitemap URL: {url}")
            memberships[url].add(sitemap_url)
    inventory = [
        {
            "url": url,
            "discovered_in": sorted(found_in),
            "disposition": "excluded",
            "reason": REASON,
            "content_retrieved_for_corpus": False,
            "direct_training_included": False,
        }
        for url, found_in in sorted(memberships.items())
    ]
    write(ROOT / "inventory.jsonl", "".join(stable_json(item) + "\n" for item in inventory))
    dispositions = [
        {
            "url": item["url"],
            "disposition": "retrieved_metadata",
            "role": item["role"],
            "sha256": item["sha256"],
        }
        for item in sorted(resources, key=lambda value: value["url"])
    ] + inventory
    write(ROOT / "url_dispositions.jsonl", "".join(stable_json(item) + "\n" for item in dispositions))
    write(ROOT / "content_records.jsonl", "")

    completed = provenance["completed_at"]
    corpus = f"""# Shibari Study corpus access notice

**Status:** excluded from direct training. **Content records:** 0. **Direct-training ready:** false. **Non-QA:** true.

The access audit found that the live [Terms and Conditions](https://shibaristudy.com/pages/terms-conditions) grant personal, noncommercial access and prohibit storing or copying Service content, compiling it into a database, and automated or manual content collection. Although the [robots policy]({ROBOTS_URL}) permits crawling selected public routes, crawl permission does not override those use restrictions. The resulting disposition is `{REASON}`.

Only robots and XML sitemap metadata were retained. No tutorial, blog, FAQ, glossary, description, transcript, subtitle, image, video, lesson, member material, procedure, or factual paraphrase was captured for training. `content_records.jsonl` is intentionally empty, and the sitemap inventory is compliance metadata only.

No taxonomy mapping or instruction inference was attempted. A future corpus requires explicit authorization or a materially changed policy, followed by a new review and a source-document split assigned before any Markdown chunk or derived QA is created.
"""
    write(ROOT / "CORPUS.md", corpus)
    report = f"""# Shibari Study exclusion build report

- Metadata snapshot completed: `{completed}`
- Policy decision: `{REASON}`
- Direct-training ready: `false`
- Canonical instructional body snapshots retained: `0`
- Content records: `0`
- Unique sitemap URLs inventoried and excluded: `{len(inventory)}`
- Sitemap metadata files captured: `{len(child_urls) + 1}`
- Robots metadata files captured: `1`
- Taxonomy mappings: `0`
- Video or image inference: `none`

## Sitemap membership counts

These counts prove metadata coverage only. URL titles and slugs are not training material.

""" + "".join(f"- `{url}`: {count}\n" for url, count in sorted(sitemap_counts.items()))
    write(ROOT / "REPORT.md", report)
    output_names = ("CORPUS.md", "REPORT.md", "content_records.jsonl", "inventory.jsonl", "url_dispositions.jsonl")
    manifest = {
        "schema_version": 1,
        "resource_id": "shibari_study",
        "artifact_role": "policy_exclusion_notice",
        "generated_at": completed,
        "policy_reason": REASON,
        "direct_training_ready": False,
        "non_qa": True,
        "content_record_count": 0,
        "inventory_record_count": len(inventory),
        "canonical_instructional_body_snapshots_retained": 0,
        "source_capture_scope": provenance["capture_scope"],
        "taxonomy_mappings": [],
        "document_disjoint_requirement": "assign canonical source document before Markdown chunking or QA derivation",
        "policy_decision": {
            "path": "policy_decision.json",
            "sha256": digest((ROOT / "policy_decision.json").read_bytes()),
        },
        "source_snapshot": [
            {
                "url": item["url"],
                "path": item["path"],
                "retrieved_at": item["retrieved_at"],
                "sha256": item["sha256"],
                "byte_length": item["byte_length"],
            }
            for item in sorted(resources, key=lambda value: value["url"])
        ],
        "outputs": {
            name: {"sha256": digest((ROOT / name).read_bytes()), "byte_length": (ROOT / name).stat().st_size}
            for name in output_names
        },
    }
    write(ROOT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build()
