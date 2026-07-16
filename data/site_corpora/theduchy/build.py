#!/usr/bin/env python3
"""Build the policy-excluded The Duchy corpus artifact deterministically."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
REASON = "robots_and_content_signal_ai_training_disallowed"
ROBOTS_URL = "https://www.theduchy.com/robots.txt"
INDEX_URL = "https://www.theduchy.com/sitemap_index.xml"
POLICY_LINE = "Content-Signal: search=yes,ai-train=no,use=reference"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def locs(data: bytes) -> list[str]:
    root = ET.fromstring(data)
    return [node.text.strip() for node in root.findall(f".//{{{SITEMAP_NS}}}loc") if node.text]


def load_and_verify_snapshot() -> tuple[dict[str, object], list[dict[str, object]]]:
    provenance = json.loads((SNAPSHOT / "provenance.json").read_text(encoding="utf-8"))
    if provenance["capture_scope"] != "robots_and_sitemap_metadata_only" or provenance["page_body_requests"] != 0:
        raise RuntimeError("snapshot provenance does not prove metadata-only capture")
    resources = provenance["resources"]
    for resource in resources:
        path = ROOT / resource["path"]
        data = path.read_bytes()
        if digest(data) != resource["sha256"] or len(data) != resource["byte_length"]:
            raise RuntimeError(f"snapshot hash/length mismatch: {path}")
    robots = (SNAPSHOT / "robots.txt").read_text(encoding="utf-8")
    for required in (POLICY_LINE, "User-agent: GPTBot", "Disallow: /"):
        if required not in robots:
            raise RuntimeError(f"required robots policy is absent: {required}")
    return provenance, resources


def build() -> None:
    provenance, resources = load_and_verify_snapshot()
    resource_by_url = {resource["url"]: resource for resource in resources}
    robots_resource = resource_by_url[ROBOTS_URL]
    index_urls = locs((SNAPSHOT / "sitemap_index.xml").read_bytes())
    if set(index_urls) != {resource["url"] for resource in resources if resource["role"] == "child_sitemap"}:
        raise RuntimeError("sitemap-index resources and captured child sitemaps differ")

    discovered_in: dict[str, set[str]] = defaultdict(set)
    sitemap_counts: dict[str, int] = {}
    for sitemap_url in sorted(index_urls):
        resource = resource_by_url[sitemap_url]
        urls = locs((ROOT / resource["path"]).read_bytes())
        sitemap_counts[sitemap_url] = len(urls)
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in {"theduchy.com", "www.theduchy.com"}:
                raise RuntimeError(f"noncanonical sitemap URL: {url}")
            discovered_in[url].add(sitemap_url)

    inventory_records = [
        {
            "url": url,
            "discovered_in": sorted(source_urls),
            "disposition": "excluded",
            "reason": REASON,
            "content_retrieved": False,
            "direct_training_included": False,
        }
        for url, source_urls in sorted(discovered_in.items())
    ]
    inventory_text = "".join(stable_json(record) + "\n" for record in inventory_records)
    write_text(ROOT / "inventory.jsonl", inventory_text)

    dispositions = [
        {
            "url": resource["url"],
            "disposition": "retrieved_metadata",
            "role": resource["role"],
            "sha256": resource["sha256"],
        }
        for resource in sorted(resources, key=lambda item: item["url"])
    ] + inventory_records
    dispositions_text = "".join(stable_json(record) + "\n" for record in dispositions)
    write_text(ROOT / "url_dispositions.jsonl", dispositions_text)

    # The intentionally empty file is the machine-readable direct-training set.
    content_text = ""
    write_text(ROOT / "content_records.jsonl", content_text)

    captured = provenance["completed_at"]
    corpus = f"""# The Duchy corpus access notice

**Status:** excluded from direct training. **Content records:** 0. **Direct-training ready:** false. **Non-QA:** true.

The metadata snapshot completed at `{captured}`. The retrieved [robots policy]({ROBOTS_URL}) declares `{POLICY_LINE}` and separately disallows GPTBot site-wide. That policy conflicts with producing a direct-training corpus, so no canonical page, tutorial, post, media asset, gated route, title, or body text was requested, copied, summarized, paraphrased, or inferred.

The [sitemap index]({INDEX_URL}) and its listed XML sitemaps were read only to create a disposition inventory. Every discovered canonical URL is marked `{REASON}` in `inventory.jsonl`; the inventory is compliance metadata, not training content. The exact robots response and XML snapshots, retrieval times, and hashes are retained in `source_snapshot/`.

Media and access gates were not assessed because doing so would require canonical-page retrieval. No UI text, image-only procedure, video instruction, purchase material, member material, or category mapping is represented.
"""
    write_text(ROOT / "CORPUS.md", corpus)

    report = f"""# The Duchy exclusion build report

- Snapshot completed: `{captured}`
- Robots policy retrieved: `{robots_resource['retrieved_at']}`
- Robots policy SHA-256: `{robots_resource['sha256']}`
- Policy disposition: `{REASON}`
- Direct-training ready: `false`
- Non-QA: `true`
- Canonical page-body requests: `0`
- Content records: `0`
- Unique canonical URLs inventoried and excluded: `{len(inventory_records)}`
- Sitemap metadata files captured: `{len(index_urls) + 1}` (one index plus {len(index_urls)} children)
- Robots metadata files captured: `1`
- Duplicate sitemap memberships retained in `discovered_in`: `true`
- Gate/media assessment: `not_performed_without_page_access`

## Sitemap membership counts

These counts validate metadata coverage only; URL slugs and page titles are deliberately not reproduced here.

""" + "".join(f"- `{url}`: {count}\n" for url, count in sorted(sitemap_counts.items()))
    write_text(ROOT / "REPORT.md", report)

    output_paths = ["CORPUS.md", "REPORT.md", "content_records.jsonl", "inventory.jsonl", "url_dispositions.jsonl"]
    manifest = {
        "schema_version": 1,
        "resource_id": "theduchy",
        "artifact_role": "policy_exclusion_notice",
        "generated_at": captured,
        "source_capture_scope": "robots_and_sitemap_metadata_only",
        "policy_line": POLICY_LINE,
        "policy_reason": REASON,
        "direct_training_ready": False,
        "non_qa": True,
        "page_body_requests": 0,
        "content_record_count": 0,
        "inventory_record_count": len(inventory_records),
        "taxonomy_mappings": [],
        "media_and_gate_handling": "not_assessed_without_canonical_page_access",
        "source_snapshot": [
            {
                "url": resource["url"],
                "path": resource["path"],
                "retrieved_at": resource["retrieved_at"],
                "sha256": resource["sha256"],
                "byte_length": resource["byte_length"],
            }
            for resource in sorted(resources, key=lambda item: item["url"])
        ],
        "outputs": {
            name: {"sha256": digest((ROOT / name).read_bytes()), "byte_length": (ROOT / name).stat().st_size}
            for name in output_paths
        },
    }
    write_text(ROOT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    build()
