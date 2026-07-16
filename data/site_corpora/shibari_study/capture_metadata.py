#!/usr/bin/env python3
"""Capture only Shibari Study robots and XML sitemap metadata.

Canonical pages are deliberately outside the allowlist. This utility is not
part of the ordinary offline build.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
ROBOTS_URL = "https://shibaristudy.com/robots.txt"
INDEX_URL = "https://shibaristudy.com/sitemap-index.xml"
ALLOWED_PATHS = {"/robots.txt", "/sitemap-index.xml", "/sitemap.xml", "/sitemap-blog.xml"}
USER_AGENT = "CodexPolicyInventory/1.0"
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_metadata_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "shibaristudy.com":
        raise ValueError(f"outside metadata host allowlist: {url}")
    if parsed.query or parsed.fragment or parsed.path not in ALLOWED_PATHS:
        raise ValueError(f"canonical page/body retrieval is forbidden: {url}")


class SafeRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        validate_metadata_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retrieve(url: str) -> tuple[bytes, dict[str, object]]:
    validate_metadata_url(url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/plain,text/xml,application/xml"},
    )
    retrieved_at = now()
    with urllib.request.build_opener(SafeRedirects()).open(req, timeout=30) as response:
        final_url = response.geturl()
        validate_metadata_url(final_url)
        data = response.read()
        status = response.status
        content_type = response.headers.get("Content-Type", "")
    if status != 200:
        raise RuntimeError(f"metadata request failed with HTTP {status}: {url}")
    return data, {
        "url": url,
        "final_url": final_url,
        "retrieved_at": retrieved_at,
        "http_status": status,
        "content_type": content_type,
        "byte_length": len(data),
        "sha256": digest(data),
    }


def locs(data: bytes) -> list[str]:
    root = ET.fromstring(data)
    return [node.text.strip() for node in root.findall(f".//{{{NS}}}loc") if node.text]


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    started_at = now()
    resources: list[dict[str, object]] = []
    for url, role, filename in (
        (ROBOTS_URL, "robots_policy", "robots.txt"),
        (INDEX_URL, "sitemap_index", "sitemap-index.xml"),
    ):
        data, record = retrieve(url)
        (SNAPSHOT / filename).write_bytes(data)
        record.update({"role": role, "path": f"source_snapshot/{filename}"})
        resources.append(record)
    index = (SNAPSHOT / "sitemap-index.xml").read_bytes()
    for url in locs(index):
        validate_metadata_url(url)
        data, record = retrieve(url)
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        (SNAPSHOT / filename).write_bytes(data)
        record.update({"role": "child_sitemap", "path": f"source_snapshot/{filename}"})
        resources.append(record)
    provenance = {
        "schema_version": 1,
        "capture_scope": "robots_and_sitemap_metadata_only",
        "canonical_page_body_requests": 0,
        "canonical_page_body_snapshots_retained": 0,
        "user_agent": USER_AGENT,
        "started_at": started_at,
        "completed_at": now(),
        "resources": resources,
    }
    (SNAPSHOT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
