#!/usr/bin/env python3
"""Capture only The Duchy's robots.txt and XML sitemap metadata.

This is intentionally separate from the deterministic corpus builder. It has a
hard path allowlist and refuses canonical content pages.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
ROBOTS_URL = "https://www.theduchy.com/robots.txt"
INDEX_URL = "https://www.theduchy.com/sitemap_index.xml"
ALLOWED_HOSTS = {"theduchy.com", "www.theduchy.com"}
USER_AGENT = "CodexPolicyInventory/1.0"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_metadata_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"outside metadata host allowlist: {url}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"query/fragment not allowed for metadata capture: {url}")
    path = parsed.path
    if path == "/robots.txt":
        return
    if re.fullmatch(r"/[a-z0-9_-]*sitemap(?:_index)?\.xml", path):
        return
    raise ValueError(f"canonical page/body retrieval is forbidden: {url}")


class SameHostRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        validate_metadata_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retrieve(url: str) -> tuple[bytes, dict[str, object]]:
    validate_metadata_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain,text/xml,application/xml"})
    opener = urllib.request.build_opener(SameHostRedirectHandler())
    retrieved_at = now()
    with opener.open(request, timeout=30) as response:
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
        "sha256": sha256(data),
    }


def sitemap_locations(data: bytes) -> list[str]:
    root = ET.fromstring(data)
    return [node.text.strip() for node in root.findall(f".//{{{SITEMAP_NS}}}loc") if node.text]


def main() -> None:
    started_at = now()
    (SNAPSHOT / "sitemaps").mkdir(parents=True, exist_ok=True)
    resources: list[dict[str, object]] = []

    robots, record = retrieve(ROBOTS_URL)
    robots_path = SNAPSHOT / "robots.txt"
    robots_path.write_bytes(robots)
    record.update({"role": "robots_policy", "path": str(robots_path.relative_to(ROOT))})
    resources.append(record)

    index, record = retrieve(INDEX_URL)
    index_path = SNAPSHOT / "sitemap_index.xml"
    index_path.write_bytes(index)
    record.update({"role": "sitemap_index", "path": str(index_path.relative_to(ROOT))})
    resources.append(record)

    child_urls = sitemap_locations(index)
    for url in child_urls:
        validate_metadata_url(url)
        data, record = retrieve(url)
        filename = urlparse(url).path.rsplit("/", 1)[-1]
        path = SNAPSHOT / "sitemaps" / filename
        path.write_bytes(data)
        record.update({"role": "child_sitemap", "path": str(path.relative_to(ROOT))})
        resources.append(record)

    provenance = {
        "schema_version": 1,
        "capture_scope": "robots_and_sitemap_metadata_only",
        "page_body_requests": 0,
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
