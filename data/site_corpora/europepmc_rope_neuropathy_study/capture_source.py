#!/usr/bin/env python3
"""Capture one authorized Europe PMC JATS document and nothing else."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
SOURCE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10294117/fullTextXML"
EXPECTED_HOST = "www.ebi.ac.uk"
EXPECTED_PATH = "/europepmc/webservices/rest/PMC10294117/fullTextXML"
USER_AGENT = "specialist-europepmc-corpus-capture/1.0"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"redirect refused outside the one-route capture contract: {newurl}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != EXPECTED_HOST
        or parsed.path != EXPECTED_PATH
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"request outside authorized EMBL-EBI fullTextXML route: {url}")


def capture() -> None:
    validate_url(SOURCE_URL)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(NoRedirect())
    with opener.open(request, timeout=60) as response:
        if response.status != 200 or response.geturl() != SOURCE_URL:
            raise RuntimeError(f"unexpected fullTextXML response: {response.status} {response.geturl()}")
        body = response.read()
        headers = response.headers

    root = ET.fromstring(body)
    if root.tag != "article":
        raise RuntimeError(f"expected JATS article root, received {root.tag!r}")
    article_ids = {
        node.attrib.get("pub-id-type"): "".join(node.itertext()).strip()
        for node in root.findall("./front/article-meta/article-id")
    }
    if article_ids.get("pmcid") != "PMC10294117" or article_ids.get("pmid") != "37384078":
        raise RuntimeError(f"unexpected article identifiers: {article_ids}")

    licenses = [
        {
            "license_type": node.attrib.get("license-type"),
            "href": node.attrib.get("{http://www.w3.org/1999/xlink}href"),
            "text": " ".join(" ".join(node.itertext()).split()),
        }
        for node in root.findall(".//license")
    ]
    if not any("creativecommons.org/licenses/by/3.0" in (item["href"] or item["text"]) for item in licenses):
        raise RuntimeError("CC BY 3.0 license statement missing from JATS")

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    xml_path = SNAPSHOT / "fullTextXML.xml"
    xml_path.write_bytes(body)
    provenance = {
        "schema": "single-route-jats-capture-v1",
        "resource_id": "europepmc_rope_neuropathy_study",
        "captured_at": now(),
        "capture_scope": "one_authorized_embl_ebi_europepmc_fullTextXML_get",
        "request_count": 1,
        "requested_url": SOURCE_URL,
        "final_url": SOURCE_URL,
        "http_status": 200,
        "content_type": headers.get("Content-Type"),
        "content_length_header": headers.get("Content-Length"),
        "last_modified": headers.get("Last-Modified"),
        "etag": headers.get("ETag"),
        "x_robots_tag": headers.get("X-Robots-Tag"),
        "content_signal": headers.get("Content-Signal"),
        "body_path": "source_snapshot/fullTextXML.xml",
        "body_byte_length": len(body),
        "body_sha256": sha256(body),
        "article_ids": article_ids,
        "discovery_metadata_discrepancy": {
            "queued_canonical_url": "https://europepmc.org/article/MED/37324199",
            "queued_pmid": "37324199",
            "jats_pmid": article_ids["pmid"],
            "resolution": "corpus identity is anchored to the authorized PMCID route and its JATS DOI/PMID; no MED or publisher route was queried",
        },
        "licenses": licenses,
        "excluded_routes": [
            "Cureus publisher endpoint",
            "United States PMC endpoint",
        ],
    }
    (SNAPSHOT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    capture()
