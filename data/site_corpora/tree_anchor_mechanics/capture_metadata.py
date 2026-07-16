#!/usr/bin/env python3
"""Capture a narrow TreeConsult rights audit and document metadata inventory.

This is intentionally not the offline corpus builder.  It may GET only the
robots, legal-notice, and downloads-index pages and may issue HEAD requests to
the resources listed in the downloads index.  It never GETs a research PDF or
retains either inspected HTML page body.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "source_snapshot"
BASE_URL = "https://www.tree-consult.org/"
ROBOTS_URL = urljoin(BASE_URL, "robots.txt")
LEGAL_URL = urljoin(BASE_URL, "legal-notice.htm")
DOWNLOADS_URL = urljoin(BASE_URL, "downloads.htm")
CATEGORY = "Climbing and rigging"
USER_AGENT = "specialist-corpus-rights-audit/1.0"
RIGHTS_FRAGMENT = (
    "no texts, excerpts of texts, images or parts of images may otherwise be used "
    "without written permission"
)


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return normalize(" ".join(self.parts))


class DownloadInventoryParser(HTMLParser):
    """Extract only the title/link/citation metadata in one named category."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.capture_tag: str | None = None
        self.buffer: list[str] = []
        self.category: str | None = None
        self.pending_title: str | None = None
        self.link_href: str | None = None
        self.link_parts: list[str] = []
        self.records: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h2", "h4"}:
            self.capture_tag = tag
            self.buffer = []
        elif tag == "a" and self.category == CATEGORY and self.pending_title:
            self.link_href = dict(attrs).get("href")
            self.link_parts = []

    def handle_data(self, data: str) -> None:
        if self.capture_tag:
            self.buffer.append(data)
        if self.link_href is not None:
            self.link_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self.capture_tag:
            value = normalize(" ".join(self.buffer))
            if tag == "h2":
                self.category = value
                if value != CATEGORY:
                    self.pending_title = None
            elif tag == "h4" and self.category == CATEGORY:
                self.pending_title = value
            self.capture_tag = None
            self.buffer = []
        elif tag == "a" and self.link_href is not None:
            if not self.pending_title:
                raise RuntimeError("download link lacks a title")
            self.records.append(
                {
                    "title_as_displayed": self.pending_title,
                    "href_as_displayed": self.link_href.strip(),
                    "citation_as_displayed": normalize(" ".join(self.link_parts)),
                }
            )
            self.pending_title = None
            self.link_href = None
            self.link_parts = []


def normalize(value: str) -> str:
    return " ".join(value.split())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def request(url: str, method: str = "GET") -> tuple[bytes, dict[str, object]]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read() if method == "GET" else b""
            headers = response.headers
            metadata = {
                "requested_url": url,
                "final_url": response.geturl(),
                "http_status": response.status,
                "content_type": headers.get("Content-Type"),
                "content_length": integer_or_none(headers.get("Content-Length")),
                "last_modified": headers.get("Last-Modified"),
                "etag": headers.get("ETag"),
                "x_robots_tag": headers.get("X-Robots-Tag"),
                "content_signal": headers.get("Content-Signal"),
            }
            return data, metadata
    except urllib.error.HTTPError as error:
        return b"", {
            "requested_url": url,
            "final_url": error.geturl(),
            "http_status": error.code,
            "content_type": error.headers.get("Content-Type"),
            "content_length": integer_or_none(error.headers.get("Content-Length")),
            "last_modified": error.headers.get("Last-Modified"),
            "etag": error.headers.get("ETag"),
            "x_robots_tag": error.headers.get("X-Robots-Tag"),
            "content_signal": error.headers.get("Content-Signal"),
        }


def integer_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def validate_get_url(url: str) -> None:
    if url not in {ROBOTS_URL, LEGAL_URL, DOWNLOADS_URL}:
        raise ValueError(f"GET is outside metadata-audit scope: {url}")


def validate_head_url(url: str) -> None:
    parsed = urlparse(url)
    internal_pdf = (
        parsed.scheme == "https"
        and parsed.hostname == "www.tree-consult.org"
        and parsed.path.startswith("/upload/mediapool/pdf/rigging_und_seilklettertechnik/")
        and parsed.path.endswith(".pdf")
        and not parsed.query
        and not parsed.fragment
    )
    external_hse = url == "https://www.hse.gov.uk/research/rrhtm/rr668.htm"
    if not (internal_pdf or external_hse):
        raise ValueError(f"HEAD is outside inventoried-resource scope: {url}")


# Manual interpretation of metadata only.  These fields do not summarize any
# document body and are not a direct-training surface.
METADATA_OVERRIDES: dict[str, dict[str, object]] = {
    "climbers_corner_only.pdf": {
        "authors_as_displayed": ["Follett, M.", "Lacigne, B.", "Detter, A.", "Göcke, L.", "Messier, C."],
        "publication_metadata": {"date_as_displayed": "February 2022", "year_claims": [2022], "version_as_displayed": None, "status": "single_date_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["rigging-point friction", "blocks and rings", "anchor loading"],
    },
    "detter-a_simple_rule_of_thumb_for_rigging_forces_from_full-size_tests.pdf": {
        "authors_as_displayed": ["Andreas Detter"],
        "publication_metadata": {"date_as_displayed": "title says 2019; citation says Autumn 2018", "year_claims": [2019, 2018], "version_as_displayed": None, "status": "conflicting_years_preserved"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["full-size tests", "rigging forces", "uncertainty and rules of thumb"],
    },
    "arb_news_damping_june_2022.pdf": {
        "authors_as_displayed": ["Follett, M.", "Lacigne, B.", "Detter, A.", "Göcke, L.", "Messier, C."],
        "publication_metadata": {"date_as_displayed": "June 2022", "year_claims": [2022], "version_as_displayed": None, "status": "single_date_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["negative rigging", "damping", "stem stress"],
    },
    "tragfaehigkeit_gruener_aeste_bmz_4_17.pdf": {
        "authors_as_displayed": ["A. Detter"],
        "publication_metadata": {"date_as_displayed": "issue 4/2017", "year_claims": [2017], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["green branches", "anchor points", "capacity uncertainty"],
    },
    "augsburg_2014_dynamic_2014-05-07.pdf": {
        "authors_as_displayed": [],
        "publication_metadata": {"date_as_displayed": "Kletterforum Augsburg 2014; filename includes 2014-05-07", "year_claims": [2014], "version_as_displayed": "filename date 2014-05-07", "status": "author_not_stated_in_index"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["dynamic loads", "climbing", "rigging"],
    },
    "koehler_detter-einfachseil_doppelstrang-2011.pdf": {
        "authors_as_displayed": ["Köhler", "Detter"],
        "publication_metadata": {"date_as_displayed": "citation says 2012 in Kletterblatt 2011", "year_claims": [2012, 2011], "version_as_displayed": None, "status": "conflicting_years_preserved"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["single-rope systems", "doubled-rope systems", "load paths"],
    },
    "howard_hitch.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "issue 6/2001", "year_claims": [2001], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "outside_requested_tree_anchor_mechanics",
        "target_topics": ["climbing friction hitch"],
    },
    "detter-klettern_mit_redirect-2012.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "issue 2/2012", "year_claims": [2012], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["climbing redirects", "load paths", "redirect selection"],
    },
    "detter-kraefte_an_umlenkungen-2011.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "issue 2/2011", "year_claims": [2011], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["redirect forces", "rope angles", "anchor loads"],
    },
    "gewichtgruenerstammstuecke.pdf": {
        "authors_as_displayed": ["Detter et al."],
        "publication_metadata": {"date_as_displayed": "AFZ-DerWald calendar 2010", "year_claims": [2010], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_supporting_load_estimation",
        "target_topics": ["green log mass", "input-load estimation", "arboriculture context"],
    },
    "rigging_folgeprojekte.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "Kletterblatt 2008", "year_claims": [2008], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["research gaps", "test limitations", "rigging research"],
    },
    "evaluation-of-current-rigging-techniques-used-to-dismantle-trees.pdf": {
        "authors_as_displayed": ["Detter et al."],
        "publication_metadata": {"date_as_displayed": "HSE Books RR668, 2008", "year_claims": [2008], "version_as_displayed": "RR668", "status": "report_identifier_and_year"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["tree dismantling", "rigging practices", "field evaluation"],
    },
    "afz_rigging-kinematics.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "issue 24/2008", "year_claims": [2008], "version_as_displayed": "part 1", "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["rigging kinematics", "tree dismantling", "dynamic events"],
    },
    "tci_mag_nov_08_climbing-loads.pdf": {
        "authors_as_displayed": ["Shoemaker"],
        "publication_metadata": {"date_as_displayed": "November 2008", "year_claims": [2008], "version_as_displayed": None, "status": "single_date_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["climber loads", "tree response", "anchor loading"],
    },
    "rr668.htm": {
        "authors_as_displayed": ["Detter et al."],
        "publication_metadata": {"date_as_displayed": "HSE Books RR668, 2008", "year_claims": [2008], "version_as_displayed": "RR668", "status": "external_landing_for_report"},
        "scope_class": "external_duplicate_landing_requires_separate_rights_review",
        "target_topics": ["tree dismantling", "rigging practices", "field evaluation"],
    },
    "rigging-research-articles_1-4.pdf": {
        "authors_as_displayed": ["Detter", "McKeown"],
        "publication_metadata": {"date_as_displayed": "essentialARB 2008", "year_claims": [2008], "version_as_displayed": "articles 1–4", "status": "series_and_year"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["rigging research", "test findings", "method limitations"],
    },
    "dehnung_und_elastizitaet.pdf": {
        "authors_as_displayed": ["Detter et al."],
        "publication_metadata": {"date_as_displayed": "issue 2/2005", "year_claims": [2005], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["rope elongation", "elasticity", "dynamic loading"],
    },
    "fangstoss.pdf": {
        "authors_as_displayed": ["Detter"],
        "publication_metadata": {"date_as_displayed": "Kletterblatt 2005", "year_claims": [2005], "version_as_displayed": None, "status": "single_year_claim"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["catch shock", "lowering", "peak loads"],
    },
    "rigging_muenchereschule.pdf": {
        "authors_as_displayed": ["Brudi et al."],
        "publication_metadata": {"date_as_displayed": "Kletterblatt 2004", "year_claims": [2004], "version_as_displayed": "Rigging 1.0", "status": "named_version_and_year"},
        "scope_class": "candidate_high_value_transferable_mechanics",
        "target_topics": ["rigging risk assessment", "system planning", "arboriculture operations"],
    },
}


def clean_citation(value: str) -> tuple[str, str | None]:
    match = re.search(r"\s*\((\d+(?:\.\d+)?\s+MiB)\)\s*$", value)
    if not match:
        return value, None
    return value[: match.start()].strip(), match.group(1)


def inventory_key(url: str) -> str:
    return Path(urlparse(url).path).name


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    started = now()

    for url in (ROBOTS_URL, LEGAL_URL, DOWNLOADS_URL):
        validate_get_url(url)
    robots, robots_http = request(ROBOTS_URL)
    legal, legal_http = request(LEGAL_URL)
    downloads, downloads_http = request(DOWNLOADS_URL)
    if any(item["http_status"] != 200 for item in (robots_http, legal_http, downloads_http)):
        raise RuntimeError("metadata-audit endpoint did not return 200")

    robots_text = robots.decode("utf-8")
    required_robots = ("User-agent: *", "Disallow: /cms/", "Disallow: /includes/")
    if not all(line in robots_text for line in required_robots):
        raise RuntimeError("robots directives changed; audit must be reviewed")

    legal_parser = TextExtractor()
    legal_parser.feed(legal.decode("utf-8"))
    legal_text = legal_parser.text()
    if RIGHTS_FRAGMENT not in legal_text:
        raise RuntimeError("required written-permission clause changed or disappeared")

    parser = DownloadInventoryParser()
    parser.feed(downloads.decode("utf-8"))
    if len(parser.records) != 19:
        raise RuntimeError(f"expected 19 listed climbing/rigging resources, got {len(parser.records)}")

    documents: list[dict[str, object]] = []
    for index, raw in enumerate(parser.records, start=1):
        url = urljoin(BASE_URL, raw["href_as_displayed"].strip())
        validate_head_url(url)
        key = inventory_key(url)
        if key not in METADATA_OVERRIDES:
            raise RuntimeError(f"manual metadata review is missing for {url}")
        citation, displayed_size = clean_citation(raw["citation_as_displayed"])
        _, access = request(url, method="HEAD")
        hosted = urlparse(url).hostname == "www.tree-consult.org"
        documents.append(
            {
                "inventory_id": f"treeconsult-rigging-{index:03d}",
                "source_order": index,
                "category_as_displayed": CATEGORY,
                "title_as_displayed": raw["title_as_displayed"],
                "url": url,
                "hosted_by_treeconsult": hosted,
                "resource_type": "pdf" if urlparse(url).path.endswith(".pdf") else "external_landing_page",
                "citation_as_displayed": citation,
                "displayed_file_size": displayed_size,
                **METADATA_OVERRIDES[key],
                "access_audit": {**access, "method": "HEAD", "audited_at": now()},
                "body_retrieved": False,
                "body_snapshot_retained": False,
            }
        )

    if len({item["url"] for item in documents}) != 19:
        raise RuntimeError("document inventory URLs are not unique")
    completed = now()

    (SNAPSHOT / "robots.txt").write_bytes(robots)
    inventory_path = SNAPSHOT / "document_inventory.json"
    inventory_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    policy = {
        "schema_version": 1,
        "audited_at": completed,
        "decision": "defer_direct_training_pending_written_permission",
        "decision_reason": "treeconsult_copyright_notice_requires_written_permission_for_text_reuse",
        "direct_training_ready": False,
        "legal_notice_url": LEGAL_URL,
        "legal_notice_http": legal_http,
        "legal_notice_body_sha256": sha256(legal),
        "legal_notice_body_byte_length": len(legal),
        "legal_notice_body_snapshot_retained": False,
        "rights_evidence_fragment": RIGHTS_FRAGMENT,
        "rights_evidence_fragment_word_count": len(RIGHTS_FRAGMENT.split()),
        "rights_interpretation": "The requested dense paraphrase corpus is deferred unless written permission is documented; public readability and robots access are not treated as reuse permission.",
        "external_resource_boundary": "The external HSE landing page is not governed by TreeConsult's notice but remains excluded from this zero-body artifact pending a separate source-specific rights audit.",
    }
    (ROOT / "policy_decision.json").write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    provenance = {
        "schema_version": 1,
        "started_at": started,
        "completed_at": completed,
        "capture_scope": "robots_and_page_metadata_plus_research_resource_head_only",
        "user_agent": USER_AGENT,
        "page_get_requests": 3,
        "research_document_get_requests": 0,
        "research_resource_head_requests": len(documents),
        "canonical_instructional_body_snapshots_retained": 0,
        "inspected_html_body_snapshots_retained": 0,
        "robots": {
            **robots_http,
            "retrieved_at": started,
            "path": "source_snapshot/robots.txt",
            "sha256": sha256(robots),
            "byte_length": len(robots),
        },
        "legal_notice": {
            **legal_http,
            "retrieved_at": started,
            "body_sha256": sha256(legal),
            "body_byte_length": len(legal),
            "body_snapshot_retained": False,
        },
        "downloads_index": {
            **downloads_http,
            "retrieved_at": started,
            "body_sha256": sha256(downloads),
            "body_byte_length": len(downloads),
            "body_snapshot_retained": False,
            "category_inspected": CATEGORY,
            "listed_resource_count": len(documents),
        },
        "document_inventory": {
            "path": "source_snapshot/document_inventory.json",
            "sha256": sha256(inventory_path.read_bytes()),
            "byte_length": inventory_path.stat().st_size,
            "record_count": len(documents),
        },
    }
    (SNAPSHOT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
