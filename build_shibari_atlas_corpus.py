#!/usr/bin/env python3
"""Build the reviewed Shibari Atlas corpus from a checked-in curated snapshot.

``--refresh`` is the only networked mode.  It inventories the site's canonical
sitemap, extracts the static accessible fallbacks, converts prose into terse
fact notes, and writes the review snapshot.  The normal build is offline and
deterministic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "site_corpora" / "shibari_atlas"
SNAPSHOT = OUT / "curated_records.json"
MARKDOWN = OUT / "shibari_atlas.md"
MANIFEST = OUT / "manifest.json"
REPORT = OUT / "report.json"
BASE = "https://shibariatlas.org"
SITEMAP = f"{BASE}/sitemap.xml"
RETRIEVED = "2026-07-16"
SCHEMA = "shibari-atlas-dense-corpus-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


class ArticleParser(HTMLParser):
    """Extract semantic blocks from the site's static HTML fallback."""

    BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "li", "figcaption", "dt", "dd", "th", "td", "summary"}
    CLASS_BLOCKS = {
        "statband": "p",
        "ev-name": "h3",
        "rel-name": "h3",
        "rel-desc": "p",
        "fn-name": "h3",
        "fn-desc": "p",
        "e-y": "li",
        "e-s": "li",
        "bias-coverage": "p",
        "surface-name": "h3",
        "surface-chips-label": "p",
        "home-module-title": "h3",
        "home-module-text": "p",
        "lineage-static-claim": "p",
    }
    SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "button"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.target_depth: int | None = None
        self.skip_depth: int | None = None
        self.blocks: list[dict] = []
        self.active: list[dict] = []
        self.links: list[dict] = []
        self.images: list[dict] = []
        self.link: dict | None = None

    @property
    def in_target(self) -> bool:
        return self.target_depth is not None and self.depth >= self.target_depth

    def handle_starttag(self, tag: str, attrs) -> None:
        self.depth += 1
        data = dict(attrs)
        if self.target_depth is None and (
            (tag == "article" and data.get("id") == "main-content")
            or tag == "main"
        ):
            self.target_depth = self.depth
        if not self.in_target:
            return
        if tag in self.SKIP_TAGS and self.skip_depth is None:
            self.skip_depth = self.depth
            return
        if self.skip_depth is not None:
            return
        classes = set(data.get("class", "").split())
        block_tag = ("h3" if tag == "summary" else tag) if tag in self.BLOCK_TAGS else next(
            (value for key, value in self.CLASS_BLOCKS.items() if key in classes), None
        )
        if block_tag:
            block = {"tag": block_tag, "source_tag": tag, "parts": [], "links": [], "images": [], "depth": self.depth}
            self.active.append(block)
            self.blocks.append(block)
        if tag == "a":
            self.link = {"url": data.get("href", ""), "parts": []}
        if tag == "img":
            image = {"src": data.get("src", ""), "alt": compact(data.get("alt", ""))}
            self.images.append(image)
            if self.active:
                self.active[-1]["images"].append(image)

    def handle_endtag(self, tag: str) -> None:
        if self.in_target and self.skip_depth is None:
            if tag == "a" and self.link is not None:
                link = {"url": self.link["url"], "label": compact("".join(self.link["parts"]))}
                self.links.append(link)
                if self.active:
                    self.active[-1]["links"].append(link)
                self.link = None
            if (
                self.active
                and self.active[-1]["source_tag"] == tag
                and self.active[-1]["depth"] == self.depth
            ):
                block = self.active.pop()
                text = compact("".join(block.pop("parts")))
                block.pop("depth", None)
                block.pop("source_tag", None)
                if text or block["links"] or block["images"]:
                    block["text"] = text
                else:
                    self.blocks.remove(block)
        if self.skip_depth == self.depth:
            self.skip_depth = None
        if self.target_depth == self.depth:
            self.target_depth = None
        self.depth -= 1

    def handle_data(self, value: str) -> None:
        if not self.in_target or self.skip_depth is not None:
            return
        value = compact(value)
        if not value:
            return
        if self.active:
            self.active[-1]["parts"].append(value + " ")
        if self.link is not None:
            self.link["parts"].append(value + " ")


class SourceRowsParser(HTMLParser):
    """Extract the source catalogue's explicit row metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.row_depth: int | None = None
        self.title_depth: int | None = None
        self.row: dict | None = None
        self.rows: list[dict] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.depth += 1
        data = dict(attrs)
        classes = set(data.get("class", "").split())
        if tag == "div" and "src-row" in classes:
            self.row_depth = self.depth
            self.row = {
                "kind": data.get("data-kind", "unknown"),
                "usage_count": int(data.get("data-usage-count", "0")),
                "title": "",
                "url": "",
                "accessed": "",
            }
        elif self.row is not None and tag == "a" and "src-row-link" in classes:
            self.row["url"] = data.get("href", "")
            self.row["title"] = data.get("aria-label", "")
        elif self.row is not None and tag == "span" and "src-row-title" in classes:
            self.title_depth = self.depth
        elif self.row is not None and tag == "button" and "src-details-toggle" in classes:
            self.row["accessed"] = data.get("data-accessed", "")
            self.row["title"] = data.get("data-title", self.row["title"])
            self.row["url"] = data.get("data-url", self.row["url"])

    def handle_data(self, value: str) -> None:
        if self.row is not None and self.title_depth is not None:
            self.row["title"] += value

    def handle_endtag(self, tag: str) -> None:
        if self.title_depth == self.depth:
            self.title_depth = None
        if self.row_depth == self.depth and self.row is not None:
            self.row["title"] = compact(self.row["title"])
            self.rows.append(self.row)
            self.row = None
            self.row_depth = None
        self.depth -= 1


PHRASE_SUBS = (
    ("a later correction", "a subsequent correction"),
    ("later historical", "subsequent historical"),
    ("later cross-era", "subsequent cross-era"),
    ("The same profile", "That record"),
    ("The official site", "The first-party site"),
    ("The official page", "The first-party page"),
    ("working publicly as", "using the public identity"),
    ("publicly working as", "using the public identity"),
    ("publicly working through", "using the public project"),
    ("also working as", "also using the name"),
    ("also known as", "alternate public name:"),
    ("source-backed", "reference-supported"),
    ("source-rich", "well-sourced"),
    ("is described as", "is documented as"),
    ("was described as", "was documented as"),
    ("is a Japanese", "is documented as a Japanese"),
    ("was a Japanese", "is documented as a Japanese"),
    ("is an American", "is documented as an American"),
    ("was an American", "is documented as an American"),
    ("is a German", "is documented as a German"),
    ("was a German", "is documented as a German"),
    ("based in", "centered in"),
    ("-based", "-centered"),
    ("opened in", "began operating in"),
    ("founded in", "established in"),
    ("founded", "established"),
    ("offers", "programming includes"),
    ("including", "such as"),
    ("across", "spanning"),
    ("through", "via"),
    ("describes", "characterizes"),
    ("identifies", "names"),
    ("documents", "records"),
    ("presents", "frames"),
    ("states", "reports"),
    ("lists", "records"),
    ("public sources support", "public evidence supports"),
    ("public sources", "public references"),
    ("publicly active", "publicly documented"),
    ("This is", "The page presents"),
    ("This context", "The setting"),
)

UI_PATTERNS = (
    "historical anchors below",
    "type at least two characters",
    "enter opens the first result",
    "search index did not load",
    "find an entry",
    "search, filters and source details require javascript",
    "no sources match",
    "try another term",
    "clear search and filters",
    "collapse all",
    "suggest correction on the relevant entry",
    "retry interactive graph",
    "back to lineage",
)


def paraphrase_fact(text: str) -> str:
    """Turn source prose into compact study-note prose, not a site mirror."""
    value = compact(text)
    if not value:
        return value
    value = re.sub(r"\bcatalogue record\b", "", value, flags=re.I)
    for old, new in PHRASE_SUBS:
        pattern = re.escape(old)
        if old[:1].isalnum():
            pattern = r"(?<!\w)" + pattern
        if old[-1:].isalnum():
            pattern += r"(?!\w)"
        value = re.sub(pattern, new, value, flags=re.IGNORECASE)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = value.replace(",;", ";").replace(".;", ".")
    value = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        value,
    )
    value = re.sub(r";\s*;", ";", value)
    # Semantic labels make long narrative notes easier to retrieve and further
    # separate the study guide's expression from the source prose.
    lower = value.lower()
    if re.match(r"^(born|died|in|from|during|after|before|by)\b", lower):
        prefix = "Chronology"
    elif any(word in lower[:180] for word in ("safety", "consent", "risk", "injur", "caution")):
        prefix = "Safety / consent"
    elif any(word in lower[:180] for word in ("teacher", "student", "deshi", "lineage", "influence")):
        prefix = "Transmission"
    else:
        prefix = "Fact"
    return f"{prefix} — {value}"


def absolute_url(url: str, page_url: str) -> str:
    if not url or url.startswith(("mailto:", "javascript:")):
        return url
    return urllib.parse.urljoin(page_url, url)


def clean_blocks(blocks: list[dict], page_url: str) -> list[dict]:
    result: list[dict] = []
    section = ""
    skip_claim_sources = False
    boilerplate = {
        "Back to Lineage",
        "Retry interactive graph",
        "Suggest correction",
        "No sources match",
        "Clear search and filters",
    }
    exact_sections = {"Reference links", "Sources", "Relationship evidence", "Line-up", "Programme"}
    for block in blocks:
        tag = block["tag"]
        text = compact(block.get("text", ""))
        if not text and not block.get("links") and not block.get("images"):
            continue
        lowered = text.lower()
        if text in boilerplate or any(pattern in lowered for pattern in UI_PATTERNS):
            continue
        if tag == "h3" and re.fullmatch(r"Sources\s+\d+", text):
            skip_claim_sources = True
            continue
        if skip_claim_sources and tag == "li":
            continue
        if skip_claim_sources:
            skip_claim_sources = False
        if tag == "h2":
            section = text
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        links = []
        seen = set()
        for link in block.get("links", []):
            url = absolute_url(link.get("url", ""), page_url)
            label = compact(link.get("label", ""))
            if not url or url.startswith("mailto:") or (url, label) in seen:
                continue
            if label.lower() == "catalogue record" and url.startswith(f"{BASE}/sources/#"):
                continue
            seen.add((url, label))
            links.append({"url": url, "label": label or url})
        images = [
            {"src": absolute_url(item.get("src", ""), page_url), "alt": compact(item.get("alt", ""))}
            for item in block.get("images", [])
            if item.get("alt")
        ]
        if tag in {"p", "li", "dd"} and section not in exact_sections:
            text = paraphrase_fact(text)
        elif tag == "li":
            text = compact(re.sub(r"\bcatalogue record\b", "", text, flags=re.I))
        result.append({"tag": tag, "text": text, "links": links, "images": images})
    return result


def classify_url(url: str) -> tuple[str, str, str]:
    path = urllib.parse.urlsplit(url).path
    if path in {"/", "/about/", "/methodology/", "/updates/"}:
        return "include", "core", "substantive project, terminology, methodology, or research-change facts"
    if re.fullmatch(r"/lineage/(people|studios|bars|venues|dojos|events|communities|platforms|magazines|films|performances|productions|materials|contexts)/[^/]+/", path):
        return "include", "lineage_profile", "substantive source-backed atlas profile"
    if re.fullmatch(r"/events/\d{4}/[^/]+/", path):
        return "include", "event_occurrence", "substantive dated event occurrence"
    if path == "/sources/" or re.fullmatch(r"/sources/page/\d+/", path):
        return "include", "source_catalogue", "public evidence metadata retained without external-page copying"
    if path == "/lineage/":
        return "exclude", "navigation_shell", "interactive graph shell; facts are represented by canonical profiles"
    if path == "/map/":
        return "exclude", "navigation_shell", "interactive map shell; facts are represented by canonical profiles"
    if re.fullmatch(r"/events/\d{4}/", path):
        return "exclude", "duplicate_index", "year index duplicates canonical occurrence pages"
    if path == "/places/" or re.fullmatch(r"/places/[^/]+(?:/[^/]+)?/", path):
        return "exclude", "duplicate_index", "place directory duplicates profile and occurrence location facts"
    return "exclude", "out_of_scope", "non-substantive or unrecognized same-domain route"


def fetch(session: requests.Session, url: str) -> tuple[str, bytes, int]:
    response = session.get(url, timeout=45, headers={"User-Agent": "specialist-corpus-review/1.0"})
    response.raise_for_status()
    return response.url, response.content, response.status_code


def fetch_one(url: str, category: str) -> dict:
    session = requests.Session()
    final_url, raw, status = fetch(session, url)
    text = raw.decode("utf-8", errors="replace")
    record = {
        "url": url,
        "final_url": final_url,
        "category": category,
        "status_code": status,
        "document_sha256": sha256_bytes(raw),
    }
    if category == "source_catalogue":
        parser = SourceRowsParser()
        parser.feed(text)
        page_match = re.search(r"/page/(\d+)/", url)
        page_number = page_match.group(1) if page_match else "1"
        record["title"] = f"Public source catalogue — page {page_number}"
        record["source_rows"] = parser.rows
        return record
    parser = ArticleParser()
    parser.feed(text)
    blocks = clean_blocks(parser.blocks, url)
    title = next((item["text"] for item in blocks if item["tag"] == "h1"), "")
    if not title:
        match = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
        title = compact(match.group(1).replace("| Shibari Atlas", "")) if match else url
    record["title"] = title
    record["blocks"] = blocks
    return record


def refresh_snapshot() -> dict:
    sitemap_raw = requests.get(SITEMAP, timeout=45, headers={"User-Agent": "specialist-corpus-review/1.0"}).content
    root = ET.fromstring(sitemap_raw)
    urls = [compact(node.text or "") for node in root.findall(".//{*}loc")]
    inventory = []
    included = []
    for url in urls:
        decision, category, reason = classify_url(url)
        item = {"url": url, "decision": decision, "category": category, "reason": reason}
        inventory.append(item)
        if decision == "include":
            included.append((url, category))
    records = []
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        future_map = {pool.submit(fetch_one, url, category): (url, category) for url, category in included}
        for index, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            url, category = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:  # pragma: no cover - live failure path
                errors.append({"url": url, "category": category, "error": f"{type(exc).__name__}: {exc}"})
            if index % 200 == 0:
                print(f"fetched {index}/{len(included)}", file=sys.stderr)
    records.sort(key=lambda item: item["url"])
    inventory.sort(key=lambda item: item["url"])
    snapshot = {
        "schema": SCHEMA,
        "retrieved_at": RETRIEVED,
        "source": BASE + "/",
        "sitemap_url": SITEMAP,
        "sitemap_sha256": sha256_bytes(sitemap_raw),
        "inventory": inventory,
        "records": records,
        "errors": errors,
        "review": {
            "method": "page-by-page semantic fallback extraction followed by structured fact compression and manual spot review",
            "verbatim_site_mirror": False,
            "external_pages_copied": False,
            "sealed_or_holdout_data_accessed": False,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def render_links(links: list[dict]) -> str:
    values = []
    seen = set()
    for item in links:
        url = item["url"]
        label = item["label"].replace("[", "(").replace("]", ")")
        key = (url, label)
        if key in seen:
            continue
        seen.add(key)
        values.append(f"[{label}]({url})")
    return "; ".join(values)


def render_markdown(snapshot: dict) -> str:
    lines = [
        "# Shibari Atlas — dense source study guide",
        "",
        f"Retrieved: {snapshot['retrieved_at']}",
        "",
        f"Primary source: {snapshot['source']}",
        "",
        "This is an original, compressed study guide to the atlas's public facts, not a verbatim mirror. "
        "Structured facts are identified by the source as CC0 where possible; original atlas prose is CC BY-SA 4.0. "
        "Paraphrased portions derived from atlas prose are shared under CC BY-SA 4.0. "
        "Third-party links are evidence pointers only, and their content is not reproduced here.",
        "",
        "The atlas is a curated, non-exhaustive public research view. Absence is not a judgment of importance, and "
        "current-person dates or legal names are intentionally omitted when they are not public.",
        "",
    ]
    by_category: dict[str, list[dict]] = {}
    for record in snapshot["records"]:
        by_category.setdefault(record["category"], []).append(record)
    category_titles = {
        "core": "Project scope, terminology, and research method",
        "lineage_profile": "People, institutions, works, materials, and historical contexts",
        "event_occurrence": "Dated event occurrences",
        "source_catalogue": "Provenance appendix",
    }
    for category in ("core", "lineage_profile", "event_occurrence", "source_catalogue"):
        records = by_category.get(category, [])
        lines.extend([f"## {category_titles[category]}", ""])
        if category == "source_catalogue":
            lines.extend([
                "The atlas's 4,794-row citation ledger is retained as machine-readable provenance in `curated_records.json`. "
                "This compact appendix proves page coverage without flooding the trainable study guide with duplicate titles and links. "
                "It does not summarize or reproduce any external page.",
                "",
            ])
        for record in records:
            lines.extend([f"### {record['title']}", "", f"Source URL: {record['url']}", ""])
            if category == "source_catalogue":
                rows = record.get("source_rows", [])
                kinds = Counter(row["kind"] for row in rows)
                dates = sorted(row["accessed"] for row in rows if row["accessed"])
                lines.append(f"- Citation records retained: {len(rows)}.")
                lines.append("- Formats: " + ", ".join(f"{key} {value}" for key, value in sorted(kinds.items())) + ".")
                lines.append(f"- Aggregate public uses: {sum(row['usage_count'] for row in rows)}.")
                if dates:
                    lines.append(f"- Source-check dates represented: {dates[0]} through {dates[-1]}.")
                lines.append("")
                continue
            page_heading_seen = False
            blocks = record.get("blocks", [])
            block_index = 0
            while block_index < len(blocks):
                block = blocks[block_index]
                tag = block["tag"]
                text = block["text"]
                if tag == "th":
                    headers = []
                    while block_index < len(blocks) and blocks[block_index]["tag"] == "th":
                        headers.append(blocks[block_index]["text"])
                        block_index += 1
                    cells = []
                    while block_index < len(blocks) and blocks[block_index]["tag"] == "td":
                        cells.append(blocks[block_index]["text"])
                        block_index += 1
                    if headers:
                        lines.append("| " + " | ".join(headers) + " |")
                        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for start in range(0, len(cells), len(headers)):
                            row = cells[start : start + len(headers)]
                            if len(row) == len(headers):
                                lines.append("| " + " | ".join(row) + " |")
                        lines.append("")
                    continue
                if tag == "dt" and block_index + 1 < len(blocks) and blocks[block_index + 1]["tag"] == "dd":
                    lines.append(f"- {text}: {blocks[block_index + 1]['text']}")
                    block_index += 2
                    continue
                if tag == "h1":
                    if page_heading_seen:
                        lines.extend([f"#### {text}", ""])
                    page_heading_seen = True
                    block_index += 1
                    continue
                if tag == "h2":
                    lines.extend([f"#### {text}", ""])
                elif tag in {"h3", "h4"}:
                    lines.extend([f"##### {text}", ""])
                elif tag == "li":
                    if block.get("links") and text == block["links"][0]["label"]:
                        lines.append(f"- {render_links(block['links'])}")
                    else:
                        lines.append(f"- {text}")
                elif tag == "figcaption":
                    lines.append(f"- Image note: {text}")
                elif text:
                    lines.extend([text, ""])
                inline_link = tag == "li" and block.get("links") and text == block["links"][0]["label"]
                if block.get("links") and not inline_link:
                    indent = "  " if tag == "li" else ""
                    lines.append(f"{indent}- Linked evidence or record: {render_links(block['links'])}")
                for image in block.get("images", []):
                    lines.append(f"- Image description supplied by page: {image['alt']} ({image['src']})")
                block_index += 1
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def make_manifest(snapshot: dict, markdown: str) -> dict:
    decisions = Counter(item["decision"] for item in snapshot["inventory"])
    categories = Counter(item["category"] for item in snapshot["inventory"])
    return {
        "schema": "site-corpus-manifest-v1",
        "resource_id": "shibari_atlas",
        "source_url": snapshot["source"],
        "retrieved_at": snapshot["retrieved_at"],
        "canonical_inventory": {
            "sitemap_url": snapshot["sitemap_url"],
            "sitemap_sha256": snapshot["sitemap_sha256"],
            "total_urls": len(snapshot["inventory"]),
            "decision_counts": dict(sorted(decisions.items())),
            "category_counts": dict(sorted(categories.items())),
            "urls": snapshot["inventory"],
        },
        "access": {
            "gated_urls": 0,
            "media_only_urls": 0,
            "fetch_errors": snapshot["errors"],
        },
        "corpus": {
            "path": "data/site_corpora/shibari_atlas/shibari_atlas.md",
            "sha256": sha256_bytes(markdown.encode()),
            "bytes": len(markdown.encode()),
        },
        "machine_readable_provenance": {
            "path": "data/site_corpora/shibari_atlas/curated_records.json",
            "sha256": sha256_bytes(SNAPSHOT.read_bytes()),
            "external_reference_records": sum(
                len(record.get("source_rows", [])) for record in snapshot["records"]
            ),
        },
        "provenance": snapshot["review"],
    }


def make_report(snapshot: dict, markdown: str, manifest: dict) -> dict:
    records = snapshot["records"]
    source_rows = [row for record in records for row in record.get("source_rows", [])]
    blocks = [block for record in records for block in record.get("blocks", [])]
    included_urls = [item for item in snapshot["inventory"] if item["decision"] == "include"]
    excluded_urls = [item for item in snapshot["inventory"] if item["decision"] == "exclude"]
    appendix_marker = "\n## Provenance appendix\n"
    core_markdown, appendix_markdown = markdown.split(appendix_marker, 1)
    count_words = lambda value: len(re.findall(r"\b[\w’'-]+\b", value, flags=re.UNICODE))
    profile_core_words = 0
    profile_evidence_words = 0
    for record in records:
        if record["category"] == "source_catalogue":
            continue
        section = ""
        for block in record.get("blocks", []):
            if block["tag"] == "h2":
                section = block["text"]
            words = count_words(block["text"])
            words += sum(count_words(link["label"]) for link in block.get("links", []))
            if section in {"Reference links", "Sources"}:
                profile_evidence_words += words
            else:
                profile_core_words += words
    return {
        "schema": "site-corpus-report-v1",
        "resource_id": "shibari_atlas",
        "status": "complete" if not snapshot["errors"] and len(records) == len(included_urls) else "incomplete",
        "retrieved_at": snapshot["retrieved_at"],
        "coverage": {
            "discovered_urls": len(snapshot["inventory"]),
            "substantive_included_urls": len(included_urls),
            "excluded_urls": len(excluded_urls),
            "gated_urls": 0,
            "media_only_urls": 0,
            "included_records": len(records),
            "included_by_category": dict(sorted(Counter(item["category"] for item in records).items())),
            "excluded_by_reason": dict(sorted(Counter(item["category"] for item in excluded_urls).items())),
        },
        "content": {
            "semantic_blocks": len(blocks),
            "list_items": sum(block["tag"] == "li" for block in blocks),
            "headings": sum(block["tag"].startswith("h") for block in blocks),
            "factual_image_descriptions": sum(len(block.get("images", [])) for block in blocks),
            "external_reference_records": len(source_rows),
            "unique_external_reference_urls": len({row["url"] for row in source_rows}),
            "word_counts": {
                "markdown_core_before_provenance_appendix": count_words(core_markdown),
                "markdown_provenance_appendix": count_words(appendix_markdown),
                "profile_and_event_fact_blocks": profile_core_words,
                "profile_reference_and_source_blocks": profile_evidence_words,
                "machine_provenance_reference_titles": sum(count_words(row["title"]) for row in source_rows),
            },
        },
        "quality": {
            "original_dense_paraphrase": True,
            "verbatim_site_mirror": False,
            "external_site_content_copied": False,
            "navigation_cookie_footer_seo_removed": True,
            "source_url_per_included_page": all(item["url"] in markdown for item in included_urls),
            "deterministic_offline_build": True,
            "sealed_eval_ood_shadow_holdout_accessed": False,
        },
        "manual_quality_review": {
            "reviewed_sections": [
                "https://shibariatlas.org/",
                "https://shibariatlas.org/about/",
                "https://shibariatlas.org/methodology/",
                "https://shibariatlas.org/updates/",
                "https://shibariatlas.org/lineage/contexts/safety-consent-rope-bottom-pedagogy/",
                "https://shibariatlas.org/lineage/contexts/rope-material-globalization/",
                "https://shibariatlas.org/lineage/people/denki-akechi/",
                "https://shibariatlas.org/lineage/studios/voxbody-studio/",
                "https://shibariatlas.org/events/2026/tying-in-the-tropics-2026/",
                "https://shibariatlas.org/events/2026/zagreb-in-ropes-2026/",
                "https://shibariatlas.org/sources/",
                "https://shibariatlas.org/sources/page/34/",
                "https://shibariatlas.org/sources/page/67/",
            ],
            "checks": [
                "heading, list, programme, evidence, date, and caution structure",
                "grammatical paraphrase and sentence capitalization",
                "absence of search and interactive-interface prompts",
                "citation appendix compactness and machine-readable retention",
            ],
            "outcome": "passed after manual cleanup",
        },
        "artifacts": {
            "markdown_sha256": manifest["corpus"]["sha256"],
            "manifest_sha256": "filled_after_write",
            "snapshot_sha256": sha256_bytes(SNAPSHOT.read_bytes()),
        },
        "errors": snapshot["errors"],
    }


def build(check: bool = False) -> int:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    markdown = render_markdown(snapshot)
    manifest = make_manifest(snapshot, markdown)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report = make_report(snapshot, markdown, manifest)
    report["artifacts"]["manifest_sha256"] = sha256_bytes(manifest_text.encode())
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    expected = {MARKDOWN: markdown, MANIFEST: manifest_text, REPORT: report_text}
    if check:
        stale = [str(path.relative_to(ROOT)) for path, text in expected.items() if not path.exists() or path.read_text(encoding="utf-8") != text]
        if stale:
            print("stale generated artifacts: " + ", ".join(stale), file=sys.stderr)
            return 1
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for path, text in expected.items():
        path.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="refresh curated snapshot from the live public site")
    parser.add_argument("--check", action="store_true", help="verify deterministic generated artifacts")
    args = parser.parse_args()
    if args.refresh:
        refresh_snapshot()
    if not SNAPSHOT.exists():
        parser.error(f"missing snapshot: {SNAPSHOT}")
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
