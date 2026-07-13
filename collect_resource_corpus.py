#!/usr/bin/env python3
"""Collect the narrowly scoped public corpus in ``rope_resources_v1.json``.

The collector is deliberately conservative:

* ``reference_only`` resources never cause a network request.
* robots.txt and ``Content-Signal: ai-train=no`` fail closed.
* the manifest's explicit ``collection.urls`` are the fetch plan (the canonical
  URL is the fallback); sitemap discovery is enabled only where a narrow URL
  policy exists.
* no HTML link traversal is performed, so a shop or event site cannot turn into a
  general-purpose crawl accidentally.
* YouTube access is isolated behind an optional adapter and never supplies login
  credentials, cookies, or paywall workarounds.

Each collected page is written as a provenance-rich JSON document.  A separate
coverage JSON is always written for every selected resource, including blocked or
unavailable resources.
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "raw" / "rope_resources_v1"
DEFAULT_COVERAGE_DIR = ROOT / "data" / "raw" / "rope_resources_v1_coverage"
DEFAULT_SUMMARY = ROOT / "data" / "rope_resources_coverage_v1.json"
USER_AGENT = (
    "specialist-resource-collector/1.0 "
    "(+https://github.com/catid/specialist; policy-respecting research crawl)"
)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SITEMAPS_PER_RESOURCE = 20
MAX_SITEMAP_URLS = 10_000
MAX_YOUTUBE_ITEMS = 200
MIN_TEXT_CHARS = 120
SUPPORTED_MODES = {
    "exact_public_page",
    "public_catalog_and_descriptions",
    "reference_only",
    "reference_plus_narrow_specs",
    "relevant_event_pages",
    "relevant_product_pages",
    "relevant_public_pages",
    "youtube_playlist",
    "youtube_video",
}

CONTENT_SIGNAL_NO_TRAIN_RE = re.compile(
    r"(?:content-signal\s*:\s*)?[^\r\n]*\bai-train\s*=\s*no\b", re.I
)
GATED_PATH_RE = re.compile(
    r"/(?:account|accounts|cart|checkout|dashboard|login|log-in|member|members|"
    r"membership|my-plan|orders|register|sign-in|signin|sign-up|signup|users)(?:/|$)",
    re.I,
)
ASSET_PATH_RE = re.compile(
    r"\.(?:avif|css|csv|gif|ico|jpe?g|js|json|m4a|mov|mp3|mp4|pdf|png|svg|"
    r"tar|tgz|webm|webp|woff2?|zip)(?:$|\?)",
    re.I,
)
COMMON_IRRELEVANT_PATH_RE = re.compile(
    r"/(?:about|affiliate|author|authors|basket|bio|blog|cart|checkout|contact|"
    r"cookie|feed|gallery|gift-card|legal|login|newsletter|partners?|privacy|"
    r"refund|returns?|search|sign-in|signin|sign-up|signup|tag|terms)(?:/|$)",
    re.I,
)
ACCESS_CHALLENGE_RE = re.compile(
    r"cf-chl-|g-recaptcha|hcaptcha|/captcha/|verify (?:that )?you are human|"
    r"sorry, we just need to make sure you(?:'|’)re not a robot|"
    r"automated access to our systems has been detected",
    re.I,
)


class CollectionError(RuntimeError):
    """Base class for expected, reportable collection failures."""


class YouTubeAdapterUnavailable(CollectionError):
    """The optional YouTube adapter or its public data is unavailable."""


class YouTubeAccessBlocked(CollectionError):
    """A YouTube video/playlist requires access the collector will not bypass."""


class YouTubeAdapter(Protocol):
    """Policy boundary for optional public YouTube metadata/caption collection."""

    name: str

    def collect(self, resource: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return public items without authentication or access-control bypasses."""


@dataclass(frozen=True)
class ResourcePolicy:
    """A bounded URL policy layered on top of the owner-supplied manifest."""

    explicit_sitemaps: tuple[str, ...] = ()
    sitemap_child_pattern: re.Pattern[str] | None = None
    page_path_pattern: re.Pattern[str] | None = None
    exact_only: bool = False

    def permits_sitemap_child(self, url: str) -> bool:
        if self.sitemap_child_pattern is None:
            return False
        return bool(self.sitemap_child_pattern.search(unquote(urlsplit(url).path)))

    def permits_page(self, url: str, canonical_url: str) -> bool:
        if normalize_url(url) == normalize_url(canonical_url):
            return True
        if self.exact_only or self.page_path_pattern is None:
            return False
        path = unquote(urlsplit(url).path)
        return bool(self.page_path_pattern.search(path))


PAGE_SITEMAP_RE = re.compile(r"/(?:page|pages)-sitemap(?:\d+)?\.xml$", re.I)
PAGE_OR_PRODUCT_SITEMAP_RE = re.compile(
    r"/(?:page|pages|product|products)-sitemap(?:\d+)?\.xml$", re.I
)
PAGE_OR_EVENT_SITEMAP_RE = re.compile(
    r"/(?:page|pages|event|events)-sitemap(?:\d+)?\.xml$", re.I
)

# These filters are intentionally URL-based and narrow.  They decide which public
# landing pages may be fetched; they do not assert that every matching page is fit
# for downstream training.
RESOURCE_POLICIES: dict[str, ResourcePolicy] = {
    "rope365": ResourcePolicy(
        explicit_sitemaps=("https://rope365.com/sitemap_index.xml",),
        sitemap_child_pattern=PAGE_SITEMAP_RE,
        # Rope365's page sitemap is itself a curriculum inventory.  The owner-
        # supplied manifest carries the explicit language/commerce/event/test
        # exclusions, so useful pages with non-taxonomic slugs are not lost.
        page_path_pattern=re.compile(r"^/", re.I),
    ),
    "shibari_study": ResourcePolicy(
        explicit_sitemaps=("https://shibaristudy.com/sitemap-index.xml",),
        sitemap_child_pattern=re.compile(r"/sitemap\.xml$", re.I),
        page_path_pattern=re.compile(
            r"^/programs/[^/?#]*(?:free|safety|suspension|upline|rope|tie|knot|"
            r"column|friction|harness|hip|chest|takate|tk)[^/?#]*$",
            re.I,
        ),
    ),
    "subspace_designs": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(r"/(?:rig|plate|ring|suspension)[^/]*", re.I),
    ),
    "tetruss": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(r"/(?:tetruss|frame|rig|suspension)[^/]*", re.I),
    ),
    "shibari_supply_bamboo": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(r"/(?:bamboo|suspension)[^/]*", re.I),
    ),
    "de_giotto_rope": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(r"/(?:rope|jute|hemp|care|condition)[^/]*", re.I),
    ),
    "rw_rope": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:upline|up-line|rope|custom|spec|ordering)[^/]*", re.I
        ),
    ),
    "chromaknotz": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:rope|nylon|synthetic|bondage|care|condition)[^/]*", re.I
        ),
    ),
    "knothead_nylon": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_PRODUCT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:rope|nylon|synthetic|bondage|care|condition)[^/]*", re.I
        ),
    ),
    "tethered_together": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_EVENT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:event|schedule|class|education|presenter|teacher|venue|ticket|"
            r"registration|attend|faq)(?:[-/]|$)",
            re.I,
        ),
    ),
    "ropecraft": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_EVENT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:event|schedule|class|education|presenter|teacher|venue|ticket|"
            r"registration|attend|faq)(?:[-/]|$)",
            re.I,
        ),
    ),
    "atx_empty_space": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_EVENT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:event|calendar|schedule|class|education|teacher|venue|ticket|"
            r"registration|faq)(?:[-/]|$)",
            re.I,
        ),
    ),
    "austin_rope_slingers": ResourcePolicy(
        sitemap_child_pattern=PAGE_OR_EVENT_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:event|calendar|meeting|schedule|class|education|teacher|venue|"
            r"registration|faq)(?:[-/]|$)",
            re.I,
        ),
    ),
    "bight_bound": ResourcePolicy(
        sitemap_child_pattern=PAGE_SITEMAP_RE,
        page_path_pattern=re.compile(
            r"/(?:learn|class|education|safety|rope|tutorial|resource|event)"
            r"(?:[-/]|$)",
            re.I,
        ),
    ),
}

@dataclass
class RobotsInfo:
    origin: str
    url: str
    status: str
    parser: RobotFileParser | None = None
    sitemaps: list[str] = field(default_factory=list)
    ai_train_no: bool = False
    http_status: int | None = None
    reason: str | None = None
    retrieved_at: str | None = None

    def allows(self, user_agent: str, url: str) -> bool:
        if self.status != "ok":
            return self.status == "not_found"
        if self.ai_train_no or self.parser is None:
            return False
        return self.parser.can_fetch(user_agent, url)

    def public_record(self) -> dict[str, Any]:
        return {
            "ai_train_no": self.ai_train_no,
            "http_status": self.http_status,
            "reason": self.reason,
            "retrieved_at": self.retrieved_at,
            "sitemaps": self.sitemaps,
            "status": self.status,
            "url": self.url,
        }


class _VisibleTextParser(HTMLParser):
    """Small dependency-free fallback when trafilatura is unavailable."""

    BLOCKED_TAGS = {"script", "style", "svg", "template", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocked_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.BLOCKED_TAGS:
            self.blocked_depth += 1
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.BLOCKED_TAGS and self.blocked_depth:
            self.blocked_depth -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.blocked_depth:
            return
        value = re.sub(r"\s+", " ", data).strip()
        if not value:
            return
        if self.in_title:
            self.title_parts.append(value)
        else:
            self.text_parts.append(value)


def extract_html(html: str, _url: str) -> tuple[str, str, str]:
    """Return ``(title, text, extractor_name)`` from a public HTML page."""

    parser = _VisibleTextParser()
    parser.feed(html)
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()[:300]
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        extracted = None
    text = extracted or "\n".join(parser.text_parts)
    text = re.sub(r"[ \t]+", " ", html_module.unescape(text or ""))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text, "trafilatura" if extracted else "html.parser"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def normalize_url(url: str, *, keep_query: bool = True) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    query = parsed.query if keep_query else ""
    return urlunsplit((scheme, netloc, path, query, ""))


def origin_for(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), "", "", ""))


def same_site(first: str, second: str) -> bool:
    def host(url: str) -> str:
        value = (urlsplit(url).hostname or "").lower()
        return value[4:] if value.startswith("www.") else value

    def port(url: str) -> int | None:
        parsed = urlsplit(url)
        return parsed.port or (443 if parsed.scheme.lower() == "https" else None)

    return (
        urlsplit(first).scheme.lower() == "https"
        and urlsplit(second).scheme.lower() == "https"
        and host(first) == host(second)
        and port(first) == port(second)
    )


def header_value(headers: Mapping[str, Any], name: str) -> str:
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value)
    return ""


def has_no_train_signal(*values: str) -> bool:
    return any(CONTENT_SIGNAL_NO_TRAIN_RE.search(value or "") for value in values)


def parse_sitemap_locations(xml_text: str) -> tuple[str, list[str]]:
    """Parse a sitemap without executing external entities."""

    if re.search(r"<!DOCTYPE|<!ENTITY", xml_text, re.I):
        raise ValueError("DTD/entity declarations are not accepted in sitemaps")
    root = ET.fromstring(xml_text)
    kind = root.tag.rsplit("}", 1)[-1].lower()
    if kind not in {"sitemapindex", "urlset"}:
        raise ValueError(f"unsupported sitemap root {kind!r}")
    # Only read the direct <loc> child of each <url> or <sitemap>.  WordPress
    # page sitemaps often nest image:image/image:loc elements; treating those
    # as pages inflated Rope365's 197-page map to more than 2,500 candidates.
    locations = []
    for container in list(root):
        for element in list(container):
            if element.tag.rsplit("}", 1)[-1].lower() == "loc" and element.text:
                locations.append(element.text.strip())
                break
    return kind, locations


def validate_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "rope-resource-manifest-v1":
        raise ValueError(f"{path}: unsupported manifest schema")
    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError(f"{path}: resources must be a non-empty list")
    seen: set[str] = set()
    for index, resource in enumerate(resources):
        location = f"{path}:resources[{index}]"
        if not isinstance(resource, dict):
            raise ValueError(f"{location}: expected an object")
        resource_id = resource.get("id")
        if not isinstance(resource_id, str) or not re.fullmatch(r"[a-z0-9_]+", resource_id):
            raise ValueError(f"{location}: invalid resource id")
        if resource_id in seen:
            raise ValueError(f"{location}: duplicate resource id {resource_id!r}")
        seen.add(resource_id)
        for field_name in ("name", "category", "purpose", "canonical_url"):
            if not isinstance(resource.get(field_name), str) or not resource[field_name].strip():
                raise ValueError(f"{location}: missing non-empty {field_name}")
        parsed = urlsplit(resource["canonical_url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username:
            raise ValueError(f"{location}: canonical_url must be a public HTTPS URL")
        collection = resource.get("collection")
        if not isinstance(collection, dict) or not isinstance(collection.get("mode"), str):
            raise ValueError(f"{location}: collection.mode must be a string")
        if collection["mode"] not in SUPPORTED_MODES:
            raise ValueError(
                f"{location}: unsupported collection mode {collection['mode']!r}"
            )
        max_pages = collection.get("max_pages", 1)
        if not isinstance(max_pages, int) or isinstance(max_pages, bool) or max_pages < 1:
            raise ValueError(f"{location}: max_pages must be a positive integer")
        for key in ("urls", "include_urls", "explicit_urls", "sitemaps", "sitemap_urls"):
            if key not in collection:
                continue
            values = collection[key]
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{location}: collection.{key} must be a list of URLs")
            for value in values:
                parsed_value = urlsplit(value)
                if (
                    parsed_value.scheme != "https"
                    or not parsed_value.hostname
                    or parsed_value.username
                    or not same_site(resource["canonical_url"], value)
                ):
                    raise ValueError(
                        f"{location}: collection.{key} contains a non-HTTPS or cross-site URL"
                    )
        for key in ("include_url_patterns", "exclude_url_patterns"):
            if key not in collection:
                continue
            patterns = collection[key]
            if not isinstance(patterns, list) or any(
                not isinstance(pattern, str) or not pattern for pattern in patterns
            ):
                raise ValueError(
                    f"{location}: collection.{key} must be a list of regex strings"
                )
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as error:
                    raise ValueError(
                        f"{location}: invalid collection.{key} regex {pattern!r}: {error}"
                    ) from error
    return manifest, resources


def new_coverage(
    resource: Mapping[str, Any], manifest_path: Path, manifest_hash: str
) -> dict[str, Any]:
    collection = resource["collection"]
    return {
        "schema": "resource-corpus-coverage-v1",
        "resource_id": resource["id"],
        "resource_name": resource["name"],
        "resource_category": resource["category"],
        "resource_purpose": resource["purpose"],
        "canonical_url": resource["canonical_url"],
        "collection_mode": collection["mode"],
        "collection_policy": dict(collection),
        "manifest": portable_path(manifest_path),
        "manifest_sha256": manifest_hash,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "counts": {"fetched": 0, "skipped": 0, "blocked": 0, "failed": 0},
        "count_semantics": (
            "fetched counts saved documents; skipped/blocked/failed count explicit "
            "page or sitemap decisions in this run"
        ),
        "results": {"fetched": [], "skipped": [], "blocked": [], "failed": []},
        "robots": None,
        "sitemaps": [],
        "limits": {
            "max_pages": int(collection.get("max_pages", 1)),
            "max_response_bytes": MAX_RESPONSE_BYTES,
            "max_youtube_items": (
                MAX_YOUTUBE_ITEMS
                if collection["mode"] in {"youtube_video", "youtube_playlist"}
                else None
            ),
        },
    }


def add_result(
    coverage: dict[str, Any],
    outcome: str,
    url: str,
    reason: str,
    **details: Any,
) -> None:
    if outcome not in {"fetched", "skipped", "blocked", "failed"}:
        raise ValueError(f"unknown outcome {outcome!r}")
    item = {"reason": reason, "url": url}
    item.update({key: value for key, value in details.items() if value is not None})
    coverage["results"][outcome].append(item)
    coverage["counts"][outcome] += 1


class ResourceCollector:
    def __init__(
        self,
        *,
        session: Any,
        manifest_path: Path,
        output_dir: Path,
        coverage_dir: Path,
        youtube_adapter: YouTubeAdapter | None = None,
        extractor: Callable[[str, str], tuple[str, str, str]] = extract_html,
        user_agent: str = USER_AGENT,
        timeout: float = 30.0,
        delay: float = 1.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        min_text_chars: int = MIN_TEXT_CHARS,
    ) -> None:
        self.session = session
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.coverage_dir = Path(coverage_dir)
        self.youtube_adapter = youtube_adapter
        self.extractor = extractor
        self.user_agent = user_agent
        self.timeout = timeout
        self.delay = max(0.0, delay)
        self.max_response_bytes = max_response_bytes
        self.min_text_chars = min_text_chars
        self.manifest_hash = file_sha256(self.manifest_path)
        auth = getattr(session, "auth", None)
        session_headers = getattr(session, "headers", {})
        cookies = getattr(session, "cookies", None)
        if auth:
            raise ValueError("authenticated HTTP sessions are not accepted")
        if isinstance(session_headers, Mapping) and any(
            str(key).lower() in {"authorization", "cookie"}
            for key in session_headers
        ):
            raise ValueError("HTTP sessions with authorization or cookie headers are not accepted")
        if cookies is not None and len(cookies):
            raise ValueError("HTTP sessions with cookies are not accepted")
        self._robots: dict[str, RobotsInfo] = {}
        self._last_request: dict[str, float] = {}
        self._origin_delays: dict[str, float] = {}

    def _wait(self, url: str) -> None:
        origin = origin_for(url)
        effective_delay = max(self.delay, self._origin_delays.get(origin, 0.0))
        if effective_delay <= 0:
            return
        last = self._last_request.get(origin)
        if last is not None:
            remaining = effective_delay - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)

    def _get(self, url: str) -> Any:
        self._wait(url)
        response = self.session.get(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xml;q=0.9,*/*;q=0.1",
            },
            timeout=self.timeout,
            allow_redirects=True,
        )
        self._last_request[origin_for(url)] = time.monotonic()
        return response

    def _response_bytes(self, response: Any) -> bytes:
        content = getattr(response, "content", None)
        if content is None:
            content = str(getattr(response, "text", "")).encode("utf-8")
        if not isinstance(content, bytes):
            content = bytes(content)
        if len(content) > self.max_response_bytes:
            raise ValueError(
                f"response exceeds {self.max_response_bytes} byte safety limit"
            )
        return content

    def robots_for(self, url: str) -> RobotsInfo:
        origin = origin_for(url)
        if origin in self._robots:
            return self._robots[origin]
        robots_url = urljoin(origin + "/", "robots.txt")
        info = RobotsInfo(origin=origin, url=robots_url, status="unavailable")
        self._robots[origin] = info
        try:
            response = self._get(robots_url)
            info.http_status = int(getattr(response, "status_code", 0))
            info.retrieved_at = utc_now()
            final_url = normalize_url(getattr(response, "url", None) or robots_url)
            if not same_site(url, final_url):
                info.status = "blocked"
                info.reason = "robots.txt redirected off site; fail closed"
                return info
            if info.http_status == 404:
                info.status = "not_found"
                info.reason = "robots.txt returned 404; no rules declared"
                return info
            if info.http_status in {401, 403, 429} or info.http_status >= 500:
                info.status = "blocked"
                info.reason = f"robots.txt unavailable with HTTP {info.http_status}; fail closed"
                return info
            if not 200 <= info.http_status < 300:
                info.status = "blocked"
                info.reason = f"unexpected robots.txt HTTP {info.http_status}; fail closed"
                return info
            body = self._response_bytes(response).decode("utf-8", errors="replace")
            signal_header = header_value(getattr(response, "headers", {}), "Content-Signal")
            info.ai_train_no = has_no_train_signal(body, signal_header)
            info.sitemaps = [
                match.strip()
                for match in re.findall(r"^\s*Sitemap\s*:\s*(\S+)\s*$", body, re.I | re.M)
                if match.strip().startswith("https://")
            ]
            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(body.splitlines())
            info.parser = parser
            crawl_delay = parser.crawl_delay(self.user_agent)
            if crawl_delay is None:
                crawl_delay = parser.crawl_delay("*")
            request_rate = parser.request_rate(self.user_agent)
            if request_rate is None:
                request_rate = parser.request_rate("*")
            policy_delay = float(crawl_delay or 0)
            if request_rate and request_rate.requests > 0:
                policy_delay = max(
                    policy_delay,
                    float(request_rate.seconds) / float(request_rate.requests),
                )
            if policy_delay:
                self._origin_delays[origin] = policy_delay
            info.status = "ok"
            if info.ai_train_no:
                info.reason = "robots.txt declares Content-Signal ai-train=no"
            return info
        except Exception as error:
            info.status = "unavailable"
            info.reason = (
                "robots.txt request failed; fail closed: "
                f"{type(error).__name__}: {error}"
            )
            return info

    def _policy_for(self, resource: Mapping[str, Any]) -> ResourcePolicy:
        mode = resource["collection"]["mode"]
        if mode in {"exact_public_page", "reference_only"}:
            return ResourcePolicy(exact_only=True)
        configured = RESOURCE_POLICIES.get(str(resource["id"]))
        if configured is not None:
            return configured
        collection = resource["collection"]
        has_declared_sitemap = any(
            isinstance(collection.get(key), list) and collection[key]
            for key in ("sitemaps", "sitemap_urls")
        )
        # A manifest-owned sitemap is sufficient for generic discovery only when
        # it also supplies a positive URL filter.  This keeps an unfamiliar shop
        # from becoming a broad crawl merely because it advertises a sitemap.
        if has_declared_sitemap and collection.get("include_url_patterns"):
            child_pattern = {
                "relevant_product_pages": PAGE_OR_PRODUCT_SITEMAP_RE,
                "reference_plus_narrow_specs": PAGE_OR_PRODUCT_SITEMAP_RE,
                "relevant_event_pages": PAGE_OR_EVENT_SITEMAP_RE,
                "public_catalog_and_descriptions": re.compile(
                    r"/(?:sitemap|page-sitemap(?:\d+)?)\.xml$", re.I
                ),
            }.get(mode, PAGE_SITEMAP_RE)
            return ResourcePolicy(
                sitemap_child_pattern=child_pattern,
                page_path_pattern=re.compile(r"."),
            )
        return ResourcePolicy(exact_only=True)

    def _explicit_urls(self, resource: Mapping[str, Any]) -> list[str]:
        collection = resource["collection"]
        configured = collection.get("urls")
        raw_urls: list[Any] = (
            list(configured)
            if isinstance(configured, list) and configured
            else [resource["canonical_url"]]
        )
        for key in ("urls", "include_urls", "explicit_urls"):
            if key == "urls":
                continue
            value = collection.get(key, [])
            if isinstance(value, list):
                raw_urls.extend(value)
        output: list[str] = []
        for value in raw_urls:
            if not isinstance(value, str):
                continue
            normalized = normalize_url(value)
            if normalized.startswith("https://") and same_site(
                resource["canonical_url"], normalized
            ):
                output.append(normalized)
        return list(dict.fromkeys(output))

    def _manifest_url_decision(
        self,
        resource: Mapping[str, Any],
        url: str,
        *,
        require_include_match: bool,
    ) -> tuple[bool, str | None]:
        """Apply owner-supplied URL regexes, with exclusions taking precedence."""

        collection = resource["collection"]
        exclude_patterns = collection.get("exclude_url_patterns", [])
        if any(re.search(pattern, url) for pattern in exclude_patterns):
            return False, "manifest_exclude_pattern"
        include_patterns = collection.get("include_url_patterns", [])
        if (
            require_include_match
            and include_patterns
            and not any(re.search(pattern, url) for pattern in include_patterns)
        ):
            return False, "manifest_include_pattern_miss"
        return True, None

    def _fetch_sitemap(
        self,
        resource: Mapping[str, Any],
        policy: ResourcePolicy,
        url: str,
        robots: RobotsInfo,
        coverage: dict[str, Any],
        seen: set[str],
        page_urls: list[str],
    ) -> None:
        normalized = normalize_url(url, keep_query=False)
        if normalized in seen:
            return
        if len(seen) >= MAX_SITEMAPS_PER_RESOURCE:
            add_result(coverage, "skipped", normalized, "sitemap_limit")
            return
        seen.add(normalized)
        if not same_site(resource["canonical_url"], normalized):
            add_result(coverage, "skipped", normalized, "cross_origin_sitemap")
            return
        if not robots.allows(self.user_agent, normalized):
            add_result(coverage, "blocked", normalized, "robots_disallow_sitemap")
            return
        sitemap_record: dict[str, Any] = {"url": normalized, "status": "fetching"}
        coverage["sitemaps"].append(sitemap_record)
        try:
            response = self._get(normalized)
            status = int(getattr(response, "status_code", 0))
            sitemap_record["http_status"] = status
            sitemap_record["retrieved_at"] = utc_now()
            final_url = normalize_url(getattr(response, "url", None) or normalized)
            if not same_site(resource["canonical_url"], final_url):
                sitemap_record["status"] = "blocked"
                sitemap_record["final_url"] = final_url
                add_result(
                    coverage,
                    "blocked",
                    normalized,
                    "cross_origin_sitemap_redirect",
                    final_url=final_url,
                )
                return
            if not robots.allows(self.user_agent, final_url):
                sitemap_record["status"] = "blocked"
                sitemap_record["final_url"] = final_url
                add_result(
                    coverage,
                    "blocked",
                    normalized,
                    "robots_disallow_sitemap_final_url",
                    final_url=final_url,
                )
                return
            if status in {401, 403, 451}:
                sitemap_record["status"] = "blocked"
                add_result(coverage, "blocked", normalized, f"sitemap_http_{status}")
                return
            if not 200 <= status < 300:
                sitemap_record["status"] = "failed"
                add_result(coverage, "failed", normalized, f"sitemap_http_{status}")
                return
            content = self._response_bytes(response)
            text = content.decode("utf-8", errors="replace")
            if has_no_train_signal(
                header_value(getattr(response, "headers", {}), "Content-Signal"),
                text[:4096],
            ):
                sitemap_record["status"] = "blocked"
                sitemap_record["content_signal"] = "ai-train=no"
                add_result(coverage, "blocked", normalized, "sitemap_content_signal_ai_train_no")
                return
            kind, locations = parse_sitemap_locations(text)
            sitemap_record.update(
                {"status": "fetched", "kind": kind, "locations": len(locations)}
            )
            if kind == "sitemapindex":
                for child in locations:
                    child_url = normalize_url(urljoin(normalized, child), keep_query=False)
                    if policy.permits_sitemap_child(child_url):
                        self._fetch_sitemap(
                            resource, policy, child_url, robots, coverage, seen, page_urls
                        )
                    else:
                        add_result(coverage, "skipped", child_url, "sitemap_out_of_scope")
                return
            for item in locations[:MAX_SITEMAP_URLS]:
                candidate = normalize_url(urljoin(normalized, item), keep_query=False)
                manifest_allowed, manifest_reason = self._manifest_url_decision(
                    resource, candidate, require_include_match=True
                )
                if not manifest_allowed:
                    add_result(
                        coverage,
                        "skipped",
                        candidate,
                        manifest_reason or "manifest_url_filter",
                    )
                elif not same_site(resource["canonical_url"], candidate):
                    add_result(coverage, "skipped", candidate, "cross_origin_page")
                elif ASSET_PATH_RE.search(candidate):
                    add_result(coverage, "skipped", candidate, "asset_url")
                elif GATED_PATH_RE.search(urlsplit(candidate).path):
                    add_result(coverage, "blocked", candidate, "gated_path")
                elif COMMON_IRRELEVANT_PATH_RE.search(urlsplit(candidate).path):
                    add_result(coverage, "skipped", candidate, "irrelevant_path")
                elif policy.permits_page(candidate, resource["canonical_url"]):
                    page_urls.append(candidate)
                else:
                    add_result(coverage, "skipped", candidate, "page_out_of_scope")
            if len(locations) > MAX_SITEMAP_URLS:
                coverage.setdefault("notes", []).append(
                    f"{len(locations) - MAX_SITEMAP_URLS} sitemap URLs omitted at safety limit"
                )
        except Exception as error:
            sitemap_record["status"] = "failed"
            sitemap_record["error"] = f"{type(error).__name__}: {error}"
            add_result(
                coverage,
                "failed",
                normalized,
                "sitemap_request_or_parse_failed",
                error=sitemap_record["error"],
            )

    def _candidate_urls(
        self,
        resource: Mapping[str, Any],
        robots: RobotsInfo,
        coverage: dict[str, Any],
    ) -> list[str]:
        policy = self._policy_for(resource)
        collection = resource["collection"]
        candidates = []
        for explicit in self._explicit_urls(resource):
            allowed, reason = self._manifest_url_decision(
                resource, explicit, require_include_match=False
            )
            if allowed:
                candidates.append(explicit)
            else:
                add_result(coverage, "skipped", explicit, reason or "manifest_url_filter")
        declared = []
        for key in ("sitemaps", "sitemap_urls"):
            value = collection.get(key, [])
            if isinstance(value, list):
                declared.extend(value)
        has_explicit_plan = bool(collection.get("urls"))
        if not policy.exact_only and (declared or not has_explicit_plan):
            roots = (
                declared
                if declared
                else list(policy.explicit_sitemaps) + list(robots.sitemaps)
            )
            seen_sitemaps: set[str] = set()
            discovered: list[str] = []
            for sitemap in dict.fromkeys(str(item) for item in roots if item):
                self._fetch_sitemap(
                    resource, policy, sitemap, robots, coverage, seen_sitemaps, discovered
                )
            candidates.extend(discovered)
        candidates = list(dict.fromkeys(candidates))
        max_pages = int(collection.get("max_pages", 1))
        for excess in candidates[max_pages:]:
            add_result(coverage, "skipped", excess, "max_pages")
        return candidates[:max_pages]

    def _document_path(self, resource_id: str, url: str) -> Path:
        suffix = sha256_text(normalize_url(url))[:20]
        return self.output_dir / f"{resource_id}__{suffix}.json"

    def _clear_previous_documents(self, resource_id: str) -> int:
        """Prevent stale documents from surviving a newly blocked/failed run."""

        if not self.output_dir.exists():
            return 0
        removed = 0
        for path in self.output_dir.glob(f"{resource_id}__*.json"):
            if path.is_file():
                path.unlink()
                removed += 1
        return removed

    def _write_document(self, record: dict[str, Any]) -> Path:
        path = self._document_path(record["resource_id"], record["url"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return path

    def _page_record(
        self,
        resource: Mapping[str, Any],
        requested_url: str,
        final_url: str,
        response: Any,
        raw: bytes,
        title: str,
        text: str,
        extractor_name: str,
        robots: RobotsInfo,
    ) -> dict[str, Any]:
        collection = resource["collection"]
        retrieved_at = utc_now()
        return {
            "schema": "resource-corpus-document-v1",
            "source": resource["id"],
            "resource_id": resource["id"],
            "resource_name": resource["name"],
            "resource_category": resource["category"],
            "resource_purpose": resource["purpose"],
            "canonical_resource_url": resource["canonical_url"],
            "url": final_url,
            "title": title,
            "text": text,
            "document_sha256": sha256_text(text),
            "raw_response_sha256": sha256_bytes(raw),
            "source_lineage": {
                "manifest": portable_path(self.manifest_path),
                "manifest_sha256": self.manifest_hash,
                "requested_url": requested_url,
                "final_url": final_url,
                "retrieved_at": retrieved_at,
                "http_status": int(getattr(response, "status_code", 0)),
                "content_type": header_value(getattr(response, "headers", {}), "Content-Type"),
                "extractor": extractor_name,
                "collection_mode": collection["mode"],
                "authenticated": False,
                "robots_url": robots.url,
                "robots_retrieved_at": robots.retrieved_at,
                "content_signal": "no ai-train=no directive observed",
            },
            "collection_constraints": {
                "content_use": collection.get("content_use"),
                "excluded": list(collection.get("exclude", [])),
                "known_conflicts": list(collection.get("known_conflicts", [])),
                "known_issues": list(collection.get("known_issues", [])),
                "owner_note": resource.get("owner_note"),
                "requires_javascript": bool(collection.get("requires_javascript", False)),
                "volatile": list(collection.get("volatile", [])),
            },
        }

    def _collect_page(
        self,
        resource: Mapping[str, Any],
        url: str,
        robots: RobotsInfo,
        coverage: dict[str, Any],
    ) -> None:
        if not robots.allows(self.user_agent, url):
            add_result(coverage, "blocked", url, "robots_disallow")
            return
        try:
            response = self._get(url)
            status = int(getattr(response, "status_code", 0))
            final_url = normalize_url(getattr(response, "url", None) or url)
            if status in {401, 403, 451}:
                add_result(coverage, "blocked", url, f"http_{status}", final_url=final_url)
                return
            if not 200 <= status < 300:
                add_result(coverage, "failed", url, f"http_{status}", final_url=final_url)
                return
            if not same_site(resource["canonical_url"], final_url):
                add_result(coverage, "blocked", url, "cross_origin_redirect", final_url=final_url)
                return
            if (
                normalize_url(final_url) != normalize_url(url)
                and GATED_PATH_RE.search(urlsplit(final_url).path)
            ):
                add_result(
                    coverage,
                    "blocked",
                    url,
                    "redirected_to_gated_path",
                    final_url=final_url,
                )
                return
            if not robots.allows(self.user_agent, final_url):
                add_result(
                    coverage,
                    "blocked",
                    url,
                    "robots_disallow_final_url",
                    final_url=final_url,
                )
                return
            content_type = header_value(getattr(response, "headers", {}), "Content-Type")
            if content_type and not any(
                item in content_type.lower()
                for item in ("text/html", "application/xhtml+xml", "text/plain")
            ):
                add_result(
                    coverage,
                    "skipped",
                    url,
                    "unsupported_content_type",
                    content_type=content_type,
                )
                return
            raw = self._response_bytes(response)
            html = raw.decode(getattr(response, "encoding", None) or "utf-8", errors="replace")
            if has_no_train_signal(
                header_value(getattr(response, "headers", {}), "Content-Signal"),
                html[:8192],
            ):
                add_result(coverage, "blocked", url, "page_content_signal_ai_train_no")
                return
            if ACCESS_CHALLENGE_RE.search(html):
                add_result(coverage, "blocked", url, "access_challenge")
                return
            if re.search(
                r"memberpress[^\n]{0,120}unauthorized|"
                r"class=[\"'][^\"']*unauthorized",
                html,
                re.I,
            ):
                add_result(coverage, "blocked", url, "member_only_page")
                return
            title, text, extractor_name = self.extractor(html, final_url)
            if len(text.strip()) < self.min_text_chars:
                add_result(
                    coverage,
                    "skipped",
                    url,
                    "insufficient_extracted_text",
                    extracted_chars=len(text.strip()),
                )
                return
            record = self._page_record(
                resource,
                url,
                final_url,
                response,
                raw,
                title,
                text.strip(),
                extractor_name,
                robots,
            )
            path = self._write_document(record)
            add_result(
                coverage,
                "fetched",
                final_url,
                "document_saved",
                requested_url=url,
                output=portable_path(path),
                document_sha256=record["document_sha256"],
                text_chars=len(record["text"]),
            )
        except Exception as error:
            add_result(
                coverage,
                "failed",
                url,
                "page_request_or_extract_failed",
                error=f"{type(error).__name__}: {error}",
            )

    def _youtube_record(
        self, resource: Mapping[str, Any], item: Mapping[str, Any]
    ) -> dict[str, Any]:
        title = str(item.get("title") or resource["name"]).strip()
        description = str(item.get("description") or "").strip()
        transcript = str(item.get("transcript") or "").strip()
        text_parts = [part for part in (title, description, transcript) if part]
        text = "\n\n".join(text_parts)
        url = normalize_url(str(item.get("url") or resource["canonical_url"]))
        parsed_url = urlsplit(url)
        youtube_host = (parsed_url.hostname or "").lower()
        if parsed_url.scheme != "https" or youtube_host not in {
            "youtube.com",
            "www.youtube.com",
            "youtu.be",
        }:
            raise ValueError(f"YouTube adapter returned an off-platform URL: {url!r}")
        return {
            "schema": "resource-corpus-document-v1",
            "source": resource["id"],
            "resource_id": resource["id"],
            "resource_name": resource["name"],
            "resource_category": resource["category"],
            "resource_purpose": resource["purpose"],
            "canonical_resource_url": resource["canonical_url"],
            "url": url,
            "title": title,
            "text": text,
            "document_sha256": sha256_text(text),
            "source_lineage": {
                "manifest": portable_path(self.manifest_path),
                "manifest_sha256": self.manifest_hash,
                "retrieved_at": utc_now(),
                "collection_mode": resource["collection"]["mode"],
                "adapter": getattr(
                    self.youtube_adapter,
                    "name",
                    type(self.youtube_adapter).__name__,
                ),
                "creator": item.get("creator"),
                "caption_kind": item.get("caption_kind"),
                "authenticated": False,
            },
            "collection_constraints": {
                "content_use": resource["collection"].get("content_use"),
                "excluded": list(resource["collection"].get("exclude", [])),
                "known_conflicts": list(
                    resource["collection"].get("known_conflicts", [])
                ),
                "known_issues": list(resource["collection"].get("known_issues", [])),
                "owner_note": resource.get("owner_note"),
                "requires_javascript": bool(
                    resource["collection"].get("requires_javascript", False)
                ),
                "volatile": list(resource["collection"].get("volatile", [])),
            },
        }

    def _collect_youtube(
        self, resource: Mapping[str, Any], coverage: dict[str, Any]
    ) -> None:
        url = resource["canonical_url"]
        coverage["robots"] = {
            "status": "adapter_managed",
            "reason": "specialized public-data adapter; no HTML crawl",
        }
        if self.youtube_adapter is None:
            add_result(
                coverage,
                "blocked",
                url,
                "youtube_adapter_unavailable",
                detail=(
                    "install yt-dlp or inject a policy-compatible adapter; "
                    "no fallback scrape attempted"
                ),
            )
            coverage["status"] = "blocked"
            return
        try:
            items = self.youtube_adapter.collect(resource)
            if not items:
                add_result(coverage, "skipped", url, "youtube_adapter_returned_no_public_items")
            for item in items[:MAX_YOUTUBE_ITEMS]:
                record = self._youtube_record(resource, item)
                if not record["text"].strip():
                    add_result(coverage, "skipped", record["url"], "empty_youtube_metadata")
                    continue
                if len(record["text"].encode("utf-8")) > self.max_response_bytes:
                    add_result(
                        coverage,
                        "skipped",
                        record["url"],
                        "youtube_item_too_large",
                    )
                    continue
                path = self._write_document(record)
                add_result(
                    coverage,
                    "fetched",
                    record["url"],
                    "document_saved",
                    output=portable_path(path),
                    document_sha256=record["document_sha256"],
                    text_chars=len(record["text"]),
                )
            for item in items[MAX_YOUTUBE_ITEMS:]:
                add_result(
                    coverage,
                    "skipped",
                    str(item.get("url") or url),
                    "youtube_item_limit",
                )
        except YouTubeAccessBlocked as error:
            add_result(coverage, "blocked", url, "youtube_access_blocked", detail=str(error))
        except YouTubeAdapterUnavailable as error:
            add_result(coverage, "blocked", url, "youtube_adapter_unavailable", detail=str(error))
        except Exception as error:
            add_result(
                coverage,
                "failed",
                url,
                "youtube_adapter_failed",
                error=f"{type(error).__name__}: {error}",
            )

    def collect_resource(self, resource: Mapping[str, Any]) -> dict[str, Any]:
        coverage = new_coverage(resource, self.manifest_path, self.manifest_hash)
        coverage["limits"]["max_response_bytes"] = self.max_response_bytes
        mode = resource["collection"]["mode"]
        url = resource["canonical_url"]
        try:
            coverage["previous_documents_removed"] = self._clear_previous_documents(
                resource["id"]
            )
            if mode == "reference_only":
                add_result(
                    coverage,
                    "blocked",
                    url,
                    "manifest_reference_only",
                    detail=resource["collection"].get("reason"),
                    policy_url=resource["collection"].get("policy_url"),
                )
                coverage["robots"] = {
                    "status": "not_requested",
                    "reason": "manifest reference_only is already a stronger restriction",
                }
            elif mode in {"youtube_video", "youtube_playlist"}:
                self._collect_youtube(resource, coverage)
            else:
                robots = self.robots_for(url)
                coverage["robots"] = robots.public_record()
                if robots.ai_train_no:
                    add_result(
                        coverage,
                        "blocked",
                        url,
                        "robots_content_signal_ai_train_no",
                    )
                elif robots.status not in {"ok", "not_found"}:
                    add_result(
                        coverage,
                        "blocked",
                        url,
                        "robots_unavailable_fail_closed",
                        detail=robots.reason,
                    )
                else:
                    candidates = self._candidate_urls(resource, robots, coverage)
                    coverage["candidate_urls"] = candidates
                    for candidate in candidates:
                        self._collect_page(resource, candidate, robots, coverage)
        except Exception as error:
            add_result(
                coverage,
                "failed",
                url,
                "unexpected_resource_collection_error",
                error=f"{type(error).__name__}: {error}",
            )
        counts = coverage["counts"]
        if counts["fetched"]:
            coverage["status"] = "complete" if not counts["failed"] else "partial"
        elif counts["failed"]:
            coverage["status"] = "failed"
        elif counts["blocked"]:
            coverage["status"] = "blocked"
        else:
            coverage["status"] = "empty"
        coverage["completed_at"] = utc_now()
        self.coverage_dir.mkdir(parents=True, exist_ok=True)
        coverage_path = self.coverage_dir / f"{resource['id']}.coverage.json"
        coverage_path.write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        return coverage

    def collect(self, resources: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        coverages = [self.collect_resource(resource) for resource in resources]
        totals = {outcome: 0 for outcome in ("fetched", "skipped", "blocked", "failed")}
        for coverage in coverages:
            for outcome in totals:
                totals[outcome] += coverage["counts"][outcome]
        return {
            "schema": "resource-corpus-run-summary-v1",
            "completed_at": utc_now(),
            "manifest": portable_path(self.manifest_path),
            "manifest_sha256": self.manifest_hash,
            "output_dir": portable_path(self.output_dir),
            "coverage_dir": portable_path(self.coverage_dir),
            "resources": len(coverages),
            "resource_statuses": {
                coverage["resource_id"]: coverage["status"] for coverage in coverages
            },
            "coverage_files": {
                coverage["resource_id"]: portable_path(
                    self.coverage_dir / f"{coverage['resource_id']}.coverage.json")
                for coverage in coverages
            },
            "counts": totals,
        }


class YtDlpAdapter:
    """Optional metadata/manual-caption adapter backed by ``yt-dlp``.

    The adapter deliberately supplies no cookies, credentials, geo bypass, or age
    gate workaround.  It reads creator-provided ``subtitles`` only and ignores
    ``automatic_captions``.
    """

    name = "yt-dlp-public-metadata-v1"

    def __init__(self, module: Any, session: Any, timeout: float = 30.0) -> None:
        self.module = module
        self.session = session
        self.timeout = timeout

    def _subtitle_text(self, entry: Mapping[str, Any]) -> tuple[str, str | None]:
        subtitles = entry.get("subtitles")
        if not isinstance(subtitles, dict) or not subtitles:
            return "", None
        languages = sorted(
            subtitles,
            key=lambda language: (not language.lower().startswith("en"), language),
        )
        for language in languages:
            tracks = subtitles.get(language)
            if not isinstance(tracks, list):
                continue
            preferred = sorted(
                (track for track in tracks if isinstance(track, dict) and track.get("url")),
                key=lambda track: str(track.get("ext")) not in {"vtt", "srv3", "json3"},
            )
            for track in preferred:
                response = self.session.get(
                    track["url"],
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    allow_redirects=True,
                )
                status = int(getattr(response, "status_code", 0))
                if status in {401, 403, 451}:
                    raise YouTubeAccessBlocked(f"creator captions returned HTTP {status}")
                if not 200 <= status < 300:
                    continue
                content = getattr(response, "content", b"")
                if len(content) > MAX_RESPONSE_BYTES:
                    continue
                if has_no_train_signal(
                    header_value(getattr(response, "headers", {}), "Content-Signal")
                ):
                    raise YouTubeAccessBlocked("creator captions declare ai-train=no")
                caption = content.decode("utf-8", errors="replace")
                caption = re.sub(r"^WEBVTT.*?$", "", caption, flags=re.I | re.M)
                caption = re.sub(
                    r"^\s*(?:\d+:)?\d{2}:\d{2}[.,]\d{3}\s+-->.*?$",
                    "",
                    caption,
                    flags=re.M,
                )
                caption = re.sub(r"<[^>]+>", "", caption)
                caption = re.sub(r"\n{3,}", "\n\n", caption).strip()
                return caption, language
        return "", None

    def collect(self, resource: Mapping[str, Any]) -> list[dict[str, Any]]:
        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "playlistend": MAX_YOUTUBE_ITEMS,
            "geo_bypass": False,
            "cookiefile": None,
        }
        try:
            with self.module.YoutubeDL(options) as downloader:
                info = downloader.extract_info(resource["canonical_url"], download=False)
        except Exception as error:
            message = str(error)
            if re.search(
                r"private|members[- ]only|sign in|login|age[- ]restricted|"
                r"not available in your country",
                message,
                re.I,
            ):
                raise YouTubeAccessBlocked(message) from error
            raise YouTubeAdapterUnavailable(message) from error
        if not isinstance(info, dict):
            return []
        raw_entries = info.get("entries")
        if raw_entries is None:
            entries = [info]
        else:
            try:
                entries = list(raw_entries)[:MAX_YOUTUBE_ITEMS]
            except TypeError:
                entries = [info]
        output = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            transcript, language = self._subtitle_text(entry)
            webpage_url = entry.get("webpage_url")
            if not webpage_url and entry.get("id"):
                webpage_url = f"https://www.youtube.com/watch?v={entry['id']}"
            output.append(
                {
                    "url": webpage_url or resource["canonical_url"],
                    "title": entry.get("title") or resource["name"],
                    "description": entry.get("description") or "",
                    "creator": entry.get("uploader") or entry.get("channel"),
                    "transcript": transcript,
                    "caption_kind": f"creator-provided:{language}" if language else None,
                }
            )
        return output


def optional_youtube_adapter(session: Any, timeout: float) -> YouTubeAdapter | None:
    try:
        import yt_dlp  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return None
    return YtDlpAdapter(yt_dlp, session, timeout)


def collect_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    coverage_dir: Path = DEFAULT_COVERAGE_DIR,
    *,
    resource_ids: Iterable[str] | None = None,
    session: Any | None = None,
    youtube_adapter: YouTubeAdapter | None = None,
    auto_youtube_adapter: bool = True,
    timeout: float = 30.0,
    delay: float = 1.0,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    min_text_chars: int = MIN_TEXT_CHARS,
    extractor: Callable[[str, str], tuple[str, str, str]] = extract_html,
) -> dict[str, Any]:
    """Collect selected resources and return an aggregate in-memory summary."""

    _, resources = validate_manifest(Path(manifest_path))
    requested = set(resource_ids or [])
    known = {resource["id"] for resource in resources}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"unknown resource ids: {unknown}")
    if requested:
        resources = [resource for resource in resources if resource["id"] in requested]
    if session is None:
        import requests

        session = requests.Session()
    adapter = youtube_adapter
    if adapter is None and auto_youtube_adapter:
        adapter = optional_youtube_adapter(session, timeout)
    collector = ResourceCollector(
        session=session,
        manifest_path=Path(manifest_path),
        output_dir=Path(output_dir),
        coverage_dir=Path(coverage_dir),
        youtube_adapter=adapter,
        extractor=extractor,
        timeout=timeout,
        delay=delay,
        max_response_bytes=max_response_bytes,
        min_text_chars=min_text_chars,
    )
    return collector.collect(resources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY,
        help="write the compact tracked run summary (use '-' to disable)",
    )
    parser.add_argument(
        "--resource",
        dest="resource_ids",
        action="append",
        help="resource ID to collect (repeatable; default: all)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument(
        "--disable-youtube",
        action="store_true",
        help="do not load the optional yt-dlp adapter; coverage will report it unavailable",
    )
    args = parser.parse_args()
    summary = collect_manifest(
        args.manifest,
        args.output_dir,
        args.coverage_dir,
        resource_ids=args.resource_ids,
        auto_youtube_adapter=not args.disable_youtube,
        timeout=args.timeout,
        delay=args.delay,
    )
    if str(args.summary) != "-":
        from build_resource_coverage_report import build as build_coverage_report
        build_coverage_report(args.manifest, args.coverage_dir, args.summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
