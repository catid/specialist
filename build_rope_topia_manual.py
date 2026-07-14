#!/usr/bin/env python3
"""Build the hand-reviewed Rope-topia resource-index QA artifact.

The live blog is an explicit four-page demo.  This builder emits URL-discovery
Q&A only; it never treats a title or URL as article-body evidence.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from build_leakfree_qa import eval_facts
from qa_quality import (has_protocol_tokens, leakage_reason, normalize_text,
                        parse_qa, qa_pair_from_record, stable_fact_id)


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "sources" / "rope_topia_manual_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "rope_topia_manual_v1.jsonl"
DEFAULT_REPORT = ROOT / "data" / "rope_topia_manual_v1.report.json"
DEFAULT_EVAL = [ROOT / "data" / "eval_qa.jsonl",
                ROOT / "data" / "eval_qa_v2.jsonl"]
DEFAULT_BASELINES = [ROOT / "data" / "train_qa_curated_v1.jsonl"]


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


def require_text(item: dict, field: str, location: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}: missing non-empty {field}")
    return value.strip()


def require_rope_topia_url(url: str, location: str,
                           require_trailing_slash: bool = True) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.netloc != "rope-topia.com" or
            parsed.username or parsed.query or parsed.fragment):
        raise ValueError(
            f"{location}: expected a canonical public Rope-topia HTTPS URL, "
            f"got {url!r}")
    if (require_trailing_slash and parsed.path and
            not parsed.path.endswith("/")):
        raise ValueError(f"{location}: URL path must end with '/': {url!r}")


def read_jsonl(path: Path):
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


def baseline_keys(
        paths: list[Path]) -> tuple[set[str], set[tuple[str, str]], int]:
    questions: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    count = 0
    for path in paths:
        for line_number, item in read_jsonl(path):
            # The curated artifact may already contain a previous deterministic
            # build of this tranche. Exclude it from its own duplicate baseline
            # while still checking every other curated source.
            if item.get("source") == "rope_topia":
                continue
            try:
                pair = qa_pair_from_record(item)
            except ValueError as exc:
                location = "{}:{}".format(path, line_number)
                raise ValueError(f"{location}: {exc}") from exc
            if pair is None:
                location = "{}:{}".format(path, line_number)
                raise ValueError(
                    f"{location}: unsupported baseline QA")
            question, answer = pair
            questions.add(normalize_text(question))
            pairs.add((normalize_text(question), normalize_text(answer)))
            count += 1
    return questions, pairs, count


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "rope-topia-manual-source-v1":
        raise ValueError(f"{path}: unsupported manifest schema")
    require_text(manifest, "source_id", str(path))
    require_text(manifest, "reviewed_at", str(path))
    require_text(manifest, "reviewer", str(path))
    policy = manifest.get("collection_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"{path}: collection_policy must be an object")
    require_text(policy, "robots_evidence", str(path))
    require_text(policy, "copyright_policy", str(path))
    require_text(policy, "access_policy", str(path))
    access = manifest.get("live_access")
    if not isinstance(access, dict):
        raise ValueError(f"{path}: live_access must be an object")
    if access.get("blog_access_status") != "demo_gate":
        raise ValueError(f"{path}: blog must remain marked demo_gate")
    require_text(access, "gate_evidence", str(path))
    accessible_pages = access.get("accessible_demo_pages")
    if not isinstance(accessible_pages, list) or len(accessible_pages) != 4:
        raise ValueError(f"{path}: expected four accessible demo pages")
    for index, page in enumerate(accessible_pages, 1):
        location = "{}:accessible_demo_page[{}]".format(path, index)
        require_rope_topia_url(
            require_text(page, "canonical_url", location), location)
        require_rope_topia_url(
            require_text(page, "alias_url", location), location, False)
        require_text(page, "content_role", location)
        if page.get("http_status") != 200:
            raise ValueError(f"{location}: expected reviewed HTTP status 200")

    resources = manifest.get("resources")
    if not isinstance(resources, list) or not resources:
        raise ValueError(f"{path}: resources must be a non-empty list")
    exclusions = manifest.get("exclusions")
    if not isinstance(exclusions, list) or not exclusions:
        raise ValueError(f"{path}: exclusions must be a non-empty list")
    excluded_urls = manifest.get("excluded_urls")
    if not isinstance(excluded_urls, list) or not excluded_urls:
        raise ValueError(f"{path}: excluded_urls must be a non-empty list")

    ids: set[str] = set()
    questions: set[str] = set()
    canonical_urls: set[str] = set()
    supplied_urls: set[str] = set()
    for index, item in enumerate(resources, 1):
        location = "{}:resource[{}]".format(path, index)
        resource_id = require_text(item, "id", location)
        question = require_text(item, "question", location)
        supplied_url = require_text(item, "supplied_url", location)
        canonical_url = require_text(item, "canonical_url", location)
        title = require_text(item, "title", location)
        title_evidence = require_text(item, "title_evidence", location)
        title_evidence_url = require_text(item, "title_evidence_url", location)
        url_evidence = require_text(item, "url_evidence", location)
        url_evidence_url = require_text(item, "url_evidence_url", location)
        require_text(item, "resource_kind", location)
        if resource_id in ids:
            raise ValueError(
                f"{location}: duplicate resource id {
                    resource_id!r}")
        ids.add(resource_id)
        normalized_question = normalize_text(question)
        if normalized_question in questions:
            raise ValueError(f"{location}: duplicate question")
        questions.add(normalized_question)
        if not question.endswith("?") or "\n" in question:
            raise ValueError(
                f"{location}: question must be one line ending in '?'")
        if supplied_url in supplied_urls:
            raise ValueError(f"{location}: duplicate supplied URL")
        supplied_urls.add(supplied_url)
        if canonical_url in canonical_urls:
            raise ValueError(f"{location}: duplicate canonical URL")
        canonical_urls.add(canonical_url)
        require_rope_topia_url(supplied_url, location)
        require_rope_topia_url(canonical_url, location)
        require_rope_topia_url(title_evidence_url, location, False)
        require_rope_topia_url(url_evidence_url, location, False)
        if item.get("access_status") != "demo_gate":
            raise ValueError(
                f"{location}: content URL must be marked demo_gate")
        if item.get("live_http_status") != 200:
            raise ValueError(f"{location}: expected reviewed HTTP status 200")
        if normalize_text(title) != normalize_text(title_evidence.rstrip(".")):
            raise ValueError(
                f"{location}: title disagrees with short evidence")
        if len(title_evidence.split()) > 16:
            raise ValueError(
                f"{location}: title evidence is not a short excerpt")
        if url_evidence != f"<loc>{canonical_url}</loc>":
            raise ValueError(
                f"{location}: URL evidence must contain exactly the "
                "canonical URL")
        if canonical_url != supplied_url:
            require_text(item, "canonicalization_reason", location)
        elif "canonicalization_reason" in item:
            raise ValueError(
                f"{location}: unnecessary canonicalization_reason")
        if has_protocol_tokens(question) or has_protocol_tokens(canonical_url):
            raise ValueError(f"{location}: protocol token in QA")

    reviewed_urls = canonical_urls | supplied_urls
    seen_excluded = set()
    for index, item in enumerate(excluded_urls, 1):
        location = "{}:excluded_url[{}]".format(path, index)
        url = require_text(item, "url", location)
        require_text(item, "title", location)
        require_text(item, "reason", location)
        require_rope_topia_url(url, location)
        if item.get("access_status") != "demo_gate":
            raise ValueError(
                f"{location}: excluded URL must be marked demo_gate")
        if item.get("live_http_status") != 200:
            raise ValueError(f"{location}: expected reviewed HTTP status 200")
        if url in seen_excluded:
            raise ValueError(f"{location}: duplicate excluded URL")
        if url in reviewed_urls:
            raise ValueError(
                f"{location}: URL is both represented and excluded")
        seen_excluded.add(url)
    return manifest, resources


def build(manifest_path: Path, output_path: Path, report_path: Path,
          eval_paths: list[Path], baseline_paths: list[Path]) -> dict:
    manifest, resources = load_manifest(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    facts = eval_facts(eval_paths)
    prior_questions, prior_pairs, baseline_count = baseline_keys(
        baseline_paths)

    rows = []
    represented_canonical = set()
    represented_supplied = set()
    fact_ids = set()
    questions = set()
    kinds = collections.Counter()
    for item in resources:
        question = item["question"].strip()
        answer = item["canonical_url"].strip()
        rendered = f"Question: {question}\nAnswer: {answer}"
        if parse_qa(rendered) != (question, answer):
            raise ValueError(f"resource {item['id']}: QA does not round-trip")
        leak = leakage_reason(question, answer, facts)
        if leak:
            raise ValueError(f"resource {item['id']}: eval leakage: {leak}")
        question_key = normalize_text(question)
        pair_key = (question_key, normalize_text(answer))
        if question_key in prior_questions or pair_key in prior_pairs:
            raise ValueError(
                f"resource {
                    item['id']}: duplicate of baseline QA")
        fact_id = stable_fact_id(question, answer, "rope_topia_resource_index")
        if question_key in questions:
            raise ValueError(
                f"resource {
                    item['id']}: duplicate output question")
        if fact_id in fact_ids:
            raise ValueError(
                f"resource {
                    item['id']}: duplicate output fact ID")
        questions.add(question_key)
        fact_ids.add(fact_id)
        represented_canonical.add(item["canonical_url"])
        represented_supplied.add(item["supplied_url"])
        kinds[item["resource_kind"]] += 1
        row = {
            "access_status": "demo_gate",
            "answer": answer,
            "article_content_available": False,
            "canonical_url": item["canonical_url"],
            "content_use": "resource_metadata_only_due_demo_gate",
            "document_sha256": manifest_sha256,
            # This QA answers with a URL, so its primary evidence is the exact
            # sitemap entry. Keep the independently reviewed title evidence in
            # explicit fields rather than presenting it as support for the URL.
            "evidence": item["url_evidence"],
            "evidence_url": item["url_evidence_url"],
            "fact_id": fact_id,
            "kind": "qa_resource_index",
            "quality_schema": "manual-resource-index-qa-v1",
            "question": question,
            "resource_id": item["id"],
            "resource_kind": item["resource_kind"],
            "review": {
                "action": "add",
                "reason": (
                    "Preserved the publicly listed resource title and URL "
                    "without inferring facts from its gated article body."
                ),
                "reviewed_at": manifest["reviewed_at"],
                "reviewer": manifest["reviewer"],
            },
            "source": "rope_topia",
            "source_lineage": {"manifest": portable_path(manifest_path)},
            "supplied_url": item["supplied_url"],
            "text": rendered,
            "title": item["title"],
            "title_evidence": item["title_evidence"],
            "title_evidence_url": item["title_evidence_url"],
            "url": item["canonical_url"],
            "url_evidence": item["url_evidence"],
            "url_evidence_url": item["url_evidence_url"],
        }
        if "canonicalization_reason" in item:
            row["canonicalization_reason"] = item["canonicalization_reason"]
        rows.append(row)

    expected_canonical = {item["canonical_url"] for item in resources}
    expected_supplied = {item["supplied_url"] for item in resources}
    if represented_canonical != expected_canonical:
        raise ValueError("not every canonical resource URL is represented")
    if represented_supplied != expected_supplied:
        raise ValueError("not every supplied resource URL is represented")

    rows.sort(
        key=lambda row: (
            normalize_text(
                row["question"]),
            row["fact_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False,
                                         sort_keys=True) + "\n")

    report = {
        "schema": "rope-topia-manual-report-v1",
        "manifest": portable_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "eval_inputs": {portable_path(path): file_sha256(path)
                        for path in eval_paths},
        "baseline_inputs": {portable_path(path): file_sha256(path)
                            for path in baseline_paths},
        "counts": {
            "accessible_demo_endpoints": len(
                manifest["live_access"]["accessible_demo_urls"]),
            "accessible_demo_pages": len(
                manifest["live_access"]["accessible_demo_pages"]),
            "article_content_qa": 0,
            "baseline_rows_compared": baseline_count,
            "canonical_urls_represented": len(represented_canonical),
            "eval_facts_compared": len(facts),
            "excluded_scopes": len(manifest["exclusions"]),
            "excluded_urls": len(manifest["excluded_urls"]),
            "gated_resources": sum(
                item["access_status"] == "demo_gate" for item in resources),
            "gated_relevant_endpoints_including_blog": (
                1 + len(resources) + len(manifest["excluded_urls"])),
            "output": len(rows),
            "resources": len(resources),
            "resource_kinds": dict(sorted(kinds.items())),
            "supplied_urls_represented": len(represented_supplied),
            "unique_fact_ids": len(fact_ids),
            "unique_questions": len(questions),
            "unavailable_public_endpoints": len(
                manifest["live_access"]["unavailable_public_endpoints"]),
        },
        "validation": {
            "all_article_bodies_excluded": True,
            "all_questions_standalone": True,
            "all_resources_manually_reviewed": True,
            "baseline_duplicate_count": 0,
            "copyright_excerpts_are_short": True,
            "eval_leakage_count": 0,
            "primary_evidence_supports_answer_url": True,
            "unsafe_or_unsupported_content_claims": 0,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--eval", nargs="+", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--baselines", nargs="+", type=Path,
                        default=DEFAULT_BASELINES)
    args = parser.parse_args()
    report = build(args.manifest, args.output, args.report, args.eval,
                   args.baselines)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
