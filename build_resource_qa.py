#!/usr/bin/env python3
"""Build deterministic, owner-curated resource recommendation QA records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from qa_quality import has_protocol_tokens, normalize_text, parse_qa, stable_fact_id


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "rope_resource_qa_v1.jsonl"
DEFAULT_REPORT = ROOT / "data" / "rope_resource_qa_v1.report.json"


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


def require_public_https(url: str, location: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username:
        raise ValueError(f"{location}: expected a public HTTPS URL, got {url!r}")


def load_manifest(path: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "rope-resource-manifest-v1":
        raise ValueError(f"{path}: unsupported manifest schema")
    resources = manifest.get("resources")
    category_questions = manifest.get("category_questions")
    if not isinstance(resources, list) or not resources:
        raise ValueError(f"{path}: resources must be a non-empty list")
    if not isinstance(category_questions, list):
        raise ValueError(f"{path}: category_questions must be a list")

    by_id = {}
    questions = set()
    for index, resource in enumerate(resources, 1):
        location = f"{path}:resource[{index}]"
        resource_id = require_text(resource, "id", location)
        if resource_id in by_id:
            raise ValueError(f"{location}: duplicate resource id {resource_id!r}")
        by_id[resource_id] = resource
        for field in ("name", "category", "purpose", "supplied_url",
                      "canonical_url", "recommendation_question"):
            require_text(resource, field, location)
        require_public_https(resource["supplied_url"], location)
        require_public_https(resource["canonical_url"], location)
        question = resource["recommendation_question"].strip()
        if not question.endswith("?"):
            raise ValueError(f"{location}: recommendation_question must end in '?'")
        normalized = normalize_text(question)
        if normalized in questions:
            raise ValueError(f"{location}: duplicate question")
        questions.add(normalized)
        if not isinstance(resource.get("collection"), dict):
            raise ValueError(f"{location}: collection must be an object")

    for index, category_question in enumerate(category_questions, 1):
        location = f"{path}:category_question[{index}]"
        question = require_text(category_question, "question", location)
        if not question.endswith("?"):
            raise ValueError(f"{location}: question must end in '?'")
        normalized = normalize_text(question)
        if normalized in questions:
            raise ValueError(f"{location}: duplicate question")
        questions.add(normalized)
        resource_ids = category_question.get("resource_ids")
        if not isinstance(resource_ids, list) or not resource_ids:
            raise ValueError(f"{location}: resource_ids must be a non-empty list")
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError(f"{location}: duplicate resource id")
        unknown = [resource_id for resource_id in resource_ids
                   if resource_id not in by_id]
        if unknown:
            raise ValueError(f"{location}: unknown resource ids {unknown}")
    return manifest, resources, category_questions


def make_record(question: str, resources: list[dict], manifest_path: Path,
                manifest_sha256: str, kind: str,
                include_purpose: bool = False) -> dict:
    answer = "; ".join(
        f"{resource['name']}"
        f"{f' ({resource['purpose']})' if include_purpose else ''}: "
        f"{resource.get('recommendation_url', resource['canonical_url'])}"
        for resource in resources
    )
    if has_protocol_tokens(question) or has_protocol_tokens(answer):
        raise ValueError(f"protocol token in resource QA: {question!r}")
    rendered = f"Question: {question}\nAnswer: {answer}"
    if parse_qa(rendered) != (question, answer):
        raise ValueError(f"resource QA does not round-trip: {question!r}")
    return {
        "answer": answer,
        "document_sha256": manifest_sha256,
        "fact_id": stable_fact_id(question, answer),
        "kind": kind,
        "quality_schema": "owner-curated-resource-qa-v1",
        "question": question,
        "resource_ids": [resource["id"] for resource in resources],
        "source": "owner_curated_rope_resource_directory",
        "source_lineage": {"manifest": portable_path(manifest_path)},
        "supplied_url": resources[0]["supplied_url"],
        "supplied_urls": [resource["supplied_url"] for resource in resources],
        "text": rendered,
        "url": resources[0].get("recommendation_url",
                                resources[0]["canonical_url"]),
        "urls": [resource.get("recommendation_url", resource["canonical_url"])
                 for resource in resources],
        "canonical_url": resources[0]["canonical_url"],
        "canonical_urls": [resource["canonical_url"] for resource in resources],
    }


def build(manifest_path: Path, output_path: Path, report_path: Path) -> dict:
    _, resources, category_questions = load_manifest(manifest_path)
    by_id = {resource["id"]: resource for resource in resources}
    manifest_sha256 = file_sha256(manifest_path)
    output = [
        make_record(resource["recommendation_question"], [resource],
                    manifest_path, manifest_sha256, "qa_resource_direct")
        for resource in resources
    ]
    output.extend(
        make_record(
            item["question"], [by_id[resource_id]
                               for resource_id in item["resource_ids"]],
            manifest_path, manifest_sha256, "qa_resource_category",
            bool(item.get("include_purpose", False)))
        for item in category_questions
    )

    fact_ids = [item["fact_id"] for item in output]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("resource QA contains duplicate fact IDs")
    represented = {
        url for item in output for url in item["canonical_urls"]
    }
    missing = [resource["id"] for resource in resources
               if resource["canonical_url"] not in represented]
    if missing:
        raise ValueError(f"resource URLs missing from QA: {missing}")
    supplied_represented = {
        url for item in output for url in item["supplied_urls"]
    }
    missing_supplied = [resource["id"] for resource in resources
                        if resource["supplied_url"] not in supplied_represented]
    if missing_supplied:
        raise ValueError(f"supplied resource URLs missing from QA: {missing_supplied}")
    rendered_destinations = {
        resource.get("recommendation_url", resource["canonical_url"])
        for resource in resources
    }
    metadata_destinations = {url for item in output for url in item["urls"]}
    if not rendered_destinations <= metadata_destinations:
        raise ValueError("rendered recommendation URL missing from URL metadata")

    output.sort(key=lambda item: (item["kind"], normalize_text(item["question"])))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as destination:
        for item in output:
            destination.write(json.dumps(item, ensure_ascii=False,
                                         sort_keys=True) + "\n")
    report = {
        "schema": "owner-curated-resource-qa-report-v1",
        "manifest": portable_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "output": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
        "counts": {
            "category_questions": len(category_questions),
            "direct_questions": len(resources),
            "output": len(output),
            "resources": len(resources),
            "represented_canonical_urls": len(represented),
            "represented_supplied_urls": len(supplied_represented),
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
    args = parser.parse_args()
    report = build(args.manifest, args.output, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
