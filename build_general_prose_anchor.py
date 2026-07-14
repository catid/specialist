#!/usr/bin/env python3
"""Build a deterministic train-only prose anchor from local HotpotQA text.

Only search-result observations are extracted.  Questions, answers, thoughts,
and actions are never copied into the output.  Protected evaluation artifacts
are inspected through an explicit safe-field allowlist which excludes answers
and model outputs; collision reports contain hashes and counts, never protected
text.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlsplit

from build_eval_v3 import normalize_source_url


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "sglang/benchmark/react/hotpotqa_100.jsonl"
DEFAULT_OUTPUT = ROOT / "data/general_prose_anchor_v1.jsonl"
DEFAULT_REPORT = ROOT / "data/general_prose_anchor_v1.report.json"
SAFE_TEXT_FIELDS = ("excerpt", "question", "text", "title")
URL_FIELDS = ("normalized_source_url", "source_url", "url")
AUTHORITATIVE_PROTECTED = (
    "eval_qa_v3.jsonl",
    "heldout_docs.jsonl",
    "ood_qa_v3.jsonl",
    "ood_prose_v3.jsonl",
)
DEFAULT_MANIFESTS = (
    ROOT / "experiments/eggroll_es_hpo/snapshots/s5/manifest.json",
)
SEARCH_ACTION = re.compile(r"^Search\[(.+)]$")
REFERENCE_MARK = re.compile(r"\[(?:citation needed|\d+)]", re.IGNORECASE)
SPACE = re.compile(r"\s+")
WORD = re.compile(r"\w+", re.UNICODE)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def canonical_json_bytes(value):
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ) + "\n"
    ).encode("utf-8")


def normalized_words(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return WORD.findall(value)


def normalized_text(value):
    return " ".join(normalized_words(value))


def text_sha256(value):
    return hashlib.sha256(normalized_text(value).encode("utf-8")).hexdigest()


def shingles(value, width):
    words = normalized_words(value)
    return {
        hashlib.sha256(" ".join(words[start:start + width]).encode()).digest()
        for start in range(0, len(words) - width + 1)
    }


def identity_label(value):
    return "".join(normalized_words(value))


def url_label(value):
    try:
        path = urlsplit(value).path
    except ValueError:
        return ""
    if not path:
        return ""
    return identity_label(unquote(path.rstrip("/").rsplit("/", 1)[-1]))


def discover_protected_artifacts(data_dir):
    paths = [Path(data_dir) / name for name in AUTHORITATIVE_PROTECTED]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "missing authoritative protected artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return [path.resolve() for path in paths]


def read_jsonl_objects(path):
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}, line {line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}, line {line_number}: expected an object"
                )
            yield line_number, row


def build_protected_index(paths, shingle_width):
    index = {
        "document_hashes": set(),
        "labels": set(),
        "normalized_urls": set(),
        "shingles": set(),
    }
    artifact_reports = []
    for path in paths:
        row_count = 0
        safe_text_count = 0
        sealed_row_count = 0
        projection = []
        for _, row in read_jsonl_objects(path):
            row_count += 1
            sealed = (
                Path(path).name.startswith("heldout")
                or row.get("split") == "heldout"
            )
            if sealed:
                sealed_row_count += 1
            projected_row = {"sealed": sealed}
            document_hash = row.get("source_document_sha256")
            if isinstance(document_hash, str) and document_hash:
                index["document_hashes"].add(document_hash.lower())
                projected_row["source_document_sha256"] = (
                    document_hash.lower()
                )
            for field in URL_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                try:
                    normalized_url = normalize_source_url(value)
                    index["normalized_urls"].add(normalized_url)
                    projected_row[field] = normalized_url
                except ValueError:
                    pass
                label = url_label(value)
                if label:
                    index["labels"].add(label)
            # Sealed rows are deliberately identity-only.  In particular,
            # question, answer, excerpt, text, and title are neither read nor
            # hashed, so changing a sealed sentinel cannot affect the anchor.
            if sealed:
                projection.append(projected_row)
                continue
            for field in SAFE_TEXT_FIELDS:
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                safe_text_count += 1
                normalized = normalized_text(value)
                index["document_hashes"].add(
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                )
                projected_row[field + "_sha256"] = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()
                index["shingles"].update(shingles(value, shingle_width))
                if field == "title":
                    label = identity_label(value)
                    if label:
                        index["labels"].add(label)
            projection.append(projected_row)
        artifact_reports.append({
            "path": report_path(path),
            "rows": row_count,
            "safe_text_fields_indexed": safe_text_count,
            "sealed_identity_only_rows": sealed_row_count,
            "identity_and_safe_content_projection_sha256": hashlib.sha256(
                b"".join(canonical_json_bytes(row) for row in projection)
            ).hexdigest(),
        })
    return index, artifact_reports


def pin_manifests(paths):
    reports = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"missing evaluation manifest: {path}")
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"evaluation manifest is not an object: {path}")
        projected_inputs = []
        for item in value.get("eval_inputs", []):
            if not isinstance(item, dict) or not isinstance(
                item.get("path"), str
            ):
                raise ValueError(f"invalid evaluation manifest input: {path}")
            projected_inputs.append({
                "path": Path(item["path"]).name,
            })
        projection = {
            "eval_inputs": projected_inputs,
            "eval_splits": value.get("eval_splits", {}),
            "schema": value.get("schema"),
        }
        reports.append({
            "path": report_path(path),
            "identity_projection_sha256": hashlib.sha256(
                canonical_json_bytes(projection)
            ).hexdigest(),
            **projection,
        })
    return reports


def clean_observation(value):
    value = REFERENCE_MARK.sub("", value)
    return SPACE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def extract_candidates(source_path, minimum_words):
    candidates = []
    rejected = {"short": 0, "non_search": 0, "duplicate_document": 0}
    seen_documents = set()
    for line_number, row in read_jsonl_objects(source_path):
        for steps in row.values():
            if not isinstance(steps, list):
                raise ValueError(
                    f"{source_path}, line {line_number}: invalid trajectory"
                )
            for step_number, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                match = SEARCH_ACTION.fullmatch(str(step.get("action", "")))
                observation = step.get("observation")
                if match is None or not isinstance(observation, str):
                    rejected["non_search"] += 1
                    continue
                topic = match.group(1).strip()
                text = clean_observation(observation)
                if len(normalized_words(text)) < minimum_words:
                    rejected["short"] += 1
                    continue
                document_label = identity_label(topic)
                if not document_label or document_label in seen_documents:
                    rejected["duplicate_document"] += 1
                    continue
                seen_documents.add(document_label)
                material = f"{line_number}\0{step_number}\0{topic}\0{text}"
                candidates.append({
                    "document_label": document_label,
                    "item_id": "anchor-v1-" + hashlib.sha256(
                        material.encode("utf-8")
                    ).hexdigest()[:20],
                    "source_line": line_number,
                    "source_step": step_number,
                    "text": text,
                    "topic": topic,
                })
    return candidates, rejected


def build_anchor(
    source_path,
    protected_paths,
    *,
    max_rows=128,
    minimum_words=80,
    shingle_width=20,
    manifest_paths=(),
):
    if max_rows <= 0 or minimum_words <= 0 or shingle_width <= 0:
        raise ValueError("row, word, and shingle limits must be positive")
    protected, protected_reports = build_protected_index(
        protected_paths, shingle_width,
    )
    candidates, extraction_rejections = extract_candidates(
        source_path, minimum_words,
    )
    accepted = []
    collision_rejections = {
        "document_hash": 0,
        "document_identity": 0,
        "protected_content_shingle": 0,
    }
    seen_text_hashes = set()
    for candidate in candidates:
        candidate_hash = text_sha256(candidate["text"])
        if (
            candidate_hash in protected["document_hashes"]
            or candidate_hash in seen_text_hashes
        ):
            collision_rejections["document_hash"] += 1
            continue
        if candidate["document_label"] in protected["labels"]:
            collision_rejections["document_identity"] += 1
            continue
        candidate_shingles = shingles(candidate["text"], shingle_width)
        if candidate_shingles & protected["shingles"]:
            collision_rejections["protected_content_shingle"] += 1
            continue
        seen_text_hashes.add(candidate_hash)
        accepted.append({
            "document_id": (
                "local-hotpotqa-search://" + candidate["document_label"]
            ),
            "item_id": candidate["item_id"],
            "quality_bucket": "train_only_general_prose",
            "source": "local_hotpotqa_search_observation",
            "source_line": candidate["source_line"],
            "source_step": candidate["source_step"],
            "split": "anchor_prose",
            "text": candidate["text"],
            "text_sha256": candidate_hash,
            "title": candidate["topic"],
        })

    # Hash order prevents the source file's question ordering from becoming a
    # topic-selection policy while remaining byte deterministic.
    accepted.sort(key=lambda row: hashlib.sha256(
        ("general-prose-anchor-v1\0" + row["item_id"]).encode()
    ).digest())
    accepted = accepted[:max_rows]
    output = b"".join(canonical_json_bytes(row) for row in accepted)
    report = {
        "schema": "general-prose-anchor-build-v1",
        "source": {
            "path": report_path(source_path),
            "sha256": file_sha256(source_path),
        },
        "policy": {
            "copied_fields": ["observation"],
            "excluded_sensitive_fields": [
                "answer", "target", "thought", "question", "model_output",
            ],
            "max_rows": max_rows,
            "minimum_words": minimum_words,
            "shingle_width_words": shingle_width,
        },
        "protected_artifacts": protected_reports,
        "evaluation_manifests": pin_manifests(manifest_paths),
        "protected_index": {
            key: len(values) for key, values in sorted(protected.items())
        },
        "extraction_rejections": extraction_rejections,
        "collision_rejections": collision_rejections,
        "candidate_rows": len(candidates),
        "output_rows": len(accepted),
        "output_sha256": hashlib.sha256(output).hexdigest(),
        "unique_documents": len({row["document_id"] for row in accepted}),
        "unique_texts": len({row["text_sha256"] for row in accepted}),
    }
    if not accepted:
        raise ValueError("collision guards rejected every anchor candidate")
    return output, report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--minimum-words", type=int, default=80)
    parser.add_argument("--shingle-width", type=int, default=20)
    parser.add_argument(
        "--manifest", action="append", type=Path,
        default=list(DEFAULT_MANIFESTS),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    protected = discover_protected_artifacts(args.data_dir)
    output, report = build_anchor(
        args.source.resolve(), protected,
        max_rows=args.max_rows,
        minimum_words=args.minimum_words,
        shingle_width=args.shingle_width,
        manifest_paths=[path.resolve() for path in args.manifest],
    )
    report_bytes = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    if args.check:
        if args.output.read_bytes() != output:
            raise SystemExit(f"stale anchor output: {args.output}")
        if args.report.read_bytes() != report_bytes:
            raise SystemExit(f"stale anchor report: {args.report}")
        return
    args.output.write_bytes(output)
    args.report.write_bytes(report_bytes)


if __name__ == "__main__":
    main()
