#!/usr/bin/env python3
"""Build a deterministic, document-disjoint S5 evaluation gate.

The historical eval-v2 and OOD inputs are read-only inputs.  Eval-v3 uses
only eval-v2 questions generated from the frozen held-out documents, removes
any document represented by a candidate training source URL, and partitions
the remaining documents (not individual questions) into development and
sealed holdout splits.  Fixed OOD probes are annotated with source identities
and audited against the same candidate training URLs.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_SPLIT_SEED = "specialist-eval-v3-s5-20260714"
TRACKING_QUERY_KEYS = {
    "dclid", "fbclid", "gclid", "mc_cid", "mc_eid", "si",
}
UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
SAFETY_PATTERN = re.compile(
    r"\b(?:anatomy|ankles?|breath(?:ing)?|circulation|danger|"
    r"emergenc(?:y|ies)|fall(?:ing)?|harm|injur(?:y|ies)|medical|neck|"
    r"nerve|numb(?:ness)?|pain|risk|safe(?:ly|ty)?|suspension|"
    r"tingl(?:e|ing)|weight|wrists?)\b|"
    r"\b(?:emt|medical|safety)\s+shears?\b|"
    r"\bcut(?:ting)?\s+(?:the\s+)?rope\b",
    re.IGNORECASE,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}, line {line_number}: expected a JSON object"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no JSONL rows")
    return rows


def _decode_unreserved(path):
    def replace(match):
        character = chr(int(match.group(1), 16))
        return character if character in UNRESERVED else match.group(0).upper()

    return re.sub(r"%([0-9a-fA-F]{2})", replace, path)


def _normalize_path(path):
    path = _decode_unreserved(path or "/")
    segments = []
    for segment in re.sub(r"/{2,}", "/", path).split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    normalized = "/" + "/".join(segments)
    return normalized if normalized == "/" else normalized.rstrip("/")


def normalize_source_url(value):
    """Return a stable document identity for a web or repository URL.

    HTTP and HTTPS are deliberately equivalent.  Host names, default ports,
    fragments, trailing slashes, dot segments, unreserved percent escapes,
    and known tracking parameters are normalized.  Semantic query parameters
    are retained and sorted.  Common YouTube URL forms share one identity.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError("source URL must be a non-empty string")
    raw = value.strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https", "repo"}:
        unsupported = scheme or "<none>"
        raise ValueError(f"unsupported source URL scheme: {unsupported}")
    host = (parsed.hostname or "").rstrip(".").lower()
    if not host:
        raise ValueError(f"source URL has no host: {value!r}")
    try:
        host = host.encode("idna").decode("ascii")
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid source URL: {value!r}") from exc
    if host.startswith("www."):
        host = host[4:]
    if host == "m.youtube.com":
        host = "youtube.com"
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        host = "{}:{}".format(host, port)

    path = _normalize_path(parsed.path)
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
            continue
        query.append((key, item))
    query.sort(key=lambda pair: (pair[0], pair[1]))

    if host == "youtu.be":
        video_id = path.strip("/").split("/", 1)[0]
        if video_id:
            host, path, query = "youtube.com", "/watch", [("v", video_id)]
    elif host == "youtube.com" and path == "/watch":
        videos = [item for key, item in query if key == "v" and item]
        if videos:
            query = [("v", videos[0])]
    elif host == "youtube.com" and path == "/playlist":
        playlists = [item for key, item in query if key == "list" and item]
        if playlists:
            query = [("list", playlists[0])]

    prefix = "repo" if scheme == "repo" else "web"
    encoded_query = urlencode(query, doseq=True, quote_via=quote)
    return "{}://{}{}".format(prefix, host, path) + (
        f"?{encoded_query}" if encoded_query else ""
    )


def source_urls(record):
    """Extract URL-valued provenance fields from one training record."""
    found = []
    for key, value in record.items():
        lowered = key.lower()
        if not (
            lowered in {"url", "urls"}
            or lowered.endswith("_url")
            or lowered.endswith("_urls")
        ):
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            item_scheme = (
                urlsplit(item.strip()).scheme.lower()
                if isinstance(item, str) else ""
            )
            if item_scheme in {"http", "https", "repo"}:
                found.append((key, item))
    return found


def jsonl_bytes(rows):
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n").encode("utf-8")


def bytes_identity(content, rows):
    return {
        "bytes": len(content),
        "rows": rows,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def candidate_review_sha256(rows):
    identities = sorted(
        (
            row["item_id"], row["normalized_source_url"],
            row["question"], row["answer"],
        )
        for row in rows
    )
    encoded = json.dumps(
        identities, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_identity(path, rows):
    path = Path(path)
    return {
        "bytes": path.stat().st_size,
        "path": str(path),
        "rows": rows,
        "sha256": file_sha256(path),
    }


def _source_from_url(normalized_url, heldout_by_url):
    try:
        return heldout_by_url[normalized_url]["source"]
    except KeyError as exc:
        raise ValueError(
            "legacy heldout evaluation URL is absent from heldout_docs: "
            f"{normalized_url}"
        ) from exc


def _partition_documents(documents_by_source, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError(
            "holdout fraction must be strictly between zero and one"
        )
    final_documents = set()
    for source, urls in sorted(documents_by_source.items()):
        ordered = sorted(
            urls,
            key=lambda url: (sha256_text(f"{seed}\0{source}\0{url}"), url),
        )
        if len(ordered) == 1:
            final_count = 0
        else:
            final_count = max(
                1, int(math.floor(len(ordered) * fraction + 0.5))
            )
            final_count = min(final_count, len(ordered) - 1)
        final_documents.update(ordered[:final_count])
    return final_documents


def _count_by(rows, field):
    return dict(sorted(Counter(row[field] for row in rows).items()))


def _overlap_details(eval_rows, train_identities):
    overlapping = [
        row for row in eval_rows
        if normalize_source_url(row["url"]) in train_identities
    ]
    return {
        "documents": len({normalize_source_url(row["url"])
                          for row in overlapping}),
        "normalized_source_urls": sorted({
            normalize_source_url(row["url"]) for row in overlapping
        }),
        "rows": len(overlapping),
    }


def build_artifacts(
    train_path,
    legacy_eval_path,
    heldout_docs_path,
    ood_qa_path,
    ood_prose_path,
    review_path,
    *,
    holdout_fraction=1 / 3,
    split_seed=DEFAULT_SPLIT_SEED,
):
    paths = [
        Path(train_path), Path(legacy_eval_path), Path(heldout_docs_path),
        Path(ood_qa_path), Path(ood_prose_path),
    ]
    train_rows, legacy_rows, heldout_docs, ood_qa, ood_prose = [
        read_jsonl(path) for path in paths
    ]
    train_identity_fields = defaultdict(set)
    for row in train_rows:
        for field, value in source_urls(row):
            train_identity_fields[normalize_source_url(value)].add(field)
    train_identities = set(train_identity_fields)
    if not train_identities:
        raise ValueError("candidate training JSONL has no source URLs")

    heldout_by_url = {}
    for row in heldout_docs:
        for required in ("source", "text", "url"):
            if not isinstance(row.get(required), str) or not row[required]:
                raise ValueError(f"heldout document is missing {required}")
        normalized = normalize_source_url(row["url"])
        if normalized in heldout_by_url:
            if heldout_by_url[normalized]["text"] != row["text"]:
                raise ValueError(
                    "conflicting heldout documents normalize to " + normalized
                )
            continue
        heldout_by_url[normalized] = row

    legacy_by_split = defaultdict(list)
    for row in legacy_rows:
        for required in ("answer", "excerpt", "question", "split", "url"):
            if not isinstance(row.get(required), str) or not row[required]:
                raise ValueError(
                    f"legacy evaluation row is missing {required}"
                )
        legacy_by_split[row["split"]].append(row)

    eligible = []
    for row in legacy_by_split.get("heldout", []):
        normalized = normalize_source_url(row["url"])
        source = _source_from_url(normalized, heldout_by_url)
        document = heldout_by_url[normalized]
        if not document["text"].startswith(row["excerpt"]):
            raise ValueError(
                "legacy excerpt is not a prefix of heldout document "
                + normalized
            )
        if normalized not in train_identities:
            eligible.append((row, normalized, source, document))
    if not eligible:
        raise ValueError("no document-disjoint heldout evaluation rows remain")

    domain_rows = []
    seen_items = set()
    for row, normalized, source, document in eligible:
        item_material = "\0".join((normalized, row["question"], row["answer"]))
        item_id = "evalv3-" + sha256_text(item_material)[:20]
        if item_id in seen_items:
            raise ValueError(f"duplicate eval-v3 item identity: {item_id}")
        seen_items.add(item_id)
        question_and_answer = f"{row['question']}\n{row['answer']}"
        domain_rows.append({
            "answer": row["answer"],
            "domain": "rope_bondage",
            "excerpt": row["excerpt"],
            "item_id": item_id,
            "legacy_split": row["split"],
            "normalized_source_url": normalized,
            "quality_bucket": (
                "safety_relevant_grounded"
                if SAFETY_PATTERN.search(question_and_answer)
                else "standard_grounded"
            ),
            "question": row["question"],
            "source": source,
            "source_document_sha256": sha256_text(document["text"]),
            "url": row["url"],
        })

    review_path = Path(review_path)
    try:
        review = json.loads(review_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load manual review {review_path}: {exc}"
        ) from exc
    if review.get("schema") != "specialist-eval-v3-manual-review-v1":
        raise ValueError("manual review has an unsupported schema")
    expected_review_sha = candidate_review_sha256(domain_rows)
    if review.get("reviewed_candidate_set_sha256") != expected_review_sha:
        raise ValueError(
            "manual review does not cover this candidate evaluation cohort"
        )
    if review.get("reviewed_rows") != len(domain_rows):
        raise ValueError("manual review row count does not match candidates")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"]:
        raise ValueError("manual review has no reviewer")
    reason_definitions = review.get("reason_definitions")
    if not isinstance(reason_definitions, dict) or not reason_definitions:
        raise ValueError("manual review has no reason definitions")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in reason_definitions.items()
    ):
        raise ValueError("manual review has an invalid reason definition")
    drop_by_id = {}
    for decision in review.get("drops", []):
        if not isinstance(decision, dict):
            raise ValueError("manual review drop must be an object")
        item_id = decision.get("item_id")
        reason = decision.get("reason_code")
        if not isinstance(item_id, str) or not isinstance(reason, str):
            raise ValueError("manual review drop lacks item_id or reason_code")
        if item_id in drop_by_id:
            raise ValueError(f"duplicate manual review drop {item_id}")
        if reason not in reason_definitions:
            raise ValueError(f"manual review uses undefined reason {reason}")
        drop_by_id[item_id] = decision
    candidate_ids = {row["item_id"] for row in domain_rows}
    unknown_drops = sorted(set(drop_by_id) - candidate_ids)
    if unknown_drops:
        raise ValueError(f"manual review has unknown drops: {unknown_drops}")
    rejected_rows = [
        row for row in domain_rows if row["item_id"] in drop_by_id
    ]
    domain_rows = [row for row in domain_rows
                   if row["item_id"] not in drop_by_id]
    if not domain_rows:
        raise ValueError("manual review rejected every domain evaluation row")

    documents_by_source = defaultdict(set)
    for row in domain_rows:
        documents_by_source[row["source"]].add(row["normalized_source_url"])
    final_documents = _partition_documents(
        documents_by_source, holdout_fraction, split_seed
    )
    for row in domain_rows:
        row["manual_review"] = "keep"
        row["split"] = (
            "heldout"
            if row["normalized_source_url"] in final_documents
            else "validation"
        )
    domain_rows.sort(key=lambda row: (
        row["split"] != "validation", row["source"],
        row["normalized_source_url"], row["item_id"],
    ))

    repository_source = normalize_source_url(
        "repo://specialist/build_ood_probes.py"
    )
    ood_qa_rows = []
    for row in ood_qa:
        if not isinstance(row.get("question"), str) or not isinstance(
            row.get("answer"), str
        ):
            raise ValueError("OOD QA row is missing question or answer")
        material = f"{row['question']}\0{row['answer']}"
        ood_qa_rows.append({
            "answer": row["answer"],
            "domain": "general_knowledge",
            "item_id": "oodqa-v3-" + sha256_text(material)[:20],
            "normalized_source_url": repository_source,
            "quality_bucket": "fixed_short_answer",
            "question": row["question"],
            "source": "repository_authored_probe",
            "split": "ood_qa",
            "url": "repo://specialist/build_ood_probes.py",
        })

    ood_prose_rows = []
    for row in ood_prose:
        if not isinstance(row.get("title"), str) or not isinstance(
            row.get("text"), str
        ):
            raise ValueError("OOD prose row is missing title or text")
        url = "https://en.wikipedia.org/wiki/" + quote(
            row["title"].replace(" ", "_"), safe="()_-"
        )
        normalized = normalize_source_url(url)
        ood_prose_rows.append({
            "domain": "general_prose",
            "item_id": "oodprose-v3-" + sha256_text(
                f"{row['title']}\0{row['text']}"
            )[:20],
            "normalized_source_url": normalized,
            "quality_bucket": "frozen_reference_prose",
            "source": "wikipedia",
            "split": "ood_prose",
            "text": row["text"],
            "title": row["title"],
            "url": url,
        })

    domain_urls = {row["normalized_source_url"] for row in domain_rows}
    validation_urls = {
        row["normalized_source_url"] for row in domain_rows
        if row["split"] == "validation"
    }
    heldout_urls = domain_urls - validation_urls
    ood_web_urls = {
        row["normalized_source_url"] for row in ood_prose_rows
    }
    collisions = {
        "candidate_train_vs_domain": sorted(train_identities & domain_urls),
        "candidate_train_vs_ood_prose": sorted(
            train_identities & ood_web_urls
        ),
        "domain_vs_ood_prose": sorted(domain_urls & ood_web_urls),
        "validation_vs_heldout": sorted(validation_urls & heldout_urls),
    }
    if any(collisions.values()):
        raise ValueError(
            "evaluation disjointness audit failed: "
            + json.dumps(collisions, sort_keys=True)
        )

    domain_content = jsonl_bytes(domain_rows)
    ood_qa_content = jsonl_bytes(ood_qa_rows)
    ood_prose_content = jsonl_bytes(ood_prose_rows)
    validation_rows = [row for row in domain_rows
                       if row["split"] == "validation"]
    final_rows = [row for row in domain_rows if row["split"] == "heldout"]
    if not validation_rows or not final_rows:
        raise ValueError(
            "both validation and sealed heldout must be non-empty"
        )

    representative_rejections = []
    rejection_reasons = sorted({
        drop_by_id[row["item_id"]]["reason_code"]
        for row in rejected_rows
    })
    for reason in rejection_reasons:
        examples = sorted(
            (
                row for row in rejected_rows
                if drop_by_id[row["item_id"]]["reason_code"] == reason
            ),
            key=lambda row: row["item_id"],
        )[:2]
        representative_rejections.extend({
            "item_id": row["item_id"],
            "question": row["question"],
            "reason_code": reason,
        } for row in examples)

    report = {
        "schema": "specialist-eval-v3-build-report-v1",
        "candidate_training_sources": {
            "normalized_source_urls": len(train_identities),
            "source_url_fields": {
                field: sum(field in fields
                           for fields in train_identity_fields.values())
                for field in sorted({
                    field for fields in train_identity_fields.values()
                    for field in fields
                })
            },
        },
        "disjointness": {
            "collisions": collisions,
            "passed": True,
            "source_identity_normalization": (
                "scheme-insensitive web URL; normalized host/default port/"
                "path/tracking query; semantic query retained; YouTube "
                "aliases folded"
            ),
        },
        "historical_eval_audit": {
            "by_legacy_split": {
                split: {
                    "documents": len({normalize_source_url(row["url"])
                                      for row in rows}),
                    "overlap_with_candidate_train": _overlap_details(
                        rows, train_identities
                    ),
                    "rows": len(rows),
                }
                for split, rows in sorted(legacy_by_split.items())
            },
            "heldout_documents_overlapping_candidate_train": sorted(
                set(heldout_by_url) & train_identities
            ),
        },
        "inputs": {
            "candidate_train": file_identity(paths[0], len(train_rows)),
            "heldout_docs": file_identity(paths[2], len(heldout_docs)),
            "legacy_eval": file_identity(paths[1], len(legacy_rows)),
            "manual_review": file_identity(
                review_path, review["reviewed_rows"]
            ),
            "ood_prose": file_identity(paths[4], len(ood_prose)),
            "ood_qa": file_identity(paths[3], len(ood_qa)),
        },
        "limitations": [
            (
                "Source-URL disjointness cannot detect copied or paraphrased "
                "content hosted at a different URL."
            ),
            (
                "Domain questions retain eval-v2's automated generation "
                "and two verifier gates; they are not a new human-authored "
                "gold set."
            ),
            (
                "The fixed OOD QA items are repository-authored probes rather "
                "than source-cited questions."
            ),
            (
                "The sealed heldout split must not be used for HPO, "
                "checkpoint selection, or early stopping."
            ),
        ],
        "manual_review": {
            "candidate_set_sha256": expected_review_sha,
            "dropped_rows": len(rejected_rows),
            "kept_rows": len(domain_rows),
            "reason_counts": dict(sorted(Counter(
                drop_by_id[row["item_id"]]["reason_code"]
                for row in rejected_rows
            ).items())),
            "reason_definitions": dict(sorted(reason_definitions.items())),
            "representative_rejections": representative_rejections,
            "reviewed_rows": review["reviewed_rows"],
            "reviewer": review["reviewer"],
        },
        "metric_policy": {
            "candidate_selection": {
                "primary": "validation mean reward",
                "secondary": [
                    "validation exact rate", "validation nonzero rate"
                ],
                "strata": ["source", "quality_bucket"],
            },
            "ood_noninferiority_gate": {
                "bootstrap": (
                    "paired document/item bootstrap, 20000 samples, fixed seed"
                ),
                "general_knowledge_qa": {
                    "exact_count_delta_minimum": -1,
                    "mean_reward_delta_95_ci_lower_minimum": -0.02,
                },
                "general_prose": {
                    "mean_token_logprob_delta_95_ci_lower_minimum": -0.02,
                },
                "rule": "reject a candidate unless every OOD condition passes",
            },
            "sealed_holdout": {
                "allowed_uses_per_frozen_cycle": 1,
                "metrics": ["mean reward", "exact rate", "nonzero rate"],
                "selection_use": "prohibited",
                "strata": ["source", "quality_bucket"],
            },
        },
        "outputs": {
            "domain_eval": bytes_identity(domain_content, len(domain_rows)),
            "ood_prose": bytes_identity(
                ood_prose_content, len(ood_prose_rows)
            ),
            "ood_qa": bytes_identity(ood_qa_content, len(ood_qa_rows)),
        },
        "parameters": {
            "holdout_fraction_by_source_documents": holdout_fraction,
            "split_seed": split_seed,
        },
        "strata": {
            "heldout": {
                "documents": len(heldout_urls),
                "quality_bucket": _count_by(final_rows, "quality_bucket"),
                "rows": len(final_rows),
                "source": _count_by(final_rows, "source"),
            },
            "ood_prose": {
                "documents": len(ood_prose_rows),
                "quality_bucket": _count_by(ood_prose_rows, "quality_bucket"),
                "rows": len(ood_prose_rows),
            },
            "ood_qa": {
                "documents": 1,
                "quality_bucket": _count_by(ood_qa_rows, "quality_bucket"),
                "rows": len(ood_qa_rows),
            },
            "validation": {
                "documents": len(validation_urls),
                "quality_bucket": _count_by(validation_rows, "quality_bucket"),
                "rows": len(validation_rows),
                "source": _count_by(validation_rows, "source"),
            },
        },
    }
    return {
        "domain": domain_content,
        "ood_qa": ood_qa_content,
        "ood_prose": ood_prose_content,
        "report": json_bytes(report),
    }, report


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path,
                        default=DATA / "train_qa_curated_v1.jsonl")
    parser.add_argument("--legacy-eval", type=Path,
                        default=DATA / "eval_qa_v2.jsonl")
    parser.add_argument("--heldout-docs", type=Path,
                        default=DATA / "heldout_docs.jsonl")
    parser.add_argument("--ood-qa", type=Path, default=DATA / "ood_qa.jsonl")
    parser.add_argument("--ood-prose", type=Path,
                        default=DATA / "ood_prose.jsonl")
    parser.add_argument("--manual-review", type=Path,
                        default=DATA / "eval_qa_v3.review.json")
    parser.add_argument("--eval-output", type=Path,
                        default=DATA / "eval_qa_v3.jsonl")
    parser.add_argument("--ood-qa-output", type=Path,
                        default=DATA / "ood_qa_v3.jsonl")
    parser.add_argument("--ood-prose-output", type=Path,
                        default=DATA / "ood_prose_v3.jsonl")
    parser.add_argument("--report-output", type=Path,
                        default=DATA / "eval_v3.report.json")
    parser.add_argument("--holdout-fraction", type=float, default=1 / 3)
    parser.add_argument("--split-seed", default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--check", action="store_true",
        help="verify that existing outputs equal a deterministic rebuild",
    )
    args = parser.parse_args()

    inputs = {
        args.train.resolve(), args.legacy_eval.resolve(),
        args.heldout_docs.resolve(), args.ood_qa.resolve(),
        args.ood_prose.resolve(), args.manual_review.resolve(),
    }
    output_paths = {
        "domain": args.eval_output,
        "ood_qa": args.ood_qa_output,
        "ood_prose": args.ood_prose_output,
        "report": args.report_output,
    }
    resolved_outputs = [path.resolve() for path in output_paths.values()]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        parser.error("output paths must be distinct")
    if inputs & set(resolved_outputs):
        parser.error("refusing to overwrite a historical or training input")

    artifacts, report = build_artifacts(
        args.train, args.legacy_eval, args.heldout_docs,
        args.ood_qa, args.ood_prose, args.manual_review,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
    )
    if args.check:
        mismatches = []
        for name, path in output_paths.items():
            if not path.exists() or path.read_bytes() != artifacts[name]:
                mismatches.append(str(path))
        if mismatches:
            parser.error(
                "deterministic output mismatch: " + ", ".join(mismatches)
            )
    else:
        for name, path in output_paths.items():
            _write_atomic(path, artifacts[name])
    print(json.dumps({
        "disjointness_passed": report["disjointness"]["passed"],
        "heldout_rows": report["strata"]["heldout"]["rows"],
        "mode": "check" if args.check else "build",
        "validation_rows": report["strata"]["validation"]["rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
