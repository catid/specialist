#!/usr/bin/env python3
"""Quarantine-aware V2 evaluation and compute boundary.

V2 never opens the quarantined Eval-V3 source.  It deterministically selects
terminal prose documents from corpus material committed after the frozen V434
training bytes, audits those documents against train/dev/OOD in memory, and
persists only opaque identities and aggregate collision counts.

The production terminal API combines the irreversible claim and source load in
one call.  The claim is durably created with O_EXCL before any selected source
is opened; every failure after that point consumes the only access.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import io
import json
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import build_eval_v3 as eval_v3
import build_train_shadow_folds_v37a as folds_v37a
import eggroll_es_train_panel_sampler_v13 as semantic_v13


ROOT = Path(__file__).resolve().parent
CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v2.json"
).resolve()
INCIDENT = (
    ROOT / "experiments/eggroll_es_hpo/incidents/"
    "protected_holdout_access_20260717_v1.json"
).resolve()
LEGACY_EVAL_COLLISION_INCIDENT = (
    ROOT / "experiments/eggroll_es_hpo/incidents/"
    "legacy_eval_collision_access_20260717_v2.json"
).resolve()
RECURSIVE_LOOKUP_INCIDENT = (
    ROOT / "experiments/eggroll_es_hpo/incidents/"
    "recursive_lookup_access_20260717_v3.json"
).resolve()
SUPERSEDED_V2_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v2_superseded_prefix_scan_20260717.json"
).resolve()
QUARANTINE_BOUNDARY_REGISTRY = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "quarantine_boundary_registry_v3.json"
).resolve()
LEGACY_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
TRAIN_REGISTRY = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v56_v434_train_disjoint_identity_registry.json"
).resolve()
TRAIN = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
)
DEV_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
)
DEV = (
    ROOT / "experiments/sft_controls/"
    "v37a_shadow_folds_v412/fold_3_shadow_dev.jsonl"
)
OOD_QA = (ROOT / "data/ood_qa_v3.jsonl").resolve()
OOD_PROSE = (ROOT / "data/ood_prose_v3.jsonl").resolve()
TERMINAL_CLAIM = (
    ROOT / ".protected_evaluation_state/"
    "recipe_evaluation_compute_contract_v2/terminal_claim.json"
).resolve()
CURATED_QA_BUILDER = (ROOT / "build_curated_qa.py").resolve()
LEGACY_EVAL_INCIDENT_BUILDER = (
    ROOT / "build_legacy_eval_collision_incident_v2.py"
).resolve()
RECURSIVE_LOOKUP_INCIDENT_BUILDER = (
    ROOT / "build_recursive_lookup_access_incident_v3.py"
).resolve()

SCHEMA = "specialist-recipe-evaluation-compute-contract-v2"
STATUS = "provisional_prose_only_resealed_after_prefix_scan_terminal_unopened"
SELECTION_SCHEMA = "specialist-recipe-selection-receipt-v2"
CLAIM_SCHEMA = "specialist-protected-terminal-claim-v2"
POOL_SCHEMA = "specialist-post-v434-terminal-prose-pool-v2"
POOL_SELECTION_SEED = "specialist-v2-terminal-prose-20260717"
PROTECTED_TARGET_ROWS = 9
FRESH_DEV_TARGET_DOCUMENTS = 4
RESEALED_CLEAR_SOURCE_COUNT = 13
HISTORICAL_V2_SELECTED_SOURCE_COUNT = 12

QUARANTINED_V1_SOURCE_PATH = ROOT / "data/eval_qa_v3.jsonl"
QUARANTINED_V1_SOURCE_SHA256 = (
    "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b"
)
QUARANTINED_V1_CONTRACT_CONTENT_SHA256 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
QUARANTINED_V1_CONTRACT_FILE_SHA256 = (
    "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
)
INCIDENT_FILE_SHA256 = (
    "e20d2129a72fc2d314002a5448a8e8332296b0975e345f40140e6895247978ae"
)
INCIDENT_CONTENT_SHA256 = (
    "df8856617f5facd6fedac21ab7b653681d2a38484f8e7d93f493bccd47932301"
)
LEGACY_EVAL_COLLISION_INCIDENT_FILE_SHA256 = (
    "351baed6bda7805d5e3d5518ed03622bc13512b90d0df639f657cd57727529e1"
)
LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256 = (
    "68b214cfad7326f15ad7f090130a14b60dab2e2ec478efbf32213e239d1097d4"
)
RECURSIVE_LOOKUP_INCIDENT_FILE_SHA256 = (
    "68652b270d44bef8c33c95df6a39e88534baab8923613a5e8e936ed309d0b550"
)
RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256 = (
    "1e98c7042d1f9279b1b18aba682a65e674f9890465c80bd0622e45c89c7a2907"
)
SUPERSEDED_V2_CONTRACT_FILE_SHA256 = (
    "3e01a95138356224e006c2661fec8dd4675b6fdb13d7ce43c2b1bcfbab656fb3"
)
SUPERSEDED_V2_CONTRACT_CONTENT_SHA256 = (
    "4121c90c79cb2aacc8a927174e9fbbab6b38f983176c525ad0517b548edc0391"
)
QUARANTINE_BOUNDARY_REGISTRY_FILE_SHA256 = (
    "3d8ef097a1419e03f4b735e6f8d30e5a876b0a8e86c4b0f1ac100114cb7daf5d"
)
QUARANTINE_BOUNDARY_REGISTRY_CONTENT_SHA256 = (
    "5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7"
)
QUARANTINED_LEGACY_EVAL_RELATIVE_PATHS = (
    "data/eval_qa.jsonl",
    "data/eval_qa_v2.jsonl",
)
RECURSIVELY_ACCESSED_RELATIVE_PREFIXES = (
    "data/manual_reviews",
    "experiments/sft_controls",
)
FORMER_UNSELECTED_CLEAR_SOURCE_OPAQUE_ITEM_IDENTITY = (
    "5b4344a0bc2ef38896ee460e447249fe0ef731370235a1c730db0e4758526454"
)
DEV_TERMINAL_PARTITION_SEED = "specialist-v2-reseal-dev-terminal-20260717"

EXPECTED = {
    "train_registry_file": (
        "907886ccf689618cd58e68eff05e8212a29826a6c7655c7698632164f9ec5bc8"
    ),
    "train_registry_content": (
        "aea5b80183b2d98cf0dff37fd5f68cd6a8573901cf71c13a4558417c851cae8a"
    ),
    "train": "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    "dev_manifest_file": (
        "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d"
    ),
    "dev_manifest_content": (
        "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb"
    ),
    "dev": "6d5b72f7506a752fd5275425739ec785e25f0ff486f5c03b68e91c8e99d7ebeb",
    "ood_qa": "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d",
    "ood_prose": (
        "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57"
    ),
}

# Every source below was introduced after the exact V434 train file was sealed
# by TRAIN_FREEZE_COMMIT.  Policy-only corpora with zero retained content are
# intentionally absent.  Selection uses hashes only; tuple order is irrelevant.
TRAIN_FREEZE_COMMIT = "eb55135f0b0b83809982ed5e081809e1da5af1ff"
POST_FREEZE_POOL = (
    ("data/site_corpora/acta_loop_knot_efficiency_experiments_2020", "d9403872359551696ceea38668e198f5a1bb655b"),
    ("data/site_corpora/coir_ropes_french_polynesia_2024", "6539cf2db167bdd93afb933fe6fbe7c5e3e58a5f"),
    ("data/site_corpora/europepmc_bdsm_fatality_review", "388c3868b502a6395c627600330f7c214f573c2b"),
    ("data/site_corpora/europepmc_entrapment_neuropathy_review", "fe49fb745f61b81814394301747fb478455ce4d2"),
    ("data/site_corpora/fhwa_t5140_34_adhesive_anchors", "33d8c8cd5ddc041a5256229af21671c6bc93d8e5"),
    ("data/site_corpora/gutenberg_brady_kedge_anchor_77729", "58fcc705abd2cb9fb5097c8766abd8deb273b28e"),
    ("data/site_corpora/hse_temporary_works_faqs", "d45d23ed0dca6f13cde8b790a624b315f1e05c61"),
    ("data/site_corpora/hse_treework_lifting_and_climbing", "5a19a02cace02b222fd7ad4ff26c6bd52ff5b84a"),
    ("data/site_corpora/innotrac_camera_visual_rope_inspection_2020", "0a5d5c244928ed8f7aabad3b90bf90de4cf8233f"),
    ("data/site_corpora/kink_education_code_of_conduct", "46cd6c1b9caf744c424065d2aee0eaa8b17f074a"),
    ("data/site_corpora/maib_throwbag_hidden_fused_joints_2019", "0fa2cfdd055475415712f91942c8f3cbdb7af114"),
    ("data/site_corpora/maib_zarga_hmpe_inspection_failure_2017", "cc028f7c9aed00f08d866297482cf90c70f5ac85"),
    ("data/site_corpora/mdpi_knot_efficiency_statistics_2022", "3541847614210ef85c4b731fe16e2eb6a1f37039"),
    ("data/site_corpora/mdpi_tree_climbing_friction_hitch_2021", "67b5109e65a19e18697a3792babf6adacb67be87"),
    ("data/site_corpora/nist_aramid_rope_sling_fatigue_1976", "20a9b1ddcc3b6b9a446e2e4c2277d8e29eebca53"),
    ("data/site_corpora/nist_manila_rope_color_serviceability_1933", "64b492666696ce40a3db1792b72e76c2be3d15e7"),
    ("data/site_corpora/nist_manila_rope_statistics_1947", "cd2aca8b200b061d84b11054fc0256f69ea6c874"),
    ("data/site_corpora/nist_manila_rope_tests_t198_1921", "f50ae726832e02ee7050479c18b48eb558c5ac8a"),
    ("data/site_corpora/nistir_6096_post_installed_anchors_review", "3847b47b46d0d550d4285acbe0cd05ec585b835a"),
    ("data/site_corpora/noaa_eight_strand_rope_structural_model_1989", "e03776758634ea92dcfdeffb766e80653a8a86ba"),
    ("data/site_corpora/noaa_synthetic_rope_deterioration_1990", "e0006163a58347abc7e2d49a1ebe23755b968100"),
    ("data/site_corpora/nps_textile_fiber_aging_appendix_k_2002", "dfe55a4ef29bb2621eac749d726af855d1123f77"),
    ("data/site_corpora/phm_synthetic_fiber_rope_condition_monitoring_review_2017", "d0762f698c6bdfa70508f87c505f2961ac2c6df4"),
    ("data/site_corpora/sage_aramid_three_strand_contact_forces_2025", "d8a9bd39b7594aa8ea1e66a1b12d742207bf2fc0"),
    ("data/site_corpora/usbr_testing_verifying_rope_access_anchors", "914225fa64def02fc465cf128a901f073bf33b64"),
    ("data/site_corpora/usfs_region5_hazard_tree_2022", "8f098e87ab4fb3d087fd3cf25a5106837dd76cb9"),
    ("data/site_corpora/usfs_rigging_for_trail_work", "2f4611f95849a80c664478b3ea05b32ad3ad5457"),
)

ROLE_ORDER = ("train", "dev", "protected_terminal", "ood_qa", "ood_prose")
QA_ROLES = frozenset(("train", "ood_qa"))
SEMANTIC_FIELD_NAMES = frozenset(
    (
        "question", "answer", "excerpt", "text", "title", "url", "urls",
        "path", "relative_path", "source_path",
    )
)
RESERVATION_SCHEMA = "specialist-v2-resealed-clear-source-reservation-v2"
DEV_RESERVATION_SCHEMA = "specialist-v2-reserved-fresh-dev-source-identities-v1"
FUTURE_DATASET_IDENTITY_SCHEMA = (
    "specialist-future-dataset-source-identity-registry-v2"
)
FUTURE_ADAPTATION_ROLES = frozenset(("train", "dev", "cpt", "sft", "qa"))
RESERVED_IDENTITY_DOMAINS = (
    "opaque_item_identity",
    "document_sha256",
    "source_path_identity_sha256",
    "normalized_url_identity_sha256",
    "raw_lineage_identity_sha256",
)
FORBIDDEN_TERMINAL_KEYS = re.compile(
    r"(?:question|answer|excerpt|prompt|completion|response|text|title|url|"
    r"per[_-]?item|item[_-]?id|raw|row)",
    re.IGNORECASE,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path}: empty or invalid JSONL")
    return rows


def _require_file(path: Path, expected: str, label: str) -> None:
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(f"{label} bytes changed: {observed}")


def _identity(schema: str, value: object) -> str:
    return canonical_sha256({"schema": schema, "value": value})


def _legacy_eval_path_identities() -> frozenset[str]:
    return frozenset(
        _identity("repository-relative-path-v1", value)
        for value in QUARANTINED_LEGACY_EVAL_RELATIVE_PATHS
    )


def _lexical_absolute_path(value: Path | str) -> Path:
    """Normalize a path without resolving, statting, hashing, or opening it."""
    return Path(os.path.abspath(os.fspath(value)))


QUARANTINED_LEGACY_EVAL_ABSOLUTE_PATHS = frozenset(
    _lexical_absolute_path(ROOT / value)
    for value in QUARANTINED_LEGACY_EVAL_RELATIVE_PATHS
)
RECURSIVELY_ACCESSED_ABSOLUTE_PREFIXES = tuple(
    _lexical_absolute_path(ROOT / value)
    for value in RECURSIVELY_ACCESSED_RELATIVE_PREFIXES
)


def _lexical_path_is_under_accessed_prefix(path: Path) -> bool:
    return any(
        path == prefix or prefix in path.parents
        for prefix in RECURSIVELY_ACCESSED_ABSOLUTE_PREFIXES
    )


def _lineage_identity(value: str) -> str:
    return _identity("raw-lineage-edge-identity-v56", value)


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[a-z0-9]+", normalized))


def _token_ngrams(tokens: tuple[str, ...], width: int = 3) -> frozenset[tuple]:
    return frozenset(
        tuple(tokens[index:index + width])
        for index in range(max(0, len(tokens) - width + 1))
    )


def _row_urls(row: dict) -> frozenset[str]:
    return frozenset(
        eval_v3.normalize_source_url(value)
        for _field, value in eval_v3.source_urls(row)
    )


def _document_ids(row: dict, role: str) -> frozenset[str]:
    if isinstance(row.get("document_sha256"), str):
        return frozenset((row["document_sha256"],))
    if isinstance(row.get("source_document_sha256"), str):
        return frozenset((row["source_document_sha256"],))
    if role == "ood_prose" and isinstance(row.get("text"), str):
        return frozenset((hashlib.sha256(row["text"].encode()).hexdigest(),))
    if role == "ood_qa":
        return frozenset((EXPECTED["ood_qa"],))
    return frozenset()


def _lineage_ids(row: dict) -> frozenset[str]:
    return frozenset(
        _lineage_identity(value)
        for value in folds_v37a.row_lineage_identities(row)
    )


def _body(row: dict) -> str:
    if isinstance(row.get("question"), str):
        return row["question"] + "\n" + str(row.get("answer", ""))
    return str(row.get("title", "")) + "\n" + str(row.get("text", ""))


def _record(row: dict, role: str) -> dict:
    body_tokens = _tokens(_body(row))
    qa_features = None
    if role in QA_ROLES:
        qa_features = (
            semantic_v13._content_tokens(str(row.get("question", ""))),
            semantic_v13._content_tokens(str(row.get("answer", ""))),
        )
    return {
        "documents": _document_ids(row, role),
        "urls": _row_urls(row),
        "lineages": _lineage_ids(row),
        "qa_features": qa_features,
        "tokens": body_tokens,
        "ngrams": _token_ngrams(body_tokens),
    }


def _near_duplicate(left: dict, right: dict) -> bool:
    if left["qa_features"] is not None and right["qa_features"] is not None:
        if semantic_v13._semantic_match(left["qa_features"], right["qa_features"]):
            return True
    if left["tokens"] == right["tokens"] and left["tokens"]:
        return True
    lgrams, rgrams = left["ngrams"], right["ngrams"]
    if not lgrams or not rgrams:
        return False
    intersection = len(lgrams & rgrams)
    union = len(lgrams | rgrams)
    containment = intersection / min(len(lgrams), len(rgrams))
    return intersection / union >= 0.80 or (
        min(len(left["tokens"]), len(right["tokens"])) >= 12
        and containment >= 0.90
    )


def _collision_reasons(left: dict, right: dict) -> frozenset[str]:
    reasons = set()
    if left["documents"] & right["documents"]:
        reasons.add("document_sha256")
    if left["urls"] & right["urls"]:
        reasons.add("normalized_url")
    if left["lineages"] & right["lineages"]:
        reasons.add("raw_lineage")
    if _near_duplicate(left, right):
        reasons.add("near_duplicate")
    return frozenset(reasons)


def audit_role_records(role_records: dict[str, list[dict]]) -> dict:
    """Return aggregate, content-free cross-role collision evidence."""
    pairs = {}
    passed = True
    for left_index, left_role in enumerate(ROLE_ORDER):
        for right_role in ROLE_ORDER[left_index + 1:]:
            reasons = defaultdict(int)
            colliding_pairs = 0
            for left in role_records[left_role]:
                for right in role_records[right_role]:
                    found = _collision_reasons(left, right)
                    if found:
                        colliding_pairs += 1
                        for reason in found:
                            reasons[reason] += 1
            pairs[f"{left_role}__{right_role}"] = {
                "colliding_row_pairs": colliding_pairs,
                "by_identity_domain": {
                    name: reasons[name]
                    for name in (
                        "document_sha256", "normalized_url",
                        "raw_lineage", "near_duplicate",
                    )
                },
            }
            passed = passed and colliding_pairs == 0
    return {"passed": passed, "pairs": pairs}


def _train_records_from_identity_registry(registry: dict) -> list[dict]:
    """Rebuild opaque train audit features without reopening train bytes."""
    records = []
    items = registry.get("items")
    if not isinstance(items, list) or len(items) != 448:
        raise RuntimeError("V434 train identity items changed")
    expected_keys = {
        "document_sha256", "normalized_urls",
        "raw_lineage_identity_sha256s", "row_sha256",
        "semantic_answer_tokens", "semantic_cluster_sha256",
        "semantic_question_tokens",
    }
    for item in items:
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise RuntimeError("V434 train identity item schema changed")
        question = item["semantic_question_tokens"]
        answer = item["semantic_answer_tokens"]
        urls = item["normalized_urls"]
        lineages = item["raw_lineage_identity_sha256s"]
        if (
            not isinstance(question, list) or not isinstance(answer, list)
            or any(not isinstance(token, str) for token in question + answer)
            or not isinstance(urls, list)
            or any(not isinstance(value, str) for value in urls)
            or not isinstance(lineages, list)
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in lineages
            )
            or not isinstance(item["document_sha256"], str)
            or not SHA256_RE.fullmatch(item["document_sha256"])
        ):
            raise RuntimeError("V434 train opaque audit features changed")
        question_features = frozenset(question)
        answer_features = frozenset(answer)
        tokens = tuple(sorted(question_features | answer_features))
        records.append({
            "documents": frozenset((item["document_sha256"],)),
            "urls": frozenset(urls),
            "lineages": frozenset(lineages),
            "qa_features": (question_features, answer_features),
            "tokens": tokens,
            "ngrams": _token_ngrams(tokens),
        })
    return records


def _validate_public_registries() -> dict:
    for label, path in (
        ("train_registry_file", TRAIN_REGISTRY),
        ("ood_qa", OOD_QA),
        ("ood_prose", OOD_PROSE),
    ):
        _require_file(path, EXPECTED[label], label)

    registry = _read_json(TRAIN_REGISTRY)
    compact = {
        key: value for key, value in registry.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        registry.get("schema") != "v434-train-disjoint-identity-registry-v56"
        or registry.get("content_sha256_before_self_field")
        != EXPECTED["train_registry_content"]
        or canonical_sha256(compact) != EXPECTED["train_registry_content"]
        or registry.get("aggregate", {}).get("rows") != 448
        or registry.get("source", {}).get("file_sha256") != EXPECTED["train"]
    ):
        raise RuntimeError("V434 train identity registry contract changed")

    # TRAIN, DEV_MANIFEST, and DEV are deliberately not resolved, statted,
    # hashed, or opened after the recursive-prefix incident.  The train role is
    # audited solely from the immutable registry outside the affected prefix;
    # the prior dev role is permanently superseded.
    return registry


def _validate_quarantine_receipt() -> tuple[dict, list[str]]:
    _require_file(INCIDENT, INCIDENT_FILE_SHA256, "V1 incident receipt")
    _require_file(
        LEGACY_CONTRACT,
        QUARANTINED_V1_CONTRACT_FILE_SHA256,
        "quarantined V1 contract",
    )
    receipt = _read_json(INCIDENT)
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    known = receipt.get("known_selected_opaque_item_identities")
    if (
        receipt.get("schema")
        != "specialist-protected-holdout-access-incident-v1"
        or receipt.get("status")
        != "confirmed_nonzero_access_entire_v1_source_quarantined"
        or receipt.get("content_sha256_before_self_field")
        != INCIDENT_CONTENT_SHA256
        or canonical_sha256(compact) != INCIDENT_CONTENT_SHA256
        or receipt.get("affected_contract", {}).get("content_sha256")
        != QUARANTINED_V1_CONTRACT_CONTENT_SHA256
        or receipt.get("affected_contract", {}).get("protected_source_file_sha256")
        != QUARANTINED_V1_SOURCE_SHA256
        or receipt.get("event", {}).get("rows_materialized_per_raw_source_parse")
        != 59
        or receipt.get("event", {}).get("legacy_heldout_candidate_rows_in_each_source_read")
        != 18
        or receipt.get("event", {}).get("v1_protected_access_count_is_nonzero")
        is not True
        or not isinstance(known, list)
        or len(known) != 12
        or any(not isinstance(item, str) or not SHA256_RE.fullmatch(item) for item in known)
    ):
        raise RuntimeError("V1 quarantine receipt changed or is incomplete")
    # Deliberately do not stat, hash, or open QUARANTINED_V1_SOURCE_PATH.
    return receipt, known


def _validate_legacy_eval_collision_incident() -> dict:
    _require_file(
        LEGACY_EVAL_COLLISION_INCIDENT,
        LEGACY_EVAL_COLLISION_INCIDENT_FILE_SHA256,
        "legacy evaluation collision incident receipt",
    )
    receipt = _read_json(LEGACY_EVAL_COLLISION_INCIDENT)
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        receipt.get("schema")
        != "specialist-legacy-evaluation-collision-access-incident-v2"
        or receipt.get("status")
        != "confirmed_nonzero_legacy_evaluation_family_access_paths_quarantined"
        or receipt.get("content_sha256_before_self_field")
        != LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
        or canonical_sha256(compact)
        != LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
        or receipt.get("classification", {}).get("irreversible_nonzero_access")
        is not True
        or receipt.get("classification", {}).get(
            "distinct_from_quarantined_v1_eval_v3"
        ) is not True
        or receipt.get("classification", {}).get(
            "distinct_from_v2_terminal_boundary"
        ) is not True
        or receipt.get("event", {}).get("aggregate_eval_fact_count_reported")
        != 505
        or receipt.get("scope", {}).get("touched_source_count") != 2
        or set(receipt.get("scope", {}).get(
            "touched_source_path_identity_sha256", ()
        )) != _legacy_eval_path_identities()
        or receipt.get("scope", {}).get("source_file_hashes")
        != "unknown_not_computed"
        or receipt.get("quarantine", {}).get(
            "source_reopen_stat_or_hash_prohibited"
        ) is not True
        or receipt.get("quarantine", {}).get("future_adaptation_roles_excluded")
        != sorted(FUTURE_ADAPTATION_ROLES)
    ):
        raise RuntimeError("legacy evaluation collision incident receipt changed")
    # Deliberately do not resolve, stat, hash, or open either touched source.
    return receipt


def _accessed_prefix_identities() -> frozenset[str]:
    return frozenset(
        _identity("repository-relative-path-prefix-v1", value)
        for value in RECURSIVELY_ACCESSED_RELATIVE_PREFIXES
    )


def _normalize_repository_relative_path(value: str) -> str:
    if "\\" in value or os.path.isabs(value):
        raise RuntimeError("future dataset source path must use root-relative slashes")
    normalized = os.path.normpath(value).replace(os.sep, "/")
    if (
        normalized in {"", ".", ".."}
        or normalized.startswith("../")
    ):
        raise RuntimeError("future dataset source path must be repository-relative")
    return normalized


def _relative_path_is_under_accessed_prefix(value: str) -> bool:
    normalized = _normalize_repository_relative_path(value)
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in RECURSIVELY_ACCESSED_RELATIVE_PREFIXES
    )


def _validate_recursive_lookup_incident() -> dict:
    _require_file(
        RECURSIVE_LOOKUP_INCIDENT,
        RECURSIVE_LOOKUP_INCIDENT_FILE_SHA256,
        "recursive lookup access incident receipt",
    )
    receipt = _read_json(RECURSIVE_LOOKUP_INCIDENT)
    compact = {
        key: value for key, value in receipt.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        receipt.get("schema")
        != "specialist-recursive-filename-lookup-access-incident-v3"
        or receipt.get("status")
        != "confirmed_nonzero_prefix_wide_tool_access_evaluation_roles_quarantined"
        or receipt.get("content_sha256_before_self_field")
        != RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256
        or canonical_sha256(compact)
        != RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256
        or receipt.get("classification", {}).get(
            "irreversible_nonzero_tool_access"
        ) is not True
        or receipt.get("classification", {}).get(
            "scope_is_not_limited_to_reported_filename_matches"
        ) is not True
        or receipt.get("event", {}).get("recursive_scan_passes") != 8
        or set(receipt.get("scope", {}).get(
            "accessed_prefix_identity_sha256", ()
        )) != _accessed_prefix_identities()
        or receipt.get("scope", {}).get("individual_file_inventory")
        != "not_enumerated_reopen_prohibited"
        or receipt.get("v2_impact", {}).get("affected_revision_file_sha256")
        != SUPERSEDED_V2_CONTRACT_FILE_SHA256
        or receipt.get("v2_impact", {}).get("affected_revision_content_sha256")
        != SUPERSEDED_V2_CONTRACT_CONTENT_SHA256
        or receipt.get("v2_impact", {}).get("v2_terminal_source_opened_by_event")
        is not False
        or receipt.get("future_worker_policy", {}).get(
            "explicit_file_allowlist_required_before_lookup"
        ) is not True
    ):
        raise RuntimeError("recursive lookup access incident receipt changed")
    return receipt


def _validate_content_free_quarantine_boundary() -> dict:
    _require_file(
        QUARANTINE_BOUNDARY_REGISTRY,
        QUARANTINE_BOUNDARY_REGISTRY_FILE_SHA256,
        "content-free quarantine boundary registry",
    )
    boundary = _read_json(QUARANTINE_BOUNDARY_REGISTRY)
    compact = {
        key: value for key, value in boundary.items()
        if key != "content_sha256_before_self_field"
    }
    expected_exact = set(_legacy_eval_path_identities()) | {
        _identity(
            "repository-relative-path-v1",
            str(QUARANTINED_V1_SOURCE_PATH.relative_to(ROOT)),
        )
    }
    policy = boundary.get("ancestor_denial_policy", {})
    if (
        boundary.get("schema")
        != "specialist-content-free-quarantine-boundary-registry-v3"
        or boundary.get("status")
        != "active_fail_closed_content_free_quarantine_boundary"
        or boundary.get("content_sha256_before_self_field")
        != QUARANTINE_BOUNDARY_REGISTRY_CONTENT_SHA256
        or canonical_sha256(compact)
        != QUARANTINE_BOUNDARY_REGISTRY_CONTENT_SHA256
        or set(boundary.get("exact_path_identity_sha256", ()))
        != expected_exact
        or set(boundary.get("prefix_identity_sha256", ()))
        != _accessed_prefix_identities()
        or policy.get("lexical_deny_before_resolution_stat_hash_or_open")
        is not True
        or policy.get(
            "lexically_allowed_resolution_rechecked_before_metadata_or_open"
        ) is not True
        or boundary.get("content_minimization", {}).get(
            "plaintext_boundary_paths_persisted"
        ) is not False
    ):
        raise RuntimeError("content-free quarantine boundary registry changed")
    return boundary


def _validate_superseded_v2_contract() -> dict:
    _require_file(
        SUPERSEDED_V2_CONTRACT,
        SUPERSEDED_V2_CONTRACT_FILE_SHA256,
        "superseded V2 contract",
    )
    contract = _read_json(SUPERSEDED_V2_CONTRACT)
    compact = {
        key: value for key, value in contract.items()
        if key != "content_sha256_before_self_field"
    }
    protected = contract.get("roles", {}).get("protected_terminal", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status")
        != "sealed_after_v1_quarantine_before_hpo_terminal_unopened"
        or contract.get("content_sha256_before_self_field")
        != SUPERSEDED_V2_CONTRACT_CONTENT_SHA256
        or canonical_sha256(compact) != SUPERSEDED_V2_CONTRACT_CONTENT_SHA256
        or protected.get("rows") != HISTORICAL_V2_SELECTED_SOURCE_COUNT
        or len(protected.get("selected_sources", ()))
        != HISTORICAL_V2_SELECTED_SOURCE_COUNT
        or protected.get("access_authorized_by_this_contract") is not False
    ):
        raise RuntimeError("superseded V2 contract bytes changed")
    return contract


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def _metadata_paths(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    ))


def _json_values(path: Path) -> list[object]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [json.loads(path.read_text(encoding="utf-8"))]


def _walk_key_values(value: object, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _walk_key_values(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_key_values(child, key)
    else:
        yield key, value


def _candidate_from_source(
    relative_directory: str,
    added_commit: str,
    *,
    include_terminal_text: bool = False,
) -> dict:
    directory = (ROOT / relative_directory).resolve()
    corpus_path = directory / "CORPUS.md"
    manifest_path = directory / "manifest.json"
    if not corpus_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("V2 source-pool member is incomplete")
    if _git("merge-base", "--is-ancestor", TRAIN_FREEZE_COMMIT, added_commit):
        # merge-base emits no stdout; success is what matters.
        pass
    added_paths = _git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", added_commit).splitlines()
    expected_addition = f"{relative_directory}/CORPUS.md"
    if expected_addition not in added_paths:
        raise RuntimeError("V2 source pool does not bind its addition commit")

    manifest = _read_json(manifest_path)
    if manifest.get("direct_training_ready") is not True:
        raise RuntimeError("V2 source pool contains a non-substantive corpus")
    metadata_paths = _metadata_paths(directory)
    metadata_registry = [
        {
            "relative_path_identity_sha256": _identity(
                "repository-relative-path-v1", str(path.relative_to(ROOT))
            ),
            "file_sha256": file_sha256(path),
        }
        for path in metadata_paths
    ]
    metadata_values = [
        value
        for path in metadata_paths
        for value in _json_values(path)
    ]
    urls = set()
    document_hashes = set()
    raw_lineages = {relative_directory, added_commit}
    for value in metadata_values:
        for key, item in _walk_key_values(value):
            lowered = key.lower()
            if isinstance(item, str):
                if (
                    lowered in {"url", "urls"}
                    or lowered.endswith("_url")
                    or lowered.endswith("_urls")
                ):
                    try:
                        urls.add(eval_v3.normalize_source_url(item))
                    except ValueError:
                        pass
                if SHA256_RE.fullmatch(item) and any(
                    marker in lowered
                    for marker in ("body", "document", "content", "audit_text", "source")
                ):
                    document_hashes.add(item)
                if lowered in {
                    "source_id", "source_ids", "resource_id", "corpus_id",
                    "package_id", "package", "source_lineage",
                }:
                    raw_lineages.add(item)

    corpus_bytes = corpus_path.read_bytes()
    corpus_sha = hashlib.sha256(corpus_bytes).hexdigest()
    try:
        corpus_text = corpus_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("V2 terminal corpus is not UTF-8") from exc
    document_hashes.add(corpus_sha)
    metadata_bundle_sha = canonical_sha256(metadata_registry)
    path_identity = _identity(
        "repository-relative-path-v1", str(corpus_path.relative_to(ROOT))
    )
    url_identity_set = sorted(
        _identity("normalized-source-url-v2", value) for value in urls
    )
    lineage_ids = frozenset(_lineage_identity(item) for item in raw_lineages)
    item_identity = canonical_sha256({
        "schema": "protected-terminal-prose-item-v2",
        "corpus_file_sha256": corpus_sha,
        "metadata_bundle_sha256": metadata_bundle_sha,
        "normalized_url_identity_set_sha256": canonical_sha256(url_identity_set),
        "source_path_identity_sha256": path_identity,
    })
    tokens = _tokens(corpus_text)
    if len(tokens) < 128:
        raise RuntimeError("V2 terminal corpus is too small for prose evaluation")
    record = {
        "documents": frozenset(document_hashes),
        "urls": frozenset(urls),
        "lineages": lineage_ids,
        "qa_features": None,
        "tokens": tokens,
        "ngrams": _token_ngrams(tokens),
    }
    return {
        "added_commit": added_commit,
        "corpus_file_sha256": corpus_sha,
        "item_identity": item_identity,
        "metadata_bundle_sha256": metadata_bundle_sha,
        "record": record,
        "relative_path": str(corpus_path.relative_to(ROOT)),
        "source_path_identity_sha256": path_identity,
        "terminal_text": corpus_text if include_terminal_text else None,
        "url_identity_set_sha256": canonical_sha256(url_identity_set),
    }


def _selected_source_binding(item: dict) -> dict:
    """Return the persisted, content-free binding for one selected source."""
    return {
        "added_commit": item["added_commit"],
        "corpus_file_sha256": item["corpus_file_sha256"],
        "metadata_bundle_sha256": item["metadata_bundle_sha256"],
        "opaque_item_identity": item["item_identity"],
        "source_path_identity_sha256": item["source_path_identity_sha256"],
        "url_identity_set_sha256": item["url_identity_set_sha256"],
    }


def _source_identity_domains(sources: list[dict]) -> dict[str, list[str]]:
    return {
        "opaque_item_identity": sorted(
            item["item_identity"] for item in sources
        ),
        "document_sha256": sorted({
            value for item in sources for value in item["record"]["documents"]
        }),
        "source_path_identity_sha256": sorted(
            item["source_path_identity_sha256"] for item in sources
        ),
        "normalized_url_identity_sha256": sorted({
            _identity("normalized-source-url-v2", value)
            for item in sources for value in item["record"]["urls"]
        }),
        "raw_lineage_identity_sha256": sorted({
            value for item in sources for value in item["record"]["lineages"]
        }),
    }


def _reservation_registry(clear_sources: list[dict]) -> dict:
    domains = _source_identity_domains(clear_sources)
    registry = {
        "schema": RESERVATION_SCHEMA,
        "status": "all_resealed_dev_terminal_sources_reserved_before_adaptation",
        "roles": sorted(FUTURE_ADAPTATION_ROLES),
        "reserved_source_count": len(clear_sources),
        "terminal_source_count": PROTECTED_TARGET_ROWS,
        "dev_source_count": FRESH_DEV_TARGET_DOCUMENTS,
        "historical_selected_source_count": HISTORICAL_V2_SELECTED_SOURCE_COUNT,
        "historical_selected_sources_remain_reserved": True,
        "terminal_eligible_unselected_sources": 0,
        "identity_domains": {
            name: {
                "count": len(domains[name]),
                "opaque_values": domains[name],
                "opaque_value_set_sha256": canonical_sha256(domains[name]),
            }
            for name in RESERVED_IDENTITY_DOMAINS
        },
        "enforcement_api": "assert_future_dataset_excludes_reserved",
        "missing_or_unverifiable_row_provenance": "reject_fail_closed",
    }
    registry["content_sha256_before_self_field"] = canonical_sha256(registry)
    return registry


def _fresh_dev_reservation_registry(dev: dict) -> dict:
    dev_sources = dev if isinstance(dev, list) else [dev]
    domains = _source_identity_domains(dev_sources)
    registry = {
        "schema": DEV_RESERVATION_SCHEMA,
        "status": "fresh_dev_source_reserved_outside_model_adaptation",
        "roles": sorted(FUTURE_ADAPTATION_ROLES),
        "dev_source_count": len(dev_sources),
        "removed_from_terminal_eligibility": True,
        "identity_domains": {
            name: {
                "count": len(domains[name]),
                "opaque_values": domains[name],
                "opaque_value_set_sha256": canonical_sha256(domains[name]),
            }
            for name in RESERVED_IDENTITY_DOMAINS
        },
        "enforcement_api": "assert_future_dataset_excludes_reserved",
        "missing_or_unverifiable_row_provenance": "reject_fail_closed",
    }
    registry["content_sha256_before_self_field"] = canonical_sha256(registry)
    return registry


def _allocate_successor_pool(
    public_records: list[dict], superseded_v2: dict
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Repartition 13 clear sources into four dev and nine terminal documents."""
    candidates = [
        _candidate_from_source(relative, commit)
        for relative, commit in POST_FREEZE_POOL
    ]
    candidates.sort(key=lambda item: canonical_sha256({
        "schema": POOL_SCHEMA,
        "seed": POOL_SELECTION_SEED,
        "opaque_item_identity": item["item_identity"],
    }))
    by_identity = {item["item_identity"]: item for item in candidates}
    prior_protected = superseded_v2["roles"]["protected_terminal"]
    prior_selected_bindings = prior_protected["selected_sources"]
    prior_selected_ids = prior_protected["selected_opaque_item_identities"]
    if (
        len(by_identity) != len(candidates)
        or len(prior_selected_ids) != HISTORICAL_V2_SELECTED_SOURCE_COUNT
        or set(prior_selected_ids) - set(by_identity)
    ):
        raise RuntimeError("superseded V2 terminal selection no longer resolves")
    historical_selected = [by_identity[value] for value in prior_selected_ids]
    if sorted(
        (_selected_source_binding(item) for item in historical_selected),
        key=lambda item: item["opaque_item_identity"],
    ) != sorted(
        prior_selected_bindings,
        key=lambda item: item["opaque_item_identity"],
    ):
        raise RuntimeError("superseded V2 terminal byte bindings changed")

    extra = by_identity.get(FORMER_UNSELECTED_CLEAR_SOURCE_OPAQUE_ITEM_IDENTITY)
    if extra is None or extra in historical_selected:
        raise RuntimeError("13th clear source allocation no longer resolves")
    clear_sources = historical_selected + [extra]
    if len({item["item_identity"] for item in clear_sources}) != RESEALED_CLEAR_SOURCE_COUNT:
        raise RuntimeError("resealed clear-source set changed")
    for candidate in clear_sources:
        relative = candidate["relative_path"]
        if any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in RECURSIVELY_ACCESSED_RELATIVE_PREFIXES
        ):
            raise RuntimeError("resealed source falls under an accessed prefix")
        if any(
            _collision_reasons(candidate["record"], other)
            for other in public_records
        ):
            raise RuntimeError("resealed clear source collides with public role")
    if any(
        _collision_reasons(left["record"], right["record"])
        for index, left in enumerate(clear_sources)
        for right in clear_sources[index + 1:]
    ):
        raise RuntimeError("resealed clear sources collide with each other")

    partitioned = sorted(clear_sources, key=lambda item: canonical_sha256({
        "schema": "v2-resealed-dev-terminal-partition-v1",
        "seed": DEV_TERMINAL_PARTITION_SEED,
        "opaque_item_identity": item["item_identity"],
    }))
    dev_sources = partitioned[:FRESH_DEV_TARGET_DOCUMENTS]
    selected = partitioned[FRESH_DEV_TARGET_DOCUMENTS:]
    if (
        len(dev_sources) != FRESH_DEV_TARGET_DOCUMENTS
        or len(selected) != PROTECTED_TARGET_ROWS
    ):
        raise RuntimeError("resealed dev/terminal partition count changed")

    excluded = defaultdict(int)
    for candidate in candidates:
        if candidate in clear_sources:
            continue
        reasons = set()
        for other in public_records:
            reasons.update(_collision_reasons(candidate["record"], other))
        for other in clear_sources:
            reasons.update(_collision_reasons(candidate["record"], other["record"]))
        if not reasons:
            raise RuntimeError(
                "unexpected unallocated source remains terminal eligible"
            )
        excluded["rows"] += 1
        for reason in reasons:
            excluded[reason] += 1
    if excluded["rows"] != 14:
        raise RuntimeError("successor pool exclusion count changed")
    selected_pair_collisions = sum(
        bool(_collision_reasons(left["record"], right["record"]))
        for index, left in enumerate(selected)
        for right in selected[index + 1:]
    )
    if selected_pair_collisions:
        raise RuntimeError("V2 selected terminal documents collide with each other")
    registry = [
        {
            "added_commit": item["added_commit"],
            "corpus_file_sha256": item["corpus_file_sha256"],
            "metadata_bundle_sha256": item["metadata_bundle_sha256"],
            "opaque_item_identity": item["item_identity"],
            "source_path_identity_sha256": item["source_path_identity_sha256"],
            "url_identity_set_sha256": item["url_identity_set_sha256"],
        }
        for item in candidates
    ]
    return selected, dev_sources, clear_sources, {
        "candidate_sources": len(candidates),
        "terminal_selected_sources": len(selected),
        "fresh_dev_allocated_sources": len(dev_sources),
        "resealed_clear_sources": len(clear_sources),
        "historical_selected_sources_still_reserved": (
            HISTORICAL_V2_SELECTED_SOURCE_COUNT
        ),
        "terminal_eligible_unselected_sources": 0,
        "dev_sources_removed_from_terminal_eligibility": True,
        "partition_seed": DEV_TERMINAL_PARTITION_SEED,
        "excluded_sources": excluded["rows"],
        "selected_source_pair_collision_count": selected_pair_collisions,
        "excluded_source_reason_counts": {
            name: excluded[name]
            for name in (
                "document_sha256", "normalized_url",
                "raw_lineage", "near_duplicate",
            )
        },
        "pool_registry_content_sha256": canonical_sha256(registry),
        "pool_source_file_sha256_set_sha256": canonical_sha256(sorted(
            item["corpus_file_sha256"] for item in candidates
        )),
    }


def build_contract() -> dict:
    registry = _validate_public_registries()
    incident, legacy_known_ids = _validate_quarantine_receipt()
    legacy_eval_incident = _validate_legacy_eval_collision_incident()
    recursive_lookup_incident = _validate_recursive_lookup_incident()
    quarantine_boundary = _validate_content_free_quarantine_boundary()
    superseded_v2 = _validate_superseded_v2_contract()
    ood_qa_rows = _read_jsonl(OOD_QA)
    ood_prose_rows = _read_jsonl(OOD_PROSE)
    if len(ood_qa_rows) != 24 or len(ood_prose_rows) != 16:
        raise RuntimeError("frozen public role row count changed")
    train_records = _train_records_from_identity_registry(registry)
    ood_qa_records = [_record(row, "ood_qa") for row in ood_qa_rows]
    ood_prose_records = [_record(row, "ood_prose") for row in ood_prose_rows]
    selected, fresh_dev_sources, clear_sources, pool_audit = _allocate_successor_pool(
        train_records + ood_qa_records + ood_prose_records,
        superseded_v2,
    )
    dev_records = [item["record"] for item in fresh_dev_sources]
    role_records = {
        "train": train_records,
        "dev": dev_records,
        "protected_terminal": [item["record"] for item in selected],
        "ood_qa": ood_qa_records,
        "ood_prose": ood_prose_records,
    }
    audit = audit_role_records(role_records)
    if not audit["passed"]:
        raise RuntimeError("V2 recipe role disjointness audit failed")
    selected_sources = [
        _selected_source_binding(item)
        for item in sorted(selected, key=lambda value: value["item_identity"])
    ]
    clear_source_bindings = [
        _selected_source_binding(item)
        for item in sorted(clear_sources, key=lambda value: value["item_identity"])
    ]
    fresh_dev_bindings = [
        _selected_source_binding(item)
        for item in sorted(
            fresh_dev_sources, key=lambda value: value["item_identity"]
        )
    ]
    selected_ids = [item["opaque_item_identity"] for item in selected_sources]
    clear_ids = sorted(item["item_identity"] for item in clear_sources)
    reservation = _reservation_registry(clear_sources)
    fresh_dev_reservation = _fresh_dev_reservation_registry(fresh_dev_sources)
    quarantined_path_identity = _identity(
        "repository-relative-path-v1",
        str(QUARANTINED_V1_SOURCE_PATH.relative_to(ROOT)),
    )
    quarantine_exclusion = {
        "candidate_source_path_matches_quarantined_v1": (
            quarantined_path_identity
            in {
                item["source_path_identity_sha256"]
                for item in clear_source_bindings
            }
        ),
        "candidate_source_hash_matches_quarantined_v1": any(
            item["corpus_file_sha256"] == QUARANTINED_V1_SOURCE_SHA256
            for item in clear_source_bindings
        ),
        "resealed_clear_identity_intersection_with_known_v1": len(
            set(clear_ids) & set(legacy_known_ids)
        ),
        "candidate_source_path_identity_intersection_with_legacy_eval_family": (
            len({
                item["source_path_identity_sha256"]
                for item in clear_source_bindings
            } & _legacy_eval_path_identities())
        ),
        "fresh_dev_identity_intersection_with_reserved_terminal": int(
            bool(set(
                item["opaque_item_identity"] for item in fresh_dev_bindings
            ) & set(selected_ids))
        ),
        "fresh_dev_or_terminal_source_under_recursively_accessed_prefix": False,
    }
    if any(quarantine_exclusion.values()):
        raise RuntimeError("V2 terminal selection intersects V1 quarantine")

    contract = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "purpose": (
            "Provisional prose-logprob-only boundary after three irreversible "
            "access incidents, with four opaque dev documents, nine terminal "
            "documents, all 13 reserved, one compute ledger, and one-shot claim."
        ),
        "supersedes": {
            "schema": "specialist-recipe-evaluation-compute-contract-v1",
            "content_sha256": QUARANTINED_V1_CONTRACT_CONTENT_SHA256,
            "status": "quarantined_fail_closed_never_reusable",
        },
        "superseded_v2_revision": {
            "path": str(SUPERSEDED_V2_CONTRACT),
            "file_sha256": SUPERSEDED_V2_CONTRACT_FILE_SHA256,
            "content_sha256": SUPERSEDED_V2_CONTRACT_CONTENT_SHA256,
            "status": "immutable_superseded_nonpromotable",
            "terminal_selection_preserved": True,
            "dev_role_reuse_prohibited": True,
        },
        "incident": {
            "path": str(INCIDENT),
            "file_sha256": INCIDENT_FILE_SHA256,
            "content_sha256": INCIDENT_CONTENT_SHA256,
            "status": incident["status"],
            "v1_access_count_nonzero": True,
        },
        "additional_legacy_evaluation_incident": {
            "path": str(LEGACY_EVAL_COLLISION_INCIDENT),
            "file_sha256": LEGACY_EVAL_COLLISION_INCIDENT_FILE_SHA256,
            "content_sha256": LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256,
            "status": legacy_eval_incident["status"],
            "access_count_nonzero": True,
            "distinct_from_v1_eval_v3_and_v2_terminal": True,
        },
        "recursive_lookup_incident": {
            "path": str(RECURSIVE_LOOKUP_INCIDENT),
            "file_sha256": RECURSIVE_LOOKUP_INCIDENT_FILE_SHA256,
            "content_sha256": RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256,
            "status": recursive_lookup_incident["status"],
            "prefix_wide_access_count_nonzero": True,
            "terminal_source_opened": False,
        },
        "content_free_quarantine_boundary": {
            "path": str(QUARANTINE_BOUNDARY_REGISTRY),
            "file_sha256": QUARANTINE_BOUNDARY_REGISTRY_FILE_SHA256,
            "content_sha256": QUARANTINE_BOUNDARY_REGISTRY_CONTENT_SHA256,
            "exact_path_identity_count": quarantine_boundary[
                "exact_path_identity_count"
            ],
            "prefix_identity_count": quarantine_boundary[
                "prefix_identity_count"
            ],
            "plaintext_boundary_paths_persisted": False,
            "deny_before_resolution_stat_hash_or_open": True,
        },
        "quarantine": {
            "passed": True,
            "scope": "entire V1 source: all 59 rows and all 18 legacy heldout candidates",
            "source_path": str(QUARANTINED_V1_SOURCE_PATH),
            "source_file_sha256": QUARANTINED_V1_SOURCE_SHA256,
            "source_opened_during_v2_build": False,
            "legacy_access_count_reset_prohibited": True,
            "legacy_contract_validation_prohibited": True,
            "legacy_terminal_loading_prohibited": True,
            "known_v1_selected_identity_count": len(legacy_known_ids),
            "known_v1_selected_identity_set_sha256": canonical_sha256(
                sorted(legacy_known_ids)
            ),
            "v1_row_level_semantic_comparison": "prohibited_not_performed",
            "v1_exclusion_proof_scope": (
                "source path identity, complete source file hash, and the 12 "
                "known opaque V1 selected identities only"
            ),
            "unknown_v1_row_semantic_disjointness_claimed": False,
            "additional_legacy_eval_source_count": 2,
            "additional_legacy_eval_source_path_identity_sha256": sorted(
                _legacy_eval_path_identities()
            ),
            "additional_legacy_eval_source_bytes_known": False,
            "additional_legacy_eval_source_reopen_stat_or_hash_prohibited": True,
            "additional_legacy_eval_future_roles_excluded": sorted(
                FUTURE_ADAPTATION_ROLES
            ),
            "curated_qa_implicit_eval_default_prohibited": True,
            "recursive_lookup_accessed_prefix_count": 2,
            "recursive_lookup_accessed_prefix_identity_sha256": sorted(
                _accessed_prefix_identities()
            ),
            "recursive_lookup_file_inventory_reconstruction_prohibited": True,
            "prior_train_dev_paths_resolved_statted_hashed_or_opened_during_reseal": False,
            "prior_dev_role_reuse_prohibited": True,
            "fresh_dev_outside_accessed_prefixes": True,
            "future_worker_explicit_file_allowlist_required": True,
            "exclusion_audit": quarantine_exclusion,
        },
        "roles": {
            "train": {
                "use": "model updates and train-only sampling statistics",
                "path": str(TRAIN),
                "file_sha256": EXPECTED["train"],
                "rows": 448,
                "source_bytes_reopened_during_reseal": False,
                "audit_features_loaded_from_opaque_registry_only": True,
                "identity_registry_path": str(TRAIN_REGISTRY),
                "identity_registry_file_sha256": EXPECTED["train_registry_file"],
                "identity_registry_content_sha256": EXPECTED["train_registry_content"],
                "source_documents": registry["aggregate"]["documents"],
                "semantic_clusters": registry["aggregate"]["semantic_clusters"],
            },
            "dev": {
                "use": (
                    "provisional prose-logprob tuning and early stopping only; "
                    "no QA or general quality promotion authority"
                ),
                "kind": "fresh_source_document_prose",
                "rows": len(fresh_dev_bindings),
                "documents": len(fresh_dev_bindings),
                "source_bindings": fresh_dev_bindings,
                "source_identity_set_sha256": canonical_sha256(sorted(
                    item["opaque_item_identity"] for item in fresh_dev_bindings
                )),
                "source_path_persisted": False,
                "source_text_or_url_persisted": False,
                "outside_recursively_accessed_prefixes": True,
                "removed_from_terminal_eligibility": True,
                "terminal_identity_intersection": 0,
                "prior_v37a_dev_reused": False,
                "prose_logprob_tuning_authorized": True,
                "multi_item_qa_dev_present": False,
                "qa_hpo_or_quality_promotion_authorized": False,
                "future_adaptation_reservation": fresh_dev_reservation,
            },
            "protected_terminal": {
                "use": "one terminal aggregate prose report after recipe freeze only",
                "kind": "source_document_prose",
                "source_pool_schema": POOL_SCHEMA,
                "source_pool_selection_seed": POOL_SELECTION_SEED,
                "source_pool_frozen_train_predecessor_commit": TRAIN_FREEZE_COMMIT,
                "source_pool_audit": pool_audit,
                "rows": len(selected_sources),
                "documents": len(selected_sources),
                "selected_sources": selected_sources,
                "selected_opaque_item_identities": selected_ids,
                "selected_identity_set_sha256": canonical_sha256(selected_ids),
                "future_adaptation_reservation": reservation,
                "access_authorized_by_this_contract": False,
                "accesses_per_frozen_recipe_program": 1,
                "selection_or_tuning_use": "prohibited",
            },
            "ood": {
                "use": "noninferiority trust-region gate; never point-score optimize",
                "qa": {
                    "path": str(OOD_QA),
                    "file_sha256": EXPECTED["ood_qa"],
                    "rows": 24,
                },
                "prose": {
                    "path": str(OOD_PROSE),
                    "file_sha256": EXPECTED["ood_prose"],
                    "rows": 16,
                },
            },
        },
        "authority": {
            "systems_and_prose_logprob_tuning_only": True,
            "qa_hpo_or_general_quality_promotion_authorized": False,
            "terminal_access_authorized": False,
            "separate_source_disjoint_multi_item_qa_dev_required": True,
        },
        "explicit_blockers": [
            {
                "code": "fresh_source_disjoint_multi_item_qa_dev_absent",
                "effect": "QA HPO and general quality promotion remain prohibited",
            },
            {
                "code": "terminal_execution_not_authorized",
                "effect": (
                    "terminal claim requires an exact frozen selection receipt "
                    "and separate explicit authority"
                ),
            },
        ],
        "disjointness": {
            "passed": True,
            "audit": audit,
            "identity_domains": [
                "document SHA-256", "normalized provenance URL",
                "raw-lineage identity", "lexical near-duplicate",
            ],
            "source_normalization": (
                "Eval V3 normalize_source_url: scheme/default-port/path/"
                "tracking normalization with YouTube alias folding"
            ),
            "near_duplicate_rule": {
                "qa_semantic": {
                    "question_jaccard_direct": 0.82,
                    "question_jaccard_joint": 0.66,
                    "answer_jaccard_joint": 0.86,
                    "implementation": "frozen V13 lexical-semantic rule",
                    "train_features_loaded_from_opaque_registry": True,
                },
                "copy_detection": {
                    "unicode": "NFKC plus casefold",
                    "token_ngram_width": 3,
                    "jaccard_threshold": 0.80,
                    "containment_threshold": 0.90,
                    "containment_minimum_tokens": 12,
                },
            },
            "future_train_refresh_rule": (
                "Any refresh must rerun all four identity domains against the "
                "same opaque terminal selection before model update."
            ),
            "future_dataset_reservation_rule": {
                "roles": sorted(FUTURE_ADAPTATION_ROLES),
                "selected_sources_reserved": PROTECTED_TARGET_ROWS,
                "fresh_dev_sources_reserved": FRESH_DEV_TARGET_DOCUMENTS,
                "historical_v2_selected_sources_reserved": (
                    HISTORICAL_V2_SELECTED_SOURCE_COUNT
                ),
                "total_resealed_sources_reserved": RESEALED_CLEAR_SOURCE_COUNT,
                "fresh_dev_removed_from_terminal_eligibility": True,
                "unselected_terminal_eligible_sources": 0,
                "identity_registry_required_before_any_dataset_write": True,
                "missing_provenance_fails_closed": True,
                "enforcement_api": "assert_future_dataset_excludes_reserved",
            },
        },
        "score_aggregation": {
            "dev_primary": (
                "uniform mean of per-document continuation token logprob over "
                "four immutable fresh prose documents; candidate minus baseline"
            ),
            "dev_secondary_order": [
                "document token-logprob paired delta",
                "deterministic segment-level paired delta",
            ],
            "ood_noninferiority": {
                "paired_bootstrap_samples": 20000,
                "bootstrap_seed": 2026071701,
                "qa_mean_reward_delta_95_lcb_minimum": -0.02,
                "qa_exact_count_delta_minimum": -1,
                "prose_mean_token_logprob_delta_95_lcb_minimum": -0.02,
                "all_conditions_required": True,
            },
            "protected_terminal": {
                "per_document_statistic": "mean continuation token logprob",
                "aggregate": "uniform mean over immutable source documents",
                "comparison": "paired candidate-minus-frozen-baseline interval",
                "aggregate_only_persistence": True,
                "result_can_change_recipe": False,
            },
        },
        "compute_accounting": {
            "charged_gpu_second": (
                "sum over physical GPUs of all model-resident monotonic-time "
                "intervals, including failed attempts after output is observed"
            ),
            "excluded_time": [
                "scheduler queue", "CPU-only preregistration",
                "CPU-only dataset audit before model allocation",
            ],
            "generated_rollout": (
                "one generated completion; count candidates, mirrored signs, "
                "repeats, dev/OOD checks, and observed failed attempts"
            ),
            "also_required": [
                "prompt count", "generated tokens", "teacher-forced tokens",
                "SFT non-padding train tokens", "checkpoint count",
            ],
            "budget_modes": {
                "estimator_control": {
                    "screen_target_generated_rollouts_per_arm": 2048,
                    "screen_gpu_second_ceiling_per_arm": 14400,
                    "confirmation_target_generated_rollouts_per_seed": 2048,
                    "confirmation_gpu_second_ceiling_per_seed": 14400,
                },
                "compute_matched_quality": {
                    "screen_target_gpu_seconds_per_arm": 14400,
                    "confirmation_target_gpu_seconds_per_seed": 14400,
                    "terminal_target_gpu_seconds_per_seed": 28800,
                    "relative_match_tolerance": 0.02,
                },
                "systems_throughput": {"quality_promotion_authorized": False},
            },
            "four_gpu_requirement": {
                "physical_gpu_ids": [0, 1, 2, 3],
                "all_model_resident_intervals_attributed": True,
                "useful_positive_activity_each_gpu_per_training_phase": True,
                "idle_capacity_does_not_count_as_useful_activity": True,
            },
        },
        "seeds": {
            "split_seed": POOL_SELECTION_SEED,
            "screen_training_seeds": [1701],
            "confirmation_training_seeds": [1701, 1702, 1703],
            "bootstrap_seed": 2026071701,
            "unregistered_seed_retry": "prohibited",
        },
        "stopping_and_promotion": {
            "screening": {
                "checks_only_at_preregistered_rung_boundaries": True,
                "protected_terminal_visible": False,
                "failed_arm_budget_reallocated": False,
            },
            "promotion_to_confirmation": {
                "prose_logprob_tuning_only": True,
                "qa_or_general_quality_promotion_authorized": False,
                "dev_tuple_must_strictly_exceed_frozen_baseline": True,
                "every_ood_noninferiority_condition_must_pass": True,
                "compute_contract_must_pass": True,
                "protocol_or_leak_counter_may_not_increase": True,
            },
            "recipe_freeze": {
                "three_registered_seeds_required": True,
                "positive_dev_primary_seeds_minimum": 2,
                "pooled_dev_primary_paired_95_lcb_minimum": 0.0,
                "all_seed_ood_gates_required": True,
                "selection_receipt_required_before_terminal_access": True,
            },
            "terminal": {
                "one_shot_after_recipe_selection": True,
                "claim_and_load_are_one_api_call": True,
                "claim_persisted_and_directory_fsynced_before_source_read": True,
                "evaluator_source_bound_in_selection_receipt": True,
                "evaluator_must_be_reviewed_module_level_repository_code": True,
                "crash_or_error_after_claim_consumes_access": True,
                "aggregate_only": True,
                "retry_reopen_reset_or_alternate_claim_path": "prohibited",
                "post_result_hpo_or_recipe_change": "prohibited",
            },
        },
        "terminal_access": {
            "claim_path": str(TERMINAL_CLAIM),
            "claim_path_identity_sha256": _identity(
                "terminal-claim-path-v2", str(TERMINAL_CLAIM)
            ),
            "claim_schema": CLAIM_SCHEMA,
            "selection_receipt_schema": SELECTION_SCHEMA,
            "production_api": "claim_and_evaluate_protected",
        },
        "content_minimization": {
            "protected_question_persisted": False,
            "protected_answer_persisted": False,
            "protected_excerpt_persisted": False,
            "protected_text_persisted": False,
            "protected_url_persisted": False,
            "protected_source_path_persisted": False,
            "protected_per_item_metric_persisted": False,
            "protected_selection_uses_opaque_hashes_only": True,
            "fresh_dev_text_url_or_source_path_persisted": False,
            "fresh_dev_selection_uses_opaque_hashes_only": True,
            "future_adaptation_reservation_uses_opaque_hashes_only": True,
            "adaptation_artifacts_may_not_bind_terminal_source": True,
        },
        "implementation_bindings": {
            "builder": str(Path(__file__).resolve()),
            "builder_file_sha256": file_sha256(Path(__file__).resolve()),
            "eval_normalizer_file_sha256": file_sha256(eval_v3.__file__),
            "lineage_rule_file_sha256": file_sha256(folds_v37a.__file__),
            "semantic_rule_file_sha256": file_sha256(semantic_v13.__file__),
            "curated_qa_boundary_file_sha256": file_sha256(CURATED_QA_BUILDER),
            "legacy_eval_incident_builder_file_sha256": file_sha256(
                LEGACY_EVAL_INCIDENT_BUILDER
            ),
            "recursive_lookup_incident_builder_file_sha256": file_sha256(
                RECURSIVE_LOOKUP_INCIDENT_BUILDER
            ),
        },
    }
    contract["content_sha256_before_self_field"] = canonical_sha256(contract)
    return contract


def _contains_semantic_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in SEMANTIC_FIELD_NAMES
            or _contains_semantic_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_semantic_key(item) for item in value)
    return False


def _validated_reservation(contract: dict) -> dict[str, frozenset[str]]:
    protected = contract.get("roles", {}).get("protected_terminal", {})
    reservation = protected.get("future_adaptation_reservation", {})
    compact = {
        key: value for key, value in reservation.items()
        if key != "content_sha256_before_self_field"
    }
    domains = reservation.get("identity_domains", {})
    if (
        reservation.get("schema") != RESERVATION_SCHEMA
        or reservation.get("status")
        != "all_resealed_dev_terminal_sources_reserved_before_adaptation"
        or reservation.get("roles") != sorted(FUTURE_ADAPTATION_ROLES)
        or reservation.get("reserved_source_count")
        != RESEALED_CLEAR_SOURCE_COUNT
        or reservation.get("terminal_source_count") != PROTECTED_TARGET_ROWS
        or reservation.get("dev_source_count") != FRESH_DEV_TARGET_DOCUMENTS
        or reservation.get("historical_selected_source_count")
        != HISTORICAL_V2_SELECTED_SOURCE_COUNT
        or reservation.get("historical_selected_sources_remain_reserved")
        is not True
        or reservation.get("terminal_eligible_unselected_sources") != 0
        or reservation.get("enforcement_api")
        != "assert_future_dataset_excludes_reserved"
        or reservation.get("missing_or_unverifiable_row_provenance")
        != "reject_fail_closed"
        or reservation.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or set(domains) != set(RESERVED_IDENTITY_DOMAINS)
        or _contains_semantic_key(reservation)
    ):
        raise RuntimeError("invalid V2 future-adaptation reservation")

    validated = {}
    for name in RESERVED_IDENTITY_DOMAINS:
        domain = domains[name]
        values = domain.get("opaque_values") if isinstance(domain, dict) else None
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in values
            )
            or domain.get("count") != len(values)
            or domain.get("opaque_value_set_sha256")
            != canonical_sha256(values)
        ):
            raise RuntimeError("invalid opaque reservation identity domain")
        validated[name] = frozenset(values)

    selected_sources = protected.get("selected_sources", ())
    dev_sources = contract.get("roles", {}).get("dev", {}).get(
        "source_bindings", ()
    )
    all_sources = list(selected_sources) + list(dev_sources)
    expected_binding_keys = {
        "added_commit", "corpus_file_sha256", "metadata_bundle_sha256",
        "opaque_item_identity", "source_path_identity_sha256",
        "url_identity_set_sha256",
    }
    if (
        not isinstance(selected_sources, list)
        or len(selected_sources) != PROTECTED_TARGET_ROWS
        or not isinstance(dev_sources, list)
        or len(dev_sources) != FRESH_DEV_TARGET_DOCUMENTS
        or len(all_sources) != RESEALED_CLEAR_SOURCE_COUNT
        or len({item.get("opaque_item_identity") for item in all_sources})
        != RESEALED_CLEAR_SOURCE_COUNT
        or any(
            not isinstance(item, dict)
            or set(item) != expected_binding_keys
            or not re.fullmatch(r"[0-9a-f]{40}", item["added_commit"])
            or any(
                not isinstance(item[key], str)
                or not SHA256_RE.fullmatch(item[key])
                for key in expected_binding_keys - {"added_commit"}
            )
            for item in all_sources
        )
        or sorted(item["opaque_item_identity"] for item in all_sources)
        != sorted(validated["opaque_item_identity"])
        or {
            item["source_path_identity_sha256"] for item in all_sources
        }
        != set(validated["source_path_identity_sha256"])
        or not {
            item["corpus_file_sha256"] for item in all_sources
        }.issubset(validated["document_sha256"])
    ):
        raise RuntimeError("selected V2 source bindings do not match reservation")
    return validated


def _validated_fresh_dev_reservation(
    contract: dict,
) -> dict[str, frozenset[str]]:
    dev = contract.get("roles", {}).get("dev", {})
    reservation = dev.get("future_adaptation_reservation", {})
    compact = {
        key: value for key, value in reservation.items()
        if key != "content_sha256_before_self_field"
    }
    domains = reservation.get("identity_domains", {})
    if (
        reservation.get("schema") != DEV_RESERVATION_SCHEMA
        or reservation.get("status")
        != "fresh_dev_source_reserved_outside_model_adaptation"
        or reservation.get("roles") != sorted(FUTURE_ADAPTATION_ROLES)
        or reservation.get("dev_source_count") != FRESH_DEV_TARGET_DOCUMENTS
        or reservation.get("removed_from_terminal_eligibility") is not True
        or reservation.get("enforcement_api")
        != "assert_future_dataset_excludes_reserved"
        or reservation.get("missing_or_unverifiable_row_provenance")
        != "reject_fail_closed"
        or reservation.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or set(domains) != set(RESERVED_IDENTITY_DOMAINS)
        or _contains_semantic_key(reservation)
    ):
        raise RuntimeError("invalid fresh dev future-adaptation reservation")
    validated = {}
    for name in RESERVED_IDENTITY_DOMAINS:
        domain = domains[name]
        values = domain.get("opaque_values") if isinstance(domain, dict) else None
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in values
            )
            or domain.get("count") != len(values)
            or domain.get("opaque_value_set_sha256")
            != canonical_sha256(values)
        ):
            raise RuntimeError("invalid fresh dev opaque identity domain")
        validated[name] = frozenset(values)
    bindings = dev.get("source_bindings", ())
    if (
        not isinstance(bindings, list)
        or len(bindings) != FRESH_DEV_TARGET_DOCUMENTS
        or {
            binding.get("opaque_item_identity") for binding in bindings
            if isinstance(binding, dict)
        } != set(validated["opaque_item_identity"])
        or {
            binding.get("source_path_identity_sha256") for binding in bindings
            if isinstance(binding, dict)
        } != set(validated["source_path_identity_sha256"])
        or not {
            binding.get("corpus_file_sha256") for binding in bindings
            if isinstance(binding, dict)
        }.issubset(validated["document_sha256"])
    ):
        raise RuntimeError("fresh dev binding does not match reservation")
    return validated


def validate_contract(contract: dict) -> None:
    content_sha = contract.get("content_sha256_before_self_field")
    compact = {
        key: value for key, value in contract.items()
        if key != "content_sha256_before_self_field"
    }
    protected = contract.get("roles", {}).get("protected_terminal", {})
    dev = contract.get("roles", {}).get("dev", {})
    selected_sources = protected.get("selected_sources", ())
    pool_audit = protected.get("source_pool_audit", {})
    disjoint_pairs = contract.get("disjointness", {}).get(
        "audit", {}
    ).get("pairs", {})
    expected_pairs = {
        f"{left}__{right}"
        for index, left in enumerate(ROLE_ORDER)
        for right in ROLE_ORDER[index + 1:]
    }
    quarantine_exclusion = contract.get("quarantine", {}).get(
        "exclusion_audit", {}
    )
    minimization = contract.get("content_minimization", {})
    implementation = contract.get("implementation_bindings", {})
    if (
        contract.get("schema") != SCHEMA
        or contract.get("status") != STATUS
        or content_sha != canonical_sha256(compact)
        or contract.get("quarantine", {}).get("passed") is not True
        or contract.get("quarantine", {}).get("source_file_sha256")
        != QUARANTINED_V1_SOURCE_SHA256
        or contract.get("incident", {}).get("content_sha256")
        != INCIDENT_CONTENT_SHA256
        or contract.get("additional_legacy_evaluation_incident", {}).get(
            "content_sha256"
        ) != LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
        or contract.get("recursive_lookup_incident", {}).get("content_sha256")
        != RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256
        or contract.get("content_free_quarantine_boundary", {}).get(
            "content_sha256"
        ) != QUARANTINE_BOUNDARY_REGISTRY_CONTENT_SHA256
        or contract.get("content_free_quarantine_boundary", {}).get(
            "exact_path_identity_count"
        ) != 3
        or contract.get("content_free_quarantine_boundary", {}).get(
            "prefix_identity_count"
        ) != 2
        or contract.get("content_free_quarantine_boundary", {}).get(
            "plaintext_boundary_paths_persisted"
        ) is not False
        or contract.get("superseded_v2_revision", {}).get("content_sha256")
        != SUPERSEDED_V2_CONTRACT_CONTENT_SHA256
        or contract.get("superseded_v2_revision", {}).get("status")
        != "immutable_superseded_nonpromotable"
        or not contract.get("disjointness", {}).get("passed")
        or dev.get("rows") != FRESH_DEV_TARGET_DOCUMENTS
        or dev.get("documents") != FRESH_DEV_TARGET_DOCUMENTS
        or dev.get("source_path_persisted") is not False
        or dev.get("source_text_or_url_persisted") is not False
        or dev.get("outside_recursively_accessed_prefixes") is not True
        or dev.get("removed_from_terminal_eligibility") is not True
        or dev.get("terminal_identity_intersection") != 0
        or dev.get("prior_v37a_dev_reused") is not False
        or dev.get("prose_logprob_tuning_authorized") is not True
        or dev.get("multi_item_qa_dev_present") is not False
        or dev.get("qa_hpo_or_quality_promotion_authorized") is not False
        or contract.get("authority", {}).get(
            "systems_and_prose_logprob_tuning_only"
        ) is not True
        or contract.get("authority", {}).get(
            "qa_hpo_or_general_quality_promotion_authorized"
        ) is not False
        or protected.get("access_authorized_by_this_contract") is not False
        or protected.get("rows") != PROTECTED_TARGET_ROWS
        or len(selected_sources) != PROTECTED_TARGET_ROWS
        or _contains_semantic_key(selected_sources)
        or pool_audit.get("candidate_sources") != len(POST_FREEZE_POOL)
        or pool_audit.get("terminal_selected_sources") != PROTECTED_TARGET_ROWS
        or pool_audit.get("fresh_dev_allocated_sources")
        != FRESH_DEV_TARGET_DOCUMENTS
        or pool_audit.get("resealed_clear_sources")
        != RESEALED_CLEAR_SOURCE_COUNT
        or pool_audit.get("historical_selected_sources_still_reserved")
        != HISTORICAL_V2_SELECTED_SOURCE_COUNT
        or pool_audit.get("terminal_eligible_unselected_sources") != 0
        or pool_audit.get("dev_sources_removed_from_terminal_eligibility")
        is not True
        or pool_audit.get("excluded_sources") != 14
        or pool_audit.get("selected_source_pair_collision_count") != 0
        or set(disjoint_pairs) != expected_pairs
        or any(
            pair.get("colliding_row_pairs") != 0
            or any(pair.get("by_identity_domain", {}).values())
            for pair in disjoint_pairs.values()
        )
        or quarantine_exclusion.get(
            "candidate_source_path_matches_quarantined_v1"
        ) is not False
        or quarantine_exclusion.get(
            "candidate_source_hash_matches_quarantined_v1"
        ) is not False
        or quarantine_exclusion.get(
            "resealed_clear_identity_intersection_with_known_v1"
        ) != 0
        or quarantine_exclusion.get(
            "candidate_source_path_identity_intersection_with_legacy_eval_family"
        ) != 0
        or quarantine_exclusion.get(
            "fresh_dev_identity_intersection_with_reserved_terminal"
        ) != 0
        or quarantine_exclusion.get(
            "fresh_dev_or_terminal_source_under_recursively_accessed_prefix"
        ) is not False
        or contract.get("quarantine", {}).get(
            "v1_row_level_semantic_comparison"
        ) != "prohibited_not_performed"
        or contract.get("quarantine", {}).get(
            "unknown_v1_row_semantic_disjointness_claimed"
        ) is not False
        or contract.get("quarantine", {}).get(
            "additional_legacy_eval_source_path_identity_sha256"
        ) != sorted(_legacy_eval_path_identities())
        or contract.get("quarantine", {}).get(
            "additional_legacy_eval_source_reopen_stat_or_hash_prohibited"
        ) is not True
        or contract.get("quarantine", {}).get(
            "additional_legacy_eval_future_roles_excluded"
        ) != sorted(FUTURE_ADAPTATION_ROLES)
        or contract.get("quarantine", {}).get(
            "curated_qa_implicit_eval_default_prohibited"
        ) is not True
        or contract.get("quarantine", {}).get(
            "recursive_lookup_accessed_prefix_identity_sha256"
        ) != sorted(_accessed_prefix_identities())
        or contract.get("quarantine", {}).get(
            "recursive_lookup_file_inventory_reconstruction_prohibited"
        ) is not True
        or contract.get("quarantine", {}).get(
            "prior_train_dev_paths_resolved_statted_hashed_or_opened_during_reseal"
        ) is not False
        or contract.get("quarantine", {}).get("prior_dev_role_reuse_prohibited")
        is not True
        or contract.get("quarantine", {}).get(
            "fresh_dev_outside_accessed_prefixes"
        ) is not True
        or contract.get("quarantine", {}).get(
            "future_worker_explicit_file_allowlist_required"
        ) is not True
        or minimization.get("protected_source_path_persisted") is not False
        or minimization.get("protected_text_persisted") is not False
        or minimization.get("protected_url_persisted") is not False
        or minimization.get("protected_per_item_metric_persisted") is not False
        or minimization.get("fresh_dev_text_url_or_source_path_persisted")
        is not False
        or minimization.get("fresh_dev_selection_uses_opaque_hashes_only")
        is not True
        or implementation.get("builder_file_sha256")
        != file_sha256(Path(__file__).resolve())
        or implementation.get("eval_normalizer_file_sha256")
        != file_sha256(eval_v3.__file__)
        or implementation.get("lineage_rule_file_sha256")
        != file_sha256(folds_v37a.__file__)
        or implementation.get("semantic_rule_file_sha256")
        != file_sha256(semantic_v13.__file__)
        or implementation.get("curated_qa_boundary_file_sha256")
        != file_sha256(CURATED_QA_BUILDER)
        or implementation.get("legacy_eval_incident_builder_file_sha256")
        != file_sha256(LEGACY_EVAL_INCIDENT_BUILDER)
        or implementation.get("recursive_lookup_incident_builder_file_sha256")
        != file_sha256(RECURSIVE_LOOKUP_INCIDENT_BUILDER)
        or contract.get("terminal_access", {}).get("claim_path")
        != str(TERMINAL_CLAIM)
    ):
        raise RuntimeError("invalid or quarantined recipe evaluation contract")
    reserved = _validated_reservation(contract)
    _validated_fresh_dev_reservation(contract)
    _validate_content_free_quarantine_boundary()
    superseded = _validate_superseded_v2_contract()
    historical_ids = set(
        superseded["roles"]["protected_terminal"][
            "selected_opaque_item_identities"
        ]
    )
    if not historical_ids.issubset(reserved["opaque_item_identity"]):
        raise RuntimeError("historical 12-source reservation was weakened")


def _add_opaque_values(
    target: set[str], value: object, *, label: str
) -> bool:
    if value is None:
        return False
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    if not values:
        return False
    for item in values:
        if not isinstance(item, str) or not SHA256_RE.fullmatch(item):
            raise RuntimeError(f"future dataset has invalid {label}")
        target.add(item)
    return True


def build_future_dataset_identity_registry(
    rows: Iterable[dict], *, role: str
) -> dict:
    """Reduce future rows to opaque source identities, rejecting missing lineage."""
    normalized_role = str(role).lower()
    if normalized_role not in FUTURE_ADAPTATION_ROLES:
        raise RuntimeError("future dataset role is outside the reservation policy")
    domains = {name: set() for name in RESERVED_IDENTITY_DOMAINS}
    row_count = 0
    for row in rows:
        row_count += 1
        if not isinstance(row, dict):
            raise RuntimeError("future dataset row is not an object")
        has_provenance = False
        for key in ("opaque_item_identity", "source_opaque_item_identity"):
            has_provenance |= _add_opaque_values(
                domains["opaque_item_identity"], row.get(key), label=key
            )
        for key in (
            "document_sha256", "source_document_sha256", "corpus_file_sha256",
            "source_document_sha256s",
        ):
            has_provenance |= _add_opaque_values(
                domains["document_sha256"], row.get(key), label=key
            )
        for key in (
            "source_path_identity_sha256", "source_path_identity_sha256s",
        ):
            has_provenance |= _add_opaque_values(
                domains["source_path_identity_sha256"], row.get(key), label=key
            )
        for key in (
            "normalized_url_identity_sha256",
            "normalized_url_identity_sha256s",
        ):
            has_provenance |= _add_opaque_values(
                domains["normalized_url_identity_sha256"], row.get(key),
                label=key,
            )
        for key in (
            "raw_lineage_identity_sha256", "raw_lineage_identity_sha256s",
        ):
            has_provenance |= _add_opaque_values(
                domains["raw_lineage_identity_sha256"], row.get(key), label=key
            )

        for key in ("source_relative_path", "corpus_relative_path"):
            value = row.get(key)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise RuntimeError("future dataset source path is invalid")
                if _relative_path_is_under_accessed_prefix(value):
                    raise RuntimeError(
                        "future dataset source falls under recursively accessed prefix"
                    )
                domains["source_path_identity_sha256"].add(
                    _identity(
                        "repository-relative-path-v1",
                        _normalize_repository_relative_path(value),
                    )
                )
                has_provenance = True
        try:
            normalized_urls = _row_urls(row)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("future dataset source URL is invalid") from exc
        if normalized_urls:
            domains["normalized_url_identity_sha256"].update(
                _identity("normalized-source-url-v2", value)
                for value in normalized_urls
            )
            has_provenance = True
        lineage_ids = _lineage_ids(row)
        if lineage_ids:
            domains["raw_lineage_identity_sha256"].update(lineage_ids)
            has_provenance = True
        if not has_provenance:
            raise RuntimeError(
                "future dataset row lacks auditable source provenance; refusing write"
            )

    opaque_domains = {
        name: sorted(domains[name]) for name in RESERVED_IDENTITY_DOMAINS
    }
    registry = {
        "schema": FUTURE_DATASET_IDENTITY_SCHEMA,
        "role": normalized_role,
        "rows": row_count,
        "all_rows_have_auditable_source_provenance": True,
        "identity_domains": opaque_domains,
    }
    registry["content_sha256_before_self_field"] = canonical_sha256(registry)
    return registry


def assert_future_dataset_excludes_reserved(
    identity_registry: dict, contract: dict
) -> dict:
    """Fail closed before any train/dev/CPT/SFT/QA artifact write."""
    validate_contract(contract)
    if not isinstance(identity_registry, dict):
        raise RuntimeError("future dataset identity registry is missing")
    compact = {
        key: value for key, value in identity_registry.items()
        if key != "content_sha256_before_self_field"
    }
    role = identity_registry.get("role")
    domains = identity_registry.get("identity_domains")
    rows = identity_registry.get("rows")
    if (
        identity_registry.get("schema") != FUTURE_DATASET_IDENTITY_SCHEMA
        or role not in FUTURE_ADAPTATION_ROLES
        or not isinstance(rows, int) or isinstance(rows, bool) or rows < 0
        or identity_registry.get("all_rows_have_auditable_source_provenance")
        is not True
        or identity_registry.get("content_sha256_before_self_field")
        != canonical_sha256(compact)
        or not isinstance(domains, dict)
        or set(domains) != set(RESERVED_IDENTITY_DOMAINS)
    ):
        raise RuntimeError("future dataset identity registry is incomplete")

    reserved = _validated_reservation(contract)
    fresh_dev_reserved = _validated_fresh_dev_reservation(contract)
    intersections = {}
    fresh_dev_intersections = {}
    for name in RESERVED_IDENTITY_DOMAINS:
        values = domains[name]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or any(
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                for value in values
            )
        ):
            raise RuntimeError("future dataset opaque identity domain is invalid")
        intersections[name] = len(set(values) & reserved[name])
        fresh_dev_intersections[name] = len(
            set(values) & fresh_dev_reserved[name]
        )
    legacy_eval_path_intersection = len(
        set(domains["source_path_identity_sha256"])
        & _legacy_eval_path_identities()
    )
    if legacy_eval_path_intersection:
        raise RuntimeError(
            "future dataset intersects quarantined legacy evaluation path identities"
        )
    if any(intersections.values()):
        raise RuntimeError(
            "future dataset intersects reserved V2 terminal source identities"
        )
    if any(fresh_dev_intersections.values()):
        raise RuntimeError(
            "future dataset intersects reserved fresh V2 dev source identities"
        )
    return {
        "passed": True,
        "role": role,
        "rows": rows,
        "selected_sources_reserved": PROTECTED_TARGET_ROWS,
        "fresh_dev_sources_reserved": FRESH_DEV_TARGET_DOCUMENTS,
        "historical_v2_selected_sources_reserved": (
            HISTORICAL_V2_SELECTED_SOURCE_COUNT
        ),
        "total_resealed_sources_reserved": RESEALED_CLEAR_SOURCE_COUNT,
        "unselected_terminal_eligible_sources": 0,
        "intersection_counts": intersections,
        "fresh_dev_intersection_counts": fresh_dev_intersections,
        "legacy_eval_path_identity_intersection_count": 0,
        "reservation_content_sha256": contract["roles"]["protected_terminal"]
        ["future_adaptation_reservation"]["content_sha256_before_self_field"],
    }


def assert_future_dataset_rows_exclude_reserved(
    rows: Iterable[dict], *, role: str, contract: dict
) -> dict:
    """Convenience boundary for builders; call before opening output files."""
    registry = build_future_dataset_identity_registry(rows, role=role)
    return assert_future_dataset_excludes_reserved(registry, contract)


def assert_adaptation_inputs(paths: Iterable[Path | str], contract: dict) -> None:
    validate_contract(contract)
    allowed_sha = contract["roles"]["train"]["file_sha256"]
    for value in paths:
        lexical = _lexical_absolute_path(value)
        if lexical in QUARANTINED_LEGACY_EVAL_ABSOLUTE_PATHS:
            raise RuntimeError(
                "quarantined legacy evaluation source cannot enter adaptation"
            )
        if lexical == _lexical_absolute_path(QUARANTINED_V1_SOURCE_PATH):
            raise RuntimeError("quarantined V1 source cannot enter adaptation")
        if (
            _lexical_path_is_under_accessed_prefix(lexical)
            and lexical != _lexical_absolute_path(TRAIN)
        ):
            raise RuntimeError(
                "adaptation path is outside the explicit post-incident allowlist"
            )
        path = Path(value).resolve()
        if path in QUARANTINED_LEGACY_EVAL_ABSOLUTE_PATHS:
            raise RuntimeError(
                "resolved adaptation input aliases quarantined legacy evaluation"
            )
        if path == QUARANTINED_V1_SOURCE_PATH:
            raise RuntimeError("quarantined V1 source cannot enter adaptation")
        if (
            _lexical_path_is_under_accessed_prefix(path)
            and path != _lexical_absolute_path(TRAIN)
        ):
            raise RuntimeError(
                "resolved adaptation path is outside the explicit allowlist"
            )
        if not path.is_file():
            raise RuntimeError("adaptation input does not exist")
        observed = file_sha256(path)
        if observed == QUARANTINED_V1_SOURCE_SHA256:
            raise RuntimeError("quarantined V1 bytes cannot enter adaptation")
        if observed != allowed_sha:
            raise RuntimeError("adaptation input is absent from audited train registry")


def charge_compute_attempt(attempt: dict) -> dict:
    by_gpu = defaultdict(list)
    for interval in attempt.get("gpu_residency_intervals", ()):
        gpu = int(interval["physical_gpu_id"])
        start, end = float(interval["start_s"]), float(interval["end_s"])
        if gpu not in (0, 1, 2, 3) or not math.isfinite(start + end) or end <= start:
            raise ValueError("invalid GPU residency interval")
        by_gpu[gpu].append((start, end))
    if set(by_gpu) != {0, 1, 2, 3}:
        raise ValueError("compute attempt must attribute all four GPUs")
    gpu_seconds = 0.0
    for intervals in by_gpu.values():
        intervals.sort()
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("overlapping residency intervals on one GPU")
        gpu_seconds += math.fsum(end - start for start, end in intervals)
    counts = {}
    for key in (
        "optimization_generated_rollouts", "evaluation_generated_rollouts",
        "generated_tokens", "teacher_forced_tokens", "sft_nonpadding_tokens",
    ):
        value = attempt.get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid compute count {key}")
        counts[key] = value
    return {"charged_gpu_seconds": gpu_seconds, **counts}


def aggregate_compute_ledger(attempts: Iterable[dict]) -> dict[str, dict]:
    totals = defaultdict(lambda: defaultdict(float))
    for attempt in attempts:
        arm = attempt.get("arm")
        if not isinstance(arm, str) or not arm:
            raise ValueError("compute attempt has no arm")
        for key, value in charge_compute_attempt(attempt).items():
            totals[arm][key] += value
    return {
        arm: {
            key: (int(value) if key != "charged_gpu_seconds" else value)
            for key, value in values.items()
        }
        for arm, values in sorted(totals.items())
    }


def validate_compute_match(
    totals: dict[str, dict], *, mode: str, contract: dict
) -> dict:
    validate_contract(contract)
    if len(totals) < 2:
        raise ValueError("compute matching requires at least two arms")
    policy = contract["compute_accounting"]["budget_modes"].get(mode)
    if policy is None:
        raise ValueError(f"unsupported compute mode {mode}")
    values = list(totals.values())
    gpu_seconds = [float(item["charged_gpu_seconds"]) for item in values]
    if any(value <= 0 for value in gpu_seconds):
        raise RuntimeError("compute matching requires positive GPU seconds")
    if mode == "estimator_control":
        target = policy["screen_target_generated_rollouts_per_arm"]
        if any(item["optimization_generated_rollouts"] != target for item in values):
            raise RuntimeError("estimator-control rollout budgets differ")
        if len({item["evaluation_generated_rollouts"] for item in values}) != 1:
            raise RuntimeError("estimator-control evaluation rollouts differ")
        if any(value > policy["screen_gpu_second_ceiling_per_arm"] for value in gpu_seconds):
            raise RuntimeError("estimator-control GPU-second ceiling exceeded")
    elif mode == "compute_matched_quality":
        if max(gpu_seconds) / min(gpu_seconds) - 1.0 > policy["relative_match_tolerance"]:
            raise RuntimeError("quality arms are not GPU-second matched")
        if len({item["evaluation_generated_rollouts"] for item in values}) != 1:
            raise RuntimeError("quality evaluation rollout counts differ")
    else:
        work_keys = (
            "optimization_generated_rollouts", "evaluation_generated_rollouts",
            "generated_tokens", "teacher_forced_tokens", "sft_nonpadding_tokens",
        )
        if any(len({item[key] for item in values}) != 1 for key in work_keys):
            raise RuntimeError("systems-throughput workloads differ")
    return {
        "passed": True,
        "mode": mode,
        "arms": sorted(totals),
        "gpu_second_min": min(gpu_seconds),
        "gpu_second_max": max(gpu_seconds),
    }


def _validate_selection_receipt(selection: dict, contract: dict) -> str:
    expected_keys = {
        "schema", "status", "content_sha256_before_self_field",
        "contract_content_sha256", "hpo_closed",
        "v2_terminal_access_count_before_selection",
        "v1_quarantined_access_count_nonzero_acknowledged",
        "incident_content_sha256", "selected_recipe_id",
        "legacy_eval_collision_incident_content_sha256",
        "recursive_lookup_incident_content_sha256",
        "superseded_v2_nonpromotable_acknowledged",
        "fresh_dev_source_identity_set_sha256",
        "selected_checkpoint_sha256", "terminal_evaluator_file_sha256",
        "terminal_evaluator_path_identity_sha256",
        "terminal_evaluator_aggregate_only_reviewed",
        "terminal_evaluator_side_effect_reviewed",
    }
    if not isinstance(selection, dict):
        raise RuntimeError("terminal access requires a valid V2 frozen selection")
    compact = {
        key: value for key, value in selection.items()
        if key != "content_sha256_before_self_field"
    }
    selection_sha = canonical_sha256(compact)
    if (
        set(selection) != expected_keys
        or selection.get("schema") != SELECTION_SCHEMA
        or selection.get("status") != "recipe_selected_frozen_hpo_closed"
        or selection.get("content_sha256_before_self_field") != selection_sha
        or selection.get("contract_content_sha256")
        != contract["content_sha256_before_self_field"]
        or selection.get("hpo_closed") is not True
        or selection.get("v2_terminal_access_count_before_selection") != 0
        or selection.get("v1_quarantined_access_count_nonzero_acknowledged") is not True
        or selection.get("incident_content_sha256") != INCIDENT_CONTENT_SHA256
        or selection.get("legacy_eval_collision_incident_content_sha256")
        != LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
        or selection.get("recursive_lookup_incident_content_sha256")
        != RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256
        or selection.get("superseded_v2_nonpromotable_acknowledged") is not True
        or selection.get("fresh_dev_source_identity_set_sha256")
        != contract["roles"]["dev"]["source_identity_set_sha256"]
        or not isinstance(selection.get("selected_recipe_id"), str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,128}", selection["selected_recipe_id"]
        )
        or not isinstance(selection.get("selected_checkpoint_sha256"), str)
        or not SHA256_RE.fullmatch(selection["selected_checkpoint_sha256"])
        or not isinstance(selection.get("terminal_evaluator_file_sha256"), str)
        or not SHA256_RE.fullmatch(selection["terminal_evaluator_file_sha256"])
        or not isinstance(
            selection.get("terminal_evaluator_path_identity_sha256"), str
        )
        or not SHA256_RE.fullmatch(
            selection["terminal_evaluator_path_identity_sha256"]
        )
        or selection.get("terminal_evaluator_aggregate_only_reviewed") is not True
        or selection.get("terminal_evaluator_side_effect_reviewed") is not True
    ):
        raise RuntimeError("terminal access requires a valid V2 frozen selection")
    return selection_sha


def _validate_evaluator_binding(
    evaluator: Callable[[list[dict], dict], dict], selection: dict
) -> None:
    source_name = inspect.getsourcefile(evaluator)
    qualified_name = getattr(evaluator, "__qualname__", "")
    if (
        not inspect.isfunction(evaluator)
        or not isinstance(source_name, str)
        or "<locals>" in qualified_name
    ):
        raise RuntimeError("terminal evaluator must be reviewed module-level code")
    source_path = Path(source_name)
    source = source_path.resolve()
    if (
        not source.is_file()
        or source_path.is_symlink()
        or not source.is_relative_to(ROOT)
    ):
        raise RuntimeError("terminal evaluator source is outside the repository")
    relative = str(source.relative_to(ROOT))
    if (
        file_sha256(source) != selection["terminal_evaluator_file_sha256"]
        or _identity("terminal-aggregate-evaluator-path-v2", relative)
        != selection["terminal_evaluator_path_identity_sha256"]
    ):
        raise RuntimeError("terminal evaluator does not match frozen selection")


def _write_exclusive_durable(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # Never remove a partial claim: any creation attempt consumes access.
        raise


def _replace_claim_state(path: Path, value: dict) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _claim_and_load_once(
    state_path: Path,
    *,
    contract_content_sha256: str,
    selection_content_sha256: str,
    selected_identity_set_sha256: str,
    loader: Callable[[], list[dict]],
) -> tuple[list[dict], dict]:
    """Synthetic-testable primitive; O_EXCL claim always precedes loader."""
    claim = {
        "schema": CLAIM_SCHEMA,
        "status": "claimed_access_irrevocably_consumed_before_source_read",
        "access_count": 1,
        "contract_content_sha256": contract_content_sha256,
        "selection_content_sha256": selection_content_sha256,
        "selected_identity_set_sha256": selected_identity_set_sha256,
        "source_open_attempted": False,
        "retry_reset_or_reopen_allowed": False,
    }
    claim["content_sha256_before_self_field"] = canonical_sha256(claim)
    _write_exclusive_durable(state_path, claim)
    try:
        rows = loader()
    except BaseException:
        failed = dict(claim)
        failed.pop("content_sha256_before_self_field", None)
        failed.update({
            "status": "source_open_failed_access_consumed_no_retry",
            "source_open_attempted": True,
        })
        failed["content_sha256_before_self_field"] = canonical_sha256(failed)
        _replace_claim_state(state_path, failed)
        raise
    consumed = dict(claim)
    consumed.pop("content_sha256_before_self_field", None)
    consumed.update({
        "status": "source_open_completed_access_consumed_no_retry",
        "source_open_attempted": True,
    })
    consumed["content_sha256_before_self_field"] = canonical_sha256(consumed)
    _replace_claim_state(state_path, consumed)
    return rows, consumed


def _load_selected_sources(contract: dict) -> list[dict]:
    expected = {
        source["source_path_identity_sha256"]: source
        for source in contract["roles"]["protected_terminal"]["selected_sources"]
    }
    rows = []
    for relative_directory, added_commit in POST_FREEZE_POOL:
        relative_path = f"{relative_directory}/CORPUS.md"
        path_identity = _identity("repository-relative-path-v1", relative_path)
        source = expected.get(path_identity)
        if source is None:
            continue
        candidate = _candidate_from_source(
            relative_directory,
            added_commit,
            include_terminal_text=True,
        )
        if _selected_source_binding(candidate) != source:
            raise RuntimeError("V2 terminal source opaque binding changed")
        rows.append({
            "item_id": source["opaque_item_identity"],
            "document_sha256": source["corpus_file_sha256"],
            "text": candidate["terminal_text"],
        })
    if len(rows) != len(expected):
        raise RuntimeError("V2 terminal opaque locator set no longer resolves")
    return rows


def _run_aggregate_evaluator(
    rows: list[dict],
    claim: dict,
    evaluator: Callable[[list[dict], dict], dict],
    validator: Callable[[dict], None],
) -> dict:
    """Suppress all evaluator output and return only a validated aggregate."""
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            receipt = evaluator(rows, claim)
    except BaseException:
        raise RuntimeError(
            "terminal evaluator failed; access is consumed and details suppressed"
        ) from None
    if stdout.getvalue() or stderr.getvalue():
        raise RuntimeError(
            "terminal evaluator emitted output; access is consumed and evidence rejected"
        )
    validator(receipt)
    return receipt


def claim_and_evaluate_protected(
    state_path: Path | str,
    contract: dict,
    selection: dict,
    evaluator: Callable[[list[dict], dict], dict],
) -> dict:
    """Claim once, evaluate in-process, and return aggregate evidence only."""
    validate_contract(contract)
    path = Path(state_path).resolve()
    if path != TERMINAL_CLAIM:
        raise RuntimeError("production terminal claim path is immutable")
    selection_sha = _validate_selection_receipt(selection, contract)
    _validate_evaluator_binding(evaluator, selection)
    rows, claim = _claim_and_load_once(
        path,
        contract_content_sha256=contract["content_sha256_before_self_field"],
        selection_content_sha256=selection_sha,
        selected_identity_set_sha256=(
            contract["roles"]["protected_terminal"]["selected_identity_set_sha256"]
        ),
        loader=lambda: _load_selected_sources(contract),
    )
    selected = set(
        contract["roles"]["protected_terminal"]["selected_opaque_item_identities"]
    )
    if len(rows) != PROTECTED_TARGET_ROWS or {row["item_id"] for row in rows} != selected:
        raise RuntimeError("V2 protected opaque selection no longer resolves")
    def validate_bound_receipt(receipt: dict) -> None:
        if (
            not isinstance(receipt, dict)
            or receipt.get("terminal_claim_state_content_sha256")
            != claim["content_sha256_before_self_field"]
        ):
            raise RuntimeError("terminal receipt is not bound to the consumed claim")
        validate_terminal_aggregate_receipt(receipt, contract)

    return _run_aggregate_evaluator(
        rows,
        claim,
        evaluator,
        validate_bound_receipt,
    )


def claim_and_load_protected_rows(*_args, **_kwargs):
    raise RuntimeError(
        "returning raw protected rows is prohibited; use "
        "claim_and_evaluate_protected for aggregate-only evidence"
    )


def claim_protected_access_once(*_args, **_kwargs):
    raise RuntimeError(
        "split claim/load is prohibited; use claim_and_evaluate_protected"
    )


def load_claimed_protected_rows(*_args, **_kwargs):
    raise RuntimeError(
        "reopening a claimed protected source is prohibited in V2"
    )


def validate_terminal_aggregate_receipt(receipt: dict, contract: dict) -> None:
    validate_contract(contract)
    root_keys = {
        "schema", "status", "aggregate_only", "contract_content_sha256",
        "terminal_claim_state_content_sha256", "documents", "baseline",
        "candidate", "paired_delta", "bootstrap",
    }
    metric_keys = {"mean_document_token_logprob"}
    delta_keys = {
        "mean_document_token_logprob", "lower_95", "upper_95",
    }
    bootstrap_keys = {"samples", "seed"}
    if not isinstance(receipt, dict) or set(receipt) != root_keys:
        raise RuntimeError("terminal receipt fields changed or are not aggregate-only")
    if any(FORBIDDEN_TERMINAL_KEYS.search(str(key)) for key in receipt):
        raise RuntimeError("terminal receipt contains a forbidden semantic field")
    if (
        receipt.get("schema")
        != "specialist-protected-terminal-aggregate-receipt-v2"
        or receipt.get("status") != "terminal_complete_access_consumed_no_retry"
        or receipt.get("aggregate_only") is not True
        or receipt.get("contract_content_sha256")
        != contract["content_sha256_before_self_field"]
        or not isinstance(receipt.get("terminal_claim_state_content_sha256"), str)
        or not SHA256_RE.fullmatch(receipt["terminal_claim_state_content_sha256"])
        or receipt.get("documents") != PROTECTED_TARGET_ROWS
    ):
        raise RuntimeError("terminal receipt is not a valid V2 aggregate")
    for label in ("baseline", "candidate"):
        value = receipt.get(label)
        if not isinstance(value, dict) or set(value) != metric_keys:
            raise RuntimeError(f"terminal receipt {label} fields changed")
        metric = value["mean_document_token_logprob"]
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(metric):
            raise RuntimeError(f"terminal receipt {label} metric is invalid")
    delta = receipt.get("paired_delta")
    if not isinstance(delta, dict) or set(delta) != delta_keys:
        raise RuntimeError("terminal receipt paired delta fields changed")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in delta.values()
    ):
        raise RuntimeError("terminal receipt paired delta is invalid")
    bootstrap = receipt.get("bootstrap")
    if not isinstance(bootstrap, dict) or set(bootstrap) != bootstrap_keys:
        raise RuntimeError("terminal receipt bootstrap fields changed")
    if (
        bootstrap.get("samples") != 20000
        or bootstrap.get("seed") != 2026071701
    ):
        raise RuntimeError("terminal receipt bootstrap contract changed")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONTRACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_contract()
    if args.check:
        if _read_json(args.output) != value:
            raise RuntimeError("persisted V2 recipe contract differs from rebuild")
    else:
        _atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "path": str(args.output.resolve()),
        "content_sha256": value["content_sha256_before_self_field"],
        "protected_rows": value["roles"]["protected_terminal"]["rows"],
        "protected_semantic_content_persisted": False,
        "quarantine_passed": value["quarantine"]["passed"],
        "disjointness_passed": value["disjointness"]["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
