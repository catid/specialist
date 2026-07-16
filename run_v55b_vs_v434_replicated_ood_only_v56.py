#!/usr/bin/env python3
"""Run the sealed replicated V55B-versus-V434 OOD QA/prose gate."""

from __future__ import annotations

import argparse
import dis
import hashlib
import json
import math
import queue
import random
import stat
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import build_eval_v3 as eval_v3
import build_v434_train_identity_registry_v56 as registry_builder
import eggroll_es_train_panel_sampler_v13 as semantic
import run_eggroll_es_equal_unit_v38a as cleanup
import run_lora_topology_probe_v40a as topology
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44c as parser_fix
import run_matched_lora_candidate_eval_v45a as ood_first
import run_sealed_candidate_eval_v39a as metrics
import stage_v49d_adapters_vllm as v434_stage
import stage_v55b_candidate_vllm_v56 as v55b_stage
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
BASE_ARMS = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS = {
    "v434_equal": ("v434_equal_a", "v434_equal_b"),
    "v55b_candidate": ("v55b_candidate_a", "v55b_candidate_b"),
}
# Interleaving is intentional: each direct pair spans adjacent independent
# actors and both logical models occupy two of the four TP1 engines.
CANDIDATE_ARMS = (
    "v434_equal_a", "v55b_candidate_a",
    "v434_equal_b", "v55b_candidate_b",
)
ARMS = BASE_ARMS + CANDIDATE_ARMS
LOGICAL_CANDIDATES = tuple(LOGICAL_REPLICAS)
STAGED_BY_LOGICAL = {
    "v434_equal": v434_stage.OUTPUTS["v434_equal"],
    "v55b_candidate": v55b_stage.OUTPUT,
}
STAGE_MANIFEST_ARMS = {
    "v434_equal": "v434_equal",
    "v55b_candidate": v55b_stage.ARM,
}
STAGED_BY_ARM = {
    arm: STAGED_BY_LOGICAL[logical]
    for logical, replicas in LOGICAL_REPLICAS.items()
    for arm in replicas
}
ADAPTER_IDS = {arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS)}
STAGE_EXPECTED = {
    "v434_equal": {
        "weights": "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a",
        "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "manifest_file": "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813",
        "manifest_content": "ea328ada018e1c0d182d329d2a9cb81f8f0375aef93738f3b9c0a00f63c82da3",
        "transformed_identity": "f210bf05e7fe38481d0a7d9c641a7f902e575521b50e98bdc021bf11b49cb1c8",
    },
    "v55b_candidate": {
        "weights": "e30ab8173b4f979e6a5b4621908042cce66411246f3e242541ded945acaa7608",
        "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "manifest_file": "43d779ddf1c37d11c0016159739832c431ea223b8b5e9345e29ac3e3ed341483",
        "manifest_content": "a39359184a96b52008eabbce50f0f83ab1bda39513f6ed6c731dbdcdb093c6ce",
        "transformed_identity": "1638967e98ef1b677a651c49f6b97abb1656c4fce1b6776423aaccc3662e3cf5",
    },
}
OOD_INPUTS = {
    "ood_qa": {
        "path": str((ROOT / "data/ood_qa_v3.jsonl").resolve()),
        "file_sha256": "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d",
    },
    "ood_prose": {
        "path": str((ROOT / "data/ood_prose_v3.jsonl").resolve()),
        "file_sha256": "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57",
    },
}
REGISTRY = registry_builder.OUTPUT
REGISTRY_FILE_SHA256 = (
    "907886ccf689618cd58e68eff05e8212a29826a6c7655c7698632164f9ec5bc8"
)
REGISTRY_CONTENT_SHA256 = (
    "aea5b80183b2d98cf0dff37fd5f68cd6a8573901cf71c13a4558417c851cae8a"
)
TUNED_TABLE_FILE_CONTENT_SHA256 = (
    "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
)
TUNED_TABLE_RUNTIME_PROJECTION_SHA256 = (
    "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
)
BOOTSTRAP_SAMPLES = 20_000
GENERATION_SEED = core.GENERATION_SEED
PAIR_BOOTSTRAP_SEEDS = {
    "replica_a": 20_260_716_01,
    "replica_b": 20_260_716_02,
}
DIRECT_PAIRS = (
    ("replica_a", "v434_equal_a", "v55b_candidate_a"),
    ("replica_b", "v434_equal_b", "v55b_candidate_b"),
)
GPU_PHASES = ("v56_ood_qa", "v56_ood_prose")
EXPERIMENT = "v56_v55b_vs_v434_replicated_ood_only"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v56.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v56.jsonl").resolve()
FAILURE = (RUN_DIR / "failure_v56.json").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "v55b_vs_v434_replicated_ood_only_v56.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v55b_vs_v434_replicated_ood_only_v56.json"
).resolve()
BUILDER = (ROOT / "build_v55b_vs_v434_ood_preregistration_v56.py").resolve()
TESTS = (ROOT / "test_v55b_vs_v434_replicated_ood_only_v56.py").resolve()


def arm_wave_plan_v56():
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS)),
    )


def _source_seal(logical: str) -> dict:
    if logical == "v434_equal":
        return v434_stage.source_seal_v49d("v434_equal")
    if logical == "v55b_candidate":
        return v55b_stage.source_seal_v56()
    raise ValueError(logical)


def canonical_stage_binding_v56(logical: str) -> dict:
    if logical not in LOGICAL_CANDIDATES:
        raise ValueError(logical)
    directory = STAGED_BY_LOGICAL[logical]
    expected = STAGE_EXPECTED[logical]
    manifest_path = directory / "stage_manifest_v44a.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    observed = {
        "logical_candidate": logical,
        "directory": str(directory),
        "weights_file_sha256": core.file_sha256(
            directory / "adapter_model.safetensors"
        ),
        "adapter_config_file_sha256": core.file_sha256(
            directory / "adapter_config.json"
        ),
        "manifest_file_sha256": core.file_sha256(manifest_path),
        "manifest_content_sha256": content,
        "transformed_identity_sha256": value.get(
            "transformed_identity", {}
        ).get("sha256"),
        "tensor_count": value.get("transformed_identity", {}).get(
            "tensor_count"
        ),
        "elements": value.get("transformed_identity", {}).get("elements"),
        "tensor_bytes_preserved_exactly": value.get(
            "transformed_identity", {}
        ).get("all_tensor_bytes_preserved_exactly"),
    }
    if (
        content != core.canonical_sha256(compact)
        or value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or value.get("arm") != STAGE_MANIFEST_ARMS[logical]
        or observed["weights_file_sha256"] != expected["weights"]
        or observed["adapter_config_file_sha256"] != expected["config"]
        or observed["manifest_file_sha256"] != expected["manifest_file"]
        or content != expected["manifest_content"]
        or observed["transformed_identity_sha256"]
        != expected["transformed_identity"]
        or observed["tensor_count"] != 70
        or observed["elements"] != 4_528_128
        or observed["tensor_bytes_preserved_exactly"] is not True
        or value.get("dataset_or_evaluation_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
        or value.get("gpu_accessed") is not False
    ):
        raise RuntimeError(f"V56 {logical} staged adapter changed")
    return observed


def replica_stage_bindings_v56() -> dict:
    logical = {
        name: canonical_stage_binding_v56(name) for name in LOGICAL_CANDIDATES
    }
    return {
        arm: {
            **logical[name],
            "replica_arm": arm,
            "adapter_id": ADAPTER_IDS[arm],
        }
        for name, replicas in LOGICAL_REPLICAS.items()
        for arm in replicas
    }


def implementation_bindings_v56() -> dict:
    paths = {
        "runtime_v56": Path(__file__).resolve(),
        "builder_v56": BUILDER,
        "tests_v56": TESTS,
        "identity_registry_builder": Path(registry_builder.__file__).resolve(),
        "identity_registry": REGISTRY,
        "stage_v55b_runtime": Path(v55b_stage.__file__).resolve(),
        "stage_v434_runtime": Path(v434_stage.__file__).resolve(),
        "canonical_stage_runtime": Path(v55b_stage.prior.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "metric_runtime": Path(metrics.__file__).resolve(),
        "ood_gate_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(parser_fix.__file__).resolve(),
        "topology_runtime": Path(topology.__file__).resolve(),
        "cleanup_runtime": Path(cleanup.__file__).resolve(),
        "url_normalization_runtime": Path(eval_v3.__file__).resolve(),
        "semantic_rule_runtime": Path(semantic.__file__).resolve(),
        "resolver_runtime_v40c": Path(core.v40c.__file__).resolve(),
        "v434_source_weights": (
            v434_stage.SOURCES["v434_equal"] / "adapter_model.safetensors"
        ),
        "v434_source_config": (
            v434_stage.SOURCES["v434_equal"] / "adapter_config.json"
        ),
        "v434_source_report": v434_stage.REPORTS["v434_equal"],
        "v434_source_attempt": v434_stage.ATTEMPTS["v434_equal"],
        "v434_source_gpu_log": v434_stage.GPU_LOGS["v434_equal"],
        "v55b_source_weights": v55b_stage.SOURCE / "adapter_model.safetensors",
        "v55b_source_config": v55b_stage.SOURCE / "adapter_config.json",
        "v55b_source_evidence": v55b_stage.EVIDENCE,
        "v55b_source_p16_gate": v55b_stage.P16_GATE,
        "v55b_source_internal_report": v55b_stage.INTERNAL_REPORT,
        "v55b_source_wrapper_report": v55b_stage.WRAPPER_REPORT,
        "v434_staged_weights": (
            STAGED_BY_LOGICAL["v434_equal"] / "adapter_model.safetensors"
        ),
        "v434_staged_config": (
            STAGED_BY_LOGICAL["v434_equal"] / "adapter_config.json"
        ),
        "v434_stage_manifest": (
            STAGED_BY_LOGICAL["v434_equal"] / "stage_manifest_v44a.json"
        ),
        "v55b_staged_weights": (
            STAGED_BY_LOGICAL["v55b_candidate"] / "adapter_model.safetensors"
        ),
        "v55b_staged_config": (
            STAGED_BY_LOGICAL["v55b_candidate"] / "adapter_config.json"
        ),
        "v55b_stage_manifest": (
            STAGED_BY_LOGICAL["v55b_candidate"] / "stage_manifest_v44a.json"
        ),
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "gitignore": ROOT / ".gitignore",
    }
    result = {name: core.file_sha256(path) for name, path in paths.items()}
    if result["identity_registry"] != REGISTRY_FILE_SHA256:
        raise RuntimeError("V56 train identity registry changed")
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["source_seals"] = {
        arm: _source_seal(arm) for arm in LOGICAL_CANDIDATES
    }
    return result


def resolver_surface_preflight_v56() -> dict:
    """Audit the bound V40C resolver API without constructing a trainer."""
    v40c_instructions = tuple(dis.get_instructions(
        core.v40c.make_trainer_v40c
    ))
    v44a_names = set(core.make_trainer_v44a.__code__.co_names)
    attaches_resolver = any(
        item.opname == "STORE_ATTR" and item.argval == "_resolve"
        for item in v40c_instructions
    )
    v44a_uses_v40c = {
        "v40c", "make_trainer_v40c"
    }.issubset(v44a_names)
    if not attaches_resolver or not v44a_uses_v40c:
        raise RuntimeError("V56 current V40C trainer resolver surface changed")
    return {
        "checked_without_model_or_gpu_creation": True,
        "v44a_make_trainer_dispatches_to_v40c": True,
        "v40c_make_trainer_attaches_resolve": True,
        "resolver_runtime_file_sha256": core.file_sha256(
            Path(core.v40c.__file__).resolve()
        ),
    }


def tuned_table_projection_preflight_v56() -> dict:
    """Reproduce V40C's normalized runtime identity without touching a GPU."""
    value = json.loads(core.TUNED_FILE.read_text(encoding="utf-8"))
    file_content = core.canonical_sha256(value)
    value.pop("triton_version", None)
    runtime_projection = core.canonical_sha256({
        int(key): item for key, item in value.items()
    })
    if (
        file_content != TUNED_TABLE_FILE_CONTENT_SHA256
        or runtime_projection != TUNED_TABLE_RUNTIME_PROJECTION_SHA256
    ):
        raise RuntimeError("V56 normalized V40C tuned-table identity changed")
    return {
        "checked_without_model_or_gpu_creation": True,
        "normalization": "remove triton_version and cast table keys to int",
        "file_content_sha256": file_content,
        "runtime_projection_sha256": runtime_projection,
    }


def load_identity_registry_v56() -> dict:
    if core.file_sha256(REGISTRY) != REGISTRY_FILE_SHA256:
        raise RuntimeError("V56 train identity registry bytes changed")
    value = json.loads(REGISTRY.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    items = value.get("items")
    if (
        content != REGISTRY_CONTENT_SHA256
        or core.canonical_sha256(compact) != content
        or value.get("schema") != "v434-train-disjoint-identity-registry-v56"
        or value.get("aggregate", {}).get("rows") != 448
        or not isinstance(items, list)
        or len(items) != 448
        or value.get("content_minimization", {}).get("question_persisted")
        is not False
        or value.get("content_minimization", {}).get("answer_persisted")
        is not False
        or value.get("content_minimization", {}).get("document_text_persisted")
        is not False
        or value.get("access_receipt", {}).get(
            "ood_shadow_holdout_or_benchmark_semantics_opened"
        )
        is not False
    ):
        raise RuntimeError("V56 train identity registry contract changed")
    return value


def _raw_lineage_v56(label: str, line_number: int) -> str:
    source = OOD_INPUTS[label]["file_sha256"]
    return registry_builder.canonical_lineage_identity_v56(
        f"raw_eval_v3:{source}:{line_number}"
    )


def _ood_identity_records_v56(
    ood_qa_rows: list[dict], ood_prose_rows: list[dict]
) -> list[dict]:
    records = []
    for index, row in enumerate(ood_qa_rows, 1):
        required = ("item_id", "question", "answer", "url", "normalized_source_url")
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise RuntimeError("V56 OOD QA identity schema changed")
        normalized = eval_v3.normalize_source_url(row["url"])
        if normalized != row["normalized_source_url"]:
            raise RuntimeError("V56 OOD QA normalized URL changed")
        q_tokens = sorted(semantic._content_tokens(row["question"]))
        a_tokens = sorted(semantic._content_tokens(row["answer"]))
        records.append({
            "label": "ood_qa",
            "item_id_sha256": hashlib.sha256(row["item_id"].encode()).hexdigest(),
            "document_sha256": OOD_INPUTS["ood_qa"]["file_sha256"],
            "normalized_url": normalized,
            "raw_lineage_identity_sha256": _raw_lineage_v56("ood_qa", index),
            "semantic_question_tokens": q_tokens,
            "semantic_answer_tokens": a_tokens,
        })
    for index, row in enumerate(ood_prose_rows, 1):
        required = ("item_id", "title", "text", "url", "normalized_source_url")
        if any(not isinstance(row.get(key), str) or not row[key] for key in required):
            raise RuntimeError("V56 OOD prose identity schema changed")
        normalized = eval_v3.normalize_source_url(row["url"])
        if normalized != row["normalized_source_url"]:
            raise RuntimeError("V56 OOD prose normalized URL changed")
        q_tokens = sorted(semantic._content_tokens(row["title"]))
        a_tokens = sorted(semantic._content_tokens(row["text"]))
        records.append({
            "label": "ood_prose",
            "item_id_sha256": hashlib.sha256(row["item_id"].encode()).hexdigest(),
            "document_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "normalized_url": normalized,
            "raw_lineage_identity_sha256": _raw_lineage_v56("ood_prose", index),
            "semantic_question_tokens": q_tokens,
            "semantic_answer_tokens": a_tokens,
        })
    if not records:
        raise RuntimeError("V56 OOD identity inventory is empty")
    return records


def prove_train_ood_disjoint_v56(
    ood_qa_rows: list[dict], ood_prose_rows: list[dict], registry: dict
) -> dict:
    records = _ood_identity_records_v56(ood_qa_rows, ood_prose_rows)
    train_items = registry["items"]
    train_documents = {item["document_sha256"] for item in train_items}
    train_urls = {
        value for item in train_items for value in item["normalized_urls"]
    }
    train_lineages = {
        value for item in train_items
        for value in item["raw_lineage_identity_sha256s"]
    }
    ood_documents = {item["document_sha256"] for item in records}
    ood_urls = {item["normalized_url"] for item in records}
    ood_lineages = {item["raw_lineage_identity_sha256"] for item in records}
    semantic_intersections = set()
    for ood in records:
        ood_features = (
            frozenset(ood["semantic_question_tokens"]),
            frozenset(ood["semantic_answer_tokens"]),
        )
        for train in train_items:
            train_features = (
                frozenset(train["semantic_question_tokens"]),
                frozenset(train["semantic_answer_tokens"]),
            )
            if semantic._semantic_match(ood_features, train_features):
                semantic_intersections.add(train["semantic_cluster_sha256"])
    intersections = {
        "document_sha256": sorted(train_documents & ood_documents),
        "normalized_url": sorted(train_urls & ood_urls),
        "raw_lineage": sorted(train_lineages & ood_lineages),
        "semantic_cluster": sorted(semantic_intersections),
    }
    counts = {key: len(value) for key, value in intersections.items()}
    safe_records = [{
        "label": item["label"],
        "item_id_sha256": item["item_id_sha256"],
        "document_sha256": item["document_sha256"],
        "normalized_url_sha256": hashlib.sha256(
            item["normalized_url"].encode()
        ).hexdigest(),
        "raw_lineage_identity_sha256": item["raw_lineage_identity_sha256"],
        "semantic_feature_sha256": core.canonical_sha256({
            "question_tokens": item["semantic_question_tokens"],
            "answer_tokens": item["semantic_answer_tokens"],
        }),
    } for item in records]
    proof = {
        "schema": "v434-train-v56-ood-four-domain-disjointness",
        "checked_before_model_creation": True,
        "train_rows_opened_by_runtime": 0,
        "identity_registry": {
            "path": str(REGISTRY),
            "file_sha256": REGISTRY_FILE_SHA256,
            "content_sha256": REGISTRY_CONTENT_SHA256,
            "train_rows": len(train_items),
        },
        "ood_rows": {
            "ood_qa": len(ood_qa_rows),
            "ood_prose": len(ood_prose_rows),
        },
        "intersection_counts": counts,
        "all_four_identity_domains_disjoint": not any(counts.values()),
        "ood_identity_record_sha256": core.canonical_sha256(safe_records),
        "semantic_rule": (
            "pairwise frozen V13 lexical thresholds against stored train feature "
            "sets; a match yields the exact train semantic-cluster identity"
        ),
        "document_identity_definitions": {
            "ood_qa": (
                "the exact sealed OOD-QA JSONL file SHA256 is the canonical "
                "source-document identity for every QA row because eval-v3 "
                "rows do not carry document_sha256"
            ),
            "ood_prose": "SHA256 of the exact prose text field",
        },
    }
    if not proof["all_four_identity_domains_disjoint"]:
        raise RuntimeError(
            "V56 train/OOD identity intersection before model creation: "
            + json.dumps(counts, sort_keys=True)
        )
    return proof


def _paired_qa_bootstrap_v56(
    reference_rows: list[dict], candidate_rows: list[dict], *, seed: int
) -> dict:
    if len(reference_rows) != len(candidate_rows) or not reference_rows:
        raise RuntimeError("V56 paired OOD QA coverage changed")
    deltas = {
        "generated_reward_f1": [],
        "generated_exact": [],
        "generated_nonzero": [],
        "answer_logprob": [],
    }
    for reference, candidate in zip(reference_rows, candidate_rows, strict=True):
        if reference["item_sha256"] != candidate["item_sha256"]:
            raise RuntimeError("V56 paired OOD QA identity changed")
        reference_reward = float(reference["reward"])
        candidate_reward = float(candidate["reward"])
        deltas["generated_reward_f1"].append(
            candidate_reward - reference_reward
        )
        deltas["generated_exact"].append(
            int(candidate["format"] == "exact")
            - int(reference["format"] == "exact")
        )
        deltas["generated_nonzero"].append(
            int(candidate_reward > 0.0) - int(reference_reward > 0.0)
        )
        deltas["answer_logprob"].append(
            float(candidate["teacher"]["mean_answer_token_logprob"])
            - float(reference["teacher"]["mean_answer_token_logprob"])
        )
    rng = random.Random(seed)
    samples = {key: [] for key in deltas}
    count = len(reference_rows)
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(count) for _ in range(count)]
        for key, values in deltas.items():
            samples[key].append(
                math.fsum(values[index] for index in indices) / count
            )
    result = {}
    for key, values in deltas.items():
        interval = [
            base.linear_percentile(samples[key], 0.025),
            base.linear_percentile(samples[key], 0.975),
        ]
        result[key] = {
            "paired_item_mean_delta": math.fsum(values) / count,
            "paired_item_bootstrap_95_ci": interval,
            "bootstrap_lcb_noninferiority_passed": interval[0] >= 0.0,
        }
    result["bootstrap"] = {
        "unit": "ood_qa_item",
        "item_count": count,
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "percentiles": [0.025, 0.975],
        "lower_bound_minimum": 0.0,
    }
    return result


def _direct_replica_gate_v56(
    ood_qa: dict, prose_details: dict, raw_sink: dict,
    *, pair: str, reference: str, candidate: str,
) -> dict:
    reference_qa = ood_qa[reference]
    candidate_qa = ood_qa[candidate]
    qa_points = {
        "generated_reward_delta": (
            candidate_qa["generated_row_mean_reward"]
            - reference_qa["generated_row_mean_reward"]
        ),
        "generated_f1_delta": (
            candidate_qa["generated_equal_unit_mean_reward"]
            - reference_qa["generated_equal_unit_mean_reward"]
        ),
        "generated_exact_count_delta": (
            candidate_qa["generated_exact_count"]
            - reference_qa["generated_exact_count"]
        ),
        "generated_nonzero_count_delta": (
            candidate_qa["generated_nonzero_count"]
            - reference_qa["generated_nonzero_count"]
        ),
        "answer_logprob_delta": (
            candidate_qa["teacher_forced_equal_unit_mean_answer_logprob"]
            - reference_qa["teacher_forced_equal_unit_mean_answer_logprob"]
        ),
    }
    qa_point_checks = {key + "_noninferior": value >= 0.0
                       for key, value in qa_points.items()}
    paired = _paired_qa_bootstrap_v56(
        raw_sink["ood_qa"][reference], raw_sink["ood_qa"][candidate],
        seed=PAIR_BOOTSTRAP_SEEDS[pair],
    )
    paired_checks = {
        key + "_bootstrap_lcb_noninferior":
        value["bootstrap_lcb_noninferiority_passed"]
        for key, value in paired.items() if key != "bootstrap"
    }
    prose = metrics.prose_gate(
        prose_details[reference], prose_details[candidate]
    )
    reference_logprob = float(prose_details[reference]["mean_token_logprob"])
    candidate_logprob = float(prose_details[candidate]["mean_token_logprob"])
    prose_points = {
        "mean_token_logprob_delta": candidate_logprob - reference_logprob,
        "perplexity_delta": (
            math.exp(-candidate_logprob) - math.exp(-reference_logprob)
        ),
    }
    prose_checks = {
        "token_logprob_point_noninferior":
            prose_points["mean_token_logprob_delta"] >= 0.0,
        "perplexity_point_noninferior": prose_points["perplexity_delta"] <= 0.0,
        "paired_document_bootstrap_lcb_noninferior":
            prose["bootstrap_lcb_non_degradation_passed"],
    }
    reference_counters = reference_qa["protocol_leak_counters"]
    candidate_counters = candidate_qa["protocol_leak_counters"]
    protocol_checks = {
        key: candidate_counters[key] <= reference_counters[key]
        for key in reference_counters
    }
    eligible = all(qa_point_checks.values()) and all(
        paired_checks.values()
    ) and all(prose_checks.values()) and all(protocol_checks.values())
    return {
        "pair": pair,
        "reference_arm": reference,
        "candidate_arm": candidate,
        "qa_point_deltas": qa_points,
        "qa_point_checks": qa_point_checks,
        "qa_paired_bootstrap": paired,
        "qa_paired_bootstrap_checks": paired_checks,
        "prose_point_deltas": prose_points,
        "prose_gate": prose,
        "prose_checks": prose_checks,
        "protocol_reference": reference_counters,
        "protocol_candidate": candidate_counters,
        "protocol_no_increase_checks": protocol_checks,
        "independently_directly_noninferior": eligible,
    }


def direct_gate_table_v56(ood_qa, prose_details, raw_sink) -> dict:
    replicas = [
        _direct_replica_gate_v56(
            ood_qa, prose_details, raw_sink,
            pair=pair, reference=reference, candidate=candidate,
        )
        for pair, reference, candidate in DIRECT_PAIRS
    ]
    return {
        "comparison": "fixed V55B candidate directly versus exact V434 master",
        "replicas": replicas,
        "both_candidate_replicas_independently_directly_noninferior": all(
            item["independently_directly_noninferior"] for item in replicas
        ),
        "promotion_authorized": False,
    }


def assert_four_raw_base_controls_v56(table: dict, raw: dict, label: str) -> dict:
    aggregate_hashes = {
        arm: core.canonical_sha256(table[arm]) for arm in BASE_ARMS
    }
    raw_hashes = {arm: core.canonical_sha256(raw[arm]) for arm in BASE_ARMS}
    if len(set(aggregate_hashes.values())) != 1 or len(set(raw_hashes.values())) != 1:
        raise RuntimeError(f"V56 four raw-base determinism controls failed: {label}")
    return {
        "label": label,
        "all_four_aggregate_outputs_exact": True,
        "all_four_raw_outputs_exact": True,
        "aggregate_sha256": aggregate_hashes["base_a"],
        "raw_sha256": raw_hashes["base_a"],
    }


def summarize_ood_gpu_v56(
    path: Path, expected_pids: dict[int, int]
) -> dict:
    """Require attributed positive activity on all GPUs in both OOD phases."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if set(expected_pids) != set(range(4)):
        raise RuntimeError("V56 physical GPU PID coverage changed")
    result = {}
    for gpu in range(4):
        selected = [row for row in rows if row.get("gpu") == gpu]
        if not selected or any(row["foreign_compute_pids"] for row in selected):
            raise RuntimeError("V56 GPU monitor coverage or exclusivity failed")
        if any(row["expected_pid"] != expected_pids[gpu] for row in selected):
            raise RuntimeError("V56 GPU PID attribution changed")
        phases = {}
        for label in GPU_PHASES:
            phase_rows = [row for row in selected if row["phase"] == label]
            resident = [
                row for row in phase_rows
                if expected_pids[gpu] in row["compute_pids"]
            ]
            positive = [
                row for row in resident if row["utilization_percent"] > 0
            ]
            if not resident or not positive:
                raise RuntimeError(f"V56 GPU {gpu} inactive in {label}")
            phases[label] = {
                "samples": len(phase_rows),
                "resident_samples": len(resident),
                "positive_samples": len(positive),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in resident
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in resident
                ),
                "mean_resident_utilization_percent": math.fsum(
                    row["utilization_percent"] for row in resident
                ) / len(resident),
            }
        result[str(gpu)] = {
            "expected_pid": expected_pids[gpu], "phases": phases,
        }
    return {
        "required_phase_labels_exact": list(GPU_PHASES),
        "all_four_attributed_positive_each_ood_phase": True,
        "shadow_phase_required_or_opened": False,
        "by_gpu": result,
    }


@contextmanager
def patched_arm_globals_v56():
    saved = (
        core.BASE_ARMS, core.CANDIDATE_ARMS, core.ARMS, core.STAGED_BY_ARM,
        core.ADAPTER_IDS_V44A, core.ENGINE_INDEX_BY_ARM_V44A,
        core.arm_wave_plan_v44a, core.EXPERIMENT, core.RUN_DIR,
    )
    core.BASE_ARMS = BASE_ARMS
    core.CANDIDATE_ARMS = CANDIDATE_ARMS
    core.ARMS = ARMS
    core.STAGED_BY_ARM = STAGED_BY_ARM
    core.ADAPTER_IDS_V44A = ADAPTER_IDS
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: engine for wave in arm_wave_plan_v56() for arm, engine in wave
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v56
    core.EXPERIMENT = EXPERIMENT
    core.RUN_DIR = RUN_DIR
    try:
        yield
    finally:
        (
            core.BASE_ARMS, core.CANDIDATE_ARMS, core.ARMS, core.STAGED_BY_ARM,
            core.ADAPTER_IDS_V44A, core.ENGINE_INDEX_BY_ARM_V44A,
            core.arm_wave_plan_v44a, core.EXPERIMENT, core.RUN_DIR,
        ) = saved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return result


def load_preregistration_v56(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V56 OOD preregistration bytes changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256(compact)
        or value.get("schema") != "v55b-vs-v434-replicated-ood-only-v56"
        or value.get("status") != "preregistered_before_single_ood_access"
        or value.get("evaluation_launch_authorized") is not True
        or value.get("training_access_authorized") is not False
        or value.get("shadow_access_authorized") is not False
        or value.get("terminal_holdout_access_authorized") is not False
        or value.get("promotion_authorized") is not False
        or value.get("single_access_inputs") != OOD_INPUTS
        or value.get("arms") != list(ARMS)
        or value.get("logical_candidate_replicas") != {
            key: list(item) for key, item in LOGICAL_REPLICAS.items()
        }
        or value.get("staged_adapters") != replica_stage_bindings_v56()
        or value.get("implementation_bindings") != implementation_bindings_v56()
        or value.get("train_identity_registry", {}).get("file_sha256")
        != REGISTRY_FILE_SHA256
        or value.get("train_identity_registry", {}).get("content_sha256")
        != REGISTRY_CONTENT_SHA256
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != TUNED_TABLE_RUNTIME_PROJECTION_SHA256
        or value.get("runtime", {}).get("tuned_table_file_content_sha256")
        != TUNED_TABLE_FILE_CONTENT_SHA256
        or value.get("runtime", {}).get("trainer_resolver_capability_preflight")
        != resolver_surface_preflight_v56()
        or value.get("runtime", {}).get("tuned_table_projection_preflight")
        != tuned_table_projection_preflight_v56()
    ):
        raise RuntimeError("V56 OOD preregistration contract changed")
    core._forbid_holdout_v44a(item["path"] for item in OOD_INPUTS.values())
    return value


class Phase:
    value = "setup"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v56(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "single_access_labels": sorted(prereg["single_access_inputs"]),
            "train_rows_opened": 0,
            "protected_semantic_access_count": 0,
            "shadow_opened": False,
            "terminal_holdout_opened": False,
            "gpu_accessed": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("V56 requires fresh immutable artifact paths")
    if subprocess.run(
        ["git", "check-ignore", "-q", str(RAW)], cwd=ROOT,
        check=False, capture_output=True,
    ).returncode != 0:
        raise RuntimeError("V56 raw protected output path is not git-ignored")
    registry = load_identity_registry_v56()
    resolver_preflight = resolver_surface_preflight_v56()
    tuned_table_preflight = tuned_table_projection_preflight_v56()
    ood_first.environment.environment_bindings_v44b()
    preflight = topology.gpu_preflight()
    attempt = core.self_hashed({
        "schema": "v56-v55b-v434-ood-only-attempt",
        "status": "launching",
        "phase": "before_model_or_ood_semantic_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "train_rows_opened": 0,
        "protected_semantic_access_count": 0,
        "shadow_opened": False,
        "terminal_holdout_opened": False,
        "preflight": preflight,
        "trainer_resolver_preflight": resolver_preflight,
        "tuned_table_projection_preflight": tuned_table_preflight,
    })
    core.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = firewall = None
    disjointness_passed = False
    model_created = False
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = Phase()
    raw_sink = {"schema": "v56-v55b-v434-ood-only-raw-local"}
    started = time.monotonic()
    try:
        saved_inputs = core.PROTECTED_INPUTS_V44A
        core.PROTECTED_INPUTS_V44A = OOD_INPUTS
        try:
            firewall = core.SingleSemanticAccessV44A(OOD_INPUTS)
            ood_qa_rows = firewall.jsonl("ood_qa")
            ood_prose_rows = firewall.jsonl("ood_prose")
            disjointness = prove_train_ood_disjoint_v56(
                ood_qa_rows, ood_prose_rows, registry
            )
            disjointness_passed = True
            ood_qa_bundle, _qa_schema = parser_fix.ood_qa_bundle_v44c(
                ood_qa_rows
            )
            _prose_schema = parser_fix.ood_prose_preflight_v44c(ood_prose_rows)
        finally:
            core.PROTECTED_INPUTS_V44A = saved_inputs
        if set(firewall.receipts) != set(OOD_INPUTS):
            raise RuntimeError("V56 source-faithful single-access coverage changed")

        base.set_seed(GENERATION_SEED)
        with patched_arm_globals_v56():
            trainer, saved = core.make_trainer_v44a(prereg)
            model_created = True
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote() for engine in trainer.engines
            ])
            worker_ids = topology._rpc_all(trainer, "runtime_identity_v40a")
            pid_map = core.actor_pid_map_v44a(actor_ids, worker_ids)
            monitor = threading.Thread(
                target=metrics.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
            )
            monitor.start()
            phase.value = "v56_ood_qa"
            ood_qa = core.evaluate_qa_v44a(
                trainer, ood_qa_bundle, raw_sink, "ood_qa"
            )
            phase.value = "v56_ood_prose"
            ood_prose, prose_details = core.evaluate_prose_v44a(
                trainer, ood_prose_rows, raw_sink
            )
        raw_sink["single_access_receipts"] = firewall.receipts
        raw_sink["train_ood_disjointness"] = disjointness
        core.atomic_json(RAW, raw_sink, mode=0o600)
        if stat.S_IMODE(RAW.stat().st_mode) != 0o600:
            raise RuntimeError("V56 raw protected output mode changed")
        raw_sha = core.file_sha256(RAW)

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("V56 GPU monitor failed")
        base_equivalence = {
            "ood_qa": assert_four_raw_base_controls_v56(
                ood_qa, raw_sink["ood_qa"], "ood_qa"
            ),
            "ood_prose": assert_four_raw_base_controls_v56(
                ood_prose, raw_sink["ood_prose"], "ood_prose"
            ),
        }
        direct = direct_gate_table_v56(ood_qa, prose_details, raw_sink)
        gpu = summarize_ood_gpu_v56(GPU_LOG, pid_map)
        closed = cleanup.strict_close_trainer_v38a(trainer)
        trainer = None
        if (
            closed.get("engine_kill_count") != 4
            or closed.get("placement_group_remove_count") != 4
            or closed.get("all_four_gcs_states_removed") is not True
        ):
            raise RuntimeError("V56 exact four-engine cleanup changed")
        import ray
        ray.shutdown()
        idle = cleanup.wait_for_gpu_idle()
        report = core.self_hashed({
            "schema": "v56-v55b-v434-replicated-ood-only-aggregate",
            "status": "complete_ood_only_no_shadow_holdout_or_promotion",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "arms": list(ARMS),
            "fixed_waves": [
                [{"arm": arm, "engine_index": engine} for arm, engine in wave]
                for wave in arm_wave_plan_v56()
            ],
            "staged_adapters": prereg["staged_adapters"],
            "actor_identities": actor_ids,
            "worker_identities": worker_ids,
            "physical_gpu_pid_map": pid_map,
            "train_ood_disjointness": disjointness,
            "base_determinism_controls": base_equivalence,
            "ood_qa": ood_qa,
            "ood_prose": ood_prose,
            "direct_v55b_vs_v434_gate": direct,
            "candidate_ood_eligible": direct[
                "both_candidate_replicas_independently_directly_noninferior"
            ],
            "single_access_receipts": firewall.receipts,
            "gpu_activity": gpu,
            "placement_group_cleanup": closed,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "trainer_resolver_preflight": resolver_preflight,
            "tuned_table_projection_preflight": tuned_table_preflight,
            "raw_local_artifact": {
                "path": str(RAW),
                "file_sha256": raw_sha,
                "mode": "0600",
                "git_eligible": False,
            },
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": core.file_sha256(GPU_LOG),
            },
            "train_rows_opened": 0,
            "protected_semantic_access_count": 2,
            "shadow_opened": False,
            "terminal_holdout_opened": False,
            "selection_or_promotion_authorized": False,
        })
        core.atomic_json(REPORT, report)
        complete = dict(attempt)
        complete.pop("content_sha256_before_self_field", None)
        complete.update({
            "status": "complete",
            "phase": "v56_ood_aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protected_semantic_access_count": 2,
            "candidate_ood_eligible": report["candidate_ood_eligible"],
            "report": str(REPORT),
            "report_sha256": core.file_sha256(REPORT),
        })
        core.atomic_json(
            ATTEMPT.with_suffix(".complete.json"), core.self_hashed(complete)
        )
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": core.file_sha256(REPORT),
            "candidate_ood_eligible": report["candidate_ood_eligible"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = core.self_hashed({
            "schema": "v56-v55b-v434-ood-only-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "train_rows_opened": 0,
            "protected_semantic_access_count": (
                0 if firewall is None else len(firewall.receipts)
            ),
            "protected_semantic_access_labels": (
                [] if firewall is None else sorted(firewall.receipts)
            ),
            "disjointness_passed_before_model_creation": disjointness_passed,
            "model_created": model_created,
            "model_creation_preceded_disjointness_pass": (
                model_created and not disjointness_passed
            ),
            "shadow_opened": False,
            "terminal_holdout_opened": False,
            "promotion_authorized": False,
        })
        core.atomic_json(FAILURE, failure)
        raise
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except Exception:
                pass
        if saved is not None:
            topology.EXPERIMENT, topology.RUN_DIR = saved
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
