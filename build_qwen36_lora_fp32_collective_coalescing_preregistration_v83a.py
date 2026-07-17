#!/usr/bin/env python3
"""Build the CPU-only V83A exact-FP32 collective coalescing preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import eggroll_es_fp32_collective_coalescing_v83a as contract


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_fp32_collective_coalescing_v83a.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_lora_fp32_collective_coalescing_v83a_cpu_evidence_20260717.md"
)

SCHEMA_V83A = "qwen36-lora-fp32-collective-coalescing-preregistration-v83a"
CREATED_AT_UTC_V83A = "2026-07-17T00:00:00+00:00"

HELPER = ROOT / "eggroll_es_fp32_collective_coalescing_v83a.py"
HELPER_SHA256 = (
    "0c1ed3a7e451da20e76d3b0ea971771b4e7acbc17d6e5f34b5d54efe4a7bc0d6"
)
FOCUSED_TEST = ROOT / "test_eggroll_es_fp32_collective_coalescing_v83a.py"
FOCUSED_TEST_SHA256 = (
    "6335c181c60644ecdc301c363e592ccaca85aed285a9b9d94c951ba4dd2aa803"
)
BUILDER_TEST = ROOT / (
    "test_build_qwen36_lora_fp32_collective_coalescing_preregistration_v83a.py"
)
BUILDER_TEST_SHA256 = (
    "4a8d41badf9de5bc1a649cb3ea6b2afb0e518755d100ce5205f4ef758e8f8a02"
)

WORKER_V72 = ROOT / "eggroll_es_worker_lora_v72.py"
WORKER_V72_SHA256 = (
    "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2"
)
FUSED_RUNTIME_V72 = ROOT / "eggroll_es_fused_structured_runtime_v72.py"
FUSED_RUNTIME_V72_SHA256 = (
    "357607f3c16b071f67d2bc3adb0317bbbd29f31f7e1db0cf1aa3030ac997df6e"
)
V82B_ORACLE = ROOT / "eggroll_es_collective_compression_v82b.py"
V82B_ORACLE_SHA256 = (
    "4fef02306ef6519a328ba024e4c4d5eec568695d7a08a80ef0bec8796ffdeb35"
)
V82B_BUILDER = ROOT / (
    "build_qwen36_lora_collective_compression_preregistration_v82b.py"
)
V82B_BUILDER_SHA256 = (
    "aa868ea06d35d8d28e722de8c76a70169d04f43e422898bb45b45bdd2da9fd6c"
)
V82B_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_lora_collective_compression_v82b.json"
)
V82B_PREREG_FILE_SHA256 = (
    "c4417708c035198959647a1b3db21dfecf7709b051658d918500793264378e50"
)
V82B_PREREG_CONTENT_SHA256 = (
    "3efcc3a59652a6dbef73a5e0a963e4a86628992ef371556314c73d61245983f4"
)


def file_sha256_v83a(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_v83a(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json_v83a(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicates
    )
    _require_v83a(isinstance(value, dict), f"JSON object required: {path}")
    return value


def inspect_bound_sources_v83a() -> dict[str, Any]:
    bindings = (
        (HELPER, HELPER_SHA256, "V83A helper"),
        (FOCUSED_TEST, FOCUSED_TEST_SHA256, "V83A focused test"),
        (BUILDER_TEST, BUILDER_TEST_SHA256, "V83A builder test"),
        (WORKER_V72, WORKER_V72_SHA256, "accepted V72 worker"),
        (FUSED_RUNTIME_V72, FUSED_RUNTIME_V72_SHA256, "accepted V72 runtime"),
        (V82B_ORACLE, V82B_ORACLE_SHA256, "accepted V82B oracle"),
        (V82B_BUILDER, V82B_BUILDER_SHA256, "accepted V82B builder"),
        (V82B_PREREG, V82B_PREREG_FILE_SHA256, "accepted V82B preregistration"),
    )
    rows = []
    for path, expected, label in bindings:
        _require_v83a(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v83a(path) == expected,
            f"sealed {label} changed",
        )
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "file_sha256": expected,
            }
        )

    v72_text = WORKER_V72.read_text(encoding="utf-8")
    runtime_text = FUSED_RUNTIME_V72.read_text(encoding="utf-8")
    helper_text = HELPER.read_text(encoding="utf-8")
    _require_v83a(
        "for key, master in self._v41_master.items():" in v72_text
        and "self.inter_pg.all_reduce(" in v72_text
        and "accumulator, out_tensor=accumulator, stream=stream" in v72_text
        and "EXPECTED_SOURCE_TENSORS_V72 = 70" in runtime_text
        and "EXPECTED_SOURCE_ELEMENTS_V72 = 4_528_128" in runtime_text
        and "self.communicator.all_reduce(" in helper_text
        and "staging, out_tensor=staging, stream=self.stream" in helper_text,
        "V83A bound source semantics changed",
    )
    return {
        "files": rows,
        "accepted_v72_call": (
            "self.inter_pg.all_reduce(accumulator, "
            "out_tensor=accumulator, stream=stream)"
        ),
        "prospective_v83a_call": (
            "self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)"
        ),
        "accepted_v72_mutated": False,
        "accepted_v82b_mutated": False,
    }


def inspect_v82b_scope_v83a() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = _load_json_v83a(V82B_PREREG)
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require_v83a(
        claimed == V82B_PREREG_CONTENT_SHA256
        and contract.canonical_sha256_v83a(body) == V82B_PREREG_CONTENT_SHA256,
        "V82B preregistration self identity changed",
    )
    master = value.get("canonical_lora_update_scope", {}).get(
        "canonical_master", {}
    )
    records = master.get("ordered_shape_manifest")
    _require_v83a(
        value.get("schema") == "qwen36-lora-collective-compression-correction-v82b"
        and master.get("tensor_count") == contract.TENSOR_COUNT_V83A
        and master.get("elements") == contract.TOTAL_ELEMENTS_V83A
        and master.get("bytes") == contract.TOTAL_BYTES_V83A
        and master.get("ordered_shape_manifest_sha256")
        == contract.EXPECTED_SOURCE_MANIFEST_SHA256_V83A,
        "V82B canonical LoRA scope changed",
    )
    canonical = contract._source_records_v83a(records)
    return value, canonical


def build_preregistration_v83a() -> dict[str, Any]:
    bindings = inspect_bound_sources_v83a()
    v82b, records = inspect_v82b_scope_v83a()
    plans = {
        choice: contract.build_bucket_plan_v83a(records, choice)
        for choice, _capacity in contract.BUCKET_CHOICES_V83A
    }
    _require_v83a(
        [plans[name]["coalesced_collective_calls"] for name, _ in contract.BUCKET_CHOICES_V83A]
        == [1, 3, 5, 10]
        and all(
            plan["total_elements"] == contract.TOTAL_ELEMENTS_V83A
            and plan["total_bytes"] == contract.TOTAL_BYTES_V83A
            for plan in plans.values()
        ),
        "V83A deterministic plan identities changed",
    )
    flat = plans["flat_all_18112512b"]
    body = {
        "schema": SCHEMA_V83A,
        "status": (
            "prospective_cpu_only_exact_fp32_coalescing_live_arm_blocked_"
            "pending_v73d_call_latency_materiality"
        ),
        "created_at_utc": CREATED_AT_UTC_V83A,
        "bead": "specialist-0j5.36",
        "authority": {
            "source_and_synthetic_cpu_only": True,
            "dataset_training_examples_or_site_corpus_opened": False,
            "evaluation_dev_ood_holdout_shadow_or_probe_opened": False,
            "model_ray_or_gpu_launched": False,
            "live_pynccl_executed": False,
            "adapter_update_or_checkpoint_written": False,
            "bucket_choice_selected": False,
            "live_arm_authorized": False,
            "training_hpo_quality_or_promotion_authorized": False,
        },
        "purpose": (
            "Prospectively replace up to 70 exact FP32 PyNccl-style calls with "
            "a bounded number of exact FP32 calls if V73D later proves call "
            "latency material, without changing payload dtype or element count."
        ),
        "source_bindings": bindings,
        "canonical_lora_surface": {
            "v82b_preregistration_content_sha256": V82B_PREREG_CONTENT_SHA256,
            "v82b_scope_content_sha256": v82b["corrected_byte_accounting"][
                "content_sha256"
            ],
            "ordered_shape_manifest_sha256": (
                contract.EXPECTED_SOURCE_MANIFEST_SHA256_V83A
            ),
            "ordered_key_sha256": contract.EXPECTED_ORDERED_KEY_SHA256_V83A,
            "tensor_count": contract.TENSOR_COUNT_V83A,
            "module_count": contract.MODULE_COUNT_V83A,
            "elements": contract.TOTAL_ELEMENTS_V83A,
            "bytes": contract.TOTAL_BYTES_V83A,
            "dtype": "float32",
            "ordered_records": records,
        },
        "deterministic_bucket_plans": plans,
        "selection": {
            "selected_choice": None,
            "selection_authorized": False,
            "one_flat_maximum_choice": "flat_all_18112512b",
            "smaller_bounded_alternatives": [
                "bounded_8mib",
                "bounded_4mib",
                "bounded_2mib",
            ],
            "flat_collective_calls": flat["coalesced_collective_calls"],
            "flat_staging_bytes": flat["maximum_bucket_bytes"],
            "native_collective_calls": contract.NATIVE_COLLECTIVE_CALLS_V83A,
            "decision_rule": (
                "Do not select or launch unless separately accepted V73D "
                "evidence ranks canonical collective-call latency material; "
                "then benchmark all registered choices against unchanged V72."
            ),
        },
        "numerical_contract": {
            "payload_dtype": "float32",
            "payload_elements_per_actor": contract.TOTAL_ELEMENTS_V83A,
            "payload_bytes_per_actor": contract.TOTAL_BYTES_V83A,
            "compression_or_quantization_used": False,
            "rank_local_tensor_arithmetic_order_changed": False,
            "synthetic_same_rank_order_bitwise_identity_proven": True,
            "live_pynccl_bitwise_identity_across_message_sizes_proven": False,
            "reason_live_bitwise_is_unproven": (
                "A changed message size may select a different live collective "
                "algorithm or reduction tree even though dtype and per-element "
                "rank inputs are unchanged."
            ),
            "live_acceptance_requires_native_vs_coalesced_candidate_and_restore_gate": True,
        },
        "runtime_semantics": {
            "actual_future_communicator": "self.inter_pg",
            "actual_future_method": "all_reduce",
            "exact_future_call_expression": (
                "self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)"
            ),
            "fill_collective_scale_and_d2h_use_one_ordered_stream": True,
            "event_recorded_after_collective_and_scale": True,
            "event_synchronized_before_host_candidate_consumption": True,
            "event_synchronized_before_staging_release_or_reuse": True,
            "return_must_alias_in_place_fp32_input": True,
            "partial_candidate_commit_forbidden": True,
            "failure_preserves_exact_original_or_terminally_poisons": True,
            "stale_transaction_retry_forbidden": True,
            "outer_v72_accept_commit_finalize_boundaries_still_required": True,
        },
        "memory_and_bandwidth": {
            "network_payload_bytes_change": 0,
            "nominal_ring_bytes_change": 0,
            "flat_sequential_staging_bytes": flat["maximum_bucket_bytes"],
            "native_maximum_accumulator_bytes": (
                contract.NATIVE_MAX_ACCUMULATOR_BYTES_V83A
            ),
            "flat_incremental_staging_bytes_versus_native_maximum": (
                flat["byte_accounting"][
                    "incremental_gpu_staging_bytes_versus_native_maximum"
                ]
            ),
            "materialized_pack_read_plus_write_hbm_bytes": 36_225_024,
            "materialized_gpu_unpack_read_plus_write_hbm_bytes": 36_225_024,
            "conservative_pack_plus_gpu_unpack_hbm_bytes": 72_450_048,
            "unchanged_d2h_source_hbm_read_bytes": 18_112_512,
            "preferred_future_implementation": (
                "directly_generate_rank_local_updates_into ordered bucket views "
                "and copy reduced slices to CPU without GPU unpack"
            ),
            "direct_fill_hbm_saving_measured": False,
            "nccl_internal_hbm_measured": False,
        },
        "synthetic_evidence": {
            "command": (
                ".venv/bin/pytest -q "
                "test_eggroll_es_fp32_collective_coalescing_v83a.py "
                "test_build_qwen36_lora_fp32_collective_coalescing_"
                "preregistration_v83a.py"
            ),
            "observed_result": "22 passed",
            "gpu_or_model_used": False,
            "fake_world_size": 4,
            "all_four_fake_ranks_exercised": True,
            "proves": [
                "ordered 70-key shape and offset coverage",
                "no gaps, overlaps, duplicate keys, or source/master alias",
                "exact same-rank-order native versus coalesced update identity",
                "exact candidate identity and exact provisional restore",
                "in-place call, stream, event, and failure ordering",
                "partial failure poisoning and stale retry rejection",
                "payload, staging, ring, pack, unpack, and HBM formulas",
            ],
            "does_not_prove": [
                "live PyNccl performance",
                "live PyNccl bitwise identity across message sizes",
                "GPU peak memory or HBM traffic",
                "training quality, HPO fitness, OOD, or promotion",
            ],
        },
        "future_gate": {
            "prerequisite": "accepted V73D canonical exact-phase profile",
            "required_v73d_finding": (
                "collective call latency is material to update execution"
            ),
            "v73d_receipt_read_or_bound_by_v83a": False,
            "gpu_launch_authorized_now": False,
            "if_prerequisite_passes": (
                "build a separate live preregistration and paired unchanged-V72 "
                "benchmark for all four bucket choices on the actual communicator"
            ),
            "speed_vram_or_bandwidth_improvement_claimed_now": False,
            "bead_remains_open": True,
        },
    }
    return {
        **body,
        "content_sha256_before_self_field": contract.canonical_sha256_v83a(body),
    }


def validate_preregistration_v83a(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = build_preregistration_v83a()
    _require_v83a(dict(value) == expected, "V83A preregistration changed")
    return copy.deepcopy(expected)


def render_json_v83a(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def render_report_v83a(value: Mapping[str, Any]) -> str:
    plans = value["deterministic_bucket_plans"]
    rows = []
    for choice, _capacity in contract.BUCKET_CHOICES_V83A:
        plan = plans[choice]
        rows.append(
            f"| `{choice}` | {plan['coalesced_collective_calls']} | "
            f"{plan['maximum_bucket_bytes']:,} | "
            f"{plan['collective_calls_eliminated']} |"
        )
    table = "\n".join(rows)
    memory = value["memory_and_bandwidth"]
    return f"""# Qwen3.6 LoRA exact-FP32 collective coalescing V83A

## Result

The source-only contract is sealed and its focused synthetic CPU suites pass
22 tests.  It binds the accepted 70-key, 4,528,128-element, 18,112,512-byte
FP32 LoRA surface and four deterministic source-order, no-split bucket plans.
No bucket is selected and no live launch is authorized.  V73D must first show
that canonical collective-call latency is material.

| Plan | Calls | Maximum staging bytes | Calls eliminated |
|---|---:|---:|---:|
{table}

The network payload and nominal ring bytes are unchanged for every plan.  The
flat plan needs {memory['flat_sequential_staging_bytes']:,} bytes of sequential
staging, {memory['flat_incremental_staging_bytes_versus_native_maximum']:,}
bytes above V72's largest single accumulator.  A materialized pack plus GPU
unpack moves at least {memory['conservative_pack_plus_gpu_unpack_hbm_bytes']:,}
local HBM bytes per actor per update, excluding noise generation, D2H, and
NCCL internals.  Therefore the intended future design is direct generation
into bucket views followed by reduced-slice D2H; that path is not measured or
implemented live here.

## Bound runtime semantics

The future canonical call is
`self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)` in FP32.
Fill, collective, scale, and D2H stay ordered on one stream.  An event is
recorded after scale and synchronized before host consumption or staging
reuse.  Any fill, collective, incompatible return, event, or unpack failure
must preserve the exact original master or terminally poison the transaction;
a partial candidate and stale retry are forbidden.

The fake-four-rank proof is bitwise exact because native and coalesced paths
use the same explicit rank order.  This does not prove live bitwise identity:
a different message size may make PyNccl/NCCL choose another reduction tree.
A future live arm therefore needs its own exact candidate/restore gate plus a
paired unchanged-V72 performance and memory comparison.

Focused command: `.venv/bin/pytest -q test_eggroll_es_fp32_collective_coalescing_v83a.py test_build_qwen36_lora_fp32_collective_coalescing_preregistration_v83a.py`

Observed result: `22 passed` (CPU only; no model, Ray, GPU, dataset, evaluation,
OOD, holdout, shadow, or probe access).

V83A content SHA-256: `{value['content_sha256_before_self_field']}`
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v83a()
    rendered = render_json_v83a(value)
    report = render_report_v83a(value)
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="ascii")
        REPORT.write_text(report, encoding="ascii")
    if args.check:
        _require_v83a(
            OUTPUT.read_text(encoding="ascii") == rendered,
            "V83A preregistration is stale",
        )
        _require_v83a(
            REPORT.read_text(encoding="ascii") == report,
            "V83A evidence report is stale",
        )
    if not args.write and not args.check:
        print(rendered, end="")
    else:
        print(
            json.dumps(
                {
                    "output": str(OUTPUT),
                    "report": str(REPORT),
                    "content_sha256": value[
                        "content_sha256_before_self_field"
                    ],
                    "live_arm_authorized": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
