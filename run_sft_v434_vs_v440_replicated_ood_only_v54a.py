#!/usr/bin/env python3
"""Run the sealed two-wave v434-equal versus v440-equal OOD comparison."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_sft_v434_sampling_midpoint_ood_only_v49d as parent
import stage_v440_equal_adapter_v54a as v440_stage
import stage_v49d_adapters_vllm as v434_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS = {
    "v434_equal": ("v434_equal_a", "v434_equal_b"),
    "v440_equal": ("v440_equal_a", "v440_equal_b"),
}
LOGICAL_CANDIDATES = tuple(LOGICAL_REPLICAS)
CANDIDATE_ARMS = tuple(
    arm for replicas in LOGICAL_REPLICAS.values() for arm in replicas
)
ARMS = BASE_ARMS + CANDIDATE_ARMS
STAGED_BY_LOGICAL = {
    "v434_equal": v434_stage.OUTPUTS["v434_equal"],
    "v440_equal": v440_stage.OUTPUT,
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
    "v440_equal": {
        "weights": "d660b5eb836b561f98d2fcfa1379309f330778c675512a88ae373fea5e620b1d",
        "config": "45c2329fad2e6ccdd5244ec456d12533875f972ca2adacc069cab3f9c90373e4",
        "manifest_file": "40d12a096299895cec5a9de8581caeaa7b29e983f81721bfe12dcecdf1e91b7e",
        "manifest_content": "cff3c35a1983354cff85066a14f60e4647ef58e605fe32d0c012d1081dc9a7fc",
        "transformed_identity": "13405c8490ebdf6ebb4b2f94a9c7c34e4ae5f7678e75c020da04ad4af9579500",
    },
}
OOD_INPUTS = dict(parent.OOD_INPUTS)
BOOTSTRAP_SAMPLES = parent.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = parent.BOOTSTRAP_SEED
EXPERIMENT = "v54a_v434_equal_vs_v440_equal_replicated_ood_only"
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT
).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v54a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v54a.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_vs_v440_equal_replicated_ood_only_v54a.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_v440_equal_replicated_ood_only_v54a.json"
).resolve()
BUILDER = (
    ROOT / "build_sft_v434_vs_v440_ood_preregistration_v54a.py"
).resolve()
TESTS = (ROOT / "test_sft_v434_vs_v440_ood_v54a.py").resolve()


def arm_wave_plan_v54a():
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS)),
    )


def _source_seal(logical: str) -> dict:
    if logical == "v434_equal":
        return v434_stage.source_seal_v49d(logical)
    if logical == "v440_equal":
        return v440_stage.source_seal_v54a(logical)
    raise ValueError(logical)


def canonical_stage_binding_v54a(logical: str) -> dict:
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
        or value.get("arm") != logical
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
    ):
        raise RuntimeError(f"V54A {logical} staged adapter changed")
    return observed


def replica_stage_bindings_v54a() -> dict:
    logical = {
        name: canonical_stage_binding_v54a(name)
        for name in LOGICAL_CANDIDATES
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


def implementation_bindings_v54a() -> dict:
    paths = {
        "runtime_v54a": Path(__file__).resolve(),
        "builder_v54a": BUILDER,
        "tests_v54a": TESTS,
        "runtime_parent_v49d": Path(parent.__file__).resolve(),
        "stage_v440_runtime": Path(v440_stage.__file__).resolve(),
        "stage_v434_runtime": Path(v434_stage.__file__).resolve(),
        "canonical_stage_runtime": Path(v440_stage.prior.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "metric_runtime": Path(parent.metrics.__file__).resolve(),
        "ood_gate_runtime": Path(parent.ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(parent.parser_fix.__file__).resolve(),
        "topology_runtime": Path(parent.topology.__file__).resolve(),
        "cleanup_runtime": Path(parent.cleanup.__file__).resolve(),
        "v434_source_weights": (
            v434_stage.SOURCES["v434_equal"] / "adapter_model.safetensors"
        ),
        "v434_source_config": (
            v434_stage.SOURCES["v434_equal"] / "adapter_config.json"
        ),
        "v434_source_report": v434_stage.REPORTS["v434_equal"],
        "v434_source_attempt": v434_stage.ATTEMPTS["v434_equal"],
        "v434_source_gpu_log": v434_stage.GPU_LOGS["v434_equal"],
        "v434_staged_weights": (
            STAGED_BY_LOGICAL["v434_equal"] / "adapter_model.safetensors"
        ),
        "v434_staged_config": (
            STAGED_BY_LOGICAL["v434_equal"] / "adapter_config.json"
        ),
        "v434_stage_manifest": (
            STAGED_BY_LOGICAL["v434_equal"] / "stage_manifest_v44a.json"
        ),
        "v440_source_weights": v440_stage.SOURCE / "adapter_model.safetensors",
        "v440_source_config": v440_stage.SOURCE / "adapter_config.json",
        "v440_source_report": v440_stage.REPORT,
        "v440_source_attempt": v440_stage.ATTEMPT,
        "v440_source_gpu_log": v440_stage.GPU_LOG,
        "v440_source_preregistration": v440_stage.PREREGISTRATION,
        "v440_staged_weights": (
            STAGED_BY_LOGICAL["v440_equal"] / "adapter_model.safetensors"
        ),
        "v440_staged_config": (
            STAGED_BY_LOGICAL["v440_equal"] / "adapter_config.json"
        ),
        "v440_stage_manifest": (
            STAGED_BY_LOGICAL["v440_equal"] / "stage_manifest_v44a.json"
        ),
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
    }
    result = {name: core.file_sha256(path) for name, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = (
        parent.ood_first.environment.environment_bindings_v44b()
    )
    result["source_seals"] = {
        arm: _source_seal(arm) for arm in LOGICAL_CANDIDATES
    }
    return result


def load_preregistration_v54a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V54A OOD preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "sft-v434-equal-vs-v440-equal-replicated-ood-only-v54a"
        or value.get("status") != "preregistered_before_fresh_ood_only_launch"
        or value.get("evaluation_launch_authorized") is not True
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("shadow_access_authorized") is not False
        or value.get("single_access_inputs") != OOD_INPUTS
        or value.get("arms") != list(ARMS)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES)
        or value.get("staged_adapters") != replica_stage_bindings_v54a()
        or value.get("implementation_bindings")
        != implementation_bindings_v54a()
    ):
        raise RuntimeError("V54A OOD preregistration content changed")
    core._forbid_holdout_v44a(item["path"] for item in OOD_INPUTS.values())
    return value


def _gate_table_v54a(ood_qa, prose_details, raw_sink):
    table = {}
    for logical, replicas in LOGICAL_REPLICAS.items():
        gates = [
            parent._replica_gate(
                ood_qa, prose_details, raw_sink, arm
            )
            for arm in replicas
        ]
        table[logical] = {
            "replicas": list(replicas),
            "replica_gates": gates,
            "both_replicas_independently_ood_eligible": all(
                gate["eligible"] for gate in gates
            ),
        }
    v434 = LOGICAL_REPLICAS["v434_equal"]
    v440 = LOGICAL_REPLICAS["v440_equal"]

    def mean_qa(arms, field):
        return sum(float(ood_qa[arm][field]) for arm in arms) / 2.0

    def mean_prose(arms):
        return sum(
            float(prose_details[arm]["mean_token_logprob"])
            for arm in arms
        ) / 2.0

    direct = {
        "comparison": "mean(v440_equal replicas)-mean(v434_equal replicas)",
        "v440_minus_v434_mean_reward": (
            mean_qa(v440, "generated_equal_unit_mean_reward")
            - mean_qa(v434, "generated_equal_unit_mean_reward")
        ),
        "v440_minus_v434_mean_exact_count": (
            mean_qa(v440, "generated_exact_count")
            - mean_qa(v434, "generated_exact_count")
        ),
        "v440_minus_v434_mean_prose_token_logprob": (
            mean_prose(v440) - mean_prose(v434)
        ),
        "paired_bootstrap_ci_role": "informational_not_a_direct_gate",
    }
    direct["reward_nonnegative"] = (
        direct["v440_minus_v434_mean_reward"] >= 0.0
    )
    direct["exact_nonnegative"] = (
        direct["v440_minus_v434_mean_exact_count"] >= 0.0
    )
    direct["prose_nonregression"] = (
        direct["v440_minus_v434_mean_prose_token_logprob"] >= 0.0
    )
    direct["all_direct_point_gates_passed"] = (
        direct["reward_nonnegative"]
        and direct["exact_nonnegative"]
        and direct["prose_nonregression"]
    )
    return table, direct


@contextmanager
def patched_parent_v54a():
    names = {
        "BASE_ARMS": BASE_ARMS,
        "LOGICAL_REPLICAS": LOGICAL_REPLICAS,
        "LOGICAL_CANDIDATES": LOGICAL_CANDIDATES,
        "CANDIDATE_ARMS": CANDIDATE_ARMS,
        "ARMS": ARMS,
        "STAGED_BY_LOGICAL": STAGED_BY_LOGICAL,
        "STAGED_BY_ARM": STAGED_BY_ARM,
        "ADAPTER_IDS": ADAPTER_IDS,
        "OOD_INPUTS": OOD_INPUTS,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "RAW": RAW,
        "GPU_LOG": GPU_LOG,
        "REPORT": REPORT,
        "DEFAULT_PREREGISTRATION": DEFAULT_PREREGISTRATION,
        "arm_wave_plan_v49d": arm_wave_plan_v54a,
        "canonical_stage_binding_v49d": canonical_stage_binding_v54a,
        "replica_stage_bindings_v49d": replica_stage_bindings_v54a,
        "implementation_bindings_v49d": implementation_bindings_v54a,
        "load_preregistration_v49d": load_preregistration_v54a,
        "_gate_table": _gate_table_v54a,
    }
    saved = {name: getattr(parent, name) for name in names}
    for name, value in names.items():
        setattr(parent, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(parent, name, value)


def _rewrite_success_v54a() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report.pop("content_sha256_before_self_field", None)
    report.update({
        "schema": "v54a-v434-equal-v440-equal-ood-only-aggregate",
        "protocol_parent": "V49D replicated OOD-only runtime",
        "status": "complete_ood_only_shadow_and_holdout_unopened",
    })
    core.atomic_json(REPORT, core.self_hashed(report))
    complete_path = ATTEMPT.with_suffix(".complete.json")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("content_sha256_before_self_field", None)
    complete.update({
        "schema": "v54a-ood-only-complete-attempt",
        "report_sha256": core.file_sha256(REPORT),
    })
    core.atomic_json(complete_path, core.self_hashed(complete))


def parser():
    return core.parser()


def main(argv: list[str] | None = None) -> int:
    effective = parser().parse_args(argv)
    with patched_parent_v54a():
        result = parent.main(argv)
    if result == 0 and not effective.dry_run:
        _rewrite_success_v54a()
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": core.file_sha256(REPORT),
            "complete_attempt": str(ATTEMPT.with_suffix(".complete.json")),
            "complete_attempt_sha256": core.file_sha256(
                ATTEMPT.with_suffix(".complete.json")
            ),
        }, sort_keys=True))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
