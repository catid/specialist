#!/usr/bin/env python3
"""Build compact evidence for the immutable V24A memory-endpoint failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import eggroll_es_hybrid_backend_preregistration_v24a as prereg


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
ATTEMPT_RELATIVE_PATH_R1 = (
    "experiments/eggroll_es_hpo/runs/"
    ".s6_v24a_hybrid_backend_train_only_compatibility_runtime.launch_attempt.json"
)
REPORT_RELATIVE_PATH_R1 = (
    "experiments/eggroll_es_hpo/runs/"
    "s6_v24a_hybrid_backend_train_only_compatibility_runtime/"
    "hybrid_backend_compatibility_v24a.json"
)
OUTPUT_PATH_R1 = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V24A_MEMORY_ENDPOINT_FAILURE_EVIDENCE_R1.json"
)
VLLM_MODEL_RUNNER_RELATIVE_PATH_R1 = (
    "es-at-scale/.venv/lib/python3.12/site-packages/vllm/v1/worker/"
    "gpu_model_runner.py"
)
VLLM_GPU_WORKER_RELATIVE_PATH_R1 = (
    "es-at-scale/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py"
)
ORIGINAL_RUNTIME_RELATIVE_PATH_R1 = "run_eggroll_es_hybrid_backend_v24a.py"
EXPECTED_ATTEMPT_FILE_SHA256_R1 = (
    "4d33a25c394efeb350dd03761e018d18bb29146315ff945152757ae01f833390"
)
EXPECTED_ATTEMPT_CONTENT_SHA256_R1 = (
    "31679c585b49256572b7e86e60b23753481d4fb7911cb1a5c863a4528b35deaa"
)
EXPECTED_SOURCE_HEAD_R1 = "cb52c1a5960d7cd68d7046cf09f84d482d71ad1e"
EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R1 = (
    "8e55cd04379382fd4e33b913fb87a934b6142a062e40144a6633aaf249a49bf2"
)
EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R1 = (
    "db0e6fc457e93e218a9801b55b88ed505f9b063a453e597868c3fda18e9c7262"
)
EXPECTED_RECIPE_CONTENT_SHA256_R1 = (
    "10b9409ce939a132f95d8da9011d9dc8e649bd4ccc47ec93d25d855032efa1c8"
)
EXPECTED_ORIGINAL_RUNTIME_FILE_SHA256_R1 = (
    "493368981dd77c0c90673ac66fc393b4841886b2da5041c34c8e628bf291209f"
)
EXPECTED_VLLM_MODEL_RUNNER_FILE_SHA256_R1 = (
    "6c92ded8468f44d6df863a617ce588f132fa6df7031feecc0cc421702a41610e"
)
EXPECTED_VLLM_GPU_WORKER_FILE_SHA256_R1 = (
    "7e00284da7b453154af47300630483ed7ea5a5d79e724c5ee61d4a24edaf930e"
)
FORBIDDEN_KEYS_R1 = {
    "traceback", "message", "question", "questions", "answer", "answers",
    "prompt", "prompts", "responses", "row_content", "token_ids",
    "model_repr", "unit_scores", "bootstrap_draws", "bootstrap_replicates",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _seal(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def _assert_compact(value):
    overlap = FORBIDDEN_KEYS_R1 & set(_recursive_keys(value))
    if overlap:
        raise RuntimeError(f"v24a-r1 evidence contains forbidden keys: {sorted(overlap)}")


def build_memory_failure_evidence_r1(
    attempt_path: Path,
    report_path: Path,
    model_runner_path: Path,
    gpu_worker_path: Path,
    original_runtime_path: Path,
):
    """Verify the failed run and the exact local vLLM load-memory semantics."""
    attempt_path = Path(attempt_path).resolve()
    report_path = Path(report_path).resolve()
    model_runner_path = Path(model_runner_path).resolve()
    gpu_worker_path = Path(gpu_worker_path).resolve()
    original_runtime_path = Path(original_runtime_path).resolve()
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    source = attempt.get("source_provenance", {})
    recipe = attempt.get("recipe", {})
    failure = attempt.get("failure", {})
    model_source = model_runner_path.read_text(encoding="utf-8")
    worker_source = gpu_worker_path.read_text(encoding="utf-8")
    runtime_source = original_runtime_path.read_text(encoding="utf-8")
    if (
        file_sha256(attempt_path) != EXPECTED_ATTEMPT_FILE_SHA256_R1
        or attempt.get("content_sha256_before_self_field")
        != EXPECTED_ATTEMPT_CONTENT_SHA256_R1
        or attempt.get("content_sha256_before_self_field")
        != prereg.canonical_sha256(_without_self(attempt))
        or attempt.get("schema") != "eggroll-es-durable-launch-attempt-v24a"
        or attempt.get("status") != "failed"
        or attempt.get("phase") != "inside_v24a_train_only_runtime"
        or failure.get("type") != "SystemExit"
        or attempt.get("model_update_applied") is not False
        or attempt.get("nontrain_surface_opened") is not False
        or source.get("git_head") != EXPECTED_SOURCE_HEAD_R1
        or source.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R1
        or source.get("content_sha256_before_self_field")
        != EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R1
        or recipe.get("content_sha256_before_self_field")
        != EXPECTED_RECIPE_CONTENT_SHA256_R1
        or recipe.get("implementation_bundle_sha256")
        != EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R1
        or source.get("files", {}).get("runtime_v24a", {}).get("file_sha256")
        != EXPECTED_ORIGINAL_RUNTIME_FILE_SHA256_R1
        or report_path.exists()
        or "configuration" in attempt
        or "report_binding" in attempt
        or file_sha256(model_runner_path)
        != EXPECTED_VLLM_MODEL_RUNNER_FILE_SHA256_R1
        or file_sha256(gpu_worker_path) != EXPECTED_VLLM_GPU_WORKER_FILE_SHA256_R1
        or file_sha256(original_runtime_path)
        != EXPECTED_ORIGINAL_RUNTIME_FILE_SHA256_R1
        or model_source.count("self.model_memory_usage = m.consumed_memory") != 1
        or "Model loading took %s GiB memory" not in model_source
        or worker_source.count(
            "weights_memory=int(self.model_runner.model_memory_usage)"
        ) != 1
        or "gpu_memory_utilization=0.82" not in runtime_source
        or '"source": "pynvml.nvmlDeviceGetMemoryInfo(handle).used"'
        not in json.dumps(recipe.get("nvml_memory_contract", {}), sort_keys=True)
    ):
        raise RuntimeError("V24A memory-endpoint failure evidence changed")

    evidence = {
        "schema": "eggroll-es-v24a-memory-endpoint-failure-evidence-r1",
        "authority": {
            "train_only_instrumentation_diagnosis": True,
            "model_selection_allowed": False,
            "evaluation_opened": False,
            "dataset_content_inspected": False,
        },
        "failed_attempt": {
            "relative_path": ATTEMPT_RELATIVE_PATH_R1,
            "file_sha256": EXPECTED_ATTEMPT_FILE_SHA256_R1,
            "content_sha256": EXPECTED_ATTEMPT_CONTENT_SHA256_R1,
            "source_git_head": EXPECTED_SOURCE_HEAD_R1,
            "source_implementation_bundle_sha256": (
                EXPECTED_SOURCE_IMPLEMENTATION_SHA256_R1
            ),
            "source_provenance_content_sha256": (
                EXPECTED_SOURCE_PROVENANCE_CONTENT_SHA256_R1
            ),
            "recipe_content_sha256": EXPECTED_RECIPE_CONTENT_SHA256_R1,
            "status": "failed",
            "phase": "inside_v24a_train_only_runtime",
            "failure_type": "SystemExit",
            "compact_report_relative_path": REPORT_RELATIVE_PATH_R1,
            "compact_report_absent": True,
            "configuration_not_persisted": True,
            "model_update_applied": False,
            "nontrain_surface_opened": False,
        },
        "invalid_original_memory_endpoint": {
            "name": "post_warmup_nvml_resident_bytes",
            "original_source": "pynvml.nvmlDeviceGetMemoryInfo(handle).used",
            "fixed_gpu_memory_utilization": 0.82,
            "includes_vllm_kv_cache_reservation": True,
            "equalized_by_fixed_gpu_memory_utilization": True,
            "contemporaneous_approximate_mib": {
                "bf16": 82_484,
                "hybrid": 82_508,
            },
            "observed_hybrid_not_lower_than_bf16": True,
            "forty_percent_reduction_gate_capable": False,
            "excluded_from_retry_memory_gate": True,
            "retained_as_diagnostic": True,
        },
        "authoritative_replacement_endpoint": {
            "name": "vllm_model_load_consumed_bytes",
            "rpc_expression": "int(self.model_runner.model_memory_usage)",
            "assignment_semantics": "self.model_memory_usage = m.consumed_memory",
            "model_runner_source": {
                "relative_path": VLLM_MODEL_RUNNER_RELATIVE_PATH_R1,
                "file_sha256": EXPECTED_VLLM_MODEL_RUNNER_FILE_SHA256_R1,
                "assignment_line": 5294,
                "load_log_line": 5307,
            },
            "gpu_worker_source": {
                "relative_path": VLLM_GPU_WORKER_RELATIVE_PATH_R1,
                "file_sha256": EXPECTED_VLLM_GPU_WORKER_FILE_SHA256_R1,
                "weights_memory_use_line": 466,
            },
            "measured_after_model_load_before_scoring": True,
            "used_for_retry_memory_gate": True,
        },
        "retry_scope": {
            "sole_semantic_repair": (
                "replace post-warmup NVML gate input with exact vLLM "
                "model-load-consumed bytes"
            ),
            "all_arms_basis_panels_schedule_quality_speed_bootstrap_and_guards_unchanged": True,
            "duplicate_backend_values_must_match_exactly": True,
            "memory_reduction_threshold": 0.40,
            "abort_before_reference_a_b_or_first_perturbation_on_failure": True,
        },
        "traceback_or_model_repr_persisted": False,
        "row_response_score_or_bootstrap_content_persisted": False,
    }
    _assert_compact(evidence)
    return _seal(evidence)


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--model-runner-path", type=Path, required=True)
    parser.add_argument("--gpu-worker-path", type=Path, required=True)
    parser.add_argument("--original-runtime-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH_R1)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    evidence = build_memory_failure_evidence_r1(
        args.attempt_path, args.report_path, args.model_runner_path,
        args.gpu_worker_path, args.original_runtime_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "file_sha256": file_sha256(args.output),
        "content_sha256": evidence["content_sha256_before_self_field"],
    }, sort_keys=True))
    return evidence


if __name__ == "__main__":
    main()
