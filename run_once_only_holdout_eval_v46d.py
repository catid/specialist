#!/usr/bin/env python3
"""V46D: launch-enabled once-only V42I holdout after V45F/V46C resolution."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_once_only_holdout_eval_v46a as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v46d_once_only_sft_v42i_sealed_holdout_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v46d.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "once_only_sft_v42i_sealed_holdout_eval_v46d.json"
).resolve()
PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "once_only_sft_v42i_sealed_holdout_eval_v46d.json"
).resolve()
V46A_PREREG = prior.PREREG
V46A_PREREG_FILE_SHA256 = (
    "c4344446cd0f8394677f55064a0a244d32eaa2bbff450214394e90f103dec64e"
)
V46A_PREREG_CONTENT_SHA256 = (
    "612e3b511a588fa6ac085dd4e29899e093cf48d8c6bc3f810e99945987a1e84b"
)
V45F_REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_ijk_replica_consensus_ood_eval_v45f.json"
).resolve()
V45F_REPORT_FILE_SHA256 = (
    "82029429c5d253ec38bc12a35ea37fbb0780ca074078c229583add81d8d22a86"
)
V45F_REPORT_CONTENT_SHA256 = (
    "aa02e782783c7423b7306fb80ed1eb9f7c476ea5591f1b4a3e1cb93e4e0c955f"
)
V46C_REPORT = (
    ROOT / "experiments/eval_reports/"
    "lora_es_v43j_vs_sft_v42i_replicated_ood_first_eval_v46c.json"
).resolve()
V46C_REPORT_FILE_SHA256 = (
    "3f8d4fcb0a5a2fcb27f50199501169b7e43f3da7e44cce375812a234559d9fd2"
)
V46C_REPORT_CONTENT_SHA256 = (
    "7f1cdba8d17315aa2fd78a969e310b65039970c21124f2befc1cf6cf0025bd6e"
)


def _compact_sha(value: dict) -> str:
    return prior.core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def prior_preregistration_v46d() -> dict:
    if prior.core.file_sha256(V46A_PREREG) != V46A_PREREG_FILE_SHA256:
        raise RuntimeError("V46D prepared V46A preregistration file changed")
    value = json.loads(V46A_PREREG.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != V46A_PREREG_CONTENT_SHA256
        or _compact_sha(value) != V46A_PREREG_CONTENT_SHA256
        or value.get("fixed_candidate_arm") != "sft_v42i"
        or value.get("candidate_selection_permitted") is not False
        or value.get("post_result_tuning_or_selection_permitted") is not False
        or value.get("holdout_access_count_before_preregistration") != 0
        or value.get("holdout_opened_or_hashed_while_building") is not False
    ):
        raise RuntimeError("V46D prepared V46A preregistration content changed")
    return value


def resolution_evidence_v46d() -> dict:
    if prior.core.file_sha256(V45F_REPORT) != V45F_REPORT_FILE_SHA256:
        raise RuntimeError("V46D V45F aggregate file changed")
    v45f = json.loads(V45F_REPORT.read_text())
    f_selection = v45f.get("selection", {})
    f_i = f_selection.get(
        "per_logical_candidate_consensus_gate_table", {}
    ).get("sft_v42i", {})
    if (
        v45f.get("content_sha256_before_self_field")
        != V45F_REPORT_CONTENT_SHA256
        or _compact_sha(v45f) != V45F_REPORT_CONTENT_SHA256
        or v45f.get("status") != "complete_aggregate_only_no_heldout_access"
        or f_selection.get("selected_logical_candidate") != "sft_v42i"
        or f_i.get("both_replicas_independently_eligible") is not True
        or f_i.get("eligible") is not True
        or v45f.get("final_gate", {}).get("passed") is not True
        or v45f.get("base_duplicate_equivalence", {}).get("all_splits")
        is not True
        or v45f.get("gpu_activity", {}).get(
            "all_four_attributed_positive_each_phase"
        ) is not True
        or v45f.get("heldout_or_holdout_opened") is not False
        or v45f.get(
            "raw_questions_answers_or_generations_persisted_in_aggregate"
        ) is not False
    ):
        raise RuntimeError("V46D V45F aggregate content changed")

    if prior.core.file_sha256(V46C_REPORT) != V46C_REPORT_FILE_SHA256:
        raise RuntimeError("V46D V46C aggregate file changed")
    v46c = json.loads(V46C_REPORT.read_text())
    c_selection = v46c.get("selection", {})
    c_table = c_selection.get("per_logical_candidate_gate_table", {})
    c_i, c_es = c_table.get("sft_v42i", {}), c_table.get("lora_es_v43j", {})
    staged_a = v46c.get("staged_adapters", {}).get("sft_v42i_a", {})
    staged_b = v46c.get("staged_adapters", {}).get("sft_v42i_b", {})
    expected_stage = prior.staged_candidate_binding_v46a()
    for staged in (staged_a, staged_b):
        if (
            staged.get("weights_file_sha256")
            != expected_stage["weights_file_sha256"]
            or staged.get("adapter_config_file_sha256")
            != expected_stage["adapter_config_file_sha256"]
            or staged.get("manifest_file_sha256")
            != expected_stage["manifest_file_sha256"]
            or staged.get("manifest_content_sha256")
            != expected_stage["manifest_content_sha256"]
            or staged.get("transformed_identity_sha256")
            != expected_stage["transformed_identity_sha256"]
            or staged.get("tensor_bytes_preserved_exactly") is not True
        ):
            raise RuntimeError("V46D V46C exact staged V42I identity changed")
    if (
        v46c.get("content_sha256_before_self_field")
        != V46C_REPORT_CONTENT_SHA256
        or _compact_sha(v46c) != V46C_REPORT_CONTENT_SHA256
        or v46c.get("status") != "complete_aggregate_only_no_heldout_access"
        or c_selection.get("selected_logical_candidate") != "sft_v42i"
        or c_selection.get("eligible_logical_candidates") != ["sft_v42i"]
        or c_i.get("both_replicas_independently_ood_eligible") is not True
        or c_i.get("eligible") is not True
        or c_es.get("eligible") is not False
        or v46c.get("final_gate", {}).get("passed") is not True
        or v46c.get("base_duplicate_equivalence", {}).get("all_splits")
        is not True
        or v46c.get("gpu_activity", {}).get(
            "all_four_attributed_positive_each_phase"
        ) is not True
        or v46c.get("heldout_or_holdout_opened") is not False
        or v46c.get(
            "raw_questions_answers_or_generations_persisted_in_aggregate"
        ) is not False
    ):
        raise RuntimeError("V46D V46C aggregate content changed")
    return {
        "schema": "v42i-final-boundary-resolution-evidence-v46d",
        "v45f": {
            "path": str(V45F_REPORT),
            "file_sha256": V45F_REPORT_FILE_SHA256,
            "content_sha256": V45F_REPORT_CONTENT_SHA256,
            "selected_logical_candidate": "sft_v42i",
            "both_v42i_replicas_independently_ood_eligible": True,
            "strict_final_gate_passed": True,
            "all_four_gpus_positive_each_phase": True,
            "heldout_or_holdout_opened": False,
        },
        "v46c": {
            "path": str(V46C_REPORT),
            "file_sha256": V46C_REPORT_FILE_SHA256,
            "content_sha256": V46C_REPORT_CONTENT_SHA256,
            "selected_logical_candidate": "sft_v42i",
            "only_eligible_logical_candidate": "sft_v42i",
            "v43j_es_eligible": False,
            "both_v42i_replicas_independently_ood_eligible": True,
            "strict_final_gate_passed": True,
            "all_four_gpus_positive_each_phase": True,
            "heldout_or_holdout_opened": False,
        },
        "fixed_candidate_resolution": "sft_v42i",
        "candidate_selection_closed_before_holdout": True,
        "raw_aggregate_only_reports_inspected": True,
        "raw_semantics_inspected": False,
        "heldout_or_holdout_opened": False,
    }


def nonprotected_bindings_v46d() -> dict:
    return {
        "runtime": prior.core.file_sha256(Path(__file__).resolve()),
        "prepared_v46a_runtime": prior.core.file_sha256(
            Path(prior.__file__).resolve()
        ),
        "prepared_v46a_nonprotected_bindings": (
            prior.nonprotected_bindings_v46a()
        ),
        "prepared_v46a_preregistration_file_sha256": V46A_PREREG_FILE_SHA256,
        "prepared_v46a_preregistration_content_sha256": (
            V46A_PREREG_CONTENT_SHA256
        ),
        "resolution_evidence": resolution_evidence_v46d(),
    }


def load_preregistration_v46d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if prior.core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V46D preregistration file changed")
    value = json.loads(path.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or _compact_sha(value) != args.preregistration_content_sha256
        or value.get("schema")
        != "once-only-fixed-sft-v42i-holdout-preregistration-v46d"
        or value.get("status")
        != "preregistered_after_v45f_v46c_before_once_only_holdout_access"
        or value.get("fixed_arms") != list(prior.ARMS_V46A)
        or value.get("fixed_candidate_arm") != "sft_v42i"
        or value.get("candidate_selection_permitted") is not False
        or value.get("post_result_tuning_or_selection_permitted") is not False
        or value.get("holdout_access_authorized_once") is not True
        or value.get("holdout_access_count_before_preregistration") != 0
        or value.get("holdout_commitment")
        != prior.holdout_report_commitment_v46a()
        or value.get("fixed_staged_candidate")
        != prior.staged_candidate_binding_v46a()
        or value.get("resolution_evidence") != resolution_evidence_v46d()
        or value.get("implementation_bindings") != nonprotected_bindings_v46d()
        or value.get("launch_interlock") != {
            "resolved": True,
            "resolution": "V45F and V46C both strict-selected SFT V42I",
            "real_launch_permitted": True,
            "candidate_identity_replaceable_after_seal": False,
        }
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("V46D preregistration content changed")
    # Deliberately do not stat, hash, or open the holdout here.
    return value


@contextmanager
def patched_prior_v46d():
    saved = {
        "EXPERIMENT": prior.EXPERIMENT, "RUN_DIR": prior.RUN_DIR,
        "ATTEMPT": prior.ATTEMPT, "GPU_LOG": prior.GPU_LOG,
        "REPORT": prior.REPORT, "load": prior.load_preregistration_v46a,
        "interlock": prior.LAUNCH_INTERLOCK_RESOLVED_V46A,
        "self_hashed": prior.core.self_hashed,
    }

    def revised_self_hashed(value: dict) -> dict:
        value = dict(value)
        schemas = {
            "once-only-fixed-sft-v42i-holdout-attempt-v46a":
                "once-only-fixed-sft-v42i-holdout-attempt-v46d",
            "once-only-fixed-sft-v42i-holdout-aggregate-v46a":
                "once-only-fixed-sft-v42i-holdout-aggregate-v46d",
            "once-only-fixed-sft-v42i-holdout-failure-v46a":
                "once-only-fixed-sft-v42i-holdout-failure-v46d",
        }
        if value.get("schema") in schemas:
            value["schema"] = schemas[value["schema"]]
        value["runtime_revision"] = "v46d_v45f_v46c_resolved"
        return saved["self_hashed"](value)

    prior.EXPERIMENT, prior.RUN_DIR = EXPERIMENT, RUN_DIR
    prior.ATTEMPT, prior.GPU_LOG, prior.REPORT = ATTEMPT, GPU_LOG, REPORT
    prior.load_preregistration_v46a = load_preregistration_v46d
    prior.LAUNCH_INTERLOCK_RESOLVED_V46A = True
    prior.core.self_hashed = revised_self_hashed
    try:
        yield
    finally:
        prior.EXPERIMENT, prior.RUN_DIR = saved["EXPERIMENT"], saved["RUN_DIR"]
        prior.ATTEMPT, prior.GPU_LOG = saved["ATTEMPT"], saved["GPU_LOG"]
        prior.REPORT = saved["REPORT"]
        prior.load_preregistration_v46a = saved["load"]
        prior.LAUNCH_INTERLOCK_RESOLVED_V46A = saved["interlock"]
        prior.core.self_hashed = saved["self_hashed"]


def main(argv: list[str] | None = None) -> int:
    args = prior.parser().parse_args(argv)
    prereg = load_preregistration_v46d(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "fixed_candidate": "sft_v42i",
            "protected_semantic_access_count": 0,
            "holdout_opened_or_hashed": False,
            "candidate_selection_performed": False,
            "real_launch_permitted": True,
            "resolution_evidence": "V45F+V46C",
        }, sort_keys=True))
        return 0
    with patched_prior_v46d():
        return prior.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
