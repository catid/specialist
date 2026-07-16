#!/usr/bin/env python3

import json

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44b as subject


def test_failed_launch_was_before_protected_access_v44b():
    value = subject.failed_launch_provenance_v44b()
    assert value["failure_before_model_creation"] is True
    assert value["protected_semantic_access_count"] == 0
    assert value["heldout_or_holdout_opened"] is False
    assert value["scientific_observations_produced"] is False
    assert not (subject.FAILED_RUN_V44A / "raw_items_v44a.json").exists()
    assert not core.REPORT.exists()


def test_environment_is_exact_es_at_scale_v44b():
    value = subject.environment_bindings_v44b()
    assert value["sys_prefix"] == str(subject.EXPECTED_ENV_PREFIX)
    assert value["vllm_importable"] is True
    assert value["packages"]["vllm"] == "0.25.0"
    assert "/es-at-scale/" in value["module_files"]["es_at_scale"]


def test_fresh_retry_paths_do_not_overlap_failed_attempt_v44b():
    paths = {subject.RUN_DIR, subject.ATTEMPT, subject.RAW,
             subject.GPU_LOG, subject.REPORT}
    assert subject.FAILED_RUN_V44A not in paths
    assert subject.FAILED_ATTEMPT_V44A not in paths
    # V44B has since been consumed by the sealed parser failure audited in
    # V44C; it must remain isolated from the earlier V44A paths.
    assert subject.RUN_DIR.exists()
    assert subject.ATTEMPT.exists()
    assert not subject.RAW.exists()
    assert not subject.REPORT.exists()


def test_prereg_load_never_hashes_protected_inputs_v44b(tmp_path, monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v44b as builder

    value = builder.build()
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(candidate):
        assert str(candidate) not in protected
        return original(candidate)

    monkeypatch.setattr(core, "file_sha256", guarded)
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    loaded = subject.load_preregistration_v44b(args)
    assert loaded["single_access_inputs"] == core.PROTECTED_INPUTS_V44A
