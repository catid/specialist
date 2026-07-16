#!/usr/bin/env python3

import json

import run_once_only_holdout_eval_v46d as subject


def test_resolution_evidence_strictly_closes_selection_v46d():
    value = subject.resolution_evidence_v46d()
    assert value["fixed_candidate_resolution"] == "sft_v42i"
    assert value["v45f"][
        "both_v42i_replicas_independently_ood_eligible"
    ] is True
    assert value["v46c"]["only_eligible_logical_candidate"] == "sft_v42i"
    assert value["v46c"]["v43j_es_eligible"] is False
    assert value["candidate_selection_closed_before_holdout"] is True
    assert value["heldout_or_holdout_opened"] is False


def test_builder_loader_and_dry_run_never_touch_holdout_v46d(tmp_path,
                                                              monkeypatch):
    import build_once_only_holdout_preregistration_v46d as builder
    original_hash = subject.prior.core.file_sha256
    original_read_bytes = subject.Path.read_bytes

    def guarded_hash(path):
        assert subject.Path(path).resolve() != subject.prior.HOLDOUT_PATH
        return original_hash(path)

    def guarded_read_bytes(path):
        assert subject.Path(path).resolve() != subject.prior.HOLDOUT_PATH
        return original_read_bytes(path)

    monkeypatch.setattr(subject.prior.core, "file_sha256", guarded_hash)
    monkeypatch.setattr(subject.Path, "read_bytes", guarded_read_bytes)
    value = builder.build()
    assert value["launch_interlock"]["real_launch_permitted"] is True
    assert value["fixed_candidate_arm"] == "sft_v42i"
    assert value["candidate_selection_permitted"] is False
    assert value["post_result_tuning_or_selection_permitted"] is False
    assert value["holdout_opened_or_hashed_while_building"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original_hash(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    assert subject.load_preregistration_v46d(args)[
        "holdout_access_count_before_preregistration"
    ] == 0


def test_runtime_patch_uses_fresh_paths_and_resolves_only_interlock_v46d():
    assert subject.RUN_DIR != subject.prior.RUN_DIR
    assert subject.ATTEMPT != subject.prior.ATTEMPT
    assert subject.REPORT != subject.prior.REPORT
    assert subject.prior.LAUNCH_INTERLOCK_RESOLVED_V46A is False
    with subject.patched_prior_v46d():
        assert subject.prior.RUN_DIR == subject.RUN_DIR
        assert subject.prior.ATTEMPT == subject.ATTEMPT
        assert subject.prior.REPORT == subject.REPORT
        assert subject.prior.LAUNCH_INTERLOCK_RESOLVED_V46A is True
    assert subject.prior.LAUNCH_INTERLOCK_RESOLVED_V46A is False
