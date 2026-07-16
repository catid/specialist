#!/usr/bin/env python3

import json
from collections import defaultdict

import pytest

import run_once_only_holdout_eval_v46a as subject


def _synthetic_eval_rows():
    rows = []
    for index in range(41):
        rows.append({
            "answer": f"va{index}", "item_id": f"v{index}",
            "normalized_source_url": f"https://validation/{index % 22}",
            "quality_bucket": "standard_grounded",
            "question": f"vq{index}", "source": "rope365",
            "split": "validation", "url": f"https://validation/{index % 22}",
        })
    sources = ["esinem"] * 3 + ["kinbakutoday"] * 8 + ["rope365"] * 6 + ["wikipedia"]
    for index, source in enumerate(sources):
        rows.append({
            "answer": f"ha{index}", "item_id": f"h{index}",
            "normalized_source_url": f"https://heldout/{index % 11}",
            "quality_bucket": (
                "safety_relevant_grounded" if index < 2 else "standard_grounded"
            ),
            "question": f"hq{index}", "source": source,
            "split": "heldout", "url": f"https://heldout/{index % 11}",
        })
    return rows


def test_synthetic_holdout_bundle_enforces_disjoint_identity_v46a():
    bundle, metadata, proof = subject.holdout_bundle_v46a(
        _synthetic_eval_rows()
    )
    assert bundle["rows"] == 18
    assert bundle["units"] == 18
    assert len(metadata) == 18
    assert proof["heldout_documents"] == 11
    assert proof["validation_vs_heldout_document_intersection_count"] == 0


def test_builder_and_loader_never_open_or_hash_holdout_v46a(tmp_path,
                                                             monkeypatch):
    import build_once_only_holdout_preregistration_v46a as builder
    original_hash = subject.core.file_sha256
    original_read_bytes = subject.Path.read_bytes

    def guarded_hash(path):
        assert subject.Path(path).resolve() != subject.HOLDOUT_PATH
        return original_hash(path)

    def guarded_read_bytes(path):
        assert subject.Path(path).resolve() != subject.HOLDOUT_PATH
        return original_read_bytes(path)

    monkeypatch.setattr(subject.core, "file_sha256", guarded_hash)
    monkeypatch.setattr(subject.Path, "read_bytes", guarded_read_bytes)
    value = builder.build()
    assert value["fixed_candidate_arm"] == "sft_v42i"
    assert value["launch_interlock"]["real_launch_permitted"] is False
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
    loaded = subject.load_preregistration_v46a(args)
    assert loaded["holdout_access_count_before_preregistration"] == 0


def test_real_launch_interlock_precedes_gpu_or_attempt_v46a(monkeypatch):
    prereg = {
        "schema": "prepared-once-only-fixed-sft-v42i-holdout-preregistration-v46a",
        "content_sha256_before_self_field": "x",
    }
    monkeypatch.setattr(subject, "load_preregistration_v46a", lambda args: prereg)
    monkeypatch.setattr(
        subject.core.v40a, "gpu_preflight",
        lambda: pytest.fail("GPU preflight must remain unreachable"),
    )
    with pytest.raises(RuntimeError, match="V43I OOD-first resolution"):
        subject.main([
            "--preregistration", "unused", "--preregistration-sha256", "x",
            "--preregistration-content-sha256", "x",
        ])


def test_fixed_gate_and_strata_are_aggregate_only_v46a():
    metrics = {}
    raw = defaultdict(list)
    metadata = []
    for index in range(2):
        identity = f"item-{index}"
        metadata.append({
            "item_sha256": identity, "source": "s",
            "quality_bucket": "q",
        })
        for arm in subject.ARMS_V46A:
            reward = 0.5 if arm == "sft_v42i" else 0.25
            raw[arm].append({
                "item_sha256": identity, "reward": reward,
                "format": "exact" if index == 0 else "partial",
            })
    for arm in subject.ARMS_V46A:
        rewards = [row["reward"] for row in raw[arm]]
        metrics[arm] = {
            "generated_row_mean_reward": sum(rewards) / 2,
            "generated_exact_count": 1,
            "generated_nonzero_count": 2,
            "protocol_leak_counters": {
                "protocol_token_emission": 0, "prompt_echo": 0,
                "empty_extracted_answer": 0,
            },
        }
    gate = subject.fixed_gate_v46a(metrics, raw)
    strata = subject.stratified_aggregate_v46a(raw, metadata)
    assert gate["passed"] is True
    assert gate["candidate_selection_performed"] is False
    assert strata["sft_v42i"]["source"]["s"]["rows"] == 2
