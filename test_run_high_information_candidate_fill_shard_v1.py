from __future__ import annotations

from pathlib import Path

import run_high_information_candidate_fill_shard_v1 as fill
import run_high_information_candidate_shard_v1 as base


def _request(**overrides):
    value = {
        "request_id": "generation-request-v1:" + "a" * 64,
        "task_family": "closed_book_application",
        "task_subtype": "application_scenario",
        "generation_mode": "positive",
        "target_verified_assistant_tokens": 192,
        "source_group_id": "synthetic-source-group",
        "gpu_shard": 0,
    }
    value.update(overrides)
    return value


def test_fill_seed_is_stable_and_distinct_from_original():
    request_id = _request()["request_id"]
    assert fill.fill_request_seed(request_id) == fill.fill_request_seed(request_id)
    assert fill.fill_request_seed(request_id) != base.request_seed(request_id)


def test_fill_paths_never_overlap_original_pass():
    for shard in range(4):
        original = set(base.shard_paths(shard, smoke=False).values())
        replacement = set(fill.fill_shard_paths(shard, smoke=False).values())
        assert not original & replacement
        assert all("generation_fill_candidates" in path.name for path in replacement)


def test_fill_prompt_adds_quality_and_application_requirements():
    messages = fill.fill_generation_messages(
        _request(),
        {"text": "If numbness occurs, stop and remove the rope."},
    )
    system = messages[0]["content"]
    assert "Every requested facet must be explicitly answered" in system
    assert "construct a concrete novel scenario" in system
    assert "bare era, date, public-identity" in system
    assert messages[-1]["content"].endswith("---\n")


def test_fill_sampling_records_pass_specific_seed():
    request = _request()
    values = fill.fill_sampling_kwargs(request)
    assert values["n"] == 1
    assert values["temperature"] == 0.4
    assert values["seed"] == fill.fill_request_seed(request["request_id"])


def test_fill_candidate_receipt_matches_actual_sampling_pass():
    request = _request()
    row = fill.fill_make_candidate_record(
        request,
        '{"examples": []}',
        finish_reason="stop",
        generated_token_count=4,
    )
    assert row["generator"]["temperature"] == 0.4
    assert row["generator"]["generation_pass"] == fill.PASS_ID
    assert row["generator"]["seed"] == fill.fill_request_seed(request["request_id"])
    assert row["content_sha256_before_self_field"] == base.candidate_content_sha256(row)


def test_configure_base_receipts_wrapper_and_changes_only_expected_surfaces():
    fill.configure_base()
    assert Path(base.__file__).resolve() == Path(fill.__file__).resolve()
    assert base.request_seed is fill.fill_request_seed
    assert base.shard_paths is fill.fill_shard_paths
    assert base.generation_messages is fill.fill_generation_messages
    assert base.sampling_kwargs is fill.fill_sampling_kwargs
    assert base.make_candidate_record is fill.fill_make_candidate_record
