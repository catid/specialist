import hashlib
import inspect
import json

import build_eggroll_es_v401_train_only_candidate_v34a as candidate


def test_v401_train_only_candidate_rebuilds_exactly_without_sealed_inputs(tmp_path):
    output = tmp_path / "candidate.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rebuilt = candidate.build_candidate(output, manifest_path)
    frozen = json.loads(candidate.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    assert output.read_bytes() == candidate.OUTPUT_CANDIDATE.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
    )
    assert rebuilt == frozen
    replay = rebuilt["train_only_replay"]
    assert replay["versions"] == [390, 401]
    assert len(replay["curation_artifacts"]) == 12
    assert replay["total_edits"] == 36
    assert replay["total_drops_or_additions"] == 0
    assert replay["collision_fact_set_count"] == 0
    assert replay["validation_heldout_ood_or_benchmark_file_opened"] is False
    assert rebuilt["candidate"]["rows"] == 531
    assert rebuilt["contains_row_content"] is False


def test_v401_materializer_has_no_sealed_file_path_or_fact_loader():
    source = inspect.getsource(candidate)
    assert "eval_qa" not in source
    assert "ood_qa" not in source
    assert "heldout." not in source
    assert "benchmark." not in source
    assert "eval_facts(" not in source
    assert "frozenset()" in inspect.getsource(candidate.build_candidate)


def test_v401_manifest_is_content_free_self_sealed_and_history_bound():
    value = json.loads(candidate.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    assert value["content_sha256_before_self_field"] == candidate.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    assert value["content_sha256_before_self_field"] == (
        "42304107e89119c10c545dc79b4f85ab08bd4b8b78efec710d286987a3e8a5af"
    )
    assert candidate.file_sha256(candidate.OUTPUT_MANIFEST) == (
        "1013032a1a4c21a2ece6e80e0930aecee8430639e07eaf48c2ac7701708e8f52"
    )
    assert value["curator_snapshot"]["commit"] == (
        "d7abea3540bd1cb43d725ec94772385821d2cee4"
    )
    assert value["source_candidate"]["freeze_commit"] == (
        "c54cbf4cdea670f4044a6dc5fb035eb20face83c"
    )
    assert value["source_candidate"]["manifest_content_sha256"] == (
        "7eb3b179b79ee499c2a1ca7676b9b2a7a4122577c42786432250761afd3bed8f"
    )


def test_v401_all_edit_artifacts_are_exact_and_contiguous():
    value = json.loads(candidate.OUTPUT_MANIFEST.read_text(encoding="utf-8"))
    records = value["train_only_replay"]["curation_artifacts"]
    assert [item["version"] for item in records] == list(range(390, 402))
    assert all(item["edit_count"] == 3 for item in records)
    assert {item["file_sha256"] for item in records} == set(
        candidate.CURATION_SHA256_BY_VERSION.values()
    )
