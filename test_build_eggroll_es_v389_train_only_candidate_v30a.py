import hashlib
import inspect
import json

import build_eggroll_es_v389_train_only_candidate_v30a as candidate


def test_v389_train_only_candidate_rebuilds_exactly_without_sealed_inputs(tmp_path):
    output = tmp_path / "candidate.jsonl"
    manifest_path = tmp_path / "manifest.json"
    rebuilt = candidate.build_candidate_v389(output, manifest_path)
    frozen = json.loads(candidate.OUTPUT_MANIFEST_V389.read_text(encoding="utf-8"))
    assert output.read_bytes() == candidate.OUTPUT_CANDIDATE_V389.read_bytes()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == (
        "4b6da77e7e1ae3d1145b3f2d29c7774b6aad2b4cb520fcea9a48af93d4322388"
    )
    assert rebuilt == frozen
    replay = rebuilt["train_only_replay"]
    assert replay["total_edits"] == 75
    assert replay["total_drops_or_additions"] == 0
    assert replay["collision_fact_set_count"] == 0
    assert replay["validation_heldout_ood_or_benchmark_file_opened"] is False
    assert rebuilt["candidate"]["rows"] == 531
    assert rebuilt["contains_row_content"] is False


def test_v389_materializer_has_no_sealed_file_path_or_eval_fact_loader():
    source = inspect.getsource(candidate)
    assert "eval_qa" not in source
    assert "ood_qa" not in source
    assert "heldout." not in source
    assert "benchmark." not in source
    assert "eval_facts(" not in source
    assert "frozenset()" in inspect.getsource(candidate.build_candidate_v389)


def test_v389_manifest_is_content_free_and_self_sealed():
    value = json.loads(candidate.OUTPUT_MANIFEST_V389.read_text(encoding="utf-8"))
    assert value["content_sha256_before_self_field"] == candidate.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    assert value["curator_snapshot"]["commit"] == (
        "2e8a6b7d02fbc77a2442f6790fe0f80f1bebc02e"
    )
    assert value["curator_snapshot"]["report_file_sha256"] == (
        "809cbe853b24129c9d957bc20219a61c0c7ba14044c49e48f5541c71a946349f"
    )
    assert len(value["train_only_replay"]["curation_artifacts"]) == 25
