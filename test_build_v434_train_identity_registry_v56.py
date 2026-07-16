from __future__ import annotations

import build_v434_train_identity_registry_v56 as subject


def test_v56_registry_is_deterministic_and_content_minimized():
    first = subject.build_registry_v56()
    second = subject.build_registry_v56()
    assert first == second
    assert first["schema"] == "v434-train-disjoint-identity-registry-v56"
    assert first["aggregate"]["rows"] == 448
    assert first["identity_domains"] == [
        "document_sha256", "normalized_url", "raw_lineage", "semantic_cluster",
    ]
    assert first["content_minimization"] == {
        "question_persisted": False,
        "answer_persisted": False,
        "evidence_persisted": False,
        "document_text_persisted": False,
        "only_hashes_normalized_provenance_and_lexical_feature_sets": True,
    }
    forbidden = {"question", "answer", "evidence", "text"}
    assert all(not (forbidden & set(item)) for item in first["items"])


def test_v56_registry_binds_exact_v434_membership_without_nontrain_access():
    value = subject.build_registry_v56()
    assert value["source"]["file_sha256"] == subject.TRAIN_FILE_SHA256
    assert value["source"]["membership_file_sha256"] == (
        subject.MEMBERSHIP_FILE_SHA256
    )
    assert value["access_receipt"][
        "ood_shadow_holdout_or_benchmark_semantics_opened"
    ] is False
    assert value["access_receipt"]["gpu_accessed"] is False
    assert value["access_receipt"]["model_outcomes_used"] is False
