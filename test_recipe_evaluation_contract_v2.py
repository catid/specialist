"""Synthetic-only regression checks for the V2 security primitives.

These tests never build the production contract and never resolve a real
terminal source path.
"""

import json
import os

import pytest

import recipe_evaluation_contract_v2 as subject


def _synthetic_module_level_evaluator(_rows, _claim):
    return {}


def _record(*, document, url, lineage, tokens):
    tokens = tuple(tokens)
    return {
        "documents": frozenset((document,)),
        "urls": frozenset((url,)),
        "lineages": frozenset((lineage,)),
        "qa_features": None,
        "tokens": tokens,
        "ngrams": subject._token_ngrams(tokens),
    }


def test_opaque_role_audit_reports_each_identity_domain_without_content():
    left = _record(
        document="d-left", url="web://left.invalid/", lineage="l-left",
        tokens=("synthetic", "left", "record"),
    )
    right = _record(
        document="d-right", url="web://right.invalid/", lineage="l-right",
        tokens=("synthetic", "right", "record"),
    )
    roles = {name: [] for name in subject.ROLE_ORDER}
    roles["train"] = [left]
    roles["protected_terminal"] = [right]
    result = subject.audit_role_records(roles)
    assert result["passed"] is True
    assert result["pairs"]["train__protected_terminal"]["colliding_row_pairs"] == 0

    roles["protected_terminal"] = [dict(right, documents=left["documents"])]
    result = subject.audit_role_records(roles)
    pair = result["pairs"]["train__protected_terminal"]
    assert result["passed"] is False
    assert pair["by_identity_domain"]["document_sha256"] == 1


def test_claim_and_load_is_one_shot_and_claim_precedes_synthetic_loader(tmp_path):
    state = tmp_path / "claim.json"
    observations = []

    def loader():
        observations.append(state.exists())
        return [{"synthetic": True}]

    rows, consumed = subject._claim_and_load_once(
        state,
        contract_content_sha256="c" * 64,
        selection_content_sha256="s" * 64,
        selected_identity_set_sha256="i" * 64,
        loader=loader,
    )
    assert rows == [{"synthetic": True}]
    assert observations == [True]
    assert consumed["status"] == "source_open_completed_access_consumed_no_retry"
    persisted = json.loads(state.read_text())
    assert persisted["access_count"] == 1
    assert persisted["source_open_attempted"] is True

    with pytest.raises(FileExistsError):
        subject._claim_and_load_once(
            state,
            contract_content_sha256="c" * 64,
            selection_content_sha256="s" * 64,
            selected_identity_set_sha256="i" * 64,
            loader=lambda: pytest.fail("retry must not invoke loader"),
        )


def test_loader_failure_consumes_claim_and_cannot_retry(tmp_path):
    state = tmp_path / "claim.json"

    def fail():
        raise SyntheticFailure("synthetic loader failure")

    class SyntheticFailure(RuntimeError):
        pass

    with pytest.raises(SyntheticFailure):
        subject._claim_and_load_once(
            state,
            contract_content_sha256="c" * 64,
            selection_content_sha256="s" * 64,
            selected_identity_set_sha256="i" * 64,
            loader=fail,
        )
    persisted = json.loads(state.read_text())
    assert persisted["status"] == "source_open_failed_access_consumed_no_retry"
    assert persisted["source_open_attempted"] is True
    with pytest.raises(FileExistsError):
        subject._claim_and_load_once(
            state,
            contract_content_sha256="c" * 64,
            selection_content_sha256="s" * 64,
            selected_identity_set_sha256="i" * 64,
            loader=lambda: [],
        )


def test_split_claim_and_reopen_apis_are_prohibited():
    with pytest.raises(RuntimeError, match="split claim/load"):
        subject.claim_protected_access_once()
    with pytest.raises(RuntimeError, match="reopening"):
        subject.load_claimed_protected_rows()
    with pytest.raises(RuntimeError, match="returning raw protected rows"):
        subject.claim_and_load_protected_rows()


def test_aggregate_evaluator_suppresses_accidental_output():
    def noisy(_rows, _claim):
        print("synthetic output must not escape")
        return {}

    with pytest.raises(RuntimeError, match="emitted output"):
        subject._run_aggregate_evaluator(
            [{"synthetic": True}],
            {"synthetic_claim": True},
            noisy,
            lambda _receipt: pytest.fail("noisy receipt must not validate"),
        )


def test_aggregate_evaluator_suppresses_exception_details():
    marker = "synthetic-sensitive-marker"

    def failing(_rows, _claim):
        raise RuntimeError(marker)

    with pytest.raises(RuntimeError, match="details suppressed") as error:
        subject._run_aggregate_evaluator(
            [{"synthetic": True}],
            {"synthetic_claim": True},
            failing,
            lambda _receipt: None,
        )
    assert marker not in str(error.value)


def test_terminal_evaluator_must_match_reviewed_module_level_source():
    source = subject.Path(__file__).resolve()
    selection = {
        "terminal_evaluator_file_sha256": subject.file_sha256(source),
        "terminal_evaluator_path_identity_sha256": subject._identity(
            "terminal-aggregate-evaluator-path-v2",
            str(source.relative_to(subject.ROOT)),
        ),
    }
    subject._validate_evaluator_binding(
        _synthetic_module_level_evaluator, selection
    )

    def nested(_rows, _claim):
        return {}

    with pytest.raises(RuntimeError, match="reviewed module-level"):
        subject._validate_evaluator_binding(nested, selection)


def test_selection_receipt_is_exact_aggregate_only_schema():
    contract = {
        "content_sha256_before_self_field": "c" * 64,
        "roles": {"dev": {"source_identity_set_sha256": "a" * 64}},
    }
    selection = {
        "schema": subject.SELECTION_SCHEMA,
        "status": "recipe_selected_frozen_hpo_closed",
        "contract_content_sha256": "c" * 64,
        "hpo_closed": True,
        "v2_terminal_access_count_before_selection": 0,
        "v1_quarantined_access_count_nonzero_acknowledged": True,
        "incident_content_sha256": subject.INCIDENT_CONTENT_SHA256,
        "legacy_eval_collision_incident_content_sha256": (
            subject.LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
        ),
        "recursive_lookup_incident_content_sha256": (
            subject.RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256
        ),
        "superseded_v2_nonpromotable_acknowledged": True,
        "fresh_dev_source_identity_set_sha256": "a" * 64,
        "selected_recipe_id": "synthetic-recipe",
        "selected_checkpoint_sha256": "d" * 64,
        "terminal_evaluator_file_sha256": "e" * 64,
        "terminal_evaluator_path_identity_sha256": "f" * 64,
        "terminal_evaluator_aggregate_only_reviewed": True,
        "terminal_evaluator_side_effect_reviewed": True,
    }
    selection["content_sha256_before_self_field"] = subject.canonical_sha256(
        selection
    )
    assert subject._validate_selection_receipt(selection, contract) == (
        selection["content_sha256_before_self_field"]
    )
    selection["text"] = "synthetic forbidden extra field"
    selection["content_sha256_before_self_field"] = subject.canonical_sha256({
        key: value for key, value in selection.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="valid V2 frozen selection"):
        subject._validate_selection_receipt(selection, contract)


def test_content_minimizer_detects_semantic_fields_but_allows_opaque_metadata():
    opaque = {
        "opaque_item_identity": "a" * 64,
        "url_identity_set_sha256": "b" * 64,
        "source_path_identity_sha256": "c" * 64,
    }
    assert subject._contains_semantic_key(opaque) is False
    assert subject._contains_semantic_key({"text": "synthetic only"}) is True
    assert subject._contains_semantic_key(
        {"relative_path": "synthetic/source.bin"}
    ) is True


def test_future_dataset_registry_rejects_rows_without_source_provenance():
    with pytest.raises(RuntimeError, match="lacks auditable source provenance"):
        subject.build_future_dataset_identity_registry(
            [{"synthetic_payload": True}], role="train"
        )


@pytest.mark.parametrize("role", sorted(subject.FUTURE_ADAPTATION_ROLES))
def test_every_future_adaptation_role_builds_only_opaque_identity_registry(role):
    registry = subject.build_future_dataset_identity_registry(
        [{"source_document_sha256": "d" * 64}], role=role
    )
    assert registry["role"] == role
    assert registry["all_rows_have_auditable_source_provenance"] is True
    assert registry["identity_domains"]["document_sha256"] == ["d" * 64]
    assert subject._contains_semantic_key(registry) is False


def test_selected_identity_is_reserved_but_unselected_identity_remains_available(
    monkeypatch,
):
    selected = "a" * 64
    unselected = "b" * 64
    reserved = {
        name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
    }
    reserved["document_sha256"] = frozenset((selected,))
    contract = {
        "roles": {
            "protected_terminal": {
                "future_adaptation_reservation": {
                    "content_sha256_before_self_field": "c" * 64,
                }
            }
        }
    }
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject, "_validated_reservation", lambda _contract: reserved
    )
    monkeypatch.setattr(
        subject,
        "_validated_fresh_dev_reservation",
        lambda _contract: {
            name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
        },
    )

    clear = subject.build_future_dataset_identity_registry(
        [{"source_document_sha256": unselected}], role="cpt"
    )
    result = subject.assert_future_dataset_excludes_reserved(clear, contract)
    assert result["passed"] is True
    assert result["unselected_terminal_eligible_sources"] == 0
    assert all(value == 0 for value in result["intersection_counts"].values())

    collision = subject.build_future_dataset_identity_registry(
        [{"source_document_sha256": selected}], role="qa"
    )
    with pytest.raises(RuntimeError, match="reserved V2 terminal"):
        subject.assert_future_dataset_excludes_reserved(collision, contract)


def test_legacy_eval_path_identity_is_quarantined_for_every_future_role(
    monkeypatch,
):
    reserved = {
        name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
    }
    contract = {
        "roles": {
            "protected_terminal": {
                "future_adaptation_reservation": {
                    "content_sha256_before_self_field": "c" * 64,
                }
            }
        }
    }
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject, "_validated_reservation", lambda _contract: reserved
    )
    monkeypatch.setattr(
        subject,
        "_validated_fresh_dev_reservation",
        lambda _contract: {
            name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
        },
    )
    for role in sorted(subject.FUTURE_ADAPTATION_ROLES):
        registry = subject.build_future_dataset_identity_registry(
            [{"source_relative_path": "data/eval_qa.jsonl"}], role=role
        )
        with pytest.raises(RuntimeError, match="quarantined legacy evaluation"):
            subject.assert_future_dataset_excludes_reserved(registry, contract)


def test_direct_legacy_eval_adaptation_input_rejects_before_path_resolution(
    monkeypatch,
):
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject.Path,
        "resolve",
        lambda *_args, **_kwargs: pytest.fail(
            "quarantined legacy eval path was resolved"
        ),
    )
    contract = {"roles": {"train": {"file_sha256": "a" * 64}}}
    with pytest.raises(RuntimeError, match="quarantined legacy evaluation"):
        subject.assert_adaptation_inputs(
            [subject.ROOT / "data/eval_qa_v2.jsonl"], contract
        )


def test_direct_v1_eval_adaptation_rejects_before_path_resolution(monkeypatch):
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject.Path,
        "resolve",
        lambda *_args, **_kwargs: pytest.fail("quarantined V1 path was resolved"),
    )
    contract = {"roles": {"train": {"file_sha256": "a" * 64}}}
    with pytest.raises(RuntimeError, match="quarantined V1"):
        subject.assert_adaptation_inputs(
            [subject.ROOT / "data/eval_qa_v3.jsonl"], contract
        )


@pytest.mark.parametrize("role", sorted(subject.FUTURE_ADAPTATION_ROLES))
def test_recursive_access_prefix_is_rejected_before_registry_write(role):
    with pytest.raises(RuntimeError, match="recursively accessed prefix"):
        subject.build_future_dataset_identity_registry(
            [{
                "source_relative_path": (
                    "experiments/sft_controls/synthetic/future.jsonl"
                )
            }],
            role=role,
        )


def test_nonallowlisted_prefix_adaptation_rejects_before_resolution(monkeypatch):
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject.Path,
        "resolve",
        lambda *_args, **_kwargs: pytest.fail(
            "nonallowlisted recursively accessed path was resolved"
        ),
    )
    contract = {"roles": {"train": {"file_sha256": "a" * 64}}}
    with pytest.raises(RuntimeError, match="explicit post-incident allowlist"):
        subject.assert_adaptation_inputs(
            [subject.ROOT / "experiments/sft_controls/synthetic/dev.jsonl"],
            contract,
        )


def test_resolved_alias_into_accessed_prefix_rejects_before_file_probe(
    monkeypatch, tmp_path
):
    alias = tmp_path / "synthetic-alias.jsonl"
    resolved = subject.ROOT / "experiments/sft_controls/synthetic/dev.jsonl"
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(subject.Path, "resolve", lambda _self: resolved)
    monkeypatch.setattr(
        subject.Path,
        "is_file",
        lambda _self: pytest.fail("resolved prefix alias reached file probe"),
    )
    contract = {"roles": {"train": {"file_sha256": "a" * 64}}}
    with pytest.raises(RuntimeError, match="resolved adaptation path"):
        subject.assert_adaptation_inputs([alias], contract)


def test_future_source_paths_require_forward_slash_lexical_form():
    with pytest.raises(RuntimeError, match="root-relative slashes"):
        subject.build_future_dataset_identity_registry(
            [{"source_relative_path": "safe\\synthetic.jsonl"}], role="train"
        )


def test_fresh_dev_identity_is_reserved_from_future_adaptation(monkeypatch):
    empty = {
        name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
    }
    fresh = dict(empty)
    fresh["opaque_item_identity"] = frozenset(("d" * 64,))
    contract = {
        "roles": {
            "protected_terminal": {
                "future_adaptation_reservation": {
                    "content_sha256_before_self_field": "f" * 64,
                }
            }
        }
    }
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject, "_validated_reservation", lambda _contract: empty
    )
    monkeypatch.setattr(
        subject, "_validated_fresh_dev_reservation", lambda _contract: fresh
    )
    registry = subject.build_future_dataset_identity_registry(
        [{"source_opaque_item_identity": "d" * 64}], role="sft"
    )
    with pytest.raises(RuntimeError, match="reserved fresh V2 dev"):
        subject.assert_future_dataset_excludes_reserved(registry, contract)


@pytest.mark.parametrize(
    "domain,row_field",
    (
        ("opaque_item_identity", "source_opaque_item_identity"),
        ("document_sha256", "source_document_sha256"),
        ("source_path_identity_sha256", "source_path_identity_sha256"),
        (
            "normalized_url_identity_sha256",
            "normalized_url_identity_sha256",
        ),
        ("raw_lineage_identity_sha256", "raw_lineage_identity_sha256"),
    ),
)
def test_every_opaque_reservation_domain_fails_closed(
    monkeypatch, domain, row_field
):
    selected = "e" * 64
    reserved = {
        name: frozenset((selected,)) if name == domain else frozenset()
        for name in subject.RESERVED_IDENTITY_DOMAINS
    }
    contract = {
        "roles": {
            "protected_terminal": {
                "future_adaptation_reservation": {
                    "content_sha256_before_self_field": "f" * 64,
                }
            }
        }
    }
    monkeypatch.setattr(subject, "validate_contract", lambda _contract: None)
    monkeypatch.setattr(
        subject, "_validated_reservation", lambda _contract: reserved
    )
    monkeypatch.setattr(
        subject,
        "_validated_fresh_dev_reservation",
        lambda _contract: {
            name: frozenset() for name in subject.RESERVED_IDENTITY_DOMAINS
        },
    )
    registry = subject.build_future_dataset_identity_registry(
        [{row_field: selected}], role="train"
    )
    with pytest.raises(RuntimeError, match="reserved V2 terminal"):
        subject.assert_future_dataset_excludes_reserved(registry, contract)


def test_compute_ledger_counts_synthetic_attempts_only():
    attempt = {
        "arm": "synthetic-a",
        "gpu_residency_intervals": [
            {"physical_gpu_id": gpu, "start_s": 1.0, "end_s": 3.0}
            for gpu in range(4)
        ],
        "optimization_generated_rollouts": 4,
        "evaluation_generated_rollouts": 2,
        "generated_tokens": 10,
        "teacher_forced_tokens": 20,
        "sft_nonpadding_tokens": 30,
    }
    totals = subject.aggregate_compute_ledger([attempt])
    assert totals["synthetic-a"]["charged_gpu_seconds"] == 8.0
    assert totals["synthetic-a"]["optimization_generated_rollouts"] == 4


def test_production_reseal_denies_accessed_prefixes_and_is_prose_only(
    monkeypatch,
):
    prefixes = tuple(
        subject.Path(os.path.abspath(subject.ROOT / value))
        for value in subject.RECURSIVELY_ACCESSED_RELATIVE_PREFIXES
    )

    def implicated(value):
        path = subject.Path(os.path.abspath(os.fspath(value)))
        return any(path == prefix or prefix in path.parents for prefix in prefixes)

    original_open = subject.Path.open
    original_stat = subject.Path.stat
    original_resolve = subject.Path.resolve

    def guarded_open(path, *args, **kwargs):
        if implicated(path):
            pytest.fail("V2 reseal opened an accessed prefix")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        if implicated(path):
            pytest.fail("V2 reseal statted an accessed prefix")
        return original_stat(path, *args, **kwargs)

    def guarded_resolve(path, *args, **kwargs):
        if implicated(path):
            pytest.fail("V2 reseal resolved an accessed prefix")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(subject.Path, "open", guarded_open)
    monkeypatch.setattr(subject.Path, "stat", guarded_stat)
    monkeypatch.setattr(subject.Path, "resolve", guarded_resolve)
    contract = subject.build_contract()
    subject.validate_contract(contract)
    assert contract["roles"]["dev"]["documents"] == 4
    assert contract["roles"]["protected_terminal"]["documents"] == 9
    assert contract["authority"][
        "qa_hpo_or_general_quality_promotion_authorized"
    ] is False
    reservation = contract["roles"]["protected_terminal"][
        "future_adaptation_reservation"
    ]
    assert reservation["reserved_source_count"] == 13
    assert reservation["historical_selected_source_count"] == 12
