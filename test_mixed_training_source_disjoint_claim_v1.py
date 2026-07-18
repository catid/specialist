from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile
from unittest import mock

import pytest

import mixed_training_source_disjoint_claim_v1 as claim


def _sealed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["content_sha256_before_self_field"] = claim.canonical_sha256(result)
    return result


def _write_artifact(path: Path, value: dict) -> bytes:
    payload = claim.artifact_bytes(value)
    path.write_bytes(payload)
    return payload


def _chat_unit(component: str, unit: str, group: str, document: str, token: int) -> dict:
    return {
        "component": component,
        "unit_id": unit,
        "source_group_id": group,
        "source_document_id": document,
        "training_format": "chat_assistant_only",
        "input_ids": [token, token + 1],
        "labels": [-100, token + 1],
        "budget_token_count": 1,
    }


def _markdown_unit(unit: str, group: str, document: str, token: int) -> dict:
    return {
        "component": "raw_markdown",
        "unit_id": unit,
        "source_group_id": group,
        "source_document_id": document,
        "training_format": "raw_markdown_causal",
        "input_ids": [token, token + 1],
        "labels": [token, token + 1],
        "budget_token_count": 2,
    }


def _candidate_units() -> dict[str, list[dict]]:
    seed = _chat_unit("seed_qa", "unit-seed", "group-seed", "doc-seed", 10)
    generated = _chat_unit(
        "generated_domain", "unit-generated", "group-generated", "doc-generated", 20
    )
    replay = _chat_unit("replay", "unit-replay", "group-replay", "doc-replay", 30)
    return {
        "protocol_core_100k": [
            copy.deepcopy(seed),
            copy.deepcopy(generated),
            _markdown_unit("unit-core", "group-core", "doc-core", 40),
            copy.deepcopy(replay),
        ],
        "full_authorized_markdown": [
            copy.deepcopy(seed),
            copy.deepcopy(generated),
            _markdown_unit("unit-full", "group-full", "doc-full", 50),
            copy.deepcopy(replay),
        ],
    }


@contextmanager
def _fixture():
    # Do not use pytest's test-name-derived temporary paths: several adversarial
    # test names deliberately contain path-class words that the firewall rejects.
    with tempfile.TemporaryDirectory(prefix="sdclaim-") as directory:
        root = Path(directory)
        paths = {
            role: root / f"input-{index:02d}-{role}.bin"
            for index, role in enumerate(claim.STATIC_INPUT_ROLES)
        }
        for index, role in enumerate(claim.STATIC_INPUT_ROLES):
            paths[role].write_bytes(f"synthetic-{index}-{role}\n".encode("utf-8"))

        decision_payload = b'{"schema":"synthetic-decision"}\n'
        paths["seed_qa_decision_bundle"].write_bytes(decision_payload)
        admitted_commitment = "a" * 64
        seed_authority = _sealed(
            {
                "schema": "synthetic-seed-authority-v1",
                "decision_bundle": {
                    "file_sha256": claim.sha256_bytes(decision_payload)
                },
                "admitted_record_identity_commitment_sha256": admitted_commitment,
            }
        )
        _write_artifact(paths["seed_qa_semantic_authority"], seed_authority)

        train_groups = frozenset(
            {"group-seed", "group-generated", "group-core", "group-full"}
        )
        train_documents = frozenset(
            {"doc-seed", "doc-generated", "doc-core", "doc-full"}
        )
        replay_groups = frozenset({"group-replay"})
        train_group_commitment = claim.canonical_sha256(sorted(train_groups))
        split_authority = _sealed(
            {
                "schema": "specialist-source-group-split-authority-v1",
                "status": "sealed_source_disjoint_assignment_launch_still_gated",
                "assignments": {
                    "train": {
                        "source_group_membership_commitment_sha256": (
                            train_group_commitment
                        ),
                        "records": [
                            {"source_group_id": item}
                            for item in sorted(train_groups)
                        ],
                    },
                    "final": {
                        "source_group_membership_commitment_sha256": "c" * 64,
                        "records_redacted": True,
                    },
                },
            }
        )
        _write_artifact(paths["source_split_authority"], split_authority)

        generated_data = b'{"schema":"synthetic-generated-row"}\n'
        paths["generated_data"].write_bytes(generated_data)
        generated_report = _sealed(
            {
                "schema": "synthetic-generated-report-v1",
                "status": "sealed",
            }
        )
        report_payload = _write_artifact(
            paths["generated_report"], generated_report
        )
        replacement = {
            "schema": "seed-qa-generated-replacement-receipt-v1",
            "replacement_assistant_tokens_required": 2,
            "replacement_assistant_tokens_selected": 2,
        }
        generated_manifest = _sealed(
            {
                "schema": "synthetic-generated-manifest-v1",
                "dataset": {"file_sha256": claim.sha256_bytes(generated_data)},
                "report": {"file_sha256": claim.sha256_bytes(report_payload)},
                "seed_qa_replacement": replacement,
            }
        )
        _write_artifact(paths["generated_manifest"], generated_manifest)

        runner = root / "claim-runner.py"
        runner.write_text("# synthetic claim runner fixture\n", encoding="utf-8")

        source_split_receipt = {
            "path": claim.SOURCE_SPLIT_DECLARED_PATH,
            "file_sha256": claim.file_sha256(paths["source_split_authority"]),
            "content_sha256": split_authority[
                "content_sha256_before_self_field"
            ],
            "train_source_group_membership_commitment_sha256": (
                train_group_commitment
            ),
            "final_source_group_membership_commitment_sha256": "c" * 64,
            "final_records_redacted": True,
        }
        generated_receipt = {
            "manifest_file_sha256": claim.file_sha256(paths["generated_manifest"]),
            "manifest_content_sha256": generated_manifest[
                "content_sha256_before_self_field"
            ],
            "report_file_sha256": claim.file_sha256(paths["generated_report"]),
            "report_content_sha256": generated_report[
                "content_sha256_before_self_field"
            ],
            "dataset_file_sha256": claim.file_sha256(paths["generated_data"]),
            "seed_replacement_receipt_sha256": claim.canonical_sha256(replacement),
        }
        seed_receipt = {
            "authority_file_sha256": claim.file_sha256(
                paths["seed_qa_semantic_authority"]
            ),
            "authority_content_sha256": seed_authority[
                "content_sha256_before_self_field"
            ],
            "decision_bundle_file_sha256": claim.file_sha256(
                paths["seed_qa_decision_bundle"]
            ),
            "admitted_record_identity_commitment_sha256": admitted_commitment,
        }
        units = _candidate_units()
        request, request_payload = claim.build_claim_request(
            static_input_paths=paths,
            expected_static_input_paths=paths,
            units_by_variant=units,
            source_split_authority=source_split_receipt,
            generated_domain_authority=generated_receipt,
            seed_qa_semantic_authority=seed_receipt,
            runner_path=runner,
            expected_runner_path=runner,
            train_source_group_ids=train_groups,
            train_source_document_ids=train_documents,
            replay_source_group_ids=replay_groups,
        )
        request_path = root / "claim-request.json"
        request_path.write_bytes(request_payload)
        yield {
            "root": root,
            "paths": paths,
            "runner": runner,
            "source_split": source_split_receipt,
            "generated": generated_receipt,
            "seed": seed_receipt,
            "units": units,
            "request": request,
            "request_payload": request_payload,
            "request_path": request_path,
            "request_sha256": claim.sha256_bytes(request_payload),
            "train_groups": train_groups,
            "train_documents": train_documents,
            "replay_groups": replay_groups,
        }


def _authorization(fixture: dict) -> tuple[dict, bytes, Path, str]:
    authorization, payload = claim.build_claim_authorization(
        request_path=fixture["request_path"],
        expected_request_path=fixture["request_path"],
        expected_request_sha256=fixture["request_sha256"],
        static_input_paths=fixture["paths"],
        expected_static_input_paths=fixture["paths"],
        units_by_variant=fixture["units"],
        source_split_authority=fixture["source_split"],
        generated_domain_authority=fixture["generated"],
        seed_qa_semantic_authority=fixture["seed"],
        runner_path=fixture["runner"],
        expected_runner_path=fixture["runner"],
        train_source_group_ids=fixture["train_groups"],
        train_source_document_ids=fixture["train_documents"],
        replay_source_group_ids=fixture["replay_groups"],
    )
    path = fixture["root"] / "claim-authorization.json"
    path.write_bytes(payload)
    return authorization, payload, path, claim.sha256_bytes(payload)


def _extension(fixture: dict) -> tuple[dict, bytes, dict, str]:
    authorization, _, authorization_path, authorization_sha = _authorization(fixture)
    extension, payload = claim.build_extension(
        request_path=fixture["request_path"],
        expected_request_path=fixture["request_path"],
        expected_request_sha256=fixture["request_sha256"],
        authorization_path=authorization_path,
        expected_authorization_path=authorization_path,
        expected_authorization_sha256=authorization_sha,
        static_input_paths=fixture["paths"],
        expected_static_input_paths=fixture["paths"],
        units_by_variant=fixture["units"],
        source_split_authority=fixture["source_split"],
        generated_domain_authority=fixture["generated"],
        seed_qa_semantic_authority=fixture["seed"],
        runner_path=fixture["runner"],
        expected_runner_path=fixture["runner"],
        train_source_group_ids=fixture["train_groups"],
        train_source_document_ids=fixture["train_documents"],
        replay_source_group_ids=fixture["replay_groups"],
    )
    return extension, payload, authorization, authorization_sha


def _reseal(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    return _sealed(result)


def test_claim_request_binds_both_variant_candidate_sets_and_every_static_input():
    with _fixture() as fixture:
        request = fixture["request"]
        assert set(request["candidate_sets"]) == set(claim.VARIANTS)
        assert [item["role"] for item in request["static_inputs"]["bindings"]] == list(
            claim.STATIC_INPUT_ROLES
        )
        assert request["static_inputs"]["bindings_commitment_sha256"] == (
            claim.canonical_sha256(request["static_inputs"]["bindings"])
        )
        serialized = claim.artifact_bytes(request)
        for forbidden in (b"unit-seed", b"group-seed", b"doc-seed"):
            assert forbidden not in serialized


def test_claim_request_rejects_unknown_or_nontrain_domain_source_group():
    with _fixture() as fixture:
        fixture["train_groups"] = frozenset(
            fixture["train_groups"] - {"group-generated"}
        )
        with pytest.raises(claim.ContractError, match="sealed split authority"):
            _authorization(fixture)


@pytest.mark.parametrize(
    "mutation",
    ("generated_data", "seed_decision_bundle", "replacement_receipt"),
)
def test_claim_request_rejects_stale_generated_seed_or_replacement_receipt(mutation):
    with _fixture() as fixture:
        if mutation == "generated_data":
            fixture["paths"]["generated_data"].write_bytes(b"changed generated bytes\n")
        elif mutation == "seed_decision_bundle":
            fixture["paths"]["seed_qa_decision_bundle"].write_bytes(
                b"changed seed decisions\n"
            )
        else:
            manifest_path = fixture["paths"]["generated_manifest"]
            manifest = json.loads(manifest_path.read_bytes())
            manifest["seed_qa_replacement"]["replacement_assistant_tokens_selected"] = 1
            manifest_path.write_bytes(claim.artifact_bytes(_reseal(manifest)))
        with pytest.raises(claim.ContractError):
            _authorization(fixture)


@pytest.mark.parametrize("forbidden", ("development", "final", "protected"))
def test_claim_request_never_opens_development_final_or_protected_paths(forbidden):
    with tempfile.TemporaryDirectory(prefix="sdclaim-") as directory:
        path = Path(directory) / forbidden / "candidate.bin"
        expected = path
        with mock.patch.object(
            Path, "lstat", side_effect=AssertionError("forbidden path was inspected")
        ):
            with pytest.raises(claim.ContractError, match="forbidden path class"):
                claim.secure_regular_file(path, expected=expected, label="synthetic")


def test_claim_runner_emits_only_exact_content_free_schema():
    with _fixture() as fixture:
        authorization, payload, _, _ = _authorization(fixture)
        assert set(authorization) == claim.AUTHORIZATION_KEYS
        assert authorization["claims"] == claim.ZERO_CLAIMS
        assert authorization["boundary"] == claim.OUTPUT_BOUNDARY
        assert authorization["opaque_receipt_sha256"] == claim.canonical_sha256(
            claim._opaque_payload(authorization)
        )
        for forbidden in (
            b"unit-seed",
            b"group-seed",
            b"doc-seed",
            b"https://",
            b'"question"',
            b'"answer"',
        ):
            assert forbidden not in payload


def test_claim_runner_rejects_candidate_commitment_mismatch():
    with _fixture() as fixture:
        fixture["units"]["protocol_core_100k"][0]["input_ids"][0] += 1
        with pytest.raises(
            claim.ContractError,
            match="different cross-variant content|live candidate inputs",
        ):
            _authorization(fixture)


@pytest.mark.parametrize("authority", ("document", "replay"))
def test_claim_runner_rejects_train_membership_miss_or_nonzero_collision(authority):
    with _fixture() as fixture:
        if authority == "document":
            fixture["train_documents"] = frozenset(
                fixture["train_documents"] - {"doc-core"}
            )
        else:
            fixture["replay_groups"] = frozenset()
        with pytest.raises(
            claim.ContractError,
            match="live candidate inputs|frozen string set|aggregate counts",
        ):
            _authorization(fixture)


@pytest.mark.parametrize("authority", ("document", "replay"))
def test_preregistered_membership_cannot_be_extended_with_candidate_only_ids(
    authority,
):
    with _fixture() as fixture:
        if authority == "document":
            for variant in claim.VARIANTS:
                fixture["units"][variant][1]["source_document_id"] = (
                    "candidate-only-document"
                )
            fixture["train_documents"] = frozenset({
                *fixture["train_documents"],
                "candidate-only-document",
            })
        else:
            for variant in claim.VARIANTS:
                fixture["units"][variant][-1]["source_group_id"] = (
                    "candidate-only-replay-group"
                )
            fixture["replay_groups"] = frozenset({
                *fixture["replay_groups"],
                "candidate-only-replay-group",
            })
        with pytest.raises(claim.ContractError, match="live candidate inputs"):
            _authorization(fixture)


def test_claim_authorization_rejects_unknown_semantic_or_protected_fields():
    with _fixture() as fixture:
        authorization, _, _, _ = _authorization(fixture)
        forged = copy.deepcopy(authorization)
        forged["question"] = "synthetic content that must never be carried"
        forged = _reseal(forged)
        with pytest.raises(claim.ContractError, match="exact field"):
            claim.validate_claim_authorization(forged)
        forged = copy.deepcopy(authorization)
        forged["boundary"]["protected_identifiers"] = ["synthetic-id"]
        forged = _reseal(forged)
        with pytest.raises(claim.ContractError, match="exact field"):
            claim.validate_claim_authorization(forged)


def test_extension_requires_independently_pinned_authorization_sha256():
    with _fixture() as fixture:
        _, _, path, _ = _authorization(fixture)
        with pytest.raises(claim.ContractError, match="independently supplied"):
            claim.build_extension(
                request_path=fixture["request_path"],
                expected_request_path=fixture["request_path"],
                expected_request_sha256=fixture["request_sha256"],
                authorization_path=path,
                expected_authorization_path=path,
                expected_authorization_sha256="0" * 64,
                static_input_paths=fixture["paths"],
                expected_static_input_paths=fixture["paths"],
                units_by_variant=fixture["units"],
                source_split_authority=fixture["source_split"],
                generated_domain_authority=fixture["generated"],
                seed_qa_semantic_authority=fixture["seed"],
                runner_path=fixture["runner"],
                expected_runner_path=fixture["runner"],
                train_source_group_ids=fixture["train_groups"],
                train_source_document_ids=fixture["train_documents"],
                replay_source_group_ids=fixture["replay_groups"],
            )


@pytest.mark.parametrize("mutation", ("static", "split", "candidate"))
def test_extension_rejects_stale_split_generated_seed_static_or_candidate_binding(
    mutation,
):
    with _fixture() as fixture:
        _, _, authorization_path, authorization_sha = _authorization(fixture)
        if mutation == "static":
            fixture["paths"]["core_markdown"].write_bytes(b"changed core bytes\n")
        elif mutation == "split":
            fixture["source_split"] = {
                **fixture["source_split"],
                "train_source_group_membership_commitment_sha256": "d" * 64,
            }
        else:
            fixture["units"]["full_authorized_markdown"][-1]["labels"][-1] += 1
            fixture["units"]["full_authorized_markdown"][-1]["input_ids"][-1] += 1
        with pytest.raises(claim.ContractError):
            claim.build_extension(
                request_path=fixture["request_path"],
                expected_request_path=fixture["request_path"],
                expected_request_sha256=fixture["request_sha256"],
                authorization_path=authorization_path,
                expected_authorization_path=authorization_path,
                expected_authorization_sha256=authorization_sha,
                static_input_paths=fixture["paths"],
                expected_static_input_paths=fixture["paths"],
                units_by_variant=fixture["units"],
                source_split_authority=fixture["source_split"],
                generated_domain_authority=fixture["generated"],
                seed_qa_semantic_authority=fixture["seed"],
                runner_path=fixture["runner"],
                expected_runner_path=fixture["runner"],
                train_source_group_ids=fixture["train_groups"],
                train_source_document_ids=fixture["train_documents"],
                replay_source_group_ids=fixture["replay_groups"],
            )


def test_extension_rejects_arbitrary_opaque_receipt_even_when_resealed():
    with _fixture() as fixture:
        authorization, _, _, _ = _authorization(fixture)
        authorization["opaque_receipt_sha256"] = "f" * 64
        authorization = _reseal(authorization)
        with pytest.raises(claim.ContractError, match="canonical claim hash"):
            claim.validate_claim_authorization(authorization)


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_extension_rejects_symlink_hardlink_or_forbidden_alias_before_read(alias_kind):
    with tempfile.TemporaryDirectory(prefix="sdclaim-") as directory:
        root = Path(directory)
        source = root / "source.bin"
        source.write_bytes(b"synthetic\n")
        alias = root / "alias.bin"
        if alias_kind == "symlink":
            alias.symlink_to(source)
        else:
            os.link(source, alias)
        with pytest.raises(claim.ContractError, match="aliases are forbidden"):
            claim.secure_regular_file(alias, expected=alias, label="synthetic alias")


def test_valid_all_synthetic_round_trip():
    with _fixture() as fixture:
        extension, payload, authorization, authorization_sha = _extension(fixture)
        assert extension["schema"] == claim.EXTENSION_SCHEMA
        assert extension["accepted_for_training"] is True
        assert extension["opaque_collision_contract"]["claims"] == claim.ZERO_CLAIMS
        extension_path = fixture["root"] / "extension.json"
        extension_path.write_bytes(payload)
        loaded = claim.load_extension(
            extension_path,
            expected_path=extension_path,
            expected_file_sha256=claim.sha256_bytes(payload),
            expected_request_sha256=fixture["request_sha256"],
            expected_authorization_sha256=authorization_sha,
            request=fixture["request"],
            authorization=authorization,
        )
        assert loaded == extension


def test_cli_real_validation_requires_external_preregistered_digests():
    with pytest.raises(SystemExit) as error:
        claim.main(["validate-authorization"])
    assert error.value.code == 2
