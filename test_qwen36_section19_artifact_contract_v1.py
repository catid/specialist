from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import qwen36_section19_artifact_contract_v1 as contract
import train_qwen36_low_regression_sft_v1 as trainer


VARIANT = "protocol_core_100k"


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: dict) -> None:
    _write(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _metadata(kind: str) -> dict:
    common = {
        "replay": False,
        "hard_negative": False,
        "rights": {"decision": "synthetic_fixture"},
        "safety_transfer_flags": [],
        "source_record_sha256": contract.canonical_sha256(
            {"synthetic_source": kind}
        ),
        "lineage": {"source_kind": "synthetic_fixture", "kind": kind},
    }
    if kind == "generated":
        return {
            **common,
            "category": "synthetic_domain",
            "task_family": "closed_book_application",
            "task_subtype": "application_scenario",
            "generation_mode": "positive",
            "verifier": {
                name: {"status": "passed"}
                for name in contract.GENERATED_VERIFIER_KEYS
            },
            "generator": {
                "lane": "synthetic",
                "semantic_shard_receipt": {"sha256": "1" * 64},
            },
        }
    if kind == "markdown":
        return {
            **common,
            "category": "raw_domain_continuation",
            "verifier": {
                "type": "sealed_text_and_token_hash_v1",
                "status": "passed",
                "text_sha256": "2" * 64,
            },
            "generator": {"type": "source_document", "status": "not_generated"},
        }
    if kind == "replay":
        return {
            **common,
            "category": "instruction_following",
            "replay": True,
            "verifier": {"type": "json_exact_v1", "status": "passed"},
            "generator": {"name": "deterministic_reference_compiler_v1"},
        }
    if kind == "seed":
        return {
            **common,
            "category": "precurated_seed_qa_opaque",
            "fact_id": "synthetic-fact",
            "verifier": {
                "type": "sealed_seed_qa_semantic_authority_v1",
                "status": "sealed_passed",
            },
            "generator": {
                "type": "precurated_source_row",
                "status": "generator_not_declared_by_source",
            },
        }
    raise AssertionError(kind)


def _sequence(
    kind: str,
    *,
    unit_id: str | None = None,
    sequence_id: str | None = None,
    source_document: str | None = None,
    source_start: int = 0,
    length: int = 3,
) -> dict:
    stream = {
        "generated": "domain_qa",
        "markdown": "raw_markdown",
        "replay": "replay",
        "seed": "domain_qa",
    }[kind]
    training_format = (
        "causal_next_token" if kind == "markdown" else "chat_assistant_only"
    )
    unit_id = unit_id or f"source-{kind}"
    sequence_id = sequence_id or f"sequence-{kind}-{source_start}"
    metadata = _metadata(kind)
    return {
        "schema": "mixed-training-packed-sequence-v1",
        "sequence_id": sequence_id,
        "stream": stream,
        "training_format": training_format,
        "source_group_id": f"group-{unit_id}",
        "source_document_id": source_document or f"document-{unit_id}",
        "segments": [
            {
                "unit_id": unit_id,
                "token_start": 0,
                "token_stop": length,
                "source_token_start": source_start,
                "source_token_stop": source_start + length,
                "budget_token_count": 2 if training_format == "chat_assistant_only" else length,
                "metadata": metadata,
            }
        ],
    }


def _authority(root: Path) -> tuple[SimpleNamespace, dict[str, dict]]:
    builder_paths = {}
    for name in ("snapshot_builder", "runtime", "section19", "trainer"):
        path = root / f"{name}.py"
        _write(path, f"# synthetic {name}\n".encode("utf-8"))
        builder_paths[name] = path

    top_path = root / "snapshot" / "manifest.json"
    _write_json(top_path, {"synthetic": True})
    variant_path = root / "snapshot" / VARIANT / "manifest.json"
    _write_json(variant_path, {"synthetic_variant": True})
    sequences = {
        row["sequence_id"]: row
        for row in (
            _sequence("generated"),
            _sequence("markdown"),
            _sequence("replay"),
            _sequence("seed"),
        )
    }
    variant_manifest = {
        "content_sha256_before_self_field": "4" * 64,
        "sequences": {"sha256": "5" * 64},
        "schedule": {"sha256": "6" * 64},
    }
    top_manifest = {
        "content_sha256_before_self_field": "3" * 64,
        "assembler": {
            "path": "snapshot_builder.py",
            "sha256": contract.file_sha256(builder_paths["snapshot_builder"]),
        },
        "variants": {
            VARIANT: {
                "manifest_path": variant_path.relative_to(root).as_posix(),
                "manifest_file_sha256": contract.file_sha256(variant_path),
            },
        },
    }
    authority = SimpleNamespace(
        variant=VARIANT,
        top_manifest_path=top_path,
        top_manifest=top_manifest,
        variant_manifest=variant_manifest,
        sequences=sequences,
        sequence_set_identity_sha256="8" * 64,
        final_cursor_commitment_sha256="9" * 64,
    )
    builders = contract.builder_receipts(
        authority,
        repository_root=root,
        additional={
            "mixed_snapshot_runtime": builder_paths["runtime"],
            "section19_contract": builder_paths["section19"],
            "sft_trainer": builder_paths["trainer"],
        },
    )
    return authority, builders


def test_dataset_manifest_derives_every_section19_field_and_binds_builders(
    tmp_path: Path,
):
    authority, builders = _authority(tmp_path)
    manifest = contract.build_dataset_manifest(
        authority,
        repository_root=tmp_path,
        builders=builders,
    )
    assert manifest["status"] == "complete_launchable"
    assert manifest["field_contract"]["required_fields"] == list(
        contract.SECTION19_FIELDS
    )
    assert manifest["segment_count"] == 4
    assert manifest["source_count"] == 4
    assert set(manifest["builders"]) == {
        "snapshot_assembler",
        "mixed_snapshot_runtime",
        "section19_contract",
        "sft_trainer",
    }
    by_type = {row["example_type"]: row for row in manifest["segments"]}
    generated = by_type["generated_domain_qa:application_scenario"]
    assert generated["closed_book"] is True
    assert generated["contains_source_context"] is False
    assert generated["deterministic_verifier"] is False
    assert by_type["raw_markdown_cpt"]["contains_source_context"] is True
    assert by_type["raw_markdown_cpt"]["deterministic_verifier"] is True
    assert by_type["general_behavior_replay"]["replay"] is True
    assert by_type["general_behavior_replay"]["deterministic_verifier"] is True
    assert by_type["precurated_seed_qa"]["verification_status"] == "sealed_passed"
    for row in manifest["segments"]:
        assert all(field in row for field in contract.SECTION19_FIELDS)
        assert row["split"] == "train"
        assert row["token_count"] == 3
        assert row["data_lineage"]["source_record_sha256"]
    assert contract.validate_dataset_manifest(
        manifest,
        authority,
        repository_root=tmp_path,
        builders=builders,
    ) == manifest


def test_pinned_replay_seed_hash_is_classified_as_deterministic(tmp_path: Path):
    authority, builders = _authority(tmp_path)
    metadata = authority.sequences["sequence-replay-0"]["segments"][0]["metadata"]
    metadata["verifier"]["type"] = "pinned_seed_hash_v1"
    manifest = contract.build_dataset_manifest(
        authority,
        repository_root=tmp_path,
        builders=builders,
    )
    replay = next(row for row in manifest["segments"] if row["replay"])
    assert replay["deterministic_verifier"] is True


@pytest.mark.parametrize(
    "missing",
    ["category", "replay", "hard_negative", "verifier", "generator", "lineage", "source_record_sha256"],
)
def test_dataset_manifest_rejects_missing_required_derivation_inputs(
    tmp_path: Path,
    missing: str,
):
    authority, builders = _authority(tmp_path)
    authority.sequences["sequence-generated-0"]["segments"][0]["metadata"].pop(missing)
    with pytest.raises(contract.Section19ContractError):
        contract.build_dataset_manifest(
            authority,
            repository_root=tmp_path,
            builders=builders,
        )


def test_dataset_manifest_rejects_forged_derivation_even_when_resealed(
    tmp_path: Path,
):
    authority, builders = _authority(tmp_path)
    manifest = contract.build_dataset_manifest(
        authority, repository_root=tmp_path, builders=builders
    )
    forged = copy.deepcopy(manifest)
    forged.pop("content_sha256_before_self_field")
    forged["segments"][0]["closed_book"] = not forged["segments"][0]["closed_book"]
    forged = contract.self_address(forged)
    with pytest.raises(
        contract.Section19ContractError, match="stale or forged"
    ):
        contract.validate_dataset_manifest(
            forged,
            authority,
            repository_root=tmp_path,
            builders=builders,
        )


@pytest.mark.parametrize("field", contract.SECTION19_FIELDS)
def test_dataset_manifest_rejects_each_missing_section19_surface_field(
    tmp_path: Path,
    field: str,
):
    authority, builders = _authority(tmp_path)
    manifest = contract.build_dataset_manifest(
        authority, repository_root=tmp_path, builders=builders
    )
    missing = copy.deepcopy(manifest)
    missing.pop("content_sha256_before_self_field")
    missing["segments"][0].pop(field)
    missing["segment_set_identity_sha256"] = contract.canonical_sha256(
        missing["segments"]
    )
    missing = contract.self_address(missing)
    with pytest.raises(contract.Section19ContractError, match="stale or forged"):
        contract.validate_dataset_manifest(
            missing,
            authority,
            repository_root=tmp_path,
            builders=builders,
        )


def test_dataset_manifest_rejects_duplicate_and_ambiguous_segment_identities(
    tmp_path: Path,
):
    authority, builders = _authority(tmp_path)
    sequence = authority.sequences["sequence-generated-0"]
    sequence["segments"].append(copy.deepcopy(sequence["segments"][0]))
    with pytest.raises(contract.Section19ContractError, match="duplicate emitted"):
        contract.build_dataset_manifest(
            authority, repository_root=tmp_path, builders=builders
        )

    authority, builders = _authority(tmp_path / "ambiguous")
    extra = _sequence(
        "markdown",
        unit_id="source-markdown",
        sequence_id="sequence-markdown-other-document",
        source_document="different-document",
        source_start=3,
    )
    authority.sequences[extra["sequence_id"]] = extra
    with pytest.raises(contract.Section19ContractError, match="duplicate identity is ambiguous"):
        contract.build_dataset_manifest(
            authority,
            repository_root=tmp_path / "ambiguous",
            builders=builders,
        )


def test_dataset_manifest_rejects_overlap_unknown_verifier_and_stale_builder(
    tmp_path: Path,
):
    authority, builders = _authority(tmp_path / "overlap")
    extra = _sequence(
        "markdown",
        unit_id="source-markdown",
        sequence_id="sequence-markdown-overlap",
        source_start=2,
    )
    authority.sequences[extra["sequence_id"]] = extra
    with pytest.raises(contract.Section19ContractError, match="overlapping"):
        contract.build_dataset_manifest(
            authority,
            repository_root=tmp_path / "overlap",
            builders=builders,
        )

    authority, builders = _authority(tmp_path / "unknown")
    authority.sequences["sequence-replay-0"]["segments"][0]["metadata"]["verifier"]["type"] = "opaque_maybe_deterministic_v1"
    with pytest.raises(contract.Section19ContractError, match="determinism is ambiguous"):
        contract.build_dataset_manifest(
            authority,
            repository_root=tmp_path / "unknown",
            builders=builders,
        )

    authority, builders = _authority(tmp_path / "stale")
    (tmp_path / "stale" / "trainer.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(contract.Section19ContractError, match="stale receipt"):
        contract.build_dataset_manifest(
            authority,
            repository_root=tmp_path / "stale",
            builders=builders,
        )


def _run_fixture(root: Path) -> tuple[Path, SimpleNamespace, dict[str, dict]]:
    authority, builders = _authority(root)
    run = root / "runs" / "synthetic"
    run.mkdir(parents=True)
    dataset_manifest = contract.build_dataset_manifest(
        authority,
        repository_root=root,
        builders=builders,
    )
    _write_json(run / "dataset_manifest.json", dataset_manifest)
    hashes = contract.build_dataset_hashes(
        authority,
        dataset_manifest_path=run / "dataset_manifest.json",
        dataset_manifest=dataset_manifest,
    )
    _write_json(run / "dataset_hashes.json", hashes)
    _write_json(
        run / "run_config.json",
        contract.self_address({
            "schema": "qwen36-low-regression-expert-lora-sft-run-v1",
            "run_id": "synthetic-run",
        }),
    )
    _write_json(
        run / "environment.txt",
        {
            "python": "synthetic",
            "platform": "synthetic",
            "executable": "/synthetic/python",
            "gpu": {"name": "synthetic"},
        },
    )
    _write(run / "pip_freeze.txt", b"synthetic-package==1.0\n")
    _write_json(
        run / "git_commits.json",
        {"specialist": "a" * 40, "dirty_paths": []},
    )
    _write_json(run / "model_config.json", {"model_type": "synthetic"})
    _write_json(run / "adapter_config.json", {"peft_type": "LORA"})
    _write(
        run / "trainable_parameters.txt",
        (json.dumps({"name": "synthetic.lora_A", "elements": 1}) + "\n").encode(),
    )
    _write(run / "training_metrics.jsonl", b"")
    _write_json(
        run / "evaluation_results.json",
        {"status": "not_run_by_training_process", "reason": "synthetic"},
    )
    _write_json(run / "routing_metrics.json", contract.pending_routing_metrics())
    _write_json(run / "memory_profile.json", contract.pending_memory_profile())
    for name in ("checkpoints", "plots", "samples"):
        (run / name).mkdir()
    return run, authority, builders


def test_run_artifact_launch_receipts_cover_exact_required_inventory(
    tmp_path: Path,
):
    run, authority, builders = _run_fixture(tmp_path)
    value = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        phase="launch_ready",
    )
    assert set(value["receipts"]) == set(contract.REQUIRED_RUN_ARTIFACTS)
    assert value["gates"]["launch_authorized"] is True
    assert value["gates"]["training_outputs_complete"] is False
    assert value["gates"]["evaluation_complete"] is False
    assert value["gates"]["selection_or_deployment_authorized"] is False
    assert all(
        receipt["schema"] == contract.ARTIFACT_RECEIPT_SCHEMA
        for receipt in value["receipts"].values()
    )
    assert contract.validate_run_artifact_contract(
        run,
        value,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        expected_phase="launch_ready",
    ) == value


def test_run_artifact_gate_rejects_missing_stale_forged_and_duplicate_receipts(
    tmp_path: Path,
):
    run, authority, builders = _run_fixture(tmp_path)
    value = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        phase="launch_ready",
    )
    (run / "model_config.json").write_text(
        '{"model_type":"tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(contract.Section19ContractError, match="stale, forged, or ambiguous"):
        contract.validate_run_artifact_contract(
            run,
            value,
            repository_root=tmp_path,
            authority=authority,
            builders=builders,
            expected_phase="launch_ready",
        )

    run, authority, builders = _run_fixture(tmp_path / "missing")
    (run / "pip_freeze.txt").unlink()
    with pytest.raises(contract.Section19ContractError, match="path is missing"):
        contract.seal_run_artifacts(
            run,
            repository_root=tmp_path / "missing",
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )

    run, authority, builders = _run_fixture(tmp_path / "forged")
    value = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path / "forged",
        authority=authority,
        builders=builders,
        phase="launch_ready",
    )
    forged = copy.deepcopy(value)
    forged.pop("content_sha256_before_self_field")
    forged["receipts"]["model_config.json"]["sha256"] = "f" * 64
    forged["receipts"]["adapter_config.json"]["artifact"] = "model_config.json"
    forged = contract.self_address(forged)
    with pytest.raises(contract.Section19ContractError, match="stale, forged, or ambiguous"):
        contract.validate_run_artifact_contract(
            run,
            forged,
            repository_root=tmp_path / "forged",
            authority=authority,
            builders=builders,
            expected_phase="launch_ready",
        )


def test_launch_recomputes_dataset_derivations_against_sequence_authority(
    tmp_path: Path,
):
    run, authority, builders = _run_fixture(tmp_path)
    manifest = json.loads((run / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("content_sha256_before_self_field")
    manifest["segments"][0]["closed_book"] = not manifest["segments"][0]["closed_book"]
    manifest["segment_set_identity_sha256"] = contract.canonical_sha256(
        manifest["segments"]
    )
    manifest = contract.self_address(manifest)
    _write_json(run / "dataset_manifest.json", manifest)
    forged_hashes = contract.build_dataset_hashes(
        authority,
        dataset_manifest_path=run / "dataset_manifest.json",
        dataset_manifest=manifest,
    )
    _write_json(run / "dataset_hashes.json", forged_hashes)
    with pytest.raises(contract.Section19ContractError, match="stale or forged"):
        contract.seal_run_artifacts(
            run,
            repository_root=tmp_path,
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )


def test_run_artifact_gate_rejects_symlink_and_hardlink_aliases(tmp_path: Path):
    run, authority, builders = _run_fixture(tmp_path / "symlink")
    os.symlink("../model_config.json", run / "checkpoints" / "alias.json")
    with pytest.raises(contract.Section19ContractError, match="forbidden"):
        contract.seal_run_artifacts(
            run,
            repository_root=tmp_path / "symlink",
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )

    run, authority, builders = _run_fixture(tmp_path / "hardlink")
    os.link(run / "model_config.json", run / "checkpoints" / "alias.json")
    with pytest.raises(contract.Section19ContractError, match="hard-link"):
        contract.seal_run_artifacts(
            run,
            repository_root=tmp_path / "hardlink",
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )


def test_run_artifact_gate_rejects_protected_or_external_run_paths(tmp_path: Path):
    root = tmp_path / "repository"
    run, authority, builders = _run_fixture(root)
    forbidden = root / "evaluation_protocol" / "synthetic"
    forbidden.parent.mkdir()
    run.rename(forbidden)
    with pytest.raises(contract.Section19ContractError, match="path class"):
        contract.seal_run_artifacts(
            forbidden,
            repository_root=root,
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )

    external = tmp_path / "external-run"
    forbidden.rename(external)
    with pytest.raises(contract.Section19ContractError, match="escapes repository"):
        contract.seal_run_artifacts(
            external,
            repository_root=root,
            authority=authority,
            builders=builders,
            phase="launch_ready",
        )


def test_training_complete_and_complete_phases_are_distinct_status_gates(
    tmp_path: Path,
):
    run, authority, builders = _run_fixture(tmp_path)
    metric = {
        "schema": "qwen36-low-regression-sft-training-metric-v1",
        "optimizer_step": 1,
    }
    _write(run / "training_metrics.jsonl", (json.dumps(metric) + "\n").encode())
    _write_json(
        run / "routing_metrics.json",
        contract.self_address({
            "schema": "qwen36-low-regression-sft-routing-summary-v1",
            "status": "complete",
        }),
    )
    _write_json(
        run / "memory_profile.json",
        contract.self_address({
            "schema": "qwen36-low-regression-sft-memory-profile-v1",
            "status": "complete",
        }),
    )
    _write(run / "checkpoints" / "checkpoint.bin", b"synthetic checkpoint")
    training_complete = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        phase="training_complete",
    )
    assert training_complete["gates"]["training_outputs_complete"] is True
    assert training_complete["gates"]["evaluation_complete"] is False
    assert training_complete["gates"]["selection_or_deployment_authorized"] is False
    with pytest.raises(contract.Section19ContractError, match="final evaluation is incomplete"):
        contract.seal_run_artifacts(
            run,
            repository_root=tmp_path,
            authority=authority,
            builders=builders,
            phase="complete",
        )

    _write_json(
        run / "evaluation_results.json",
        {
            "schema": "synthetic-evaluation-results-v1",
            "status": "complete",
        },
    )
    complete = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        phase="complete",
    )
    assert complete["gates"]["evaluation_complete"] is True
    assert complete["gates"]["selection_or_deployment_authorized"] is True


def test_resume_gate_preserves_immutable_receipts_but_allows_dynamic_progress(
    tmp_path: Path,
):
    run, authority, builders = _run_fixture(tmp_path)
    launch = contract.seal_run_artifacts(
        run,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
        phase="launch_ready",
    )
    _write(
        run / "training_metrics.jsonl",
        (
            json.dumps({
                "schema": "qwen36-low-regression-sft-training-metric-v1",
                "optimizer_step": 1,
            })
            + "\n"
        ).encode(),
    )
    assert contract.validate_immutable_run_artifacts(
        run,
        launch,
        repository_root=tmp_path,
        authority=authority,
        builders=builders,
    ) == launch

    (run / "adapter_config.json").write_text(
        '{"peft_type":"FORGED"}\n', encoding="utf-8"
    )
    with pytest.raises(contract.Section19ContractError, match="immutable artifact changed"):
        contract.validate_immutable_run_artifacts(
            run,
            launch,
            repository_root=tmp_path,
            authority=authority,
            builders=builders,
        )


def test_trainer_static_writer_materializes_and_validates_launch_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repository_root = tmp_path
    assembler = repository_root / "build_mixed_training_snapshot_v1.py"
    synthetic_runtime = repository_root / "qwen36_mixed_training_runtime_v1.py"
    synthetic_contract = repository_root / "qwen36_section19_artifact_contract_v1.py"
    synthetic_trainer = repository_root / "train_qwen36_low_regression_sft_v1.py"
    for path in (assembler, synthetic_runtime, synthetic_contract, synthetic_trainer):
        _write(path, f"# synthetic {path.name}\n".encode("utf-8"))
    monkeypatch.setattr(trainer, "ROOT", repository_root)
    monkeypatch.setattr(trainer, "__file__", str(synthetic_trainer))
    monkeypatch.setattr(trainer.mixed, "__file__", str(synthetic_runtime))
    monkeypatch.setattr(trainer.section19, "__file__", str(synthetic_contract))
    top_path = repository_root / "snapshot" / "manifest.json"
    _write_json(top_path, {"synthetic": True})
    variant_path = repository_root / "snapshot" / VARIANT / "manifest.json"
    _write_json(variant_path, {"synthetic_variant": True})
    sequences = {
        row["sequence_id"]: row
        for row in (
            _sequence("generated"),
            _sequence("markdown"),
            _sequence("replay"),
            _sequence("seed"),
        )
    }
    authority = SimpleNamespace(
        variant=VARIANT,
        top_manifest_path=top_path,
        top_manifest={
            "content_sha256_before_self_field": "3" * 64,
            "assembler": {
                "path": assembler.relative_to(repository_root).as_posix(),
                "sha256": contract.file_sha256(assembler),
            },
            "variants": {
                VARIANT: {
                    "manifest_path": variant_path.relative_to(
                        repository_root
                    ).as_posix(),
                    "manifest_file_sha256": contract.file_sha256(variant_path),
                },
            },
        },
        variant_manifest={
            "content_sha256_before_self_field": "4" * 64,
            "sequences": {"sha256": "5" * 64},
            "schedule": {"sha256": "6" * 64},
        },
        sequences=sequences,
        sequence_set_identity_sha256="8" * 64,
        final_cursor_commitment_sha256="9" * 64,
    )

    def fake_run(arguments, **_kwargs):
        if "pip" in arguments:
            return SimpleNamespace(stdout="synthetic-package==1.0\n")
        if arguments[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        if arguments[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="")
        raise AssertionError(arguments)

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)
    monkeypatch.setattr(trainer.platform, "platform", lambda: "synthetic-platform")
    run_config = trainer._self_addressed({
        "schema": trainer.RUN_SCHEMA,
        "run_id": "synthetic-static-writer",
    })

    class FakeConfig:
        @staticmethod
        def to_dict():
            return {"model_type": "synthetic"}

    class FakeLoraConfig:
        @staticmethod
        def to_dict():
            return {"peft_type": "LORA", "target_modules": ["synthetic"]}

    output = tmp_path / "run"
    output.mkdir()
    trainer._write_static_artifacts(
        output,
        run_config,
        authority,
        SimpleNamespace(config=FakeConfig()),
        FakeLoraConfig(),
        {
            "trainable_parameters": [
                {
                    "name": "synthetic.lora_A",
                    "shape": [1],
                    "elements": 1,
                    "dtype": "torch.bfloat16",
                }
            ]
        },
        {"name": "synthetic", "total_memory_gib": 96.0},
    )
    artifact_contract = trainer._load_self_addressed(
        output / "run_artifact_receipts.json"
    )
    assert artifact_contract["status"] == "launch_ready"
    expected_builders = trainer._section19_builder_receipts(authority)
    assert contract.validate_run_artifact_contract(
        output,
        artifact_contract,
        repository_root=repository_root,
        authority=authority,
        builders=expected_builders,
        expected_phase="launch_ready",
    ) == artifact_contract
    dataset_manifest = trainer._load_self_addressed(
        output / "dataset_manifest.json"
    )
    assert dataset_manifest["segment_count"] == 4
    assert all(
        set(contract.SECTION19_FIELDS).issubset(row)
        for row in dataset_manifest["segments"]
    )
