from __future__ import annotations

import copy

import build_high_information_fill_semantic_judge_contract_v1 as builder


def static_shard(shard: int) -> dict:
    prompts = {
        "schema": "fill-semantic-judge-rendered-prompt-receipt-root-v1",
        "candidate_grouping": builder.fill_judge.CANDIDATE_GROUPING,
        "rendered_prompts": 2,
        "receipts_sha256": str(shard) * 64,
        "ordered_receipt_fields": ["messages_sha256"],
        "every_rendered_system_and_user_message_bound": True,
        "prompt_text_persisted": False,
    }
    return {
        "gpu_shard": shard,
        "packets": 10 + shard,
        "request_groups": 10 + shard,
        "generation_pass": {
            "id": builder.fill_judge.PASS_ID,
            "contract_sha256": str(shard) * 64,
        },
        "structural_review": {"review_file_sha256": "b" * 64},
        "rendered_prompt_receipts": prompts,
        "required_sealed_fill_nli": {
            "output": f"nli-gpu{shard}.jsonl",
            "report": f"nli-gpu{shard}.report.json",
            "receipt": f"nli-gpu{shard}.receipt.json",
        },
        "planned_fill_judge_outputs": {
            "output": f"judge-gpu{shard}.jsonl",
            "report": f"judge-gpu{shard}.report.json",
            "receipt": f"judge-gpu{shard}.receipt.json",
        },
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def state(shard: int, present: bool) -> dict:
    return {
        "gpu_shard": shard,
        "artifacts": {
            name: {
                "path": f"nli-gpu{shard}.{name}",
                "regular_non_symlink_file_present": present,
            }
            for name in ("output", "report", "receipt")
        },
        "sealed_trio_present": present,
    }


def patch_static(monkeypatch, *, present: list[bool]):
    reused = builder.fill_judge.validate_reused_judge_semantics()
    monkeypatch.setattr(builder.fill_judge, "validate_reused_judge_semantics", lambda: reused)
    monkeypatch.setattr(
        builder.fill_judge,
        "validate_nli_launch_contract",
        lambda: ({}, {
            "path": "fill-nli.launch.json",
            "file_sha256": "c" * 64,
            "content_sha256_before_self_field": "d" * 64,
        }),
    )
    monkeypatch.setattr(builder, "_static_shard", static_shard)
    monkeypatch.setattr(builder, "_nli_trio_state", lambda shard: state(shard, present[shard]))


def test_blocked_scaffold_emits_no_commands_until_every_nli_trio_is_present(monkeypatch):
    patch_static(monkeypatch, present=[True, True, True, False])
    contract = builder.build_contract()
    assert contract["status"] == "blocked_pending_all_four_sealed_fill_nli_outputs"
    assert contract["commands_released"] is False
    assert "missing_sealed_fill_nli_output:gpu3" in contract["blockers"]
    assert contract["inference"]["request_batch_size"] == 16
    assert contract["inference"]["two_pass_sequences_per_full_batch"] == 32
    assert contract["fill_judge_wrapper_receipt"]["path"] == (
        "run_high_information_fill_semantic_judge_v1.py"
    )
    assert contract["exact_runtime"]["python_executable"].endswith("es-at-scale/.venv/bin/python")
    assert contract["exact_runtime"]["ld_library_path"] == builder.CU13_LIBRARY_PATH
    assert contract["eventual_exact_hash_binding"]["output_or_report_without_matching_receipt_is_valid"] is False
    for shard in contract["shards"]:
        assert "run_command" not in shard
        assert "preflight_command" not in shard
        assert shard["rendered_prompt_receipts"]["receipts_sha256"]
        assert shard["training_rows_emitted"] is False
    unsigned = dict(contract)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert declared == builder.corpus.canonical_sha256(unsigned)


def test_all_four_validated_preflights_release_exact_es_runtime_commands(monkeypatch):
    patch_static(monkeypatch, present=[True, True, True, True])
    monkeypatch.setenv("LD_LIBRARY_PATH", builder.CU13_LIBRARY_PATH)

    def preflight(args):
        static = static_shard(args.shard_index)
        return {
            "sealed_fill_nli": {
                "output_sha256": "e" * 64,
                "receipt_self_sha256": "f" * 64,
            },
            "run_contract": {
                "gpu_shard": args.shard_index,
                "generation_pass": {
                    "contract_sha256": static["generation_pass"]["contract_sha256"]
                },
                "judge_protocol": {
                    "rendered_prompt_receipts": static["rendered_prompt_receipts"]
                },
                "content_sha256_before_self_field": "1" * 64,
            },
        }

    monkeypatch.setattr(builder.fill_judge, "preflight", preflight)
    contract = builder.build_contract()
    assert contract["status"] == "ready_for_explicit_four_gpu_fill_semantic_judge_launch"
    assert contract["commands_released"] is True
    assert contract["blockers"] == []
    for shard, item in enumerate(contract["shards"]):
        assert item["run_command"] == (
            f"CUDA_VISIBLE_DEVICES={shard} "
            f"LD_LIBRARY_PATH={builder.CU13_LIBRARY_PATH} "
            f"{builder.PYTHON_EXECUTABLE} "
            "run_high_information_fill_semantic_judge_v1.py "
            f"--shard-index {shard} --gpu-index {shard} "
            "--request-batch-size 16 --max-model-len 16384 "
            "--max-tokens 3072 --gpu-memory-utilization 0.90"
        )
        assert item["post_run_receipt_required"] is True
        assert item["sealed_fill_nli"]["output_sha256"] == "e" * 64
        assert item["training_rows_emitted"] is False


def test_present_but_invalid_nli_preflight_remains_command_blocked(monkeypatch):
    patch_static(monkeypatch, present=[True, True, True, True])
    monkeypatch.setenv("LD_LIBRARY_PATH", builder.CU13_LIBRARY_PATH)
    monkeypatch.setattr(
        builder.fill_judge,
        "preflight",
        lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic invalid receipt")),
    )
    contract = builder.build_contract()
    assert contract["commands_released"] is False
    assert contract["blockers"] == ["fill_nli_or_judge_preflight_invalid:gpu0"]
    assert all("run_command" not in shard for shard in contract["shards"])
