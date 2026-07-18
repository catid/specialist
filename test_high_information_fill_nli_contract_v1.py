from __future__ import annotations

import build_high_information_fill_nli_contract_v1 as contract_builder


def test_launch_contract_uses_synthetic_preflights_and_seals_four_commands(
    monkeypatch,
):
    model = {
        "id": "synthetic-nli",
        "revision": "synthetic-revision",
        "file_receipts": {
            "model": {
                "snapshot_blob_id": "synthetic-blob",
                "file_sha256": "a" * 64,
                "file_bytes": 1,
                "required_for_local_runtime": True,
            }
        },
    }
    runtime = {
        "transformers_version": contract_builder.fill_nli.TRANSFORMERS_VERSION,
        "torch_version": contract_builder.fill_nli.TORCH_VERSION,
        "dtype": "bfloat16",
    }
    implementations = [
        {"path": "synthetic-worker.py", "file_sha256": "b" * 64}
    ]
    fill_receipts = [
        {"path": "fill-wrapper.py", "file_sha256": "c" * 64},
        {"path": "base-worker.py", "file_sha256": "d" * 64},
    ]

    def synthetic_preflight(args):
        shard = args.shard_index
        run_contract = {
            "model": model,
            "runtime": runtime,
            "implementation_receipts": implementations,
            "generation_pass": {"runtime_worker_receipts": fill_receipts},
            "content_sha256_before_self_field": str(shard) * 64,
        }
        return {
            "packets": 100 + shard,
            "positive_packets": 90 + shard,
            "hard_negative_packets": 10,
            "generation_pass_contract_sha256": "e" * 64,
            "structural_review_sha256": "f" * 64,
            "planned_outputs": {
                "output": f"synthetic-gpu{shard}.jsonl",
                "report": f"synthetic-gpu{shard}.report.json",
                "receipt": f"synthetic-gpu{shard}.receipt.json",
            },
            "run_contract": run_contract,
        }

    monkeypatch.setattr(contract_builder.fill_nli, "preflight", synthetic_preflight)
    contract = contract_builder.build_contract()
    assert len(contract["shards"]) == 4
    assert contract["generation_pass"]["runtime_worker_receipts"] == fill_receipts
    assert len(contract["generation_pass"]["runtime_worker_receipts"]) == 2
    assert contract["post_run_hash_binding"][
        "output_or_report_without_matching_receipt_is_valid"
    ] is False
    for shard, item in enumerate(contract["shards"]):
        assert item["gpu_shard"] == shard
        assert item["run_command"] == (
            f"CUDA_VISIBLE_DEVICES={shard} "
            f"LD_LIBRARY_PATH={contract_builder.CU13_LIBRARY_PATH} "
            f"{contract_builder.PYTHON_EXECUTABLE} "
            "run_high_information_fill_nli_prefilter_v1.py "
            f"--shard-index {shard} --gpu-index {shard} --batch-size 64"
        )
        assert item["post_run_receipt_required"] is True
        assert item["training_rows_emitted"] is False
    assert contract["runtime_environment"] == {
        "python_executable": "es-at-scale/.venv/bin/python",
        "ld_library_path": contract_builder.CU13_LIBRARY_PATH,
    }
    assert contract["policy"]["gpu_job_launched_by_contract_builder"] is False
    assert contract["policy"]["training_rows_emitted"] is False
    unsigned = dict(contract)
    unsigned.pop("content_sha256_before_self_field")
    assert contract["content_sha256_before_self_field"] == (
        contract_builder.corpus.canonical_sha256(unsigned)
    )
