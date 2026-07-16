from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import probe_vllm_synthetic_repeat_determinism_v62b as probe


def _outputs(count: int = probe.ROWS) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[index, index + 1])])
        for index in range(count)
    ]


def test_token_receipts_are_hashes_and_lengths_only() -> None:
    receipts = probe.token_hashes(_outputs())
    assert len(receipts) == 68
    assert set(receipts[0]) == {"token_count", "token_ids_sha256"}
    assert receipts[0]["token_count"] == 2
    assert len(receipts[0]["token_ids_sha256"]) == 64
    assert "token_ids" not in receipts[0]


def test_token_receipts_fail_closed_on_shape_change() -> None:
    with pytest.raises(RuntimeError, match="row count changed"):
        probe.token_hashes(_outputs(67))


def test_probe_defaults_preserve_exact_v62_runtime_shape() -> None:
    args = probe.parser().parse_args(["--output", "x", "--actor-label", "a"])
    assert args.graph is False
    assert args.max_num_seqs == 68
    assert args.torch_deterministic is False
    assert args.warmup_calls == 1


def test_probe_source_is_synthetic_and_numeric_receipt_only() -> None:
    source = Path(probe.__file__).read_text()
    assert "Synthetic deterministic evaluator probe item" in source
    assert "source_dataset_rows_opened\": 0" in source
    assert "prompt_or_generation_text_persisted\": False" in source
    assert "token_ids_persisted\": False" in source
    assert "protected_ood_shadow_or_terminal_opened\": False" in source
    assert "train_qa" not in source
    assert "holdout" not in source.lower()

