import copy
import json
from pathlib import Path

import pytest

import eggroll_es_adapter_transport_precision_v81 as transport


ROOT = Path(__file__).resolve().parent
VLLM = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
PROJECTION_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fused_structured_runtime_v72.json"
)


def _sources():
    return {
        "config_source": (VLLM / "config/lora.py").read_text(encoding="utf-8"),
        "base_linear_source": (
            VLLM / "lora/layers/base_linear.py"
        ).read_text(encoding="utf-8"),
        "fused_moe_source": (
            VLLM / "lora/layers/fused_moe.py"
        ).read_text(encoding="utf-8"),
        "model_manager_source": (
            VLLM / "lora/model_manager.py"
        ).read_text(encoding="utf-8"),
    }


def _manifest():
    return json.loads(PROJECTION_PREREG.read_text(encoding="utf-8"))[
        "production_projection"
    ]["manifest"]


def _capability():
    return transport.attest_vllm_lora_sources_v81(**_sources())


def _plan():
    return transport.build_transport_plan_v81(_manifest(), _capability())


def _sha(character):
    return character * 64


def _stage_and_issue(fence, generation=1, candidate_id="candidate-1"):
    fence.stage(
        generation=generation,
        candidate_id=candidate_id,
        bank_storage_token="stable-pinned-bank",
        values_sha256=_sha("a"),
    )
    return fence.issue_async_copies(
        pinned=True,
        non_blocking=True,
        direct_to_runtime_views=True,
        copy_count=transport.EXPECTED_RUNTIME_VIEWS_V81,
        h2d_bytes=transport.EXPECTED_RUNTIME_BYTES_V81,
        device_staging_bytes=0,
        stream_token="copy-stream",
        event_token=f"event-{generation}",
    )


def _complete_publish(fence, generation=1):
    fence.observe_completion(event_token=f"event-{generation}", complete=True)
    fence.exact_runtime_audit(
        event_token=f"event-{generation}",
        runtime_values_sha256=_sha("a"),
        d2h_bytes=transport.EXPECTED_RUNTIME_BYTES_V81,
        d2h_calls=1,
    )
    return fence.publish()


def test_installed_vllm_source_attests_only_fp16_and_bf16_execution():
    receipt = _capability()
    assert receipt["supported_lora_dtypes"] == ["auto", "float16", "bfloat16"]
    assert receipt["byte_lower_than_bfloat16_supported"] is False
    assert receipt["fp8_lora_execution_supported"] is False
    assert receipt["dense_slot_allocations_use_lora_dtype"] == 2
    assert receipt["fused_moe_slot_allocations_use_lora_dtype"] == 6
    assert receipt["packed_cpu_lora_pinning_present"] is True


def test_source_attestation_fails_if_fp8_is_claimed_supported():
    sources = _sources()
    sources["config_source"] = sources["config_source"].replace(
        'Literal["auto", "float16", "bfloat16"]',
        'Literal["auto", "float16", "bfloat16", "float8_e4m3fn"]',
    )
    with pytest.raises(RuntimeError, match="dtype surface changed"):
        transport.attest_vllm_lora_sources_v81(**sources)


def test_source_attestation_fails_if_pinning_or_direct_copy_disappears():
    sources = _sources()
    sources["model_manager_source"] = sources["model_manager_source"].replace(
        ".pin_memory()", ".contiguous()"
    )
    with pytest.raises(RuntimeError, match="pinning surface changed"):
        transport.attest_vllm_lora_sources_v81(**sources)

    sources = _sources()
    sources["base_linear_source"] = sources["base_linear_source"].replace(
        "non_blocking=True", "non_blocking=False"
    )
    with pytest.raises(RuntimeError, match="allocation/copy surface changed"):
        transport.attest_vllm_lora_sources_v81(**sources)


def test_dtype_resolution_accepts_only_installed_half_precision_surface():
    assert transport.resolve_execution_dtype_v81("auto", "bf16") == "bfloat16"
    assert transport.resolve_execution_dtype_v81("torch.float16", "bf16") == "float16"
    with pytest.raises(RuntimeError, match="unsupported"):
        transport.resolve_execution_dtype_v81("float8_e4m3fn", "bfloat16")
    with pytest.raises(RuntimeError, match="unsupported"):
        transport.resolve_execution_dtype_v81("int8", "bfloat16")


def test_production_plan_preserves_fp32_authority_and_exact_byte_ledger():
    plan = _plan()
    assert plan["canonical_authority"] == {
        "dtype": "float32",
        "location": "cpu",
        "tensor_count": 70,
        "elements": 4_528_128,
        "bytes": 18_112_512,
        "roles": [
            "canonical_master",
            "perturbation_and_update_arithmetic",
            "optimizer_state",
            "checkpoint_authority",
        ],
    }
    assert plan["execution_view"]["resolved_dtype"] == "bfloat16"
    assert plan["execution_view"]["persistent_device_bytes"] == 9_842_688
    assert plan["control_v71"]["materialization_hbm_read_write_lower_bound_bytes"] == 29_528_064
    assert plan["challenger_pinned_direct"]["materialization_hbm_read_write_lower_bound_bytes"] == 9_842_688
    assert plan["challenger_pinned_direct"]["device_staging_bytes"] == 0
    assert plan["exact_delta"] == {
        "h2d_bytes_saved_per_transition": 0,
        "h2d_copy_calls_saved_per_transition": 0,
        "device_to_device_payload_bytes_saved_per_transition": 9_842_688,
        "hbm_read_write_lower_bound_bytes_saved_per_transition": 19_685_376,
        "hbm_materialization_lower_bound_fraction_saved": 2 / 3,
        "logical_transient_device_bytes_saved_at_peak": 524_288,
        "host_staging_bytes_saved_at_peak": 0,
        "per_16_candidate_hbm_bytes_saved": 314_966_016,
    }
    compact = dict(plan)
    claimed = compact.pop("content_sha256_before_self_field")
    assert transport.canonical_sha256_v81(compact) == claimed


def test_byte_neutral_fp16_is_not_a_memory_plan_and_fp8_fails_prelaunch():
    with pytest.raises(RuntimeError, match="must retain.*BF16"):
        transport.build_transport_plan_v81(
            _manifest(), _capability(), requested_dtype="float16"
        )
    with pytest.raises(RuntimeError, match="unsupported"):
        transport.build_transport_plan_v81(
            _manifest(), _capability(), requested_dtype="float8_e4m3fn"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("runtime_bytes", 1),
        lambda value: value["projections"].pop(),
        lambda value: value["projections"][0].__setitem__("runtime_key", "changed"),
        lambda value: value.__setitem__("runtime_dtype", "torch.float16"),
    ],
)
def test_projection_manifest_tampering_fails_closed(mutation):
    manifest = copy.deepcopy(_manifest())
    mutation(manifest)
    with pytest.raises(RuntimeError, match="projection manifest changed"):
        transport.validate_projection_manifest_v81(manifest)


def test_happy_path_requires_event_then_exact_audit_and_reuses_same_bank():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    first = _stage_and_issue(fence)
    assert first["h2d_bytes"] == 9_842_688
    publication = _complete_publish(fence)
    assert publication["generation_may_begin"] is True
    retired = fence.retire(candidate_id="candidate-1")
    assert retired["bank_reuse_authorized"] is True

    _stage_and_issue(fence, generation=2, candidate_id="candidate-2")
    _complete_publish(fence, generation=2)
    fence.retire(candidate_id="candidate-2")
    final = fence.final_receipt()
    assert final["completed_generations"] == 2
    assert final["clean_idle"] is True
    assert fence.bank_storage_token == "stable-pinned-bank"
    assert fence.bank_version == 2


def test_bank_cannot_be_reused_before_current_candidate_retires():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        fence.stage(
            generation=2,
            candidate_id="candidate-2",
            bank_storage_token="stable-pinned-bank",
            values_sha256=_sha("b"),
        )
    _complete_publish(fence)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        fence.stage(
            generation=2,
            candidate_id="candidate-2",
            bank_storage_token="stable-pinned-bank",
            values_sha256=_sha("b"),
        )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"pinned": False}, "requires a pinned"),
        ({"non_blocking": False}, "requires a pinned"),
        ({"direct_to_runtime_views": False}, "forbids an intermediate"),
        ({"copy_count": 81}, "partial or byte-mismatched"),
        ({"h2d_bytes": 9_842_686}, "partial or byte-mismatched"),
        ({"device_staging_bytes": 2}, "partial or byte-mismatched"),
    ],
)
def test_pageable_unfenced_partial_or_staged_copy_is_rejected(overrides, match):
    fence = transport.StreamSafePublicationFenceV81(_plan())
    fence.stage(
        generation=1,
        candidate_id="candidate-1",
        bank_storage_token="stable-pinned-bank",
        values_sha256=_sha("a"),
    )
    arguments = {
        "pinned": True,
        "non_blocking": True,
        "direct_to_runtime_views": True,
        "copy_count": 82,
        "h2d_bytes": 9_842_688,
        "device_staging_bytes": 0,
        "stream_token": "stream",
        "event_token": "event-1",
    }
    arguments.update(overrides)
    with pytest.raises(RuntimeError, match=match):
        fence.issue_async_copies(**arguments)


def test_pending_or_stale_event_cannot_reach_audit_or_publication():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    pending = fence.observe_completion(event_token="event-1", complete=False)
    assert pending["action"] == "completion_pending"
    with pytest.raises(RuntimeError, match="preceded copy completion"):
        fence.exact_runtime_audit(
            event_token="event-1",
            runtime_values_sha256=_sha("a"),
            d2h_bytes=9_842_688,
            d2h_calls=1,
        )
    with pytest.raises(RuntimeError, match="before exact audit"):
        fence.publish()
    with pytest.raises(RuntimeError, match="stale or foreign"):
        fence.observe_completion(event_token="event-0", complete=True)


def test_exact_runtime_mismatch_or_partial_readback_blocks_publication():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    fence.observe_completion(event_token="event-1", complete=True)
    with pytest.raises(RuntimeError, match="differs from staging"):
        fence.exact_runtime_audit(
            event_token="event-1",
            runtime_values_sha256=_sha("b"),
            d2h_bytes=9_842_688,
            d2h_calls=1,
        )

    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    fence.observe_completion(event_token="event-1", complete=True)
    with pytest.raises(RuntimeError, match="readback coverage changed"):
        fence.exact_runtime_audit(
            event_token="event-1",
            runtime_values_sha256=_sha("a"),
            d2h_bytes=9_842_686,
            d2h_calls=1,
        )


def test_uncertain_partial_copy_exact_restores_without_accepting_reward():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    receipt = fence.recover_uncertain(
        exact_restore_succeeded=True,
        restored_master_sha256=_sha("c"),
        expected_master_sha256=_sha("c"),
    )
    assert receipt["action"] == "uncertain_copy_exact_restore"
    assert receipt["reward_or_update_accepted"] is False
    assert receipt["bank_reuse_authorized"] is True
    assert fence.phase == "idle"


@pytest.mark.parametrize(
    "restore_succeeded,restored",
    [(False, None), (True, "d" * 64)],
)
def test_uncertain_copy_poisons_if_exact_restore_is_not_proven(
    restore_succeeded, restored
):
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    receipt = fence.recover_uncertain(
        exact_restore_succeeded=restore_succeeded,
        restored_master_sha256=restored,
        expected_master_sha256=_sha("c"),
    )
    assert receipt["action"] == "uncertain_copy_poison"
    assert receipt["bank_reuse_authorized"] is False
    with pytest.raises(RuntimeError, match="terminally poisoned"):
        fence.stage(
            generation=2,
            candidate_id="candidate-2",
            bank_storage_token="stable-pinned-bank",
            values_sha256=_sha("b"),
        )


def test_final_receipt_rejects_inflight_or_published_state():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    _stage_and_issue(fence)
    with pytest.raises(RuntimeError, match="clean idle"):
        fence.final_receipt()
    _complete_publish(fence)
    with pytest.raises(RuntimeError, match="clean idle"):
        fence.final_receipt()


def test_bool_cannot_spoof_copy_count():
    fence = transport.StreamSafePublicationFenceV81(_plan())
    fence.stage(
        generation=1,
        candidate_id="candidate-1",
        bank_storage_token="stable-pinned-bank",
        values_sha256=_sha("a"),
    )
    with pytest.raises(ValueError, match="copy count"):
        fence.issue_async_copies(
            pinned=True,
            non_blocking=True,
            direct_to_runtime_views=True,
            copy_count=True,
            h2d_bytes=9_842_688,
            device_staging_bytes=0,
            stream_token="stream",
            event_token="event-1",
        )
