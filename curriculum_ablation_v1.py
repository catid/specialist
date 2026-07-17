#!/usr/bin/env python3
"""CPU-only contract checks for the CPT -> SFT -> mirrored-ES curriculum.

The module validates identities and accounting, never training content.  A
preview may be sealed before the source-disjoint extension audit exists, but
``require_launch_ready=True`` fails until that opaque four-domain audit is
attached.  This prevents a filename such as ``train`` from becoming implicit
authorization for a new adaptation source.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


PLAN_SCHEMA_V1 = "specialist-cpt-sft-es-curriculum-ablation-v1"
EXTENSION_SCHEMA_V1 = "specialist-curriculum-source-disjoint-extension-v1"
RECEIPT_SCHEMA_V1 = "specialist-curriculum-execution-receipt-v1"
ALLOWED_RIGHTS_V1 = frozenset({
    "explicit_open_license",
    "federal_text_public_domain_presumption",
    "public_domain_in_usa_source_with_trademark_and_jurisdiction_limits",
})
ALLOWED_SFT_KINDS_V1 = frozenset({
    "qa_verified",
    "qa_manual",
    "qa_resource_manual_fact",
    "qa_resource_direct",
    "qa_resource_category",
})
ALLOWED_SFT_SCHEMAS_V1 = frozenset({
    "curated-qa-v1",
    "manual-resource-fact-v1",
})
EXPECTED_ARMS_V1 = (
    "cpt_sft_es",
    "direct_sft_es",
    "direct_sft",
    "direct_es",
)
IDENTITY_DOMAINS_V1 = (
    "document_sha256",
    "normalized_url",
    "raw_lineage",
    "near_duplicate",
)
CPT_UNIT_FIELDS_V1 = frozenset({
    "resource_id",
    "source_document_identity_sha256",
    "markdown_path",
    "markdown_sha256",
    "available_tokens",
    "selected_tokens",
    "token_start",
    "token_stop",
    "selected_token_ids_sha256",
    "rights_status",
    "promotion_gate",
    "direct_training_ready",
    "training_role",
    "training_format",
    "assistant_supervision",
})
SFT_UNIT_FIELDS_V1 = frozenset({
    "fact_id",
    "row_sha256",
    "document_sha256",
    "source",
    "kind",
    "quality_schema",
    "conflict_unit_identity_sha256",
    "conflict_unit_rows",
    "prompt_tokens",
    "answer_tokens",
    "verified_instructional",
    "url_trivia",
    "raw_markdown",
    "training_role",
    "training_format",
})
ES_ITEM_FIELDS_V1 = frozenset({
    "unit_identity_sha256",
    "row_sha256",
    "source",
    "category",
    "position",
})
STAGE_FIELDS_V1 = frozenset({
    "stage_id",
    "dataset_content_sha256",
    "target_gpu_seconds",
    "optimizer_updates",
    "nonpadding_tokens",
    "generated_rollouts",
    "checkpoint_input",
    "transition_mode",
})
EVALUATION_FIELDS_V1 = frozenset({
    "path",
    "file_sha256",
    "content_sha256",
    "disjointness_passed",
    "protected_access_authorized",
    "protected_selection_identity_sha256",
    "current_adaptation_allowlist",
    "new_curriculum_inputs_currently_authorized",
})
EXTENSION_FIELDS_V1 = frozenset({
    "schema",
    "status",
    "parent_evaluation_contract_content_sha256",
    "protected_selection_identity_sha256",
    "identity_domains",
    "required_new_input_content_sha256s",
    "audit_file_sha256",
    "audit_content_sha256",
    "cross_role_collision_counts",
    "audit_passed",
    "launch_eligible",
    "protected_semantics_opened",
    "protected_content_persisted",
})
AUTHORIZATION_FIELDS_V1 = frozenset({
    "gpu_launch",
    "new_adaptation_inputs",
    "protected_holdout_access",
    "protected_holdout_selection_or_tuning",
    "future_launch_requires_disjoint_extension_and_gpu_throughput_calibration",
})
ACCESS_FIELDS_V1 = frozenset({
    "CPT_train_candidate_markdown_opened",
    "SFT_registered_train_semantics_opened",
    "dev_or_ood_semantics_opened",
    "protected_holdout_semantics_opened",
    "model_weights_loaded",
    "gpu_launched",
    "live_run_directory_touched",
})
EXECUTION_RECEIPT_FIELDS_V1 = frozenset({
    "schema",
    "arm_id",
    "stage_receipts",
    "total_charged_gpu_seconds",
    "useful_physical_gpu_ids",
    "protected_holdout_opened",
    "final_checkpoint_sha256",
})
STAGE_RECEIPT_FIELDS_V1 = frozenset({
    "stage_id",
    "input_checkpoint_sha256",
    "output_checkpoint_sha256",
    "dataset_content_sha256",
    "optimizer_updates",
    "nonpadding_tokens",
    "generated_rollouts",
    "charged_gpu_seconds",
    "resume_segments",
})
RESUME_SEGMENT_FIELDS_V1 = frozenset({
    "segment_index",
    "input_checkpoint_sha256",
    "output_checkpoint_sha256",
    "input_training_state_sha256",
    "output_training_state_sha256",
    "starting_update",
    "ending_update",
})


def canonical_sha256_v1(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_v1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"curriculum {label} must be lowercase SHA-256")
    return value


def _identity_v1(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"curriculum {label} must be a nonempty short string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"curriculum {label} must be opaque ASCII") from error
    if any(character.isspace() for character in value):
        raise ValueError(f"curriculum {label} may not contain whitespace")
    return value


def _nonnegative_int_v1(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"curriculum {label} must be an exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"curriculum {label} must be nonnegative")
    return result


def _positive_number_v1(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"curriculum {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"curriculum {label} must be finite and positive")
    return result


def _validate_section_hash_v1(section: Mapping[str, Any], label: str) -> str:
    compact = dict(section)
    observed = compact.pop("content_sha256", None)
    _sha256_v1(observed, f"{label} content identity")
    if canonical_sha256_v1(compact) != observed:
        raise RuntimeError(f"curriculum {label} content identity changed")
    return observed


def _validate_cpt_v1(cpt: Mapping[str, Any]) -> str:
    content_sha = _validate_section_hash_v1(cpt, "CPT dataset")
    if (
        cpt.get("schema") != "specialist-cpt-raw-markdown-input-v1"
        or cpt.get("training_role") != "causal_language_modeling"
        or cpt.get("assistant_supervision") is not False
        or cpt.get("source_token_cap") != 4_845
        or cpt.get("target_nonpadding_tokens") != 99_216
        or cpt.get("chunk_token_cap") != 1_024
        or not math.isclose(
            float(cpt.get("maximum_source_token_fraction", -1.0)),
            4_845 / 99_216,
            rel_tol=0.0,
            abs_tol=1e-18,
        )
    ):
        raise ValueError("curriculum CPT stage contract changed")
    units = cpt.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("curriculum CPT needs raw Markdown units")
    resources, documents, markdown_hashes = set(), set(), set()
    selected_total = 0
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != CPT_UNIT_FIELDS_V1:
            raise ValueError("curriculum CPT unit schema leaked another stage")
        resource = _identity_v1(unit["resource_id"], "CPT resource")
        document = _sha256_v1(
            unit["source_document_identity_sha256"], "CPT source document"
        )
        markdown_sha = _sha256_v1(unit["markdown_sha256"], "CPT Markdown")
        if (
            resource in resources
            or document in documents
            or markdown_sha in markdown_hashes
        ):
            raise ValueError("curriculum CPT duplicated a source unit")
        resources.add(resource)
        documents.add(document)
        markdown_hashes.add(markdown_sha)
        available = _nonnegative_int_v1(
            unit["available_tokens"], "CPT available tokens"
        )
        selected = _nonnegative_int_v1(
            unit["selected_tokens"], "CPT selected tokens"
        )
        start = _nonnegative_int_v1(unit["token_start"], "CPT token start")
        stop = _nonnegative_int_v1(unit["token_stop"], "CPT token stop")
        if (
            selected <= 0
            or selected > cpt["source_token_cap"]
            or stop - start != selected
            or stop > available
            or unit["rights_status"] not in ALLOWED_RIGHTS_V1
            or not str(unit["promotion_gate"]).startswith("ready_")
            or unit["direct_training_ready"] is not True
            or unit["training_role"] != "cpt_raw_markdown"
            or unit["training_format"] != "causal_next_token_markdown"
            or unit["assistant_supervision"] is not False
            or not str(unit["markdown_path"]).startswith("data/site_corpora/")
            or not str(unit["markdown_path"]).endswith(".md")
        ):
            raise ValueError("curriculum CPT unit eligibility or token cap changed")
        _sha256_v1(
            unit["selected_token_ids_sha256"], "CPT selected token IDs"
        )
        selected_total += selected
    if (
        len(units) != cpt.get("eligible_source_documents")
        or selected_total != cpt["target_nonpadding_tokens"]
        or selected_total != cpt.get("selected_nonpadding_tokens")
        or max(unit["selected_tokens"] for unit in units) / selected_total
        > float(cpt.get("maximum_source_token_fraction", -1.0)) + 1e-15
    ):
        raise ValueError("curriculum CPT source/total token accounting changed")
    return content_sha


def _validate_sft_v1(sft: Mapping[str, Any]) -> str:
    content_sha = _validate_section_hash_v1(sft, "SFT dataset")
    if (
        sft.get("schema") != "specialist-verified-instruction-sft-input-v1"
        or sft.get("training_role") != "assistant_only_supervision"
        or sft.get("raw_markdown_as_assistant_answer") is not False
        or sft.get("base_pass_nonpadding_tokens") != 33_072
        or sft.get("maximum_answer_tokens") != 128
        or sft.get("maximum_sequence_tokens") != 1_024
        or sft.get("source_multiplier_min") != 2 / 3
        or sft.get("source_multiplier_max") != 3 / 2
        or sft.get("unit_objective_multiplier_cap") != 3 / 2
    ):
        raise ValueError("curriculum SFT stage contract changed")
    units = sft.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("curriculum SFT needs verified instruction units")
    facts, row_hashes = set(), set()
    by_conflict_unit: Counter[str] = Counter()
    prompt_tokens = answer_tokens = 0
    for unit in units:
        if not isinstance(unit, Mapping) or set(unit) != SFT_UNIT_FIELDS_V1:
            raise ValueError("curriculum SFT unit schema leaked raw content")
        fact = _identity_v1(unit["fact_id"], "SFT fact")
        row_hash = _sha256_v1(unit["row_sha256"], "SFT row")
        if fact in facts or row_hash in row_hashes:
            raise ValueError("curriculum SFT duplicated a source unit")
        facts.add(fact)
        row_hashes.add(row_hash)
        _sha256_v1(unit["document_sha256"], "SFT document")
        conflict = _sha256_v1(
            unit["conflict_unit_identity_sha256"], "SFT conflict unit"
        )
        _identity_v1(unit["source"], "SFT source")
        prompt = _nonnegative_int_v1(unit["prompt_tokens"], "SFT prompt tokens")
        answer = _nonnegative_int_v1(unit["answer_tokens"], "SFT answer tokens")
        if (
            prompt <= 0
            or answer <= 0
            or answer > sft["maximum_answer_tokens"]
            or prompt + answer > sft.get("maximum_sequence_tokens")
            or unit["kind"] not in ALLOWED_SFT_KINDS_V1
            or unit["quality_schema"] not in ALLOWED_SFT_SCHEMAS_V1
            or unit["verified_instructional"] is not True
            or unit["url_trivia"] is not False
            or unit["raw_markdown"] is not False
            or unit["training_role"] != "sft_verified_instruction"
            or unit["training_format"] != "masked_prompt_qa_chat"
        ):
            raise ValueError("curriculum SFT verification or stage role changed")
        declared_rows = _nonnegative_int_v1(
            unit["conflict_unit_rows"], "SFT conflict-unit rows"
        )
        if declared_rows <= 0:
            raise ValueError("curriculum SFT conflict unit is empty")
        by_conflict_unit[conflict] += 1
        prompt_tokens += prompt
        answer_tokens += answer
    if any(
        unit["conflict_unit_rows"]
        != by_conflict_unit[unit["conflict_unit_identity_sha256"]]
        for unit in units
    ):
        raise ValueError("curriculum SFT conflict-unit coverage changed")
    if (
        len(units) != sft.get("eligible_rows")
        or len(by_conflict_unit) != sft.get("conflict_units")
        or prompt_tokens != sft.get("base_pass_prompt_tokens")
        or answer_tokens != sft.get("base_pass_answer_tokens")
        or prompt_tokens + answer_tokens != sft["base_pass_nonpadding_tokens"]
        or sft.get("excluded_rows") != 15
        or sft.get("exclusion_reasons") != {"canonical_url_trivia": 15}
    ):
        raise ValueError("curriculum SFT row/token/exclusion accounting changed")
    return content_sha


def _validate_es_v1(es: Mapping[str, Any]) -> str:
    content_sha = _validate_section_hash_v1(es, "ES dataset")
    if (
        es.get("schema") != "specialist-mirrored-es-curriculum-input-v1"
        or es.get("variant") != "category_stratified"
        or es.get("rows_per_update") != 64
        or es.get("signed_population") != 16
        or es.get("directions") != 8
        or es.get("source_row_cap") != 15
        or es.get("rollouts_per_update") != 1_024
    ):
        raise ValueError("curriculum ES input contract changed")
    items = es.get("items")
    if not isinstance(items, list) or len(items) != 64:
        raise ValueError("curriculum ES panel coverage changed")
    units, rows = set(), set()
    sources: Counter[str] = Counter()
    for expected_position, item in enumerate(items):
        if not isinstance(item, Mapping) or set(item) != ES_ITEM_FIELDS_V1:
            raise ValueError("curriculum ES item schema changed")
        unit = _sha256_v1(item["unit_identity_sha256"], "ES conflict unit")
        row = _sha256_v1(item["row_sha256"], "ES row")
        source = _identity_v1(item["source"], "ES source")
        if unit in units or row in rows:
            raise ValueError("curriculum ES duplicated a source unit")
        if item["position"] != expected_position:
            raise ValueError("curriculum ES panel order changed")
        units.add(unit)
        rows.add(row)
        sources[source] += 1
    if max(sources.values()) > es["source_row_cap"]:
        raise ValueError("curriculum ES source cap exceeded")
    return content_sha


def _validate_extension_v1(
    extension: Mapping[str, Any], required_inputs: set[str], require_launch_ready: bool
) -> bool:
    if set(extension) != EXTENSION_FIELDS_V1 or extension.get(
        "schema"
    ) != EXTENSION_SCHEMA_V1:
        raise ValueError("curriculum disjoint extension schema changed")
    _sha256_v1(
        extension.get("parent_evaluation_contract_content_sha256"),
        "extension parent contract",
    )
    _sha256_v1(
        extension.get("protected_selection_identity_sha256"),
        "opaque protected selection",
    )
    if (
        set(extension.get("required_new_input_content_sha256s", ()))
        != required_inputs
        or tuple(extension.get("identity_domains", ())) != IDENTITY_DOMAINS_V1
        or extension.get("protected_semantics_opened") is not False
        or extension.get("protected_content_persisted") is not False
    ):
        raise ValueError("curriculum disjoint extension boundary changed")
    passed = extension.get("status") == "passed"
    if passed:
        _sha256_v1(extension.get("audit_file_sha256"), "extension audit file")
        _sha256_v1(extension.get("audit_content_sha256"), "extension audit content")
        collisions = extension.get("cross_role_collision_counts")
        if (
            not isinstance(collisions, Mapping)
            or set(collisions) != set(IDENTITY_DOMAINS_V1)
            or any(value != 0 for value in collisions.values())
            or extension.get("audit_passed") is not True
            or extension.get("launch_eligible") is not True
        ):
            raise ValueError("curriculum source-disjoint audit did not pass")
    elif (
        extension.get("status") != "required_not_built"
        or extension.get("audit_file_sha256") is not None
        or extension.get("audit_content_sha256") is not None
        or extension.get("cross_role_collision_counts") is not None
        or extension.get("audit_passed") is not False
        or extension.get("launch_eligible") is not False
    ):
        raise ValueError("curriculum pending extension state changed")
    if require_launch_ready and not passed:
        raise RuntimeError(
            "curriculum launch requires a fresh four-domain source-disjoint audit"
        )
    return passed


def _validate_arms_v1(plan: Mapping[str, Any], dataset_hashes: Mapping[str, str]) -> None:
    arms = plan.get("arms")
    if not isinstance(arms, list) or [arm.get("arm_id") for arm in arms] != list(
        EXPECTED_ARMS_V1
    ):
        raise ValueError("curriculum arm inventory or order changed")
    compute = plan["compute_contract"]
    target_total = _positive_number_v1(
        compute.get("target_gpu_seconds_per_arm"), "arm GPU-second target"
    )
    expected_sequences = {
        "cpt_sft_es": ("cpt", "sft", "es"),
        "direct_sft_es": ("sft", "es"),
        "direct_sft": ("sft",),
        "direct_es": ("es",),
    }
    expected_native = {
        "cpt_sft_es": ((99_216, 96, 0), (99_216, 48, 0), (0, 2, 2_048)),
        "direct_sft_es": ((198_432, 96, 0), (0, 2, 2_048)),
        "direct_sft": ((396_864, 192, 0),),
        "direct_es": ((0, 4, 4_096),),
    }
    for arm in arms:
        arm_id = arm["arm_id"]
        stages = arm.get("stages")
        if (
            not isinstance(stages, list)
            or tuple(stage.get("stage_id") for stage in stages)
            != expected_sequences[arm_id]
            or len({stage.get("stage_id") for stage in stages}) != len(stages)
        ):
            raise ValueError("curriculum stage order or leakage changed")
        total_gpu_seconds = 0.0
        for index, (stage, expected) in enumerate(
            zip(stages, expected_native[arm_id], strict=True)
        ):
            if not isinstance(stage, Mapping) or set(stage) != STAGE_FIELDS_V1:
                raise ValueError("curriculum stage schema changed")
            stage_id = stage["stage_id"]
            if stage["dataset_content_sha256"] != dataset_hashes[stage_id]:
                raise RuntimeError("curriculum stage dataset identity changed")
            tokens = _nonnegative_int_v1(
                stage["nonpadding_tokens"], "stage nonpadding tokens"
            )
            updates = _nonnegative_int_v1(
                stage["optimizer_updates"], "stage optimizer updates"
            )
            rollouts = _nonnegative_int_v1(
                stage["generated_rollouts"], "stage generated rollouts"
            )
            if (tokens, updates, rollouts) != expected:
                raise ValueError("curriculum native token/update/rollout budget changed")
            if stage_id == "sft" and tokens % plan["datasets"]["sft"][
                "base_pass_nonpadding_tokens"
            ]:
                raise ValueError("curriculum SFT budget is not complete sealed passes")
            if stage_id == "es" and rollouts != updates * plan["datasets"]["es"][
                "rollouts_per_update"
            ]:
                raise ValueError("curriculum ES rollout/update budget changed")
            expected_input = "initial_adapter" if index == 0 else "previous_stage_output"
            expected_transition = (
                "fresh_from_initial" if index == 0
                else "weights_only_new_stage_optimizer"
            )
            if (
                stage["checkpoint_input"] != expected_input
                or stage["transition_mode"] != expected_transition
            ):
                raise ValueError("curriculum checkpoint stage transition changed")
            total_gpu_seconds += _positive_number_v1(
                stage["target_gpu_seconds"], "stage GPU-second target"
            )
        if (
            arm.get("total_target_gpu_seconds") != target_total
            or not math.isclose(total_gpu_seconds, target_total, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("curriculum arms do not have equal total compute")


def validate_plan_v1(
    plan: Mapping[str, Any], *, require_launch_ready: bool = False
) -> dict:
    if not isinstance(plan, Mapping):
        raise ValueError("curriculum plan must be a mapping")
    sealed = dict(plan)
    observed_content_sha = sealed.pop("content_sha256_before_self_field", None)
    _sha256_v1(observed_content_sha, "plan content identity")
    if canonical_sha256_v1(sealed) != observed_content_sha:
        raise RuntimeError("curriculum plan content identity changed")
    if plan.get("schema") != PLAN_SCHEMA_V1:
        raise ValueError("curriculum plan schema changed")
    evaluation = plan.get("evaluation_contract", {})
    if (
        set(evaluation) != EVALUATION_FIELDS_V1
        or evaluation.get("protected_access_authorized") is not False
        or evaluation.get("disjointness_passed") is not True
        or evaluation.get("new_curriculum_inputs_currently_authorized") is not False
    ):
        raise ValueError("curriculum parent evaluation firewall changed")
    _sha256_v1(evaluation.get("file_sha256"), "evaluation contract file")
    parent_content = _sha256_v1(
        evaluation.get("content_sha256"), "evaluation contract content"
    )
    protected_identity = _sha256_v1(
        evaluation.get("protected_selection_identity_sha256"),
        "protected selection identity",
    )
    allowlist = evaluation.get("current_adaptation_allowlist")
    if (
        not isinstance(allowlist, list)
        or len(allowlist) != 1
        or _sha256_v1(allowlist[0], "current adaptation allowlist") != allowlist[0]
    ):
        raise ValueError("curriculum parent adaptation allowlist changed")
    datasets = plan.get("datasets", {})
    if set(datasets) != {"cpt", "sft", "es"}:
        raise ValueError("curriculum dataset stage inventory changed")
    dataset_hashes = {
        "cpt": _validate_cpt_v1(datasets["cpt"]),
        "sft": _validate_sft_v1(datasets["sft"]),
        "es": _validate_es_v1(datasets["es"]),
    }
    extension = plan.get("source_disjoint_extension", {})
    if (
        extension.get("parent_evaluation_contract_content_sha256")
        != parent_content
        or extension.get("protected_selection_identity_sha256")
        != protected_identity
    ):
        raise RuntimeError("curriculum extension parent identity changed")
    extension_passed = _validate_extension_v1(
        extension, set(dataset_hashes.values()), require_launch_ready
    )
    authorization = plan.get("authorization", {})
    expected_gpu_authorization = extension_passed
    if (
        set(authorization) != AUTHORIZATION_FIELDS_V1
        or authorization.get(
            "future_launch_requires_disjoint_extension_and_gpu_throughput_calibration"
        ) is not True
        or authorization.get("protected_holdout_access") is not False
        or authorization.get("protected_holdout_selection_or_tuning") is not False
        or authorization.get("gpu_launch") is not expected_gpu_authorization
        or authorization.get("new_adaptation_inputs") is not extension_passed
    ):
        raise ValueError("curriculum authorization exceeds its disjoint audit")
    compute = plan.get("compute_contract", {})
    if (
        compute.get("mode") != "compute_matched_quality"
        or compute.get("target_gpu_seconds_per_arm") != 14_400
        or compute.get("relative_match_tolerance") != 0.02
        or compute.get("physical_gpu_ids") != [0, 1, 2, 3]
        or compute.get("useful_activity_each_gpu_required") is not True
    ):
        raise ValueError("curriculum equal-compute contract changed")
    _validate_arms_v1(plan, dataset_hashes)
    checkpoint = plan.get("checkpoint_contract", {})
    _sha256_v1(checkpoint.get("initial_adapter_state_sha256"), "initial adapter")
    _sha256_v1(checkpoint.get("adapter_surface_sha256"), "adapter surface")
    if (
        checkpoint.get("cross_stage_transition")
        != "weights_only_exact_previous_output_new_optimizer_scheduler"
        or checkpoint.get("same_stage_resume_requires")
        != [
            "exact_model_state",
            "exact_optimizer_state",
            "exact_scheduler_state",
            "exact_rng_state",
            "exact_dataloader_cursor",
        ]
        or checkpoint.get("unregistered_checkpoint_fallback") != "prohibited"
    ):
        raise ValueError("curriculum checkpoint identity contract changed")
    access = plan.get("access_receipt", {})
    if (
        set(access) != ACCESS_FIELDS_V1
        or access.get("CPT_train_candidate_markdown_opened") is not True
        or access.get("SFT_registered_train_semantics_opened") is not True
        or access.get("protected_holdout_semantics_opened") is not False
        or access.get("dev_or_ood_semantics_opened") is not False
        or access.get("gpu_launched") is not False
        or access.get("model_weights_loaded") is not False
        or access.get("live_run_directory_touched") is not False
    ):
        raise ValueError("curriculum CPU access receipt changed")
    if require_launch_ready and plan.get("status") != "sealed_launch_ready":
        raise RuntimeError("curriculum launch-ready status is absent")
    if not require_launch_ready and plan.get("status") not in {
        "sealed_cpu_preview_launch_ineligible",
        "sealed_launch_ready",
    }:
        raise ValueError("curriculum plan status changed")
    return dict(plan)


def validate_execution_receipts_v1(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict:
    """Validate native budgets, exact resume chains, and equal GPU seconds."""
    plan = validate_plan_v1(plan, require_launch_ready=True)
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise ValueError("curriculum receipts must be a sequence")
    by_arm = {}
    for receipt in receipts:
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != EXECUTION_RECEIPT_FIELDS_V1
            or receipt.get("schema") != RECEIPT_SCHEMA_V1
        ):
            raise ValueError("curriculum execution receipt schema changed")
        arm_id = receipt.get("arm_id")
        if arm_id in by_arm or arm_id not in EXPECTED_ARMS_V1:
            raise ValueError("curriculum execution arm coverage changed")
        if (
            receipt.get("protected_holdout_opened") is not False
            or receipt.get("useful_physical_gpu_ids") != [0, 1, 2, 3]
        ):
            raise ValueError("curriculum execution access/GPU coverage changed")
        by_arm[arm_id] = receipt
    if set(by_arm) != set(EXPECTED_ARMS_V1):
        raise ValueError("curriculum execution receipt matrix is incomplete")
    initial = plan["checkpoint_contract"]["initial_adapter_state_sha256"]
    tolerance = plan["compute_contract"]["relative_match_tolerance"]
    target = plan["compute_contract"]["target_gpu_seconds_per_arm"]
    totals = []
    arm_plans = {arm["arm_id"]: arm for arm in plan["arms"]}
    for arm_id in EXPECTED_ARMS_V1:
        receipt = by_arm[arm_id]
        stages = receipt.get("stage_receipts")
        expected_stages = arm_plans[arm_id]["stages"]
        if not isinstance(stages, list) or len(stages) != len(expected_stages):
            raise ValueError("curriculum stage receipt coverage changed")
        expected_input = initial
        stage_gpu_total = 0.0
        for expected_stage, observed in zip(expected_stages, stages, strict=True):
            if (
                not isinstance(observed, Mapping)
                or set(observed) != STAGE_RECEIPT_FIELDS_V1
                or observed.get("stage_id") != expected_stage["stage_id"]
            ):
                raise ValueError("curriculum stage receipt order changed")
            input_checkpoint = _sha256_v1(
                observed.get("input_checkpoint_sha256"), "stage input checkpoint"
            )
            output_checkpoint = _sha256_v1(
                observed.get("output_checkpoint_sha256"), "stage output checkpoint"
            )
            if input_checkpoint != expected_input:
                raise RuntimeError("curriculum resumed/checkpoint chain drifted")
            if observed.get("dataset_content_sha256") != expected_stage[
                "dataset_content_sha256"
            ]:
                raise RuntimeError("curriculum execution dataset identity drifted")
            for field in ("optimizer_updates", "nonpadding_tokens", "generated_rollouts"):
                if observed.get(field) != expected_stage[field]:
                    raise ValueError(f"curriculum execution {field} budget drifted")
            segments = observed.get("resume_segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError("curriculum stage needs a resume-chain receipt")
            segment_checkpoint = input_checkpoint
            training_state = None
            ending_update = 0
            for index, segment in enumerate(segments):
                if (
                    not isinstance(segment, Mapping)
                    or set(segment) != RESUME_SEGMENT_FIELDS_V1
                    or segment.get("segment_index") != index
                ):
                    raise ValueError("curriculum resume segment order changed")
                segment_input = _sha256_v1(
                    segment.get("input_checkpoint_sha256"),
                    "resume segment input checkpoint",
                )
                segment_output = _sha256_v1(
                    segment.get("output_checkpoint_sha256"),
                    "resume segment output checkpoint",
                )
                input_state = _sha256_v1(
                    segment.get("input_training_state_sha256"),
                    "resume input training state",
                )
                output_state = _sha256_v1(
                    segment.get("output_training_state_sha256"),
                    "resume output training state",
                )
                if (
                    segment_input != segment_checkpoint
                    or segment.get("starting_update") != ending_update
                    or (index > 0 and input_state != training_state)
                ):
                    raise RuntimeError("curriculum same-stage resume state drifted")
                ending_update = _nonnegative_int_v1(
                    segment.get("ending_update"), "resume ending update"
                )
                if ending_update <= segment.get("starting_update"):
                    raise ValueError("curriculum resume segment did not advance")
                segment_checkpoint = segment_output
                training_state = output_state
            if (
                segment_checkpoint != output_checkpoint
                or ending_update != expected_stage["optimizer_updates"]
            ):
                raise RuntimeError("curriculum resume chain did not reach stage output")
            stage_gpu = _positive_number_v1(
                observed.get("charged_gpu_seconds"), "stage charged GPU seconds"
            )
            if abs(stage_gpu - expected_stage["target_gpu_seconds"]) / expected_stage[
                "target_gpu_seconds"
            ] > tolerance:
                raise ValueError("curriculum stage GPU-second budget drifted")
            stage_gpu_total += stage_gpu
            expected_input = output_checkpoint
        total_gpu = _positive_number_v1(
            receipt.get("total_charged_gpu_seconds"), "arm charged GPU seconds"
        )
        if (
            not math.isclose(total_gpu, stage_gpu_total, rel_tol=0, abs_tol=1e-9)
            or abs(total_gpu - target) / target > tolerance
            or receipt.get("final_checkpoint_sha256") != expected_input
        ):
            raise ValueError("curriculum arm compute/final checkpoint receipt drifted")
        totals.append(total_gpu)
    if (max(totals) - min(totals)) / target > tolerance:
        raise ValueError("curriculum arms are not equal-compute")
    result = {
        "schema": "specialist-curriculum-execution-validation-v1",
        "arm_gpu_seconds": dict(zip(EXPECTED_ARMS_V1, totals, strict=True)),
        "all_native_budgets_exact": True,
        "all_checkpoint_and_resume_chains_exact": True,
        "all_four_gpus_useful_per_arm": True,
        "protected_holdout_opened": False,
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result
