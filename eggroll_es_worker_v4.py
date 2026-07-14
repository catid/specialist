"""Layer-restricted, partition-audited worker methods for EGGROLL-ES v4.

V4 keeps v3's four-engine, strided-seed FP32 all-reduce, but only a frozen
set of model parameters may be perturbed, retained as exact CPU references,
or updated.  Every unselected parameter is instead protected by one immutable
origin identity.  Population verification, update completion, state
inspection, and abort all re-hash that partition and fail closed on any drift.

The only production plans accepted by this module are the two checked-in,
parameter-matched dense plans.  Tests may replace ``FROZEN_LAYER_PLANS_V4``
on a tiny model; runtime code cannot opt out through an RPC argument.
"""

import hashlib
import json
import math
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

from eggroll_es_worker_v3 import (
    REQUIRED_ENGINE_COUNT,
    DistributedExactAuditWorkerExtensionV3,
    accumulate_seed_terms_v3,
    canonical_sha256_v3,
    coefficient_sha256_v3,
    seed_shard_v3,
    validate_seed_coefficients_v3,
)


FROZEN_LAYER_PLANS_V4 = {
    "6af34ef41187d8b08f53b9dab1e40102744b954c80146c130bd2c053fc3f52cb": {
        "plan": "front_back",
        "file_sha256": (
            "8e855cbd0d6130278e87b1af348e39dd0f683b8575d9abcb9260f3fe7b29d824"
        ),
        "source_unit_count": 70,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "selected_byte_count": 571_998_208,
        "checkpoint_to_runtime_mapping_sha256": (
            "0a1b84e8ed53ef56c174e7fcac728a4820293505647ab6b9ea02bc86a012b3b1"
        ),
        "runtime_selected_name_sha256": (
            "417b3867ba9a56f909d01b1e7bb0b8bb04f903c3ec49438a6675239a7bab270f"
        ),
    },
    "b5e4e162116695e5d2544e24c2e0cdfb49ca8783aa6f9d707ef41d6f725ca5e0": {
        "plan": "middle_matched",
        "file_sha256": (
            "f2b38054e3cdaf41619cce579d3ba2e030fa3cfa87fd42b50543f655ff5f6dc0"
        ),
        "source_unit_count": 70,
        "runtime_selected_parameter_count": 46,
        "selected_element_count": 285_999_104,
        "selected_byte_count": 571_998_208,
        "checkpoint_to_runtime_mapping_sha256": (
            "d6f43de81bb5c41318a38f077b8a3e6272676801752ff68d4772977ac72182f7"
        ),
        "runtime_selected_name_sha256": (
            "a7df9257f81c05a3fb3e858209486bd930aad0ddb94d7398e1644b779fb8b70d"
        ),
    },
}

# Each engine owns a full 35B checkpoint.  A serial SHA-256 pass leaves its
# GPU waiting on one CPU core for minutes.  Hash independent parameter records
# in parallel, then SHA-256 their ordered record digests into the partition
# root.  The construction is deterministic across worker counts and retains a
# cryptographic byte commitment for every tensor while bounding host staging.
PARTITION_HASH_WORKERS_V4 = 8
PARTITION_HASH_MAX_INFLIGHT_V4 = 16


def _require_sha256_v4(name, value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity")
    return value


def _strict_json_loads_v4(raw):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        return json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("layer plan is not strict UTF-8 JSON") from error


def validate_frozen_layer_plan_v4(
    raw_layer_plan_bytes,
    expected_file_sha256,
    expected_plan_sha256,
):
    """Independently validate exact bytes and canonical semantic identity."""
    if not isinstance(raw_layer_plan_bytes, bytes):
        raise TypeError("raw layer plan must be provided as bytes")
    expected_file_sha256 = _require_sha256_v4(
        "layer plan file SHA-256", expected_file_sha256,
    )
    expected_plan_sha256 = _require_sha256_v4(
        "layer plan SHA-256", expected_plan_sha256,
    )
    frozen = FROZEN_LAYER_PLANS_V4.get(expected_plan_sha256)
    if not isinstance(frozen, dict):
        raise ValueError("layer plan semantic identity is not frozen for v4")
    if frozen.get("file_sha256") != expected_file_sha256:
        raise ValueError("layer plan file/semantic identity pair is not frozen")
    actual_file_sha256 = hashlib.sha256(raw_layer_plan_bytes).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError("raw layer plan file identity changed")
    plan = _strict_json_loads_v4(raw_layer_plan_bytes)
    if not isinstance(plan, dict):
        raise ValueError("layer plan JSON root must be an object")
    claimed_plan_sha256 = plan.get("plan_sha256")
    if claimed_plan_sha256 != expected_plan_sha256:
        raise ValueError("layer plan claimed a different semantic identity")
    canonical_payload = dict(plan)
    canonical_payload.pop("plan_sha256", None)
    actual_plan_sha256 = canonical_sha256_v3(canonical_payload)
    if actual_plan_sha256 != expected_plan_sha256:
        raise ValueError("layer plan canonical semantic identity changed")
    if plan.get("schema") != "qwen36-es-layer-plan-v1":
        raise ValueError("unsupported layer plan schema")
    if plan.get("plan") != frozen.get("plan"):
        raise ValueError("frozen layer plan name changed")
    if plan.get("groups") != ["dense"]:
        raise ValueError("v4 requires the frozen dense unit group")
    units = plan.get("units")
    if (
        not isinstance(units, list)
        or not units
        or any(not isinstance(name, str) or not name for name in units)
        or len(set(units)) != len(units)
        or units != sorted(units)
    ):
        raise ValueError("layer plan units must be unique sorted names")
    if (
        isinstance(plan.get("num_units"), bool)
        or plan.get("num_units") != len(units)
        or len(units) != frozen.get("source_unit_count")
    ):
        raise ValueError("layer plan selected parameter count changed")
    include_regex = plan.get("include_regex")
    if not isinstance(include_regex, str) or not include_regex:
        raise ValueError("layer plan include regex is missing")
    try:
        compiled = re.compile(include_regex)
    except re.error as error:
        raise ValueError("layer plan include regex is invalid") from error
    if any(compiled.fullmatch(name) is None for name in units):
        raise ValueError("layer plan regex does not select every declared unit")
    return plan, dict(frozen)


_CHECKPOINT_LAYER_PREFIX_V4 = "model.language_model.layers."
_RUNTIME_LAYER_PREFIX_V4 = "language_model.model.layers."


def _checkpoint_layer_parts_v4(checkpoint_name):
    if not isinstance(checkpoint_name, str):
        raise ValueError("checkpoint unit name is not a string")
    match = re.fullmatch(
        r"model\.language_model\.layers\.(\d+)\.(.+)", checkpoint_name,
    )
    if match is None:
        raise ValueError(
            f"unsupported checkpoint unit name for v4: {checkpoint_name!r}"
        )
    return match.groups()


def checkpoint_to_runtime_parameter_v4(checkpoint_name):
    """Map one frozen HF checkpoint unit to its packed vLLM parameter."""
    layer, suffix = _checkpoint_layer_parts_v4(checkpoint_name)
    replacements = {
        "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkvz.weight",
        "linear_attn.in_proj_z.weight": "linear_attn.in_proj_qkvz.weight",
        "linear_attn.in_proj_b.weight": "linear_attn.in_proj_ba.weight",
        "linear_attn.in_proj_a.weight": "linear_attn.in_proj_ba.weight",
        "self_attn.q_proj.weight": "self_attn.qkv_proj.weight",
        "self_attn.k_proj.weight": "self_attn.qkv_proj.weight",
        "self_attn.v_proj.weight": "self_attn.qkv_proj.weight",
        "mlp.shared_expert.gate_proj.weight": (
            "mlp.shared_expert.gate_up_proj.weight"
        ),
        "mlp.shared_expert.up_proj.weight": (
            "mlp.shared_expert.gate_up_proj.weight"
        ),
    }
    allowed_unpacked = {
        "linear_attn.out_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.gate.weight",
    }
    if suffix in replacements:
        runtime_suffix = replacements[suffix]
    elif suffix in allowed_unpacked:
        runtime_suffix = suffix
    else:
        raise ValueError(
            f"checkpoint unit has no audited v4 runtime mapping: {checkpoint_name!r}"
        )
    return f"{_RUNTIME_LAYER_PREFIX_V4}{layer}.{runtime_suffix}"


def checkpoint_runtime_mapping_v4(checkpoint_units):
    """Return a complete, deterministic 70-source to 46-runtime mapping."""
    if (
        not isinstance(checkpoint_units, list)
        or any(not isinstance(name, str) for name in checkpoint_units)
        or len(set(checkpoint_units)) != len(checkpoint_units)
    ):
        raise ValueError("checkpoint units must be a unique list")
    rows = [{
        "checkpoint_unit": name,
        "runtime_parameter": checkpoint_to_runtime_parameter_v4(name),
    } for name in sorted(checkpoint_units)]
    reverse = {}
    for row in rows:
        reverse.setdefault(row["runtime_parameter"], []).append(
            row["checkpoint_unit"]
        )

    # Packed parameters are safe to perturb as a whole only when every source
    # shard that constitutes the runtime tensor is selected.
    for runtime_name, sources in reverse.items():
        source_suffixes = {
            _checkpoint_layer_parts_v4(name)[1]
            for name in sources
        }
        if runtime_name.endswith("linear_attn.in_proj_qkvz.weight"):
            expected = {
                "linear_attn.in_proj_qkv.weight",
                "linear_attn.in_proj_z.weight",
            }
        elif runtime_name.endswith("linear_attn.in_proj_ba.weight"):
            expected = {
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
            }
        elif runtime_name.endswith("self_attn.qkv_proj.weight"):
            expected = {
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            }
        elif runtime_name.endswith("mlp.shared_expert.gate_up_proj.weight"):
            expected = {
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
            }
        else:
            expected = source_suffixes
        if source_suffixes != expected:
            raise ValueError(
                f"layer plan selects a partial packed runtime parameter: "
                f"{runtime_name}"
            )
    runtime_names = sorted(reverse)
    return {
        "schema": "eggroll-es-checkpoint-runtime-mapping-v4",
        "rows": rows,
        "mapping_sha256": canonical_sha256_v3(rows),
        "source_unit_count": len(rows),
        "runtime_selected_parameter_count": len(runtime_names),
        "runtime_selected_names": runtime_names,
        "runtime_selected_name_sha256": canonical_sha256_v3(runtime_names),
    }


def validate_dense_reward_identity_v4(config, expected_sha256):
    expected_sha256 = _require_sha256_v4(
        "dense reward SHA-256", expected_sha256,
    )
    if (
        not isinstance(config, dict)
        or config.get("schema") != "eggroll-es-dense-qa-reward-v1"
    ):
        raise ValueError("unsupported dense reward configuration")
    actual = canonical_sha256_v3(config)
    if actual != expected_sha256:
        raise ValueError("dense reward configuration identity changed")
    return json.loads(json.dumps(config, allow_nan=False)), actual


def update_manifest_v4(
    *,
    coefficient_sha256,
    population_size,
    world_size,
    reference_generation,
    plan_id,
    update_sequence,
    previous_alpha,
    target_alpha,
    expected_base_sha256,
    layer_plan_file_sha256,
    layer_plan_sha256,
    checkpoint_to_runtime_mapping_sha256,
    source_unit_count,
    runtime_selected_name_sha256,
    selected_parameter_manifest_sha256,
    runtime_selected_parameter_count,
    selected_element_count,
    unselected_origin_sha256,
    dense_reward_sha256,
):
    """Build the complete controller/worker contract for one v4 increment."""
    return {
        "schema": "eggroll-es-layer-restricted-update-manifest-v4",
        "coefficient_sha256": str(coefficient_sha256),
        "population_size": int(population_size),
        "world_size": int(world_size),
        "reference_generation": int(reference_generation),
        "plan_id": str(plan_id),
        "update_sequence": int(update_sequence),
        "previous_alpha": float(previous_alpha),
        "target_alpha": float(target_alpha),
        "expected_base_sha256": str(expected_base_sha256),
        "layer_plan_file_sha256": str(layer_plan_file_sha256),
        "layer_plan_sha256": str(layer_plan_sha256),
        "checkpoint_to_runtime_mapping_sha256": str(
            checkpoint_to_runtime_mapping_sha256
        ),
        "source_unit_count": int(source_unit_count),
        "runtime_selected_name_sha256": str(runtime_selected_name_sha256),
        "selected_parameter_manifest_sha256": str(
            selected_parameter_manifest_sha256
        ),
        "runtime_selected_parameter_count": int(
            runtime_selected_parameter_count
        ),
        "selected_element_count": int(selected_element_count),
        "unselected_origin_sha256": str(unselected_origin_sha256),
        "dense_reward_sha256": str(dense_reward_sha256),
    }


class LayerRestrictedExactAuditWorkerExtensionV4(
    DistributedExactAuditWorkerExtensionV3,
):
    """Selected-only exact ES operations with an immutable complement guard."""

    def _require_layer_plan_v4(self):
        if getattr(self, "_v4_layer_plan_installed", False) is not True:
            raise RuntimeError("v4 frozen layer plan has not been installed")

    def _parameter_layout_v4(self):
        parameters = []
        seen = set()
        for name, parameter in self.model_runner.model.named_parameters():
            name = str(name)
            if name in seen:
                raise RuntimeError(f"duplicate model parameter name: {name!r}")
            if not parameter.data.is_contiguous():
                raise RuntimeError(
                    f"v4 cannot byte-audit noncontiguous parameter {name!r}"
                )
            seen.add(name)
            parameters.append((name, parameter))
        if not parameters:
            raise RuntimeError("layer-restricted model has no parameters")
        manifest = [{
            "name": name,
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
        } for name, parameter in parameters]
        return parameters, manifest, canonical_sha256_v3(manifest)

    def _validated_parameters_v4(self):
        self._require_layer_plan_v4()
        parameters, manifest, manifest_sha = self._parameter_layout_v4()
        if manifest_sha != self._v4_full_parameter_manifest_sha256:
            raise RuntimeError("model parameter layout changed after plan install")
        selected = [
            (name, parameter) for name, parameter in parameters
            if name in self._v4_selected_name_set
        ]
        if [name for name, _ in selected] != self._v4_selected_names:
            raise RuntimeError("selected parameter order changed")
        selected_elements = sum(int(parameter.numel()) for _, parameter in selected)
        if (
            len(selected) != self._v4_selected_parameter_count
            or selected_elements != self._v4_selected_element_count
        ):
            raise RuntimeError("selected parameter count or elements changed")
        return parameters, selected

    def _partition_prefix_v4(self, partition):
        return {
            "schema": "eggroll-es-parameter-partition-prefix-v4",
            "partition": partition,
            "layer_plan_file_sha256": self._v4_layer_plan_file_sha256,
            "layer_plan_sha256": self._v4_layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                self._v4_checkpoint_to_runtime_mapping_sha256
            ),
            "runtime_selected_name_sha256": (
                self._v4_runtime_selected_name_sha256
            ),
            "selected_parameter_manifest_sha256": (
                self._v4_selected_parameter_manifest_sha256
            ),
            "dense_reward_sha256": self._v4_dense_reward_sha256,
        }

    def _partition_identity_v4(
        self, partition, named_tensors, chunk_bytes,
    ):
        chunk_bytes = int(chunk_bytes)
        if chunk_bytes <= 0:
            raise ValueError("digest chunk size must be positive")
        named_tensors = list(named_tensors)
        if not named_tensors:
            raise RuntimeError(f"v4 {partition} partition is empty")
        digest = hashlib.sha256()
        prefix = json.dumps(
            self._partition_prefix_v4(partition),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        digest.update(len(prefix).to_bytes(8, "little"))
        digest.update(prefix)
        digest.update(len(named_tensors).to_bytes(8, "little"))
        total_bytes = 0
        total_elements = sum(int(tensor.numel()) for _, tensor in named_tensors)

        def tensor_record(name, cpu_tensor):
            record = hashlib.sha256()
            byte_count = self._update_tensor_hash(
                record, name, cpu_tensor, chunk_bytes,
            )
            return record.digest(), byte_count

        worker_count = min(PARTITION_HASH_WORKERS_V4, len(named_tensors))
        inflight_limit = max(
            worker_count, min(PARTITION_HASH_MAX_INFLIGHT_V4,
                              len(named_tensors)),
        )
        pending = deque()

        def consume_oldest():
            nonlocal total_bytes
            record_digest, byte_count = pending.popleft().result()
            digest.update(len(record_digest).to_bytes(8, "little"))
            digest.update(record_digest)
            total_bytes += byte_count

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for name, tensor in named_tensors:
                # D2H copies remain ordered and bounded, while OpenSSL SHA-256
                # runs on earlier CPU copies in parallel with later transfers.
                cpu_tensor = tensor.detach().to(device="cpu", copy=True)
                pending.append(executor.submit(
                    tensor_record, name, cpu_tensor,
                ))
                if len(pending) >= inflight_limit:
                    consume_oldest()
            while pending:
                consume_oldest()
        return {
            "schema": "eggroll-es-parameter-partition-sha256-v4",
            "partition": partition,
            "sha256": digest.hexdigest(),
            "parameter_count": len(named_tensors),
            "total_elements": total_elements,
            "total_bytes": total_bytes,
            "layer_plan_file_sha256": self._v4_layer_plan_file_sha256,
            "layer_plan_sha256": self._v4_layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                self._v4_checkpoint_to_runtime_mapping_sha256
            ),
            "runtime_selected_name_sha256": (
                self._v4_runtime_selected_name_sha256
            ),
            "selected_parameter_manifest_sha256": (
                self._v4_selected_parameter_manifest_sha256
            ),
            "dense_reward_sha256": self._v4_dense_reward_sha256,
        }

    def _combined_identity_v4(self, selected, unselected):
        payload = {
            "schema": "eggroll-es-partitioned-weight-payload-v4",
            "layer_plan_file_sha256": self._v4_layer_plan_file_sha256,
            "layer_plan_sha256": self._v4_layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                self._v4_checkpoint_to_runtime_mapping_sha256
            ),
            "source_unit_count": self._v4_source_unit_count,
            "runtime_selected_name_sha256": (
                self._v4_runtime_selected_name_sha256
            ),
            "selected_parameter_manifest_sha256": (
                self._v4_selected_parameter_manifest_sha256
            ),
            "dense_reward_sha256": self._v4_dense_reward_sha256,
            "selected": selected,
            "unselected": unselected,
        }
        return {
            "schema": "eggroll-es-partitioned-weight-state-v4",
            "sha256": canonical_sha256_v3(payload),
            "layer_plan_file_sha256": self._v4_layer_plan_file_sha256,
            "layer_plan_sha256": self._v4_layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                self._v4_checkpoint_to_runtime_mapping_sha256
            ),
            "source_unit_count": self._v4_source_unit_count,
            "runtime_selected_name_sha256": (
                self._v4_runtime_selected_name_sha256
            ),
            "selected_parameter_manifest_sha256": (
                self._v4_selected_parameter_manifest_sha256
            ),
            "dense_reward_sha256": self._v4_dense_reward_sha256,
            "selected": selected,
            "unselected": unselected,
        }

    def _record_full_audit_v4(self, identity, phase):
        if (
            not isinstance(identity, dict)
            or identity.get("schema")
            != "eggroll-es-partitioned-weight-state-v4"
            or not isinstance(phase, str)
            or not phase
        ):
            raise RuntimeError("v4 full-audit cache input is invalid")
        self._v4_last_full_audit_identity = dict(identity)
        self._v4_last_full_audit_phase = phase
        return identity

    def _require_full_audit_v4(self, phase):
        identity = getattr(self, "_v4_last_full_audit_identity", None)
        current = getattr(self, "_v3_current_identity", None)
        if (
            getattr(self, "_v4_last_full_audit_phase", None) != phase
            or not isinstance(identity, dict)
            or (
                current is None
                and phase != "layer_plan_install"
            )
            or (current is not None and identity != current)
        ):
            raise RuntimeError(
                f"v4 cached state lacks required full audit phase {phase!r}"
            )
        return dict(identity)

    def _partitioned_weight_state_v4(
        self, chunk_bytes=64 * 1024 * 1024, require_unselected_origin=True,
    ):
        parameters, selected = self._validated_parameters_v4()
        selected_names = self._v4_selected_name_set
        unselected = [
            (name, parameter) for name, parameter in parameters
            if name not in selected_names
        ]
        selected_identity = self._partition_identity_v4(
            "selected", selected, chunk_bytes,
        )
        unselected_identity = self._partition_identity_v4(
            "unselected", unselected, chunk_bytes,
        )
        origin = getattr(self, "_v4_unselected_origin_identity", None)
        if require_unselected_origin:
            if not isinstance(origin, dict):
                raise RuntimeError("unselected origin identity is missing")
            if unselected_identity != origin:
                raise RuntimeError(
                    "unselected parameter bytes drifted from immutable origin"
                )
        return self._combined_identity_v4(
            selected_identity, unselected_identity,
        )

    def _unselected_identity_v4(self, chunk_bytes=64 * 1024 * 1024):
        parameters, _ = self._validated_parameters_v4()
        return self._partition_identity_v4(
            "unselected",
            [
                (name, parameter) for name, parameter in parameters
                if name not in self._v4_selected_name_set
            ],
            chunk_bytes,
        )

    def _verify_unselected_origin_v4(
        self, phase, chunk_bytes=64 * 1024 * 1024,
    ):
        current = self._unselected_identity_v4(chunk_bytes=chunk_bytes)
        origin = getattr(self, "_v4_unselected_origin_identity", None)
        if not isinstance(origin, dict) or current != origin:
            raise RuntimeError(
                f"unselected parameter bytes drifted from immutable origin "
                f"{phase}"
            )
        return current

    def _binding_fields_v4(self):
        self._require_layer_plan_v4()
        return {
            "layer_plan_file_sha256": self._v4_layer_plan_file_sha256,
            "layer_plan_sha256": self._v4_layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                self._v4_checkpoint_to_runtime_mapping_sha256
            ),
            "source_unit_count": self._v4_source_unit_count,
            "runtime_selected_name_sha256": (
                self._v4_runtime_selected_name_sha256
            ),
            "selected_parameter_manifest_sha256": (
                self._v4_selected_parameter_manifest_sha256
            ),
            "runtime_selected_parameter_count": (
                self._v4_selected_parameter_count
            ),
            "selected_element_count": self._v4_selected_element_count,
            "unselected_origin_sha256": (
                self._v4_unselected_origin_identity["sha256"]
            ),
            "dense_reward_sha256": self._v4_dense_reward_sha256,
        }

    def install_layer_plan_v4(
        self,
        raw_layer_plan_bytes,
        expected_file_sha256,
        expected_plan_sha256,
        dense_reward_config,
        expected_dense_reward_sha256,
        chunk_bytes=64 * 1024 * 1024,
    ):
        """Install the immutable plan before the first exact reference."""
        if getattr(self, "_v3_pending_update", None) is not None:
            raise RuntimeError("cannot install a layer plan during an update")
        reference_present = bool(
            getattr(self, "exact_reference_weights", None)
        ) or isinstance(
            getattr(self, "exact_reference_identity", None), dict,
        ) or isinstance(
            getattr(self, "_v3_reference_identity", None), dict,
        )
        reference_generation = int(
            getattr(self, "_v3_reference_generation", 0)
        )
        if reference_present or reference_generation != 0:
            raise RuntimeError(
                "v4 layer plan must be installed before exact reference capture"
            )
        communicator = self._communicator_state_v3(REQUIRED_ENGINE_COUNT)
        plan, frozen = validate_frozen_layer_plan_v4(
            raw_layer_plan_bytes,
            expected_file_sha256,
            expected_plan_sha256,
        )
        mapping = checkpoint_runtime_mapping_v4(plan["units"])
        if (
            mapping["source_unit_count"] != frozen["source_unit_count"]
            or mapping["runtime_selected_parameter_count"]
            != frozen["runtime_selected_parameter_count"]
            or mapping["mapping_sha256"]
            != frozen["checkpoint_to_runtime_mapping_sha256"]
            or mapping["runtime_selected_name_sha256"]
            != frozen["runtime_selected_name_sha256"]
        ):
            raise RuntimeError(
                "checkpoint/runtime mapping changed from frozen plan"
            )
        dense_reward_config, dense_reward_sha256 = (
            validate_dense_reward_identity_v4(
                dense_reward_config, expected_dense_reward_sha256,
            )
        )
        installation_identity = canonical_sha256_v3({
            "schema": "eggroll-es-layer-installation-v4",
            "layer_plan_file_sha256": expected_file_sha256,
            "layer_plan_sha256": expected_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                mapping["mapping_sha256"]
            ),
            "dense_reward_sha256": dense_reward_sha256,
        })
        if getattr(self, "_v4_layer_plan_installed", False):
            if installation_identity != self._v4_installation_identity:
                raise RuntimeError("v4 layer plan cannot be reconfigured")
            current = self._partitioned_weight_state_v4(
                chunk_bytes=chunk_bytes, require_unselected_origin=True,
            )
            if current != self._v4_installed_identity:
                raise RuntimeError(
                    "v4 model bytes changed after layer plan installation"
                )
            return {
                "schema": "eggroll-es-layer-plan-installed-v4",
                "installed": True,
                "idempotent": True,
                "installation_identity": installation_identity,
                "reference_present_before_install": False,
                "reference_generation_before_install": 0,
                "rank": communicator["rank"],
                "world_size": communicator["world_size"],
                "selected_byte_count": self._v4_selected_byte_count,
                "plan": self._v4_layer_plan["plan"],
                "full_parameter_manifest_sha256": (
                    self._v4_full_parameter_manifest_sha256
                ),
                "initial_identity": current,
                **self._binding_fields_v4(),
            }

        parameters, full_manifest, full_manifest_sha = (
            self._parameter_layout_v4()
        )
        model_names = [name for name, _ in parameters]
        model_name_set = set(model_names)
        runtime_selected_names = mapping["runtime_selected_names"]
        runtime_selected_set = set(runtime_selected_names)
        missing = runtime_selected_set - model_name_set
        if missing:
            raise RuntimeError(
                "layer plan mapped to unknown runtime parameters: "
                + ", ".join(sorted(missing))
            )
        selected_parameters = [
            (name, parameter) for name, parameter in parameters
            if name in runtime_selected_set
        ]
        selected_names = [name for name, _ in selected_parameters]
        selected_elements = sum(
            int(parameter.numel()) for _, parameter in selected_parameters
        )
        selected_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for _, parameter in selected_parameters
        )
        if (
            len(selected_parameters)
            != frozen["runtime_selected_parameter_count"]
            or selected_elements != frozen["selected_element_count"]
            or selected_bytes != frozen["selected_byte_count"]
            or any(
                parameter.dtype != torch.bfloat16
                for _, parameter in selected_parameters
            )
        ):
            raise RuntimeError(
                "model does not match frozen BF16 selected count/elements/bytes"
            )
        selected_manifest = [
            item for item in full_manifest
            if item["name"] in runtime_selected_set
        ]
        selected_manifest_sha = canonical_sha256_v3({
            "schema": "eggroll-es-selected-parameter-manifest-v4",
            "layer_plan_file_sha256": expected_file_sha256,
            "layer_plan_sha256": expected_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                mapping["mapping_sha256"]
            ),
            "source_unit_count": mapping["source_unit_count"],
            "runtime_selected_name_sha256": (
                mapping["runtime_selected_name_sha256"]
            ),
            "parameters": selected_manifest,
        })

        installed_attributes = {
            "_v4_layer_plan_installed": True,
            "_v4_installation_identity": installation_identity,
            "_v4_layer_plan": plan,
            "_v4_layer_plan_file_sha256": expected_file_sha256,
            "_v4_layer_plan_sha256": expected_plan_sha256,
            "_v4_checkpoint_runtime_mapping": mapping,
            "_v4_checkpoint_to_runtime_mapping_sha256": (
                mapping["mapping_sha256"]
            ),
            "_v4_source_unit_count": mapping["source_unit_count"],
            "_v4_runtime_selected_name_sha256": (
                mapping["runtime_selected_name_sha256"]
            ),
            "_v4_dense_reward_config": dense_reward_config,
            "_v4_dense_reward_sha256": dense_reward_sha256,
            "_v4_full_parameter_manifest_sha256": full_manifest_sha,
            "_v4_selected_parameter_manifest_sha256": selected_manifest_sha,
            "_v4_selected_names": selected_names,
            "_v4_selected_name_set": frozenset(selected_names),
            "_v4_selected_parameter_count": len(selected_parameters),
            "_v4_selected_element_count": selected_elements,
            "_v4_selected_byte_count": selected_bytes,
        }
        for name, value in installed_attributes.items():
            setattr(self, name, value)
        try:
            initial = self._partitioned_weight_state_v4(
                chunk_bytes=chunk_bytes, require_unselected_origin=False,
            )
            self._v4_unselected_origin_identity = dict(initial["unselected"])
            self._v4_installed_identity = dict(initial)
            self._record_full_audit_v4(initial, "layer_plan_install")
        except Exception:
            for name in (
                *installed_attributes,
                "_v4_unselected_origin_identity",
                "_v4_installed_identity",
                "_v4_last_full_audit_identity",
                "_v4_last_full_audit_phase",
            ):
                if hasattr(self, name):
                    delattr(self, name)
            raise
        return {
            "schema": "eggroll-es-layer-plan-installed-v4",
            "installed": True,
            "idempotent": False,
            "installation_identity": installation_identity,
            "reference_present_before_install": False,
            "reference_generation_before_install": 0,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "plan": plan["plan"],
            "full_parameter_manifest_sha256": full_manifest_sha,
            "selected_byte_count": selected_bytes,
            "initial_identity": initial,
            **self._binding_fields_v4(),
        }

    def save_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        """Capture selected tensors only; never recapture unselected origin."""
        self._require_layer_plan_v4()
        self._require_no_pending_update_v3()
        first_capture = int(getattr(self, "_v3_reference_generation", 0)) == 0
        if first_capture:
            installed = self._require_full_audit_v4("layer_plan_install")
            if installed != self._v4_installed_identity:
                raise RuntimeError("v4 installed identity changed before capture")
            unselected_current = dict(self._v4_unselected_origin_identity)
        else:
            unselected_current = self._verify_unselected_origin_v4(
                "before exact reference capture", chunk_bytes=chunk_bytes,
            )
        _, selected = self._validated_parameters_v4()
        references = {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in selected
        }
        selected_reference = self._partition_identity_v4(
            "selected", list(references.items()), chunk_bytes,
        )
        if (
            first_capture
            and selected_reference != self._v4_installed_identity["selected"]
        ):
            raise RuntimeError(
                "selected weights changed between install and first capture"
            )
        identity = self._combined_identity_v4(
            selected_reference, unselected_current,
        )
        self.exact_reference_weights = references
        self.exact_reference_identity = dict(identity)
        self._v4_selected_reference_identity = dict(selected_reference)
        generation = int(getattr(self, "_v3_reference_generation", 0)) + 1
        self._v3_reference_generation = generation
        self._v3_reference_fresh = True
        self._v3_reference_identity = dict(identity)
        self._v3_current_identity = dict(identity)
        self._v3_update_session = None
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_pending_update = None
        # The first capture reuses the immediately preceding full install
        # audit after independently re-hashing all selected CPU references.
        # Later captures perform a fresh complement audit above.
        self._v4_last_full_audit_identity = dict(identity)
        self._v4_last_full_audit_phase = "exact_reference"
        communicator = self._communicator_state_v3(REQUIRED_ENGINE_COUNT)
        return {
            "schema": "eggroll-es-selected-exact-reference-state-v4",
            "reference_generation": generation,
            "fresh_for_population": True,
            "identity": dict(identity),
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            **self._binding_fields_v4(),
        }

    def _restore_selected_reference_v4(
        self, *, verify_partitions, chunk_bytes=64 * 1024 * 1024,
    ):
        self._require_layer_plan_v4()
        references = getattr(self, "exact_reference_weights", None)
        if not isinstance(references, dict) or not references:
            raise RuntimeError("selected exact reference has not been captured")
        if set(references) != self._v4_selected_name_set:
            raise RuntimeError("selected exact reference names changed")
        _, selected = self._validated_parameters_v4()
        with torch.no_grad():
            for name, parameter in selected:
                reference = references[name]
                if (
                    reference.dtype != parameter.dtype
                    or tuple(reference.shape) != tuple(parameter.shape)
                ):
                    raise RuntimeError(
                        f"selected exact reference metadata changed for {name!r}"
                    )
                parameter.data.copy_(reference, non_blocking=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verify_partitions:
            current = self._partitioned_weight_state_v4(
                chunk_bytes=chunk_bytes, require_unselected_origin=True,
            )
            if current != self._v3_reference_identity:
                raise RuntimeError(
                    "selected rollback did not reproduce partitioned reference"
                )
            self._v3_current_identity = dict(current)
            return current
        self._v3_current_identity = dict(self._v3_reference_identity)
        return None

    def restore_self_weights_exact(self):
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        self._restore_selected_reference_v4(verify_partitions=False)
        self._v4_last_full_audit_identity = None
        self._v4_last_full_audit_phase = "population_restore_pending_audit"
        return True

    def verify_self_exact_reference(self, chunk_bytes=64 * 1024 * 1024):
        """Verify selected restore cheaply; the population boundary audits all."""
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        _, selected = self._validated_parameters_v4()
        current = self._partition_identity_v4(
            "selected", selected, chunk_bytes,
        )
        if current != self._v4_selected_reference_identity:
            raise RuntimeError("selected weights differ from exact reference")
        return {
            "schema": "eggroll-es-selected-exact-reference-check-v4",
            "passed": True,
            "reference_generation": int(self._v3_reference_generation),
            "reference": dict(self._v4_selected_reference_identity),
            "current": current,
            "unselected_audit": "deferred_to_population_completion_v4",
            **self._binding_fields_v4(),
        }

    def weight_state_sha256(self, chunk_bytes=64 * 1024 * 1024):
        current = self._partitioned_weight_state_v4(
            chunk_bytes=chunk_bytes, require_unselected_origin=True,
        )
        return self._record_full_audit_v4(current, "explicit_weight_state")

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        """Apply v3-compatible seeded noise to selected parameters only."""
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        scale = float(noise_scale)
        if not math.isfinite(scale):
            raise ValueError("perturbation scale must be finite")
        sign = -1.0 if negate else 1.0
        self._set_seed(int(seed))
        self._v4_last_full_audit_identity = None
        self._v4_last_full_audit_phase = "population_perturbed"
        _, selected = self._validated_parameters_v4()
        with torch.no_grad():
            for _, parameter in selected:
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(int(seed))
                noise = torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                parameter.data.add_(sign * scale * noise)
                del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _allocation_readiness_preflight_v4(self):
        """Touch the actual selected-only accumulation peak before NCCL."""
        _, selected = self._validated_parameters_v4()
        largest = max(
            (
                int(parameter.numel())
                * (2 * torch.empty((), dtype=torch.float32).element_size()
                   + int(parameter.element_size())),
                name,
                parameter,
            )
            for name, parameter in selected
        )
        footprint, name, parameter = largest
        accumulator = native_noise = converted_noise = None
        try:
            accumulator = torch.empty_like(
                parameter.data, dtype=torch.float32,
            )
            native_noise = torch.empty_like(parameter.data)
            converted_noise = torch.empty_like(
                parameter.data, dtype=torch.float32,
            )
            accumulator.zero_()
            native_noise.zero_()
            converted_noise.zero_()
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
        finally:
            del accumulator, native_noise, converted_noise
        return {
            "schema": "eggroll-es-selected-allocation-preflight-v4",
            "passed": True,
            "parameter_count": self._v4_selected_parameter_count,
            "element_count": self._v4_selected_element_count,
            "largest_parameter_name": name,
            "largest_parameter_shape": list(parameter.shape),
            "parameter_dtype": str(parameter.dtype),
            "accumulator_dtype": "torch.float32",
            "simulated_peak_temporary_bytes": footprint,
            "scratch_freed_before_collectives": True,
            "collectives_created": False,
            "rng_consumed": False,
            "weights_changed": False,
            **self._binding_fields_v4(),
        }

    def inspect_distributed_update_state_v4(self, expected_world_size):
        """Re-hash both partitions before reporting controller-visible state."""
        communicator = self._communicator_state_v3(expected_world_size)
        if not isinstance(getattr(self, "_v3_reference_identity", None), dict):
            raise RuntimeError("v4 exact reference has not been captured")
        current = self._partitioned_weight_state_v4(
            require_unselected_origin=True,
        )
        if current != self._v3_current_identity:
            raise RuntimeError("v4 current selected identity changed unexpectedly")
        self._record_full_audit_v4(current, "explicit_state_inspection")
        return self._worker_state_report_v4(communicator, current)

    def _worker_state_report_v4(self, communicator, current):
        return {
            "schema": "eggroll-es-layer-restricted-worker-state-v4",
            "communicator": communicator,
            "reference_generation": int(self._v3_reference_generation),
            "reference_fresh_for_population": bool(self._v3_reference_fresh),
            "reference_identity": dict(self._v3_reference_identity),
            "current_identity": current,
            "update_session": self._v3_update_session,
            "update_sequence": int(self._v3_update_sequence),
            "accepted_alpha": float(self._v3_accepted_alpha),
            "pending": self._v3_pending_update is not None,
            **self._binding_fields_v4(),
        }

    def inspect_cached_distributed_update_state_v4(
        self, expected_world_size, required_full_audit_phase,
    ):
        """Report a just-audited state without immediately hashing 68GB again."""
        self._require_no_pending_update_v3()
        communicator = self._communicator_state_v3(expected_world_size)
        current = self._require_full_audit_v4(required_full_audit_phase)
        return self._worker_state_report_v4(communicator, current)

    def audit_population_completion_v4(
        self,
        expected_world_size,
        reference_generation,
        expected_reference_sha256,
        chunk_bytes=64 * 1024 * 1024,
    ):
        """One full partition audit after all selected-only rollouts finish."""
        self._require_no_pending_update_v3()
        self._require_fresh_population_reference_v3()
        communicator = self._communicator_state_v3(expected_world_size)
        if int(reference_generation) != int(self._v3_reference_generation):
            raise RuntimeError("post-population audit used a stale reference")
        expected_reference_sha256 = str(expected_reference_sha256)
        if self._v3_reference_identity.get("sha256") != expected_reference_sha256:
            raise RuntimeError("post-population expected reference changed")
        current = self._partitioned_weight_state_v4(
            chunk_bytes=chunk_bytes, require_unselected_origin=True,
        )
        if current["selected"] != self._v4_selected_reference_identity:
            raise RuntimeError(
                "selected weights were not exactly restored after population"
            )
        if current != self._v3_reference_identity:
            raise RuntimeError(
                "partitioned weights differ from post-population reference"
            )
        self._v3_current_identity = dict(current)
        self._record_full_audit_v4(current, "population_completion")
        return {
            "schema": "eggroll-es-post-population-audit-v4",
            "passed": True,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "reference_generation": int(self._v3_reference_generation),
            "reference_sha256": expected_reference_sha256,
            "current_identity": current,
            **self._binding_fields_v4(),
        }

    def _validate_binding_arguments_v4(
        self,
        layer_plan_file_sha256,
        layer_plan_sha256,
        checkpoint_to_runtime_mapping_sha256,
        source_unit_count,
        runtime_selected_name_sha256,
        selected_parameter_manifest_sha256,
        runtime_selected_parameter_count,
        selected_element_count,
        unselected_origin_sha256,
        dense_reward_sha256,
    ):
        expected = self._binding_fields_v4()
        supplied = {
            "layer_plan_file_sha256": layer_plan_file_sha256,
            "layer_plan_sha256": layer_plan_sha256,
            "checkpoint_to_runtime_mapping_sha256": (
                checkpoint_to_runtime_mapping_sha256
            ),
            "source_unit_count": source_unit_count,
            "runtime_selected_name_sha256": runtime_selected_name_sha256,
            "selected_parameter_manifest_sha256": (
                selected_parameter_manifest_sha256
            ),
            "runtime_selected_parameter_count": (
                runtime_selected_parameter_count
            ),
            "selected_element_count": selected_element_count,
            "unselected_origin_sha256": unselected_origin_sha256,
            "dense_reward_sha256": dense_reward_sha256,
        }
        if supplied != expected:
            raise RuntimeError("controller v4 plan/reward bindings changed")
        return expected

    def prepare_sharded_seed_update_v4(
        self,
        seeds,
        coefficients,
        coefficient_sha256,
        population_size,
        expected_world_size,
        reference_generation,
        plan_id,
        update_sequence,
        previous_alpha,
        target_alpha,
        expected_base_sha256,
        layer_plan_file_sha256,
        layer_plan_sha256,
        checkpoint_to_runtime_mapping_sha256,
        source_unit_count,
        runtime_selected_name_sha256,
        selected_parameter_manifest_sha256,
        runtime_selected_parameter_count,
        selected_element_count,
        unselected_origin_sha256,
        dense_reward_sha256,
    ):
        """Prepare a fully plan-bound update without entering a collective."""
        self._require_no_pending_update_v3()
        bindings = self._validate_binding_arguments_v4(
            layer_plan_file_sha256,
            layer_plan_sha256,
            checkpoint_to_runtime_mapping_sha256,
            source_unit_count,
            runtime_selected_name_sha256,
            selected_parameter_manifest_sha256,
            runtime_selected_parameter_count,
            selected_element_count,
            unselected_origin_sha256,
            dense_reward_sha256,
        )
        communicator = self._communicator_state_v3(expected_world_size)
        seeds, coefficients = validate_seed_coefficients_v3(
            seeds, coefficients, population_size, expected_world_size,
        )
        actual_coefficient_sha = coefficient_sha256_v3(seeds, coefficients)
        if actual_coefficient_sha != coefficient_sha256:
            raise ValueError("coefficient identity changed before v4 update")
        reference_generation = int(reference_generation)
        if reference_generation != int(self._v3_reference_generation):
            raise RuntimeError("v4 update used a stale reference generation")
        current = self._partitioned_weight_state_v4(
            require_unselected_origin=True,
        )
        if current != self._v3_current_identity:
            raise RuntimeError("v4 update current identity changed before prepare")
        expected_base_sha256 = str(expected_base_sha256)
        if current["sha256"] != expected_base_sha256:
            raise RuntimeError("v4 distributed update base hash changed")
        self._record_full_audit_v4(current, "update_prepare")
        update_sequence = int(update_sequence)
        if update_sequence != int(self._v3_update_sequence) + 1:
            raise RuntimeError("v4 update sequence is stale or skipped")
        if self._v3_update_sequence == 0 and not self._v3_reference_fresh:
            raise RuntimeError("first v4 update has a stale population reference")
        if self._v3_update_sequence > 0 and self._v3_reference_fresh:
            raise RuntimeError("continued v4 update has a fresh reference")
        previous_alpha = float(previous_alpha)
        target_alpha = float(target_alpha)
        if (
            not math.isfinite(previous_alpha)
            or not math.isfinite(target_alpha)
            or target_alpha <= previous_alpha
            or previous_alpha != float(self._v3_accepted_alpha)
        ):
            raise ValueError("v4 alpha transition is not monotonic")
        plan_id = str(plan_id)
        if not plan_id:
            raise ValueError("v4 distributed update plan id is empty")
        if self._v3_update_session not in (None, plan_id):
            raise RuntimeError("a different v4 update plan is active")
        if self._v3_update_sequence > 0 and self._v3_update_session != plan_id:
            raise RuntimeError("v4 update session changed mid-search")
        manifest = update_manifest_v4(
            coefficient_sha256=coefficient_sha256,
            population_size=population_size,
            world_size=expected_world_size,
            reference_generation=reference_generation,
            plan_id=plan_id,
            update_sequence=update_sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha256,
            **bindings,
        )
        manifest_sha = canonical_sha256_v3(manifest)
        shard = seed_shard_v3(
            seeds, coefficients, communicator["rank"], communicator["world_size"],
        )
        expected_count = int(population_size) // int(expected_world_size)
        if len(shard["indices"]) != expected_count:
            raise RuntimeError("v4 distributed seed shard is not balanced")
        preflight = self._allocation_readiness_preflight_v4()
        self._v3_pending_update = {
            "schema": "eggroll-es-layer-restricted-pending-update-v4",
            "phase": "prepared",
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "shard": shard,
        }
        return {
            "schema": "eggroll-es-layer-restricted-update-prepared-v4",
            "prepared": True,
            "manifest_sha256": manifest_sha,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "shard_indices": list(shard["indices"]),
            "shard_seeds": list(shard["seeds"]),
            "shard_pair_sha256": canonical_sha256_v3({
                "seeds": shard["seeds"],
                "coefficients": shard["coefficients"],
            }),
            "base_sha256": current["sha256"],
            "reference_generation": reference_generation,
            "update_sequence": update_sequence,
            "allocation_preflight": preflight,
            **bindings,
        }

    def execute_prepared_seed_update_v4(self, manifest_sha256):
        """All-reduce and update exactly the frozen selected partition."""
        pending = getattr(self, "_v3_pending_update", None)
        if (
            not isinstance(pending, dict)
            or pending.get("schema")
            != "eggroll-es-layer-restricted-pending-update-v4"
            or pending.get("phase") != "prepared"
        ):
            raise RuntimeError("v4 distributed update was not prepared")
        if pending.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("prepared v4 manifest changed")
        manifest = pending["manifest"]
        if canonical_sha256_v3(manifest) != manifest_sha256:
            raise RuntimeError("prepared v4 manifest payload changed")
        communicator = self._communicator_state_v3(manifest["world_size"])
        self._validate_binding_arguments_v4(
            manifest["layer_plan_file_sha256"],
            manifest["layer_plan_sha256"],
            manifest["checkpoint_to_runtime_mapping_sha256"],
            manifest["source_unit_count"],
            manifest["runtime_selected_name_sha256"],
            manifest["selected_parameter_manifest_sha256"],
            manifest["runtime_selected_parameter_count"],
            manifest["selected_element_count"],
            manifest["unselected_origin_sha256"],
            manifest["dense_reward_sha256"],
        )
        shard = pending["shard"]
        parameter_count = 0
        reduced_element_count = 0
        scale = (
            float(manifest["target_alpha"])
            - float(manifest["previous_alpha"])
        ) / float(manifest["population_size"])
        try:
            base_identity = self._require_full_audit_v4("update_prepare")
            if (
                base_identity != self._v3_current_identity
                or base_identity["sha256"] != manifest["expected_base_sha256"]
            ):
                raise RuntimeError(
                    "v4 update base changed between prepare and execute"
                )
            _, selected = self._validated_parameters_v4()
            with torch.no_grad():
                for _, parameter in selected:
                    accumulator = accumulate_seed_terms_v3(
                        parameter, shard["seeds"], shard["coefficients"],
                    )
                    if accumulator.dtype != torch.float32:
                        raise RuntimeError("v4 update accumulator is not FP32")
                    stream = (
                        torch.cuda.current_stream()
                        if accumulator.is_cuda else None
                    )
                    reduced = self.inter_pg.all_reduce(
                        accumulator, out_tensor=accumulator, stream=stream,
                    )
                    if reduced is None:
                        raise RuntimeError("v4 PyNccl all-reduce returned no tensor")
                    if (
                        reduced.dtype != torch.float32
                        or reduced.device != parameter.device
                        or tuple(reduced.shape) != tuple(parameter.shape)
                    ):
                        raise RuntimeError(
                            "v4 PyNccl all-reduce returned an incompatible tensor"
                        )
                    reduced.mul_(scale)
                    parameter.data.add_(reduced.to(parameter.dtype))
                    parameter_count += 1
                    reduced_element_count += int(reduced.numel())
                    del accumulator, reduced
            if (
                parameter_count != self._v4_selected_parameter_count
                or reduced_element_count != self._v4_selected_element_count
            ):
                raise RuntimeError("v4 collective did not cover exact selection")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            final_identity = self._partitioned_weight_state_v4(
                require_unselected_origin=True,
            )
            self._record_full_audit_v4(
                final_identity, f"update_execute:{manifest_sha256}",
            )
        except Exception as error:
            self._v3_pending_update = None
            try:
                restored = self._restore_selected_reference_v4(
                    verify_partitions=True,
                )
                self._v3_current_identity = dict(restored)
                self._v3_reference_fresh = True
            except Exception as rollback_error:
                raise RuntimeError(
                    f"v4 update failed ({error}); exact selected rollback or "
                    f"unselected-origin audit also failed ({rollback_error})"
                ) from error
            raise
        pending["phase"] = "executed"
        pending["final_identity"] = dict(final_identity)
        return {
            "schema": "eggroll-es-layer-restricted-update-executed-v4",
            "executed": True,
            "manifest_sha256": manifest_sha256,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "parameter_count": parameter_count,
            "reduced_element_count": reduced_element_count,
            "collective_dtype": "torch.float32",
            "final_identity": dict(final_identity),
            **self._binding_fields_v4(),
        }

    def commit_prepared_seed_update_v4(
        self, manifest_sha256, expected_final_sha256,
    ):
        pending = getattr(self, "_v3_pending_update", None)
        if (
            not isinstance(pending, dict)
            or pending.get("schema")
            != "eggroll-es-layer-restricted-pending-update-v4"
            or pending.get("phase") != "executed"
        ):
            raise RuntimeError("v4 distributed update was not executed")
        if pending.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError("executed v4 manifest changed")
        if canonical_sha256_v3(pending.get("manifest")) != manifest_sha256:
            raise RuntimeError("executed v4 manifest payload changed")
        final_identity = pending.get("final_identity")
        if final_identity.get("sha256") != expected_final_sha256:
            raise RuntimeError("v4 final partitioned identity changed")
        if final_identity["unselected"] != self._v4_unselected_origin_identity:
            raise RuntimeError("v4 commit observed unselected parameter drift")
        current = self._partitioned_weight_state_v4(
            require_unselected_origin=True,
        )
        if current != final_identity:
            raise RuntimeError("v4 weights changed between execute and commit")
        self._record_full_audit_v4(
            current, f"update_commit:{manifest_sha256}",
        )
        manifest = pending["manifest"]
        self._v3_current_identity = dict(final_identity)
        self._v3_update_session = manifest["plan_id"]
        self._v3_update_sequence = int(manifest["update_sequence"])
        self._v3_accepted_alpha = float(manifest["target_alpha"])
        self._v3_reference_fresh = False
        self._v3_pending_update = None
        return {
            "schema": "eggroll-es-layer-restricted-update-committed-v4",
            "committed": True,
            "manifest_sha256": manifest_sha256,
            "rank": int(self.inter_pg.rank),
            "final_sha256": expected_final_sha256,
            "reference_generation": int(self._v3_reference_generation),
            "reference_fresh_for_population": False,
            "update_sequence": int(self._v3_update_sequence),
            "accepted_alpha": float(self._v3_accepted_alpha),
            **self._binding_fields_v4(),
        }

    def abort_distributed_update_v4(self, plan_id, reference_generation):
        """Restore selected base and prove the complement never changed."""
        if int(reference_generation) != int(self._v3_reference_generation):
            raise RuntimeError("cannot abort v4 through a stale reference")
        active_plan = self._v3_update_session
        pending = getattr(self, "_v3_pending_update", None)
        pending_plan = (
            pending.get("manifest", {}).get("plan_id")
            if isinstance(pending, dict) else None
        )
        if (
            active_plan is not None or pending_plan is not None
        ) and str(plan_id) not in (active_plan, pending_plan):
            raise RuntimeError("abort plan does not match v4 update session")
        self._verify_unselected_origin_v4(
            "before selected distributed abort"
        )
        self._v3_pending_update = None
        restored = self._restore_selected_reference_v4(
            verify_partitions=True,
        )
        self._record_full_audit_v4(restored, "distributed_abort")
        self._v3_current_identity = dict(restored)
        self._v3_reference_fresh = True
        self._v3_update_session = None
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        return {
            "schema": "eggroll-es-layer-restricted-update-abort-v4",
            "aborted": True,
            "rank": int(self.inter_pg.rank),
            "reference_generation": int(self._v3_reference_generation),
            "restored_identity": restored,
            **self._binding_fields_v4(),
        }

    @staticmethod
    def _forbid_inherited_full_model_path_v4(path):
        raise RuntimeError(
            f"inherited full-model path {path} is forbidden in v4"
        )

    def save_self_initial_weights(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "save_self_initial_weights"
        )

    def update_weights_from_seeds(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "update_weights_from_seeds"
        )

    def save_self_weights_to_disk(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "save_self_weights_to_disk"
        )

    def load_weights_from_disk(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "load_weights_from_disk"
        )

    def broadcast_all_weights(self, src_rank):
        if getattr(self, "_v4_layer_plan_installed", False):
            return self._forbid_inherited_full_model_path_v4(
                "broadcast_all_weights after v4 plan installation"
            )
        return super().broadcast_all_weights(src_rank)

    def inspect_distributed_update_state_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "inspect_distributed_update_state_v3"
        )

    def _allocation_readiness_preflight_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "_allocation_readiness_preflight_v3"
        )

    def prepare_sharded_seed_update_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "prepare_sharded_seed_update_v3"
        )

    def execute_prepared_seed_update_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "execute_prepared_seed_update_v3"
        )

    def commit_prepared_seed_update_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "commit_prepared_seed_update_v3"
        )

    def abort_distributed_update_v3(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_inherited_full_model_path_v4(
            "abort_distributed_update_v3"
        )


FrozenLayerPlanAuditWorkerExtensionV4 = (
    LayerRestrictedExactAuditWorkerExtensionV4
)
