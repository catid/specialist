#!/usr/bin/env python3
"""Heterogeneous four-arm, update-closed worker extension for V23A."""

from __future__ import annotations

import os
from types import FunctionType

import torch

import eggroll_es_worker_v4 as worker_v4
import eggroll_es_worker_v13 as worker_v13


FROZEN_LAYER_PLANS_V23A = {
    "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f": {
        "plan": "v23a_base_middle_late",
        "file_sha256": "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747",
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552, "selected_byte_count": 285_999_104,
        "checkpoint_to_runtime_mapping_sha256": (
            "7ebd15cb9c04cfe8dab67009dc2af5c0054131a11c15a6c8e83f277ffef4585c"
        ),
        "runtime_selected_name_sha256": (
            "5e5996a2865f55dad42d876fdcc8f22efbb3584901ef6e963f390f5dc839aa68"
        ),
    },
    "3197998ea32ed4c36be64353e699d7b18c9c08c7211f544461d65c3eb847e354": {
        "plan": "v23a_insert_front_e005",
        "file_sha256": "4c0f451545f01ec53cc17800e24f331d1038bbc3ce96fe352a9cc2b96c822e29",
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552, "selected_byte_count": 285_999_104,
        "checkpoint_to_runtime_mapping_sha256": (
            "9c51558f4e134249997aacff139b035860ccf888d6458fc0ab26edaa6c79f80f"
        ),
        "runtime_selected_name_sha256": (
            "eedc8a0a630c31dea73a3e7dce53393acbc834a7c7dbc20ebf2be6211917515c"
        ),
    },
    "395796311319ed63d033131996e77dcd5faf636cdb940d49ea5fe463369edc8d": {
        "plan": "v23a_insert_middle_e005",
        "file_sha256": "9f343cb136a5d4883ae81878ecec005e028b7f5e492ca0cc64b1f9e1945c112a",
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552, "selected_byte_count": 285_999_104,
        "checkpoint_to_runtime_mapping_sha256": (
            "7ebd15cb9c04cfe8dab67009dc2af5c0054131a11c15a6c8e83f277ffef4585c"
        ),
        "runtime_selected_name_sha256": (
            "5e5996a2865f55dad42d876fdcc8f22efbb3584901ef6e963f390f5dc839aa68"
        ),
    },
    "b539e4e70710ade7ae5b1dbf21814c728514e5a1e3f0a3cf36945821bd06c77e": {
        "plan": "v23a_insert_back_e005",
        "file_sha256": "21a0100d2bf729ce5ce88ea83ea668086cfb512ff5684413050d03d796c7820e",
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552, "selected_byte_count": 285_999_104,
        "checkpoint_to_runtime_mapping_sha256": (
            "cdfeae8da8b703d48355dfc182188f850d3a1a9cda01e7989437c791682c3b5a"
        ),
        "runtime_selected_name_sha256": (
            "f0fe344311a820f9bee5a103dcad17e899ac79825a3dbe9f549b778769b63fd7"
        ),
    },
}


def validate_frozen_layer_plan_v23a(raw, expected_file, expected_plan):
    original = worker_v4.FROZEN_LAYER_PLANS_V4
    try:
        worker_v4.FROZEN_LAYER_PLANS_V4 = FROZEN_LAYER_PLANS_V23A
        return worker_v4.validate_frozen_layer_plan_v4(
            raw, expected_file, expected_plan
        )
    finally:
        worker_v4.FROZEN_LAYER_PLANS_V4 = original


def _clone_with_globals(function, replacements, name):
    cloned = FunctionType(
        function.__code__, {**function.__globals__, **replacements},
        name, function.__defaults__, function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    return cloned


class InsertionLocationAuditWorkerExtensionV23A(
    worker_v13.TrainPanelDiagnosticWorkerExtensionV13,
):
    """Keep exact perturb/restore but freeze one rank-specific motif plan."""

    install_layer_plan_v4 = _clone_with_globals(
        worker_v4.LayerRestrictedExactAuditWorkerExtensionV4.install_layer_plan_v4,
        {"validate_frozen_layer_plan_v4": validate_frozen_layer_plan_v23a},
        "install_layer_plan_v23a",
    )

    @staticmethod
    def _forbid_mutation_surface_v23a(*_args, **_kwargs):
        raise RuntimeError("v23a closes update checkpoint and legacy restore surfaces")

    broadcast_all_weights = _forbid_mutation_surface_v23a
    abort_distributed_update_v4 = _forbid_mutation_surface_v23a
    restore_self_weights = _forbid_mutation_surface_v23a

    def runtime_device_identity_v23a(self, expected_arm):
        communicator = self._communicator_state_v3(4)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        value = {
            "schema": "eggroll-es-runtime-device-identity-v23a",
            "rank": communicator["rank"], "world_size": communicator["world_size"],
            "arm": str(expected_arm), "cuda_visible_devices": visible,
            "runtime_cuda_device": int(torch.cuda.current_device()),
            "device_name": str(properties.name),
            "device_total_memory": int(properties.total_memory),
            "update_surfaces_closed": True,
        }
        return value
