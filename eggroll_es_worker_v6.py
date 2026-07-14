"""Four-arm edge-split worker allowlist for EGGROLL-ES v6.

V6 deliberately reuses the audited v4 worker bytecode.  Only the immutable
layer-plan validator's allowlist is replaced, using cloned functions with
private globals rather than mutating the v4 module.  Every partition hash,
exact restore, two-phase update, four-rank audit, and forbidden full-model
path therefore remains the v4 implementation.
"""

from __future__ import annotations

from types import FunctionType

import eggroll_es_worker_v4 as worker_v4


SELECTED_ELEMENT_COUNT_V6 = 142_999_552
SELECTED_BYTE_COUNT_V6 = 285_999_104
FROZEN_LAYER_PLANS_V6 = {
    "af9dcf4e5c932aeb192ee0d195e9c4fee9d4a510467d850d7cf26f6db4c2d823": {
        "plan": "front",
        "file_sha256": (
            "02e5ce4cc2e20cf6b0910578a3e7982569d323b3458593000f77c624a8db62bf"
        ),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": SELECTED_ELEMENT_COUNT_V6,
        "selected_byte_count": SELECTED_BYTE_COUNT_V6,
        "checkpoint_to_runtime_mapping_sha256": (
            "d7d86497a7a0e445f19806990e761f0956c52e6fe1a72e242819bd7943fe2336"
        ),
        "runtime_selected_name_sha256": (
            "c385b0cb62f0bad29907c9efc69d30a1bf1c6f82bb1f30c0b00c966bc32fcfa0"
        ),
    },
    "d72624f2ef55b49b40aa8e52910394f079827a2d848bacc1ee42abb82c47846d": {
        "plan": "middle_early",
        "file_sha256": (
            "1496184e483071537cd95e10fd8cd051d7bd18c947df1b1e76d72f7d47bafab1"
        ),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": SELECTED_ELEMENT_COUNT_V6,
        "selected_byte_count": SELECTED_BYTE_COUNT_V6,
        "checkpoint_to_runtime_mapping_sha256": (
            "07e065a5304dcc1d9ecbcc7f2ad26cfa0613e1ac980535cfb2b0f1eabd120ecf"
        ),
        "runtime_selected_name_sha256": (
            "d7549779b080bd7c4d77c331d103ea8afb3f223285f2e126eed784b85c698cdc"
        ),
    },
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9": {
        "plan": "middle_late",
        "file_sha256": (
            "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
        ),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": SELECTED_ELEMENT_COUNT_V6,
        "selected_byte_count": SELECTED_BYTE_COUNT_V6,
        "checkpoint_to_runtime_mapping_sha256": (
            "7ebd15cb9c04cfe8dab67009dc2af5c0054131a11c15a6c8e83f277ffef4585c"
        ),
        "runtime_selected_name_sha256": (
            "5e5996a2865f55dad42d876fdcc8f22efbb3584901ef6e963f390f5dc839aa68"
        ),
    },
    "6da92a4db760676acda1bcbcaec4a925a6dd7b641c250a58a3fe4837d97ac93a": {
        "plan": "back",
        "file_sha256": (
            "73bfc82ba057908c0071d3c5e190581fecf6147cc398f06a994231f31908187e"
        ),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": SELECTED_ELEMENT_COUNT_V6,
        "selected_byte_count": SELECTED_BYTE_COUNT_V6,
        "checkpoint_to_runtime_mapping_sha256": (
            "690c604b60ac5802bb62a26e64dfeaf0ede1fd1e925f870329f435de9dba22cb"
        ),
        "runtime_selected_name_sha256": (
            "7cfcde51588c250e44607691e54cf2cb920c285e9096bcfc8047935798c8b790"
        ),
    },
}


def _clone_with_globals(function, replacements, name):
    globals_v6 = dict(function.__globals__)
    globals_v6.update(replacements)
    clone = FunctionType(
        function.__code__, globals_v6, name, function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    clone.__module__ = __name__
    clone.__qualname__ = name
    return clone


validate_frozen_layer_plan_v6 = _clone_with_globals(
    worker_v4.validate_frozen_layer_plan_v4,
    {"FROZEN_LAYER_PLANS_V4": FROZEN_LAYER_PLANS_V6},
    "validate_frozen_layer_plan_v6",
)

_install_layer_plan_v6 = _clone_with_globals(
    worker_v4.LayerRestrictedExactAuditWorkerExtensionV4.install_layer_plan_v4,
    {"validate_frozen_layer_plan_v4": validate_frozen_layer_plan_v6},
    "install_layer_plan_v4",
)


class FrozenEdgeSplitAuditWorkerExtensionV6(
    worker_v4.LayerRestrictedExactAuditWorkerExtensionV4,
):
    """The exact v4 worker with only the four v6 plan identities admitted."""

    install_layer_plan_v4 = _install_layer_plan_v6

