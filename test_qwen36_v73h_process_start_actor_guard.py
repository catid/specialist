from __future__ import annotations

import json
import os
import subprocess

import qwen36_v73h_exact_phase_profiler_contract as contract


def _subprocess(code: str, updates: dict[str, str]) -> dict:
    environment = dict(os.environ)
    for name in (
        "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD",
        "SPECIALIST_V73E_CONTROLLER_GUARD_PID",
        contract.ACTOR_BOOTSTRAP_ENV,
        contract.ACTOR_GUARD_SHA_ENV,
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = f"{contract.GUARD_DIRECTORY}:{contract.ROOT}"
    environment.update(updates)
    completed = subprocess.run(
        [str(contract.REQUIRED_PYTHON), "-c", code],
        cwd=contract.ROOT,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    return json.loads(lines[-1])


def test_controller_sitecustomize_covers_preloaded_parent():
    value = _subprocess(
        """
import json
import eggroll_es_worker_lora_v72
import eggroll_es_worker_lora_v73h as worker
r = worker.LoRAAdapterStateWorkerExtensionV73E.systems_only_path_guard_receipt_v73e(None)
print(json.dumps(r, sort_keys=True))
""",
        {"SPECIALIST_V73E_SYSTEMS_ONLY_GUARD": "1"},
    )
    assert value["process_role"] == "controller_worker_contract_validation"
    assert value["bootstrap_mechanism"] == "controller_sitecustomize"
    assert value["guard_was_preinstalled"] is True
    assert value["parent_modules_absent_before_guard_install"] is False


def test_actor_sitecustomize_covers_realistic_preloaded_parent():
    value = _subprocess(
        """
import json
import eggroll_es_worker_lora_v72
import eggroll_es_worker_lora_v73h as worker
r = worker.LoRAAdapterStateWorkerExtensionV73E.systems_only_path_guard_receipt_v73e(None)
print(json.dumps(r, sort_keys=True))
""",
        {
            "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD": "1",
            "SPECIALIST_V73E_CONTROLLER_GUARD_PID": "1",
            contract.ACTOR_BOOTSTRAP_ENV: "1",
            contract.ACTOR_GUARD_SHA_ENV: contract.file_sha256(contract.GUARD),
        },
    )
    assert value["process_role"] == "ray_actor_worker_extension"
    assert value["bootstrap_mechanism"] == contract.ACTOR_GUARD_MECHANISM
    assert value["guard_was_preinstalled"] is True
    assert value["actor_sitecustomize_installed_before_runtime_imports"] is True
    assert value["parent_modules_covered_by_process_start_guard"] is True
    assert value[
        "historical_reference_modules_covered_by_process_start_guard"
    ] is True
    assert value["parent_modules_absent_before_guard_install"] is False
    guard = value["guard_process_receipt"]
    assert guard["installation_mechanism"] == contract.ACTOR_GUARD_MECHANISM
    assert guard["successful_protected_opens"] == 0
    assert guard["successful_protected_resolves"] == 0
    assert guard["successful_protected_metadata"] == 0
    assert guard["successful_protected_enumerations"] == 0


def test_actor_sitecustomize_wrong_hash_fails_before_worker_parent_import():
    value = _subprocess(
        """
import json
import sys
failed = False
message = ''
try:
    import eggroll_es_worker_lora_v73h
except RuntimeError as error:
    failed = True
    message = str(error)
print(json.dumps({
    'failed': failed,
    'message': message,
    'v72_loaded': 'eggroll_es_worker_lora_v72' in sys.modules,
}, sort_keys=True))
""",
        {
            "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD": "1",
            "SPECIALIST_V73E_CONTROLLER_GUARD_PID": "1",
            contract.ACTOR_BOOTSTRAP_ENV: "1",
            contract.ACTOR_GUARD_SHA_ENV: "0" * 64,
        },
    )
    assert value["failed"] is True
    assert value["v72_loaded"] is False


def test_actor_flag_without_inherited_controller_pid_fails_closed():
    value = _subprocess(
        """
import json
failed = False
try:
    import eggroll_es_worker_lora_v73h
except RuntimeError:
    failed = True
print(json.dumps({'failed': failed}, sort_keys=True))
""",
        {
            "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD": "1",
            contract.ACTOR_BOOTSTRAP_ENV: "1",
            contract.ACTOR_GUARD_SHA_ENV: contract.file_sha256(contract.GUARD),
        },
    )
    assert value["failed"] is True
