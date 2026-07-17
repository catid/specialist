from __future__ import annotations

import json
import os
import subprocess

import qwen36_v73g_exact_phase_profiler_contract as contract


def _subprocess(code: str, updates: dict[str, str]) -> dict:
    environment = dict(os.environ)
    for name in (
        "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD",
        "SPECIALIST_V73E_CONTROLLER_GUARD_PID",
        contract.ACTOR_BOOTSTRAP_ENV,
        contract.ACTOR_GUARD_SHA_ENV,
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(contract.ROOT)
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


def test_controller_validation_accepts_parents_imported_after_sitecustomize():
    value = _subprocess(
        """
import json
import eggroll_es_worker_lora_v72
import eggroll_es_worker_lora_v73g as worker
r = worker.LoRAAdapterStateWorkerExtensionV73E.systems_only_path_guard_receipt_v73e(None)
print(json.dumps(r, sort_keys=True))
""",
        {
            "PYTHONPATH": f"{contract.GUARD_DIRECTORY}:{contract.ROOT}",
            "SPECIALIST_V73E_SYSTEMS_ONLY_GUARD": "1",
        },
    )

    assert value["process_role"] == "controller_worker_contract_validation"
    assert value["bootstrap_mechanism"] == "controller_sitecustomize"
    assert value["guard_was_preinstalled"] is True
    assert value["parent_modules_absent_before_guard_install"] is False
    assert value["guard_process_receipt"]["successful_protected_opens"] == 0
    assert value["guard_process_receipt"]["successful_protected_resolves"] == 0
    assert value["guard_process_receipt"]["successful_protected_metadata"] == 0
    assert value["guard_process_receipt"]["successful_protected_enumerations"] == 0


def test_actor_still_rejects_preimported_parent():
    value = _subprocess(
        """
import json
import eggroll_es_worker_lora_v72
failed = False
message = ''
try:
    import eggroll_es_worker_lora_v73g
except RuntimeError as error:
    failed = True
    message = str(error)
print(json.dumps({'failed': failed, 'message': message}, sort_keys=True))
""",
        {
            contract.ACTOR_BOOTSTRAP_ENV: "1",
            contract.ACTOR_GUARD_SHA_ENV: contract.file_sha256(contract.GUARD),
        },
    )

    assert value["failed"] is True
    assert "actor bootstrap environment or import order changed" in value["message"]
