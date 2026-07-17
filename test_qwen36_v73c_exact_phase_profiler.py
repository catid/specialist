from __future__ import annotations

import ast
import copy
import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

import build_qwen36_v73c_exact_phase_profiler_preregistration as prereg
import qwen36_v73c_exact_phase_profiler_contract as contract
import run_lora_es_v71_v72_profile_calibration_v73c as target
import run_qwen36_v73c_exact_phase_profiler as profiler


def _global_pid(pid: int) -> int:
    return pid << 24


def _global_tid(pid: int, tid: int | None = None) -> int:
    return (pid << 24) | (pid if tid is None else tid)


def _self_hash(value: dict) -> dict:
    result = copy.deepcopy(value)
    result["content_sha256_before_self_field"] = contract.canonical_sha256(result)
    return result


def _phase_receipt() -> dict:
    rows = []
    for index, phase in enumerate(contract.PHASES):
        start = index * 100
        rows.append({
            "phase": phase,
            "epoch": index,
            "controller_pid": 9000,
            "started_monotonic_ns": start,
            "ended_monotonic_ns": start + 50,
            "elapsed_ns": 50,
        })
    return _self_hash({
        "schema": "eggroll-es-exact-phase-range-receipt-v73c",
        "arm": "timeline",
        "complete": True,
        "nvtx_domain": contract.PHASE_DOMAIN,
        "expected_phase_order": list(contract.PHASES),
        "observed_phase_order": list(contract.PHASES),
        "phase_count": len(contract.PHASES),
        "one_controller_pid": True,
        "rows": rows,
        "contains_prompts_questions_answers_or_outputs": False,
    })


def _synthetic_sqlite(
    path: Path,
    *,
    crossing_kernel: bool = False,
    omit_kernel_gpu: int | None = None,
    nonempty_stream: bool = False,
) -> tuple[set[int], dict]:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS(
          start INTEGER, end INTEGER, eventType INTEGER, globalTid INTEGER,
          domainId INTEGER, textId INTEGER, text TEXT
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
          start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER,
          demangledName INTEGER, shortName INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
          start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER,
          bytes INTEGER, copyKind INTEGER
        );
        CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER, label TEXT);
        CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS(
          start INTEGER, globalPid INTEGER, deviceId INTEGER, bytes INTEGER,
          memoryOperationType INTEGER
        );
        CREATE TABLE ENUM_CUDA_DEV_MEM_EVENT_OPER(id INTEGER, label TEXT);
        CREATE TABLE ProcessStreams(
          globalPid INTEGER, filenameId INTEGER, contentId INTEGER
        );
        """
    )
    strings: list[tuple[int, str]] = []
    next_id = 1

    def string_id(value: str) -> int:
        nonlocal next_id
        found = next((key for key, item in strings if item == value), None)
        if found is not None:
            return found
        result = next_id
        next_id += 1
        strings.append((result, value))
        return result

    controller_pid = 9000
    actor_pids = {100, 101, 102, 103}
    phase_domain = 7
    nccl_domain = 8
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)",
        (1, None, profiler.NVTX_DOMAIN_CREATE, _global_tid(controller_pid),
         phase_domain, None, contract.PHASE_DOMAIN),
    )
    for pid in sorted(actor_pids):
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)",
            (2, None, profiler.NVTX_DOMAIN_CREATE, _global_tid(pid),
             nccl_domain, None, "NCCL"),
        )

    phase_windows = {}
    for index, phase in enumerate(contract.PHASES):
        start = 10_000 + index * 1_000_000
        end = start + 900_000
        phase_windows[phase] = (start, end)
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)",
            (start, end, 59, _global_tid(controller_pid), phase_domain,
             string_id(phase), None),
        )
        for gpu, pid in enumerate(sorted(actor_pids)):
            if gpu != omit_kernel_gpu:
                kernel_start = start + 100 + gpu * 10
                kernel_end = kernel_start + 5
                if crossing_kernel and index == 0 and gpu == 0:
                    kernel_end = end + 1
                if "materialize" in phase:
                    name = "materialize_pack_kernel"
                elif phase == "pair_difference_update_execute_all_actors":
                    name = "ncclDevKernel_AllReduce"
                else:
                    name = "elementwise_kernel"
                name_id = string_id(name)
                connection.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                    (kernel_start, kernel_end, gpu, _global_pid(pid),
                     name_id, name_id),
                )
            copy_start = start + 300 + gpu * 10
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?,?,?)",
                (copy_start, copy_start + 7, gpu, _global_pid(pid),
                 1024 + gpu, 1),
            )

    update_start, _ = phase_windows[
        "pair_difference_update_execute_all_actors"
    ]
    nccl_text = string_id("ncclAllReduce size=18112512")
    for pid in sorted(actor_pids):
        for index in range(70):
            start = update_start + 2_000 + index * 10
            connection.execute(
                "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?,?,?)",
                (start, start + 4, 59, _global_tid(pid), nccl_domain,
                 nccl_text, None),
            )

    setup_start, _ = phase_windows["setup"]
    cleanup_start, _ = phase_windows["cleanup_all_actors"]
    for gpu, pid in enumerate(sorted(actor_pids)):
        connection.execute(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES(?,?,?,?,?)",
            (setup_start + 500 + gpu, _global_pid(pid), gpu, 4096, 1),
        )
        connection.execute(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES(?,?,?,?,?)",
            (cleanup_start + 500 + gpu, _global_pid(pid), gpu, 4096, 2),
        )
        filename = string_id(f"/tmp/pid_{pid}_stdout.log")
        content = string_id("semantic output" if nonempty_stream and gpu == 0 else "")
        connection.execute(
            "INSERT INTO ProcessStreams VALUES(?,?,?)",
            (_global_pid(pid), filename, content),
        )
    connection.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'HtoD')")
    connection.execute(
        "INSERT INTO ENUM_CUDA_DEV_MEM_EVENT_OPER VALUES(1,'allocation')"
    )
    connection.execute(
        "INSERT INTO ENUM_CUDA_DEV_MEM_EVENT_OPER VALUES(2,'deallocation')"
    )
    connection.executemany("INSERT INTO StringIds VALUES(?,?)", strings)
    connection.commit()
    connection.close()
    return actor_pids, _phase_receipt()


def test_runtime_roots_do_not_import_preregistration_builder():
    for path in (contract.TARGET, contract.LAUNCHER, Path(contract.__file__)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "build_qwen36_v73c_exact_phase_profiler_preregistration" not in imports


def test_preregistration_separates_timeline_and_permission_blocked_hbm():
    value = prereg.build_preregistration_v73c()
    contract.validate_generated_preregistration_v73c(value)
    timeline = value["arms"]["timeline"]
    hbm = value["arms"]["hbm_metrics"]
    assert timeline["launch_authorized_by_this_file_after_identity_checks"] is True
    assert hbm["launch_authorized_by_this_file_after_identity_checks"] is False
    assert hbm["observed_driver_parameter"]["value"] == 1
    assert hbm["observed_tool_failure_class"] == "ERR_NVGPUCTRPERM"
    command = timeline["command_template"]
    assert command[:7] == [
        "/usr/bin/bash", "--noprofile", "--norc", "-i", "-c",
        contract.BASHRC_EXEC_SCRIPT, "v73c-nsys-launch",
    ]
    assert "--nccl-trace=none" in command
    assert "--nccl-trace=all" not in command
    assert "NCCL_DEBUG=INFO" in command
    assert "NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL" in command
    output_index = next(
        index for index, item in enumerate(command) if item.startswith("--output=")
    )
    env_index = command.index(str(contract.ENV))
    target_index = command.index(str(contract.REQUIRED_PYTHON))
    assert command.index(str(contract.NSYS)) < output_index < env_index < target_index
    assert command[env_index + 1].startswith("PYTHONPATH=")
    assert "--gpu-metrics-set=gb20x-top" not in command
    assert "--gpu-metrics-set=gb20x-top" in hbm[
        "command_template_after_future_reseal"
    ]


def test_exact_lora_geometry_and_topology_are_not_overinterpreted():
    value = prereg.build_preregistration_v73c()
    geometry = value["exact_lora_collective_geometry"]
    assert geometry == {
        "canonical_fp32_tensor_count": 70,
        "canonical_fp32_element_count": 4_528_128,
        "logical_payload_bytes_per_rank": 18_112_512,
        "logical_payload_bytes_all_four_ranks": 72_450_048,
        "nominal_ring_schedule_bytes_per_rank": 27_168_768,
        "nominal_ring_schedule_is_not_measured_physical_link_bytes": True,
        "legacy_full_weight_23_tensor_geometry_used": False,
        "v71_sampled_pcie_rx_attributed_to_this_collective": False,
    }
    topology = value["physical_gpu_topology"]
    assert topology["all_off_diagonal_gpu_pairs_report_node"] is True
    assert topology["all_cross_gpu_peer_read_status_ok"] is True
    assert topology["all_cross_gpu_peer_write_status_ok"] is True
    assert topology["all_cross_gpu_nvlink_status_not_supported"] is True
    assert topology["actual_nccl_transport_or_path_inferred_from_node"] is False


def test_phase_ledger_emits_exact_nonoverlapping_order():
    class Base:
        def __init__(self, initial="setup"):
            self._phase = initial
            self._epoch = 0

        def snapshot(self):
            return self._phase, self._epoch

        @property
        def value(self):
            return self._phase

        @value.setter
        def value(self, phase):
            self._epoch += 1
            self._phase = phase

    class Domain:
        def __init__(self):
            self.events = []

        def push_range(self, **kwargs):
            self.events.append(("push", kwargs["message"]))

        def pop_range(self):
            self.events.append(("pop", None))

    target._PHASE_INSTANCE = None
    domain = Domain()
    Phase = target.phase_class_v73c(Base, domain)
    instance = Phase()
    for phase in contract.PHASES[1:]:
        instance.value = phase
    instance.close_v73c_range()
    receipt = target.phase_receipt_v73c("timeline", complete=True)
    assert receipt["observed_phase_order"] == list(contract.PHASES)
    assert len(domain.events) == 2 * len(contract.PHASES)
    target._PHASE_INSTANCE = None


def test_hbm_execute_rejects_before_paths_or_subprocess(tmp_path, monkeypatch):
    value = prereg.build_preregistration_v73c()
    paths = value["arms"]["hbm_metrics"]["artifacts"]
    for key in paths:
        if key == "nccl_debug_pattern":
            paths[key] = str(tmp_path / "profile" / "nccl_debug.*.log")
        else:
            paths[key] = str(tmp_path / key)
    monkeypatch.setattr(
        profiler.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not be reached"),
    )
    with pytest.raises(RuntimeError, match="HBM metrics arm blocked"):
        profiler.execute_profile_v73c(value, "a" * 64, "hbm_metrics")
    assert list(tmp_path.iterdir()) == []


def test_changed_bashrc_identity_rejects_before_paths_or_subprocess(
    tmp_path, monkeypatch
):
    value = prereg.build_preregistration_v73c()
    paths = value["arms"]["timeline"]["artifacts"]
    for key in paths:
        if key == "nccl_debug_pattern":
            paths[key] = str(tmp_path / "profile" / "nccl_debug.*.log")
        else:
            paths[key] = str(tmp_path / key)
    value["toolchain"]["launch_shell"]["bashrc"]["file_sha256"] = "0" * 64
    monkeypatch.setattr(
        profiler.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not be reached"),
    )
    with pytest.raises(RuntimeError, match="launch bashrc"):
        profiler.execute_profile_v73c(value, "a" * 64, "timeline")
    assert list(tmp_path.iterdir()) == []


def test_prelaunch_toolchain_identity_binds_all_timeline_runtime_inputs():
    value = prereg.build_preregistration_v73c()
    receipt = profiler.validate_prelaunch_toolchain_v73c(value)
    assert receipt["passed"] is True
    assert receipt["checked_source_count"] == 12
    assert receipt["environment_values_persisted"] is False
    assert {row["label"] for row in receipt["checked_sources"]} == {
        "NCCL library",
        "NCCL package metadata",
        "env",
        "launch bash",
        "launch bashrc",
        "nsight-systems invocation",
        "nsight-systems native binary",
        "nvToolsExt library",
        "nvidia-smi",
        "python nvtx native extension",
        "python nvtx package",
        "target Python",
    }


def test_fresh_path_collision_fails_closed(tmp_path):
    paths = {
        "application_attempt": str(tmp_path / "attempt"),
        "run_directory": str(tmp_path / "run"),
        "profile_attempt": str(tmp_path / "profile-attempt"),
        "profile_directory": str(tmp_path / "profile"),
    }
    Path(paths["run_directory"]).mkdir()
    with pytest.raises(RuntimeError, match="fresh application/profile"):
        profiler._fresh_launch(paths)


def test_synthetic_timeline_exact_phase_parser(tmp_path):
    sqlite_path = tmp_path / "trace.sqlite"
    pids, receipt = _synthetic_sqlite(sqlite_path)
    result = profiler.analyze_sqlite_v73c(
        sqlite_path, receipt, {pid: pid - 100 for pid in pids}
    )
    assert result["cuda_kernel_event_count"] == 4 * len(contract.PHASES)
    assert result["cuda_memcpy_bytes"] > 0
    assert result["process_streams"]["all_stream_content_empty"] is True
    assert result["hbm_dram_metrics"]["collected"] is False
    nccl = result["nccl"]
    assert nccl["exact_logical_collective_payload_bytes_per_rank"] == 18_112_512
    assert nccl["nominal_ring_schedule_bytes_per_rank"] == 27_168_768
    assert nccl["observed_exactly_70_api_ranges_per_actor"] is True
    assert nccl["physical_link_bytes_inferred_from_algorithm"] is False
    named = result["explicit_named_kernel_evidence"]
    assert named["nccl_explicit_symbol"]["classification_available"] is True
    assert named[
        "pack_flatten_or_materialize_explicit_symbol"
    ]["classification_available"] is True
    assert named["actual_nccl_link_path_or_link_only_time"]["available"] is False


def test_actor_receipt_maps_process_local_cupti_device_zero_to_physical_gpus(
    tmp_path,
):
    sqlite_path = tmp_path / "trace.sqlite"
    pids, receipt = _synthetic_sqlite(sqlite_path)
    connection = sqlite3.connect(sqlite_path)
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUDA_GPU_MEMORY_USAGE_EVENTS",
    ):
        connection.execute(f"UPDATE {table} SET deviceId = 0")
    connection.commit()
    connection.close()
    result = profiler.analyze_sqlite_v73c(
        sqlite_path, receipt, {pid: pid - 100 for pid in pids}
    )
    assert set(result["cuda_by_phase_and_gpu"]["setup"]) == {"0", "1", "2", "3"}


def test_point_event_at_phase_transition_uses_half_open_next_phase():
    phases = [
        {"text": "left", "start_ns": 0, "end_ns": 10},
        {"text": "right", "start_ns": 10, "end_ns": 20},
    ]
    assert profiler._phase_for_event(10, 10, phases, "point") == "right"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"crossing_kernel": True}, "crossed or fell outside"),
        ({"omit_kernel_gpu": 3}, "omitted a GPU"),
        ({"nonempty_stream": True}, "process stream was nonempty"),
    ],
)
def test_synthetic_timeline_fail_closed(tmp_path, kwargs, message):
    sqlite_path = tmp_path / "trace.sqlite"
    pids, receipt = _synthetic_sqlite(sqlite_path, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        profiler.analyze_sqlite_v73c(
            sqlite_path, receipt, {pid: pid - 100 for pid in pids}
        )


def test_nccl_debug_transport_requires_explicit_content_free_lines(tmp_path):
    pids = {100, 101, 102, 103}
    for pid in pids:
        (tmp_path / f"nccl_debug.host.{pid}.log").write_text(
            f"host:{pid}:1 [{pid - 100}] NCCL INFO Channel 00 via P2P/IPC\n",
            encoding="utf-8",
        )
    result = profiler._analyze_nccl_debug_logs_v73c(
        str(tmp_path / "nccl_debug.*.log"), pids
    )
    assert result["actual_transport_evidence_available"] is True
    assert {row["transport_token"] for row in result["explicit_transport_evidence"]} == {
        "P2P/IPC"
    }
    assert result["transport_token_used_to_infer_link_only_time_or_bytes"] is False


def test_nccl_debug_rejects_semantic_markers(tmp_path):
    pids = {100, 101, 102, 103}
    for pid in pids:
        content = "host NCCL INFO Channel via P2P/IPC\n"
        if pid == 100:
            content += "<|im_start|>\n"
        (tmp_path / f"nccl_debug.host.{pid}.log").write_text(
            content, encoding="utf-8"
        )
    with pytest.raises(RuntimeError, match="semantic payload marker"):
        profiler._analyze_nccl_debug_logs_v73c(
            str(tmp_path / "nccl_debug.*.log"), pids
        )


def test_prelaunch_four_gpu_gate_records_identity_and_no_path_inference(monkeypatch):
    value = prereg.build_preregistration_v73c()
    topology = value["physical_gpu_topology"]
    monkeypatch.setattr(profiler.builder, "gpu_topology_v73c", lambda: topology)
    gpu_rows = "\n".join(
        f"{row['physical_gpu_id']}, {row['uuid']}, {row['pci_bus_id']}, 4, 0"
        for row in value["physical_gpu_identity"]
    )
    monkeypatch.setattr(
        profiler,
        "_run_text",
        lambda command: (
            ""
            if any(item.startswith("--query-compute-apps=") for item in command)
            else gpu_rows
        ),
    )
    result = profiler.four_gpu_idle_preflight_v73c(value)
    assert result["passed"] is True
    assert result["no_compute_processes"] is True
    assert result["all_cross_gpu_peer_read_write_ok_nvlink_not_supported"] is True
    assert result["actual_nccl_transport_or_path_inferred_from_node"] is False


def test_generated_contract_tamper_fails():
    value = prereg.build_preregistration_v73c()
    value["arms"]["timeline"]["artifacts"]["run_directory"] += "-tampered"
    body = dict(value)
    body.pop("content_sha256_before_self_field")
    value["content_sha256_before_self_field"] = contract.canonical_sha256(body)
    with pytest.raises(RuntimeError, match="generated runtime contract changed"):
        contract.validate_generated_preregistration_v73c(value)


def _guard_subprocess(code: str) -> dict:
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": f"{contract.GUARD_DIRECTORY}:{contract.ROOT}",
        "SPECIALIST_V73C_SYSTEMS_ONLY_GUARD": "1",
    })
    completed = subprocess.run(
        [str(contract.REQUIRED_PYTHON), "-c", code],
        cwd=contract.ROOT,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return json.loads(completed.stdout)


def test_sitecustomize_guard_denies_every_claimed_open_and_resolution_api():
    value = _guard_subprocess(
        """
import builtins
import io
import json
import os
from pathlib import Path
import v73c_path_open_guard as guard
p = Path('/home/catid/specialist/data/eval_qa_v3.jsonl')
open_denials = {}
for name, operation in (
    ('builtins.open', lambda: builtins.open(p, 'rb')),
    ('io.open', lambda: io.open(p, 'rb')),
    ('os.open', lambda: os.open(p, os.O_RDONLY)),
    ('Path.open', lambda: p.open('rb')),
):
    try:
        operation()
    except PermissionError:
        open_denials[name] = True
    else:
        open_denials[name] = False
path_resolve = p.resolve()
os_realpath = os.path.realpath(p)
r = guard.receipt()
r['open_denials'] = open_denials
r['path_resolve_lexical_only'] = str(path_resolve) == str(p.absolute())
r['os_realpath_lexical_only'] = os_realpath == str(p.absolute())
print(json.dumps(r, sort_keys=True))
"""
    )
    assert value["installed_before_runtime_imports"] is True
    assert value["open_denials"] == {
        "Path.open": True,
        "builtins.open": True,
        "io.open": True,
        "os.open": True,
    }
    assert value["path_resolve_lexical_only"] is True
    assert value["os_realpath_lexical_only"] is True
    assert value["protected_open_attempts_denied"] >= 4
    assert value["protected_resolve_attempts_denied"] >= 2
    assert value["successful_protected_opens"] == 0
    assert value["successful_protected_resolves"] == 0


def test_guarded_runtime_import_records_denied_legacy_resolution_not_open():
    value = _guard_subprocess(
        """
import json
import run_lora_es_v71_v72_profile_calibration_v73c
import v73c_path_open_guard as guard
print(json.dumps(guard.receipt(), sort_keys=True))
"""
    )
    assert value["installed_before_runtime_imports"] is True
    assert value["protected_resolve_attempts_denied"] >= 1
    assert value["protected_open_attempts_denied"] == 0
    assert value["successful_protected_opens"] == 0
    assert value["successful_protected_resolves"] == 0
