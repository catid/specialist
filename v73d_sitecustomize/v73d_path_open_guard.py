"""Content-free V3 quarantine denial for V73D systems-only tracing."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import pathlib
import threading
from typing import Any


ROOT = pathlib.Path(__file__).absolute().parents[1]
EXACT_PATH_IDENTITY_SHA256 = frozenset({
    "07b0ef637696b230de90504ae6cd798422d806e9b9465f3d683478714a527d96",
    "1c82c6288eda8e538d45d29f63d84d3983fa7c9b1c00dd1dac3d8d8c1f9f4aa5",
    "1f4cf1a6a1890fd7810a547fbdce51553a1ae2f191a6cc8b6edae4989fe5f4b5",
})
PREFIX_IDENTITY_SHA256 = frozenset({
    "39a0e5b5693317f52d9f539a6666d8af6f33d3269bf263d0e2393d24beaa0a42",
    "7af72f106b10b5ab698436550819f1b1906232fd5f075ef011f8cd958d81b128",
})
EXACT_IDENTITY_SCHEMA = "repository-relative-path-v1"
PREFIX_IDENTITY_SCHEMA = "repository-relative-path-prefix-v1"
BOUNDARY_REGISTRY_FILE_SHA256 = (
    "3d8ef097a1419e03f4b735e6f8d30e5a876b0a8e86c4b0f1ac100114cb7daf5d"
)
BOUNDARY_REGISTRY_CONTENT_SHA256 = (
    "5a8c6dbd3b0c225992bb2cc4ae5c02a5ec7774a175fe1a138c4e78260eed7fc7"
)
BOUNDARY_POLICY_SHA256 = (
    "3758e272396ebc0cbed5a933469665530275c9eabe99f1c0d6210464e7d1a48c"
)

_LOCK = threading.Lock()
_TLS = threading.local()
_INSTALLED = False
_INSTALLATION_MECHANISM: str | None = None
_INSTALLATION_PID: int | None = None
_COUNTERS = {
    "open_events_seen": 0,
    "resolve_events_seen": 0,
    "metadata_events_seen": 0,
    "enumeration_events_seen": 0,
    "protected_open_attempts_denied": 0,
    "protected_resolve_attempts_denied": 0,
    "protected_metadata_attempts_denied": 0,
    "protected_enumeration_attempts_denied": 0,
    "successful_protected_opens": 0,
    "successful_protected_resolves": 0,
    "successful_protected_metadata": 0,
    "successful_protected_enumerations": 0,
}
_ORIGINALS: dict[str, Any] = {}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _identity(schema: str, value: str) -> str:
    return canonical_sha256({"schema": schema, "value": value})


def _lexical_classification(value: Any) -> dict[str, Any]:
    if isinstance(value, int):
        return {"scope": "descriptor", "denied": False, "path": value}
    try:
        raw = os.fspath(value)
    except TypeError:
        return {"scope": "non_path", "denied": False, "path": value}
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not os.path.isabs(raw) and ".." in pathlib.PurePath(raw).parts:
        return {
            "scope": "parent_traversal",
            "denied": True,
            "path": raw,
        }
    absolute = os.path.normpath(os.path.abspath(raw))
    try:
        within_root = os.path.commonpath((str(ROOT), absolute)) == str(ROOT)
    except ValueError:
        within_root = False
    if not within_root:
        return {"scope": "outside_repository", "denied": False, "path": raw}
    relative = os.path.relpath(absolute, str(ROOT)).replace(os.sep, "/")
    if relative == ".":
        relative = ""
    exact = _identity(EXACT_IDENTITY_SCHEMA, relative)
    parts = relative.split("/") if relative else []
    ancestors = ["/".join(parts[:index]) for index in range(1, len(parts) + 1)]
    prefix_match = any(
        _identity(PREFIX_IDENTITY_SCHEMA, ancestor) in PREFIX_IDENTITY_SHA256
        for ancestor in ancestors
    )
    return {
        "scope": "repository",
        "denied": exact in EXACT_PATH_IDENTITY_SHA256 or prefix_match,
        "path": absolute,
    }


def _count(name: str) -> None:
    with _LOCK:
        _COUNTERS[name] += 1


def _realpath_original(value: Any) -> str:
    previous = getattr(_TLS, "internal_resolution", False)
    _TLS.internal_resolution = True
    try:
        return _ORIGINALS["realpath"](value)
    finally:
        _TLS.internal_resolution = previous


def _gate(value: Any, event: str) -> tuple[dict[str, Any], Any]:
    _count(f"{event}_events_seen")
    lexical = _lexical_classification(value)
    if lexical["denied"]:
        _count(f"protected_{event}_attempts_denied")
        return lexical, value
    if lexical["scope"] != "repository":
        return lexical, value
    resolved = _realpath_original(lexical["path"])
    classified = _lexical_classification(resolved)
    if classified["scope"] != "repository" or classified["denied"]:
        _count(f"protected_{event}_attempts_denied")
        return {**classified, "denied": True}, value
    return classified, resolved


def _deny(label: str) -> None:
    raise PermissionError(f"V73D systems-only protected {label} denied")


def install(mechanism: str) -> None:
    global _INSTALLED, _INSTALLATION_MECHANISM, _INSTALLATION_PID
    if mechanism not in {
        "controller_sitecustomize",
        "ray_actor_worker_extension_pre_parent_import",
    }:
        raise RuntimeError("V73D path guard installation mechanism is invalid")
    with _LOCK:
        if _INSTALLED:
            if (
                _INSTALLATION_MECHANISM != mechanism
                or _INSTALLATION_PID != os.getpid()
            ):
                raise RuntimeError(
                    "V73D path guard cannot change installation mechanism"
                )
            return
        _ORIGINALS.update({
            "builtins_open": builtins.open,
            "io_open": io.open,
            "os_open": os.open,
            "os_stat": os.stat,
            "os_lstat": os.lstat,
            "os_listdir": os.listdir,
            "os_scandir": os.scandir,
            "path_resolve": pathlib.Path.resolve,
            "realpath": os.path.realpath,
        })

        def guarded_builtin_open(file, *args, **kwargs):
            classified, target = _gate(file, "open")
            if classified["denied"]:
                _deny("open")
            return _ORIGINALS["builtins_open"](target, *args, **kwargs)

        def guarded_io_open(file, *args, **kwargs):
            classified, target = _gate(file, "open")
            if classified["denied"]:
                _deny("open")
            return _ORIGINALS["io_open"](target, *args, **kwargs)

        def guarded_os_open(file, *args, **kwargs):
            if kwargs.get("dir_fd") is not None and not os.path.isabs(
                os.fsdecode(os.fspath(file))
            ):
                _count("open_events_seen")
                _count("protected_open_attempts_denied")
                _deny("dir-fd open")
            classified, target = _gate(file, "open")
            if classified["denied"]:
                _deny("open")
            return _ORIGINALS["os_open"](target, *args, **kwargs)

        def guarded_stat(path, *args, **kwargs):
            if getattr(_TLS, "internal_resolution", False):
                return _ORIGINALS["os_stat"](path, *args, **kwargs)
            if kwargs.get("dir_fd") is not None and not isinstance(path, int):
                raw = os.fsdecode(os.fspath(path))
                if not os.path.isabs(raw):
                    _count("metadata_events_seen")
                    _count("protected_metadata_attempts_denied")
                    _deny("dir-fd metadata")
            classified, target = _gate(path, "metadata")
            if classified["denied"]:
                _deny("metadata")
            return _ORIGINALS["os_stat"](target, *args, **kwargs)

        def guarded_lstat(path, *args, **kwargs):
            if getattr(_TLS, "internal_resolution", False):
                return _ORIGINALS["os_lstat"](path, *args, **kwargs)
            if kwargs.get("dir_fd") is not None and not isinstance(path, int):
                raw = os.fsdecode(os.fspath(path))
                if not os.path.isabs(raw):
                    _count("metadata_events_seen")
                    _count("protected_metadata_attempts_denied")
                    _deny("dir-fd metadata")
            classified, target = _gate(path, "metadata")
            if classified["denied"]:
                _deny("metadata")
            return _ORIGINALS["os_lstat"](target, *args, **kwargs)

        def guarded_listdir(path="."):
            classified, target = _gate(path, "enumeration")
            if classified["denied"]:
                _deny("enumeration")
            return _ORIGINALS["os_listdir"](target)

        def guarded_scandir(path="."):
            classified, target = _gate(path, "enumeration")
            if classified["denied"]:
                _deny("enumeration")
            return _ORIGINALS["os_scandir"](target)

        def guarded_resolve(self, *args, **kwargs):
            classified, target = _gate(self, "resolve")
            if classified["denied"]:
                return pathlib.Path(os.path.abspath(os.fspath(self)))
            if classified["scope"] != "repository":
                return _ORIGINALS["path_resolve"](self, *args, **kwargs)
            previous = getattr(_TLS, "internal_resolution", False)
            _TLS.internal_resolution = True
            try:
                resolved = _ORIGINALS["path_resolve"](
                    pathlib.Path(target), *args, **kwargs
                )
            finally:
                _TLS.internal_resolution = previous
            rechecked = _lexical_classification(resolved)
            if rechecked["scope"] != "repository" or rechecked["denied"]:
                _count("protected_resolve_attempts_denied")
                return pathlib.Path(os.path.abspath(os.fspath(self)))
            return resolved

        def guarded_realpath(filename, *args, **kwargs):
            if getattr(_TLS, "internal_resolution", False):
                return _ORIGINALS["realpath"](filename, *args, **kwargs)
            classified, target = _gate(filename, "resolve")
            if classified["denied"]:
                return os.path.abspath(os.fspath(filename))
            return _ORIGINALS["realpath"](target, *args, **kwargs)

        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        os.open = guarded_os_open
        os.stat = guarded_stat
        os.lstat = guarded_lstat
        os.listdir = guarded_listdir
        os.scandir = guarded_scandir
        pathlib.Path.resolve = guarded_resolve
        os.path.realpath = guarded_realpath
        _INSTALLED = True
        _INSTALLATION_MECHANISM = mechanism
        _INSTALLATION_PID = os.getpid()


def installed() -> bool:
    return _INSTALLED


def installation_mechanism() -> str | None:
    return _INSTALLATION_MECHANISM


def receipt() -> dict[str, Any]:
    with _LOCK:
        counters = dict(_COUNTERS)
    result = {
        "schema": "qwen36-v73d-systems-only-path-guard-process-v1",
        "pid": os.getpid(),
        "installed_before_runtime_imports": _INSTALLED,
        "installation_mechanism": _INSTALLATION_MECHANISM,
        "installation_pid": _INSTALLATION_PID,
        "boundary_registry_file_sha256": BOUNDARY_REGISTRY_FILE_SHA256,
        "boundary_registry_content_sha256": BOUNDARY_REGISTRY_CONTENT_SHA256,
        "boundary_policy_sha256": BOUNDARY_POLICY_SHA256,
        "exact_path_identity_count": len(EXACT_PATH_IDENTITY_SHA256),
        "exact_path_identity_set_sha256": canonical_sha256(
            sorted(EXACT_PATH_IDENTITY_SHA256)
        ),
        "prefix_identity_count": len(PREFIX_IDENTITY_SHA256),
        "prefix_identity_set_sha256": canonical_sha256(
            sorted(PREFIX_IDENTITY_SHA256)
        ),
        **counters,
        "protected_path_values_persisted": False,
        "quality_hpo_or_promotion_authorized": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result
