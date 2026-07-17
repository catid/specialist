"""Content-free protected-path open/resolve denial for V73C systems tracing."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import pathlib
import threading
from typing import Any


# SHA-256 identities of normalized absolute paths.  Paths and protected bytes
# are intentionally absent from this runtime module and from every receipt.
PROHIBITED_PATH_IDENTITY_SHA256 = frozenset({
    "15e5376fd03c8b15555cfcf2332f3ddd0e7da2d6334f65319f92e682c26ecfc2",
    "3254bb909744e185f6a8f16bdbe60cc500f787cb2f819461d0f6aa341fd86c8e",
    "3780b2d1003397ca4de78cbc37f027151ede10fdaa70cbdc7bc6641f9f2780f3",
    "52bfa1275791f8baa29ed2dc9752f2919371e25f1315802cbf12359fdeea1b56",
    "54960685d271dd6163ba46c1f1d063321858e8a74083441b646ac6539a3b6332",
    "5660ecb57b62fb02b1e8eeeabdfee5284773e191a4616c4423e7dc7b21fd0475",
    "ab805360ba5fbc9fcef3aba422ab7be2490c2c9b36664c282cb7ae37babb3f5b",
    "bb99d780975c35071a1802a0c55a16bb23d220c1d6c0cee19825b04271d4182e",
    "f4ada3505bd876d33b49c0f7d98e4858932d5057b2dcce660ed9aeb259c541e1",
})

_LOCK = threading.Lock()
_INSTALLED = False
_COUNTERS = {
    "open_events_seen": 0,
    "resolve_events_seen": 0,
    "protected_open_attempts_denied": 0,
    "protected_resolve_attempts_denied": 0,
    "successful_protected_opens": 0,
    "successful_protected_resolves": 0,
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


def _path_identity(value: Any) -> str | None:
    if isinstance(value, int):
        return None
    try:
        path = os.fspath(value)
    except TypeError:
        return None
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    normalized = os.path.normcase(os.path.abspath(path))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _prohibited(value: Any) -> bool:
    identity = _path_identity(value)
    return identity in PROHIBITED_PATH_IDENTITY_SHA256


def _count(name: str) -> None:
    with _LOCK:
        _COUNTERS[name] += 1


def install() -> None:
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        _ORIGINALS.update({
            "builtins_open": builtins.open,
            "io_open": io.open,
            "os_open": os.open,
            "path_resolve": pathlib.Path.resolve,
            "realpath": os.path.realpath,
        })

        def guarded_builtin_open(file, *args, **kwargs):
            _count("open_events_seen")
            if _prohibited(file):
                _count("protected_open_attempts_denied")
                raise PermissionError("V73C systems-only protected open denied")
            return _ORIGINALS["builtins_open"](file, *args, **kwargs)

        def guarded_io_open(file, *args, **kwargs):
            _count("open_events_seen")
            if _prohibited(file):
                _count("protected_open_attempts_denied")
                raise PermissionError("V73C systems-only protected open denied")
            return _ORIGINALS["io_open"](file, *args, **kwargs)

        def guarded_os_open(file, *args, **kwargs):
            _count("open_events_seen")
            if _prohibited(file):
                _count("protected_open_attempts_denied")
                raise PermissionError("V73C systems-only protected open denied")
            return _ORIGINALS["os_open"](file, *args, **kwargs)

        def guarded_resolve(self, *args, **kwargs):
            _count("resolve_events_seen")
            if _prohibited(self):
                _count("protected_resolve_attempts_denied")
                # Return only a lexical absolute path.  The filesystem-backed
                # resolution is denied, while legacy import-time constants can
                # remain inert and the later open gate remains fail-closed.
                return pathlib.Path(os.path.abspath(os.fspath(self)))
            return _ORIGINALS["path_resolve"](self, *args, **kwargs)

        def guarded_realpath(filename, *args, **kwargs):
            _count("resolve_events_seen")
            if _prohibited(filename):
                _count("protected_resolve_attempts_denied")
                return os.path.abspath(os.fspath(filename))
            return _ORIGINALS["realpath"](filename, *args, **kwargs)

        builtins.open = guarded_builtin_open
        io.open = guarded_io_open
        os.open = guarded_os_open
        pathlib.Path.resolve = guarded_resolve
        os.path.realpath = guarded_realpath
        _INSTALLED = True


def installed() -> bool:
    return _INSTALLED


def receipt() -> dict[str, Any]:
    with _LOCK:
        counters = dict(_COUNTERS)
    result = {
        "schema": "qwen36-v73c-systems-only-path-guard-process-v1",
        "pid": os.getpid(),
        "installed_before_runtime_imports": _INSTALLED,
        "prohibited_path_identity_count": len(PROHIBITED_PATH_IDENTITY_SHA256),
        "prohibited_path_identity_set_sha256": canonical_sha256(
            sorted(PROHIBITED_PATH_IDENTITY_SHA256)
        ),
        **counters,
        "protected_path_values_persisted": False,
        "quality_hpo_or_promotion_authorized": False,
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result
