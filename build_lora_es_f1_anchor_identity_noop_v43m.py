#!/usr/bin/env python3
"""Seal the CPU-only V43M identity/no-op decision."""

from __future__ import annotations

import json
from pathlib import Path

import lora_es_f1_anchor_projection_v43m as subject


OUTPUT = (
    subject.ROOT / "experiments/eggroll_es_hpo/projections/"
    "lora_es_three_anchor_identity_noop_v43m.json"
).resolve()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    value = subject.build_identity_noop_v43m()
    subject.v43i.v40a.atomic_json(OUTPUT, value)
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": subject.v43i.v40a.file_sha256(OUTPUT),
        "content_sha256": value["content_sha256_before_self_field"],
        "decision": value["decision"],
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
