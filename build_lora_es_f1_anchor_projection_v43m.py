#!/usr/bin/env python3
"""Materialize the sealed CPU-only V43M projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lora_es_f1_anchor_projection_v43m as projection
import run_lora_es_multi_anchor_v43i as v43i


OUTPUT = (
    projection.ROOT / "experiments/eggroll_es_hpo/projections/"
    "lora_es_three_anchor_projection_v43m.json"
).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = projection.build_projection_v43m()
    v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "required_anchors": value["required_centered_rank_anchors"],
        "population_resampled": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
