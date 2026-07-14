#!/usr/bin/env python3
"""Build or verify the frozen V13 train-only panel manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import eggroll_es_train_panel_sampler_v13 as sampler


DEFAULT_OUTPUT = Path(
    "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
)


def encoded(manifest):
    return json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", default=str(sampler.DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)

    rows, source_sha = sampler.load_frozen_train(args.train_source)
    manifest = sampler.build_manifest(rows, args.train_source, source_sha)
    raw = encoded(manifest)
    output = Path(args.output)
    if args.verify_existing:
        if output.read_bytes() != raw:
            raise RuntimeError("existing V13 panel manifest is not reproducible")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as destination:
            destination.write(raw)
    print(json.dumps({
        "schema": "eggroll-es-train-panel-build-v13",
        "output": str(output),
        "sha256": sampler.file_sha256(output),
        "content_sha256": manifest["content_sha256_before_self_field"],
        "panels": len(manifest["panels"]),
        "rows_per_panel": sampler.PANEL_SIZE,
        "gpu_launched": False,
    }, sort_keys=True))
    return manifest


if __name__ == "__main__":
    main()
