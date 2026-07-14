#!/usr/bin/env python3
"""Deterministically render the four frozen Qwen3.6 edge-split v6 plans.

The existing front/back manifests and the two middle controls all select one
complete [linear, linear, linear, full] four-layer motif.  Rendering is kept
separate from the frozen runtime allowlist: this utility can reproduce or
check artifacts, but it cannot make an unpinned plan executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import es_layer_plan


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "Qwen3.6-35B-A3B"
PLAN_LAYERS_V6 = {
    "front": (0, 1, 2, 3),
    "middle_early": (16, 17, 18, 19),
    "middle_late": (20, 21, 22, 23),
    "back": (36, 37, 38, 39),
}
PLAN_FILES_V6 = {
    "front": "front_dense.json",
    "middle_early": "middle_early_dense_v6.json",
    "middle_late": "middle_late_dense_v6.json",
    "back": "back_dense.json",
}


def canonical_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_plan_v6(name, model_path=MODEL):
    if name not in PLAN_LAYERS_V6:
        raise ValueError(f"unknown v6 edge-split plan: {name!r}")
    manifest = es_layer_plan.plan_manifest(
        Path(model_path), "front", ["dense"], PLAN_LAYERS_V6[name],
    )
    manifest["plan"] = name
    manifest.pop("plan_sha256", None)
    manifest["plan_sha256"] = canonical_sha256(manifest)
    return manifest


def render_plan_v6(name, model_path=MODEL):
    return json.dumps(
        build_plan_v6(name, model_path), indent=2, sort_keys=True,
        allow_nan=False,
    ) + "\n"


def check_directory_v6(directory, model_path=MODEL):
    directory = Path(directory)
    results = {}
    for name, filename in PLAN_FILES_V6.items():
        path = directory / filename
        expected = render_plan_v6(name, model_path).encode("utf-8")
        actual = path.read_bytes()
        if actual != expected:
            raise ValueError(f"v6 plan bytes differ: {path}")
        results[name] = {
            "path": str(path.resolve()),
            "file_sha256": hashlib.sha256(actual).hexdigest(),
            "plan_sha256": build_plan_v6(name, model_path)["plan_sha256"],
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--plan", choices=tuple(PLAN_LAYERS_V6))
    parser.add_argument("--check-directory", type=Path)
    args = parser.parse_args(argv)
    if (args.plan is None) == (args.check_directory is None):
        parser.error("choose exactly one of --plan or --check-directory")
    if args.check_directory is not None:
        print(json.dumps(
            check_directory_v6(args.check_directory, args.model),
            indent=2, sort_keys=True,
        ))
    else:
        print(render_plan_v6(args.plan, args.model), end="")


if __name__ == "__main__":
    main()
