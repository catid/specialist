#!/usr/bin/env python3
"""Summarize schema-v2 ES journals and their human-readable probe logs."""
import argparse
import json
import re
from pathlib import Path


RESUME_PROBE = re.compile(
    r"\[gen (\d+)\] probe (?:F1|reward) train ([0-9.]+) heldout ([0-9.]+)")
PERIODIC_PROBE = re.compile(
    r"\[gen (\d+)\].*probe_train_(?:F1|reward) ([0-9.]+) "
    r"probe_held_(?:F1|reward) ([0-9.]+)")


def parse_probes(log):
    """Return probe states keyed by the number of committed generations.

    A resume/start probe printed as ``[gen N] probe`` sees N commits.  A
    periodic probe on the generation-N result line runs after that commit, so
    it sees N+1.  Tracking that distinction prevents an eval-every boundary
    from being mislabeled as the final state of a longer resumed run.
    """
    probes = []
    for line in log.splitlines():
        match = RESUME_PROBE.search(line)
        if match:
            probes.append({
                "committed_generations": int(match.group(1)),
                "train": float(match.group(2)),
                "heldout": float(match.group(3)),
            })
            continue
        match = PERIODIC_PROBE.search(line)
        if match:
            probes.append({
                "committed_generations": int(match.group(1)) + 1,
                "train": float(match.group(2)),
                "heldout": float(match.group(3)),
            })
    return probes


def summarize(journal_path):
    with journal_path.open() as journal:
        rows = [json.loads(line) for line in journal if line.strip()]
    if not rows:
        raise ValueError(f"empty journal: {journal_path}")
    log_path = journal_path.with_suffix(".log")
    log = log_path.read_text() if log_path.exists() else ""
    probes = parse_probes(log)
    first = min(
        enumerate(probes),
        key=lambda indexed: (indexed[1]["committed_generations"], indexed[0]),
    )[1] if probes else None
    last = max(
        enumerate(probes),
        key=lambda indexed: (indexed[1]["committed_generations"], indexed[0]),
    )[1] if probes else None
    states = [row.get("replica_state") for row in rows
              if row.get("replica_state") is not None]
    return {
        "name": journal_path.stem,
        "journal": str(journal_path.resolve()),
        "log": str(log_path.resolve()) if log_path.exists() else None,
        "generations": len(rows),
        "generation_range": [rows[0]["gen"], rows[-1]["gen"]],
        "layer_plan": rows[0].get("layer_plan", {}).get("plan"),
        "layers": rows[0].get("layer_plan", {}).get("layers"),
        "num_planned_units": rows[0].get("layer_plan", {}).get("num_units"),
        "start_probe_after_generations": (
            first["committed_generations"] if first else None),
        "start_train_probe": first["train"] if first else None,
        "start_heldout_probe": first["heldout"] if first else None,
        "end_probe_after_generations": (
            last["committed_generations"] if last else None),
        "end_train_probe": last["train"] if last else None,
        "end_heldout_probe": last["heldout"] if last else None,
        "heldout_delta": (last["heldout"] - first["heldout"]
                          if first and last else None),
        "mean_member_fitness_last": rows[-1]["mean_fit"],
        "max_member_fitness_all": max(row["max_fit"] for row in rows),
        "verified_commits": len(states),
        "all_verified_unit_counts": sorted({state["num_units"] for state in states}),
        "target_manifest_hashes": sorted(
            {state["target_manifest_hash"] for state in states}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "es-experiment-summary-v1",
        "experiments": [summarize(path) for path in args.journals],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
