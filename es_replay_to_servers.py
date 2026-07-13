#!/usr/bin/env python
"""Re-apply journal commits to freshly restarted ES servers (resume support)."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, "/home/catid/specialist")
from es_common import perturb


def _resolve_include_regex(row, line_number, legacy_include_regex,
                           allow_legacy_full_model):
    if "include_regex" in row:
        include_regex = row["include_regex"]
        if include_regex is not None and not isinstance(include_regex, str):
            raise ValueError(
                f"journal line {line_number} has non-string include_regex")
        if include_regex == "":
            raise ValueError(
                f"journal line {line_number} has an empty include_regex; "
                "empty targets are ambiguous")
        return include_regex

    if row.get("schema_version", 1) >= 2:
        raise ValueError(
            f"journal line {line_number} declares schema_version >= 2 but "
            "does not contain include_regex")
    if legacy_include_regex is not None:
        if legacy_include_regex == "":
            raise ValueError("legacy include_regex must be non-empty")
        return legacy_include_regex
    if allow_legacy_full_model:
        return None
    raise ValueError(
        f"journal line {line_number} predates per-generation include_regex; "
        "its target is ambiguous. Refusing to replay. Pass "
        "--legacy-include-regex REGEX for a targeted legacy run, or "
        "--allow-legacy-full-model only after confirming it updated all units.")


def prepare_replay(journal, legacy_include_regex=None,
                   allow_legacy_full_model=False):
    """Validate every row before making the first mutating server request."""
    prepared = []
    with Path(journal).open() as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid journal JSON on line {line_number}: {exc}") from exc
            expected_gen = len(prepared)
            if row.get("gen") != expected_gen:
                raise ValueError(
                    f"journal line {line_number} has generation "
                    f"{row.get('gen')!r}; expected {expected_gen}")
            try:
                ops = [(int(seed), float(coeff)) for seed, coeff in row["ops"]]
                sigma = float(row["sigma"])
                rank = int(row["rank"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"journal line {line_number} has invalid replay fields") from exc
            include_regex = _resolve_include_regex(
                row, line_number, legacy_include_regex, allow_legacy_full_model)
            prepared.append((expected_gen, ops, sigma, rank, include_regex))
    return prepared


def replay_journal(journal, ports, legacy_include_regex=None,
                   allow_legacy_full_model=False):
    generations = prepare_replay(
        journal, legacy_include_regex=legacy_include_regex,
        allow_legacy_full_model=allow_legacy_full_model)
    for generation, ops, sigma, rank, include_regex in generations:
        for port in ports:
            perturb(port, ops, sigma, rank=rank, mode="commit",
                    include_regex=include_regex)
        target = include_regex if include_regex is not None else "<all units>"
        print(f"gen {generation} replayed (target: {target})", flush=True)
    return len(generations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="/home/catid/specialist/es_journal.jsonl")
    ap.add_argument("--ports", type=int, nargs="+", default=[30001, 30002, 30003, 30004])
    legacy = ap.add_mutually_exclusive_group()
    legacy.add_argument(
        "--legacy-include-regex", metavar="REGEX",
        help="target to use for every legacy row missing include_regex")
    legacy.add_argument(
        "--allow-legacy-full-model", action="store_true",
        help="explicitly confirm that legacy rows updated the full model")
    args = ap.parse_args()
    try:
        replay_journal(
            args.journal, args.ports,
            legacy_include_regex=args.legacy_include_regex,
            allow_legacy_full_model=args.allow_legacy_full_model)
    except ValueError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
