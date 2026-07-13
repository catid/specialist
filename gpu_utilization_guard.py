#!/usr/bin/env python3
"""Monitor every GPU and fail if any stays underutilized during an experiment."""
import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path


QUERY = ("index,name,memory.used,utilization.gpu,power.draw")


def sample():
    output = subprocess.check_output([
        "nvidia-smi", f"--query-gpu={QUERY}",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        index, name, memory, utilization, power = (
            value.strip() for value in line.split(",", 4))
        rows.append({
            "index": int(index), "name": name, "memory_mib": int(memory),
            "utilization_percent": int(utilization), "power_watts": float(power),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--min-util", type=int, default=20)
    parser.add_argument("--grace-samples", type=int, default=3)
    parser.add_argument("--samples", type=int, default=0,
                        help="stop after N samples; 0 monitors until interrupted")
    parser.add_argument("--log", type=Path,
                        default=Path("experiments/gpu_utilization.jsonl"))
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    low_counts = {}
    iteration = 0
    with args.log.open("a") as log:
        while not args.samples or iteration < args.samples:
            rows = sample()
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            log.write(json.dumps({"timestamp": timestamp, "gpus": rows}) + "\n")
            log.flush()
            status = []
            for row in rows:
                index = row["index"]
                low_counts[index] = (low_counts.get(index, 0) + 1
                                     if row["utilization_percent"] < args.min_util
                                     else 0)
                status.append(f"{index}:{row['utilization_percent']}%")
            print(timestamp, " ".join(status), flush=True)
            offenders = [index for index, count in low_counts.items()
                         if count >= args.grace_samples]
            if offenders:
                raise SystemExit(
                    f"GPUs below {args.min_util}% for {args.grace_samples} samples: "
                    f"{offenders}")
            iteration += 1
            if not args.samples or iteration < args.samples:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
