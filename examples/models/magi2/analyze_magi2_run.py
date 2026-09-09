#!/usr/bin/env python3
"""Validate and summarize MAGI-2 JSONL integration-training evidence."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the JSONL validation and summary arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected-start", type=int, required=True)
    parser.add_argument("--expected-end", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Validate one run's metrics and write a compact summary report."""
    args = parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line]
    expected_steps = list(range(args.expected_start, args.expected_end + 1))
    actual_steps = [int(row["step"]) for row in rows]
    if actual_steps != expected_steps:
        raise ValueError(f"steps differ: expected {expected_steps}, got {actual_steps}")
    if any(not row["losses_finite"] for row in rows):
        raise ValueError("non-finite loss detected")
    if any(not row["grad_norm_finite"] for row in rows):
        raise ValueError("non-finite gradient norm detected")
    if any(row["skipped_iteration"] for row in rows):
        raise ValueError("skipped optimizer iteration detected")

    losses = [float(row["losses"]["flow_mse"]) for row in rows]
    grad_norms = [float(row["grad_norm"]) for row in rows]
    throughputs = [float(row["global_tokens_per_second"]) for row in rows]
    measured_throughputs = throughputs[min(args.warmup_steps, len(rows) - 1) :]
    values = losses + grad_norms + throughputs
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite scalar detected")

    report = {
        "status": "passed",
        "steps": {"start": actual_steps[0], "end": actual_steps[-1], "count": len(rows)},
        "loss": {
            "first": losses[0],
            "last": losses[-1],
            "min": min(losses),
            "max": max(losses),
            "mean": statistics.fmean(losses),
        },
        "grad_norm": {"min": min(grad_norms), "max": max(grad_norms)},
        "global_tokens_per_second_after_warmup": {
            "mean": statistics.fmean(measured_throughputs),
            "median": statistics.median(measured_throughputs),
        },
        "peak_memory_allocated_gib": max(float(row["memory_allocated_gib"]["max"]) for row in rows),
        "peak_memory_reserved_gib": max(float(row["memory_reserved_gib"]["max"]) for row in rows),
        "skipped_iterations": 0,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
