#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


def parse_benchmark_output(text: str):
    # Example line:
    # matmul 256x256  mean=1.2345ms  min=1.1111ms  iters=20
    pattern = re.compile(r"^\s*(.+?)\s+mean=([0-9.]+)ms\s+min=([0-9.]+)ms", re.MULTILINE)
    parsed = {}
    for name, mean_ms, min_ms in pattern.findall(text):
        parsed[name.strip()] = {"mean_ms": float(mean_ms), "min_ms": float(min_ms)}
    return parsed


def main():
    if len(sys.argv) != 3:
        print("Usage: perf_gate.py <benchmark_output.txt> <baseline.json>", file=sys.stderr)
        return 2

    output_path = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2])

    observed = parse_benchmark_output(output_path.read_text())
    baseline = json.loads(baseline_path.read_text())

    failures = []
    for bench_name, cfg in baseline.items():
        if bench_name not in observed:
            failures.append(f"Missing benchmark in output: {bench_name}")
            continue
        max_allowed = float(cfg["max_min_ms"])
        got = observed[bench_name]["min_ms"]
        if got > max_allowed:
            failures.append(
                f"{bench_name}: min={got:.4f}ms exceeds gate {max_allowed:.4f}ms"
            )

    if failures:
        print("Performance gate FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1

    print("Performance gate PASSED")
    for bench_name, cfg in baseline.items():
        got = observed[bench_name]["min_ms"]
        print(f" - {bench_name}: {got:.4f}ms <= {cfg['max_min_ms']:.4f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
