#!/usr/bin/env python3
"""Preflight checks before starting SLE Level 3 stability counter."""

from __future__ import annotations

import argparse
import re
import shutil
import statistics
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def parse_l2_metrics(output: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    pattern = re.compile(r"(\w+)=([^\s]+)")
    for line in output.splitlines():
        if not line.startswith("[SLE-L2]") or "all gates passed" in line:
            continue
        kv = dict(pattern.findall(line))
        rows.append(
            {
                "family": kv["family"],
                "seed": float(kv["seed"]),
                "profile": kv["profile"],
                "fidelity": float(kv["fidelity"]),
                "teacher_acc": float(kv["teacher_acc"]),
                "sle_acc": float(kv["sle_acc"]),
                "sle_gate_count": float(kv["sle_gate_count"]),
                "compact_nodes": float(kv["compact_nodes"]),
                "teacher_p95_us": float(kv["teacher_p95_us"]),
                "sle_p95_us": float(kv["sle_p95_us"]),
            }
        )
    return rows


def decision_audit(rows: list[dict[str, float | str]]) -> str:
    fidelity_min = min(float(r["fidelity"]) for r in rows)
    teacher_acc = statistics.mean(float(r["teacher_acc"]) for r in rows)
    sle_acc = statistics.mean(float(r["sle_acc"]) for r in rows)

    latency_reg = []
    for r in rows:
        t = float(r["teacher_p95_us"])
        s = float(r["sle_p95_us"])
        latency_reg.append((s - t) / max(t, 1e-12) * 100.0)

    compression_wins = [
        float(r["sle_gate_count"]) < float(r["compact_nodes"])
        for r in rows
    ]

    lines = [
        "Decision audit:",
        f"  - Fidelity min: {fidelity_min:.6f} (hard gate >= 0.95, margin {fidelity_min - 0.95:+.6f})",
        f"  - Accuracy delta (SLE - teacher): {sle_acc - teacher_acc:+.6f}",
        f"  - p95 latency regression range: [{min(latency_reg):+.2f}%, {max(latency_reg):+.2f}%]",
    ]

    if all(compression_wins):
        lines.append("  - Win axis: compression profile only (SLE gate_count < compact model node count for all seeds).")
    elif any(compression_wins):
        lines.append("  - Win axis: mixed (compression win present, but not for every seed).")
    else:
        lines.append("  - Win axis: none detected against compact model.")

    if max(latency_reg) > 0.0 and all(compression_wins):
        lines.append("  - Verdict: weak win (size-only win with positive latency regression).")

    return "\n".join(lines)


def perf_stat(cmd: list[str], cwd: Path) -> str:
    if shutil.which("perf") is None:
        return "perf not available in environment"

    perf_cmd = [
        "perf",
        "stat",
        "-e",
        "cycles,instructions,cache-references,cache-misses,branches,branch-misses",
    ] + cmd
    proc = run_cmd(perf_cmd, cwd)
    if proc.returncode != 0:
        return f"perf failed (exit={proc.returncode}): {proc.stderr.strip()}"
    return proc.stderr.strip()


def configure_and_build(cwd: Path, build_dir: Path, extra: list[str]) -> tuple[int, str]:
    cfg = run_cmd(["cmake", "-S", ".", "-B", str(build_dir), *extra], cwd)
    if cfg.returncode != 0:
        return cfg.returncode, cfg.stdout + "\n" + cfg.stderr
    bld = run_cmd(["cmake", "--build", str(build_dir), "-j4"], cwd)
    return bld.returncode, cfg.stdout + cfg.stderr + bld.stdout + bld.stderr


def ctest_one(cwd: Path, build_dir: Path) -> tuple[int, str]:
    t = run_cmd([
        "ctest",
        "--test-dir",
        str(build_dir),
        "-R",
        "test_sle_level2_utility",
        "--output-on-failure",
    ], cwd)
    return t.returncode, t.stdout + t.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SLE Level 3 preflight checks")
    parser.add_argument("--manifest", default="tests/data/sle_level2_manifest.csv")
    parser.add_argument("--run-sanitizers", action="store_true")
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    binary = repo / "build" / "test_sle_level2_utility"
    if not binary.exists():
        print("ERROR: build/test_sle_level2_utility not found. Build tests first.")
        return 2

    l2 = run_cmd([str(binary), args.manifest], repo)
    if l2.returncode != 0:
        print(l2.stdout)
        print(l2.stderr)
        return l2.returncode

    rows = parse_l2_metrics(l2.stdout)
    if not rows:
        print("ERROR: no [SLE-L2] metric rows parsed")
        return 3

    print(decision_audit(rows))
    print("\nHPC micro-arch check (perf stat around test_sle_level2_utility):")
    print(perf_stat([str(binary), args.manifest], repo))

    if args.run_sanitizers:
        print("\nSanitizer / warning pre-sanitize checks:")
        checks = [
            ("asan_ubsan", [
                "-DUNIFIED_ML_BUILD_TESTS=ON",
                "-DUNIFIED_ML_ENABLE_ASAN=ON",
                "-DUNIFIED_ML_ENABLE_UBSAN=ON",
                "-DUNIFIED_ML_ENABLE_WERROR=ON",
            ]),
            ("msan", [
                "-DUNIFIED_ML_BUILD_TESTS=ON",
                "-DUNIFIED_ML_ENABLE_MSAN=ON",
                "-DUNIFIED_ML_ENABLE_WERROR=ON",
            ]),
        ]

        for name, flags in checks:
            bdir = repo / f"build_{name}"
            rc, out = configure_and_build(repo, bdir, flags)
            print(f"[{name}] configure+build exit={rc}")
            if rc == 0:
                trc, tout = ctest_one(repo, bdir)
                print(f"[{name}] test_sle_level2_utility exit={trc}")
                if trc != 0:
                    print(tout)
            else:
                print(out.splitlines()[-30:])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
