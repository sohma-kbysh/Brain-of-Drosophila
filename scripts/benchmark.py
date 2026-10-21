#!/usr/bin/env python
"""Scaling benchmark (addendum sections 6, 22).

Measures wall time, CPU utilisation, peak RSS and throughput for representative
workloads across worker counts, so that a large run is sized from measurement
rather than from assumption.

Deliberately capped: 128/256/384-worker points are NOT run automatically. This
is a shared machine and the load-check refuses to start a large sweep when
others are using it. Pass --max-workers explicitly and only when the machine is
quiet.

    python scripts/benchmark.py --workloads quantum --max-workers 16
    python scripts/benchmark.py --workloads all --max-workers 64 --force
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bioquantumfly.config import Config
from bioquantumfly.experiments.registry import make_runner
from bioquantumfly.runtime.profiles import load_warnings, resolve_profile
from bioquantumfly.runtime.resources import detect_resources
from bioquantumfly.runtime.sweep import run_sweep

#: Worker counts that are safe to explore on a shared machine without asking.
DEFAULT_WORKER_LADDER = (1, 2, 4, 8, 16, 32, 64)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _quantum_workload(scale: int) -> tuple[str, Config]:
    cfg = Config()
    cfg.quantum.T2_seconds = [4e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3] * scale
    cfg.quantum.n_carriers = 64
    return "feasibility_map", cfg


def _radical_pair_workload(scale: int) -> tuple[str, Config]:
    cfg = Config()
    cfg.quantum.T2_seconds = [1e-6] * (4 * scale)
    return "eve_ancilla_persistence", cfg


def _neural_workload(scale: int) -> tuple[str, Config]:
    cfg = Config()
    cfg.connectome.synthetic_n_neurons = 500 * scale
    cfg.host_selection.n_hosts = 16
    cfg.neural.duration_s = 0.2
    return "neural_baseline", cfg


def _hybrid_workload(scale: int) -> tuple[str, Config]:
    cfg = Config()
    cfg.connectome.synthetic_n_neurons = 400 * scale
    cfg.host_selection.n_hosts = 16
    cfg.neural.duration_s = 0.15
    cfg.coupling.gain_sweep = [0.0, 0.1, 1.0]
    return "hybrid_tamper_propagation", cfg


WORKLOADS: dict[str, Callable[[int], tuple[str, Config]]] = {
    "quantum": _quantum_workload,
    "radical_pair": _radical_pair_workload,
    "neural": _neural_workload,
    "hybrid": _hybrid_workload,
}


def benchmark_one(name: str, scale: int, workers: int, outdir: Path) -> dict[str, Any]:
    experiment, cfg = WORKLOADS[name](scale)
    from bioquantumfly.experiments.registry import get_experiment

    specs = get_experiment(experiment).build_specs(cfg)
    runner = make_runner(experiment, cfg)

    cpu_before = time.process_time()
    wall_start = time.perf_counter()
    df = run_sweep(specs, runner, outdir / f"{name}-w{workers}",
                   workers=workers, blas_threads=1, resume=False)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_before

    return {
        "workload": name,
        "experiment": experiment,
        "scale": scale,
        "workers": workers,
        "n_runs": len(specs),
        "n_rows": len(df),
        "wall_seconds": wall,
        "parent_cpu_seconds": cpu,
        "runs_per_second": len(specs) / wall if wall > 0 else float("nan"),
        "rows_per_second": len(df) / wall if wall > 0 else float("nan"),
        "peak_rss_mb": peak_rss_mb(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default="quantum",
                        help="comma-separated, or 'all': " + ", ".join(WORKLOADS))
    parser.add_argument("--max-workers", type=int, default=8,
                        help="highest worker count to test (default 8)")
    parser.add_argument("--scale", type=int, default=1, help="problem-size multiplier")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--force", action="store_true",
                        help="run even when the machine is already busy")
    args = parser.parse_args()

    resources = detect_resources(Path.cwd())
    profile = resolve_profile("development", workers=args.max_workers, resources=resources)
    for warning in load_warnings(resources, profile):
        print(f"warning: {warning}")

    if resources.is_heavily_loaded and not args.force:
        print("\nMachine is heavily loaded. Benchmarking now would measure "
              "contention, not scaling. Re-run when quiet, or pass --force.")
        return 1
    if args.max_workers > 64 and not args.force:
        print(f"\n{args.max_workers} workers is above the unattended ceiling of 64 "
              "on this shared machine. Pass --force if you have checked the load.")
        return 1

    names = list(WORKLOADS) if args.workloads == "all" else args.workloads.split(",")
    for name in names:
        if name not in WORKLOADS:
            parser.error(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}")

    ladder = [w for w in DEFAULT_WORKER_LADDER if w <= args.max_workers]
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for name in names:
        baseline = None
        for workers in ladder:
            row = benchmark_one(name, args.scale, workers, args.output)
            baseline = baseline or row["wall_seconds"]
            row["speedup_vs_1_worker"] = baseline / row["wall_seconds"]
            row["parallel_efficiency"] = row["speedup_vs_1_worker"] / workers
            rows.append(row)
            print(f"{name:14s} workers={workers:3d}  {row['wall_seconds']:7.2f}s  "
                  f"speedup {row['speedup_vs_1_worker']:5.2f}x  "
                  f"efficiency {row['parallel_efficiency']:.0%}  "
                  f"peak RSS {row['peak_rss_mb']:.0f} MB")

    df = pd.DataFrame(rows)
    df.to_csv(args.output / "benchmark.csv", index=False)
    (args.output / "hardware.json").write_text(resources.to_json())
    print(f"\nwrote {args.output / 'benchmark.csv'}")

    best = df.loc[df.groupby("workload")["wall_seconds"].idxmin()]
    print("\nfastest configuration per workload:")
    for row in best.itertuples():
        print(f"  {row.workload:14s} {row.workers} workers "
              f"({row.wall_seconds:.2f}s, efficiency {row.parallel_efficiency:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
