"""Resumable, parallel parameter sweeps (addendum sections 7, 8, 14).

Design constraints that shaped this module:

* Sweeps must survive SSH disconnection and partial failure, so state lives in
  a manifest on disk, not in memory.
* `/home` is NFS here, which is poor at many-tiny-file workloads, so results
  are batched into Parquet rather than written one file per run.
* Worker count must not change results, so every run seeds itself from its own
  `RunSpec` (see `bioquantumfly.runtime.seeds`).
"""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from bioquantumfly.runtime.blas import worker_initializer
from bioquantumfly.runtime.runspec import RunSpec


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    rows: list[dict[str, Any]]
    error: str | None = None
    wall_seconds: float = 0.0


class SweepManifest:
    """Lightweight on-disk record of which runs are done.

    JSON Lines, appended to. One line per status transition; the last line for
    a run id wins. Append-only keeps concurrent writes from corrupting earlier
    entries, which matters when a sweep is interrupted mid-write.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a killed process
                self._state[rec["run_id"]] = rec

    def record(self, run_id: str, status: RunStatus, **extra: Any) -> None:
        rec = {"run_id": run_id, "status": status.value, "time": time.time(), **extra}
        self._state[run_id] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def status_of(self, run_id: str) -> RunStatus:
        rec = self._state.get(run_id)
        return RunStatus(rec["status"]) if rec else RunStatus.PENDING

    def is_complete(self, run_id: str) -> bool:
        return self.status_of(run_id) is RunStatus.COMPLETE

    def summary(self) -> dict[str, int]:
        out = {s.value: 0 for s in RunStatus}
        for rec in self._state.values():
            out[rec["status"]] += 1
        return out


def _execute(spec_and_fn: tuple[RunSpec, Callable[[RunSpec], list[dict[str, Any]]]]) -> RunResult:
    """Run one spec in a worker process, converting failure into a result.

    A failed run must not kill the sweep: it is recorded and can be rerun later
    with --resume while completed runs are skipped.
    """
    spec, fn = spec_and_fn
    start = time.perf_counter()
    try:
        rows = fn(spec)
    except Exception:  # noqa: BLE001 - deliberately broad: isolate run failures
        return RunResult(
            spec.run_id, RunStatus.FAILED, [],
            error=traceback.format_exc(limit=8),
            wall_seconds=time.perf_counter() - start,
        )
    base = spec.to_dict()
    # A per-run row index makes deduplication experiment-agnostic. Guessing at
    # identifying columns instead (an earlier version keyed on run_id plus
    # whichever of t_over_T2/eta happened to exist) silently collapsed every
    # other dimension a run emitted -- 900 rows became 60.
    return RunResult(
        spec.run_id, RunStatus.COMPLETE,
        [{**base, "_row_index": i, **row} for i, row in enumerate(rows)],
        wall_seconds=time.perf_counter() - start,
    )


def run_sweep(
    specs: Sequence[RunSpec],
    fn: Callable[[RunSpec], list[dict[str, Any]]],
    output_dir: str | Path,
    *,
    workers: int = 4,
    blas_threads: int = 1,
    resume: bool = True,
    batch_size: int = 200,
    progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Execute `fn` over `specs`, writing results incrementally.

    Parameters
    ----------
    batch_size
        How many completed runs to accumulate before flushing a Parquet part.
        Batching exists because the project lives on NFS, where writing one
        small file per run is slow and leaves tens of thousands of files.
    resume
        Skip runs the manifest already marks complete.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    manifest = SweepManifest(output_dir / "manifest.jsonl")

    todo = [s for s in specs if not (resume and manifest.is_complete(s.run_id))]
    skipped = len(specs) - len(todo)
    if progress:
        progress(0, len(todo), f"{skipped} already complete, {len(todo)} to run")

    buffer: list[dict[str, Any]] = []
    part_index = len(list(parts_dir.glob("part-*.parquet")))
    done = 0

    def flush() -> None:
        nonlocal buffer, part_index
        if not buffer:
            return
        pd.DataFrame(buffer).to_parquet(parts_dir / f"part-{part_index:05d}.parquet", index=False)
        part_index += 1
        buffer = []

    if workers == 1:
        # Serial path: identical results, easier debugging, and the reference
        # against which the parallel path is tested for determinism.
        worker_initializer(blas_threads)
        for spec in todo:
            result = _execute((spec, fn))
            manifest.record(result.run_id, result.status,
                            wall_seconds=result.wall_seconds, error=result.error)
            buffer.extend(result.rows)
            done += 1
            if progress:
                progress(done, len(todo), result.status.value)
            if len(buffer) >= batch_size:
                flush()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_initializer,
            initargs=(blas_threads,),
        ) as pool:
            futures = {pool.submit(_execute, (spec, fn)): spec for spec in todo}
            for fut in as_completed(futures):
                result = fut.result()
                manifest.record(result.run_id, result.status,
                                wall_seconds=result.wall_seconds, error=result.error)
                buffer.extend(result.rows)
                done += 1
                if progress:
                    progress(done, len(todo), result.status.value)
                if len(buffer) >= batch_size:
                    flush()
    flush()
    return load_sweep_results(output_dir)


def load_sweep_results(output_dir: str | Path) -> pd.DataFrame:
    """Read every Parquet part back into one DataFrame."""
    parts = sorted(Path(output_dir).glob("parts/part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    # A resumed sweep can re-emit a run whose part was written but whose
    # manifest line was lost; keep the last copy of each row. (run_id,
    # _row_index) identifies a row exactly, whatever columns the experiment
    # chose to emit.
    if {"run_id", "_row_index"} <= set(df.columns):
        df = df.drop_duplicates(subset=["run_id", "_row_index"], keep="last")
        df = df.reset_index(drop=True)
    return df


def failed_runs(output_dir: str | Path) -> list[dict[str, Any]]:
    """Failed runs with their tracebacks, for debugging after a sweep."""
    manifest = SweepManifest(Path(output_dir) / "manifest.jsonl")
    return [r for r in manifest._state.values() if r["status"] == RunStatus.FAILED.value]
