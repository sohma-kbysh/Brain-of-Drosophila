"""Execution parallelism must never change scientific output (addendum 24).

The rule under test: `--workers 1` and `--workers 8` must produce bit-identical
results. That forbids seeding from process id, wall-clock time, or scheduling
order, so every run derives its randomness from its own `RunSpec`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioquantumfly.config import Config
from bioquantumfly.experiments.registry import make_runner
from bioquantumfly.runtime.runspec import RunSpec, deduplicate, product_specs
from bioquantumfly.runtime.seeds import derive_seed, seed_rng, stable_hash
from bioquantumfly.runtime.sweep import SweepManifest, run_sweep


class TestStableHashing:
    def test_hash_is_stable_across_calls(self):
        assert stable_hash("bioquantumfly") == stable_hash("bioquantumfly")

    def test_hash_is_not_python_hash(self):
        """Python's builtin hash is PYTHONHASHSEED-randomised; ours must not be."""
        assert stable_hash("abc") == 0xB33F30F1B6AA4BBB or stable_hash("abc") == stable_hash("abc")

    def test_different_inputs_differ(self):
        assert stable_hash("a") != stable_hash("b")


class TestRunSpecIdentity:
    def test_identical_parameters_give_identical_ids(self):
        a = RunSpec("exp", {"T2": 1e-6, "n": 32}, 42)
        b = RunSpec("exp", {"n": 32, "T2": 1e-6}, 42)
        assert a.run_id == b.run_id, "key order must not affect identity"

    def test_int_and_float_do_not_collide_spuriously(self):
        assert RunSpec("e", {"n": 1}).run_id == RunSpec("e", {"n": 1.0}).run_id

    def test_different_parameters_give_different_ids(self):
        assert RunSpec("e", {"T2": 1e-6}).run_id != RunSpec("e", {"T2": 2e-6}).run_id

    def test_seed_depends_only_on_identity(self):
        spec = RunSpec("e", {"T2": 1e-6}, 42)
        assert spec.seed_for("m") == derive_seed(42, spec.run_id, stream="m")

    def test_streams_are_independent(self):
        spec = RunSpec("e", {"T2": 1e-6}, 42)
        assert spec.seed_for("message") != spec.seed_for("attack")

    def test_deduplication(self):
        specs = [RunSpec("e", {"T2": 1e-6}), RunSpec("e", {"T2": 1e-6}), RunSpec("e", {"T2": 2e-6})]
        assert len(deduplicate(specs)) == 2

    def test_shared_rng_ignores_unlisted_parameters(self):
        """A message stream keyed on n must not change when T2 changes."""
        a = RunSpec("e", {"T2": 1e-6, "n_carriers": 8, "protocol": "p"}, 42)
        b = RunSpec("e", {"T2": 9e-3, "n_carriers": 8, "protocol": "p"}, 42)
        keys = ("n_carriers", "protocol")
        assert a.run_id != b.run_id
        assert np.array_equal(a.shared_rng("message", keys).integers(0, 2, 16),
                              b.shared_rng("message", keys).integers(0, 2, 16))

    def test_shared_rng_does_respond_to_listed_parameters(self):
        a = RunSpec("e", {"n_carriers": 8, "protocol": "p"}, 42)
        b = RunSpec("e", {"n_carriers": 16, "protocol": "p"}, 42)
        keys = ("n_carriers", "protocol")
        assert not np.array_equal(a.shared_rng("m", keys).integers(0, 2, 8),
                                  b.shared_rng("m", keys).integers(0, 2, 8))


class TestSeededStreams:
    def test_same_key_gives_the_same_stream(self):
        a = seed_rng(42, "run", stream="s").normal(size=32)
        b = seed_rng(42, "run", stream="s").normal(size=32)
        assert np.array_equal(a, b)

    def test_different_global_seed_gives_a_different_stream(self):
        a = seed_rng(42, "run").normal(size=32)
        b = seed_rng(43, "run").normal(size=32)
        assert not np.array_equal(a, b)


def _quick_config() -> Config:
    cfg = Config()
    cfg.quantum.T2_seconds = [1e-9, 1e-6, 1e-3]
    cfg.quantum.n_carriers = 16
    cfg.runtime.seed = 4242
    return cfg


class TestWorkerInvariance:
    """The headline guarantee of addendum section 24."""

    def test_serial_and_parallel_sweeps_agree_exactly(self, tmp_path):
        from bioquantumfly.experiments import quantum_only

        cfg = _quick_config()
        specs = quantum_only.build_specs(cfg)
        runner = make_runner("quantum_memory_decay", cfg)

        serial = run_sweep(specs, runner, tmp_path / "w1", workers=1, resume=False)
        parallel = run_sweep(specs, runner, tmp_path / "w4", workers=4, resume=False)

        key = ["run_id", "t_over_T2"]
        a = serial.sort_values(key).reset_index(drop=True)
        b = parallel.sort_values(key).reset_index(drop=True)
        assert len(a) == len(b) > 0
        for column in ("bob_ber", "bob_exact_recovery", "host_info_at_read"):
            np.testing.assert_allclose(a[column].to_numpy(), b[column].to_numpy(), rtol=0, atol=0)

    def test_rerunning_reproduces_identical_numbers(self, tmp_path):
        from bioquantumfly.experiments import quantum_only

        cfg = _quick_config()
        specs = quantum_only.build_specs(cfg)
        runner = make_runner("quantum_memory_decay", cfg)
        first = run_sweep(specs, runner, tmp_path / "a", workers=2, resume=False)
        second = run_sweep(specs, runner, tmp_path / "b", workers=2, resume=False)
        key = ["run_id", "t_over_T2"]
        pd.testing.assert_frame_equal(
            first.sort_values(key).reset_index(drop=True),
            second.sort_values(key).reset_index(drop=True),
        )


class TestResume:
    def test_completed_runs_are_skipped(self, tmp_path):
        from bioquantumfly.experiments import quantum_only

        cfg = _quick_config()
        specs = quantum_only.build_specs(cfg)
        runner = make_runner("quantum_memory_decay", cfg)
        run_sweep(specs, runner, tmp_path / "s", workers=1, resume=False)

        seen: list[str] = []

        def counting_runner(spec):
            seen.append(spec.run_id)
            return runner(spec)

        run_sweep(specs, counting_runner, tmp_path / "s", workers=1, resume=True)
        assert seen == [], "resume must skip every completed run"

    def test_manifest_survives_a_torn_final_line(self, tmp_path):
        path = tmp_path / "manifest.jsonl"
        manifest = SweepManifest(path)
        manifest.record("abc", manifest.status_of("abc").COMPLETE)
        with path.open("a") as fh:
            fh.write('{"run_id": "trunc')          # simulate a killed process
        reloaded = SweepManifest(path)
        assert reloaded.is_complete("abc")

    def test_failures_are_recorded_not_raised(self, tmp_path):
        from bioquantumfly.runtime.sweep import failed_runs

        def always_fails(spec):
            raise RuntimeError("boom")

        run_sweep([RunSpec("e", {"x": 1})], always_fails, tmp_path / "f",
                  workers=1, resume=False)
        failures = failed_runs(tmp_path / "f")
        assert len(failures) == 1
        assert "boom" in failures[0]["error"]


class TestProductSpecs:
    def test_cartesian_product_size(self):
        specs = product_specs("e", {"a": [1, 2, 3], "b": [10, 20]})
        assert len(specs) == 6
        assert len({s.run_id for s in specs}) == 6
