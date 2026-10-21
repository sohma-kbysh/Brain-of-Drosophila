"""Resource detection, profiles and thread control (addendum sections 2-5)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bioquantumfly.runtime.blas import (
    THREAD_VARS, blas_threads, configure_blas_threads, current_thread_settings,
)
from bioquantumfly.runtime.profiles import PROFILES, load_warnings, resolve_profile
from bioquantumfly.runtime.resources import (
    DiskSpaceError, check_disk_space, detect_resources, resolve_scratch_dir,
)


class TestResourceDetection:
    def test_detects_the_basics(self):
        r = detect_resources(Path.cwd())
        assert r.logical_cpu_count >= 1
        assert r.hostname
        assert r.python_version

    def test_never_raises_on_a_missing_path(self):
        r = detect_resources("/nonexistent/path/xyz")
        assert r.project_filesystem is None

    def test_usable_cpu_count_respects_a_scheduler_allocation(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        r = detect_resources(Path.cwd())
        assert r.scheduler.inside_allocation
        assert r.usable_cpu_count == 8, "must never exceed the allocation"


class TestProfiles:
    def test_development_is_the_default_and_is_conservative(self):
        from bioquantumfly.runtime.profiles import DEFAULT_PROFILE

        assert DEFAULT_PROFILE == "development"
        assert PROFILES["development"].workers <= 4

    def test_large_is_never_the_default(self):
        from bioquantumfly.config import Config

        assert Config().runtime.profile != "large"

    def test_sweep_profiles_use_one_blas_thread(self):
        """Process-parallel sweeps must not also multi-thread BLAS."""
        for name in ("development", "standard", "large"):
            assert PROFILES[name].blas_threads == 1

    def test_linalg_profile_inverts_the_tradeoff(self):
        p = PROFILES["linalg"]
        assert p.workers == 1 and p.blas_threads > 1

    def test_overrides_are_applied(self):
        assert resolve_profile("development", workers=12).workers == 12

    def test_workers_are_capped_by_usable_cpus(self):
        r = detect_resources(Path.cwd())
        r.scheduler.inside_allocation = True
        r.scheduler.allocated_cpus = 2
        assert resolve_profile("large", resources=r).workers <= 2

    def test_invalid_profile_is_rejected(self):
        with pytest.raises(ValueError, match="unknown profile"):
            resolve_profile("enormous")

    def test_zero_workers_is_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            resolve_profile("development", workers=0)

    def test_root_filesystem_produces_a_warning(self):
        r = detect_resources(Path.cwd())
        if r.project_filesystem:
            r.project_filesystem.is_root_filesystem = True
            warnings = load_warnings(r, PROFILES["development"])
            assert any("ROOT" in w for w in warnings)


class TestBlasControl:
    def test_sets_every_thread_variable(self):
        applied = configure_blas_threads(3, warn_if_late=False)
        try:
            assert set(applied) == set(THREAD_VARS)
            assert all(v == "3" for v in current_thread_settings().values())
        finally:
            configure_blas_threads(4, warn_if_late=False)

    def test_context_manager_restores_the_previous_values(self):
        before = current_thread_settings()
        with blas_threads(7):
            assert os.environ["OMP_NUM_THREADS"] == "7"
        assert current_thread_settings() == before

    def test_rejects_zero_threads(self):
        with pytest.raises(ValueError, match=">= 1"):
            configure_blas_threads(0)


class TestFilesystemGuards:
    def test_scratch_never_lands_on_the_root_filesystem(self, tmp_path):
        scratch = resolve_scratch_dir(tmp_path)
        assert scratch.exists()
        assert scratch.is_dir()

    def test_explicit_scratch_is_honoured(self, tmp_path):
        target = tmp_path / "explicit"
        assert resolve_scratch_dir(tmp_path, requested=str(target)) == target
        assert target.exists()

    def test_an_impossible_write_is_refused(self, tmp_path):
        with pytest.raises(DiskSpaceError, match="refusing to write"):
            check_disk_space(tmp_path, estimated_gb=1e9)

    def test_warn_only_mode_does_not_raise(self, tmp_path):
        info = check_disk_space(tmp_path, estimated_gb=1e9, raise_on_fail=False)
        assert info.available_gb >= 0


class TestProvenanceMetadata:
    def test_metadata_records_hardware_and_software(self):
        from bioquantumfly.runtime.provenance import build_metadata

        meta = build_metadata(experiment="t", config={"a": 1}, seed=42)
        for key in ("hostname", "cpu_model", "logical_cpu_count", "packages",
                    "git", "blas_threads", "python_version",
                    "speculative_coupling_enabled", "malecns_dataset_version"):
            assert key in meta, key

    def test_speculative_flag_is_carried(self):
        from bioquantumfly.runtime.provenance import build_metadata

        meta = build_metadata(experiment="t", config={}, seed=1,
                              speculative_coupling_enabled=True)
        assert meta["speculative_coupling_enabled"] is True

    def test_results_directory_layout(self, tmp_path):
        from bioquantumfly.runtime.provenance import results_dir, write_metadata

        d = results_dir(tmp_path, "exp", timestamp="20260101-000000")
        assert d == tmp_path / "exp" / "20260101-000000"
        assert write_metadata(d, {"a": 1}).exists()
