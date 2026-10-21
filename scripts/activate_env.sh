#!/usr/bin/env bash
# BioQuantumFly environment activation helper.
#
# Usage:   source scripts/activate_env.sh
#
# Activates the project Conda environment and exports conservative thread
# limits. This machine has 384 logical CPUs and is a shared research server,
# so we deliberately do NOT let BLAS/OpenMP grab every core.
#
# This script does not modify any shell startup file. It only affects the
# shell you source it from.

BQF_ENV_PREFIX="${BQF_ENV_PREFIX:-$HOME/conda-envs/bioquantumfly}"

if [ ! -d "$BQF_ENV_PREFIX" ]; then
    echo "ERROR: conda environment not found at $BQF_ENV_PREFIX" >&2
    return 1 2>/dev/null || exit 1
fi

# Prefer the real conda activation mechanism (sets CONDA_PREFIX, runs
# activate.d hooks). Fall back to PATH manipulation only if conda is absent.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda activate "$BQF_ENV_PREFIX" || {
        echo "ERROR: conda activate failed for $BQF_ENV_PREFIX" >&2
        return 1 2>/dev/null || exit 1
    }
else
    echo "WARNING: 'conda' not on PATH; falling back to PATH prepend." >&2
    export PATH="$BQF_ENV_PREFIX/bin:$PATH"
    export CONDA_PREFIX="$BQF_ENV_PREFIX"
fi

# --- Conservative CPU thread policy (shared server) ---
# Capture the true CPU count BEFORE exporting OMP_NUM_THREADS: nproc honours
# that variable and would otherwise just echo our own limit back at us.
BQF_TOTAL_CPUS="$(nproc --all 2>/dev/null || nproc)"
BQF_THREADS="${BQF_THREADS:-16}"
export OMP_NUM_THREADS="$BQF_THREADS"
export OPENBLAS_NUM_THREADS="$BQF_THREADS"
export MKL_NUM_THREADS="$BQF_THREADS"
export NUMEXPR_NUM_THREADS="$BQF_THREADS"
export VECLIB_MAXIMUM_THREADS="$BQF_THREADS"

# --- Keep caches and scratch off the (nearly full) root filesystem ---
# Resolve this script's location under both bash and zsh (the user's shell).
if [ -n "${BASH_SOURCE:-}" ]; then
    _bqf_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _bqf_self="${(%):-%x}"
else
    _bqf_self="$0"
fi
BQF_ROOT="$(cd "$(dirname "$_bqf_self")/.." && pwd)"
unset _bqf_self
export BQF_ROOT
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"

# Override TMPDIR unconditionally. The default (/tmp/user/$UID) is backed by
# the root filesystem, which has only ~52 GB free; a large intermediate would
# fill it. Use BQF_TMPDIR to point somewhere else -- but keep it off "/".
export TMPDIR="${BQF_TMPDIR:-$BQF_ROOT/data/cache/tmp}"
mkdir -p "$TMPDIR"

echo "BioQuantumFly environment active"
echo "  prefix : $BQF_ENV_PREFIX"
echo "  python : $(python --version 2>&1)  ($(command -v python))"
echo "  threads: $BQF_THREADS / $BQF_TOTAL_CPUS logical CPUs"
echo "  TMPDIR : $TMPDIR"
if [ -n "${NEUPRINT_TOKEN:-}" ]; then
    echo "  neuPrint token: present"
else
    echo "  neuPrint token: NOT set (export NEUPRINT_TOKEN=... when needed)"
fi
