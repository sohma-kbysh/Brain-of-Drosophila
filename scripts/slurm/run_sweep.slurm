#!/bin/bash
#SBATCH --job-name=bqf-sweep
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#
# BioQuantumFly parameter sweep under SLURM.
#
# NOT submitted automatically. Review the resource request, then:
#     sbatch scripts/slurm/run_sweep.slurm
#
# The code reads SLURM_CPUS_PER_TASK and will never exceed the allocation,
# so --cpus-per-task above is the single place to change the worker count.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/activate_env.sh

# One BLAS thread per worker: with 32 processes, 16 BLAS threads each would
# put ~512 runnable threads on the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p results/slurm

echo "job ${SLURM_JOB_ID:-none} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
bioquantumfly check-resources

# --resume makes the sweep restartable: requeue this script and completed runs
# are skipped rather than repeated.
bioquantumfly sweep \
    --config configs/sweeps.yaml \
    --experiment feasibility_map \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --resume
