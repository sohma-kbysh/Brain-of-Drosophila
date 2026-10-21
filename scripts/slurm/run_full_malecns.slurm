#!/bin/bash
#SBATCH --job-name=bqf-malecns
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#
# Tier 3: whole-MaleCNS neural simulation (166,691 neurons).
#
# PREREQUISITES -- this script does not fetch anything:
#   1. NEUPRINT_TOKEN exported in the submitting environment
#   2. the connectome already preprocessed:
#        bioquantumfly fetch-subgraph --seed-type <type> --hops 3
#        bioquantumfly preprocess-malecns --input-dir data/cache/neuprint/<hash>
#   3. connectome.source set to 'processed' in the config
#
# Read addendum section 27 before running: Tier 3 is the LAST step, after
# Tiers 0-2 are validated and benchmarked. Sufficient RAM is not a reason to
# jump straight to a whole-CNS hybrid run.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/activate_env.sh

# Whole-CNS work is one large Brian2 simulation rather than many small ones, so
# this is the profile where multi-threaded BLAS is worth benchmarking.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p results/slurm

if [ ! -f data/processed/male_cns/metadata.json ]; then
    echo "ERROR: data/processed/male_cns is missing. Run fetch-subgraph and" >&2
    echo "       preprocess-malecns first; this script does not download." >&2
    exit 1
fi

bioquantumfly check-resources
bioquantumfly simulate-neural --config configs/male_cns_subgraph.yaml --workers 1
