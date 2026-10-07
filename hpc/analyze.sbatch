#!/bin/bash
# (Re-)analyze the samples saved in $OUT: MBAR, (T, p) phase diagram, plots.
# run.sbatch already does this at the end; use this to analyze a running or
# finished run again (e.g. with another --discard).
# Submit from the repository root:   sbatch hpc/analyze.sbatch
#
#SBATCH --job-name=fe110_h_analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=CHANGE_ME   # EDIT
##SBATCH --account=CHANGE_ME    # EDIT
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/config.sh
setup_environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -u "$SCRIPT" --analyze-only --out "$OUT" --discard "${DISCARD:-0.2}"
