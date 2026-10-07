#!/bin/bash
# Short pre-flight job: environment check + the example's --check (potential sanity
# checks, H2 reference, H binding scans, mu grid, timing and projected cost).
# Submit from the repository root:   sbatch hpc/check.sbatch
#
#SBATCH --job-name=fe110_h_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=CHANGE_ME   # EDIT (a short/dev partition is ideal)
##SBATCH --account=CHANGE_ME    # EDIT
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/config.sh
setup_environment

GRACE_FLAG=()
[[ " ${POTENTIAL_ARGS[*]} " == *" grace/fs "* ]] && GRACE_FLAG=(--grace)
python hpc/check_env.py "${GRACE_FLAG[@]}"

# --workers only sets the projection in the timing line: use run.sbatch's --cpus-per-task
python -u "$SCRIPT" --check "${POTENTIAL_ARGS[@]}" "${EXTRA_ARGS[@]}" --steps "$STEPS" --workers 42
