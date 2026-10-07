#!/bin/bash
# Production REGC run of examples/13_fe_h_eam_surface.py on one node.
# Submit from the repository root:   sbatch hpc/run.sbatch
#
# The job always continues the run saved in $OUT (--resume; a fresh start if there
# is none), stops cleanly ~10 min before its wall time, and resubmits itself until
# the run finishes (all steps, or rho(u) converged); then it runs the analysis.
#
#SBATCH --job-name=fe110_h_regc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=42      # EDIT: one worker per replica (6 T x 7 p = 42); <= cores per node
#SBATCH --time=24:00:00         # EDIT: within your partition's limit
#SBATCH --partition=CHANGE_ME   # EDIT
##SBATCH --account=CHANGE_ME    # EDIT (uncomment if your cluster needs it)
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/config.sh
setup_environment
mkdir -p "$OUT"

# remaining wall time (squeue %L: [D-]HH:MM:SS, MM:SS or UNLIMITED) -> seconds
remaining_seconds() {
    local t d=0 h=0 m=0 s=0
    t=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)
    [[ -z "$t" || "$t" == UNLIMITED || "$t" == INVALID ]] && { echo ""; return; }
    [[ "$t" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r -a p <<< "$t"
    case ${#p[@]} in
        3) h=${p[0]}; m=${p[1]}; s=${p[2]} ;;
        2) m=${p[0]}; s=${p[1]} ;;
        1) s=${p[0]} ;;
    esac
    echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

LIMIT_ARGS=()
REM=$(remaining_seconds)
if [[ -n "$REM" ]]; then
    LIMIT_ARGS=(--time-limit $(( REM - 600 )))   # 10 min margin for the last checkpoint + analysis
fi

echo "job $SLURM_JOB_ID on $(hostname): $SLURM_CPUS_PER_TASK workers, resubmission ${RESUBMIT_COUNT:-0}, remaining ${REM:-?} s"
set +e
python -u "$SCRIPT" --run --resume "${POTENTIAL_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    --steps "$STEPS" --chunk "$CHUNK" --workers "$SLURM_CPUS_PER_TASK" --out "$OUT" \
    "${LIMIT_ARGS[@]}"
rc=$?
set -e

if [[ $rc -eq 75 ]]; then
    n=$(( ${RESUBMIT_COUNT:-0} + 1 ))
    if (( n > MAX_RESUBMITS )); then
        echo "stopped for the wall time, but MAX_RESUBMITS=$MAX_RESUBMITS reached; resubmit by hand"
        exit 0
    fi
    echo "stopped for the wall time; resubmitting (#$n)"
    sbatch --export=ALL,RESUBMIT_COUNT=$n hpc/run.sbatch
    exit 0
fi
exit $rc
