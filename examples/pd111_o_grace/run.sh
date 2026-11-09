#!/usr/bin/env bash
# O on Pd(111) with a GRACE model through LAMMPS grace/runtime (pd111_o_grace.yaml).
#
#   ./run.sh check      setup only: lattice constant, the model's O2 and D0, the mu ladder
#   ./run.sh run        sampling + phase diagram
#   ./run.sh resume     continue the checkpointed run in output_pd111_o_grace/
#   ./run.sh analyze    phase diagram from the saved samples
#
# Run `check` first and look at the printed D0 (experiment: 5.12 eV) and mu range.
#
# Environment variables:
#   GRACE_MODEL   registry name (default GRACE-2L-OMAT-medium-ft-E) or a path to a
#                 kokkos.npz; linked here as model.npz, the file the config names
#   WORKERS       worker processes (default: the config's n_workers, i.e. all cores)
#   TIME_LIMIT    seconds: stop cleanly before this (exit code 75: resume later)
#
# Needs the activated regc environment with the grace-runtime LAMMPS
# (bash hpc/build_lammps.sh); to download models, the grace-runtime package
# (pip install grace-runtime).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG=pd111_o_grace.yaml
MODEL="${GRACE_MODEL:-GRACE-2L-OMAT-medium-ft-E}"
[ -f "$MODEL" ] && MODEL="$(realpath "$MODEL")"   # a path given relative to the caller's directory
MODE="${1:-}"

# One replica per process: no hidden thread pools.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND="${MPLBACKEND:-Agg}"

cd "$HERE"

link_model() {
    local src
    if [ -f "$MODEL" ]; then
        src="$MODEL"
    else
        src="${GRACE_CACHE:-$HOME/.cache/grace}/$MODEL/kokkos.npz"
        if [ ! -f "$src" ]; then
            if ! command -v grace-runtime >/dev/null; then
                echo "No $src. Install grace-runtime (pip install grace-runtime) to download" \
                     "it, or set GRACE_MODEL=/path/to/kokkos.npz." >&2
                exit 1
            fi
            grace-runtime models download "$MODEL"
        fi
    fi
    ln -sfn "$(realpath "$src")" model.npz
    echo "model.npz -> $(readlink model.npz)"
}

check_lammps() {
    if ! python "$REPO/hpc/check_env.py" --lammps-name grace --grace-runtime >/dev/null 2>&1; then
        python "$REPO/hpc/check_env.py" --lammps-name grace --grace-runtime || true
        echo "The grace-runtime LAMMPS is missing or incomplete: bash $REPO/hpc/build_lammps.sh" >&2
        exit 1
    fi
}

workers=(${WORKERS:+--workers "$WORKERS"})
case "$MODE" in
    check)   check_lammps; link_model; regc-run "$CONFIG" --check ;;
    run)     check_lammps; link_model
             regc-run "$CONFIG" "${workers[@]}" ${TIME_LIMIT:+--time-limit "$TIME_LIMIT"} ;;
    resume)  check_lammps; link_model
             regc-run "$CONFIG" --resume "${workers[@]}" ${TIME_LIMIT:+--time-limit "$TIME_LIMIT"} ;;
    analyze) regc-run "$CONFIG" --analyze-only ;;
    *)       sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
