# Shared settings for the Slurm jobs in this directory (sourced by check/run/analyze.sbatch).
# Edit the lines marked EDIT; nothing else should need changing.

# ---- environment -------------------------------------------------------------------
# EDIT: modules and Python environment of your cluster (example for a conda/mamba env).
setup_environment() {
    # module purge
    # module load compiler/gnu mpi/openmpi          # whatever your LAMMPS was built with
    # source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate gcmc_re
}

# ---- paths -----------------------------------------------------------------------------
# EDIT: where the gcmc_re checkout lives, and where results go. On clusters with
# workspaces/scratch, put OUT there (large checkpoint.pkl files, many writes).
PROJECT_DIR="$HOME/gcmc_re"
OUT="$PROJECT_DIR/runs/fe110_h"

# ---- potential (EDIT: one of the two) -----------------------------------------------
# (a) an Fe-H EAM file (.eam.fs / .eam.alloy), e.g. the Wen (2021) EAM shipped in examples/:
POTENTIAL_ARGS=(--potential "$PROJECT_DIR/examples/FeH_Wen.eam.fs")
# (b) a GRACE-FS model through LAMMPS's grace/fs (needs a GRACE-enabled LAMMPS):
# POTENTIAL_ARGS=(--pair-style grace/fs --pair-coeff "* * $PROJECT_DIR/potentials/model.yaml Fe H"
#                 --min-distance 0.6)

# ---- run -------------------------------------------------------------------------------
STEPS=20000                   # total REGC steps (the run may stop earlier on rho(u) convergence)
CHUNK=500                     # steps between checkpoints (checkpoint.pkl + samples.npz)
EXTRA_ARGS=(--d0 self)        # e.g. (--d0 experimental) or (--diffusion hmc)
MAX_RESUBMITS=20              # safety cap on automatic resubmission of run.sbatch

# ---- threading: one replica per process, no hidden thread pools --------------------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export MPLBACKEND=Agg         # headless plotting

SCRIPT="$PROJECT_DIR/examples/13_fe_h_eam_surface.py"
