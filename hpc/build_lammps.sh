#!/usr/bin/env bash
# Build LAMMPS as a shared library + Python module for gcmc_re's LammpsCalculator.
#
# Run inside the activated gcmc_re environment, on a node with the compilers you
# want (login node is usually fine):    bash hpc/build_lammps.sh
#
# Styles gcmc_re needs (hpc/check_env.py verifies them after the build):
#   pair  eam/fs, eam/alloy           -> package MANYBODY
#   pair  grace/fs (GRACE-FS models)  -> ML-PACE with GRACE support: follow the GRACE /
#                                        tensorpotential documentation for the LAMMPS
#                                        source/version and options that provide it
#   fix   temp/csvr, wall/reflect, nve; compute temp
set -euo pipefail

LAMMPS_SRC="${LAMMPS_SRC:-$HOME/src/lammps}"   # EDIT: a LAMMPS checkout that provides grace/fs
BUILD_DIR="${BUILD_DIR:-$LAMMPS_SRC/build-gcmc}"
NPROC="${NPROC:-8}"

if [ ! -d "$LAMMPS_SRC/cmake" ]; then
    echo "No LAMMPS source at $LAMMPS_SRC (set LAMMPS_SRC=...)" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake "$LAMMPS_SRC/cmake" \
    -D CMAKE_BUILD_TYPE=Release \
    -D BUILD_SHARED_LIBS=yes \
    -D LAMMPS_EXCEPTIONS=yes \
    -D BUILD_MPI=no \
    -D BUILD_OMP=no \
    -D PKG_MANYBODY=yes \
    -D PKG_EXTRA-FIX=yes \
    -D PKG_ML-PACE=yes \
    -D CMAKE_INSTALL_PREFIX="$CONDA_PREFIX"
    # add the GRACE-specific options here if the GRACE docs require any
make -j "$NPROC"
make install-python          # installs the `lammps` Python module into the active env

python "$(dirname "$0")/check_env.py"
