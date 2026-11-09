#!/usr/bin/env bash
# Build LAMMPS with grace-runtime (https://grace-runtime.readthedocs.io/) as a shared
# library for regc's LammpsCalculator, next to the env's default LAMMPS.
#
# Run inside the activated regc environment (it needs network access: the configure
# step downloads grace-runtime and the ML-PACE library):
#
#     bash hpc/build_lammps.sh
#
# The result is liblammps_grace.so (and the executable lmp_grace) in $CONDA_PREFIX,
# selected in regc with LammpsCalculator(..., lammps_name="grace"), the config key
# calculator.lammps_name: grace, or REGC_LAMMPS_NAME=grace. The env's own LAMMPS
# (liblammps.so from conda-forge) stays the default and is untouched.
#
# Styles provided (hpc/check_env.py --lammps-name grace --grace-runtime verifies them):
#   pair  grace/runtime  GRACE-1L/2L/3L models exported as kokkos.npz (CPU, OpenMP)
#   pair  pace/runtime   PACE (pacemaker .yaml) through grace-runtime
#   pair  grace/fs, pace (ML-PACE), eam/fs, eam/alloy (MANYBODY)
#   fix   temp/csvr, wall/reflect, nve; compute temp (what regc's MD uses)
#
# Settings (environment variables):
#   LAMMPS_SRC      LAMMPS checkout; cloned from LAMMPS_REPO/LAMMPS_BRANCH if absent
#                   (default ~/src/lammps-grace)
#   MACHINE         library suffix: liblammps_$MACHINE.so (default grace). MACHINE= (empty)
#                   builds a plain liblammps.so, for an env without another LAMMPS.
#   INSTALL_PYTHON  1: also install this LAMMPS's Python module into the env (replaces
#                   conda-forge's; only for an env whose sole LAMMPS this is). Default 0.
#   ARCH_FLAGS      compiler flags (default -march=native: run on the CPU type you build
#                   on; on a cluster build on a compute node, or use e.g. -march=x86-64-v3)
#   NPROC           parallel compile jobs (default: all cores)
set -euo pipefail

: "${CONDA_PREFIX:?activate the regc environment first}"
LAMMPS_REPO="${LAMMPS_REPO:-https://github.com/yury-lysogorskiy/lammps.git}"
LAMMPS_BRANCH="${LAMMPS_BRANCH:-grace-runtime}"
LAMMPS_SRC="${LAMMPS_SRC:-$HOME/src/lammps-grace}"
MACHINE="${MACHINE-grace}"
BUILD_DIR="${BUILD_DIR:-$LAMMPS_SRC/build-${MACHINE:-default}}"
INSTALL_PYTHON="${INSTALL_PYTHON:-0}"
ARCH_FLAGS="${ARCH_FLAGS:--march=native}"
NPROC="${NPROC:-$(nproc)}"

if [ ! -d "$LAMMPS_SRC/cmake" ]; then
    git clone --depth 1 -b "$LAMMPS_BRANCH" "$LAMMPS_REPO" "$LAMMPS_SRC"
fi

# The env's compilers (conda-forge cxx-compiler) if present, so the library links
# against the env's libstdc++/libgomp, which the Python process loads anyway.
CXX="${CXX:-$(command -v x86_64-conda-linux-gnu-g++ || command -v g++)}"
CC="${CC:-$(command -v x86_64-conda-linux-gnu-gcc || command -v gcc)}"

cmake -S "$LAMMPS_SRC/cmake" -B "$BUILD_DIR" \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_CXX_COMPILER="$CXX" -D CMAKE_C_COMPILER="$CC" \
    -D CMAKE_CXX_FLAGS="$ARCH_FLAGS" \
    -D CMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$CONDA_PREFIX/lib" \
    -D CMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath-link,$CONDA_PREFIX/lib" \
    -D CMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -D CMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -D BUILD_SHARED_LIBS=yes \
    -D LAMMPS_MACHINE="$MACHINE" \
    -D BUILD_MPI=no \
    -D BUILD_OMP=yes \
    -D PKG_MANYBODY=yes \
    -D PKG_EXTRA-FIX=yes \
    -D PKG_ML-PACE=yes
cmake --build "$BUILD_DIR" -j "$NPROC"

# Install only the library and the executable: a full `cmake --install` would also
# overwrite the headers and potential files of the env's conda-forge LAMMPS.
suffix="${MACHINE:+_$MACHINE}"
cp -av "$BUILD_DIR"/liblammps"$suffix".so* "$CONDA_PREFIX/lib/"
cp -av "$BUILD_DIR/lmp$suffix" "$CONDA_PREFIX/bin/"
if [ "$INSTALL_PYTHON" = 1 ]; then
    cmake --build "$BUILD_DIR" --target install-python
fi

python "$(dirname "$0")/check_env.py" ${MACHINE:+--lammps-name "$MACHINE"} --grace-runtime
