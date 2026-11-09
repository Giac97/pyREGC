# Running the Fe–H REGC example on a Slurm cluster

Production setup for `examples/13_fe_h_eam_surface.py` (H on Fe(110), REGC with
LAMMPS doing energies, forces and MD). Everything cluster-specific is in
`hpc/config.sh` and the `#SBATCH` headers marked `EDIT`.

## Files

| File | Purpose |
|---|---|
| `config.sh` | **edit this**: environment activation, potential, output directory, run length |
| `environment.yml` | conda/mamba environment (Python side; LAMMPS is built separately) |
| `build_lammps.sh` | builds the grace-runtime LAMMPS into the env as `liblammps_grace.so` |
| `check_env.py` | verifies Python packages and every LAMMPS style the run needs |
| `check.sbatch` | 30-min pre-flight: `check_env.py` + the example's `--check` (sanity checks, timing) |
| `run.sbatch` | the production run: resumes automatically, stops before the wall time, resubmits itself |
| `analyze.sbatch` | (re-)analysis of saved samples (MBAR, (T, p) phase diagram) |

## One-time setup

```bash
git clone <your repository> ~/gcmc_re && cd ~/gcmc_re      # or copy the directory
mamba env create -f hpc/environment.yml
conda activate gcmc_re
pip install -e .
bash hpc/build_lammps.sh           # only for GRACE-1L/2L/3L; see "LAMMPS" below
python hpc/check_env.py            # --grace: GRACE-FS; --lammps-name grace --grace-runtime
```

Then edit `hpc/config.sh` (`setup_environment`, `PROJECT_DIR`, `OUT`,
`POTENTIAL_ARGS`) and the `#SBATCH --partition/--account` lines of the three
`.sbatch` files.

### LAMMPS

`regc.LammpsCalculator` uses the LAMMPS **Python module** and a LAMMPS shared
library. The conda-forge `lammps` package from `environment.yml` is the default
(`liblammps.so`), with EAM, ACE (`pace`) and GRACE-FS (`grace/fs`).

For GRACE-1L/2L/3L models, `build_lammps.sh` builds the grace-runtime fork of
LAMMPS (https://github.com/yury-lysogorskiy/lammps, branch `grace-runtime`; see
https://grace-runtime.readthedocs.io/). It uses the env's compilers and
installs `liblammps_grace.so` and `lmp_grace` into the env, without touching
the default build:

```bash
bash hpc/build_lammps.sh                     # ~10 min; needs network at configure time
python hpc/check_env.py --lammps-name grace --grace-runtime
export REGC_LAMMPS_NAME=grace                # in setup_environment(), for the jobs
```

Settings: `LAMMPS_SRC` (default `~/src/lammps-grace`), `MACHINE` (default
`grace`), `ARCH_FLAGS` (default `-march=native`: build on the CPU type you
run on, e.g. a compute node), `NPROC`. A cluster without conda-forge's LAMMPS
can use `MACHINE= INSTALL_PYTHON=1` to make this build the env's only one.

Required styles:

- `pair_style eam/fs` / `eam/alloy` (package MANYBODY) for EAM potentials;
- `pair_style grace/fs` for GRACE-FS models (ML-PACE; both builds have it);
- `pair_style grace/runtime` for GRACE-1L/2L/3L models exported as
  `kokkos.npz` (grace-runtime build only); `pace/runtime` for pacemaker ACE;
- `fix nve`, `fix temp/csvr` (Bussi thermostat), `fix wall/reflect`, `compute temp`.

`check_env.py` lists what is present (`--lammps-name NAME` for a named build).
One process can load only one LAMMPS build. The engine runs serially per replica
(no MPI inside LAMMPS): parallelism is one process per replica.

## Workflow

Submit everything from the repository root.

```bash
sbatch hpc/check.sbatch        # 1. pre-flight; read its output before going on
sbatch hpc/run.sbatch          # 2. production; resubmits itself until finished
sbatch hpc/analyze.sbatch      # 3. optional: re-analyze (DISCARD=0.3 sbatch ... to change burn-in)
```

What to look for in the `check` output:

- **Lattice constant**: the potential's bcc Fe value (exp. 2.8665 Å).
- **H₂ from the potential**: r₀ and D₀ vs experiment (0.741 Å, 4.478 eV). If far off,
  set `EXTRA_ARGS=(--d0 experimental)` in `config.sh`.
- **Isolated H energy** `e_atom`: the zero of the GCMC energy scale (0 for most EAMs
  and for GRACE foundation models, but it is computed, not assumed).
- **H binding scans**: adsorption energies vs ½H₂ at a few surface sites; minima
  flagged `[AT SCAN EDGE]` mean H prefers to go deeper/higher than scanned.
- **Timing**: ms per REGC step per replica and the projected wall time.

### Resources

- **One node, one task, one worker process per replica**: `--cpus-per-task` = number
  of replicas (default grid 6 T × 7 p = 42), at most the cores per node. Fewer cores
  also work (replicas are distributed over the workers); more do not help.
- Threading is pinned to 1 per process in `config.sh` (`OMP_NUM_THREADS=1`, ...).
- Each worker loads the potential once (LAMMPS instances are cached per process),
  so even a large GRACE model is not reloaded every step.
- Multi-node runs are not supported (the replica pool is a single-node process pool).

### Restarts and wall time

`run.sbatch` always passes `--resume`: a fresh start if `$OUT` has no run, otherwise
it continues the saved one **bit-identically** (replica configurations, all random
streams and the samples are in `$OUT/checkpoint.pkl`, written atomically after every
`CHUNK` steps). It reads the remaining wall time from `squeue`, stops cleanly
~10 min before it (exit code 75) and resubmits itself (at most `MAX_RESUBMITS`
times). The run ends when `STEPS` are done or ρ(u) has converged (the paper's
criterion), and the analysis runs at the end of the last job.

To start over in the same `$OUT`, delete it (or add `--overwrite` by hand).

### Outputs (`$OUT`)

| File | Content |
|---|---|
| `run_metadata.json` | potential, lattice constant, H₂ reference (D₀, e_atom), grid, protocol (checked on resume) |
| `checkpoint.pkl` | full restart state (`ReplicaExchangeGCMC.save_checkpoint`) |
| `samples.npz` | recorded (U, N) per replica (input to MBAR) |
| `phase_diagram.npz` | coverage and most stable N_H on a fine (T, p) mesh |
| `fe110_h_Tp_phase_diagram.png` | the plots |

## Tested

The scripts were exercised end to end with a mocked Slurm (`squeue`/`sbatch` stubs)
and a synthetic toy Fe–H EAM: fresh run finishing in one job; a job stopped for the
wall time, resubmitted and resumed to the same final state as the uninterrupted run;
and the pre-flight job. Not tested on a real cluster or with `grace/fs` (not
available where this was written).
