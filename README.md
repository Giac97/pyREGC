# gcmc_re

Multidimensional replica-exchange grand canonical Monte Carlo: a 2D ladder
of replicas over (temperature, chemical potential), a pluggable energy
backend (fast pairwise Lennard-Jones, or *any* ASE-compatible potential --
EAM, MACE, NequIP, Allegro, ...), optional process-pool parallelism across
replicas, and MBAR-based reweighting into smooth coverage(T, mu)/(T, p)
surface phase diagrams.

## Install

### Everything, with mamba (recommended)

```bash
mamba env create -f environment.yml     # from the repository root
mamba activate gcmc_re
pytest                                  # optional: the test suite (~1 min)
```

This installs Python 3.11 with numpy, scipy, ASE, pymbar, matplotlib, PyYAML,
pytest and **LAMMPS from conda-forge**, a serial CPU build with the Python
module that `LammpsCalculator` uses. It also installs this package in
editable mode (`pip install -e .`), so edits to `gcmc_re/` take effect
immediately. After changing `environment.yml`, run
`mamba env update -f environment.yml --prune`.

conda-forge's LAMMPS has no `pair_style grace/fs`. For GRACE-FS models in
LAMMPS, build LAMMPS yourself (`hpc/build_lammps.sh`, see `hpc/README.md`).
GRACE's own ASE calculator (`tensorpotential`, with TensorFlow) is an
optional extra, commented out in `environment.yml`.

### With pip

```bash
pip install .                 # core only (numpy)
pip install ".[all]"          # + ase, pymbar/scipy, matplotlib, pyyaml
pip install -e ".[dev]"       # editable, + pytest and build
```

Extras: `ase` (EAM/EMT/MLIP calculators, structures), `mbar` (pymbar +
scipy: phase diagrams), `plot` (matplotlib), `yaml` (YAML configs), `all`,
`dev`. LAMMPS is not a pip extra: install it from conda-forge or build it.

The core (engine, LJ calculators, geometries, units, reservoirs, driver,
sample I/O) needs only numpy. Every optional dependency is imported lazily,
inside the function that needs it, so `import gcmc_re` works with numpy
alone.

### Command line

The install also provides `gcmc-re-run config.yaml`: the config-driven
pipeline (structure file -> REGC -> MBAR -> (T, p) phase diagram), see
`examples/pd111_emt_example.yaml`.

## Layout

```
gcmc_re/
├── calculators.py     Calculator, LJCalculator, PairLJCalculator, ASECalculatorAdapter
├── lammps_backend.py  LammpsCalculator (any LAMMPS pair style, native MD),
│                      eam_pair_commands, lammps_ase_calculator
├── systems.py         ParticleSystem (mobile/substrate/frozen), BulkSystem, SlabSystem
│                      (historical aliases: LJSystem, SlabLJSystem)
├── state.py           ThermoState
├── replica.py         Replica (one grid point: GCMC moves, MC/HMC/MD diffusion)
├── engine.py          ReplicaExchangeGCMC (grid, swaps incl. diagonals, run/run_regc,
│                      parallelism, checkpoints)
├── driver.py          run_with_checkpoints, RunStatus (chunked, resumable, wall-time aware)
├── io.py              save_samples / load_samples -> SampleSet; atomic pickling
├── units.py           kelvin_to_eV, eV_to_kelvin, ThermalWavelengthCubed, pressure units
├── pressure.py        simple (reduced-units) mu(p, T) <-> p conventions
├── reservoir.py       MonatomicIdealGasReservoir, DiatomicReservoir (e.g. H2), low-level
│                      diatomic functions (self-consistent D0 and isolated-atom energy)
├── structures.py      cubic_lattice_constant, slab_geometry_from_atoms, SlabGeometry,
│                      insertion_energy_scan
├── analysis.py        MBARSurfaceAnalysis (MBAR phase diagrams, lowest-energy maps)
├── plotting.py        plot_phase_diagram
├── trajectory.py      extended-XYZ trajectory snapshots
└── runner.py          config-file-driven pipeline (structure file -> phase diagram)

docs/         REGC_algorithm.md/.pdf: the algorithm with every acceptance rule
examples/     runnable scripts, cheapest first (+ examples/data/ for shipped structure files)
hpc/          Slurm setup for production runs
tests/        pytest checks, incl. NumPy-docstring validation of the package
```

## Typical real-units workflow

```python
from ase.build import bcc110
from gcmc_re import (DiatomicReservoir, LammpsCalculator, ReplicaExchangeGCMC,
                     cubic_lattice_constant, eam_pair_commands, kelvin_to_eV,
                     lammps_ase_calculator, run_with_checkpoints, slab_geometry_from_atoms,
                     load_samples, MBARSurfaceAnalysis)

pair = eam_pair_commands("Fe-H.eam.fs", ["Fe", "H"])
ase_calc = lammps_ase_calculator(pair, ["Fe", "H"])           # same potential, for setup scans
a = cubic_lattice_constant(ase_calc, "Fe", "bcc")
geometry = slab_geometry_from_atoms(bcc110("Fe", size=(3, 4, 6), a=a, vacuum=None, orthogonal=True),
                                    n_frozen_layers=3, headroom=5.0)
h2 = DiatomicReservoir.from_calculator(ase_calc, "H")         # D0 and e_atom from the potential

T_K, p_bar = [300, 500, 700], [1e-6, 1e-3, 1, 1e3]
sim = ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], h2.mu_grid(T_K, p_bar, unit="bar"),
                          lambda rng: geometry.make_system(LammpsCalculator(pair, ["Fe", "H"]), "H"),
                          lambda_cubed_of_T=h2.lambda_cubed_of_T(), n_workers=12)
run_with_checkpoints(sim, "run", n_steps=20000, chunk=1000, x0=0.9, diffusion="md",
                     diffusion_steps=40, record_every=10, timestep_fs=0.5, taut_fs=20.0)
analysis = MBARSurfaceAnalysis(load_samples("run/samples.npz", discard=0.2),
                               reservoir=h2, pressure_unit="bar")
```

`examples/13_fe_h_eam_surface.py` is this workflow with checks, a CLI and plots.

## Real physical units (EMT, EAM, MLIPs)

A Calculator like ASE's EMT (or a real EAM potential, or a trained MLIP)
reports energies in eV using positions in Angstrom -- not the reduced
Lennard-Jones units (sigma = epsilon = 1) the LJ examples use. Two things
in `Replica`'s GCMC formulas need real values to match:

- **Temperature**: `ThermoState.beta = 1/temperature` needs `temperature`
  in eV, since it multiplies energies the calculator reports in eV. Use
  `gcmc_re.units.kelvin_to_eV(T_kelvin)`.
- **Thermal de Broglie wavelength**: the insertion/deletion acceptance
  formulas contain a `log(V / ((N+1) * Lambda^3))` term. The historical
  default `Lambda^3 = 1` is a correct, self-consistent convention in
  reduced LJ units, but not in real units -- a real particle's Lambda
  depends on its mass and T. Pass `lambda_cubed_of_T` to
  `ReplicaExchangeGCMC` using `gcmc_re.units.thermal_wavelength_cubed_angstrom3`.

See `examples/05_emt_metal_surface.py` for both in use, on a real Au(111)
slab (built with `ase.build.fcc111`) with H gas via ASE's EMT calculator --
including two important, concretely-checked caveats about what EMT+H
actually predicts there (steep repulsion, then a shallow, implausibly deep
secondary minimum) before trusting any number it reports.

`SlabSystem` also gained `insertion_z_range`, for restricting random
insertion attempts (and the volume used to weight them) to a thin region
near a real surface instead of sampling uniformly across a mostly-empty
box -- worth using whenever the interesting region is much smaller than
the accessible volume, which is generically true for a real substrate.

## Adsorbate sampling

The engine as is samples by default the number of gas atoms in the box at the end of the diffusion
process. In order to sample the actually adsorbed atoms an extra parameter
has to be passed to the ReplicaExchangeGCMC object specifying
the cutoff distance above which inserted atoms have to be considered
in the gasous state rather than actually adsorbed. In the previous
example, to add the condition of sampling the adsorbed atoms:
```python
sim = ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], h2.mu_grid(T_K, p_bar, unit="bar"),
                          lambda rng: geometry.make_system(LammpsCalculator(pair, ["Fe", "H"]), "H"),
                          lambda_cubed_of_T=h2.lambda_cubed_of_T(), n_workers=12)
```


## Parallelizing (`n_workers > 1`) with ASE calculators

`n_workers > 1` pickles whole `Replica` objects to ship them to worker
processes -- which means pickling whatever calculator each one holds.
Many ASE calculators become unpicklable once actually *used*: EMT builds a
`collections.defaultdict` with a lambda default factory the first time
`initialize()` runs (the first energy evaluation), and lambdas can't be
pickled by the standard library's pickler. The symptom is a cryptic
failure deep inside `concurrent.futures`/`multiprocessing`:

```
AttributeError: Can't pickle local object 'EMT.initialize.<locals>.<lambda>'
```

not any obviously-wrong line in your own code. `ASECalculatorAdapter` fixes
this by accepting a zero-argument **factory** instead of a live instance:
the class itself, if it needs no constructor arguments
(`ASECalculatorAdapter(EMT, cell=..., pbc=...)` -- note: `EMT`, not
`EMT()`), or `functools.partial` for one that does
(`ASECalculatorAdapter(functools.partial(EAM, potential='Fe_H.eam.fs'),
cell=..., pbc=...)`). It then constructs its own calculator lazily, the
first time it's actually needed, separately in each worker process, so
nothing already-initialized (and therefore potentially unpicklable) ever
has to cross a process boundary. All of `examples/05-09` use this pattern
already. Passing a live instance still works exactly as before for
`n_workers=1`; trying to actually use one with `n_workers > 1` now raises a
clear `TypeError` pointing at this fix, instead of the cryptic pickling
traceback above.

## Hybrid Monte Carlo (`hmc_enabled`)

Single-particle random MC displacement is a blind random walk; it's fine
for a dilute gas, but inefficient whenever particles need to move
*together* -- a dense/structured fluid, and eventually (see the design
notes this grew out of) surface reconstruction. `ReplicaExchangeGCMC(...,
hmc_enabled=True, hmc_leapfrog_steps=10, hmc_dt=0.005)` replaces every
replica's displacement move with Hybrid Monte Carlo instead: resample
momenta from Maxwell-Boltzmann, run a short leapfrog trajectory for *all*
mobile particles at once (proposals that follow the actual force field,
not a blind guess), and accept/reject the whole trajectory on the total
(kinetic + potential) energy change.

The property that makes this safe to use at all: correctness never
depends on getting the integrator or timestep right, only efficiency does
-- the Metropolis step absorbs whatever integration error a too-large
timestep introduces. Validated directly in `tests/test_hmc.py`: HMC-
augmented GCMC reproduces the same equilibrium `<N>`/`<U>` as plain MC on
the same system, and that result is unchanged even with a 4x larger,
deliberately sloppy timestep (acceptance rate drops, the answer doesn't) --
see `examples/10_hmc_demo.py` to watch this directly. Needs a `Calculator`
with `supports_forces = True` (`LJCalculator` and `ASECalculatorAdapter`
both qualify; the latter via ASE's own `get_forces()`); a per-species
`mass_of_species` dict on `ParticleSystem` feeds the kinetic-energy term
(falling back to real atomic masses via ASE's data tables, or 1.0 in
reduced units, for anything not given explicitly).

Current limitation worth knowing: the momentum kicks use only
`calculator.forces()` -- not any `external_energy` field (a continuum
wall, or a hard z-boundary). This doesn't break correctness (the final
accept/reject step still checks the true total energy, including that
field, so a trajectory ending somewhere forbidden is still always
rejected), but a trajectory isn't steered away from such a boundary the
way real forces would, so acceptance efficiency near one may be poor.
Fine for a bulk system or a real explicit substrate with no continuum
field; something to watch with `SlabSystem`'s `wall_epsilon > 0`.

Because HMC always resamples momenta fresh before each trial (rather than
carrying velocities as persistent state across the whole simulation),
there's no velocity state that ever crosses a replica-exchange swap --
unlike temperature replica exchange in MD, no velocity rescaling is
needed here.

## Displaceable substrate (surface reconstruction)

Real surfaces relax and reconstruct in response to adsorbates; a
`frozen_positions`/`frozen_species` substrate (rigid, never moves at all)
can't capture that. `ParticleSystem` also has a third particle category
for this -- `substrate_positions`/`substrate_species` -- atoms that enter
the energy evaluation and CAN move (MC or HMC), but, unlike the mobile
species, are never inserted or deleted; a fixed inventory that's free to
rearrange. `full_configuration()` always orders the three categories as
frozen, then substrate, then mobile.

Two ways to move substrate atoms:
- **`p_displace_substrate`** (a `Replica.sweep`/`ReplicaExchangeGCMC.run`
  argument, default 0 i.e. off): independent single-particle random MC
  moves for substrate atoms, via the generic clone-and-recompute path
  (correct for any calculator, same as the mobile species' generic path).
- **Joint HMC** (when `hmc_enabled=True`): substrate atoms move *together*
  with the mobile species, in one leapfrog trajectory under one
  accept/reject decision -- the right way to capture correlated
  adsorbate-substrate relaxation, since both then move under the same
  force field at the same time, rather than as independent, uncoordinated
  proposals. With no substrate atoms present this reduces to exactly the
  mobile-only HMC move from the previous section. The two mechanisms
  aren't mutually exclusive -- `p_displace_substrate` can add extra plain
  MC substrate moves on top of whatever HMC already does.

Needs a calculator whose `forces()` accounts for substrate atoms for the
HMC path specifically -- `ASECalculatorAdapter` does (via the full
configuration); `LJCalculator` does not (a documented limitation, since it
only ever sees `system.positions`), and combining it with substrate atoms
+ HMC raises a clear, specific error rather than silently giving wrong
physics. Plain MC substrate moves (`p_displace_substrate`) work with any
calculator via the generic path regardless.

`ParticleSystem.substrate_masses()` (parallel to `masses()`) feeds HMC's
kinetic-energy term for substrate atoms, same `mass_of_species` lookup
mechanism -- worth actually setting for a real substrate, since real
substrate atoms are typically much heavier than the mobile species (e.g.
`{'Fe': 55.845, 'H': 1.008}`), which matters for HMC's momentum scale.

Validated in `tests/test_substrate.py` on the case where the right answer
is known exactly: two substrate atoms, started well away from a real (ASE)
Lennard-Jones potential's known equilibrium separation, relax toward it --
via joint HMC and, independently, via plain MC -- not just "doesn't
crash". See `examples/11_substrate_reconstruction_demo.py` to watch this
directly. A genuinely interesting demo -- adsorbate-induced reconstruction,
where GCMC-controlled gas coverage measurably shifts the substrate's own
equilibrium geometry -- needs more careful (T, mu, mass, timestep) tuning
than that example attempts; a first try at combining both, with
too-favorable mu, gave overwhelming gas coverage that disrupted the
substrate dynamics entirely rather than gently perturbing them. Real,
unfinished tuning work, called out as such rather than papered over.

## Referencing a diatomic gas reservoir (`reservoir.py`)

GCMC here inserts/deletes single atoms, but a real experiment usually
controls a *diatomic* gas's pressure (H2, O2, N2, ...): mu_atom =
(1/2) mu_molecule(p, T). That conversion needs mu_molecule referenced to
"two separated ground-state atoms" as its zero of energy, i.e. shifted by
the molecule's dissociation energy D0 -- and if your Calculator is only
approximate (EMT, EAM, an MLIP), D0 must be computed with THAT SAME
calculator, not taken from an experimental table, or you silently bake in
an offset error (the calculator's own isolated-atom energy has no absolute
physical meaning to begin with; this is the standard "atomistic
thermodynamics" approach used throughout surface science). `reservoir.py`
provides:

- `diatomic_dissociation_energy_eV(ase_calculator, symbol)` -- self-
  consistent D0 (bond-length scan + harmonic zero-point correction) for
  ANY ASE-compatible calculator and ANY homonuclear diatomic, not just H2.
- `mu_atomic_from_diatomic_reservoir_eV` / `pressure_from_atomic_mu_reservoir_pascal`
  -- convert between a real (T, p) and the corresponding mu_atom, combining
  that self-consistent D0 with REAL gas-phase rotational/vibrational
  spectroscopy (which is universal physics, not calculator-dependent).

Both conversions also need the calculator's **isolated-atom energy**
`e_atom_eV` (a required keyword argument): GCMC energy differences are
measured on the calculator's own scale, where an isolated gas atom has
energy e_atom, so

    mu_atom(T, p) = (1/2) [mu_molecule(T, p) - D0] + e_atom

`diatomic_dissociation_energy_eV` returns it as `'e_atom_eV'`; it is
+3.21 eV for EMT's H and exactly 0 for the native `LJCalculator`. Leaving
it out shifts every mu_atom by -e_atom. For EMT+H that made lab pressures
look like ~10^55 bar in earlier versions of these examples. With it,
1e-4 to 1e6 atm H2 at 350-450 K is mu_H ~ +0.3 to +0.75 eV, inside the
mu_H windows examples 05-08 simulate (see
`examples/06_real_units_phase_diagram.py`). The YAML runner evaluates
e_atom with the configured calculator unless `diatomic_reservoir.e_atom_eV`
is given.

## A general (T, mu) -> (T, p) conversion for `MBARSurfaceAnalysis`

The built-in `isotherm_p`/`phase_diagram_Tp`/`locate_coexistence_p` assume
the simple atomic-gas convention `mu = T*ln(p/p0(T))`. A diatomic reservoir
doesn't fit that shape (it has an extra rotational/vibrational entropy term
and a dissociation-energy offset), so `MBARSurfaceAnalysis` also accepts a
fully general override:

```python
analysis = MBARSurfaceAnalysis(sim, mu_of_Tp=my_mu_of_Tp, p_of_Tmu=my_p_of_Tmu)
```

`mu_of_Tp(T, p) -> mu` replaces the simple conversion entirely; every method
built on it (`isotherm_p`, `phase_diagram_Tp`, `locate_coexistence_p`,
`coexistence_curve_Tp`) picks it up automatically, unchanged. `p_of_Tmu(T,
mu) -> p` is its inverse, used by `p_from_mu`; supply it when you have a
closed form (as `reservoir.py`'s H2 functions do) for an exact, cheap
answer. If you only give `mu_of_Tp`, `p_from_mu` falls back to numerically
inverting it (`invert_mu_of_Tp` in `pressure.py`, log-space root-finding
with automatic bracket expansion) -- robust even when the root sits at a
very low or very high pressure. `examples/07_general_Tp_phase_diagram.py`
runs the examples/05/06 setup with the real H2 reservoir wired in as
`mu_of_Tp`/`p_of_Tmu` and plots coverage on a pressure axis: the simulated
mu_H window (-0.2 to 0.8 eV) maps to roughly 1e-22 to 1e8 bar, which
contains the lab-relevant 1e-4 to 1e6 atm range.

(Getting that plot to render correctly across such an extreme range also
exposed a real bug in `plot_phase_diagram`: `pcolormesh` infers cell edges
by linearly averaging neighboring coordinates regardless of the axis's
display scale, so log-spaced pressure data spanning many decades had its
lower-pressure columns visually collapse to nothing. Fixed by plotting
against log10(p) directly with reformatted tick labels, rather than
plotting p on a matplotlib log-scale axis.)

## A real structure through the config pipeline: `examples/08_reasonable_Tp_phase_diagram.py`

Example 08 runs Pd(111) + H via EMT (structure read from
`examples/data/pd111_slab.xyz`) through `gcmc_re.runner`, on the realistic
1e-4 to 1e6 atm window, with D0 and e_atom both computed from EMT. The
whole window lies inside the simulated mu_H grid. Expect large, unphysical H
uptake: roughly 10-25 H on the 2x2 Pd(111) cell, i.e. several H per surface
Pd. EMT binds H about 3.1 eV below an isolated atom, against about 2.5 eV
per H in EMT's own H2, and its H parameters are documented as "not for any
serious use". The short demo run is also not equilibrated (<N> is still
rising at the end), so scale it up and check that coverage plateaus before
reading numbers off it. Use a real Pd-H EAM or an MLIP for physical
results. `examples/09_config_driven_run.py` + `pd111_emt_example.yaml` is
the file-driven version.

## The paper's LJ surface: `examples/12_paper_lj_surface.py`

Sets up the Lennard-Jones surface of Zhou, Scheffler & Ghiringhelli, PRB 100,
174106 (2019), Sec. III A, as specified there:
- a frozen 2-layer fcc(111) A18 slab in its hexagonal 3x3 cell, built with
  `SlabSystem(None, None, Lz, ..., cell_xy=...)`;
- the A-B and B-B LJ parameters, through the multi-species
  `PairLJCalculator`, which, unlike `LJCalculator`, sees frozen and
  substrate atoms;
- the system volume equal to the paper's 48 A surface zone. B must be
  confined to it: otherwise it slips through the top layer's hollows and
  is absorbed inside the slab;
- the 10 x 16 (T, mu) grid;
- `run_regc` with x0 = 0.99 and ten MC moves per step (`diffusion='mc_moves'`).

The paper does not give B's mass; 16.48 amu reproduces its Fig. 3 caption.
The script saves samples every 5000 steps and then compares MBAR@REGC and
aiAT@REGC phase diagrams against the paper's findings, in (T, mu) and, like
the paper's Fig. 3, in (T, p). The (T, p) figure hatches conditions outside
the sampled (T, mu) window, where MBAR extrapolates. Run it with
`--analyze-only` to re-analyze saved samples. A full 1.2e5-step run takes
hours.

## LAMMPS backend (`gcmc_re.LammpsCalculator`)

`LammpsCalculator(pair_commands, species)` runs any LAMMPS pair style:
EAM (`eam/fs`, `eam/alloy`), GRACE-FS (`grace/fs`), ACE (`pace`), and so
on. You give it the `pair_style`/`pair_coeff` lines and the species order
they use.

- **Energies and forces:** all atoms are loaded into a persistent LAMMPS
  instance and evaluated with `run 0`, so GCMC and MC/HMC moves work
  unchanged.
- **MD:** `Replica.diffuse('md')` runs each segment natively in LAMMPS
  (`fix nve` + `fix temp/csvr`, i.e. Bussi, + `fix wall/reflect`, with
  frozen atoms excluded) instead of through ASE.
- **Speed:** for a 72-atom Fe(110) slab with EAM, a full REGC step (GC
  attempt plus 40 MD steps) takes about 4 ms, against about 4 s through
  ASE's pure-Python EAM.
- **Hexagonal cells:** supported through LAMMPS triclinic boxes.
- **Workers:** the LAMMPS handle is rebuilt in each worker process.
- **`min_distance`:** an optional hard core that keeps GCMC insertions
  out of unphysical configurations where machine-learned potentials are
  unreliable.

The native `LJCalculator`/`PairLJCalculator` use the minimum-image
convention. They raise an error if a cutoff exceeds the cell's
minimum-image radius (`system.min_image_radius`), where further periodic
images would be silently missed. LAMMPS counts all images.

## H on Fe(110): `examples/13_fe_h_eam_surface.py`

A prepared production setup for a real metal-hydrogen system.

- **Potential:** not included. Use an Fe-H EAM in LAMMPS format (`.eam.fs`
  or `.eam.alloy`, e.g. from the NIST Interatomic Potentials Repository)
  via `--potential`, or any LAMMPS pair style via `--pair-style` /
  `--pair-coeff`, e.g. a GRACE-FS model:
  `--pair-style grace/fs --pair-coeff "* * model.yaml Fe H" --min-distance 0.6`.
  GRACE-FS needs a LAMMPS built with GRACE support.
- **Backend:** LAMMPS by default when its Python module is installed;
  `--backend ase` uses ASE's EAM instead.
- **Slab:** Fe(110), 3x4 cell, 6 layers. The bottom 3 are frozen; the top 3
  can relax under H. The lattice constant comes from the potential.
- **H region:** H is confined above the frozen block but may enter the
  relaxing subsurface.
- **Reservoir:** H2 with D0 and the isolated-H energy from the same
  potential, or `--d0 experimental` for potentials with a poor H2 molecule.
- **Grid and protocol:** a temperature x H2-pressure grid; `run_regc` with
  x0 = 0.9 and Bussi NVT MD (0.5 fs), or exact HMC.

Run `--check` first. It reports the lattice constant, the potential's own
H2 (r0, D0), e_atom, H binding scans at a few surface sites, the mu grid,
and a timing estimate. `--run` does the simulation with checkpoints and
rho(u) convergence checks; `--analyze-only` re-analyzes saved samples.
`--run --resume` continues a saved run bit-identically, and `--time-limit`
stops cleanly before a wall time (exit code 75). `hpc/` contains a ready
Slurm setup: environment, LAMMPS build, pre-flight check, and a
self-resubmitting production job. See `hpc/README.md`.

## Quickstart

```python
import numpy as np
from gcmc_re import BulkSystem, LJCalculator, ReplicaExchangeGCMC

def make_bulk(rng):
    return BulkSystem(box_length=7.0, calculator=LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))

temperatures = list(np.linspace(0.9, 1.3, 4))
mus = list(np.linspace(-3.8, -3.2, 4))

with ReplicaExchangeGCMC(temperatures, mus, make_bulk, seed=7, n_workers=1) as sim:
    sim.run(n_cycles=150, sweeps_per_cycle=150, record_every=5, burn_in_cycles=20)
    for row in sim.summary():
        print(row)
```

See `examples/` for: validating a new Calculator against the fast native
path, a surface-adsorption + MBAR phase diagram, a frozen-substrate
(e.g. Fe slab + mobile H) setup, a real multi-element ASE calculator
(EMT) driving a Au(111) slab + H gas in real physical units, that same
system's phase diagram properly referenced against a real H2 gas
reservoir, that reservoir wired in as a full (T, p) phase diagram, the
realistic 1e-4 to 1e6 atm window applied to a *real* structure and
potential (Pd(111) + H via EMT, with trajectory saving -- large, unphysical
H uptake, because of EMT's H parameters), the config-file-driven version of
that last one, and HMC
against plain MC on the same GCMC system (including the timestep-
independence check described above), and displaceable-substrate relaxation
(both HMC and plain MC) toward a real potential's known equilibrium
separation.

## Trajectory saving

`ReplicaExchangeGCMC.run(...)` accepts `trajectory_dir` and (optionally)
`trajectory_every`: every that many cycles, each replica's current
configuration (frozen + mobile atoms, via `full_configuration()`) is
appended as one frame to its own extended-XYZ file,
`<trajectory_dir>/replica_T{it}_mu{imu}.xyz` -- readable by ASE, OVITO,
VMD, or a text editor. Requires `pip install ase`. `trajectory_every`
defaults to `record_every` if omitted. `ParticleSystem` subclasses need
`cell`/`pbc` properties for this (both `BulkSystem` and `SlabSystem`
already provide them, with the correct periodicity for their geometry --
a bulk cube is periodic in all three axes, a slab only in x, y).

## Config-file-driven runs (`runner.py`)

For running a real structure end-to-end without writing Python each time:
`gcmc_re.runner.run_from_config` takes one config dict (or load one from a
JSON or YAML file with `load_config`) specifying an ASE-readable structure
file, the gas species, the (T, mu) replica ladder, and the (T, p) mesh for
the final phase diagram, and does everything -- read the structure, run
GCMC (with optional trajectory saving), compute D0 self-consistently (for
EMT; other calculators need `diatomic_reservoir.D0_eV` given explicitly)
and the isolated-atom energy e_atom (with the configured calculator, unless
`diatomic_reservoir.e_atom_eV` is given), run MBAR, and save the plot plus
raw `.npz` data. See
`examples/09_config_driven_run.py` (the CLI wrapper: `python
09_config_driven_run.py your_config.yaml`) and
`examples/pd111_emt_example.yaml` (an annotated example config; see its
own comments for why Pd+EMT+H gives unphysically large coverage) for the
full schema. A relative `structure.file` is resolved against the config
file's directory. The structure's cell must be orthogonal
(diagonal) -- build one via e.g. `ase.build.fcc111(..., orthogonal=True)`
if you don't have one already; the runner raises a clear error otherwise.

One YAML gotcha worth knowing if you write your own config: PyYAML only
recognizes scientific notation as a float when the exponent has an
explicit sign -- `1.0e-4` parses fine, but `1.0e6` (no `+`) silently parses
as the *string* `"1.0e6"`. The runner defensively re-parses the fields most
likely to hit this (pressures, D0) using Python's own `float()` (which has
no such restriction), so this specific mistake self-corrects rather than
causing a confusing failure -- but it's simplest to just always write the
sign (`1.0e+6`) in fields you add yourself.

## Extending

- **Docstrings**: every function and method in `gcmc_re/` uses numpy-style
  docstrings (Parameters/Returns/Raises sections), validated structurally
  with `numpydoc.validate`. If you add a new function, match that
  convention -- `help(gcmc_re.SomeClass.some_method)` is the quick way to
  check the rendered result looks right.
- **New potential**: subclass `Calculator`. If it's pairwise, implement
  `pair_energy_of` and set `supports_incremental = True` for the fast path
  (see `LJCalculator`). If it's many-body (EAM) or a black box (an MLIP),
  just implement `total_energy` and leave `supports_incremental = False`
  -- `ParticleSystem`'s generic clone-and-recompute path handles the rest
  correctly, automatically. `ASECalculatorAdapter` already does this for
  any ASE-compatible calculator (see `examples/01_validate_calculator.py`).
- **New geometry**: subclass `ParticleSystem` and implement `_min_image`,
  `volume`, `random_insertion_position`, `apply_move`, `apply_insert`
  (`external_energy` and the frozen-substrate machinery are optional).
- **Real substrate + adsorbate** (e.g. Fe + H): give `SlabSystem`
  `frozen_positions`/`frozen_species` for the rigid substrate, set
  `wall_epsilon=0`, and use `ASECalculatorAdapter(EAM(...))` or an MLIP as
  the calculator -- see `examples/03_frozen_substrate.py`. To connect to a
  real H2 gas reservoir, remember mu_H = (1/2) mu_H2(p, T): half the
  *molecular* chemical potential, since two H atoms come from one H2
  molecule at equilibrium; `pressure.py`'s p0(T) = T convention only
  captures translational motion, so for quantitative work get mu_H2(p, T)
  from NIST-JANAF tables or H2's rotational/vibrational partition function
  instead.
- **Run the tests** (`pytest tests/`) after any change to `systems.py` or
  `calculators.py` in particular -- they check that the generic
  clone-and-recompute path still agrees with the fast analytic path to
  floating-point precision, and that parallel and serial runs stay
  bit-identical. `tests/test_trajectory.py` and `tests/test_runner.py`
  cover the trajectory-saving and config-driven pieces respectively, using
  the fast LJCalculator so they run in seconds even though real usage of
  both is normally paired with EMT/EAM/an MLIP. `tests/test_hmc.py` checks
  forces against numerical gradients, and -- the check that actually
  matters for a new force implementation -- that HMC-augmented GCMC
  reproduces plain MC's equilibrium; if you add `forces()` to a new
  Calculator, extend this test rather than trusting it by inspection.
  `tests/test_substrate.py` similarly checks substrate atoms relax toward
  a real potential's known equilibrium separation, not just that nothing
  crashes.

Real, public Fe-H EAM potentials (LAMMPS `eam/fs` format, drop into
`ase.calculators.eam.EAM(potential='...')`):
- Ramasubramaniam, Kim & Curtin, *Phys. Rev. B* 79, 174101 (2009):
  `github.com/EACcodes/EAMpotentials` (`solid/Fe+H` folder)
- Wen (2021): `openkim.org/id/EAM_Dynamo_Wen_2021_FeH__MO_634187028437_001`
- General repository: `ctcms.nist.gov/potentials`
