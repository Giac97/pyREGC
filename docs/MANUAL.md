# regc user manual

`regc` computes surface phase diagrams: the coverage of an adsorbate on
a slab as a function of temperature T and gas pressure p. It runs
replica-exchange grand-canonical Monte Carlo (REGC) on a grid of
(T, mu) replicas, as in Zhou, Scheffler and Ghiringhelli, Phys. Rev. B
100, 174106 (2019), referred to below as [ZSG]. It then reweights all
samples with MBAR to a continuous θ(T, p) map.

You can use it in two ways:

- **As a Python library** (Part I): you build the structure, potential,
  gas reservoir and replica grid yourself, and keep full control.
- **As a standalone program** (Part II): you describe the study in one
  YAML (or JSON) file and run `regc-run config.yaml`.

Both use the same code, and the same study gives the same kind of
output either way. Part III collects practical advice and
troubleshooting.

The algorithm, with every acceptance rule, is in
[`REGC_algorithm.md`](REGC_algorithm.md). Every public class and
function, with all its parameters, is listed in
[`API_REFERENCE.md`](API_REFERENCE.md), which is generated from the
docstrings (`python docs/build_api_reference.py`). This manual explains
how the pieces fit together.

## Contents

- [0. Installation and conventions](#0-installation-and-conventions)
- [Concepts](#concepts)
- [Part I: the library](#part-i-the-library)
  - [1. A first run in 20 seconds](#1-a-first-run-in-20-seconds)
  - [2. Structures](#2-structures)
  - [3. Calculators (potentials)](#3-calculators-potentials)
  - [4. Gas reservoirs](#4-gas-reservoirs)
  - [5. The replica grid](#5-the-replica-grid)
  - [6. Running the simulation](#6-running-the-simulation)
  - [7. Long runs: checkpoints and batch jobs](#7-long-runs-checkpoints-and-batch-jobs)
  - [8. Saving and loading samples](#8-saving-and-loading-samples)
  - [9. Analysis with MBAR](#9-analysis-with-mbar)
  - [10. Several gas species](#10-several-gas-species)
  - [11. Worked examples](#11-worked-examples)
- [Part II: the standalone program (YAML)](#part-ii-the-standalone-program-yaml)
  - [12. A first run from a config file](#12-a-first-run-from-a-config-file)
  - [13. Command-line reference](#13-command-line-reference)
  - [14. Config file reference](#14-config-file-reference)
  - [15. Output files](#15-output-files)
  - [16. Batch jobs with the runner](#16-batch-jobs-with-the-runner)
  - [17. Calling the runner from Python](#17-calling-the-runner-from-python)
- [Part III: practice](#part-iii-practice)
  - [18. A workflow for a new system](#18-a-workflow-for-a-new-system)
  - [19. Choosing settings](#19-choosing-settings)
  - [20. Troubleshooting](#20-troubleshooting)
- [API reference](API_REFERENCE.md) (separate file)

---

## 0. Installation and conventions

### Installing

With mamba (recommended; includes LAMMPS from conda-forge):

```bash
mamba env create -f environment.yml     # from the repository root
mamba activate gcmc_re
pytest                                  # optional: the test suite, about 1 minute
```

The environment keeps the package's former name, `gcmc_re`. The package
itself is `regc`. `import gcmc_re` and `gcmc-re-run` still work as
deprecated aliases, so old scripts and checkpoints keep loading.

With pip:

```bash
pip install .                 # core only: numpy
pip install ".[all]"          # + ase, pymbar/scipy, matplotlib, pyyaml
pip install -e ".[dev]"       # editable install + pytest
```

| Extra | Brings | Needed for |
|---|---|---|
| `ase` | ASE | EMT/EAM/MLIP calculators, `ase.build` structures, trajectories |
| `mbar` | pymbar, scipy | `MBARSurfaceAnalysis` (any phase diagram) |
| `plot` | matplotlib | `plot_phase_diagram` |
| `yaml` | PyYAML | `.yaml`/`.yml` configs (JSON needs nothing) |

LAMMPS is not a pip extra. Install it from conda-forge, or build it
yourself (`hpc/build_lammps.sh`) for pair styles that conda-forge lacks,
such as `grace/fs`.

The core needs only numpy. Every optional dependency is imported inside
the function that needs it, so `import regc` always works.

### Units

| Quantity | Unit | Notes |
|---|---|---|
| Energy, chemical potential μ | eV | Real-units calculators (EMT, EAM, LAMMPS, MLIPs). |
| Length | Å | |
| Mass | amu | |
| Temperature **inside the engine** | eV (k_B T) | Convert with `kelvin_to_eV(T_K)`. Every `ReplicaExchangeGCMC` temperature and every analysis temperature is in eV. |
| Temperature at the edges | K | Reservoir methods (`mu_eV`, `mu_grid`) and the YAML file take Kelvin. |
| Pressure | Pa, bar or atm | Chosen with a `unit=` / `pressure_unit=` argument (`PRESSURE_UNITS`). |
| MD time step | fs | `timestep_fs`. |
| HMC time step `hmc_dt` | internal unit, 1 = 10.1805 fs | 0.5 fs is `0.5 / 10.1805`. The YAML key `hmc.timestep_fs` converts for you. |

The Lennard-Jones examples use reduced units instead (σ = ε = 1,
Λ³ = 1); see [§5](#5-the-replica-grid).

### The energy zero

Each calculator has its own energy zero. The chemical potential must sit
on the same scale. For a gas atom, that means adding the calculator's
energy of the isolated atom, `e_atom`. For EMT this is +3.21 eV for H.
Leaving it out shifts μ by that amount and makes every pressure absurd.
The reservoirs in [§4](#4-gas-reservoirs) handle this when they are built
from the calculator.

---

## Concepts

This section explains what the package computes, in enough depth to
choose settings and read results. The full derivations are in
[`REGC_algorithm.md`](REGC_algorithm.md).

### The system

A simulation box holds a slab and a variable number N of gas atoms
above it, periodic in x and y. The slab has two kinds of atoms:

- **frozen atoms** never move;
- **substrate atoms**, if any, relax.

The gas atoms are the *mobile* or *exchanged* particles. Hard walls
keep them between z = 0 (the top frozen layer) and z = Lz.

### The grand-canonical ensemble

At fixed temperature T and chemical potential μ, the box exchanges
atoms with an ideal reservoir. A configuration R with N gas atoms has
probability

```
P(R, N)  ∝  V^N / (N! Λ³ᴺ(T)) · exp(−β [E(R) − μ N]),     β = 1 / k_B T
```

where E is the calculator's energy and Λ the thermal de Broglie
wavelength of the gas atom. Grand-canonical Monte Carlo samples this
with three kinds of moves.

- **Insertion:** a new atom is placed at a random point in the insertion
  volume V_ex and accepted with probability
  `min(1, V_ex / (Λ³ (N_ex+1)) · exp(β [μ − ΔE]))`.
- **Deletion:** a random atom inside V_ex is removed with probability
  `min(1, Λ³ N_ex / V_ex · exp(−β [μ + ΔE]))`.
- **Diffusion** at fixed N: MC displacements, HMC or MD. This moves the
  atoms between insertions and relaxes the substrate.

Here N_ex counts the atoms inside V_ex. The insertion window changes
only how efficiently the ensemble is sampled, not which ensemble is
sampled.

### Why a grid of replicas

At low temperature or high coverage a single GCMC run gets stuck: the
energy barriers between surface phases are large compared with k_B T.
REGC runs one replica at each point of a (T, μ) grid and lets
neighbouring replicas swap configurations. The swap obeys detailed
balance in the combined ensemble of all replicas:

```
P_swap = min(1, (Λ³(T_a)/Λ³(T_b))^(N_a−N_b) · exp{(β_a−β_b)(E_a−E_b) − (β_a μ_a − β_b μ_b)(N_a−N_b)})
```

A configuration can then melt at high T, travel across the grid and
settle at low T in a different phase. This works only if neighbouring
replicas overlap: their energy and N distributions must share some
range. The swap acceptance measures this.

### From samples to a phase diagram: MBAR

Each replica records (U, N) every `record_every` steps. MBAR (the
multistate Bennett acceptance ratio) uses all samples from all replicas
together. It estimates the relative free energy of every sampled state,
and from that the average of any observable at any (T, μ), including
points between replicas, together with a statistical uncertainty. That
is why a 6×6 grid of replicas yields a smooth 150×120 map.

Two kinds of maps come out:

- **Mean coverage** θ(T, p) = ⟨N⟩ / area: a smooth function. Phase
  transitions appear as steep steps.
- **Most probable N**, the maximum of the distribution P(N): a
  piecewise-constant map whose regions are the stable phases. This is
  [ZSG]'s Fig. 3. Where P(N) has two peaks, two phases coexist.

### From pressure to chemical potential

Experiments control T and the gas pressure p, while the simulation
controls μ. The reservoir converts between them, on the calculator's
energy scale:

- **Monatomic gas:** `μ(T, p) = k_B T ln(p Λ³ / k_B T) + e_atom`.
- **Diatomic gas X2 (H2, O2, ...)**, which supplies atoms:
  `μ_X(T, p) = ½ [μ_X2(T, p) − D0] + e_atom`. Here μ_X2 is the
  molecule's ideal-gas chemical potential from its translational,
  rotational and vibrational partition functions, measured from its
  ground state. D0 is the zero-point-corrected dissociation energy.

Both D0 and e_atom must come from the potential that computes the slab.
Otherwise μ is offset from the energies it is compared with, and the
predicted pressures are wrong by orders of magnitude.

A decade of pressure changes μ by k_B T ln 10 for a monatomic gas, and
by half that for atoms from a diatomic gas. At 300 K, that is 0.060 eV
and 0.030 eV per decade.

### What "coverage" counts

- **N** counts every gas atom in the box, including atoms in the gas
  phase above the surface.
- **N_ads** counts only atoms within `adsorbed_cutoff` of a slab atom.
  Use it when the headroom is large, or at high pressure.
- **Normalisation:** per surface atom (monolayers, ML), per Å², or as
  an absolute count.

---

# Part I: the library

## 1. A first run in 20 seconds

This script runs a complete study on a toy system. The system is H on a
rigid 3×3 Cu(100) slab, with a pairwise Lennard-Jones potential and a
monatomic gas. It needs `ase`, `pymbar` and `matplotlib`, and takes about
20 seconds on one core.

```python
import os
import numpy as np
from ase.build import fcc100
from regc import (MBARSurfaceAnalysis, MonatomicIdealGasReservoir, PairLJCalculator,
                     ReplicaExchangeGCMC, kelvin_to_eV, load_samples, plot_phase_diagram,
                     save_samples, slab_geometry_from_atoms)

# 1. Structure: a rigid 3x3 Cu(100) slab with 4 A of space above it
slab = fcc100("Cu", size=(3, 3, 2), a=3.6, vacuum=None, orthogonal=True)
geometry = slab_geometry_from_atoms(slab, headroom=4.0)
print(geometry.summary())

# 2. Potential: a toy pairwise Lennard-Jones model, {(a, b): (epsilon_eV, sigma_A)}
pair_params = {("Cu", "H"): (0.15, 1.4), ("H", "H"): (0.01, 2.2)}

def make_system(rng):
    """A fresh system (and calculator) for every replica."""
    return geometry.make_system(PairLJCalculator(pair_params, r_cut_factor=1.7), "H")

# 3. Reservoir: a monatomic ideal gas of H atoms (toy: real H comes from H2)
gas = MonatomicIdealGasReservoir(mass_amu=1.008)
T_K = [300.0, 400.0, 500.0]
p_bar = [1e-3, 1e-1, 1e1, 1e3]
mus = gas.mu_grid(T_K, p_bar, unit="bar")           # shape (3, 4), eV
print(np.round(mus, 3))

# 4. Sampling: the REGC loop on a 3 x 4 replica grid
os.makedirs("tutorial", exist_ok=True)
with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], mus, make_system, seed=1,
                         lambda_cubed_of_T=gas.lambda_cubed_of_T(),
                         adsorbed_cutoff=2.5) as sim:
    sim.run_regc(n_steps=2000, x0=0.9, diffusion="mc", diffusion_steps=2,
                 record_every=10, burn_in_steps=300, verbose=False)
    save_samples(sim, "tutorial/samples.npz")
    print("swap acceptance:", sim.swap_acceptance())

# 5. Analysis: MBAR reweighting to a fine (T, p) mesh
analysis = MBARSurfaceAnalysis(load_samples("tutorial/samples.npz"),
                               reservoir=gas, pressure_unit="bar")
T_grid_K = np.linspace(300, 500, 21)
p_grid = np.geomspace(1e-3, 1e3, 31)
theta = analysis.phase_diagram_Tp([kelvin_to_eV(T) for T in T_grid_K], p_grid,
                                  area=geometry.n_surface, observable="N_ads")
print(f"coverage: {theta.min():.2f} to {theta.max():.2f} H per surface Cu")
plot_phase_diagram(T_grid_K, p_grid, theta, save_path="tutorial/theta.png",
                   colorbar_label="H per surface Cu", xlabel="p (bar)", ylabel="T (K)")
```

Output (acceptance values rounded):

```
18 frozen + 0 displaceable slab atoms, 9 per surface layer, cell |a1| = 7.637 A, |a2| = 7.637 A (rectangular), mobile species in z in (0, 4.00] A
[[-0.453 -0.334 -0.215 -0.096]
 [-0.629 -0.47  -0.311 -0.153]
 [-0.81  -0.612 -0.413 -0.215]]
swap acceptance: {'T': 0.49, 'mu': 0.28, 'cross_left': 0.15, 'cross_right': 0.38}
coverage: 0.00 to 1.54 H per surface Cu
Saved phase diagram to tutorial/theta.png
```

Every real study has the same five steps:

1. **Structure:** an ASE slab, laid out by `slab_geometry_from_atoms`.
2. **Calculator:** the potential, created fresh for every replica inside
   a *system factory* `make_system(rng)`.
3. **Reservoir:** maps (T, p) of the gas to μ, and supplies the thermal
   wavelength Λ³(T).
4. **Sampling:** `ReplicaExchangeGCMC` over the (T, μ) grid, run with
   `run_regc` (or `run_with_checkpoints` for long runs).
5. **Analysis:** `MBARSurfaceAnalysis` on the saved samples, given the
   same reservoir.

The rest of Part I covers each step in detail.

## 2. Structures

### 2.1 `slab_geometry_from_atoms`

```python
geometry = slab_geometry_from_atoms(atoms, n_frozen_layers=None, headroom=5.0, layer_tol=1e-3)
```

This takes an ASE `Atoms` slab and returns a `SlabGeometry`.

- **Layers** are found by z, grouping atoms within `layer_tol` Å.
- **`n_frozen_layers`**: the bottom layers kept rigid. Layers above
  them are *displaceable substrate*: MC (with `displace_substrate`),
  HMC and MD move them. The default `None` freezes every layer.
- **Placement:** the slab is shifted so the top *frozen* layer sits at
  z = 0. The box has a hard wall at z = 0, so gas atoms can reach the
  displaceable layers and subsurface sites above the frozen block but
  never enter it.
- **`headroom`**: the space in Å for gas atoms above the topmost atom.
  The box height is `Lz = top_z + headroom`.
- **Cell:** a1 must lie along x and a2 in the xy plane, the form that
  `ase.build` surface functions produce. Oblique cells, such as a
  hexagonal fcc(111) cell, work. A cell in any other orientation raises
  `ValueError: ... a1 along x ...`.

Useful attributes of `SlabGeometry`:

| Attribute | Meaning |
|---|---|
| `cell_xy` | 2×2 in-plane cell (rows a1, a2), Å |
| `Lz` | box height, Å |
| `top_z` | z of the topmost slab atom |
| `n_surface` | atoms in the top layer, for coverage per surface atom |
| `area` | in-plane area, Å² |
| `orthogonal` | True for a rectangular cell |
| `frozen_positions`, `substrate_positions` | the two atom blocks |
| `summary()` | one-line description |
| `make_system(calculator, exchange_species, mass_of_species=None, insertion_z_range=None)` | the `SlabSystem` |

`make_system` is what your system factory calls:

```python
system = geometry.make_system(calc, "H", mass_of_species={"H": 1.008},
                              insertion_z_range=(geometry.top_z + 1.0, geometry.top_z + 3.0))
```

`insertion_z_range` restricts insertion attempts, and the volume that
weights them, to a z window. Use it when the interesting region is a
thin layer above the surface. Insertions and deletions then only happen
inside the window. The default is the whole box.

### 2.2 Building the slab

Use any `ase.build` surface function. Pass `vacuum=None`, since the
geometry adds the headroom itself, and `orthogonal=True` where the
function supports it:

```python
from ase.build import bcc100, fcc111
slab = bcc100("Fe", size=(3, 3, 4), a=2.855, vacuum=None, orthogonal=True)
slab = fcc111("Pd", size=(2, 2, 4), a=3.89, vacuum=None)          # hexagonal cell: fine
slab = ase.io.read("my_slab.xyz")                                  # or any file ASE reads
```

### 2.3 The lattice constant of the potential

Build the slab at the potential's own lattice constant, otherwise the
slab carries strain:

```python
from regc import cubic_lattice_constant
a = cubic_lattice_constant(ase_calc, "Fe", crystal="bcc", a_range=(2.3, 3.5))
```

The function scans `a_range` and refines the minimum. Widen `a_range`
for fcc metals with large lattice constants, for example `(2.3, 5.5)`.

### 2.4 Checking a site before a run

`insertion_energy_scan(system, xy_points, z_values)` returns the energy
of inserting one atom along vertical lines. Use it on the bare slab to
check that the potential binds the adsorbate where you expect, at a
sensible height, before spending CPU time.

### 2.5 Other systems

- **`SlabSystem(Lx, Ly, Lz, calculator, ...)`** builds a slab by hand,
  with `frozen_positions`, `substrate_positions`, `insertion_z_range`,
  an optional continuum wall (`wall_epsilon`, `wall_sigma`, `wall_cut`)
  and `cell_xy` for oblique cells. `geometry.make_system` does this for
  you.
- **`BulkSystem(box_length, calculator, ...)`** is a fully periodic
  cubic box, used for bulk-gas validation.

`LJSystem` and `SlabLJSystem` are older aliases of `BulkSystem` and
`SlabSystem`.

## 3. Calculators (potentials)

All calculators subclass `Calculator`.

| Calculator | Use it for | HMC | MD | Notes |
|---|---|---|---|---|
| `LJCalculator(sigma, epsilon, r_cut)` | reduced-unit LJ tests | yes | no | **Ignores the slab:** mobile–mobile only. |
| `PairLJCalculator(pair_params, r_cut_factor=2.5)` | toy models that see the slab | yes | no | `pair_params = {(a, b): (epsilon_eV, sigma_A)}` |
| `LammpsCalculator(pair_commands, species, ...)` | EAM, ACE/PACE, GRACE, any LAMMPS pair style | yes | yes, native | The fastest choice for real potentials. |
| `ASECalculatorAdapter(factory, cell, pbc)` | any ASE calculator: EMT, EAM, MACE, NequIP, ... | if ASE gives forces | yes, ASE MD | Needs a *factory* for `n_workers > 1`. |

### 3.1 Cutoffs and the minimum image

The pairwise calculators use the minimum-image convention. Every cutoff
must be shorter than half the shortest in-plane cell width
(`system.min_image_radius`). A larger cutoff raises an error naming both
numbers. Lower `r_cut` / `r_cut_factor`, or use a larger cell.

### 3.2 LAMMPS

```python
from regc import LammpsCalculator, eam_pair_commands, lammps_ase_calculator

pair = eam_pair_commands("FeH_Wen.eam.fs", ["Fe", "H"])    # style from the file name
pair = ["pair_style pace", "pair_coeff * * FeH.yace Fe H"]   # ...or any pair style
calc = LammpsCalculator(pair, ["Fe", "H"], masses={"H": 1.008},
                        min_distance=None, z_padding=5.0)
ase_calc = lammps_ase_calculator(pair, ["Fe", "H"])         # same potential, ASE interface
```

- **`species`:** the LAMMPS atom-type order; it must match `pair_coeff`.
- **`min_distance`:** a hard core in Å. Insertions closer than this to
  any atom are rejected. Use it for machine-learned potentials, which
  can be wrong at short range, where they have no training data.
- **`z_padding`:** extra box height LAMMPS adds beyond the atoms.
- **Startup:** LAMMPS starts lazily on first use, so a
  `LammpsCalculator` can be pickled to worker processes before it has
  run.
- **`ase_calc`:** an ordinary ASE calculator running the same potential.
  Use it for setup work: `cubic_lattice_constant`,
  `DiatomicReservoir.from_calculator` and site scans.

### 3.3 ASE calculators and parallel runs

```python
import functools
from ase.calculators.eam import EAM
from regc import ASECalculatorAdapter

factory = functools.partial(EAM, potential="FeH_Wen.eam.fs")
calc = ASECalculatorAdapter(factory, cell=[geometry.cell_xy[0, 0], geometry.cell_xy[1, 1], geometry.Lz],
                            pbc=(True, True, False))
```

With `n_workers > 1`, replicas are pickled to worker processes. Many ASE
calculators cannot be pickled once they have been used. EMT, for
example, fails with `Can't pickle local object 'EMT.initialize.<locals>.<lambda>'`.
To avoid this, pass the adapter a zero-argument factory (a class, or a
`functools.partial`) instead of a live instance. It then builds the
calculator inside each worker. A live instance is fine for
`n_workers=1`.

For an oblique cell, pass the full 3×3 cell as `cell`.

### 3.4 Writing your own calculator

Subclass `Calculator` and implement `total_energy(system)`. It returns
the energy of `system.full_configuration()`, the mobile, substrate and
frozen atoms together. Optionally add:

- `forces(system)` and `supports_forces = True`, to enable HMC and MD;
- `pair_energy_of(system, pos, exclude=None, species=None)` and
  `supports_incremental = True`, for fast single-particle moves.

Without `pair_energy_of`, the engine recomputes the total energy for
each move. That is always correct, only slower.

## 4. Gas reservoirs

A reservoir converts the gas state (T, p) into the chemical potential of
the exchanged atom. It also gives the atom's thermal wavelength Λ³(T).
Build it from the same potential as the slab.

### 4.1 Diatomic gases (H2, O2, ...)

```python
from regc import DiatomicReservoir

h2 = DiatomicReservoir.from_calculator(ase_calc, "H", d0="self")
print(h2.D0_eV, h2.e_atom_eV, h2.calculator_info)
```

`from_calculator` evaluates the potential's own molecule and isolated
atom. **e_atom** (the zero of the energy scale) always comes from the
calculator. **D0** (the zero-point-corrected dissociation energy)
depends on `d0`:

| `d0` | D0 used | When |
|---|---|---|
| `"self"` (default) | the potential's own molecule | Consistent; use it when the potential's molecule is reasonable. |
| `"experimental"` | 4.478 eV (H2 only) | The potential describes the molecule poorly. |
| a number | that value, in eV | A corrected or literature value. |

`calculator_info` holds the diagnostics: `r0_angstrom`, `D0_eV`,
`zpe_eV`, `d0_source` and more. Compare them with experiment before a
production run.

For elements other than H, pass the real molecule's spectroscopic
constants:

```python
o2 = DiatomicReservoir.from_calculator(ase_calc, "O", bond_length_angstrom=1.2075,
                                       vib_wavenumber_cm=1580.2)
```

Without a calculator, construct a reservoir directly:

```python
h2 = DiatomicReservoir.hydrogen(D0_eV=4.478, e_atom_eV=0.0)
res = DiatomicReservoir(D0_eV, e_atom_eV, mass_molecule_amu, bond_length_angstrom,
                        vib_wavenumber_cm, symmetry_number=2)
```

O2 has a triplet ground state that the reservoir does not include. It
shifts μ_O by −½ kT ln 3, about −0.02 eV at 500 K.

### 4.2 Monatomic gases

```python
from regc import MonatomicIdealGasReservoir
ar = MonatomicIdealGasReservoir(mass_amu=39.95, e_atom_eV=0.0)
```

### 4.3 What a reservoir gives you

| Method | Returns |
|---|---|
| `mu_eV(T_kelvin, p_pascal)` | μ at one point |
| `pressure_pascal(T_kelvin, mu_eV)` | the inverse |
| `mu_grid(temperatures_K, pressures, unit="Pa")` | the (n_T, n_p) μ array for a pressure ladder |
| `lambda_cubed_of_T()` | Λ³ as a function of k_B T in eV, for the engine |
| `mu_at_T_eV(T_eV, p, unit="Pa")` | μ with the engine's temperature unit |
| `analysis_functions(unit="Pa")` | `(mu_of_Tp, p_of_Tmu)` for MBAR |

For lower-level work, `regc.reservoir` also has
`diatomic_dissociation_energy_eV`, `diatomic_gas_mu_eV`,
`mu_atomic_from_diatomic_reservoir_eV` and
`pressure_from_atomic_mu_reservoir_pascal`.

## 5. The replica grid

```python
sim = ReplicaExchangeGCMC(temperatures, mus, system_factory,
                          seed=0, max_disp=0.2, n_workers=1,
                          lambda_cubed_of_T=..., hmc_enabled=False,
                          hmc_leapfrog_steps=10, hmc_dt=0.005,
                          adsorbed_cutoff=None)
```

| Argument | Meaning |
|---|---|
| `temperatures` | The T ladder, in **eV**: `[kelvin_to_eV(T) for T in T_K]`. |
| `mus` | 1D (n_μ, shared by all T) or 2D (n_T, n_μ). A pressure ladder is 2D: `reservoir.mu_grid(T_K, p, unit)`. With S species, add a last axis of length S. |
| `system_factory` | `factory(rng) -> ParticleSystem`, called once per replica. **Create a new calculator inside it**, so replicas never share one. |
| `seed` | Master seed. Each replica draws its own stream from it, so runs are reproducible. Parallel and serial runs give identical results. |
| `max_disp` | Largest single-atom MC displacement per axis, Å. |
| `n_workers` | Worker processes. Use up to one per replica; more brings no gain. |
| `lambda_cubed_of_T` | Λ³(T): `reservoir.lambda_cubed_of_T()` in real units. The default, Λ³ = 1, is correct only in reduced LJ units. |
| `hmc_enabled` | Sweep scheme only: use HMC in place of single-atom displacements. |
| `hmc_leapfrog_steps`, `hmc_dt` | HMC trajectory length and step (internal time unit, 1 = 10.1805 fs). Used by `diffusion="hmc"` and by the sweep scheme's HMC. |
| `adsorbed_cutoff` | If set (Å), each sample also records `N_ads`, the number of gas atoms within this distance of a slab atom. |

`ReplicaExchangeGCMC` is a context manager. `with ... as sim:` shuts the
worker pool down at the end; without `with`, call `sim.close()`.

**Choosing the grid.** Neighbouring replicas must overlap in energy and
particle number, or swaps between them are never accepted. Space
temperatures geometrically, for example `np.geomspace(300, 900, 6)`.
Space pressures by decades or less. If `swap_acceptance()` reports
values near zero for a pattern, tighten the grid along that axis.

## 6. Running the simulation

### 6.1 The REGC loop (`run_regc`)

```python
finished = sim.run_regc(n_steps, x0=0.99, diffusion="mc", diffusion_steps=10,
                        record_every=100, burn_in_steps=0,
                        swap_types=("T", "mu", "cross_left", "cross_right"),
                        convergence_every=None, convergence_tol=0.05, convergence_bins=30,
                        verbose=True, trajectory_dir=None, trajectory_every=None,
                        **diffusion_kwargs)
```

One REGC step:

1. **Choose the move.** With probability `x0`, every replica makes one
   insertion or deletion attempt. Otherwise one replica-exchange pattern
   is drawn from `swap_types`, with an offset of 0 or 1, and all its
   disjoint pairs are attempted.
2. **Diffuse.** Every replica moves at fixed N for `diffusion_steps`
   units, in parallel.
3. **Record.** Every `record_every` steps after `burn_in_steps`, each
   replica records U, N, and `N_ads` / `N_s` where available.
4. **Check convergence.** Every `convergence_every` steps, the run
   compares the reduced-energy densities ρ(u) with the previous check.
   It stops when the largest change is below `convergence_tol`.

The return value is True if the run converged, otherwise False. [ZSG]
use x0 = 0.99 for the LJ surface and 0.9 for Si clusters. 0.9 is a good
start for a real surface.

**Diffusion modes**

| `diffusion` | One unit of `diffusion_steps` is | Needs | Keyword arguments |
|---|---|---|---|
| `"mc_moves"` | one single-atom displacement attempt | – | – |
| `"mc"` | one MC cycle: as many attempts as movable atoms | – | `displace_substrate=True` to move substrate atoms too |
| `"hmc"` | one HMC trajectory (all atoms, Metropolis-corrected) | forces | set via `hmc_leapfrog_steps`, `hmc_dt` on the constructor |
| `"md"` | one NVT MD step | `LammpsCalculator` or `ASECalculatorAdapter` | `timestep_fs=1.0`, `thermostat="bussi"` or `"langevin"`, `taut_fs=20.0`, `friction_per_fs=0.01` |

- **MD needs a full-featured calculator.** With `LammpsCalculator` it
  runs natively in LAMMPS, about 1000× faster than ASE for an EAM slab.
  With `ASECalculatorAdapter` it uses ASE's MD. `LJCalculator` and
  `PairLJCalculator` raise
  `ValueError: diffusion='md' ... needs an ASECalculatorAdapter`; use
  `"mc"` or `"hmc"` with them.
- **Substrate motion:** HMC and MD always move the displaceable
  substrate.
- **MD sampling error:** MD has no accept/reject step, so the samples
  carry its time-step error. Keep `timestep_fs` small: [ZSG] use 1 fs,
  and 0.5 fs is safer with H.

**Swap patterns.** `"T"` pairs neighbours in T, `"mu"` neighbours in μ,
and `"cross_left"` / `"cross_right"` the two diagonals. Restricting
`swap_types` is mostly useful for tests.

**Trajectories.** Pass `trajectory_dir` (and optionally
`trajectory_every`, default `record_every`). Each replica's
configuration is appended as an extended-XYZ frame to
`<dir>/replica_T{i}_mu{j}.xyz`. This needs ASE.

### 6.2 The sweep scheme (`run`)

The older scheme mixes displacement, insertion and deletion moves in
each cycle. Replicas are exchanged every `exchange_every` cycles.

```python
sim.run(n_cycles, sweeps_per_cycle=200, exchange_every=1, record_every=1, burn_in_cycles=0,
        p_displace=0.5, p_insert=0.25, p_delete=0.25, p_displace_substrate=0.0,
        diagonal_swaps=True, trajectory_dir=None, trajectory_every=None, verbose=True)
```

It is often more efficient for cheap force fields. Prefer `run_regc` for
anything you want to compare with [ZSG]. Only `run_regc` supports
checkpoints.

### 6.3 Diagnostics

- `sim.swap_acceptance()`: acceptance per swap pattern. [ZSG] report
  about 25%.
- `sim.summary()`: one dict per grid point with T, μ, N, energy, sample
  count and acceptance rates.
- `sim[(i_T, i_mu)].acceptance_rates()`: the move acceptance of one
  replica.
- `sim.rho_u_change(n_prev)`: the convergence measure used by
  `run_regc`.

## 7. Long runs: checkpoints and batch jobs

```python
from regc import run_with_checkpoints, EXIT_RESUBMIT

status = run_with_checkpoints(sim, "run_dir", n_steps=20000, chunk=1000,
                              x0=0.9, diffusion="md", diffusion_steps=40, record_every=10,
                              convergence_every=2000, convergence_tol=0.05,
                              time_limit=None, resume=True, overwrite=False,
                              metadata={"potential": "FeH_Wen.eam.fs", "a": a},
                              timestep_fs=0.5, taut_fs=20.0)
print(status.steps_done, status.finished, status.converged, status.resumed_from)
```

This runs `run_regc` in chunks of `chunk` steps. After each chunk it
writes:

- `samples.npz`, all samples so far;
- `checkpoint.pkl`, the complete simulation state, including every
  random number generator;
- `run_metadata.json`, your `metadata`.

How restarts behave:

- **`resume=True`** continues the run in `run_dir` if there is one. The
  checkpoint must match `sim` (grid, species, seed), and every key of
  `metadata` must equal the saved value. Otherwise you get a
  `ValueError`, so a resumed run can never silently change its setup.
  `n_steps` may be raised to extend a finished run.
- **A directory that already holds a run** raises `FileExistsError`
  unless you resume, or pass `resume=False, overwrite=True`.
- **`time_limit`** (seconds from the call): the function stops before
  starting a chunk that would end later, and returns
  `finished=False`. A batch script then exits with `EXIT_RESUBMIT`
  (75) and resubmits itself; `hpc/run.sbatch` does exactly this.
- **Early stops:** a run stopped before its first chunk completes
  writes nothing, and the next call starts afresh.
- **Burn-in:** discard equilibration in the analysis
  (`load_samples(..., discard=0.2)`). `burn_in_steps` would apply to
  every chunk.

For finer control there are `sim.save_checkpoint(path, extra=None)` and
`sim.load_checkpoint(path)`.

## 8. Saving and loading samples

```python
from regc import save_samples, load_samples
save_samples(sim, "samples.npz")
samples = load_samples("samples.npz", discard=0.2)   # drop the first 20% per replica
```

The file stores, for each replica, the recorded U, N, `N_ads` (when
recorded) and per-species counts `N_s`. It also stores the temperature
ladder, the μ grid and Λ³, so the analysis needs nothing else. The
result is a `SampleSet` that `MBARSurfaceAnalysis` accepts like a
finished simulation. `samples.discard(fraction)` drops more data later.

## 9. Analysis with MBAR

```python
from regc import MBARSurfaceAnalysis
analysis = MBARSurfaceAnalysis(samples, reservoir=h2, pressure_unit="bar")
```

MBAR combines all replicas' samples into one estimate of the free energy
of every sampled state. It can then predict averages at any (T, μ)
where the samples overlap, including points between replicas.

- **Input:** a finished simulation or a `SampleSet`.
- **Pressure mapping:** give `reservoir` and `pressure_unit` to convert
  pressures to μ the same way the sampling did. Alternatively, pass
  `mu_of_Tp(T_eV, p)` and optionally its inverse `p_of_Tmu`.
- **Λ³:** taken from the samples.
- **Units:** temperatures are k_B T in eV; pressures are in
  `pressure_unit`.

| Method | Gives |
|---|---|
| `phase_diagram_Tp(T_grid, p_grid, area=1.0, observable="N")` | ⟨observable⟩ / area on the (T, p) mesh: the coverage map |
| `phase_diagram(T_grid, mu_grid, area, observable)` | the same on a (T, μ) mesh |
| `most_probable_N_Tp(T_grid, p_grid, observable="N")` | the most probable N: the stable phase, labelled by its N ([ZSG] Fig. 3) |
| `most_probable_N(T_grid, mu_grid, observable)` | the same on (T, μ) |
| `mean_N(T, mu)`, `mean_N_p(T, p)` | ⟨N⟩ and its uncertainty at one point |
| `isotherm(T, mu_grid, area)`, `isotherm_p(T, p_grid, area)` | coverage along one isotherm |
| `pN(T, mu, observable)`, `pN_p(T, p)` | the reweighted distribution P(N) |
| `lowest_energy_N(mu_values)`, `lowest_energy_N_Tp(T_grid, p_grid)` | the atomistic-thermodynamics (aiAT) diagram from the lowest sampled energy per N, for comparison |
| `locate_coexistence_p(T, p_bracket)`, `coexistence_curve_Tp(T_grid, p_bracket)` | the 2D condensation line, where P(N) is bimodal |
| `mean_N_species`, `most_probable_composition` | several species ([§10](#10-several-gas-species)) |

**Observables.**

| `observable` | Counts | Needs |
|---|---|---|
| `"N"` | every gas atom in the box | – |
| `"N_ads"` | atoms within `adsorbed_cutoff` of a slab atom | a run with `adsorbed_cutoff` |
| `"N_<species>"` | one species, e.g. `"N_Cl"` | a multi-species run |

Choose `N_ads` when the headroom holds gas-phase atoms that are not
adsorbed.

**Normalisation.** `area=geometry.area` gives atoms per Å².
`area=geometry.n_surface` gives atoms per surface atom (monolayers).
`area=1` gives absolute counts.

**Plotting.**

```python
from regc import plot_phase_diagram
plot_phase_diagram(T_grid_K, p_grid, theta, save_path="theta.png",
                   colorbar_label="H per surface Fe", xlabel="p(H2) (bar)", ylabel="T (K)",
                   coexistence=None)
```

The x axis is log10(p). `coexistence` overlays a list of (T, p) points.

**Extrapolation.** Reweighting is reliable only where the sampled
energies and particle numbers overlap with the target state. Inside the
sampled (T, p) window this is usually true. Beyond it, the map is an
extrapolation; mark it as such or extend the ladder.

## 10. Several gas species

Pass a list of species to exchange several at once, for example O and
Cl from an O2/Cl2 feed. Each species gets its own μ and Λ³. Insertions
and deletions choose a species. The swap rule and MBAR sum over species.
A one-species run uses the original code path, with identical results.

The ladder stays 2D. Its second axis becomes a path through
(μ_O, μ_Cl) space; the natural path is a feed of fixed composition at
increasing total pressure.

```python
x = {"O": 0.999, "Cl": 0.001}                                    # feed mole fractions
o2  = DiatomicReservoir.from_calculator(ase_calc, "O",  bond_length_angstrom=1.2075, vib_wavenumber_cm=1580.2)
cl2 = DiatomicReservoir.from_calculator(ase_calc, "Cl", bond_length_angstrom=1.988,  vib_wavenumber_cm=559.7)
p = np.array(p_bar)
mus = np.stack([o2.mu_grid(T_K, x["O"] * p, unit="bar"),
                cl2.mu_grid(T_K, x["Cl"] * p, unit="bar")], axis=-1)     # (n_T, n_p, 2)

sim = ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], mus,
                          lambda rng: geometry.make_system(LammpsCalculator(pair, ["Ag", "O", "Cl"]), ["O", "Cl"]),
                          lambda_cubed_of_T=[o2.lambda_cubed_of_T(), cl2.lambda_cubed_of_T()])

def mu_of_Tp(T, p):                                              # the same path, for MBAR
    return np.array([o2.mu_at_T_eV(T, x["O"] * p, unit="bar"),
                     cl2.mu_at_T_eV(T, x["Cl"] * p, unit="bar")])

analysis = MBARSurfaceAnalysis(load_samples("run/samples.npz"), mu_of_Tp=mu_of_Tp)
theta_cl = analysis.phase_diagram_Tp(T_grid, p_grid, area=geometry.n_surface, observable="N_Cl")
```

- **Species order:** `mus` and `lambda_cubed_of_T` follow the order of
  `exchange_species`.
- **Analysis:** `mean_N_species(T, mu_vec)` gives every species' mean.
  `most_probable_composition` labels phases by (N_O, N_Cl), which
  separates an O-rich and a Cl-rich phase with the same total coverage.
- **Coexistence:** `locate_coexistence_mu` is disabled with several
  species; scan `pN` along the path instead.
- **Scope:** only atoms are exchanged. Molecules such as CO, and swaps
  of substrate atoms, are not implemented.

## 11. Worked examples

Each example below is a complete script. The examples in §11.1–11.5
were run as printed, and the output shown is their real output. Run
them from a directory where they may write files. The
`examples/` folder of the repository has more, numbered from cheapest
to most expensive.

### 11.1 A real calculator from a structure file, in parallel

H on Pd(111) with ASE's EMT. The slab is read from a file, the H2
reservoir comes from EMT itself, insertions are limited to 1–3 Å above
the surface, and two worker processes run the replicas. It takes about
2 minutes. Run it from the repository root, or change the path to the
structure file.

```python
import numpy as np
from ase.io import read
from ase.calculators.emt import EMT
from regc import (ASECalculatorAdapter, DiatomicReservoir, MBARSurfaceAnalysis,
                     ReplicaExchangeGCMC, kelvin_to_eV, load_samples, save_samples,
                     slab_geometry_from_atoms)

slab = read("examples/data/pd111_slab.xyz")                     # 2x2 Pd(111)
geometry = slab_geometry_from_atoms(slab, headroom=5.0)         # all layers rigid
print(geometry.summary())

h2 = DiatomicReservoir.from_calculator(EMT(), "H", d0="self")   # EMT's own H2 and H atom
print(f"D0 = {h2.D0_eV:.4f} eV, e_atom = {h2.e_atom_eV:.4f} eV")

cell = np.zeros((3, 3))
cell[:2, :2], cell[2, 2] = geometry.cell_xy, geometry.Lz
window = (geometry.top_z + 1.0, geometry.top_z + 3.0)           # insert 1-3 A above the top

def make_system(rng):
    calc = ASECalculatorAdapter(EMT, cell=cell, pbc=(True, True, False))   # a factory: picklable
    return geometry.make_system(calc, "H", insertion_z_range=window)

T_K, p_atm = [350.0, 450.0], [1e-4, 1e-1, 1e2, 1e5]
with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], h2.mu_grid(T_K, p_atm, unit="atm"),
                         make_system, seed=3, n_workers=2, max_disp=0.3,
                         lambda_cubed_of_T=h2.lambda_cubed_of_T()) as sim:
    sim.run_regc(n_steps=400, x0=0.9, diffusion="mc", diffusion_steps=1, record_every=5,
                 burn_in_steps=100, verbose=False)
    save_samples(sim, "emt_samples.npz")
    for row in sim.summary()[:2]:
        print({k: row[k] for k in ("T_index", "mu_index", "N", "energy")})

analysis = MBARSurfaceAnalysis(load_samples("emt_samples.npz"), reservoir=h2, pressure_unit="atm")
N, dN = analysis.mean_N_p(kelvin_to_eV(400.0), 1.0)
print(f"<N_H> at 400 K, 1 atm: {N:.2f} +- {dN:.2f}  ({N / geometry.n_surface:.2f} per surface Pd)")
```

```
8 frozen + 0 displaceable slab atoms, 4 per surface layer, cell |a1| = 5.501 A, |a2| = 4.764 A (rectangular), mobile species in z in (0, 5.00] A
D0 = 4.9179 eV, e_atom = 3.2100 eV
{'T_index': 0, 'mu_index': 0, 'N': 14, 'energy': 2.899}
{'T_index': 0, 'mu_index': 1, 'N': 24, 'energy': 5.217}
<N_H> at 400 K, 1 atm: 12.69 +- 0.10  (3.17 per surface Pd)
```

What to notice:

- **The energy zero.** `e_atom` is 3.21 eV: EMT puts an isolated H atom
  3.21 eV above its zero. Without it, μ would be off by that much.
- **The cell.** The cell built from the 3×3 matrix works for any slab
  orientation, so the same code handles hexagonal cells.
- **The ± error** is MBAR's statistical uncertainty. It does not include
  slow equilibration, which only longer runs reveal.
- **The physics.** Three H per surface Pd is far beyond any real Pd–H
  surface phase. EMT over-binds H, and its H parameters are not meant
  for serious use. The pipeline is fine; the potential is not.

### 11.2 A relaxing substrate: MC, HMC and MD diffusion

A 3-layer Cu(100) slab whose top layer relaxes (`n_frozen_layers=2`),
with a Cu–Cu pair term so the substrate holds together. The script runs
the same grid with substrate MC moves and with HMC.

```python
import numpy as np
from ase.build import fcc100
from regc import (MonatomicIdealGasReservoir, PairLJCalculator, ReplicaExchangeGCMC,
                     kelvin_to_eV, slab_geometry_from_atoms)

slab = fcc100("Cu", size=(3, 3, 3), a=3.6, vacuum=None, orthogonal=True)
geometry = slab_geometry_from_atoms(slab, n_frozen_layers=2, headroom=4.0)   # top layer relaxes
print(geometry.summary())

pair_params = {("Cu", "Cu"): (0.40, 2.30), ("Cu", "H"): (0.15, 1.4), ("H", "H"): (0.01, 2.2)}
make_system = lambda rng: geometry.make_system(PairLJCalculator(pair_params, r_cut_factor=1.6), "H")
gas = MonatomicIdealGasReservoir(mass_amu=1.008)
T_K, p_bar = [300.0, 450.0], [1e-2, 1e2]

FS = 10.1805                                   # fs per internal time unit
for mode, kwargs in [("mc", {"displace_substrate": True}), ("hmc", {})]:
    with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], gas.mu_grid(T_K, p_bar, unit="bar"),
                             make_system, seed=0, lambda_cubed_of_T=gas.lambda_cubed_of_T(),
                             hmc_leapfrog_steps=10, hmc_dt=1.0 / FS) as sim:      # 1 fs steps
        sim.run_regc(n_steps=200, x0=0.9, diffusion=mode, diffusion_steps=2, record_every=10,
                     verbose=False, **kwargs)
        rep = sim[(0, 1)]
        top = rep.system.substrate_positions[:, 2]
        print(f"{mode:3s}: N = {rep.system.n}, top-layer z spread = {top.std():.3f} A, "
              f"acceptance = { {k: round(v, 2) for k, v in rep.acceptance_rates().items()} }")
```

```
18 frozen + 9 displaceable slab atoms, 9 per surface layer, cell |a1| = 7.637 A, |a2| = 7.637 A (rectangular), mobile species in z in (0, 5.80] A
mc : N = 8, top-layer z spread = 0.028 A, acceptance = {'displace': 0.48, 'insert': 0.14, 'delete': 0.11, 'hmc': 0.0, 'displace_substrate': 0.05}
hmc: N = 7, top-layer z spread = 0.056 A, acceptance = {'displace': 0.0, 'insert': 0.21, 'delete': 0.13, 'hmc': 0.96, 'displace_substrate': 0.0}
```

What to notice:

- **The box height.** With a relaxing top layer, the box runs from the
  top *frozen* layer (z = 0) to 4 Å above the top atom, so here
  z ≤ 5.80 Å. H can reach the relaxing layer and the sites below it.
- **Substrate MC.** Only 5% of the substrate moves are accepted
  (`max_disp` is 0.2 Å by default, too large for stiff Cu–Cu bonds).
  Lower `max_disp`, or use HMC.
- **HMC** moves all atoms together and accepts 96% of trajectories.
  That is a sign that `hmc_dt` could be larger: aim for 60–90%.
- **MD** would need a `LammpsCalculator` or an `ASECalculatorAdapter`.
  With a pair-LJ calculator, `diffusion="md"` raises an error.

### 11.3 Checkpoints: resume, extend, and the safety checks

```python
from ase.build import fcc100
from regc import (MonatomicIdealGasReservoir, PairLJCalculator, ReplicaExchangeGCMC,
                     kelvin_to_eV, run_with_checkpoints, slab_geometry_from_atoms)

geometry = slab_geometry_from_atoms(fcc100("Cu", size=(3, 3, 2), a=3.6, vacuum=None,
                                           orthogonal=True), headroom=4.0)
pair_params = {("Cu", "H"): (0.15, 1.4), ("H", "H"): (0.01, 2.2)}
gas = MonatomicIdealGasReservoir(mass_amu=1.008)
T_K, p_bar = [300.0, 400.0, 500.0], [1e-3, 1e-1, 1e1, 1e3]

def build():
    return ReplicaExchangeGCMC(
        [kelvin_to_eV(T) for T in T_K], gas.mu_grid(T_K, p_bar, unit="bar"),
        lambda rng: geometry.make_system(PairLJCalculator(pair_params, r_cut_factor=1.7), "H"),
        seed=1, lambda_cubed_of_T=gas.lambda_cubed_of_T())

meta = {"system": "toy Cu(100)+H", "pressures_bar": p_bar}
common = dict(chunk=250, x0=0.9, diffusion="mc", diffusion_steps=2, record_every=10,
              metadata=meta, progress=None)

with build() as sim:                                   # first job: 500 steps
    st = run_with_checkpoints(sim, "ckpt_run", n_steps=500, **common)
print("first: ", st)

with build() as sim:                                   # later: extend the same run to 1000
    st = run_with_checkpoints(sim, "ckpt_run", n_steps=1000, **common)
print("extend:", st)

try:                                                   # a changed setup is refused
    with build() as sim:
        run_with_checkpoints(sim, "ckpt_run", n_steps=1500,
                             **dict(common, metadata=dict(meta, system="something else")))
except ValueError as err:
    print("refused:", err)

try:                                                   # an existing run is never overwritten silently
    with build() as sim:
        run_with_checkpoints(sim, "ckpt_run", n_steps=1000, resume=False, **common)
except FileExistsError as err:
    print("refused:", err)
```

```
first:  RunStatus(steps_done=500, finished=True, converged=False, rho_u_changes=[], resumed_from=0)
extend: RunStatus(steps_done=1000, finished=True, converged=False, rho_u_changes=[], resumed_from=500)
refused: cannot resume: 'system' is 'something else' now but 'toy Cu(100)+H' in ckpt_run/run_metadata.json
refused: ckpt_run/checkpoint.pkl exists: resume=True to continue it, or overwrite=True
```

What to notice:

- **Each job builds a fresh simulation.** `run_with_checkpoints` loads
  the saved state into it, including every random number generator, so
  the extended run continues exactly where the first stopped.
- **Put into `metadata`** everything that defines the physics: the
  potential file, lattice constant, D0, e_atom and the ladder. A resumed
  job with a different setup is then refused instead of silently mixing
  two studies.
- **`converged=False`** only means no convergence check was requested.
  Pass `convergence_every` to get `rho_u_changes`.

### 11.4 The analysis toolbox

The samples of §1, analysed in more ways:

```python
import numpy as np
from regc import MBARSurfaceAnalysis, MonatomicIdealGasReservoir, kelvin_to_eV, load_samples

gas = MonatomicIdealGasReservoir(mass_amu=1.008)
samples = load_samples("tutorial/samples.npz")
print("samples per replica:", samples.n_samples)
analysis = MBARSurfaceAnalysis(samples, reservoir=gas, pressure_unit="bar")
T = kelvin_to_eV(400.0)
n_surface = 9

# one point, with its statistical uncertainty
N, dN = analysis.mean_N_p(T, 1.0)
print(f"<N> at 400 K, 1 bar: {N:.2f} +- {dN:.2f}")

# an isotherm
p = np.geomspace(1e-3, 1e3, 7)
theta, dtheta = analysis.isotherm_p(T, p, area=n_surface)
print("isotherm 400 K:", np.round(theta, 2))

# the distribution P(N) at one point
N_values, prob = analysis.pN_p(T, 10.0)
print("P(N) at 400 K, 10 bar: most likely N =", N_values[np.argmax(prob)])

# phase maps: mean coverage, most probable N, and the aiAT comparison
T_grid = np.array([kelvin_to_eV(t) for t in (300, 400, 500)])
p_grid = np.geomspace(1e-3, 1e3, 4)
print("most probable N_ads:\n", analysis.most_probable_N_Tp(T_grid, p_grid, observable="N_ads"))
print("aiAT (lowest energy per N):\n", analysis.lowest_energy_N_Tp(T_grid, p_grid))

# convergence: does dropping the first half change the answer?
half = MBARSurfaceAnalysis(load_samples("tutorial/samples.npz", discard=0.5),
                           reservoir=gas, pressure_unit="bar")
a = analysis.phase_diagram_Tp(T_grid, p_grid, area=n_surface)
b = half.phase_diagram_Tp(T_grid, p_grid, area=n_surface)
print(f"largest coverage change when discarding half: {np.abs(a - b).max():.3f}")
```

```
samples per replica: 170
<N> at 400 K, 1 bar: 0.70 +- 0.04
isotherm 400 K: [0.   0.   0.01 0.08 0.53 0.87 1.33]
P(N) at 400 K, 10 bar: most likely N = 5.0
most probable N_ads:
 [[ 0  1  9 15]
 [ 0  0  5 10]
 [ 0  0  1  9]]
aiAT (lowest energy per N):
 [[ 0  9  9 16]
 [ 0  0  9 10]
 [ 0  0  0  9]]
largest coverage change when discarding half: 0.147
```

What to notice:

- **Mean and most probable N differ.** The mean is a smooth average.
  The most probable N jumps between phases.
- **aiAT vs REGC.** The aiAT construction keeps only the lowest energy
  per N, so it ignores entropy. Where it disagrees with REGC, finite
  temperature matters.
- **This run is not converged.** Dropping the first half changes the
  coverage by up to 0.15 per surface atom. A converged run changes by
  much less than the features you want to resolve. Here, run longer.
- **pymbar warnings.** pymbar may print warnings about "statistical
  inefficiency" or "divide by zero encountered in log". With short runs
  they are expected; treat them as a prompt to check convergence.

### 11.5 Two gas species

A toy feed of 20% H and 80% He over Cu(100). Each species has its own
reservoir. The ladder runs along the total pressure.

```python
import numpy as np
from ase.build import fcc100
from regc import (MBARSurfaceAnalysis, MonatomicIdealGasReservoir, PairLJCalculator,
                     ReplicaExchangeGCMC, kelvin_to_eV, load_samples, save_samples,
                     slab_geometry_from_atoms)

geometry = slab_geometry_from_atoms(fcc100("Cu", size=(3, 3, 2), a=3.6, vacuum=None,
                                           orthogonal=True), headroom=4.0)
pair_params = {("Cu", "H"): (0.15, 1.4), ("Cu", "He"): (0.06, 1.8),
               ("H", "H"): (0.01, 2.2), ("He", "He"): (0.005, 2.2), ("H", "He"): (0.005, 2.2)}
species = ["H", "He"]
gases = {"H": MonatomicIdealGasReservoir(1.008), "He": MonatomicIdealGasReservoir(4.0026)}
x = {"H": 0.2, "He": 0.8}                                  # feed mole fractions

T_K, p_bar = [300.0, 450.0], np.array([1e-2, 1.0, 1e2])
mus = np.stack([gases[s].mu_grid(T_K, x[s] * p_bar, unit="bar") for s in species], axis=-1)
print("mus shape:", mus.shape)                             # (n_T, n_p, 2)

with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], mus,
                         lambda rng: geometry.make_system(
                             PairLJCalculator(pair_params, r_cut_factor=1.7), species),
                         seed=2, lambda_cubed_of_T=[gases[s].lambda_cubed_of_T() for s in species]) as sim:
    sim.run_regc(n_steps=1500, x0=0.9, diffusion="mc", diffusion_steps=2, record_every=10,
                 burn_in_steps=200, verbose=False)
    save_samples(sim, "multi_samples.npz")

def mu_of_Tp(T, p):                                        # the same feed path, for MBAR
    return np.array([gases[s].mu_at_T_eV(T, x[s] * p, unit="bar") for s in species])

analysis = MBARSurfaceAnalysis(load_samples("multi_samples.npz"), mu_of_Tp=mu_of_Tp)
T = kelvin_to_eV(350.0)
means, errs = analysis.mean_N_species(T, mu_of_Tp(T, 10.0))
print(f"350 K, 10 bar total: <N_H> = {means[0]:.2f}, <N_He> = {means[1]:.2f}")
T_grid = np.array([kelvin_to_eV(t) for t in (300, 450)])
p_grid = np.geomspace(1e-2, 1e2, 3)
print("H per surface Cu:\n", np.round(analysis.phase_diagram_Tp(T_grid, p_grid, area=9, observable="N_H"), 2))
```

```
mus shape: (2, 3, 2)
350 K, 10 bar total: <N_H> = 3.90, <N_He> = 0.14
H per surface Cu:
 [[0.01 0.4  0.85]
 [0.   0.   0.35]]
```

What to notice:

- **Selectivity.** He is 80% of the feed but hardly adsorbs: it binds
  more weakly. Per-species observables (`N_H`, `N_He`) show what the
  total N would hide.
- **The same path for MBAR.** `mu_of_Tp` must describe the same feed
  the ladder was built from, returning one μ per species.

### 11.6 A production recipe: H on Fe with an EAM in LAMMPS

This is `examples/13_fe_h_eam_surface.py` in short form. Every quantity
comes from the potential itself. It needs LAMMPS and a cluster node:
36 replicas × 20,000 steps of 40 MD steps each.

```python
import numpy as np
from ase.build import bcc100
from regc import (DiatomicReservoir, LammpsCalculator, MBARSurfaceAnalysis, ReplicaExchangeGCMC,
                     cubic_lattice_constant, eam_pair_commands, kelvin_to_eV,
                     lammps_ase_calculator, load_samples, run_with_checkpoints,
                     slab_geometry_from_atoms)

species = ["Fe", "H"]
pair = eam_pair_commands("FeH_Wen.eam.fs", species)
ase_calc = lammps_ase_calculator(pair, species)               # for the setup only

a = cubic_lattice_constant(ase_calc, "Fe", "bcc")             # 2.855 A for this potential
slab = bcc100("Fe", size=(3, 3, 4), a=a, vacuum=None, orthogonal=True)
geometry = slab_geometry_from_atoms(slab, n_frozen_layers=2, headroom=5.0)
h2 = DiatomicReservoir.from_calculator(ase_calc, "H", d0="self")   # D0 = 4.458 eV here

T_K = list(np.geomspace(300, 900, 6))
p_bar = list(np.logspace(-8, 4, 6))

def make_system(rng):
    return geometry.make_system(LammpsCalculator(pair, species, masses={"H": 1.008}), "H",
                                mass_of_species={"H": 1.008})

with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in T_K], h2.mu_grid(T_K, p_bar, unit="bar"),
                         make_system, seed=0, n_workers=36, max_disp=0.3,
                         lambda_cubed_of_T=h2.lambda_cubed_of_T(), adsorbed_cutoff=2.5) as sim:
    status = run_with_checkpoints(sim, "run_fe100", n_steps=20000, chunk=1000, x0=0.9,
                                  diffusion="md", diffusion_steps=40, record_every=10,
                                  convergence_every=2000,
                                  metadata={"a": a, "D0_eV": h2.D0_eV, "e_atom_eV": h2.e_atom_eV,
                                            "n_surface": geometry.n_surface},
                                  timestep_fs=0.5, thermostat="bussi", taut_fs=20.0)

samples = load_samples("run_fe100/samples.npz", discard=0.2)
analysis = MBARSurfaceAnalysis(samples, reservoir=h2, pressure_unit="bar")
T_grid_K = np.linspace(300, 900, 120)
p_grid = np.logspace(-8, 4, 150)
T_grid = [kelvin_to_eV(T) for T in T_grid_K]
theta = analysis.phase_diagram_Tp(T_grid, p_grid, area=geometry.n_surface, observable="N_ads")
n_star = analysis.most_probable_N_Tp(T_grid, p_grid, observable="N_ads")
```

To analyse in a later session without the calculator, rebuild the
reservoir from the saved numbers:
`DiatomicReservoir.hydrogen(D0_eV=meta["D0_eV"], e_atom_eV=meta["e_atom_eV"])`.

---

# Part II: the standalone program (YAML)

The `regc-run` command (installed with the package) runs the whole
pipeline from one config file: structure → calculator → reservoir →
REGC → MBAR → plot. Configs can be YAML (`.yaml`/`.yml`, needs PyYAML)
or JSON (`.json`).

## 12. A first run from a config file

This is the study of [§1](#1-a-first-run-in-20-seconds) as a config file.
Save it as `tutorial.yaml`:

```yaml
structure:
  build: {surface: fcc100, symbol: Cu, size: [3, 3, 2], a: 3.6, vacuum: null, orthogonal: true}
  headroom_angstrom: 4.0

gas:
  species: H
  mass_amu: 1.008

calculator:
  type: PairLJ
  pair_params: [[Cu, H, 0.15, 1.4], [H, H, 0.01, 2.2]]
  r_cut_factor: 1.7

reservoir:
  type: monatomic

replica_exchange:
  temperatures_K: [300, 400, 500]
  pressures: [1.0e-3, 1.0e-1, 1.0e+1, 1.0e+3]
  pressure_unit: bar
  scheme: regc
  n_steps: 2000
  x0: 0.9
  diffusion: mc
  diffusion_steps: 2
  record_every: 10
  burn_in_steps: 300
  adsorbed_cutoff: 2.5
  seed: 1

phase_diagram:
  T_min_K: 300
  T_max_K: 500
  n_T: 21
  p_min: 1.0e-3
  p_max: 1.0e+3
  n_p: 31
  observable: N_ads
  coverage_per: surface_atom

output:
  directory: tutorial_yaml
```

First check the setup. This builds everything and prints the μ ladder,
but samples nothing:

```
$ regc-run tutorial.yaml --check
slab: 18 frozen + 0 displaceable slab atoms, 9 per surface layer, cell |a1| = 7.637 A, |a2| = 7.637 A (rectangular), mobile species in z in (0, 4.00] A
H reservoir (MonatomicIdealGasReservoir): e_atom = 0.0000 eV
replica grid: 3 T x 4 p = 12 replicas

mu_H (eV): rows T (K), columns p (bar)
         1.00e-03  1.00e-01  1.00e+01  1.00e+03
   300     -0.453    -0.334    -0.215    -0.096
   400     -0.629    -0.470    -0.311    -0.153
   500     -0.810    -0.612    -0.413    -0.215

bare-slab energy: 0.0000 eV (PAIRLJ calculator works)
```

Then run it:

```
$ regc-run tutorial.yaml
...
Saved phase diagram to tutorial_yaml/phase_diagram.png
$ ls tutorial_yaml
phase_diagram_data.npz  phase_diagram.png  run_metadata.json  samples.npz
```

Two annotated example configs cover every key:

- `examples/pd111_emt_example.yaml`: Pd(111) + H with EMT, read from a
  file. Small; runs in minutes.
- `examples/fe100_h_eam_example.yaml`: Fe(100) + H with the Wen EAM in
  LAMMPS, with displaceable layers, MD diffusion and checkpoints. The
  production setup of example 13.

### 12.1 More config files

Each of these was run as shown.

**Two gas species with a feed composition.** This is the YAML version
of §11.5. `gas.species` is a list, `gas.composition` gives the feed
mole fractions, and the `reservoir` section is keyed by species. The
pressure ladder and the phase diagram are in *total* pressure.

```yaml
structure:
  build: {surface: fcc100, symbol: Cu, size: [3, 3, 2], a: 3.6, vacuum: null, orthogonal: true}
  headroom_angstrom: 4.0

gas:
  species: [H, He]
  composition: {H: 0.2, He: 0.8}          # feed mole fractions

calculator:
  type: PairLJ
  pair_params: [[Cu, H, 0.15, 1.4], [Cu, He, 0.06, 1.8], [H, H, 0.01, 2.2],
                [He, He, 0.005, 2.2], [H, He, 0.005, 2.2]]
  r_cut_factor: 1.7

reservoir:                                 # keyed by species
  H: {type: monatomic}
  He: {type: monatomic}

replica_exchange:
  temperatures_K: [300, 450]
  pressures: [1.0e-2, 1.0, 1.0e+2]         # total pressure of the feed
  pressure_unit: bar
  scheme: regc
  n_steps: 1500
  x0: 0.9
  diffusion: mc
  diffusion_steps: 2
  record_every: 10
  burn_in_steps: 200
  seed: 2

phase_diagram:
  T_min_K: 300
  T_max_K: 450
  n_T: 16
  p_min: 1.0e-2
  p_max: 1.0e+2
  n_p: 21
  observable: N_H                          # H only; N for both together
  coverage_per: surface_atom

output:
  directory: multi_out
```

It finishes in about 7 seconds and prints
`<N_H> per surface_atom: 3.144e-05 to 0.7343`.

**The sweep scheme with an explicit μ ladder.** Without `pressures`
there is no reservoir to build (unless a `phase_diagram` asks for one),
and without a `phase_diagram` section the run only samples.

```yaml
structure:
  build: {surface: fcc100, symbol: Cu, size: [3, 3, 2], a: 3.6, vacuum: null, orthogonal: true}
  headroom_angstrom: 4.0

gas:
  species: H

calculator:
  type: PairLJ
  pair_params: [[Cu, H, 0.15, 1.4], [H, H, 0.01, 2.2]]
  r_cut_factor: 1.7

replica_exchange:
  temperatures_K: [300, 400, 500]
  mu_eV: [-0.6, -0.45, -0.3, -0.15]        # explicit mu, shared by every temperature
  scheme: sweep
  n_cycles: 60
  sweeps_per_cycle: 20
  exchange_every: 1
  burn_in_cycles: 10
  record_every: 2
  p_displace: 0.5
  p_insert: 0.25
  p_delete: 0.25
  seed: 4

output:
  directory: sweep_out
```

It ends with `Done (no phase_diagram section: sampling only). Outputs in sweep_out/`.
To analyse it, load `sweep_out/samples.npz` in Python and use
`MBARSurfaceAnalysis.phase_diagram(T_grid, mu_grid)` on a (T, μ) mesh
(§9).

**Any ASE calculator class, with checkpoints and trajectories.** The
calculator is imported by its dotted path and built inside each worker.
The structure file is found relative to the config file.

```yaml
structure:
  file: pd111_slab.xyz                     # relative to this file
  headroom_angstrom: 5.0

gas:
  species: H
  insertion_z_range_above_surface: [1.0, 3.0]

calculator:
  type: ASE
  class: ase.calculators.emt.EMT           # any importable ASE calculator class
  kwargs: {}                               # passed to the class

reservoir:
  type: diatomic
  d0: self

replica_exchange:
  temperatures_K: [350, 450]
  pressures: [1.0e-4, 1.0, 1.0e+4]
  pressure_unit: atm
  scheme: regc
  n_steps: 200
  x0: 0.9
  diffusion: mc
  diffusion_steps: 1
  record_every: 5
  n_workers: 2
  seed: 3

checkpoint:
  chunk: 100

trajectory:
  save: true
  every: 50

phase_diagram:
  T_min_K: 350
  T_max_K: 450
  n_T: 6
  p_min: 1.0e-4
  p_max: 1.0e+4
  n_p: 9
  coverage_per: surface_atom
  most_probable: true
  discard: 0.2

output:
  directory: ase_out
```

Its progress lines report the swap acceptance per pattern:

```
     100/200 steps         7 s  mean N =    8.00  swap acc: T 0.50, mu 0.00, cross_left 0.00, cross_right 1.00
     200/200 steps        28 s  mean N =   14.50  swap acc: T 0.50, mu 0.00, cross_left 0.00, cross_right 1.00
```

The pressure ladder here jumps four decades between replicas, and the
μ swaps are never accepted (`mu 0.00`): the replicas do not overlap.
For a real run, use one decade per replica or less (§19). The output
directory then holds `checkpoint.pkl`, `samples.npz`,
`run_metadata.json`, both plots, `phase_diagram_data.npz` and
`trajectories/replica_T*_mu*.xyz`, with one frame every 50 steps.

## 13. Command-line reference

```
regc-run CONFIG [--check | --analyze-only] [--resume] [--overwrite]
                   [--time-limit SECONDS] [--workers N] [-q]
```

| Option | Effect |
|---|---|
| *(none)* | Set up, sample, write samples and metadata, then analyse if there is a `phase_diagram` section. |
| `--check` | Build the setup and print the lattice constant, slab, reservoir (D0, e_atom), the μ ladder and the bare-slab energy. Samples nothing. Use it before every new run. |
| `--analyze-only` | Skip sampling. Recompute the phase diagram from `samples.npz` and `run_metadata.json` in `output.directory`, for example after changing the mesh, `observable`, `coverage_per` or `discard`. |
| `--resume` | Continue a checkpointed run (needs a `checkpoint` section). Refuses a checkpoint whose setup differs from the config. |
| `--overwrite` | Start afresh in an output directory that already holds a checkpointed run. |
| `--time-limit S` | Stop cleanly before a chunk that would end after S seconds. Overrides `checkpoint.time_limit_s`. |
| `--workers N` | Override `replica_exchange.n_workers`. |
| `-q`, `--quiet` | No progress output. |

**Exit codes:** 0 when finished. 75 (`EXIT_RESUBMIT`) when a
checkpointed run stopped for the time limit: resubmit with `--resume`.
Any other non-zero code is an error.

`python examples/09_config_driven_run.py CONFIG ...` is the same
program.

## 14. Config file reference

General rules:

- **Unknown keys are an error,** and so are keys that belong to the
  other scheme. A typo cannot silently do nothing.
- **Exponents:** write them with a sign: `1.0e+6`, not `1.0e6`. PyYAML
  reads `1.0e6` as a string. The runner converts it back, but don't rely
  on that.
- **Relative paths:** `structure.file`, `calculator.potential`, files
  named in `pair_coeff` / `pair_commands`, and string values in
  `calculator.kwargs` are resolved against the config file's directory
  when the file exists there. `output.directory` is relative to the
  directory you run from.
- **Required sections:** `structure`, `gas` and `replica_exchange`. The
  others are optional.

### 14.1 `structure`

Give exactly one of `file` or `build`.

| Key | Default | Meaning |
|---|---|---|
| `file` | – | Any ASE-readable slab file. a1 along x, a2 in the xy plane. |
| `build` | – | A mapping for an `ase.build` surface function (below). |
| `n_frozen_layers` | all layers | Bottom layers kept rigid; the rest are displaceable. |
| `headroom_angstrom` | 5.0 | Space above the topmost atom. |
| `layer_tol` | 0.001 | z tolerance (Å) for grouping atoms into layers. |

`build` keys:

| Key | Meaning |
|---|---|
| `surface` | The `ase.build` function: `fcc111`, `fcc100`, `fcc110`, `bcc100`, `bcc110`, `hcp0001`, ... |
| `symbol` | Element. |
| `size` | `[nx, ny, nlayers]` |
| `a` | Lattice constant in Å, or `auto` to fit it to the potential (fcc and bcc only; needs a calculator with an ASE counterpart: EMT, LAMMPS or ASE). |
| `a_range` | Search range for `a: auto`; default `[2.3, 5.5]`. |
| *anything else* | Passed to the `ase.build` function, e.g. `orthogonal: true`, `vacuum: null`, `c`. |

### 14.2 `gas`

| Key | Default | Meaning |
|---|---|---|
| `species` | – | The exchanged element, or a list such as `[O, Cl]`. |
| `mass_amu` | ASE tables | Atomic mass: a number, or a mapping per species. |
| `insertion_z_range_above_surface` | whole box | `[lo, hi]` in Å above the topmost slab atom. Insertions and deletions happen only there. It must lie inside the box; raise `headroom_angstrom` if not. |
| `composition` | – | Feed mole fractions per species, e.g. `{O: 0.999, Cl: 0.001}`. Required with several species when pressures are used. |

### 14.3 `calculator`

`type` selects the potential. The default is `EMT`.

| `type` | Keys | Notes |
|---|---|---|
| `EMT` | – | ASE's EMT. Fine for testing; its H description is not physical. |
| `LJ` | `lj_sigma` (1.0), `lj_epsilon` (0.1), `lj_r_cut` (3.0) | Ignores the slab entirely; pipeline tests only. A warning is printed. |
| `PairLJ` | `pair_params`: list of `[a, b, epsilon_eV, sigma_A]`; `r_cut_factor` (2.5) | Pairwise LJ that sees the slab. |
| `LAMMPS` | `potential` (an `*.eam.fs` / `*.eam.alloy` file), **or** `pair_style` + `pair_coeff`, **or** `pair_commands` (list of lines); `species` (atom-type order; default slab element(s) then gas); `min_distance` (Å hard core); `z_padding` (5.0) | Needs the LAMMPS Python module. Native MD for `diffusion: md`. |
| `ASE` | `class`: dotted import path, e.g. `ase.calculators.eam.EAM`; `kwargs`: mapping passed to it | Any ASE calculator. Built in each worker, so `n_workers > 1` works. |

```yaml
calculator: {type: LAMMPS, potential: FeH_Wen.eam.fs, species: [Fe, H]}

calculator:
  type: LAMMPS
  pair_style: pace
  pair_coeff: "* * FeH.yace Fe H"
  species: [Fe, H]
  min_distance: 0.6

calculator:
  type: ASE
  class: ase.calculators.eam.EAM
  kwargs: {potential: FeH_Wen.eam.fs}
```

### 14.4 `reservoir`

For one species, write the keys directly under `reservoir`. With several
species, key the section by species: `reservoir: {O: {...}, Cl: {...}}`.

| Key | Default | Meaning |
|---|---|---|
| `type` | `diatomic` | `diatomic` (atoms from X2 gas) or `monatomic`. |
| `d0` | `self` | Diatomic: `self` (the potential's molecule), `experimental` (H2 only, 4.478 eV) or a number in eV. |
| `D0_eV` | – | Same as giving `d0` a number. Required for `LJ` and `PairLJ`, which have no ASE counterpart to evaluate the molecule. |
| `e_atom_eV` | from the calculator | Isolated-atom energy, the zero of μ. |
| `bond_length_angstrom`, `vibrational_wavenumber_cm` | H2 values for H | Required for other elements. |
| `symmetry_number` | 2 | Rotational symmetry number. |
| `mass_molecule_amu` | 2 × atomic mass | Molecular mass. |
| `mass_amu` | gas mass | Monatomic: the atom's mass. |

The reservoir is built when the ladder uses `pressures`, or when there
is a `phase_diagram` or `reservoir` section.

### 14.5 `replica_exchange`

The ladder: give `temperatures_K` and **exactly one** of `pressures` or
`mu_eV`.

| Key | Default | Meaning |
|---|---|---|
| `temperatures_K` | – | Temperature ladder. |
| `pressures` | – | Pressure ladder; μ comes from the reservoir at every (T, p). |
| `pressure_unit` | `bar` | `Pa`, `bar` or `atm`. |
| `mu_eV` | – | Explicit μ: a list (shared by all T) or one row per T. |
| `scheme` | `regc` if `n_steps` is given, else `sweep` | Sampling scheme. |
| `seed` | 0 | Master seed. |
| `max_disp` | 0.3 | MC displacement, Å. |
| `n_workers` | 1 | Worker processes, or `auto` for all cores. |
| `adsorbed_cutoff` | – | Record `N_ads` with this cutoff (Å). |
| `record_every` | 100 (regc), 5 (sweep) | Steps or cycles between samples. |
| `hmc` | – | `{leapfrog_steps: 10, timestep_fs: 0.5}` (or `dt` in internal units); `enabled: true` turns on HMC in the sweep scheme. |

`scheme: regc` keys:

| Key | Default | Meaning |
|---|---|---|
| `n_steps` | – | REGC steps (the maximum with convergence checks). |
| `x0` | 0.99 | Probability of a grand-canonical step. |
| `diffusion` | `mc` | `mc`, `mc_moves`, `hmc` or `md`. |
| `diffusion_steps` | 10 | Units of diffusion per step. |
| `burn_in_steps` | 0 | Steps before recording. Not allowed together with checkpoint chunks; use `phase_diagram.discard`. |
| `convergence_every`, `convergence_tol` | off, 0.05 | ρ(u) convergence checks. |
| `swap_types` | all four | Subset of `[T, mu, cross_left, cross_right]`. |
| `displace_substrate` | false | With `mc`: move displaceable substrate atoms too. |
| `md` | – | `{timestep_fs: 1.0, thermostat: bussi, taut_fs: 20.0, friction_per_fs: 0.01}` for `diffusion: md`. |

`scheme: sweep` keys: `n_cycles`, `sweeps_per_cycle` (100),
`exchange_every` (1), `burn_in_cycles` (0), `diagonal_swaps` (true),
`p_displace` (0.5), `p_insert` (0.25), `p_delete` (0.25),
`p_displace_substrate` (0).

Some runner defaults differ from the library's: `max_disp` 0.3 here,
0.2 in `ReplicaExchangeGCMC`. Set them explicitly if you compare the
two.

### 14.6 `checkpoint` (regc only)

| Key | Default | Meaning |
|---|---|---|
| `chunk` | `n_steps` | Steps between checkpoints. |
| `time_limit_s` | – | Stop cleanly before this many seconds. `--time-limit` overrides it. |

With this section the run goes through `run_with_checkpoints`
([§7](#7-long-runs-checkpoints-and-batch-jobs)): `checkpoint.pkl` and
`samples.npz` are written after every chunk. Use `--resume` and
`--time-limit`.

### 14.7 `trajectory`

| Key | Default | Meaning |
|---|---|---|
| `save` | false | Write one extended-XYZ file per replica. |
| `directory` | `trajectories` | Under `output.directory`. |
| `every` | `record_every` | Steps (regc) or cycles (sweep) between frames. |

### 14.8 `phase_diagram`

| Key | Default | Meaning |
|---|---|---|
| `T_min_K`, `T_max_K`, `n_T` | –, –, 15 | Temperature mesh (linear). |
| `p_min`, `p_max`, `n_p` | –, –, 40 | Pressure mesh (geometric). |
| `pressure_unit` | the ladder's unit, else `bar` | Unit of `p_min` / `p_max`. |
| `observable` | `N` | `N`, `N_ads` (needs `adsorbed_cutoff`) or `N_<species>`. |
| `coverage_per` | `area` | `area` (per Å²), `surface_atom` (per top-layer atom) or `none` (absolute count). |
| `most_probable` | false | Also compute and plot the most probable N (`<plot>_most_probable.png`). |
| `discard` | 0.0 | Fraction of each replica's samples dropped as equilibration. |

This section is checked before sampling starts, so a typo here does not
cost you the run.

### 14.9 `output`

| Key | Default | Meaning |
|---|---|---|
| `directory` | `output` | Where everything is written. |
| `plot_filename` | `phase_diagram.png` | The coverage plot. |
| `save_raw_data` | true | Write `phase_diagram_data.npz`. |

### 14.10 Older configs

Configs from earlier versions still run:

| Old key | Current key |
|---|---|
| `diatomic_reservoir` | `reservoir` with `type: diatomic` |
| `phase_diagram.p_min_atm` / `p_max_atm` | `p_min` / `p_max` (the unit defaults to atm when the old keys are used) |

The scheme is still inferred from `n_steps` (regc) or `n_cycles` (sweep).
One key must be renamed by hand: `structure.extra_vacuum_angstrom` is now
`headroom_angstrom`, and the old name is rejected as unknown. Two
behaviours changed:

- The slab is now laid out by `slab_geometry_from_atoms`: the top frozen
  layer sits at z = 0.
- Without `insertion_z_range_above_surface`, insertions use the whole
  box.

## 15. Output files

Everything goes to `output.directory`:

| File | Contents |
|---|---|
| `samples.npz` | Per replica: U, N, N_ads, N_s, plus the ladder, μ grid and Λ³. Readable with `load_samples`. |
| `run_metadata.json` | The setup: species, masses, temperatures, μ grid, pressures and unit, composition, reservoirs (D0, e_atom, ...), structure, calculator, lattice constant, `n_surface`, area, insertion window and sampling settings. `--analyze-only` and `--resume` rely on it. |
| `checkpoint.pkl` | With a `checkpoint` section: the full simulation state. |
| `phase_diagram.png` | The coverage map. |
| `phase_diagram_most_probable.png` | With `most_probable: true`. |
| `phase_diagram_data.npz` | `T_K`, `p`, `pressure_unit`, `p_atm`, `theta` (n_T × n_p), `observable`, `coverage_per`, and `most_probable_N` if computed. |
| `trajectories/replica_T*_mu*.xyz` | With `trajectory.save`. |

To replot or post-process:

```python
d = np.load("output/phase_diagram_data.npz")
theta, T_K, p = d["theta"], d["T_K"], d["p"]
```

## 16. Batch jobs with the runner

A job that resubmits itself until the run is complete:

```bash
#!/bin/bash
#SBATCH --cpus-per-task=36
#SBATCH --time=24:00:00
set -uo pipefail
regc-run fe100.yaml --resume --workers "$SLURM_CPUS_PER_TASK" --time-limit 85800
rc=$?
if [ $rc -eq 75 ]; then sbatch "$0"; exit 0; fi      # not finished: next job continues
exit $rc
```

- **The first job:** `--resume` starts a fresh run when there is no
  checkpoint yet, so the same script works for every job.
- **The time limit** leaves a margin for the last chunk and the
  analysis: here, 10 minutes off a 24 h limit.
- **Size a chunk** so that it takes a few minutes, not hours.
- **One worker per replica** is the most that helps.
- **A fuller version:** `hpc/run.sbatch` reads the remaining wall time
  from Slurm and caps the number of resubmissions (it drives example
  13, but the pattern is the same).

## 17. Calling the runner from Python

The runner is an ordinary module. A config is just a dict:

```python
from regc.runner import (load_config, validate_config, setup_from_config,
                            run_from_config, analyze_from_config, check_config)

config = load_config("tutorial.yaml")        # YAML or JSON; resolves relative paths
config["replica_exchange"]["n_steps"] = 5000  # edit freely
result = run_from_config(config)              # verbose=True, resume=False, overwrite=False,
                                              # time_limit=None, n_workers=None
print(result["finished"], result["theta_grid"].shape)
```

`run_from_config` returns a dict with:

- `sim`, `setup` and `status`;
- `finished`, `D0_eV` and `e_atom_eV` (per species with several);
- the analysis results, when there is a `phase_diagram` section:
  `analysis`, `theta_grid`, `T_grid_K`, `p_grid`, `pressure_unit`,
  `p_grid_atm`, `coverage_per`, `observable`, and `most_probable_N`.

The other functions:

- `setup_from_config(config)` builds everything but samples nothing. It
  returns the geometry, system factory, μ grid, Λ³ and metadata, so you
  can take over from there.
- `analyze_from_config(config)` re-runs only the analysis.
- `check_config(config)` prints what `--check` prints.
- `examples/08_reasonable_Tp_phase_diagram.py` drives the runner with
  an inline dict.

---

# Part III: practice

## 18. A workflow for a new system

1. **Check the potential.**
   - The lattice constant: `cubic_lattice_constant` or `a: auto`.
   - The molecule: `reservoir.calculator_info` or `--check`. If its D0
     is far from experiment, use `d0: experimental` or a corrected
     value.
   - Adsorption energies on the bare slab: `insertion_energy_scan`.
2. **Choose the window.** Set T and p to cover the conditions you care
   about. Look at the μ ladder `--check` prints: the range of μ should
   span empty to saturated coverage.
3. **Run small first.** Use a small cell and short runs, check the
   swap acceptance (aim for about 20–50% per pattern), and look at the
   trajectories.
4. **Scale up.** Use the production cell, `diffusion: md` or `hmc` for
   relaxing substrates, checkpoints, and `n_workers` equal to the number
   of replicas.
5. **Converge.** Use `convergence_every`. Re-analyse with increasing
   `discard`, and repeat with a different `seed`: θ(T, p) should not
   change.
6. **Read the map critically.** Use `N_ads` when the headroom holds
   gas. Distrust regions outside the sampled window. Low temperatures
   converge slowest.

## 19. Choosing settings

These are starting points; the convergence checks of §18 decide.

### The cell

- **Large enough for the phases.** A phase with 7/8 coverage needs a
  cell with a multiple of 8 surface sites to exist. Small cells allow
  only the coverages they can tile.
- **Large enough for the cutoff.** Every pair cutoff must stay below
  half the shortest cell width.
- **Layers.** Two frozen bottom layers and one or two relaxing top
  layers are a common choice for metals.
- **Headroom.** 4–6 Å above the top atom. More only adds gas-phase atoms
  and cost; restrict insertions with `insertion_z_range` /
  `insertion_z_range_above_surface` if needed.

### The (T, p) ladder

- **Cover the window you want to report.** MBAR interpolates well
  between replicas and extrapolates poorly beyond them.
- **Pressure spacing.** One decade or less between replicas for atoms
  from a diatomic gas (0.03 eV in μ at 300 K, more at higher T). Denser
  near expected transitions. Check `swap_acceptance()["mu"]`.
- **Temperature spacing.** Geometric, for example 6 temperatures over a
  factor of 3. Low temperatures need closer spacing. Check
  `swap_acceptance()["T"]`.
- **Size.** n_T × n_p replicas cost n_T × n_p times one replica; with
  `n_workers` equal to the number of replicas, the wall time is about
  that of one replica. 6 × 6 = 36 replicas fits a typical cluster node.
- **Target acceptance:** 20–50% per swap pattern. [ZSG] report about
  25%.

### The REGC step

| Setting | Start with | Why |
|---|---|---|
| `x0` | 0.9 | 10% swap steps. [ZSG] use 0.99 for cheap LJ runs and 0.9 for Si clusters. |
| `diffusion` | `"md"` with LAMMPS, `"hmc"` with forces, `"mc"` otherwise | Collective moves relax the substrate and adsorbate layer far faster than single-atom MC. |
| `diffusion_steps` | MD: 20–50 steps; HMC: 1–5 trajectories; MC: 1–10 cycles | Enough motion between grand-canonical moves to decorrelate. |
| MD `timestep_fs` | 0.5 with H, 1–2 otherwise | MD carries time-step bias; H vibrates fast. |
| `hmc_dt`, `hmc_leapfrog_steps` | 0.5–1 fs (divided by 10.1805), 10 steps | Aim for 60–90% HMC acceptance. |
| `max_disp` | 0.2–0.3 Å | Aim for 30–50% displacement acceptance. |
| `record_every` | 10 | Samples closer than a few steps are strongly correlated. |
| `n_steps` | 10⁴–10⁵ | Until the convergence checks pass. |
| `convergence_every` | `n_steps / 10` | ρ(u) changes below `convergence_tol` = 0.05 stop the run. |

### Timing

Measure the cost of one REGC step per replica. `examples/13_fe_h_eam_surface.py --check`
times `Replica.regc_local_step` on one replica. With the runner, time a
short run (a few hundred steps, one worker per replica). Multiply by
`n_steps` for the wall time with one worker per replica. The cost grows with coverage, because
there are more atoms, so time the high-pressure, low-temperature
corner.

### Analysis

- **Discard** the first 10–30% of samples (`discard=`). Increase it
  until the map stops changing.
- **Mesh:** 100–200 points per axis is cheap. MBAR evaluation dominates
  only for very large meshes.
- **Observable:** `N_ads` whenever the headroom holds gas atoms, or
  when you compare with experiments that count adsorbed species.

## 20. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `interaction cutoff ... exceeds the minimum-image radius ...` | A pair cutoff exceeds half the cell width. | Lower `r_cut` / `r_cut_factor`, or use a larger cell. |
| `ValueError: ... a1 along x ...` | The cell is not in surface form. | Use an `ase.build` slab, or rotate the cell so a1 ∥ x and a2 lies in xy. |
| `Can't pickle local object 'EMT.initialize...'` | A live ASE calculator with `n_workers > 1`. | Pass a factory to `ASECalculatorAdapter` (the runner does this). |
| `FileExistsError` | The output directory already holds a checkpointed run. | `--resume` to continue, or `--overwrite` to start afresh. |
| `ValueError` on resume | The config or metadata changed since the checkpoint. | Restore the setup, or start a new run in another directory. |
| `unknown key(s) [...] in config section ...` | A typo, or a key of the other scheme. | Check §14. |
| `reservoir D0_eV for H must be given explicitly` | `d0: self` with `LJ`/`PairLJ`, which cannot evaluate a molecule. | Give `D0_eV`, or `d0: experimental` for H. |
| `... energy has no minimum for a in [...]` | The lattice constant is outside `a_range`. | Set `structure.build.a_range` (or `a_range=` in Python). |
| All pressures give an empty or full surface | The μ window misses the transition, often because e_atom is wrong. | Compare the μ table from `--check` with the adsorption energies. |
| A swap pattern has acceptance near 0 | Neighbouring replicas do not overlap. | Use more, closer temperatures or pressures. |
| Coverage of many atoms per surface atom with EMT | EMT over-binds H. | Use a real EAM or a machine-learned potential. |
| `1.0e6` read as text | A PyYAML quirk. | Write `1.0e+6`. |
| `diffusion='md' ... needs an ASECalculatorAdapter` | MD with `LJ` or `PairLJ`. | Use `mc` or `hmc`, or a LAMMPS/ASE calculator. |
| `diffusion='hmc'` or HMC raises a forces error | The calculator has no forces. | Use `mc`, or a calculator with forces. |
| pymbar warns about "statistical inefficiency" or "divide by zero encountered in log" | Short or poorly overlapping runs. | Warnings, not errors. Check convergence (`discard`, a second seed) and run longer. |
| The map changes when `discard` grows | The run is not equilibrated. | Run longer, improve diffusion, or tighten the ladder. |
| Coverage jumps at the edge of the map | MBAR extrapolating beyond the sampled window. | Extend the ladder, or restrict the map to the sampled window. |
| HMC acceptance near 100% or near 0% | `hmc_dt` too small or too large. | Aim for 60–90%. |
| `swap_types` / `md` / `n_cycles` rejected in YAML | A key of the other scheme. | Check `scheme`; each scheme has its own keys. |
| The run stops with exit code 75 | The time limit was reached, as intended. | Resubmit with `--resume`. |
