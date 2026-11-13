# regc API reference

Generated from the docstrings of regc by `docs/build_api_reference.py`; do not edit by hand. For how to use these pieces together, see [MANUAL.md](MANUAL.md).

Everything below is importable as `from regc import <name>`, except the runner functions at the end.

## Contents

- [Calculators (`regc.calculators`)](#calculators-regccalculators): `Calculator`, `LJCalculator`, `PairLJCalculator`, `ASECalculatorAdapter`
- [LAMMPS backend (`regc.lammps_backend`)](#lammps-backend-regclammps_backend): `LammpsCalculator`, `eam_pair_commands`, `lammps_ase_calculator`, `open_lammps`
- [Systems (`regc.systems`)](#systems-regcsystems): `ParticleSystem`, `BulkSystem`, `SlabSystem`, `ClusterSystem`, `LJSystem`, `SlabLJSystem`
- [Structures and slab geometry (`regc.structures`)](#structures-and-slab-geometry-regcstructures): `cubic_lattice_constant`, `slab_geometry_from_atoms`, `SlabGeometry`, `insertion_energy_scan`, `cluster_geometry_from_atoms`, `ClusterGeometry`, `coordination_numbers`, `reference_lattice_constant`
- [Thermodynamic state (`regc.state`)](#thermodynamic-state-regcstate): `ThermoState`
- [Replicas (`regc.replica`)](#replicas-regcreplica): `Replica`
- [The replica-exchange engine (`regc.engine`)](#the-replica-exchange-engine-regcengine): `ReplicaExchangeGCMC`, `SWAP_TYPES`
- [Checkpointed runs (`regc.driver`)](#checkpointed-runs-regcdriver): `run_with_checkpoints`, `RunStatus`, `EXIT_RESUBMIT`
- [Sample I/O (`regc.io`)](#sample-io-regcio): `save_samples`, `load_samples`, `SampleSet`
- [Units (`regc.units`)](#units-regcunits): `kelvin_to_eV`, `eV_to_kelvin`, `pascal_per`, `PRESSURE_UNITS`, `ThermalWavelengthCubed`, `SpeciesLambdaCubed`, `thermal_wavelength_angstrom`, `thermal_wavelength_cubed_angstrom3`, `default_mass_amu`
- [Reduced-unit pressure conventions (`regc.pressure`)](#reduced-unit-pressure-conventions-regcpressure): `p0_reduced_units`, `p0_ideal_gas_wavelength`, `mu_from_pressure`, `pressure_from_mu`, `mu_grid_from_pressures`, `invert_mu_of_Tp`
- [Gas reservoirs (`regc.reservoir`)](#gas-reservoirs-regcreservoir): `Reservoir`, `MonatomicIdealGasReservoir`, `DiatomicReservoir`, `diatomic_dissociation_energy_eV`, `diatomic_gas_mu_eV`, `mu_atomic_from_diatomic_reservoir_eV`, `pressure_from_atomic_mu_reservoir_pascal`, `H2_BOND_LENGTH_ANGSTROM`, `H2_VIBRATIONAL_WAVENUMBER_CM`, `H_MASS_AMU`, `H2_MASS_AMU`, `H2_D0_EXPERIMENTAL_EV`
- [MBAR analysis (`regc.analysis`)](#mbar-analysis-regcanalysis): `MBARSurfaceAnalysis`, `heat_capacity_ridges`, `export_phase_configurations`
- [Plotting (`regc.plotting`)](#plotting-regcplotting): `plot_phase_diagram`, `plot_map`, `plot_radial_distribution`
- [Per-sample records and other potentials (`regc.records`)](#per-sample-records-and-other-potentials-regcrecords): `RecordWriter`, `Records`, `attach_energies`, `reevaluate`
- [Config runner (`regc.runner`)](#config-runner-regcrunner): `load_config`, `validate_config`, `setup_from_config`, `run_from_config`, `analyze_from_config`, `check_config`, `reevaluate_from_config`, `main`, `reevaluate_main`

## Calculators (`regc.calculators`)

### `Calculator`

```python
Calculator(*args, **kwargs)
```

Energy-evaluation backend. `ParticleSystem` delegates all physics to
one of these; everything else (geometry, boundaries, GCMC bookkeeping)
is completely agnostic to what potential is used.

`supports_incremental = True` means the calculator also implements
`pair_energy_of(system, pos, exclude=None)`, the energy of a single
point's interaction with the rest of the system -- unlocking the fast
incremental single-particle GCMC path. Leave it False (the default) for
any potential where that isn't available or safe to hand-derive; Replica
then uses the generic clone-and-recompute path automatically, which is
always correct.

`supports_forces = True` means the calculator also implements
`forces(system)`, returning the force (eV/Angstrom, or the matching
reduced-unit force, per system) on each MOBILE particle, in
`system.positions` order -- what `Replica`'s HMC move needs (see
`replica.py`). Leave it False for any potential with no cheap/available
force evaluation; HMC then just isn't available for that calculator.

#### `Calculator.total_energy`

```python
Calculator.total_energy(system: ParticleSystem) -> float
```

Compute the total interatomic energy for a configuration.

**Parameters**

- `system` (*ParticleSystem*): The system whose current configuration (positions, species, and any frozen/substrate atoms) should be evaluated.

**Returns**

- *float*: The calculator's own contribution to the total energy (i.e. not including any `ParticleSystem.external_energy` field, which `ParticleSystem.total_energy` adds separately).

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `Calculator.forces`

```python
Calculator.forces(system: ParticleSystem) -> np.ndarray
```

Compute forces on the movable particles, for HMC.

**Parameters**

- `system` (*ParticleSystem*): The system whose current configuration should be evaluated.

**Returns**

- *numpy.ndarray*: The force on each movable particle, in an array-specific order documented by the concrete subclass (see `LJCalculator.forces` and `ASECalculatorAdapter.forces`).

**Raises**

- *NotImplementedError*: Always, on the base class, and on any subclass with `supports_forces = False` (the default) that doesn't override this method.

---

### `LJCalculator`

```python
LJCalculator(sigma: float = 1.0, epsilon: float = 1.0, r_cut: float = 2.5)
```

The original fast, incremental pairwise Lennard-Jones potential.

**Constructor**

Construct a Lennard-Jones calculator.

**Parameters**

- `sigma` (*float, optional*): The Lennard-Jones length parameter (the separation at which the potential crosses zero). Default is 1.0.
- `epsilon` (*float, optional*): The Lennard-Jones energy parameter (the well depth). Default is 1.0.
- `r_cut` (*float, optional*): The cutoff radius beyond which interactions are ignored. The potential is shifted so that U(r_cut) = 0, avoiding an energy discontinuity at the cutoff. Default is 2.5.

#### `LJCalculator.pair_energy_of`

```python
LJCalculator.pair_energy_of(system: ParticleSystem, pos: np.ndarray, exclude: Optional[int] = None, species: Optional[str] = None) -> float
```

Compute the pairwise energy of a single point with the system.

**Parameters**

- `system` (*ParticleSystem*): The system supplying the other particles and the minimum-image convention (`system._min_image`).
- `pos` (*numpy.ndarray*): The (3,) Cartesian position of the point to evaluate.
- `exclude` (*int, optional*): If given, the index of a particle in `system.positions` to exclude from the sum -- used when `pos` is (a trial position for) that particle itself, so it doesn't interact with its own unmoved copy. Default is None (no exclusion).
- `species` (*str, optional*): Species of a new particle at `pos`. Ignored: this calculator treats every mobile particle alike. Default is None.

**Returns**

- *float*: The sum of pairwise Lennard-Jones interactions between `pos` and every other (non-excluded) mobile particle within `r_cut`, using the minimum-image convention.

#### `LJCalculator.total_energy`

```python
LJCalculator.total_energy(system: ParticleSystem) -> float
```

Compute the total pairwise Lennard-Jones energy of the system.

**Parameters**

- `system` (*ParticleSystem*): The system whose mobile particles (`system.positions`) should be evaluated. Frozen and substrate atoms are NOT included -- a documented limitation of `LJCalculator` (see the class docstring); use `ASECalculatorAdapter` for a potential that accounts for them.

**Returns**

- *float*: The sum of pairwise Lennard-Jones interactions over every pair of mobile particles within `r_cut`.

#### `LJCalculator.forces`

```python
LJCalculator.forces(system: ParticleSystem) -> np.ndarray
```

Compute forces on every mobile particle, fully vectorized.

Every pair at once (an (N, N, 3) displacement tensor,
minimum-imaged), no per-particle Python loop -- validated to match
a per-particle loop exactly, and about 10x faster, which matters
since HMC calls this every leapfrog half-step.

**Parameters**

- `system` (*ParticleSystem*): The system whose mobile particles (`system.positions`) should be evaluated. As with `total_energy`, frozen and substrate atoms are not included.

**Returns**

- *numpy.ndarray*: Array of shape (system.n, 3): the net Lennard-Jones force on each mobile particle, in `system.positions` order.

---

### `PairLJCalculator`

```python
PairLJCalculator(pair_params: dict, r_cut_factor: float = 2.5)
```

Multi-species, shifted 12-6 Lennard-Jones with per-pair parameters.

Unlike `LJCalculator`, this one sees every atom category -- frozen,
displaceable substrate and mobile -- so a rigid LJ surface (e.g. the
A18 fcc(111) slab of Zhou, Scheffler & Ghiringhelli, PRB 100, 174106)
is evaluated natively, on the fast incremental path. Pairs of species
with no entry in `pair_params` do not interact (the paper defines no
A-A interaction, for instance).

Each pair (a, b) has phi(r) = 4 eps [(sig/r)^12 - (sig/r)^6] - phi(r_c)
for r < r_c = r_cut_factor * sig, and 0 beyond (shifted so phi(r_c) = 0).

**Constructor**

Construct the calculator.

**Parameters**

- `pair_params` (*dict*): `{(species_a, species_b): (epsilon, sigma)}`; order of the two species does not matter.
- `r_cut_factor` (*float, optional*): Cutoff of each pair in units of its sigma. Default is 2.5.

#### `PairLJCalculator.pair_energy_of`

```python
PairLJCalculator.pair_energy_of(system: ParticleSystem, pos: np.ndarray, exclude: Optional[int] = None, species: Optional[str] = None) -> float
```

Energy of a (trial) mobile particle at `pos` with everything else.

**Parameters**

- `system` (*ParticleSystem*): The system.
- `pos` (*numpy.ndarray*): The (3,) position.
- `exclude` (*int, optional*): Index of the mobile particle `pos` belongs to (its species is used, and it is left out of the sum). None means a new particle.
- `species` (*str, optional*): Species of the new particle (when `exclude` is None). Default None: `system.exchange_species`.

**Returns**

- *float*: Its interaction energy with frozen, substrate and other mobile atoms.

#### `PairLJCalculator.total_energy`

```python
PairLJCalculator.total_energy(system: ParticleSystem) -> float
```

Total pair energy of every interacting pair in the system.

**Parameters**

- `system` (*ParticleSystem*): The system (frozen + substrate + mobile atoms).

**Returns**

- *float*: Sum over pairs i < j (including frozen-frozen pairs if their species pair is defined; constant, since they never move).

#### `PairLJCalculator.forces`

```python
PairLJCalculator.forces(system: ParticleSystem) -> np.ndarray
```

Forces on the substrate and mobile atoms.

**Parameters**

- `system` (*ParticleSystem*): The system.

**Returns**

- *numpy.ndarray*: Shape (n_substrate + n_mobile, 3), substrate rows first -- the layout `Replica._move_hmc` expects. Frozen atoms act as sources only.

---

### `ASECalculatorAdapter`

```python
ASECalculatorAdapter(ase_calculator_or_factory, cell: Sequence[float], pbc: Tuple[bool, bool, bool] = (True, True, False))
```

Wraps any ASE-compatible Calculator -- EAM, an MLIP (MACE, NequIP,
Allegro, CHGNet, ...), ASE's own EMT/LennardJones, even DFT if you're
patient -- so it can drive this GCMC engine. `supports_incremental` stays
False: for a general potential there is no safe shortcut, so every trial
move goes through `ParticleSystem.generic_*_dU` (clone, mutate, evaluate
total energy twice). This is correct for ANY potential, many-body ones
included; it is simply not the fastest possible implementation.

Why many-body potentials need this instead of the fast path: the fast
LJ path works because a *pairwise* potential's single-particle energy is
a cheap, clean sum over neighbors. EAM is explicitly NOT pairwise:
E = sum_i F(rho_i) + (1/2) sum_i,j!=i phi(r_ij), where rho_i is a local
electron density contributed to by every neighbor -- moving one atom
changes the embedding energy of all its neighbors too, not just its own
energy. Most MLIPs are even less local: a single forward pass gives the
whole configuration's energy, with no cheap way to extract "just the
change from one atom." Cloning the configuration and diffing two full
evaluations is the one formulation that is always correct regardless of
the potential's structure.

Example -- a tabulated EAM potential (LAMMPS/DYNAMO format):
    from ase.calculators.eam import EAM
    calc = ASECalculatorAdapter(EAM(potential='Fe_H.eam.fs'),
                                 cell=[Lx, Ly, Lz], pbc=(True, True, False))

Example -- a trained MLIP via its ASE wrapper:
    from mace.calculators import MACECalculator
    calc = ASECalculatorAdapter(
        MACECalculator(model_paths='fe_h_model.model', device='cuda'),
        cell=[Lx, Ly, Lz], pbc=(True, True, False))

Both plug into `ParticleSystem`/`SlabSystem` exactly like `LJCalculator`
-- nothing else in this package needs to know which one is in use.

Parallelism (`n_workers > 1` in `ReplicaExchangeGCMC`) pickles whole
`Replica` objects to send them to worker processes, which means pickling
whatever calculator each one holds. Many ASE calculators become
unpicklable once they've actually been used: EMT, for instance, builds a
`collections.defaultdict` with a lambda default factory the first time
`initialize()` runs (i.e. the first energy evaluation), and lambdas
can't be pickled by the standard library's pickler -- so a live,
already-used EMT instance fails with a cryptic
`AttributeError: Can't pickle local object 'EMT.initialize.<locals>.<lambda>'`
deep inside `concurrent.futures`, not any obvious line of your own code.

To make this safe, pass a zero-argument FACTORY instead of a live
instance whenever you want `n_workers > 1`: the class itself, if it
needs no constructor arguments (`ASECalculatorAdapter(EMT, cell=...,
pbc=...)`), or `functools.partial` for one that does
(`ASECalculatorAdapter(functools.partial(EAM, potential='Fe_H.eam.fs'),
cell=..., pbc=...)`). The adapter then constructs its own fresh
calculator instance lazily, the first time it's actually needed --
separately in each worker process, so nothing initialized (and
therefore potentially unpicklable) ever needs to cross a process
boundary. Passing a live instance still works exactly as before for
`n_workers=1`; trying to pickle one (i.e. actually using it with
`n_workers > 1`) raises a clear `TypeError` pointing you at this fix,
rather than the cryptic pickling traceback.

**Constructor**

Wrap an ASE calculator (or a factory for one) as a `Calculator`.

**Parameters**

- `ase_calculator_or_factory` (*ase.calculators.calculator.Calculator or callable*): Either a live, already-constructed ASE calculator instance (fine for single-process use, i.e. `n_workers=1`), or a zero-argument factory -- a class needing no constructor arguments, or a `functools.partial` for one that does -- which is lazily called to build the calculator the first time it's actually needed. A factory is required for `n_workers > 1`; see this class's docstring for why.
- `cell` (*sequence of float*): The (Lx, Ly, Lz) orthorhombic cell dimensions to build the ASE `Atoms` object with on every evaluation.
- `pbc` (*tuple of bool, optional*): Periodicity along each of the three cell axes, passed to the ASE `Atoms` constructor. Default is `(True, True, False)` (periodic in x, y; open in z, matching `SlabSystem`'s geometry).

#### `ASECalculatorAdapter.ase_calculator` (property)

The underlying, lazily-constructed ASE calculator instance.

**Returns**

- *ase.calculators.calculator.Calculator*: The live ASE calculator, built from the factory on first access if that's how this adapter was constructed (and cached for subsequent accesses); otherwise the instance passed at construction, directly.

#### `ASECalculatorAdapter.total_energy`

```python
ASECalculatorAdapter.total_energy(system: ParticleSystem) -> float
```

Compute the total energy of the full configuration via ASE.

**Parameters**

- `system` (*ParticleSystem*): The system whose full configuration (frozen + substrate + mobile particles, via `system.full_configuration()`) should be evaluated.

**Returns**

- *float*: The potential energy reported by the wrapped ASE calculator for the full configuration, or 0.0 if the configuration is empty.

#### `ASECalculatorAdapter.forces`

```python
ASECalculatorAdapter.forces(system: ParticleSystem) -> np.ndarray
```

Compute forces on the substrate + mobile particles via ASE.

Both substrate and mobile particles can move; only frozen atoms
are excluded. `full_configuration()` always puts frozen atoms
first (see `ParticleSystem`), so dropping the first
`frozen_positions.shape[0]` rows leaves [substrate; mobile], in
that order -- what `Replica`'s HMC move expects.

**Parameters**

- `system` (*ParticleSystem*): The system whose full configuration should be evaluated.

**Returns**

- *numpy.ndarray*: Array of shape (n_substrate + n_mobile, 3): the force on each substrate atom (first) and mobile particle (last), as reported by the wrapped ASE calculator's `get_forces()`.

## LAMMPS backend (`regc.lammps_backend`)

### `LammpsCalculator`

```python
LammpsCalculator(pair_commands: Sequence[str], species: Sequence[str], masses: Optional[dict] = None, min_distance: Optional[float] = None, z_padding: float = 5.0, lammps_name: Optional[str] = None)
```

Energies, forces and native MD from LAMMPS, for any pair style.

**Parameters**

- `pair_commands` (*sequence of str*): The LAMMPS commands defining the interaction, e.g. `["pair_style eam/fs", "pair_coeff * * Fe-H.eam.fs Fe H"]` or `["pair_style grace/fs", "pair_coeff * * model.yaml Fe H"]` or, with a grace-runtime build (`lammps_name="grace"`), `["pair_style grace/runtime", "pair_coeff * * kokkos.npz Fe H"]`.
- `species` (*sequence of str*): Species in LAMMPS atom-type order (type 1 = species[0], ...); must match the element order in `pair_commands`.
- `masses` (*dict, optional*): Mass (amu) per species; defaults to ASE's tables.
- `min_distance` (*float, optional*): Hard core: the energy is +inf if any mobile atom is closer than this to any other atom (an exact modification of the potential). Guards GCMC insertions against unphysical, off-training-set configurations of machine-learned potentials. Choose it below any physically reachable separation. Default None (off).
- `z_padding` (*float, optional*): Extra space (A) beyond the outermost atoms/walls along every non-periodic direction of the LAMMPS box (z for a slab, all three for a cluster). Default 5.0.
- `lammps_name` (*str, optional*): Which LAMMPS shared library to load: `"grace"` loads `liblammps_grace.so`, `""` the default `liblammps.so`. Default None: the environment variable `REGC_LAMMPS_NAME` of the process that evaluates (worker processes inherit it), else the default library. See `open_lammps`.

**Constructor**

Store the potential definition; LAMMPS starts lazily on first use.

**Parameters**

- `pair_commands` (*sequence of str*): LAMMPS `pair_style`/`pair_coeff` lines.
- `species` (*sequence of str*): Species in LAMMPS atom-type order.
- `masses` (*dict, optional*): Mass (amu) per species. Default None (ASE tables).
- `min_distance` (*float, optional*): Hard-core distance (A). Default None (off).
- `z_padding` (*float, optional*): Extra box height (A) beyond the atoms/walls. Default 5.0.
- `lammps_name` (*str, optional*): LAMMPS library name. Default None (`REGC_LAMMPS_NAME`, else the default library).

#### `LammpsCalculator.total_energy`

```python
LammpsCalculator.total_energy(system) -> float
```

Potential energy of the full configuration (`run 0`).

The energies of the last two configurations are cached (exact
match of positions, species and cell), so the energy before a
trial move -- the configuration evaluated at the previous move --
costs nothing; see also `_load` for the cheap re-evaluation.

**Parameters**

- `system` (*ParticleSystem*): The configuration (frozen + substrate + mobile atoms).

**Returns**

- *float*: The LAMMPS potential energy in eV (0 for an empty system; +inf if the optional hard core is violated).

#### `LammpsCalculator.forces`

```python
LammpsCalculator.forces(system) -> np.ndarray
```

Forces on substrate + mobile atoms (frozen rows dropped).

**Parameters**

- `system` (*ParticleSystem*): The configuration.

**Returns**

- *numpy.ndarray*: Shape (n_substrate + n_mobile, 3), in eV/A.

#### `LammpsCalculator.energy_after_md`

```python
LammpsCalculator.energy_after_md(system) -> float
```

Potential energy of the configuration `run_md` just produced.

**Parameters**

- `system` (*ParticleSystem*): The system after its positions were updated from `run_md`.

**Returns**

- *float*: The LAMMPS potential energy of the segment's last step (no re-evaluation), +inf if the optional hard core is violated.

#### `LammpsCalculator.run_md`

```python
LammpsCalculator.run_md(system, n_steps: int, temperature_eV: float, timestep_fs: float, taut_fs: float, seed: int, thermostat: str = 'bussi', friction_per_fs: Optional[float] = None) -> np.ndarray
```

One NVT MD segment inside LAMMPS.

Fresh Maxwell-Boltzmann velocities at T for the moving atoms, drawn
here (not with `velocity create`, which rescales the draw to
exactly T and fails for a single atom), so the start is an exact
canonical momentum sample; `fix nve` + `fix temp/csvr` (Bussi;
or `langevin`), `fix wall/reflect` at the system's hard walls
(`md_wall_bounds`: the z-walls of a slab, the six faces of a
cluster box), frozen atoms excluded from integration. The Bussi
thermostat's temperature counts 3N degrees of freedom (`extra/dof
0`): frozen atoms and walls break momentum conservation, and the
ASE path (`Replica._diffuse_md`) counts the same.

**Parameters**

- `system` (*ParticleSystem*): The system to propagate (not modified here).
- `n_steps` (*int*): Number of MD steps.
- `temperature_eV` (*float*): k_B T in eV.
- `timestep_fs, taut_fs` (*float*): Time step and Bussi thermostat time constant, in fs.
- `seed` (*int*): Random seed (velocities and thermostat).
- `thermostat` (*str, optional*): `'bussi'` (default) or `'langevin'`.
- `friction_per_fs` (*float, optional*): Langevin friction in 1/fs (damping time 1/friction), as in the ASE path. Default None: damping time `taut_fs`.

**Returns**

- *numpy.ndarray*: New positions of the substrate + mobile atoms, shape (n_substrate + n_mobile, 3), unwrapped along periodic axes.

**Raises**

- *ValueError*: If `thermostat` is unknown.

---

### `eam_pair_commands`

```python
eam_pair_commands(potential_file: str, species: Sequence[str]) -> list
```

LAMMPS pair commands for an EAM file, inferring the style from its name.

**Parameters**

- `potential_file` (*str*): A `*.eam.fs` (Finnis-Sinclair) or `*.eam.alloy` (setfl) file.
- `species` (*sequence of str*): Element names in LAMMPS type order, as they appear in the file.

**Returns**

- *list of str*: `['pair_style eam/fs', 'pair_coeff * * /abs/path Fe H']` or the `eam/alloy` equivalent.

**Raises**

- *ValueError*: If the extension is neither `.fs` nor `.alloy`.

---

### `lammps_ase_calculator`

```python
lammps_ase_calculator(pair_commands: Sequence[str], species: Sequence[str], lammps_name: Optional[str] = None)
```

An ASE calculator running the same LAMMPS potential.

For setup work done through ASE (lattice-constant scans, the
reservoir's molecule, ...), so it uses exactly the potential the
simulation's `LammpsCalculator` uses.

**Parameters**

- `pair_commands` (*sequence of str*): LAMMPS `pair_style`/`pair_coeff` lines.
- `species` (*sequence of str*): Species in LAMMPS type order.
- `lammps_name` (*str, optional*): LAMMPS library to load (`"grace"` -> `liblammps_grace.so`). Default None: `REGC_LAMMPS_NAME`, else the default library.

**Returns**

- *ase.calculators.lammpslib.LAMMPSlib*: The calculator.

---

### `open_lammps`

```python
open_lammps(lammps_name: Optional[str] = None, cmdargs: Sequence[str] = ('-log', 'none', '-screen', 'none', '-nocite'))
```

Start a LAMMPS instance from a chosen shared library.

The LAMMPS Python module loads `liblammps_<name>.so` (`liblammps.so`
for an empty name) from its own directory or, failing that, from the
dynamic loader's search path (the environment's `lib/`,
`LD_LIBRARY_PATH`). Several LAMMPS builds can so live in one
environment, e.g. conda-forge's as the default and a grace-runtime build
as `liblammps_grace.so` (`hpc/build_lammps.sh`).

A process can use only one LAMMPS library: asking for a second, different
one raises instead of loading it (two builds in one process share
symbols and corrupt each other's memory). Compare builds in separate
processes.

The module refuses a library of another LAMMPS version than its own.
The C library interface regc uses (commands, atom creation and
gathering, thermo output) is stable across versions, so when a named
library has a different version this function warns and turns the
module's version check off for the rest of the process.

**Parameters**

- `lammps_name` (*str, optional*): Library name (see `resolve_lammps_name`). Default None (`REGC_LAMMPS_NAME`, else the default library).
- `cmdargs` (*sequence of str, optional*): LAMMPS command-line arguments. Default: no log file, no screen output, no citation reminder.

**Returns**

- *lammps.lammps*: The instance.

**Raises**

- *OSError*: If the library cannot be found or loaded.
- *RuntimeError*: If this process already uses another LAMMPS library.

## Systems (`regc.systems`)

### `ParticleSystem`

```python
ParticleSystem(calculator: Calculator, exchange_species: Union[str, Sequence[str]] = 'X', positions: Optional[np.ndarray] = None, species: Optional[np.ndarray] = None, frozen_positions: Optional[np.ndarray] = None, frozen_species: Optional[np.ndarray] = None, substrate_positions: Optional[np.ndarray] = None, substrate_species: Optional[np.ndarray] = None, mass_of_species: Optional[dict] = None)
```

Geometry, boundary conditions, and species bookkeeping.

Owns positions and species for all three particle categories
described in this module's docstring (mobile, substrate, frozen). All
physics is delegated to `calculator`.

**Constructor**

Construct a particle system.

**Parameters**

- `calculator` (*Calculator*): The energy-evaluation backend all physics is delegated to.
- `exchange_species` (*str or sequence of str, optional*): The species label given to newly inserted mobile particles (and used as the default `species` for any `positions` given without an explicit `species` array). Default is `'X'`. A sequence of several labels (e.g. `['O', 'Cl']`) exchanges each with its own reservoir; see `exchange_species_list`. A one-element sequence is the same as its single label.
- `positions` (*numpy.ndarray, optional*): Initial mobile (GCMC-exchangeable) particle positions, shape (N, 3). Default is None (no mobile particles).
- `species` (*numpy.ndarray, optional*): Species label for each mobile particle in `positions`, shape (N,). Default is None, which fills every entry with `exchange_species` (required if several species are exchanged and `positions` is not empty).
- `frozen_positions` (*numpy.ndarray, optional*): Positions of fully rigid atoms that enter the energy evaluation but never move, shape (M, 3). Default is None (no frozen atoms).
- `frozen_species` (*numpy.ndarray, optional*): Species label for each frozen atom, shape (M,). Default is None (empty).
- `substrate_positions` (*numpy.ndarray, optional*): Positions of displaceable-but-non-exchangeable substrate atoms, shape (K, 3). Default is None (no substrate atoms).
- `substrate_species` (*numpy.ndarray, optional*): Species label for each substrate atom, shape (K,). Default is None (empty).
- `mass_of_species` (*dict, optional*): Mapping from species label to mass (in atomic mass units), used by `masses()`/`substrate_masses()` for HMC. Species not present in this dict fall back to `units.default_mass_amu`. Default is None (empty; every species falls back).

**Raises**

- *ValueError*: If `exchange_species` is an empty or repeated sequence, names several species while `positions` is given without `species`, or is a single species while `species` contains other labels.

#### `ParticleSystem.n` (property)

int: The number of mobile (GCMC-exchangeable) particles.

#### `ParticleSystem.exchange_species_list` (property)

tuple of str: The exchanged species, in reservoir order (length 1 for one).

#### `ParticleSystem.n_exchange_species` (property)

int: The number of exchanged species (1 unless `exchange_species` is a list).

#### `ParticleSystem.species_counts`

```python
ParticleSystem.species_counts() -> np.ndarray
```

Count the mobile particles of each exchanged species.

**Returns**

- *numpy.ndarray*: Integer array of length `n_exchange_species`, in the order of `exchange_species_list`. With a single exchanged species this is `[n]` (every mobile particle).

#### `ParticleSystem.n_adsorbed`

```python
ParticleSystem.n_adsorbed(d_max: float) -> int
```

Count the adsorbed mobile particles.

A mobile (GCMC-exchangeable) particle is adsorbed if it lies within
`d_max` (minimum-image distance) of at least one substrate or
frozen atom; each particle is counted once, however many atoms
are nearby.

**Parameters**

- `d_max` (*float*): Largest particle-atom distance that still counts as adsorbed.

**Returns**

- *int*: The number of adsorbed mobile particles (0 if there are no mobile particles or no explicit substrate/frozen atoms).

#### `ParticleSystem.n_substrate` (property)

int: The number of displaceable substrate atoms.

#### `ParticleSystem.masses`

```python
ParticleSystem.masses() -> np.ndarray
```

Get the per-mobile-particle mass array, for HMC.

**Returns**

- *numpy.ndarray*: Array of shape (n,), in `system.positions` order. Looked up from `mass_of_species` if given for that particle's species; otherwise a best-effort real atomic mass (via ASE, if installed) or 1.0 (reduced units) as a fallback -- see `units.default_mass_amu`.

#### `ParticleSystem.substrate_masses`

```python
ParticleSystem.substrate_masses() -> np.ndarray
```

Get the per-substrate-atom mass array, for HMC.

**Returns**

- *numpy.ndarray*: Array of shape (n_substrate,), in `system.substrate_positions` order, using the same lookup as `masses()`. Substrate atoms are typically much heavier than the mobile species, which `mass_of_species` should reflect for realistic HMC dynamics (e.g. `{'Fe': 55.845, 'H': 1.008}`).

#### `ParticleSystem.min_image_radius` (property)

float: Largest interaction cutoff for which the minimum-image
convention (`_min_image`) counts every periodic image: half the
smallest lattice-plane spacing. `inf` on the base class (no
periodicity known).

#### `ParticleSystem.volume` (property)

float: The accessible volume used in the GCMC insertion/deletion
acceptance formulas.

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `ParticleSystem.cell` (property)

tuple of float: (Lx, Ly, Lz)-style orthorhombic cell dimensions,
for anything that needs to write this system's geometry out (e.g.
trajectory saving in `engine.py`).

Not required for the simulation itself -- only for that optional
feature -- so subclasses that skip it just can't use it.

**Raises**

- *NotImplementedError*: On the base class, and on any subclass that doesn't define a cell.

#### `ParticleSystem.pbc` (property)

tuple of bool: Periodicity along each cell axis, paired with
`cell` for anything that writes this system's geometry out.

Subclasses must override this alongside `cell` -- there is no
sensible shared default (a bulk cube and a slab have genuinely
different periodicity).

**Raises**

- *NotImplementedError*: On the base class, and on any subclass that doesn't define periodicity.

#### `ParticleSystem.external_energy`

```python
ParticleSystem.external_energy(pos: np.ndarray) -> float
```

Evaluate any one-body external field at a position.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position to evaluate the field at.

**Returns**

- *float*: The field energy at `pos` (e.g. from a continuum wall). Default (on the base class) is always 0.0, i.e. no field. `numpy.inf` marks a forbidden region -- moves there are rejected for free by the Metropolis criterion.

#### `ParticleSystem.random_insertion_position`

```python
ParticleSystem.random_insertion_position(rng: np.random.Generator) -> np.ndarray
```

Draw a random trial position for a GCMC insertion move.

**Parameters**

- `rng` (*numpy.random.Generator*): The random number generator to draw from.

**Returns**

- *numpy.ndarray*: A (3,) Cartesian position, uniformly distributed over the geometry's accessible insertion volume.

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `ParticleSystem.apply_move`

```python
ParticleSystem.apply_move(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th mobile particle.

**Parameters**

- `i` (*int*): Index of the mobile particle (into `self.positions`) to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to, wrapped through the geometry's own boundary conditions before being stored.

**Returns**

- *None*

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `ParticleSystem.apply_move_substrate`

```python
ParticleSystem.apply_move_substrate(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th substrate atom.

Through the same geometry-specific wrapping `apply_move` uses.
Only needed if you actually have substrate atoms and want to move
them (MC or HMC).

**Parameters**

- `i` (*int*): Index of the substrate atom (into `self.substrate_positions`) to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to.

**Returns**

- *None*

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `ParticleSystem.apply_insert`

```python
ParticleSystem.apply_insert(pos: np.ndarray, species: Optional[str] = None) -> None
```

Commit the insertion of a new mobile particle.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position of the new particle, wrapped through the geometry's own boundary conditions before being stored.
- `species` (*str, optional*): Its species. Default None: `self.exchange_species` (required when several species are exchanged).

**Returns**

- *None*

**Raises**

- *NotImplementedError*: Always, on the base class -- subclasses must override this.

#### `ParticleSystem.apply_delete`

```python
ParticleSystem.apply_delete(i: int) -> None
```

Commit the deletion of a mobile particle.

**Parameters**

- `i` (*int*): Index of the mobile particle (into `self.positions` and `self.species`) to remove.

**Returns**

- *None*

#### `ParticleSystem.exchange_zone_indices`

```python
ParticleSystem.exchange_zone_indices(species: Optional[str] = None) -> np.ndarray
```

Indices of the mobile particles inside the insertion/deletion zone.

GCMC deletion may only pick particles from the same region that
`random_insertion_position` samples (whose size is `volume`), and
the acceptance rules use this count, not `n`, for detailed balance.
The base-class zone is the whole system.

**Parameters**

- `species` (*str, optional*): Only particles of this species. Default None (all species).

**Returns**

- *numpy.ndarray*: Integer indices into `self.positions`.

#### `ParticleSystem.md_z_bounds`

```python
ParticleSystem.md_z_bounds() -> Optional[Tuple[float, float]]
```

Hard z-boundaries that MD must enforce by reflection, if any.

MD integrates the calculator's forces only, so it cannot see the
infinite `external_energy` outside the box; `Replica._diffuse_md`
reflects positions/velocities at these bounds instead. The base
class has none (fully periodic).

**Returns**

- *tuple of float or None*: `(z_min, z_max)`, or None.

#### `ParticleSystem.md_wall_bounds`

```python
ParticleSystem.md_wall_bounds() -> list
```

Hard walls along each axis that MD must enforce by reflection.

The general form of `md_z_bounds`: MD (ASE or LAMMPS) reflects
positions and velocities at these planes. The base class turns
`md_z_bounds` into it, so geometries with z-walls only need not
override this.

**Returns**

- *list*: Three entries (x, y, z), each `(lo, hi)` or None (no wall).

#### `ParticleSystem.full_configuration`

```python
ParticleSystem.full_configuration() -> Tuple[np.ndarray, np.ndarray]
```

Assemble the full configuration for a Calculator to evaluate.

Combines the frozen atoms, the displaceable substrate, and the
mobile particles (in that order, skipping any empty category).
`Calculator.forces()` relies on this exact order to know which
rows of its own output are frozen (and so can be dropped) -- see
`ASECalculatorAdapter.forces`.

**Returns**

- *tuple of numpy.ndarray*: `(positions, species)`: the concatenated positions, shape (n_frozen + n_substrate + n_mobile, 3), and species, shape (n_frozen + n_substrate + n_mobile,).

#### `ParticleSystem.pair_energy_of`

```python
ParticleSystem.pair_energy_of(pos: np.ndarray, exclude: Optional[int] = None, species: Optional[str] = None) -> float
```

Compute the energy of a single point with the rest of the system.

Only usable when `self.calculator.supports_incremental` is True.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position to evaluate.
- `exclude` (*int, optional*): If given, the index of a mobile particle to exclude from the sum (used when `pos` is a trial position for that particle itself). Default is None.
- `species` (*str, optional*): Species of a new particle at `pos` (when `exclude` is None). Default None: the calculator assumes `exchange_species`. Only passed on to the calculator when given, so single-species calculators need not accept it.

**Returns**

- *float*: The calculator's pairwise energy of `pos` with the rest of the mobile particles, plus any `external_energy` at `pos` (or, if that field is infinite there, just that infinity, without evaluating the calculator at all).

**Raises**

- *NotImplementedError*: If `self.calculator.supports_incremental` is False -- use the generic `generic_*_dU` methods instead.

#### `ParticleSystem.total_energy`

```python
ParticleSystem.total_energy() -> float
```

Compute the total energy of the current configuration.

**Returns**

- *float*: The calculator's total energy for the full configuration, plus the sum of `external_energy` over every mobile and substrate particle (frozen atoms don't call `external_energy`, since they never move and so can never enter a forbidden region).

#### `ParticleSystem.external_energy_total`

```python
ParticleSystem.external_energy_total() -> float
```

Sum of `external_energy` over every mobile and substrate particle.

**Returns**

- *float*: The external-field part of `total_energy` (frozen atoms excluded).

#### `ParticleSystem.generic_displace_dU`

```python
ParticleSystem.generic_displace_dU(i: int, new_pos: np.ndarray) -> float
```

Compute the energy change of displacing a mobile particle.

Via clone-mutate-recompute: correct for any calculator, regardless
of whether it supports the fast incremental path.

**Parameters**

- `i` (*int*): Index of the mobile particle (into `self.positions`) to displace.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian trial position.

**Returns**

- *float*: The total energy change (calculator + external field) of moving particle `i` to `new_pos`. `numpy.inf` if `new_pos` is in a forbidden region (`external_energy` returns non-finite there), without evaluating the calculator at all.

#### `ParticleSystem.generic_displace_substrate_dU`

```python
ParticleSystem.generic_displace_substrate_dU(i: int, new_pos: np.ndarray) -> float
```

Compute the energy change of displacing a substrate atom.

Like `generic_displace_dU`, but for the i-th substrate atom.

**Parameters**

- `i` (*int*): Index of the substrate atom (into `self.substrate_positions`) to displace.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian trial position.

**Returns**

- *float*: The total energy change (calculator + external field) of moving substrate atom `i` to `new_pos`. `numpy.inf` if `new_pos` is in a forbidden region.

#### `ParticleSystem.generic_insert_dU`

```python
ParticleSystem.generic_insert_dU(pos: np.ndarray, species: Optional[str] = None) -> float
```

Compute the energy change of inserting a new mobile particle.

Via clone-mutate-recompute: correct for any calculator.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian trial position for the new particle.
- `species` (*str, optional*): Its species. Default None: `self.exchange_species` (required when several species are exchanged).

**Returns**

- *float*: The total energy change (calculator + external field) of adding a particle at `pos`. `numpy.inf` if `pos` is in a forbidden region, without evaluating the calculator at all.

#### `ParticleSystem.generic_delete_dU`

```python
ParticleSystem.generic_delete_dU(i: int) -> float
```

Compute the energy change of deleting a mobile particle.

Via clone-mutate-recompute: correct for any calculator.

**Parameters**

- `i` (*int*): Index of the mobile particle (into `self.positions`) to remove.

**Returns**

- *float*: The total energy change (calculator + external field) of removing particle `i`.

---

### `BulkSystem`

```python
BulkSystem(box_length: float, calculator: Calculator, positions: Optional[np.ndarray] = None, substrate_positions: Optional[np.ndarray] = None, substrate_species: Optional[np.ndarray] = None, mass_of_species: Optional[dict] = None)
```

Bulk, fully-periodic cubic box.

**Constructor**

Construct a bulk, fully-periodic cubic system.

**Parameters**

- `box_length` (*float*): The cube's edge length, L (periodic along all three axes).
- `calculator` (*Calculator*): The energy-evaluation backend.
- `positions` (*numpy.ndarray, optional*): Initial mobile particle positions, shape (N, 3). Default is None (no mobile particles).
- `substrate_positions` (*numpy.ndarray, optional*): Positions of displaceable substrate atoms, shape (K, 3). Default is None (no substrate atoms).
- `substrate_species` (*numpy.ndarray, optional*): Species label for each substrate atom, shape (K,). Default is None (empty).
- `mass_of_species` (*dict, optional*): Mapping from species label to mass (in atomic mass units), for HMC. Default is None (empty).

#### `BulkSystem.min_image_radius` (property)

float: Half the box length, L/2.

#### `BulkSystem.volume` (property)

float: The cube's volume, L^3.

#### `BulkSystem.cell` (property)

tuple of float: (L, L, L), the cube's edge lengths.

#### `BulkSystem.pbc` (property)

tuple of bool: `(True, True, True)` -- periodic along all three
axes.

#### `BulkSystem.random_insertion_position`

```python
BulkSystem.random_insertion_position(rng: np.random.Generator) -> np.ndarray
```

Draw a uniformly-distributed random position in the box.

**Parameters**

- `rng` (*numpy.random.Generator*): The random number generator to draw from.

**Returns**

- *numpy.ndarray*: A (3,) Cartesian position, uniform over [0, L)^3.

#### `BulkSystem.apply_move`

```python
BulkSystem.apply_move(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th mobile particle.

**Parameters**

- `i` (*int*): Index of the mobile particle to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to; wrapped into the primary cell before being stored.

**Returns**

- *None*

#### `BulkSystem.apply_move_substrate`

```python
BulkSystem.apply_move_substrate(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th substrate atom.

**Parameters**

- `i` (*int*): Index of the substrate atom to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to; wrapped into the primary cell before being stored.

**Returns**

- *None*

#### `BulkSystem.apply_insert`

```python
BulkSystem.apply_insert(pos: np.ndarray, species: Optional[str] = None) -> None
```

Commit the insertion of a new mobile particle.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position of the new particle; wrapped into the primary cell before being stored.
- `species` (*str, optional*): Its species. Default None: `self.exchange_species` (required when several species are exchanged).

**Returns**

- *None*

---

### `SlabSystem`

```python
SlabSystem(Lx: float, Ly: float, Lz: float, calculator: Calculator, exchange_species: str = 'X', wall_epsilon: float = 0.0, wall_sigma: float = 1.0, wall_cut: Optional[float] = None, frozen_positions: Optional[np.ndarray] = None, frozen_species: Optional[np.ndarray] = None, substrate_positions: Optional[np.ndarray] = None, substrate_species: Optional[np.ndarray] = None, positions: Optional[np.ndarray] = None, species: Optional[np.ndarray] = None, insertion_z_range: Optional[Tuple[float, float]] = None, mass_of_species: Optional[dict] = None, cell_xy: Optional[np.ndarray] = None)
```

Periodic in x, y only; particles live between z=0 and z=Lz.

`wall_epsilon > 0` adds a continuum 9-3 wall -- use this for a fast
LJ-only demo with no explicit substrate atoms. For a real substrate
(e.g. Fe or Au), instead pass `frozen_positions`/`frozen_species`
(rigid) and/or `substrate_positions`/`substrate_species` (displaceable
-- what a reconstructing surface needs) and leave `wall_epsilon = 0`:
the substrate's own interaction with the mobile species, through
whatever Calculator you plug in, supplies the attraction/repulsion
instead of an artificial continuum field. Don't use both at once --
that would double-count the substrate.

`insertion_z_range=(z_lo, z_hi)` restricts random insertion attempts to
a sub-slab (and correspondingly narrows the `volume` used in the
insertion/deletion acceptance formulas to match, and restricting
deletion to the particles inside that sub-slab, see
`exchange_zone_indices` -- together these keep the ensemble exact) instead of sampling z uniformly across the whole box. This
matters whenever most of the box is somewhere insertion attempts would
essentially never be accepted anyway -- e.g. deep inside a solid
substrate, or in a large vacuum region far from a real (rather than
continuum-wall) surface -- since without it, GCMC would waste nearly
every insertion attempt on that dead volume. Defaults to the full
(0, Lz) range, unchanged from previous behavior.

**Constructor**

Construct a surface-adsorption slab system.

**Parameters**

- `Lx, Ly, Lz` (*float*): The cell dimensions. Periodic along x and y; particles live between z=0 and z=Lz (open, not periodic, along z). Pass `Lx = Ly = None` together with `cell_xy` for a non-orthogonal (e.g. hexagonal) surface cell.
- `calculator` (*Calculator*): The energy-evaluation backend.
- `exchange_species` (*str, optional*): The species label for newly inserted mobile particles. Default is `'X'`.
- `wall_epsilon` (*float, optional*): Well depth of an optional continuum 9-3 wall at z=0 (a fast LJ-only stand-in for an explicit substrate). Default is 0.0 (no wall). Must be 0 if `frozen_positions` or `substrate_positions` are given (see class docstring).
- `wall_sigma` (*float, optional*): Length scale of the continuum wall. Default is 1.0.
- `wall_cut` (*float, optional*): Cutoff distance beyond which the continuum wall's energy is zero. Default is None, which uses `3.0 * wall_sigma`.
- `frozen_positions` (*numpy.ndarray, optional*): Positions of fully rigid substrate atoms, shape (M, 3). Default is None (none).
- `frozen_species` (*numpy.ndarray, optional*): Species label for each frozen atom, shape (M,). Default is None (empty).
- `substrate_positions` (*numpy.ndarray, optional*): Positions of displaceable substrate atoms, shape (K, 3). Default is None (none).
- `substrate_species` (*numpy.ndarray, optional*): Species label for each substrate atom, shape (K,). Default is None (empty).
- `positions` (*numpy.ndarray, optional*): Initial mobile particle positions, shape (N, 3). Default is None (none).
- `species` (*numpy.ndarray, optional*): Species label for each mobile particle, shape (N,). Default is None, which fills every entry with `exchange_species`.
- `insertion_z_range` (*tuple of float, optional*): (z_lo, z_hi) sub-range of [0, Lz] to restrict random insertion attempts (and the `volume` used in the insertion/deletion acceptance formulas) to. Default is None, which uses the full (0, Lz) range.
- `mass_of_species` (*dict, optional*): Mapping from species label to mass (in atomic mass units), for HMC. Default is None (empty).
- `cell_xy` (*array_like, optional*): 2x2 matrix whose rows are the in-plane lattice vectors a1, a2 (z components zero), e.g. a hexagonal fcc(111) supercell. When given, `Lx` and `Ly` must be None. Default is None (the rectangular Lx x Ly cell).

**Raises**

- *ValueError*: If `wall_epsilon` > 0 and either `frozen_positions` or `substrate_positions` is also given (double-counts the substrate), or if `insertion_z_range` doesn't satisfy `0 <= z_lo < z_hi <= Lz`.

#### `SlabSystem.md_z_bounds`

```python
SlabSystem.md_z_bounds() -> Optional[Tuple[float, float]]
```

Hard z-walls of `external_energy`, for MD reflection.

**Returns**

- *tuple of float*: `(0, Lz)`.

#### `SlabSystem.exchange_zone_indices`

```python
SlabSystem.exchange_zone_indices(species: Optional[str] = None) -> np.ndarray
```

Indices of the mobile particles with `z_lo <= z < z_hi`.

The same region `random_insertion_position` samples (x, y are
always inside, being periodic). Particles that have diffused out of
`insertion_z_range` stay in the system but can't be deleted until
they diffuse back in.

**Parameters**

- `species` (*str, optional*): Only particles of this species. Default None (all species).

**Returns**

- *numpy.ndarray*: Integer indices into `self.positions`.

#### `SlabSystem.volume` (property)

float: The accessible volume, `area * (z_hi - z_lo)` over
`insertion_z_range`.

#### `SlabSystem.min_image_radius` (property)

float: Half the smallest in-plane lattice-plane spacing,
`min(area/|a1|, area/|a2|) / 2` (`min(Lx, Ly)/2` if orthogonal).

#### `SlabSystem.area` (property)

float: The surface-cell area, `|det(cell_xy)|` (`Lx * Ly` if
orthogonal).

#### `SlabSystem.cell` (property)

tuple or list: (Lx, Ly, Lz) for an orthogonal cell, else the 3x3
cell matrix (rows a1, a2, (0, 0, Lz)).

#### `SlabSystem.pbc` (property)

tuple of bool: `(True, True, False)` -- periodic in x, y; open
in z.

#### `SlabSystem.external_energy`

```python
SlabSystem.external_energy(pos: np.ndarray) -> float
```

Evaluate the hard z-boundary and optional continuum wall.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position to evaluate.

**Returns**

- *float*: `numpy.inf` if `pos`'s z coordinate is outside (0, Lz]. Otherwise, the continuum 9-3 wall's energy at that z (0.0 if `wall_epsilon` is 0 or z is beyond `wall_cut`).

#### `SlabSystem.random_insertion_position`

```python
SlabSystem.random_insertion_position(rng: np.random.Generator) -> np.ndarray
```

Draw a uniformly-distributed random position within
`insertion_z_range`.

**Parameters**

- `rng` (*numpy.random.Generator*): The random number generator to draw from.

**Returns**

- *numpy.ndarray*: A (3,) Cartesian position, uniform over the surface cell (`[0, Lx) x [0, Ly)` if orthogonal) times `[z_lo, z_hi)`.

#### `SlabSystem.apply_move`

```python
SlabSystem.apply_move(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th mobile particle.

**Parameters**

- `i` (*int*): Index of the mobile particle to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to; x, y wrapped into the primary cell, z left as given.

**Returns**

- *None*

#### `SlabSystem.apply_move_substrate`

```python
SlabSystem.apply_move_substrate(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th substrate atom.

**Parameters**

- `i` (*int*): Index of the substrate atom to move.
- `new_pos` (*numpy.ndarray*): The (3,) Cartesian position to move it to; x, y wrapped into the primary cell, z left as given.

**Returns**

- *None*

#### `SlabSystem.apply_insert`

```python
SlabSystem.apply_insert(pos: np.ndarray, species: Optional[str] = None) -> None
```

Commit the insertion of a new mobile particle.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) Cartesian position of the new particle; x, y wrapped into the primary cell.
- `species` (*str, optional*): Its species. Default None: `self.exchange_species` (required when several species are exchanged).

**Returns**

- *None*

#### `SlabSystem.coverage`

```python
SlabSystem.coverage() -> float
```

Compute the mobile-particle areal coverage.

**Returns**

- *float*: The number of mobile particles per unit area, `n / (Lx * Ly)`.

---

### `ClusterSystem`

```python
ClusterSystem(box: Union[float, Sequence[float]], calculator: Calculator, exchange_species: Union[str, Sequence[str]] = 'X', frozen_positions: Optional[np.ndarray] = None, frozen_species: Optional[np.ndarray] = None, substrate_positions: Optional[np.ndarray] = None, substrate_species: Optional[np.ndarray] = None, positions: Optional[np.ndarray] = None, species: Optional[np.ndarray] = None, insertion_shell: Optional[Tuple[float, float]] = None, center: Optional[Sequence[float]] = None, mass_of_species: Optional[dict] = None)
```

A cluster (nanoparticle) in a box with no periodicity: hard walls on all faces.

Particles live in `(0, Lx] x (0, Ly] x (0, Lz]`; `external_energy` is
infinite outside, and MD reflects at the six faces (`md_wall_bounds`).
The cluster's own atoms are the frozen and displaceable substrate
atoms, as for a slab; freeze at least the central atom, so the
particle cannot drift away from the insertion shell.

Insertions are drawn uniformly over the whole box, or, with
`insertion_shell=(r_in, r_out)`, over the spherical shell
`r_in <= |r - center| < r_out`. That shell is where adsorption
happens; the rest of the box (vacuum, the cluster's interior) would
waste almost every insertion attempt. As for `SlabSystem`'s
`insertion_z_range`, the acceptance rules use the shell's volume and
deletions pick only particles inside it (`exchange_zone_indices`), so
the ensemble stays exact; particles that wander outside the shell stay
in the system and can be deleted once they are back.

**Constructor**

Construct a cluster system.

**Parameters**

- `box` (*float or sequence of 3 floats*): Box edge lengths (Lx, Ly, Lz), or one length for a cube. Not periodic in any direction.
- `calculator` (*Calculator*): The energy-evaluation backend.
- `exchange_species` (*str or sequence of str, optional*): The exchanged species (see `ParticleSystem`). Default `'X'`.
- `frozen_positions, frozen_species` (*numpy.ndarray, optional*): Rigid cluster atoms (e.g. the central atom), shape (M, 3) and (M,). Default None (none).
- `substrate_positions, substrate_species` (*numpy.ndarray, optional*): Displaceable cluster atoms, shape (K, 3) and (K,). Default None.
- `positions, species` (*numpy.ndarray, optional*): Initial mobile particles, shape (N, 3) and (N,). Default None.
- `insertion_shell` (*tuple of float, optional*): `(r_in, r_out)`: insert in this spherical shell around `center` (0 <= r_in < r_out; the shell must fit in the box). Default None: the whole box.
- `center` (*sequence of 3 floats, optional*): Centre of the insertion shell. Default: the box centre.
- `mass_of_species` (*dict, optional*): Masses (amu) by species, for MD/HMC. Default None (ASE tables).

**Raises**

- *ValueError*: If a box length is not positive, or the shell is empty or does not fit in the box.

#### `ClusterSystem.volume` (property)

float: The insertion volume: the shell's `4/3 pi (r_out^3 -
r_in^3)`, or the box volume without a shell.

#### `ClusterSystem.cell` (property)

tuple of float: The box, (Lx, Ly, Lz).

#### `ClusterSystem.pbc` (property)

tuple of bool: `(False, False, False)`: no periodicity.

#### `ClusterSystem.external_energy`

```python
ClusterSystem.external_energy(pos: np.ndarray) -> float
```

The hard walls of the box.

**Parameters**

- `pos` (*numpy.ndarray*): A (3,) Cartesian position.

**Returns**

- *float*: `numpy.inf` outside `(0, L]` along any axis, else 0.0.

#### `ClusterSystem.random_insertion_position`

```python
ClusterSystem.random_insertion_position(rng: np.random.Generator) -> np.ndarray
```

Draw a position uniformly over the insertion shell (or the box).

**Parameters**

- `rng` (*numpy.random.Generator*): The random number generator to draw from.

**Returns**

- *numpy.ndarray*: A (3,) Cartesian position.

#### `ClusterSystem.exchange_zone_indices`

```python
ClusterSystem.exchange_zone_indices(species: Optional[str] = None) -> np.ndarray
```

Indices of the mobile particles inside the insertion shell.

**Parameters**

- `species` (*str, optional*): Only particles of this species. Default None (all species).

**Returns**

- *numpy.ndarray*: Integer indices into `self.positions` (every particle when there is no shell).

#### `ClusterSystem.md_z_bounds`

```python
ClusterSystem.md_z_bounds() -> Optional[Tuple[float, float]]
```

The z-walls (see `md_wall_bounds` for all three axes).

**Returns**

- *tuple of float*: `(0, Lz)`.

#### `ClusterSystem.md_wall_bounds`

```python
ClusterSystem.md_wall_bounds() -> list
```

The six box faces, for MD reflection.

**Returns**

- *list*: `[(0, Lx), (0, Ly), (0, Lz)]`.

#### `ClusterSystem.apply_move`

```python
ClusterSystem.apply_move(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th mobile particle (no wrapping).

**Parameters**

- `i` (*int*): Index of the mobile particle.
- `new_pos` (*numpy.ndarray*): Its new (3,) position.

**Returns**

- *None*

#### `ClusterSystem.apply_move_substrate`

```python
ClusterSystem.apply_move_substrate(i: int, new_pos: np.ndarray) -> None
```

Commit a displacement of the i-th substrate atom (no wrapping).

**Parameters**

- `i` (*int*): Index of the substrate atom.
- `new_pos` (*numpy.ndarray*): Its new (3,) position.

**Returns**

- *None*

#### `ClusterSystem.apply_insert`

```python
ClusterSystem.apply_insert(pos: np.ndarray, species: Optional[str] = None) -> None
```

Commit the insertion of a new mobile particle.

**Parameters**

- `pos` (*numpy.ndarray*): The (3,) position of the new particle.
- `species` (*str, optional*): Its species. Default None: `self.exchange_species` (required when several species are exchanged).

**Returns**

- *None*

---

### `LJSystem`

Alias of [`BulkSystem`](#bulksystem), kept for older scripts.

---

### `SlabLJSystem`

Alias of [`SlabSystem`](#slabsystem), kept for older scripts.

## Structures and slab geometry (`regc.structures`)

### `cubic_lattice_constant`

```python
cubic_lattice_constant(ase_calculator, symbol: str, crystal: str = 'bcc', a_range: Optional[Tuple[float, float]] = None, n_coarse: int = 61, n_fine: int = 21) -> float
```

Equilibrium lattice constant of a cubic crystal for a given potential.

A coarse energy scan of the conventional cubic cell locates the minimum,
a finer scan around it is fitted with a parabola.

**Parameters**

- `ase_calculator` (*ase.calculators.calculator.Calculator*): The potential (e.g. ASE's EAM, or `regc.lammps_backend.lammps_ase_calculator`).
- `symbol` (*str*): Element symbol.
- `crystal` (*{'bcc', 'fcc', 'sc'}, optional*): Crystal structure. Default `'bcc'`.
- `a_range` (*tuple of float, optional*): Range (A) of the coarse scan. Default: 0.8 to 1.2 times the element's experimental lattice constant (`reference_lattice_constant`), or (2.3, 5.5) if ASE has none for this crystal. Keep the range near the real value for machine-learned potentials: far below it they can have unphysical energy holes, which a wide scan finds.
- `n_coarse, n_fine` (*int, optional*): Points of the coarse and fine scans. Defaults 61 and 21.

**Returns**

- *float*: The lattice constant, in Angstrom.

**Raises**

- *ValueError*: If the energy has no minimum inside `a_range` (wrong potential or range) or is not convex around it.

---

### `slab_geometry_from_atoms`

```python
slab_geometry_from_atoms(atoms, n_frozen_layers: Optional[int] = None, headroom: float = 5.0, layer_tol: float = 0.001) -> SlabGeometry
```

Lay out an ASE slab for `SlabSystem`.

Layers are identified by z. The slab is shifted so the top *frozen*
layer is at z = 0: the box's hard wall then keeps the exchanged species
out of the frozen block, while it can still reach the displaceable
layers above (subsurface sites included).

**Parameters**

- `atoms` (*ase.Atoms*): The slab, with a1 along x and a2 in the xy plane (as `ase.build` surface functions give), periodic in x and y.
- `n_frozen_layers` (*int, optional*): Number of bottom layers kept rigid; the rest are displaceable substrate. Default None: all layers frozen (a rigid surface).
- `headroom` (*float, optional*): Space (A) for the exchanged species above the topmost atom. Default 5.0.
- `layer_tol` (*float, optional*): z tolerance (A) for grouping atoms into layers. Default 1e-3.

**Returns**

- *SlabGeometry*: The geometry.

**Raises**

- *ValueError*: If the cell is not in the restricted (a1 along x, a2 in xy) form, or `n_frozen_layers` is out of range.

---

### `SlabGeometry`

```python
SlabGeometry(cell_xy: np.ndarray, Lz: float, top_z: float, frozen_positions: np.ndarray, frozen_species: np.ndarray, substrate_positions: np.ndarray, substrate_species: np.ndarray, layer_z: np.ndarray, n_surface: int = 0) -> None
```

A slab laid out for `SlabSystem`: frozen block at z <= 0, substrate above.

**Attributes**

- `cell_xy` (*numpy.ndarray*): 2x2 in-plane cell (rows a1, a2).
- `Lz` (*float*): Height of the box the exchanged species lives in, (0, Lz].
- `top_z` (*float*): z of the topmost slab atom.
- `frozen_positions, substrate_positions` (*numpy.ndarray*): Positions of the rigid and the displaceable slab atoms, (M, 3).
- `frozen_species, substrate_species` (*numpy.ndarray*): Their species labels.
- `layer_z` (*numpy.ndarray*): z of every atomic layer, ascending (after the shift).
- `n_surface` (*int*): Number of atoms in the top layer (for coverages per surface atom).

#### `SlabGeometry.orthogonal` (property)

bool: Whether the in-plane cell is rectangular.

#### `SlabGeometry.area` (property)

float: Surface-cell area, in A^2.

#### `SlabGeometry.make_system`

```python
SlabGeometry.make_system(calculator: Calculator, exchange_species: Union[str, Sequence[str]], mass_of_species: Optional[dict] = None, insertion_z_range: Optional[Tuple[float, float]] = None) -> SlabSystem
```

Build the `SlabSystem` for this geometry.

**Parameters**

- `calculator` (*Calculator*): The energy backend (each system needs its own instance).
- `exchange_species` (*str or sequence of str*): Species GCMC inserts and deletes (several, e.g. `['O', 'Cl']`, each with its own reservoir; see `ParticleSystem`).
- `mass_of_species` (*dict, optional*): Masses (amu) by species, for MD/HMC. Default None (ASE tables).
- `insertion_z_range` (*tuple of float, optional*): Restrict insertions to this z-range (see `SlabSystem`); default None, the whole box (0, Lz].

**Returns**

- *SlabSystem*: An empty (no mobile particles) system.

#### `SlabGeometry.summary`

```python
SlabGeometry.summary() -> str
```

One-line description.

**Returns**

- *str*: Atom counts, cell and box height.

---

### `insertion_energy_scan`

```python
insertion_energy_scan(system, xy_points: Sequence[Sequence[float]], z_values: Sequence[float]) -> np.ndarray
```

Energy of inserting one particle along vertical lines.

For each lateral point and height, the energy change of inserting one
`system.exchange_species` particle there (the system is unchanged).
With `xy_points` above adsorption sites, the minimum over z of each row
is the site's unrelaxed binding energy relative to the calculator's
zero (subtract a reservoir reference, e.g. 1/2 E(H2), for adsorption
energies).

**Parameters**

- `system` (*ParticleSystem*): The system (typically a bare slab).
- `xy_points` (*sequence of (float, float)*): Lateral positions, in A.
- `z_values` (*sequence of float*): Heights, in A (positions outside the box give +inf).

**Returns**

- *numpy.ndarray*: Shape (len(xy_points), len(z_values)): insertion energies in eV. Points where the calculator returns NaN (e.g. exactly on top of an atom, r = 0) are reported as +inf, so `argmin` finds real minima.

---

### `cluster_geometry_from_atoms`

```python
cluster_geometry_from_atoms(atoms, headroom: float = 6.0, frozen: Union[str, int] = 'center', full_coordination: int = 12, cn_cutoff: Optional[float] = None) -> ClusterGeometry
```

Lay out a cluster for `ClusterSystem`.

The cluster is translated so its centroid is at the centre of a cubic
box of edge `2 (radius + headroom)`: a spherical insertion shell up
to `headroom` beyond the outermost atom then fits inside the box.

**Parameters**

- `atoms` (*ase.Atoms*): The cluster (its cell and periodicity are ignored).
- `headroom` (*float, optional*): Space (A) between the outermost atom and the box walls along the axes. Default 6.0.
- `frozen` (*{'center', 'core', 'none'} or int, optional*): Which atoms are rigid: the one nearest the centroid (default; pins the particle in place while it relaxes), every fully coordinated atom (`coordination >= full_coordination`), none, or the given number of atoms nearest the centroid.
- `full_coordination` (*int, optional*): Coordination of an interior atom: 12 for fcc, hcp and icosahedral particles, 14 for bcc with the default cutoff. Atoms below it are surface atoms. Default 12.
- `cn_cutoff` (*float, optional*): Neighbour cutoff for the coordination numbers (see `coordination_numbers`). Default None (automatic).

**Returns**

- *ClusterGeometry*: The geometry.

**Raises**

- *ValueError*: If `frozen` is not understood, or the cluster has no atoms.

---

### `ClusterGeometry`

```python
ClusterGeometry(box: float, center: np.ndarray, radius: float, frozen_positions: np.ndarray, frozen_species: np.ndarray, substrate_positions: np.ndarray, substrate_species: np.ndarray, coordination: np.ndarray, n_surface: int = 0) -> None
```

A cluster laid out for `ClusterSystem`: centred in a cubic, non-periodic box.

**Attributes**

- `box` (*float*): Edge of the cubic box, in A.
- `center` (*numpy.ndarray*): The box centre (3,), where the cluster's centroid sits.
- `radius` (*float*): Largest distance of a cluster atom from `center`, in A.
- `frozen_positions, substrate_positions` (*numpy.ndarray*): Positions of the rigid and the displaceable cluster atoms, (M, 3).
- `frozen_species, substrate_species` (*numpy.ndarray*): Their species labels.
- `coordination` (*numpy.ndarray*): Coordination number of every cluster atom (frozen first, then substrate, as in `full_configuration`).
- `n_surface` (*int*): Number of surface atoms (coordination below full), for coverages per surface atom.

#### `ClusterGeometry.n_atoms` (property)

int: Number of cluster atoms (frozen + displaceable).

#### `ClusterGeometry.area` (property)

None: a cluster has no surface-cell area (coverages per atom instead).

#### `ClusterGeometry.shell`

```python
ClusterGeometry.shell(d_in: float, d_out: float) -> Tuple[float, float]
```

Insertion-shell radii from distances relative to the cluster's surface.

**Parameters**

- `d_in, d_out` (*float*): Inner and outer distance from `radius` (A); negative values reach into the cluster.

**Returns**

- *tuple of float*: `(max(0, radius + d_in), radius + d_out)`, for `ClusterSystem(insertion_shell=...)`.

**Raises**

- *ValueError*: If the shell is empty or does not fit in the box.

#### `ClusterGeometry.make_system`

```python
ClusterGeometry.make_system(calculator: Calculator, exchange_species: Union[str, Sequence[str]], mass_of_species: Optional[dict] = None, insertion_shell: Optional[Tuple[float, float]] = None) -> ClusterSystem
```

Build the `ClusterSystem` for this geometry.

**Parameters**

- `calculator` (*Calculator*): The energy backend (each system needs its own instance).
- `exchange_species` (*str or sequence of str*): Species GCMC inserts and deletes.
- `mass_of_species` (*dict, optional*): Masses (amu) by species, for MD/HMC. Default None (ASE tables).
- `insertion_shell` (*tuple of float, optional*): `(r_in, r_out)` around the centre (see `shell`). Default None: the whole box.

**Returns**

- *ClusterSystem*: An empty (no mobile particles) system.

#### `ClusterGeometry.summary`

```python
ClusterGeometry.summary() -> str
```

One-line description.

**Returns**

- *str*: Atom counts, radius and box.

---

### `coordination_numbers`

```python
coordination_numbers(positions: np.ndarray, cutoff: Optional[float] = None) -> np.ndarray
```

Coordination number of every atom of a finite (non-periodic) structure.

**Parameters**

- `positions` (*numpy.ndarray*): Atom positions, (N, 3).
- `cutoff` (*float, optional*): Neighbour cutoff, in A. Default: halfway between the fcc first and second neighbour shells, `(1 + sqrt 2)/2` times the median nearest-neighbour distance (12 neighbours inside an fcc or icosahedral particle; 14 inside a bcc one).

**Returns**

- *numpy.ndarray*: Integer coordination numbers, (N,).

---

### `reference_lattice_constant`

```python
reference_lattice_constant(symbol: str, crystal: str) -> Optional[float]
```

The experimental cubic lattice constant of an element, from ASE's tables.

**Parameters**

- `symbol` (*str*): Element symbol.
- `crystal` (*{'bcc', 'fcc', 'sc'}*): Crystal structure.

**Returns**

- *float or None*: `ase.data.reference_states` lattice constant (A) when the element's reference structure is `crystal`, else None.

## Thermodynamic state (`regc.state`)

### `ThermoState`

```python
ThermoState(temperature: float, mu: Union[float, np.ndarray], label: Tuple[int, int], lambda_cubed: Union[float, np.ndarray] = 1.0) -> None
```

The thermodynamic label attached to one replica-exchange grid point.

Stays fixed to its grid slot; it is the *configuration* that gets
swapped between slots during replica exchange (see `Replica` in
`replica.py`).

With several exchanged species (`system.exchange_species` a list,
e.g. `['O', 'Cl']`), `mu` and `lambda_cubed` are arrays with one
entry per species, in that order; with one species they are floats,
exactly as before.

**Attributes**

- `temperature` (*float*): Temperature, in the same energy-like units as the Calculator's reported energies (e.g. eV for EMT/EAM/MLIP calculators, or reduced units for `LJCalculator`).
- `mu` (*float or numpy.ndarray*): Chemical potential, in the same units as `temperature`; one per exchanged species if there are several.
- `label` (*tuple of int*): The (temperature_index, mu_index) position of this state in the replica-exchange grid.
- `lambda_cubed` (*float or numpy.ndarray, optional*): Thermal de Broglie wavelength cubed, in whatever length units the system's volume/positions use (one per exchanged species if there are several). Default is 1.0, the correct, self-consistent choice in reduced Lennard-Jones units (it's equivalent to defining the LJ length unit sigma as the wavelength scale too) -- but it is NOT generally correct when a Calculator works in real physical units (e.g. ASE's EMT or a real EAM/MLIP, which report energies in eV using positions in Angstrom). There, supply the real value (see `regc.units.thermal_wavelength_cubed_angstrom3`) so the insertion/deletion acceptance formulas in `replica.py` stay dimensionally correct.

#### `ThermoState.beta` (property)

Inverse temperature.

**Returns**

- *float*: 1 / `temperature`.

#### `ThermoState.mu_vec` (property)

numpy.ndarray: `mu` as a 1D array (length 1 for one species).

#### `ThermoState.lambda_cubed_vec` (property)

numpy.ndarray: `lambda_cubed` as a 1D array, broadcast to `mu_vec`.

## Replicas (`regc.replica`)

### `Replica`

```python
Replica(state: ThermoState, system: ParticleSystem, rng: np.random.Generator, max_disp: float = 0.2, hmc_enabled: bool = False, hmc_leapfrog_steps: int = 10, hmc_dt: float = 0.005)
```

Note the design choice: the ThermoState stays attached to the grid
slot; it is the *configuration* (`self.system`) that gets swapped between
slots during replica exchange (see `engine.py`). Each slot keeps its own
RNG stream so move statistics and pseudo-random sequences stay
reproducible per grid point regardless of how sweeps are scheduled
(serially or across a process pool).

`hmc_enabled=True` replaces the single-particle random MC displacement
move with Hybrid Monte Carlo (HMC): resample momenta from Maxwell-
Boltzmann, run a short leapfrog trajectory for ALL mobile particles at
once, and accept/reject the whole trajectory on the total (kinetic +
potential) energy change. This is exact regardless of integrator error
or timestep -- the Metropolis step, not the integrator, carries the
correctness guarantee, so a too-large `hmc_dt` costs you acceptance
rate (efficiency), never correctness. Collective, force-directed moves
like this explore configuration space far more efficiently than
single-particle random displacement whenever particles need to move
together, which is the whole motivation (surface reconstruction is the
eventual target; a dense/structured fluid benefits too). Needs a
Calculator with `supports_forces = True` (LJCalculator and
ASECalculatorAdapter both qualify); raises immediately at construction
otherwise, rather than failing confusingly on the first move.

Current scope/limitation worth knowing: `hmc_dt`'s momentum kicks use
only `calculator.forces()`, i.e. the interatomic contribution -- not
any `external_energy` field (a continuum wall, or a hard z-boundary).
Correctness is unaffected (the final accept/reject step still checks
the true total energy including that field, so a trajectory that ends
somewhere forbidden is still always rejected), but a trajectory isn't
steered away from such a boundary the way real forces would, so
acceptance efficiency near one may be poor. Fine for a bulk system with
no external field (the common case); something to watch for a
wall-bounded or hard-boundary geometry.

**Constructor**

Construct a replica-exchange grid slot.

**Parameters**

- `state` (*ThermoState*): The (temperature, chemical potential) label for this grid slot.
- `system` (*ParticleSystem*): The initial configuration to attach to this slot.
- `rng` (*numpy.random.Generator*): This slot's own random number generator stream, kept separate per grid point so move statistics and pseudo-random sequences stay reproducible regardless of how sweeps are scheduled (serially or across a process pool).
- `max_disp` (*float, optional*): Maximum single-particle displacement (per axis) for plain MC displacement moves (`_move_displace`, `_move_displace_substrate`). Default is 0.2.
- `hmc_enabled` (*bool, optional*): If True, replace the single-particle random MC displacement move with Hybrid Monte Carlo (see this class's docstring for the full rationale). Default is False.
- `hmc_leapfrog_steps` (*int, optional*): Number of leapfrog integration steps per HMC trial. Default is 10.
- `hmc_dt` (*float, optional*): Leapfrog integration timestep for HMC. Default is 0.005.

**Raises**

- *ValueError*: If `hmc_enabled` is True but `system.calculator` doesn't have `supports_forces = True`.

#### `Replica.gc_attempt`

```python
Replica.gc_attempt() -> None
```

Make one particle insertion/removal attempt ([ZSG] Sec. II A 1).

Removal if a uniform random number is below 0.5, insertion
otherwise -- equal numbers of insertion and removal attempts, as
detailed balance requires.

**Returns**

- *None*

#### `Replica.regc_local_step`

```python
Replica.regc_local_step(gc: bool, mode: str, n: int, **kwargs) -> Replica
```

The per-replica part of one REGC step: optional GC attempt, then
diffusion.

**Parameters**

- `gc` (*bool*): Whether to make a `gc_attempt` first (this step drew the GC branch rather than replica exchange).
- `mode, n, **kwargs`: Passed to `diffuse`.

**Returns**

- *Replica*: `self`.

#### `Replica.regc_local_block`

```python
Replica.regc_local_block(gc_flags, first_step: int, record_every: int, burn_in_steps: int, mode: str, n: int, adsorbed_cutoff: Optional[float] = None, observables=None, record_configurations: bool = False, **kwargs) -> Replica
```

Several consecutive REGC steps with no replica exchange in between.

Between two RE moves the replicas are independent, so the engine
hands each one a whole block of steps in a single (possibly remote)
call instead of one call per step. The result is identical to
calling `regc_local_step` once per step.

**Parameters**

- `gc_flags` (*sequence of bool*): One entry per step: whether that step starts with a GC attempt.
- `first_step` (*int*): 1-based index of the block's first step in the whole run.
- `record_every, burn_in_steps` (*int*): Record (energy, N) after step k if `k > burn_in_steps` and `k % record_every == 0`, as `ReplicaExchangeGCMC.run_regc` does.
- `adsorbed_cutoff` (*float, optional*): If given, each sample also carries `system.n_adsorbed(adsorbed_cutoff)`; else None. Default None.
- `observables` (*sequence of regc.observables.Observable, optional*): Evaluated on the configuration of every recorded sample. Default None.
- `record_configurations` (*bool, optional*): If True, each sample also carries a copy of the configuration (`ParticleSystem.full_configuration` plus the group sizes). Default False.
- `mode, n, **kwargs`: Passed to `diffuse`.

**Returns**

- *Replica*: `self`, with the samples of this block in `self.block_samples` as a list of `(energy, N, N_ads, N_s, obs, config)`; `N_s` is `system.species_counts()` with several exchanged species, else None; `obs` is `{name: values}` (empty without observables); `config` is `(positions, species, n_frozen, n_substrate)` or None.

#### `Replica.diffuse`

```python
Replica.diffuse(mode: str = 'mc', n: int = 10, displace_substrate: bool = False, **md_kwargs) -> Replica
```

Sample the canonical (fixed N, V, T) ensemble ([ZSG] Sec. II A 3).

**Parameters**

- `mode` (*str, optional*): `'mc_moves'`: `n` single-atom displacement attempts (Eq. (3)). `'mc'`: `n` MC cycles, each of as many single-atom displacement attempts (Eq. (3)) as there are movable atoms, so each is attempted once on average. `'hmc'`: `n` HMC trajectories (`_move_hmc`; needs forces). `'md'`: `n` steps of ASE NVT molecular dynamics (`_diffuse_md`). Default is `'mc'`.
- `n` (*int, optional*): Number of cycles / trajectories / MD steps. Default is 10.
- `displace_substrate` (*bool, optional*): For `'mc'`: also count displaceable substrate atoms as movable (each attempt picks uniformly among mobile + substrate atoms). HMC and MD always move substrate atoms. Default False.
- `**md_kwargs`: Passed to `_diffuse_md` for `'md'`.

**Returns**

- *Replica*: `self`.

**Raises**

- *ValueError*: If `mode` is unknown, or `'hmc'` is requested with a calculator without forces.

#### `Replica.sweep`

```python
Replica.sweep(n_steps: int, p_displace: float = 0.5, p_insert: float = 0.25, p_delete: float = 0.25, p_displace_substrate: float = 0.0) -> Replica
```

Run a batch of randomly-chosen Monte Carlo moves.

`p_displace_substrate` (default 0, i.e. off) adds an independent
single-particle MC move for substrate atoms, on top of whatever
substrate motion already happens via HMC's joint trajectory when
`hmc_enabled=True` -- the two aren't mutually exclusive. With no
substrate atoms present, this is a no-op regardless of the
probability given (see `_move_displace_substrate`).

**Parameters**

- `n_steps` (*int*): Number of moves to attempt in this sweep.
- `p_displace` (*float, optional*): Relative probability of a displacement move (single-particle MC, or the joint HMC trajectory if `self.hmc_enabled` is True). Default is 0.5.
- `p_insert` (*float, optional*): Relative probability of a GCMC insertion move. Default is 0.25.
- `p_delete` (*float, optional*): Relative probability of a GCMC deletion move; must equal `p_insert`. Default is 0.25.
- `p_displace_substrate` (*float, optional*): Relative probability of an independent substrate displacement move. Default is 0.0 (off).

**Returns**

- *Replica*: `self`, returned so a process-pool worker can send the mutated replica back to the main process.

**Raises**

- *ValueError*: If `p_insert` differs from `p_delete` (the acceptance rules assume equal attempt probabilities).

#### `Replica.check_energy`

```python
Replica.check_energy(tol: float = 1e-06) -> Replica
```

Compare the running energy with a full recomputation, and resync it.

MC moves update the energy incrementally (`energy += dU`); this
catches any drift (round-off, or a calculator whose incremental and
full energies disagree). A drift larger than `tol` per atom is
stored in `last_energy_drift` and the energy is reset to the
recomputed value; a smaller one is only recorded, so runs whose
energies agree to round-off stay bit-identical.

**Parameters**

- `tol` (*float, optional*): Tolerance per atom (substrate + mobile), in energy units. Default 1e-6.

**Returns**

- *Replica*: `self` (for process-pool workers), with `last_energy_drift` (running minus recomputed energy) and `energy_resynced`.

#### `Replica.acceptance_rates`

```python
Replica.acceptance_rates() -> Dict[str, float]
```

Compute the acceptance rate of each move type attempted so far.

**Returns**

- *dict of str to float*: One entry per move type (`'displace'`, `'insert'`, `'delete'`, `'hmc'`, `'displace_substrate'`, and with several exchanged species `'insert_<s>'`/`'delete_<s>'`): the fraction of attempts of that type that were accepted (0.0 for a move type never attempted).

## The replica-exchange engine (`regc.engine`)

### `ReplicaExchangeGCMC`

```python
ReplicaExchangeGCMC(temperatures: List[float], mus, system_factory: Callable[[np.random.Generator], ParticleSystem], seed: int = 0, max_disp: float = 0.2, n_workers: int = 1, lambda_cubed_of_T: Callable[[float], float] = <function ReplicaExchangeGCMC.<lambda>>, hmc_enabled: bool = False, hmc_leapfrog_steps: int = 10, hmc_dt: float = 0.005, adsorbed_cutoff: Optional[float] = None, observables: Optional[Sequence[Observable]] = None, configurations_dir: Optional[str] = None, configurations_every: int = 1, skip_empty_swaps: bool = True, records_path: Optional[str] = None, records_potential: str = 'sampling', records_every: int = 1)
```

Manages a 2D ladder of replicas over (temperature, chemical potential).
`system_factory(rng) -> ParticleSystem` builds a fresh system for each
grid point -- this is what lets the same exchange engine drive any
geometry/potential combination: the exchange acceptance criterion only
ever needs (beta, mu, U, N), never anything about the geometry or
calculator in use.

`mus` can be either:
  * 1D, length n_mu -- the same chemical potentials used at every
    temperature.
  * 2D, shape (n_T, n_mu) -- an explicit mu for every (T, mu) grid
    point (what a *pressure* ladder needs; see `pressure.py`).

Samples for MBAR are recorded per grid point as plain (U, N) pairs after
`burn_in_cycles`, every `record_every` cycles (see `analysis.py`).

Several exchanged species (`system.exchange_species = ['O', 'Cl']`):
each grid point then has one mu per species, so `mus` gets a trailing
species axis -- shape (n_mu, n_species), shared across temperatures, or
(n_T, n_mu, n_species). The ladder stays 2D: its second axis is a path
through (mu_O, mu_Cl, ...) space, e.g. a gas feed of fixed composition
at increasing pressure. `lambda_cubed_of_T` then returns one Lambda^3
per species (pass a list of per-species callables, or a
`SpeciesLambdaCubed`), and samples also record the per-species counts
under `'N_s'`. With one species everything is exactly as before.

`lambda_cubed_of_T(T) -> float` sets each replica's thermal de Broglie
wavelength cubed (see `ThermoState.lambda_cubed`); the default (always
1.0) is correct for reduced LJ units and a no-op for existing code. Real-
units setups (e.g. an ASE EMT/EAM/MLIP calculator) should pass the true
value -- see `regc.units.thermal_wavelength_cubed_angstrom3`.

`hmc_enabled=True` (with `hmc_leapfrog_steps`/`hmc_dt`) replaces every
replica's single-particle random displacement move with Hybrid Monte
Carlo -- collective, force-directed proposals instead of a blind random
walk. Needs a Calculator with `supports_forces = True` (see
`Replica`'s docstring for the full explanation, correctness guarantee,
and current limitations).

Call `.close()` (or use as a context manager) to shut down the worker
pool when you're done -- otherwise the worker processes linger.

**Constructor**

Construct the replica-exchange ladder.

**Parameters**

- `temperatures` (*list of float*): The temperature ladder, length n_T.
- `mus` (*array_like*): Either 1D (length n_mu, shared across all temperatures) or 2D (shape (n_T, n_mu), an explicit mu for every grid point -- what a pressure ladder needs; see `pressure.py`). With several exchanged species, shape (n_mu, n_species) or (n_T, n_mu, n_species), species in `system.exchange_species` order.
- `system_factory` (*callable*): A function `system_factory(rng) -> ParticleSystem` that builds a fresh system for each grid point, given that grid point's own random number generator.
- `seed` (*int, optional*): Master random seed; each grid point's own RNG stream is derived from it deterministically. Default is 0.
- `max_disp` (*float, optional*): Maximum single-particle displacement (per axis) for plain MC displacement moves, passed to every `Replica`. Default is 0.2.
- `n_workers` (*int, optional*): Number of worker processes for parallel local sweeps. Default is 1 (serial, no process pool).
- `lambda_cubed_of_T` (*callable or sequence of callable, optional*): A function `lambda_cubed_of_T(T) -> float` giving each replica's thermal de Broglie wavelength cubed (see `ThermoState.lambda_cubed`). Default always returns 1.0, correct for reduced LJ units. With several exchanged species it may return one value per species, or be a list of one callable per species (wrapped in `SpeciesLambdaCubed`).
- `hmc_enabled` (*bool, optional*): If True, every replica uses Hybrid Monte Carlo instead of plain single-particle MC displacement (see `Replica`'s docstring). Default is False.
- `hmc_leapfrog_steps` (*int, optional*): Number of leapfrog integration steps per HMC trial, passed to every `Replica`. Default is 10.
- `hmc_dt` (*float, optional*): Leapfrog integration timestep for HMC, passed to every `Replica`. Default is 0.005.
- `adsorbed_cutoff` (*float, optional*): If given, every sample also records `system.n_adsorbed(adsorbed_cutoff)` under `'N_ads'`, so the adsorbed count can be reweighted like N (see `MBARSurfaceAnalysis`, `observable='N_ads'`). Default None (not recorded).
- `observables` (*sequence of regc.observables.Observable, optional*): Structural observables (coordination numbers, RDFs, ...) evaluated on every recorded sample and stored under `samples[key]['obs:<name>']`, for MBAR reweighting like N. Default None.
- `configurations_dir` (*str, optional*): If given, every `configurations_every`-th recorded sample's configuration is appended to `<configurations_dir>/configs_T{it}_mu{imu}.extxyz`, tagged with its sample index, so structures can be picked out of the phase diagram afterwards (`MBARSurfaceAnalysis.representative_samples`, `regc.analysis.export_phase_configurations`). Default None (off); see also `save_configurations`.
- `configurations_every` (*int, optional*): Store every this many samples' configurations. Default 1 (all).
- `skip_empty_swaps` (*bool, optional*): In `run_regc`, draw the replica-exchange move only among the (pattern, offset) combinations that have at least one pair on this grid, instead of among all of them (on an n_T x 1 grid, six of the eight draws would otherwise attempt nothing). The choice does not depend on the configurations, so either way the sampling is exact; True wastes no RE steps. Default True; a run resumed from a checkpoint keeps the setting it was started with (False for checkpoints written before this option existed), so it continues bit-identically.
- `records_path` (*str, optional*): If given, every recorded sample is appended to this HDF5 file (`regc.records`): grid point, sample index, walker, N, the energy under the sampling potential and (every `records_every`-th sample) the configuration, so the samples can later be re-evaluated with another potential (`regc.records.reevaluate`) and reweighted to it (`MBARSurfaceAnalysis.with_potential`). Needs h5py. Default None (off); see also `save_records`.
- `records_potential` (*str, optional*): Name of the sampling potential in the records file. Default `'sampling'`.
- `records_every` (*int, optional*): Store the configuration of every this many samples (the energy and N of all of them). Default 1.

**Raises**

- *ValueError*: If `mus` doesn't have one of the shapes above (for the number of exchanged species the systems declare), if the systems don't all exchange the same species, or if `lambda_cubed_of_T` doesn't give one value (or one per species).

#### `ReplicaExchangeGCMC.close`

```python
ReplicaExchangeGCMC.close() -> None
```

Shut down the worker pool, if one was created.

**Returns**

- *None*

#### `ReplicaExchangeGCMC.swap_pairs`

```python
ReplicaExchangeGCMC.swap_pairs(swap_type: str, parity: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]
```

List the disjoint replica pairs of one swap pattern ([ZSG] Fig. 2).

**Parameters**

- `swap_type` (*str*): One of `SWAP_TYPES`: `'T'` pairs (l, m)-(l+1, m); `'mu'` pairs (l, m)-(l, m+1); `'cross_left'` pairs (l, m+1)-(l+1, m); `'cross_right'` pairs (l, m)-(l+1, m+1).
- `parity` (*int*): 0 or 1: the offset of the first pair along the axis being paired (T for 'T' and both diagonals, mu for 'mu'). Alternating it is what connects e.g. T_2 with T_3 -- with a single offset the ladder would split into disconnected blocks.

**Returns**

- *list of tuple*: `(key_a, key_b)` pairs; no grid point appears more than once.

**Raises**

- *ValueError*: If `swap_type` is not one of `SWAP_TYPES`.

#### `ReplicaExchangeGCMC.check_energies`

```python
ReplicaExchangeGCMC.check_energies(tol: float = 1e-06) -> Dict[Tuple[int, int], Tuple[float, bool]]
```

Check every replica's running energy against a full recomputation.

See `Replica.check_energy`; runs in the worker processes of a
parallel run. `regc.driver.run_with_checkpoints` calls it after
every chunk.

**Parameters**

- `tol` (*float, optional*): Tolerance per atom. Default 1e-6.

**Returns**

- *dict*: `{key: (drift, resynced)}`.

#### `ReplicaExchangeGCMC.pair_acceptance`

```python
ReplicaExchangeGCMC.pair_acceptance() -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, float]]
```

Swap attempts and acceptance ratio of every neighbouring pair.

A pattern's average can hide a single pair that never swaps and so
cuts the ladder in two; this lists them one by one.

**Returns**

- *dict*: `{(key_a, key_b): (attempts, acceptance)}`.

#### `ReplicaExchangeGCMC.round_trip_counts`

```python
ReplicaExchangeGCMC.round_trip_counts() -> Dict[str, np.ndarray]
```

Completed round trips of every walker along T and along mu.

A walker that never completes a round trip along T has not carried
a low-temperature configuration through the hot replicas and back:
the low-T replicas may still remember their start.

**Returns**

- *dict*: `{'T': array, 'mu': array}`, one entry per walker (completed end-to-end-and-back trips).

#### `ReplicaExchangeGCMC.run`

```python
ReplicaExchangeGCMC.run(n_cycles: int, sweeps_per_cycle: int = 200, exchange_every: int = 1, record_every: int = 1, burn_in_cycles: int = 0, verbose: bool = True, trajectory_dir: Optional[str] = None, trajectory_every: Optional[int] = None, p_displace: float = 0.5, p_insert: float = 0.25, p_delete: float = 0.25, p_displace_substrate: float = 0.0, diagonal_swaps: bool = True) -> None
```

Run the replica-exchange simulation.

`trajectory_dir`, if given, saves each replica's configuration
(positions + species, as an extxyz frame -- see trajectory.py) to
its own file under that directory every `trajectory_every` cycles
(defaulting to `record_every`). One file per grid point:
`<trajectory_dir>/replica_T{it}_mu{imu}.xyz`. Requires the
system's geometry to define a `cell` (`BulkSystem` and
`SlabSystem` both do; pip install ase is also needed, for
writing the file itself).

`p_displace`/`p_insert`/`p_delete`/`p_displace_substrate` set the
relative probability of each move type per step (renormalized to
sum to 1, same as `Replica.sweep`); `p_displace_substrate` only
does anything if the system actually has substrate atoms (see
`Replica._move_displace_substrate`) and is independent of whatever
substrate motion already happens via HMC's joint trajectory when
`hmc_enabled=True` was passed at construction.

**Parameters**

- `n_cycles` (*int*): Total number of cycles to run.
- `sweeps_per_cycle` (*int, optional*): Number of Monte Carlo moves per replica per cycle. Default is 200.
- `exchange_every` (*int, optional*): Attempt a full round of replica exchanges every this many cycles. Default is 1 (every cycle).
- `record_every` (*int, optional*): Record an (U, N) sample per grid point every this many cycles (after `burn_in_cycles`). Default is 1 (every cycle).
- `burn_in_cycles` (*int, optional*): Number of initial cycles to run without recording samples. Default is 0.
- `verbose` (*bool, optional*): If True, print progress roughly ten times over the run. Default is True.
- `trajectory_dir` (*str, optional*): Directory to save trajectory frames to; if None, no trajectories are saved. Default is None.
- `trajectory_every` (*int, optional*): Save a trajectory frame every this many cycles. Default is None, which uses `record_every` if `trajectory_dir` is given.
- `p_displace` (*float, optional*): Relative probability of a displacement move. Default is 0.5.
- `p_insert` (*float, optional*): Relative probability of a GCMC insertion move. Default is 0.25.
- `p_delete` (*float, optional*): Relative probability of a GCMC deletion move. Default is 0.25.
- `p_displace_substrate` (*float, optional*): Relative probability of an independent substrate displacement move. Default is 0.0 (off).
- `diagonal_swaps` (*bool, optional*): If True, each exchange round also sweeps the two diagonal patterns ('cross_left', 'cross_right' of [ZSG] Fig. 2) after the T and mu ones. Default is True.

**Returns**

- *None*

#### `ReplicaExchangeGCMC.run_regc`

```python
ReplicaExchangeGCMC.run_regc(n_steps: int, x0: float = 0.99, diffusion: str = 'mc', diffusion_steps: int = 10, record_every: int = 100, burn_in_steps: int = 0, swap_types: Tuple[str, ...] = ('T', 'mu', 'cross_left', 'cross_right'), convergence_every: Optional[int] = None, convergence_tol: float = 0.05, convergence_bins: int = 30, verbose: bool = True, trajectory_dir: Optional[str] = None, trajectory_every: Optional[int] = None, **diffusion_kwargs) -> bool
```

Run the REGC scheme exactly as in [ZSG] Sec. II A / Fig. 1.

One REGC step is:

1. Draw `rand` uniform in [0, 1). If `rand < x0`, every replica
   makes ONE particle insertion/removal attempt (removal or
   insertion with probability 1/2 each, Eqs. (4)-(5) of
   docs/REGC_algorithm.md). Otherwise make ONE replica-exchange
   move: draw one pattern from `swap_types` and an offset (0/1),
   each uniformly (with `skip_empty_swaps`: one uniformly among
   the combinations that have pairs on this grid), and attempt all
   of that pattern's disjoint swaps (Eq. (8)).
2. Diffusion at fixed (N, V, T), in parallel over replicas (see
   `Replica.diffuse`).
3. Every `record_every` steps after `burn_in_steps`, record (U, N)
   of every replica -- i.e. after the last diffusion move, before
   the next GC/RE step.
4. Every `convergence_every` steps, test convergence of the
   reduced-energy densities rho(u): compare the samples recorded
   since the previous check with those of the interval before it
   (disjoint blocks, see `rho_u_change`); stop if they agree,
   otherwise continue up to `n_steps`.

`x0` follows the paper's TEXT: it is the probability of a GC
attempt ([ZSG]: x0 = 0.99 with "swaps attempted every 100 REGC
steps"; x0 = 0.9 for the Si clusters). [ZSG] Fig. 1 draws the
opposite branch labels (rand <= x0 -> replica exchange); to use
that convention pass `1 - x0`.

**Parameters**

- `n_steps` (*int*): Maximum number of REGC steps.
- `x0` (*float, optional*): Probability of a particle insertion/removal step (vs. a replica exchange step). Default is 0.99, as in [ZSG] Sec. III A.
- `diffusion` (*str, optional*): `'mc_moves'`, `'mc'`, `'hmc'` or `'md'`; see `Replica.diffuse`. Default is `'mc'`.
- `diffusion_steps` (*int, optional*): MC moves, MC cycles, HMC trajectories, or MD steps per REGC step. Default is 10 ([ZSG]: "about ten"; for their LJ surface, ten MC moves, i.e. `diffusion='mc_moves'`).
- `record_every` (*int, optional*): Record a sample every this many REGC steps. Default is 100.
- `burn_in_steps` (*int, optional*): Steps before recording starts. Default is 0.
- `swap_types` (*tuple of str, optional*): Swap patterns to draw from. Default is all four of `SWAP_TYPES` (including the diagonals).
- `convergence_every` (*int, optional*): Check convergence every this many steps (rounded to a multiple of `record_every`). Default is None (no check; run `n_steps`).
- `convergence_tol` (*float, optional*): Converged when the largest total-variation distance between the rho(u) of the two latest inter-check blocks, over all grid points, is below this. Default is 0.05. Two blocks of n samples from the same distribution still differ by roughly sqrt(convergence_bins / (pi n)) from sampling noise alone, so the tolerance must lie above that floor (fewer bins or longer blocks). The first verdict needs three checks.
- `convergence_bins` (*int, optional*): Histogram bins for rho(u). Default is 30.
- `verbose` (*bool, optional*): Print progress about ten times over the run. Default is True.
- `trajectory_dir` (*str, optional*): As in `run`. Default is None.
- `trajectory_every` (*int, optional*): As in `run`; defaults to `record_every`.
- `**diffusion_kwargs`: Passed on to `Replica.diffuse` (e.g. `timestep_fs`, `thermostat`, `taut_fs` for MD; `displace_substrate` for MC).

**Returns**

- *bool*: True if the convergence criterion was met before `n_steps`.

**Raises**

- *ValueError*: If `x0` is outside [0, 1] or `swap_types` contains an unknown pattern.

#### `ReplicaExchangeGCMC.rho_u_change`

```python
ReplicaExchangeGCMC.rho_u_change(n_prev: int, bins: int = 30, n_start: int = 0) -> float
```

Largest change of the reduced-energy density rho(u) between blocks.

For each grid point, u = beta*(U - mu*N) + N*ln(Lambda^3) of its
recorded samples (Eq. (9) of docs/REGC_algorithm.md; summed over
species if several are exchanged) is histogrammed on common bins
for two disjoint blocks: samples `n_start:n_prev` (the interval
before the previous check) and samples `n_prev:` (since it).
Returns the maximum over grid points of the total-variation
distance 0.5 * sum |p_new - p_old| -- [ZSG]'s criterion, "no change
in the density of reduced-energy states rho(U) with increasing
simulation steps".

The blocks are disjoint on purpose: comparing the first `n_prev`
samples with all samples (which contain them) shrinks like 1/k
after k equal checks whatever the data do, so a drifting run would
pass. Two blocks of n samples drawn from one distribution still
differ by about sqrt(bins / (pi n)) (sampling noise), so a tolerance
must lie above that.

**Parameters**

- `n_prev` (*int*): Number of samples per grid point at the previous check: the end of the old block, the start of the new one.
- `bins` (*int, optional*): Number of histogram bins. Default 30.
- `n_start` (*int, optional*): Start of the old block (the sample count at the check before the previous one). Default 0.

**Returns**

- *float*: The maximum total-variation distance, in [0, 1].

#### `ReplicaExchangeGCMC.fingerprint`

```python
ReplicaExchangeGCMC.fingerprint() -> dict
```

What a checkpoint must agree with to be loaded into this simulation.

**Returns**

- *dict*: The temperature ladder, the mu grid and Lambda^3 on the ladder.

#### `ReplicaExchangeGCMC.save_checkpoint`

```python
ReplicaExchangeGCMC.save_checkpoint(path: str, extra: Optional[dict] = None) -> None
```

Save everything needed to continue this simulation exactly.

Replica configurations, energies and random streams, the engine's
random stream, the samples, swap statistics and convergence history,
plus `extra` (anything picklable, e.g. a driver's step counter). The
write is atomic. A run continued with `load_checkpoint` is
bit-identical to an uninterrupted one.

**Parameters**

- `path` (*str*): Checkpoint file (pickle).
- `extra` (*dict, optional*): Additional state to store and get back from `load_checkpoint`.

#### `ReplicaExchangeGCMC.load_checkpoint`

```python
ReplicaExchangeGCMC.load_checkpoint(path: str) -> dict
```

Restore a state saved by `save_checkpoint` into this simulation.

The simulation must have been constructed with the same temperature
ladder, mu grid, Lambda^3(T) and `adsorbed_cutoff`. The
checkpoint supplies the configurations, energies, random streams
and samples; the calculators and the move settings (`max_disp`,
HMC) are this simulation's, built from the current setup. So a
checkpoint survives a moved potential file, and a changed potential
setting (e.g. a new hard core) takes effect: each restored energy is
recomputed, and resynchronised with a warning if it changed.

**Parameters**

- `path` (*str*): Checkpoint file. Only load files you created (pickle).

**Returns**

- *dict*: The `extra` dict given to `save_checkpoint`.

**Raises**

- *ValueError*: If the checkpoint's temperature ladder, mu grid, Lambda^3 or `adsorbed_cutoff` differ from this simulation's.

#### `ReplicaExchangeGCMC.swap_acceptance`

```python
ReplicaExchangeGCMC.swap_acceptance() -> Dict[str, float]
```

Replica-exchange acceptance ratio per swap pattern.

[ZSG] Sec. III A tunes the (T, mu) spacing to about 25%.

**Returns**

- *dict of str to float*: Accepted / attempted swaps for each of `SWAP_TYPES` (0.0 if never attempted).

#### `ReplicaExchangeGCMC.save_configurations`

```python
ReplicaExchangeGCMC.save_configurations(directory: Optional[str], every: int = 1) -> None
```

Store the configurations of recorded samples from now on (or stop).

**Parameters**

- `directory` (*str or None*): Directory for `configs_T{it}_mu{imu}.extxyz` (one per grid point); None stops storing.
- `every` (*int, optional*): Store every this many samples (by sample index). Default 1.

**Raises**

- *ValueError*: If `every` is not positive.

#### `ReplicaExchangeGCMC.save_records`

```python
ReplicaExchangeGCMC.save_records(path: Optional[str], potential: str = 'sampling', every: int = 1, positions_dtype: str = 'float64') -> None
```

Append every recorded sample to an HDF5 records file from now on (or stop).

See `regc.records`. The file is opened at the first recorded
sample and flushed at the end of every `run`/`run_regc` call and
at every checkpoint.

**Parameters**

- `path` (*str or None*): The records file; None stops recording (and closes the file).
- `potential` (*str, optional*): Name of the sampling potential. Default `'sampling'`.
- `every` (*int, optional*): Store the configuration of every this many samples (by sample index). Default 1.
- `positions_dtype` (*str, optional*): `'float64'` (default) or `'float32'`.

**Raises**

- *ValueError*: If `every` is not positive.

#### `ReplicaExchangeGCMC.n_substrate` (property)

int: Displaceable (non-frozen) substrate atoms per replica (for the kinetic heat capacity).

#### `ReplicaExchangeGCMC.observables_meta` (property)

dict: `{name: metadata}` of the recorded observables (see `regc.observables`).

#### `ReplicaExchangeGCMC.summary`

```python
ReplicaExchangeGCMC.summary() -> List[dict]
```

Build a per-grid-point summary of the current simulation state.

**Returns**

- *list of dict*: One dict per grid point (sorted by grid coordinates), with keys `'T_index'`, `'mu_index'`, `'T'`, `'mu'`, `'N'`, `'energy'`, `'n_samples'`, `'exchange_acc'`, `'displace_acc'`, `'insert_acc'`, `'delete_acc'`, plus `'N_s'` (counts per species) with several exchanged species.

---

### `SWAP_TYPES`

```python
SWAP_TYPES = ('T', 'mu', 'cross_left', 'cross_right')
```

## Checkpointed runs (`regc.driver`)

### `run_with_checkpoints`

```python
run_with_checkpoints(sim, out_dir: str, n_steps: int, chunk: int = 1000, x0: float = 0.99, diffusion: str = 'mc', diffusion_steps: int = 10, record_every: int = 100, convergence_every: Optional[int] = None, convergence_tol: float = 0.05, time_limit: Optional[float] = None, resume: bool = True, overwrite: bool = False, metadata: Optional[dict] = None, progress: Optional[Callable[[str], None]] = <built-in function print>, energy_tol: float = 1e-06, **diffusion_kwargs) -> RunStatus
```

Run `sim.run_regc` in checkpointed chunks.

**Parameters**

- `sim` (*ReplicaExchangeGCMC*): The simulation (freshly constructed; on resume its state is replaced by the checkpoint's).
- `out_dir` (*str*): Directory for `samples.npz`, `checkpoint.pkl` and `run_metadata.json`.
- `n_steps` (*int*): Total REGC steps of the run (over all resumptions).
- `chunk` (*int, optional*): Steps between checkpoints; a multiple of `record_every` (and of `trajectory_every` if given). Default 1000.
- `x0, diffusion, diffusion_steps, record_every`: Passed to `ReplicaExchangeGCMC.run_regc`.
- `convergence_every` (*int, optional*): Check rho(u) convergence every this many steps (rounded up to a multiple of `chunk`); also at the end. Default None: no checks.
- `convergence_tol` (*float, optional*): Converged when rho(u) of the samples since the last check and of the interval before it differ (largest total-variation distance) by less than this; see `ReplicaExchangeGCMC.rho_u_change` for the noise floor. Default 0.05.
- `time_limit` (*float, optional*): Seconds: stop (`finished=False`) before starting a chunk that is expected to end later than this, measured from the call. Default None (no limit).
- `resume` (*bool, optional*): Continue the run saved in `out_dir` if there is one. Default True.
- `overwrite` (*bool, optional*): Start afresh even if `out_dir` holds a run (with `resume=False`). Default False.
- `metadata` (*dict, optional*): JSON-serialisable description of the run (potential, lattice constant, protocol, ...). Saved on a fresh start; on resume every key must match the saved value.
- `progress` (*callable, optional*): Called with a one-line status after every chunk. Default `print`; None for silence.
- `energy_tol` (*float, optional*): After every chunk each replica's running energy is compared with a full recomputation (`ReplicaExchangeGCMC.check_energies`); a drift above this, per atom, is resynced with a warning. Default 1e-6.
- `**diffusion_kwargs`: Passed to `run_regc` (e.g. `timestep_fs`, `taut_fs`).

**Returns**

- *RunStatus*: Steps done, whether finished/converged, and the convergence history.

**Raises**

- *FileExistsError*: If `out_dir` holds a run and neither `resume` nor `overwrite` is set.
- *ValueError*: If `chunk` is not a multiple of `record_every`; on resume, if the checkpoint or `metadata` do not match `sim`.

---

### `RunStatus`

```python
RunStatus(steps_done: int, finished: bool, converged: bool, rho_u_changes: List[float] = <factory>, resumed_from: int = 0) -> None
```

Outcome of `run_with_checkpoints`.

**Attributes**

- `steps_done` (*int*): REGC steps completed so far (over all resumptions).
- `finished` (*bool*): True if the run reached its step count or converged; False if it stopped for the time limit (continue with `resume=True`).
- `converged` (*bool*): True if the rho(u) convergence criterion was met.
- `rho_u_changes` (*list of float*): The convergence measure at every check.
- `resumed_from` (*int*): Step the run was resumed from (0 for a fresh start).

---

### `EXIT_RESUBMIT`

```python
EXIT_RESUBMIT = 75
```

## Sample I/O (`regc.io`)

### `save_samples`

```python
save_samples(sim, path: str) -> None
```

Write a simulation's (or `SampleSet`'s) samples to an `.npz` file.

The write is atomic (temporary file + rename). The Lambda^3 convention
is stored as the particle mass for a `ThermalWavelengthCubed` (the
per-species masses for a `SpeciesLambdaCubed` of them), as "unit" for
Lambda^3 = 1; any other callable is stored by its values on the ladder
and must be passed again to `load_samples`.

**Parameters**

- `sim` (*ReplicaExchangeGCMC or SampleSet*): Anything with `temperatures`, `mu_grid`, `samples` and `lambda_cubed_of_T`.
- `path` (*str*): Output file (`.npz`).

**Raises**

- *ValueError*: For a merged set (`merge_samples`): save the runs separately.

---

### `load_samples`

```python
load_samples(path: str, discard: float = 0.0, lambda_cubed_of_T: Optional[Callable[[float], float]] = None, temperatures: Optional[List[float]] = None, mu_grid: Optional[np.ndarray] = None) -> SampleSet
```

Read samples written by `save_samples`.

**Parameters**

- `path` (*str*): The `.npz` file.
- `discard` (*float, optional*): Fraction of each replica's samples to drop as equilibration. Default 0.0.
- `lambda_cubed_of_T` (*callable, optional*): Lambda^3(T); needed only if the file was written with a custom callable, or holds only (U, N). Default None (use what the file records).
- `temperatures` (*list of float, optional*): The temperature ladder; needed only for files holding only (U, N) (written before this format). Default None (from the file).
- `mu_grid` (*numpy.ndarray, optional*): The mu grid, shape (n_T, n_mu) or (n_mu,) (with several species, (n_T, n_mu, n_species) or (n_mu, n_species)); as `temperatures`.

**Returns**

- *SampleSet*: The samples, ready for `MBARSurfaceAnalysis`.

**Raises**

- *ValueError*: If the file lacks the ladder/grid and they are not given.

---

### `SampleSet`

```python
SampleSet(temperatures: List[float], mu_grid: np.ndarray, samples: Dict[Tuple[int, int], Dict[str, list]], lambda_cubed_of_T: Callable[[float], float] = <function _unit_lambda_cubed>, exchange_species: Optional[Tuple[str, ...]] = None, observables_meta: Optional[dict] = None, n_substrate: Optional[int] = None, first_index: Union[int, Dict] = 0, state_table: Optional[Dict] = None, run_labels: Optional[List[str]] = None) -> None
```

Recorded REGC samples, detached from the simulation that produced them.

Has exactly the attributes `MBARSurfaceAnalysis` reads from a
`ReplicaExchangeGCMC`, so either can be analysed.

**Attributes**

- `temperatures` (*list of float*): The temperature ladder (k_B*T in eV, or reduced units).
- `mu_grid` (*numpy.ndarray*): Chemical potentials, shape (n_T, n_mu), or (n_T, n_mu, n_species) with several exchanged species.
- `samples` (*dict*): `{(it, imu): {'U': [...], 'N': [...]}}`, plus `'N_ads'` if the run recorded it (`adsorbed_cutoff`) and `'N_s'` (counts per species) with several exchanged species.
- `lambda_cubed_of_T` (*callable*): Lambda^3 as a function of temperature (see `ThermalWavelengthCubed`, `SpeciesLambdaCubed`).
- `exchange_species` (*tuple of str, optional*): The exchanged species, in the order of the last axis of `mu_grid` and of `'N_s'`. None if unknown (files from older versions).
- `observables_meta` (*dict, optional*): `{name: metadata}` of the observables recorded under `samples[key]['obs:<name>']` (see `regc.observables`).
- `n_substrate` (*int, optional*): Displaceable substrate atoms per replica (kinetic part of the heat capacity). None if unknown (files from older versions).
- `first_index` (*int or dict*): Index, in the run's own numbering, of the first sample kept per grid point (`discard` drops the earlier ones); maps samples to the stored configurations. A dict `{key: index}` when it differs between states (merged runs). Default 0.
- `state_table` (*dict, optional*): `{key: (T, mu)}`: the thermodynamic state of every key, for sets that are not one rectangular (temperature x mu) grid -- the union of several runs (`merge_samples`), whose keys are `(run, it, imu)`. Default None: `temperatures[key[0]]` and `mu_grid[key]`.
- `run_labels` (*list of str, optional*): With `state_table` from `merge_samples`: a label per run (e.g. its output directory), indexed by `key[0]`. Default None.

#### `SampleSet.lambda_cubed_of_T`

```python
SampleSet.lambda_cubed_of_T(T: float) -> float
```

Lambda^3 = 1 at every temperature (the reduced-units convention).

**Parameters**

- `T` (*float*): Temperature (unused).

**Returns**

- *float*: 1.0.

#### `SampleSet.n_samples` (property)

int: Samples per grid point (the smallest, if they differ).

#### `SampleSet.discard`

```python
SampleSet.discard(fraction: float) -> SampleSet
```

Drop the first part of every replica's samples (equilibration).

**Parameters**

- `fraction` (*float*): Fraction in [0, 1) of each replica's samples to drop.

**Returns**

- *SampleSet*: A new set with the remaining samples.

**Raises**

- *ValueError*: If `fraction` is not in [0, 1).

## Units (`regc.units`)

### `kelvin_to_eV`

```python
kelvin_to_eV(T_kelvin: float) -> float
```

Convert a real temperature to the eV energy scale.

Computes k_B * T, in eV -- the natural energy-scale temperature for a
Calculator (like EMT, or most EAM/MLIP potentials) that reports
energies in eV. `ThermoState.beta = 1/temperature` needs `temperature`
in the same units as the energies it multiplies, so a real Kelvin
temperature must be converted via this function before use in real
(rather than reduced Lennard-Jones) units.

**Parameters**

- `T_kelvin` (*float*): Temperature in Kelvin.

**Returns**

- *float*: k_B * T_kelvin, in eV. At 300 K this is about 0.02585 eV (the textbook "kT ~ 25.7 meV at room temperature").

**See Also**

thermal_wavelength_angstrom : the companion conversion for
    `ThermoState.lambda_cubed`.

---

### `eV_to_kelvin`

```python
eV_to_kelvin(T_eV: float) -> float
```

Convert an energy-scale temperature k_B*T (eV) back to Kelvin.

The inverse of `kelvin_to_eV`, valid for any value (not only for the
temperatures of a replica ladder).

**Parameters**

- `T_eV` (*float*): k_B * T, in eV.

**Returns**

- *float*: The temperature in Kelvin.

---

### `pascal_per`

```python
pascal_per(unit: str) -> float
```

Pascal per pressure unit.

**Parameters**

- `unit` (*str*): One of the keys of `PRESSURE_UNITS` (`'Pa'`, `'bar'`, `'atm'`).

**Returns**

- *float*: How many Pascal one `unit` is.

**Raises**

- *ValueError*: If `unit` is not a known pressure unit.

---

### `PRESSURE_UNITS`

```python
PRESSURE_UNITS = {'Pa': 1.0, 'bar': 100000.0, 'atm': 101325.0}
```

---

### `ThermalWavelengthCubed`

```python
ThermalWavelengthCubed(mass_amu: float)
```

Lambda^3 of a particle of given mass as a function of k_B*T in eV.

The callable `ReplicaExchangeGCMC`'s and `MBARSurfaceAnalysis`'s
`lambda_cubed_of_T` argument expects in real units: temperatures in
this package are k_B*T in eV, lengths in Angstrom. Unlike an ad-hoc
lambda it is picklable, valid at any temperature (MBAR evaluates it off
the replica ladder), and records its mass, so saved samples can restore
it (see `regc.io`).

**Parameters**

- `mass_amu` (*float*): Mass of the exchanged particle, in atomic mass units.

**Examples**

>>> lam3 = ThermalWavelengthCubed(1.008)          # H atoms
>>> round(lam3(kelvin_to_eV(300.0)), 4)           # Angstrom^3 at 300 K
1.0119

**Constructor**

Store the particle mass.

**Parameters**

- `mass_amu` (*float*): Mass of the exchanged particle, in atomic mass units.

---

### `SpeciesLambdaCubed`

```python
SpeciesLambdaCubed(per_species)
```

Lambda^3(T) of several exchanged species, as one callable.

What `ReplicaExchangeGCMC`'s `lambda_cubed_of_T` takes when more than
one species is exchanged (e.g. O and Cl): calling it returns one
Lambda^3 per species, in the order of `system.exchange_species`.
Picklable, and saved/restored by `regc.io` when every entry is a
`ThermalWavelengthCubed`.

**Parameters**

- `per_species` (*sequence of callable*): One `lambda_cubed_of_T(T) -> float` per exchanged species, e.g. `[ThermalWavelengthCubed(15.999), ThermalWavelengthCubed(35.45)]` or each reservoir's `Reservoir.lambda_cubed_of_T()`.

**Examples**

>>> lam3 = SpeciesLambdaCubed([ThermalWavelengthCubed(15.999),
...                            ThermalWavelengthCubed(35.45)])
>>> lam3(kelvin_to_eV(300.0)).shape
(2,)

**Constructor**

Store the per-species callables.

**Parameters**

- `per_species` (*sequence of callable*): One `lambda_cubed_of_T(T) -> float` per exchanged species.

#### `SpeciesLambdaCubed.masses_amu` (property)

list of float or None: per-species masses, if all are `ThermalWavelengthCubed`.

---

### `thermal_wavelength_angstrom`

```python
thermal_wavelength_angstrom(T_kelvin: float, mass_amu: float) -> float
```

Compute the 3D ideal-gas thermal de Broglie wavelength.

Evaluates Lambda(T) = h / sqrt(2*pi*m*k_B*T), in Angstrom, for a
particle of the given mass at the given temperature.

**Parameters**

- `T_kelvin` (*float*): Temperature in Kelvin.
- `mass_amu` (*float*): Particle mass in atomic mass units.

**Returns**

- *float*: The thermal de Broglie wavelength Lambda(T), in Angstrom.

**See Also**

thermal_wavelength_cubed_angstrom3 : Lambda(T) cubed, what
    `ThermoState.lambda_cubed` actually wants.

---

### `thermal_wavelength_cubed_angstrom3`

```python
thermal_wavelength_cubed_angstrom3(T_kelvin: float, mass_amu: float) -> float
```

Compute the thermal de Broglie wavelength cubed, in cubic Angstrom.

**Parameters**

- `T_kelvin` (*float*): Temperature in Kelvin.
- `mass_amu` (*float*): Particle mass in atomic mass units.

**Returns**

- *float*: Lambda(T)^3, in Angstrom^3 -- what `ThermoState.lambda_cubed` wants when positions/volume are in Angstrom (e.g. for an ASE-backed Calculator).

---

### `default_mass_amu`

```python
default_mass_amu(species: str) -> float
```

Look up a best-effort mass for a species symbol.

Used by `ParticleSystem.masses()`/`substrate_masses()` for any species
not given an explicit `mass_of_species` entry.

**Parameters**

- `species` (*str*): A species label -- a real element symbol (e.g. `'H'`, `'Fe'`) or a generic reduced-units placeholder (e.g. `'X'`).

**Returns**

- *float*: The real atomic mass in atomic mass units, looked up via ASE's data tables, if `species` is a recognized element symbol and ASE is installed. Otherwise 1.0 -- the reduced-units convention this package's LJ examples already assume, where mass=1 alongside sigma=epsilon=1.

## Reduced-unit pressure conventions (`regc.pressure`)

### `p0_reduced_units`

```python
p0_reduced_units(T: float) -> float
```

Reference pressure for the reduced-units (Lambda=1) convention.

**Parameters**

- `T` (*float*): Temperature, in reduced units.

**Returns**

- *float*: p0(T) = T. Consistent with the Lambda=1 convention already built into this package's GCMC moves -- use this unless you have a specific reason to calibrate p0(T) differently.

---

### `p0_ideal_gas_wavelength`

```python
p0_ideal_gas_wavelength(T: float, mass: float = 1.0, hbar: float = 1.0) -> float
```

Reference pressure for a literal 3D ideal-gas thermal wavelength.

**Parameters**

- `T` (*float*): Temperature, in reduced units.
- `mass` (*float, optional*): Particle mass, in reduced units. Default is 1.0.
- `hbar` (*float, optional*): Reduced Planck constant, in reduced units. Default is 1.0.

**Returns**

- *float*: p0(T) = T / Lambda(T)^3 with the literal 3D ideal-gas thermal de Broglie wavelength Lambda(T) = sqrt(2*pi*hbar^2/(mass*T)) (reduced units, k_B = 1). Gives p0(T) ~ T^{5/2}, in contrast to `p0_reduced_units`'s p0(T) ~ T^1.

---

### `mu_from_pressure`

```python
mu_from_pressure(T: float, p: float, p0_of_T: Callable[[float], float] = <function p0_reduced_units>) -> float
```

Convert a (temperature, pressure) pair to a chemical potential.

**Parameters**

- `T` (*float*): Temperature.
- `p` (*float*): Pressure, in whatever units `p0_of_T` is calibrated for.
- `p0_of_T` (*callable, optional*): A function `p0_of_T(T) -> float` giving the reference pressure at temperature `T`. Default is `p0_reduced_units`.

**Returns**

- *float*: mu(p, T) = k_B*T*ln(p/p0(T)), with k_B = 1 in these reduced units.

**See Also**

pressure_from_mu : the inverse conversion.

---

### `pressure_from_mu`

```python
pressure_from_mu(T: float, mu: float, p0_of_T: Callable[[float], float] = <function p0_reduced_units>) -> float
```

Convert a (temperature, chemical potential) pair to a pressure.

**Parameters**

- `T` (*float*): Temperature.
- `mu` (*float*): Chemical potential.
- `p0_of_T` (*callable, optional*): A function `p0_of_T(T) -> float` giving the reference pressure at temperature `T`. Default is `p0_reduced_units`.

**Returns**

- *float*: p = p0(T) * exp(mu/(k_B*T)), the inverse of `mu_from_pressure`.

**See Also**

mu_from_pressure : the forward conversion.

---

### `mu_grid_from_pressures`

```python
mu_grid_from_pressures(temperatures: List[float], pressures: List[float], p0_of_T: Callable[[float], float] = <function p0_reduced_units>) -> np.ndarray
```

Build a 2D chemical-potential grid for a shared pressure ladder.

Holding p fixed across T generally means mu varies with T (through
both the explicit T*ln(p/p0) and p0(T) itself), which is why
`ReplicaExchangeGCMC` accepts a 2D mu grid instead of only a flat list.

**Parameters**

- `temperatures` (*list of float*): The temperature ladder, length n_T.
- `pressures` (*list of float*): The shared pressure ladder, length n_p.
- `p0_of_T` (*callable, optional*): A function `p0_of_T(T) -> float` giving the reference pressure at temperature `T`. Default is `p0_reduced_units`.

**Returns**

- *numpy.ndarray*: The (n_T, n_p) array of chemical potentials `ReplicaExchangeGCMC` needs to run a replica ladder at this shared pressure ladder.

---

### `invert_mu_of_Tp`

```python
invert_mu_of_Tp(mu_of_Tp: Callable[[float, float], float], T: float, mu_target: float, p_bracket: Tuple[float, float] = (1e-40, 1e+100), max_bracket_expansions: int = 400, xtol: float = 1e-12) -> float
```

Numerically invert an arbitrary mu(T, p) relationship for p.

Solves `mu_of_Tp(T, p) == mu_target` for p, given only a black-box
`mu_of_Tp` callable -- for reservoirs (e.g. a diatomic gas via
`reservoir.py`) that aren't a simple closed-form `mu_from_pressure` and
so have no ready-made analytic inverse like `pressure_from_mu`.

Assumes `mu_of_Tp(T, .)` is monotonically increasing in p, which
holds for any physically sensible reservoir (more gas pressure never
lowers the chemical potential at fixed T). Solves in log(p) rather
than p directly -- p can easily need to span dozens of orders of
magnitude (see `examples/07_general_Tp_phase_diagram.py`), where
root-finding in p itself is numerically hopeless.

**Parameters**

- `mu_of_Tp` (*callable*): A function `mu_of_Tp(T, p) -> mu`, monotonically increasing in p at fixed T.
- `T` (*float*): Temperature at which to invert.
- `mu_target` (*float*): The chemical potential to solve for the corresponding pressure of.
- `p_bracket` (*tuple of float, optional*): Initial (p_lo, p_hi) bracket to search within. Expanded geometrically (in each direction independently, in log-space) until it brackets a sign change, so a generous default is fine even when the true root sits at a wildly unphysical pressure -- which is itself sometimes the point. Default is `(1e-40, 1e100)`.
- `max_bracket_expansions` (*int, optional*): Maximum number of geometric bracket expansions to attempt, in each direction. Default is 400.
- `xtol` (*float, optional*): Absolute tolerance (in log(p)) passed to `scipy.optimize.brentq`. Default is 1e-12.

**Returns**

- *float*: The pressure p such that `mu_of_Tp(T, p) == mu_target`.

**Raises**

- *RuntimeError*: If `p_bracket`, even after `max_bracket_expansions` geometric expansions in each direction, does not bracket a sign change.

## Gas reservoirs (`regc.reservoir`)

### `Reservoir`

```python
Reservoir(*args, **kwargs)
```

A gas reservoir fixing the exchanged species' chemical potential.

Subclasses implement the (T, p) <-> mu relation of a particular gas;
this base class turns it into the pieces the rest of the package needs:
a replica-ladder mu grid (`mu_grid`), the `lambda_cubed_of_T`
callable of the exchanged particle (`lambda_cubed_of_T`), and the
`mu_of_Tp`/`p_of_Tmu` pair `MBARSurfaceAnalysis` takes
(`analysis_functions`). Temperatures are in Kelvin, pressures in Pascal,
chemical potentials in eV on the calculator's energy scale.

**Attributes**

- `mass_exchanged_amu` (*float*): Mass of the particle GCMC inserts and deletes (sets Lambda).

#### `Reservoir.mu_eV`

```python
Reservoir.mu_eV(T_kelvin: float, p_pascal: float) -> float
```

Chemical potential of the exchanged species at (T, p).

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `p_pascal` (*float*): Reservoir pressure, in Pascal.

**Returns**

- *float*: mu, in eV.

**Raises**

- *NotImplementedError*: On the base class.

#### `Reservoir.pressure_pascal`

```python
Reservoir.pressure_pascal(T_kelvin: float, mu_eV: float) -> float
```

Reservoir pressure giving chemical potential `mu_eV` at T.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `mu_eV` (*float*): Chemical potential of the exchanged species, in eV.

**Returns**

- *float*: Pressure, in Pascal.

**Raises**

- *NotImplementedError*: On the base class.

#### `Reservoir.mu_grid`

```python
Reservoir.mu_grid(temperatures_K: Sequence[float], pressures: Sequence[float], unit: Optional[str] = None) -> np.ndarray
```

Chemical potentials of a (temperature x pressure) replica ladder.

**Parameters**

- `temperatures_K` (*sequence of float*): Temperatures, in Kelvin.
- `pressures` (*sequence of float*): Pressures, in `unit`.
- `unit` (*str, optional*): Pressure unit (see `regc.units.PRESSURE_UNITS`). Give it explicitly: the default None means Pa, with a FutureWarning (bar is the runner's default, so values in bar would otherwise be read as Pa, a factor 1e5).

**Returns**

- *numpy.ndarray*: Shape (len(temperatures_K), len(pressures)): mu in eV, ready for `ReplicaExchangeGCMC`'s 2D `mus` argument.

#### `Reservoir.lambda_cubed_of_T`

```python
Reservoir.lambda_cubed_of_T() -> ThermalWavelengthCubed
```

Lambda^3(T) of the exchanged particle, as a function of k_B*T in eV.

**Returns**

- *ThermalWavelengthCubed*: Pass it as `lambda_cubed_of_T` to `ReplicaExchangeGCMC` (and, through the simulation, to `MBARSurfaceAnalysis`).

#### `Reservoir.mu_at_T_eV`

```python
Reservoir.mu_at_T_eV(T_eV: float, p: float, unit: Optional[str] = None) -> float
```

`mu_eV` with the engine's temperature unit (k_B*T in eV).

**Parameters**

- `T_eV` (*float*): k_B * T, in eV.
- `p` (*float*): Pressure, in `unit`.
- `unit` (*str, optional*): Pressure unit. Default None: Pa, with a FutureWarning.

**Returns**

- *float*: mu, in eV.

#### `Reservoir.pressure_at_T_eV`

```python
Reservoir.pressure_at_T_eV(T_eV: float, mu_eV: float, unit: Optional[str] = None) -> float
```

`pressure_pascal` with the engine's temperature unit, in `unit`.

**Parameters**

- `T_eV` (*float*): k_B * T, in eV.
- `mu_eV` (*float*): Chemical potential, in eV.
- `unit` (*str, optional*): Pressure unit of the result. Default None: Pa, with a FutureWarning.

**Returns**

- *float*: Pressure, in `unit`.

#### `Reservoir.analysis_functions`

```python
Reservoir.analysis_functions(unit: Optional[str] = None) -> Tuple[Callable, Callable]
```

The `(mu_of_Tp, p_of_Tmu)` pair for `MBARSurfaceAnalysis`.

**Parameters**

- `unit` (*str, optional*): Pressure unit the analysis will be given/return. Default None: Pa, with a FutureWarning.

**Returns**

- *tuple of callable*: `mu_of_Tp(T_eV, p)` and `p_of_Tmu(T_eV, mu)` (picklable).

---

### `MonatomicIdealGasReservoir`

```python
MonatomicIdealGasReservoir(mass_amu: float, e_atom_eV: float = 0.0) -> None
```

Ideal gas of the exchanged atoms themselves ([ZSG] Sec. III A).

mu = k_B T ln(p Lambda^3 / k_B T) + e_atom, i.e. beta*mu =
beta*mu0 + ln(beta*p) with mu0 = k_B T ln(Lambda^3) (Eq. (13) of
docs/REGC_algorithm.md), shifted by the calculator's isolated-atom
energy.

**Parameters**

- `mass_amu` (*float*): Mass of the gas atoms, in atomic mass units.
- `e_atom_eV` (*float, optional*): The calculator's energy of one isolated atom (the zero of the GCMC energy scale). Default 0.0 (pair potentials).

#### `MonatomicIdealGasReservoir.mass_exchanged_amu` (property)

float: The gas atom's mass (it is also the exchanged particle).

#### `MonatomicIdealGasReservoir.mu_eV`

```python
MonatomicIdealGasReservoir.mu_eV(T_kelvin: float, p_pascal: float) -> float
```

Chemical potential of the gas atoms at (T, p).

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `p_pascal` (*float*): Pressure, in Pascal.

**Returns**

- *float*: mu, in eV.

#### `MonatomicIdealGasReservoir.pressure_pascal`

```python
MonatomicIdealGasReservoir.pressure_pascal(T_kelvin: float, mu_eV: float) -> float
```

Pressure giving chemical potential `mu_eV` at T.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `mu_eV` (*float*): Chemical potential, in eV.

**Returns**

- *float*: Pressure, in Pascal.

---

### `DiatomicReservoir`

```python
DiatomicReservoir(D0_eV: float, e_atom_eV: float, mass_molecule_amu: float, bond_length_angstrom: float, vib_wavenumber_cm: float, symmetry_number: int = 2, electronic_degeneracy: int = 1, calculator_info: Optional[dict] = None) -> None
```

A homonuclear diatomic gas (e.g. H2) exchanging single atoms.

mu_atom = (1/2) [mu_molecule(T, p) - D0] + e_atom, with mu_molecule from
real gas-phase spectroscopy (`diatomic_gas_mu_eV`) and D0, e_atom
consistent with the calculator (see this module's docstring and
`from_calculator`).

**Parameters**

- `D0_eV` (*float*): Zero-point-corrected dissociation energy of the molecule, in eV.
- `e_atom_eV` (*float*): The calculator's energy of one isolated atom, in eV.
- `mass_molecule_amu` (*float*): Molecular mass, in atomic mass units.
- `bond_length_angstrom` (*float*): Equilibrium bond length (rotational partition function), in A.
- `vib_wavenumber_cm` (*float*): Vibrational wavenumber, in cm^-1.
- `symmetry_number` (*int, optional*): Rotational symmetry number. Default 2 (homonuclear).
- `electronic_degeneracy` (*int, optional*): Degeneracy of the molecule's electronic ground state (3 for O2). Default 1; `from_calculator` picks it from the element.
- `calculator_info` (*dict, optional*): Diagnostics of the potential's own molecule (r0, De, D0, ZPE, e_atom), as filled in by `from_calculator`. Default None.

#### `DiatomicReservoir.mass_exchanged_amu` (property)

float: Mass of the exchanged atom (half the molecule).

#### `DiatomicReservoir.mu_eV`

```python
DiatomicReservoir.mu_eV(T_kelvin: float, p_pascal: float) -> float
```

Chemical potential of the atoms at (T, p) of the molecular gas.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `p_pascal` (*float*): Molecular-gas pressure, in Pascal.

**Returns**

- *float*: mu_atom, in eV.

#### `DiatomicReservoir.pressure_pascal`

```python
DiatomicReservoir.pressure_pascal(T_kelvin: float, mu_eV: float) -> float
```

Molecular-gas pressure giving atomic chemical potential `mu_eV`.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `mu_eV` (*float*): Atomic chemical potential, in eV.

**Returns**

- *float*: Pressure, in Pascal.

#### `DiatomicReservoir.hydrogen` (classmethod)

```python
DiatomicReservoir.hydrogen(D0_eV: float = 4.478, e_atom_eV: float = 0.0) -> DiatomicReservoir
```

An H2 reservoir with experimental spectroscopic constants.

**Parameters**

- `D0_eV` (*float, optional*): Dissociation energy. Default: experiment (4.478 eV).
- `e_atom_eV` (*float, optional*): The calculator's isolated-H energy. Default 0.0.

**Returns**

- *DiatomicReservoir*: The reservoir.

#### `DiatomicReservoir.from_calculator` (classmethod)

```python
DiatomicReservoir.from_calculator(ase_calculator, symbol: str = 'H', d0: Union[str, float] = 'self', bond_length_angstrom: Optional[float] = None, vib_wavenumber_cm: Optional[float] = None, mass_molecule_amu: Optional[float] = None, symmetry_number: int = 2, r_scan: Optional[np.ndarray] = None, electronic_degeneracy: Optional[int] = None) -> DiatomicReservoir
```

A reservoir consistent with a given potential.

Evaluates the potential's own molecule and isolated atom
(`diatomic_dissociation_energy_eV`): e_atom always comes from the
calculator; D0 does too unless `d0` says otherwise.

**Parameters**

- `ase_calculator` (*ase.calculators.calculator.Calculator*): The potential the simulation uses (e.g. ASE's EAM, or `regc.lammps_backend.lammps_ase_calculator`).
- `symbol` (*str, optional*): Element of the homonuclear molecule. Default `'H'`.
- `d0` (*{'self', 'experimental'} or float, optional*): `'self'`: the potential's D0 (default, consistent); `'experimental'`: 4.478 eV (H2 only), for potentials whose molecule is poor; a number: that D0 in eV.
- `bond_length_angstrom, vib_wavenumber_cm` (*float, optional*): Spectroscopic constants of the real molecule; default to H2's for `symbol='H'`, required otherwise.
- `mass_molecule_amu` (*float, optional*): Molecular mass; default twice the atomic mass from ASE's tables.
- `symmetry_number` (*int, optional*): Rotational symmetry number. Default 2.
- `r_scan` (*numpy.ndarray, optional*): Bond lengths (A) for the coarse scan; default 0.4-3.0 A.
- `electronic_degeneracy` (*int, optional*): Electronic ground-state degeneracy of the molecule. Default None: `default_electronic_degeneracy` (3 for O2, 1 otherwise).

**Returns**

- *DiatomicReservoir*: The reservoir, with the potential's diagnostics in `calculator_info` (plus `'d0_source'`).

**Raises**

- *ValueError*: If spectroscopic constants are missing for a non-H element, or `d0` is `'experimental'` for a non-H element or unknown.

---

### `diatomic_dissociation_energy_eV`

```python
diatomic_dissociation_energy_eV(ase_calculator, symbol: str, mass_amu: Optional[float] = None, r_scan: Optional[np.ndarray] = None, cell: float = 20.0) -> Dict[str, float]
```

Compute a self-consistent, zero-point-corrected dissociation energy.

Computes D0 of the homonuclear diatomic `symbol`+`symbol`, as
`ase_calculator` sees it: a coarse bond-length scan locates the well,
a local scan and iterated parabolic interpolation pin the minimum down
(independent of the scan grid), and a central finite difference gives
the harmonic force constant (hence the vibrational frequency and
zero-point energy).

**Parameters**

- `ase_calculator` (*ase.calculators.calculator.Calculator*): The ASE calculator to evaluate the diatomic and isolated-atom energies with. D0 must come from this same calculator (not an experimental table) for approximate potentials -- see this module's docstring.
- `symbol` (*str*): The element symbol of the homonuclear diatomic (e.g. `'H'` for H2).
- `mass_amu` (*float, optional*): Atomic mass, in atomic mass units. Default is None, which looks it up via ASE's data tables for `symbol`.
- `r_scan` (*numpy.ndarray, optional*): Bond lengths (in Angstrom) for the initial coarse scan. Default is None, which uses `numpy.linspace(0.4, 2.5, 60)`.
- `cell` (*float, optional*): Cubic cell edge length (in Angstrom) for the (non-periodic) evaluation cell. Default is 20.0.

**Returns**

- *dict*: Keys `'r0_angstrom'` (equilibrium bond length), `'e_min_eV'` (energy at that bond length), `'e_atom_eV'` (isolated-atom energy), `'force_constant_eV_per_A2'` (harmonic force constant), `'omega_rad_per_s'` (vibrational angular frequency), `'zpe_eV'` (zero-point energy), `'De_eV'` (well depth, 2*e_atom - e_min), and `'D0_eV'` (zero-point-corrected dissociation energy, De - zpe). Inspect these (particularly r0_angstrom and De_eV against known experimental values) before trusting the potential's gas-phase description at all.

---

### `diatomic_gas_mu_eV`

```python
diatomic_gas_mu_eV(T_kelvin: float, p_pascal: float, mass_amu: float, bond_length_angstrom: float, vib_wavenumber_cm: float, symmetry_number: int = 2, electronic_degeneracy: int = 1) -> float
```

Compute the chemical potential of a real diatomic gas at (T, p).

Chemical potential (eV) of a real, classical-rotor/harmonic-oscillator
ideal gas of homonuclear diatomic molecules at (T, p), referenced to
the molecule's own ground rotational/vibrational/electronic state
(i.e. NOT yet shifted to separated atoms -- combine with a
self-consistent D0, see `mu_atomic_from_diatomic_reservoir_eV`, to get
that).

Uses the classical (high-temperature) rotational partition function,
valid whenever T is well above the rotational temperature theta_rot
(H2: 87.6 K; O2: 2.08 K). Against NIST-JANAF at 1 bar it is within
~7 meV per H2 and ~2 meV per O2 (with `electronic_degeneracy=3`) at
300-1000 K.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `p_pascal` (*float*): Gas pressure, in Pascal.
- `mass_amu` (*float*): Molecular mass, in atomic mass units.
- `bond_length_angstrom` (*float*): Equilibrium bond length, in Angstrom.
- `vib_wavenumber_cm` (*float*): Vibrational wavenumber, in cm^-1.
- `symmetry_number` (*int, optional*): Rotational symmetry number (2 for a homonuclear diatomic). Default is 2.
- `electronic_degeneracy` (*int, optional*): Degeneracy of the electronic ground state: 1 for closed-shell molecules (H2, N2, Cl2), 3 for the triplet ground state of O2. Default is 1.

**Returns**

- *float*: The molecule's chemical potential, in eV.

---

### `mu_atomic_from_diatomic_reservoir_eV`

```python
mu_atomic_from_diatomic_reservoir_eV(T_kelvin: float, p_pascal: float, D0_eV: float, mass_molecule_amu: float, bond_length_angstrom: float, vib_wavenumber_cm: float, symmetry_number: int = 2, *, e_atom_eV: float, electronic_degeneracy: int = 1) -> float
```

Compute an atomic species' chemical potential from a diatomic reservoir.

mu_atom(T, p) = (1/2) * [mu_molecule(T, p) - D0] + e_atom, with
mu_molecule from real gas-phase spectroscopy, and D0 and e_atom
computed self-consistently with whatever Calculator will evaluate the
adsorbed atoms (see `diatomic_dissociation_energy_eV` and this module's
docstring). This is the number to feed `ReplicaExchangeGCMC` as `mu`
for an atomic species in equilibrium with a real diatomic gas
reservoir.

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `p_pascal` (*float*): Reservoir gas pressure, in Pascal.
- `D0_eV` (*float*): Self-consistent, zero-point-corrected dissociation energy, in eV (see `diatomic_dissociation_energy_eV`).
- `mass_molecule_amu` (*float*): Molecular mass, in atomic mass units.
- `bond_length_angstrom` (*float*): Equilibrium bond length, in Angstrom.
- `vib_wavenumber_cm` (*float*): Vibrational wavenumber, in cm^-1.
- `symmetry_number` (*int, optional*): Rotational symmetry number (2 for a homonuclear diatomic). Default is 2.
- `e_atom_eV` (*float*): The calculator's energy of one isolated atom (keyword-only, required): the zero of the GCMC energy scale. `'e_atom_eV'` from `diatomic_dissociation_energy_eV`; 0 for `LJCalculator`.
- `electronic_degeneracy` (*int, optional*): Degeneracy of the molecule's electronic ground state (keyword-only; 3 for O2). Default is 1.

**Returns**

- *float*: The atomic species' chemical potential, in eV, on the calculator's energy scale.

---

### `pressure_from_atomic_mu_reservoir_pascal`

```python
pressure_from_atomic_mu_reservoir_pascal(T_kelvin: float, mu_atom_eV: float, D0_eV: float, mass_molecule_amu: float, bond_length_angstrom: float, vib_wavenumber_cm: float, symmetry_number: int = 2, *, e_atom_eV: float, electronic_degeneracy: int = 1) -> float
```

Invert `mu_atomic_from_diatomic_reservoir_eV` for the reservoir pressure.

Computes the real gas pressure (Pa) whose reservoir would produce the
given atomic mu_atom at temperature T. Useful for translating a mu
range you've already explored (e.g. by direct GCMC scanning) into
"what real pressure would this correspond to".

**Parameters**

- `T_kelvin` (*float*): Temperature, in Kelvin.
- `mu_atom_eV` (*float*): The atomic species' chemical potential, in eV.
- `D0_eV` (*float*): Self-consistent, zero-point-corrected dissociation energy, in eV.
- `mass_molecule_amu` (*float*): Molecular mass, in atomic mass units.
- `bond_length_angstrom` (*float*): Equilibrium bond length, in Angstrom.
- `vib_wavenumber_cm` (*float*): Vibrational wavenumber, in cm^-1.
- `symmetry_number` (*int, optional*): Rotational symmetry number (2 for a homonuclear diatomic). Default is 2.
- `e_atom_eV` (*float*): The calculator's isolated-atom energy (keyword-only, required); see `mu_atomic_from_diatomic_reservoir_eV`.
- `electronic_degeneracy` (*int, optional*): Degeneracy of the molecule's electronic ground state (keyword-only; 3 for O2). Default is 1.

**Returns**

- *float*: The reservoir gas pressure, in Pascal.

---

### `H2_BOND_LENGTH_ANGSTROM`

```python
H2_BOND_LENGTH_ANGSTROM = 0.741
```

---

### `H2_VIBRATIONAL_WAVENUMBER_CM`

```python
H2_VIBRATIONAL_WAVENUMBER_CM = 4401.2
```

---

### `H_MASS_AMU`

```python
H_MASS_AMU = 1.008
```

---

### `H2_MASS_AMU`

```python
H2_MASS_AMU = 2.016
```

---

### `H2_D0_EXPERIMENTAL_EV`

```python
H2_D0_EXPERIMENTAL_EV = 4.478
```

## MBAR analysis (`regc.analysis`)

### `MBARSurfaceAnalysis`

```python
MBARSurfaceAnalysis(sim: ReplicaExchangeGCMC, solver_protocol: str = 'robust', p0_of_T: Callable[[float], float] = <function p0_reduced_units>, mu_of_Tp: Optional[Callable[[float, float], float]] = None, p_of_Tmu: Optional[Callable[[float, float], float]] = None, lambda_cubed_of_T: Optional[Callable[[float], float]] = None, reservoir=None, pressure_unit: Optional[str] = None, initial_f_k=None, use_pymbar: Optional[bool] = None, subsample=None)
```

Reweight recorded (U, N) samples from a finished
`ReplicaExchangeGCMC` run into a smooth coverage(T, mu) / coverage(T, p)
surface phase diagram, at (T, mu)/(T, p) values that were never
directly simulated.

**Constructor**

Fit an MBAR reweighting model to a finished simulation's samples.

The simplest way to get pressures right in real units is to pass the
gas reservoir the simulation's mu grid came from (`reservoir`, a
`regc.reservoir.Reservoir`): it sets `mu_of_Tp`/`p_of_Tmu`, with
pressures in `pressure_unit`. The lower-level options below remain
for other conventions.

`p0_of_T` sets the simple atomic-ideal-gas convention
mu = T*ln(p/p0(T)) (the default; unchanged from before). For a
reservoir that doesn't fit that closed form -- e.g. a diatomic gas
via `reservoir.py`, which adds a rotational/vibrational entropy
term and a dissociation-energy offset -- pass `mu_of_Tp(T, p) ->
mu` instead; it overrides `p0_of_T` entirely. `p_of_Tmu(T, mu) ->
p` is its inverse, used by `p_from_mu` and everything built on it
(`locate_coexistence_p`, `coexistence_curve_Tp`); if you have a
closed-form inverse (like
`pressure_from_atomic_mu_reservoir_pascal`) supplying it is exact
and cheap. If you only have `mu_of_Tp`, `p_from_mu` falls back to
numerically inverting it (`invert_mu_of_Tp`) -- robust even when
the root sits at an extreme pressure, just more expensive per call.

**Parameters**

- `sim` (*ReplicaExchangeGCMC or regc.io.SampleSet*): The finished simulation (or its saved samples, `load_samples`); must have recorded samples (`sim.samples[key]['U'/'N']`) for every grid point.
- `solver_protocol` (*str, optional*): The pymbar solver protocol to use. Default is `"robust"`.
- `p0_of_T` (*callable, optional*): A function `p0_of_T(T) -> float` giving the reference pressure at temperature T, for the default simple ideal-gas mu-pressure relationship. Default is `p0_reduced_units`. Ignored if `mu_of_Tp` is given.
- `mu_of_Tp` (*callable, optional*): A function `mu_of_Tp(T, p) -> mu` overriding `p0_of_T` entirely, for a reservoir with a non-simple mu-pressure relationship. Default is None.
- `p_of_Tmu` (*callable, optional*): A function `p_of_Tmu(T, mu) -> p`, the inverse of `mu_of_Tp`. Default is None, which falls back to numerically inverting `mu_of_Tp` (or, with neither given, the closed-form inverse of `p0_of_T`'s relationship).
- `lambda_cubed_of_T` (*callable, optional*): A function `lambda_cubed_of_T(T) -> float` giving the thermal de Broglie wavelength cubed used during sampling. It enters the reduced potential as `N * ln(Lambda^3(T))` (see `_u_of_state`). Default is None, which uses `sim.lambda_cubed_of_T` if the simulation has one, else Lambda^3 = 1 (reduced units).
- `reservoir` (*regc.reservoir.Reservoir, optional*): Sets `mu_of_Tp` and `p_of_Tmu` (unless given explicitly) from `Reservoir.analysis_functions`. Default None.
- `pressure_unit` (*str, optional*): Pressure unit for `reservoir` (`'Pa'`, `'bar'`, `'atm'`). Default None: Pa, with a FutureWarning (give it explicitly).
- `initial_f_k` (*array_like, optional*): Starting free energies for the MBAR solver (e.g. those of the full data set when analysing a bootstrap resample). Default None (a Newton estimate).
- `use_pymbar` (*bool, optional*): Finish (and check) the solve with pymbar, which also provides analytical uncertainties. False keeps the Newton solution only (it agrees with pymbar to round-off), for the many solves of a bootstrap. Default None: True for one run, False for merged runs (`regc.io.merge_samples`), whose duplicated or barely overlapping states can stall pymbar's solver for many minutes; use the bootstrap for their uncertainties.
- `subsample` (*{None, 'g'} or int, optional*): Thin every replica before the analysis: `'g'` keeps every g-th sample, g being the replica's statistical inefficiency (`regc.diagnostics.replica_inefficiencies`), so the kept samples are roughly independent and pymbar's analytical uncertainties (`mean_N`'s std error) become meaningful; an integer keeps every that-many-th sample. Thinning discards information: the estimates get noisier, so prefer the full data with `regc.diagnostics.bootstrap` error bars for maps, and use this for quick, honest single-number errors. Default None.

**Raises**

- *ValueError*: If any grid point in `sim` has no recorded samples, or several species are exchanged but per-species counts were not recorded.

#### `MBARSurfaceAnalysis.resampled`

```python
MBARSurfaceAnalysis.resampled(indices: dict) -> MBARSurfaceAnalysis
```

The same analysis on a resampled data set (for the bootstrap).

**Parameters**

- `indices` (*dict*): `{state_key: integer index array}`: the samples of each state to keep, with repetition (see `regc.diagnostics.bootstrap`).

**Returns**

- *MBARSurfaceAnalysis*: Built with the options of this one, the MBAR solve warm-started from its free energies.

#### `MBARSurfaceAnalysis.with_potential`

```python
MBARSurfaceAnalysis.with_potential(name: str, reservoir=None, pressure_unit: Optional[str] = None, mu_of_Tp: Optional[Callable] = None, p_of_Tmu: Optional[Callable] = None) -> MBARSurfaceAnalysis
```

The same analysis reweighted to another potential.

The samples were drawn with potential A (the sampling one); their
energies under B (`samples[key]['E:<name>']`, from
`regc.records.reevaluate` and `regc.records.attach_energies`)
give B's weight at any (T, mu),

    w_n^B(T, mu) ~ exp(-beta U_B(n) + beta mu N_n - N_n ln Lambda^3)
                   / sum_k N_k exp(f_k - u_k^A(n)),

the denominator being A's MBAR mixture over the sampled states
(Shirts & Chodera 2008, J. Chem. Phys. 129, 124105, Sec. on
reweighting to unsampled states). Every map of the returned
analysis (coverage, C_V, observables, effective sample size,
configurations) is then B's. The effective sample size measures
how well A's samples cover B's distribution: where it is low, B
favours configurations A rarely visited, and only a run with B
itself can say what happens there.

Samples without a B energy (re-evaluation of a subset) get zero
weight; choose the subset independently of the configurations
(every k-th sample) so the estimates stay unbiased.

Chemical potentials keep their meaning only if both potentials
place the reservoir's reference (e.g. the isolated molecule) at the
same energy: if B binds the gas molecule differently, give B's own
`reservoir` (or `mu_of_Tp`/`p_of_Tmu`) for the pressure axes.

**Parameters**

- `name` (*str*): Potential name (the samples hold `'E:<name>'`).
- `reservoir` (*regc.reservoir.Reservoir, optional*): B's gas reservoir (its pressure <-> mu mapping). Default None: keep this analysis's mapping.
- `pressure_unit` (*str, optional*): Pressure unit for `reservoir`. Default None (Pa with a warning).
- `mu_of_Tp, p_of_Tmu` (*callable, optional*): Explicit mapping for B (override `reservoir`). Default None.

**Returns**

- *MBARSurfaceAnalysis*: A copy sharing A's free energies, with U = U_B. Its `mbar` gives no analytical uncertainties (use `regc.diagnostics.bootstrap`, which resamples and reweights again); its overlap matrix is A's (the sampled states).

**Raises**

- *ValueError*: If the samples hold no `'E:<name>'` energies, or none are finite.

#### `MBARSurfaceAnalysis.grand_potential_shift`

```python
MBARSurfaceAnalysis.grand_potential_shift(T: float, mu) -> Tuple[float, float]
```

Omega_B - Omega_A at one (T, mu), by exponential averaging.

For an analysis from `with_potential`:
beta (Omega_B - Omega_A) = -ln < exp(-beta (U_B - U_A)) >_A(T, mu)
(Zwanzig's free-energy perturbation, in the grand ensemble), with
the A average MBAR-reweighted to (T, mu) over the evaluated
samples. Reliable only while the effective sample size stays large.

**Parameters**

- `T` (*float*): Temperature.
- `mu` (*float or array_like*): Chemical potential (as in this analysis).

**Returns**

- *tuple of float*: `(dOmega, ess)`: the shift in energy units, and the Kish effective sample size of B's weights there.

**Raises**

- *ValueError*: On the sampling analysis itself.

#### `MBARSurfaceAnalysis.mean_N`

```python
MBARSurfaceAnalysis.mean_N(T: float, mu) -> Tuple[float, float]
```

Compute the reweighted mean particle number at arbitrary (T, mu).

With several exchanged species, N is the total (see
`mean_N_species` for each species).

**Parameters**

- `T` (*float*): Temperature to reweight to.
- `mu` (*float or array_like*): Chemical potential to reweight to (one per species with several exchanged species).

**Returns**

- *tuple of float*: `(mean, std_error)`: the MBAR-reweighted mean of N, and its standard error, at (T, mu).

#### `MBARSurfaceAnalysis.mean_N_species`

```python
MBARSurfaceAnalysis.mean_N_species(T: float, mu) -> Tuple[np.ndarray, np.ndarray]
```

Reweighted mean count of each exchanged species at arbitrary (T, mu).

**Parameters**

- `T` (*float*): Temperature to reweight to.
- `mu` (*array_like*): One chemical potential per species (a float for one species).

**Returns**

- *tuple of numpy.ndarray*: `(means, std_errors)`, each of shape (n_species,), in the order of `exchange_species`.

#### `MBARSurfaceAnalysis.isotherm`

```python
MBARSurfaceAnalysis.isotherm(T: float, mu_grid: np.ndarray, area: float = 1.0) -> Tuple[np.ndarray, np.ndarray]
```

Compute one adsorption isotherm: coverage vs mu at fixed T.

**Parameters**

- `T` (*float*): Temperature of the isotherm.
- `mu_grid` (*numpy.ndarray*): Chemical potentials to evaluate the isotherm at (rows of one mu per species, shape (n_mu, n_species), with several).
- `area` (*float, optional*): Surface area to divide N by, giving areal coverage rather than raw particle count. Default is 1.0 (i.e. N itself).

**Returns**

- *tuple of numpy.ndarray*: `(means, sigmas)`, each of shape matching `mu_grid`: the reweighted mean coverage and its standard error at each mu.

#### `MBARSurfaceAnalysis.phase_diagram`

```python
MBARSurfaceAnalysis.phase_diagram(T_grid: np.ndarray, mu_grid: np.ndarray, area: float = 1.0, observable: str = 'N') -> np.ndarray
```

Compute a reweighted coverage(T, mu) surface on a fine mesh.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures to evaluate the surface at.
- `mu_grid` (*numpy.ndarray*): Chemical potentials to evaluate the surface at; with several species a path of mu vectors, shape (n_mu, n_species).
- `area` (*float, optional*): Surface area to divide N by. Default is 1.0.
- `observable` (*str, optional*): `'N'` (all mobile particles, the default), `'N_ads'` (adsorbed ones only; needs a run with `adsorbed_cutoff`) or `'N_<species>'`.

**Returns**

- *numpy.ndarray*: Array of shape (len(T_grid), len(mu_grid)): the reweighted mean coverage at each (T, mu) point.

#### `MBARSurfaceAnalysis.mu_from_p`

```python
MBARSurfaceAnalysis.mu_from_p(T: float, p: float) -> float
```

Convert a (temperature, pressure) pair to a chemical potential.

**Parameters**

- `T` (*float*): Temperature.
- `p` (*float*): Pressure.

**Returns**

- *float*: The chemical potential at (T, p), via `mu_of_Tp` if given at construction, otherwise `pressure.mu_from_pressure` with `p0_of_T`.

#### `MBARSurfaceAnalysis.p_from_mu`

```python
MBARSurfaceAnalysis.p_from_mu(T: float, mu: float, p_bracket: Tuple[float, float] = (1e-40, 1e+100)) -> float
```

Convert a (temperature, chemical potential) pair to a pressure.

**Parameters**

- `T` (*float*): Temperature.
- `mu` (*float*): Chemical potential.
- `p_bracket` (*tuple of float, optional*): Initial pressure bracket for numerical inversion, used only when `p_of_Tmu` is not given but `mu_of_Tp` is (see `pressure.invert_mu_of_Tp`). Default is `(1e-40, 1e100)`.

**Returns**

- *float*: The pressure at (T, mu), via `p_of_Tmu` if given at construction, else by numerically inverting `mu_of_Tp` if that was given, else `pressure.pressure_from_mu` with `p0_of_T`.

**Raises**

- *ValueError*: With several exchanged species and no `p_of_Tmu`.

#### `MBARSurfaceAnalysis.mean_N_p`

```python
MBARSurfaceAnalysis.mean_N_p(T: float, p: float) -> Tuple[float, float]
```

Compute the reweighted mean particle number at arbitrary (T, p).

**Parameters**

- `T` (*float*): Temperature to reweight to.
- `p` (*float*): Pressure to reweight to (converted to mu via `mu_from_p`).

**Returns**

- *tuple of float*: `(mean, std_error)`: the MBAR-reweighted mean of N, and its standard error, at (T, p).

#### `MBARSurfaceAnalysis.isotherm_p`

```python
MBARSurfaceAnalysis.isotherm_p(T: float, p_grid: np.ndarray, area: float = 1.0) -> Tuple[np.ndarray, np.ndarray]
```

Compute coverage vs pressure at fixed T -- the isotherm as it's
normally plotted.

`p_grid` need not include, or be limited to, pressures that were
actually part of the simulated ladder.

**Parameters**

- `T` (*float*): Temperature of the isotherm.
- `p_grid` (*numpy.ndarray*): Pressures to evaluate the isotherm at.
- `area` (*float, optional*): Surface area to divide N by. Default is 1.0.

**Returns**

- *tuple of numpy.ndarray*: `(means, sigmas)`, each of shape matching `p_grid`: the reweighted mean coverage and its standard error at each pressure.

#### `MBARSurfaceAnalysis.phase_diagram_Tp`

```python
MBARSurfaceAnalysis.phase_diagram_Tp(T_grid: np.ndarray, p_grid: np.ndarray, area: float = 1.0, observable: str = 'N') -> np.ndarray
```

Compute the reweighted coverage(T, p) surface phase diagram.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures to evaluate the surface at.
- `p_grid` (*numpy.ndarray*): Pressures to evaluate the surface at.
- `area` (*float, optional*): Surface area to divide N by. Default is 1.0.
- `observable` (*str, optional*): `'N'` (all mobile particles, the default) or `'N_ads'` (adsorbed ones only; needs a run with `adsorbed_cutoff`).

**Returns**

- *numpy.ndarray*: Array of shape (len(T_grid), len(p_grid)): the reweighted mean coverage at each (T, p) point -- the temperature-pressure surface phase diagram for the number of adsorbed particles.

#### `MBARSurfaceAnalysis.effective_sample_size`

```python
MBARSurfaceAnalysis.effective_sample_size(T_grid: np.ndarray, grid: np.ndarray, pressure: bool = False) -> np.ndarray
```

Kish effective number of samples, 1 / sum_n w_n^2, on a mesh.

It counts how many samples effectively carry the reweighted
average at a target state: about the number of (independent)
samples of a simulated state there, and ~1 where one sample
dominates (extrapolation beyond the sampled ladder, or a state
between poorly overlapping replicas). Mask cells below a few tens
before trusting a map; correlated samples make the true number
smaller still (by the statistical inefficiency).

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures.
- `grid` (*numpy.ndarray*): Pressures (`pressure=True`) or chemical potentials.
- `pressure` (*bool, optional*): Whether `grid` holds pressures. Default False.

**Returns**

- *numpy.ndarray*: Shape (len(T_grid), len(grid)).

#### `MBARSurfaceAnalysis.observable_map`

```python
MBARSurfaceAnalysis.observable_map(name: str, T_grid: np.ndarray, grid: np.ndarray, pressure: bool = False) -> np.ndarray
```

Reweighted mean of every component of a recorded observable on a mesh.

**Parameters**

- `name` (*str*): Observable name (`observables=` of the run).
- `T_grid` (*numpy.ndarray*): Temperatures.
- `grid` (*numpy.ndarray*): Pressures (`pressure=True`) or chemical potentials.
- `pressure` (*bool, optional*): Whether `grid` holds pressures. Default False.

**Returns**

- *numpy.ndarray*: Shape (len(T_grid), len(grid), size).

#### `MBARSurfaceAnalysis.observable_mean`

```python
MBARSurfaceAnalysis.observable_mean(name: str, T: float, mu) -> np.ndarray
```

Reweighted mean of every component of a recorded observable at one (T, mu).

**Parameters**

- `name` (*str*): Observable name.
- `T` (*float*): Temperature.
- `mu` (*float or array_like*): Chemical potential (one per species with several).

**Returns**

- *numpy.ndarray*: Shape (size,).

#### `MBARSurfaceAnalysis.mean_coordination`

```python
MBARSurfaceAnalysis.mean_coordination(name: str, T_grid: np.ndarray, grid: np.ndarray, pressure: bool = False) -> np.ndarray
```

Reweighted mean coordination number, <bonds> / <centers>, on a mesh.

**Parameters**

- `name` (*str*): A `regc.observables.CoordinationNumber` observable.
- `T_grid` (*numpy.ndarray*): Temperatures.
- `grid` (*numpy.ndarray*): Pressures (`pressure=True`) or chemical potentials.
- `pressure` (*bool, optional*): Whether `grid` holds pressures. Default False.

**Returns**

- *numpy.ndarray*: Shape (len(T_grid), len(grid)); NaN where there are (almost) no centre atoms (<centers> < 1e-6).

#### `MBARSurfaceAnalysis.coordination_distribution`

```python
MBARSurfaceAnalysis.coordination_distribution(name: str, T: float, mu) -> np.ndarray
```

Probability that a centre atom has coordination 0, 1, ..., max_cn.

**Parameters**

- `name` (*str*): A `regc.observables.CoordinationHistogram` observable.
- `T` (*float*): Temperature.
- `mu` (*float or array_like*): Chemical potential.

**Returns**

- *numpy.ndarray*: Shape (max_cn + 1,), summing to 1 (all zeros without centres); the last entry is `P(CN >= max_cn)`.

#### `MBARSurfaceAnalysis.radial_distribution`

```python
MBARSurfaceAnalysis.radial_distribution(name: str, T: float, mu, normalize: str = 'per_center') -> Tuple[np.ndarray, np.ndarray]
```

Reweighted radial distribution at one (T, mu).

**Parameters**

- `name` (*str*): A `regc.observables.RadialDistribution` observable.
- `T` (*float*): Temperature.
- `mu` (*float or array_like*): Chemical potential.
- `normalize` (*{'per_center', 'density', 'g'}, optional*): `'per_center'`: n(r) = <H(r)> / (<n_a> dr), neighbours of a centre per unit distance (its integral to r_c is the mean coordination; for surfaces and clusters). `'density'`: n(r) / (4 pi r^2), local density of b around a (1/A^3). `'g'`: density / (<n_b> / V), the bulk g(r) (V includes any vacuum, so only meaningful in bulk). Default `'per_center'`.

**Returns**

- *tuple of numpy.ndarray*: `(r, values)`: bin centres (A) and the distribution (zeros if there are no a atoms).

**Raises**

- *ValueError*: On an unknown `normalize`.

#### `MBARSurfaceAnalysis.heat_capacity`

```python
MBARSurfaceAnalysis.heat_capacity(T_grid: np.ndarray, grid: np.ndarray, pressure: bool = False, kinetic: bool = True) -> np.ndarray
```

Heat capacity C_V = (<E^2> - <E>^2) / (k_B T^2) on a mesh, in units of k_B.

[ZSG] PRL 128, 246101, Eq. (1): E is the total energy, potential
plus kinetic, and the averages are MBAR-reweighted to every (T, mu)
or (T, p). Ridges of C_V along T at fixed p, or along p at fixed T,
mark first-order phase boundaries (`heat_capacity_ridges`); a
change of coverage without a ridge is a smooth crossover.

The kinetic energy of the n moving atoms (mobile + displaceable
substrate) is classical and independent of the configuration given
n: <K | n> = (3/2) n k_B T and Var(K | n) = (3/2) n (k_B T)^2. So,
with x = U + (3/2) k_B T n,

    Var(E) = Var(x) + (3/2) (k_B T)^2 <n>,

computed from the recorded U and N alone (MC diffusion never
samples momenta, and it does not need to).

In the grand ensemble (momenta included, Lambda^3 proportional to
T^(-3/2)) this fluctuation formula is exactly d<E>/dT at fixed
beta*mu -- not at fixed mu, p or N; its ridges still locate the
transitions, which is what it is for. It is a variance, so it needs
more samples than <N> to converge: check `effective_sample_size`
and compare halves of the run.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures (k_B T, in the energy unit of U).
- `grid` (*numpy.ndarray*): Pressures (`pressure=True`) or chemical potentials.
- `pressure` (*bool, optional*): Whether `grid` holds pressures. Default False.
- `kinetic` (*bool, optional*): Include the kinetic energy (as [ZSG]). Default True; False gives the configurational part Var(U) / (k_B T)^2.

**Returns**

- *numpy.ndarray*: C_V / k_B, shape (len(T_grid), len(grid)).

#### `MBARSurfaceAnalysis.sample_location`

```python
MBARSurfaceAnalysis.sample_location(index: int) -> Tuple[Tuple[int, int], int]
```

Grid point and run sample index of a sample (in the analysis order).

**Parameters**

- `index` (*int*): Index into `U_all`.

**Returns**

- *tuple*: `(key, sample)`: the grid point it was recorded at and its index there in the run's numbering (as written with the stored configurations, `regc.trajectory.load_configurations`).

#### `MBARSurfaceAnalysis.run_indices`

```python
MBARSurfaceAnalysis.run_indices(k: int) -> np.ndarray
```

Run sample indices of the samples of state `k` (in the analysis order).

Accounts for discarded equilibration (`first_index`) and for
thinning (`subsample`).

**Parameters**

- `k` (*int*): Index into `states`.

**Returns**

- *numpy.ndarray*: One run sample index per sample of that state.

#### `MBARSurfaceAnalysis.representative_samples`

```python
MBARSurfaceAnalysis.representative_samples(T: np.ndarray, mu: np.ndarray, target_N: np.ndarray, observable: str = 'N', candidates: Optional[np.ndarray] = None) -> np.ndarray
```

The most probable recorded sample with a given N at each target state.

At each (T, mu) the sample of largest MBAR weight among those whose
`observable` equals the target value (or, if none has it, the
closest value): a typical configuration of the phase shown there.

**Parameters**

- `T, mu` (*numpy.ndarray*): Target states, common shape S (`mu` with a species axis with several species).
- `target_N` (*numpy.ndarray*): Target values of `observable`, shape S (e.g. the most probable N).
- `observable` (*str, optional*): `'N'`, `'N_ads'`, ... Default `'N'`.
- `candidates` (*numpy.ndarray, optional*): Boolean mask over all samples allowed to be picked (e.g. those whose configuration was stored). Default None (all).

**Returns**

- *numpy.ndarray*: Integer indices into `U_all`, shape S (see `sample_location`).

**Raises**

- *ValueError*: If `candidates` allows no sample.

#### `MBARSurfaceAnalysis.most_probable_N`

```python
MBARSurfaceAnalysis.most_probable_N(T_grid: np.ndarray, mu_grid: np.ndarray, observable: str = 'N') -> np.ndarray
```

The most stable phase, labelled by N, on a (T, mu) mesh.

[ZSG] Sec. II B: with phases defined by the number of particles,
f_N = -kT ln P(N), so the stable phase is argmax_N P(N).

**Parameters**

- `T_grid, mu_grid` (*numpy.ndarray*): The mesh axes (`mu_grid` a path of mu vectors, shape (n_mu, n_species), with several species).
- `observable` (*str, optional*): `'N'` (all mobile particles, the default), `'N_ads'` (adsorbed ones only; needs a run with `adsorbed_cutoff`) or `'N_<species>'`.

**Returns**

- *numpy.ndarray*: Integer array of shape (len(T_grid), len(mu_grid)).

#### `MBARSurfaceAnalysis.most_probable_composition`

```python
MBARSurfaceAnalysis.most_probable_composition(T_grid: np.ndarray, mu_grid: np.ndarray) -> np.ndarray
```

The most stable phase, labelled by its composition (N_1, ..., N_S).

`most_probable_N` for several exchanged species: the argmax of the
reweighted joint P(N_1, ..., N_S), e.g. telling an O-rich from a
Cl-rich phase of the same total coverage.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures, length n_T.
- `mu_grid` (*numpy.ndarray*): Path of mu vectors, shape (n_mu, n_species) (length n_mu for one species).

**Returns**

- *numpy.ndarray*: Integer array of shape (n_T, n_mu, n_species).

#### `MBARSurfaceAnalysis.most_probable_N_Tp`

```python
MBARSurfaceAnalysis.most_probable_N_Tp(T_grid: np.ndarray, p_grid: np.ndarray, observable: str = 'N') -> np.ndarray
```

`most_probable_N` on a (T, p) mesh ([ZSG] Fig. 3).

Each (T, p) is converted with `mu_from_p` (so `mu_of_Tp`/`p0_of_T`
given at construction set the reservoir).

**Parameters**

- `T_grid, p_grid` (*numpy.ndarray*): The mesh axes (p in the units `mu_from_p` expects).
- `observable` (*str, optional*): `'N'` (all mobile particles, the default) or `'N_ads'` (adsorbed ones only; needs a run with `adsorbed_cutoff`).

**Returns**

- *numpy.ndarray*: Integer array of shape (len(T_grid), len(p_grid)).

#### `MBARSurfaceAnalysis.lowest_energy_per_N`

```python
MBARSurfaceAnalysis.lowest_energy_per_N() -> Tuple[np.ndarray, np.ndarray]
```

Lowest sampled energy for every observed particle number.

**Returns**

- *tuple of numpy.ndarray*: `(N_values, E_min)`: the distinct N in the samples and the lowest potential energy sampled with that N. With several species, `N_values` holds the distinct compositions, shape (n_compositions, n_species).

#### `MBARSurfaceAnalysis.lowest_energy_N`

```python
MBARSurfaceAnalysis.lowest_energy_N(mu_values: Sequence[float]) -> np.ndarray
```

Phase diagram from lowest energies only ([ZSG] "aiAT@REGC", Eq. 13).

For each mu, the N minimising E_min(N) - mu*N, with E_min(N) the
lowest sampled energy at that N (`lowest_energy_per_N`): the
ab initio atomistic-thermodynamics construction with all vibrational
(and other entropic) contributions neglected. Compare with
`most_probable_N` to see their effect. Independent of T at fixed mu.

**Parameters**

- `mu_values` (*sequence of float*): Chemical potentials (mu vectors, shape (n, n_species), with several species).

**Returns**

- *numpy.ndarray*: Integer array of the stable N, one per mu (with several species, the stable composition, shape (n, n_species)).

#### `MBARSurfaceAnalysis.lowest_energy_N_Tp`

```python
MBARSurfaceAnalysis.lowest_energy_N_Tp(T_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray
```

`lowest_energy_N` on a (T, p) mesh.

**Parameters**

- `T_grid, p_grid` (*numpy.ndarray*): The mesh axes (p in the units `mu_from_p` expects).

**Returns**

- *numpy.ndarray*: Integer array of shape (len(T_grid), len(p_grid)) (plus a trailing species axis with several species).

#### `MBARSurfaceAnalysis.pN`

```python
MBARSurfaceAnalysis.pN(T: float, mu, n_bins: Optional[int] = None, observable: str = 'N') -> Tuple[np.ndarray, np.ndarray]
```

Compute the reweighted probability mass function of N at (T, mu).

A bimodal P(N) is the signature of two-phase coexistence (dilute
film vs. condensed islands/layer) near a 2D condensation
transition.

**Parameters**

- `T` (*float*): Temperature to reweight to.
- `mu` (*float or array_like*): Chemical potential to reweight to (one per species with several).
- `n_bins` (*int, optional*): Number of histogram bins. Default is None, which uses one bin per integer N from 0 to the observed maximum.
- `observable` (*str, optional*): Count to histogram: `'N'` (default), `'N_ads'` or `'N_<species>'` (see `phase_diagram`).

**Returns**

- *tuple of numpy.ndarray*: `(centers, probs)`: the bin centers (in N) and the reweighted probability mass in each bin.

#### `MBARSurfaceAnalysis.pN_p`

```python
MBARSurfaceAnalysis.pN_p(T: float, p: float, n_bins: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]
```

Compute the reweighted P(N) at (T, p).

**Parameters**

- `T` (*float*): Temperature to reweight to.
- `p` (*float*): Pressure to reweight to (converted to mu via `mu_from_p`).
- `n_bins` (*int, optional*): Number of histogram bins. Default is None (see `pN`).

**Returns**

- *tuple of numpy.ndarray*: `(centers, probs)` -- see `pN`.

#### `MBARSurfaceAnalysis.locate_coexistence_mu`

```python
MBARSurfaceAnalysis.locate_coexistence_mu(T: float, mu_bracket: Tuple[float, float], n_iter: int = 25, n_bins: Optional[int] = None) -> float
```

Locate the approximate 2D-condensation coexistence mu at fixed T.

Via the equal-weight rule: bisect mu until reweighted P(N) has
equal integrated probability in its low-N ('dilute') and high-N
('condensed') basins. The basins are split at one fixed N, midway
between the valleys of the bimodal P(N) at the two bracket ends, so
the bisection stays well defined where P(N) is unimodal.

Caveats (this is a diagnostic tool, not a black box): it assumes
P(N) genuinely becomes bimodal somewhere in `mu_bracket` -- i.e.
the potential/conditions you chose actually support 2D
condensation -- and that the replica ladder has enough mu-points
near the transition for MBAR to have good sample overlap there. If
P(N) never shows two peaks across the bracket, this raises rather
than returning a misleading number.

**Parameters**

- `T` (*float*): Temperature at which to locate coexistence.
- `mu_bracket` (*tuple of float*): (mu_lo, mu_hi) bracket to bisect within; must contain the transition, with a bimodal P(N) detected at both ends.
- `n_iter` (*int, optional*): Number of bisection iterations. Default is 25.
- `n_bins` (*int, optional*): Number of histogram bins for `pN`. Default is None.

**Returns**

- *float*: The coexistence chemical potential.

**Raises**

- *RuntimeError*: If a bimodal P(N) crossing can't be bracketed within `mu_bracket`.
- *ValueError*: With several exchanged species (no single mu to bisect; scan `pN` along a mu path instead).

#### `MBARSurfaceAnalysis.locate_coexistence_p`

```python
MBARSurfaceAnalysis.locate_coexistence_p(T: float, p_bracket: Tuple[float, float], n_iter: int = 25, n_bins: Optional[int] = None) -> float
```

Locate the 2D-condensation coexistence pressure at fixed T.

`locate_coexistence_mu`, expressed in pressure: one point on the
2D condensation line in the (T, p) phase diagram.

**Parameters**

- `T` (*float*): Temperature at which to locate coexistence.
- `p_bracket` (*tuple of float*): (p_lo, p_hi) bracket to search within (converted to a mu bracket via `mu_from_p`).
- `n_iter` (*int, optional*): Number of bisection iterations. Default is 25.
- `n_bins` (*int, optional*): Number of histogram bins for `pN`. Default is None.

**Returns**

- *float*: The coexistence pressure.

**Raises**

- *RuntimeError*: If a bimodal P(N) crossing can't be bracketed within `p_bracket` (via `locate_coexistence_mu`).

#### `MBARSurfaceAnalysis.coexistence_curve_Tp`

```python
MBARSurfaceAnalysis.coexistence_curve_Tp(T_grid: np.ndarray, p_bracket, n_iter: int = 25, n_bins: Optional[int] = None, verbose: bool = True) -> List[Tuple[float, float]]
```

Trace the 2D condensation line p_coex(T) across a temperature grid.

The phase boundary on the (T, p) diagram.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures to trace the coexistence curve at.
- `p_bracket` (*tuple of float or callable*): Either a fixed (p_lo, p_hi) tuple used at every T, or a callable `T -> (p_lo, p_hi)` for a bracket that needs to move with T.
- `n_iter` (*int, optional*): Number of bisection iterations per T. Default is 25.
- `n_bins` (*int, optional*): Number of histogram bins for `pN`. Default is None.
- `verbose` (*bool, optional*): If True, print a message for any T where no bimodal crossing is found. Default is True.

**Returns**

- *list of tuple of float*: `(T, p_coex)` pairs, one per T in `T_grid` where a coexistence pressure was found. Skips (and reports, if `verbose`) any T where no bimodal crossing is found, rather than raising.

---

### `heat_capacity_ridges`

```python
heat_capacity_ridges(T_grid: np.ndarray, grid: np.ndarray, cv: np.ndarray, valid: Optional[np.ndarray] = None, prominence: float = 0.2, ladder: Optional[Sequence[float]] = None, ladder_tol: Optional[float] = None) -> dict
```

Local maxima of C_V along each axis: candidate phase-boundary points.

[ZSG] PRL 128, 246101: a phase boundary connects the (T, p) points
where C_V(p) at fixed T, or C_V(T) at fixed p, has a peak; a ridge
that fades out ends at a critical point.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures, length n_T.
- `grid` (*numpy.ndarray*): Pressures or chemical potentials, length n_y.
- `cv` (*numpy.ndarray*): Heat capacity, shape (n_T, n_y) (`MBARSurfaceAnalysis.heat_capacity`).
- `valid` (*numpy.ndarray, optional*): Boolean mask of trustworthy cells (e.g. effective sample size above a threshold); peaks in other cells are dropped. Default None (all).
- `prominence` (*float, optional*): Minimum relative height of a peak over the deeper of its two flanking minima: a peak counts if C_V there exceeds that minimum by this fraction (found on ln C_V, so it is the same for a small bump on a large background and for one near C_V = 0). Default 0.2 (20 %); raise it for noisy maps from short runs.
- `ladder` (*sequence of float, optional*): The sampled values along `grid` (the replica ladder's p or mu). C_V is a variance: between replicas whose samples barely overlap, MBAR rests on few effective samples and underestimates it, so spurious peaks appear *on* the sampled values. With `ladder` given, each point is flagged (`'on_ladder'`) if it lies within `ladder_tol` of a sampled value, and a warning is issued when most do. Real boundaries move with T; artefacts stay on the ladder. For a pressure grid, give the ladder's pressures. Default None.
- `ladder_tol` (*float, optional*): Distance counting as "on" the ladder. Default: a quarter of the smallest ladder spacing (in log10 for a positive ladder spanning more than a decade, i.e. pressures).

**Returns**

- *dict*: `'along_y'`: (i_T, j_y) index pairs of peaks of C_V(y) at fixed T; `'along_T'`: the same for C_V(T) at fixed y; `'points'`: the union as `(T, y)` values, shape (n, 2), and `'cells'` the same as index pairs; `'along_y_bases'`, `'along_T_bases'`: the indices of the two flanking minima of each peak along its axis (for `regc.diagnostics.ridge_support`); with `ladder`, also `'on_ladder'`.

---

### `export_phase_configurations`

```python
export_phase_configurations(analysis: MBARSurfaceAnalysis, configurations_dir: str, T_grid: np.ndarray, grid: np.ndarray, path: str, pressure: bool = False, mode: str = 'phases', observable: str = 'N', min_ess: float = 0.0, temperature_K: bool = False, area: Optional[float] = None, tile_path: Optional[str] = None, min_cells: int = 1) -> List[dict]
```

Write the configurations visible in a phase diagram as extended XYZ.

The run must have stored configurations (`configurations_dir` of
`ReplicaExchangeGCMC`, `configurations: save` in a config file). On
the (T, y) mesh, the most probable value of `observable` labels the
phase of every cell. Then:

* `mode='phases'`: one frame per connected phase region, taken at
  the cell of the region where that phase is most probable;
* `mode='grid'`: one frame per mesh cell (use a coarse mesh).

Each frame is the recorded sample with the largest MBAR weight at that
cell among those with the phase's value of `observable` (and a stored
configuration): a typical structure of the phase, as pySNOW shows when
a point of the diagram is picked. The comment line carries the cell's
T, p or mu, the phase label, the probability of that label there, the
sample's weight, its origin (grid point, sample index, energy) and the
effective sample size; per-atom `group` is 0 frozen, 1 substrate,
2 mobile.

**Parameters**

- `analysis` (*MBARSurfaceAnalysis*): The analysis of the run.
- `configurations_dir` (*str or sequence of str*): Where the run stored `configs_T{it}_mu{imu}.extxyz`; for merged runs (`regc.io.merge_samples`), one directory per run, in the order the runs were merged.
- `T_grid` (*numpy.ndarray*): Temperatures (k_B T, as given to the analysis).
- `grid` (*numpy.ndarray*): Pressures (`pressure=True`) or chemical potentials.
- `path` (*str*): Output `.extxyz` file (multi-frame).
- `pressure` (*bool, optional*): Whether `grid` holds pressures. Default False.
- `mode` (*{'phases', 'grid'}, optional*): See above. Default `'phases'`.
- `observable` (*str, optional*): Phase label: `'N'` (default), `'N_ads'`, `'N_<species>'`.
- `min_ess` (*float, optional*): Ignore cells whose effective sample size is below this (their phase is extrapolated). Default 0.0.
- `temperature_K` (*bool, optional*): Also write `T_K` (temperatures given as k_B T in eV). Default False.
- `area` (*float, optional*): If given, also write `coverage` = label / area. Default None.
- `tile_path` (*str, optional*): If given, also write all exported frames side by side in one frame (rows: T, columns: p or mu), a visual phase diagram. Default None.
- `min_cells` (*int, optional*): `mode='phases'`: skip regions smaller than this many mesh cells (where the label changes almost continuously, as in a fluid, every cell is its own region). Default 1.

**Returns**

- *list of dict*: One entry per exported frame: its comment-line values.

**Raises**

- *FileNotFoundError*: If no configurations were stored.
- *ValueError*: On an unknown `mode`, or if no cell passes `min_ess`.

## Plotting (`regc.plotting`)

### `plot_phase_diagram`

```python
plot_phase_diagram(T_grid: np.ndarray, p_grid: np.ndarray, theta: np.ndarray, coexistence: Optional[List[Tuple[float, float]]] = None, save_path: str = 'phase_diagram.png', colorbar_label: str = 'coverage (particles / area)', xlabel: str = 'pressure p', ylabel: str = 'temperature T', title: str = 'Reweighted surface coverage $\\theta(T, p)$') -> None
```

Plot and save a coverage(T, p) heatmap.

With the (optional) coexistence line overlaid. Requires matplotlib;
does nothing but print a note if it isn't installed.

Plots against log10(p) directly, with tick labels reformatted as
10^x, rather than plotting p on a matplotlib log-scale axis:
pcolormesh infers cell edges by linearly averaging adjacent coordinate
values regardless of the axis's display scale, and for log-spaced p
that linear average is completely dominated by the larger neighbor
whenever p spans more than a couple of decades -- visually collapsing
every lower-pressure column into an imperceptible sliver even though
the underlying data is fine. Working in log10(p) directly sidesteps
this: the coordinates are then genuinely evenly spaced, so the (still
linear) edge-averaging is exactly right.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperature grid, length n_T.
- `p_grid` (*numpy.ndarray*): Pressure grid, length n_p.
- `theta` (*numpy.ndarray*): Coverage array, shape (n_T, n_p) (e.g. from `MBARSurfaceAnalysis.phase_diagram_Tp`).
- `coexistence` (*list of tuple of float, optional*): `(T, p)` coexistence-line points to overlay (e.g. from `MBARSurfaceAnalysis.coexistence_curve_Tp`). Default is None (no overlay).
- `save_path` (*str, optional*): Path to save the figure to. Default is `"phase_diagram.png"`.
- `colorbar_label` (*str, optional*): Colorbar label. Default `"coverage (particles / area)"`.
- `xlabel, ylabel` (*str, optional*): Axis labels. Defaults `"pressure p"` and `"temperature T"`.
- `title` (*str, optional*): Plot title. Default: reweighted surface coverage theta(T, p).

**Returns**

- *None*: Saves the figure to `save_path` and returns nothing.

---

### `plot_map`

```python
plot_map(T_grid: np.ndarray, grid: np.ndarray, values: np.ndarray, save_path: str, log_x: bool = True, valid: Optional[np.ndarray] = None, points: Optional[np.ndarray] = None, discrete: bool = False, log_color: bool = False, colorbar_label: str = '', xlabel: str = 'pressure p', ylabel: str = 'temperature T', title: str = '', cmap: str = 'viridis', invalid_label: str = 'too few effective samples', vmin: Optional[float] = None, vmax: Optional[float] = None, contour_levels: Optional[Sequence[float]] = None, contour_label: str = '', points_label: str = '$C_V$ peak', figsize: Tuple[float, float] = (6.0, 4.0), dpi: int = 300) -> None
```

Plot a (T, p) or (T, mu) map, with unreliable cells shown as gaps.

Cells where `valid` is False are drawn grey and hatched (not
interpolated or hidden, and distinct from any colour of the map), and
named in a legend placed outside the axes so it never covers data.
Integer label maps (`discrete=True`) get one colour per value up to
20 values; beyond that the colours are continuous but the boundaries
between values are outlined. On a logarithmic colour scale, values
at or below zero are not hidden: they take the colour below the scale
(black) and get their own legend entry.

**Parameters**

- `T_grid` (*numpy.ndarray*): Temperatures (y axis), length n_T.
- `grid` (*numpy.ndarray*): Pressures or chemical potentials (x axis), length n_y.
- `values` (*numpy.ndarray*): Shape (n_T, n_y).
- `save_path` (*str*): Output image.
- `log_x` (*bool, optional*): Plot log10 of `grid` (pressures). Default True.
- `valid` (*numpy.ndarray, optional*): Boolean mask of trustworthy cells (e.g. effective sample size above a threshold). Default None (all valid).
- `points` (*numpy.ndarray, optional*): `(T, x)` points to overlay (e.g. heat-capacity ridges, x in the units of `grid`). Default None.
- `discrete` (*bool, optional*): Integer labels (e.g. most probable N). Default False.
- `log_color` (*bool, optional*): Logarithmic colour scale (for quantities spanning decades, like C_V). Default False.
- `colorbar_label, xlabel, ylabel, title` (*str, optional*): Labels.
- `cmap` (*str, optional*): Matplotlib colormap (perceptually uniform by default). Default `'viridis'`.
- `invalid_label` (*str, optional*): Legend text for the hatched cells.
- `vmin, vmax` (*float, optional*): Colour range. Default None: the data range (for `log_color`, from the 2nd percentile of the positive values; the colorbar arrow marks anything below).
- `contour_levels` (*sequence of float, optional*): Draw these contour levels of `values` in white (e.g. an effective sample size threshold). Default None.
- `contour_label` (*str, optional*): Legend text for the contours.
- `points_label` (*str, optional*): Legend text for `points`. Default `'$C_V$ peak'`.
- `figsize` (*tuple of float, optional*): Figure size in inches. Default (6.0, 4.0).
- `dpi` (*int, optional*): Resolution of the saved image. Default 300.

**Returns**

- *None*: Saves the figure.

---

### `plot_radial_distribution`

```python
plot_radial_distribution(curves: dict, save_path: str, ylabel: str = '$n(r)$ (Å$^{-1}$)', xlabel: str = '$r$ (Å)', title: str = '', xlim: Optional[Tuple[float, float]] = None, figsize: Tuple[float, float] = (5.0, 3.2), dpi: int = 300) -> None
```

Plot reweighted radial distributions at several states.

Each curve gets a colour-blind-safe colour and its own line style, so
the curves stay distinguishable in greyscale.

**Parameters**

- `curves` (*dict*): `{label: (r, values)}`, e.g. from `MBARSurfaceAnalysis.radial_distribution`.
- `save_path` (*str*): Output image.
- `ylabel, xlabel, title` (*str, optional*): Labels.
- `xlim` (*tuple of float, optional*): x range. Default None: from the first nonzero value to the end.
- `figsize` (*tuple of float, optional*): Figure size in inches. Default (5.0, 3.2).
- `dpi` (*int, optional*): Resolution of the saved image. Default 300.

**Returns**

- *None*: Saves the figure.

## Per-sample records and other potentials (`regc.records`)

### `RecordWriter`

```python
RecordWriter(path: str, sampling_potential: str, species: Sequence[str], lattice: np.ndarray, pbc: Sequence[bool], n_frozen: int, n_substrate: int, temperatures, mu_grid, n_species: int = 1, positions_dtype: str = 'float64')
```

Append per-sample records to an HDF5 file.

Records are queued by `add` and written by `flush` (called by the
engine at checkpoints and at the end of every run call, and here
every `FLUSH_EVERY` records, so memory stays bounded).

**Parameters**

- `path` (*str*): The file (created if missing, appended to otherwise).
- `sampling_potential` (*str*): Name under which the sampling energies are stored (`E/<name>`).
- `species` (*sequence of str*): All atom labels that can occur (frozen, substrate and exchanged).
- `lattice` (*numpy.ndarray*): Cell matrix, shape (3, 3).
- `pbc` (*sequence of bool*): Periodicity.
- `n_frozen, n_substrate` (*int*): Sizes of the first two atom groups (constant during a run).
- `temperatures, mu_grid` (*array_like*): The run's ladder (stored for reference).
- `n_species` (*int, optional*): Exchanged species (`/samples/N_s` with more than one). Default 1.
- `positions_dtype` (*str, optional*): `'float64'` (default, exact re-evaluation) or `'float32'` (half the size; energies change by ~1e-6 relative).

**Constructor**

Open or create the file.

**Parameters**

- `path` (*str*): See the class docstring (likewise below).
- `sampling_potential` (*str*): Energy column of the sampling potential.
- `species` (*sequence of str*): Atom labels.
- `lattice` (*numpy.ndarray*): Cell matrix.
- `pbc` (*sequence of bool*): Periodicity.
- `n_frozen, n_substrate` (*int*): Group sizes.
- `temperatures, mu_grid` (*array_like*): The ladder.
- `n_species` (*int, optional*): Exchanged species.
- `positions_dtype` (*str, optional*): Position precision.

**Raises**

- *ValueError*: If an existing file was written for another sampling potential or species set.

#### `RecordWriter.add`

```python
RecordWriter.add(key: Tuple[int, int], sample: int, walker: int, energy: float, n: int, config: Optional[tuple], n_s: Optional[Sequence[int]] = None) -> None
```

Queue one record (written by `flush`).

**Parameters**

- `key` (*tuple of int*): Grid point.
- `sample` (*int*): Sample index at that grid point.
- `walker` (*int*): Walker id (-1 if unknown).
- `energy` (*float*): Energy under the sampling potential.
- `n` (*int*): Mobile atoms.
- `config` (*tuple or None*): `regc.trajectory.snapshot` of the configuration, or None (not stored for this sample).
- `n_s` (*sequence of int, optional*): Counts per exchanged species.

#### `RecordWriter.flush`

```python
RecordWriter.flush() -> None
```

Write the queued records and flush the file to disk.

#### `RecordWriter.close`

```python
RecordWriter.close() -> None
```

Flush and close the file.

---

### `Records`

```python
Records(path: str, writable: bool = False)
```

Read access to a records file (read-only unless `writable`).

**Parameters**

- `path` (*str*): The file.
- `writable` (*bool, optional*): Open for adding energy columns (`reevaluate`). Default False.

**Constructor**

Open the file and index its records.

**Parameters**

- `path, writable`: See the class docstring.

#### `Records.potentials` (property)

list of str: Names of the stored energy columns.

#### `Records.energies`

```python
Records.energies(name: str) -> Dict[Tuple[Tuple[int, int], int], float]
```

Energies under one potential.

**Parameters**

- `name` (*str*): Potential name (`E/<name>`).

**Returns**

- *dict*: `{((it, imu), sample): energy}` (nan where not evaluated).

#### `Records.configuration`

```python
Records.configuration(key: Tuple[int, int], sample: int) -> Optional[dict]
```

One stored configuration.

**Parameters**

- `key` (*tuple of int*): Grid point.
- `sample` (*int*): Sample index there.

**Returns**

- *dict or None*: Frame-like dict (`positions`, `species`, `lattice`, `pbc`, `info`, `arrays` with `group`), as from `regc.trajectory.read_extxyz_frames`; None if not stored.

#### `Records.stored`

```python
Records.stored(key: Tuple[int, int]) -> set
```

Sample indices of one grid point that have a stored configuration.

**Parameters**

- `key` (*tuple of int*): Grid point.

**Returns**

- *set of int*: The sample indices.

#### `Records.close`

```python
Records.close() -> None
```

Close the file.

---

### `attach_energies`

```python
attach_energies(samples, records_path: str, names: Optional[Sequence[str]] = None)
```

Add energies under other potentials to a sample set, sample by sample.

**Parameters**

- `samples` (*SampleSet*): Loaded with `regc.io.load_samples` (any `discard`).
- `records_path` (*str*): The run's records file.
- `names` (*sequence of str, optional*): Potentials to attach. Default None: every one except the sampling potential.

**Returns**

- *SampleSet*: The same set, with `samples[key]['E:<name>']` lists (nan where a sample has not been evaluated).

**Raises**

- *ValueError*: If a sample of the set is missing from the records.

---

### `reevaluate`

```python
reevaluate(records_path: str, make_system: Callable, name: str, subset: Optional[Callable[[Tuple[int, int], int], bool]] = None, n_workers: int = 1, chunk: int = 200, progress: Optional[Callable[[int, int], None]] = None) -> int
```

Evaluate the stored configurations with another potential.

Every stored configuration not yet evaluated under `name` is rebuilt in
a system from `make_system` (same geometry, the new calculator) and its
`total_energy` written to `E/<name>`. The energies are flushed to disk
after every `chunk` configurations, so an interrupted call loses at
most one chunk and rerunning continues where it stopped. Use a
`subset` that does not depend on the configurations (e.g. every k-th
sample), so that reweighting over the evaluated samples stays unbiased.

**Parameters**

- `records_path` (*str*): The records file (not open for writing elsewhere: HDF5 locks it).
- `make_system` (*callable*): `make_system(rng)`: a system of the run's geometry with the new calculator. Called once per worker (picklable for `n_workers > 1`).
- `name` (*str*): Name of the new potential.
- `subset` (*callable, optional*): `subset(key, sample) -> bool`: evaluate only these. Default None (all stored configurations).
- `n_workers` (*int, optional*): Worker processes. Default 1.
- `chunk` (*int, optional*): Configurations per task and between flushes. Default 200.
- `progress` (*callable, optional*): Called with `(done, total)` after every chunk. Default None.

**Returns**

- *int*: Number of configurations evaluated in this call.

**Raises**

- *ValueError*: If `name` is the sampling potential.

## `regc.observables`

### `Observable`

```python
Observable(*args, **kwargs)
```

Base class: a named, fixed-length vector function of a configuration.

Subclasses set `name`, implement `labels` (one per component) and
`__call__` (system -> numpy array of that length), and describe
themselves in `metadata` (JSON-able; stored in `samples.npz` so the
analysis can interpret the components without the original objects).

#### `Observable.labels`

```python
Observable.labels() -> List[str]
```

Names of the components.

**Returns**

- *list of str*: One label per component.

**Raises**

- *NotImplementedError*: On the base class.

#### `Observable.size` (property)

int: Number of components.

#### `Observable.metadata`

```python
Observable.metadata() -> dict
```

JSON-able description (type, parameters, component labels).

**Returns**

- *dict*: At least `'type'` and `'labels'`.

---

### `CoordinationNumber`

```python
CoordinationNumber(name: str, r_cut: float, center: str = 'mobile', neighbor: str = 'substrate', z_range: Optional[Tuple[float, float]] = None)
```

Mean coordination number of a set of atoms.

Records `[bonds, centers]`: the number of (centre, neighbour) pairs
closer than `r_cut` and the number of centre atoms. The reweighted
mean coordination is <bonds> / <centers>
(`MBARSurfaceAnalysis.mean_coordination`).

**Parameters**

- `name` (*str*): Observable name (sample key `'obs:<name>'`).
- `r_cut` (*float*): Bond cutoff, in Angstrom (e.g. the first minimum of the corresponding RDF; [ZSG] use 2.0 A for D-Si).
- `center` (*str, optional*): Atoms whose coordination is counted. Default `'mobile'`.
- `neighbor` (*str, optional*): Atoms counted as neighbours. Default `'substrate'`.
- `z_range` (*tuple of float, optional*): Restrict the centre atoms to `z_min <= z < z_max`. Default None.

**Constructor**

Store the definition.

**Parameters**

- `name, r_cut, center, neighbor, z_range`: See the class docstring.

#### `CoordinationNumber.labels`

```python
CoordinationNumber.labels() -> List[str]
```

Component names.

**Returns**

- *list of str*: `['bonds', 'centers']`.

#### `CoordinationNumber.metadata`

```python
CoordinationNumber.metadata() -> dict
```

JSON-able description.

**Returns**

- *dict*: Type, labels and parameters.

#### `CoordinationNumber.coordination`

```python
CoordinationNumber.coordination(system) -> np.ndarray
```

Coordination number of every centre atom.

**Parameters**

- `system` (*ParticleSystem*): The configuration.

**Returns**

- *numpy.ndarray*: Integer array, one entry per centre atom.

---

### `CoordinationHistogram`

```python
CoordinationHistogram(name: str, r_cut: float, max_cn: int = 12, center: str = 'substrate', neighbor: str = 'all', z_range: Optional[Tuple[float, float]] = None)
```

Distribution of the coordination numbers of a set of atoms.

Records how many centre atoms have coordination 0, 1, ..., `max_cn`
(the last bin collects `>= max_cn`): [ZSG]'s coordination histogram
H_coord of the top-layer Si atoms, a structural phase descriptor.
`MBARSurfaceAnalysis.coordination_distribution` normalises the
reweighted counts to a probability per centre atom.

**Parameters**

- `name` (*str*): Observable name.
- `r_cut` (*float*): Bond cutoff, in Angstrom.
- `max_cn` (*int, optional*): Largest coordination with its own bin. Default 12.
- `center, neighbor, z_range`: As in `CoordinationNumber` (defaults `'substrate'` and `'all'`: the substrate atoms' bonds to anything).

**Constructor**

Store the definition.

**Parameters**

- `name, r_cut, max_cn, center, neighbor, z_range`: See the class docstring.

#### `CoordinationHistogram.labels`

```python
CoordinationHistogram.labels() -> List[str]
```

Component names.

**Returns**

- *list of str*: `['cn0', 'cn1', ..., 'cn<max_cn>']`.

#### `CoordinationHistogram.metadata`

```python
CoordinationHistogram.metadata() -> dict
```

JSON-able description.

**Returns**

- *dict*: Type, labels and parameters.

---

### `RadialDistribution`

```python
RadialDistribution(name: str, r_max: float, n_bins: int = 100, a: str = 'mobile', b: str = 'substrate', z_range: Optional[Tuple[float, float]] = None)
```

Pair-distance histogram between two atom sets (for an RDF).

Records the number of (a, b) pairs per distance bin (self-pairs
excluded), followed by `n_a`, `n_b` and the system volume.
`MBARSurfaceAnalysis.radial_distribution` turns the reweighted
histogram into

* `'per_center'`: n(r) = <H(r)> / (<n_a> dr), neighbours of a centre
  per unit distance -- its integral up to r_c is the mean coordination
  (the natural form for surfaces and clusters, where no uniform
  reference density exists);
* `'density'`: n(r) / (4 pi r^2), the local density of b around a;
* `'g'`: that density over <n_b> / V, the bulk-normalised g(r)
  (meaningful for bulk systems only: V includes a slab's vacuum).

**Parameters**

- `name` (*str*): Observable name.
- `r_max` (*float*): Largest distance, in Angstrom (below the minimum-image radius).
- `n_bins` (*int, optional*): Number of distance bins. Default 100.
- `a, b` (*str, optional*): The two atom sets. Default `'mobile'` and `'substrate'`.
- `z_range` (*tuple of float, optional*): Restrict the a atoms to `z_min <= z < z_max`. Default None.

**Constructor**

Store the definition.

**Parameters**

- `name, r_max, n_bins, a, b, z_range`: See the class docstring.

#### `RadialDistribution.edges` (property)

numpy.ndarray: The n_bins + 1 bin edges, in Angstrom.

#### `RadialDistribution.labels`

```python
RadialDistribution.labels() -> List[str]
```

Component names.

**Returns**

- *list of str*: `['h0', ..., 'h<n_bins-1>', 'n_a', 'n_b', 'volume']`.

#### `RadialDistribution.metadata`

```python
RadialDistribution.metadata() -> dict
```

JSON-able description.

**Returns**

- *dict*: Type, labels and parameters (incl. the bin edges).

---

### `FunctionObservable`

```python
FunctionObservable(name: str, func: Callable, labels: Optional[Sequence[str]] = None)
```

Wrap a user function `f(system) -> float or array` as an observable.

The function must be picklable (a module-level function, not a lambda)
to run in parallel workers, and must always return the same number of
components.

**Parameters**

- `name` (*str*): Observable name.
- `func` (*callable*): `func(system)` returning a float or a 1-D array.
- `labels` (*sequence of str, optional*): Component labels. Default None: `['value']` (one component).

**Constructor**

Store the function.

**Parameters**

- `name, func, labels`: See the class docstring.

#### `FunctionObservable.labels`

```python
FunctionObservable.labels() -> List[str]
```

Component names.

**Returns**

- *list of str*: As given (default `['value']`).

#### `FunctionObservable.metadata`

```python
FunctionObservable.metadata() -> dict
```

JSON-able description.

**Returns**

- *dict*: Type, labels and the function's qualified name.

## Config runner (`regc.runner`)

Not exported at the package top level: `from regc.runner import ...`. The config file format is in MANUAL.md, Part II.

### `load_config`

```python
load_config(path: str) -> Dict[str, Any]
```

Load a JSON or YAML config file.

Format is inferred from the extension.

**Parameters**

- `path` (*str*): Path to the config file. `.yaml`/`.yml` is parsed as YAML (requires `pip install pyyaml`); anything else is parsed as JSON.

**Returns**

- *dict*: The parsed config. A relative `structure.file` is resolved against the config file's own directory, and so are the potential files of the `calculator` section when they exist there (see the module docstring), so the config works from any working directory.

---

### `validate_config`

```python
validate_config(config: Dict[str, Any]) -> None
```

Check a config's sections and keys before anything expensive runs.

**Parameters**

- `config` (*dict*): The parsed config.

**Raises**

- *ValueError*: On an unknown section or key, a missing required section, or keys of the other sampling scheme.

---

### `setup_from_config`

```python
setup_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]
```

Everything a run needs, built from a config but nothing sampled yet.

**Parameters**

- `config` (*dict*): The parsed config (see the module docstring).
- `verbose` (*bool, optional*): Print the setup's key numbers. Default True.

**Returns**

- *dict*: `geometry` (`SlabGeometry` or `ClusterGeometry`), `lattice_constant` (A, or None), `calc_spec`, `species`, `masses`, `feed` (`GasFeed`, or None when the run needs no reservoir), `temperatures_K`, `mu_grid`, `pressures`/`pressure_unit` (None for an explicit mu ladder), `lambda_cubed_of_T`, `insertion_z_range` (slab) or `insertion_shell` (cluster), `system_factory` and `metadata` (JSON-able; what a resumed run must agree with, and what `analyze_from_config` reads).

**Raises**

- *ValueError*: On an invalid config (see `validate_config`) or an insertion z-range outside the box.

---

### `run_from_config`

```python
run_from_config(config: Dict[str, Any], verbose: bool = True, resume: bool = False, overwrite: bool = False, time_limit: Optional[float] = None, n_workers: Optional[int] = None) -> Dict[str, Any]
```

Run the whole config-driven pipeline: structure to phase diagram.

See this module's docstring and `examples/pd111_emt_example.yaml`
for the config schema. Writes `samples.npz`, `run_metadata.json`
(and, with a `checkpoint` section, `checkpoint.pkl`), trajectories
if asked for, and -- with a `phase_diagram` section -- the phase
diagram plot(s) and raw grid (.npz), all to `output.directory`.

**Parameters**

- `config` (*dict*): The parsed config (see `load_config`).
- `verbose` (*bool, optional*): If True, print progress messages. Default is True.
- `resume` (*bool, optional*): With a `checkpoint` section: continue the run saved in the output directory, if there is one. Default False.
- `overwrite` (*bool, optional*): With a `checkpoint` section: start afresh even if the output directory holds a run. Default False.
- `time_limit` (*float, optional*): With a `checkpoint` section: seconds after which to stop cleanly (overrides `checkpoint.time_limit_s`). Default None.
- `n_workers` (*int, optional*): Worker processes (overrides `replica_exchange.n_workers`). Default None.

**Returns**

- *dict*: `'sim'` (the `ReplicaExchangeGCMC`), `'setup'` (see `setup_from_config`), `'status'` (`RunStatus` of a checkpointed run, else None), `'finished'` (False if a checkpointed run stopped for its time limit; no analysis is done then), `'D0_eV'`/`'e_atom_eV'` (of the single species' reservoir, or dicts by species; None without a reservoir), plus everything `analyze_from_config` returns when the config has a `phase_diagram` section and the run finished.

**Raises**

- *ValueError*: On an invalid config (see `validate_config`), or `burn_in_steps` with a checkpointed run of several chunks.
- *FileExistsError*: With a `checkpoint` section, if the output directory holds a run and neither `resume` nor `overwrite` is set.

---

### `analyze_from_config`

```python
analyze_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]
```

MBAR (T, p) phase diagram from the samples a run saved.

Reads `samples.npz` and `run_metadata.json` from
`output.directory` (no calculator is needed: the reservoirs are
rebuilt from the metadata), so a finished run can be re-analysed with
other `phase_diagram` settings (`regc-run --analyze-only`).

**Parameters**

- `config` (*dict*): The parsed config; its `phase_diagram` and `output` sections are used.
- `verbose` (*bool, optional*): Print a summary. Default True.

**Returns**

- *dict*: `'analysis'` (`MBARSurfaceAnalysis`), `'theta_grid'` (shape (n_T, n_p)), `'T_grid_K'`, `'p_grid'` (in `'pressure_unit'`), `'p_grid_atm'`, `'coverage_per'`, `'observable'` and, with `most_probable: true`, `'most_probable_N'`.

**Raises**

- *ValueError*: If the config has no `phase_diagram` section, the run had no reservoir, or `coverage_per` is unknown.
- *FileNotFoundError*: If the output directory holds no saved run.

---

### `check_config`

```python
check_config(config: Dict[str, Any]) -> Dict[str, Any]
```

Build the setup and print it, with the mu ladder, without sampling.

**Parameters**

- `config` (*dict*): The parsed config.

**Returns**

- *dict*: The setup (see `setup_from_config`).

---

### `reevaluate_from_config`

```python
reevaluate_from_config(config: Dict[str, Any], potential: Dict[str, Any], name: str, every: int = 1, n_workers: int = 1, verbose: bool = True) -> int
```

Re-evaluate a run's recorded configurations with another potential.

The run's config with its `calculator` section (and any other
top-level section given in `potential`, e.g. `reservoir`) replaced
builds the same geometry with the new potential; its energies go to
`E/<name>` in the records file, and its gas reservoirs (whose
reference energies depend on the potential) to the file's attribute
`reservoirs:<name>`, for the pressure axes of the reweighted phase
diagram (`phase_diagram.potentials`).

**Parameters**

- `config` (*dict*): The run's parsed config (`records: {save: true}` must have been on).
- `potential` (*dict*): Top-level sections to replace, at least `calculator`.
- `name` (*str*): Name of the new potential.
- `every` (*int, optional*): Evaluate every this many stored samples (by sample index). Default 1.
- `n_workers` (*int, optional*): Worker processes. Default 1.
- `verbose` (*bool, optional*): Print progress. Default True.

**Returns**

- *int*: Configurations evaluated.

**Raises**

- *FileNotFoundError*: If the run has no records file.
- *ValueError*: If `potential` has no calculator, or the new setup's cell differs from the recorded one (fix the lattice constant in `structure`).

---

### `main`

```python
main(argv: Optional[list] = None) -> int
```

Command-line entry point: `regc-run config.yaml`.

Runs the structure -> REGC -> MBAR -> (T, p) phase diagram pipeline of
`run_from_config` for one config file; `--check` only builds and
prints the setup, `--analyze-only` re-analyses saved samples.

**Parameters**

- `argv` (*list of str, optional*): Arguments (without the program name). Default None: `sys.argv[1:]`.

**Returns**

- *int*: Process exit status: 0 on success, `EXIT_RESUBMIT` (75) if a checkpointed run stopped for its time limit.

---

### `reevaluate_main`

```python
reevaluate_main(argv: Optional[list] = None) -> int
```

Command-line entry point: `regc-reevaluate config.yaml potential.yaml --name B`.

Re-evaluates the configurations a run recorded (`records: {save:
true}`) with the potential described in `potential.yaml` (a
`calculator` section, plus any other top-level section to replace);
then list `B` under `phase_diagram.potentials` and run
`regc-run --analyze-only` for the reweighted phase diagram.

**Parameters**

- `argv` (*list of str, optional*): Arguments. Default None: `sys.argv[1:]`.

**Returns**

- *int*: 0.
