"""
gcmc_re: multidimensional replica-exchange grand-canonical Monte Carlo.

An implementation of the REGC method of Zhou, Scheffler & Ghiringhelli,
PRB 100, 174106 (2019): a (temperature x chemical potential) grid of
grand-canonical replicas with configuration swaps, MC/HMC/MD diffusion,
any potential, and MBAR reweighting into (T, p) phase diagrams. The
algorithm, with every acceptance rule, is documented in
``docs/REGC_algorithm.md``.

Core (numpy only)
-----------------
calculators  Calculator, LJCalculator, PairLJCalculator, ASECalculatorAdapter
systems      ParticleSystem, BulkSystem, SlabSystem (aliases LJSystem, SlabLJSystem)
state        ThermoState
replica      Replica -- one grid point: GCMC moves and diffusion
engine       ReplicaExchangeGCMC -- the grid, swaps, `run` and `run_regc`
units        kelvin_to_eV, eV_to_kelvin, ThermalWavelengthCubed, ...
pressure     simple (reduced-units) mu <-> p conventions

Real units, potentials and gas reservoirs
-----------------------------------------
reservoir       MonatomicIdealGasReservoir, DiatomicReservoir (e.g. H2), and the
                low-level diatomic functions
structures      cubic_lattice_constant, slab_geometry_from_atoms, SlabGeometry,
                insertion_energy_scan  (needs ase)
lammps_backend  LammpsCalculator (any LAMMPS pair style; native MD),
                eam_pair_commands, lammps_ase_calculator  (needs the lammps module)

Running and analysing
---------------------
driver    run_with_checkpoints, RunStatus, EXIT_RESUBMIT -- long/batch runs
io        save_samples, load_samples, SampleSet
analysis  MBARSurfaceAnalysis  (needs pymbar)
plotting  plot_phase_diagram   (needs matplotlib)
runner    run_from_config      -- the YAML/JSON-driven pipeline

Optional dependencies are imported lazily, inside the functions that need
them, so ``import gcmc_re`` succeeds with numpy alone.
"""

from .calculators import ASECalculatorAdapter, Calculator, LJCalculator, PairLJCalculator
from .lammps_backend import LammpsCalculator, eam_pair_commands, lammps_ase_calculator
from .systems import BulkSystem, LJSystem, ParticleSystem, SlabLJSystem, SlabSystem
from .state import ThermoState
from .replica import Replica
from .engine import SWAP_TYPES, ReplicaExchangeGCMC
from .pressure import (
    invert_mu_of_Tp,
    mu_from_pressure,
    mu_grid_from_pressures,
    p0_ideal_gas_wavelength,
    p0_reduced_units,
    pressure_from_mu,
)
from .analysis import MBARSurfaceAnalysis
from .plotting import plot_phase_diagram
from .units import (
    PRESSURE_UNITS,
    ThermalWavelengthCubed,
    default_mass_amu,
    eV_to_kelvin,
    kelvin_to_eV,
    pascal_per,
    thermal_wavelength_angstrom,
    thermal_wavelength_cubed_angstrom3,
)
from .reservoir import (
    H2_BOND_LENGTH_ANGSTROM,
    H2_D0_EXPERIMENTAL_EV,
    H2_MASS_AMU,
    H2_VIBRATIONAL_WAVENUMBER_CM,
    H_MASS_AMU,
    DiatomicReservoir,
    MonatomicIdealGasReservoir,
    Reservoir,
    diatomic_dissociation_energy_eV,
    diatomic_gas_mu_eV,
    mu_atomic_from_diatomic_reservoir_eV,
    pressure_from_atomic_mu_reservoir_pascal,
)
from .structures import (
    SlabGeometry,
    cubic_lattice_constant,
    insertion_energy_scan,
    slab_geometry_from_atoms,
)
from .io import SampleSet, load_samples, save_samples
from .driver import EXIT_RESUBMIT, RunStatus, run_with_checkpoints

__all__ = [
    # calculators and systems
    "Calculator", "LJCalculator", "PairLJCalculator", "ASECalculatorAdapter", "LammpsCalculator",
    "eam_pair_commands", "lammps_ase_calculator",
    "ParticleSystem", "BulkSystem", "SlabSystem", "LJSystem", "SlabLJSystem",
    # engine
    "ThermoState", "Replica", "ReplicaExchangeGCMC", "SWAP_TYPES",
    # units and pressure conventions
    "kelvin_to_eV", "eV_to_kelvin", "pascal_per", "PRESSURE_UNITS", "ThermalWavelengthCubed",
    "thermal_wavelength_angstrom", "thermal_wavelength_cubed_angstrom3", "default_mass_amu",
    "p0_reduced_units", "p0_ideal_gas_wavelength",
    "mu_from_pressure", "pressure_from_mu", "mu_grid_from_pressures", "invert_mu_of_Tp",
    # reservoirs
    "Reservoir", "MonatomicIdealGasReservoir", "DiatomicReservoir",
    "diatomic_dissociation_energy_eV", "diatomic_gas_mu_eV",
    "mu_atomic_from_diatomic_reservoir_eV", "pressure_from_atomic_mu_reservoir_pascal",
    "H2_BOND_LENGTH_ANGSTROM", "H2_VIBRATIONAL_WAVENUMBER_CM", "H_MASS_AMU", "H2_MASS_AMU",
    "H2_D0_EXPERIMENTAL_EV",
    # structures
    "cubic_lattice_constant", "slab_geometry_from_atoms", "SlabGeometry", "insertion_energy_scan",
    # running and analysis
    "run_with_checkpoints", "RunStatus", "EXIT_RESUBMIT",
    "save_samples", "load_samples", "SampleSet",
    "MBARSurfaceAnalysis", "plot_phase_diagram",
]

__version__ = "0.2.0"
