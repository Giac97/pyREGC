"""
Referencing a GCMC-exchangeable atomic species (e.g. H) against a real
gas-phase diatomic reservoir (e.g. H2) at a given (T, p).

The subtlety this module exists to handle: GCMC here inserts/deletes single
atoms, but a real experiment controls a diatomic gas's pressure, and
mu_atom = (1/2) mu_molecule(p, T) only once mu_molecule is referenced to
"two separated ground-state atoms" as its zero of energy -- which requires
subtracting the molecule's dissociation energy D0. If your Calculator is an
approximation (EMT, EAM, an MLIP), D0 must come from THAT SAME calculator,
not from an experimental table: the calculator has its own energy scale
(e.g. EMT's isolated-atom reference has no absolute physical meaning), and
mixing in an external D0 would silently introduce an offset error into
every mu_atom you compute. This is the standard "atomistic thermodynamics"
approach used throughout surface science (Reuter & Scheffler-style ab
initio phase diagrams) -- the gas-phase reservoir's rotational/vibrational
entropy is real physics (universal, spectroscopic, calculator-independent),
but the energy reference connecting gas-phase molecule to separated atoms
must be computed self-consistently with whatever potential will evaluate
the adsorbed atoms.

`diatomic_dissociation_energy_eV` computes that self-consistent D0 (well
depth minus zero-point energy, via a bond-length scan and a local harmonic
fit) for ANY ASE-compatible calculator and ANY homonuclear diatomic --
not just H2/EMT, though that's this package's running example.

The second ingredient is the calculator's isolated-atom energy e_atom.
GCMC acceptance rules measure energies on the calculator's own scale, on
which an isolated gas atom has energy e_atom, not 0 (EMT's isolated H
atom: +3.21 eV). The reservoir mu_atom must be on that same scale:

    mu_atom = (1/2) * [mu_molecule(T, p) - D0] + e_atom

(equivalently, (1/2) of the molecule's chemical potential measured from
the calculator's own energy of the relaxed molecule plus its zero-point
energy). Dropping e_atom shifts every mu_atom by -e_atom. Both conversion
functions below therefore take `e_atom_eV` as a required keyword
argument; `diatomic_dissociation_energy_eV` returns it as ``'e_atom_eV'``,
and it is exactly 0 for the native `LJCalculator`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .units import (AMU, BOLTZMANN_K_SI, EV_IN_JOULES, PLANCK_H, ThermalWavelengthCubed,
                    eV_to_kelvin, kelvin_to_eV, pascal_per, thermal_wavelength_angstrom,
                    thermal_wavelength_cubed_angstrom3)

PLANCK_HBAR = PLANCK_H / (2.0 * math.pi)

# Standard experimental H2 spectroscopic constants (calculator-independent --
# real physics of the gas-phase molecule, not tied to any particular
# potential). Bond length and vibrational wavenumber are well-established
# textbook values.
H2_BOND_LENGTH_ANGSTROM = 0.741
H2_VIBRATIONAL_WAVENUMBER_CM = 4401.2
H_MASS_AMU = 1.008
H2_MASS_AMU = 2.016
H2_D0_EXPERIMENTAL_EV = 4.478   # zero-point-corrected dissociation energy of H2


def diatomic_dissociation_energy_eV(ase_calculator, symbol: str,
                                     mass_amu: Optional[float] = None,
                                     r_scan: Optional[np.ndarray] = None,
                                     cell: float = 20.0) -> Dict[str, float]:
    """
    Compute a self-consistent, zero-point-corrected dissociation energy.

    Computes D0 of the homonuclear diatomic ``symbol``+``symbol``, as
    `ase_calculator` sees it: a coarse bond-length scan locates the well,
    a local scan and iterated parabolic interpolation pin the minimum down
    (independent of the scan grid), and a central finite difference gives
    the harmonic force constant (hence the vibrational frequency and
    zero-point energy).

    Parameters
    ----------
    ase_calculator : ase.calculators.calculator.Calculator
        The ASE calculator to evaluate the diatomic and isolated-atom
        energies with. D0 must come from this same calculator (not an
        experimental table) for approximate potentials -- see this
        module's docstring.
    symbol : str
        The element symbol of the homonuclear diatomic (e.g. ``'H'`` for
        H2).
    mass_amu : float, optional
        Atomic mass, in atomic mass units. Default is None, which looks
        it up via ASE's data tables for `symbol`.
    r_scan : numpy.ndarray, optional
        Bond lengths (in Angstrom) for the initial coarse scan. Default is
        None, which uses ``numpy.linspace(0.4, 2.5, 60)``.
    cell : float, optional
        Cubic cell edge length (in Angstrom) for the (non-periodic)
        evaluation cell. Default is 20.0.

    Returns
    -------
    dict
        Keys ``'r0_angstrom'`` (equilibrium bond length),
        ``'e_min_eV'`` (energy at that bond length),
        ``'e_atom_eV'`` (isolated-atom energy),
        ``'force_constant_eV_per_A2'`` (harmonic force constant),
        ``'omega_rad_per_s'`` (vibrational angular frequency),
        ``'zpe_eV'`` (zero-point energy),
        ``'De_eV'`` (well depth, 2*e_atom - e_min), and
        ``'D0_eV'`` (zero-point-corrected dissociation energy,
        De - zpe). Inspect these (particularly r0_angstrom and De_eV
        against known experimental values) before trusting the
        potential's gas-phase description at all.
    """
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers

    if mass_amu is None:
        mass_amu = float(atomic_masses[atomic_numbers[symbol]])
    if r_scan is None:
        r_scan = np.linspace(0.4, 2.5, 60)

    def energy_at(r: float) -> float:
        atoms = Atoms(f"{symbol}2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
                       cell=[cell, cell, cell], pbc=False)
        atoms.calc = ase_calculator
        return float(atoms.get_potential_energy())

    atom = Atoms(symbol, positions=[[0.0, 0.0, 0.0]], cell=[cell, cell, cell], pbc=False)
    atom.calc = ase_calculator
    e_atom = float(atom.get_potential_energy())

    coarse_e = np.array([energy_at(r) for r in r_scan])
    r0_guess = float(r_scan[int(np.argmin(coarse_e))])

    fine_r = np.linspace(max(r_scan[0], r0_guess - 0.05), min(r_scan[-1], r0_guess + 0.05), 21)
    fine_e = np.array([energy_at(r) for r in fine_r])
    i_fine = int(np.argmin(fine_e))
    r0 = float(fine_r[i_fine])
    # refine the minimum off the grid (iterated parabolic interpolation), so the
    # result does not depend on the scan grid
    step = float(fine_r[1] - fine_r[0])
    for _ in range(4):
        e_m, e_0, e_p = energy_at(r0 - step), energy_at(r0), energy_at(r0 + step)
        curvature = e_m - 2.0 * e_0 + e_p
        if curvature <= 0.0:
            break
        r0 += 0.5 * step * (e_m - e_p) / curvature
        step /= 10.0
    e_min = float(energy_at(r0))

    h = 0.01
    k_eV_per_A2 = (energy_at(r0 + h) - 2.0 * e_min + energy_at(r0 - h)) / h ** 2

    mu_reduced_kg = (mass_amu * AMU) / 2.0
    k_SI = k_eV_per_A2 * EV_IN_JOULES / (1.0e-10) ** 2
    omega = math.sqrt(k_SI / mu_reduced_kg) if k_SI > 0 else 0.0
    zpe_eV = 0.5 * PLANCK_HBAR * omega / EV_IN_JOULES

    De = 2.0 * e_atom - e_min
    D0 = De - zpe_eV
    return {
        "r0_angstrom": r0, "e_min_eV": e_min, "e_atom_eV": e_atom,
        "force_constant_eV_per_A2": k_eV_per_A2, "omega_rad_per_s": omega,
        "zpe_eV": zpe_eV, "De_eV": De, "D0_eV": D0,
    }


def diatomic_gas_mu_eV(T_kelvin: float, p_pascal: float, mass_amu: float,
                        bond_length_angstrom: float, vib_wavenumber_cm: float,
                        symmetry_number: int = 2) -> float:
    """
    Compute the chemical potential of a real diatomic gas at (T, p).

    Chemical potential (eV) of a real, classical-rotor/harmonic-oscillator
    ideal gas of homonuclear diatomic molecules at (T, p), referenced to
    the molecule's own ground rotational/vibrational/electronic state
    (i.e. NOT yet shifted to separated atoms -- combine with a
    self-consistent D0, see `mu_atomic_from_diatomic_reservoir_eV`, to get
    that).

    Uses the classical (high-temperature) rotational partition function,
    valid whenever T is well above the rotational temperature theta_rot
    (true for H2 -- theta_rot ~ 87 K -- at any remotely ordinary
    temperature).

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    p_pascal : float
        Gas pressure, in Pascal.
    mass_amu : float
        Molecular mass, in atomic mass units.
    bond_length_angstrom : float
        Equilibrium bond length, in Angstrom.
    vib_wavenumber_cm : float
        Vibrational wavenumber, in cm^-1.
    symmetry_number : int, optional
        Rotational symmetry number (2 for a homonuclear diatomic).
        Default is 2.

    Returns
    -------
    float
        The molecule's chemical potential, in eV.
    """
    T_eV = kelvin_to_eV(T_kelvin)
    lam_m = thermal_wavelength_angstrom(T_kelvin, mass_amu) * 1.0e-10
    T_J = BOLTZMANN_K_SI * T_kelvin
    p0_pascal = T_J / lam_m ** 3

    r_m = bond_length_angstrom * 1.0e-10
    mu_reduced_kg = (mass_amu * AMU) / 2.0
    I = mu_reduced_kg * r_m ** 2
    theta_rot = PLANCK_HBAR ** 2 / (2.0 * I * BOLTZMANN_K_SI)
    q_rot = T_kelvin / (symmetry_number * theta_rot)

    theta_vib = PLANCK_H * (vib_wavenumber_cm * 2.99792458e10) / BOLTZMANN_K_SI
    q_vib = 1.0 / (1.0 - math.exp(-theta_vib / T_kelvin))

    return T_eV * math.log(p_pascal / p0_pascal) - T_eV * math.log(q_rot * q_vib)


def mu_atomic_from_diatomic_reservoir_eV(T_kelvin: float, p_pascal: float, D0_eV: float,
                                          mass_molecule_amu: float, bond_length_angstrom: float,
                                          vib_wavenumber_cm: float, symmetry_number: int = 2,
                                          *, e_atom_eV: float) -> float:
    """
    Compute an atomic species' chemical potential from a diatomic reservoir.

    mu_atom(T, p) = (1/2) * [mu_molecule(T, p) - D0] + e_atom, with
    mu_molecule from real gas-phase spectroscopy, and D0 and e_atom
    computed self-consistently with whatever Calculator will evaluate the
    adsorbed atoms (see `diatomic_dissociation_energy_eV` and this module's
    docstring). This is the number to feed `ReplicaExchangeGCMC` as `mu`
    for an atomic species in equilibrium with a real diatomic gas
    reservoir.

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    p_pascal : float
        Reservoir gas pressure, in Pascal.
    D0_eV : float
        Self-consistent, zero-point-corrected dissociation energy, in eV
        (see `diatomic_dissociation_energy_eV`).
    mass_molecule_amu : float
        Molecular mass, in atomic mass units.
    bond_length_angstrom : float
        Equilibrium bond length, in Angstrom.
    vib_wavenumber_cm : float
        Vibrational wavenumber, in cm^-1.
    symmetry_number : int, optional
        Rotational symmetry number (2 for a homonuclear diatomic).
        Default is 2.
    e_atom_eV : float
        The calculator's energy of one isolated atom (keyword-only,
        required): the zero of the GCMC energy scale. ``'e_atom_eV'`` from
        `diatomic_dissociation_energy_eV`; 0 for `LJCalculator`.

    Returns
    -------
    float
        The atomic species' chemical potential, in eV, on the calculator's
        energy scale.
    """
    mu_molecule = diatomic_gas_mu_eV(T_kelvin, p_pascal, mass_molecule_amu,
                                      bond_length_angstrom, vib_wavenumber_cm, symmetry_number)
    return 0.5 * (mu_molecule - D0_eV) + e_atom_eV


def pressure_from_atomic_mu_reservoir_pascal(T_kelvin: float, mu_atom_eV: float, D0_eV: float,
                                              mass_molecule_amu: float, bond_length_angstrom: float,
                                              vib_wavenumber_cm: float, symmetry_number: int = 2,
                                              *, e_atom_eV: float) -> float:
    """
    Invert `mu_atomic_from_diatomic_reservoir_eV` for the reservoir pressure.

    Computes the real gas pressure (Pa) whose reservoir would produce the
    given atomic mu_atom at temperature T. Useful for translating a mu
    range you've already explored (e.g. by direct GCMC scanning) into
    "what real pressure would this correspond to".

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    mu_atom_eV : float
        The atomic species' chemical potential, in eV.
    D0_eV : float
        Self-consistent, zero-point-corrected dissociation energy, in eV.
    mass_molecule_amu : float
        Molecular mass, in atomic mass units.
    bond_length_angstrom : float
        Equilibrium bond length, in Angstrom.
    vib_wavenumber_cm : float
        Vibrational wavenumber, in cm^-1.
    symmetry_number : int, optional
        Rotational symmetry number (2 for a homonuclear diatomic).
        Default is 2.
    e_atom_eV : float
        The calculator's isolated-atom energy (keyword-only, required);
        see `mu_atomic_from_diatomic_reservoir_eV`.

    Returns
    -------
    float
        The reservoir gas pressure, in Pascal.
    """
    T_eV = kelvin_to_eV(T_kelvin)
    lam_m = thermal_wavelength_angstrom(T_kelvin, mass_molecule_amu) * 1.0e-10
    T_J = BOLTZMANN_K_SI * T_kelvin
    p0_pascal = T_J / lam_m ** 3

    r_m = bond_length_angstrom * 1.0e-10
    mu_reduced_kg = (mass_molecule_amu * AMU) / 2.0
    I = mu_reduced_kg * r_m ** 2
    theta_rot = PLANCK_HBAR ** 2 / (2.0 * I * BOLTZMANN_K_SI)
    q_rot = T_kelvin / (symmetry_number * theta_rot)

    theta_vib = PLANCK_H * (vib_wavenumber_cm * 2.99792458e10) / BOLTZMANN_K_SI
    q_vib = 1.0 / (1.0 - math.exp(-theta_vib / T_kelvin))

    mu_molecule = 2.0 * (mu_atom_eV - e_atom_eV) + D0_eV
    return p0_pascal * math.exp(mu_molecule / T_eV) * q_rot * q_vib


# =============================================================================
# Reservoir objects: one interface for every gas reservoir
# =============================================================================
class Reservoir:
    """
    A gas reservoir fixing the exchanged species' chemical potential.

    Subclasses implement the (T, p) <-> mu relation of a particular gas;
    this base class turns it into the pieces the rest of the package needs:
    a replica-ladder mu grid (`mu_grid`), the ``lambda_cubed_of_T``
    callable of the exchanged particle (`lambda_cubed_of_T`), and the
    ``mu_of_Tp``/``p_of_Tmu`` pair `MBARSurfaceAnalysis` takes
    (`analysis_functions`). Temperatures are in Kelvin, pressures in Pascal,
    chemical potentials in eV on the calculator's energy scale.

    Attributes
    ----------
    mass_exchanged_amu : float
        Mass of the particle GCMC inserts and deletes (sets Lambda).
    """

    mass_exchanged_amu: float = 1.0

    def mu_eV(self, T_kelvin: float, p_pascal: float) -> float:
        """
        Chemical potential of the exchanged species at (T, p).

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Reservoir pressure, in Pascal.

        Returns
        -------
        float
            mu, in eV.

        Raises
        ------
        NotImplementedError
            On the base class.
        """
        raise NotImplementedError

    def pressure_pascal(self, T_kelvin: float, mu_eV: float) -> float:
        """
        Reservoir pressure giving chemical potential `mu_eV` at T.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        mu_eV : float
            Chemical potential of the exchanged species, in eV.

        Returns
        -------
        float
            Pressure, in Pascal.

        Raises
        ------
        NotImplementedError
            On the base class.
        """
        raise NotImplementedError

    def mu_grid(self, temperatures_K: Sequence[float], pressures: Sequence[float],
                unit: str = "Pa") -> np.ndarray:
        """
        Chemical potentials of a (temperature x pressure) replica ladder.

        Parameters
        ----------
        temperatures_K : sequence of float
            Temperatures, in Kelvin.
        pressures : sequence of float
            Pressures, in `unit`.
        unit : str, optional
            Pressure unit (see `gcmc_re.units.PRESSURE_UNITS`). Default ``'Pa'``.

        Returns
        -------
        numpy.ndarray
            Shape (len(temperatures_K), len(pressures)): mu in eV, ready for
            `ReplicaExchangeGCMC`'s 2D ``mus`` argument.
        """
        f = pascal_per(unit)
        return np.array([[self.mu_eV(T, p * f) for p in pressures] for T in temperatures_K])

    def lambda_cubed_of_T(self) -> ThermalWavelengthCubed:
        """
        Lambda^3(T) of the exchanged particle, as a function of k_B*T in eV.

        Returns
        -------
        ThermalWavelengthCubed
            Pass it as ``lambda_cubed_of_T`` to `ReplicaExchangeGCMC` (and,
            through the simulation, to `MBARSurfaceAnalysis`).
        """
        return ThermalWavelengthCubed(self.mass_exchanged_amu)

    def mu_at_T_eV(self, T_eV: float, p: float, unit: str = "Pa") -> float:
        """
        `mu_eV` with the engine's temperature unit (k_B*T in eV).

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        p : float
            Pressure, in `unit`.
        unit : str, optional
            Pressure unit. Default ``'Pa'``.

        Returns
        -------
        float
            mu, in eV.
        """
        return self.mu_eV(eV_to_kelvin(T_eV), p * pascal_per(unit))

    def pressure_at_T_eV(self, T_eV: float, mu_eV: float, unit: str = "Pa") -> float:
        """
        `pressure_pascal` with the engine's temperature unit, in `unit`.

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        mu_eV : float
            Chemical potential, in eV.
        unit : str, optional
            Pressure unit of the result. Default ``'Pa'``.

        Returns
        -------
        float
            Pressure, in `unit`.
        """
        return self.pressure_pascal(eV_to_kelvin(T_eV), mu_eV) / pascal_per(unit)

    def analysis_functions(self, unit: str = "Pa") -> Tuple[Callable, Callable]:
        """
        The ``(mu_of_Tp, p_of_Tmu)`` pair for `MBARSurfaceAnalysis`.

        Parameters
        ----------
        unit : str, optional
            Pressure unit the analysis will be given/return. Default ``'Pa'``.

        Returns
        -------
        tuple of callable
            ``mu_of_Tp(T_eV, p)`` and ``p_of_Tmu(T_eV, mu)`` (picklable).
        """
        return (functools.partial(self.mu_at_T_eV, unit=unit),
                functools.partial(self.pressure_at_T_eV, unit=unit))


@dataclass
class MonatomicIdealGasReservoir(Reservoir):
    """
    Ideal gas of the exchanged atoms themselves ([ZSG] Sec. III A).

    mu = k_B T ln(p Lambda^3 / k_B T) + e_atom, i.e. beta*mu =
    beta*mu0 + ln(beta*p) with mu0 = k_B T ln(Lambda^3) (Eq. (13) of
    docs/REGC_algorithm.md), shifted by the calculator's isolated-atom
    energy.

    Parameters
    ----------
    mass_amu : float
        Mass of the gas atoms, in atomic mass units.
    e_atom_eV : float, optional
        The calculator's energy of one isolated atom (the zero of the GCMC
        energy scale). Default 0.0 (pair potentials).
    """

    mass_amu: float
    e_atom_eV: float = 0.0

    @property
    def mass_exchanged_amu(self) -> float:
        """float: The gas atom's mass (it is also the exchanged particle)."""
        return self.mass_amu

    def _p0_pascal(self, T_kelvin: float) -> float:
        """
        Reference pressure k_B T / Lambda^3.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.

        Returns
        -------
        float
            k_B T / Lambda^3(T), in Pascal.
        """
        lam3_m3 = thermal_wavelength_cubed_angstrom3(T_kelvin, self.mass_amu) * 1e-30
        return BOLTZMANN_K_SI * T_kelvin / lam3_m3

    def mu_eV(self, T_kelvin: float, p_pascal: float) -> float:
        """
        Chemical potential of the gas atoms at (T, p).

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Pressure, in Pascal.

        Returns
        -------
        float
            mu, in eV.
        """
        return kelvin_to_eV(T_kelvin) * math.log(p_pascal / self._p0_pascal(T_kelvin)) + self.e_atom_eV

    def pressure_pascal(self, T_kelvin: float, mu_eV: float) -> float:
        """
        Pressure giving chemical potential `mu_eV` at T.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        mu_eV : float
            Chemical potential, in eV.

        Returns
        -------
        float
            Pressure, in Pascal.
        """
        return self._p0_pascal(T_kelvin) * math.exp((mu_eV - self.e_atom_eV) / kelvin_to_eV(T_kelvin))


@dataclass
class DiatomicReservoir(Reservoir):
    """
    A homonuclear diatomic gas (e.g. H2) exchanging single atoms.

    mu_atom = (1/2) [mu_molecule(T, p) - D0] + e_atom, with mu_molecule from
    real gas-phase spectroscopy (`diatomic_gas_mu_eV`) and D0, e_atom
    consistent with the calculator (see this module's docstring and
    `from_calculator`).

    Parameters
    ----------
    D0_eV : float
        Zero-point-corrected dissociation energy of the molecule, in eV.
    e_atom_eV : float
        The calculator's energy of one isolated atom, in eV.
    mass_molecule_amu : float
        Molecular mass, in atomic mass units.
    bond_length_angstrom : float
        Equilibrium bond length (rotational partition function), in A.
    vib_wavenumber_cm : float
        Vibrational wavenumber, in cm^-1.
    symmetry_number : int, optional
        Rotational symmetry number. Default 2 (homonuclear).
    calculator_info : dict, optional
        Diagnostics of the potential's own molecule (r0, De, D0, ZPE,
        e_atom), as filled in by `from_calculator`. Default None.
    """

    D0_eV: float
    e_atom_eV: float
    mass_molecule_amu: float
    bond_length_angstrom: float
    vib_wavenumber_cm: float
    symmetry_number: int = 2
    calculator_info: Optional[dict] = field(default=None, repr=False)

    @property
    def mass_exchanged_amu(self) -> float:
        """float: Mass of the exchanged atom (half the molecule)."""
        return 0.5 * self.mass_molecule_amu

    def mu_eV(self, T_kelvin: float, p_pascal: float) -> float:
        """
        Chemical potential of the atoms at (T, p) of the molecular gas.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Molecular-gas pressure, in Pascal.

        Returns
        -------
        float
            mu_atom, in eV.
        """
        return mu_atomic_from_diatomic_reservoir_eV(
            T_kelvin, p_pascal, self.D0_eV, self.mass_molecule_amu, self.bond_length_angstrom,
            self.vib_wavenumber_cm, self.symmetry_number, e_atom_eV=self.e_atom_eV)

    def pressure_pascal(self, T_kelvin: float, mu_eV: float) -> float:
        """
        Molecular-gas pressure giving atomic chemical potential `mu_eV`.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        mu_eV : float
            Atomic chemical potential, in eV.

        Returns
        -------
        float
            Pressure, in Pascal.
        """
        return pressure_from_atomic_mu_reservoir_pascal(
            T_kelvin, mu_eV, self.D0_eV, self.mass_molecule_amu, self.bond_length_angstrom,
            self.vib_wavenumber_cm, self.symmetry_number, e_atom_eV=self.e_atom_eV)

    @classmethod
    def hydrogen(cls, D0_eV: float = H2_D0_EXPERIMENTAL_EV, e_atom_eV: float = 0.0
                 ) -> "DiatomicReservoir":
        """
        An H2 reservoir with experimental spectroscopic constants.

        Parameters
        ----------
        D0_eV : float, optional
            Dissociation energy. Default: experiment (4.478 eV).
        e_atom_eV : float, optional
            The calculator's isolated-H energy. Default 0.0.

        Returns
        -------
        DiatomicReservoir
            The reservoir.
        """
        return cls(D0_eV, e_atom_eV, H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM,
                   H2_VIBRATIONAL_WAVENUMBER_CM)

    @classmethod
    def from_calculator(cls, ase_calculator, symbol: str = "H",
                        d0: Union[str, float] = "self",
                        bond_length_angstrom: Optional[float] = None,
                        vib_wavenumber_cm: Optional[float] = None,
                        mass_molecule_amu: Optional[float] = None,
                        symmetry_number: int = 2,
                        r_scan: Optional[np.ndarray] = None) -> "DiatomicReservoir":
        """
        A reservoir consistent with a given potential.

        Evaluates the potential's own molecule and isolated atom
        (`diatomic_dissociation_energy_eV`): e_atom always comes from the
        calculator; D0 does too unless `d0` says otherwise.

        Parameters
        ----------
        ase_calculator : ase.calculators.calculator.Calculator
            The potential the simulation uses (e.g. ASE's EAM, or
            `gcmc_re.lammps_backend.lammps_ase_calculator`).
        symbol : str, optional
            Element of the homonuclear molecule. Default ``'H'``.
        d0 : {'self', 'experimental'} or float, optional
            ``'self'``: the potential's D0 (default, consistent);
            ``'experimental'``: 4.478 eV (H2 only), for potentials whose
            molecule is poor; a number: that D0 in eV.
        bond_length_angstrom, vib_wavenumber_cm : float, optional
            Spectroscopic constants of the real molecule; default to H2's
            for ``symbol='H'``, required otherwise.
        mass_molecule_amu : float, optional
            Molecular mass; default twice the atomic mass from ASE's tables.
        symmetry_number : int, optional
            Rotational symmetry number. Default 2.
        r_scan : numpy.ndarray, optional
            Bond lengths (A) for the coarse scan; default 0.4-3.0 A.

        Returns
        -------
        DiatomicReservoir
            The reservoir, with the potential's diagnostics in
            `calculator_info` (plus ``'d0_source'``).

        Raises
        ------
        ValueError
            If spectroscopic constants are missing for a non-H element, or
            `d0` is ``'experimental'`` for a non-H element or unknown.
        """
        if symbol == "H":
            bond_length_angstrom = bond_length_angstrom or H2_BOND_LENGTH_ANGSTROM
            vib_wavenumber_cm = vib_wavenumber_cm or H2_VIBRATIONAL_WAVENUMBER_CM
        if bond_length_angstrom is None or vib_wavenumber_cm is None:
            raise ValueError(f"give bond_length_angstrom and vib_wavenumber_cm for {symbol}2")
        if mass_molecule_amu is None:
            from .units import default_mass_amu
            mass_molecule_amu = 2.0 * default_mass_amu(symbol)
        info = diatomic_dissociation_energy_eV(
            ase_calculator, symbol, mass_amu=0.5 * mass_molecule_amu,
            r_scan=np.linspace(0.4, 3.0, 80) if r_scan is None else r_scan)
        if d0 == "self":
            D0 = info["D0_eV"]
        elif d0 == "experimental":
            if symbol != "H":
                raise ValueError("d0='experimental' is only tabulated for H2; pass a number")
            D0 = H2_D0_EXPERIMENTAL_EV
        elif isinstance(d0, (int, float)):
            D0 = float(d0)
        else:
            raise ValueError(f"d0 must be 'self', 'experimental' or a number, got {d0!r}")
        info = dict(info, d0_source=d0 if isinstance(d0, str) else "given")
        return cls(D0, info["e_atom_eV"], mass_molecule_amu, bond_length_angstrom,
                   vib_wavenumber_cm, symmetry_number, calculator_info=info)
