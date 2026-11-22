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
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .units import (AMU, BOLTZMANN_K_SI, EV_IN_JOULES, PLANCK_H, ThermalWavelengthCubed,
                    eV_to_kelvin, kelvin_to_eV, pascal_per, thermal_wavelength_angstrom,
                    thermal_wavelength_cubed_angstrom3)

PLANCK_HBAR = PLANCK_H / (2.0 * math.pi)
SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10

# Standard experimental H2 spectroscopic constants (calculator-independent --
# real physics of the gas-phase molecule, not tied to any particular
# potential). Bond length and vibrational wavenumber are well-established
# textbook values.
H2_BOND_LENGTH_ANGSTROM = 0.741
H2_VIBRATIONAL_WAVENUMBER_CM = 4401.2
H_MASS_AMU = 1.008
H2_MASS_AMU = 2.016
H2_D0_EXPERIMENTAL_EV = 4.478   # zero-point-corrected dissociation energy of H2

# Degeneracy of the electronic ground state of homonuclear diatomics that are not
# closed-shell singlets (O2: triplet 3Sigma_g^-); every other element defaults to 1.
ELECTRONIC_GROUND_STATE_DEGENERACY = {"O": 3}


def default_electronic_degeneracy(symbol: str) -> int:
    """
    Electronic ground-state degeneracy of the X2 molecule of element `symbol`.

    Parameters
    ----------
    symbol : str
        Element symbol.

    Returns
    -------
    int
        3 for O (triplet O2), 1 otherwise.
    """
    return ELECTRONIC_GROUND_STATE_DEGENERACY.get(symbol, 1)


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


def _diatomic_p0_and_q_internal(T_kelvin: float, mass_molecule_amu: float,
                                bond_length_angstrom: float, vib_wavenumber_cm: float,
                                symmetry_number: int, electronic_degeneracy: int
                                ) -> Tuple[float, float]:
    """
    Reference pressure and internal partition function of a homonuclear diatomic.

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    mass_molecule_amu : float
        Molecular mass, in atomic mass units.
    bond_length_angstrom : float
        Equilibrium bond length, in Angstrom.
    vib_wavenumber_cm : float
        Vibrational wavenumber, in cm^-1.
    symmetry_number : int
        Rotational symmetry number.
    electronic_degeneracy : int
        Degeneracy of the electronic ground state.

    Returns
    -------
    tuple of float
        ``(p0_pascal, q_internal)``: k_B T / Lambda_mol^3 in Pascal, and
        g_el * q_rot * q_vib (q_vib measured from the v = 0 level), so that
        mu_molecule = k_B T ln(p / p0) - k_B T ln(q_internal).
    """
    lam_m = thermal_wavelength_angstrom(T_kelvin, mass_molecule_amu) * 1.0e-10
    p0_pascal = BOLTZMANN_K_SI * T_kelvin / lam_m ** 3

    # homonuclear: reduced mass = m_atom / 2 = m_molecule / 4
    r_m = bond_length_angstrom * 1.0e-10
    reduced_mass_kg = (mass_molecule_amu * AMU) / 4.0
    I = reduced_mass_kg * r_m ** 2
    theta_rot = PLANCK_HBAR ** 2 / (2.0 * I * BOLTZMANN_K_SI)
    q_rot = T_kelvin / (symmetry_number * theta_rot)

    theta_vib = PLANCK_H * (vib_wavenumber_cm * SPEED_OF_LIGHT_CM_PER_S) / BOLTZMANN_K_SI
    q_vib = 1.0 / (1.0 - math.exp(-theta_vib / T_kelvin))
    return p0_pascal, electronic_degeneracy * q_rot * q_vib


def diatomic_gas_mu_eV(T_kelvin: float, p_pascal: float, mass_amu: float,
                        bond_length_angstrom: float, vib_wavenumber_cm: float,
                        symmetry_number: int = 2, electronic_degeneracy: int = 1) -> float:
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
    (H2: 87.6 K; O2: 2.08 K). Against NIST-JANAF at 1 bar it is within
    ~7 meV per H2 and ~2 meV per O2 (with ``electronic_degeneracy=3``) at
    300-1000 K.

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
    electronic_degeneracy : int, optional
        Degeneracy of the electronic ground state: 1 for closed-shell
        molecules (H2, N2, Cl2), 3 for the triplet ground state of O2.
        Default is 1.

    Returns
    -------
    float
        The molecule's chemical potential, in eV.
    """
    p0_pascal, q_int = _diatomic_p0_and_q_internal(
        T_kelvin, mass_amu, bond_length_angstrom, vib_wavenumber_cm, symmetry_number,
        electronic_degeneracy)
    T_eV = kelvin_to_eV(T_kelvin)
    return T_eV * math.log(p_pascal / p0_pascal) - T_eV * math.log(q_int)


def mu_atomic_from_diatomic_reservoir_eV(T_kelvin: float, p_pascal: float, D0_eV: float,
                                          mass_molecule_amu: float, bond_length_angstrom: float,
                                          vib_wavenumber_cm: float, symmetry_number: int = 2,
                                          *, e_atom_eV: float,
                                          electronic_degeneracy: int = 1) -> float:
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
    electronic_degeneracy : int, optional
        Degeneracy of the molecule's electronic ground state (keyword-only;
        3 for O2). Default is 1.

    Returns
    -------
    float
        The atomic species' chemical potential, in eV, on the calculator's
        energy scale.
    """
    mu_molecule = diatomic_gas_mu_eV(T_kelvin, p_pascal, mass_molecule_amu,
                                      bond_length_angstrom, vib_wavenumber_cm, symmetry_number,
                                      electronic_degeneracy)
    return 0.5 * (mu_molecule - D0_eV) + e_atom_eV


def pressure_from_atomic_mu_reservoir_pascal(T_kelvin: float, mu_atom_eV: float, D0_eV: float,
                                              mass_molecule_amu: float, bond_length_angstrom: float,
                                              vib_wavenumber_cm: float, symmetry_number: int = 2,
                                              *, e_atom_eV: float,
                                              electronic_degeneracy: int = 1) -> float:
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
    electronic_degeneracy : int, optional
        Degeneracy of the molecule's electronic ground state (keyword-only;
        3 for O2). Default is 1.

    Returns
    -------
    float
        The reservoir gas pressure, in Pascal.
    """
    p0_pascal, q_int = _diatomic_p0_and_q_internal(
        T_kelvin, mass_molecule_amu, bond_length_angstrom, vib_wavenumber_cm, symmetry_number,
        electronic_degeneracy)
    mu_molecule = 2.0 * (mu_atom_eV - e_atom_eV) + D0_eV
    return p0_pascal * math.exp(mu_molecule / kelvin_to_eV(T_kelvin)) * q_int


def _pressure_unit(unit: Optional[str]) -> str:
    """
    The pressure unit to use; warns when none was given.

    Parameters
    ----------
    unit : str or None
        The unit the caller gave.

    Returns
    -------
    str
        `unit`, or ``'Pa'`` (with a FutureWarning) for None.
    """
    if unit is None:
        warnings.warn("no pressure unit given: assuming Pa. Pass unit= ('Pa', 'bar', 'atm', ...) "
                      "explicitly; the implicit default will be removed", FutureWarning,
                      stacklevel=3)
        return "Pa"
    return unit


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
                unit: Optional[str] = None) -> np.ndarray:
        """
        Chemical potentials of a (temperature x pressure) replica ladder.

        Parameters
        ----------
        temperatures_K : sequence of float
            Temperatures, in Kelvin.
        pressures : sequence of float
            Pressures, in `unit`.
        unit : str, optional
            Pressure unit (see `regc.units.PRESSURE_UNITS`). Give it
            explicitly: the default None means Pa, with a FutureWarning
            (bar is the runner's default, so values in bar would otherwise
            be read as Pa, a factor 1e5).

        Returns
        -------
        numpy.ndarray
            Shape (len(temperatures_K), len(pressures)): mu in eV, ready for
            `ReplicaExchangeGCMC`'s 2D ``mus`` argument.
        """
        f = pascal_per(_pressure_unit(unit))
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

    def mu_at_T_eV(self, T_eV: float, p: float, unit: Optional[str] = None) -> float:
        """
        `mu_eV` with the engine's temperature unit (k_B*T in eV).

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        p : float
            Pressure, in `unit`.
        unit : str, optional
            Pressure unit. Default None: Pa, with a FutureWarning.

        Returns
        -------
        float
            mu, in eV.
        """
        return self.mu_eV(eV_to_kelvin(T_eV), p * pascal_per(_pressure_unit(unit)))

    def pressure_at_T_eV(self, T_eV: float, mu_eV: float, unit: Optional[str] = None) -> float:
        """
        `pressure_pascal` with the engine's temperature unit, in `unit`.

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        mu_eV : float
            Chemical potential, in eV.
        unit : str, optional
            Pressure unit of the result. Default None: Pa, with a
            FutureWarning.

        Returns
        -------
        float
            Pressure, in `unit`.
        """
        return self.pressure_pascal(eV_to_kelvin(T_eV), mu_eV) / pascal_per(_pressure_unit(unit))

    def analysis_functions(self, unit: Optional[str] = None) -> Tuple[Callable, Callable]:
        """
        The ``(mu_of_Tp, p_of_Tmu)`` pair for `MBARSurfaceAnalysis`.

        Parameters
        ----------
        unit : str, optional
            Pressure unit the analysis will be given/return. Default None:
            Pa, with a FutureWarning.

        Returns
        -------
        tuple of callable
            ``mu_of_Tp(T_eV, p)`` and ``p_of_Tmu(T_eV, mu)`` (picklable).
        """
        unit = _pressure_unit(unit)
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
class BulkSolidReservoir(Reservoir):
    """
    The bulk crystal of the exchanged element as its reservoir (e.g. bulk Pd).

    Removing a surface metal atom puts it back into the bulk, as when a
    surface oxide forms that holds fewer metal atoms than the bare surface
    (Pd(100)-(sqrt5 x sqrt5)R27-O has one Pd fewer per cell; Rogal, Reuter,
    Scheffler, PRB 75, 205433 (2007)). The chemical potential is the free
    energy per atom of the classical harmonic crystal, independent of the
    gas pressure:

        mu(T) = e_bulk + 3 k_B T ln(hbar omega_g / k_B T),

    omega_g the geometric mean of its phonon frequencies. Classical, like
    the sampled atoms (whose momenta give the Lambda^3 of the GC weight), so
    the classical momentum and vibrational terms of reservoir and surface
    are counted the same way; pV of the solid is neglected.

    Parameters
    ----------
    mass_amu : float
        Atomic mass, in amu.
    e_bulk_eV : float
        Potential energy per atom of the relaxed crystal, in eV (same
        potential and energy zero as the simulation).
    mean_wavenumber_cm : float
        Geometric-mean phonon wavenumber, in cm^-1 (`from_calculator`
        computes it from the potential's Hessian).
    info : dict, optional
        Diagnostics (crystal, lattice constant, supercell). Default empty.
    """

    mass_amu: float
    e_bulk_eV: float
    mean_wavenumber_cm: float
    info: dict = field(default_factory=dict, compare=False)

    @property
    def mass_exchanged_amu(self) -> float:
        """float: The atom's mass."""
        return self.mass_amu

    def mu_eV(self, T_kelvin: float, p_pascal: float = 0.0) -> float:
        """
        Chemical potential of an atom in the bulk crystal at T.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float, optional
            Ignored (a solid reservoir has no gas pressure). Default 0.

        Returns
        -------
        float
            mu, in eV.
        """
        kT = kelvin_to_eV(T_kelvin)
        hw = (PLANCK_H * SPEED_OF_LIGHT_CM_PER_S * self.mean_wavenumber_cm) / EV_IN_JOULES
        return self.e_bulk_eV + 3.0 * kT * math.log(hw / kT)

    def pressure_pascal(self, T_kelvin: float, mu_eV: float) -> float:
        """
        Not defined: a solid reservoir has no pressure.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        mu_eV : float
            Chemical potential, in eV.

        Returns
        -------
        float
            Never returns.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError("a bulk-solid reservoir has a fixed mu(T), no pressure")

    @classmethod
    def from_calculator(cls, ase_calculator, symbol: str, crystal: str = "fcc",
                        a: Optional[float] = None, supercell: int = 2, delta: float = 0.01,
                        mass_amu: Optional[float] = None) -> "BulkSolidReservoir":
        """
        Energy per atom and mean phonon frequency of the bulk crystal, from a potential.

        Builds the cubic cell of `crystal` repeated `supercell` times, takes
        its energy per atom, and the Gamma-point phonons of that supercell
        from a finite-difference Hessian (3 translations dropped). A 2x2x2
        cubic fcc cell (32 atoms) gives the geometric-mean frequency to a
        few per cent, i.e. mu to about 3 kT x 0.03.

        Parameters
        ----------
        ase_calculator : ase.calculators.calculator.Calculator
            The potential.
        symbol : str
            The element.
        crystal : str, optional
            ``ase.build.bulk`` crystal structure. Default ``'fcc'``.
        a : float, optional
            Lattice constant (A); default None: ASE's tabulated one. Pass the
            slab's own (the potential's equilibrium) lattice constant.
        supercell : int, optional
            Repetitions of the cubic cell. Default 2.
        delta : float, optional
            Finite-difference displacement, A. Default 0.01.
        mass_amu : float, optional
            Atomic mass; default None: ASE's.

        Returns
        -------
        BulkSolidReservoir
            The reservoir.

        Raises
        ------
        ValueError
            If the crystal has imaginary modes at this lattice constant.
        """
        from ase.build import bulk
        atoms = bulk(symbol, crystal, a=a, cubic=True).repeat(int(supercell))
        if mass_amu is not None:
            atoms.set_masses([float(mass_amu)] * len(atoms))
        atoms.calc = ase_calculator
        n = len(atoms)
        e_bulk = float(atoms.get_potential_energy()) / n
        x0 = atoms.get_positions()
        hess = np.zeros((3 * n, 3 * n))
        for i in range(n):
            for k in range(3):
                f = []
                for sgn in (1.0, -1.0):
                    x = x0.copy()
                    x[i, k] += sgn * delta
                    atoms.set_positions(x)
                    f.append(atoms.get_forces().ravel())
                hess[3 * i + k] = -(f[0] - f[1]) / (2.0 * delta)
        atoms.set_positions(x0)
        hess = 0.5 * (hess + hess.T)
        m3 = np.repeat(atoms.get_masses(), 3)
        lam = np.sort(np.linalg.eigvalsh(hess / np.sqrt(np.outer(m3, m3))))
        modes = lam[np.argsort(np.abs(lam))][3:]                 # drop the 3 translations
        if np.any(modes <= 0.0):
            raise ValueError(f"bulk {symbol} ({crystal}, a = {a}) has imaginary modes with this "
                             f"potential: not a minimum at this lattice constant")
        omega = np.sqrt(modes * EV_IN_JOULES / (1.0e-20 * AMU))
        wn = omega / (2.0 * math.pi * SPEED_OF_LIGHT_CM_PER_S)
        mean_wn = float(np.exp(np.mean(np.log(wn))))
        info = {"crystal": crystal, "a_angstrom": None if a is None else float(a),
                "supercell": int(supercell), "n_atoms": n,
                "max_wavenumber_cm": float(wn.max()), "min_wavenumber_cm": float(wn.min())}
        return cls(float(atoms.get_masses()[0]), e_bulk, mean_wn, info)


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
    electronic_degeneracy : int, optional
        Degeneracy of the molecule's electronic ground state (3 for O2).
        Default 1; `from_calculator` picks it from the element.
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
    electronic_degeneracy: int = 1
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
            self.vib_wavenumber_cm, self.symmetry_number, e_atom_eV=self.e_atom_eV,
            electronic_degeneracy=self.electronic_degeneracy)

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
            self.vib_wavenumber_cm, self.symmetry_number, e_atom_eV=self.e_atom_eV,
            electronic_degeneracy=self.electronic_degeneracy)

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
                        r_scan: Optional[np.ndarray] = None,
                        electronic_degeneracy: Optional[int] = None) -> "DiatomicReservoir":
        """
        A reservoir consistent with a given potential.

        Evaluates the potential's own molecule and isolated atom
        (`diatomic_dissociation_energy_eV`): e_atom always comes from the
        calculator; D0 does too unless `d0` says otherwise.

        Parameters
        ----------
        ase_calculator : ase.calculators.calculator.Calculator
            The potential the simulation uses (e.g. ASE's EAM, or
            `regc.lammps_backend.lammps_ase_calculator`).
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
        electronic_degeneracy : int, optional
            Electronic ground-state degeneracy of the molecule. Default None:
            `default_electronic_degeneracy` (3 for O2, 1 otherwise).

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
        if electronic_degeneracy is None:
            electronic_degeneracy = default_electronic_degeneracy(symbol)
        return cls(D0, info["e_atom_eV"], mass_molecule_amu, bond_length_angstrom,
                   vib_wavenumber_cm, symmetry_number, int(electronic_degeneracy),
                   calculator_info=info)


# =============================================================================
# Polyatomic gases and the elemental chemical potentials of a gas feed
# =============================================================================
#: Rotational symmetry numbers of common gas molecules (point-group order of rotations).
SYMMETRY_NUMBERS = {"H2": 2, "N2": 2, "O2": 2, "F2": 2, "Cl2": 2, "CO": 1, "NO": 1, "HCl": 1,
                    "HF": 1, "CO2": 2, "H2O": 2, "H2S": 2, "SO2": 2, "NO2": 2, "N2O": 1,
                    "NH3": 3, "CH4": 12, "C2H2": 2, "C2H4": 4, "C2H6": 6}
#: Spin multiplicity (electronic ground-state degeneracy) of open-shell gas molecules.
SPIN_MULTIPLICITIES = {"O2": 3, "NO": 2, "NO2": 2}


#: Vibration models of `MolecularGas`: 'classical' matches the sampler, 'quantum' is NIST-JANAF.
VIBRATION_MODELS = ("classical", "quantum")


def check_vibrations(vibrations: str) -> str:
    """
    Validate a vibration model name.

    Parameters
    ----------
    vibrations : str
        ``'classical'`` or ``'quantum'``.

    Returns
    -------
    str
        The name, lower case.

    Raises
    ------
    ValueError
        If it is neither.
    """
    v = str(vibrations).strip().lower()
    if v not in VIBRATION_MODELS:
        raise ValueError(f"vibrations must be one of {list(VIBRATION_MODELS)}, not {vibrations!r}")
    return v


def molecular_gas_mu_eV(T_kelvin: float, p_pascal: float, energy_eV: float, mass_amu: float,
                        moments_amu_A2: Sequence[float], vib_wavenumbers_cm: Sequence[float],
                        symmetry_number: int = 1, spin_multiplicity: int = 1,
                        vibrations: str = "classical") -> float:
    """
    Chemical potential of an ideal polyatomic gas on a potential's energy scale.

    mu = E + E_0 + k_B T ln(p / p0) - k_B T ln(g q_rot q_vib), with p0 =
    k_B T / Lambda^3 of the molecule and a classical rigid rotor (linear or
    not). With ``vibrations='classical'`` (the reference consistent with
    sampling the internal coordinates classically, as the molecular moves
    do) q_vib = prod_i k_B T / (h nu_i) from the potential minimum, E_0 = 0
    and g = 1: an ideal gas of flexible molecules sampled with exp(-beta U)
    at this mu has exactly the density p / k_B T (harmonic, rigid-rotor
    level). With ``vibrations='quantum'`` q_vib = prod_i 1/(1 - exp(-h nu_i /
    k_B T)), E_0 = ZPE and g = `spin_multiplicity` (the physical gas,
    NIST-JANAF); for a diatomic this is exactly `diatomic_gas_mu_eV` shifted
    by E + ZPE = 2 e_atom - D0. The two differ by k_B T ln R, R = prod_i
    [2 sinh(h nu_i / 2 k_B T) / (h nu_i / k_B T)] / g.

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    p_pascal : float
        Partial pressure of the gas, in Pascal.
    energy_eV : float
        Energy of the relaxed molecule with the potential, in eV.
    mass_amu : float
        Molecular mass, in amu.
    moments_amu_A2 : sequence of float
        Principal moments of inertia, amu A^2 (all ~0 for an atom, one ~0
        for a linear molecule).
    vib_wavenumbers_cm : sequence of float
        Vibrational wavenumbers, cm^-1 (3n - 5 or 3n - 6 of them).
    symmetry_number : int, optional
        Rotational symmetry number. Default 1.
    spin_multiplicity : int, optional
        Electronic ground-state degeneracy (used by ``'quantum'`` only).
        Default 1.
    vibrations : str, optional
        ``'classical'`` (default) or ``'quantum'``.

    Returns
    -------
    float
        mu of the molecule, in eV.
    """
    p0, q_int, e0 = _molecular_p0_q_zpe(T_kelvin, mass_amu, moments_amu_A2, vib_wavenumbers_cm,
                                       symmetry_number, spin_multiplicity, vibrations)
    return energy_eV + e0 + kelvin_to_eV(T_kelvin) * (math.log(p_pascal / p0) - math.log(q_int))


def _molecular_p0_q_zpe(T_kelvin: float, mass_amu: float, moments_amu_A2: Sequence[float],
                        vib_wavenumbers_cm: Sequence[float], symmetry_number: int,
                        spin_multiplicity: int, vibrations: str = "classical"
                        ) -> Tuple[float, float, float]:
    """
    Reference pressure, internal partition function and vibrational ground-state energy.

    Parameters
    ----------
    T_kelvin : float
        Temperature, in Kelvin.
    mass_amu : float
        Molecular mass, in amu.
    moments_amu_A2 : sequence of float
        Principal moments of inertia, amu A^2.
    vib_wavenumbers_cm : sequence of float
        Vibrational wavenumbers, cm^-1.
    symmetry_number : int
        Rotational symmetry number.
    spin_multiplicity : int
        Electronic ground-state degeneracy (``'quantum'`` only).
    vibrations : str, optional
        ``'classical'`` (default) or ``'quantum'``; see `molecular_gas_mu_eV`.

    Returns
    -------
    tuple of float
        ``(p0_pascal, q_internal, e0_eV)``; e0 is the ZPE for
        ``'quantum'`` and 0 for ``'classical'``.

    Raises
    ------
    ValueError
        If `vibrations` is unknown.
    """
    quantum = check_vibrations(vibrations) == "quantum"
    lam = thermal_wavelength_angstrom(T_kelvin, mass_amu) * 1.0e-10
    p0 = BOLTZMANN_K_SI * T_kelvin / lam ** 3
    moments = np.sort(np.asarray(moments_amu_A2, dtype=float))
    to_si = AMU * 1.0e-20
    q_rot = 1.0
    if moments[-1] > 1e-8:
        theta = PLANCK_HBAR ** 2 / (2.0 * np.maximum(moments, 1e-30) * to_si * BOLTZMANN_K_SI)
        if moments[0] < 1e-3 * moments[-1]:            # linear
            q_rot = T_kelvin / (symmetry_number * theta[-1])
        else:
            q_rot = (math.sqrt(math.pi) / symmetry_number
                     * math.sqrt(T_kelvin ** 3 / float(np.prod(theta))))
    q_vib, e0 = 1.0, 0.0
    for w in vib_wavenumbers_cm:
        theta_v = PLANCK_H * float(w) * SPEED_OF_LIGHT_CM_PER_S / BOLTZMANN_K_SI
        if quantum:
            q_vib /= 1.0 - math.exp(-theta_v / T_kelvin)
            e0 += 0.5 * PLANCK_H * float(w) * SPEED_OF_LIGHT_CM_PER_S / EV_IN_JOULES
        else:                                          # classical oscillator from the minimum
            q_vib *= T_kelvin / theta_v
    # the sampled Hamiltonian has one electronic state: no degeneracy for 'classical'
    g = spin_multiplicity if quantum else 1
    return p0, g * q_rot * q_vib, e0


@dataclass
class MolecularGas(Reservoir):
    """
    An ideal polyatomic (or monatomic) gas, referenced to the simulation potential.

    The molecule's energy is that of its relaxed geometry with the same
    potential that evaluates the adsorbed atoms (the role D0 and e_atom
    play for `DiatomicReservoir`); translation, rotation, vibration and
    spin are standard ideal-gas statistical mechanics
    (`molecular_gas_mu_eV`). The relaxed template also defines the shape
    of the molecular insertion move (`regc.molecules.MoleculeMove`).

    The default ``vibrations='classical'`` is the reference consistent with
    the sampler, which treats the internal coordinates of molecules (in the
    gas zone or adsorbed intact) classically with exp(-beta U) on a single
    electronic surface: classical vibrations from the minimum, no ZPE, no
    spin degeneracy. ``'quantum'`` (quantum harmonic vibrations, ZPE and
    `spin_multiplicity`) is the physical gas of NIST-JANAF; with it the
    sampled intact molecules are over-stabilised by k_B T ln R (R in
    `molecular_gas_mu_eV`; ~0.07 eV for CO, 0.5-0.7 eV for NH3 and CH4 at
    300 K).

    Parameters
    ----------
    name : str
        Label, e.g. ``'CO'``.
    symbols : sequence of str
        Element of every atom.
    positions_angstrom : sequence
        Relaxed positions, shape (n, 3), Angstrom.
    energy_eV : float
        Energy of the relaxed molecule with the potential, eV.
    vib_wavenumbers_cm : sequence of float
        Vibrational wavenumbers, cm^-1.
    symmetry_number : int
        Rotational symmetry number.
    spin_multiplicity : int, optional
        Electronic ground-state degeneracy (``'quantum'`` only). Default 1.
    masses_amu : sequence of float, optional
        Atomic masses. Default None: ASE's table.
    vibrations : str, optional
        ``'classical'`` (default) or ``'quantum'``; see above.
    calculator_info : dict, optional
        Diagnostics from `from_calculator`. Default None.
    """

    name: str
    symbols: Tuple[str, ...]
    positions_angstrom: Sequence
    energy_eV: float
    vib_wavenumbers_cm: Tuple[float, ...]
    symmetry_number: int
    spin_multiplicity: int = 1
    masses_amu: Optional[Tuple[float, ...]] = None
    vibrations: str = "classical"
    calculator_info: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self):
        """
        Normalise the containers and fill in the masses.

        Raises
        ------
        ValueError
            If symbols and positions or masses disagree in length, or
            `vibrations` is unknown.
        """
        from .units import default_mass_amu
        self.symbols = tuple(str(s) for s in self.symbols)
        self.positions_angstrom = np.asarray(self.positions_angstrom, dtype=float).reshape(-1, 3)
        if len(self.positions_angstrom) != len(self.symbols):
            raise ValueError(f"gas {self.name}: {len(self.symbols)} symbols for "
                             f"{len(self.positions_angstrom)} positions")
        self.vib_wavenumbers_cm = tuple(float(w) for w in self.vib_wavenumbers_cm)
        self.masses_amu = (tuple(default_mass_amu(s) for s in self.symbols)
                           if self.masses_amu is None else tuple(float(m) for m in self.masses_amu))
        if len(self.masses_amu) != len(self.symbols):
            raise ValueError(f"gas {self.name}: {len(self.masses_amu)} masses for "
                             f"{len(self.symbols)} atoms")
        self.symmetry_number = int(self.symmetry_number)
        self.spin_multiplicity = int(self.spin_multiplicity)
        self.vibrations = check_vibrations(self.vibrations)

    @property
    def composition(self) -> Dict[str, int]:
        """dict: Number of atoms of each element."""
        out: Dict[str, int] = {}
        for s in self.symbols:
            out[s] = out.get(s, 0) + 1
        return out

    @property
    def mass_exchanged_amu(self) -> float:
        """float: Molecular mass, amu."""
        return float(sum(self.masses_amu))

    def principal_moments(self) -> np.ndarray:
        """
        Principal moments of inertia of the relaxed geometry.

        Returns
        -------
        numpy.ndarray
            Three moments, ascending, amu A^2.
        """
        m = np.asarray(self.masses_amu)
        r = self.positions_angstrom - (m[:, None] * self.positions_angstrom).sum(0) / m.sum()
        tensor = np.einsum("i,ij,ik->jk", m, r, r)
        inertia = np.trace(tensor) * np.eye(3) - tensor
        return np.sort(np.linalg.eigvalsh(inertia))

    def mu_eV(self, T_kelvin: float, p_pascal: float) -> float:
        """
        Chemical potential of the molecule at (T, p).

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Partial pressure, in Pascal.

        Returns
        -------
        float
            mu, in eV on the potential's energy scale.
        """
        return molecular_gas_mu_eV(T_kelvin, p_pascal, self.energy_eV, self.mass_exchanged_amu,
                                   self.principal_moments(), self.vib_wavenumbers_cm,
                                   self.symmetry_number, self.spin_multiplicity, self.vibrations)

    def pressure_pascal(self, T_kelvin: float, mu_eV: float) -> float:
        """
        Partial pressure giving the molecular chemical potential `mu_eV`.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        mu_eV : float
            Chemical potential of the molecule, in eV.

        Returns
        -------
        float
            Pressure, in Pascal.
        """
        p0, q_int, e0 = _molecular_p0_q_zpe(T_kelvin, self.mass_exchanged_amu,
                                           self.principal_moments(), self.vib_wavenumbers_cm,
                                           self.symmetry_number, self.spin_multiplicity,
                                           self.vibrations)
        return p0 * q_int * math.exp((mu_eV - self.energy_eV - e0) / kelvin_to_eV(T_kelvin))

    def to_dict(self) -> dict:
        """
        JSON-able form (diagnostics dropped).

        Returns
        -------
        dict
            The constructor arguments, plus ``type: molecular``. A quantum
            gas omits ``vibrations``, as written before the option existed.
        """
        d = {"type": "molecular", "name": self.name, "symbols": list(self.symbols),
             "positions_angstrom": self.positions_angstrom.tolist(),
             "energy_eV": float(self.energy_eV),
             "vib_wavenumbers_cm": list(self.vib_wavenumbers_cm),
             "symmetry_number": self.symmetry_number,
             "spin_multiplicity": self.spin_multiplicity, "masses_amu": list(self.masses_amu)}
        if self.vibrations != "quantum":
            d["vibrations"] = self.vibrations
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MolecularGas":
        """
        Rebuild a gas written by `to_dict`.

        Parameters
        ----------
        d : dict
            The dict; without ``vibrations`` (older runs) the gas is quantum.

        Returns
        -------
        MolecularGas
            The gas.
        """
        d = {k: v for k, v in d.items() if k != "type"}
        d.setdefault("vibrations", "quantum")
        return cls(**d)

    @classmethod
    def from_calculator(cls, ase_calculator, template, name: Optional[str] = None,
                        symmetry_number: Optional[int] = None,
                        spin_multiplicity: Optional[int] = None,
                        vib_wavenumbers_cm: Optional[Sequence[float]] = None,
                        fmax: float = 1e-3, steps: int = 1000, delta: float = 0.005,
                        cell: float = 20.0, masses_amu=None,
                        vibrations: str = "classical") -> "MolecularGas":
        """
        Relax a gas molecule with a potential and take its vibrations from the Hessian.

        Parameters
        ----------
        ase_calculator : ase.calculators.calculator.Calculator
            The simulation's potential, as an ASE calculator (forces needed).
        template : str or ase.Atoms
            A name of `ase.build.molecule` (``'CO'``, ``'CH4'``, ...) or a
            starting geometry.
        name : str, optional
            Label. Default None: the template name or chemical formula.
        symmetry_number : int, optional
            Default None: from `SYMMETRY_NUMBERS`.
        spin_multiplicity : int, optional
            Default None: from `SPIN_MULTIPLICITIES` (3 for O2), else 1.
        vib_wavenumbers_cm : sequence of float, optional
            Use these instead of the potential's (e.g. experimental ones).
            Default None.
        fmax : float, optional
            Force tolerance of the relaxation, eV/A. Default 1e-3.
        steps : int, optional
            Maximum relaxation steps. Default 1000.
        delta : float, optional
            Finite-difference displacement for the Hessian, A. Default 0.005.
        cell : float, optional
            Edge of the (non-periodic) box the molecule sits in, A. Default 20.
        masses_amu : dict or sequence of float, optional
            ``{element: mass}`` (elements not in it keep the template's)
            or one mass per atom, amu; give the exchanged atoms' masses
            (isotopes such as D). They enter Lambda, the moments of inertia
            and the Hessian frequencies. Default None: the template's
            (ASE's table).
        vibrations : str, optional
            ``'classical'`` (default) or ``'quantum'``; see the class.

        Returns
        -------
        MolecularGas
            The gas, with ``calculator_info`` (relaxed energy, all
            Hessian frequencies, the largest residual force).

        Raises
        ------
        ValueError
            If the symmetry number is unknown and not given, the relaxed
            geometry is not a minimum (imaginary vibrational modes), or
            `masses_amu` does not fit the template.
        """
        from ase.build import molecule
        from ase.optimize import BFGS

        if isinstance(template, str):
            label = template
            atoms = molecule(template)
        else:
            atoms = template.copy()
            label = atoms.get_chemical_formula()
        name = name or label
        if masses_amu is not None:
            if isinstance(masses_amu, dict):
                masses_amu = [masses_amu.get(s, m) for s, m in
                              zip(atoms.get_chemical_symbols(), atoms.get_masses())]
            if len(masses_amu) != len(atoms):
                raise ValueError(f"gas {name}: {len(masses_amu)} masses for {len(atoms)} atoms")
            atoms.set_masses([float(m) for m in masses_amu])
        atoms.set_cell([cell] * 3)
        atoms.set_pbc(False)
        atoms.center()
        atoms.calc = ase_calculator
        if len(atoms) > 1:
            BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        energy = float(atoms.get_potential_energy())
        if symmetry_number is None:
            if len(atoms) == 1:
                symmetry_number = 1
            elif name in SYMMETRY_NUMBERS or label in SYMMETRY_NUMBERS:
                symmetry_number = SYMMETRY_NUMBERS.get(name, SYMMETRY_NUMBERS.get(label))
            else:
                raise ValueError(f"no tabulated symmetry number for {name}: give symmetry_number")
        if spin_multiplicity is None:
            spin_multiplicity = SPIN_MULTIPLICITIES.get(name, SPIN_MULTIPLICITIES.get(label, 1))
        masses = atoms.get_masses()
        info = {"energy_eV": energy,
                "max_force_eV_per_A": float(np.abs(atoms.get_forces()).max())}
        if len(atoms) > 1:
            n = len(atoms)
            hess = np.zeros((3 * n, 3 * n))
            x0 = atoms.get_positions()
            for i in range(n):
                for a in range(3):
                    f = []
                    for sgn in (1.0, -1.0):
                        x = x0.copy()
                        x[i, a] += sgn * delta
                        atoms.set_positions(x)
                        f.append(atoms.get_forces().ravel())
                    hess[3 * i + a] = -(f[0] - f[1]) / (2.0 * delta)
            atoms.set_positions(x0)
            hess = 0.5 * (hess + hess.T)
            m3 = np.repeat(masses, 3)
            lam = np.linalg.eigvalsh(hess / np.sqrt(np.outer(m3, m3)))
            omega = np.sign(lam) * np.sqrt(np.abs(lam) * EV_IN_JOULES / (1.0e-20 * AMU))
            wn = omega / (2.0 * math.pi * SPEED_OF_LIGHT_CM_PER_S)
            gas = cls(name, atoms.get_chemical_symbols(), x0, energy, (), symmetry_number,
                      spin_multiplicity, tuple(masses), vibrations)
            linear = gas.principal_moments()[0] < 1e-3 * gas.principal_moments()[-1]
            n_rigid = 5 if linear else 6
            modes = wn[np.argsort(np.abs(wn))][n_rigid:]
            info["all_wavenumbers_cm"] = wn.tolist()
            if vib_wavenumbers_cm is None:
                if np.any(modes <= 0.0):
                    raise ValueError(f"{name} relaxed with this potential is not a minimum: "
                                     f"vibrational modes {np.round(np.sort(modes), 1).tolist()} "
                                     f"cm^-1 (negative = imaginary); give vib_wavenumbers_cm or "
                                     f"another template")
                vib_wavenumbers_cm = sorted(modes.tolist())
        else:
            x0 = atoms.get_positions()
            vib_wavenumbers_cm = ()
        return cls(name, atoms.get_chemical_symbols(), x0, energy, tuple(vib_wavenumbers_cm),
                   int(symmetry_number), int(spin_multiplicity), tuple(masses),
                   vibrations, calculator_info=info)


class MolecularGasFeed:
    """
    A gas mixture at fixed mole fractions, as elemental chemical potentials.

    The simulation exchanges atoms with chemical potentials mu_e (one per
    element); a gas g of composition nu_g fixes nu_g . mu = mu_g(T, x_g p).
    With one gas per independent composition this is the constrained
    equilibrium of atomistic thermodynamics (Reuter and Scheffler): the
    surface is in equilibrium with every gas, the gases not necessarily
    with each other. The mu_e are the minimum-norm solution; where fewer
    gases than elements leave them underdetermined (pure CO fixes only
    mu_C + mu_O), only combinations that the moves use -- checked by
    `check_moves` -- enter the sampling, and those are unique.

    Parameters
    ----------
    elements : sequence of str
        The exchanged elements, in `ParticleSystem.exchange_species` order.
    gases : sequence of MolecularGas
        The gases.
    fractions : sequence of float
        Their mole fractions (normalised here).
    element_masses_amu : dict, optional
        Atomic masses for Lambda_e. Default None: ASE's table. They should
        be the gases' atom masses: the activities prod_e z_e^{nu_e} =
        exp(beta mu_g) / prod_e Lambda_e^{3 nu_e} of a molecular move put a
        classical gas at its pressure only then (a mismatch is warned about).

    Raises
    ------
    ValueError
        If a gas contains an element that is not exchanged, or the gas
        compositions are linearly dependent (e.g. CO, O2 and CO2: their
        chemical potentials agree only at gas-phase equilibrium).
    """

    def __init__(self, elements: Sequence[str], gases: Sequence[MolecularGas],
                 fractions: Sequence[float], element_masses_amu: Optional[Dict[str, float]] = None):
        """
        Build the composition matrix and its pseudo-inverse.

        Parameters
        ----------
        elements, gases, fractions, element_masses_amu
            See the class docstring.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        from .units import default_mass_amu
        self.species = [str(e) for e in elements]
        self.gases = list(gases)
        x = np.asarray(fractions, dtype=float)
        if len(x) != len(self.gases) or np.any(x <= 0):
            raise ValueError("one positive mole fraction per gas is needed")
        self.fractions = list(x / x.sum())
        for g in self.gases:
            foreign = sorted(set(g.composition) - set(self.species))
            if foreign:
                raise ValueError(f"gas {g.name} contains {foreign}, which are not exchanged "
                                 f"species {self.species}")
        self.A = np.array([[g.composition.get(e, 0) for e in self.species] for g in self.gases],
                          dtype=float)
        if np.linalg.matrix_rank(self.A) < len(self.gases):
            raise ValueError(
                f"the gas compositions {[g.name for g in self.gases]} are linearly dependent: a "
                f"fixed feed of them has no common elemental chemical potentials unless the gases "
                f"are in mutual equilibrium (e.g. CO + O2 <-> CO2). Keep one gas per independent "
                f"composition (constrained equilibrium)")
        self.pinv = np.linalg.pinv(self.A)
        masses = dict(element_masses_amu or {})
        self.element_masses_amu = {e: float(masses.get(e, default_mass_amu(e)))
                                   for e in self.species}
        # z_e = exp(beta mu_e) / Lambda_e^3 reproduces a gas only if its atoms have these masses
        for g in self.gases:
            off = sorted({s for s, m in zip(g.symbols, g.masses_amu)
                          if abs(m - self.element_masses_amu[s]) > 1e-6 * m})
            if off:
                warnings.warn(f"gas {g.name}: masses of {off} differ from the exchanged atoms' "
                              f"{ {s: self.element_masses_amu[s] for s in off} }; the sampled gas "
                              f"is then not at the reservoir pressure", stacklevel=2)

    @property
    def reservoirs(self) -> List[MolecularGas]:
        """list of MolecularGas: The gases (the name `GasFeed` uses)."""
        return self.gases

    def determines(self, composition: Dict[str, int]) -> bool:
        """
        Whether the feed fixes the chemical potential of a composition uniquely.

        Parameters
        ----------
        composition : dict
            ``{element: count}`` of a move (one atom, or a molecule).

        Returns
        -------
        bool
            True if the composition is a combination of the gases'.
        """
        v = np.array([composition.get(e, 0) for e in self.species], dtype=float)
        return bool(np.allclose(v @ self.pinv @ self.A, v, atol=1e-9))

    def check_moves(self, compositions: Dict[str, Dict[str, int]]) -> None:
        """
        Check that every exchange move's chemical potential is fixed by the gases.

        Parameters
        ----------
        compositions : dict
            ``{move label: {element: count}}``.

        Raises
        ------
        ValueError
            Naming the moves whose chemical potential the feed leaves
            undetermined (e.g. atomic C with a pure CO feed).
        """
        bad = [k for k, c in compositions.items() if not self.determines(c)]
        if bad:
            raise ValueError(f"the gas feed {[g.name for g in self.gases]} does not fix the "
                             f"chemical potential of the moves {bad}: add a gas containing them, "
                             f"or drop those moves")

    def gas_mus_eV(self, T_kelvin: float, p_pascal: float) -> np.ndarray:
        """
        Chemical potential of every gas at its partial pressure.

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Total pressure, in Pascal.

        Returns
        -------
        numpy.ndarray
            mu_g, one per gas, eV.
        """
        return np.array([g.mu_eV(T_kelvin, x * p_pascal)
                         for g, x in zip(self.gases, self.fractions)])

    def mu_elements_eV(self, T_kelvin: float, p_pascal: float) -> np.ndarray:
        """
        Elemental chemical potentials at (T, total p).

        Parameters
        ----------
        T_kelvin : float
            Temperature, in Kelvin.
        p_pascal : float
            Total pressure, in Pascal.

        Returns
        -------
        numpy.ndarray
            mu_e, one per element (minimum-norm solution), eV.
        """
        return self.pinv @ self.gas_mus_eV(T_kelvin, p_pascal)

    def mu_at_T_eV(self, T_eV: float, p: float, unit: str = "Pa"):
        """
        Elemental chemical potential(s) at temperature k_B*T and total pressure p.

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        p : float
            Total pressure, in `unit`.
        unit : str, optional
            Pressure unit. Default ``'Pa'``.

        Returns
        -------
        float or numpy.ndarray
            mu (eV) for one element, an array of one per element otherwise.
        """
        mu = self.mu_elements_eV(eV_to_kelvin(T_eV), p * pascal_per(unit))
        return float(mu[0]) if len(mu) == 1 else mu

    def mu_grid(self, temperatures_K: Sequence[float], pressures: Sequence[float],
                unit: str = "Pa") -> np.ndarray:
        """
        Elemental chemical potentials of a (temperature x pressure) ladder.

        Parameters
        ----------
        temperatures_K : sequence of float
            Temperatures, in Kelvin.
        pressures : sequence of float
            Total pressures, in `unit`.
        unit : str, optional
            Pressure unit. Default ``'Pa'``.

        Returns
        -------
        numpy.ndarray
            Shape (n_T, n_p) for one element, (n_T, n_p, n_elements) otherwise.
        """
        f = pascal_per(unit)
        grid = np.array([[self.mu_elements_eV(T, p * f) for p in pressures]
                         for T in temperatures_K])
        return grid[..., 0] if len(self.species) == 1 else grid

    def lambda_cubed_of_T(self):
        """
        Lambda^3(T) of every exchanged element (atoms are the sampled particles).

        Returns
        -------
        ThermalWavelengthCubed or SpeciesLambdaCubed
            For `ReplicaExchangeGCMC`'s ``lambda_cubed_of_T``.
        """
        from .units import SpeciesLambdaCubed
        lams = [ThermalWavelengthCubed(self.element_masses_amu[e]) for e in self.species]
        return lams[0] if len(lams) == 1 else SpeciesLambdaCubed(lams)

    def to_dict(self) -> dict:
        """
        JSON-able form, for ``run_metadata.json``.

        Returns
        -------
        dict
            Elements, gases, fractions and element masses.
        """
        return {"type": "molecular_feed", "elements": self.species,
                "gases": [g.to_dict() for g in self.gases], "fractions": self.fractions,
                "element_masses_amu": self.element_masses_amu}

    @classmethod
    def from_dict(cls, d: dict) -> "MolecularGasFeed":
        """
        Rebuild a feed written by `to_dict`.

        Parameters
        ----------
        d : dict
            The dict.

        Returns
        -------
        MolecularGasFeed
            The feed.
        """
        return cls(d["elements"], [MolecularGas.from_dict(g) for g in d["gases"]],
                   d["fractions"], d.get("element_masses_amu"))
