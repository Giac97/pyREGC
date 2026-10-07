"""
Unit-conversion helpers for driving this package's GCMC engine with a
Calculator that works in real physical units (eV, Angstrom, Kelvin) --
ASE's EMT, a real EAM potential, or a trained MLIP -- rather than the
reduced Lennard-Jones units (sigma = epsilon = 1) used in the LJ examples.

Why this matters: `Replica`'s insertion/deletion acceptance formulas
contain a term log(V / ((N+1) * Lambda^3)) that reduces to log(V/(N+1))
only when the thermal de Broglie wavelength Lambda is taken as exactly 1
length unit. That is a harmless, self-consistent convention in reduced LJ
units (equivalent to defining sigma as the wavelength scale too) but is
NOT correct in real units: a real particle's Lambda depends on its mass and
the temperature, and for a light atom like H at typical temperatures it is
generally well under 1 Angstrom. Use `ThermoState.lambda_cubed` (via
`ReplicaExchangeGCMC`'s `lambda_cubed_of_T` argument) to supply the correct
value; `thermal_wavelength_cubed_angstrom3` below computes it.

Similarly, `kelvin_to_eV` converts a real temperature into the eV energy
scale a Calculator like EMT reports energies in, since `ThermoState.beta =
1/temperature` needs `temperature` in the same units as the energies it
multiplies.
"""

from __future__ import annotations

import math

# CODATA 2018 values (the same ones ASE's own ase.units module uses)
PLANCK_H = 6.62607015e-34          # J s
BOLTZMANN_K_SI = 1.380649e-23      # J / K
AMU = 1.66053906660e-27            # kg
EV_IN_JOULES = 1.602176634e-19     # J

KB_EV_PER_KELVIN = BOLTZMANN_K_SI / EV_IN_JOULES  # ~8.617333262e-5 eV/K

#: Pascal per unit, for the pressure units accepted across the package.
PRESSURE_UNITS = {"Pa": 1.0, "bar": 1.0e5, "atm": 101325.0}


def kelvin_to_eV(T_kelvin: float) -> float:
    """
    Convert a real temperature to the eV energy scale.

    Computes k_B * T, in eV -- the natural energy-scale temperature for a
    Calculator (like EMT, or most EAM/MLIP potentials) that reports
    energies in eV. `ThermoState.beta = 1/temperature` needs `temperature`
    in the same units as the energies it multiplies, so a real Kelvin
    temperature must be converted via this function before use in real
    (rather than reduced Lennard-Jones) units.

    Parameters
    ----------
    T_kelvin : float
        Temperature in Kelvin.

    Returns
    -------
    float
        k_B * T_kelvin, in eV. At 300 K this is about 0.02585 eV (the
        textbook "kT ~ 25.7 meV at room temperature").

    See Also
    --------
    thermal_wavelength_angstrom : the companion conversion for
        `ThermoState.lambda_cubed`.
    """
    return KB_EV_PER_KELVIN * T_kelvin


def eV_to_kelvin(T_eV: float) -> float:
    """
    Convert an energy-scale temperature k_B*T (eV) back to Kelvin.

    The inverse of `kelvin_to_eV`, valid for any value (not only for the
    temperatures of a replica ladder).

    Parameters
    ----------
    T_eV : float
        k_B * T, in eV.

    Returns
    -------
    float
        The temperature in Kelvin.
    """
    return T_eV / KB_EV_PER_KELVIN


def pascal_per(unit: str) -> float:
    """
    Pascal per pressure unit.

    Parameters
    ----------
    unit : str
        One of the keys of `PRESSURE_UNITS` (``'Pa'``, ``'bar'``, ``'atm'``).

    Returns
    -------
    float
        How many Pascal one `unit` is.

    Raises
    ------
    ValueError
        If `unit` is not a known pressure unit.
    """
    try:
        return PRESSURE_UNITS[unit]
    except KeyError:
        raise ValueError(f"unknown pressure unit {unit!r}; use one of {sorted(PRESSURE_UNITS)}")


def thermal_wavelength_angstrom(T_kelvin: float, mass_amu: float) -> float:
    """
    Compute the 3D ideal-gas thermal de Broglie wavelength.

    Evaluates Lambda(T) = h / sqrt(2*pi*m*k_B*T), in Angstrom, for a
    particle of the given mass at the given temperature.

    Parameters
    ----------
    T_kelvin : float
        Temperature in Kelvin.
    mass_amu : float
        Particle mass in atomic mass units.

    Returns
    -------
    float
        The thermal de Broglie wavelength Lambda(T), in Angstrom.

    See Also
    --------
    thermal_wavelength_cubed_angstrom3 : Lambda(T) cubed, what
        `ThermoState.lambda_cubed` actually wants.
    """
    m_kg = mass_amu * AMU
    lam_m = PLANCK_H / math.sqrt(2.0 * math.pi * m_kg * BOLTZMANN_K_SI * T_kelvin)
    return lam_m * 1.0e10  # meters -> Angstrom


def thermal_wavelength_cubed_angstrom3(T_kelvin: float, mass_amu: float) -> float:
    """
    Compute the thermal de Broglie wavelength cubed, in cubic Angstrom.

    Parameters
    ----------
    T_kelvin : float
        Temperature in Kelvin.
    mass_amu : float
        Particle mass in atomic mass units.

    Returns
    -------
    float
        Lambda(T)^3, in Angstrom^3 -- what `ThermoState.lambda_cubed`
        wants when positions/volume are in Angstrom (e.g. for an
        ASE-backed Calculator).
    """
    return thermal_wavelength_angstrom(T_kelvin, mass_amu) ** 3


def default_mass_amu(species: str) -> float:
    """
    Look up a best-effort mass for a species symbol.

    Used by `ParticleSystem.masses()`/`substrate_masses()` for any species
    not given an explicit `mass_of_species` entry.

    Parameters
    ----------
    species : str
        A species label -- a real element symbol (e.g. ``'H'``, ``'Fe'``)
        or a generic reduced-units placeholder (e.g. ``'X'``).

    Returns
    -------
    float
        The real atomic mass in atomic mass units, looked up via ASE's
        data tables, if `species` is a recognized element symbol and ASE
        is installed. Otherwise 1.0 -- the reduced-units convention this
        package's LJ examples already assume, where mass=1 alongside
        sigma=epsilon=1.
    """
    try:
        from ase.data import atomic_masses, atomic_numbers
        return float(atomic_masses[atomic_numbers[species]])
    except (ImportError, KeyError):
        return 1.0


class ThermalWavelengthCubed:
    """
    Lambda^3 of a particle of given mass as a function of k_B*T in eV.

    The callable `ReplicaExchangeGCMC`'s and `MBARSurfaceAnalysis`'s
    ``lambda_cubed_of_T`` argument expects in real units: temperatures in
    this package are k_B*T in eV, lengths in Angstrom. Unlike an ad-hoc
    lambda it is picklable, valid at any temperature (MBAR evaluates it off
    the replica ladder), and records its mass, so saved samples can restore
    it (see `gcmc_re.io`).

    Parameters
    ----------
    mass_amu : float
        Mass of the exchanged particle, in atomic mass units.

    Examples
    --------
    >>> lam3 = ThermalWavelengthCubed(1.008)          # H atoms
    >>> round(lam3(kelvin_to_eV(300.0)), 4)           # Angstrom^3 at 300 K
    1.0119
    """

    def __init__(self, mass_amu: float):
        """
        Store the particle mass.

        Parameters
        ----------
        mass_amu : float
            Mass of the exchanged particle, in atomic mass units.
        """
        self.mass_amu = float(mass_amu)

    def __call__(self, T_eV: float) -> float:
        """
        Evaluate Lambda^3 at an energy-scale temperature.

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.

        Returns
        -------
        float
            Lambda(T)^3, in Angstrom^3.
        """
        return thermal_wavelength_cubed_angstrom3(eV_to_kelvin(T_eV), self.mass_amu)

    def __repr__(self) -> str:
        """
        Representation showing the mass.

        Returns
        -------
        str
            ``ThermalWavelengthCubed(mass_amu=...)``.
        """
        return f"ThermalWavelengthCubed(mass_amu={self.mass_amu})"
