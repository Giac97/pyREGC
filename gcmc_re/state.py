"""A single (temperature, chemical potential) grid point's thermodynamic label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ThermoState:
    """
    The thermodynamic label attached to one replica-exchange grid point.

    Stays fixed to its grid slot; it is the *configuration* that gets
    swapped between slots during replica exchange (see `Replica` in
    `replica.py`).

    Attributes
    ----------
    temperature : float
        Temperature, in the same energy-like units as the Calculator's
        reported energies (e.g. eV for EMT/EAM/MLIP calculators, or
        reduced units for `LJCalculator`).
    mu : float
        Chemical potential, in the same units as `temperature`.
    label : tuple of int
        The (temperature_index, mu_index) position of this state in the
        replica-exchange grid.
    lambda_cubed : float, optional
        Thermal de Broglie wavelength cubed, in whatever length units the
        system's volume/positions use. Default is 1.0, the correct,
        self-consistent choice in reduced Lennard-Jones units (it's
        equivalent to defining the LJ length unit sigma as the wavelength
        scale too) -- but it is NOT generally correct when a Calculator
        works in real physical units (e.g. ASE's EMT or a real EAM/MLIP,
        which report energies in eV using positions in Angstrom). There,
        supply the real value (see
        `gcmc_re.units.thermal_wavelength_cubed_angstrom3`) so the
        insertion/deletion acceptance formulas in `replica.py` stay
        dimensionally correct.
    """

    temperature: float
    mu: float
    label: Tuple[int, int]
    lambda_cubed: float = 1.0

    @property
    def beta(self) -> float:
        """
        Inverse temperature.

        Returns
        -------
        float
            1 / `temperature`.
        """
        return 1.0 / self.temperature
