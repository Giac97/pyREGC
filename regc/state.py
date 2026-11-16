"""A single (temperature, chemical potential) grid point's thermodynamic label."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class ThermoState:
    """
    The thermodynamic label attached to one replica-exchange grid point.

    Stays fixed to its grid slot; it is the *configuration* that gets
    swapped between slots during replica exchange (see `Replica` in
    `replica.py`).

    With several exchanged species (``system.exchange_species`` a list,
    e.g. ``['O', 'Cl']``), `mu` and `lambda_cubed` are arrays with one
    entry per species, in that order; with one species they are floats,
    exactly as before.

    Attributes
    ----------
    temperature : float
        Temperature, in the same energy-like units as the Calculator's
        reported energies (e.g. eV for EMT/EAM/MLIP calculators, or
        reduced units for `LJCalculator`).
    mu : float or numpy.ndarray
        Chemical potential, in the same units as `temperature`; one per
        exchanged species if there are several.
    label : tuple of int
        The (temperature_index, mu_index) position of this state in the
        replica-exchange grid.
    lambda_cubed : float or numpy.ndarray, optional
        Thermal de Broglie wavelength cubed, in whatever length units the
        system's volume/positions use (one per exchanged species if there
        are several). Default is 1.0, the correct,
        self-consistent choice in reduced Lennard-Jones units (it's
        equivalent to defining the LJ length unit sigma as the wavelength
        scale too) -- but it is NOT generally correct when a Calculator
        works in real physical units (e.g. ASE's EMT or a real EAM/MLIP,
        which report energies in eV using positions in Angstrom). There,
        supply the real value (see
        `regc.units.thermal_wavelength_cubed_angstrom3`) so the
        insertion/deletion acceptance formulas in `replica.py` stay
        dimensionally correct.
    sgc_dmu : numpy.ndarray, optional
        Semi-grand-canonical field of the substrate (see
        `ParticleSystem.sgc_species`): one chemical-potential difference
        mu_k - mu_ref per entry of ``system.sgc_species``, in the units of
        `temperature`; the first (reference) entry is 0. The weight of a
        configuration gains exp(beta * sum_k sgc_dmu[k] * N_k), N_k the
        number of swappable substrate atoms of species k. Configurational
        (mass-free): a T-independent value describes equilibrium with a
        bulk alloy whose excess free energy is T-independent (regular
        solution). Default None: no identity swaps.
    """

    temperature: float
    mu: Union[float, np.ndarray]
    label: Tuple[int, int]
    lambda_cubed: Union[float, np.ndarray] = 1.0
    sgc_dmu: Optional[np.ndarray] = None

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

    @property
    def mu_vec(self) -> np.ndarray:
        """numpy.ndarray: `mu` as a 1D array (length 1 for one species)."""
        return np.atleast_1d(np.asarray(self.mu, dtype=float))

    @property
    def lambda_cubed_vec(self) -> np.ndarray:
        """numpy.ndarray: `lambda_cubed` as a 1D array, broadcast to `mu_vec`."""
        return np.broadcast_to(np.asarray(self.lambda_cubed, dtype=float),
                               self.mu_vec.shape).copy()
