"""
Pressure <-> chemical potential:  mu(p, T) = k_B*T*ln(p/p0(T))

Classically, a GCMC insertion/deletion move only ever sees the *activity*
z = exp(beta*mu)/Lambda(T)^3 -- mu and the thermal wavelength Lambda always
appear in that one combination. `Replica`'s insertion/deletion moves already
assume Lambda = 1 throughout, which is exactly the same as fixing
p0(T) = k_B*T (with k_B = 1 in these reduced units): mu = T*ln(p/T), so
p/p0 = p/T = exp(mu/T) = z. That is the *consistent* default,
`p0_reduced_units`.

`p0_ideal_gas_wavelength` is provided for users who want p0(T) to track a
literal 3D ideal-gas thermal wavelength (p0 ~ T^{5/2} instead of T^1). This
does NOT require touching the Monte Carlo acceptance rules -- it is a
different (equally valid) convention for how mu is labeled in terms of p --
but pressures reported under the two conventions are on different scales
and shouldn't be compared to each other. If you are calibrating p0(T)
against a real reservoir's measured equation of state (e.g. mu_H2(p, T) for
a real H2 gas reservoir), supply your own `p0_of_T`.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import math

import numpy as np


def p0_reduced_units(T: float) -> float:
    """
    Reference pressure for the reduced-units (Lambda=1) convention.

    Parameters
    ----------
    T : float
        Temperature, in reduced units.

    Returns
    -------
    float
        p0(T) = T. Consistent with the Lambda=1 convention already built
        into this package's GCMC moves -- use this unless you have a
        specific reason to calibrate p0(T) differently.
    """
    return T


def p0_ideal_gas_wavelength(T: float, mass: float = 1.0, hbar: float = 1.0) -> float:
    """
    Reference pressure for a literal 3D ideal-gas thermal wavelength.

    Parameters
    ----------
    T : float
        Temperature, in reduced units.
    mass : float, optional
        Particle mass, in reduced units. Default is 1.0.
    hbar : float, optional
        Reduced Planck constant, in reduced units. Default is 1.0.

    Returns
    -------
    float
        p0(T) = T / Lambda(T)^3 with the literal 3D ideal-gas thermal de
        Broglie wavelength Lambda(T) = sqrt(2*pi*hbar^2/(mass*T)) (reduced
        units, k_B = 1). Gives p0(T) ~ T^{5/2}, in contrast to
        `p0_reduced_units`'s p0(T) ~ T^1.
    """
    Lambda3 = (2.0 * np.pi * hbar ** 2 / (mass * T)) ** 1.5
    return T / Lambda3


def mu_from_pressure(T: float, p: float,
                      p0_of_T: Callable[[float], float] = p0_reduced_units) -> float:
    """
    Convert a (temperature, pressure) pair to a chemical potential.

    Parameters
    ----------
    T : float
        Temperature.
    p : float
        Pressure, in whatever units `p0_of_T` is calibrated for.
    p0_of_T : callable, optional
        A function ``p0_of_T(T) -> float`` giving the reference pressure
        at temperature `T`. Default is `p0_reduced_units`.

    Returns
    -------
    float
        mu(p, T) = k_B*T*ln(p/p0(T)), with k_B = 1 in these reduced units.

    See Also
    --------
    pressure_from_mu : the inverse conversion.
    """
    return T * np.log(p / p0_of_T(T))


def pressure_from_mu(T: float, mu: float,
                      p0_of_T: Callable[[float], float] = p0_reduced_units) -> float:
    """
    Convert a (temperature, chemical potential) pair to a pressure.

    Parameters
    ----------
    T : float
        Temperature.
    mu : float
        Chemical potential.
    p0_of_T : callable, optional
        A function ``p0_of_T(T) -> float`` giving the reference pressure
        at temperature `T`. Default is `p0_reduced_units`.

    Returns
    -------
    float
        p = p0(T) * exp(mu/(k_B*T)), the inverse of `mu_from_pressure`.

    See Also
    --------
    mu_from_pressure : the forward conversion.
    """
    return p0_of_T(T) * np.exp(mu / T)


def mu_grid_from_pressures(temperatures: List[float], pressures: List[float],
                            p0_of_T: Callable[[float], float] = p0_reduced_units) -> np.ndarray:
    """
    Build a 2D chemical-potential grid for a shared pressure ladder.

    Holding p fixed across T generally means mu varies with T (through
    both the explicit T*ln(p/p0) and p0(T) itself), which is why
    `ReplicaExchangeGCMC` accepts a 2D mu grid instead of only a flat list.

    Parameters
    ----------
    temperatures : list of float
        The temperature ladder, length n_T.
    pressures : list of float
        The shared pressure ladder, length n_p.
    p0_of_T : callable, optional
        A function ``p0_of_T(T) -> float`` giving the reference pressure
        at temperature `T`. Default is `p0_reduced_units`.

    Returns
    -------
    numpy.ndarray
        The (n_T, n_p) array of chemical potentials `ReplicaExchangeGCMC`
        needs to run a replica ladder at this shared pressure ladder.
    """
    T_arr = np.asarray(temperatures, dtype=float)
    p_arr = np.asarray(pressures, dtype=float)
    p0 = np.array([p0_of_T(T) for T in T_arr])
    return T_arr[:, None] * np.log(p_arr[None, :] / p0[:, None])


def invert_mu_of_Tp(mu_of_Tp: Callable[[float, float], float], T: float, mu_target: float,
                     p_bracket: Tuple[float, float] = (1e-40, 1e100),
                     max_bracket_expansions: int = 400, xtol: float = 1e-12) -> float:
    """
    Numerically invert an arbitrary mu(T, p) relationship for p.

    Solves ``mu_of_Tp(T, p) == mu_target`` for p, given only a black-box
    `mu_of_Tp` callable -- for reservoirs (e.g. a diatomic gas via
    `reservoir.py`) that aren't a simple closed-form `mu_from_pressure` and
    so have no ready-made analytic inverse like `pressure_from_mu`.

    Assumes ``mu_of_Tp(T, .)`` is monotonically increasing in p, which
    holds for any physically sensible reservoir (more gas pressure never
    lowers the chemical potential at fixed T). Solves in log(p) rather
    than p directly -- p can easily need to span dozens of orders of
    magnitude (see ``examples/07_general_Tp_phase_diagram.py``), where
    root-finding in p itself is numerically hopeless.

    Parameters
    ----------
    mu_of_Tp : callable
        A function ``mu_of_Tp(T, p) -> mu``, monotonically increasing
        in p at fixed T.
    T : float
        Temperature at which to invert.
    mu_target : float
        The chemical potential to solve for the corresponding pressure of.
    p_bracket : tuple of float, optional
        Initial (p_lo, p_hi) bracket to search within. Expanded
        geometrically (in each direction independently, in log-space)
        until it brackets a sign change, so a generous default is fine
        even when the true root sits at a wildly unphysical pressure --
        which is itself sometimes the point. Default is ``(1e-40, 1e100)``.
    max_bracket_expansions : int, optional
        Maximum number of geometric bracket expansions to attempt, in
        each direction. Default is 400.
    xtol : float, optional
        Absolute tolerance (in log(p)) passed to `scipy.optimize.brentq`.
        Default is 1e-12.

    Returns
    -------
    float
        The pressure p such that ``mu_of_Tp(T, p) == mu_target``.

    Raises
    ------
    RuntimeError
        If `p_bracket`, even after `max_bracket_expansions` geometric
        expansions in each direction, does not bracket a sign change.
    """
    from scipy.optimize import brentq

    def f_log(x: float) -> float:
        return mu_of_Tp(T, math.exp(x)) - mu_target

    lo, hi = math.log(p_bracket[0]), math.log(p_bracket[1])
    f_lo, f_hi = f_log(lo), f_log(hi)
    n = 0
    while f_lo > 0.0 and n < max_bracket_expansions:
        lo -= math.log(10.0)
        f_lo = f_log(lo)
        n += 1
    while f_hi < 0.0 and n < max_bracket_expansions:
        hi += math.log(10.0)
        f_hi = f_log(hi)
        n += 1
    if f_lo > 0.0 or f_hi < 0.0:
        raise RuntimeError(
            f"Could not bracket p for mu_target={mu_target} at T={T} within "
            f"{max_bracket_expansions} geometric bracket expansions from {p_bracket}."
        )
    x_root = brentq(f_log, lo, hi, xtol=xtol)
    return math.exp(x_root)
