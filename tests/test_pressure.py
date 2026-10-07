"""mu(p, T) <-> p round-trips, and mu_grid_from_pressures shape/values."""

import numpy as np
import pytest

from gcmc_re import (
    invert_mu_of_Tp,
    mu_from_pressure,
    mu_grid_from_pressures,
    p0_ideal_gas_wavelength,
    p0_reduced_units,
    pressure_from_mu,
)


@pytest.mark.parametrize("p0_of_T", [p0_reduced_units, p0_ideal_gas_wavelength])
def test_round_trip(p0_of_T):
    T, p = 0.85, 3.7e-3
    mu = mu_from_pressure(T, p, p0_of_T)
    p_back = pressure_from_mu(T, mu, p0_of_T)
    assert p_back == pytest.approx(p, rel=1e-10)


def test_reduced_units_convention_is_p_equals_activity():
    # with p0(T) = T, mu = T*ln(p/T) means p/T = exp(mu/T) = activity z
    T, mu = 1.0, -3.0
    p = pressure_from_mu(T, mu, p0_reduced_units)
    z = np.exp(mu / T)
    assert p / T == pytest.approx(z, rel=1e-10)


def test_mu_grid_shape_and_values():
    temperatures = [0.8, 1.0, 1.2]
    pressures = [1e-3, 1e-2]
    grid = mu_grid_from_pressures(temperatures, pressures, p0_of_T=p0_reduced_units)
    assert grid.shape == (3, 2)
    for it, T in enumerate(temperatures):
        for ip, p in enumerate(pressures):
            assert grid[it, ip] == pytest.approx(mu_from_pressure(T, p, p0_reduced_units))


def test_invert_mu_of_Tp_matches_analytic_inverse():
    T, p_true = 1.1, 3.7e-3
    mu_target = mu_from_pressure(T, p_true, p0_reduced_units)
    p_numeric = invert_mu_of_Tp(lambda T, p: mu_from_pressure(T, p, p0_reduced_units), T, mu_target)
    assert p_numeric == pytest.approx(p_true, rel=1e-8)


def test_invert_mu_of_Tp_handles_extreme_scales():
    """The root can legitimately sit many tens of orders of magnitude away
    from a 'reasonable' bracket -- this is exactly the diatomic-reservoir
    situation from reservoir.py, so the default bracket must actually reach
    that far without the caller doing anything special."""
    T = 1.0
    # a contrived mu_of_Tp whose root sits around p ~ 1e70
    mu_of_Tp = lambda T, p: T * np.log(p) - 300.0
    p_numeric = invert_mu_of_Tp(mu_of_Tp, T, mu_target=0.0)
    assert p_numeric == pytest.approx(np.exp(300.0), rel=1e-8)
