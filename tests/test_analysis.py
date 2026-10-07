"""
MBARSurfaceAnalysis only reads sim.samples / sim.temperatures / sim.mu_grid
in its constructor, so we can test its (T,p) <-> (T,mu) routing logic with a
small synthetic fixture instead of an expensive real GCMC run -- these tests
are about the plumbing (does mu_of_Tp/p_of_Tmu actually get used, does the
default stay backward compatible), not about MBAR's statistics.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pymbar", reason="pip install pymbar to run this test")

from gcmc_re import (
    MBARSurfaceAnalysis,
    mu_from_pressure,
    p0_reduced_units,
    pressure_from_mu,
)


@pytest.fixture
def tiny_sim():
    """A minimal object with exactly the attributes MBARSurfaceAnalysis reads:
    two temperatures x two mus, a handful of noisy synthetic (U, N) samples
    per state -- just enough for pymbar's solver to have something to fit."""
    rng = np.random.default_rng(0)
    temperatures = [1.0, 1.2]
    mu_grid = np.array([[-3.0, -2.0], [-3.0, -2.0]])
    samples = {}
    for it in range(2):
        for imu in range(2):
            n_mean = 5.0 + 3.0 * imu + it  # some variation across states
            N = np.clip(rng.normal(n_mean, 1.0, size=100).round(), 0, None)
            U = -0.5 * N + rng.normal(0, 0.1, size=100)
            samples[(it, imu)] = {"U": U.tolist(), "N": N.tolist()}
    return SimpleNamespace(temperatures=temperatures, mu_grid=mu_grid, samples=samples)


def test_default_p0_of_T_behavior_unchanged(tiny_sim):
    analysis = MBARSurfaceAnalysis(tiny_sim, p0_of_T=p0_reduced_units)
    T, p = 1.1, 0.01
    assert analysis.mu_from_p(T, p) == pytest.approx(mu_from_pressure(T, p, p0_reduced_units))
    mu = -2.5
    assert analysis.p_from_mu(T, mu) == pytest.approx(pressure_from_mu(T, mu, p0_reduced_units))


def test_mu_of_Tp_override_is_used(tiny_sim):
    calls = []

    def custom_mu(T, p):
        calls.append((T, p))
        return 2.0 * T * np.log(p)  # some arbitrary but monotonic-in-p relation

    analysis = MBARSurfaceAnalysis(tiny_sim, mu_of_Tp=custom_mu)
    result = analysis.mu_from_p(1.1, 0.03)
    assert result == pytest.approx(custom_mu(1.1, 0.03))
    assert calls  # confirms the override, not the default, actually ran


def test_p_of_Tmu_override_is_used_in_preference_to_numeric_inverse(tiny_sim):
    def custom_mu(T, p):
        return T * np.log(p)

    def custom_p_inverse(T, mu):
        return -999.0  # deliberately NOT the true inverse, to prove this path is taken

    analysis = MBARSurfaceAnalysis(tiny_sim, mu_of_Tp=custom_mu, p_of_Tmu=custom_p_inverse)
    assert analysis.p_from_mu(1.0, 0.5) == -999.0


def test_p_from_mu_falls_back_to_numeric_inverse_of_mu_of_Tp(tiny_sim):
    def custom_mu(T, p):
        return T * np.log(p / 2.0)  # analytically invertible: p = 2*exp(mu/T)

    analysis = MBARSurfaceAnalysis(tiny_sim, mu_of_Tp=custom_mu)  # no p_of_Tmu given
    T, mu = 1.3, -1.0
    p_numeric = analysis.p_from_mu(T, mu)
    p_expected = 2.0 * np.exp(mu / T)
    assert p_numeric == pytest.approx(p_expected, rel=1e-6)


def test_isotherm_p_and_locate_coexistence_p_use_the_override(tiny_sim):
    """Confirm the override propagates through the higher-level methods
    that are built on mu_from_p/p_from_mu, not just those two directly."""

    def custom_mu(T, p):
        return T * np.log(p)

    analysis = MBARSurfaceAnalysis(tiny_sim, mu_of_Tp=custom_mu)
    p_grid = np.array([np.exp(-3.0), np.exp(-2.0)])  # -> mu = -3, -2 at T=1.0, matching mu_grid
    theta, _ = analysis.isotherm_p(1.0, p_grid, area=1.0)
    theta_direct, _ = analysis.isotherm(1.0, np.array([-3.0, -2.0]), area=1.0)
    assert theta == pytest.approx(theta_direct)


def test_pN_fast_path_matches_mbar_expectations(tiny_sim):
    analysis = MBARSurfaceAnalysis(tiny_sim)
    T, mu = 1.1, -2.5
    centers, probs = analysis.pN(T, mu)
    assert probs.sum() == pytest.approx(1.0)
    u = analysis._u_of_state(T, mu)
    for c, p in zip(centers, probs):
        ind = (np.abs(analysis.N_all - c) < 0.5).astype(float)
        ref = analysis.mbar.compute_expectations(ind, u_kn=u, compute_uncertainty=False)['mu'][0]
        assert p == pytest.approx(ref, abs=1e-10)
    m = analysis.most_probable_N(np.array([T]), np.array([mu]))
    assert m[0, 0] == int(round(centers[np.argmax(probs)]))


def test_most_probable_N_Tp_routes_through_mu_from_p(tiny_sim):
    analysis = MBARSurfaceAnalysis(tiny_sim, p0_of_T=p0_reduced_units)
    T, ps = 1.1, np.array([1e-2, 1e-1, 1.0])
    mus = np.array([mu_from_pressure(T, p, p0_reduced_units) for p in ps])
    expected = analysis.most_probable_N(np.array([T]), mus)
    assert np.array_equal(analysis.most_probable_N_Tp(np.array([T]), ps), expected)


def test_lbfgs_warm_start_matches_pymbar_solution(tiny_sim):
    from pymbar import MBAR

    from gcmc_re.analysis import _mbar_free_energies_lbfgs

    analysis = MBARSurfaceAnalysis(tiny_sim)
    reference = MBAR(analysis.u_kn, analysis.Nk).f_k
    assert np.allclose(_mbar_free_energies_lbfgs(analysis.u_kn, analysis.Nk), reference, atol=1e-6)
