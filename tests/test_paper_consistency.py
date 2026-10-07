"""
Consistency checks of the REGC acceptance rules against exact ideal-gas
results (see docs/REGC_algorithm.md). For a non-interacting gas the
grand-canonical particle number in a volume V is Poisson with mean
<N> = exp(beta*mu) * V / Lambda^3, so any inconsistency between the
insert/delete rules, the replica-exchange rule, and the MBAR reduced
potential shows up as a biased <N>.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gcmc_re import LJSystem, ReplicaExchangeGCMC, SlabLJSystem
from gcmc_re.calculators import Calculator
from gcmc_re.replica import Replica
from gcmc_re.state import ThermoState


class ZeroCalculator(Calculator):
    """Ideal gas: every configuration has zero interaction energy."""

    def total_energy(self, system) -> float:
        return 0.0


class FixedRNG:
    """Stands in for a replica's rng in `_attempt_exchange`: always draws `u`."""

    def __init__(self, u: float):
        self.u = u

    def random(self) -> float:
        return self.u


def _expected_swap_prob(beta_a, beta_b, mu_a, mu_b, Ua, Ub, Na, Nb, lam3_a, lam3_b):
    log_acc = ((beta_a - beta_b) * (Ua - Ub) - (beta_a * mu_a - beta_b * mu_b) * (Na - Nb)
               + (Na - Nb) * np.log(lam3_a / lam3_b))
    return min(1.0, float(np.exp(log_acc)))


@pytest.mark.parametrize("lambda_cubed_of_T", [lambda T: 1.0, lambda T: 0.3 * T ** -1.5],
                         ids=["reduced-units", "physical-Lambda"])
def test_exchange_acceptance_uses_same_lambda_as_gcmc(lambda_cubed_of_T):
    """Fix 1: the swap prefactor is (Lambda_a^3/Lambda_b^3)^(Na-Nb), with the
    same Lambda^3 the insert/delete moves use -- 1 in reduced units, the
    paper's (beta_a/beta_b)^(3/2 (Na-Nb)) for a physical Lambda(T)."""
    temperatures = [1.0, 1.6]
    sim = ReplicaExchangeGCMC(temperatures, [-1.0],
                              lambda rng: LJSystem(4.0, ZeroCalculator()),
                              lambda_cubed_of_T=lambda_cubed_of_T)
    ra, rb = sim.grid[(0, 0)], sim.grid[(1, 0)]
    ra.system.positions = np.zeros((7, 3))
    ra.system.species = np.array(["X"] * 7)
    rb.system.positions = np.zeros((3, 3))
    rb.system.species = np.array(["X"] * 3)
    ra.energy, rb.energy = -12.0, 0.0

    p = _expected_swap_prob(ra.state.beta, rb.state.beta, ra.state.mu, rb.state.mu,
                            ra.energy, rb.energy, 7, 3,
                            ra.state.lambda_cubed, rb.state.lambda_cubed)
    assert 0.0 < p < 1.0  # the test is only meaningful if p isn't saturated

    if lambda_cubed_of_T(1.0) != 1.0:
        beta_ratio_factor = (ra.state.beta / rb.state.beta) ** (1.5 * (7 - 3))
        lam_factor = (ra.state.lambda_cubed / rb.state.lambda_cubed) ** (7 - 3)
        assert lam_factor == pytest.approx(beta_ratio_factor)

    ra.rng = FixedRNG(p * 1.001)
    assert not sim._attempt_exchange((0, 0), (1, 0))
    ra.rng = FixedRNG(p * 0.999)
    assert sim._attempt_exchange((0, 0), (1, 0))


def test_replica_exchange_ideal_gas_density():
    """Fix 1, end to end: with frequent exchanges between two temperatures
    in reduced units (Lambda = 1), each grid point must still sample the
    exact ideal-gas <N> = exp(beta*mu) * V. mu is scaled with T so both
    states have the same density: every correct swap is then accepted, and
    a spurious (beta_a/beta_b)^(3/2 dN) factor would bias <N> by ~15%."""
    L = 3.0
    temperatures = [1.0, 1.5]
    mu_grid = [[-1.0], [-1.5]]
    sim = ReplicaExchangeGCMC(temperatures, mu_grid,
                              lambda rng: LJSystem(L, ZeroCalculator()), seed=3)
    sim.run(n_cycles=3000, sweeps_per_cycle=4, exchange_every=1, record_every=1,
            burn_in_cycles=100, verbose=False, p_displace=0.0, p_insert=0.5, p_delete=0.5)
    expected = np.exp(-1.0) * L ** 3
    for it in range(len(temperatures)):
        measured = np.mean(sim.samples[(it, 0)]["N"])
        assert measured == pytest.approx(expected, rel=0.05), (it, measured, expected)


def test_deletion_restricted_to_insertion_zone_ideal_gas():
    """Fix 3: with insertion restricted to a sub-slab, deletion only picks
    particles inside it and the acceptance rules use the zone's count.
    Particles diffuse freely over the whole slab, so the exact result is
    the grand-canonical ideal gas over the *whole* accessible volume:
    total <N> = exp(beta*mu) * V_box / Lambda^3, of which a fraction
    V_zone / V_box is in the zone."""
    Lx = Ly = 2.0
    Lz = 10.0
    z_range = (4.0, 6.0)
    mu, T = np.log(0.5), 1.0
    system = SlabLJSystem(Lx, Ly, Lz, ZeroCalculator(), wall_epsilon=0.0,
                          insertion_z_range=z_range)
    replica = Replica(ThermoState(T, mu, label=(0, 0)), system,
                      np.random.default_rng(11), max_disp=1.0)

    n_total, n_zone = [], []
    replica.sweep(20000, p_displace=0.8, p_insert=0.1, p_delete=0.1)  # burn-in
    for _ in range(3000):
        replica.sweep(100, p_displace=0.8, p_insert=0.1, p_delete=0.1)
        n_total.append(system.n)
        n_zone.append(len(system.exchange_zone_indices()))

    density = np.exp(mu / T)
    assert np.mean(n_total) == pytest.approx(density * Lx * Ly * Lz, rel=0.07)
    assert np.mean(n_zone) == pytest.approx(density * system.volume, rel=0.1)


def test_mbar_includes_lambda_cubed_term():
    """Fix 2: MBAR reweighting across temperatures needs the N*ln(Lambda^3(T))
    term of the reduced potential. Feed it exact Poisson ideal-gas samples
    generated with a T-dependent Lambda and check the reweighted <N> at an
    unsampled (T, mu)."""
    pytest.importorskip("pymbar", reason="pip install pymbar to run this test")
    from gcmc_re import MBARSurfaceAnalysis

    V = 10.0
    lam3 = lambda T: 0.2 * T ** -1.5
    temperatures = [1.0, 1.5]
    mu_grid = np.array([[-2.0, -1.0], [-3.0, -2.0]])
    rng = np.random.default_rng(0)
    samples = {}
    for it, T in enumerate(temperatures):
        for imu, mu in enumerate(mu_grid[it]):
            N = rng.poisson(np.exp(mu / T) * V / lam3(T), size=3000).astype(float)
            samples[(it, imu)] = {"U": np.zeros_like(N).tolist(), "N": N.tolist()}
    sim = SimpleNamespace(temperatures=temperatures, mu_grid=mu_grid, samples=samples,
                          lambda_cubed_of_T=lam3)

    analysis = MBARSurfaceAnalysis(sim)
    T, mu = 1.25, -2.0
    mean_n, _ = analysis.mean_N(T, mu)
    assert mean_n == pytest.approx(np.exp(mu / T) * V / lam3(T), rel=0.03)
