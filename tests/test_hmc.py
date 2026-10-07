"""
Hybrid Monte Carlo: forces, per-species masses, and -- the check that
actually matters -- that HMC-augmented GCMC reproduces the same
equilibrium <N> and <U> as plain single-particle MC on the same system.
That comparison only means anything if both sides are actually
equilibrated; the fixed seeds/conditions below were chosen by explicitly
checking first-half vs second-half agreement within a run before trusting
them (see the accompanying design discussion -- an earlier, less careful
attempt at this exact test gave wildly inconsistent <N> across seeds
because the system hadn't equilibrated, which is exactly the failure mode
to guard against, not paper over with a looser tolerance).
"""

import numpy as np
import pytest

from gcmc_re import LJCalculator, LJSystem, ThermoState, Replica


# ---------------------------------------------------------------------------
# Forces
# ---------------------------------------------------------------------------
def test_lj_force_matches_numerical_gradient():
    sigma, epsilon, r_cut = 1.0, 1.0, 2.5
    calc = LJCalculator(sigma=sigma, epsilon=epsilon, r_cut=r_cut)
    rng = np.random.default_rng(0)
    system = LJSystem(7.0, calc, positions=rng.random((10, 3)) * 7.0)

    F = calc.forces(system)
    assert F.shape == (10, 3)

    # cross-check particle 3's force against a numerical gradient of total_energy
    i = 3
    h = 1e-6
    F_numeric = np.zeros(3)
    for d in range(3):
        pos_plus = system.positions[i].copy(); pos_plus[d] += h
        pos_minus = system.positions[i].copy(); pos_minus[d] -= h
        u_plus = calc.pair_energy_of(system, pos_plus, exclude=i)
        u_minus = calc.pair_energy_of(system, pos_minus, exclude=i)
        F_numeric[d] = -(u_plus - u_minus) / (2 * h)

    assert np.allclose(F[i], F_numeric, atol=1e-5)


def test_lj_force_vectorized_matches_per_particle_reference():
    """The vectorized (N,N,3) implementation must agree exactly with an
    unambiguous, slower per-particle reference computed via pair_energy_of's
    own displacement convention."""
    calc = LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5)
    rng = np.random.default_rng(1)
    system = LJSystem(7.0, calc, positions=rng.random((15, 3)) * 7.0)

    F_vectorized = calc.forces(system)

    F_reference = np.zeros((15, 3))
    for i in range(15):
        others_mask = np.ones(15, dtype=bool)
        others_mask[i] = False
        dr = system._min_image(system.positions[i] - system.positions[others_mask])
        r2 = np.einsum('ij,ij->i', dr, dr)
        inside = r2 < calc.r_cut2
        dr, r2 = dr[inside], r2[inside]
        s = (calc.sigma ** 2) / r2
        coeff = (24.0 * calc.epsilon / r2) * (2 * s ** 6 - s ** 3)
        F_reference[i] = (coeff[:, None] * dr).sum(axis=0)

    assert np.allclose(F_vectorized, F_reference, atol=1e-12)


def test_ase_calculator_adapter_forces_match_ase_lj():
    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.lj import LennardJones as ASELennardJones

    from gcmc_re import ASECalculatorAdapter

    sigma, epsilon, r_cut = 1.0, 1.0, 2.5
    rng = np.random.default_rng(2)
    L = 6.0
    positions = rng.random((12, 3)) * L

    native = LJSystem(L, LJCalculator(sigma, epsilon, r_cut), positions=positions.copy())
    ase_calc = ASECalculatorAdapter(
        ASELennardJones(sigma=sigma, epsilon=epsilon, rc=r_cut, smooth=False, shift=True),
        cell=[L, L, L], pbc=(True, True, True))
    ase_system = LJSystem(L, ase_calc, positions=positions.copy())

    F_native = native.calculator.forces(native)
    F_ase = ase_system.calculator.forces(ase_system)
    assert np.allclose(F_native, F_ase, atol=1e-6)


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------
def test_default_mass_amu_real_element_vs_fallback():
    from gcmc_re.units import default_mass_amu

    assert default_mass_amu("H") == pytest.approx(1.008, abs=0.01)
    assert default_mass_amu("Fe") == pytest.approx(55.845, abs=0.01)
    assert default_mass_amu("X") == 1.0  # generic reduced-units species -> fallback


def test_particle_system_masses_uses_explicit_override_first():
    calc = LJCalculator()
    system = LJSystem(7.0, calc, positions=np.zeros((3, 3)))
    system.mass_of_species = {"X": 2.5}
    masses = system.masses()
    assert masses.shape == (3,)
    assert np.all(masses == 2.5)


def test_particle_system_masses_default_is_one_for_generic_species():
    calc = LJCalculator()
    system = LJSystem(7.0, calc, positions=np.zeros((2, 3)))
    assert np.all(system.masses() == 1.0)


# ---------------------------------------------------------------------------
# HMC construction and bookkeeping
# ---------------------------------------------------------------------------
def test_hmc_requires_a_force_capable_calculator():
    from gcmc_re.calculators import Calculator

    class NoForcesCalculator(Calculator):
        def total_energy(self, system):
            return 0.0

    system = LJSystem(7.0, NoForcesCalculator())
    state = ThermoState(1.0, -3.0, (0, 0))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="supports_forces"):
        Replica(state, system, rng, hmc_enabled=True)


def test_hmc_energy_bookkeeping_stays_exact():
    """After many HMC moves, the incrementally-tracked energy must still
    match a fresh total_energy() recomputation -- same discipline as the
    existing generic-path bookkeeping tests."""
    rng = np.random.default_rng(3)
    system = LJSystem(7.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5),
                       positions=rng.random((20, 3)) * 7.0)
    state = ThermoState(temperature=1.2, mu=-3.0, label=(0, 0))
    replica = Replica(state, system, rng, hmc_enabled=True, hmc_leapfrog_steps=8, hmc_dt=0.01)

    replica.sweep(60, p_displace=1.0, p_insert=0.0, p_delete=0.0)  # HMC-only moves

    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)


def test_hmc_acceptance_rate_is_tracked_separately_from_displace():
    rng = np.random.default_rng(4)
    system = LJSystem(7.0, LJCalculator(), positions=rng.random((10, 3)) * 7.0)
    state = ThermoState(1.0, -3.0, (0, 0))
    replica = Replica(state, system, rng, hmc_enabled=True)
    replica.sweep(20, p_displace=1.0, p_insert=0.0, p_delete=0.0)
    acc = replica.acceptance_rates()
    assert replica.n_attempts['hmc'] == 20
    assert replica.n_attempts['displace'] == 0  # HMC replaced the displace slot, not added to it


# ---------------------------------------------------------------------------
# The check that actually matters: HMC-augmented GCMC reproduces the same
# equilibrium as plain MC-augmented GCMC.
# ---------------------------------------------------------------------------
MU = -3.6  # chosen (see module docstring) so the system equilibrates quickly and
           # fluctuates around a stable mean rather than slowly growing from empty


def _make_replica(seed, hmc_enabled, hmc_dt=0.01):
    rng = np.random.default_rng(seed)
    system = LJSystem(7.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))
    state = ThermoState(temperature=1.0, mu=MU, label=(0, 0))
    return Replica(state, system, rng, max_disp=0.2,
                    hmc_enabled=hmc_enabled, hmc_leapfrog_steps=10, hmc_dt=hmc_dt)


def _run(seed, hmc_enabled, n_cycles=250, sweeps=30, hmc_dt=0.01):
    r = _make_replica(seed, hmc_enabled, hmc_dt=hmc_dt)
    Ns, Us = [], []
    for c in range(n_cycles):
        r.sweep(sweeps)
        if c > n_cycles * 0.3:
            Ns.append(r.system.n); Us.append(r.energy)
    return np.mean(Ns), np.mean(Us)


def test_hmc_reproduces_the_same_equilibrium_as_plain_mc():
    N_mc, U_mc = _run(seed=50, hmc_enabled=False)
    N_hmc, U_hmc = _run(seed=51, hmc_enabled=True)
    # generous tolerance: this is a stochastic comparison of finite MCMC
    # samples, not an exact identity -- the point is to catch a genuinely
    # broken force/HMC implementation (which would give wildly different,
    # not just noisily different, results), not to chase statistical noise
    assert abs(N_hmc - N_mc) < 5.0
    assert abs(U_hmc - U_mc) < 3.0


def test_hmc_correctness_is_independent_of_timestep():
    """The point of doing this via Metropolis-corrected HMC rather than
    plain uncorrected MD: a much larger, sloppier timestep should still
    give the correct equilibrium (just possibly lower acceptance), not a
    biased one."""
    N_mc, U_mc = _run(seed=60, hmc_enabled=False)
    N_hmc_sloppy, U_hmc_sloppy = _run(seed=61, hmc_enabled=True, hmc_dt=0.04)
    assert abs(N_hmc_sloppy - N_mc) < 5.0
    assert abs(U_hmc_sloppy - U_mc) < 3.0
