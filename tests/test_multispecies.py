"""
Several exchanged species (``exchange_species=['O', 'Cl']``): per-species
GCMC moves, the species-summed replica-exchange rule and MBAR reduced
potential, sample I/O and checkpoints -- checked against exact results:

* ideal gas: <N_s> = V exp(beta mu_s) / Lambda_s^3 at every grid point,
  directly and after MBAR reweighting off the grid;
* two labels of the SAME interacting particle: the total-N statistics equal
  those of one species with activity z = z_1 + z_2, z_s = exp(beta mu_s) /
  Lambda_s^3 (the grand partition function only sees the sum);
* a one-element species list is exactly the single-species code path.
"""

import numpy as np
import pytest

from gcmc_re import (LJCalculator, PairLJCalculator, ReplicaExchangeGCMC, SlabSystem,
                     SpeciesLambdaCubed, ThermalWavelengthCubed, ThermoState, load_samples,
                     save_samples)
from gcmc_re.calculators import Calculator
from gcmc_re.replica import Replica

SPECIES = ("O", "Cl")


class ZeroCalculator(Calculator):
    """Ideal gas: every configuration has zero interaction energy (fast path)."""

    supports_incremental = True

    def total_energy(self, system) -> float:
        return 0.0

    def pair_energy_of(self, system, pos, exclude=None, species=None) -> float:
        return 0.0


def _ideal_slab(rng=None, species=SPECIES):
    return SlabSystem(4.0, 4.0, 2.0, ZeroCalculator(), exchange_species=list(species))


def _ideal_mean(T, mu, lam3, volume=32.0):
    return volume * np.exp(np.asarray(mu) / T) / np.asarray(lam3)


# ---------------------------------------------------------------- bookkeeping
def test_system_species_bookkeeping():
    sysm = _ideal_slab()
    assert sysm.exchange_species_list == SPECIES and sysm.n_exchange_species == 2
    sysm.apply_insert(np.array([1.0, 1.0, 1.0]), species="Cl")
    sysm.apply_insert(np.array([2.0, 1.0, 1.0]), species="O")
    sysm.apply_insert(np.array([3.0, 1.0, 1.0]), species="Cl")
    assert sysm.species_counts().tolist() == [1, 2]
    assert sysm.exchange_zone_indices("Cl").tolist() == [0, 2]
    with pytest.raises(ValueError, match="name the one to insert"):
        sysm.apply_insert(np.array([1.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="species="):
        SlabSystem(4.0, 4.0, 2.0, ZeroCalculator(), exchange_species=["O", "Cl"],
                   positions=[[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="distinct"):
        SlabSystem(4.0, 4.0, 2.0, ZeroCalculator(), exchange_species=["O", "O"])
    # a one-element list is the single label
    assert SlabSystem(4.0, 4.0, 2.0, ZeroCalculator(), exchange_species=["H"]).exchange_species == "H"


def test_incremental_insert_energy_uses_the_inserted_species():
    class Generic(PairLJCalculator):
        supports_incremental = False

    params = {("A", "O"): (1.0, 1.0), ("A", "Cl"): (2.0, 1.2), ("O", "Cl"): (0.5, 1.1),
              ("Cl", "Cl"): (0.7, 1.3)}
    frozen = np.array([[1.0, 1.0, 0.5], [3.0, 3.0, 0.5]])

    def make(calc):
        sysm = SlabSystem(8.0, 8.0, 4.0, calc, exchange_species=["O", "Cl"],
                          frozen_positions=frozen, frozen_species=np.array(["A", "A"]))
        sysm.apply_insert(np.array([2.0, 2.0, 1.5]), species="Cl")
        return sysm

    fast, slow = make(PairLJCalculator(params)), make(Generic(params))
    pos = np.array([1.2, 1.4, 1.6])
    for sp in SPECIES:
        assert fast.pair_energy_of(pos, species=sp) == pytest.approx(
            slow.generic_insert_dU(pos, species=sp), rel=1e-12, abs=1e-12)
    assert fast.pair_energy_of(pos, species="O") != pytest.approx(
        fast.pair_energy_of(pos, species="Cl"))


def test_engine_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="n_mu, 2"):
        ReplicaExchangeGCMC([1.0], [-1.0, -2.0, -3.0], _ideal_slab)
    calls = []

    def mixed(rng):  # the second system exchanges different species
        calls.append(1)
        return _ideal_slab(species=SPECIES if len(calls) == 1 else ("O", "N"))

    with pytest.raises(ValueError, match="same species"):
        ReplicaExchangeGCMC([1.0, 2.0], [[-1.0, -2.0]], mixed)
    with pytest.raises(ValueError, match="1 or 2"):
        ReplicaExchangeGCMC([1.0], [[-1.0, -2.0]], _ideal_slab,
                            lambda_cubed_of_T=lambda T: np.ones(3))



# ---------------------------------------------------------------- ideal gas
def test_ideal_gas_single_replica_matches_poisson_means():
    T, mu, lam3 = 1.0, np.log([4.0 / 32.0, 2.0 * 2.0 / 32.0]), np.array([1.0, 2.0])
    rep = Replica(ThermoState(T, mu, (0, 0), lambda_cubed=lam3), _ideal_slab(),
                  np.random.default_rng(1))
    counts = []
    for _ in range(6000):
        rep.sweep(10, p_displace=0.0, p_insert=0.5, p_delete=0.5)
        counts.append(rep.system.species_counts())
    mean = np.mean(counts[500:], axis=0)
    np.testing.assert_allclose(mean, [4.0, 2.0], atol=0.15)
    acc = rep.acceptance_rates()
    assert {"insert_O", "insert_Cl", "delete_O", "delete_Cl"} <= set(acc)


def test_ideal_gas_ladder_with_species_and_T_dependent_lambda():
    # Lambda_Cl depends (steeply) on T, so swaps along T exercise the
    # per-species Lambda term of the exchange rule; with few local GC moves
    # per swap, dropping that term biases <N_s> by ~50%
    temperatures = [1.0, 1.15]
    lam = [lambda T: 1.0, lambda T: 2.0 * T ** -6]
    mus = np.array([[-1.9, -2.3], [-1.75, -2.15]])  # path of (mu_O, mu_Cl)
    sim = ReplicaExchangeGCMC(temperatures, mus, _ideal_slab, seed=4, lambda_cubed_of_T=lam)
    assert sim.mu_grid.shape == (2, 2, 2) and sim.exchange_species == SPECIES
    sim.run(6000, sweeps_per_cycle=2, burn_in_cycles=300, verbose=False,
            p_displace=0.0, p_insert=0.5, p_delete=0.5)
    assert all(acc > 0.05 for acc in sim.swap_acceptance().values())
    for (it, imu), smp in sim.samples.items():
        T = temperatures[it]
        expected = _ideal_mean(T, mus[imu], [1.0, lam[1](T)])
        got = np.mean(smp["N_s"], axis=0)
        np.testing.assert_allclose(got, expected, rtol=0.08, atol=0.1)
        assert np.array_equal(np.sum(smp["N_s"], axis=1), smp["N"])

    from gcmc_re import MBARSurfaceAnalysis
    pytest.importorskip("pymbar")
    an = MBARSurfaceAnalysis(sim)
    T, mu = 1.08, np.array([-1.8, -2.2])  # off the grid
    means, sigmas = an.mean_N_species(T, mu)
    np.testing.assert_allclose(means, _ideal_mean(T, mu, [1.0, lam[1](T)]), rtol=0.1)
    total, _ = an.mean_N(T, mu)
    assert total == pytest.approx(means.sum(), rel=1e-6)
    pd_cl = an.phase_diagram([T], [mu], observable="N_Cl")
    assert pd_cl.shape == (1, 1) and pd_cl[0, 0] == pytest.approx(means[1], rel=1e-6)
    comp = an.most_probable_composition(temperatures, mus)
    assert comp.shape == (2, 2, 2)


# ---------------------------------------------------------- interacting, exact
def _lj_slab(species):
    # LJCalculator treats every mobile particle alike: the labels are only
    # names, so the two-species system is one species with z = z_O + z_Cl
    return SlabSystem(5.0, 5.0, 3.0, LJCalculator(), exchange_species=species,
                      wall_epsilon=2.0, wall_sigma=1.0)


def _total_N_U_fraction(species, mu, lam3, T=2.0, seed=7, n_cycles=1500):
    rep = Replica(ThermoState(T, mu, (0, 0), lambda_cubed=lam3), _lj_slab(species),
                  np.random.default_rng(seed))
    N, U, frac = [], [], []
    for k in range(n_cycles):
        rep.sweep(40)
        if k >= 300:
            N.append(rep.system.n)
            U.append(rep.energy)
            frac.append(rep.system.species_counts()[-1] / max(1, rep.system.n))
    return np.mean(N), np.mean(U), np.mean(frac)


def test_two_labels_of_one_particle_equal_one_species_with_summed_activity():
    # supercritical film (T = 2), so the chains decorrelate quickly; seeds
    # fixed. Dropping Lambda_s from the per-species rules would shift <N> by
    # ~15% here, counting all species in N_s by far more.
    T, mu_single, lam3 = 2.0, -3.0, np.array([1.0, 0.6])
    z = np.exp(mu_single / T) * np.array([1.0, 2.0]) / 3.0  # z_Cl = 2 z_O, sum = z_single
    mu = T * np.log(z * lam3)
    N2, U2, frac_cl = _total_N_U_fraction(["O", "Cl"], mu, lam3)
    N1, U1, _ = _total_N_U_fraction("O", mu_single, 1.0, seed=107)
    assert N1 > 10  # an interacting film, not a dilute gas
    assert N2 == pytest.approx(N1, rel=0.06)
    assert U2 == pytest.approx(U1, rel=0.10)
    # each particle carries label s with probability z_s / sum(z), exactly
    assert frac_cl == pytest.approx(2.0 / 3.0, abs=0.02)


# ------------------------------------------------------- exactness of the S=1 path
def test_one_element_species_list_is_the_single_species_code_path():
    def run(species):
        sim = ReplicaExchangeGCMC([1.0, 1.2], [-3.0, -2.5], lambda rng: _lj_slab(species), seed=2)
        sim.run_regc(150, x0=0.9, diffusion="mc", diffusion_steps=1, record_every=5,
                     verbose=False)
        return sim

    a, b = run("H"), run(["H"])
    assert b.n_species == 1 and "N_s" not in b.samples[(0, 0)]
    for key in a.samples:
        assert a.samples[key] == b.samples[key]


# ------------------------------------------------------- I/O and checkpoints
def test_save_load_samples_roundtrip(tmp_path):
    pytest.importorskip("pymbar")
    from gcmc_re import MBARSurfaceAnalysis
    lam = SpeciesLambdaCubed([ThermalWavelengthCubed(16.0), ThermalWavelengthCubed(35.45)])
    T = [1.0, 1.4]
    mus = np.array([[-2.0, -2.5], [-1.0, -1.5]]) + np.log(lam(1.2))
    sim = ReplicaExchangeGCMC(T, mus, _ideal_slab, seed=1, lambda_cubed_of_T=lam)
    sim.run_regc(400, x0=0.9, diffusion="mc_moves", diffusion_steps=1, record_every=4,
                 verbose=False)
    path = str(tmp_path / "s.npz")
    save_samples(sim, path)
    ss = load_samples(path)
    assert ss.exchange_species == SPECIES
    assert isinstance(ss.lambda_cubed_of_T, SpeciesLambdaCubed)
    assert ss.lambda_cubed_of_T.masses_amu == [16.0, 35.45]
    assert np.array_equal(ss.mu_grid, sim.mu_grid)
    assert ss.samples[(1, 1)]["N_s"] == sim.samples[(1, 1)]["N_s"]
    T_grid, path_mu = [1.0, 1.2, 1.4], mus + 0.1
    np.testing.assert_allclose(MBARSurfaceAnalysis(ss).phase_diagram(T_grid, path_mu, observable="N_O"),
                               MBARSurfaceAnalysis(sim).phase_diagram(T_grid, path_mu, observable="N_O"),
                               rtol=1e-10)


def test_checkpoint_continuation_is_bit_identical(tmp_path):
    def make():
        return ReplicaExchangeGCMC([1.0, 1.3], [[-2.0, -2.5], [-1.5, -2.0]], _ideal_slab, seed=9,
                                   lambda_cubed_of_T=[lambda T: 1.0, lambda T: 0.5])

    def go(sim):
        sim.run_regc(100, x0=0.8, diffusion="mc_moves", diffusion_steps=1, record_every=2,
                     verbose=False)

    ref = make(); go(ref); go(ref)
    first = make(); go(first)
    path = str(tmp_path / "c.pkl")
    first.save_checkpoint(path)
    resumed = make()
    resumed.load_checkpoint(path)
    go(resumed)
    for key in ref.samples:
        assert resumed.samples[key] == ref.samples[key]
