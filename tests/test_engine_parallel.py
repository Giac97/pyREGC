"""
Serial (n_workers=1) and process-pool-parallel (n_workers>1) runs must be
bit-identical for the same seed: nothing about parallelizing the sweep
phase should change the algorithm's statistics, only which core executes
each replica's (already-independent) RNG-driven moves. Run this after
touching engine.py.
"""

import numpy as np
import pytest

from gcmc_re import LJCalculator, LJSystem, ReplicaExchangeGCMC


def make_bulk(rng: np.random.Generator) -> LJSystem:
    return LJSystem(7.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def _run(n_workers: int):
    temperatures = list(np.linspace(0.9, 1.3, 3))
    mus = list(np.linspace(-3.8, -3.2, 3))
    with ReplicaExchangeGCMC(temperatures, mus, make_bulk, seed=42,
                              max_disp=0.2, n_workers=n_workers) as sim:
        sim.run(n_cycles=20, sweeps_per_cycle=40, exchange_every=1,
                 record_every=5, burn_in_cycles=0, verbose=False)
    return sim.summary()


def test_parallel_matches_serial_bit_for_bit():
    rows_serial = _run(n_workers=1)
    rows_parallel = _run(n_workers=2)
    for rs, rp in zip(rows_serial, rows_parallel):
        assert rs["N"] == rp["N"]
        assert rs["energy"] == rp["energy"]


def test_mu_grid_accepts_1d_and_2d():
    """1D mus (shared across T) and an equivalent 2D mu_grid must produce
    the same per-replica states."""
    temperatures = [1.0, 1.1]
    mus_1d = [-3.0, -2.5]

    def factory(rng):
        return LJSystem(6.0, LJCalculator())

    sim_1d = ReplicaExchangeGCMC(temperatures, mus_1d, factory, seed=0)
    mus_2d = np.tile(np.array(mus_1d), (len(temperatures), 1))
    sim_2d = ReplicaExchangeGCMC(temperatures, mus_2d, factory, seed=0)

    for key in sim_1d.grid:
        assert sim_1d.grid[key].state.mu == sim_2d.grid[key].state.mu
        assert sim_1d.grid[key].state.temperature == sim_2d.grid[key].state.temperature


def test_lambda_cubed_defaults_to_one():
    def factory(rng):
        return LJSystem(6.0, LJCalculator())

    sim = ReplicaExchangeGCMC([1.0], [-3.0], factory, seed=0)
    assert sim.grid[(0, 0)].state.lambda_cubed == 1.0


def test_lambda_cubed_of_T_is_applied_per_temperature():
    def factory(rng):
        return LJSystem(6.0, LJCalculator())

    temperatures = [1.0, 2.0]
    sim = ReplicaExchangeGCMC(temperatures, [-3.0], factory, seed=0,
                               lambda_cubed_of_T=lambda T: 10.0 * T)
    assert sim.grid[(0, 0)].state.lambda_cubed == pytest.approx(10.0)
    assert sim.grid[(1, 0)].state.lambda_cubed == pytest.approx(20.0)


def test_insertion_z_range_restricts_sampling_and_volume():
    from gcmc_re import SlabLJSystem

    system = SlabLJSystem(Lx=10.0, Ly=10.0, Lz=20.0, calculator=LJCalculator(),
                           insertion_z_range=(5.0, 8.0))
    assert system.volume == pytest.approx(10.0 * 10.0 * 3.0)

    rng = np.random.default_rng(0)
    zs = np.array([system.random_insertion_position(rng)[2] for _ in range(200)])
    assert zs.min() >= 5.0
    assert zs.max() <= 8.0


def test_insertion_z_range_defaults_to_full_box():
    from gcmc_re import SlabLJSystem

    system = SlabLJSystem(Lx=5.0, Ly=5.0, Lz=12.0, calculator=LJCalculator())
    assert system.insertion_z_range == (0.0, 12.0)
    assert system.volume == pytest.approx(5.0 * 5.0 * 12.0)


def test_parallel_emt_via_factory_does_not_reproduce_the_pickling_bug():
    """Regression test for a real bug report: n_workers > 1 with an
    ASECalculatorAdapter wrapping a live (already-used) EMT instance fails
    with 'Can't pickle local object EMT.initialize.<locals>.<lambda>' deep
    inside concurrent.futures. Passing the EMT class itself (a factory)
    instead of EMT() must run cleanly end to end through the real
    ReplicaExchangeGCMC + ProcessPoolExecutor machinery, not just in an
    isolated pickle check."""
    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    from gcmc_re import ASECalculatorAdapter, SlabLJSystem, kelvin_to_eV

    slab = fcc111("Pd", size=(2, 2, 2), vacuum=6.0, orthogonal=True, periodic=False)
    Lx, Ly, Lz = slab.cell[0, 0], slab.cell[1, 1], slab.cell[2, 2]
    frozen_positions = slab.positions.copy()
    frozen_species = np.array(slab.get_chemical_symbols())

    def make_system(rng):
        calc = ASECalculatorAdapter(EMT, cell=[Lx, Ly, Lz], pbc=(True, True, False))
        return SlabLJSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                             frozen_positions=frozen_positions, frozen_species=frozen_species,
                             insertion_z_range=(0.5, 2.5))

    T_eV = kelvin_to_eV(400.0)
    with ReplicaExchangeGCMC([T_eV], [0.0, 0.5], make_system, seed=1,
                              max_disp=0.3, n_workers=2) as sim:
        sim.run(n_cycles=4, sweeps_per_cycle=5, record_every=2, verbose=False)
    # reaching here at all is the test; also sanity-check it produced samples
    assert len(sim.samples[(0, 0)]["N"]) > 0
