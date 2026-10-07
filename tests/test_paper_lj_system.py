"""
Building blocks for the Lennard-Jones surface of Zhou, Scheffler &
Ghiringhelli, PRB 100, 174106 (2019), Sec. III A: the multi-species
PairLJCalculator (sees frozen atoms), the hexagonal surface cell of
SlabLJSystem, the 'mc_moves' diffusion mode, and the block execution of
run_regc (must equal a naive step-by-step loop exactly).
"""

import itertools

import numpy as np
import pytest

from gcmc_re import (LJCalculator, LJSystem, PairLJCalculator, ReplicaExchangeGCMC,
                     SlabLJSystem)
from gcmc_re.calculators import Calculator
from gcmc_re.engine import SWAP_TYPES
from gcmc_re.replica import Replica
from gcmc_re.state import ThermoState

PAPER_PARAMS = {("A", "B"): (0.66, 1.91), ("B", "B"): (0.01, 1.2)}
L_HEX = 11.489
HEX = np.array([[L_HEX, 0.0], [L_HEX / 2, L_HEX * np.sqrt(3) / 2]])


def _hex_slab(calc, rng, n_mobile=12, n_frozen=9, Lz=12.0):
    frozen = np.column_stack([rng.random((n_frozen, 2)) @ HEX, np.full(n_frozen, 1.0)])
    mobile = np.column_stack([rng.random((n_mobile, 2)) @ HEX, 2.0 + 6.0 * rng.random(n_mobile)])
    return SlabLJSystem(None, None, Lz, calc, exchange_species="B", cell_xy=HEX,
                        frozen_positions=frozen, frozen_species=np.array(["A"] * n_frozen),
                        positions=mobile, species=np.array(["B"] * n_mobile))


def _brute_min_image(system, dr):
    best = None
    for i, j in itertools.product(range(-3, 4), repeat=2):
        cand = dr.copy()
        cand[:2] += i * system._A[0] + j * system._A[1]
        if best is None or cand @ cand < best @ best:
            best = cand
    return best


def test_hex_min_image_is_exact():
    rng = np.random.default_rng(0)
    system = _hex_slab(ZeroCalc(), rng)
    for _ in range(500):
        dr = np.concatenate([(rng.random(2) * 4 - 2) @ HEX, [rng.random()]])
        mi = system._min_image(dr)
        assert np.allclose(mi, _brute_min_image(system, dr))


def test_hex_wrap_insertion_and_volume():
    rng = np.random.default_rng(1)
    system = _hex_slab(ZeroCalc(), rng)
    system.insertion_z_range = (3.0, 7.0)
    assert system.area == pytest.approx(L_HEX ** 2 * np.sqrt(3) / 2)
    assert system.volume == pytest.approx(system.area * 4.0)
    pts = np.array([system.random_insertion_position(rng) for _ in range(4000)])
    frac = pts[:, :2] @ system._Ainv
    assert np.all((frac >= 0) & (frac < 1))
    assert np.all((pts[:, 2] >= 3.0) & (pts[:, 2] < 7.0))
    assert np.allclose(frac.mean(axis=0), 0.5, atol=0.02)  # uniform in the cell
    w = system._wrap(np.array([-3.0, 25.0, 4.0]))
    f = w[:2] @ system._Ainv
    assert np.all((f >= 0) & (f < 1)) and w[2] == 4.0


def test_cell_arguments_validated():
    with pytest.raises(ValueError):
        SlabLJSystem(5.0, 5.0, 10.0, ZeroCalc(), cell_xy=HEX)
    with pytest.raises(ValueError):
        SlabLJSystem(None, None, 10.0, ZeroCalc())


class ZeroCalc(Calculator):
    def total_energy(self, system):
        return 0.0


def _brute_energy(system, calc):
    pos, sp = system.full_configuration()
    e = 0.0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            par = calc.params.get((sp[i], sp[j]))
            if par is None:
                continue
            eps, sig, rc2, shift = par
            d = _brute_min_image(system, pos[i] - pos[j])
            r2 = d @ d
            if r2 < rc2:
                sr6 = (sig * sig / r2) ** 3
                e += 4 * eps * (sr6 * sr6 - sr6) - shift
    return e


def test_pair_lj_total_energy_matches_brute_force_with_frozen_atoms():
    rng = np.random.default_rng(2)
    calc = PairLJCalculator(PAPER_PARAMS)
    system = _hex_slab(calc, rng)
    assert system.total_energy() == pytest.approx(_brute_energy(system, calc), rel=1e-10)


def test_pair_lj_incremental_moves_match_total_energy_differences():
    rng = np.random.default_rng(3)
    calc = PairLJCalculator(PAPER_PARAMS)
    system = _hex_slab(calc, rng)
    e0 = system.total_energy()
    i, new = 4, system.positions[4] + np.array([0.3, -0.2, 0.4])
    du = system.pair_energy_of(new, exclude=i) - system.pair_energy_of(system.positions[i], exclude=i)
    assert du == pytest.approx(system.generic_displace_dU(i, new), abs=1e-10)
    trial = np.array([2.0, 3.0, 4.0])
    assert system.pair_energy_of(trial) == pytest.approx(system.generic_insert_dU(trial), abs=1e-10)
    assert -system.pair_energy_of(system.positions[7], exclude=7) == pytest.approx(
        system.generic_delete_dU(7), abs=1e-10)
    assert system.total_energy() == pytest.approx(e0)


def test_pair_lj_forces_are_minus_gradient():
    rng = np.random.default_rng(4)
    calc = PairLJCalculator(PAPER_PARAMS)
    system = _hex_slab(calc, rng, n_mobile=6)
    F = calc.forces(system)
    assert F.shape == (system.n, 3)
    h = 1e-6
    for i in range(system.n):
        for k in range(3):
            system.positions[i, k] += h
            ep = calc.total_energy(system)
            system.positions[i, k] -= 2 * h
            em = calc.total_energy(system)
            system.positions[i, k] += h
            assert F[i, k] == pytest.approx(-(ep - em) / (2 * h), abs=1e-5)


def test_pair_lj_reduces_to_lj_calculator_for_one_species():
    rng = np.random.default_rng(5)
    pos = rng.random((25, 3)) * 6.0
    a = LJSystem(6.0, LJCalculator(1.0, 1.0, 2.5), positions=pos.copy())
    b = LJSystem(6.0, PairLJCalculator({("X", "X"): (1.0, 1.0)}), positions=pos.copy())
    assert a.total_energy() == pytest.approx(b.total_energy(), rel=1e-12)
    assert np.allclose(a.calculator.forces(a), b.calculator.forces(b))


def test_hex_ideal_gas_density_exact():
    """GCMC + 'mc_moves' diffusion of an ideal gas in the hexagonal slab:
    <N> = exp(beta mu) * area * Lz (whole box, zone restricted)."""
    rng = np.random.default_rng(6)
    system = SlabLJSystem(None, None, 6.0, ZeroCalc(), exchange_species="B", cell_xy=HEX,
                          insertion_z_range=(1.0, 5.0))
    rep = Replica(ThermoState(1.0, np.log(0.02), (0, 0)), system, rng, max_disp=1.0)
    ns = []
    for k in range(6000):
        rep.regc_local_step(gc=True, mode='mc_moves', n=10)
        if k > 500:
            ns.append(system.n)
    assert np.mean(ns) == pytest.approx(0.02 * system.area * 6.0, rel=0.07)


def _naive_run_regc(sim, n_steps, x0, record_every, mode, n):
    """Reference: the step-by-step loop that run_regc's blocks must equal."""
    samples = {k: {'U': [], 'N': []} for k in sim.grid}
    for step in range(1, n_steps + 1):
        do_gc = bool(sim.rng.random() < x0)
        if not do_gc:
            t = SWAP_TYPES[int(sim.rng.integers(len(SWAP_TYPES)))]
            sim._exchange_round(t, int(sim.rng.integers(2)))
        for r in sim.grid.values():
            r.regc_local_step(do_gc, mode, n)
        if step % record_every == 0:
            for k, r in sim.grid.items():
                samples[k]['U'].append(r.energy)
                samples[k]['N'].append(r.system.n)
    return samples


@pytest.mark.parametrize("x0", [0.99, 0.5])
def test_block_execution_equals_naive_loop(x0):
    def make(rng):
        return LJSystem(5.0, LJCalculator(1.0, 1.0, 2.5))
    args = ([0.9, 1.2], [[-3.5, -3.0], [-3.5, -3.0]], make)
    ref = ReplicaExchangeGCMC(*args, seed=21)
    expected = _naive_run_regc(ref, 400, x0, 7, 'mc_moves', 3)
    sim = ReplicaExchangeGCMC(*args, seed=21)
    sim.run_regc(n_steps=400, x0=x0, diffusion='mc_moves', diffusion_steps=3,
                 record_every=7, verbose=True)  # verbose adds block boundaries too
    assert sim.samples == expected
    assert sim.swap_stats == ref.swap_stats
