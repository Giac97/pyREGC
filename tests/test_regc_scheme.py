"""
The [ZSG] Fig. 1 scheme (`ReplicaExchangeGCMC.run_regc`): the four swap
patterns of Fig. 2 (including diagonals), exact ideal-gas statistics of the
full GC / RE / diffusion loop, the rho(u) convergence stop, and ASE NVT MD
as the diffusion step.
"""

import itertools

import numpy as np
import pytest

from gcmc_re import LJSystem, ReplicaExchangeGCMC, SlabLJSystem
from gcmc_re.calculators import Calculator
from gcmc_re.engine import SWAP_TYPES


class ZeroCalculator(Calculator):
    """Ideal gas: every configuration has zero interaction energy."""

    def total_energy(self, system) -> float:
        return 0.0


def _ideal_sim(temperatures, mu_grid, L=3.0, seed=0):
    return ReplicaExchangeGCMC(temperatures, mu_grid,
                               lambda rng: LJSystem(L, ZeroCalculator()), seed=seed)


@pytest.mark.parametrize("shape", [(4, 4), (3, 5), (2, 1), (1, 3)])
def test_swap_patterns_disjoint_and_cover_all_eight_neighbours(shape):
    n_T, n_mu = shape
    sim = _ideal_sim(list(np.linspace(1.0, 2.0, n_T)), np.zeros((n_T, n_mu)))
    edges = set()
    for swap_type, parity in itertools.product(SWAP_TYPES, (0, 1)):
        pairs = sim.swap_pairs(swap_type, parity)
        keys = [k for pair in pairs for k in pair]
        assert len(keys) == len(set(keys)), "a replica appears twice in one round"
        for a, b in pairs:
            dl, dm = b[0] - a[0], b[1] - a[1]
            expected = {'T': (1, 0), 'mu': (0, 1), 'cross_left': (1, -1),
                        'cross_right': (1, 1)}[swap_type]
            assert (dl, dm) == expected
            edges.add(frozenset((a, b)))
    all_neighbours = {frozenset(((l, m), (l + dl, m + dm)))
                      for l in range(n_T) for m in range(n_mu)
                      for dl, dm in ((1, 0), (0, 1), (1, 1), (1, -1))
                      if 0 <= l + dl < n_T and 0 <= m + dm < n_mu}
    assert edges == all_neighbours


def test_fig2_4x4_round_involves_all_replicas():
    """[ZSG]: one swap type at offset 0 involves all replicas when L and M
    are even (the diagonals leave out only the two corners not on them)."""
    sim = _ideal_sim([1.0, 1.2, 1.4, 1.6], np.zeros((4, 4)))
    assert len(sim.swap_pairs('T', 0)) == 8
    assert len(sim.swap_pairs('mu', 0)) == 8
    assert len(sim.swap_pairs('cross_left', 0)) == 6   # Fig. 2(c): 3 arrows x 2 rows
    assert len(sim.swap_pairs('cross_right', 0)) == 6  # Fig. 2(d)


@pytest.mark.parametrize("swap_types", [SWAP_TYPES, ('cross_left', 'cross_right')],
                         ids=["all", "diagonal-only"])
def test_run_regc_ideal_gas_exact(swap_types):
    """Full Fig. 1 loop on a 2x2 ideal-gas grid with frequent exchanges:
    every state must sample <N> = exp(beta*mu) * V. mu is scaled with T so
    the two columns have equal density across T and swaps are accepted
    often, including along the diagonals."""
    L = 3.0
    temperatures = [1.0, 1.5]
    mu_grid = [[-1.0, -0.5], [-1.5, -0.75]]
    sim = _ideal_sim(temperatures, mu_grid, L=L, seed=5)
    sim.run_regc(n_steps=20000, x0=0.5, diffusion='mc', diffusion_steps=0,
                 record_every=1, burn_in_steps=200, swap_types=swap_types, verbose=False)
    for (it, imu), T in zip(itertools.product(range(2), range(2)), np.repeat(temperatures, 2)):
        expected = np.exp(mu_grid[it][imu] / T) * L ** 3
        measured = np.mean(sim.samples[(it, imu)]["N"])
        assert measured == pytest.approx(expected, rel=0.05), ((it, imu), measured, expected)
    acc = sim.swap_acceptance()
    for t in swap_types:
        assert sim.swap_stats[t][0] > 0 and acc[t] > 0.1


def test_run_regc_x0_controls_gc_vs_exchange():
    sim = _ideal_sim([1.0, 1.5], [[-1.0], [-1.5]])
    sim.run_regc(n_steps=2000, x0=0.9, diffusion='mc', diffusion_steps=0,
                 record_every=10, verbose=False)
    n_gc = sum(r.n_attempts['insert'] + r.n_attempts['delete'] for r in sim.grid.values())
    n_re_rounds_with_pairs = sim.swap_stats['T'][0]  # only 'T' has pairs on a 2x1 grid
    assert n_gc / len(sim.grid) == pytest.approx(0.9 * 2000, rel=0.05)
    # each RE step draws 'T' w.p. 1/4 and offset 0 (the only one with a pair) w.p. 1/2
    assert n_re_rounds_with_pairs == pytest.approx(0.1 * 2000 / 8, rel=0.35)


def test_run_regc_stops_on_rho_u_convergence():
    sim = _ideal_sim([1.0, 1.5], [[-1.0], [-1.5]])
    converged = sim.run_regc(n_steps=200000, x0=0.9, diffusion='mc', diffusion_steps=0,
                             record_every=5, convergence_every=2000, convergence_tol=0.05,
                             verbose=False)
    assert converged
    assert sim.convergence_history and sim.convergence_history[-1] < 0.05
    assert len(sim.samples[(0, 0)]["U"]) < 200000 // 5


def test_md_diffusion_with_ase_slab():
    """diffusion='md': ASE NVT MD moves substrate + mobile atoms, keeps
    frozen atoms fixed, respects the hard z-walls by reflection, and leaves
    Replica.energy equal to the true total energy."""
    pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    from gcmc_re import ASECalculatorAdapter, kelvin_to_eV

    slab = fcc111("Pd", size=(2, 2, 2), vacuum=None, orthogonal=True, periodic=True)
    slab.positions[:, 2] += 1.0
    Lx, Ly = slab.cell[0, 0], slab.cell[1, 1]
    z_top = slab.positions[:, 2].max()
    Lz = z_top + 3.0  # a thin vacuum region: H atoms will hit the top wall
    bottom = slab.positions[:, 2] < slab.positions[:, 2].mean()
    symbols = np.array(slab.get_chemical_symbols())

    def make_system(rng):
        calc = ASECalculatorAdapter(EMT, cell=[Lx, Ly, Lz], pbc=(True, True, False))
        h = np.column_stack([rng.random(3) * Lx, rng.random(3) * Ly, z_top + 1.0 + rng.random(3)])
        return SlabLJSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                            frozen_positions=slab.positions[bottom], frozen_species=symbols[bottom],
                            substrate_positions=slab.positions[~bottom],
                            substrate_species=symbols[~bottom], positions=h,
                            insertion_z_range=(z_top + 0.5, Lz))

    T_eV = kelvin_to_eV(2000.0)
    sim = ReplicaExchangeGCMC([T_eV], [0.0], make_system, seed=2)
    r = sim.grid[(0, 0)]
    frozen0 = r.system.frozen_positions.copy()
    substrate0 = r.system.substrate_positions.copy()
    sim.run_regc(n_steps=10, x0=1.0, diffusion='md', diffusion_steps=20, record_every=1,
                 verbose=False, timestep_fs=1.0, thermostat='bussi', taut_fs=20.0)

    assert np.array_equal(r.system.frozen_positions, frozen0)
    assert not np.allclose(r.system.substrate_positions, substrate0)
    z = np.concatenate([r.system.substrate_positions[:, 2], r.system.positions[:, 2]])
    assert np.all((z > 0.0) & (z <= Lz))
    assert np.isfinite(r.energy)
    assert r.energy == pytest.approx(r.system.total_energy(), abs=1e-8)


def test_md_reflects_at_hard_walls():
    """Fast H atoms in a 2.5 A-thick box cross it many times in 300 fs;
    without reflection they would leave (0, Lz]."""
    pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.emt import EMT

    from gcmc_re import ASECalculatorAdapter, kelvin_to_eV

    Lx = Ly = 6.0
    Lz = 2.5

    def make_system(rng):
        calc = ASECalculatorAdapter(EMT, cell=[Lx, Ly, Lz], pbc=(True, True, False))
        return SlabLJSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                            positions=[[1.0, 1.0, 1.0], [4.0, 4.0, 1.5]])

    sim = ReplicaExchangeGCMC([kelvin_to_eV(3000.0)], [0.0], make_system, seed=4)
    r = sim.grid[(0, 0)]
    r.diffuse('md', n=300, timestep_fs=1.0, thermostat='langevin')
    assert np.all((r.system.positions[:, 2] > 0.0) & (r.system.positions[:, 2] <= Lz))
    assert np.isfinite(r.energy)


def test_md_rejects_non_ase_calculator():
    sim = _ideal_sim([1.0], [0.0])
    with pytest.raises(ValueError, match="ASECalculatorAdapter"):
        sim.run_regc(n_steps=1, x0=1.0, diffusion='md', verbose=False)


def _bulk_lj(rng):
    from gcmc_re import LJCalculator
    return LJSystem(6.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def test_run_regc_parallel_matches_serial_bit_for_bit():
    def run(n_workers):
        with ReplicaExchangeGCMC([0.9, 1.1], [[-3.6, -3.2], [-3.6, -3.2]], _bulk_lj, seed=9,
                                 n_workers=n_workers) as sim:
            sim.run_regc(n_steps=60, x0=0.7, diffusion='mc', diffusion_steps=2,
                         record_every=5, verbose=False)
        return sim.samples, sim.swap_stats
    assert run(1) == run(2)
