"""
The displaceable substrate: a third particle category (alongside mobile/
GCMC-exchangeable and frozen/permanently-rigid) that enters the energy
evaluation and CAN move (MC or, jointly with the mobile species, HMC), but
is never inserted or deleted. This is what a reconstructing surface needs.

The check that actually matters here isn't "does it run without crashing"
-- it's whether substrate dynamics produce genuine physics: do substrate
atoms actually relax toward a lower-energy configuration under real forces,
the same way a real reconstructing surface would? Both the joint-HMC test
and the plain-MC test below start two substrate atoms stretched well away
from the LJ potential's known equilibrium separation and confirm they
relax toward it -- not just "doesn't crash".
"""

import numpy as np
import pytest

from gcmc_re import LJCalculator, LJSystem, ThermoState, Replica


# ---------------------------------------------------------------------------
# full_configuration() / bookkeeping
# ---------------------------------------------------------------------------
def test_full_configuration_orders_frozen_substrate_mobile():
    calc = LJCalculator()
    system = LJSystem(10.0, calc, positions=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    system.frozen_positions = np.array([[0.0, 0.0, 0.0]])
    system.frozen_species = np.array(["F"])
    system.substrate_positions = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    system.substrate_species = np.array(["S", "S"])

    pos, sp = system.full_configuration()
    assert pos.shape == (5, 3)
    assert list(sp) == ["F", "S", "S", "X", "X"]
    assert np.allclose(pos[0], [0, 0, 0])       # frozen first
    assert np.allclose(pos[1:3], [[3, 3, 3], [4, 4, 4]])  # substrate next
    assert np.allclose(pos[3:5], [[1, 1, 1], [2, 2, 2]])  # mobile last


def test_generic_displace_substrate_dU_energy_bookkeeping_stays_exact():
    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.lj import LennardJones as ASELennardJones
    from gcmc_re import ASECalculatorAdapter

    calc = ASECalculatorAdapter(
        ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
        cell=[10.0, 10.0, 10.0], pbc=(True, True, True))
    rng = np.random.default_rng(3)
    system = LJSystem(10.0, calc, positions=rng.random((3, 3)) * 10.0)
    system.substrate_positions = rng.random((4, 3)) * 10.0
    system.substrate_species = np.array(["X"] * 4)

    state = ThermoState(temperature=1.0, mu=-3.0, label=(0, 0))
    replica = Replica(state, system, rng, max_disp=0.2)
    for _ in range(200):
        replica._move_displace_substrate()

    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)


def test_substrate_masses_and_mass_of_species():
    calc = LJCalculator()
    system = LJSystem(10.0, calc)
    system.substrate_positions = np.zeros((3, 3))
    system.substrate_species = np.array(["Fe", "Fe", "H"])
    system.mass_of_species = {"H": 1.008}
    masses = system.substrate_masses()
    assert masses.shape == (3,)
    assert masses[0] == pytest.approx(55.845, abs=0.01)  # real Fe mass, via ASE data
    assert masses[2] == pytest.approx(1.008)             # explicit override


# ---------------------------------------------------------------------------
# Forces on substrate atoms (ASECalculatorAdapter)
# ---------------------------------------------------------------------------
def test_ase_calculator_adapter_substrate_forces_match_numerical_gradient():
    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.lj import LennardJones as ASELennardJones
    from gcmc_re import ASECalculatorAdapter

    calc = ASECalculatorAdapter(
        ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
        cell=[10.0, 10.0, 10.0], pbc=(True, True, True))
    rng = np.random.default_rng(5)
    system = LJSystem(10.0, calc, positions=rng.random((3, 3)) * 10.0)
    system.substrate_positions = rng.random((4, 3)) * 10.0
    system.substrate_species = np.array(["X"] * 4)

    F = calc.forces(system)
    assert F.shape == (7, 3)  # substrate (4) + mobile (3), substrate first

    i_sub = 1
    h = 1e-6
    F_numeric = np.zeros(3)
    for d in range(3):
        saved = system.substrate_positions[i_sub, d]
        system.substrate_positions[i_sub, d] = saved + h
        u_plus = system.total_energy()
        system.substrate_positions[i_sub, d] = saved - h
        u_minus = system.total_energy()
        system.substrate_positions[i_sub, d] = saved
        F_numeric[d] = -(u_plus - u_minus) / (2 * h)

    assert np.allclose(F[i_sub], F_numeric, atol=1e-5)


def test_lj_calculator_with_substrate_and_hmc_fails_clearly_not_silently():
    """LJCalculator doesn't account for substrate atoms at all (documented
    limitation). Combining it with substrate atoms + HMC must fail loudly
    and clearly, not silently give wrong physics."""
    rng = np.random.default_rng(0)
    system = LJSystem(10.0, LJCalculator(), positions=rng.random((3, 3)) * 10.0)
    system.substrate_positions = rng.random((2, 3)) * 10.0
    system.substrate_species = np.array(["X", "X"])
    state = ThermoState(1.0, -3.0, (0, 0))
    replica = Replica(state, system, rng, hmc_enabled=True)

    with pytest.raises(ValueError, match="substrate"):
        replica._move_hmc()


# ---------------------------------------------------------------------------
# The check that actually matters: substrate atoms genuinely relax under
# real forces, via both mechanisms (joint HMC, and plain MC).
# ---------------------------------------------------------------------------
def _stretched_two_atom_system(r_start, mass=1.0):
    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.lj import LennardJones as ASELennardJones
    from gcmc_re import ASECalculatorAdapter

    calc = ASECalculatorAdapter(
        ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
        cell=[10.0, 10.0, 10.0], pbc=(True, True, True))
    system = LJSystem(10.0, calc)
    system.substrate_positions = np.array([[5.0, 5.0, 5.0], [5.0 + r_start, 5.0, 5.0]])
    system.substrate_species = np.array(["X", "X"])
    system.mass_of_species = {"X": mass}
    return system


R_NATURAL = 2 ** (1 / 6)  # LJ minimum separation for sigma=1
R_START = 1.8             # deliberately far from equilibrium


def test_joint_hmc_relaxes_stretched_substrate_toward_lj_minimum():
    system = _stretched_two_atom_system(R_START, mass=10.0)
    state = ThermoState(temperature=0.05, mu=-100.0, label=(0, 0))  # low T: relax, don't just diffuse
    rng = np.random.default_rng(7)
    replica = Replica(state, system, rng, hmc_enabled=True, hmc_leapfrog_steps=15, hmc_dt=0.02)

    initial_sep = R_START
    for _ in range(200):
        replica._move_hmc()
    final_sep = np.linalg.norm(system.substrate_positions[1] - system.substrate_positions[0])

    assert abs(final_sep - R_NATURAL) < abs(initial_sep - R_NATURAL) / 3  # substantially closer
    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)


def test_plain_mc_substrate_moves_also_relax_stretched_substrate():
    system = _stretched_two_atom_system(R_START)
    state = ThermoState(temperature=0.05, mu=-100.0, label=(0, 0))
    rng = np.random.default_rng(8)
    replica = Replica(state, system, rng, max_disp=0.1, hmc_enabled=False)

    for _ in range(1500):
        replica._move_displace_substrate()
    final_sep = np.linalg.norm(system.substrate_positions[1] - system.substrate_positions[0])

    assert abs(final_sep - R_NATURAL) < abs(R_START - R_NATURAL) / 3
    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)


# ---------------------------------------------------------------------------
# Engine-level wiring: p_displace_substrate reaches Replica.sweep via run()
# ---------------------------------------------------------------------------
def test_engine_run_forwards_p_displace_substrate():
    from gcmc_re import ReplicaExchangeGCMC

    ase = pytest.importorskip("ase", reason="pip install ase to run this test")
    from ase.calculators.lj import LennardJones as ASELennardJones
    from gcmc_re import ASECalculatorAdapter

    def make_system(rng):
        calc = ASECalculatorAdapter(
            ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
            cell=[10.0, 10.0, 10.0], pbc=(True, True, True))
        system = LJSystem(10.0, calc)
        system.substrate_positions = np.array([[5.0, 5.0, 5.0], [5.0 + R_START, 5.0, 5.0]])
        system.substrate_species = np.array(["X", "X"])
        return system

    with ReplicaExchangeGCMC([0.1], [-100.0], make_system, seed=1, max_disp=0.1,
                              n_workers=1) as sim:
        sim.run(n_cycles=30, sweeps_per_cycle=20, record_every=5, verbose=False,
                 p_displace=0.0, p_insert=0.0, p_delete=0.0, p_displace_substrate=1.0)

    replica = sim.grid[(0, 0)]
    assert replica.n_attempts["displace_substrate"] > 0
    sep = np.linalg.norm(replica.system.substrate_positions[1] - replica.system.substrate_positions[0])
    assert abs(sep - R_NATURAL) < abs(R_START - R_NATURAL) / 2
