"""
The core correctness guarantee of this package: the generic
clone-and-recompute path (used for any Calculator with
supports_incremental=False, e.g. ASECalculatorAdapter wrapping EAM or an
MLIP) must give exactly the same energies and move-dUs as the fast native
LJCalculator path, on the same configuration. Run this after touching
systems.py or calculators.py.
"""

import numpy as np
import pytest

from gcmc_re import LJCalculator, LJSystem, ASECalculatorAdapter

ase = pytest.importorskip("ase", reason="pip install ase to run this test")
from ase.calculators.lj import LennardJones as ASELennardJones  # noqa: E402


@pytest.fixture
def matched_systems():
    rng = np.random.default_rng(0)
    L = 6.0
    sigma, epsilon, r_cut = 1.0, 1.0, 2.5
    positions = rng.random((30, 3)) * L

    sys_fast = LJSystem(L, LJCalculator(sigma, epsilon, r_cut), positions=positions.copy())
    ase_calc = ASECalculatorAdapter(
        ASELennardJones(sigma=sigma, epsilon=epsilon, rc=r_cut, smooth=False, shift=True),
        cell=[L, L, L], pbc=(True, True, True),
    )
    sys_generic = LJSystem(L, ase_calc, positions=positions.copy())
    return sys_fast, sys_generic, positions


def test_total_energy_matches(matched_systems):
    sys_fast, sys_generic, _ = matched_systems
    assert sys_fast.total_energy() == pytest.approx(sys_generic.total_energy(), abs=1e-8)


def test_displace_dU_matches(matched_systems):
    sys_fast, sys_generic, positions = matched_systems
    i = 5
    new_pos = positions[i] + np.array([0.3, -0.2, 0.1])
    du_fast = (sys_fast.pair_energy_of(new_pos, exclude=i)
               - sys_fast.pair_energy_of(positions[i], exclude=i))
    du_generic = sys_generic.generic_displace_dU(i, new_pos)
    assert du_fast == pytest.approx(du_generic, abs=1e-8)


def test_insert_dU_matches(matched_systems):
    sys_fast, sys_generic, positions = matched_systems
    rng = np.random.default_rng(1)
    trial_pos = rng.random(3) * 6.0
    du_fast = sys_fast.pair_energy_of(trial_pos)
    du_generic = sys_generic.generic_insert_dU(trial_pos)
    assert du_fast == pytest.approx(du_generic, abs=1e-8)


def test_delete_dU_matches(matched_systems):
    sys_fast, sys_generic, positions = matched_systems
    j = 10
    du_fast = -sys_fast.pair_energy_of(positions[j], exclude=j)
    du_generic = sys_generic.generic_delete_dU(j)
    assert du_fast == pytest.approx(du_generic, abs=1e-8)


def test_frozen_substrate_energy_bookkeeping_stays_exact():
    """Species + frozen-substrate plumbing: after many generic-path moves,
    the incrementally-tracked energy must still match a fresh total_energy()
    recomputation (no silent drift bug)."""
    from gcmc_re import Replica, SlabLJSystem, ThermoState

    rng = np.random.default_rng(3)
    nx, ny, a = 4, 4, 1.2
    xs, ys = np.meshgrid(np.arange(nx) * a, np.arange(ny) * a)
    frozen_pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)], axis=1)
    frozen_species = np.array(["Fe"] * frozen_pos.shape[0])

    Lx = Ly = nx * a
    Lz = 8.0
    calc = ASECalculatorAdapter(
        ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
        cell=[Lx, Ly, Lz], pbc=(True, True, False),
    )
    system = SlabLJSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                           frozen_positions=frozen_pos, frozen_species=frozen_species)
    state = ThermoState(temperature=1.2, mu=-2.0, label=(0, 0))
    replica = Replica(state, system, rng, max_disp=0.3)

    replica.sweep(300, p_displace=0.4, p_insert=0.3, p_delete=0.3)

    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)
    pos, sp = system.full_configuration()
    assert (sp == "Fe").sum() == frozen_pos.shape[0]
    assert pos.shape[0] == frozen_pos.shape[0] + system.n


def test_emt_au_h_smoke():
    """Not a physics check (see examples/05_emt_metal_surface.py for the
    honest caveats about EMT+H) -- just confirms the real-units plumbing
    (a real multi-element ASE calculator, a built ase.build.fcc111 slab as
    a frozen substrate, insertion_z_range, lambda_cubed) actually runs
    end to end without error, fast (a handful of moves, not a full sim)."""
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    from gcmc_re import (Replica, SlabLJSystem, ThermoState,
                          kelvin_to_eV, thermal_wavelength_cubed_angstrom3)

    slab = fcc111("Au", size=(2, 2, 2), vacuum=6.0, orthogonal=True, periodic=False)
    Lx, Ly, Lz = slab.cell[0, 0], slab.cell[1, 1], slab.cell[2, 2]
    top_z = slab.positions[:, 2].max()

    calc = ASECalculatorAdapter(EMT(), cell=[Lx, Ly, Lz], pbc=(True, True, False))
    system = SlabLJSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                           frozen_positions=slab.positions.copy(),
                           frozen_species=np.array(slab.get_chemical_symbols()),
                           insertion_z_range=(top_z - 0.5, top_z + 5.0))

    T_K = 400.0
    state = ThermoState(temperature=kelvin_to_eV(T_K), mu=0.3, label=(0, 0),
                         lambda_cubed=thermal_wavelength_cubed_angstrom3(T_K, 1.008))
    rng = np.random.default_rng(0)
    replica = Replica(state, system, rng, max_disp=0.3)
    replica.sweep(6, p_displace=0.4, p_insert=0.3, p_delete=0.3)

    assert replica.energy == pytest.approx(system.total_energy(), abs=1e-6)


def test_ase_calculator_adapter_accepts_class_as_factory():
    """A bare class (no constructor args needed) works as a factory: lazily
    constructed on first use, exactly like passing a live instance for
    single-process use."""
    from ase.calculators.emt import EMT

    calc = ASECalculatorAdapter(EMT, cell=[10.0, 10.0, 10.0], pbc=(True, True, False))
    assert calc._calc_instance is None  # not yet constructed
    instance = calc.ase_calculator
    assert instance is not None
    assert calc.ase_calculator is instance  # cached, not rebuilt every access


def test_ase_calculator_adapter_accepts_functools_partial_as_factory():
    import functools

    from ase.calculators.emt import EMT

    calc = ASECalculatorAdapter(functools.partial(EMT), cell=[10.0, 10.0, 10.0],
                                 pbc=(True, True, False))
    assert calc.ase_calculator is not None


def test_ase_calculator_adapter_factory_survives_pickling_after_use():
    """The specific bug this guards against: EMT builds an unpicklable
    defaultdict-with-a-lambda the first time it's actually used (on
    initialize()). A factory-built adapter must still pickle fine even
    after that has happened, because it never pickles the live instance."""
    import pickle

    from ase import Atoms
    from ase.calculators.emt import EMT

    calc = ASECalculatorAdapter(EMT, cell=[10.0, 10.0, 10.0], pbc=(True, True, False))
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=False)
    atoms.calc = calc.ase_calculator
    atoms.get_potential_energy()  # triggers EMT.initialize() -> unpicklable state, if any

    restored = pickle.loads(pickle.dumps(calc))
    assert restored._calc_instance is None  # fresh, not carrying the old (unpicklable) state


def test_ase_calculator_adapter_live_instance_raises_clear_error_on_pickle():
    """A live instance is fine for single-process use but must fail loudly
    (not with a cryptic AttributeError from deep inside concurrent.futures)
    if something tries to pickle it for multiprocessing."""
    import pickle

    from ase import Atoms
    from ase.calculators.emt import EMT

    calc = ASECalculatorAdapter(EMT(), cell=[10.0, 10.0, 10.0], pbc=(True, True, False))
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=False)
    atoms.calc = calc.ase_calculator
    atoms.get_potential_energy()

    with pytest.raises(TypeError, match="factory"):
        pickle.dumps(calc)
