"""
LammpsCalculator: LAMMPS as energy/force engine and native MD engine.

Checked against ASE's own EAM implementation on a synthetic (physically
meaningless) Fe-H EAM written at test time, so no potential file is needed.
"""

import pickle

import numpy as np
import pytest

lammps = pytest.importorskip("lammps", reason="LAMMPS Python module not installed")
pytest.importorskip("ase", reason="pip install ase")

from gcmc_re import ReplicaExchangeGCMC, SlabLJSystem, kelvin_to_eV  # noqa: E402
from gcmc_re.lammps_backend import LammpsCalculator  # noqa: E402
from gcmc_re.replica import Replica  # noqa: E402
from gcmc_re.state import ThermoState  # noqa: E402


def _write_toy_eam(path):
    rc, nr, nrho, drho = 5.0, 2000, 2000, 0.02
    dr = rc / nr
    r = np.arange(nr) * dr
    d = np.arange(nrho) * drho
    cut = np.where(r < rc, (1 - (r / rc) ** 4) ** 2, 0.0)

    def morse(D, a, r0):
        return D * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0))) * cut

    rho = {'Fe': np.exp(-1.2 * r) * cut, 'H': 0.5 * np.exp(-1.5 * r) * cut}
    F = {'Fe': -0.5 * np.sqrt(d), 'H': -0.3 * np.sqrt(d)}
    phi = [morse(0.4, 1.4, 2.6), morse(0.6, 1.6, 1.7), morse(4.7, 1.9, 0.74)]
    with open(path, 'w') as f:
        f.write("toy Fe-H EAM for tests\n\n\n2 Fe H\n")
        f.write(f"{nrho} {drho} {nr} {dr} {rc}\n")
        for el, Z, m in (('Fe', 26, 55.845), ('H', 1, 1.008)):
            f.write(f"{Z} {m} 2.866 bcc\n")
            np.savetxt(f, F[el].reshape(-1, 5))
            np.savetxt(f, rho[el].reshape(-1, 5))
        for p in phi:
            np.savetxt(f, (r * p).reshape(-1, 5))


@pytest.fixture(scope="module")
def eam_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("eam") / "toy_FeH.eam.alloy"
    _write_toy_eam(str(path))
    return str(path)


def _slab(calc, hexagonal=False, n_h=6, seed=0):
    rng = np.random.default_rng(seed)
    a = 2.62
    if hexagonal:
        L = 3 * a
        cell_xy = np.array([[L, 0.0], [L / 2, L * np.sqrt(3) / 2]])
        frac = np.array([[i / 3, j / 3] for i in range(3) for j in range(3)])
        layer = frac @ cell_xy
        kw = dict(cell_xy=cell_xy)
        Lx = Ly = None
    else:
        Lx, Ly = 3 * a, 3 * a
        layer = np.array([[i * a, j * a] for i in range(3) for j in range(3)])
        kw = {}
    # frozen block below z = 0 (as in the examples), one displaceable layer above
    frozen = np.vstack([np.column_stack([layer + 0.5 * a * k, np.full(9, -a * k / 2)])
                        for k in (1, 0)])
    substrate = np.column_stack([layer + 0.25 * a, np.full(9, a / 2)])
    span = cell_xy if hexagonal else np.diag([Lx, Ly])
    h = np.column_stack([rng.random((n_h, 2)) @ span, a + 0.8 + 2.0 * rng.random(n_h)])
    return SlabLJSystem(Lx, Ly, 8.0, calc, exchange_species="H",
                        frozen_positions=frozen, frozen_species=np.array(["Fe"] * len(frozen)),
                        substrate_positions=substrate,
                        substrate_species=np.array(["Fe"] * len(substrate)),
                        positions=h, **kw)


def _ase_reference(system, eam_file):
    from ase import Atoms
    from ase.calculators.eam import EAM
    pos, sp = system.full_configuration()
    cell = np.asarray(system.cell) if np.ndim(system.cell) == 2 else np.diag(system.cell)
    atoms = Atoms(list(sp), positions=pos, cell=cell, pbc=system.pbc)
    atoms.calc = EAM(potential=eam_file)
    n_fr = system.frozen_positions.shape[0]
    return atoms.get_potential_energy(), atoms.get_forces()[n_fr:]


def _lmp_calc(eam_file, **kw):
    return LammpsCalculator(["pair_style eam/alloy", f"pair_coeff * * {eam_file} Fe H"],
                            ["Fe", "H"], **kw)


def _bcc110_slab(calc, n_h=4, seed=0):
    """A physical geometry: bcc(110) slab (frozen block below z = 0), H >= ~1.4 A
    from Fe, so tabulated-EAM interpolation differences stay negligible."""
    from ase.build import bcc110
    slab = bcc110("Fe", size=(2, 2, 4), a=2.62, vacuum=None, orthogonal=True)
    pos = slab.positions.copy()
    layers = np.unique(np.round(pos[:, 2], 4))
    pos[:, 2] -= layers[1]
    frozen = pos[:, 2] <= 1e-6
    Lx, Ly = slab.cell[0, 0], slab.cell[1, 1]
    top = pos[:, 2].max()
    rng = np.random.default_rng(seed)
    h = np.column_stack([rng.random(n_h) * Lx, rng.random(n_h) * Ly, top + 1.6 + rng.random(n_h)])
    return SlabLJSystem(Lx, Ly, top + 5.0, calc, exchange_species="H",
                        frozen_positions=pos[frozen], frozen_species=np.array(["Fe"] * frozen.sum()),
                        substrate_positions=pos[~frozen],
                        substrate_species=np.array(["Fe"] * (~frozen).sum()), positions=h)


def test_energy_and_forces_match_ase_eam(eam_file):
    system = _bcc110_slab(_lmp_calc(eam_file))
    e_ref, f_ref = _ase_reference(system, eam_file)
    assert system.calculator.total_energy(system) == pytest.approx(e_ref, abs=1e-6)
    assert np.allclose(system.calculator.forces(system), f_ref, atol=1e-5)


@pytest.mark.parametrize("hexagonal", [False, True], ids=["orthogonal", "hexagonal"])
def test_box_handling_matches_pair_lj_exactly(hexagonal):
    """Analytic LJ (no tables): LAMMPS lj/cut (shifted) must equal the native
    PairLJCalculator to round-off, incl. a triclinic hexagonal cell and frozen
    atoms below z = 0."""
    from gcmc_re import PairLJCalculator
    # cutoffs (2.5 sigma <= 3.25 A) inside the cells' minimum-image radius (>= 3.40 A)
    params = {("Fe", "H"): (0.66, 1.3), ("H", "H"): (0.01, 1.2), ("Fe", "Fe"): (0.3, 1.3)}
    lmp_calc = LammpsCalculator(
        ["pair_style lj/cut 3.25", "pair_modify shift yes",
         f"pair_coeff 1 1 0.3 1.3 {2.5 * 1.3}", f"pair_coeff 1 2 0.66 1.3 {2.5 * 1.3}",
         f"pair_coeff 2 2 0.01 1.2 {2.5 * 1.2}"], ["Fe", "H"])
    a = _slab(lmp_calc, hexagonal=hexagonal)
    b = _slab(PairLJCalculator(params), hexagonal=hexagonal)
    assert lmp_calc.total_energy(a) == pytest.approx(b.calculator.total_energy(b), rel=1e-10, abs=1e-10)
    assert np.allclose(lmp_calc.forces(a), b.calculator.forces(b), atol=1e-8)


def test_generic_moves_consistent_with_total_energies(eam_file):
    system = _slab(_lmp_calc(eam_file))
    e0 = system.total_energy()
    trial = np.array([1.0, 2.0, 3.5])
    du = system.generic_insert_dU(trial)
    system.apply_insert(trial)
    assert system.total_energy() - e0 == pytest.approx(du, abs=1e-8)
    du_del = system.generic_delete_dU(system.n - 1)
    system.apply_delete(system.n - 1)
    assert system.total_energy() - (e0 + du) == pytest.approx(du_del, abs=1e-8)
    # trial position outside the periodic cell in x, y is wrapped, not lost
    i, new = 0, system.positions[0] + np.array([30.0, -20.0, 0.1])
    du_mv = system.generic_displace_dU(i, new)
    system.apply_move(i, new)
    assert system.total_energy() - e0 == pytest.approx(du_mv, abs=1e-8)


def test_native_md_segment(eam_file):
    calc = _lmp_calc(eam_file)
    system = _slab(calc)
    T = kelvin_to_eV(600.0)
    rep = Replica(ThermoState(T, -2.0, (0, 0)), system, np.random.default_rng(3))
    frozen0 = system.frozen_positions.copy()
    sub0 = system.substrate_positions.copy()
    rep.diffuse('md', n=200, timestep_fs=0.5, taut_fs=20.0)
    assert np.array_equal(system.frozen_positions, frozen0)
    assert not np.allclose(system.substrate_positions, sub0)
    z = np.concatenate([system.substrate_positions[:, 2], system.positions[:, 2]])
    assert np.all((z > 0.0) & (z <= 8.0))
    assert rep.energy == pytest.approx(system.total_energy(), abs=1e-8)


def test_native_md_thermostat_reaches_temperature(eam_file):
    """Bussi thermostat: the time-averaged kinetic temperature of the moving
    atoms over a long segment is the target T."""
    calc = _lmp_calc(eam_file)
    system = _bcc110_slab(calc, n_h=0)
    T_K = 500.0
    calc.run_md(system, 10, kelvin_to_eV(T_K), 0.5, 20.0, seed=7)  # sets up the instance
    lmp = calc._lmp
    n_fr = calc._load(system)
    lmp.commands_list(["reset_timestep 0", f"group frozen id 1:{n_fr}",
                       "group mobile subtract all frozen",
                       f"velocity mobile create {T_K} 11 dist gaussian mom no rot no",
                       "timestep 0.0005", "fix a mobile nve",
                       f"fix b mobile temp/csvr {T_K} {T_K} 0.02 12",
                       "compute tm mobile temp", "fix ave all ave/time 1 4000 4000 c_tm",
                       "run 4000 post no"])
    t_avg = lmp.extract_fix("ave", 0, 0)
    assert t_avg == pytest.approx(T_K, rel=0.1)


def test_hard_core_and_md_revert(eam_file):
    calc = _lmp_calc(eam_file, min_distance=0.5)
    system = _slab(calc)
    near = system.frozen_positions[-1] + np.array([0.0, 0.0, 0.3])
    near[2] = 0.3  # inside the box, 0.3 A above a frozen atom
    assert np.isinf(system.generic_insert_dU(near))


def test_pickle_and_parallel_equals_serial(eam_file):
    calc = _lmp_calc(eam_file)
    clone = pickle.loads(pickle.dumps(_slab(calc)))
    assert clone.calculator._lmp is None
    assert clone.total_energy() == pytest.approx(_slab(_lmp_calc(eam_file)).total_energy())

    def run(n_workers):
        T = [kelvin_to_eV(400.0), kelvin_to_eV(600.0)]
        with ReplicaExchangeGCMC(T, [[-2.0], [-2.0]], lambda rng: _slab(_lmp_calc(eam_file), seed=1),
                                 seed=5, n_workers=n_workers) as sim:
            sim.run_regc(n_steps=12, x0=0.7, diffusion='md', diffusion_steps=5, record_every=2,
                         verbose=False, timestep_fs=0.5, taut_fs=20.0)
        return sim.samples
    serial, parallel = run(1), run(2)
    for k in serial:
        assert serial[k]['N'] == parallel[k]['N']
        assert np.allclose(serial[k]['U'], parallel[k]['U'], atol=1e-9)


def test_native_lj_refuses_cutoff_beyond_min_image_radius():
    from gcmc_re import LJCalculator, LJSystem, PairLJCalculator
    big = SlabLJSystem(None, None, 8.0, PairLJCalculator({("B", "B"): (0.01, 2.3)}),
                       exchange_species="B", cell_xy=np.array([[7.86, 0.0], [3.93, 6.81]]),
                       positions=[[1.0, 1.0, 2.0], [3.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="minimum-image radius"):
        big.total_energy()
    small_box = LJSystem(4.0, LJCalculator(1.0, 1.0, 2.5), positions=[[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError, match="minimum-image radius"):
        small_box.total_energy()


def test_lammps_instances_shared_within_a_process(eam_file):
    """Unpickled calculators (as in a worker process) reuse one LAMMPS
    instance per potential/box instead of reloading the potential."""
    a = _bcc110_slab(_lmp_calc(eam_file))
    b = pickle.loads(pickle.dumps(_bcc110_slab(_lmp_calc(eam_file), seed=3)))
    ea, eb = a.total_energy(), b.total_energy()
    assert a.calculator._lmp is b.calculator._lmp
    # interleaved evaluations stay correct (every call reloads its own atoms)
    assert a.total_energy() == pytest.approx(ea) and b.total_energy() == pytest.approx(eb)
    assert ea != pytest.approx(eb)
