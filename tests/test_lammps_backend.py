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

from regc import ReplicaExchangeGCMC, SlabLJSystem, kelvin_to_eV  # noqa: E402
from regc.lammps_backend import MVV2E_METAL, LammpsCalculator  # noqa: E402
from regc.replica import Replica  # noqa: E402
from regc.state import ThermoState  # noqa: E402


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
    from regc import PairLJCalculator
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


def _few_atoms_on_frozen_block(n_mobile):
    """Free particles (pair_style zero) above a frozen block, inside reflecting
    z-walls: no momentum is conserved, so all 3N degrees of freedom are thermal."""
    calc = LammpsCalculator(["pair_style zero 3.0", "pair_coeff * *"], ["Fe", "H"])
    a = 2.62
    layer = np.array([[i * a, j * a, 0.0] for i in range(3) for j in range(3)])
    rng = np.random.default_rng(0)
    h = np.column_stack([rng.random((n_mobile, 2)) * 3 * a, 2.0 + 3.0 * rng.random(n_mobile)])
    return SlabLJSystem(3 * a, 3 * a, 8.0, calc, exchange_species="H",
                        frozen_positions=layer, frozen_species=np.array(["Fe"] * 9),
                        positions=h)


@pytest.mark.parametrize("thermostat", ["bussi", "langevin"])
def test_native_md_kinetic_temperature_of_few_atoms(thermostat):
    """run_md's own velocities and thermostat give <KE> = (3N/2) kT, also for
    N = 2 moving atoms. ``velocity create`` (rescaled to exactly T) and the
    default 3N - 3 thermostat dof gave <KE> / (3N/2 kT) = 0.55 here."""
    system = _few_atoms_on_frozen_block(2)
    calc = system.calculator
    T = kelvin_to_eV(500.0)
    n_fr = system.frozen_positions.shape[0]
    ke = []
    for seg in range(300):
        calc.run_md(system, 40, T, 1.0, 20.0, seed=seg, thermostat=thermostat)
        v = calc._gather("v", n_fr + system.n)[n_fr:]
        m = calc._mass("H")
        ke.append(0.5 * m * MVV2E_METAL * np.sum(v ** 2))
    ratio = np.mean(ke) / (1.5 * system.n * T)
    assert ratio == pytest.approx(1.0, abs=0.12)


def test_native_md_with_a_single_moving_atom():
    """One moving atom over a frozen block: ``velocity create`` used to abort
    ("Attempting to rescale a 0.0 temperature")."""
    system = _few_atoms_on_frozen_block(1)
    rep = Replica(ThermoState(kelvin_to_eV(300.0), -2.0, (0, 0)), system,
                  np.random.default_rng(1))
    z0 = system.positions.copy()
    rep.diffuse('md', n=50, timestep_fs=1.0, taut_fs=20.0)
    assert not np.allclose(system.positions, z0)


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
    from regc import LJCalculator, LJSystem, PairLJCalculator
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


# -- choosing the LAMMPS library (lammps_name / REGC_LAMMPS_NAME) -------------------------

def test_resolve_lammps_name(monkeypatch):
    from regc.lammps_backend import resolve_lammps_name
    monkeypatch.delenv("REGC_LAMMPS_NAME", raising=False)
    assert resolve_lammps_name() == ""
    assert resolve_lammps_name("grace") == "grace"
    monkeypatch.setenv("REGC_LAMMPS_NAME", "grace")
    assert resolve_lammps_name() == "grace"
    assert resolve_lammps_name("") == ""          # an explicit name beats the variable


def test_default_library_by_name(eam_file, monkeypatch):
    """lammps_name="" and the variable set to "" both give the default library."""
    monkeypatch.setenv("REGC_LAMMPS_NAME", "")
    e_env = _bcc110_slab(_lmp_calc(eam_file)).total_energy()
    e_explicit = _bcc110_slab(_lmp_calc(eam_file, lammps_name="")).total_energy()
    assert e_env == pytest.approx(e_explicit)


def _in_fresh_process(code, env_name):
    """Run `code` in a new interpreter with REGC_LAMMPS_NAME=env_name (a process can
    hold only one LAMMPS library); returns the completed process."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, REGC_LAMMPS_NAME=env_name)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          timeout=300)


def test_missing_library_is_reported():
    res = _in_fresh_process("from regc import open_lammps; open_lammps()", "no_such_build")
    assert res.returncode != 0 and "liblammps_no_such_build" in res.stderr


def test_second_library_in_one_process_is_refused(eam_file):
    _bcc110_slab(_lmp_calc(eam_file, lammps_name="")).total_energy()   # default loaded
    with pytest.raises(RuntimeError, match="one lammps_name per process"):
        _bcc110_slab(_lmp_calc(eam_file, lammps_name="other_build")).total_energy()


def test_old_pickles_without_lammps_name_load(eam_file):
    calc = _lmp_calc(eam_file)
    del calc.__dict__["lammps_name"]                 # as pickled before the option existed
    calc = pickle.loads(pickle.dumps(calc))
    assert np.isfinite(_bcc110_slab(calc).total_energy())


_EAM_ENERGY_FORCES = """
import sys, warnings
import numpy as np
sys.path.insert(0, {testdir!r})
warnings.simplefilter("ignore")
from test_lammps_backend import _bcc110_slab, _lmp_calc
slab = _bcc110_slab(_lmp_calc({eam!r}))
np.save({out!r}, np.concatenate([[slab.total_energy()], slab.calculator.forces(slab).ravel()]))
"""


def test_named_build_matches_default_for_eam(eam_file, tmp_path):
    """A second LAMMPS build (hpc/build_lammps.sh: liblammps_grace.so), if installed,
    gives the same EAM energies and forces as the default one (each in its own process)."""
    import os
    results = {}
    for name in ("", "grace"):
        out = str(tmp_path / f"ef_{name or 'default'}.npy")
        res = _in_fresh_process(_EAM_ENERGY_FORCES.format(
            testdir=os.path.dirname(__file__), eam=eam_file, out=out), name)
        if res.returncode != 0:
            if name and "liblammps_grace" in res.stderr:
                pytest.skip("no liblammps_grace.so (hpc/build_lammps.sh)")
            raise AssertionError(res.stderr[-2000:])
        results[name] = np.load(out)
    np.testing.assert_allclose(results["grace"], results[""], atol=1e-8)


def test_fast_path_equals_full_setup_over_a_random_walk(eam_file, monkeypatch):
    """Same atoms: positions scattered + `run 1 pre no`; after insert/delete/MD: a full
    reload. Every energy must equal a full setup on a separate LAMMPS instance, including
    moves across the periodic boundary and beyond half the neighbour skin."""
    from regc.lammps_backend import _LOADED
    system = _bcc110_slab(_lmp_calc(eam_file), n_h=4)
    calc = system.calculator
    ref = _lmp_calc(eam_file, z_padding=5.0001)          # another box: another instance

    def reference():
        ref._load(system)
        ref._lmp.command("run 0 post no")
        return float(ref._lmp.get_thermo("pe"))
    rng = np.random.default_rng(3)
    Lx = system.cell[0]
    n_fast = 0
    for k in range(400):
        r = rng.random()
        if r < 0.55 and system.n:
            i = rng.integers(system.n)
            system.apply_move(i, system.positions[i] + rng.uniform(-0.2, 0.2, 3))
        elif r < 0.65 and system.n:
            i = rng.integers(system.n)
            system.apply_move(i, system.positions[i] + rng.uniform(-1.5, 1.5, 3))
        elif r < 0.72 and system.n:
            i = rng.integers(system.n)
            p = system.positions[i].copy()
            p[0] = Lx + 0.02                              # wraps to the other side
            system.apply_move(i, p)
        elif r < 0.82:
            system.apply_insert(system.random_insertion_position(rng))
        elif r < 0.92 and system.n > 1:
            system.apply_delete(rng.integers(system.n))
        else:
            new = calc.run_md(system, 10, kelvin_to_eV(500.0), 0.5, 20.0, seed=k)
            for j in range(system.n_substrate):
                system.apply_move_substrate(j, new[j])
            for j in range(system.n):
                system.apply_move(j, new[system.n_substrate + j])
            assert calc.energy_after_md(system) == pytest.approx(reference(), abs=1e-9)
        calc.__dict__.pop("_energy_cache", None)          # test the evaluation itself
        e = calc.total_energy(system)
        n_fast += "pre no" in calc._run0
        assert e == pytest.approx(reference(), abs=1e-9)
    assert n_fast > 150
    # the switch restores the full path, and the cache returns what it stored
    monkeypatch.setenv("REGC_LAMMPS_FAST", "0")
    assert calc.total_energy(system) == pytest.approx(reference(), abs=1e-9)
    assert "pre no" not in calc._run0
    monkeypatch.delenv("REGC_LAMMPS_FAST")
    e1 = calc.total_energy(system)
    assert calc.total_energy(system) == e1


def test_md_end_energy_matches_a_full_evaluation(eam_file):
    calc = _lmp_calc(eam_file)
    system = _slab(calc)
    rep = Replica(ThermoState(kelvin_to_eV(600.0), -2.0, (0, 0)), system, np.random.default_rng(4))
    rep.diffuse('md', n=50, timestep_fs=0.5, taut_fs=20.0)
    fresh = _lmp_calc(eam_file, z_padding=5.0002)
    fresh._load(system)
    fresh._lmp.command("run 0 post no")
    assert rep.energy == pytest.approx(float(fresh._lmp.get_thermo("pe")) +
                                       system.external_energy_total(), abs=1e-8)
