"""
ClusterSystem: a nanoparticle in a non-periodic box (hard walls on all faces),
with insertions in the whole box or a spherical shell around the particle.
"""

import json

import numpy as np
import pytest

from regc import ClusterSystem
from regc.calculators import Calculator
from regc.replica import Replica
from regc.state import ThermoState


class ZeroCalculator(Calculator):
    """Ideal gas: every configuration has zero interaction energy."""

    def total_energy(self, system) -> float:
        return 0.0


# -- geometry ---------------------------------------------------------------------------

def test_box_walls_volume_and_periodicity():
    s = ClusterSystem([10.0, 12.0, 14.0], ZeroCalculator())
    assert s.pbc == (False, False, False)
    assert s.cell == (10.0, 12.0, 14.0)
    assert s.volume == pytest.approx(10 * 12 * 14)
    assert s.md_wall_bounds() == [(0.0, 10.0), (0.0, 12.0), (0.0, 14.0)]
    assert s.external_energy(np.array([5.0, 6.0, 7.0])) == 0.0
    for bad in ([0.0, 6, 7], [5, 12.1, 7], [5, 6, -1.0]):
        assert s.external_energy(np.array(bad, dtype=float)) == np.inf
    np.testing.assert_array_equal(s._min_image(np.array([[9.0, -11.0, 13.0]])),
                                  [[9.0, -11.0, 13.0]])           # no periodic images
    assert s.min_image_radius == np.inf


def test_shell_insertions_are_uniform_in_the_shell():
    s = ClusterSystem(20.0, ZeroCalculator(), insertion_shell=(4.0, 8.0))
    assert s.volume == pytest.approx(4 / 3 * np.pi * (8 ** 3 - 4 ** 3))
    rng = np.random.default_rng(0)
    r = np.array([np.linalg.norm(s.random_insertion_position(rng) - 10.0) for _ in range(20000)])
    assert r.min() >= 4.0 and r.max() < 8.0
    # uniform in volume: P(r < r0) = (r0^3 - r_in^3) / (r_out^3 - r_in^3)
    assert np.mean(r < 6.0) == pytest.approx((6 ** 3 - 4 ** 3) / (8 ** 3 - 4 ** 3), abs=0.01)


def test_exchange_zone_is_the_shell():
    s = ClusterSystem(20.0, ZeroCalculator(), exchange_species="O", insertion_shell=(4.0, 8.0),
                      positions=[[10, 10, 15], [10, 10, 19], [10, 10, 10.5]])
    np.testing.assert_array_equal(s.exchange_zone_indices(), [0])
    assert len(ClusterSystem(20.0, ZeroCalculator(), positions=[[1, 1, 1], [3, 3, 3]])
               .exchange_zone_indices()) == 2


def test_shell_must_fit_in_the_box():
    with pytest.raises(ValueError, match="does not fit"):
        ClusterSystem(10.0, ZeroCalculator(), insertion_shell=(1.0, 5.5))
    with pytest.raises(ValueError, match="r_in < r_out"):
        ClusterSystem(10.0, ZeroCalculator(), insertion_shell=(3.0, 3.0))


def test_shell_ideal_gas_is_exact():
    """Insertions and deletions restricted to the shell, displacements over the
    whole box: the exact result is the ideal gas in the whole box, of which the
    fraction V_shell / V_box is in the shell (cf. the slab's insertion_z_range)."""
    L, shell = 4.0, (0.6, 1.8)
    mu, T = np.log(0.5), 1.0
    system = ClusterSystem(L, ZeroCalculator(), insertion_shell=shell)
    replica = Replica(ThermoState(T, mu, label=(0, 0)), system, np.random.default_rng(5),
                      max_disp=1.0)
    n_total, n_zone = [], []
    replica.sweep(20000, p_displace=0.8, p_insert=0.1, p_delete=0.1)  # burn-in
    for _ in range(3000):
        replica.sweep(100, p_displace=0.8, p_insert=0.1, p_delete=0.1)
        n_total.append(system.n)
        n_zone.append(len(system.exchange_zone_indices()))
    density = np.exp(mu / T)
    assert np.mean(n_total) == pytest.approx(density * L ** 3, rel=0.07)
    assert np.mean(n_zone) == pytest.approx(density * system.volume, rel=0.1)
    assert np.all(system.positions > 0) and np.all(system.positions <= L)


# -- cluster layout -----------------------------------------------------------------------

def test_pt55_icosahedron_layout():
    pytest.importorskip("ase")
    from ase.cluster import Icosahedron
    from regc import cluster_geometry_from_atoms
    g = cluster_geometry_from_atoms(Icosahedron("Pt", 3, latticeconstant=3.92), headroom=6.0)
    assert g.n_atoms == 55 and g.n_surface == 42          # 12 vertices + 30 edge atoms
    assert sorted(np.bincount(g.coordination)[[6, 8, 12]]) == [12, 13, 30]
    assert len(g.frozen_positions) == 1                   # the centre atom
    np.testing.assert_allclose(g.frozen_positions[0], g.center, atol=1e-6)
    assert g.box == pytest.approx(2 * (g.radius + 6.0))
    assert np.all(np.linalg.norm(g.substrate_positions - g.center, axis=1) <= g.radius + 1e-9)
    core = cluster_geometry_from_atoms(Icosahedron("Pt", 3, latticeconstant=3.92), frozen="core")
    assert len(core.frozen_positions) == 13 and len(core.substrate_positions) == 42
    assert len(cluster_geometry_from_atoms(Icosahedron("Pt", 3), frozen=0).frozen_positions) == 0
    assert g.shell(-1.0, 3.0) == pytest.approx((g.radius - 1.0, g.radius + 3.0))
    with pytest.raises(ValueError, match="half the box"):
        g.shell(0.0, 7.0)
    system = g.make_system(ZeroCalculator(), "O", insertion_shell=g.shell(0.5, 3.0))
    assert isinstance(system, ClusterSystem) and system.n_substrate == 54


# -- calculators on a cluster ---------------------------------------------------------------

def _small_cluster(calc, cell_species="Fe"):
    """A 13-atom icosahedron (1 frozen) with two H atoms nearby, in its box."""
    from ase.cluster import Icosahedron
    from regc import cluster_geometry_from_atoms
    g = cluster_geometry_from_atoms(Icosahedron(cell_species, 2, latticeconstant=3.6),
                                    headroom=5.0)
    s = g.make_system(calc, "H", insertion_shell=g.shell(0.5, 3.0))
    for d in ([1.0, 0.2, 0.1], [-0.3, -1.0, 0.4]):
        d = np.asarray(d) / np.linalg.norm(d)
        s.apply_insert(g.center + (g.radius + 1.6) * d)
    return g, s


def test_emt_md_stays_inside_the_box():
    """ASE MD reflects at all six faces of the box."""
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT
    from regc.calculators import ASECalculatorAdapter
    g, s = _small_cluster(None, "Pt")
    s.calculator = ASECalculatorAdapter(EMT, cell=s.cell, pbc=s.pbc)
    rep = Replica(ThermoState(0.3, -1.0, label=(0, 0)), s, np.random.default_rng(2))
    rep.energy = s.total_energy()
    rep.diffuse("md", 30, timestep_fs=2.0)
    pos, _ = s.full_configuration()
    assert np.isfinite(rep.energy)
    assert np.all(pos > 0) and np.all(pos <= g.box)
    np.testing.assert_allclose(s.frozen_positions[0], g.center)   # the centre atom stays


@pytest.fixture(scope="module")
def eam_file(tmp_path_factory):
    pytest.importorskip("lammps")
    from test_lammps_backend import _write_toy_eam
    path = tmp_path_factory.mktemp("eam") / "toy_FeH.eam.alloy"
    _write_toy_eam(str(path))
    return str(path)


def test_lammps_matches_ase_eam_on_a_cluster(eam_file):
    """LAMMPS with boundary f f f (box padded around the cluster) gives ASE's
    non-periodic EAM energy and forces."""
    pytest.importorskip("ase")
    from ase.calculators.eam import EAM
    from regc import LammpsCalculator
    from regc.calculators import ASECalculatorAdapter
    lmp = LammpsCalculator(["pair_style eam/alloy", f"pair_coeff * * {eam_file} Fe H"],
                           ["Fe", "H"])
    g, s = _small_cluster(lmp)
    ref = ASECalculatorAdapter(EAM(potential=eam_file), cell=s.cell, pbc=s.pbc)
    e_lmp, f_lmp = lmp.total_energy(s), lmp.forces(s)
    assert e_lmp == pytest.approx(ref.total_energy(s), abs=2e-4)
    np.testing.assert_allclose(f_lmp, ref.forces(s), atol=2e-3)
    assert lmp._lo == pytest.approx(np.full(3, -lmp.z_padding))   # walls at 0 and L, padded
    assert lmp._hi == pytest.approx(np.full(3, g.box + lmp.z_padding))


def test_lammps_md_reflects_at_all_faces(eam_file):
    from regc import LammpsCalculator
    lmp = LammpsCalculator(["pair_style eam/alloy", f"pair_coeff * * {eam_file} Fe H"],
                           ["Fe", "H"])
    g, s = _small_cluster(lmp)
    s.apply_insert(np.array([0.3, 0.3, g.box - 0.3]))   # a gas atom next to a corner
    new = lmp.run_md(s, 400, temperature_eV=0.2, timestep_fs=1.0, taut_fs=20.0, seed=4)
    assert np.all(new > 0) and np.all(new <= g.box)


def test_lammps_fully_periodic_box_uses_the_cell(eam_file):
    """Regression: with z periodic, the LAMMPS box spans the cell (it used to
    span the atoms' z-extent plus padding, giving the wrong periodic length)."""
    pytest.importorskip("ase")
    from ase.build import bulk
    from ase.calculators.eam import EAM
    from regc import LammpsCalculator, SlabSystem
    from regc.calculators import ASECalculatorAdapter

    class PeriodicBox(SlabSystem):
        @property
        def pbc(self):
            return (True, True, True)

        def md_z_bounds(self):
            return None

    atoms = bulk("Fe", "bcc", a=2.62, cubic=True).repeat((2, 2, 2))
    atoms.rattle(0.05, seed=1)
    L = atoms.cell[0, 0]
    lmp = LammpsCalculator(["pair_style eam/alloy", f"pair_coeff * * {eam_file} Fe H"],
                           ["Fe", "H"])
    s = PeriodicBox(L, L, L, lmp, exchange_species="H", substrate_positions=atoms.positions,
                    substrate_species=np.array(["Fe"] * len(atoms)),
                    positions=[[0.4 * L, 0.5 * L, 0.6 * L]])
    ref = ASECalculatorAdapter(EAM(potential=eam_file), cell=(L, L, L), pbc=(True, True, True))
    assert lmp.total_energy(s) == pytest.approx(ref.total_energy(s), abs=2e-4)
    assert lmp._hi[2] == pytest.approx(L)


# -- the config runner -------------------------------------------------------------------------

def _cluster_config(tmp_path, **gas):
    return {
        "structure": {"cluster": {"shape": "Icosahedron", "symbol": "Pt", "noshells": 2,
                                  "a": 3.92}, "headroom_angstrom": 5.0},
        "gas": {"species": "O", "insertion_shell_around_cluster": [0.5, 3.0], **gas},
        "calculator": {"type": "EMT"},
        "reservoir": {"type": "diatomic", "d0": "self", "bond_length_angstrom": 1.2075,
                      "vibrational_wavenumber_cm": 1580.2},
        "replica_exchange": {"temperatures_K": [400, 600], "pressures": [1.0e-6, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 40, "x0": 0.9,
                             "diffusion": "mc", "diffusion_steps": 1, "record_every": 1,
                             "adsorbed_cutoff": 2.6, "seed": 0, "n_workers": 1},
        "phase_diagram": {"T_min_K": 400, "T_max_K": 600, "n_T": 3, "p_min": 1.0e-6,
                          "p_max": 1.0, "n_p": 3, "observable": "N_ads"},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_runner_builds_and_runs_a_cluster(tmp_path):
    pytest.importorskip("ase")
    pytest.importorskip("pymbar")
    pytest.importorskip("matplotlib")
    from regc.runner import run_from_config, setup_from_config
    cfg = _cluster_config(tmp_path)
    setup = setup_from_config(cfg, verbose=False)
    g = setup["geometry"]
    assert g.n_atoms == 13 and g.n_surface == 12
    assert setup["insertion_shell"] == pytest.approx(g.shell(0.5, 3.0))
    system = setup["system_factory"](np.random.default_rng(0))
    assert isinstance(system, ClusterSystem) and system.calculator.pbc == (False, False, False)
    result = run_from_config(cfg, verbose=False)
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["geometry"] == "cluster" and meta["n_surface"] == 12 and meta["n_atoms"] == 13
    assert meta["area_angstrom2"] is None
    assert result["coverage_per"] == "surface_atom"        # the cluster default
    assert np.all(np.isfinite(result["theta_grid"]))


def test_runner_rejects_slab_keys_for_a_cluster(tmp_path):
    from regc.runner import validate_config
    cfg = _cluster_config(tmp_path)
    cfg["structure"]["n_frozen_layers"] = 2
    with pytest.raises(ValueError, match="n_frozen_layers"):
        validate_config(cfg)
    cfg = _cluster_config(tmp_path, insertion_z_range_above_surface=[0, 2])
    with pytest.raises(ValueError, match="insertion_z_range_above_surface"):
        validate_config(cfg)
    cfg = _cluster_config(tmp_path)
    cfg["phase_diagram"]["coverage_per"] = "area"
    with pytest.raises(ValueError, match="area needs a slab"):
        validate_config(cfg)
