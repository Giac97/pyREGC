"""
The fixed / thermostat / free substrate (`surface_slab`, ``structure.layers``):
layer bookkeeping, the partially thermostatted MD integrator (NVE where
there is no thermostat, the thermostat touching only the bath atoms), and the
YAML route.
"""

import numpy as np
import pytest

pytest.importorskip("ase", reason="pip install ase")

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator as ASECalc, all_changes  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402

from regc import (ASECalculatorAdapter, Replica, ThermoState, kelvin_to_eV,  # noqa: E402
                  slab_geometry_from_atoms, substrate_layers, surface_slab)
from regc.runner import setup_from_config  # noqa: E402


def test_surface_slab_regions():
    g = surface_slab("Pd", "fcc100", size=(3, 3), layers=(2, 1, 2), a=3.9)
    assert g.layers == (2, 1, 2)
    assert len(g.frozen_positions) == 18 and len(g.substrate_positions) == 27
    mask = g.thermostat_mask
    assert mask.sum() == 9
    z = g.substrate_positions[:, 2]
    assert z[mask].max() < z[~mask].min()               # the bath sits below the free layers
    assert "18 fixed + 9 thermostatted + 18 free" in g.summary() and "(2 / 1 / 2 layers)" in g.summary()
    system = g.make_system(ASECalculatorAdapter(EMT(), [3 * 3.9 / np.sqrt(2)] * 2 + [g.Lz]),
                           "H")
    assert np.array_equal(system.thermostat_mask, mask)


def test_substrate_layers_parsing():
    assert substrate_layers({"fixed": 2, "thermostat": 2, "free": 3}) == (2, 2, 3)
    assert substrate_layers([1, 1, 1]) == (1, 1, 1)
    for bad, msg in (({"fixed": 2, "thermostat": 2}, "missing"), ((2, 0, 2), "at least one"),
                     ((2, 2), "three"), ((2, 1.5, 2), "whole")):
        with pytest.raises(ValueError, match=msg):
            substrate_layers(bad)


def test_layer_counts_must_leave_a_free_layer():
    from ase.build import fcc100
    atoms = fcc100("Pd", size=(2, 2, 4), a=3.9)
    with pytest.raises(ValueError, match="free layer"):
        slab_geometry_from_atoms(atoms, n_frozen_layers=2, n_thermostat_layers=2)
    g = slab_geometry_from_atoms(atoms, n_frozen_layers=2)
    assert g.thermostat_mask is None and g.layers is None and "displaceable" in g.summary()


def test_set_thermostat_region_checks_length():
    g = surface_slab("Pd", "fcc100", size=(2, 2), layers=(1, 1, 1), a=3.9)
    system = g.make_system(ASECalculatorAdapter(EMT(), [2 * 3.9 / np.sqrt(2)] * 2 + [g.Lz]), "H")
    with pytest.raises(ValueError, match="entries"):
        system.set_thermostat_region([True])
    with pytest.raises(ValueError, match="no substrate"):
        system.set_thermostat_region(np.zeros(8, bool))
    system.set_thermostat_region(None)
    assert system.thermostat_mask is None


class _Free(ASECalc):
    """No forces at all: atoms fly straight unless a thermostat touches them."""
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def _replica(T_K=500.0):
    g = surface_slab("Pd", "fcc100", size=(2, 2), layers=(1, 1, 1), a=3.9)
    system = g.make_system(ASECalculatorAdapter(EMT(), [2 * 3.9 / np.sqrt(2)] * 2 + [g.Lz]), "H")
    return Replica(ThermoState(kelvin_to_eV(T_K), -2.0, label=(0, 0)), system,
                   np.random.default_rng(3))


@pytest.mark.parametrize("thermostat", ["bussi", "langevin"])
def test_thermostat_touches_only_the_bath(thermostat):
    rep = _replica()
    atoms = Atoms("Pd12", positions=np.random.default_rng(0).random((12, 3)) * 5,
                  cell=[20, 20, 20], pbc=False)
    atoms.set_constraint(FixAtoms(indices=range(4)))
    atoms.calc = _Free()
    v0 = np.random.default_rng(1).standard_normal((12, 3)) * 0.01
    v0[:4] = 0.0
    atoms.set_velocities(v0)
    x0 = atoms.get_positions()
    bath = np.arange(4, 8)
    rep._md_partial_thermostat(atoms, 4, bath, 50, 0.1, [], thermostat, 0.5, 0.5)
    v = atoms.get_velocities()
    assert np.allclose(v[8:], v0[8:])                     # free atoms: untouched, ballistic
    assert np.allclose(v[:4], 0.0)                        # fixed atoms: still
    assert not np.allclose(v[4:8], v0[4:8])               # the bath was thermostatted
    x = atoms.get_positions()
    assert np.allclose(x[8:], x0[8:] + 50 * 0.1 * v0[8:])  # straight lines


@pytest.mark.parametrize("thermostat", ["bussi", "langevin"])
def test_bath_reaches_the_temperature(thermostat):
    """Bath kinetic energy averages (3/2) n kT; the free atoms keep theirs."""
    rep = _replica(400.0)
    kT = rep.state.temperature
    atoms = Atoms("Pd10", positions=np.zeros((10, 3)), cell=[20, 20, 20], pbc=False)
    atoms.calc = _Free()
    v = np.zeros((10, 3))
    v[:6] = 1e-3      # Bussi rescales: it needs some kinetic energy to start from
    atoms.set_velocities(v)
    bath = np.arange(6)
    m = atoms.get_masses()[:6, None]
    kin = []
    for _ in range(400):
        rep._md_partial_thermostat(atoms, 0, bath, 5, 0.5, [], thermostat, 0.3, 0.3)
        kin.append(0.5 * np.sum(m * atoms.get_velocities()[:6] ** 2))
    assert np.mean(kin[50:]) == pytest.approx(1.5 * 6 * kT, rel=0.12)
    assert np.allclose(atoms.get_velocities()[6:], 0.0)


def test_partial_integrator_is_nve_without_coupling():
    """With the coupling switched off (tau -> inf) the integrator is velocity
    Verlet: the total energy of an EMT slab is conserved."""
    rep = _replica()
    from ase import units
    from ase.build import fcc100
    atoms = fcc100("Pd", size=(2, 2, 3), a=3.9, vacuum=6.0)
    atoms = atoms[np.argsort(-atoms.get_tags(), kind="stable")]   # bottom layer (tag 3) first
    atoms.rattle(0.05, seed=2)
    n_fr = 4
    atoms.set_constraint(FixAtoms(indices=range(n_fr)))
    atoms.calc = EMT()
    v = np.random.default_rng(4).standard_normal((12, 3)) * 0.02
    v[:n_fr] = 0.0
    atoms.set_velocities(v)
    e0 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    rep._md_partial_thermostat(atoms, n_fr, np.arange(4, 8), 200, 1.0 * units.fs, [],
                               "bussi", 0.0, 0.0)
    e1 = atoms.get_potential_energy() + atoms.get_kinetic_energy()
    assert abs(e1 - e0) < 2e-3, e1 - e0


def test_ase_md_diffusion_with_regions():
    rep = _replica(700.0)
    sysm = rep.system
    sysm.apply_insert(np.array([1.0, 1.0, sysm.substrate_positions[:, 2].max() + 1.2]))
    rep.energy = sysm.total_energy()
    frozen0, sub0 = sysm.frozen_positions.copy(), sysm.substrate_positions.copy()
    rep.diffuse('md', n=20, timestep_fs=1.0, thermostat='bussi', taut_fs=20.0)
    assert np.array_equal(sysm.frozen_positions, frozen0)
    assert not np.allclose(sysm.substrate_positions, sub0)
    assert np.isfinite(rep.energy) and rep.energy == pytest.approx(sysm.total_energy())


def _yaml_config(tmp_path, **struct):
    cfg = {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [2, 2], "a": 3.9},
                      "layers": {"fixed": 1, "thermostat": 1, "free": 2}, **struct},
        "gas": {"species": "H"},
        "calculator": {"type": "EMT"},
        "replica_exchange": {"temperatures_K": [500], "mu_eV": [[-2.0]], "scheme": "regc",
                             "n_steps": 4, "diffusion": "md", "diffusion_steps": 5,
                             "md": {"timestep_fs": 1.0}},
        "output": {"directory": str(tmp_path / "out")},
    }
    return cfg


def test_yaml_layers(tmp_path):
    setup = setup_from_config(_yaml_config(tmp_path), verbose=False)
    g = setup["geometry"]
    assert g.layers == (1, 1, 2) and len(g.layer_z) == 4
    system = setup["system_factory"](np.random.default_rng(0))
    assert system.thermostat_mask.sum() == 4 and len(system.thermostat_mask) == 12


def test_yaml_layers_errors(tmp_path):
    from regc.runner import validate_config
    cfg = _yaml_config(tmp_path, n_frozen_layers=1)
    with pytest.raises(ValueError, match="drop structure.n_frozen_layers"):
        validate_config(cfg)
    cfg = _yaml_config(tmp_path)
    cfg["structure"]["build"]["size"] = [2, 2, 5]
    with pytest.raises(ValueError, match="adds up to 4"):
        validate_config(cfg)
    cfg = _yaml_config(tmp_path)
    cfg["replica_exchange"].pop("md")
    cfg["replica_exchange"]["diffusion"] = "mc"
    with pytest.warns(UserWarning, match="only on MD"):
        validate_config(cfg)


def test_yaml_layers_run(tmp_path):
    from regc.runner import run_from_config
    run_from_config(_yaml_config(tmp_path), verbose=False)
    assert (tmp_path / "out" / "samples.npz").exists()


@pytest.mark.parametrize("surface, symbol, size, ppl", [
    ("fcc211", "Pd", (3, 2), 3), ("hcp10m10", "Ru", (2, 2), 2), ("fcc110", "Pd", (2, 2), 1)])
def test_layers_on_open_and_stepped_surfaces(surface, symbol, size, ppl):
    """A layer is an ase.build layer (several atomic planes on stepped surfaces)."""
    g = surface_slab(symbol, surface, size=size, layers=(2, 2, 2))
    assert g.layers == (2, 2, 2) and g.planes_per_layer == ppl
    assert len(g.layer_z) == 6 * ppl
    n_layer = len(g.frozen_positions) // 2            # atoms per layer
    assert g.thermostat_mask.sum() == 2 * n_layer and len(g.substrate_positions) == 4 * n_layer
    z = g.substrate_positions[:, 2]
    assert z[g.thermostat_mask].max() < z[~g.thermostat_mask].min()
    assert g.frozen_positions[:, 2].max() == pytest.approx(0.0)
    if ppl > 1:
        assert f"of {ppl} atomic planes each" in g.summary()


def test_planes_must_divide_into_layers():
    from ase.build import fcc100
    from regc import layer_planes
    atoms = fcc100("Pd", size=(2, 2, 5), a=3.9)
    assert layer_planes(atoms, 5) == 1
    with pytest.raises(ValueError, match="do not divide"):
        layer_planes(atoms, 4)
    with pytest.raises(ValueError, match="do not group"):
        slab_geometry_from_atoms(atoms, n_frozen_layers=1, planes_per_layer=2)


def test_yaml_layers_stepped_surface(tmp_path):
    cfg = _yaml_config(tmp_path)
    cfg["structure"]["build"] = {"surface": "fcc211", "symbol": "Pd", "size": [3, 2], "a": 3.9}
    g = setup_from_config(cfg, verbose=False)["geometry"]
    assert g.layers == (1, 1, 2) and g.planes_per_layer == 3


def test_yaml_layers_mismatch_names_the_key(tmp_path):
    from ase.build import fcc100
    from ase.io import write
    write(tmp_path / "slab.traj", fcc100("Pd", size=(2, 2, 5), a=3.9, vacuum=5.0))
    cfg = _yaml_config(tmp_path)
    cfg["structure"] = {"file": str(tmp_path / "slab.traj"),
                        "layers": {"fixed": 1, "thermostat": 1, "free": 2}}
    with pytest.raises(ValueError, match="structure.layers adds up to 4"):
        setup_from_config(cfg, verbose=False)


def test_ase_md_friction_none_means_one_over_taut():
    rep = _replica(600.0)
    rep.diffuse('md', n=5, timestep_fs=1.0, thermostat='bussi', taut_fs=20.0,
                friction_per_fs=None)
    rep.diffuse('md', n=5, timestep_fs=1.0, thermostat='langevin', taut_fs=20.0,
                friction_per_fs=None)
    assert np.isfinite(rep.energy)


class _Blowup(ASECalc):
    """Finite forces, but every energy after the first call is infinite."""
    implemented_properties = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {"energy": 0.0 if self.calls == 1 else np.inf,
                        "forces": np.zeros((len(self.atoms), 3))}


@pytest.mark.parametrize("regions", [True, False])
def test_ase_md_discards_a_non_finite_end_point(regions):
    g = surface_slab("Pd", "fcc100", size=(2, 2), layers=(1, 1, 1), a=3.9)
    blow = _Blowup()
    system = g.make_system(ASECalculatorAdapter(blow, [2 * 3.9 / np.sqrt(2)] * 2 + [g.Lz]), "H")
    if not regions:
        system.set_thermostat_region(None)
    rep = Replica(ThermoState(kelvin_to_eV(500.0), -2.0, label=(0, 0)), system,
                  np.random.default_rng(0))
    rep.energy = 0.0
    sub0 = system.substrate_positions.copy()
    rep.diffuse('md', n=3, timestep_fs=1.0)
    assert rep.n_attempts.get('md_reverted') == 1
    assert np.array_equal(system.substrate_positions, sub0) and rep.energy == 0.0
