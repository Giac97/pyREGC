"""
Starting a run from given configurations (``replica_exchange.initial_configurations``).

* a PairLJ Pd(100) slab (frozen, displaceable and exchangeable layers; O and
  Pd exchanged) writes trajectories; a second run starts from their last
  frames plus a hand-made frame with one exchangeable Pd removed and two O
  added: the energies after setup are the calculator's energies of those
  frames, the frames land where ``by_count`` / ``cyclic`` put them, and the
  metadata names every file;
* the assignment rules on their own;
* incompatible frames (cell, frozen atoms, foreign species, overlaps, outside
  the box) raise a ValueError naming the file; bad settings raise;
* resume: the checkpoint's configurations win, and a changed or vanished
  ``initial_configurations`` does not stop a resume.
"""

import copy
import json
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("ase")
from ase.io import read, write  # noqa: E402

from regc.initial_configurations import (assign_configurations,  # noqa: E402
                                         check_configuration, expand_files)
from regc.runner import (load_config, run_from_config, setup_from_config,  # noqa: E402
                         validate_config)

pytestmark = pytest.mark.filterwarnings("ignore:structure.exchangeable_top_layers")


def _config(out):
    return {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [3, 3, 4],
                                "a": 3.94},
                      "n_frozen_layers": 2, "exchangeable_top_layers": 1,
                      "headroom_angstrom": 5.0},
        "gas": {"species": ["O", "Pd"], "insertion_z_range_above_surface": [-1.0, 3.5]},
        "calculator": {"type": "PairLJ", "r_cut_factor": 1.6,
                       "pair_params": [["Pd", "Pd", 0.4, 2.48], ["Pd", "O", 0.6, 1.9],
                                       ["O", "O", 0.05, 2.6]]},
        "reservoir": {"O": {"type": "diatomic", "D0_eV": 5.12, "e_atom_eV": 0.0,
                            "bond_length_angstrom": 1.2075,
                            "vibrational_wavenumber_cm": 1580.2},
                      "Pd": {"type": "bulk", "e_bulk_eV": -2.3, "mean_wavenumber_cm": 110.0}},
        "replica_exchange": {"temperatures_K": [500, 900], "pressures": [1e-3, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 20,
                             "x0": 0.7, "diffusion": "mc", "diffusion_steps": 1,
                             "displace_substrate": True, "record_every": 2, "seed": 3},
        "checkpoint": {"chunk": 20},
        "phase_diagram": {"observable": "N_O"},   # (only silences the N-includes-Pd warning)
        "output": {"directory": str(out)},
    }


def _without_analysis(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("phase_diagram")
    return cfg


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    """Run 1 (writes trajectories) and a hand-made frame: one Pd fewer, two O more."""
    tmp = tmp_path_factory.mktemp("initial")
    cfg = _config(tmp / "run1")
    cfg["trajectory"] = {"save": True, "every": 20}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_from_config(_without_analysis(cfg), verbose=False)
    traj = sorted((tmp / "run1" / "trajectories").glob("*.xyz"))
    assert len(traj) == 4
    atoms = read(str(traj[0]), index=-1)
    top = [i for i, s in enumerate(atoms.get_chemical_symbols())
           if s == "Pd" and i >= 27]                     # the exchangeable (mobile) Pd
    assert len(top) >= 5
    z_top = atoms.positions[top, 2].max()
    hollow = atoms.positions[top[0]] + np.array([1.393, 1.393, 0.0])
    removed = atoms.copy()
    del removed[top[-1]]
    removed.extend(type(atoms)("O2", positions=[[hollow[0], hollow[1], z_top + 1.1],
                                                [hollow[0] + 2.786, hollow[1], z_top + 1.1]]))
    made = tmp / "made.extxyz"
    write(str(made), removed, format="extxyz")
    return {"tmp": tmp, "cfg": cfg, "traj": [str(t) for t in traj], "made": str(made),
            "atoms": atoms, "n_Pd_made": len(top) - 1}


def _start(cfg, **kw):
    """Build the simulation and stop before the first step (time limit 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_from_config(_without_analysis(cfg), verbose=False, time_limit=0.0, **kw)
    assert not res["finished"] and res["status"].steps_done == 0
    return res


def _frame_energy(setup, path):
    """The calculator's energy of a frame, put into a fresh system by hand."""
    atoms = read(path, index=-1)
    sysm = setup["system_factory"](np.random.default_rng(0))
    n_f, n_s = len(sysm.frozen_positions), len(sysm.substrate_positions)
    sp = np.array(atoms.get_chemical_symbols())
    sysm.substrate_positions = atoms.positions[n_f:n_f + n_s].copy()
    sysm.substrate_species = sp[n_f:n_f + n_s]
    sysm.positions, sysm.species = atoms.positions[n_f + n_s:].copy(), sp[n_f + n_s:]
    return sysm.total_energy(), sysm.species_counts()


def test_start_from_trajectories_and_a_modified_frame(first_run, capsys):
    cfg = copy.deepcopy(first_run["cfg"])
    cfg.pop("trajectory")
    cfg["output"]["directory"] = str(first_run["tmp"] / "run2")
    files = [str(first_run["tmp"] / "run1" / "trajectories" / "*.xyz"), first_run["made"]]
    cfg["replica_exchange"]["initial_configurations"] = {"files": files, "species": "O"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_from_config(_without_analysis(cfg), verbose=True, time_limit=0.0)
    printed = capsys.readouterr().out
    sim, setup = res["sim"], res["setup"]
    meta = json.loads((first_run["tmp"] / "run2" / "run_metadata.json").read_text())
    rec = meta["initial_configurations"]
    assert rec["assign"] == "by_count" and rec["species"] == "O" and rec["index"] == -1
    assert [c["file"] for c in rec["configurations"]] == first_run["traj"] + [first_run["made"]]
    assert all(len(c["sha256"]) == 64 for c in rec["configurations"])
    made = rec["configurations"][4]
    n_O = [c["counts"]["O"] for c in rec["configurations"]]
    assert made["counts"]["Pd"] == first_run["n_Pd_made"]
    assert made["counts"]["O"] == max(n_O) >= 2
    # 5 configurations on 4 grid points: sorted by N_O, column by column, hot row first;
    # the made frame (most O) goes to the last visit, (T0, mu1)
    table = np.array(rec["assignment"])
    assert table[0, 1] == 4
    visits = [table[it, im] for im in range(2) for it in (1, 0)]
    assert [n_O[k] for k in visits] == sorted(n_O[k] for k in visits)
    for key, r in sim.grid.items():             # energies are those of the frames
        conf = rec["configurations"][table[key]]
        energy, counts = _frame_energy(setup, conf["file"])
        assert r.energy == pytest.approx(energy, rel=1e-10, abs=1e-10)
        assert r.energy == pytest.approx(r.system.total_energy(), rel=1e-12, abs=1e-12)
        assert list(r.system.species_counts()) == list(counts)
    assert list(sim.grid[(0, 1)].system.species_counts()) == [made["counts"]["O"],
                                                              first_run["n_Pd_made"]]
    assert all(not bad for _, bad in sim.check_energies().values())
    assert "initial configurations: 5 file(s), frame -1, assign by_count" in printed
    # the slab's frozen atoms stay the template's
    np.testing.assert_array_equal(sim.grid[(0, 1)].system.frozen_positions,
                                  setup["geometry"].frozen_positions)


def test_cyclic_assignment_and_a_short_run(first_run):
    cfg = copy.deepcopy(first_run["cfg"])
    cfg.pop("trajectory")
    cfg["output"]["directory"] = str(first_run["tmp"] / "cyclic")
    cfg["replica_exchange"]["initial_configurations"] = {
        "files": [first_run["made"], first_run["traj"][1]], "assign": "cyclic"}
    cfg["replica_exchange"]["n_workers"] = 2          # the replicas travel to the workers
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_from_config(_without_analysis(cfg), verbose=False)
    assert res["finished"] and len(res["sim"].samples[(0, 0)]["U"]) == 10
    assert all(not bad for _, bad in res["sim"].check_energies().values())
    meta = json.loads((first_run["tmp"] / "cyclic" / "run_metadata.json").read_text())
    assert meta["initial_configurations"]["assignment"] == [[0, 1], [0, 1]]   # s mod 2
    assert meta["initial_configurations"]["species"] == "O"                  # the default


def test_assignment_rules():
    counts = [5, 0, 3, 1, 4, 2]
    # cyclic: row-major grid order, modulo the number of configurations
    np.testing.assert_array_equal(assign_configurations(counts, (2, 4), "cyclic"),
                                  [[0, 1, 2, 3], [4, 5, 0, 1]])
    # by_count, as many configurations as grid points: column by column, hot row first
    np.testing.assert_array_equal(assign_configurations(counts, (2, 3)),
                                  [[3, 2, 0], [1, 5, 4]])
    # fewer: neighbours share, counts never decrease along the visits
    t = assign_configurations([2, 0, 1], (2, 3))
    np.testing.assert_array_equal(t, [[1, 2, 0], [1, 2, 0]])
    # more: spread over the whole sorted list, both ends used
    t = assign_configurations(list(range(10)), (1, 3))
    np.testing.assert_array_equal(t, [[0, 4, 9]])
    np.testing.assert_array_equal(assign_configurations([7, 1], (1, 1)), [[1]])
    with pytest.raises(ValueError, match="assign"):
        assign_configurations(counts, (2, 3), "random")
    with pytest.raises(ValueError, match="no configurations"):
        assign_configurations([], (2, 3))


def _bad_frame(atoms, kind):
    a = atoms.copy()
    if kind == "cell":
        a.set_cell(a.cell[:] * np.diag([1.05, 1.0, 1.0]), scale_atoms=False)
    elif kind == "frozen":
        del a[0]
    elif kind == "shifted":
        a.positions[:, 2] += 1.0
    elif kind == "foreign":
        a.extend(type(a)("H", positions=[[1.0, 1.0, a.positions[:, 2].max() + 2.0]]))
    elif kind == "overlap":
        p = a.positions[-1] + np.array([0.0, 0.0, 0.3])
        a.extend(type(a)("O", positions=[p]))
    elif kind == "outside":
        a.extend(type(a)("O", positions=[[1.0, 1.0, a.positions[:, 2].max() + 30.0]]))
    elif kind == "substrate":
        sp = a.get_chemical_symbols()
        sp[20] = "O"
        a.set_chemical_symbols(sp)
    return a


@pytest.mark.parametrize("kind, match", [
    ("cell", "cell"), ("frozen", "frozen atoms"), ("shifted", "template positions"),
    ("foreign", "not exchanged species"), ("overlap", "apart"), ("outside", "outside the box"),
    ("substrate", "displaceable substrate atom"),
])
def test_incompatible_frames_raise(first_run, tmp_path, kind, match):
    bad = tmp_path / f"bad_{kind}.extxyz"
    write(str(bad), _bad_frame(first_run["atoms"], kind), format="extxyz")
    cfg = copy.deepcopy(first_run["cfg"])
    cfg.pop("trajectory")
    cfg["output"]["directory"] = str(tmp_path / "out")
    cfg["replica_exchange"]["initial_configurations"] = {"files": [first_run["traj"][0],
                                                                   str(bad)]}
    with pytest.raises(ValueError, match=match) as err:
        _start(cfg)
    assert str(bad) in str(err.value)


def test_check_configuration_directly(first_run):
    setup = setup_from_config(_without_analysis(first_run["cfg"]), verbose=False)
    sysm = setup["system_factory"](np.random.default_rng(0))
    a = first_run["atoms"]
    out = check_configuration(sysm, a.positions, a.get_chemical_symbols(), a.cell[:])
    assert len(out["substrate_positions"]) == 9 and set(out["counts"]) == {"O", "Pd"}
    # a plain-XYZ frame (no cell) is not cell-checked; a hard core of 3 A is violated
    check_configuration(sysm, a.positions, a.get_chemical_symbols(), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="closer than 3"):
        check_configuration(sysm, a.positions, a.get_chemical_symbols(), min_distance=3.0)


@pytest.mark.parametrize("section, match", [
    ({"files": []}, "files"), ({"files": "x.xyz", "assign": "random"}, "assign"),
    ({"files": "x.xyz", "species": "Ag"}, "not an exchanged species"),
    ({"files": "x.xyz", "index": 1.5}, "index"), ({"files": "x.xyz", "frame": 0}, "unknown"),
    ("x.xyz", "mapping"),
])
def test_bad_settings_raise(tmp_path, section, match):
    cfg = _config(tmp_path)
    cfg["replica_exchange"]["initial_configurations"] = section
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_missing_files_raise_and_paths_resolve(first_run, tmp_path):
    with pytest.raises(ValueError, match="no file matches"):
        expand_files([str(tmp_path / "none*.xyz")])
    files = expand_files([first_run["made"], str(first_run["tmp"] / "run1" / "trajectories"
                                                  / "replica_T*_mu*.xyz")])
    assert files == [first_run["made"]] + first_run["traj"]
    yaml = pytest.importorskip("yaml")
    cfg = _config("out")
    cfg["replica_exchange"]["initial_configurations"] = {"files": ["run1/*.xyz", "a.xyz"]}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    loaded = load_config(str(path))["replica_exchange"]["initial_configurations"]["files"]
    assert loaded == [str(tmp_path / "run1" / "*.xyz"), str(tmp_path / "a.xyz")]


def test_resume_keeps_the_checkpoint_configurations(first_run, tmp_path):
    cfg = copy.deepcopy(first_run["cfg"])
    cfg.pop("trajectory")
    cfg["output"]["directory"] = str(tmp_path / "run")
    made = tmp_path / "made_copy.extxyz"
    made.write_bytes(open(first_run["made"], "rb").read())
    cfg["replica_exchange"]["initial_configurations"] = {"files": [str(made)]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_from_config(_without_analysis(cfg), verbose=False)
    saved = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    os.remove(made)                                    # the file is gone: resume still works
    cfg["replica_exchange"]["n_steps"] = 40
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # unchanged section: no warning
        warnings.filterwarnings("ignore", message=".*timeseries.*")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*exchangeable_top")
        res = run_from_config(_without_analysis(cfg), verbose=False, resume=True)
    assert res["status"].resumed_from == 20 and res["status"].steps_done == 40
    after = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert after["initial_configurations"] == saved["initial_configurations"]
    # a changed (or dropped) section is ignored with a warning, not refused
    cfg["replica_exchange"]["initial_configurations"]["assign"] = "cyclic"
    cfg["replica_exchange"]["n_steps"] = 60
    with pytest.warns(UserWarning, match="initial_configurations differs"):
        res = run_from_config(_without_analysis(cfg), verbose=False, resume=True)
    assert res["status"].steps_done == 60
    cfg["replica_exchange"].pop("initial_configurations")
    cfg["replica_exchange"]["n_steps"] = 80
    with pytest.warns(UserWarning, match="initial_configurations differs"):
        assert run_from_config(_without_analysis(cfg), verbose=False,
                               resume=True)["status"].steps_done == 80
