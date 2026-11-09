"""
End-to-end tests for regc.runner, using the fast native LJCalculator
(not EMT) so these run in a couple of seconds. Real EMT usage is exercised
manually in examples/09 (too slow for the routine test suite).
"""

import json

import numpy as np
import pytest

ase = pytest.importorskip("ase", reason="pip install ase to run this test")
pytest.importorskip("pymbar", reason="pip install pymbar to run this test")

from regc.runner import (analyze_from_config, load_config, main, run_from_config,  # noqa: E402
                            setup_from_config)


@pytest.fixture
def tiny_structure(tmp_path):
    """A minimal 'substrate' (2 atoms) written to disk, just to exercise
    ase.io.read + the orthogonal-cell check -- not meant to be physically
    meaningful (see the LJ-ignores-frozen-substrate warning this triggers)."""
    from ase import Atoms

    atoms = Atoms("Fe2", positions=[[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
                   cell=[6.0, 6.0, 15.0], pbc=(True, True, False))
    path = tmp_path / "substrate.xyz"
    from ase.io import write
    write(str(path), atoms)
    return str(path)


@pytest.fixture
def base_config(tiny_structure, tmp_path):
    return {
        "structure": {"file": tiny_structure},
        "gas": {"species": "H", "insertion_z_range_above_surface": [0.3, 2.0]},
        "calculator": {"type": "LJ", "lj_sigma": 1.0, "lj_epsilon": 0.3, "lj_r_cut": 3.0},
        "replica_exchange": {
            "temperatures_K": [350, 450],
            "mu_eV": [0.5, 1.5],
            "n_cycles": 20,
            "sweeps_per_cycle": 15,
            "burn_in_cycles": 5,
            "record_every": 5,
            "seed": 1,
        },
        "trajectory": {"save": True, "directory": "trajectories", "every": 5},
        "diatomic_reservoir": {"D0_eV": 0.4},
        "phase_diagram": {
            "T_min_K": 350, "T_max_K": 450, "n_T": 3,
            "p_min_atm": 1e-4, "p_max_atm": 1e6, "n_p": 5,
        },
        "output": {"directory": str(tmp_path / "output")},
    }


def test_load_config_json(tmp_path, base_config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_config))
    loaded = load_config(str(path))
    assert loaded["gas"]["species"] == "H"


def test_load_config_resolves_structure_relative_to_config(tmp_path, base_config, monkeypatch):
    import shutil
    sub = tmp_path / "cfgdir"
    (sub / "data").mkdir(parents=True)
    shutil.copy(base_config["structure"]["file"], sub / "data" / "slab.xyz")
    base_config["structure"]["file"] = "data/slab.xyz"
    (sub / "config.json").write_text(json.dumps(base_config))
    monkeypatch.chdir(tmp_path)  # not the config's directory
    loaded = load_config(str(sub / "config.json"))
    assert loaded["structure"]["file"] == str(sub / "data" / "slab.xyz")


def test_run_from_config_produces_expected_outputs(base_config, tmp_path):
    result = run_from_config(base_config, verbose=False)

    assert result["theta_grid"].shape == (3, 5)
    assert np.all(np.isfinite(result["theta_grid"]))

    out_dir = tmp_path / "output"
    assert (out_dir / "phase_diagram.png").exists()
    assert (out_dir / "phase_diagram_data.npz").exists()

    traj_dir = out_dir / "trajectories"
    traj_files = sorted(p.name for p in traj_dir.glob("*.xyz"))
    assert traj_files == ["replica_T0_mu0.xyz", "replica_T0_mu1.xyz",
                            "replica_T1_mu0.xyz", "replica_T1_mu1.xyz"]


def test_run_from_config_regc_scheme(base_config, tmp_path):
    re_cfg = base_config["replica_exchange"]
    for key in ("n_cycles", "sweeps_per_cycle", "burn_in_cycles"):
        re_cfg.pop(key)
    re_cfg.update({"scheme": "regc", "n_steps": 60, "x0": 0.8, "diffusion": "mc",
                   "diffusion_steps": 2, "record_every": 5, "burn_in_steps": 10})
    result = run_from_config(base_config, verbose=False)
    assert np.all(np.isfinite(result["theta_grid"]))


def test_oblique_cell_is_accepted(tmp_path, base_config):
    """Hexagonal/oblique surface cells (a1 along x, a2 in xy) are supported."""
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("Fe", positions=[[0, 0, 0]],
                  cell=[[7.0, 0, 0], [3.5, 6.06, 0], [0, 0, 15.0]], pbc=(True, True, False))
    path = tmp_path / "hex.xyz"
    write(str(path), atoms)
    base_config["structure"]["file"] = str(path)
    base_config["calculator"]["lj_r_cut"] = 2.5
    result = run_from_config(base_config, verbose=False)
    assert not result["setup"]["geometry"].orthogonal
    assert np.all(np.isfinite(result["theta_grid"]))


def test_rejects_cell_not_in_surface_form(tmp_path, base_config):
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("Fe", positions=[[0, 0, 0]],
                  cell=[[6.0, 1.0, 0], [0, 6.0, 0], [0, 0, 15.0]], pbc=(True, True, False))
    path = tmp_path / "skewed.xyz"
    write(str(path), atoms)
    base_config["structure"]["file"] = str(path)
    with pytest.raises(ValueError, match="a1 along x"):
        run_from_config(base_config, verbose=False)


def test_missing_D0_without_emt_raises(base_config):
    del base_config["diatomic_reservoir"]["D0_eV"]
    with pytest.raises(ValueError, match="D0_eV"):
        run_from_config(base_config, verbose=False)


def test_yaml_unsigned_exponent_gotcha_is_silently_fixed(base_config, tmp_path):
    """PyYAML parses '1.0e-4' as a float but '1.0e6' (no explicit sign on
    the exponent) as the STRING '1.0e6' -- a classic gotcha. `_as_float`
    re-parses via Python's own (more permissive) float(), which has no such
    sign requirement, so this should come out correct rather than error."""
    import yaml

    path = tmp_path / "config.yaml"
    text = yaml.safe_dump(base_config).replace("p_max_atm: 1000000.0", "p_max_atm: 1.0e6")
    path.write_text(text)

    loaded = load_config(str(path))
    assert loaded["phase_diagram"]["p_max_atm"] == "1.0e6"  # confirms the gotcha is real

    result = run_from_config(loaded, verbose=False)
    assert result["p_grid_atm"].max() == pytest.approx(1.0e6)


def test_as_float_rejects_genuinely_invalid_values(base_config):
    """Unlike the YAML gotcha above, a value that isn't number-shaped at all
    should still fail clearly rather than propagate into np.geomspace."""
    base_config["phase_diagram"]["p_max_atm"] = "not-a-number"
    with pytest.raises(ValueError, match="p_max_atm"):
        run_from_config(base_config, verbose=False)


def test_unknown_key_is_rejected(base_config):
    base_config["replica_exchange"]["n_cylces"] = 10
    with pytest.raises(ValueError, match="n_cylces"):
        run_from_config(base_config, verbose=False)


def test_keys_of_the_other_scheme_are_rejected(base_config):
    base_config["replica_exchange"]["scheme"] = "regc"
    base_config["replica_exchange"]["n_steps"] = 10
    with pytest.raises(ValueError, match="n_cycles"):
        run_from_config(base_config, verbose=False)


def test_bad_phase_diagram_fails_before_sampling(base_config, monkeypatch):
    import regc.engine
    monkeypatch.setattr(regc.engine.ReplicaExchangeGCMC, "run",
                        lambda *a, **k: pytest.fail("sampled before validating"))
    base_config["phase_diagram"]["coverage_per"] = "volume"
    with pytest.raises(ValueError, match="coverage_per"):
        run_from_config(base_config, verbose=False)


# ---------------------------------------------------------------- current schema
def _pair_lj_config(tmp_path):
    """Cu(100)-like slab built with ase.build, H through a PairLJ potential that
    sees the slab (fast incremental path), on a pressure ladder."""
    return {
        "structure": {"build": {"surface": "fcc100", "symbol": "Cu", "size": [3, 3, 2],
                                "a": 3.6, "orthogonal": True},
                      "n_frozen_layers": 2, "headroom_angstrom": 4.0},
        "gas": {"species": "H", "insertion_z_range_above_surface": [0.5, 3.5]},
        "calculator": {"type": "PairLJ",
                       "pair_params": [["Cu", "H", 0.15, 1.4], ["H", "H", 0.02, 1.0]]},
        "reservoir": {"type": "diatomic", "D0_eV": 4.478, "e_atom_eV": 0.0},
        "replica_exchange": {
            "temperatures_K": [300, 500], "pressures": [1.0e-4, 1.0e+2], "pressure_unit": "bar",
            "scheme": "regc", "n_steps": 300, "x0": 0.9, "diffusion": "mc",
            "diffusion_steps": 2, "record_every": 10, "seed": 3, "adsorbed_cutoff": 2.5,
        },
        "phase_diagram": {"T_min_K": 300, "T_max_K": 500, "n_T": 3, "p_min": 1.0e-4,
                          "p_max": 1.0e+2, "n_p": 4, "observable": "N_ads",
                          "coverage_per": "surface_atom", "most_probable": True},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_pressure_ladder_pair_lj_n_ads(tmp_path):
    from regc import DiatomicReservoir, kelvin_to_eV
    cfg = _pair_lj_config(tmp_path)
    result = run_from_config(cfg, verbose=False)
    setup = result["setup"]
    assert setup["geometry"].n_surface == 9
    res = DiatomicReservoir.hydrogen(D0_eV=4.478, e_atom_eV=0.0)
    np.testing.assert_allclose(setup["mu_grid"], res.mu_grid([300, 500], [1e-4, 1e2], unit="bar"))
    assert result["sim"].mu_grid.shape == (2, 2)
    assert result["theta_grid"].shape == (3, 4)
    assert result["most_probable_N"].shape == (3, 4)
    out = tmp_path / "out"
    for name in ("samples.npz", "run_metadata.json", "phase_diagram.png",
                 "phase_diagram_most_probable.png", "phase_diagram_data.npz"):
        assert (out / name).exists(), name
    assert "N_ads" in np.load(out / "samples.npz")
    # re-analysis from disk alone gives the same numbers
    again = analyze_from_config(cfg, verbose=False)
    np.testing.assert_allclose(again["theta_grid"], result["theta_grid"])
    assert result["p_grid_atm"][0] == pytest.approx(1e-4 * 1e5 / 101325.0)
    # the analysis reservoir is the run's: mu at the ladder's own (T, p) matches
    assert again["analysis"].mu_from_p(kelvin_to_eV(300.0), 1e-4) == pytest.approx(
        setup["mu_grid"][0, 0])


def test_displaceable_layers_and_hmc_diffusion(tmp_path):
    cfg = _pair_lj_config(tmp_path)
    cfg["structure"]["n_frozen_layers"] = 1
    cfg["calculator"]["pair_params"].append(["Cu", "Cu", 0.3, 2.3])
    cfg["calculator"]["r_cut_factor"] = 1.6   # cutoffs within this small cell's image radius
    cfg["replica_exchange"].update({"diffusion": "hmc", "n_steps": 40,
                                    "hmc": {"leapfrog_steps": 5, "timestep_fs": 0.5}})
    del cfg["phase_diagram"]
    result = run_from_config(cfg, verbose=False)
    assert len(result["setup"]["geometry"].substrate_positions) == 9
    rep = result["sim"].grid[(0, 0)]
    assert rep.hmc_dt == pytest.approx(0.5 / 10.1805)
    assert rep.n_attempts["hmc"] > 0


def test_checkpointed_run_resumes_and_main_exit_codes(tmp_path):
    import yaml
    cfg = _pair_lj_config(tmp_path)
    cfg["checkpoint"] = {"chunk": 100}
    cfg["replica_exchange"]["n_steps"] = 100
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main([str(path), "-q"]) == 0
    assert (tmp_path / "out" / "checkpoint.pkl").exists()
    with pytest.raises(FileExistsError):          # a saved run is never clobbered silently
        main([str(path), "-q"])
    # extend the run: n_steps is not part of the setup a resume must match
    cfg["replica_exchange"]["n_steps"] = 300
    path.write_text(yaml.safe_dump(cfg))
    assert main([str(path), "-q", "--resume", "--time-limit", "0"]) == 75   # "resubmit me"
    result = run_from_config(load_config(str(path)), verbose=False, resume=True)
    assert result["status"].resumed_from == 100 and result["status"].steps_done == 300
    assert result["finished"] and "theta_grid" in result
    assert main([str(path), "-q", "--analyze-only"]) == 0


def test_resume_refuses_a_changed_setup(tmp_path):
    cfg = _pair_lj_config(tmp_path)
    cfg["checkpoint"] = {"chunk": 100}
    cfg["replica_exchange"]["n_steps"] = 100
    run_from_config(cfg, verbose=False)
    cfg["replica_exchange"]["pressures"] = [1.0e-3, 1.0e+2]
    with pytest.raises(ValueError, match="cannot resume"):
        run_from_config(cfg, verbose=False, resume=True)


def test_two_species_gas_feed(tmp_path):
    cfg = _pair_lj_config(tmp_path)
    cfg["gas"].update({"species": ["H", "D"], "composition": {"H": 0.75, "D": 0.25},
                       "mass_amu": {"H": 1.008, "D": 2.014}})
    cfg["calculator"]["pair_params"] = [["Cu", "H", 0.15, 1.4], ["Cu", "D", 0.15, 1.4],
                                        ["H", "H", 0.02, 1.0], ["D", "D", 0.02, 1.0],
                                        ["H", "D", 0.02, 1.0]]
    cfg["reservoir"] = {"H": {"type": "monatomic", "e_atom_eV": 0.0},
                        "D": {"type": "monatomic", "e_atom_eV": 0.0}}
    cfg["phase_diagram"]["observable"] = "N_D"
    cfg["phase_diagram"]["most_probable"] = False
    result = run_from_config(cfg, verbose=False)
    mus = result["setup"]["mu_grid"]
    assert mus.shape == (2, 2, 2)
    assert result["sim"].exchange_species == ("H", "D")
    assert result["theta_grid"].shape == (3, 4)
    assert set(result["e_atom_eV"]) == {"H", "D"}


def test_several_species_need_a_composition(tmp_path):
    cfg = _pair_lj_config(tmp_path)
    cfg["gas"]["species"] = ["H", "D"]
    cfg["reservoir"] = {"H": {"type": "monatomic"}, "D": {"type": "monatomic"}}
    with pytest.raises(ValueError, match="composition"):
        setup_from_config(cfg, verbose=False)


def test_build_with_lattice_constant_from_emt_and_check(tmp_path, capsys):
    """a: auto fits the lattice constant to the potential; --check prints the
    setup and the mu ladder without sampling."""
    import yaml
    cfg = {
        "structure": {"build": {"surface": "fcc111", "symbol": "Pd", "size": [2, 2, 3],
                                "a": "auto", "orthogonal": True}, "n_frozen_layers": 2},
        "gas": {"species": "H"},
        "calculator": {"type": "EMT"},
        "reservoir": {"d0": "self"},
        "replica_exchange": {"temperatures_K": [400], "pressures": [1.0, 10.0],
                             "scheme": "regc", "n_steps": 10},
        "output": {"directory": str(tmp_path / "out")},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main([str(path), "--check"]) == 0
    text = capsys.readouterr().out
    assert "lattice constant from the potential" in text and "mu_H (eV)" in text
    setup = setup_from_config(cfg, verbose=False)
    assert 3.8 < setup["lattice_constant"] < 4.1      # EMT Pd: ~3.94 A
    res = setup["feed"].reservoirs[0]
    assert res.e_atom_eV == pytest.approx(3.21, abs=0.05)  # EMT's isolated H
    assert not (tmp_path / "out" / "samples.npz").exists()


def test_load_config_resolves_potential_paths(tmp_path):
    (tmp_path / "pot").mkdir()
    (tmp_path / "pot" / "Fe-H.eam.fs").write_text("")
    (tmp_path / "cfg.json").write_text(json.dumps({
        "calculator": {"type": "LAMMPS", "potential": "pot/Fe-H.eam.fs",
                       "pair_coeff": "* * pot/Fe-H.eam.fs Fe H"}}))
    calc = load_config(str(tmp_path / "cfg.json"))["calculator"]
    assert calc["potential"] == str(tmp_path / "pot" / "Fe-H.eam.fs")
    assert calc["pair_coeff"] == f"* * {tmp_path / 'pot' / 'Fe-H.eam.fs'} Fe H"


def test_lammps_name_reaches_the_calculator(tmp_path):
    pytest.importorskip("lammps")
    from regc.runner import _CalculatorSpec
    pot = tmp_path / "x.eam.fs"
    pot.write_text("")
    spec = _CalculatorSpec({"type": "LAMMPS", "potential": str(pot), "lammps_name": "grace"},
                           ["Fe", "H"], {"H": 1.008})
    assert spec.make(np.eye(3)).lammps_name == "grace"
    default = _CalculatorSpec({"type": "LAMMPS", "potential": str(pot)}, ["Fe", "H"], {})
    assert default.make(np.eye(3)).lammps_name is None   # REGC_LAMMPS_NAME, else liblammps.so
