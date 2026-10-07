"""
End-to-end tests for gcmc_re.runner, using the fast native LJCalculator
(not EMT) so these run in a couple of seconds. Real EMT usage is exercised
manually in examples/09 (too slow for the routine test suite).
"""

import json

import numpy as np
import pytest

ase = pytest.importorskip("ase", reason="pip install ase to run this test")
pytest.importorskip("pymbar", reason="pip install pymbar to run this test")

from gcmc_re.runner import load_config, run_from_config  # noqa: E402


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


def test_rejects_non_orthogonal_cell(tmp_path, base_config):
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("Fe", positions=[[0, 0, 0]],
                   cell=[[6.0, 0, 0], [1.5, 6.0, 0], [0, 0, 15.0]], pbc=(True, True, False))
    path = tmp_path / "skewed.xyz"
    write(str(path), atoms)
    base_config["structure"]["file"] = str(path)

    with pytest.raises(ValueError, match="orthogonal"):
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
