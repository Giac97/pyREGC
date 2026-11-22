"""
Exchanged slab atoms with a bulk-crystal reservoir: `BulkSolidReservoir`,
``structure.exchangeable_top_layers`` and the runner's ``reservoir: {type: bulk}``.
"""

import math

import numpy as np
import pytest

pytest.importorskip("ase", reason="pip install ase")

from ase.calculators.emt import EMT  # noqa: E402

from regc import BulkSolidReservoir, kelvin_to_eV, slab_geometry_from_atoms  # noqa: E402
from regc.runner import setup_from_config  # noqa: E402


def test_mu_is_the_classical_harmonic_crystal():
    r = BulkSolidReservoir(106.42, -3.9, 150.0)
    for T in (300.0, 900.0):
        kT = kelvin_to_eV(T)
        hw = 150.0 * 1.239841984e-4                       # eV per cm^-1
        assert r.mu_eV(T) == pytest.approx(-3.9 + 3 * kT * math.log(hw / kT), rel=1e-6)
        assert r.mu_eV(T, 1e5) == r.mu_eV(T)              # no pressure dependence
    with pytest.raises(ValueError, match="no pressure"):
        r.pressure_pascal(300.0, -3.9)


def test_from_calculator_emt_pd():
    r = BulkSolidReservoir.from_calculator(EMT(), "Pd", a=3.94, supercell=2)
    from ase.build import bulk
    at = bulk("Pd", "fcc", a=3.94, cubic=True)
    at.calc = EMT()
    assert r.e_bulk_eV == pytest.approx(at.get_potential_energy() / len(at))
    assert 60.0 < r.mean_wavenumber_cm < 200.0 and r.info["n_atoms"] == 32
    assert r.mass_exchanged_amu == pytest.approx(106.42, abs=0.01)


def test_exchangeable_top_layers_geometry():
    from ase.build import fcc100
    atoms = fcc100("Pd", size=(3, 3, 4), a=3.94)
    g = slab_geometry_from_atoms(atoms, n_frozen_layers=2, n_exchangeable_layers=1)
    assert len(g.frozen_positions) == 18 and len(g.substrate_positions) == 9
    assert len(g.mobile_positions) == 9 and set(g.mobile_species) == {"Pd"}
    assert g.mobile_positions[:, 2].min() > g.substrate_positions[:, 2].max()
    assert "+ 9 exchangeable" in g.summary()
    with pytest.raises(ValueError, match="n_exchangeable_layers"):
        slab_geometry_from_atoms(atoms, n_frozen_layers=2, n_exchangeable_layers=3)


def _config(tmp_path, **pd_res):
    return {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [2, 2, 4], "a": 3.94},
                      "n_frozen_layers": 2, "exchangeable_top_layers": 1},
        "gas": {"species": ["O", "Pd"]},
        "calculator": {"type": "EMT"},
        "reservoir": {"O": {"type": "diatomic", "D0_eV": 5.12, "e_atom_eV": 0.0,
                            "bond_length_angstrom": 1.2075, "vibrational_wavenumber_cm": 1580.2},
                      "Pd": dict({"type": "bulk"}, **pd_res)},
        "replica_exchange": {"temperatures_K": [600, 900], "pressures": [1e-6, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 30,
                             "diffusion": "mc", "diffusion_steps": 2, "record_every": 5},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_runner_bulk_reservoir_and_exchangeable_layer(tmp_path):
    setup = setup_from_config(_config(tmp_path), verbose=False)
    mu = np.asarray(setup["mu_grid"])                     # (n_T, n_p, 2): O, Pd
    assert mu.shape == (2, 2, 2)
    assert np.allclose(mu[:, 0, 1], mu[:, 1, 1])          # mu_Pd independent of p_O2
    assert not np.isclose(mu[0, 0, 0], mu[0, 1, 0])       # mu_O is not
    res = setup["metadata"]["reservoirs"]["Pd"]
    assert res["type"] == "bulk" and res["info"]["a_angstrom"] == pytest.approx(3.94)
    system = setup["system_factory"](np.random.default_rng(0))
    assert system.n == 4 and set(system.species) == {"Pd"} and system.n_substrate == 4
    from regc.runner import _reservoir_from_dict, _reservoir_to_dict
    from regc.reservoir import BulkSolidReservoir
    r = _reservoir_from_dict(res)
    assert isinstance(r, BulkSolidReservoir) and _reservoir_to_dict(r)["type"] == "bulk"


def test_given_bulk_numbers_and_errors(tmp_path):
    setup = setup_from_config(_config(tmp_path, e_bulk_eV=0.0, mean_wavenumber_cm=110.0),
                              verbose=False)
    T = setup["temperatures_K"][0]
    assert setup["mu_grid"][0, 0, 1] == pytest.approx(BulkSolidReservoir(106.42, 0.0, 110.0).mu_eV(T))
    with pytest.raises(ValueError, match="both e_bulk_eV and mean_wavenumber_cm"):
        setup_from_config(_config(tmp_path, e_bulk_eV=0.0), verbose=False)
    cfg = _config(tmp_path)
    cfg["gas"]["species"] = ["O"]
    cfg["reservoir"] = cfg["reservoir"]["O"]
    with pytest.raises(ValueError, match="not exchanged species"):
        setup_from_config(cfg, verbose=False)


def test_pd_exchange_run(tmp_path):
    """A short run: Pd atoms of the top layer can leave into the bulk and come back."""
    from regc.runner import run_from_config
    cfg = _config(tmp_path)
    cfg["reservoir"]["Pd"] = {"type": "bulk", "e_bulk_eV": -1.0, "mean_wavenumber_cm": 110.0}
    cfg["phase_diagram"] = {"T_min_K": 600, "T_max_K": 900, "n_T": 4, "p_min": 1e-6, "p_max": 1.0,
                            "n_p": 4, "pressure_unit": "bar", "observable": "N_O",
                            "coverage_per": "surface_atom", "extra_observables": ["N_Pd"]}
    res = run_from_config(cfg, verbose=False)             # mu_Pd far below: Pd leaves
    from regc.io import load_samples
    s = load_samples(str(tmp_path / "out" / "samples.npz"))
    n_s = np.asarray(next(iter(s.samples.values()))["N_s"]).reshape(-1, 2)
    assert n_s[-1, 1] < 4                                 # some Pd went to the bulk
    th, removed = res["theta_N_Pd"], res["removed_Pd"]
    assert np.allclose(th + removed, 1.0)                 # 4 of 4 surface sites start as Pd
    assert np.nanmax(removed) > 0
    assert (tmp_path / "out" / "phase_diagram_Pd_removed.png").exists()
    import json
    assert json.load(open(tmp_path / "out" / "run_metadata.json"))["initial_mobile_counts"] == {"Pd": 4}
