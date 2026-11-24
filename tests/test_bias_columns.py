"""
Bias columns (``replica_exchange.bias_columns``): extra mu columns after the
highest pressure in which one exchanged species' chemical potential is
lowered by a multiple of kT, connected to the physical states by ordinary
replica exchange and included in MBAR.

* the grid: last physical column's mu for every species but the biased one,
  mu - shift * k_B T for it; setup, metadata and the printed summary;
* exactness on an ideal two-species gas: direct averages at every state
  (bias columns too) and MBAR at the physical states equal
  <N_s> = V exp(beta mu_s) / Lambda_s^3, and agree with a run without bias
  columns; the swap acceptance along the bias columns is the hand formula;
* the YAML pipeline: a PairLJ slab with an exchangeable Pd layer (bulk
  reservoir) and O, 3 x 3 grid + 2 bias columns: run, analysis (physical
  phase diagram + <N_Pd> on the sampled states), resume; bad configs raise.
"""

import json

import numpy as np
import pytest

from regc import ReplicaExchangeGCMC, SlabSystem, eV_to_kelvin, kelvin_to_eV
from regc.calculators import Calculator
from regc.runner import bias_column_mu_grid, setup_from_config, validate_config


class ZeroCalculator(Calculator):
    """Ideal gas: zero interaction energy (fast incremental path)."""

    supports_incremental = True

    def total_energy(self, system) -> float:
        return 0.0

    def pair_energy_of(self, system, pos, exclude=None, species=None) -> float:
        return 0.0


SPECIES = ["O", "Pd"]
VOLUME = 32.0


def _ideal(rng=None):
    return SlabSystem(4.0, 4.0, 2.0, ZeroCalculator(), exchange_species=SPECIES)


# ---------------------------------------------------------------- the grid
def test_mu_grid_construction_single_and_several_species():
    T_K = [300.0, 600.0]
    mu = np.array([[[-1.0, -3.0], [-0.5, -3.0]], [[-1.2, -3.1], [-0.7, -3.1]]])  # (2, 2, 2)
    ext, shift = bias_column_mu_grid(mu, T_K, SPECIES, {"Pd": [1.5, 3.0, 4.5]})
    assert ext.shape == (2, 5, 2) and shift.shape == (2, 5, 2)
    np.testing.assert_array_equal(ext[:, :2], mu)                      # physical untouched
    for j, s in enumerate([1.5, 3.0, 4.5]):
        np.testing.assert_array_equal(ext[:, 2 + j, 0], mu[:, -1, 0])  # O: last column
        np.testing.assert_allclose(ext[:, 2 + j, 1],
                                   mu[:, -1, 1] - s * np.array([kelvin_to_eV(T) for T in T_K]))
        assert np.all(shift[:, 2 + j, 1] == s) and np.all(shift[:, 2 + j, 0] == 0)
    assert np.all(shift[:, :2] == 0)
    # one species: (n_T, n_p) in, (n_T, n_p + k) out
    ext1, shift1 = bias_column_mu_grid(mu[..., 1], T_K, ["Pd"], {"Pd": [2.0]})
    assert ext1.shape == (2, 3) and shift1.shape == (2, 3, 1)
    assert ext1[1, 2] == pytest.approx(mu[1, -1, 1] - 2.0 * kelvin_to_eV(600.0))
    with pytest.raises(ValueError):
        bias_column_mu_grid(mu, T_K, SPECIES, {"Ag": [1.0]})
    with pytest.raises(ValueError):
        bias_column_mu_grid(mu, [300.0], SPECIES, {"Pd": [1.0]})


def test_engine_n_physical_mu_is_checked():
    with pytest.raises(ValueError, match="n_physical_mu"):
        ReplicaExchangeGCMC([1.0], [[[-1.0, -1.0]]], _ideal, n_physical_mu=2)


# ---------------------------------------------------------------- exactness
def _ideal_grid(bias: bool):
    """2 T x 2 p physical states (+ 2 Pd bias columns); <N> = V exp(beta mu)."""
    T = [1.0, 1.5]
    target = np.array([[1.0, 3.0], [2.0, 3.0]])             # columns: <N_O>, <N_Pd>
    mu = np.array([[T_ * np.log(target[j] / VOLUME) for j in range(2)] for T_ in T])
    if not bias:
        return T, mu, None
    ext, shift = bias_column_mu_grid(mu, [eV_to_kelvin(t) for t in T], SPECIES,
                                     {"Pd": [1.0, 2.0]})
    return T, ext, shift


def _exact(T, mu):
    return VOLUME * np.exp(np.asarray(mu) / T)


@pytest.fixture(scope="module")
def ideal_runs():
    out = {}
    for bias in (True, False):
        T, mu, _ = _ideal_grid(bias)
        sim = ReplicaExchangeGCMC(T, mu, _ideal, seed=7, n_physical_mu=2 if bias else None)
        sim.run_regc(n_steps=12000, x0=0.5, diffusion="mc", diffusion_steps=0, record_every=1,
                     burn_in_steps=300, verbose=False)
        out[bias] = sim
    return out


def test_ideal_gas_exact_on_every_state_and_by_mbar(ideal_runs):
    from regc import MBARSurfaceAnalysis
    sim = ideal_runs[True]
    T, mu, _ = _ideal_grid(True)
    assert sim.shape == (2, 4)
    for (it, im), d in sim.samples.items():              # direct: the moves use each state's mu
        Ns = np.asarray(d["N_s"], dtype=float).reshape(-1, 2).mean(axis=0)
        np.testing.assert_allclose(Ns, _exact(T[it], mu[it, im]), rtol=0.08, atol=0.05,
                                   err_msg=str((it, im)))
    a_bias = MBARSurfaceAnalysis(sim)
    a_plain = MBARSurfaceAnalysis(ideal_runs[False])
    assert len(a_bias.states) == 8                       # bias columns enter MBAR
    for it in range(2):
        for im in range(2):                              # the physical states
            exact = _exact(T[it], mu[it, im])
            m_bias, e_bias = a_bias.mean_N_species(T[it], mu[it, im])
            m_plain, e_plain = a_plain.mean_N_species(T[it], mu[it, im])
            np.testing.assert_allclose(m_bias, exact, rtol=0.06)
            assert np.all(np.abs(m_bias - m_plain) < 5 * np.hypot(e_bias, e_plain) + 0.05)
    # off-grid Pd chemical potential between the physical and the first bias column
    mu_mid = mu[0, 1] - np.array([0.0, 0.5 * T[0]])
    np.testing.assert_allclose(a_bias.mean_N_species(T[0], mu_mid)[0], _exact(T[0], mu_mid),
                               rtol=0.06)


def test_mixing_round_trips_cover_the_bias_columns(ideal_runs):
    trips = ideal_runs[True].round_trip_counts()
    assert set(trips) == {"T", "mu", "bias"} and trips["bias"].sum() > 0
    assert trips["mu"].sum() > 0                         # the mu axis is the extended one
    assert "bias" not in ideal_runs[False].round_trip_counts()
    pairs = ideal_runs[True].pair_acceptance()
    assert ((0, 1), (0, 2)) in pairs and ((0, 2), (0, 3)) in pairs   # physical -> bias -> bias


def test_swap_acceptance_along_bias_columns_is_the_hand_formula():
    T, mu, shift = _ideal_grid(True)
    sim = ReplicaExchangeGCMC(T, mu, _ideal, seed=1, n_physical_mu=2)
    assert (((0, 1), (0, 2)) in sim.swap_pairs("mu", 1)
            and ((0, 2), (0, 3)) in sim.swap_pairs("mu", 0))
    assert ((0, 2), (1, 3)) in sim.swap_pairs("cross_right", 0)

    def fill(key, n_O, n_Pd):
        sysm = sim[key].system
        rng = np.random.default_rng(0)
        for sp, n in (("O", n_O), ("Pd", n_Pd)):
            for _ in range(n):
                sysm.apply_insert(rng.uniform(0.1, 1.9, 3), species=sp)
        sim[key].energy = sysm.total_energy()

    for it in range(2):
        fill((it, 2), 2, 5)
        fill((it, 3), 2, 1)
    for it in range(2):                                  # same exp(-dshift * dN) at both T
        log_acc = sim.swap_log_acceptance(sim[(it, 2)], sim[(it, 3)])
        assert log_acc == pytest.approx((shift[it, 2, 1] - shift[it, 3, 1]) * (5 - 1))
    # a T swap within a bias column: the shift (the same in kT) cancels
    fill((0, 1), 2, 5)
    fill((1, 1), 2, 1)
    phys = sim.swap_log_acceptance(sim[(0, 1)], sim[(1, 1)])
    assert sim.swap_log_acceptance(sim[(0, 2)], sim[(1, 2)]) == pytest.approx(phys)


# ---------------------------------------------------------------- the YAML pipeline
def _config(tmp_path, **bias):
    return {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [3, 3, 3],
                                "a": 3.94},
                      "n_frozen_layers": 2, "exchangeable_top_layers": 1,
                      "headroom_angstrom": 5.0},
        "gas": {"species": ["O", "Pd"], "insertion_z_range_above_surface": [-1.0, 3.5]},
        "calculator": {"type": "PairLJ", "r_cut_factor": 1.6,
                       "pair_params": [["Pd", "Pd", 0.4, 2.48], ["Pd", "O", 0.6, 1.9],
                                       ["O", "O", 0.05, 2.6]]},
        "reservoir": {"O": {"type": "diatomic", "D0_eV": 5.12, "e_atom_eV": 0.0,
                            "bond_length_angstrom": 1.2075, "vibrational_wavenumber_cm": 1580.2},
                      "Pd": {"type": "bulk", "e_bulk_eV": -2.3, "mean_wavenumber_cm": 110.0}},
        "replica_exchange": {"temperatures_K": [500, 700, 900], "pressures": [1e-6, 1e-3, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 100,
                             "x0": 0.7, "diffusion": "mc", "diffusion_steps": 1,
                             "record_every": 2, "seed": 1,
                             "bias_columns": bias or {"species": "Pd", "shifts_kT": [5.0, 10.0]}},
        "checkpoint": {"chunk": 100},
        "phase_diagram": {"T_min_K": 500, "T_max_K": 900, "n_T": 4, "p_min": 1e-6,
                          "p_max": 1.0, "n_p": 4, "pressure_unit": "bar", "observable": "N_O",
                          "coverage_per": "surface_atom", "extra_observables": ["N_Pd"],
                          "heat_capacity": True},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_setup_grid_metadata_and_summary(tmp_path, capsys):
    setup = setup_from_config(_config(tmp_path), verbose=True)
    printed = capsys.readouterr().out
    mu = np.asarray(setup["mu_grid"])
    assert mu.shape == (3, 5, 2) and setup["pressures"] == [1e-6, 1e-3, 1.0]
    kT = np.array([kelvin_to_eV(T) for T in (500, 700, 900)])
    np.testing.assert_allclose(mu[:, 3:, 0], mu[:, 2:3, 0].repeat(2, axis=1))   # O: p_max
    np.testing.assert_allclose(mu[:, 3, 1], mu[:, 2, 1] - 5.0 * kT)
    np.testing.assert_allclose(mu[:, 4, 1], mu[:, 2, 1] - 10.0 * kT)
    meta = setup["metadata"]
    assert meta["bias_columns"] == {"species": ["Pd"], "shifts_kT": {"Pd": [5.0, 10.0]},
                                    "n_physical_columns": 3, "n_bias_columns": 2}
    assert np.asarray(meta["mu_grid_eV"]).shape == (3, 5, 2)
    assert np.allclose(meta["pressure_grid"], [[1e-6, 1e-3, 1.0, 1.0, 1.0]] * 3)
    assert np.asarray(meta["bias_shift_kT_grid"])[0, :, 1].tolist() == [0, 0, 0, 5.0, 10.0]
    assert meta["column_labels"][3] == "1 bar, mu_Pd - 5 kT"
    assert "3 physical + 2 bias columns" in printed
    assert f"900 K: largest shift mu_Pd {-10 * kT[2]:+.4f} eV" in printed
    # the mapping form gives the same grid
    other = setup_from_config(_config(tmp_path, shifts_kT={"Pd": [5.0, 10.0]}), verbose=False)
    np.testing.assert_allclose(other["mu_grid"], mu)


@pytest.mark.parametrize("change, match", [
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Ag", "shifts_kT": [1.0]}),
     "not an exchanged species"),
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Pd",
                                                          "shifts_kT": [2.0, 1.0]}),
     "strictly increasing"),
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Pd",
                                                          "shifts_kT": [0.0, 1.0]}),
     "positive"),
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Pd", "shifts_kT": []}),
     "at least one"),
    (lambda c: c["replica_exchange"].update(bias_columns={"shifts_kT": {"Pd": [1.0],
                                                                        "O": [1.0, 2.0]}}),
     "same number"),
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Pd"}), "shifts_kT"),
    (lambda c: c["replica_exchange"].update(bias_columns={"species": "Pd", "shift": [1.0]}),
     "unknown"),
    (lambda c: c["replica_exchange"].update(adapt_ladder={"chunks": 1}),
     "cannot be combined with replica_exchange.adapt_ladder"),
    (lambda c: c["phase_diagram"].update(suggest_ladder=True), "phase_diagram.suggest_ladder"),
    (lambda c: c["phase_diagram"].update(merge=["elsewhere"]), "phase_diagram.merge"),
    (lambda c: (c["replica_exchange"].pop("pressures"),
                c["replica_exchange"].update(mu_eV=[[[-1, -2]] * 3] * 3)), "pressure ladder"),
], ids=["unknown-species", "not-increasing", "zero", "empty", "lengths", "no-shifts",
        "bad-key", "adapt-ladder", "suggest-ladder", "merge", "mu-ladder"])
def test_bad_configs_raise(tmp_path, change, match):
    cfg = _config(tmp_path)
    change(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_yaml_run_analysis_and_resume(tmp_path):
    yaml = pytest.importorskip("yaml")
    from regc.runner import main, run_from_config
    cfg = _config(tmp_path)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert main([str(path), "-q"]) == 0
    out = tmp_path / "out"
    status = json.loads((out / "status.json").read_text())
    assert status["shape"] == [3, 5] and len(status["labels"]["column_labels"]) == 5
    assert len(status["round_trips_bias"]) == 15
    cfg["replica_exchange"]["n_steps"] = 200                # extend: resume from step 100
    path.write_text(yaml.safe_dump(cfg))
    res = run_from_config(cfg, verbose=False, resume=True)
    assert res["status"].resumed_from == 100 and res["status"].steps_done == 200
    assert res["sim"].mu_grid.shape == (3, 5, 2)
    # the physical phase diagram, and <N_Pd> on every sampled state
    assert res["theta_grid"].shape == (4, 4) and res["theta_N_Pd"].shape == (4, 4)
    states = res["states_N_Pd"]
    assert states.shape == (3, 5) and np.isfinite(states).all()
    assert states[:, 3:].mean() < states[:, :3].mean()      # Pd leaves where mu_Pd is lowered
    for name in ("phase_diagram.png", "phase_diagram_N_Pd.png", "phase_diagram_Pd_removed.png",
                 "phase_diagram_states_N_Pd.png", "phase_diagram_heat_capacity.png"):
        assert (out / name).exists(), name
    raw = np.load(out / "phase_diagram_data.npz")
    assert raw["states_N_Pd"].shape == (3, 5) and "round_trips_bias" in raw
    mixing = json.loads((out / "mixing.json").read_text())
    assert len(mixing["round_trips_bias"]) == 15
    assert any(p["a"][1] == 2 and p["b"][1] == 3 for p in mixing["pairs"])   # phys -> bias
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["bias_columns"]["n_physical_columns"] == 3
    # a changed bias ladder cannot resume
    cfg["replica_exchange"]["bias_columns"]["shifts_kT"] = [5.0, 9.0]
    with pytest.raises(ValueError, match="cannot resume"):
        run_from_config(cfg, verbose=False, resume=True)
    # merge / suggest_ladder refused on re-analysis too
    from regc.runner import analyze_from_config
    cfg["replica_exchange"]["bias_columns"]["shifts_kT"] = [5.0, 10.0]
    cfg["phase_diagram"]["suggest_ladder"] = True
    with pytest.raises(ValueError, match="bias_columns"):
        analyze_from_config(cfg, verbose=False)


# ---------------------------------------------------------------- bias columns at their own pressure
def test_bias_pressure_grid_labels_metadata_and_summary(tmp_path, capsys):
    cfg = _config(tmp_path, species="Pd", shifts_kT=[5.0, 10.0], pressure=1.0e4)
    setup = setup_from_config(cfg, verbose=True)
    printed = capsys.readouterr().out
    feed, T_K = setup["feed"], [500.0, 700.0, 900.0]
    mu = np.asarray(setup["mu_grid"])
    kT = np.array([kelvin_to_eV(T) for T in T_K])
    at_p = np.asarray(feed.mu_grid(T_K, [1.0e4], unit="bar"))[:, 0]       # (n_T, n_species)
    assert mu.shape == (3, 5, 2)
    np.testing.assert_array_equal(mu[:, :3], feed.mu_grid(T_K, [1e-6, 1e-3, 1.0], unit="bar"))
    for j, s in enumerate([5.0, 10.0]):
        np.testing.assert_allclose(mu[:, 3 + j, 0], at_p[:, 0], rtol=0, atol=1e-12)   # O at p_bias
        np.testing.assert_allclose(mu[:, 3 + j, 1], at_p[:, 1] - s * kT, rtol=0, atol=1e-12)
        for it, T in enumerate(T_K):                       # the same through mu_at_T_eV
            np.testing.assert_allclose(mu[it, 3 + j],
                                       feed.mu_at_T_eV(kT[it], 1.0e4, unit="bar")
                                       - np.array([0.0, s * kT[it]]), atol=1e-12)
    assert np.all(mu[:, 3, 0] > mu[:, 2, 0])               # O: higher pressure, higher mu
    meta = setup["metadata"]
    assert meta["bias_columns"]["pressure"] == 1.0e4
    assert np.allclose(meta["pressure_grid"], [[1e-6, 1e-3, 1.0, 1e4, 1e4]] * 3)
    assert meta["column_labels"][3:] == ["1e+04 bar, mu_Pd - 5 kT", "1e+04 bar, mu_Pd - 10 kT"]
    assert meta["column_labels"][2] == "1 bar"
    assert "bias columns at p = 10000 bar (p_max = 1 bar)" in printed
    # pressure = p_max is the default grid; without it, no 'pressure' key (older runs resume)
    same = setup_from_config(_config(tmp_path, species="Pd", shifts_kT=[5.0, 10.0],
                                     pressure=1.0), verbose=False)
    plain = setup_from_config(_config(tmp_path), verbose=False)
    np.testing.assert_allclose(same["mu_grid"], plain["mu_grid"], rtol=0, atol=1e-12)
    assert "pressure" not in plain["metadata"]["bias_columns"]
    assert plain["bias_columns"]["pressure"] is None


def test_bias_column_mu_grid_base_argument():
    T_K = [300.0, 600.0]
    mu = np.array([[[-1.0, -3.0], [-0.5, -3.0]], [[-1.2, -3.1], [-0.7, -3.1]]])
    base = np.array([[-0.2, -2.9], [-0.4, -3.0]])
    ext, _ = bias_column_mu_grid(mu, T_K, SPECIES, {"Pd": [1.0]}, base=base)
    np.testing.assert_allclose(ext[:, 2, 0], base[:, 0])
    np.testing.assert_allclose(ext[:, 2, 1], base[:, 1] - np.array([kelvin_to_eV(T) for T in T_K]))
    ext1, _ = bias_column_mu_grid(mu[..., 0], T_K, ["O"], {"O": [2.0]}, base=base[:, 0])
    assert ext1.shape == (2, 3) and ext1[0, 2] == pytest.approx(-0.2 - 2 * kelvin_to_eV(300.0))
    with pytest.raises(ValueError, match="base"):
        bias_column_mu_grid(mu, T_K, SPECIES, {"Pd": [1.0]}, base=base[:, :1])


@pytest.mark.parametrize("pressure, match", [(0.5, "at least the last physical pressure"),
                                             ("lots", "pressure")])
def test_bias_pressure_validation(tmp_path, pressure, match):
    cfg = _config(tmp_path, species="Pd", shifts_kT=[5.0], pressure=pressure)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_bias_pressure_run_and_resume(tmp_path):
    from regc.runner import run_from_config
    cfg = _config(tmp_path, species="Pd", shifts_kT=[5.0], pressure=1.0e3)
    cfg.pop("phase_diagram")
    cfg["replica_exchange"].update(temperatures_K=[500, 900], pressures=[1e-3, 1.0], n_steps=20)
    cfg["checkpoint"] = {"chunk": 20}
    res = run_from_config(cfg, verbose=False)
    assert res["finished"] and res["sim"].mu_grid.shape == (2, 3, 2)
    meta = json.loads((tmp_path / "out" / "run_metadata.json").read_text())
    assert meta["bias_columns"]["pressure"] == 1.0e3
    assert meta["pressure_grid"][0] == [1e-3, 1.0, 1.0e3]
    status = json.loads((tmp_path / "out" / "status.json").read_text())
    assert status["labels"]["column_labels"][2] == "1e+03 bar, mu_Pd - 5 kT"
    cfg["replica_exchange"]["n_steps"] = 40
    assert run_from_config(cfg, verbose=False, resume=True)["status"].steps_done == 40
    cfg["replica_exchange"]["bias_columns"]["pressure"] = 1.0e2     # moved: refused
    cfg["replica_exchange"]["n_steps"] = 60
    with pytest.raises(ValueError, match="cannot resume"):
        run_from_config(cfg, verbose=False, resume=True)
