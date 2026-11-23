"""
The adaptive replica ladder (regc.ladder): the calibrated acceptance model,
the constrained proposals, `ReplicaExchangeGCMC.set_ladder`, checkpoints of
an adapted run, and the YAML pipeline with ``adapt_ladder``.
"""

import json
import math

import numpy as np
import pytest

pytest.importorskip("scipy", reason="pip install scipy to run this test")

from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC  # noqa: E402
from regc.ladder import (AcceptanceModel, CovarianceField, LadderAdapter,  # noqa: E402
                         effective_resistance, propose_ladder, window_counts)

NT, NM = 4, 5


def _mu_of(T, y):
    """Ideal-gas-like mu: kT ln(10) y (Lambda^3 = 1)."""
    return T * math.log(10.0) * y


def _lam(T):
    return 1.0


def _synthetic(seed=0, nT=NT, nm=NM):
    """A ladder, SPD covariances per state, and measured counts of every neighbour pair."""
    rng = np.random.default_rng(seed)
    T = np.geomspace(1.0, 3.0, nT)
    Y = np.tile(np.linspace(-3.0, 1.0, nm), (nT, 1))
    A = rng.normal(size=(nT, nm, 2, 2))
    C = np.einsum("lmij,lmkj->lmik", A, A) + 0.3 * np.eye(2)
    pairs = ([((l, m), (l + 1, m)) for l in range(nT - 1) for m in range(nm)]
             + [((l, m), (l, m + 1)) for l in range(nT) for m in range(nm - 1)]
             + [((l, m), (l + 1, m + 1)) for l in range(nT - 1) for m in range(nm - 1)]
             + [((l, m + 1), (l + 1, m)) for l in range(nT - 1) for m in range(nm - 1)])
    att = rng.integers(50, 200, len(pairs))
    acc = np.round(att * rng.uniform(0.02, 0.9, len(pairs))).astype(int)
    return T, Y, C, pairs, att, acc


def _gaps(T, Y):
    return np.diff(np.log(T)), np.diff(Y, axis=1)


# ---- the model --------------------------------------------------------------------------
def test_calibration_reproduces_the_measured_rates():
    T, Y, C, pairs, att, acc = _synthetic()
    model = AcceptanceModel(T, Y, C, pairs, att, acc, _mu_of, _lam)
    np.testing.assert_allclose(model.acceptance(T, Y), acc / att, rtol=0, atol=1e-10)
    # a pair never accepted is calibrated to its clip, 0.5 / attempts
    acc[0] = 0
    model = AcceptanceModel(T, Y, C, pairs, att, acc, _mu_of, _lam)
    assert model.acceptance(T, Y)[0] == pytest.approx(0.5 / att[0], rel=1e-8)
    with pytest.raises(ValueError, match="attempts"):
        AcceptanceModel(T, Y, C, pairs, np.where(np.arange(len(att)) == 3, 0, att), acc,
                        _mu_of, _lam)


def test_covariance_field_is_exact_on_the_grid_and_clamped_outside():
    T, Y, C, *_ = _synthetic()
    Y = Y + np.linspace(0.0, 0.5, NT)[:, None] * np.array([0, 1, 1, 1, 0])  # rows differ
    f = CovarianceField(T, Y, C)
    for l in range(NT):
        np.testing.assert_allclose(f(np.full(NM, T[l]), Y[l]), C[l], rtol=0, atol=1e-14)
    np.testing.assert_allclose(f([0.1 * T[0]], [Y[0, 0] - 5]), C[:1, 0], atol=1e-14)
    mid = f([math.sqrt(T[0] * T[1])], [Y[0, 0]])[0]       # halfway in ln T, first column
    np.testing.assert_allclose(mid, 0.5 * (C[0, 0] + C[1, 0]), atol=1e-12)


def test_effective_resistance_in_series_and_parallel():
    # 0 -1- 1 -2- 2 (series: 1 + 1/2), plus a parallel edge 0-2 of conductance 1
    ia, ib = np.array([0, 1]), np.array([1, 2])
    assert effective_resistance(3, ia, ib, np.array([1.0, 2.0]), [0], [2]) == pytest.approx(1.5)
    r = effective_resistance(3, np.array([0, 1, 0]), np.array([1, 2, 2]),
                             np.array([1.0, 2.0, 1.0]), [0], [2])
    assert r == pytest.approx(1.0 / (1.0 / 1.5 + 1.0))


def test_window_counts_keep_only_the_current_ladder():
    intervals = [{"ladder_version": 0, "pairs": {((0, 0), (1, 0)): (10, 5)}},
                 {"ladder_version": 1, "pairs": {((0, 0), (1, 0)): (7, 2)}},
                 {"ladder_version": 1, "pairs": {((0, 0), (1, 0)): (3, 1)}}]
    assert window_counts(intervals, 1) == {((0, 0), (1, 0)): (10, 3)}
    assert window_counts(intervals, 0) == {((0, 0), (1, 0)): (10, 5)}


# ---- proposals -----------------------------------------------------------------------------
def test_proposal_keeps_corners_order_trust_and_coverage():
    T, Y, C, pairs, att, acc = _synthetic(1)
    trust, lo, hi = 1.3, 0.5, 2.0
    kw = dict(trust=trust, min_gap_ratio=lo, max_gap_ratio=hi, max_evals=1500)
    p = propose_ladder(T, Y, C, pairs, att, acc, _mu_of, _lam, **kw)
    assert p.improved and sum(map(math.log, p.resistance_after)) < sum(
        map(math.log, p.resistance_before))
    assert p.temperatures[0] == T[0] and p.temperatures[-1] == T[-1]
    np.testing.assert_array_equal(p.log10p[:, [0, -1]], Y[:, [0, -1]])
    gT, gY = _gaps(p.temperatures, p.log10p)
    gT0, gY0 = _gaps(T, Y)
    assert np.all(gT > 0) and np.all(gY > 0)
    eps = 1e-9
    for new, old in ((gT, gT0), (gY, gY0)):
        assert np.all(new / old <= trust + eps) and np.all(new / old >= 1 / trust - eps)
    assert not np.allclose(p.log10p, p.log10p[:1])     # per row: rows differ
    np.testing.assert_allclose(p.predicted_before, acc / att, atol=1e-10)
    assert p.mu_grid.shape == (NT, NM)
    np.testing.assert_allclose(p.mu_grid[2, 3], _mu_of(p.temperatures[2], p.log10p[2, 3]))
    # a second step from there: coverage is relative to the INITIAL ladder, cumulatively
    rng = np.random.default_rng(2)
    acc2 = np.round(att * rng.uniform(0.02, 0.9, len(att))).astype(int)
    p2 = propose_ladder(p.temperatures, p.log10p, C, pairs, att, acc2, _mu_of, _lam,
                        initial_temperatures=T, initial_log10p=Y, **kw)
    gT2, gY2 = _gaps(p2.temperatures, p2.log10p)
    for new, prev, init in ((gT2, gT, gT0), (gY2, gY, gY0)):
        assert np.all(new / prev <= trust + eps) and np.all(new / prev >= 1 / trust - eps)
        assert np.all(new / init <= hi + eps) and np.all(new / init >= lo - eps)


def test_product_mode_gives_one_shared_pressure_ladder():
    T, Y, C, pairs, att, acc = _synthetic(3)
    p = propose_ladder(T, Y[0], C, pairs, att, acc, _mu_of, _lam, per_row=False, max_evals=800)
    np.testing.assert_allclose(p.log10p, np.tile(p.log10p[0], (NT, 1)), rtol=0, atol=0)
    with pytest.raises(ValueError, match="shared"):
        propose_ladder(T, p.log10p + np.arange(NT)[:, None] * [0, .1, .1, .1, 0], C, pairs,
                       att, acc, _mu_of, _lam, per_row=False)


def test_bad_proposal_inputs_raise():
    T, Y, C, pairs, att, acc = _synthetic()
    with pytest.raises(ValueError, match="corners"):
        propose_ladder(T * 1.1, Y, C, pairs, att, acc, _mu_of, _lam, initial_temperatures=T)
    with pytest.raises(ValueError, match="trust"):
        propose_ladder(T, Y, C, pairs, att, acc, _mu_of, _lam, trust=0.5)
    with pytest.raises(ValueError, match="monotonic"):
        propose_ladder(T[[0, 2, 1, 3]], Y, C, pairs, att, acc, _mu_of, _lam)
    with pytest.raises(ValueError, match="at least 3"):
        LadderAdapter(_mu_of, _lam, [1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="trust"):
        LadderAdapter(_mu_of, _lam, T, Y[0], trust=1.0)


# ---- the engine -----------------------------------------------------------------------------
def _lj(rng):
    return BulkSystem(4.0, LJCalculator(1.0, 1.0, 2.0))


def _sim(T=(0.8, 1.0, 1.3), mus=((-3.0, -2.5, -2.0),) * 3, seed=4):
    return ReplicaExchangeGCMC(list(T), np.array(mus, dtype=float), _lj, seed=seed,
                               lambda_cubed_of_T=lambda T: 0.5 * T ** -1.5)


def test_set_ladder_moves_the_states_and_drops_the_samples(tmp_path):
    with _sim() as sim:
        sim.run_regc(300, x0=0.8, diffusion_steps=5, record_every=10, verbose=False)
        sim.snapshot_pair_stats(300)
        n_before = len(sim.samples[(0, 0)]["U"])
        energies = {k: r.energy for k, r in sim.grid.items()}
        systems = {k: r.system for k, r in sim.grid.items()}
        sim.convergence_history.append(1.0)
        T_new = [0.8, 1.1, 1.3]
        mu_new = np.array([[-3.0, -2.7, -2.0], [-3.0, -2.4, -2.0], [-3.0, -2.6, -2.0]])
        sim.set_ladder(T_new, mu_new, info={"note": "x"}, step=300)
        for (l, m), r in sim.grid.items():
            assert r.state.temperature == T_new[l] and r.state.mu == mu_new[l, m]
            assert r.state.lambda_cubed == pytest.approx(0.5 * T_new[l] ** -1.5)
            assert r.energy == energies[(l, m)] and r.system is systems[(l, m)]
        assert sim.temperatures == T_new and np.array_equal(sim.mu_grid, mu_new)
        assert all(len(v) == 0 for s in sim.samples.values() for v in s.values())
        assert sim.first_index == n_before and sim.convergence_history == []
        assert sim.ladder_version == 1 and sim.ladder_history == [{"note": "x", "version": 1}]
        # the swap criterion sees the new states
        ra, rb = sim.grid[(0, 1)], sim.grid[(1, 1)]
        Na, Nb = ra.system.n, rb.system.n
        expected = ((1 / 0.8 - 1 / 1.1) * (ra.energy - rb.energy)
                    - (-2.7 / 0.8 + 2.4 / 1.1) * (Na - Nb)
                    + (Na - Nb) * math.log((0.8 / 1.1) ** -1.5))
        assert sim.swap_log_acceptance(ra, rb) == pytest.approx(expected, rel=1e-12)
        drift = sim.check_energies()
        assert max(abs(d) for d, _ in drift.values()) < 1e-9
        # sampling continues on the new ladder, numbered after the discarded samples
        sim.run_regc(100, x0=0.8, diffusion_steps=5, record_every=10, verbose=False)
        sim.snapshot_pair_stats(400)
        chunks = sim.pair_acceptance_by_chunk()
        assert [c["ladder_version"] for c in chunks] == [0, 1]
        assert len(sim.samples[(0, 0)]["U"]) == 10
        with pytest.raises(ValueError, match="set_ladder needs"):
            sim.set_ladder(T_new[:2], mu_new)
        # checkpoint: a sim built with the ORIGINAL ladder accepts it and gets the new one
        path = str(tmp_path / "ck.pkl")
        sim.save_checkpoint(path)
        U = [list(v["U"]) for v in sim.samples.values()]
    with _sim() as again:
        again.load_checkpoint(path)
        assert again.temperatures == T_new and np.array_equal(again.mu_grid, mu_new)
        assert again.grid[(1, 1)].state.temperature == 1.1
        assert again.ladder_version == 1 and again.first_index == n_before
        assert [list(v["U"]) for v in again.samples.values()] == U
    with _sim(T=(0.8, 1.05, 1.3)) as other, pytest.raises(ValueError, match="temperatures"):
        other.load_checkpoint(path)


# ---- the YAML pipeline ------------------------------------------------------------------------
def _config(tmp_path, n_steps=300, **adapt):
    return {
        "structure": {"build": {"surface": "fcc100", "symbol": "Cu", "size": [3, 3, 2],
                                "a": 3.6, "orthogonal": True},
                      "n_frozen_layers": 2, "headroom_angstrom": 4.0},
        "gas": {"species": "H", "insertion_z_range_above_surface": [0.5, 3.5]},
        "calculator": {"type": "PairLJ",
                       "pair_params": [["Cu", "H", 0.15, 1.4], ["H", "H", 0.02, 1.0]]},
        "reservoir": {"type": "monatomic", "e_atom_eV": 0.0},
        "replica_exchange": {
            "temperatures_K": [300, 400, 500], "pressures": [1.0e-6, 1.0e-2, 1.0e+2],
            "pressure_unit": "bar", "scheme": "regc", "n_steps": n_steps, "x0": 0.5,
            "diffusion": "mc", "diffusion_steps": 2, "record_every": 10, "seed": 3,
            "adapt_ladder": {"chunks": 2, "min_attempts": 2, "max_evals": 300, **adapt}},
        "checkpoint": {"chunk": 100},
        "phase_diagram": {"T_min_K": 300, "T_max_K": 500, "n_T": 3, "p_min": 1.0e-6,
                          "p_max": 1.0e+2, "n_p": 4},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_adaptive_ladder_run_freezes_resumes_and_is_analysed(tmp_path, capsys):
    pytest.importorskip("ase")
    pytest.importorskip("pymbar")
    from regc.diagnostics import ladder_pairs
    from regc.runner import run_from_config
    from regc.units import kelvin_to_eV
    out = tmp_path / "out"
    res = run_from_config(_config(tmp_path), verbose=True)
    printed = capsys.readouterr().out
    assert "adaptive ladder: after each of the first 2 chunks" in printed
    assert "ladder 1/2 at step 100" in printed and "ladder frozen" in printed
    lad = json.loads((out / "ladder.json").read_text())
    assert lad["frozen"] and [h["step"] for h in lad["history"]] == [100, 200]
    assert any(h["changed"] for h in lad["history"])
    last = lad["history"][-1]
    T_K = last["temperatures_K"]
    assert T_K[0] == pytest.approx(300.0) and T_K[-1] == pytest.approx(500.0)
    # the outputs describe the final ladder
    s = np.load(out / "samples.npz")
    np.testing.assert_allclose(s["temperatures"], [kelvin_to_eV(T) for T in T_K])
    np.testing.assert_allclose(s["mu_grid"], last["mu_grid"])
    assert int(s["ladder_version"]) == lad["version"] and s["U"].shape == (9, 10)
    assert lad["version"] == 2 and int(s["first_index"]) == 20   # 10 samples dropped twice
    meta = json.loads((out / "run_metadata.json").read_text())
    np.testing.assert_allclose(meta["temperatures_K"], T_K)
    np.testing.assert_allclose(meta["pressure_grid"], last["pressure_grid"])
    assert meta["pressures"] == [1.0e-6, 1.0e-2, 1.0e+2]
    assert meta["ladder_adaptation"]["original"]["temperatures_K"] == [300, 400, 500]
    assert (out / "samples_ladder0.npz").exists()
    mix = json.loads((out / "mixing.json").read_text())
    assert [c["ladder_version"] for c in mix["chunks"]][0] == 0
    # analysis on the adapted grid: MBAR, overlap, and every column paired along T
    assert np.all(np.isfinite(res["theta_grid"])) and res["theta_grid"].shape == (3, 4)
    pairs = ladder_pairs(res["analysis"], pressure=True)
    assert len(pairs["T"]) == 2 * 3 and len(pairs["y"]) == 3 * 2
    # resume after the adaptation: the ladder is restored and the run goes on
    cfg = _config(tmp_path, n_steps=500)
    again = run_from_config(cfg, verbose=False, resume=True)
    assert again["status"].resumed_from == 300 and again["status"].steps_done == 500
    assert again["sim"].temperatures == pytest.approx([kelvin_to_eV(T) for T in T_K])
    assert again["sim"].ladder_version == lad["version"]
    assert json.loads((out / "ladder.json").read_text())["history"] == lad["history"]
    # ... exactly like an uninterrupted run
    straight = _config(tmp_path, n_steps=500)
    straight["output"]["directory"] = str(tmp_path / "straight")
    ref = run_from_config(straight, verbose=False)
    np.testing.assert_array_equal(np.load(tmp_path / "straight" / "samples.npz")["U"],
                                  np.load(out / "samples.npz")["U"])
    assert ref["sim"].temperatures == again["sim"].temperatures
    # a genuinely different setup is still refused
    cfg["replica_exchange"]["pressures"] = [1.0e-6, 1.0e-3, 1.0e+2]
    with pytest.raises(ValueError, match="cannot resume"):
        run_from_config(cfg, verbose=False, resume=True)


@pytest.mark.parametrize("change, match", [
    (lambda c: c.pop("checkpoint"), "checkpoint section"),
    (lambda c: c["checkpoint"].update(chunk=300), "chunk"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(speed=1), "unknown key"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(trust=1.0), "trust"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(min_gap_ratio=2.0), "min_gap_ratio"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(chunks=0), "chunks"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(per_row="yes"), "per_row"),
    (lambda c: c["replica_exchange"].update(temperatures_K=[300, 500],
                                            pressures=[1e-6, 1e2]), "at least 3"),
    (lambda c: c["replica_exchange"].update(mu_eV=[-1.0, -0.5, 0.0]) or
     c["replica_exchange"].pop("pressures"), "pressure ladder"),
])
def test_bad_adapt_ladder_configs_raise(tmp_path, change, match):
    from regc.runner import validate_config
    cfg = _config(tmp_path)
    change(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)
