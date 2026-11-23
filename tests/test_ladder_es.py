"""
Model-free ladder optimisation by an evolution strategy (`regc.ladder.LadderES`).

Mutation, fitness, selection, the snapshot/restore of the trial runs
(serial and with worker processes) and the YAML pipeline end to end.
"""

import json
import math

import numpy as np
import pytest

from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC
from regc.ladder import (ACCEPT_FLOOR, LadderES, acceptance_by_type, effective_resistance,
                         fitness_noise, fitness_resistance, fitness_uniform,
                         measured_resistances, pair_type, select_survivor)


def _mu_of(T, y):
    return -3.0 + T * y * math.log(10.0)


def _lam(T):
    return 0.5 * T ** -1.5


def _ladder(nT=4, nm=5):
    T = np.array([0.8, 1.0, 1.3, 1.9][:nT] if nT <= 4 else np.geomspace(0.8, 2.0, nT))
    Y = np.array([np.linspace(-2.0 - 0.1 * l, 1.0 + 0.2 * l, nm) for l in range(nT)])
    return T, Y


def _gap_fractions(x):
    g = np.diff(x, axis=-1)
    return g / g.sum(axis=-1, keepdims=True)


# ---- mutation ---------------------------------------------------------------------------------
def test_mutation_keeps_corners_and_order_and_is_reproducible():
    T, Y = _ladder()
    es = LadderES(_mu_of, _lam, T, Y, sigma=1.0)
    for seed in range(5):
        Tn, Yn = es.mutate(T, Y, np.random.default_rng(seed))
        assert Tn[0] == T[0] and Tn[-1] == T[-1] and np.all(np.diff(Tn) > 0)
        np.testing.assert_array_equal(Yn[:, [0, -1]], Y[:, [0, -1]])
        assert np.all(np.diff(Yn, axis=1) > 0)
        assert not np.allclose(Tn, T) and not np.allclose(Yn, Y)
        T2, Y2 = es.mutate(T, Y, np.random.default_rng(seed))
        np.testing.assert_array_equal(Tn, T2)
        np.testing.assert_array_equal(Yn, Y2)
    # the rows move independently
    Tn, Yn = es.mutate(T, Y, np.random.default_rng(1))
    f = _gap_fractions(Yn)
    assert not np.allclose(f[0], f[1])


def test_product_mode_keeps_one_shared_pressure_ladder():
    T, Y = _ladder()
    Y = np.tile(Y[0], (len(T), 1))
    es = LadderES(_mu_of, _lam, T, Y, per_row=False, sigma=1.0)
    Tn, Yn = es.mutate(T, Y, np.random.default_rng(3))
    np.testing.assert_array_equal(Yn, np.tile(Yn[0], (len(T), 1)))
    assert not np.allclose(Yn, Y)
    with pytest.raises(ValueError, match="shared"):
        LadderES(_mu_of, _lam, *_ladder(), per_row=False)


def test_min_gap_fraction_is_honoured():
    T, Y = _ladder(4, 5)
    es = LadderES(_mu_of, _lam, T, Y, sigma=3.0, min_gap_fraction=0.2)
    for seed in range(20):
        Tn, Yn = es.mutate(T, Y, np.random.default_rng(seed))
        assert _gap_fractions(np.log(Tn)).min() >= 0.2 - 1e-9
        assert _gap_fractions(Yn).min() >= 0.2 - 1e-9
    free = LadderES(_mu_of, _lam, T, Y, sigma=3.0)
    assert min(_gap_fractions(free.mutate(T, Y, np.random.default_rng(s))[1]).min()
               for s in range(20)) < 0.2
    with pytest.raises(ValueError, match="min_gap_fraction"):
        LadderES(_mu_of, _lam, T, Y, min_gap_fraction=0.25)   # 4 pressure gaps


@pytest.mark.parametrize("kwargs, match", [
    ({"sigma": 0.0}, "sigma"), ({"population": 0}, "population"),
    ({"fitness": "fast"}, "fitness"), ({"eval_steps": 0}, "eval_steps"),
])
def test_bad_es_settings_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LadderES(_mu_of, _lam, *_ladder(), **kwargs)
    with pytest.raises(ValueError, match="at least 3"):
        LadderES(_mu_of, _lam, [1.0, 2.0], [0.0, 1.0])


# ---- fitness ----------------------------------------------------------------------------------
def _counts(nT, nm, acc_of):
    out = {}
    for l in range(nT):
        for m in range(nm):
            for b in ((l + 1, m), (l, m + 1), (l + 1, m + 1)):
                if b[0] < nT and b[1] < nm:
                    out[((l, m), b)] = (100, int(round(100 * acc_of((l, m), b))))
            if l + 1 < nT and m >= 1:
                out[((l, m), (l + 1, m - 1))] = (100, int(round(100 * acc_of((l, m), None))))
    return out


def test_pair_types():
    assert pair_type(((0, 0), (1, 0))) == "T" and pair_type(((1, 0), (0, 0))) == "T"
    assert pair_type(((2, 1), (2, 2))) == "mu"
    assert pair_type(((0, 1), (1, 0))) == "cross_left"
    assert pair_type(((0, 0), (1, 1))) == "cross_right"
    with pytest.raises(ValueError, match="neighbours"):
        pair_type(((0, 0), (2, 0)))


def test_uniform_fitness():
    even = _counts(3, 3, lambda a, b: 0.3)
    assert fitness_uniform(even) == 0.0
    # the patterns are scored separately: different but even means still give 0
    by_kind = {"T": 0.5, "mu": 0.2, "cross_left": 0.1, "cross_right": 0.05}
    split = {p: (100, int(100 * by_kind[pair_type(p)])) for p in even}
    assert fitness_uniform(split) == pytest.approx(0.0, abs=1e-15)
    assert set(acceptance_by_type(split)) == set(by_kind)
    # Eq. 6 within one pattern: two T pairs at 0.5 and 0.1 -> 2 * 0.2^2
    counts = {((0, 0), (1, 0)): (10, 5), ((1, 0), (2, 0)): (10, 1), ((0, 0), (0, 1)): (0, 0)}
    assert fitness_uniform(counts) == pytest.approx(0.08)
    assert acceptance_by_type(counts) == {"T": [0.5, 0.1]}   # unattempted pairs left out


def test_resistance_fitness_matches_effective_resistance():
    rng = np.random.default_rng(0)
    counts = {p: (int(a), int(rng.integers(0, a + 1)))
              for p, a in zip(_counts(3, 4, lambda a, b: 0.0),
                              rng.integers(20, 60, size=len(_counts(3, 4, lambda a, b: 0.0))))}
    pairs = sorted(counts)
    att = np.array([counts[p][0] for p in pairs], float)
    acc = np.array([counts[p][1] for p in pairs], float)
    g = np.maximum(acc / att, ACCEPT_FLOOR) * att / att.sum()
    ia = np.array([a[0] * 4 + a[1] for a, _ in pairs])
    ib = np.array([b[0] * 4 + b[1] for _, b in pairs])
    R_T = effective_resistance(12, ia, ib, g, range(4), range(8, 12))
    R_mu = effective_resistance(12, ia, ib, g, range(0, 12, 4), range(3, 12, 4))
    assert measured_resistances((3, 4), counts) == pytest.approx((R_T, R_mu), rel=1e-12)
    assert fitness_resistance((3, 4), counts) == pytest.approx(math.log(R_T) + math.log(R_mu))
    # a better connected ladder has a lower fitness; a missing T pair keeps the graph connected
    better = {p: (a, min(a, c + 5)) for p, (a, c) in counts.items()}
    assert fitness_resistance((3, 4), better) < fitness_resistance((3, 4), counts)
    del counts[((0, 0), (1, 0))]
    assert np.isfinite(fitness_resistance((3, 4), counts))


def test_fitness_noise_shrinks_with_the_attempts():
    rng = np.random.default_rng(1)
    acc = {p: rng.uniform(0.1, 0.6) for p in _counts(3, 3, lambda a, b: 0.0)}
    small = {p: (20, int(round(20 * a))) for p, a in acc.items()}
    large = {p: (2000, int(round(2000 * a))) for p, a in acc.items()}
    se_s, se_l = fitness_noise((3, 3), small), fitness_noise((3, 3), large)
    assert all(x > 0 for x in se_s) and all(x > 0 for x in se_l)
    assert se_l[0] < se_s[0] / 4 and se_l[1] < se_s[1] / 4
    assert fitness_noise((3, 3), small) == se_s          # its own fixed seed
    assert fitness_noise((3, 3), {}) == (0.0, 0.0)


# ---- selection --------------------------------------------------------------------------------
def test_select_survivor():
    assert select_survivor(1.0, [2.0, 0.5, 0.7]) == 1
    assert select_survivor(1.0, [1.0, 2.0]) is None            # ties keep the parent
    assert select_survivor(None, [3.0, 2.0]) == 1
    assert select_survivor(1.0, [float("nan"), 0.9]) == 1
    assert select_survivor(1.0, [float("nan")]) is None


def _lj(rng):
    return BulkSystem(4.0, LJCalculator(1.0, 1.0, 2.0))


def _sim(nT=3, nm=3, seed=4, n_workers=1):
    T, Y = _ladder(nT, nm)
    mus = np.array([[_mu_of(T[l], Y[l, m]) for m in range(nm)] for l in range(nT)])
    return ReplicaExchangeGCMC(list(T), mus, _lj, seed=seed, n_workers=n_workers,
                               lambda_cubed_of_T=_lam), T, Y


def test_selection_with_a_stubbed_evaluator():
    sim, T, Y = _sim()
    with sim:
        es = LadderES(_mu_of, _lam, T, Y, generations=2, population=3, sigma=0.5)
        pairs = [((0, 0), (1, 0)), ((1, 0), (2, 0)), ((0, 0), (0, 1)), ((0, 1), (0, 2))]
        uneven = {p: (10, k) for k, p in enumerate(pairs)}
        even = {p: (10, 4) for p in pairs}
        calls = []

        def stub(scores):
            def evaluate(T_, Y_, mu_):
                calls.append(np.array(T_))
                return scores[len(calls) - 1]
            return evaluate
        # generation 1: parent uneven, candidate 2 even -> it survives
        rec = es.run_generation(sim, evaluate=stub([uneven, uneven, even, uneven]))
        assert [e["label"] for e in rec["evaluations"]] == ["parent", "candidate 1",
                                                            "candidate 2", "candidate 3"]
        assert rec["changed"] and rec["survivor"] == 2 and rec["evaluations"][2]["survived"]
        assert rec["fitness"] == 0.0 and rec["parent_fitness"] > 0
        np.testing.assert_allclose(sim.temperatures, rec["evaluations"][2]["temperatures"])
        np.testing.assert_allclose(sim.temperatures, calls[2])
        np.testing.assert_allclose(sim.mu_grid, es.mu_grid(rec["temperatures"], rec["log10p"]))
        assert sim.ladder_version == 1 and es.current_log10p(sim).tolist() == rec["log10p"]
        assert sim.grid[(1, 1)].state.temperature == sim.temperatures[1]
        # generation 2: every candidate worse than the re-evaluated parent -> kept
        calls.clear()
        rec2 = es.run_generation(sim, evaluate=stub([even, uneven, uneven, uneven]))
        assert not rec2["changed"] and rec2["survivor"] == 0 and rec2["fitness"] == 0.0
        np.testing.assert_allclose(sim.temperatures, rec["temperatures"])
        assert sim.ladder_version == 1 and len(sim.ladder_history) == 2
        assert not es.pending(sim) and es.to_json(sim)["frozen"]
        assert "parent kept" in es.describe(rec2) and "ladder frozen" in es.describe(rec2)
        assert "candidate 2 survives" in es.describe(rec)
        with pytest.raises(ValueError, match="frozen"):
            es.run_generation(sim, evaluate=stub([even] * 4))
        # the ES stream continues where the history left it, reproducibly
        assert rec2["rng_state"] != rec["rng_state"]


def test_parent_not_reevaluated_uses_its_stored_fitness():
    sim, T, Y = _sim()
    with sim:
        es = LadderES(_mu_of, _lam, T, Y, generations=2, population=2,
                      reevaluate_parent=False)
        pairs = [((0, 0), (1, 0)), ((1, 0), (2, 0))]
        a = {pairs[0]: (10, 1), pairs[1]: (10, 9)}       # uniform fitness 0.32
        b = {pairs[0]: (10, 4), pairs[1]: (10, 6)}       # 0.02
        seq = iter([a, b, a, a, a])
        rec = es.run_generation(sim, evaluate=lambda *_: next(seq))
        assert len(rec["evaluations"]) == 3 and rec["changed"] and rec["survivor"] == 1
        rec2 = es.run_generation(sim, evaluate=lambda *_: next(seq))
        assert [e["label"] for e in rec2["evaluations"]] == ["candidate 1", "candidate 2"]
        # the stored 0.02 beats both candidates at 0.32
        assert not rec2["changed"] and rec2["parent_fitness"] == rec["fitness"]


# ---- snapshot / restore -----------------------------------------------------------------------
def _configs(sim):
    return {k: (r.system.positions.copy(), r.energy, r.system.n) for k, r in sim.grid.items()}


def _same_configs(a, b):
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k][0], b[k][0])
        assert a[k][1] == b[k][1] and a[k][2] == b[k][2]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_snapshot_restore_and_continuation_from_the_survivor(n_workers):
    sim, T, Y = _sim(n_workers=n_workers)
    kw = dict(x0=0.7, diffusion_steps=3, record_every=10)
    with sim:
        sim.run_regc(60, verbose=False, **kw)
        sim.snapshot_pair_stats(60)
        before = _configs(sim)
        samples = json.dumps({str(k): v for k, v in sim.samples.items()})
        pair_stats = {k: list(v) for k, v in sim.pair_stats.items()}
        rng_state = sim.rng.bit_generator.state
        blob = sim.snapshot_state()
        # a trial run on another ladder, then back
        es = LadderES(_mu_of, _lam, T, Y, generations=1, population=2, sigma=0.4)
        Tc, Yc = es.mutate(T, Y, np.random.default_rng(0))
        sim.assign_states(Tc, es.mu_grid(Tc, Yc))
        sim.run_regc(40, verbose=False, **kw)
        sim.restore_state(blob)
        _same_configs(before, _configs(sim))
        assert json.dumps({str(k): v for k, v in sim.samples.items()}) == samples
        assert {k: list(v) for k, v in sim.pair_stats.items()} == pair_stats
        assert sim.rng.bit_generator.state == rng_state
        np.testing.assert_allclose(sim.temperatures, T)
        drift = sim.check_energies(1e-9)
        assert max(abs(d) for d, _ in drift.values()) < 1e-9
        # a whole generation: samples and counts as before, configurations the survivor's
        rec = es.run_generation(sim, dict(kw), eval_steps=40)
        assert len(rec["evaluations"]) == 3 and sum(e["survived"] for e in rec["evaluations"]) == 1
        assert all(sum(e["attempts"]) > 0 for e in rec["evaluations"])
        # every evaluation sees the same swap schedule
        assert len({tuple(e["attempts"]) for e in rec["evaluations"]}) == 1
        if rec["changed"]:   # set_ladder: the samples of the old states are dropped
            assert all(len(v["U"]) == 0 for v in sim.samples.values()) and sim.first_index == 6
        else:
            assert json.dumps({str(k): v for k, v in sim.samples.items()}) == samples
        assert {k: list(v) for k, v in sim.pair_stats.items()} == pair_stats
        drift = sim.check_energies(1e-9)
        assert max(abs(d) for d, _ in drift.values()) < 1e-9
        after = _configs(sim)
        win = rec["evaluations"][rec["survivor"]]
        # redo the survivor's evaluation by hand from the snapshot: same end configurations
        sim.restore_state(blob)
        sim.assign_states(win["temperatures"], es.mu_grid(win["temperatures"], win["log10p"]))
        sim.run_regc(40, burn_in_steps=40, verbose=False, **kw)
        _same_configs(after, _configs(sim))


def test_warmup_and_evaluations_extended_to_min_attempts():
    sim, T, Y = _sim()
    kw = dict(x0=0.7, diffusion_steps=3, record_every=10)
    with sim:
        sim.run_regc(20, verbose=False, **kw)
        sim.snapshot_pair_stats(20)
        samples = json.dumps({str(k): v for k, v in sim.samples.items()})
        pair_stats = {k: list(v) for k, v in sim.pair_stats.items()}
        es = LadderES(_mu_of, _lam, T, Y, generations=2, population=2, eval_steps=10,
                      warmup_steps=30, min_attempts_per_pair=4, eval_steps_max=200)
        lines = []
        rec = es.run_generation(sim, dict(kw), progress=lines.append)
        assert rec["warmup"]["steps"] == 30 and "warm-up" in lines[0]
        steps = {e["steps"] for e in rec["evaluations"]}
        assert len(steps) == 1 and steps.pop() % 10 == 0     # same length for every evaluation
        for e in rec["evaluations"]:
            assert e["min_attempts"] >= 4 or e["steps"] == 200
            assert len(e["attempts"]) == 20                  # every pair of the 3 x 3 grid
            assert e["fitness_se"] == e["fitness_uniform_se"] > 0
        assert "+-" in lines[1] and ">= " in lines[1]
        if not rec["changed"]:
            assert json.dumps({str(k): v for k, v in sim.samples.items()}) == samples
        assert {k: list(v) for k, v in sim.pair_stats.items()} == pair_stats
        rec2 = es.run_generation(sim, dict(kw))
        assert rec2["warmup"] is None                        # generation 1 only
        assert "+-" in es.describe(rec2)


def test_generation_is_independent_of_the_worker_count():
    recs = []
    for n_workers in (1, 2):
        sim, T, Y = _sim(n_workers=n_workers, seed=7)
        with sim:
            sim.run_regc(30, x0=0.7, diffusion_steps=3, record_every=10, verbose=False)
            es = LadderES(_mu_of, _lam, T, Y, generations=1, population=2, seed=5)
            recs.append(es.run_generation(sim, dict(x0=0.7, diffusion_steps=3, record_every=10),
                                          eval_steps=30))
            recs[-1] = (recs[-1], _configs(sim))
    for key in ("survivor", "changed", "temperatures", "log10p", "fitness"):
        assert recs[0][0][key] == recs[1][0][key]
    assert ([e["accepted"] for e in recs[0][0]["evaluations"]]
            == [e["accepted"] for e in recs[1][0]["evaluations"]])
    _same_configs(recs[0][1], recs[1][1])


def _timeless(x):
    """A ladder history without its wall times."""
    if isinstance(x, dict):
        return {k: _timeless(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [_timeless(v) for v in x]
    return x


# ---- the YAML pipeline ------------------------------------------------------------------------
def _config(tmp_path, n_steps=200, out="out", **es):
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
            "adapt_ladder": {"method": "es", "generations": 2, "population": 2,
                             "eval_steps": 40, "warmup_steps": 20, **es}},
        "checkpoint": {"chunk": 100},
        "phase_diagram": {"T_min_K": 300, "T_max_K": 500, "n_T": 3, "p_min": 1.0e-6,
                          "p_max": 1.0e+2, "n_p": 4},
        "output": {"directory": str(tmp_path / out)},
    }


def test_es_ladder_run_is_analysed_and_resumes_mid_es(tmp_path, capsys, monkeypatch):
    pytest.importorskip("ase")
    pytest.importorskip("pymbar")
    from regc.runner import run_from_config
    from regc.units import kelvin_to_eV
    out = tmp_path / "out"
    res = run_from_config(_config(tmp_path), verbose=True)
    printed = capsys.readouterr().out
    assert "ladder ES (model free): 2 generations x (2 candidates + parent) x 40 steps" in printed
    assert "= 240 REGC steps + 20 warm-up before the run" in printed
    assert "ladder ES warm-up: 20 steps" in printed
    assert "ladder ES 1/2 parent: fitness uniform" in printed
    assert "ladder ES generation 2/2" in printed and "ladder frozen" in printed
    lad = json.loads((out / "ladder.json").read_text())
    assert lad["method"] == "es" and lad["frozen"] and lad["settings"]["eval_steps"] == 40
    assert lad["history"][0]["warmup"]["steps"] == 20 and lad["history"][1]["warmup"] is None
    assert [h["generation"] for h in lad["history"]] == [1, 2]
    for h in lad["history"]:
        assert h["step"] == 0 and len(h["evaluations"]) == 3
        assert sum(e["survived"] for e in h["evaluations"]) == 1
        for e in h["evaluations"]:
            assert {"fitness_uniform", "fitness_resistance", "R_T", "R_mu", "attempts",
                    "accepted", "pairs", "log10p", "temperatures_K", "fitness_uniform_se",
                    "fitness_resistance_se", "min_attempts", "steps"} <= set(e)
    last = lad["history"][-1]
    T_K = last["temperatures_K"]
    assert T_K[0] == pytest.approx(300.0) and T_K[-1] == pytest.approx(500.0)
    # the production run sampled the final ladder; ES steps are not in n_steps
    s = np.load(out / "samples.npz")
    np.testing.assert_allclose(s["temperatures"], [kelvin_to_eV(T) for T in T_K])
    np.testing.assert_allclose(s["mu_grid"], last["mu_grid"])
    assert s["U"].shape == (9, 20) and int(s["first_index"]) == 0
    assert int(s["ladder_version"]) == lad["version"] == res["sim"].ladder_version
    assert res["status"].steps_done == 200
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["sampling"]["adapt_ladder"]["method"] == "es"
    if lad["version"]:
        np.testing.assert_allclose(meta["temperatures_K"], T_K)
        np.testing.assert_allclose(meta["pressure_grid"], last["pressure_grid"])
    mix = json.loads((out / "mixing.json").read_text())
    assert [c["ladder_version"] for c in mix["chunks"]] == [lad["version"]] * 2
    assert sum(sum(c["attempts"]) for c in mix["chunks"]) == sum(
        p["attempts"] for p in mix["pairs"])   # ES trial swaps are not in mixing.json
    assert np.all(np.isfinite(res["theta_grid"])) and res["theta_grid"].shape == (3, 4)

    # interrupted after the first generation (its checkpoint written), then resumed
    from regc import ladder as ladder_mod
    real = ladder_mod.LadderES.run_generation
    calls = []

    def interrupted(self, sim, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(self, sim, *a, **k)
    monkeypatch.setattr(ladder_mod.LadderES, "run_generation", interrupted)
    cfg = _config(tmp_path, out="cut")
    with pytest.raises(KeyboardInterrupt):
        run_from_config(cfg, verbose=False)
    monkeypatch.setattr(ladder_mod.LadderES, "run_generation", real)
    cut = tmp_path / "cut"
    assert len(json.loads((cut / "ladder.json").read_text())["history"]) == 1
    again = run_from_config(cfg, verbose=False, resume=True)
    assert again["status"].resumed_from == 0 and again["status"].steps_done == 200
    assert (_timeless(json.loads((cut / "ladder.json").read_text())["history"])
            == _timeless(lad["history"]))
    np.testing.assert_array_equal(np.load(cut / "samples.npz")["U"], s["U"])
    # resuming the finished production run keeps the frozen ladder
    longer = run_from_config(_config(tmp_path, n_steps=300), verbose=False, resume=True)
    assert longer["status"].resumed_from == 200 and longer["sim"].ladder_version == lad["version"]
    assert json.loads((out / "ladder.json").read_text())["history"] == lad["history"]


@pytest.mark.parametrize("change, match", [
    (lambda c: c.pop("checkpoint"), "checkpoint section"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(trust=1.5), "unknown key"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(method="genetic"), "method"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(sigma=0), "sigma"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(population=0), "population"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(generations=1.5), "generations"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(eval_steps=0), "eval_steps"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(fitness="fast"), "fitness"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(per_row="yes"), "per_row"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(reevaluate_parent=1),
     "reevaluate_parent"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(min_gap_fraction=0.5),
     "min_gap_fraction"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(warmup_steps=-1), "warmup_steps"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(min_attempts_per_pair=0),
     "min_attempts_per_pair"),
    (lambda c: c["replica_exchange"]["adapt_ladder"].update(eval_steps_max=10), "eval_steps_max"),
    (lambda c: c["replica_exchange"].update(temperatures_K=[300, 500],
                                            pressures=[1e-6, 1e2]), "at least 3"),
    (lambda c: c["replica_exchange"].update(mu_eV=[-1.0, -0.5, 0.0]) or
     c["replica_exchange"].pop("pressures"), "pressure ladder"),
    (lambda c: c["replica_exchange"].update(adapt_ladder={"method": "model", "sigma": 0.3}),
     "unknown key"),
])
def test_bad_es_configs_raise(tmp_path, change, match):
    from regc.runner import validate_config
    cfg = _config(tmp_path)
    change(cfg)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_es_settings_defaults(tmp_path):
    from regc.runner import _adapt_ladder_settings
    cfg = _config(tmp_path)
    cfg["replica_exchange"]["adapt_ladder"] = {"method": "es"}
    out = _adapt_ladder_settings(cfg)
    assert out == {"method": "es", "generations": 3, "population": 4, "sigma": 0.3,
                   "eval_steps": 100, "fitness": "uniform", "per_row": True,
                   "reevaluate_parent": True, "min_gap_fraction": 0.0, "warmup_steps": 0,
                   "min_attempts_per_pair": None, "eval_steps_max": None}
    cfg["replica_exchange"]["adapt_ladder"] = {"method": "model"}
    assert "method" not in _adapt_ladder_settings(cfg)   # older runs' metadata unchanged
