"""
Merging runs into one MBAR analysis, the ladder advisor and warm-started fill runs,
checked on the ideal gas, where <N> = V exp(beta mu) / Lambda^3 is exact.
"""

import numpy as np
import pytest

from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC, ThermalWavelengthCubed
from regc.io import merge_samples

pytest.importorskip("pymbar")
from regc import MBARSurfaceAnalysis  # noqa: E402
from regc.diagnostics import (ladder_pairs, neighbour_overlap, suggest_ladder,  # noqa: E402
                              system_factory_from_configurations)

V = 27.0


def _ideal(rng):
    return BulkSystem(3.0, LJCalculator(1.0, 0.0, 1.4))


def _run(mus, seed, n_cycles=300, **kw):
    with ReplicaExchangeGCMC([1.0], [list(mus)], _ideal, seed=seed, **kw) as sim:
        sim.run(n_cycles=n_cycles, sweeps_per_cycle=20, record_every=1, verbose=False)
    return sim


def test_merged_runs_reproduce_the_exact_answer_between_them():
    low, high = _run([-2.0, -1.6], 1), _run([-1.0, -0.6], 2)
    merged = merge_samples(low, high, labels=["low", "high"])
    assert sorted(merged.samples) == [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)]
    assert merged.state_table[(1, 0, 1)][1] == pytest.approx(-0.6)
    a = MBARSurfaceAnalysis(merged)
    for mu in (-2.0, -1.3, -0.6):
        assert a.mean_N(1.0, mu)[0] == pytest.approx(V * np.exp(mu), rel=0.1)
    # each run alone is far from its other end
    alone = MBARSurfaceAnalysis(low).mean_N(1.0, -0.6)[0]
    assert abs(alone - V * np.exp(-0.6)) > abs(a.mean_N(1.0, -0.6)[0] - V * np.exp(-0.6))
    # bootstrap and discard work on merged sets
    d = merged.discard(0.5)
    assert d.first_index[(1, 0, 0)] == len(merged.samples[(1, 0, 0)]["U"]) // 2
    assert MBARSurfaceAnalysis(d).Nk.sum() < a.Nk.sum()


def test_merge_refuses_incompatible_runs_and_drops_differing_observables():
    from regc import CoordinationNumber
    a = _run([-1.0], 3, n_cycles=20)
    b = _run([-0.8], 4, n_cycles=20, lambda_cubed_of_T=ThermalWavelengthCubed(20.0))
    with pytest.raises(ValueError, match="Lambda"):
        merge_samples(a, b)
    c = _run([-0.8], 5, n_cycles=20, observables=[CoordinationNumber("cn", 1.0)])
    with pytest.warns(UserWarning, match="cn"):
        m = merge_samples(a, c)
    assert not any(k.startswith("obs:") for v in m.samples.values() for k in v)
    with pytest.raises(ValueError, match="two"):
        merge_samples(a)


def test_ladder_pairs_and_overlap_of_merged_runs():
    m = merge_samples(_run([-2.0, -1.6], 6), _run([-1.8], 7))
    a = MBARSurfaceAnalysis(m)
    pairs = ladder_pairs(a)
    # one temperature: the three mu values in order, -2.0 | -1.8 | -1.6
    assert pairs["T"] == []
    assert [(p[0], p[1]) for p in pairs["y"]] == [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (0, 0, 1))]
    ov = neighbour_overlap(a)
    assert set(ov["pairs"]) == set(pairs["y"]) and all(0 < o <= 1 for o in ov["pairs"].values())


def test_advisor_fills_a_planted_gap_and_nothing_else():
    sim = _run([-3.0, -2.8, -0.4], 8)          # a wide gap between -2.8 and -0.4
    a = MBARSurfaceAnalysis(sim)
    s = suggest_ladder(a, target_overlap=0.1)
    assert len(s.gaps) == 1 and s.gaps[0]["axis"] == "y"
    assert all(-2.8 < y < -0.4 for y in s.new_y) and len(s.new_y) >= 2
    assert s.fill_runs[0]["T"].tolist() == [1.0]
    text = s.to_config(kelvin=False)
    assert "mu_eV" in text and "temperatures:" in text
    # one fill round lifts every pair above the 0.03 adequacy guideline (near the target);
    # asking again shows what is left
    fill = _run(s.new_y, 9)
    merged = MBARSurfaceAnalysis(merge_samples(sim, fill))
    assert min(neighbour_overlap(merged)["pairs"].values()) > 0.05
    again = suggest_ladder(merged, target_overlap=0.1)
    assert all(g["overlap"] > 0.05 for g in again.gaps)
    near = suggest_ladder(MBARSurfaceAnalysis(_run([-1.0, -0.9], 10)), target_overlap=0.1)
    assert not near.gaps and near.fill_runs == []


def test_fill_run_starts_from_the_nearest_stored_configuration(tmp_path):
    cfg = str(tmp_path / "cfg")
    with ReplicaExchangeGCMC([1.0], [[-2.0, 0.0]], _ideal, seed=11,
                             configurations_dir=cfg) as sim:
        sim.run(n_cycles=50, sweeps_per_cycle=20, record_every=5, verbose=False)
    a = MBARSurfaceAnalysis(sim)
    factory = system_factory_from_configurations(_ideal, cfg, a, [1.0], [[-1.9, 0.1]])
    with ReplicaExchangeGCMC([1.0], [[-1.9, 0.1]], factory, seed=12) as new:
        n_start = [new.grid[(0, j)].system.n for j in range(2)]
    n_old = [sim.grid[(0, j)].system.n for j in range(2)]
    assert n_start == n_old                    # each new replica: its neighbour's last structure


def test_merged_warm_start_reaches_the_same_free_energies():
    from regc.analysis import _merged_initial_f_k, _mbar_free_energies_newton
    m = merge_samples(_run([-2.0, -1.6], 13), _run([-1.8, -1.2], 14))
    a = MBARSurfaceAnalysis(m)
    assert not hasattr(a.mbar, "compute_free_energy_differences")   # Newton only by default
    cold = _mbar_free_energies_newton(a.u_kn, a.Nk)
    warm0 = _merged_initial_f_k(a.u_kn, a.Nk, [k[0] for k in a.states])
    warm = _mbar_free_energies_newton(a.u_kn, a.Nk, f0=warm0)
    assert np.allclose(warm, cold, atol=1e-6)
    assert np.allclose(a.mbar.f_k - a.mbar.f_k[0], cold, atol=1e-6)
