"""
Reliability diagnostics: statistical inefficiency, block bootstrap (with a
coverage check on an exactly solvable case), MBAR overlap, swap mixing and
the support of heat-capacity peaks.
"""

import json
import os

import numpy as np
import pytest

from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC, run_with_checkpoints
from regc.diagnostics import (BootstrapResult, _block_resample, neighbour_overlap,
                              replica_inefficiencies, ridge_support, statistical_inefficiency)


def _ar1(phi, n, rng):
    x = np.empty(n)
    x[0] = rng.normal()
    e = rng.normal(size=n) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


# -- statistical inefficiency ------------------------------------------------------
def test_statistical_inefficiency_of_known_series():
    rng = np.random.default_rng(0)
    assert statistical_inefficiency(rng.normal(size=50000)) == pytest.approx(1.0, abs=0.1)
    phi = 0.8   # AR(1): g = (1 + phi) / (1 - phi) = 9
    assert statistical_inefficiency(_ar1(phi, 200000, rng)) == pytest.approx(9.0, rel=0.1)
    assert statistical_inefficiency(np.ones(100)) == 1.0
    assert statistical_inefficiency([1.0, 2.0]) == 1.0


def test_block_resample_draws_consecutive_blocks():
    rng = np.random.default_rng(1)
    idx = _block_resample(103, 10, rng)
    assert idx.size == 103 and idx.min() >= 0 and idx.max() < 103
    starts = idx[::10]
    for k, s in enumerate(starts[:-1]):
        assert np.array_equal(idx[10 * k:10 * k + 10], np.arange(s, s + 10))


# -- the ideal gas: exact <N>, so intervals can be checked ----------------------------
def _ideal(rng):
    return BulkSystem(3.0, LJCalculator(1.0, 0.0, 1.4))


def _ideal_run(seed, n_cycles=300):
    with ReplicaExchangeGCMC([1.0, 1.0], [[-1.0], [-0.7]], _ideal, seed=seed) as sim:
        sim.run(n_cycles=n_cycles, sweeps_per_cycle=20, record_every=1, verbose=False)
    return sim


def test_bootstrap_intervals_cover_the_exact_answer():
    """90 % intervals of <N> at an unsampled mu contain the exact value in
    about 90 % of independent runs (here: at least 11 of 16)."""
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    from regc.diagnostics import bootstrap
    mu, exact = -0.85, 27.0 * np.exp(-0.85)
    hits, widths = 0, []
    for seed in range(16):
        a = MBARSurfaceAnalysis(_ideal_run(seed))
        b = bootstrap(a, lambda an: np.array([an.phase_diagram([1.0], [mu])[0, 0]]), n_boot=60,
                      seed=seed)
        lo, hi = b.interval(0.90)
        hits += int(lo[0] <= exact <= hi[0])
        widths.append((hi[0] - lo[0]) / exact)
    assert hits >= 11
    assert np.median(widths) < 0.3            # informative, not trivially wide


def test_bootstrap_is_reproducible_and_estimate_is_the_full_data_value():
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    from regc.diagnostics import bootstrap
    a = MBARSurfaceAnalysis(_ideal_run(3, n_cycles=150))

    def stat(an):
        return an.phase_diagram([1.0], [-1.0, -0.85, -0.7])[0]
    b1, b2 = bootstrap(a, stat, n_boot=10, seed=5), bootstrap(a, stat, n_boot=10, seed=5)
    assert np.array_equal(b1.replicates, b2.replicates)
    assert np.allclose(b1.estimate, stat(a))
    assert b1.replicates.shape == (10, 3)
    assert set(b1.block_lengths) == {(0, 0), (1, 0)}


def test_fast_resample_solver_matches_pymbar():
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    sim = _ideal_run(4, n_cycles=150)
    a = MBARSurfaceAnalysis(sim)
    idx = {k: np.arange(len(v["U"])) for k, v in sim.samples.items()}   # identity "resample"
    r = a.resampled(idx)
    assert np.allclose(r.mbar.f_k - r.mbar.f_k[0], a.mbar.f_k - a.mbar.f_k[0], atol=1e-8)
    assert r.mean_N(1.0, -0.85)[0] == pytest.approx(a.mean_N(1.0, -0.85)[0], rel=1e-8)
    O_fast, O_ref = r.mbar.compute_overlap()["matrix"], a.mbar.compute_overlap()["matrix"]
    assert np.allclose(O_fast, O_ref, atol=1e-8)


def test_replica_inefficiencies_and_overlap_gaps():
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    near = _ideal_run(5, n_cycles=150)
    ineff = replica_inefficiencies(near)
    assert set(ineff) == {(0, 0), (1, 0)}
    assert all(v["g"] >= 1.0 and v["n_independent"] <= v["n"] for v in ineff.values())
    ov = neighbour_overlap(MBARSurfaceAnalysis(near))
    assert ov["pairs"][((0, 0), (1, 0))] > 0.2 and not ov["gaps"]
    with ReplicaExchangeGCMC([1.0, 1.0], [[-3.0], [1.0]], _ideal, seed=6) as far:
        far.run(n_cycles=150, sweeps_per_cycle=20, record_every=1, verbose=False)
    gaps = neighbour_overlap(MBARSurfaceAnalysis(far))["gaps"]
    assert ((0, 0), (1, 0)) in gaps            # <N> = 1.3 vs 73: no common samples


# -- replica-exchange mixing ------------------------------------------------------------
def test_pair_acceptance_and_round_trips():
    """Two identical states: every swap is accepted, so each walker crosses
    the 2-replica T ladder at every exchange round."""
    with ReplicaExchangeGCMC([1.0, 1.0], [[-1.0], [-1.0]], _ideal, seed=7) as sim:
        ids = {k: r.system.walker_id for k, r in sim.grid.items()}
        sim.run(n_cycles=10, sweeps_per_cycle=2, record_every=1, verbose=False)
        pairs = sim.pair_acceptance()
        assert pairs == {((0, 0), (1, 0)): (10, 1.0)}
        assert sum(n for n, _ in pairs.values()) == sum(a for a, _ in sim.swap_stats.values())
        assert sorted(r.system.walker_id for r in sim.grid.values()) == sorted(ids.values())
        trips = sim.round_trip_counts()["T"]
        # the start counts as a visited end: 10 crossings = 5 round trips each
        assert list(trips) == [5, 5]


def test_driver_writes_mixing_and_resume_keeps_it(tmp_path):
    def make():
        return ReplicaExchangeGCMC([1.0, 1.0], [[-1.0], [-1.0]], _ideal, seed=8)
    # on a 2 x 1 grid only the T pattern with parity 0 has a pair: use many RE steps
    kw = dict(chunk=100, x0=0.1, diffusion="mc_moves", diffusion_steps=1, record_every=5,
              progress=None)
    with make() as sim:
        run_with_checkpoints(sim, str(tmp_path), n_steps=100, **kw)
        first = sim.pair_acceptance()
    with open(os.path.join(tmp_path, "mixing.json")) as f:
        mix = json.load(f)
    assert mix["worst_pair"]["acceptance"] == 1.0 and len(mix["round_trips_T"]) == 2
    with make() as sim:
        run_with_checkpoints(sim, str(tmp_path), n_steps=200, **kw)
        assert sim.pair_acceptance()[((0, 0), (1, 0))][0] > first[((0, 0), (1, 0))][0]


# -- heat-capacity peaks ----------------------------------------------------------------
def test_ridge_support_separates_real_from_noise_peaks():
    from regc import heat_capacity_ridges
    rng = np.random.default_rng(9)
    y = np.linspace(-1, 1, 41)
    base = 1.0 + 3.0 * np.exp(-(y / 0.1) ** 2)                 # a real peak at y = 0
    est = base.copy()
    est[30] += 0.6                                            # a noise spike in this sample only
    reps = base[None, :] + 0.05 * rng.normal(size=(200, y.size))
    r = heat_capacity_ridges([1.0], y, est[None], prominence=0.1)
    sup = dict(zip(map(tuple, r["cells"]), ridge_support(r, reps[:, None, :])))
    assert sup[(0, 20)] == 1.0
    assert sup[(0, 30)] < 0.5


def test_bootstrap_result_agreement_for_labels():
    b = BootstrapResult(np.array([1, 2]), np.array([[1, 2], [1, 3], [1, 2], [0, 2]]), {})
    assert list(b.agreement()) == [0.75, 0.75]


# -- quick wins: swap draw, energy resync, subsampling, explicit pressure units ----------------
def test_checkpoint_keeps_the_swap_draw_of_the_run(tmp_path):
    path = str(tmp_path / "c.pkl")
    with ReplicaExchangeGCMC([1.0, 1.2], [[-1.0], [-1.0]], _ideal, seed=10,
                             skip_empty_swaps=False) as old:
        old.run_regc(n_steps=20, x0=0.5, diffusion="mc_moves", record_every=5, verbose=False)
        old.save_checkpoint(path)
    with ReplicaExchangeGCMC([1.0, 1.2], [[-1.0], [-1.0]], _ideal, seed=10) as new:
        assert new.skip_empty_swaps
        new.load_checkpoint(path)
        assert not new.skip_empty_swaps          # continues with the draw it started with
    import pickle
    st = pickle.load(open(path, "rb"))
    del st["skip_empty_swaps"]                   # a checkpoint from before the option
    pickle.dump(st, open(path, "wb"))
    with ReplicaExchangeGCMC([1.0, 1.2], [[-1.0], [-1.0]], _ideal, seed=10) as new:
        new.load_checkpoint(path)
        assert not new.skip_empty_swaps


def test_energy_check_resyncs_only_real_drift():
    def lj(rng):
        return BulkSystem(4.0, LJCalculator(1.0, 1.0, 1.9))
    with ReplicaExchangeGCMC([1.0], [-2.0], lj, seed=11) as sim:
        sim.run(n_cycles=20, sweeps_per_cycle=20, record_every=5, verbose=False)
        before = {k: r.energy for k, r in sim.grid.items()}
        res = sim.check_energies()
        assert not any(resynced for _, resynced in res.values())
        assert {k: r.energy for k, r in sim.grid.items()} == before   # bit-identical
        key = next(iter(sim.grid))
        sim.grid[key].energy += 0.5
        drift, resynced = sim.check_energies()[key]
        assert resynced and drift == pytest.approx(0.5)
        assert sim.grid[key].energy == pytest.approx(sim.grid[key].system.total_energy())


def test_subsampled_analysis_gives_independent_samples_and_maps_back():
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    from regc.diagnostics import bootstrap
    sim = _ideal_run(12, n_cycles=600)
    full = MBARSurfaceAnalysis(sim)
    thin = MBARSurfaceAnalysis(sim, subsample="g")
    ineff = replica_inefficiencies(sim)
    for k, key in enumerate(thin.states):
        assert thin.Nk[k] == len(range(0, ineff[key]["n"], max(1, int(np.ceil(ineff[key]["g"])))))
    key, local = thin.sample_location(int(thin._offsets[1]) + 2)
    assert key == thin.states[1] and local == int(thin._kept[key][2])
    mu = -0.85
    m_thin, s_thin = thin.mean_N(1.0, mu)
    b = bootstrap(full, lambda a: np.array([a.mean_N(1.0, mu)[0]]), n_boot=60, seed=2)
    assert m_thin == pytest.approx(27.0 * np.exp(mu), rel=0.1)
    assert 0.3 < s_thin / b.std()[0] < 3.0       # an honest error, same order as the bootstrap


def test_pressure_unit_must_be_explicit():
    from regc import MonatomicIdealGasReservoir
    res = MonatomicIdealGasReservoir(4.0)
    with pytest.warns(FutureWarning, match="pressure unit"):
        res.mu_grid([300.0], [1.0])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.mu_grid([300.0], [1.0], unit="bar")
        res.analysis_functions("Pa")
