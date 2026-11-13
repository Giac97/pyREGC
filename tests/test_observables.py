"""
Structural observables (coordination, RDF), the heat capacity of [ZSG] PRL
128, 246101 Eq. (1), and the configurations behind a phase diagram.
"""

import os

import numpy as np
import pytest

from regc import (BulkSystem, CoordinationHistogram, CoordinationNumber, FunctionObservable,
                  LJCalculator, PairLJCalculator, RadialDistribution, ReplicaExchangeGCMC,
                  SlabSystem, ThermalWavelengthCubed, load_samples, save_samples)
from regc.observables import observable_from_config


def _flat_slab(calc=None, positions=None):
    # 4 x 4 frozen layer at z = 0 (spacing 2), periodic 8 x 8 cell
    xy = np.array([[x, y] for x in range(4) for y in range(4)], dtype=float) * 2.0
    frozen = np.column_stack([xy, np.zeros(len(xy))])
    calc = calc or PairLJCalculator({("Fe", "X"): (1.0, 1.2), ("X", "X"): (0.5, 1.0),
                                     ("Fe", "Fe"): (1.0, 1.0)})
    return SlabSystem(8.0, 8.0, 6.0, calc, frozen_positions=frozen,
                      frozen_species=np.array(["Fe"] * len(xy)), positions=positions)


# -- observables on hand-built configurations ---------------------------------------
def test_coordination_and_rdf_exact_on_a_hand_built_slab():
    # on top of (0, 0) at height 1 (1 neighbour within 1.5); a four-fold hollow at
    # (1, 1, 1): 4 neighbours at sqrt(3); across the periodic boundary at (7.5, 0, 1):
    # only (0, 0, 0) is within 1.5 (via x-periodicity, distance sqrt(1.25))
    system = _flat_slab(positions=[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [7.5, 0.0, 1.0]])
    cn = CoordinationNumber("cn", r_cut=1.5)(system)
    assert list(cn) == [2.0, 3.0]  # 1 + 0 + 1 bonds over 3 centres
    cn2 = CoordinationNumber("cn", r_cut=1.8)(system)
    assert list(cn2) == [6.0, 3.0]  # + the 4 hollow bonds
    hist = CoordinationHistogram("h", r_cut=1.8, max_cn=4, center="mobile", neighbor="substrate")
    assert list(hist(system)) == [0, 2, 0, 0, 1]
    # z_range restricts the centres
    assert list(CoordinationNumber("c", 1.8, z_range=(0.5, 0.9))(system)) == [0.0, 0.0]
    rdf = RadialDistribution("g", r_max=3.9, n_bins=39, a="mobile", b="mobile")
    v = rdf(system)
    h, n_a, n_b = v[:39], v[39], v[40]
    assert (n_a, n_b) == (3.0, 3.0)
    assert h.sum() == 6  # 3 pairs, each counted from both sides (self-pairs excluded)
    # mobile-mobile distances: |(0,0)-(1,1)| = sqrt2, |(0,0)-(7.5,0)| = 0.5 (periodic),
    # |(1,1)-(7.5,0)| = sqrt(1.5^2 + 1) = 1.80
    expected = np.histogram([np.sqrt(2), 0.5, np.hypot(1.5, 1.0)] * 2, bins=rdf.edges)[0]
    assert np.array_equal(h, expected)


def test_observable_validation_and_config_factory():
    with pytest.raises(ValueError, match="selection"):
        CoordinationNumber("c", 1.0, center="nonsense:H")
    with pytest.raises(ValueError, match="reserved"):
        ReplicaExchangeGCMC([1.0], [-1.0], lambda rng: BulkSystem(5.0, LJCalculator()),
                            observables=[CoordinationNumber("N", 1.0)])
    o = observable_from_config({"type": "rdf", "name": "g", "r_max": 2.0, "n_bins": 10})
    assert isinstance(o, RadialDistribution) and o.size == 13
    with pytest.raises(ValueError, match="type"):
        observable_from_config({"type": "bogus", "name": "x"})
    big = _flat_slab()
    with pytest.raises(ValueError, match="minimum-image"):
        RadialDistribution("g", r_max=5.0)(big)


def _slab_factory(rng):
    return _flat_slab()


def _height(system):
    return float(system.positions[:, 2].mean()) if system.n else 0.0


# -- recording, parallel workers, save/load --------------------------------------------
def _obs():
    return [CoordinationNumber("cn", 1.8), RadialDistribution("rdf", 3.9, 13, a="mobile", b="all"),
            FunctionObservable("height", _height)]


@pytest.mark.parametrize("scheme", ["run", "run_regc"])
def test_observables_are_recorded_saved_and_reweighted(scheme, tmp_path):
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis

    def go(n_workers):
        with ReplicaExchangeGCMC([0.5, 0.8], [-4.0, -2.0], _slab_factory, seed=2, observables=_obs(),
                                 n_workers=n_workers) as sim:
            if scheme == "run":
                sim.run(n_cycles=60, sweeps_per_cycle=10, record_every=2, verbose=False)
            else:
                sim.run_regc(n_steps=300, x0=0.7, diffusion="mc_moves", diffusion_steps=3,
                             record_every=5, verbose=False)
        return sim
    sim = go(1)
    key = (1, 1)
    n = len(sim.samples[key]['U'])
    assert len(sim.samples[key]['obs:cn']) == n and len(sim.samples[key]['obs:rdf'][0]) == 16
    if scheme == "run_regc":  # workers evaluate the observables: identical results
        assert go(2).samples[key]['obs:rdf'] == sim.samples[key]['obs:rdf']
    path = str(tmp_path / "s.npz")
    save_samples(sim, path)
    loaded = load_samples(path, discard=0.25)
    assert loaded.first_index == int(0.25 * n) and loaded.n_substrate == 0
    assert set(loaded.observables_meta) == {"cn", "rdf", "height"}
    assert loaded.samples[key]['obs:cn'] == sim.samples[key]['obs:cn'][loaded.first_index:]
    analysis = MBARSurfaceAnalysis(loaded)
    # at a sampled state, MBAR reproduces the direct averages closely
    direct = np.mean(loaded.samples[key]['obs:cn'], axis=0)
    rew = analysis.observable_mean("cn", 0.8, -2.0)
    assert np.allclose(rew, direct, rtol=0.25, atol=0.2)
    cn = analysis.mean_coordination("cn", [0.5, 0.8], [-4.0, -2.0])
    assert cn.shape == (2, 2) and np.all((cn[np.isfinite(cn)] >= 0) & (cn[np.isfinite(cn)] < 5))
    r, g = analysis.radial_distribution("rdf", 0.8, -2.0)
    assert r.shape == g.shape == (13,) and np.all(g >= 0)
    assert analysis.phase_diagram([0.8], [-2.0], observable="height").shape == (1, 1)
    with pytest.raises(ValueError, match="components"):
        analysis.phase_diagram([0.8], [-2.0], observable="cn")
    assert analysis.phase_diagram([0.8], [-2.0], observable="cn[centers]")[0, 0] == pytest.approx(
        analysis.mean_N(0.8, -2.0)[0], rel=1e-8)
    ess = analysis.effective_sample_size([0.8, 2.0], [-2.0])
    assert ess[0, 0] > 5 * ess[1, 0]  # far outside the ladder: few effective samples


def test_resume_refuses_changed_observables(tmp_path):
    path = str(tmp_path / "c.pkl")
    with ReplicaExchangeGCMC([0.8], [-2.0], _slab_factory, seed=2, observables=_obs()) as sim:
        sim.run_regc(n_steps=20, x0=0.7, diffusion="mc_moves", record_every=5, verbose=False)
        sim.save_checkpoint(path)
    with ReplicaExchangeGCMC([0.8], [-2.0], _slab_factory, seed=2,
                             observables=[CoordinationNumber("cn", 2.5)]) as other:
        with pytest.raises(ValueError, match="observables"):
            other.load_checkpoint(path)


# -- heat capacity ------------------------------------------------------------------
def _bulk(rng):
    return BulkSystem(4.0, LJCalculator(1.0, 1.0, 1.9))


def test_heat_capacity_is_the_temperature_derivative_at_fixed_beta_mu():
    """Var(E) / kT^2 = d<E>/dT at fixed beta*mu (Lambda^3 ~ T^-3/2): an exact
    identity of the MBAR estimator, checked by finite differences."""
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis
    lam = ThermalWavelengthCubed(20.0)
    with ReplicaExchangeGCMC([0.9, 1.3], [[-3.0, -2.0], [-3.0, -2.0]], _bulk, seed=1,
                             lambda_cubed_of_T=lam) as sim:
        sim.run(n_cycles=150, sweeps_per_cycle=20, record_every=1, verbose=False)
    a = MBARSurfaceAnalysis(sim)
    T, bmu, h = 1.1, -2.2, 1e-4

    def mean_E(t):
        m = a._reweight(np.array([t]), np.array([bmu * t]), np.column_stack([a.U_all, a.N_all]))[0]
        return m[0] + 1.5 * t * m[1]
    fd = (mean_E(T + h) - mean_E(T - h)) / (2 * h)
    cv = a.heat_capacity([T], [bmu * T])[0, 0]
    assert cv == pytest.approx(fd, rel=1e-5)
    conf = a.heat_capacity([T], [bmu * T], kinetic=False)[0, 0]
    assert 0.0 <= conf < cv


def test_ideal_gas_heat_capacity_is_15_4_N():
    """U = 0: E = (3/2) N kT with Poisson N, so Var(E) / (kT)^2 = 15/4 <N>."""
    pytest.importorskip("pymbar")
    from regc import MBARSurfaceAnalysis

    def ideal(rng):
        return BulkSystem(3.0, LJCalculator(1.0, 0.0, 1.4))
    # enough GC attempts per sample that N decorrelates (else Var(N) is noisy)
    with ReplicaExchangeGCMC([1.0, 1.2], [[-1.0], [-1.0]], ideal, seed=3) as sim:
        sim.run(n_cycles=2500, sweeps_per_cycle=60, record_every=1, verbose=False)
    a = MBARSurfaceAnalysis(sim)
    n_mean = a.mean_N(1.0, -1.0)[0]
    assert n_mean == pytest.approx(27 * np.exp(-1.0), rel=0.05)
    assert a.heat_capacity([1.0], [-1.0])[0, 0] == pytest.approx(3.75 * n_mean, rel=0.1)
    assert a.heat_capacity([1.0], [-1.0], kinetic=False)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_heat_capacity_ridges_find_a_synthetic_boundary():
    from regc import heat_capacity_ridges
    T = np.linspace(1.0, 2.0, 21)
    y = np.linspace(-3.0, 3.0, 61)
    boundary = 0.5 * (T - 1.5)                          # y position of the ridge at each T
    cv = 1.0 + 5.0 * np.exp(-((y[None, :] - boundary[:, None]) / 0.2) ** 2)
    r = heat_capacity_ridges(T, y, cv)
    found = {i: y[j] for i, j in r["along_y"]}
    assert len(found) == len(T)
    assert all(abs(found[i] - boundary[i]) <= 0.1 + 1e-9 for i in found)
    valid = np.ones_like(cv, bool)
    valid[:5] = False
    assert all(i >= 5 for i, _ in heat_capacity_ridges(T, y, cv, valid=valid)["along_y"])


# -- configurations behind the diagram -------------------------------------------------
def test_configurations_are_stored_and_exported_by_phase(tmp_path):
    pytest.importorskip("pymbar")
    pytest.importorskip("scipy")
    from regc import MBARSurfaceAnalysis, export_phase_configurations
    from regc.trajectory import load_configurations, read_extxyz_frames

    cfg = str(tmp_path / "cfg")
    with ReplicaExchangeGCMC([0.5, 0.8], [-4.0, -1.0], _slab_factory, seed=4,
                             configurations_dir=cfg, configurations_every=2) as sim:
        sim.run_regc(n_steps=400, x0=0.7, diffusion="mc_moves", diffusion_steps=3,
                     record_every=5, verbose=False)
    key = (0, 1)
    frames = load_configurations(cfg, key)
    n = len(sim.samples[key]['N'])
    assert sorted(frames) == list(range(0, n, 2))
    for k, fr in frames.items():  # each frame is the sample's configuration
        assert fr["info"]["N"] == sim.samples[key]['N'][k]
        assert fr["info"]["U"] == pytest.approx(sim.samples[key]['U'][k])
        assert int(np.sum(fr["arrays"]["group"] == 2)) == fr["info"]["N"]

    path = str(tmp_path / "s.npz")
    save_samples(sim, path)
    a = MBARSurfaceAnalysis(load_samples(path, discard=0.2))
    T, mu = np.array([0.5, 0.8]), np.linspace(-4.0, -1.0, 7)
    out = str(tmp_path / "phases.extxyz")
    tiled = str(tmp_path / "tiled.extxyz")
    entries = export_phase_configurations(a, cfg, T, mu, out, tile_path=tiled)
    most = a.most_probable_N(T, mu)
    assert len(entries) >= len(np.unique(most))
    written = list(read_extxyz_frames(out))
    assert len(written) == len(entries)
    for e, fr in zip(entries, written):
        assert fr["info"]["phase_N"] == most[e["T_index"], e["y_index"]]
        # the exported structure has the phase's N (or the closest stored one)
        assert int(np.sum(fr["arrays"]["group"] == 2)) == e["N"]
        src = load_configurations(cfg, (e["source_T_index"], e["source_mu_index"]))
        assert e["source_sample"] in src and e["source_sample"] >= a.first_index
    grid = export_phase_configurations(a, cfg, T, mu[::3], str(tmp_path / "grid.extxyz"),
                                       mode="grid")
    assert len(grid) == 2 * 3
    ase = pytest.importorskip("ase")
    from ase.io import read
    assert len(read(out, index=":")) == len(entries)
    assert len(read(tiled)) == sum(len(fr["species"]) for fr in written)
    del ase


def test_resumed_run_overrides_stale_configuration_frames(tmp_path):
    from regc.trajectory import append_configuration, load_configurations
    path_dir = str(tmp_path)
    cfg_a = (np.zeros((1, 3)), np.array(["X"]), 0, 0)
    cfg_b = (np.ones((1, 3)), np.array(["X"]), 0, 0)
    from regc.trajectory import configuration_path
    p = configuration_path(path_dir, (0, 0))
    append_configuration(p, cfg_a, np.eye(3), (True, True, False), {"sample": 3, "N": 1})
    append_configuration(p, cfg_b, np.eye(3), (True, True, False), {"sample": 3, "N": 1})
    frames = load_configurations(path_dir, (0, 0))
    assert np.allclose(frames[3]["positions"], 1.0)
    assert os.path.isfile(p)


def test_heat_capacity_ridges_flag_peaks_on_the_sampled_ladder():
    """C_V peaks sitting on the sampled mu values (an overlap artefact) are flagged."""
    from regc import heat_capacity_ridges
    T = np.linspace(1.0, 2.0, 11)
    y = np.linspace(-3.0, 3.0, 61)
    ladder = np.array([-2.0, 0.0, 2.0])
    cv = 1.0 + sum(np.exp(-((y - m) / 0.15) ** 2) for m in ladder)[None, :] * np.ones((11, 1))
    with pytest.warns(UserWarning, match="overlap artefact"):
        r = heat_capacity_ridges(T, y, cv, ladder=ladder)
    assert r["on_ladder"].all()
    moving = 1.0 + np.exp(-((y[None, :] - 1.0 - 0.6 * (T[:, None] - 1.5)) / 0.15) ** 2)  # 0.7..1.3
    r2 = heat_capacity_ridges(T, y, moving, ladder=ladder)
    assert not r2["on_ladder"].any()
