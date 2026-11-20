"""
Noise-aware rho(u) convergence (`rho_u_noise_tv`, `ReplicaExchangeGCMC.rho_u_noise_check`,
``convergence_mode='noise'``) and the status reporting of the checkpointed driver.

* calibration: for a stationary series, independent or autocorrelated, the
  TV distance of the two halves is about its noise floor (ratio <~ 1); for a
  drifting series it is far above;
* the old fixed-block measure does not shrink with run length for a
  correlated stationary chain, the noise-aware ratio's floor does;
* an ideal-gas REGC run stops on the noise criterion, and not before every
  half holds enough independent samples;
* run_with_checkpoints publishes its checks in the engine history,
  checkpoint and status.json (the monitor), also after a resume.
"""

import json

import numpy as np
import pytest

from regc import LJSystem, ReplicaExchangeGCMC
from regc.calculators import Calculator
from regc.diagnostics import rho_u_noise_tv
from regc.driver import run_with_checkpoints


class ZeroCalculator(Calculator):
    """Ideal gas."""

    def total_energy(self, system) -> float:
        return 0.0


def _ideal_sim(seed=0):
    return ReplicaExchangeGCMC([1.0, 1.5], [[-1.0], [-1.5]],
                               lambda rng: LJSystem(3.0, ZeroCalculator()), seed=seed)


def _ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal()
    return x


# ---------------------------------------------------------------- the statistic
def test_noise_floor_calibration_independent_and_correlated():
    rng = np.random.default_rng(0)
    for phi in (0.0, 0.9):
        r = np.array([rho_u_noise_tv(_ar1(4000, phi, rng))["ratio"] for _ in range(150)])
        assert 0.85 < r.mean() < 1.2, (phi, r.mean())
        assert np.percentile(r, 95) < 2.2, (phi, np.percentile(r, 95))
    # correlation is accounted for: ignoring g would inflate the ratio by ~sqrt(g)
    out = rho_u_noise_tv(_ar1(4000, 0.9, rng))
    assert 10 < out["g"] < 30 and out["n_eff"] == pytest.approx(2000 / out["g"])
    # ignoring the correlation (shuffling destroys it) would give the same TV a ~3x smaller floor
    x = _ar1(4000, 0.9, rng)
    assert rho_u_noise_tv(x)["floor"] > 2.0 * rho_u_noise_tv(rng.permutation(x))["floor"]


def test_drift_is_detected():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000)
    x[2000:] += 1.0
    assert rho_u_noise_tv(x)["ratio"] > 5.0


def test_short_or_constant_series():
    assert rho_u_noise_tv(np.arange(5.0))["n_eff"] == 0.0
    assert rho_u_noise_tv(np.ones(100))["ratio"] == 0.0


def test_fixed_blocks_do_not_shrink_but_the_noise_floor_does():
    """The old measure: two consecutive 200-sample blocks; the new: growing halves."""
    rng = np.random.default_rng(2)
    x = _ar1(40000, 0.98, rng)                                       # g ~ 100
    tv_blocks, floors = [], []
    for n in (2000, 10000, 40000):
        a, b = x[n - 400:n - 200], x[n - 200:n]
        e = np.histogram_bin_edges(np.concatenate([a, b]), bins=10)
        pa, _ = np.histogram(a, e)
        pb, _ = np.histogram(b, e)
        tv_blocks.append(0.5 * np.abs(pa / pa.sum() - pb / pb.sum()).sum())
        floors.append(rho_u_noise_tv(x[:n])["floor"])
    assert floors[2] < floors[0] / 3                                  # ~ 1/sqrt(n)
    assert min(tv_blocks) > 0.15                                      # blocks stay noisy


# ---------------------------------------------------------------- engine
def test_run_regc_stops_on_the_noise_criterion():
    sim = _ideal_sim()
    converged = sim.run_regc(n_steps=200000, x0=0.9, diffusion='mc', diffusion_steps=0,
                             record_every=5, convergence_every=5000, convergence_mode="noise",
                             convergence_min_neff=30, verbose=False)
    assert converged
    last = sim.convergence_details[-1]
    assert last["converged"] and last["max_ratio"] < 3.0 and last["median_ratio"] < 1.3
    assert last["min_neff"] >= 30
    assert sim.convergence_history[-1] == last["max_ratio"]
    assert len(sim.convergence_history) < 40
    assert sim.convergence_settings["mode"] == "noise" and sim.convergence_settings["tol"] == 3.0


def test_min_neff_prevents_a_premature_stop():
    sim = _ideal_sim(seed=3)
    converged = sim.run_regc(n_steps=3000, x0=0.9, diffusion='mc', diffusion_steps=0,
                             record_every=5, convergence_every=500, convergence_mode="noise",
                             convergence_min_neff=1e6, verbose=False)
    assert not converged and len(sim.convergence_details) == 6
    assert all(not c["converged"] for c in sim.convergence_details)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="convergence_mode"):
        _ideal_sim().run_regc(n_steps=10, convergence_every=5, convergence_mode="foo",
                              verbose=False)


# ---------------------------------------------------------------- driver and status.json
@pytest.mark.parametrize("mode", ["blocks", "noise"])
def test_driver_publishes_checks_in_status_and_checkpoint(tmp_path, mode):
    out = tmp_path / mode
    sim = _ideal_sim()
    sim.save_status(str(out / "status.json"), interval_s=1e9)        # only forced writes
    kw = dict(n_steps=3000, chunk=500, x0=0.9, diffusion="mc", diffusion_steps=0,
              record_every=5, convergence_every=500, convergence_mode=mode,
              convergence_min_neff=1e6, progress=None, metadata={"t": 1})
    st = run_with_checkpoints(sim, str(out), **kw)
    assert st.finished and len(st.rho_u_changes) >= 4
    assert sim.convergence_history == pytest.approx(st.rho_u_changes)
    status = json.load(open(out / "status.json"))
    assert status["convergence_history"] == pytest.approx(st.rho_u_changes)
    assert status["convergence"]["settings"]["mode"] == mode
    assert len(status["convergence"]["checks"]) == len(st.rho_u_changes)
    # resume: the history comes back from the checkpoint and keeps growing
    sim2 = _ideal_sim()
    sim2.save_status(str(out / "status.json"), interval_s=1e9)
    st2 = run_with_checkpoints(sim2, str(out), **dict(kw, n_steps=4000))
    assert st2.resumed_from == 3000 and len(sim2.convergence_history) == len(st2.rho_u_changes)
    assert len(json.load(open(out / "status.json"))["convergence_history"]) == len(st2.rho_u_changes)


def test_runner_convergence_keys():
    import warnings
    from regc.runner import _convergence_settings
    assert _convergence_settings({})["convergence_mode"] == "noise"
    assert _convergence_settings({"convergence_mode": "blocks", "convergence_tol": 0.05}) == {
        "convergence_mode": "blocks", "convergence_tol": 0.05, "convergence_bins": None,
        "convergence_min_neff": 10.0, "convergence_median_tol": 1.3, "convergence_discard": 0.2}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _convergence_settings({"convergence_every": 1000, "convergence_tol": 0.0})
    assert any("noise floor" in str(x.message) for x in w)
    with pytest.raises(ValueError, match="convergence_mode"):
        _convergence_settings({"convergence_mode": "x"})


def test_median_catches_weak_drift_shared_by_the_grid():
    """A 0.5-sigma linear drift in every replica stays under the max tolerance but not the median."""
    rng = np.random.default_rng(4)
    drifting = np.array([rho_u_noise_tv(rng.standard_normal(2000) + np.linspace(0, 0.5, 2000))["ratio"]
                         for _ in range(64)])
    stationary = np.array([rho_u_noise_tv(rng.standard_normal(2000))["ratio"] for _ in range(64)])
    assert np.median(drifting) > 1.3 > np.median(stationary)
    assert stationary.max() < 3.0
