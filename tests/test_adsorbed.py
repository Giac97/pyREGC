"""
n_adsorbed: the geometric count, its recording during a run (adsorbed_cutoff),
its round trip through save/load_samples, and its reweighting in
MBARSurfaceAnalysis (observable='N_ads').
"""

import numpy as np
import pytest

from gcmc_re import LJCalculator, ReplicaExchangeGCMC, SlabSystem


def _slab(rng=None):
    # one layer of frozen atoms at z = 1 on a 4x4 grid; mobile particles above
    xy = np.array([[x, y] for x in range(4) for y in range(4)], dtype=float) * 2.0 + 0.5
    frozen = np.column_stack([xy, np.ones(len(xy))])
    return SlabSystem(8.0, 8.0, 10.0, LJCalculator(), frozen_positions=frozen,
                      frozen_species=np.array(["Fe"] * len(xy)))


def test_n_adsorbed_counts_each_particle_once_with_min_image():
    system = _slab()
    system.positions = np.array([[0.5, 0.5, 2.0],    # right above an atom
                                 [1.5, 1.5, 2.0],    # hollow: near 4 atoms, counted once
                                 [7.9, 0.5, 2.0],    # near (0.5, 0.5) only through x-periodicity
                                 [0.5, 0.5, 8.0]])   # gas
    assert system.n_adsorbed(1.8) == 3
    assert system.n_adsorbed(0.5) == 0


@pytest.mark.parametrize("mode", ["run", "run_regc"])
def test_adsorbed_cutoff_records_N_ads_alongside_N(mode):
    with ReplicaExchangeGCMC([1.0, 1.5], [-2.0, 0.0], _slab, seed=3,
                             adsorbed_cutoff=2.0) as sim:
        if mode == "run":
            sim.run(n_cycles=20, sweeps_per_cycle=5, record_every=2, verbose=False)
        else:
            sim.run_regc(n_steps=200, x0=0.9, diffusion_steps=2, record_every=5,
                         verbose=False)
    for v in sim.samples.values():
        assert len(v["N_ads"]) == len(v["N"]) > 0
        assert all(0 <= a <= n for a, n in zip(v["N_ads"], v["N"]))


def test_N_ads_survives_save_load_and_reweights(tmp_path):
    pytest.importorskip("pymbar", reason="pip install pymbar to run this test")
    from gcmc_re import MBARSurfaceAnalysis
    from gcmc_re.io import load_samples, save_samples

    with ReplicaExchangeGCMC([1.0, 1.5], [-2.0, 0.0], _slab, seed=3,
                             adsorbed_cutoff=2.0) as sim:
        sim.run(n_cycles=40, sweeps_per_cycle=5, record_every=2, verbose=False)
    path = str(tmp_path / "samples.npz")
    save_samples(sim, path)
    loaded = load_samples(path, discard=0.25)
    assert all("N_ads" in v for v in loaded.samples.values())

    analysis = MBARSurfaceAnalysis(loaded)
    T, mu = np.array([1.0, 1.5]), np.array([-2.0, 0.0])
    theta = analysis.phase_diagram(T, mu)
    theta_ads = analysis.phase_diagram(T, mu, observable="N_ads")
    assert theta_ads.shape == theta.shape
    assert np.all(theta_ads <= theta + 1e-12)
    assert analysis.most_probable_N(T, mu, observable="N_ads").shape == theta.shape


def test_N_ads_missing_is_a_clear_error():
    pytest.importorskip("pymbar", reason="pip install pymbar to run this test")
    from gcmc_re import MBARSurfaceAnalysis

    with ReplicaExchangeGCMC([1.0], [-2.0, 0.0], _slab, seed=3) as sim:
        sim.run(n_cycles=10, sweeps_per_cycle=5, record_every=2, verbose=False)
    with pytest.raises(ValueError, match="adsorbed_cutoff"):
        MBARSurfaceAnalysis(sim).phase_diagram([1.0], [-2.0], observable="N_ads")
