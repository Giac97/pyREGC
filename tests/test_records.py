"""
Per-sample records (`regc.records`): the HDF5 file written during a run,
re-evaluation with another potential, and MBAR reweighting to it
(`MBARSurfaceAnalysis.with_potential`).

Exact checks: re-evaluating with the sampling potential reproduces every
recorded energy; a potential B = A + c N is A at mu - c, so reweighting to B
must equal the A analysis at shifted mu to round-off; B = A + c0 shifts the
grand potential by exactly c0.
"""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pymbar")

from regc import LJSystem, ReplicaExchangeGCMC  # noqa: E402
from regc.analysis import MBARSurfaceAnalysis  # noqa: E402
from regc.calculators import LJCalculator  # noqa: E402
from regc.records import Records, attach_energies, reevaluate  # noqa: E402

L = 5.0
TEMPS = [1.0, 1.2]
MUS = [[-3.4, -3.0], [-3.6, -3.2]]


class ShiftedLJ(LJCalculator):
    """LJ plus c0 + c1 * N: B = A + c1 N is A at mu - c1."""

    def __init__(self, c0=0.0, c1=0.0):
        super().__init__(sigma=1.0, epsilon=1.0, r_cut=2.5)
        self.c0, self.c1 = c0, c1

    def total_energy(self, system) -> float:
        return super().total_energy(system) + self.c0 + self.c1 * system.n


def make_A(rng):
    return LJSystem(L, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def make_same(rng):
    return LJSystem(L, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def make_shift_N(rng):
    return LJSystem(L, ShiftedLJ(c1=0.15))


def make_shift_const(rng):
    return LJSystem(L, ShiftedLJ(c0=0.7))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rec") / "records.h5")
    with ReplicaExchangeGCMC(TEMPS, MUS, make_A, seed=3, records_path=path,
                             records_potential="LJ") as sim:
        sim.run_regc(n_steps=3000, x0=0.8, diffusion="mc", diffusion_steps=2, record_every=10,
                     burn_in_steps=200, verbose=False)
    return sim, path


def test_records_match_samples(run):
    sim, path = run
    rec = Records(path)
    try:
        assert rec.potentials == ["LJ"]
        assert len(rec) == sum(len(v["U"]) for v in sim.samples.values())
        e = rec.energies("LJ")
        for key, v in sim.samples.items():
            for s, (U, N) in enumerate(zip(v["U"], v["N"])):
                assert e[(key, s)] == U
                fr = rec.configuration(key, s)
                assert fr["info"]["N"] == N
                assert (fr["arrays"]["group"] == 2).sum() == N
    finally:
        rec.close()


def test_reevaluate_with_same_potential_reproduces_energies(run, tmp_path):
    sim, path = run
    import shutil
    p = str(tmp_path / "r.h5")
    shutil.copy(path, p)
    n = reevaluate(p, make_same, "same", chunk=50)
    assert n == sum(len(v["U"]) for v in sim.samples.values())
    assert reevaluate(p, make_same, "same") == 0  # resumable: nothing left
    rec = Records(p)
    try:
        a, b = rec.energies("LJ"), rec.energies("same")
        assert max(abs(a[k] - b[k]) for k in a) < 1e-9
    finally:
        rec.close()
    with pytest.raises(ValueError, match="sampling potential"):
        reevaluate(p, make_same, "LJ")


def test_reevaluate_in_parallel_matches_serial(run, tmp_path):
    import shutil
    _, path = run
    p1, p2 = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
    shutil.copy(path, p1)
    shutil.copy(path, p2)
    reevaluate(p1, make_shift_N, "B", chunk=40)
    reevaluate(p2, make_shift_N, "B", chunk=40, n_workers=2)
    r1, r2 = Records(p1), Records(p2)
    try:
        assert r1.energies("B") == r2.energies("B")
    finally:
        r1.close()
        r2.close()


@pytest.fixture(scope="module")
def reweighted(run, tmp_path_factory):
    import shutil
    sim, path = run
    p = str(tmp_path_factory.mktemp("rw") / "records.h5")
    shutil.copy(path, p)
    reevaluate(p, make_shift_N, "shiftN")
    reevaluate(p, make_shift_const, "shift0")
    attach_energies(sim, p)
    return MBARSurfaceAnalysis(sim)


def test_potential_linear_in_N_equals_shifted_mu(reweighted):
    A = reweighted
    B = A.with_potential("shiftN")
    T_grid = np.array([1.0, 1.1, 1.2])
    mu_grid = np.linspace(-3.4, -3.0, 5)
    nB = B.phase_diagram(T_grid, mu_grid)
    nA = A.phase_diagram(T_grid, mu_grid - 0.15)
    assert np.allclose(nB, nA, rtol=1e-10, atol=1e-12)
    # same distribution, but B's energy is U_A + c N: C_V = Var(U_A + c N) / T^2
    T_mesh, mu_mesh = np.meshgrid(T_grid, mu_grid - 0.15, indexing="ij")
    E = A.U_all + 0.15 * A.N_all - A.U_all.mean()
    m = A._reweight(T_mesh, mu_mesh, np.column_stack([E, E ** 2]))
    assert np.allclose(B.heat_capacity(T_grid, mu_grid, kinetic=False),
                       (m[..., 1] - m[..., 0] ** 2) / T_mesh ** 2, rtol=1e-8)
    assert np.allclose(B.effective_sample_size(T_grid, mu_grid),
                       A.effective_sample_size(T_grid, mu_grid - 0.15), rtol=1e-8)
    assert B.mean_N(1.1, -3.2)[0] == pytest.approx(A.mean_N(1.1, -3.35)[0], rel=1e-10)


def test_constant_offset_shifts_grand_potential_exactly(reweighted):
    A = reweighted
    B = A.with_potential("shift0")
    d, ess = B.grand_potential_shift(1.1, -3.2)
    assert d == pytest.approx(0.7, abs=1e-10)
    assert ess == pytest.approx(float(A.effective_sample_size([1.1], [-3.2])[0, 0]), rel=1e-10)
    # same distribution: maps unchanged
    assert np.allclose(B.phase_diagram([1.05], [-3.3]), A.phase_diagram([1.05], [-3.3]))
    with pytest.raises(ValueError, match="with_potential"):
        A.grand_potential_shift(1.1, -3.2)
    with pytest.raises(ValueError, match="already reweighted"):
        B.with_potential("shiftN")


def test_unknown_potential_lists_available(reweighted):
    with pytest.raises(ValueError, match="available"):
        reweighted.with_potential("nope")


def test_subset_reevaluation_gets_zero_weight(run, tmp_path):
    import copy
    import shutil
    sim, path = run
    p = str(tmp_path / "s.h5")
    shutil.copy(path, p)
    reevaluate(p, make_shift_N, "B", subset=lambda key, s: s % 2 == 0)
    samples = copy.deepcopy(sim.samples)
    try:
        attach_energies(sim, p)
        A = MBARSurfaceAnalysis(sim)
        B = A.with_potential("B")
        assert B._evaluated.sum() == sum((len(v["U"]) + 1) // 2 for v in sim.samples.values())
        ess_B = B.effective_sample_size([1.1], [-3.2])[0, 0]
        ess_A = A.effective_sample_size([1.1], [-3.35])[0, 0]
        assert 0.3 * ess_A < ess_B < 0.7 * ess_A
        # every other sample: close to the full answer
        assert B.mean_N(1.1, -3.2)[0] == pytest.approx(A.mean_N(1.1, -3.35)[0], rel=0.1)
        # a bootstrap resample keeps the reweighting
        idx = {k: np.arange(len(v["U"])) for k, v in sim.samples.items()}
        R = B.resampled(idx)
        assert R.potential == "B"
        assert R.mean_N(1.1, -3.2)[0] == pytest.approx(B.mean_N(1.1, -3.2)[0], rel=1e-8)
    finally:
        sim.samples = samples


def test_resume_keeps_last_record_of_each_sample(tmp_path):
    """Samples recorded after a checkpoint are written again when the run
    resumes from it; the reader keeps the last record of each."""
    path, ckpt = str(tmp_path / "r.h5"), str(tmp_path / "c.pkl")
    with ReplicaExchangeGCMC(TEMPS, MUS, make_A, seed=1, records_path=path) as sim:
        sim.run_regc(n_steps=400, x0=0.8, diffusion="mc", diffusion_steps=1, record_every=10,
                     verbose=False)
        sim.save_checkpoint(ckpt)
        sim.run_regc(n_steps=300, x0=0.8, diffusion="mc", diffusion_steps=1, record_every=10,
                     verbose=False)  # "lost" in the crash
    with ReplicaExchangeGCMC(TEMPS, MUS, make_A, seed=1, records_path=path) as sim2:
        sim2.load_checkpoint(ckpt)
        sim2.run_regc(n_steps=300, x0=0.8, diffusion="mc", diffusion_steps=1, record_every=10,
                      verbose=False)
    assert sim2.samples == sim.samples  # bit-identical continuation
    rec = Records(path)
    try:
        n = sum(len(v["U"]) for v in sim2.samples.values())
        assert rec.f["samples/sample"].shape[0] > n  # duplicates on disk
        assert len(rec) == n
        e = rec.energies("sampling")
        assert all(e[(k, s)] == U for k, v in sim2.samples.items() for s, U in enumerate(v["U"]))
    finally:
        rec.close()


def test_records_every_stores_some_configurations(tmp_path):
    path = str(tmp_path / "r.h5")
    with ReplicaExchangeGCMC([1.0], [-3.2], make_A, seed=2, records_path=path,
                             records_every=3) as sim:
        sim.run(n_cycles=20, sweeps_per_cycle=5, record_every=1, verbose=False)
    rec = Records(path)
    try:
        assert len(rec) == 20
        assert rec.stored((0, 0)) == {0, 3, 6, 9, 12, 15, 18}
        assert rec.configuration((0, 0), 1) is None
    finally:
        rec.close()
    with pytest.raises(ValueError, match="records_every"):
        ReplicaExchangeGCMC([1.0], [-3.2], make_A, records_path=path, records_every=0)
