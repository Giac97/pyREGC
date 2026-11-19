"""
Semi-grand-canonical (identity-swapping) substrate: the identity move, the
replica-exchange term and the MBAR reduced potential, checked against an
exact result.

Swappable atoms with site energies only (no interactions) are independent
two-state systems: with site energies h_A, h_B and the field dmu = mu_B -
mu_A, each atom is B with probability

    x_B(T) = 1 / (1 + exp((h_B - h_A - dmu) / T)),

at every grid point and, after MBAR reweighting, at every T in between.
Replica exchange along T only gives that if the swap rule carries the
-(beta_a - beta_b) dmu (N_B,a - N_B,b) term (docs/REGC_algorithm.md, Sec. 9).
"""

import numpy as np
import pytest

from regc import ReplicaExchangeGCMC, SlabSystem
from regc.analysis import MBARSurfaceAnalysis
from regc.calculators import Calculator
from regc.observables import SGC_COUNTS_NAME, LayerComposition, SGCCounts, check_observables

# site energies (X: the exchanged gas, kept out) and the field mu_B - mu_A:
# h_B - h_A - dmu = 1, but dmu alone is large, so a swap rule without the
# field term would visibly bias the composition
H = {"A": 0.0, "B": 10.3, "X": 0.0}
DMU = 9.3
LADDER = (0.8, 1.0, 1.25)
N_SUB = 20


class SiteCalculator(Calculator):
    """U = sum of per-atom site energies: no interactions."""

    supports_incremental = False

    def total_energy(self, system) -> float:
        _, species = system.full_configuration()
        return float(sum(H[str(s)] for s in species))


def _exact_xB(T):
    return 1.0 / (1.0 + np.exp((H["B"] - H["A"] - DMU) / np.asarray(T)))


def _slab(rng=None, n_frozen=0):
    xy = np.random.default_rng(1).random((N_SUB, 3)) * [8.0, 8.0, 1.0]
    sysm = SlabSystem(8.0, 8.0, 3.0, SiteCalculator(), exchange_species="X",
                      substrate_positions=xy, substrate_species=np.array(["A"] * N_SUB),
                      frozen_positions=np.zeros((n_frozen, 3)) - 1.0,
                      frozen_species=np.array(["B"] * n_frozen))
    sysm.set_sgc_species(["A", "B"])
    return sysm


def _sim(dmu=(0.0, DMU), temperatures=LADDER, seed=3, attempts=1):
    # mu = -50: no gas particle is ever inserted
    return ReplicaExchangeGCMC(list(temperatures), [-50.0], _slab, seed=seed,
                               sgc_dmu=list(dmu), sgc_attempts=attempts)


# ---------------------------------------------------------------- bookkeeping
def test_system_sgc_bookkeeping():
    sysm = _slab(n_frozen=3)
    assert sysm.sgc_species == ("A", "B")
    assert sysm.sgc_counts().tolist() == [N_SUB, 0]      # frozen B atoms do not count
    assert len(sysm.sgc_candidates()) == N_SUB
    du = sysm.generic_identity_dU(4, "B")
    assert du == pytest.approx(H["B"] - H["A"])
    assert sysm.substrate_species[4] == "A"              # the trial left no trace
    sysm.apply_identity(4, "B")
    assert sysm.sgc_counts().tolist() == [N_SUB - 1, 1]
    with pytest.raises(ValueError, match="distinct"):
        sysm.set_sgc_species(["A"])


def test_engine_validation():
    with pytest.raises(ValueError, match="pass sgc_dmu"):
        ReplicaExchangeGCMC([1.0], [-50.0], _slab)
    with pytest.raises(ValueError, match="2 values"):
        ReplicaExchangeGCMC([1.0], [-50.0], _slab, sgc_dmu=[0.1])
    sim = _sim(dmu=(5.0, 5.0 + DMU))                     # only the difference matters
    assert np.allclose(sim.sgc_dmu, [0.0, DMU])
    assert SGC_COUNTS_NAME in sim.observables_meta
    assert sim.observables_meta[SGC_COUNTS_NAME]["dmu"] == pytest.approx([0.0, DMU])
    with pytest.raises(ValueError, match="sweep scheme"):
        sim.run(n_cycles=1, verbose=False)
    with pytest.raises(ValueError, match="reserved"):
        check_observables([LayerComposition(SGC_COUNTS_NAME, ["A"], [0.0])])
    check_observables([SGCCounts(["A", "B"], [0.0, 1.0])])


def test_swap_log_acceptance_has_the_field_term():
    sim = _sim(temperatures=(0.5, 2.0))
    ra, rb = sim.grid[(0, 0)], sim.grid[(1, 0)]
    for i in range(10):                    # replica b: 10 B atoms, a: none
        rb.system.apply_identity(i, "B")
    rb.energy = rb.system.total_energy()
    ba, bb = 1 / 0.5, 1 / 2.0
    expected = (ba - bb) * (ra.energy - rb.energy) - (ba - bb) * DMU * (0 - 10)
    assert expected == pytest.approx(-15.0)             # (beta_a - beta_b)(h - dmu)(N_a - N_b)
    assert sim.swap_log_acceptance(ra, rb) == pytest.approx(expected)
    assert sim.swap_log_acceptance(rb, ra) == pytest.approx(expected)   # symmetric in a, b
    # the inverse move (swapping the configurations back) has the opposite log ratio
    ra.system, rb.system = rb.system, ra.system
    ra.energy, rb.energy = rb.energy, ra.energy
    assert sim.swap_log_acceptance(ra, rb) == pytest.approx(-expected)


# ---------------------------------------------------------------- exact statistics
@pytest.fixture(scope="module")
def sampled():
    # swap-dominated: one identity attempt per step and replica exchange in 90 % of
    # the steps, so a wrong swap rule would bias the composition visibly
    sim = _sim()
    sim.run_regc(n_steps=30000, x0=0.1, diffusion="mc", diffusion_steps=1, record_every=5,
                 burn_in_steps=500, verbose=False)
    return sim


def test_identity_swaps_and_exchange_are_exact(sampled):
    sim = sampled
    for it, T in enumerate(sim.temperatures):
        counts = np.asarray(sim.samples[(it, 0)]["obs:" + SGC_COUNTS_NAME])
        assert np.all(counts.sum(axis=1) == N_SUB)
        xB = counts[:, 1].mean() / N_SUB
        # binomial sampling error with a generous correlation allowance
        err = np.sqrt(_exact_xB(T) * (1 - _exact_xB(T)) / N_SUB / (len(counts) / 10))
        assert xB == pytest.approx(_exact_xB(T), abs=4 * err), (T, xB, _exact_xB(T))
    acc = sim.grid[(1, 0)].acceptance_rates()
    assert 0.0 < acc["identity"] < 1.0
    assert sim.swap_acceptance()["T"] > 0.1
    for r in sim.grid.values():           # running energy == recomputed energy
        assert r.energy == pytest.approx(r.system.total_energy())


def test_mbar_reweighting_uses_the_field(sampled):
    analysis = MBARSurfaceAnalysis(sampled)
    assert analysis.sgc is not None and np.allclose(analysis.sgc["dmu"], [0.0, DMU])
    for T in (0.9, 1.1):
        xB = analysis.observable_mean(SGC_COUNTS_NAME, T, -50.0)[1] / N_SUB
        assert xB == pytest.approx(_exact_xB(T), abs=0.02), (T, xB, _exact_xB(T))
    # without the field in the reduced potential the reweighted answer is wrong
    analysis_wrong = MBARSurfaceAnalysis(sampled)
    analysis_wrong.U_all = analysis_wrong.U_phys_all
    k = analysis_wrong._log_den  # keep the solved denominator: only the target u changes
    xB_wrong = (np.exp(-analysis_wrong._u_of_state(0.9, -50.0) - k)
                @ analysis_wrong.obs_all[SGC_COUNTS_NAME][:, 1])
    xB_wrong /= np.exp(-analysis_wrong._u_of_state(0.9, -50.0) - k).sum() * N_SUB
    assert abs(xB_wrong - _exact_xB(0.9)) > 0.02


# ---------------------------------------------------------------- observables, config
def test_layer_composition_counts_slab_atoms_only():
    sysm = SlabSystem(8.0, 8.0, 6.0, SiteCalculator(), exchange_species="X",
                      positions=[[1.0, 1.0, 4.5]],
                      frozen_positions=[[0.0, 0.0, -2.0], [2.0, 0.0, -2.0]],
                      frozen_species=np.array(["A", "B"]),
                      substrate_positions=[[0.0, 0.0, 0.1], [2.0, 0.0, -0.2], [1.0, 1.0, 2.3]],
                      substrate_species=np.array(["B", "B", "A"]))
    obs = LayerComposition("layers", ["A", "B"], layer_z=[-2.0, 0.0, 2.2])
    assert obs.labels() == ["L0:A", "L0:B", "L1:A", "L1:B", "L2:A", "L2:B"]
    assert obs(sysm).tolist() == [1, 1, 0, 2, 1, 0]   # the X gas atom at z = 4.5 is not counted


def test_alloy_substitution_is_exact():
    from ase.build import fcc111
    from regc.runner import _alloy_fractions, _substitute
    atoms = fcc111("Pd", size=(4, 4, 4), a=3.9)
    _substitute(atoms, _alloy_fractions({"alloy": {"Pt": 0.5}}), seed=0)
    sym = np.array(atoms.get_chemical_symbols())
    assert (sym == "Pt").sum() == 32 and (sym == "Pd").sum() == 32
    with pytest.raises(ValueError, match="fractions"):
        _alloy_fractions({"alloy": {"Pt": 1.2}})


def test_checkpoint_round_trip_is_bit_identical(tmp_path):
    def run(sim, n):
        sim.run_regc(n_steps=n, x0=0.5, diffusion="mc", diffusion_steps=1, record_every=5,
                     verbose=False)

    straight = _sim(seed=11)
    run(straight, 400)
    first = _sim(seed=11)
    run(first, 200)
    first.save_checkpoint(str(tmp_path / "ck.pkl"))
    resumed = _sim(seed=11)
    resumed.load_checkpoint(str(tmp_path / "ck.pkl"))
    run(resumed, 200)
    key = "obs:" + SGC_COUNTS_NAME
    for k in straight.samples:
        assert np.array_equal(straight.samples[k][key], resumed.samples[k][key])
        assert np.allclose(straight.samples[k]["U"], resumed.samples[k]["U"])
    # a different field cannot continue the samples
    other = _sim(dmu=(0.0, DMU + 0.1), seed=11)
    with pytest.raises(ValueError, match="observables"):
        other.load_checkpoint(str(tmp_path / "ck.pkl"))


def test_status_and_monitor_report_the_composition(tmp_path):
    from regc.io import save_samples
    from regc.monitor import Run
    sim = _sim(seed=5)
    sim.observables.append(LayerComposition("layers", ["A", "B"], layer_z=[0.0, 0.5]))
    for k in sim.samples:
        sim.samples[k]["obs:layers"] = []
    sim.save_status(str(tmp_path / "status.json"))
    sim.run_regc(n_steps=300, x0=0.5, diffusion="mc", diffusion_steps=1, record_every=5,
                 verbose=False)
    st = sim.status(300, 300)
    comp = st["composition"]
    assert comp["species"] == "B" and comp["sgc_species"] == ["A", "B"]
    xB = np.array(comp["grids"]["mean_swappable"], dtype=float)
    assert xB.shape == (3, 1) and np.all((xB > 0) & (xB < 1))
    acc = np.array(comp["grids"]["identity_acceptance"], dtype=float)
    assert np.all((acc > 0) & (acc < 1))
    cur = sim.grid[(0, 0)].system.sgc_counts()
    assert comp["grids"]["current_swappable"][0][0] == pytest.approx(cur[1] / cur.sum())
    assert len(comp["traces"]["swappable"]) == 3 and comp["traces"]["top"] is not None
    # the monitor: live from status.json, and from samples.npz without a heartbeat
    save_samples(sim, str(tmp_path / "samples.npz"))
    d = Run(str(tmp_path)).collect()
    assert d["composition"]["source"].startswith("status.json")
    (tmp_path / "status.json").unlink()
    d = Run(str(tmp_path)).collect()
    assert d["composition"]["source"].startswith("samples.npz")
    assert np.allclose(np.array(d["composition"]["grids"]["mean_swappable"], dtype=float), xB)


def test_no_composition_without_sgc(tmp_path):
    from regc.monitor import Run
    assert Run(str(tmp_path)).collect()["composition"] is None


def test_bulk_composition_term_acts_in_the_identity_moves_only():
    # reservoir with x_B / x_A = 1/3: the field at T is dmu + T ln(1/3), so
    # x_B(T) = 1 / (1 + exp((h_B - h_A - dmu) / T - ln(1/3)))
    c = np.log(1 / 3)
    sim = ReplicaExchangeGCMC(list(LADDER), [-50.0], _slab, seed=4, sgc_dmu=[0.0, DMU],
                              sgc_attempts=1, sgc_log_ratio=[0.0, c])
    for it, T in enumerate(LADDER):
        assert sim.grid[(it, 0)].state.sgc_dmu[1] == pytest.approx(DMU + T * c)
    meta = sim.observables_meta[SGC_COUNTS_NAME]
    assert meta["dmu"] == pytest.approx([0.0, DMU]) and meta["log_ratio"][1] == pytest.approx(c)
    sim.run_regc(n_steps=30000, x0=0.1, diffusion="mc", diffusion_steps=1, record_every=5,
                 burn_in_steps=500, verbose=False)

    def exact(T):
        return 1.0 / (1.0 + np.exp((H["B"] - H["A"] - DMU) / np.asarray(T) - c))
    for it, T in enumerate(LADDER):
        counts = np.asarray(sim.samples[(it, 0)]["obs:" + SGC_COUNTS_NAME])
        xB = counts[:, 1].mean() / N_SUB
        err = np.sqrt(exact(T) * (1 - exact(T)) / N_SUB / (len(counts) / 10))
        assert xB == pytest.approx(exact(T), abs=4 * err), (T, xB, exact(T))
    analysis = MBARSurfaceAnalysis(sim)   # the T ln(x) term is state-independent in u: cancels
    for T in (0.9, 1.1):
        xB = analysis.observable_mean(SGC_COUNTS_NAME, T, -50.0)[1] / N_SUB
        assert xB == pytest.approx(exact(T), abs=0.02), (T, xB, exact(T))


def test_per_layer_alloy_is_exact_in_every_layer():
    from ase.build import fcc111
    from regc.runner import _substitute
    atoms = fcc111("Pd", size=(4, 4, 4), a=3.9)
    _substitute(atoms, {"Pt": 0.25}, seed=2, per_layer=True)
    sym = np.array(atoms.get_chemical_symbols())
    z = np.round(atoms.positions[:, 2], 3)
    for v in np.unique(z):
        assert (sym[z == v] == "Pt").sum() == 4
    with pytest.raises(ValueError, match="whole number"):
        _substitute(fcc111("Pd", size=(4, 4, 4), a=3.9), {"Pt": 0.3}, seed=0, per_layer=True)


# ---------------------------------------------------------------- canonical pair swaps
class TwoSiteClassCalculator(Calculator):
    """U = EPS for every B atom on an 'outer' site (the first N_OUT substrate atoms)."""

    supports_incremental = False
    EPS = 0.7

    def total_energy(self, system) -> float:
        return float(self.EPS * np.count_nonzero(np.asarray(system.substrate_species[:N_OUT]) == "B"))


N_OUT, N_IN, N_B = 8, 12, 5


def _canonical_slab(rng=None):
    pos = np.random.default_rng(2).random((N_OUT + N_IN, 3)) * [8.0, 8.0, 1.0]
    species = np.array(["B"] * N_B + ["A"] * (N_OUT + N_IN - N_B))   # B start outside
    sysm = SlabSystem(8.0, 8.0, 3.0, TwoSiteClassCalculator(), exchange_species="X",
                      substrate_positions=pos, substrate_species=species)
    sysm.set_sgc_species(["A", "B"], canonical=True)
    return sysm


def _exact_outer_B(T):
    from math import comb
    k = np.arange(0, min(N_B, N_OUT) + 1)
    w = np.array([comb(N_OUT, int(j)) * comb(N_IN, N_B - int(j)) for j in k], dtype=float)
    w = w * np.exp(-TwoSiteClassCalculator.EPS * k / T)
    return float((k * w).sum() / w.sum())


def test_canonical_pair_swaps_are_exact():
    sysm = _canonical_slab()
    assert sysm.sgc_canonical
    du = sysm.generic_pair_swap_dU(0, N_OUT)              # outer B <-> inner A
    assert du == pytest.approx(-TwoSiteClassCalculator.EPS)
    assert sysm.substrate_species[0] == "B"               # the trial left no trace
    with pytest.raises(ValueError, match="zero"):
        ReplicaExchangeGCMC([1.0], [-50.0], _canonical_slab, sgc_dmu=[0.0, 1.0])
    temps = (0.3, 0.45, 0.7)
    sim = ReplicaExchangeGCMC(list(temps), [-50.0], _canonical_slab, seed=5, sgc_attempts=1)
    assert np.allclose(sim.sgc_dmu, 0.0) and sim.sgc_canonical
    sim.run_regc(n_steps=20000, x0=0.5, diffusion="mc", diffusion_steps=1, record_every=5,
                 burn_in_steps=500, verbose=False)
    for it, T in enumerate(temps):
        r = sim.grid[(it, 0)]
        counts = np.asarray(sim.samples[(it, 0)]["obs:" + SGC_COUNTS_NAME])
        assert np.all(counts == [N_OUT + N_IN - N_B, N_B])          # composition fixed
        k = np.asarray(sim.samples[(it, 0)]["U"]) / TwoSiteClassCalculator.EPS
        assert k.mean() == pytest.approx(_exact_outer_B(T), abs=0.08), (T, k.mean())
        assert r.energy == pytest.approx(r.system.total_energy())
    assert 0.0 < sim.grid[(1, 0)].acceptance_rates()["identity"] < 1.0


def test_runner_canonical_section_validation():
    from regc.runner import _sgc_settings, validate_config
    base = {"structure": {"build": {}}, "gas": {"species": "O"},
            "replica_exchange": {"temperatures_K": [300], "mu_eV": [-1.0], "scheme": "regc"}}
    with pytest.raises(ValueError, match="drop dmu_eV"):
        validate_config({**base, "sgc": {"species": ["Au", "Rh"], "ensemble": "canonical",
                                         "dmu_eV": {"Rh": 0.1}}})
    with pytest.raises(ValueError, match="ensemble"):
        validate_config({**base, "sgc": {"species": ["Au", "Rh"], "ensemble": "grand"}})

    class G:
        substrate_species = np.array(["Au"] * 6 + ["Rh"] * 2)
    s = _sgc_settings({"sgc": {"species": ["Au", "Rh"], "ensemble": "canonical"}}, G())
    assert s["canonical"] and s["dmu"] == [0.0, 0.0] and s["counts"] == [6, 2]


def test_shell_composition_splits_core_and_surface():
    from ase.cluster import Icosahedron
    from regc.observables import ShellComposition, composition_fractions, observable_from_config
    atoms = Icosahedron("Au", noshells=3)                  # 55 atoms: 13 core, 42 surface
    pos = atoms.positions - atoms.positions.mean(axis=0)
    r = np.linalg.norm(pos, axis=1)
    species = np.where(r < 1.0, "Rh", "Au")                # Rh at the centre
    species[np.argmax(r)] = "Rh"                           # and one Rh vertex
    sysm = SlabSystem(30.0, 30.0, 30.0, SiteCalculator(), exchange_species="X",
                      substrate_positions=pos + 15.0, substrate_species=species,
                      positions=np.array([[15.0, 15.0, 15.0 + r.max() + 2.0]]),
                      species=np.array(["X"]))
    obs = observable_from_config({"type": "shell_composition", "name": "shells",
                                  "species": ["Au", "Rh"]})
    assert isinstance(obs, ShellComposition)
    assert obs.labels() == ["core:Au", "core:Rh", "surface:Au", "surface:Rh"]
    counts = obs(sysm)                                     # the gas atom is not counted
    assert counts.tolist() == [12.0, 1.0, 41.0, 1.0]
    meta = {"shells": obs.metadata(), "sgc_counts": SGCCounts(["Au", "Rh"], [0.0, 0.0]).metadata()}
    f = composition_fractions(meta, {"shells": counts[None], "sgc_counts": np.array([[53.0, 2.0]])},
                              species="Rh")
    assert f["top"][0] == pytest.approx(1 / 42)
