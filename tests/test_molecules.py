"""
Molecular insertion/deletion (`regc.molecules`) and polyatomic gas reservoirs.

The moves are checked against exact results:

* the proposal densities integrate to 1 (sampled uniformly over the shell);
* ideal gas (zero energy): the number of CO, O2 and CH4 molecules follows
  P(n) ~ (prod_e z_e^{nu_e} V^{n_atoms})^n / prod_e N_e!, which tests the
  unit counting, the anchor and partner-ordering sums and the chirality
  mixture; with atomic moves as well, N_C and N_O are independent Poisson;
* an interacting gas: atomic and atomic + molecular exchange sample the
  same distribution, and the running energy stays exact;
* `MolecularGas` reduces to `DiatomicReservoir` for O2, and a feed's
  elemental chemical potentials reproduce every gas's.
"""

import json
import math

import numpy as np
import pytest

from regc import ClusterSystem, PairLJCalculator, ReplicaExchangeGCMC, SlabSystem, ThermoState
from regc.calculators import Calculator
from regc.molecules import MoleculeMove, uniform_direction
from regc.observables import MOLECULE_COUNTS_NAME, MoleculeCounts
from regc.replica import Replica
from regc.reservoir import DiatomicReservoir, MolecularGas, MolecularGasFeed

CH4_TEMPLATE = [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]


class ZeroCalculator(Calculator):
    """Ideal gas: zero energy for every configuration."""

    supports_incremental = True

    def total_energy(self, system) -> float:
        return 0.0

    def pair_energy_of(self, system, pos, exclude=None, species=None) -> float:
        return 0.0


def _co(**kw):
    return MoleculeMove("CO", ["C", "O"], [[0, 0, 0], [0, 0, 1.13]], **kw)


def _counts(species, mols, atomic, mu, n_moves, L=6.0, seed=1, max_disp=3.0):
    """Species counts every 100 moves of an ideal-gas slab with molecular moves."""
    sysm = SlabSystem(L, L, L, ZeroCalculator(), exchange_species=list(species))
    sysm.set_molecules(mols, atomic_exchange=atomic)
    mu = np.asarray(mu, dtype=float)
    state = ThermoState(1.0, mu if len(species) > 1 else float(mu[0]), (0, 0))
    rep = Replica(state, sysm, np.random.default_rng(seed), max_disp=max_disp)
    out = []
    for _ in range(n_moves // 100):
        rep.sweep(100)
        out.append(sysm.species_counts().copy())
    return np.array(out), rep


def _exact_mean(logw):
    w = np.exp(np.array(logw) - max(logw))
    return float((np.arange(len(w)) * w).sum() / w.sum())


# ---------------------------------------------------------------- the move itself
def test_template_roles_anchor_and_validation():
    co = _co()
    assert co.anchor == "C" and co.partners == ("O",) and co.composition == {"C": 1, "O": 1}
    ch4 = MoleculeMove("CH4", ["C"] + ["H"] * 4, CH4_TEMPLATE)
    assert ch4.anchor == "C" and ch4.partners == ("H",) * 4 and ch4._chiral == 2
    o2 = MoleculeMove("O2", ["O", "O"], [[0, 0, 0], [0, 0, 1.2]])
    assert o2.anchor == "O" and o2._needs_dedup()
    co2 = MoleculeMove("CO2", ["O", "C", "O"], [[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]])
    assert co2.anchor == "C" and co2.partners == ("O", "O")
    with pytest.raises(ValueError, match="one atom"):
        MoleculeMove("X", ["C"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="star-shaped"):   # ethane: H on the other C
        from ase.build import molecule
        eth = molecule("C2H6")
        MoleculeMove("C2H6", eth.get_chemical_symbols(), eth.positions)
    with pytest.raises(ValueError, match="anchor"):
        MoleculeMove("CO", ["C", "O"], [[0, 0, 0], [0, 0, 1.1]], anchor="N")


@pytest.mark.parametrize("make", [
    lambda: _co(shell=(0.8, 1.6), bond_sigma=0.3),
    lambda: MoleculeMove("H2O", ["O", "H", "H"], [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]],
                         shell=(0.6, 1.4), bond_sigma=0.3, angle_sigma_deg=50),
    lambda: MoleculeMove("CH4", ["C"] + ["H"] * 4, CH4_TEMPLATE, shell=(0.7, 1.5),
                         bond_sigma=0.3, angle_sigma_deg=60),
])
def test_proposal_density_is_normalised(make):
    """E_uniform-in-shell^k [q] * V_shell^k = integral of q = 1."""
    mol = make()
    rng = np.random.default_rng(0)
    r_min, r_max = mol.shell
    v_sh = 4.0 / 3.0 * np.pi * (r_max ** 3 - r_min ** 3)
    k, n = len(mol.partners), 60_000
    vals = np.empty(n)
    for i in range(n):
        r = (r_min ** 3 + rng.random(k) * (r_max ** 3 - r_min ** 3)) ** (1 / 3)
        vecs = np.array([ri * uniform_direction(rng) for ri in r])
        vals[i] = np.exp(mol.log_q_roles(vecs))
    est, err = vals.mean() * v_sh ** k, vals.std() / np.sqrt(n) * v_sh ** k
    assert abs(est - 1.0) < max(4 * err, 0.02), (est, err)


def test_proposals_lie_in_the_shell_with_finite_density():
    mol = MoleculeMove("CH4", ["C"] + ["H"] * 4, CH4_TEMPLATE)
    rng = np.random.default_rng(1)
    for _ in range(200):
        v = mol.propose(rng)
        r = np.linalg.norm(v, axis=1)
        assert np.all((r >= mol.shell[0]) & (r <= mol.shell[1]))
        assert np.isfinite(mol.log_q_roles(v))


def test_unit_counting_and_density_symmetry():
    """Counting, random picks and Q of the same unit from both sides of a move."""
    sysm = SlabSystem(8.0, 8.0, 8.0, ZeroCalculator(), exchange_species=["O"])
    o2 = MoleculeMove("O2", ["O", "O"], [[0, 0, 0], [0, 0, 1.2]])
    sysm.set_molecules([o2])
    # three O in a row 1.2 A apart: units (0,1) and (1,2)
    sysm.apply_insert_many(np.array([[1, 1, 1], [1, 1, 2.2], [1, 1, 3.4]]), ["O"] * 3)
    P, S, Z = sysm.positions, sysm.species, sysm.in_exchange_zone(sysm.positions)
    assert o2.count_units(P, S, Z, sysm._min_image) == 2
    unit, m = o2.random_unit(P, S, Z, sysm._min_image, np.random.default_rng(0))
    assert m == 2 and unit in ((0, 1), (1, 2))
    # Q sums both anchors for O2: twice one ordering
    q = o2.log_unit_density(P, S, Z, (0, 1), sysm._min_image, sysm.volume)
    one = -np.log(sysm.volume) + o2.log_q_roles(np.array([[0, 0, 1.2]]))
    assert q == pytest.approx(one + np.log(2.0))
    # periodic images: a CO across the x boundary is one unit
    s2 = SlabSystem(6.0, 6.0, 6.0, ZeroCalculator(), exchange_species=["C", "O"])
    co = _co()
    s2.set_molecules([co])
    s2.apply_insert_many(np.array([[0.5, 1, 1], [5.37, 1, 1]]), ["C", "O"])
    Z2 = s2.in_exchange_zone(s2.positions)
    assert co.count_units(s2.positions, s2.species, Z2, s2._min_image) == 1


def test_set_molecules_validation_and_zones():
    sysm = SlabSystem(6.0, 6.0, 6.0, ZeroCalculator(), exchange_species=["C", "O"],
                      insertion_z_range=(1.0, 4.0))
    with pytest.raises(ValueError, match="not exchanged"):
        sysm.set_molecules([MoleculeMove("NO", ["N", "O"], [[0, 0, 0], [0, 0, 1.15]])])
    with pytest.raises(ValueError, match="minimum-image"):
        sysm.set_molecules([_co(shell=(0.5, 3.5))])
    with pytest.raises(ValueError, match="atomic_exchange"):
        sysm.set_molecules([_co()], atomic_exchange=["H"])
    sysm.set_molecules([_co()])
    assert sysm.atomic_exchange == ()
    sysm.set_molecules([])
    assert sysm.atomic_exchange == ("C", "O")
    rng = np.random.default_rng(0)
    pts = np.array([sysm.random_insertion_position(rng) for _ in range(50)])
    sysm.apply_insert_many(np.vstack([pts, [[1, 1, 0.5], [1, 1, 5.0]]]), ["C"] * 52)
    assert sysm.in_exchange_zone(sysm.positions).nonzero()[0].tolist() == \
        sysm.exchange_zone_indices().tolist()
    cl = ClusterSystem(12.0, ZeroCalculator(), exchange_species=["C", "O"],
                       insertion_shell=(2.0, 5.0))
    pts = np.array([cl.random_insertion_position(rng) for _ in range(50)])
    cl.apply_insert_many(np.vstack([pts, [[6, 6, 6]]]), ["O"] * 51)
    assert cl.in_exchange_zone(cl.positions).nonzero()[0].tolist() == \
        cl.exchange_zone_indices().tolist()


# ---------------------------------------------------------------- exact ideal-gas statistics
def test_ideal_gas_co_number_distribution():
    V, y = 6.0 ** 3, 6.0
    mu = 0.5 * (math.log(y) - 2 * math.log(V))
    c, rep = _counts(["C", "O"], [_co()], [], [mu, mu], 150_000)
    assert np.all(c[:, 0] == c[:, 1])
    exact = _exact_mean([n * math.log(y) - 2 * math.lgamma(n + 1) for n in range(15)])
    assert c[:, 0].mean() == pytest.approx(exact, abs=0.12)
    assert rep.n_accepts["insert_mol:CO"] > 100


def test_ideal_gas_o2_number_distribution():
    """Anchor and partner of the same element: both anchors count."""
    V, y = 6.0 ** 3, 30.0
    mu = 0.5 * (math.log(y) - 2 * math.log(V))
    o2 = MoleculeMove("O2", ["O", "O"], [[0, 0, 0], [0, 0, 1.21]])
    c, _ = _counts(["O"], [o2], [], [mu], 120_000)
    assert np.all(c[:, 0] % 2 == 0)
    exact = _exact_mean([n * math.log(y) - math.lgamma(2 * n + 1) for n in range(12)])
    assert (c[:, 0] // 2).mean() == pytest.approx(exact, abs=0.12)


def test_ideal_gas_ch4_number_distribution():
    """Four identical partners (24 orderings) and the chirality mixture."""
    L, y = 4.0, 200.0
    ch4 = MoleculeMove("CH4", ["C"] + ["H"] * 4, CH4_TEMPLATE, shell=(0.5, 1.9),
                       bond_sigma=0.5, angle_sigma_deg=60)
    mu_h = (math.log(y) - 5 * math.log(L ** 3)) / 4
    c, _ = _counts(["C", "H"], [ch4], [], [0.0, mu_h], 60_000, L=L, max_disp=2.0)
    assert np.all(4 * c[:, 0] == c[:, 1])
    exact = _exact_mean([n * math.log(y) - math.lgamma(n + 1) - math.lgamma(4 * n + 1)
                         for n in range(6)])
    assert c[:, 0].mean() == pytest.approx(exact, abs=0.05)


def test_ideal_gas_atomic_plus_molecular_channels():
    """With atomic moves too, N_C and N_O are independent Poisson (z_e V)."""
    V = 6.0 ** 3
    zc, zo = 1.5 / V, 2.5 / V
    c, rep = _counts(["C", "O"], [_co()], ["C", "O"], [math.log(zc), math.log(zo)], 150_000)
    assert c[:, 0].mean() == pytest.approx(1.5, abs=0.1)
    assert c[:, 1].mean() == pytest.approx(2.5, abs=0.12)
    assert c[:, 1].var() == pytest.approx(2.5, rel=0.15)
    assert rep.n_attempts["insert_mol:CO"] > 1000 and rep.n_attempts["insert_C"] > 1000


# ---------------------------------------------------------------- interacting gas
def test_interacting_gas_atomic_and_molecular_exchange_agree():
    params = {("C", "O"): (3.0, 1.0), ("C", "C"): (0.2, 1.1), ("O", "O"): (0.2, 1.1)}
    co = _co(shell=(0.85, 1.6), bond_sigma=0.1)
    means = {}
    for label, mols in (("atomic", []), ("molecular", [co])):
        res = []
        for seed in range(3):
            sysm = SlabSystem(6.0, 6.0, 6.0, PairLJCalculator(params), exchange_species=["C", "O"])
            sysm.set_molecules(mols, atomic_exchange=["C", "O"])
            rep = Replica(ThermoState(1.0, np.array([-6.5, -6.5]), (0, 0)), sysm,
                          np.random.default_rng(seed), max_disp=0.6)
            counter, out = MoleculeCounts([co]), []
            for k in range(800):
                rep.sweep(50)
                if k >= 40:
                    out.append([*sysm.species_counts(), counter(sysm)[0]])
            rep.check_energy()
            assert abs(rep.last_energy_drift) < 1e-8
            res.append(np.mean(out, axis=0))
        means[label] = np.mean(res, axis=0)
    assert means["molecular"] == pytest.approx(means["atomic"], abs=0.04)
    assert means["atomic"][2] > 0.02          # molecules do form


def test_engine_records_molecule_counts_and_checks_systems():
    def factory(rng):
        s = SlabSystem(6.0, 6.0, 6.0, ZeroCalculator(), exchange_species=["C", "O"])
        s.set_molecules([_co()])
        return s
    sim = ReplicaExchangeGCMC([1.0, 1.5], np.array([[-8.0, -8.0], [-7.0, -7.0]]), factory, seed=0)
    sim.run_regc(60, x0=0.9, diffusion="mc", diffusion_steps=2, record_every=10, burn_in_steps=0,
                 verbose=False)
    samples = sim.samples[(0, 0)]
    assert "obs:" + MOLECULE_COUNTS_NAME in samples
    assert sim.observables_meta[MOLECULE_COUNTS_NAME]["labels"] == ["CO"]
    counter = {"n": 0}

    def mixed(rng):
        counter["n"] += 1
        s = SlabSystem(6.0, 6.0, 6.0, ZeroCalculator(), exchange_species=["C", "O"])
        s.set_molecules([_co()] if counter["n"] == 1 else [_co(weight=2.0)])
        return s
    with pytest.raises(ValueError, match="same molecules"):
        ReplicaExchangeGCMC([1.0, 1.5], np.array([[-8.0, -8.0]]), mixed)


def test_energy_bookkeeping_with_emt():
    """A real potential (EMT Pt + CO): the running energy equals a recomputation."""
    pytest.importorskip("ase")
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from regc.structures import slab_geometry_from_atoms
    from regc.calculators import ASECalculatorAdapter
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=None, orthogonal=False)
    geo = slab_geometry_from_atoms(slab, n_frozen_layers=2, headroom=6.0)
    cell = np.zeros((3, 3))
    cell[:2, :2], cell[2, 2] = geo.cell_xy, geo.Lz
    calc = ASECalculatorAdapter(EMT, cell, pbc=(True, True, False))
    sysm = geo.make_system(calc, ["C", "O"],
                           insertion_z_range=(geo.top_z + 0.5, geo.top_z + 4.0))
    sysm.set_molecules([_co()])
    rep = Replica(ThermoState(0.05, np.array([0.0, 0.0]), (0, 0)), sysm,
                  np.random.default_rng(0), max_disp=0.2)
    for _ in range(4):
        rep.sweep(40, p_displace=0.4, p_insert=0.3, p_delete=0.3)
    rep.check_energy(tol=1e-9)
    assert abs(rep.last_energy_drift) < 1e-8
    assert rep.n_accepts.get("insert_mol:CO", 0) > 0 and sysm.n >= 2


# ---------------------------------------------------------------- reservoirs and feeds
def _o2_pair():
    D0, ea, m, r, w = 5.1, -0.3, 31.998, 1.21, 1580.0
    zpe = 0.5 * 6.62607015e-34 * w * 2.99792458e10 / 1.602176634e-19
    dia = DiatomicReservoir(D0, ea, m, r, w, 2, 3)
    gas = MolecularGas("O2", ["O", "O"], [[0, 0, 0], [0, 0, r]], 2 * ea - D0 - zpe, (w,), 2, 3,
                       (m / 2, m / 2))
    return dia, gas


def test_molecular_gas_reduces_to_the_diatomic_reservoir():
    dia, gas = _o2_pair()
    for T, p in [(300.0, 1e5), (800.0, 1e-3), (1500.0, 10.0)]:
        assert gas.mu_eV(T, p) == pytest.approx(2.0 * dia.mu_eV(T, p), abs=1e-12)
        assert gas.pressure_pascal(T, gas.mu_eV(T, p)) == pytest.approx(p, rel=1e-10)
    assert MolecularGas.from_dict(json.loads(json.dumps(gas.to_dict()))).mu_eV(500, 1.0) == \
        pytest.approx(gas.mu_eV(500, 1.0))


def test_nonlinear_rotor_and_ch4_partition_function():
    """Rotational entropy of CH4 (sigma 12, experimental constants) against tables."""
    from regc.units import kelvin_to_eV
    gas = MolecularGas("CH4", ["C"] + ["H"] * 4, CH4_TEMPLATE, 0.0, (), 12)
    m = gas.principal_moments()
    assert m[0] == pytest.approx(m[2], rel=1e-6) and m[0] == pytest.approx(3.2, rel=0.05)
    # S_rot = k [ln q_rot + 3/2]: tabulated ~ 43.6 J/mol/K at 298 K (vib-free, ideal)
    T, dT = 298.15, 0.01
    mu = lambda t: gas.mu_eV(t, 1e5)
    s_total = -(mu(T + dT) - mu(T - dT)) / (2 * dT) / kelvin_to_eV(1.0) * 8.314462618
    assert s_total == pytest.approx(186.3, abs=1.5)    # standard entropy of CH4 (vib ~0)


def test_from_calculator_with_emt():
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT
    co = MolecularGas.from_calculator(EMT(), "CO")
    assert len(co.vib_wavenumbers_cm) == 1 and co.vib_wavenumbers_cm[0] > 0
    assert co.symmetry_number == 1 and co.principal_moments()[0] < 1e-6
    o2 = MolecularGas.from_calculator(EMT(), "O2")
    assert o2.spin_multiplicity == 3 and o2.symmetry_number == 2
    with pytest.raises(ValueError, match="not a minimum"):
        MolecularGas.from_calculator(EMT(), "CO2")


def test_feed_elemental_chemical_potentials():
    _, o2 = _o2_pair()
    co = MolecularGas("CO", ["C", "O"], [[0, 0, 0], [0, 0, 1.13]], -15.0, (2143.0,), 1)
    feed = MolecularGasFeed(["C", "O"], [co, o2], [0.25, 0.75])
    mu = feed.mu_elements_eV(500.0, 2e5)
    assert mu.sum() == pytest.approx(co.mu_eV(500.0, 0.5e5))
    assert 2 * mu[1] == pytest.approx(o2.mu_eV(500.0, 1.5e5))
    grid = feed.mu_grid([400.0, 600.0], [1.0, 10.0], unit="bar")
    assert grid.shape == (2, 2, 2)
    back = MolecularGasFeed.from_dict(json.loads(json.dumps(feed.to_dict())))
    assert back.mu_elements_eV(500.0, 2e5) == pytest.approx(mu)
    pure = MolecularGasFeed(["C", "O"], [co], [1.0])
    assert pure.determines({"C": 1, "O": 1}) and not pure.determines({"C": 1})
    with pytest.raises(ValueError, match="does not fix"):
        pure.check_moves({"atomic C": {"C": 1}})
    co2 = MolecularGas("CO2", ["O", "C", "O"], [[0, 0, -1.16], [0, 0, 0], [0, 0, 1.16]], -22.0,
                       (667.0, 667.0, 1333.0, 2349.0), 2)
    with pytest.raises(ValueError, match="linearly dependent"):
        MolecularGasFeed(["C", "O"], [co, o2, co2], [1, 1, 1])
    with pytest.raises(ValueError, match="not exchanged"):
        MolecularGasFeed(["O"], [co], [1.0])


# ---------------------------------------------------------------- config runner
def _co_config(tmp_path, **molecules):
    return {
        "structure": {"build": {"surface": "fcc111", "symbol": "Pt", "size": [2, 2, 3],
                                "a": 3.92, "orthogonal": False},
                      "n_frozen_layers": 2, "headroom_angstrom": 6.0},
        "gas": {"species": ["C", "O"], "insertion_z_range_above_surface": [0.5, 4.0]},
        "molecules": molecules or {"CO": {"template": "CO"}},
        "calculator": {"type": "EMT"},
        "replica_exchange": {"temperatures_K": [400, 600], "pressures": [1e-6, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 40,
                             "x0": 0.9, "diffusion": "mc", "diffusion_steps": 1,
                             "record_every": 5, "burn_in_steps": 0, "seed": 1},
        "output": {"directory": str(tmp_path / "out")},
    }


def test_runner_validates_the_molecules_section(tmp_path):
    from regc.runner import setup_from_config, validate_config
    cfg = _co_config(tmp_path)
    cfg["reservoir"] = {"type": "diatomic"}
    with pytest.raises(ValueError, match="drop reservoir"):
        validate_config(cfg)
    cfg = _co_config(tmp_path)
    cfg["gas"]["composition"] = {"C": 0.5, "O": 0.5}
    with pytest.raises(ValueError, match="fraction"):
        validate_config(cfg)
    with pytest.raises(ValueError, match="unknown|not allowed|Unknown"):
        validate_config(_co_config(tmp_path, CO={"template": "CO", "shel": [1, 2]}))
    cfg = _co_config(tmp_path)
    cfg["gas"]["atomic_exchange"] = ["N"]
    with pytest.raises(ValueError, match="atomic_exchange"):
        validate_config(cfg)
    cfg = _co_config(tmp_path)
    cfg["gas"]["atomic_exchange"] = ["C"]           # pure CO cannot fix mu_C alone
    with pytest.raises(ValueError, match="does not fix"):
        setup_from_config(cfg, verbose=False)


def test_runner_co_and_o2_feed_end_to_end(tmp_path):
    pytest.importorskip("pymbar")
    from regc.runner import _feed_from_meta, analyze_from_config, run_from_config
    cfg = _co_config(tmp_path, CO={"template": "CO", "fraction": 0.5},
                     O2={"template": "O2", "fraction": 0.5, "insert": False})
    cfg["gas"]["atomic_exchange"] = ["O"]
    result = run_from_config(cfg, verbose=False)
    setup = result["setup"]
    assert [m.name for m in setup["molecules"]] == ["CO"] and setup["atomic_exchange"] == ("O",)
    mu = np.asarray(setup["mu_grid"])
    feed = setup["feed"]
    co, o2 = feed.gases
    assert mu[0, 1].sum() == pytest.approx(co.mu_eV(400.0, 0.5e5))
    assert 2 * mu[0, 1, 1] == pytest.approx(o2.mu_eV(400.0, 0.5e5))
    with open(tmp_path / "out" / "run_metadata.json") as f:
        meta = json.load(f)
    assert meta["atomic_exchange"] == ["O"] and meta["molecules"][0]["name"] == "CO"
    back = _feed_from_meta(meta)
    assert back.mu_elements_eV(500.0, 1e5) == pytest.approx(feed.mu_elements_eV(500.0, 1e5))
    acc = result["sim"].grid[(1, 1)].n_attempts
    assert acc.get("insert_mol:CO", 0) > 0 and acc.get("insert_O", 0) > 0
    cfg["phase_diagram"] = {"T_min_K": 400, "T_max_K": 600, "n_T": 3, "p_min": 1e-6, "p_max": 1.0,
                            "n_p": 3, "observable": "N_C", "coverage_per": "surface_atom",
                            "discard": 0.0}
    out = analyze_from_config(cfg, verbose=False)
    assert np.all(np.isfinite(out["theta_grid"]))
