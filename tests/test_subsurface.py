"""
Subsurface insertion/removal with NCMC (`regc.subsurface`).

* the interstitial sites of an fcc(100) slab are its octahedral and
  tetrahedral holes;
* the proposal density integrates to the probability of landing in the
  region;
* exact one-body test: with a potential well f(z) for the mobile atoms the
  mean number in any region R is z A int_R exp(-beta f) dz, whatever the
  NCMC protocol (T, kernel) -- this checks the work, the kernels, q and the
  region counts together;
* the running energy stays exact with a displaceable substrate;
* the runner's species rules (homonuclear diatomics atomic below the
  surface, heteronuclear molecules only above).
"""

import math

import numpy as np
import pytest

from regc import ReplicaExchangeGCMC, SlabSystem, ThermoState
from regc.calculators import Calculator
from regc.replica import Replica
from regc.subsurface import Subsurface, ncmc

EPS, Z0, WIDTH = 2.0, 3.0, 0.6
L, LZ = 6.0, 10.0
FROZEN = np.array([[i * 2.0, j * 2.0, k * 2.0 + 1.0] for i in range(3) for j in range(3)
                   for k in range(3)])          # simple cubic layers at z = 1, 3, 5


class OneBody(Calculator):
    """f(z) = -EPS exp(-(z - Z0)^2 / 2 WIDTH^2) on every mobile atom, nothing else."""

    supports_incremental = False
    supports_forces = True

    def total_energy(self, system) -> float:
        z = system.positions[:, 2]
        return float(np.sum(-EPS * np.exp(-(z - Z0) ** 2 / (2 * WIDTH ** 2))))

    def forces(self, system) -> np.ndarray:
        z = system.positions[:, 2]
        f = np.zeros((system.n_substrate + system.n, 3))
        f[system.n_substrate:, 2] = (-EPS * np.exp(-(z - Z0) ** 2 / (2 * WIDTH ** 2))
                                     * (z - Z0) / WIDTH ** 2)
        return f


def _slab(calc=None, **sub):
    s = SlabSystem(L, L, LZ, calc or OneBody(), exchange_species=["X"], frozen_positions=FROZEN,
                   frozen_species=np.array(["M"] * len(FROZEN)))
    s.set_subsurface(Subsurface(5.0, 1.5, ["X"], delta=0.5, site_min_distance=0.5, **sub))
    return s


def _exact(lo, hi, mu):
    from scipy.integrate import quad
    w = quad(lambda t: math.exp(EPS * math.exp(-(t - Z0) ** 2 / (2 * WIDTH ** 2))), lo, hi,
             points=[Z0])[0]
    return math.exp(mu) * L * L * w


# ---------------------------------------------------------------- sites and proposal
def test_fcc100_sites_are_the_octahedral_and_tetrahedral_holes():
    pytest.importorskip("ase")
    from ase.build import fcc100
    from regc.structures import slab_geometry_from_atoms
    geo = slab_geometry_from_atoms(fcc100("Pd", (3, 3, 4), a=3.89, vacuum=None, orthogonal=True),
                                   n_frozen_layers=2, headroom=6.0)
    s = geo.make_system(Calculator(), "O")
    sub = Subsurface(geo.top_z, geo.top_z - 4.0, ["O"])
    sites = sub.sites(s)
    metal = np.vstack([s.frozen_positions, s.substrate_positions])
    d = np.linalg.norm(s._min_image(sites[:, None] - metal[None]), axis=-1).min(axis=1)
    octa, tetra = np.isclose(d, 3.89 / 2, atol=0.02), np.isclose(d, 3.89 * math.sqrt(3) / 4, atol=0.02)
    assert np.all(octa | tetra)
    assert octa.sum() == 2 * 9 and tetra.sum() == 2 * 18      # two layers, two gaps in the region
    assert np.array_equal(sites, sub.sites(s))                 # deterministic


def test_proposal_density_integrates_to_the_landing_probability():
    s = _slab()
    sub = s.subsurface
    sites = sub.sites(s)
    assert len(sites) == 18
    rng = np.random.default_rng(0)
    landed = np.mean([sub.in_region(sub.propose(sites, rng)[None])[0] for _ in range(4000)])
    # integral of q over the region, by uniform sampling of the region
    n = 200_000
    pts = np.column_stack([rng.random(n) * L, rng.random(n) * L, 1.5 + rng.random(n) * 3.5])
    vol = L * L * 3.5
    q = np.array([math.exp(sub.log_q(p, sites, s)) if np.isfinite(sub.log_q(p, sites, s))
                  else 0.0 for p in pts[:20000]])
    assert q.mean() * vol == pytest.approx(landed, abs=0.03)


def test_set_subsurface_confines_the_gas_zone_and_validates():
    s = SlabSystem(L, L, LZ, OneBody(), exchange_species=["X"], frozen_positions=FROZEN,
                   frozen_species=np.array(["M"] * len(FROZEN)))
    full = s.insertion_z_range
    s.set_subsurface(Subsurface(5.0, 1.5, ["X"]))
    assert s.insertion_z_range == (5.0, full[1])
    s.set_subsurface(None)
    assert s.insertion_z_range == full
    with pytest.raises(ValueError, match="not exchanged"):
        s.set_subsurface(Subsurface(5.0, 1.5, ["H"]))
    with pytest.raises(ValueError, match="below the top"):
        s.set_subsurface(Subsurface(LZ + 1.0, 1.5, ["X"]))
    with pytest.raises(ValueError, match="z_min < z_cut"):
        Subsurface(1.0, 2.0, ["X"])


def test_ncmc_work_without_relaxation_is_the_energy_change():
    s = _slab(ncmc_steps=4, kernel_steps=0)
    s.apply_insert(np.array([1.0, 1.0, 3.0]), species="X")
    u_with = s.total_energy()
    w, (uw, uo) = ncmc(s, 0, True, s.subsurface, 1.0, np.random.default_rng(0))
    assert w == pytest.approx(u_with - 0.0) and uw == pytest.approx(u_with) and uo == 0.0
    w2, _ = ncmc(s, 0, False, s.subsurface, 1.0, np.random.default_rng(0))
    assert w2 == pytest.approx(-w)


def test_langevin_shadow_work_is_the_energy_drift_without_friction():
    """At gamma = 0 the shadow work is the integrator's energy error, and it vanishes as dt^2."""
    from regc.subsurface import _Alchemy, _kernel_langevin
    drift = []
    for dt in (0.1, 0.05):
        s = _slab(kernel="langevin", leapfrog_steps=20, dt=dt, friction=0.0)
        for p in ([1.0, 1.0, 2.6], [3.0, 3.0, 3.4], [5.0, 1.0, 2.9]):
            s.apply_insert(np.array(p), species="X")
        alch, rng = _Alchemy(s, 0), np.random.default_rng(0)
        m = np.concatenate([s.substrate_masses(), s.masses()])[:, None]
        v = rng.standard_normal((len(m), 3)) / np.sqrt(m)
        e0, lam = alch.energies(), 0.5
        h0 = 0.5 * np.sum(m * v * v) + (1 - lam) * e0[1] + lam * e0[0]
        e1, v1, shadow = _kernel_langevin(alch, lam, e0, v, s.subsurface, 1.0, rng)
        h1 = 0.5 * np.sum(m * v1 * v1) + (1 - lam) * e1[1] + lam * e1[0]
        assert shadow == pytest.approx(h1 - h0, abs=1e-12)
        drift.append(abs(shadow))
    assert drift[1] < drift[0] / 2.5            # ~ dt^2


# ---------------------------------------------------------------- exact statistics
@pytest.mark.parametrize("sub", [dict(ncmc_steps=1),
                                 dict(ncmc_steps=4, kernel="hmc", leapfrog_steps=4, dt=0.05),
                                 dict(ncmc_steps=4, kernel="langevin", leapfrog_steps=4, dt=0.05,
                                      friction=2.0),
                                 dict(ncmc_steps=4, kernel="mc", max_disp=0.3)])
def test_one_body_well_exact_region_averages(sub):
    mu = -4.0
    exact_sub, exact_gas = _exact(1.5, 5.0, mu), _exact(5.0, LZ, mu)
    res = []
    for seed in range(3):
        s = _slab(**sub)
        rep = Replica(ThermoState(1.0, mu, (0, 0)), s, np.random.default_rng(10 + seed),
                      max_disp=1.5)
        out = []
        for k in range(700):
            rep.sweep(20)
            if k >= 100:
                z = s.positions[:, 2]
                out.append([np.sum((z >= 1.5) & (z < 5.0)), np.sum(z >= 5.0)])
        rep.check_energy()
        assert abs(rep.last_energy_drift) < 1e-9
        assert rep.n_accepts.get("insert_sub:X", 0) > 20 and rep.n_accepts.get("delete_sub:X", 0) > 20
        res.append(np.mean(out, axis=0))
    m = np.mean(res, axis=0)
    assert m[0] == pytest.approx(exact_sub, rel=0.12)
    assert m[1] == pytest.approx(exact_gas, rel=0.12)


def test_energy_bookkeeping_with_a_displaceable_substrate():
    """PairLJ metal-X attraction: NCMC moves relax the substrate; the running energy stays exact."""
    from regc import PairLJCalculator
    calc = PairLJCalculator({("M", "X"): (0.5, 1.6), ("M", "M"): (0.3, 1.8), ("X", "X"): (0.1, 1.5)},
                            r_cut_factor=1.6)
    top = FROZEN[FROZEN[:, 2] > 4.0]
    s = SlabSystem(L, L, LZ, calc, exchange_species=["X"], frozen_positions=FROZEN[FROZEN[:, 2] < 4.0],
                   frozen_species=np.array(["M"] * 18), substrate_positions=top,
                   substrate_species=np.array(["M"] * 9))
    s.set_subsurface(Subsurface(5.5, 1.5, ["X"], delta=0.4, ncmc_steps=3, kernel="hmc",
                                leapfrog_steps=3, dt=0.02, site_min_distance=0.5, weight=3.0))
    rep = Replica(ThermoState(0.5, 0.0, (0, 0)), s, np.random.default_rng(1), max_disp=0.2)
    for _ in range(15):
        rep.sweep(20, p_displace=0.2, p_insert=0.4, p_delete=0.4)
    rep.check_energy()
    assert abs(rep.last_energy_drift) < 1e-8
    assert rep.n_attempts.get("insert_sub:X", 0) > 0


def test_engine_requires_the_same_settings_everywhere():
    counter = {"n": 0}

    def factory(rng):
        counter["n"] += 1
        return _slab(ncmc_steps=2 if counter["n"] == 1 else 3)
    with pytest.raises(ValueError, match="same subsurface"):
        ReplicaExchangeGCMC([1.0, 1.5], np.array([-4.0]), factory)


# ---------------------------------------------------------------- runner
def _pd100_config(tmp_path, **extra):
    cfg = {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [2, 2, 4], "a": 3.89,
                                "orthogonal": True},
                      "n_frozen_layers": 2, "headroom_angstrom": 5.0},
        "gas": {"species": "O"},
        "calculator": {"type": "EMT"},
        "reservoir": {"type": "diatomic", "D0_eV": 5.1, "bond_length_angstrom": 1.2075,
                      "vibrational_wavenumber_cm": 1580.2},
        "subsurface": {"depth_angstrom": 2.5, "ncmc_steps": 3, "kernel_steps": 1,
                       "leapfrog_steps": 2, "timestep_fs": 0.5},
        "replica_exchange": {"temperatures_K": [500], "pressures": [1e-6, 1.0], "pressure_unit": "bar",
                             "scheme": "regc", "n_steps": 12, "x0": 0.9, "diffusion": "mc",
                             "diffusion_steps": 1, "record_every": 4, "burn_in_steps": 0, "seed": 0},
        "output": {"directory": str(tmp_path / "out")},
    }
    cfg.update(extra)
    return cfg


def test_runner_subsurface_species_rules(tmp_path):
    pytest.importorskip("ase")
    from regc.runner import setup_from_config
    setup = setup_from_config(_pd100_config(tmp_path), verbose=False)
    sub = setup["subsurface"]
    assert sub.species == ("O",) and sub.z_cut == pytest.approx(setup["geometry"].top_z)
    assert setup["metadata"]["subsurface"]["species"] == ["O"]
    # O2 as a molecule: its subsurface attempts are atomic O
    cfg = _pd100_config(tmp_path, molecules={"O2": {"template": "O2"}})
    del cfg["reservoir"]
    setup = setup_from_config(cfg, verbose=False)
    assert setup["subsurface"].species == ("O",) and setup["atomic_exchange"] == ()
    # CO only: exchanged only above the surface, so no subsurface species qualifies
    cfg = _pd100_config(tmp_path, molecules={"CO": {"template": "CO"}})
    del cfg["reservoir"]
    cfg["gas"] = {"species": ["C", "O"]}
    with pytest.raises(ValueError, match="no species qualifies"):
        setup_from_config(cfg, verbose=False)
    cfg["subsurface"]["species"] = ["O"]          # and O alone is not fixed by a CO feed
    with pytest.raises(ValueError, match="does not fix"):
        setup_from_config(cfg, verbose=False)


def test_runner_subsurface_end_to_end(tmp_path):
    pytest.importorskip("ase")
    from regc.runner import run_from_config
    result = run_from_config(_pd100_config(tmp_path), verbose=False)
    sim = result["sim"]
    assert sim.subsurface is not None
    attempts = sum(r.n_attempts.get("insert_sub:O", 0) + r.n_attempts.get("delete_sub:O", 0)
                   for r in sim.grid.values())
    assert attempts > 0
    for r in sim.grid.values():
        r.check_energy()
        assert abs(r.last_energy_drift) < 1e-8
