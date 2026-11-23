"""
Site-exchange channels (`regc.subsurface.Subsurface` generalised).

* trilateration sites find a vacancy of a metal surface and the hollow
  above it, deterministically and whatever the order of the atoms;
* the defensive mixture: propose() samples exactly the density log_q gives;
* exact one-body test with trilateration sites (which depend on the mobile
  atoms, so q must be evaluated on the configuration without the removed
  atom), a defensive fraction and the held switched atom, as a partition
  and as an overlay of the uniform gas moves;
* several channels at once keep the energy bookkeeping exact;
* the runner's ``site_exchange`` section.
"""

import math

import numpy as np
import pytest

from regc import ReplicaExchangeGCMC, SlabSystem, ThermoState
from regc.replica import Replica
from regc.subsurface import Subsurface, channel_label, system_channels

from test_subsurface import FROZEN, LZ, L, OneBody, _exact

R0_PD = 3.89 / math.sqrt(2.0)


def _pd100(n_exchangeable=1):
    pytest.importorskip("ase")
    from ase.build import fcc100
    from regc.calculators import Calculator
    from regc.structures import slab_geometry_from_atoms
    geo = slab_geometry_from_atoms(fcc100("Pd", (3, 3, 4), a=3.89, vacuum=None, orthogonal=True),
                                   n_frozen_layers=2, headroom=6.0,
                                   n_exchangeable_layers=n_exchangeable)
    return geo, geo.make_system(Calculator(), ["Pd"])


def _tri(top, **kw):
    return Subsurface(top + 3.0, top - 1.0, ["Pd"], site_method="trilateration",
                      bond_length=R0_PD, site_min_distance=0.6 * R0_PD, **kw)


# ---------------------------------------------------------------- sites
def test_trilateration_finds_a_vacancy_and_the_adatom_hollows():
    geo, s = _pd100()
    sub = _tri(geo.top_z)
    sites = sub.sites(s)
    # perfect surface: the 9 fourfold hollows one interlayer spacing above the top layer
    assert len(sites) == 9
    assert np.allclose(sites[:, 2], geo.top_z + 3.89 / 2, atol=1e-6)
    d = np.linalg.norm(s._min_image(sites[:, None, :] - s.positions[None, :, :]), axis=-1)
    assert np.allclose(np.sort(d, axis=1)[:, :4], R0_PD, atol=1e-6)   # R0 from 4 top atoms
    # a removed top-layer atom leaves a site at its place
    y = s.positions[4].copy()
    s.positions, s.species = np.delete(s.positions, 4, axis=0), np.delete(s.species, 4)
    sites = sub.sites(s)
    assert np.min(np.linalg.norm(s._min_image(sites - y), axis=-1)) < 0.3
    assert np.array_equal(sites, sub.sites(s))                       # deterministic
    # and independent of the order of the atoms
    perm = np.random.default_rng(0).permutation(s.n)
    s.positions, s.species = s.positions[perm], s.species[perm]
    assert np.allclose(sites, sub.sites(s), atol=1e-9)


def test_voronoi_default_is_unchanged():
    geo, s = _pd100(0)
    old = Subsurface(geo.top_z, geo.top_z - 4.0, ["Pd"])
    assert old.site_method == "voronoi" and old.defensive_fraction == 0.0
    assert old.partition_gas_zone and old.merge_tol == 0.25
    assert len(old.sites(s)) == 2 * 9 + 2 * 18
    with pytest.raises(ValueError, match="bond_length"):
        Subsurface(1.0, 0.5, ["Pd"], site_method="trilateration")
    with pytest.raises(ValueError, match="site_method"):
        Subsurface(1.0, 0.5, ["Pd"], site_method="delaunay")
    with pytest.raises(ValueError, match="defensive_fraction"):
        Subsurface(1.0, 0.5, ["Pd"], defensive_fraction=1.0)


def test_settings_pickled_before_the_new_fields_still_work():
    """A Subsurface unpickled from an old checkpoint lacks the new attributes: class defaults."""
    import pickle
    s = _slab_one_body([dict()])
    sub = s.subsurface
    for k in ("site_method", "bond_length", "site_merge", "defensive_fraction",
              "partition_gas_zone"):
        del sub.__dict__[k]
    old = pickle.loads(pickle.dumps(sub))
    assert old.site_method == "voronoi" and old.defensive_fraction == 0.0 and old.partition_gas_zone
    assert np.array_equal(old.sites(s), Subsurface(5.0, 1.5, ["X"], delta=0.5,
                                                   site_min_distance=0.5).sites(s))
    del s.subsurface_channels, s.subsurface_labels          # a system from before the channels
    assert system_channels(s) == (s.subsurface,) and channel_label(s, 0) == "sub"


# ---------------------------------------------------------------- proposal density
def test_defensive_mixture_proposal_matches_log_q():
    s = _slab_one_body([dict(defensive_fraction=0.3)])
    sub = s.subsurface
    sites = sub.sites(s)
    rng = np.random.default_rng(1)
    n = 200_000
    ys = np.array([s.wrap_position(sub.propose(sites, rng, s)) for _ in range(n)])
    vol = L * L * 3.5
    # a box inside a site ball (both parts) and one away from every site (uniform part only)
    site = sites[len(sites) // 2]
    far = np.array([0.0, 0.0, 3.0])
    assert np.min(np.linalg.norm(s._min_image(sites - far), axis=-1)) > sub.delta + 0.9
    for centre, half in ((site, 0.2), (far, 0.5)):
        inside = np.all(np.abs(s._min_image(ys - centre)) < half, axis=1)
        pts = centre + (rng.random((4000, 3)) - 0.5) * 2 * half
        q = np.mean([math.exp(sub.log_q(p, sites, s)) for p in pts])
        expected = q * (2 * half) ** 3
        assert inside.mean() == pytest.approx(expected, rel=4.0 / math.sqrt(expected * n))
    # the far box sees exactly the uniform part
    assert sub.log_q(far, sites, s) == pytest.approx(math.log(0.3 / vol))
    # q integrates to the probability of landing in the region
    pts = np.column_stack([rng.random(20000) * L, rng.random(20000) * L,
                           1.5 + rng.random(20000) * 3.5])
    q = np.array([math.exp(sub.log_q(p, sites, s)) for p in pts])
    assert q.mean() * vol == pytest.approx(np.mean(sub.in_region(ys)), abs=0.02)
    # no site at all: the uniform part takes all the weight
    assert sub.log_q(far, np.zeros((0, 3)), s) == pytest.approx(-math.log(vol))
    assert sub.in_region(sub.propose(np.zeros((0, 3)), rng, s)[None])[0]


# ---------------------------------------------------------------- exact statistics
def _slab_one_body(channels, frozen=FROZEN):
    s = SlabSystem(L, L, LZ, OneBody(), exchange_species=["X"], frozen_positions=frozen,
                   frozen_species=np.array(["M"] * len(frozen)))
    s.set_subsurface([Subsurface(5.0, 1.5, ["X"], **dict(dict(delta=0.5, site_min_distance=0.5),
                                                         **c)) for c in channels])
    return s


# trilateration R0 = 2 on the simple cubic toy lattice (spacing 2): the sites depend on the
# mobile atoms too, and eps > 0 lets atoms away from every site be removed
TRI = dict(site_method="trilateration", bond_length=2.0, site_min_distance=1.0,
           defensive_fraction=0.1, delta=0.6, ncmc_steps=3, kernel="langevin", leapfrog_steps=3,
           dt=0.05, friction=2.0, freeze_switched=True)


@pytest.mark.parametrize("channel", [dict(TRI), dict(TRI, partition_gas_zone=False, weight=3.0)],
                         ids=["partition", "overlay"])
def test_one_body_well_exact_with_trilateration_sites(channel):
    mu = -4.0
    exact_sub, exact_gas = _exact(1.5, 5.0, mu), _exact(5.0, LZ, mu)
    res = []
    for seed in range(3):
        s = _slab_one_body([channel], frozen=FROZEN[FROZEN[:, 2] < 4.0])  # fewer atoms: faster
        if not channel.get("partition_gas_zone", True):
            assert s.subsurface is None and s.insertion_z_range == (0.0, LZ)
        rep = Replica(ThermoState(1.0, mu, (0, 0)), s, np.random.default_rng(20 + seed),
                      max_disp=1.5)
        out = []
        for k in range(1100):     # 6 seeds x 2900 sweeps: within 1.5% of exact, seed sd 3%
            rep.sweep(20)
            if k >= 100:
                z = s.positions[:, 2]
                out.append([np.sum((z >= 1.5) & (z < 5.0)), np.sum(z >= 5.0)])
        rep.check_energy()
        assert abs(rep.last_energy_drift) < 1e-9
        label = channel_label(s, 0)
        assert label == ("sub" if channel.get("partition_gas_zone", True) else "site0")
        assert rep.n_accepts.get(f"insert_{label}:X", 0) > 20
        assert rep.n_accepts.get(f"delete_{label}:X", 0) > 20
        res.append(np.mean(out, axis=0))
    m = np.mean(res, axis=0)
    assert m[0] == pytest.approx(exact_sub, rel=0.12)
    assert m[1] == pytest.approx(exact_gas, rel=0.12)


def test_two_channels_energy_bookkeeping_and_engine():
    """A partitioning Voronoi channel plus a trilateration overlay, displaceable substrate."""
    from regc import PairLJCalculator
    from regc.observables import SUBSURFACE_COUNTS_NAME

    def factory(rng=None, steps=3):
        calc = PairLJCalculator({("M", "X"): (0.5, 1.6), ("M", "M"): (0.3, 1.8),
                                 ("X", "X"): (0.1, 1.5)}, r_cut_factor=1.6)
        top = FROZEN[FROZEN[:, 2] > 4.0]
        s = SlabSystem(L, L, LZ, calc, exchange_species=["X"],
                       frozen_positions=FROZEN[FROZEN[:, 2] < 4.0],
                       frozen_species=np.array(["M"] * 18), substrate_positions=top,
                       substrate_species=np.array(["M"] * 9))
        s.set_subsurface([
            Subsurface(4.5, 1.5, ["X"], delta=0.4, ncmc_steps=steps, kernel="hmc",
                       leapfrog_steps=3, dt=0.02, site_min_distance=0.5, weight=2.0),
            Subsurface(7.5, 4.0, ["X"], site_method="trilateration", bond_length=2.0,
                       site_min_distance=1.0, defensive_fraction=0.2, delta=0.5,
                       partition_gas_zone=False, ncmc_steps=steps, kernel="langevin",
                       leapfrog_steps=2, dt=0.02, friction=1.0, freeze_switched=True,
                       weight=2.0)])
        return s

    s = factory()
    assert s.insertion_z_range == (4.5, LZ)            # only the partitioning channel raises it
    assert s.subsurface is s.subsurface_channels[0] and s.subsurface_labels == ("sub", "site1")
    rep = Replica(ThermoState(0.5, 0.0, (0, 0)), s, np.random.default_rng(3), max_disp=0.2)
    for _ in range(20):
        rep.sweep(20, p_displace=0.2, p_insert=0.4, p_delete=0.4)
    rep.check_energy()
    assert abs(rep.last_energy_drift) < 1e-9
    for key in ("insert_sub:X", "delete_sub:X", "insert_site1:X", "delete_site1:X"):
        assert rep.n_attempts.get(key, 0) > 0, key
    assert sum(rep.n_accepts.get(k, 0) for k in ("insert_site1:X", "delete_site1:X")) > 0
    # the engine: same channels everywhere, the count observable of the partitioning one
    sim = ReplicaExchangeGCMC([0.5, 0.6], np.array([[0.0], [0.0]]), factory, seed=0)
    counts = [o for o in sim.observables if o.name == SUBSURFACE_COUNTS_NAME]
    assert len(counts) == 1 and len(sim.subsurface_channels) == 2
    assert sim.subsurface is not None and sim.subsurface.z_cut == 4.5
    counter = {"n": 0}

    def mismatched(rng):
        counter["n"] += 1
        return factory(steps=3 if counter["n"] == 1 else 4)
    with pytest.raises(ValueError, match="same subsurface"):
        ReplicaExchangeGCMC([0.5, 0.6], np.array([0.0]), mismatched)


def test_overlay_only_records_no_subsurface_count():
    from regc.observables import SUBSURFACE_COUNTS_NAME

    def factory(rng):
        return _slab_one_body([dict(TRI, partition_gas_zone=False)])
    sim = ReplicaExchangeGCMC([1.0], np.array([[-4.0]]), factory, seed=0)
    assert sim.subsurface is None
    assert SUBSURFACE_COUNTS_NAME not in [o.name for o in sim.observables]
    sim.run_regc(30, x0=0.5, diffusion="mc", diffusion_steps=1, record_every=10,
                 burn_in_steps=0, verbose=False)
    r = sim.grid[(0, 0)]
    assert r.n_attempts.get("insert_site0:X", 0) + r.n_attempts.get("delete_site0:X", 0) > 0


# ---------------------------------------------------------------- runner
def _pd_o_config(tmp_path, **extra):
    cfg = {
        "structure": {"build": {"surface": "fcc100", "symbol": "Pd", "size": [2, 2, 4], "a": 3.94},
                      "n_frozen_layers": 2, "exchangeable_top_layers": 1},
        "gas": {"species": ["O", "Pd"], "insertion_z_range_above_surface": [-1.0, 4.0]},
        "calculator": {"type": "EMT"},
        "reservoir": {"O": {"type": "diatomic", "D0_eV": 5.12, "e_atom_eV": 0.0,
                            "bond_length_angstrom": 1.2075, "vibrational_wavenumber_cm": 1580.2},
                      "Pd": {"type": "bulk", "e_bulk_eV": -0.2, "mean_wavenumber_cm": 150.0}},
        "site_exchange": [{"species": ["Pd"], "z_range_above_surface": [-1.0, 3.0],
                           "ncmc_steps": 2, "kernel": "langevin", "leapfrog_steps": 2,
                           "weight": 2.0}],
        "replica_exchange": {"temperatures_K": [600], "pressures": [1e-6, 1.0],
                             "pressure_unit": "bar", "scheme": "regc", "n_steps": 12,
                             "diffusion": "mc", "diffusion_steps": 1, "record_every": 4,
                             "burn_in_steps": 0, "seed": 0},
        "output": {"directory": str(tmp_path / "out")},
    }
    cfg.update(extra)
    return cfg


def test_runner_site_exchange_setup_and_run(tmp_path, capsys):
    pytest.importorskip("ase")
    from regc.runner import run_from_config, setup_from_config
    setup = setup_from_config(_pd_o_config(tmp_path))
    assert "site exchange 0: ['Pd']" in capsys.readouterr().out
    (c,) = setup["site_exchange"]
    top = setup["geometry"].top_z
    assert c.site_method == "trilateration" and c.bond_length == pytest.approx(3.94 / math.sqrt(2))
    assert (c.z_min, c.z_cut) == pytest.approx((top - 1.0, top + 3.0))
    assert c.defensive_fraction == 0.05 and not c.partition_gas_zone and c.freeze_switched
    assert setup["subsurface"] is None
    assert setup["metadata"]["site_exchange"][0]["site_method"] == "trilateration"
    s = setup["system_factory"](np.random.default_rng(0))
    assert s.insertion_z_range == pytest.approx((top - 1.0, top + 4.0))   # overlay: unchanged
    # together with a subsurface section: that one is channel 0 ('sub'), partitioning
    cfg = _pd_o_config(tmp_path, subsurface={"species": ["O"], "z_cut_below_surface": 1.0,
                                             "depth_angstrom": 2.0, "ncmc_steps": 2})
    setup = setup_from_config(cfg, verbose=False)
    s = setup["system_factory"](np.random.default_rng(0))
    assert s.subsurface_labels == ("sub", "site1") and s.subsurface.species == ("O",)
    result = run_from_config(cfg, verbose=False)
    for r in result["sim"].grid.values():
        assert r.n_attempts.get("insert_site1:Pd", 0) + r.n_attempts.get("delete_site1:Pd", 0) > 0
        r.check_energy()
        assert abs(r.last_energy_drift) < 1e-8


@pytest.mark.parametrize("entry, match", [
    ({"species": ["Pd"], "z_range_above_surface": [-1, 3], "colour": 1}, "unknown key"),
    ({"species": ["Pd"], "z_range_above_surface": [-1, 3], "sites": "delaunay"}, "sites must be"),
    ({"species": ["Au"], "z_range_above_surface": [-1, 3]}, "not exchanged"),
    ({"species": ["Pd"]}, "z_range_above_surface"),
    ({"species": ["Pd"], "z_range_above_surface": [3, -1]}, "lo < hi"),
    ({"species": ["Pd"], "z_range_above_surface": [-1, 3], "partition_gas_zone": "yes"},
     "true or false"),
])
def test_runner_site_exchange_validation(tmp_path, entry, match):
    pytest.importorskip("ase")
    from regc.runner import setup_from_config
    with pytest.raises(ValueError, match=match):
        setup_from_config(_pd_o_config(tmp_path, site_exchange=[entry]), verbose=False)
    with pytest.raises(ValueError, match="non-empty list"):
        setup_from_config(_pd_o_config(tmp_path, site_exchange={"species": "Pd"}), verbose=False)
