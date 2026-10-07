"""
The self-contained package API: units, reservoirs, structures, sample I/O,
checkpoints and the chunked driver.
"""

import pickle

import numpy as np
import pytest

from gcmc_re import (BulkSystem, DiatomicReservoir, LJCalculator, LJSystem,
                     MonatomicIdealGasReservoir, ReplicaExchangeGCMC, SampleSet, SlabLJSystem,
                     SlabSystem, ThermalWavelengthCubed, eV_to_kelvin, kelvin_to_eV, load_samples,
                     pascal_per, run_with_checkpoints, save_samples,
                     thermal_wavelength_cubed_angstrom3)


def test_unit_helpers():
    assert eV_to_kelvin(kelvin_to_eV(523.0)) == pytest.approx(523.0)
    assert pascal_per("bar") == 1e5 and pascal_per("atm") == 101325.0
    with pytest.raises(ValueError):
        pascal_per("psi")
    lam = ThermalWavelengthCubed(1.008)
    assert lam(kelvin_to_eV(400.0)) == pytest.approx(thermal_wavelength_cubed_angstrom3(400.0, 1.008))
    assert pickle.loads(pickle.dumps(lam)).mass_amu == 1.008


def test_old_system_names_are_aliases():
    assert LJSystem is BulkSystem and SlabLJSystem is SlabSystem


@pytest.mark.parametrize("reservoir", [
    MonatomicIdealGasReservoir(16.48),
    MonatomicIdealGasReservoir(4.0, e_atom_eV=-1.2),
    DiatomicReservoir.hydrogen(e_atom_eV=3.21),
], ids=["monatomic", "monatomic-e_atom", "H2"])
def test_reservoir_round_trips_and_analysis_functions(reservoir):
    T, p = 450.0, 2.5e4
    mu = reservoir.mu_eV(T, p)
    assert reservoir.pressure_pascal(T, mu) == pytest.approx(p, rel=1e-10)
    mu_of_Tp, p_of_Tmu = pickle.loads(pickle.dumps(reservoir.analysis_functions("bar")))
    assert mu_of_Tp(kelvin_to_eV(T), p / 1e5) == pytest.approx(mu)
    assert p_of_Tmu(kelvin_to_eV(T), mu) == pytest.approx(p / 1e5, rel=1e-10)
    grid = reservoir.mu_grid([300.0, 450.0], [p / 1e5, 1.0], unit="bar")
    assert grid.shape == (2, 2) and grid[1, 0] == pytest.approx(mu)
    assert reservoir.lambda_cubed_of_T().mass_amu == pytest.approx(reservoir.mass_exchanged_amu)


def test_monatomic_reservoir_reproduces_paper_fig3_caption():
    """[ZSG] Fig. 3: (200 K, -0.9 eV) is 2.03e-17 atm for their LJ gas (m_B = 16.48 amu)."""
    r = MonatomicIdealGasReservoir(16.48)
    assert r.pressure_at_T_eV(kelvin_to_eV(200.0), -0.9, unit="atm") == pytest.approx(2.03e-17, rel=0.01)


def test_diatomic_reservoir_from_calculator():
    pytest.importorskip("ase")
    from ase.calculators.emt import EMT
    r = DiatomicReservoir.from_calculator(EMT(), "H")
    assert r.e_atom_eV == pytest.approx(3.21, abs=0.01)
    assert r.D0_eV == pytest.approx(r.calculator_info["D0_eV"])
    r_exp = DiatomicReservoir.from_calculator(EMT(), "H", d0="experimental")
    assert r_exp.D0_eV == pytest.approx(4.478)
    with pytest.raises(ValueError):
        DiatomicReservoir.from_calculator(EMT(), "H", d0="bogus")


def test_slab_geometry_layout():
    pytest.importorskip("ase")
    from ase.build import bcc110, fcc111

    from gcmc_re import PairLJCalculator, slab_geometry_from_atoms
    g = slab_geometry_from_atoms(bcc110("Fe", size=(3, 4, 6), a=2.86, vacuum=None, orthogonal=True),
                                 n_frozen_layers=3, headroom=5.0)
    assert len(g.frozen_positions) == 36 and len(g.substrate_positions) == 36
    assert g.frozen_positions[:, 2].max() == pytest.approx(0.0)  # top frozen layer at the wall
    assert g.n_surface == 12 and g.orthogonal
    assert g.Lz == pytest.approx(g.top_z + 5.0)
    h = slab_geometry_from_atoms(fcc111("Cu", size=(3, 3, 2), a=5.416, vacuum=None, orthogonal=False),
                                 headroom=48.0)
    assert not h.orthogonal and len(h.substrate_positions) == 0
    s = h.make_system(PairLJCalculator({("Cu", "B"): (0.66, 1.91), ("B", "B"): (0.01, 1.2)}), "B")
    assert s.volume == pytest.approx(h.area * 48.0)


def _bulk(rng):
    return BulkSystem(5.0, LJCalculator(1.0, 1.0, 2.5))


def _sim(seed=4):
    return ReplicaExchangeGCMC([0.9, 1.2], [[-3.5, -3.0], [-3.5, -3.0]], _bulk, seed=seed,
                               lambda_cubed_of_T=ThermalWavelengthCubed(20.0))


def test_samples_round_trip_and_discard(tmp_path):
    sim = _sim()
    sim.run_regc(n_steps=200, x0=0.8, diffusion='mc_moves', diffusion_steps=3, record_every=10,
                 verbose=False)
    path = str(tmp_path / "s.npz")
    save_samples(sim, path)
    loaded = load_samples(path)
    assert isinstance(loaded, SampleSet) and loaded.samples == sim.samples
    assert loaded.lambda_cubed_of_T.mass_amu == 20.0
    assert np.allclose(loaded.mu_grid, sim.mu_grid)
    assert load_samples(path, discard=0.5).n_samples == loaded.n_samples - loaded.n_samples // 2


def test_checkpoint_refuses_a_different_simulation(tmp_path):
    sim = _sim()
    sim.run_regc(n_steps=50, x0=0.8, diffusion='mc_moves', diffusion_steps=2, record_every=10,
                 verbose=False)
    path = str(tmp_path / "c.pkl")
    sim.save_checkpoint(path, extra={"k": 1})
    assert _sim().load_checkpoint(path) == {"k": 1}
    other = ReplicaExchangeGCMC([0.9, 1.3], [[-3.5, -3.0], [-3.5, -3.0]], _bulk,
                                lambda_cubed_of_T=ThermalWavelengthCubed(20.0))
    with pytest.raises(ValueError, match="temperatures"):
        other.load_checkpoint(path)


def test_driver_resume_is_bit_identical(tmp_path):
    kw = dict(n_steps=300, chunk=100, x0=0.8, diffusion='mc_moves', diffusion_steps=3,
              record_every=10, convergence_every=100, progress=None, metadata={"potential": "LJ"})
    full = run_with_checkpoints(_sim(), str(tmp_path / "a"), **kw)
    assert full.finished and full.steps_done == 300
    first = run_with_checkpoints(_sim(), str(tmp_path / "b"), **dict(kw, n_steps=100))
    assert first.steps_done == 100
    rest = run_with_checkpoints(_sim(), str(tmp_path / "b"), **kw)
    assert rest.resumed_from == 100 and rest.steps_done == 300
    a, b = load_samples(str(tmp_path / "a" / "samples.npz")), load_samples(str(tmp_path / "b" / "samples.npz"))
    assert a.samples == b.samples
    assert full.rho_u_changes == rest.rho_u_changes


def test_driver_guards(tmp_path):
    kw = dict(n_steps=100, chunk=50, x0=0.8, diffusion='mc_moves', diffusion_steps=2,
              record_every=10, progress=None, metadata={"potential": "LJ"})
    run_with_checkpoints(_sim(), str(tmp_path), **kw)
    with pytest.raises(FileExistsError):
        run_with_checkpoints(_sim(), str(tmp_path), resume=False, **kw)
    with pytest.raises(ValueError, match="potential"):
        run_with_checkpoints(_sim(), str(tmp_path), **dict(kw, metadata={"potential": "EAM"}))
    stopped = run_with_checkpoints(_sim(), str(tmp_path / "t"), time_limit=0.0,
                                   **dict(kw, n_steps=10 ** 6))
    assert not stopped.finished and stopped.steps_done == 0


def test_lowest_energy_map():
    pytest.importorskip("pymbar")
    from gcmc_re import MBARSurfaceAnalysis
    samples = {(0, 0): {'U': [0.0, -1.0, -1.5, -2.2], 'N': [0, 1, 2, 3]},
               (0, 1): {'U': [-0.5, -1.4, -2.3, -2.9], 'N': [1, 2, 3, 3]}}
    a = MBARSurfaceAnalysis(SampleSet([1.0], np.array([[-1.0, 0.0]]), samples))
    Ns, E = a.lowest_energy_per_N()
    assert list(Ns) == [0, 1, 2, 3] and list(E) == [0.0, -1.0, -1.5, -2.9]
    # E - mu*N: at mu = -2: N=0 wins; at mu = 0: N=3 wins
    assert list(a.lowest_energy_N([-2.0, 0.0])) == [0, 3]
