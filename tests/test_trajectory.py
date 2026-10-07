"""Trajectory saving: file layout, frame count, and content correctness."""

import os

import numpy as np
import pytest

ase = pytest.importorskip("ase", reason="pip install ase to run this test")
from ase.io import read  # noqa: E402

from gcmc_re import LJCalculator, LJSystem, ReplicaExchangeGCMC, SlabLJSystem


def test_trajectory_files_and_frame_count(tmp_path):
    def make_bulk(rng):
        return LJSystem(6.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))

    traj_dir = tmp_path / "traj"
    sim = ReplicaExchangeGCMC([1.0, 1.2], [-3.5, -3.0], make_bulk, seed=1, n_workers=1)
    sim.run(n_cycles=30, sweeps_per_cycle=15, record_every=5,
            trajectory_dir=str(traj_dir), trajectory_every=10, verbose=False)

    files = sorted(os.listdir(traj_dir))
    assert files == [f"replica_T{it}_mu{imu}.xyz" for it in range(2) for imu in range(2)]

    for f in files:
        frames = read(traj_dir / f, index=":")
        assert len(frames) == 3  # 30 cycles / trajectory_every=10
        assert np.allclose(frames[-1].cell[:].diagonal(), [6.0, 6.0, 6.0])
        assert all(frames[-1].pbc)  # bulk: periodic in all three


def test_trajectory_defaults_to_record_every(tmp_path):
    def make_bulk(rng):
        return LJSystem(6.0, LJCalculator())

    traj_dir = tmp_path / "traj"
    sim = ReplicaExchangeGCMC([1.0], [-3.0], make_bulk, seed=1, n_workers=1)
    sim.run(n_cycles=20, sweeps_per_cycle=10, record_every=4,
            trajectory_dir=str(traj_dir), verbose=False)  # trajectory_every omitted

    frames = read(traj_dir / "replica_T0_mu0.xyz", index=":")
    assert len(frames) == 20 // 4


def test_trajectory_includes_frozen_substrate(tmp_path):
    frozen_pos = np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]])
    frozen_species = np.array(["Fe", "Fe"])

    def make_slab(rng):
        calc = LJCalculator(sigma=1.0, epsilon=0.3, r_cut=2.5)
        # 6 x 6 cell: the 2.5 cutoff must stay within the minimum-image radius (3.0)
        return SlabLJSystem(6.0, 6.0, 8.0, calc, exchange_species="H", wall_epsilon=0.0,
                             frozen_positions=frozen_pos, frozen_species=frozen_species,
                             insertion_z_range=(0.5, 3.0))

    traj_dir = tmp_path / "traj"
    sim = ReplicaExchangeGCMC([1.0], [-2.0], make_slab, seed=1, n_workers=1)
    sim.run(n_cycles=10, sweeps_per_cycle=10, record_every=5,
            trajectory_dir=str(traj_dir), trajectory_every=5, verbose=False)

    frames = read(traj_dir / "replica_T0_mu0.xyz", index=":")
    last = frames[-1]
    assert list(last.get_chemical_symbols()).count("Fe") == 2
    assert list(last.pbc) == [True, True, False]
    assert np.allclose(last.cell[:].diagonal(), [6.0, 6.0, 8.0])
