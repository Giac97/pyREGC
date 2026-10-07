"""
Trajectory saving. Writes extended-XYZ frames (ASE's most portable format --
readable by ASE itself, OVITO, VMD, and plain text editors) so a replica's
configuration history can be inspected, visualized, or restarted from later.

Requires: pip install ase (imported lazily -- importing this module, or
running a simulation without trajectory saving enabled, never needs it).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def append_trajectory_frame(path: str, positions: np.ndarray, species: np.ndarray,
                             cell: Tuple[float, float, float],
                             pbc: Sequence[bool] = (True, True, False)) -> None:
    """
    Append one frame to an extended-XYZ trajectory file.

    Creates the file (and any missing parent directories) if it doesn't
    exist yet; otherwise appends a new frame to the existing trajectory.

    Parameters
    ----------
    path : str
        Path to the trajectory file to append to (or create).
    positions : numpy.ndarray
        Array of shape (N, 3) with the Cartesian coordinates of the N
        atoms in this frame.
    species : numpy.ndarray
        Array of shape (N,) with the species label (element symbol, or a
        generic placeholder) of each atom, matching `positions` in order.
    cell : tuple of float
        The (Lx, Ly, Lz) orthorhombic cell dimensions for this frame.
    pbc : sequence of bool, optional
        Periodicity along each of the three cell axes. Default is
        ``(True, True, False)`` (periodic in x, y; open in z, matching
        `SlabSystem`'s geometry).

    Returns
    -------
    None
        This function writes to disk and returns nothing.
    """
    import os

    from ase import Atoms
    from ase.io import write

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    atoms = Atoms(symbols=list(species), positions=positions, cell=list(cell), pbc=list(pbc))
    write(path, atoms, format="extxyz", append=True)
