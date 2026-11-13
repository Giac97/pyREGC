"""
Trajectory saving. Writes extended-XYZ frames (ASE's most portable format --
readable by ASE itself, OVITO, VMD, and plain text editors) so a replica's
configuration history can be inspected, visualized, or restarted from later.

Requires: pip install ase (imported lazily -- importing this module, or
running a simulation without trajectory saving enabled, never needs it).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

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


# =============================================================================
# Configurations tagged with their sample (for picking structures out of a
# phase diagram, see regc.analysis.MBARSurfaceAnalysis.representative_samples)
# =============================================================================
def snapshot(system) -> tuple:
    """
    A copy of a system's configuration, cheap to ship between processes.

    Parameters
    ----------
    system : ParticleSystem
        The system.

    Returns
    -------
    tuple
        ``(positions, species, n_frozen, n_substrate)``: all atoms in
        `full_configuration` order, and the sizes of the first two groups.
    """
    positions, species = system.full_configuration()
    return (np.array(positions, dtype=float).reshape(-1, 3), np.asarray(species).astype(str),
            int(system.frozen_positions.shape[0]), int(system.substrate_positions.shape[0]))


def lattice_of(system) -> np.ndarray:
    """
    The 3x3 cell matrix (rows a1, a2, a3) of a system.

    Parameters
    ----------
    system : ParticleSystem
        The system (its ``cell`` is (Lx, Ly, Lz) or a 3x3 matrix).

    Returns
    -------
    numpy.ndarray
        Shape (3, 3).
    """
    cell = np.asarray(system.cell, dtype=float)
    return np.diag(cell) if cell.ndim == 1 else cell.reshape(3, 3)


def _format_value(v) -> str:
    """
    One extxyz comment-line value.

    Parameters
    ----------
    v : object
        A number, bool or string.

    Returns
    -------
    str
        Its text (quoted if it contains spaces).
    """
    if isinstance(v, (bool, np.bool_)):
        return "T" if v else "F"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    s = str(v)
    return f'"{s}"' if (" " in s or not s) else s


def format_extxyz_frame(positions: np.ndarray, species: Sequence[str], lattice: np.ndarray,
                        pbc: Sequence[bool], info: Optional[dict] = None,
                        arrays: Optional[Dict[str, np.ndarray]] = None) -> str:
    """
    One extended-XYZ frame as text (readable by ASE, OVITO, VMD, pySNOW).

    Parameters
    ----------
    positions : numpy.ndarray
        Shape (n, 3), Angstrom.
    species : sequence of str
        Per-atom labels.
    lattice : numpy.ndarray
        Cell matrix, shape (3, 3).
    pbc : sequence of bool
        Periodicity of the three axes.
    info : dict, optional
        Per-frame key=value pairs (numbers or strings). Default None.
    arrays : dict, optional
        Extra per-atom integer or float columns. Default None.

    Returns
    -------
    str
        The frame, ending with a newline.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    arrays = arrays or {}
    props = "species:S:1:pos:R:3" + "".join(
        f":{k}:{'I' if np.issubdtype(np.asarray(v).dtype, np.integer) else 'R'}:1"
        for k, v in arrays.items())
    head = [f'Lattice="{" ".join(repr(float(x)) for x in np.asarray(lattice).ravel())}"',
            f"Properties={props}", f'pbc="{" ".join("T" if p else "F" for p in pbc)}"']
    head += [f"{k}={_format_value(v)}" for k, v in (info or {}).items()]
    cols = [np.asarray(v) for v in arrays.values()]
    lines = [str(len(positions)), " ".join(head)]
    for i, (s, x) in enumerate(zip(species, positions)):
        extra = "".join(f" {int(c[i])}" if np.issubdtype(c.dtype, np.integer) else f" {float(c[i]):.8g}"
                        for c in cols)
        lines.append(f"{s} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}{extra}")
    return "\n".join(lines) + "\n"


def append_configuration(path: str, config: tuple, lattice: np.ndarray, pbc: Sequence[bool],
                         info: dict) -> None:
    """
    Append one `snapshot` to an extxyz file, with a per-atom ``group`` column.

    ``group`` is 0 for frozen, 1 for displaceable substrate and 2 for mobile
    (exchanged) atoms.

    Parameters
    ----------
    path : str
        The file (created with its directory if missing).
    config : tuple
        From `snapshot`.
    lattice : numpy.ndarray
        Cell matrix, shape (3, 3).
    pbc : sequence of bool
        Periodicity.
    info : dict
        Per-frame values (e.g. ``sample``, ``U``, ``N``).
    """
    import os

    positions, species, n_fr, n_sub = config
    groups = np.concatenate([np.zeros(n_fr, int), np.ones(n_sub, int),
                             np.full(len(species) - n_fr - n_sub, 2, int)])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(format_extxyz_frame(positions, species, lattice, pbc, info, {"group": groups}))


_KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def _parse_value(text: str):
    """
    Parse one extxyz comment-line value.

    Parameters
    ----------
    text : str
        The raw value (possibly quoted).

    Returns
    -------
    int, float, bool or str
        The value.
    """
    text = text.strip('"')
    for conv in (int, float):
        try:
            return conv(text)
        except ValueError:
            pass
    return {"T": True, "F": False}.get(text, text)


def read_extxyz_frames(path: str):
    """
    Iterate over the frames of an extxyz file written by this module.

    Parameters
    ----------
    path : str
        The file.

    Yields
    ------
    dict
        ``positions`` (n, 3), ``species`` (n,), ``lattice`` (3, 3), ``pbc``
        (3,), ``info`` (dict) and ``arrays`` (dict of extra columns).
    """
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                return
            n = int(line.split()[0])
            raw = dict(_KV.findall(f.readline()))
            lattice = np.array(raw.pop("Lattice", '"0 0 0 0 0 0 0 0 0"').strip('"').split(),
                               dtype=float).reshape(3, 3)
            pbc = [p == "T" for p in raw.pop("pbc", '"F F F"').strip('"').split()]
            fields = raw.pop("Properties", "species:S:1:pos:R:3").split(":")
            cols = [(fields[i], fields[i + 1], int(fields[i + 2])) for i in range(0, len(fields), 3)]
            rows = [f.readline().split() for _ in range(n)]
            out, k = {}, 0
            for name, kind, width in cols:
                vals = [r[k:k + width] for r in rows]
                k += width
                if kind == "S":
                    out[name] = np.array([v[0] for v in vals], dtype=str)
                else:
                    arr = np.array(vals, dtype=int if kind == "I" else float)
                    out[name] = arr[:, 0] if width == 1 else arr
            yield {"positions": out.pop("pos").reshape(-1, 3), "species": out.pop("species"),
                   "lattice": lattice, "pbc": pbc,
                   "info": {k: _parse_value(v) for k, v in raw.items()}, "arrays": out}


def configuration_path(directory: str, key: Tuple[int, int]) -> str:
    """
    The configuration file of one grid point.

    Parameters
    ----------
    directory : str
        The configurations directory.
    key : tuple of int
        (temperature index, mu index).

    Returns
    -------
    str
        ``<directory>/configs_T{it}_mu{imu}.extxyz``.
    """
    import os
    return os.path.join(directory, f"configs_T{key[0]}_mu{key[1]}.extxyz")


def load_configurations(directory: str, key: Tuple[int, int]) -> Dict[int, dict]:
    """
    The stored configurations of one grid point, by sample index.

    Frames written after the last checkpoint of an interrupted run are
    rewritten when it resumes; the last frame of each sample index wins.

    Parameters
    ----------
    directory : str
        The configurations directory.
    key : tuple of int
        (temperature index, mu index).

    Returns
    -------
    dict
        ``{sample_index: frame}`` (frames as from `read_extxyz_frames`);
        empty if the file does not exist.
    """
    import os
    path = configuration_path(directory, key)
    if not os.path.isfile(path):
        return {}
    return {int(fr["info"]["sample"]): fr for fr in read_extxyz_frames(path)}
