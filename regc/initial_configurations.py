"""
Start the replicas of a run from given configurations instead of the template.

A long run started from the bare template spends its first part building up
adsorbate layers, oxides or vacancies that an earlier run already reached.
`read_configurations` reads frames (by default the last one) of extended-XYZ
files -- for instance an earlier run's ``trajectories/replica_T*_mu*.xyz`` --
and `start_from_configurations` checks each against the run's slab (or
cluster), assigns them to the grid points and puts them into the replicas,
with energies recomputed by the replicas' own calculators.

A frame lists the atoms in `ParticleSystem.full_configuration` order, as
regc writes them: the frozen atoms (at the template positions), the
displaceable substrate (same count and species, anywhere), then the
exchangeable (mobile) particles of the exchanged species, any number --
fewer exchangeable slab atoms (removed Pd) and extra adsorbates are fine.

Only the initial condition changes: every move and swap keeps its
acceptance rule, so the stationary distribution is the same, and the
samples still need an equilibration period discarded.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

#: The assignment rules of `assign_configurations`.
ASSIGN_MODES = ("by_count", "cyclic")
#: Closest allowed distance (A) between atoms of a configuration without
#: a calculator hard core (``calculator.min_distance``).
DEFAULT_MIN_DISTANCE = 0.5


def _natural_key(path: str) -> list:
    """
    Sort key that orders embedded numbers by value (T2 before T10).

    Parameters
    ----------
    path : str
        A file name.

    Returns
    -------
    list
        Alternating text and integer parts.
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def expand_files(files: Union[str, Sequence[str]], base: Optional[str] = None) -> List[str]:
    """
    The configuration files named by a glob pattern or a list of paths/patterns.

    Parameters
    ----------
    files : str or sequence of str
        A path or glob pattern, or a list of them. Each pattern's matches
        are sorted naturally (``replica_T2_mu0`` before ``replica_T10_mu0``);
        the list order is kept.
    base : str, optional
        Directory relative paths are taken from. Default None: the
        working directory.

    Returns
    -------
    list of str
        The files, in order (a file named twice is kept once).

    Raises
    ------
    ValueError
        If `files` is empty or not strings, or a pattern matches no file.
    """
    entries = [files] if isinstance(files, str) else list(files or [])
    if not entries or not all(isinstance(e, str) and e for e in entries):
        raise ValueError(f"initial configurations: 'files' must be a path, a glob pattern or "
                         f"a list of them, got {files!r}")
    out: List[str] = []
    for entry in entries:
        path = entry if base is None or os.path.isabs(entry) else os.path.join(base, entry)
        matches = sorted(glob.glob(path), key=_natural_key) if glob.has_magic(path) else (
            [path] if os.path.isfile(path) else [])
        if not matches:
            raise ValueError(f"initial configurations: no file matches {entry!r}"
                             + (f" (looked for {path})" if path != entry else ""))
        out.extend(m for m in matches if m not in out)
    return out


def _sha256(path: str) -> str:
    """
    SHA-256 of a file's contents.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    str
        Hex digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_configurations(files: Sequence[str], index: int = -1) -> List[Dict[str, Any]]:
    """
    Read one frame of every configuration file.

    Parameters
    ----------
    files : sequence of str
        Extended-XYZ (or any ASE-readable) files, e.g. from `expand_files`.
    index : int, optional
        The frame of each file (Python indexing). Default -1: the last.

    Returns
    -------
    list of dict
        One per file: ``file``, ``index``, ``sha256``, ``positions``
        (n, 3), ``species`` (n,) and ``lattice`` (3, 3; zero when the file
        has no cell).

    Raises
    ------
    ValueError
        If a file cannot be read or has no such frame.
    """
    from ase.io import read
    out = []
    for path in files:
        try:
            atoms = read(path, index=int(index))
        except Exception as err:  # noqa: BLE001 - name the file whatever ASE raises
            raise ValueError(f"initial configuration {path}: cannot read frame {index}: "
                             f"{type(err).__name__}: {err}") from None
        out.append({"file": path, "index": int(index), "sha256": _sha256(path),
                    "positions": np.array(atoms.positions, dtype=float).reshape(-1, 3),
                    "species": np.array(atoms.get_chemical_symbols(), dtype=str),
                    "lattice": np.array(atoms.cell[:], dtype=float).reshape(3, 3)})
    return out


def check_configuration(system, positions: np.ndarray, species: Sequence[str],
                        lattice: Optional[np.ndarray] = None,
                        min_distance: float = DEFAULT_MIN_DISTANCE,
                        name: str = "configuration", cell_tol: float = 1e-3,
                        position_tol: float = 1e-3) -> Dict[str, np.ndarray]:
    """
    Check a configuration against a run's template system and split it.

    The atoms must come in `ParticleSystem.full_configuration` order:
    the frozen atoms of `system` (same species, positions within
    `position_tol`, minimum image), the displaceable substrate (same count,
    the same species except that semi-grand-canonical species may have
    swapped among themselves; positions free), then mobile particles of
    exchanged species only (any number). Every moving atom must lie where
    `system.external_energy` is finite (in the box), and no two atoms may
    be closer than `min_distance` (frozen pairs excepted).

    Parameters
    ----------
    system : ParticleSystem
        The template (a freshly built system of the run).
    positions : numpy.ndarray
        All atoms, shape (n, 3), in the system's coordinates (as regc
        writes trajectories: the top frozen layer at z = 0 for a slab).
    species : sequence of str
        Their species.
    lattice : numpy.ndarray, optional
        The frame's cell (3, 3). Its periodic rows (all rows for a system
        without periodicity) must match the system's within `cell_tol`.
        None, or all zero (plain XYZ): not checked.
    min_distance : float, optional
        Closest allowed distance (A). Default `DEFAULT_MIN_DISTANCE`.
    name : str, optional
        What to call the configuration in error messages (its file).
    cell_tol, position_tol : float, optional
        Tolerances (A) of the cell and the frozen positions. Default 1e-3.

    Returns
    -------
    dict
        ``substrate_positions``, ``substrate_species``, ``positions`` and
        ``species`` (the mobile particles, xy wrapped into the cell) and
        ``counts`` ({exchanged species: number of mobile particles}).

    Raises
    ------
    ValueError
        If the cell, the frozen atoms, the substrate atoms or the mobile
        species do not fit the system, an atom is outside the box, or two
        atoms are too close; the message names `name`.
    """
    from .trajectory import lattice_of
    pos = np.asarray(positions, dtype=float).reshape(-1, 3)
    sp = np.asarray(species).astype(str)
    if sp.shape[0] != pos.shape[0]:
        raise ValueError(f"{name}: {pos.shape[0]} positions but {sp.shape[0]} species")
    where = f"initial configuration {name}"
    if lattice is not None and np.any(np.asarray(lattice, dtype=float) != 0.0):
        mine, theirs = lattice_of(system), np.asarray(lattice, dtype=float).reshape(3, 3)
        rows = [k for k, periodic in enumerate(system.pbc) if periodic] or [0, 1, 2]
        if not np.allclose(mine[rows], theirs[rows], rtol=0.0, atol=cell_tol):
            raise ValueError(f"{where}: its cell {theirs[rows].round(4).tolist()} (periodic rows) "
                             f"is not the run's {mine[rows].round(4).tolist()}")
    n_f, n_s = system.frozen_positions.shape[0], system.substrate_positions.shape[0]
    if pos.shape[0] < n_f + n_s:
        raise ValueError(f"{where}: {pos.shape[0]} atoms, fewer than the slab's {n_f} frozen + "
                         f"{n_s} displaceable atoms")
    if n_f:
        if not np.array_equal(sp[:n_f], np.asarray(system.frozen_species).astype(str)):
            raise ValueError(f"{where}: its first {n_f} atoms are not the frozen atoms of the run "
                             f"(species differ): it was not written for this slab, or not in "
                             f"frozen / substrate / mobile order")
        d = system._min_image(pos[:n_f] - system.frozen_positions)
        worst = float(np.sqrt(np.einsum("ij,ij->i", d, d)).max())
        if worst > position_tol:
            raise ValueError(f"{where}: the frozen atoms are up to {worst:.3g} A from the run's "
                             f"template positions (tolerance {position_tol:g} A): another slab, "
                             f"or another z origin (regc puts the top frozen layer at z = 0)")
    sub_sp, sub_pos = sp[n_f:n_f + n_s], pos[n_f:n_f + n_s]
    template = np.asarray(system.substrate_species).astype(str)
    sgc = tuple(getattr(system, "sgc_species", ()) or ())
    same = (sub_sp == template) | (np.isin(sub_sp, sgc) & np.isin(template, sgc))
    if not np.all(same):
        k = int(np.flatnonzero(~same)[0])
        raise ValueError(f"{where}: displaceable substrate atom {k} (atom {n_f + k}) is {sub_sp[k]}, "
                         f"the run's is {template[k]}: wrong slab, wrong atom count or order")
    if sgc and getattr(system, "sgc_canonical", False):
        counts = [int(np.sum(sub_sp == s)) for s in sgc]
        if counts != [int(c) for c in system.sgc_counts()]:
            raise ValueError(f"{where}: canonical identity swaps keep the substrate composition "
                             f"{dict(zip(sgc, system.sgc_counts().tolist()))}, the frame has "
                             f"{dict(zip(sgc, counts))}")
    mob_sp, mob_pos = sp[n_f + n_s:], pos[n_f + n_s:]
    exchanged = tuple(system.exchange_species_list)
    foreign = sorted(set(mob_sp.tolist()) - set(exchanged))
    if foreign:
        raise ValueError(f"{where}: atoms {foreign} after the slab's {n_f + n_s} frozen and "
                         f"displaceable atoms are not exchanged species {list(exchanged)}")
    mob_pos = np.array([system.wrap_position(p) for p in mob_pos]).reshape(-1, 3)
    for label, block, off in (("displaceable", sub_pos, n_f), ("mobile", mob_pos, n_f + n_s)):
        for k, p in enumerate(block):
            if not np.isfinite(system.external_energy(p)):
                raise ValueError(f"{where}: {label} atom {off + k} ({sp[off + k]}) at "
                                 f"{np.round(p, 3).tolist()} is outside the box the run allows "
                                 f"(a slab: 0 < z <= Lz = {getattr(system, 'Lz', float('nan')):.3f} A)")
    full = np.vstack([pos[:n_f], sub_pos, mob_pos])
    for i in range(n_f, full.shape[0]):
        d = system._min_image(full - full[i])
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r[i] = np.inf
        j = int(np.argmin(r))
        if r[j] < min_distance:
            raise ValueError(f"{where}: atoms {min(i, j)} ({sp[min(i, j)]}) and {max(i, j)} "
                             f"({sp[max(i, j)]}) are {r[j]:.3f} A apart, closer than "
                             f"{min_distance:g} A")
    return {"substrate_positions": sub_pos.copy(), "substrate_species": sub_sp,
            "positions": mob_pos, "species": mob_sp,
            "counts": {s: int(np.sum(mob_sp == s)) for s in exchanged}}


def assign_configurations(counts: Sequence[float], shape: Tuple[int, int],
                          assign: str = "by_count") -> np.ndarray:
    """
    Which configuration each grid point starts from.

    ``cyclic``: grid point number s in row-major order ((T0, mu0), (T0,
    mu1), ..., (T1, mu0), ...) gets configuration s mod F (F
    configurations, in the given order).

    ``by_count``: the configurations are sorted by `counts` (ascending;
    ties keep the given order), and the grid points are visited column by
    column in increasing mu index (physical columns, then bias columns),
    within a column from the highest to the lowest temperature row (the
    coverage usually falls with T). Visit number s of R = n_T * n_mu gets
    sorted configuration floor(s F / R) when F <= R (each used about R / F
    times, neighbours sharing), else round(s (F - 1) / (R - 1)) (spread
    over the whole sorted list, both ends included). The count then never
    decreases along this order: low counts at low mu.

    Parameters
    ----------
    counts : sequence of float
        One number per configuration (e.g. its count of one species).
    shape : tuple of int
        The grid, (n_T, n_mu).
    assign : str, optional
        ``'by_count'`` (default) or ``'cyclic'``.

    Returns
    -------
    numpy.ndarray
        Configuration indices into `counts`, shape `shape`.

    Raises
    ------
    ValueError
        On an unknown `assign` or no configurations.
    """
    n_T, n_mu = (int(x) for x in shape)
    F, R = len(counts), n_T * n_mu
    if F == 0:
        raise ValueError("initial configurations: no configurations to assign")
    out = np.zeros((n_T, n_mu), dtype=int)
    if assign == "cyclic":
        for s, (it, imu) in enumerate(itertools.product(range(n_T), range(n_mu))):
            out[it, imu] = s % F
        return out
    if assign != "by_count":
        raise ValueError(f"initial configurations: assign must be one of {ASSIGN_MODES}, "
                         f"got {assign!r}")
    order = np.argsort(np.asarray(counts, dtype=float), kind="stable")
    slots = [(it, imu) for imu in range(n_mu) for it in reversed(range(n_T))]
    for s, (it, imu) in enumerate(slots):
        k = s * F // R if F <= R else (0 if R == 1 else int(round(s * (F - 1) / (R - 1))))
        out[it, imu] = order[k]
    return out


def start_from_configurations(sim, configurations: Sequence[Dict[str, Any]],
                              assign: str = "by_count", species: Optional[str] = None,
                              min_distance: float = DEFAULT_MIN_DISTANCE) -> Dict[str, Any]:
    """
    Put given configurations into the replicas of a freshly built simulation.

    Each configuration is checked against the template (`check_configuration`,
    with the system of the first grid point), assigned to grid points
    (`assign_configurations`), copied into the replicas' systems (substrate
    and mobile particles; the frozen atoms stay) and the replica energies
    are recomputed with each replica's own calculator. Call it before the
    first step; a resumed run takes its configurations from the checkpoint
    instead.

    Parameters
    ----------
    sim : ReplicaExchangeGCMC
        The simulation (replicas in the main process).
    configurations : sequence of dict
        From `read_configurations` (``file``, ``index``, ``sha256``,
        ``positions``, ``species``, ``lattice``).
    assign : str, optional
        ``'by_count'`` (default) or ``'cyclic'`` (`assign_configurations`).
    species : str, optional
        The species counted by ``by_count``. Default None: the first
        exchanged species.
    min_distance : float, optional
        Closest allowed distance between atoms (A). Default
        `DEFAULT_MIN_DISTANCE`.

    Returns
    -------
    dict
        JSON-able record: ``assign``, ``species``, ``configurations``
        (``file``, ``index``, ``sha256`` and ``counts`` of each) and
        ``assignment`` (n_T x n_mu configuration indices).

    Raises
    ------
    ValueError
        If a configuration does not fit the run (the message names its
        file), `species` is not exchanged, or a replica's energy is not
        finite after the replacement.
    """
    exchanged = list(sim.exchange_species)
    species = exchanged[0] if species is None else str(species)
    if species not in exchanged:
        raise ValueError(f"initial configurations: species {species!r} is not an exchanged "
                         f"species {exchanged}")
    keys = sorted(sim.grid)
    template = sim.grid[keys[0]].system
    checked = [check_configuration(template, c["positions"], c["species"], c.get("lattice"),
                                   min_distance=min_distance, name=str(c["file"]))
               for c in configurations]
    table = assign_configurations([c["counts"][species] for c in checked], sim.shape, assign)
    for key in keys:
        conf, src = checked[table[key]], configurations[table[key]]
        sysm = sim.grid[key].system
        sysm.substrate_positions = conf["substrate_positions"].copy()
        sysm.substrate_species = conf["substrate_species"].astype(sysm.substrate_species.dtype)
        sysm.positions = conf["positions"].copy()
        sysm.species = conf["species"].copy()
        energy = sysm.total_energy()
        if not np.isfinite(energy):
            raise ValueError(f"initial configuration {src['file']}: the energy of grid point "
                             f"{key} is {energy} (a hard core, or an atom outside the box)")
        sim.grid[key].energy = float(energy)
    return {"assign": assign, "species": species,
            "configurations": [{"file": str(c["file"]), "index": int(c.get("index", -1)),
                                "sha256": c.get("sha256"), "counts": k["counts"]}
                               for c, k in zip(configurations, checked)],
            "assignment": table.tolist()}
