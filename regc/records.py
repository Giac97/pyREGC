"""
Per-sample records: one HDF5 file holding every recorded sample with its
configuration and its energy under every evaluated potential.

This is the backbone for analysing samples under potentials other than the
one that generated them (`TASKS_proposal.md`, items B1-B3):

* during a run, `ReplicaExchangeGCMC(records_path=...)` appends one record per
  recorded sample: grid point, sample index, walker, N (and counts per
  species), the energy under the sampling potential, and the configuration
  (positions as one array with an offset and a length per sample, since the
  atom count varies);
* `reevaluate` computes the energies of the stored configurations with
  another calculator (resumable, in parallel) and stores them as
  ``E/<name>``;
* `attach_energies` adds those energies to a `SampleSet` as ``'E:<name>'``,
  aligned sample by sample, and `MBARSurfaceAnalysis.with_potential` reweights
  to the other potential.

Layout (``schema`` 1)::

    /samples/state         (n, 2) int32   grid point (it, imu)
    /samples/sample        (n,)   int64   index of the sample at its grid point
    /samples/walker        (n,)   int32   configuration id (replica-exchange walker)
    /samples/N             (n,)   int32   mobile atoms
    /samples/N_s           (n, S) int32   per exchanged species (several species only)
    /samples/E/<name>      (n,)   float64 energy under potential <name> (nan: not evaluated)
    /config/offset         (n,)   int64   first row in /config/positions (-1: not stored)
    /config/length         (n,)   int32   number of atoms
    /config/positions      (m, 3)         all atoms, in `full_configuration` order
    /config/species        (m,)   uint8   index into attribute ``species``
    attributes: schema, sampling_potential, cell (3x3), pbc, n_frozen, n_substrate,
                species (list), temperatures, mu_grid

Frames written after the last checkpoint of an interrupted run are written
again when it resumes; readers keep the last record of each (state, sample).

Requires h5py (``pip install regc[records]``), imported lazily.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 1


def _h5py():
    """
    Import h5py with a helpful message.

    Returns
    -------
    module
        h5py.

    Raises
    ------
    ImportError
        If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("per-sample records need h5py: pip install h5py "
                          "(or pip install regc[records])") from err
    except ValueError as err:  # pragma: no cover - library clash
        raise ImportError("h5py failed to load after another HDF5 library (e.g. LAMMPS's) "
                          "was loaded; import h5py (or regc.records) before LAMMPS starts") from err
    return h5py


class RecordWriter:
    """
    Append per-sample records to an HDF5 file.

    Records are queued by `add` and written by `flush` (called by the
    engine at checkpoints and at the end of every run call, and here
    every `FLUSH_EVERY` records, so memory stays bounded).

    Parameters
    ----------
    path : str
        The file (created if missing, appended to otherwise).
    sampling_potential : str
        Name under which the sampling energies are stored (``E/<name>``).
    species : sequence of str
        All atom labels that can occur (frozen, substrate and exchanged).
    lattice : numpy.ndarray
        Cell matrix, shape (3, 3).
    pbc : sequence of bool
        Periodicity.
    n_frozen, n_substrate : int
        Sizes of the first two atom groups (constant during a run).
    temperatures, mu_grid : array_like
        The run's ladder (stored for reference).
    n_species : int, optional
        Exchanged species (``/samples/N_s`` with more than one). Default 1.
    positions_dtype : str, optional
        ``'float64'`` (default, exact re-evaluation) or ``'float32'``
        (half the size; energies change by ~1e-6 relative).
    """

    FLUSH_EVERY = 2048

    def __init__(self, path: str, sampling_potential: str, species: Sequence[str],
                 lattice: np.ndarray, pbc: Sequence[bool], n_frozen: int, n_substrate: int,
                 temperatures, mu_grid, n_species: int = 1, positions_dtype: str = "float64"):
        """
        Open or create the file.

        Parameters
        ----------
        path : str
            See the class docstring (likewise below).
        sampling_potential : str
            Energy column of the sampling potential.
        species : sequence of str
            Atom labels.
        lattice : numpy.ndarray
            Cell matrix.
        pbc : sequence of bool
            Periodicity.
        n_frozen, n_substrate : int
            Group sizes.
        temperatures, mu_grid : array_like
            The ladder.
        n_species : int, optional
            Exchanged species.
        positions_dtype : str, optional
            Position precision.

        Raises
        ------
        ValueError
            If an existing file was written for another sampling potential
            or species set.
        """
        h5py = _h5py()
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = h5py.File(path, "a")
        a = self.f.attrs
        if "schema" in a:
            if a["sampling_potential"] != sampling_potential:
                raise ValueError(f"{path} records potential {a['sampling_potential']!r}, not "
                                 f"{sampling_potential!r}")
            if list(a["species"]) != list(map(str, species)):
                raise ValueError(f"{path} records species {list(a['species'])}, not {list(species)}")
        else:
            a.update(schema=SCHEMA, sampling_potential=sampling_potential,
                     species=[str(s) for s in species], cell=np.asarray(lattice, dtype=float),
                     pbc=np.asarray(pbc, dtype=bool), n_frozen=int(n_frozen),
                     n_substrate=int(n_substrate),
                     temperatures=np.asarray(temperatures, dtype=float),
                     mu_grid=np.asarray(mu_grid, dtype=float))
            g = self.f.create_group("samples")
            for name, shape, dt in (("state", (0, 2), "i4"), ("sample", (0,), "i8"),
                                    ("walker", (0,), "i4"), ("N", (0,), "i4")):
                g.create_dataset(name, shape=shape, maxshape=(None,) + shape[1:], dtype=dt,
                                 chunks=True)
            if n_species > 1:
                g.create_dataset("N_s", shape=(0, n_species), maxshape=(None, n_species),
                                 dtype="i4", chunks=True)
            g.create_group("E").create_dataset(sampling_potential, shape=(0,), maxshape=(None,),
                                               dtype="f8", chunks=True)
            c = self.f.create_group("config")
            c.create_dataset("offset", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
            c.create_dataset("length", shape=(0,), maxshape=(None,), dtype="i4", chunks=True)
            c.create_dataset("positions", shape=(0, 3), maxshape=(None, 3), dtype=positions_dtype,
                             chunks=(4096, 3))
            c.create_dataset("species", shape=(0,), maxshape=(None,), dtype="u1", chunks=(4096,))
        self.species_index = {s: i for i, s in enumerate(self.f.attrs["species"])}
        self.potential = sampling_potential
        self._pending: List[tuple] = []

    def add(self, key: Tuple[int, int], sample: int, walker: int, energy: float, n: int,
            config: Optional[tuple], n_s: Optional[Sequence[int]] = None) -> None:
        """
        Queue one record (written by `flush`).

        Parameters
        ----------
        key : tuple of int
            Grid point.
        sample : int
            Sample index at that grid point.
        walker : int
            Walker id (-1 if unknown).
        energy : float
            Energy under the sampling potential.
        n : int
            Mobile atoms.
        config : tuple or None
            `regc.trajectory.snapshot` of the configuration, or None (not
            stored for this sample).
        n_s : sequence of int, optional
            Counts per exchanged species.
        """
        self._pending.append((key, sample, walker, energy, n, config, n_s))
        if len(self._pending) >= self.FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """Write the queued records and flush the file to disk."""
        if not self._pending:
            return
        recs = self._pending
        self._pending = []
        g, c = self.f["samples"], self.f["config"]
        n0 = g["sample"].shape[0]
        k = len(recs)

        def append(ds, values):
            values = np.asarray(values)
            ds.resize(ds.shape[0] + len(values), axis=0)
            ds[-len(values):] = values
        append(g["state"], [r[0] for r in recs])
        append(g["sample"], [r[1] for r in recs])
        append(g["walker"], [r[2] for r in recs])
        append(g["N"], [r[4] for r in recs])
        if "N_s" in g:
            append(g["N_s"], [r[6] for r in recs])
        for name, ds in g["E"].items():
            vals = [r[3] if name == self.potential else np.nan for r in recs]
            append(ds, vals)
        start = c["positions"].shape[0]
        offsets, lengths, pos, spec = [], [], [], []
        for r in recs:
            cfg = r[5]
            if cfg is None:
                offsets.append(-1)
                lengths.append(0)
                continue
            offsets.append(start)
            lengths.append(len(cfg[1]))
            start += len(cfg[1])
            pos.append(np.asarray(cfg[0], dtype=float))
            for label in map(str, cfg[1]):
                if label not in self.species_index:  # a label the setup did not list
                    self.species_index[label] = len(self.species_index)
                    self.f.attrs["species"] = list(self.species_index)
            spec.append(np.array([self.species_index[str(s)] for s in cfg[1]], dtype=np.uint8))
        append(c["offset"], offsets)
        append(c["length"], lengths)
        if pos:
            append(c["positions"], np.concatenate(pos))
            append(c["species"], np.concatenate(spec))
        assert g["sample"].shape[0] == n0 + k
        self.f.flush()

    def close(self) -> None:
        """Flush and close the file."""
        self.flush()
        self.f.close()


class Records:
    """
    Read access to a records file (read-only unless `writable`).

    Parameters
    ----------
    path : str
        The file.
    writable : bool, optional
        Open for adding energy columns (`reevaluate`). Default False.
    """

    def __init__(self, path: str, writable: bool = False):
        """
        Open the file and index its records.

        Parameters
        ----------
        path, writable
            See the class docstring.
        """
        h5py = _h5py()
        self.path = path
        self.f = h5py.File(path, "r+" if writable else "r")
        g = self.f["samples"]
        self.state = g["state"][:]
        self.sample = g["sample"][:]
        # the last record of each (state, sample) wins (rewritten after a resume)
        latest = {}
        for i, (st, sm) in enumerate(zip(map(tuple, self.state), self.sample)):
            latest[(st, int(sm))] = i
        self.index = latest
        self.species = [str(s) for s in self.f.attrs["species"]]
        self.attrs = dict(self.f.attrs)

    def __len__(self) -> int:
        """
        Number of distinct (state, sample) records.

        Returns
        -------
        int
            The count (rewritten records counted once).
        """
        return len(self.index)

    @property
    def potentials(self) -> List[str]:
        """list of str: Names of the stored energy columns."""
        return list(self.f["samples/E"].keys())

    def energies(self, name: str) -> Dict[Tuple[Tuple[int, int], int], float]:
        """
        Energies under one potential.

        Parameters
        ----------
        name : str
            Potential name (``E/<name>``).

        Returns
        -------
        dict
            ``{((it, imu), sample): energy}`` (nan where not evaluated).
        """
        e = self.f[f"samples/E/{name}"][:]
        return {k: float(e[i]) for k, i in self.index.items()}

    def configuration(self, key: Tuple[int, int], sample: int) -> Optional[dict]:
        """
        One stored configuration.

        Parameters
        ----------
        key : tuple of int
            Grid point.
        sample : int
            Sample index there.

        Returns
        -------
        dict or None
            Frame-like dict (``positions``, ``species``, ``lattice``,
            ``pbc``, ``info``, ``arrays`` with ``group``), as from
            `regc.trajectory.read_extxyz_frames`; None if not stored.
        """
        i = self.index.get((tuple(key), int(sample)))
        if i is None:
            return None
        off = int(self.f["config/offset"][i])
        if off < 0:
            return None
        n = int(self.f["config/length"][i])
        pos = self.f["config/positions"][off:off + n].astype(float)
        sp = np.array([self.species[j] for j in self.f["config/species"][off:off + n]])
        nf, ns = int(self.attrs["n_frozen"]), int(self.attrs["n_substrate"])
        groups = np.concatenate([np.zeros(nf, int), np.ones(ns, int), np.full(n - nf - ns, 2, int)])
        info = {"sample": int(sample), "T_index": int(key[0]), "mu_index": int(key[1]),
                "N": int(self.f["samples/N"][i]),
                "U": float(self.f[f"samples/E/{self.attrs['sampling_potential']}"][i])}
        return {"positions": pos, "species": sp, "lattice": np.asarray(self.attrs["cell"]),
                "pbc": [bool(p) for p in self.attrs["pbc"]], "info": info,
                "arrays": {"group": groups}}

    def stored(self, key: Tuple[int, int]) -> set:
        """
        Sample indices of one grid point that have a stored configuration.

        Parameters
        ----------
        key : tuple of int
            Grid point.

        Returns
        -------
        set of int
            The sample indices.
        """
        off = self.f["config/offset"][:]
        return {s for (k, s), i in self.index.items() if k == tuple(key) and off[i] >= 0}

    def close(self) -> None:
        """Close the file."""
        self.f.close()


def attach_energies(samples, records_path: str, names: Optional[Sequence[str]] = None):
    """
    Add energies under other potentials to a sample set, sample by sample.

    Parameters
    ----------
    samples : SampleSet
        Loaded with `regc.io.load_samples` (any ``discard``).
    records_path : str
        The run's records file.
    names : sequence of str, optional
        Potentials to attach. Default None: every one except the sampling
        potential.

    Returns
    -------
    SampleSet
        The same set, with ``samples[key]['E:<name>']`` lists (nan where a
        sample has not been evaluated).

    Raises
    ------
    ValueError
        If a sample of the set is missing from the records.
    """
    from .io import first_index_of
    rec = Records(records_path)
    try:
        if names is None:
            names = [p for p in rec.potentials if p != rec.attrs["sampling_potential"]]
        for name in names:
            e = rec.energies(name)
            for key, v in samples.samples.items():
                grid_key = tuple(key[-2:])
                first = first_index_of(samples, key)
                vals = []
                for j in range(len(v["U"])):
                    s = first + j
                    if (grid_key, s) not in e:
                        raise ValueError(f"sample {s} of {grid_key} is not in {records_path}")
                    vals.append(e[(grid_key, s)])
                v["E:" + name] = vals
    finally:
        rec.close()
    return samples


def _energy_of_frame(system, frame: dict) -> float:
    """
    Total energy of a stored configuration, rebuilt in `system`.

    Parameters
    ----------
    system : ParticleSystem
        A system of the run's geometry (its calculator is used).
    frame : dict
        From `Records.configuration`.

    Returns
    -------
    float
        ``system.total_energy()`` with the frame's atoms.
    """
    groups = frame["arrays"]["group"]
    pos, sp = frame["positions"], frame["species"]
    system.frozen_positions, system.frozen_species = pos[groups == 0].copy(), sp[groups == 0].copy()
    system.substrate_positions = pos[groups == 1].copy()
    system.substrate_species = sp[groups == 1].copy()
    system.positions, system.species = pos[groups == 2].copy(), sp[groups == 2].copy()
    return float(system.total_energy())


_WORKER_SYSTEM = None


def _worker_init(make_system) -> None:
    """
    Build the worker's system once (pool initializer).

    Parameters
    ----------
    make_system : callable
        ``make_system(rng)``.
    """
    global _WORKER_SYSTEM
    _WORKER_SYSTEM = make_system(np.random.default_rng(0))


def _evaluate_frames(items) -> list:
    """
    Worker: energies of a list of configurations.

    Parameters
    ----------
    items : list of tuple
        ``[(row, frame), ...]``.

    Returns
    -------
    list of tuple
        ``[(row, energy), ...]``.
    """
    return [(row, _energy_of_frame(_WORKER_SYSTEM, fr)) for row, fr in items]


def reevaluate(records_path: str, make_system: Callable, name: str,
               subset: Optional[Callable[[Tuple[int, int], int], bool]] = None,
               n_workers: int = 1, chunk: int = 200,
               progress: Optional[Callable[[int, int], None]] = None) -> int:
    """
    Evaluate the stored configurations with another potential.

    Every stored configuration not yet evaluated under `name` is rebuilt in
    a system from `make_system` (same geometry, the new calculator) and its
    `total_energy` written to ``E/<name>``. The energies are flushed to disk
    after every `chunk` configurations, so an interrupted call loses at
    most one chunk and rerunning continues where it stopped. Use a
    `subset` that does not depend on the configurations (e.g. every k-th
    sample), so that reweighting over the evaluated samples stays unbiased.

    Parameters
    ----------
    records_path : str
        The records file (not open for writing elsewhere: HDF5 locks it).
    make_system : callable
        ``make_system(rng)``: a system of the run's geometry with the new
        calculator. Called once per worker (picklable for
        ``n_workers > 1``).
    name : str
        Name of the new potential.
    subset : callable, optional
        ``subset(key, sample) -> bool``: evaluate only these. Default None
        (all stored configurations).
    n_workers : int, optional
        Worker processes. Default 1.
    chunk : int, optional
        Configurations per task and between flushes. Default 200.
    progress : callable, optional
        Called with ``(done, total)`` after every chunk. Default None.

    Returns
    -------
    int
        Number of configurations evaluated in this call.

    Raises
    ------
    ValueError
        If `name` is the sampling potential.
    """
    rec = Records(records_path, writable=True)
    try:
        if name == rec.attrs["sampling_potential"]:
            raise ValueError(f"{name!r} is the sampling potential of {records_path}")
        g = rec.f["samples/E"]
        n_rows = rec.f["samples/sample"].shape[0]
        if name not in g:
            g.create_dataset(name, data=np.full(n_rows, np.nan), maxshape=(None,), chunks=True)
        ds = g[name]
        if ds.shape[0] < n_rows:  # samples recorded since the last evaluation
            old = ds.shape[0]
            ds.resize(n_rows, axis=0)
            ds[old:] = np.nan
        done = ds[:]
        off = rec.f["config/offset"][:]
        todo = sorted((i, k, s) for (k, s), i in rec.index.items()
                      if off[i] >= 0 and np.isnan(done[i]) and (subset is None or subset(k, s)))
        chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
        total, n_done = len(todo), 0

        def frames(c):
            return [(row, rec.configuration(k, s)) for row, k, s in c]

        def store(results):
            nonlocal n_done
            for row, e in results:
                ds[row] = e
            rec.f.flush()
            n_done += len(results)
            if progress is not None:
                progress(n_done, total)

        if n_workers <= 1 or len(chunks) <= 1:
            _worker_init(make_system)
            for c in chunks:
                store(_evaluate_frames(frames(c)))
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(n_workers, initializer=_worker_init,
                                     initargs=(make_system,)) as pool:
                pending = []
                for c in chunks:  # at most 2 chunks per worker in flight
                    pending.append(pool.submit(_evaluate_frames, frames(c)))
                    while len(pending) >= 2 * n_workers:
                        store(pending.pop(0).result())
                for fut in pending:
                    store(fut.result())
    finally:
        global _WORKER_SYSTEM
        _WORKER_SYSTEM = None
        rec.close()
    return n_done
