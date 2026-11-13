"""
Saving and loading REGC samples and checkpoints.

* `save_samples` / `load_samples` -- the recorded (U, N) of every replica,
  with the temperature ladder, the mu grid and the Lambda^3(T) convention,
  in one ``.npz`` file. `load_samples` returns a `SampleSet`, which
  `MBARSurfaceAnalysis` accepts in place of a live `ReplicaExchangeGCMC`, so
  a run can be analysed (and re-analysed) without re-running it.
* `atomic_pickle` / `load_pickle` -- crash-safe pickling for checkpoints
  (see `ReplicaExchangeGCMC.save_checkpoint` and `regc.driver`).
"""

from __future__ import annotations

import json
import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .units import SpeciesLambdaCubed, ThermalWavelengthCubed


def _unit_lambda_cubed(T: float) -> float:
    """
    Lambda^3 = 1 at every temperature (the reduced-units convention).

    Parameters
    ----------
    T : float
        Temperature (unused).

    Returns
    -------
    float
        1.0.
    """
    return 1.0


@dataclass
class SampleSet:
    """
    Recorded REGC samples, detached from the simulation that produced them.

    Has exactly the attributes `MBARSurfaceAnalysis` reads from a
    `ReplicaExchangeGCMC`, so either can be analysed.

    Attributes
    ----------
    temperatures : list of float
        The temperature ladder (k_B*T in eV, or reduced units).
    mu_grid : numpy.ndarray
        Chemical potentials, shape (n_T, n_mu), or (n_T, n_mu, n_species)
        with several exchanged species.
    samples : dict
        ``{(it, imu): {'U': [...], 'N': [...]}}``, plus ``'N_ads'`` if
        the run recorded it (``adsorbed_cutoff``) and ``'N_s'`` (counts
        per species) with several exchanged species.
    lambda_cubed_of_T : callable
        Lambda^3 as a function of temperature (see `ThermalWavelengthCubed`,
        `SpeciesLambdaCubed`).
    exchange_species : tuple of str, optional
        The exchanged species, in the order of the last axis of `mu_grid`
        and of ``'N_s'``. None if unknown (files from older versions).
    observables_meta : dict, optional
        ``{name: metadata}`` of the observables recorded under
        ``samples[key]['obs:<name>']`` (see `regc.observables`).
    n_substrate : int, optional
        Displaceable substrate atoms per replica (kinetic part of the heat
        capacity). None if unknown (files from older versions).
    first_index : int or dict
        Index, in the run's own numbering, of the first sample kept per
        grid point (`discard` drops the earlier ones); maps samples to the
        stored configurations. A dict ``{key: index}`` when it differs
        between states (merged runs). Default 0.
    state_table : dict, optional
        ``{key: (T, mu)}``: the thermodynamic state of every key, for sets
        that are not one rectangular (temperature x mu) grid -- the union
        of several runs (`merge_samples`), whose keys are
        ``(run, it, imu)``. Default None: ``temperatures[key[0]]`` and
        ``mu_grid[key]``.
    run_labels : list of str, optional
        With `state_table` from `merge_samples`: a label per run (e.g. its
        output directory), indexed by ``key[0]``. Default None.
    """

    temperatures: List[float]
    mu_grid: np.ndarray
    samples: Dict[Tuple[int, int], Dict[str, list]]
    lambda_cubed_of_T: Callable[[float], float] = _unit_lambda_cubed
    exchange_species: Optional[Tuple[str, ...]] = None
    observables_meta: Optional[dict] = None
    n_substrate: Optional[int] = None
    first_index: Union[int, Dict] = 0
    state_table: Optional[Dict] = None
    run_labels: Optional[List[str]] = None

    @property
    def n_samples(self) -> int:
        """int: Samples per grid point (the smallest, if they differ)."""
        return min(len(v['U']) for v in self.samples.values())

    def discard(self, fraction: float) -> "SampleSet":
        """
        Drop the first part of every replica's samples (equilibration).

        Parameters
        ----------
        fraction : float
            Fraction in [0, 1) of each replica's samples to drop.

        Returns
        -------
        SampleSet
            A new set with the remaining samples.

        Raises
        ------
        ValueError
            If `fraction` is not in [0, 1).
        """
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"fraction must be in [0, 1), got {fraction}")
        drop = {key: int(fraction * len(v['U'])) for key, v in self.samples.items()}
        kept = {key: {name: list(vals[drop[key]:]) for name, vals in v.items()}
                for key, v in self.samples.items()}
        if isinstance(self.first_index, dict) or len(set(drop.values())) > 1:
            first = {key: first_index_of(self, key) + drop[key] for key in self.samples}
        else:
            first = self.first_index + next(iter(drop.values()))
        return SampleSet(list(self.temperatures), self.mu_grid if self.mu_grid is None
                         else np.array(self.mu_grid), kept, self.lambda_cubed_of_T,
                         self.exchange_species, self.observables_meta, self.n_substrate, first,
                         self.state_table, self.run_labels)


def state_of(source, key) -> Tuple[float, np.ndarray]:
    """
    The (T, mu) of one state of a simulation or sample set.

    Parameters
    ----------
    source : ReplicaExchangeGCMC or SampleSet
        Anything with ``samples``; ``state_table`` if not a grid.
    key : tuple
        The state's key.

    Returns
    -------
    tuple
        ``(T, mu)``, mu a float (one species) or an array (several).
    """
    table = getattr(source, "state_table", None)
    if table is not None:
        return table[key]
    return source.temperatures[key[0]], np.asarray(source.mu_grid)[key]


def first_index_of(source, key) -> int:
    """
    Run sample index of the first kept sample of one state.

    Parameters
    ----------
    source : ReplicaExchangeGCMC or SampleSet
        The samples.
    key : tuple
        The state's key.

    Returns
    -------
    int
        0 for a simulation; ``first_index`` (or its entry) for a set.
    """
    first = getattr(source, "first_index", 0)
    return int(first[key]) if isinstance(first, dict) else int(first)


def merge_samples(*sets, labels: Optional[Sequence[str]] = None) -> SampleSet:
    """
    Combine the samples of several runs of the same system into one set.

    MBAR needs only each sample's energy and particle numbers and every
    state's (T, mu), not a rectangular grid, so a run that fills the gaps
    of another (`regc.diagnostics.suggest_ladder`) is analysed together
    with it and no sample is thrown away. Keys become ``(run, it, imu)``
    and the states are listed in ``state_table``.

    The runs must sample the same ensemble: same exchanged species, same
    Lambda^3(T) convention, same number of displaceable substrate atoms
    (and, which this function cannot see, the same structure and
    potential -- `regc.runner` checks those from ``run_metadata.json``).
    Observables recorded by only some runs, or defined differently, are
    dropped with a warning.

    Parameters
    ----------
    *sets : SampleSet or ReplicaExchangeGCMC
        The runs (already with their equilibration discarded).
    labels : sequence of str, optional
        A label per run (e.g. its output directory), stored as
        ``run_labels``. Default None: ``'run0'``, ``'run1'``, ...

    Returns
    -------
    SampleSet
        The union, with ``state_table`` and a per-state ``first_index``.

    Raises
    ------
    ValueError
        With fewer than two sets, or if they are not compatible.
    """
    if len(sets) < 2:
        raise ValueError("merge_samples needs at least two runs")
    ref = sets[0]
    species = getattr(ref, "exchange_species", None)
    temps = sorted({float(state_of(s, k)[0]) for s in sets for k in s.samples})
    lam_ref = np.array([np.asarray(ref.lambda_cubed_of_T(T), dtype=float) for T in temps])
    for i, other in enumerate(sets[1:], start=1):
        if getattr(other, "exchange_species", None) != species:
            raise ValueError(f"run {i} exchanges {getattr(other, 'exchange_species', None)}, "
                             f"run 0 {species}")
        lam = np.array([np.asarray(other.lambda_cubed_of_T(T), dtype=float) for T in temps])
        if lam.shape != lam_ref.shape or not np.allclose(lam, lam_ref, rtol=1e-10):
            raise ValueError(f"run {i} uses another Lambda^3(T) convention than run 0")
        if getattr(other, "n_substrate", None) != getattr(ref, "n_substrate", None):
            raise ValueError(f"run {i} has {getattr(other, 'n_substrate', None)} displaceable "
                             f"substrate atoms, run 0 {getattr(ref, 'n_substrate', None)}")
    metas = [dict(getattr(s, "observables_meta", None) or {}) for s in sets]
    common = {name: m for name, m in metas[0].items() if all(mm.get(name) == m for mm in metas[1:])}
    dropped = sorted(set().union(*metas) - set(common))
    if dropped:
        warnings.warn(f"observables {dropped} are not recorded identically by every run: dropped "
                      f"from the merged set", stacklevel=2)
    samples, table, first = {}, {}, {}
    for r, src in enumerate(sets):
        for key, v in src.samples.items():
            T, mu = state_of(src, key)
            new = (r,) + tuple(key[-2:])
            samples[new] = {name: list(vals) for name, vals in v.items()
                            if not (name.startswith("obs:") and name[4:] not in common)}
            table[new] = (float(T), np.array(mu, dtype=float) if np.ndim(mu) else float(mu))
            first[new] = first_index_of(src, key)
    return SampleSet(temps, None, samples, ref.lambda_cubed_of_T, species, common or None,
                     getattr(ref, "n_substrate", None), first, table,
                     list(labels) if labels is not None else [f"run{r}" for r in range(len(sets))])


def save_samples(sim, path: str) -> None:
    """
    Write a simulation's (or `SampleSet`'s) samples to an ``.npz`` file.

    The write is atomic (temporary file + rename). The Lambda^3 convention
    is stored as the particle mass for a `ThermalWavelengthCubed` (the
    per-species masses for a `SpeciesLambdaCubed` of them), as "unit" for
    Lambda^3 = 1; any other callable is stored by its values on the ladder
    and must be passed again to `load_samples`.

    Parameters
    ----------
    sim : ReplicaExchangeGCMC or SampleSet
        Anything with ``temperatures``, ``mu_grid``, ``samples`` and
        ``lambda_cubed_of_T``.
    path : str
        Output file (``.npz``).

    Raises
    ------
    ValueError
        For a merged set (`merge_samples`): save the runs separately.
    """
    if getattr(sim, "state_table", None) is not None:
        raise ValueError("a merged sample set is not saved: save each run and merge them again "
                         "with merge_samples after loading")
    keys = sorted(sim.samples)
    lam = getattr(sim, 'lambda_cubed_of_T', _unit_lambda_cubed)
    if isinstance(lam, ThermalWavelengthCubed):
        lam_kind, lam_mass = "mass", lam.mass_amu
    elif isinstance(lam, SpeciesLambdaCubed) and lam.masses_amu is not None:
        lam_kind, lam_mass = "masses", np.array(lam.masses_amu, dtype=float)
    elif all(np.all(np.asarray(lam(T)) == 1.0) for T in sim.temperatures):
        lam_kind, lam_mass = "unit", np.nan
    else:
        lam_kind, lam_mass = "custom", np.nan
    extra = {}
    if all('N_ads' in v for v in sim.samples.values()):
        extra['N_ads'] = np.array([sim.samples[k]['N_ads'] for k in keys], dtype=float)
    if all('N_s' in v for v in sim.samples.values()):
        extra['N_s'] = np.array([sim.samples[k]['N_s'] for k in keys], dtype=float)
    species = getattr(sim, 'exchange_species', None)
    if species is not None:
        extra['exchange_species'] = np.array(list(species), dtype=str)
    meta = getattr(sim, 'observables_meta', None) or {}
    if meta:
        extra['observables_meta'] = np.array(json.dumps(meta))
        for name in meta:
            extra['obs:' + name] = np.array([sim.samples[k]['obs:' + name] for k in keys],
                                            dtype=float)
    n_substrate = getattr(sim, 'n_substrate', None)
    if n_substrate is not None:
        extra['n_substrate'] = np.array(int(n_substrate))
    extra['first_index'] = np.array(int(getattr(sim, 'first_index', 0)))
    tmp = path + ".tmp.npz"
    with open(tmp, "wb") as f:
        np.savez(f, **extra, U=np.array([sim.samples[k]['U'] for k in keys], dtype=float),
                 N=np.array([sim.samples[k]['N'] for k in keys], dtype=float),
                 keys=np.array(keys, dtype=int),
                 temperatures=np.asarray(sim.temperatures, dtype=float),
                 mu_grid=np.asarray(sim.mu_grid, dtype=float), lambda_kind=lam_kind,
                 lambda_mass_amu=lam_mass,
                 lambda_cubed_ladder=np.array([np.asarray(lam(T), dtype=float)
                                               for T in sim.temperatures]))
        _flush_to_disk(f)
    _durable_replace(tmp, path)


def load_samples(path: str, discard: float = 0.0,
                 lambda_cubed_of_T: Optional[Callable[[float], float]] = None,
                 temperatures: Optional[List[float]] = None,
                 mu_grid: Optional[np.ndarray] = None) -> SampleSet:
    """
    Read samples written by `save_samples`.

    Parameters
    ----------
    path : str
        The ``.npz`` file.
    discard : float, optional
        Fraction of each replica's samples to drop as equilibration.
        Default 0.0.
    lambda_cubed_of_T : callable, optional
        Lambda^3(T); needed only if the file was written with a custom
        callable, or holds only (U, N). Default None (use what the file
        records).
    temperatures : list of float, optional
        The temperature ladder; needed only for files holding only (U, N)
        (written before this format). Default None (from the file).
    mu_grid : numpy.ndarray, optional
        The mu grid, shape (n_T, n_mu) or (n_mu,) (with several species,
        (n_T, n_mu, n_species) or (n_mu, n_species)); as `temperatures`.

    Returns
    -------
    SampleSet
        The samples, ready for `MBARSurfaceAnalysis`.

    Raises
    ------
    ValueError
        If the file lacks the ladder/grid and they are not given.
    """
    d = np.load(path, allow_pickle=False)
    keys = [tuple(int(x) for x in k) for k in d["keys"]]
    samples = {k: {'U': list(d["U"][i]), 'N': list(d["N"][i])} for i, k in enumerate(keys)}
    if "N_ads" in d:
        for i, k in enumerate(keys):
            samples[k]['N_ads'] = list(d["N_ads"][i])
    if "N_s" in d:
        for i, k in enumerate(keys):
            samples[k]['N_s'] = d["N_s"][i].tolist()
    meta = json.loads(str(d["observables_meta"])) if "observables_meta" in d else None
    for name in meta or {}:
        for i, k in enumerate(keys):
            samples[k]['obs:' + name] = d['obs:' + name][i].tolist()
    n_substrate = int(d["n_substrate"]) if "n_substrate" in d else None
    first_index = int(d["first_index"]) if "first_index" in d else 0
    species = tuple(str(x) for x in d["exchange_species"]) if "exchange_species" in d else None
    kind = str(d["lambda_kind"]) if "lambda_kind" in d else "unit"
    if lambda_cubed_of_T is None:
        if kind == "mass":
            lambda_cubed_of_T = ThermalWavelengthCubed(float(d["lambda_mass_amu"]))
        elif kind == "masses":
            lambda_cubed_of_T = SpeciesLambdaCubed(
                [ThermalWavelengthCubed(float(m)) for m in d["lambda_mass_amu"]])
        elif kind == "unit":
            lambda_cubed_of_T = _unit_lambda_cubed
        else:
            warnings.warn(f"{path} was written with a custom Lambda^3(T); pass "
                          f"lambda_cubed_of_T= to reweight off the ladder. Using Lambda^3 = 1.")
            lambda_cubed_of_T = _unit_lambda_cubed
    if temperatures is None:
        if "temperatures" not in d:
            raise ValueError(f"{path} holds only (U, N): pass temperatures= and mu_grid=")
        temperatures = list(d["temperatures"])
    if mu_grid is None:
        if "mu_grid" not in d:
            raise ValueError(f"{path} holds only (U, N): pass temperatures= and mu_grid=")
        mu_grid = d["mu_grid"]
    mu_grid = np.asarray(mu_grid, dtype=float)
    several = species is not None and len(species) > 1
    if mu_grid.ndim == (2 if several else 1):
        mu_grid = np.tile(mu_grid, (len(temperatures),) + (1,) * mu_grid.ndim)
    out = SampleSet(list(temperatures), mu_grid, samples, lambda_cubed_of_T, species, meta,
                    n_substrate, first_index)
    return out.discard(discard) if discard else out


def atomic_pickle(obj: Any, path: str) -> None:
    """
    Pickle `obj` to `path` so that a crash never leaves a broken file.

    Parameters
    ----------
    obj : object
        Anything picklable.
    path : str
        Destination; written as ``path + '.tmp'`` then renamed over it.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        _flush_to_disk(f)
    _durable_replace(tmp, path)


def _flush_to_disk(f) -> None:
    """
    Flush an open file's buffers and fsync it.

    Parameters
    ----------
    f : file object
        Opened for writing.
    """
    f.flush()
    os.fsync(f.fileno())


def _durable_replace(tmp: str, path: str) -> None:
    """
    Rename `tmp` over `path` and fsync the directory, so the rename survives a crash.

    Parameters
    ----------
    tmp : str
        The fully written (and fsynced) temporary file.
    path : str
        Destination.
    """
    os.replace(tmp, path)
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some file systems do not support fsync on directories
    finally:
        os.close(fd)


def load_pickle(path: str) -> Any:
    """
    Unpickle a file written by `atomic_pickle`.

    Only load files you created: unpickling runs code.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    object
        The unpickled object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
