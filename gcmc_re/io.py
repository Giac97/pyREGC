"""
Saving and loading REGC samples and checkpoints.

* `save_samples` / `load_samples` -- the recorded (U, N) of every replica,
  with the temperature ladder, the mu grid and the Lambda^3(T) convention,
  in one ``.npz`` file. `load_samples` returns a `SampleSet`, which
  `MBARSurfaceAnalysis` accepts in place of a live `ReplicaExchangeGCMC`, so
  a run can be analysed (and re-analysed) without re-running it.
* `atomic_pickle` / `load_pickle` -- crash-safe pickling for checkpoints
  (see `ReplicaExchangeGCMC.save_checkpoint` and `gcmc_re.driver`).
"""

from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    """

    temperatures: List[float]
    mu_grid: np.ndarray
    samples: Dict[Tuple[int, int], Dict[str, list]]
    lambda_cubed_of_T: Callable[[float], float] = _unit_lambda_cubed
    exchange_species: Optional[Tuple[str, ...]] = None

    @property
    def n_samples(self) -> int:
        """int: Samples per grid point."""
        return len(next(iter(self.samples.values()))['U'])

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
        k = int(fraction * self.n_samples)
        kept = {key: {name: list(vals[k:]) for name, vals in v.items()} for key, v in self.samples.items()}
        return SampleSet(list(self.temperatures), np.array(self.mu_grid), kept, self.lambda_cubed_of_T,
                         self.exchange_species)


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
    """
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
    tmp = path + ".tmp.npz"
    np.savez(tmp, **extra, U=np.array([sim.samples[k]['U'] for k in keys], dtype=float),
             N=np.array([sim.samples[k]['N'] for k in keys], dtype=float),
             keys=np.array(keys, dtype=int), temperatures=np.asarray(sim.temperatures, dtype=float),
             mu_grid=np.asarray(sim.mu_grid, dtype=float), lambda_kind=lam_kind,
             lambda_mass_amu=lam_mass,
             lambda_cubed_ladder=np.array([np.asarray(lam(T), dtype=float) for T in sim.temperatures]))
    os.replace(tmp, path)


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
    out = SampleSet(list(temperatures), mu_grid, samples, lambda_cubed_of_T, species)
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
    os.replace(tmp, path)


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
