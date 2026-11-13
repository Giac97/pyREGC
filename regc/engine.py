"""
ReplicaExchangeGCMC: a 2D (temperature x chemical potential) ladder of
replicas, with a generalized GCMC exchange criterion and optional
process-pool parallelism across replicas.

Parallelization notes
----------------------
Between exchange rounds, every replica's local sweep only touches its own
configuration -- the classic "embarrassingly parallel" structure of replica
exchange. `n_workers > 1` runs these sweeps across a `ProcessPoolExecutor`.
The exchange step stays serial in the main process, but it is cheap: it
only ever needs four scalars (beta, mu, U, N) per replica to decide
accept/reject, and only moves a full configuration between processes when a
swap is *accepted*.

This changes only *which CPU core* does each replica's work and *when* --
the statistics are exactly the algorithm's, unaffected by execution order,
since every replica's random-number stream is independent of the others
regardless of scheduling. (Concretely: running the same seed with
n_workers=1 vs n_workers=4 gives bit-identical results.)

Speedup requires actual spare CPU cores and a per-replica cost that
dominates the pool's own overhead (a few ms of pickling + dispatch per
cycle) -- worthwhile once you're using an EAM or MLIP calculator, not
necessarily worthwhile for the fast native `LJCalculator`. If your
calculator's backend (NumPy/BLAS, or PyTorch for an MLIP) does its own
internal multithreading, remember to pin it to one thread per worker
process (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`torch.set_num_threads(1)`), or you'll oversubscribe your cores and lose
the speedup you're expecting.

For GPU-resident MLIPs, consider a persistent-worker architecture instead
(each worker builds its own model once, from a checkpoint path, and
thereafter exchanges only small messages) rather than re-pickling a live
calculator every cycle; for multi-node scaling, `mpi4py` with one rank per
replica is the standard approach. Neither is implemented here.
"""

from __future__ import annotations

import itertools
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .replica import Replica
from .state import ThermoState
from .systems import ParticleSystem
from .observables import Observable, check_observables, evaluate_observables
from .trajectory import snapshot
from .units import SpeciesLambdaCubed


def _call_worker(payload: Tuple[Replica, str, dict]) -> Replica:
    """
    Call one `Replica` method in a worker process.

    Parameters
    ----------
    payload : tuple
        ``(replica, method_name, kwargs)``.

    Returns
    -------
    Replica
        The same replica, mutated by the call, ready to be sent back to the
        main process.
    """
    replica, method_name, kwargs = payload
    getattr(replica, method_name)(**kwargs)
    return replica


# The four neighbour-swap patterns of [ZSG] Fig. 2 (see docs/REGC_algorithm.md):
# 'T' = vertical, 'mu' = horizontal, 'cross_left' = (l, m+1) <-> (l+1, m),
# 'cross_right' = (l, m) <-> (l+1, m+1).
SWAP_TYPES = ('T', 'mu', 'cross_left', 'cross_right')


class ReplicaExchangeGCMC:
    """
    Manages a 2D ladder of replicas over (temperature, chemical potential).
    `system_factory(rng) -> ParticleSystem` builds a fresh system for each
    grid point -- this is what lets the same exchange engine drive any
    geometry/potential combination: the exchange acceptance criterion only
    ever needs (beta, mu, U, N), never anything about the geometry or
    calculator in use.

    `mus` can be either:
      * 1D, length n_mu -- the same chemical potentials used at every
        temperature.
      * 2D, shape (n_T, n_mu) -- an explicit mu for every (T, mu) grid
        point (what a *pressure* ladder needs; see `pressure.py`).

    Samples for MBAR are recorded per grid point as plain (U, N) pairs after
    `burn_in_cycles`, every `record_every` cycles (see `analysis.py`).

    Several exchanged species (``system.exchange_species = ['O', 'Cl']``):
    each grid point then has one mu per species, so `mus` gets a trailing
    species axis -- shape (n_mu, n_species), shared across temperatures, or
    (n_T, n_mu, n_species). The ladder stays 2D: its second axis is a path
    through (mu_O, mu_Cl, ...) space, e.g. a gas feed of fixed composition
    at increasing pressure. `lambda_cubed_of_T` then returns one Lambda^3
    per species (pass a list of per-species callables, or a
    `SpeciesLambdaCubed`), and samples also record the per-species counts
    under ``'N_s'``. With one species everything is exactly as before.

    `lambda_cubed_of_T(T) -> float` sets each replica's thermal de Broglie
    wavelength cubed (see `ThermoState.lambda_cubed`); the default (always
    1.0) is correct for reduced LJ units and a no-op for existing code. Real-
    units setups (e.g. an ASE EMT/EAM/MLIP calculator) should pass the true
    value -- see `regc.units.thermal_wavelength_cubed_angstrom3`.

    `hmc_enabled=True` (with `hmc_leapfrog_steps`/`hmc_dt`) replaces every
    replica's single-particle random displacement move with Hybrid Monte
    Carlo -- collective, force-directed proposals instead of a blind random
    walk. Needs a Calculator with `supports_forces = True` (see
    `Replica`'s docstring for the full explanation, correctness guarantee,
    and current limitations).

    Call `.close()` (or use as a context manager) to shut down the worker
    pool when you're done -- otherwise the worker processes linger.
    """

    def __init__(self, temperatures: List[float], mus,
                 system_factory: Callable[[np.random.Generator], ParticleSystem],
                 seed: int = 0, max_disp: float = 0.2, n_workers: int = 1,
                 lambda_cubed_of_T: Callable[[float], float] = lambda T: 1.0,
                 hmc_enabled: bool = False, hmc_leapfrog_steps: int = 10,
                 hmc_dt: float = 0.005, adsorbed_cutoff: Optional[float] = None,
                 observables: Optional[Sequence[Observable]] = None,
                 configurations_dir: Optional[str] = None, configurations_every: int = 1,
                 skip_empty_swaps: bool = True, records_path: Optional[str] = None,
                 records_potential: str = "sampling", records_every: int = 1):
        """
        Construct the replica-exchange ladder.

        Parameters
        ----------
        temperatures : list of float
            The temperature ladder, length n_T.
        mus : array_like
            Either 1D (length n_mu, shared across all temperatures) or 2D
            (shape (n_T, n_mu), an explicit mu for every grid point --
            what a pressure ladder needs; see `pressure.py`). With several
            exchanged species, shape (n_mu, n_species) or (n_T, n_mu,
            n_species), species in ``system.exchange_species`` order.
        system_factory : callable
            A function ``system_factory(rng) -> ParticleSystem`` that
            builds a fresh system for each grid point, given that grid
            point's own random number generator.
        seed : int, optional
            Master random seed; each grid point's own RNG stream is
            derived from it deterministically. Default is 0.
        max_disp : float, optional
            Maximum single-particle displacement (per axis) for plain MC
            displacement moves, passed to every `Replica`. Default is 0.2.
        n_workers : int, optional
            Number of worker processes for parallel local sweeps. Default
            is 1 (serial, no process pool).
        lambda_cubed_of_T : callable or sequence of callable, optional
            A function ``lambda_cubed_of_T(T) -> float`` giving each
            replica's thermal de Broglie wavelength cubed (see
            `ThermoState.lambda_cubed`). Default always returns 1.0,
            correct for reduced LJ units. With several exchanged species
            it may return one value per species, or be a list of one
            callable per species (wrapped in `SpeciesLambdaCubed`).
        hmc_enabled : bool, optional
            If True, every replica uses Hybrid Monte Carlo instead of
            plain single-particle MC displacement (see `Replica`'s
            docstring). Default is False.
        hmc_leapfrog_steps : int, optional
            Number of leapfrog integration steps per HMC trial, passed to
            every `Replica`. Default is 10.
        hmc_dt : float, optional
            Leapfrog integration timestep for HMC, passed to every
            `Replica`. Default is 0.005.
        adsorbed_cutoff : float, optional
            If given, every sample also records
            ``system.n_adsorbed(adsorbed_cutoff)`` under ``'N_ads'``, so
            the adsorbed count can be reweighted like N (see
            `MBARSurfaceAnalysis`, ``observable='N_ads'``). Default None
            (not recorded).
        observables : sequence of regc.observables.Observable, optional
            Structural observables (coordination numbers, RDFs, ...)
            evaluated on every recorded sample and stored under
            ``samples[key]['obs:<name>']``, for MBAR reweighting like N.
            Default None.
        configurations_dir : str, optional
            If given, every `configurations_every`-th recorded sample's
            configuration is appended to
            ``<configurations_dir>/configs_T{it}_mu{imu}.extxyz``, tagged
            with its sample index, so structures can be picked out of the
            phase diagram afterwards
            (`MBARSurfaceAnalysis.representative_samples`,
            `regc.analysis.export_phase_configurations`). Default None
            (off); see also `save_configurations`.
        configurations_every : int, optional
            Store every this many samples' configurations. Default 1 (all).
        skip_empty_swaps : bool, optional
            In `run_regc`, draw the replica-exchange move only among the
            (pattern, offset) combinations that have at least one pair on
            this grid, instead of among all of them (on an n_T x 1 grid,
            six of the eight draws would otherwise attempt nothing). The
            choice does not depend on the configurations, so either way the
            sampling is exact; True wastes no RE steps. Default True; a run
            resumed from a checkpoint keeps the setting it was started with
            (False for checkpoints written before this option existed), so
            it continues bit-identically.
        records_path : str, optional
            If given, every recorded sample is appended to this HDF5 file
            (`regc.records`): grid point, sample index, walker, N, the
            energy under the sampling potential and (every
            `records_every`-th sample) the configuration, so the samples
            can later be re-evaluated with another potential
            (`regc.records.reevaluate`) and reweighted to it
            (`MBARSurfaceAnalysis.with_potential`). Needs h5py. Default
            None (off); see also `save_records`.
        records_potential : str, optional
            Name of the sampling potential in the records file. Default
            ``'sampling'``.
        records_every : int, optional
            Store the configuration of every this many samples (the energy
            and N of all of them). Default 1.

        Raises
        ------
        ValueError
            If `mus` doesn't have one of the shapes above (for the number
            of exchanged species the systems declare), if the systems
            don't all exchange the same species, or if `lambda_cubed_of_T`
            doesn't give one value (or one per species).
        """
        self.temperatures = list(temperatures)
        n_T = len(self.temperatures)
        mus_arr = np.asarray(mus, dtype=float)
        if not callable(lambda_cubed_of_T):
            lambda_cubed_of_T = SpeciesLambdaCubed(lambda_cubed_of_T)
        self.n_workers = n_workers
        self.lambda_cubed_of_T = lambda_cubed_of_T  # also read by MBARSurfaceAnalysis
        self.adsorbed_cutoff = adsorbed_cutoff
        self.skip_empty_swaps = bool(skip_empty_swaps)
        self.observables = check_observables(observables)
        self.save_configurations(configurations_dir, configurations_every)
        self._records = None
        self.save_records(records_path, records_potential, records_every)
        self.save_status(None)
        self._executor: Optional[ProcessPoolExecutor] = None

        # Build the systems first (same seeds, same order as before): they
        # tell how many species are exchanged, which fixes the shape of `mus`
        master_rng = np.random.default_rng(seed)
        rng = np.random.default_rng(master_rng.integers(0, 2 ** 62))
        built = [((0, 0), rng, system_factory(rng))]
        species = built[0][2].exchange_species_list
        self.exchange_species: Tuple[str, ...] = species
        self.n_species = len(species)
        self.mu_grid = self._parse_mus(mus_arr, n_T, self.n_species)
        self.shape = self.mu_grid.shape[:2]
        for it, imu in itertools.product(range(self.shape[0]), range(self.shape[1])):
            if (it, imu) != (0, 0):
                rng = np.random.default_rng(master_rng.integers(0, 2 ** 62))
                built.append(((it, imu), rng, system_factory(rng)))
        if any(sysm.exchange_species_list != species for _, _, sysm in built):
            raise ValueError("every system from system_factory must exchange the same species")

        self.grid: Dict[Tuple[int, int], Replica] = {}
        self.samples: Dict[Tuple[int, int], Dict[str, list]] = {}
        for (it, imu), rng, system in built:
            T = self.temperatures[it]
            if self.n_species == 1:
                mu, lam3 = self.mu_grid[it, imu], lambda_cubed_of_T(T)
            else:
                mu = self.mu_grid[it, imu].copy()
                lam3 = np.asarray(lambda_cubed_of_T(T), dtype=float)
                if lam3.ndim > 1 or lam3.size not in (1, self.n_species):
                    raise ValueError(f"lambda_cubed_of_T must give 1 or {self.n_species} "
                                     f"values, got shape {lam3.shape}")
                lam3 = np.broadcast_to(lam3, (self.n_species,)).copy()
            state = ThermoState(T, mu, label=(it, imu), lambda_cubed=lam3)
            self.grid[(it, imu)] = Replica(state, system, rng, max_disp=max_disp,
                                            hmc_enabled=hmc_enabled,
                                            hmc_leapfrog_steps=hmc_leapfrog_steps,
                                            hmc_dt=hmc_dt)
            self.samples[(it, imu)] = ({'U': [], 'N': []} if adsorbed_cutoff is None
                                       else {'U': [], 'N': [], 'N_ads': []})
            if self.n_species > 1:
                self.samples[(it, imu)]['N_s'] = []
            for o in self.observables:
                self.samples[(it, imu)]['obs:' + o.name] = []
        # drawn after the replicas' seeds, so their streams are unchanged by it
        self.rng = np.random.default_rng(master_rng.integers(0, 2 ** 62))
        self.swap_stats = {t: [0, 0] for t in SWAP_TYPES}  # type -> [attempts, accepts]
        self.convergence_history: List[float] = []
        self._init_walkers()
    @staticmethod
    def _parse_mus(mus_arr: np.ndarray, n_T: int, n_species: int) -> np.ndarray:
        """
        Expand the `mus` argument to one mu (vector) per grid point.

        Parameters
        ----------
        mus_arr : numpy.ndarray
            The `mus` given to the constructor.
        n_T : int
            Length of the temperature ladder.
        n_species : int
            Number of exchanged species.

        Returns
        -------
        numpy.ndarray
            Shape (n_T, n_mu) for one species, (n_T, n_mu, n_species) for
            several.

        Raises
        ------
        ValueError
            If `mus_arr` has none of the accepted shapes.
        """
        if n_species == 1:
            if mus_arr.ndim == 1:
                return np.tile(mus_arr, (n_T, 1))
            if mus_arr.ndim == 2:
                if mus_arr.shape[0] != n_T:
                    raise ValueError("mus.shape[0] must equal len(temperatures)")
                return mus_arr
            raise ValueError("mus must be 1D or 2D (n_T, n_mu)")
        if mus_arr.ndim == 2 and mus_arr.shape[1] == n_species:
            return np.tile(mus_arr, (n_T, 1, 1))
        if mus_arr.ndim == 3 and mus_arr.shape[0] == n_T and mus_arr.shape[2] == n_species:
            return mus_arr
        raise ValueError(f"with {n_species} exchanged species, mus must have shape "
                         f"(n_mu, {n_species}) or (n_T, n_mu, {n_species}); got {mus_arr.shape}")

    def __getitem__(self, index: Tuple[int, int]) -> Replica:
        """
        The replica at one grid point.

        Parameters
        ----------
        index : tuple of int
            ``(temperature_index, mu_index)``.

        Returns
        -------
        Replica
            ``self.grid[index]``.
        """
        return self.grid[index]

    def _get_executor(self) -> ProcessPoolExecutor:
        """
        Get (creating if necessary) the process pool executor.

        Returns
        -------
        concurrent.futures.ProcessPoolExecutor
            The lazily-created executor, with `self.n_workers` workers.
        """
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.n_workers)
        return self._executor

    def close(self) -> None:
        """
        Shut down the worker pool, if one was created.

        Returns
        -------
        None
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self._records is not None:
            self._records.close()
            self._records = None

    def __enter__(self) -> "ReplicaExchangeGCMC":
        """
        Enter the context manager.

        Returns
        -------
        ReplicaExchangeGCMC
            `self`.
        """
        return self

    def __exit__(self, *exc) -> None:
        """
        Exit the context manager, shutting down the worker pool.

        Parameters
        ----------
        *exc
            The exception type, value, and traceback, if the context
            block raised (unused -- `close()` is called unconditionally).

        Returns
        -------
        None
        """
        self.close()

    def _local_sweep(self, n_steps: int, p_displace: float = 0.5,
                      p_insert: float = 0.25, p_delete: float = 0.25,
                      p_displace_substrate: float = 0.0) -> None:
        """
        Run one local sweep on every replica, serially or in parallel.

        Parameters
        ----------
        n_steps : int
            Number of moves each replica should attempt.
        p_displace : float, optional
            Relative probability of a displacement move. Default is 0.5.
        p_insert : float, optional
            Relative probability of a GCMC insertion move. Default is
            0.25.
        p_delete : float, optional
            Relative probability of a GCMC deletion move. Default is
            0.25.
        p_displace_substrate : float, optional
            Relative probability of an independent substrate displacement
            move. Default is 0.0 (off).

        Returns
        -------
        None
        """
        self._local_call('sweep', n_steps=n_steps, p_displace=p_displace, p_insert=p_insert,
                         p_delete=p_delete, p_displace_substrate=p_displace_substrate)

    def _local_call(self, method_name: str, **kwargs) -> None:
        """
        Call one `Replica` method on every replica, serially or in parallel.

        Parameters
        ----------
        method_name : str
            Name of the `Replica` method to call (e.g. ``'gc_attempt'``,
            ``'diffuse'``).
        **kwargs
            Keyword arguments passed to that method.

        Returns
        -------
        None
        """
        if self.n_workers <= 1:
            for replica in self.grid.values():
                getattr(replica, method_name)(**kwargs)
            return
        executor = self._get_executor()
        keys = list(self.grid.keys())
        payloads = [(self.grid[k], method_name, kwargs) for k in keys]
        for k, updated_replica in zip(keys, executor.map(_call_worker, payloads)):
            self.grid[k] = updated_replica

    def _attempt_exchange(self, key_a: Tuple[int, int], key_b: Tuple[int, int]) -> bool:
        """
        Attempt one replica-exchange swap between two grid points.

        Parameters
        ----------
        key_a : tuple of int
            Grid coordinates (temperature_index, mu_index) of the first
            replica.
        key_b : tuple of int
            Grid coordinates of the second replica.

        Returns
        -------
        bool
            True if the swap was accepted (in which case `key_a` and
            `key_b`'s configurations and energies have already been
            exchanged, in place); False otherwise.
        """
        #ra, rb = self.grid[key_a], self.grid[key_b]
        ra, rb = self[key_a], self[key_b]
        beta_a, beta_b = ra.state.beta, rb.state.beta
        mu_a, mu_b = ra.state.mu, rb.state.mu
        Ua, Ub = ra.energy, rb.energy
        if self.n_species == 1:
            Na, Nb = ra.system.n, rb.system.n
            # log of the swap acceptance ratio (see docs/REGC_algorithm.md, Eq. (8)).
            # The Lambda^3 prefactor uses the same lambda_cubed the insert/delete
            # moves use, so it reduces to the paper's (beta_a/beta_b)^(3/2 dN)
            # for a physical Lambda(T) and to 1 for the reduced-units Lambda = 1.
            log_acc = ((beta_a - beta_b) * (Ua - Ub)
                       - (beta_a * mu_a - beta_b * mu_b) * (Na - Nb)
                       + (Na - Nb) * np.log(ra.state.lambda_cubed / rb.state.lambda_cubed))
        else:
            # Eq. (8) summed over species (docs/REGC_algorithm.md, Sec. 8)
            dN = ra.system.species_counts() - rb.system.species_counts()
            log_acc = float((beta_a - beta_b) * (Ua - Ub)
                            - np.dot(beta_a * ra.state.mu_vec - beta_b * rb.state.mu_vec, dN)
                            + np.dot(dN, np.log(ra.state.lambda_cubed_vec
                                                / rb.state.lambda_cubed_vec)))

        ra.exchange_attempts += 1
        rb.exchange_attempts += 1
        accept = bool(log_acc >= 0.0 or ra.rng.random() < np.exp(log_acc))
        if accept:
            ra.system, rb.system = rb.system, ra.system
            ra.energy, rb.energy = rb.energy, ra.energy
            ra.exchange_accepts += 1
            rb.exchange_accepts += 1
        return accept

    def swap_pairs(self, swap_type: str, parity: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        List the disjoint replica pairs of one swap pattern ([ZSG] Fig. 2).

        Parameters
        ----------
        swap_type : str
            One of `SWAP_TYPES`: ``'T'`` pairs (l, m)-(l+1, m); ``'mu'``
            pairs (l, m)-(l, m+1); ``'cross_left'`` pairs (l, m+1)-(l+1, m);
            ``'cross_right'`` pairs (l, m)-(l+1, m+1).
        parity : int
            0 or 1: the offset of the first pair along the axis being
            paired (T for 'T' and both diagonals, mu for 'mu'). Alternating
            it is what connects e.g. T_2 with T_3 -- with a single offset the
            ladder would split into disconnected blocks.

        Returns
        -------
        list of tuple
            ``(key_a, key_b)`` pairs; no grid point appears more than once.

        Raises
        ------
        ValueError
            If `swap_type` is not one of `SWAP_TYPES`.
        """
        n_T, n_mu = self.shape
        if swap_type == 'T':
            return [((l, m), (l + 1, m)) for m in range(n_mu) for l in range(parity, n_T - 1, 2)]
        if swap_type == 'mu':
            return [((l, m), (l, m + 1)) for l in range(n_T) for m in range(parity, n_mu - 1, 2)]
        if swap_type == 'cross_left':
            return [((l, m + 1), (l + 1, m)) for l in range(parity, n_T - 1, 2) for m in range(n_mu - 1)]
        if swap_type == 'cross_right':
            return [((l, m), (l + 1, m + 1)) for l in range(parity, n_T - 1, 2) for m in range(n_mu - 1)]
        raise ValueError(f"swap_type must be one of {SWAP_TYPES}, got {swap_type!r}")

    def _exchange_round(self, swap_type: str, parity: int) -> None:
        """
        Attempt every swap of one pattern (see `swap_pairs`).

        Parameters
        ----------
        swap_type : str
            One of `SWAP_TYPES`.
        parity : int
            0 or 1, see `swap_pairs`.

        Returns
        -------
        None
        """
        stats = self.swap_stats[swap_type]
        for key_a, key_b in self.swap_pairs(swap_type, parity):
            accepted = self._attempt_exchange(key_a, key_b)
            stats[0] += 1
            stats[1] += int(accepted)
            pair = self.pair_stats.setdefault((key_a, key_b), [0, 0])
            pair[0] += 1
            pair[1] += int(accepted)
        self._track_walkers()

    # -- replica-exchange mixing diagnostics (bookkeeping only: no random numbers) --------
    def _init_walkers(self) -> None:
        """
        Label every configuration with a walker id and reset the mixing statistics.

        A walker is one configuration followed as replica exchange moves it
        across the grid (`ParticleSystem` objects travel with accepted swaps).
        """
        self.pair_stats: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[int]] = {}
        for k, key in enumerate(sorted(self.grid)):
            self.grid[key].system.walker_id = k
        n = len(self.grid)
        self.walker_last_end = {"T": [None] * n, "mu": [None] * n}
        self.round_trips = {"T": [0] * n, "mu": [0] * n}
        self._track_walkers()

    def _track_walkers(self) -> None:
        """
        Update the round-trip counts after an exchange round.

        A round trip along an axis is a walker going from one end of the
        ladder (index 0) to the other and back. It is counted when the
        walker reaches the end opposite to the last one it visited, and a
        full round trip is two such traversals.
        """
        n_axis = {"T": self.mu_grid.shape[0], "mu": self.mu_grid.shape[1]}
        for key, r in self.grid.items():
            w = getattr(r.system, "walker_id", None)
            if w is None:
                continue
            for axis, pos in (("T", key[0]), ("mu", key[1])):
                if n_axis[axis] < 2:
                    continue
                end = "low" if pos == 0 else "high" if pos == n_axis[axis] - 1 else None
                last = self.walker_last_end[axis][w]
                if end is not None and end != last:
                    if last is not None:
                        self.round_trips[axis][w] += 1   # one traversal = half a round trip
                    self.walker_last_end[axis][w] = end

    def check_energies(self, tol: float = 1e-6) -> Dict[Tuple[int, int], Tuple[float, bool]]:
        """
        Check every replica's running energy against a full recomputation.

        See `Replica.check_energy`; runs in the worker processes of a
        parallel run. `regc.driver.run_with_checkpoints` calls it after
        every chunk.

        Parameters
        ----------
        tol : float, optional
            Tolerance per atom. Default 1e-6.

        Returns
        -------
        dict
            ``{key: (drift, resynced)}``.
        """
        self._local_call("check_energy", tol=tol)
        return {k: (r.last_energy_drift, r.energy_resynced) for k, r in self.grid.items()}

    def pair_acceptance(self) -> Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, float]]:
        """
        Swap attempts and acceptance ratio of every neighbouring pair.

        A pattern's average can hide a single pair that never swaps and so
        cuts the ladder in two; this lists them one by one.

        Returns
        -------
        dict
            ``{(key_a, key_b): (attempts, acceptance)}``.
        """
        return {pair: (att, acc / max(1, att)) for pair, (att, acc) in self.pair_stats.items()}

    def round_trip_counts(self) -> Dict[str, np.ndarray]:
        """
        Completed round trips of every walker along T and along mu.

        A walker that never completes a round trip along T has not carried
        a low-temperature configuration through the hot replicas and back:
        the low-T replicas may still remember their start.

        Returns
        -------
        dict
            ``{'T': array, 'mu': array}``, one entry per walker (completed
            end-to-end-and-back trips).
        """
        return {axis: np.array(v) // 2 for axis, v in self.round_trips.items()}

    def run(self, n_cycles: int, sweeps_per_cycle: int = 200, exchange_every: int = 1,
            record_every: int = 1, burn_in_cycles: int = 0, verbose: bool = True,
            trajectory_dir: Optional[str] = None, trajectory_every: Optional[int] = None,
            p_displace: float = 0.5, p_insert: float = 0.25, p_delete: float = 0.25,
            p_displace_substrate: float = 0.0, diagonal_swaps: bool = True
            ) -> None:
        """
        Run the replica-exchange simulation.

        `trajectory_dir`, if given, saves each replica's configuration
        (positions + species, as an extxyz frame -- see trajectory.py) to
        its own file under that directory every `trajectory_every` cycles
        (defaulting to `record_every`). One file per grid point:
        `<trajectory_dir>/replica_T{it}_mu{imu}.xyz`. Requires the
        system's geometry to define a `cell` (`BulkSystem` and
        `SlabSystem` both do; pip install ase is also needed, for
        writing the file itself).

        `p_displace`/`p_insert`/`p_delete`/`p_displace_substrate` set the
        relative probability of each move type per step (renormalized to
        sum to 1, same as `Replica.sweep`); `p_displace_substrate` only
        does anything if the system actually has substrate atoms (see
        `Replica._move_displace_substrate`) and is independent of whatever
        substrate motion already happens via HMC's joint trajectory when
        `hmc_enabled=True` was passed at construction.

        Parameters
        ----------
        n_cycles : int
            Total number of cycles to run.
        sweeps_per_cycle : int, optional
            Number of Monte Carlo moves per replica per cycle. Default is
            200.
        exchange_every : int, optional
            Attempt a full round of replica exchanges every this many
            cycles. Default is 1 (every cycle).
        record_every : int, optional
            Record an (U, N) sample per grid point every this many cycles
            (after `burn_in_cycles`). Default is 1 (every cycle).
        burn_in_cycles : int, optional
            Number of initial cycles to run without recording samples.
            Default is 0.
        verbose : bool, optional
            If True, print progress roughly ten times over the run.
            Default is True.
        trajectory_dir : str, optional
            Directory to save trajectory frames to; if None, no
            trajectories are saved. Default is None.
        trajectory_every : int, optional
            Save a trajectory frame every this many cycles. Default is
            None, which uses `record_every` if `trajectory_dir` is given.
        p_displace : float, optional
            Relative probability of a displacement move. Default is 0.5.
        p_insert : float, optional
            Relative probability of a GCMC insertion move. Default is
            0.25.
        p_delete : float, optional
            Relative probability of a GCMC deletion move. Default is
            0.25.
        p_displace_substrate : float, optional
            Relative probability of an independent substrate displacement
            move. Default is 0.0 (off).
        diagonal_swaps : bool, optional
            If True, each exchange round also sweeps the two diagonal
            patterns ('cross_left', 'cross_right' of [ZSG] Fig. 2) after
            the T and mu ones. Default is True.

        Returns
        -------
        None
        """
        if trajectory_dir is not None and trajectory_every is None:
            trajectory_every = record_every

        for cycle in range(1, n_cycles + 1):
            self._local_sweep(sweeps_per_cycle, p_displace, p_insert, p_delete,
                               p_displace_substrate)

            if cycle % exchange_every == 0:
                for swap_type in (SWAP_TYPES if diagonal_swaps else SWAP_TYPES[:2]):
                    for parity in (0, 1):
                        self._exchange_round(swap_type, parity)

            if cycle > burn_in_cycles and cycle % record_every == 0:
                for key, r in self.grid.items():
                    self.samples[key]['U'].append(r.energy)
                    self.samples[key]['N'].append(r.system.n)
                    if self.adsorbed_cutoff is not None:
                        self.samples[key]['N_ads'].append(r.system.n_adsorbed(self.adsorbed_cutoff))
                    if self.n_species > 1:
                        self.samples[key]['N_s'].append(r.system.species_counts().tolist())
                    for name, v in evaluate_observables(self.observables, r.system).items():
                        self.samples[key]['obs:' + name].append(v)
                    if self.configurations_dir is not None or self.records_path is not None:
                        sample = len(self.samples[key]['U']) - 1
                        config = snapshot(r.system)
                        if self.configurations_dir is not None:
                            self._store_configuration(key, sample, r.energy, config)
                        self._record(key, sample, r.energy, r.system.n,
                                     self.samples[key]['N_s'][-1] if self.n_species > 1 else None,
                                     config)
            if trajectory_dir is not None and cycle % trajectory_every == 0:
                for key, r in self.grid.items():
                    self._save_trajectory_frame(trajectory_dir, key, r)

            if verbose and (cycle % max(1, n_cycles // 10) == 0 or cycle == n_cycles):
                mean_n = np.mean([r.system.n for r in self.grid.values()])
                print(f"cycle {cycle:6d}/{n_cycles} | mean N across grid = {mean_n:6.2f}")
            self._maybe_write_status(cycle, n_cycles, "sweep", force=cycle == n_cycles)
        self._flush_records()
    def run_regc(self, n_steps: int, x0: float = 0.99, diffusion: str = 'mc',
                 diffusion_steps: int = 10, record_every: int = 100, burn_in_steps: int = 0,
                 swap_types: Tuple[str, ...] = SWAP_TYPES,
                 convergence_every: Optional[int] = None, convergence_tol: float = 0.05,
                 convergence_bins: int = 30, verbose: bool = True,
                 trajectory_dir: Optional[str] = None, trajectory_every: Optional[int] = None,
                 **diffusion_kwargs) -> bool:
        """
        Run the REGC scheme exactly as in [ZSG] Sec. II A / Fig. 1.

        One REGC step is:

        1. Draw ``rand`` uniform in [0, 1). If ``rand < x0``, every replica
           makes ONE particle insertion/removal attempt (removal or
           insertion with probability 1/2 each, Eqs. (4)-(5) of
           docs/REGC_algorithm.md). Otherwise make ONE replica-exchange
           move: draw one pattern from `swap_types` and an offset (0/1),
           each uniformly (with ``skip_empty_swaps``: one uniformly among
           the combinations that have pairs on this grid), and attempt all
           of that pattern's disjoint swaps (Eq. (8)).
        2. Diffusion at fixed (N, V, T), in parallel over replicas (see
           `Replica.diffuse`).
        3. Every `record_every` steps after `burn_in_steps`, record (U, N)
           of every replica -- i.e. after the last diffusion move, before
           the next GC/RE step.
        4. Every `convergence_every` steps, test convergence of the
           reduced-energy densities rho(u): compare the samples recorded
           since the previous check with those of the interval before it
           (disjoint blocks, see `rho_u_change`); stop if they agree,
           otherwise continue up to `n_steps`.

        `x0` follows the paper's TEXT: it is the probability of a GC
        attempt ([ZSG]: x0 = 0.99 with "swaps attempted every 100 REGC
        steps"; x0 = 0.9 for the Si clusters). [ZSG] Fig. 1 draws the
        opposite branch labels (rand <= x0 -> replica exchange); to use
        that convention pass ``1 - x0``.

        Parameters
        ----------
        n_steps : int
            Maximum number of REGC steps.
        x0 : float, optional
            Probability of a particle insertion/removal step (vs. a replica
            exchange step). Default is 0.99, as in [ZSG] Sec. III A.
        diffusion : str, optional
            ``'mc_moves'``, ``'mc'``, ``'hmc'`` or ``'md'``; see
            `Replica.diffuse`. Default is ``'mc'``.
        diffusion_steps : int, optional
            MC moves, MC cycles, HMC trajectories, or MD steps per REGC
            step. Default is 10 ([ZSG]: "about ten"; for their LJ surface,
            ten MC moves, i.e. ``diffusion='mc_moves'``).
        record_every : int, optional
            Record a sample every this many REGC steps. Default is 100.
        burn_in_steps : int, optional
            Steps before recording starts. Default is 0.
        swap_types : tuple of str, optional
            Swap patterns to draw from. Default is all four of `SWAP_TYPES`
            (including the diagonals).
        convergence_every : int, optional
            Check convergence every this many steps (rounded to a multiple
            of `record_every`). Default is None (no check; run `n_steps`).
        convergence_tol : float, optional
            Converged when the largest total-variation distance between the
            rho(u) of the two latest inter-check blocks, over all grid
            points, is below this. Default is 0.05. Two blocks of n samples
            from the same distribution still differ by roughly
            sqrt(convergence_bins / (pi n)) from sampling noise alone, so the
            tolerance must lie above that floor (fewer bins or longer blocks).
            The first verdict needs three checks.
        convergence_bins : int, optional
            Histogram bins for rho(u). Default is 30.
        verbose : bool, optional
            Print progress about ten times over the run. Default is True.
        trajectory_dir : str, optional
            As in `run`. Default is None.
        trajectory_every : int, optional
            As in `run`; defaults to `record_every`.
        **diffusion_kwargs
            Passed on to `Replica.diffuse` (e.g. ``timestep_fs``,
            ``thermostat``, ``taut_fs`` for MD; ``displace_substrate`` for
            MC).

        Returns
        -------
        bool
            True if the convergence criterion was met before `n_steps`.

        Raises
        ------
        ValueError
            If `x0` is outside [0, 1] or `swap_types` contains an unknown
            pattern.
        """
        if not 0.0 <= x0 <= 1.0:
            raise ValueError(f"x0 must be in [0, 1], got {x0}")
        for t in swap_types:
            if t not in SWAP_TYPES:
                raise ValueError(f"unknown swap type {t!r}; choose from {SWAP_TYPES}")
        if trajectory_dir is not None and trajectory_every is None:
            trajectory_every = record_every
        if convergence_every is not None:
            convergence_every = max(record_every, (convergence_every // record_every) * record_every)
        n_at_last_check = n_at_check_before = None

        def needs_main_process(step: int) -> bool:
            # steps after which the main process must act on the replicas
            return ((trajectory_dir is not None and step % trajectory_every == 0)
                    or (verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps))
                    or (convergence_every is not None and step > burn_in_steps
                        and step % convergence_every == 0))

        usable = [(t, par) for t in swap_types for par in (0, 1) if self.swap_pairs(t, par)]

        def draw_swap() -> Tuple[str, int]:
            if self.skip_empty_swaps and usable:  # one draw, configuration-independent
                return usable[int(self.rng.integers(len(usable)))]
            swap_type = swap_types[int(self.rng.integers(len(swap_types)))]
            return swap_type, int(self.rng.integers(2))

        # Steps are executed in blocks: a block starts with (at most) one RE
        # move and continues with GC steps until the next RE draw or a step
        # needing the main process. Between RE moves replicas are independent,
        # so each runs its whole block in one call; the random draws and their
        # order are exactly those of a step-by-step loop.
        step = 0
        print_every = max(1, n_steps // 10)
        next_print = print_every
        pending_swap: Optional[Tuple[str, int]] = None
        while step < n_steps:
            first = step + 1
            if pending_swap is None:
                do_gc = bool(self.rng.random() < x0)
                if not do_gc:
                    pending_swap = draw_swap()
            else:
                do_gc = False
            if pending_swap is not None:
                self._exchange_round(*pending_swap)
                pending_swap = None
            flags = [do_gc]
            while first + len(flags) - 1 < n_steps and not needs_main_process(first + len(flags) - 1):
                if self.rng.random() < x0:
                    flags.append(True)
                else:
                    pending_swap = draw_swap()  # RE opens the next block
                    break
            self._local_call('regc_local_block', gc_flags=flags, first_step=first,
                             record_every=record_every, burn_in_steps=burn_in_steps,
                             adsorbed_cutoff=self.adsorbed_cutoff, observables=self.observables,
                             record_configurations=(self.configurations_dir is not None
                                                    or self.records_path is not None),
                             mode=diffusion, n=diffusion_steps, **diffusion_kwargs)
            step = first + len(flags) - 1
            for key, r in self.grid.items():
                for U, N, N_ads, N_s, obs, config in r.block_samples:
                    self.samples[key]['U'].append(U)
                    self.samples[key]['N'].append(N)
                    if N_ads is not None:
                        self.samples[key]['N_ads'].append(N_ads)
                    if N_s is not None:
                        self.samples[key]['N_s'].append(N_s)
                    for name, v in obs.items():
                        self.samples[key]['obs:' + name].append(v)
                    sample = len(self.samples[key]['U']) - 1
                    if config is not None and self.configurations_dir is not None:
                        self._store_configuration(key, sample, U, config)
                    self._record(key, sample, U, N, N_s, config)
                r.block_samples = []  # do not ship the configurations back to the workers

            if trajectory_dir is not None and step % trajectory_every == 0:
                for key, r in self.grid.items():
                    self._save_trajectory_frame(trajectory_dir, key, r)

            if verbose and (step >= next_print or step == n_steps):  # blocks skip multiples
                next_print = (step // print_every + 1) * print_every
                mean_n = np.mean([r.system.n for r in self.grid.values()])
                print(f"step {step:7d}/{n_steps} | mean N across grid = {mean_n:6.2f}", flush=True)
            self._maybe_write_status(step, n_steps, "regc", force=step == n_steps)

            # check if a convergence check is set and terminate run if reached and at least the burn_in steps 
            # have already been performed
            if (convergence_every is not None and step > burn_in_steps
                    and step % convergence_every == 0):
                n_now = len(next(iter(self.samples.values()))['U'])
                if (n_at_check_before is not None and n_at_last_check - n_at_check_before >= 2
                        and n_now - n_at_last_check >= 2):
                    change = self.rho_u_change(n_at_last_check, convergence_bins,
                                               n_start=n_at_check_before)
                    self.convergence_history.append(change)
                    if verbose:
                        print(f"step {step:7d} | max change of rho(u) = {change:.4f}")
                    if change < convergence_tol:
                        self._flush_records()
                        self._maybe_write_status(step, n_steps, "regc", force=True, converged=True)
                        return True
                n_at_check_before, n_at_last_check = n_at_last_check, n_now
        self._flush_records()
        return False

    def rho_u_change(self, n_prev: int, bins: int = 30, n_start: int = 0) -> float:
        """
        Largest change of the reduced-energy density rho(u) between blocks.

        For each grid point, u = beta*(U - mu*N) + N*ln(Lambda^3) of its
        recorded samples (Eq. (9) of docs/REGC_algorithm.md; summed over
        species if several are exchanged) is histogrammed on common bins
        for two disjoint blocks: samples ``n_start:n_prev`` (the interval
        before the previous check) and samples ``n_prev:`` (since it).
        Returns the maximum over grid points of the total-variation
        distance 0.5 * sum |p_new - p_old| -- [ZSG]'s criterion, "no change
        in the density of reduced-energy states rho(U) with increasing
        simulation steps".

        The blocks are disjoint on purpose: comparing the first `n_prev`
        samples with all samples (which contain them) shrinks like 1/k
        after k equal checks whatever the data do, so a drifting run would
        pass. Two blocks of n samples drawn from one distribution still
        differ by about sqrt(bins / (pi n)) (sampling noise), so a tolerance
        must lie above that.

        Parameters
        ----------
        n_prev : int
            Number of samples per grid point at the previous check: the end
            of the old block, the start of the new one.
        bins : int, optional
            Number of histogram bins. Default 30.
        n_start : int, optional
            Start of the old block (the sample count at the check before
            the previous one). Default 0.

        Returns
        -------
        float
            The maximum total-variation distance, in [0, 1].
        """
        worst = 0.0
        for key, r in self.grid.items():
            U = np.asarray(self.samples[key]['U'], dtype=float)
            if self.n_species == 1:
                N = np.asarray(self.samples[key]['N'], dtype=float)
                u = r.state.beta * (U - r.state.mu * N) + N * np.log(r.state.lambda_cubed)
            else:
                N_s = np.asarray(self.samples[key]['N_s'], dtype=float).reshape(-1, self.n_species)
                u = (r.state.beta * U
                     + N_s @ (np.log(r.state.lambda_cubed_vec) - r.state.beta * r.state.mu_vec))
            old, new = u[n_start:n_prev], u[n_prev:]
            if len(old) == 0 or len(new) == 0:
                continue
            edges = np.histogram_bin_edges(np.concatenate([old, new]), bins=bins)
            if edges[0] == edges[-1]:
                continue  # a single repeated value: no change measurable
            p_new, _ = np.histogram(new, bins=edges)
            p_old, _ = np.histogram(old, bins=edges)
            tv = 0.5 * np.abs(p_new / p_new.sum() - p_old / p_old.sum()).sum()
            worst = max(worst, float(tv))
        return worst

    def fingerprint(self) -> dict:
        """
        What a checkpoint must agree with to be loaded into this simulation.

        Returns
        -------
        dict
            The temperature ladder, the mu grid and Lambda^3 on the ladder.
        """
        return {"temperatures": [float(T) for T in self.temperatures],
                "mu_grid": np.asarray(self.mu_grid, dtype=float).tolist(),
                "lambda_cubed": [np.asarray(self.lambda_cubed_of_T(T), dtype=float).tolist()
                                 for T in self.temperatures]}

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """
        Save everything needed to continue this simulation exactly.

        Replica configurations, energies and random streams, the engine's
        random stream, the samples, swap statistics and convergence history,
        plus `extra` (anything picklable, e.g. a driver's step counter). The
        write is atomic. A run continued with `load_checkpoint` is
        bit-identical to an uninterrupted one.

        Parameters
        ----------
        path : str
            Checkpoint file (pickle).
        extra : dict, optional
            Additional state to store and get back from `load_checkpoint`.
        """
        from .io import atomic_pickle
        self._flush_records()  # the records hold at least what the checkpoint does
        atomic_pickle({"fingerprint": self.fingerprint(), "grid": self.grid,
                       "adsorbed_cutoff": self.adsorbed_cutoff,
                       "observables_meta": self.observables_meta,
                       "samples": self.samples, "swap_stats": self.swap_stats,
                       "pair_stats": self.pair_stats, "round_trips": self.round_trips,
                       "walker_last_end": self.walker_last_end,
                       "skip_empty_swaps": self.skip_empty_swaps,
                       "convergence_history": self.convergence_history,
                       "rng_state": self.rng.bit_generator.state, "extra": extra or {}}, path)

    def load_checkpoint(self, path: str) -> dict:
        """
        Restore a state saved by `save_checkpoint` into this simulation.

        The simulation must have been constructed with the same temperature
        ladder, mu grid, Lambda^3(T) and ``adsorbed_cutoff``. The
        checkpoint supplies the configurations, energies, random streams
        and samples; the calculators and the move settings (``max_disp``,
        HMC) are this simulation's, built from the current setup. So a
        checkpoint survives a moved potential file, and a changed potential
        setting (e.g. a new hard core) takes effect: each restored energy is
        recomputed, and resynchronised with a warning if it changed.

        Parameters
        ----------
        path : str
            Checkpoint file. Only load files you created (pickle).

        Returns
        -------
        dict
            The `extra` dict given to `save_checkpoint`.

        Raises
        ------
        ValueError
            If the checkpoint's temperature ladder, mu grid, Lambda^3 or
            ``adsorbed_cutoff`` differ from this simulation's.
        """
        from .io import load_pickle
        st = load_pickle(path)
        mine, theirs = self.fingerprint(), st["fingerprint"]
        for key in mine:
            if not np.allclose(np.asarray(mine[key]), np.asarray(theirs[key]), rtol=1e-12, atol=1e-12):
                raise ValueError(f"checkpoint {path}: '{key}' differs from this simulation's")
        has_n_ads = all('N_ads' in v for v in st["samples"].values())
        if self.adsorbed_cutoff is not None and not has_n_ads:
            raise ValueError(f"checkpoint {path} has no 'N_ads' samples; it was written "
                             f"without adsorbed_cutoff and can't be continued with it")
        if self.adsorbed_cutoff is None and has_n_ads:
            raise ValueError(f"checkpoint {path} records 'N_ads' (adsorbed_cutoff was set); "
                             f"continue it with the same adsorbed_cutoff")
        saved_cutoff = st.get("adsorbed_cutoff", self.adsorbed_cutoff)  # absent in old files
        if saved_cutoff != self.adsorbed_cutoff:
            raise ValueError(f"checkpoint {path}: adsorbed_cutoff was {saved_cutoff}, now "
                             f"{self.adsorbed_cutoff}; N_ads would mix two definitions")
        saved_obs = st.get("observables_meta", {})
        if saved_obs != self.observables_meta:
            raise ValueError(f"checkpoint {path} recorded observables {sorted(saved_obs)}, this "
                             f"simulation {sorted(self.observables_meta)} (with the same "
                             f"definitions); they must match to continue the samples")
        fresh = self.grid
        resynced = []
        for key, r in st["grid"].items():
            f = fresh[key]
            r.system.calculator = f.system.calculator
            r.max_disp, r.hmc_enabled = f.max_disp, f.hmc_enabled
            r.hmc_leapfrog_steps, r.hmc_dt = f.hmc_leapfrog_steps, f.hmc_dt
            energy = r.system.total_energy()
            if not np.isclose(energy, r.energy, rtol=1e-9, atol=1e-8):
                resynced.append((key, r.energy, energy))
                r.energy = energy
        if resynced:
            key, old, new = resynced[0]
            warnings.warn(f"checkpoint {path}: the current calculator gives different energies "
                          f"for {len(resynced)} of {len(fresh)} restored configurations "
                          f"(e.g. {key}: {old:.6g} -> {new:.6g}); continuing with the current "
                          f"calculator's energies", stacklevel=2)
        self.grid, self.samples = st["grid"], st["samples"]
        self.swap_stats, self.convergence_history = st["swap_stats"], st["convergence_history"]
        self.rng.bit_generator.state = st["rng_state"]
        # keep the RE-move draw of the run being continued (bit-identical continuation)
        self.skip_empty_swaps = bool(st.get("skip_empty_swaps", False))
        if "round_trips" in st and all(hasattr(r.system, "walker_id") for r in self.grid.values()):
            self.pair_stats = st["pair_stats"]
            self.round_trips, self.walker_last_end = st["round_trips"], st["walker_last_end"]
        else:  # a checkpoint from before walker tracking: start counting now
            self._init_walkers()
        return st["extra"]

    def swap_acceptance(self) -> Dict[str, float]:
        """
        Replica-exchange acceptance ratio per swap pattern.

        [ZSG] Sec. III A tunes the (T, mu) spacing to about 25%.

        Returns
        -------
        dict of str to float
            Accepted / attempted swaps for each of `SWAP_TYPES` (0.0 if
            never attempted).
        """
        return {t: acc / max(1, att) for t, (att, acc) in self.swap_stats.items()}

    def save_configurations(self, directory: Optional[str], every: int = 1) -> None:
        """
        Store the configurations of recorded samples from now on (or stop).

        Parameters
        ----------
        directory : str or None
            Directory for ``configs_T{it}_mu{imu}.extxyz`` (one per grid
            point); None stops storing.
        every : int, optional
            Store every this many samples (by sample index). Default 1.

        Raises
        ------
        ValueError
            If `every` is not positive.
        """
        if int(every) < 1:
            raise ValueError(f"configurations_every must be >= 1, got {every}")
        self.configurations_dir = directory
        self.configurations_every = int(every)

    def _store_configuration(self, key: Tuple[int, int], sample: int, U: float,
                             config: tuple) -> None:
        """
        Append one recorded sample's configuration to its grid point's file.

        Parameters
        ----------
        key : tuple of int
            Grid point.
        sample : int
            Index of the sample in ``samples[key]``.
        U : float
            Its energy.
        config : tuple
            From `regc.trajectory.snapshot`.
        """
        if sample % self.configurations_every:
            return
        from .trajectory import append_configuration, configuration_path, lattice_of
        r = self.grid[key]
        n_mobile = len(config[1]) - config[2] - config[3]
        info = {"sample": sample, "T_index": key[0], "mu_index": key[1],
                "T": float(r.state.temperature), "U": float(U), "N": int(n_mobile)}
        if self.n_species == 1:
            info["mu"] = float(r.state.mu)
        append_configuration(configuration_path(self.configurations_dir, key), config,
                             lattice_of(r.system), r.system.pbc, info)

    def save_records(self, path: Optional[str], potential: str = "sampling", every: int = 1,
                     positions_dtype: str = "float64") -> None:
        """
        Append every recorded sample to an HDF5 records file from now on (or stop).

        See `regc.records`. The file is opened at the first recorded
        sample and flushed at the end of every `run`/`run_regc` call and
        at every checkpoint.

        Parameters
        ----------
        path : str or None
            The records file; None stops recording (and closes the file).
        potential : str, optional
            Name of the sampling potential. Default ``'sampling'``.
        every : int, optional
            Store the configuration of every this many samples (by sample
            index). Default 1.
        positions_dtype : str, optional
            ``'float64'`` (default) or ``'float32'``.

        Raises
        ------
        ValueError
            If `every` is not positive.
        """
        if int(every) < 1:
            raise ValueError(f"records_every must be >= 1, got {every}")
        if getattr(self, "_records", None) is not None:
            self._records.close()
            self._records = None
        self.records_path = path
        self.records_potential = str(potential)
        self.records_every = int(every)
        self.records_positions_dtype = positions_dtype

    def _record(self, key: Tuple[int, int], sample: int, U: float, N: int,
                N_s: Optional[Sequence[int]], config: Optional[tuple]) -> None:
        """
        Queue one recorded sample for the records file (if recording).

        Parameters
        ----------
        key : tuple of int
            Grid point.
        sample : int
            Its index there.
        U : float
            Energy under the sampling potential.
        N : int
            Mobile atoms.
        N_s : sequence of int or None
            Counts per species (several species).
        config : tuple or None
            `regc.trajectory.snapshot`.
        """
        if self.records_path is None:
            return
        if self._records is None:
            from .records import RecordWriter
            from .trajectory import lattice_of
            system = self.grid[key].system
            labels = set(map(str, system.frozen_species)) | set(map(str, system.substrate_species))
            species = sorted(labels | set(self.exchange_species))
            self._records = RecordWriter(self.records_path, self.records_potential, species,
                                         lattice_of(system), system.pbc,
                                         system.frozen_positions.shape[0], system.n_substrate,
                                         self.temperatures, self.mu_grid, self.n_species,
                                         self.records_positions_dtype)
        walker = getattr(self.grid[key].system, "walker_id", -1)
        keep = config is not None and sample % self.records_every == 0
        self._records.add(key, sample, -1 if walker is None else walker, U, N,
                          config if keep else None, N_s)

    def save_status(self, path: Optional[str], interval_s: float = 5.0,
                    extra: Optional[dict] = None) -> None:
        """
        Write a JSON heartbeat of the running simulation from now on (or stop).

        At most every `interval_s` seconds (and at the end of every run
        call) the file is replaced atomically with the progress, the
        per-replica sample counts and averages, the latest stretch of every
        replica's N and U, the swap statistics and the convergence history:
        what `regc.monitor` shows while the run is going.

        Parameters
        ----------
        path : str or None
            The status file (``status.json``); None stops writing it.
        interval_s : float, optional
            Minimum time between writes. Default 5 s.
        extra : dict, optional
            Copied into the file under ``'labels'`` (e.g. the ladder in
            kelvin and the pressures, for the monitor's axes). Default None.
        """
        self.status_path = path
        self.status_interval_s = float(interval_s)
        self.status_extra = dict(extra or {})
        self._status_last = 0.0
        self._status_started = time.time()
        self._status_offset, self._status_total = 0, None  # set by run_with_checkpoints

    TRACE_LENGTH = 400

    def status(self, step: int = 0, n_steps: int = 0, scheme: str = "regc") -> dict:
        """
        A JSON-serialisable snapshot of the simulation's progress and data.

        Parameters
        ----------
        step, n_steps : int, optional
            Progress of the current run call (offset by the steps a
            checkpointed driver already did). Default 0.
        scheme : str, optional
            ``'regc'`` or ``'sweep'``. Default ``'regc'``.

        Returns
        -------
        dict
            See `save_status`.
        """
        keys = sorted(self.grid)
        shape = [int(self.shape[0]), int(self.shape[1])]

        def grid_of(values):
            out = np.full(shape, np.nan)
            for k, v in zip(keys, values):
                out[k] = v
            return [[None if not np.isfinite(x) else float(x) for x in row] for row in out]

        n_samples, mean_N, std_N, mean_U = [], [], [], []
        traces_N, traces_U = [], []
        stride = 1
        for k in keys:
            v = self.samples[k]
            N = np.asarray(v["N"], dtype=float)
            U = np.asarray(v["U"], dtype=float)
            n_samples.append(N.size)
            mean_N.append(N.mean() if N.size else np.nan)
            std_N.append(N.std() if N.size > 1 else np.nan)
            mean_U.append(U.mean() if U.size else np.nan)
            stride = max(stride, int(np.ceil(N.size / self.TRACE_LENGTH)))
        for k in keys:
            v = self.samples[k]
            traces_N.append([float(x) for x in np.asarray(v["N"], dtype=float)[::stride]])
            traces_U.append([float(x) for x in np.asarray(v["U"], dtype=float)[::stride]])
        total = self._status_total if self._status_total is not None else n_steps
        trips = self.round_trip_counts()
        att = {t: int(a) for t, (a, _) in self.swap_stats.items()}
        return {
            "schema": 1, "time": time.time(), "started": self._status_started,
            "scheme": scheme, "step": int(self._status_offset + step), "n_steps": int(total),
            "shape": shape, "temperatures": [float(T) for T in self.temperatures],
            "mu_grid": np.asarray(self.mu_grid, dtype=float).tolist(),
            "exchange_species": list(self.exchange_species), "keys": [list(k) for k in keys],
            "replicas": {"n_samples": grid_of(n_samples), "mean_N": grid_of(mean_N),
                         "std_N": grid_of(std_N), "mean_U": grid_of(mean_U),
                         "current_N": grid_of([self.grid[k].system.n for k in keys]),
                         "current_U": grid_of([self.grid[k].energy for k in keys])},
            "traces": {"stride": stride, "N": traces_N, "U": traces_U},
            "swap_acceptance": self.swap_acceptance(), "swap_attempts": att,
            "pairs": [{"a": list(a), "b": list(b), "attempts": int(n), "acceptance": float(acc)}
                      for (a, b), (n, acc) in sorted(self.pair_acceptance().items())],
            "round_trips_T": trips["T"].tolist(), "round_trips_mu": trips["mu"].tolist(),
            "convergence_history": [float(c) for c in self.convergence_history],
            "records": None if self.records_path is None else {
                "path": os.path.abspath(self.records_path), "potential": self.records_potential,
                "every": self.records_every},
            "configurations_dir": self.configurations_dir,
            "labels": self.status_extra,
        }

    def _maybe_write_status(self, step: int, n_steps: int, scheme: str, force: bool = False,
                            **flags) -> None:
        """
        Write the status file if it is due (or `force`).

        Parameters
        ----------
        step, n_steps : int
            Progress of the current run call.
        scheme : str
            ``'regc'`` or ``'sweep'``.
        force : bool, optional
            Write regardless of the interval. Default False.
        **flags
            Extra top-level entries (e.g. ``converged=True``).
        """
        if self.status_path is None:
            return
        now = time.time()
        if not force and now - self._status_last < self.status_interval_s:
            return
        self._status_last = now
        data = self.status(step, n_steps, scheme)
        data.update(flags)
        tmp = self.status_path + ".tmp"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.status_path)), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(data, f, allow_nan=False, default=float)
            os.replace(tmp, self.status_path)
        except (OSError, ValueError) as err:  # never stop a run for its heartbeat
            warnings.warn(f"status file {self.status_path} not written: {err}", stacklevel=2)

    def _flush_records(self) -> None:
        """Write the queued records to disk."""
        if self._records is not None:
            self._records.flush()

    @property
    def n_substrate(self) -> int:
        """int: Displaceable (non-frozen) substrate atoms per replica (for the kinetic heat capacity)."""
        return int(next(iter(self.grid.values())).system.n_substrate)

    @property
    def observables_meta(self) -> dict:
        """dict: ``{name: metadata}`` of the recorded observables (see `regc.observables`)."""
        return {o.name: o.metadata() for o in self.observables}

    def _save_trajectory_frame(self, trajectory_dir: str, key: Tuple[int, int], replica: Replica) -> None:
        """
        Append one trajectory frame for a single replica.

        Parameters
        ----------
        trajectory_dir : str
            Directory to save the trajectory file under.
        key : tuple of int
            The replica's (temperature_index, mu_index) grid coordinates,
            used to name its trajectory file.
        replica : Replica
            The replica whose current full configuration should be
            appended as a new frame.

        Returns
        -------
        None
        """
        import os

        from .trajectory import append_trajectory_frame

        it, imu = key
        path = os.path.join(trajectory_dir, f"replica_T{it}_mu{imu}.xyz")
        positions, species = replica.system.full_configuration()
        append_trajectory_frame(path, positions, species, replica.system.cell,
                                 pbc=replica.system.pbc)

    def summary(self) -> List[dict]:
        """
        Build a per-grid-point summary of the current simulation state.

        Returns
        -------
        list of dict
            One dict per grid point (sorted by grid coordinates), with
            keys ``'T_index'``, ``'mu_index'``, ``'T'``, ``'mu'``, ``'N'``,
            ``'energy'``, ``'n_samples'``, ``'exchange_acc'``,
            ``'displace_acc'``, ``'insert_acc'``, ``'delete_acc'``, plus
            ``'N_s'`` (counts per species) with several exchanged species.
        """
        rows = []
        for (it, imu), r in sorted(self.grid.items()):
            acc = r.acceptance_rates()
            rows.append({
                'T_index': it, 'mu_index': imu,
                'T': r.state.temperature, 'mu': r.state.mu,
                'N': r.system.n, 'energy': round(r.energy, 3),
                'n_samples': len(self.samples[(it, imu)]['N']),
                'exchange_acc': round(r.exchange_accepts / max(1, r.exchange_attempts), 3),
                'displace_acc': round(acc['displace'], 3),
                'insert_acc': round(acc['insert'], 3),
                'delete_acc': round(acc['delete'], 3),
            })
            if self.n_species > 1:
                rows[-1]['N_s'] = r.system.species_counts().tolist()
        return rows
