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
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

from matplotlib.cbook import index_of
import numpy as np

from .replica import Replica
from .state import ThermoState
from .systems import ParticleSystem


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

    `lambda_cubed_of_T(T) -> float` sets each replica's thermal de Broglie
    wavelength cubed (see `ThermoState.lambda_cubed`); the default (always
    1.0) is correct for reduced LJ units and a no-op for existing code. Real-
    units setups (e.g. an ASE EMT/EAM/MLIP calculator) should pass the true
    value -- see `gcmc_re.units.thermal_wavelength_cubed_angstrom3`.

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
                 hmc_dt: float = 0.005):
        """
        Construct the replica-exchange ladder.

        Parameters
        ----------
        temperatures : list of float
            The temperature ladder, length n_T.
        mus : array_like
            Either 1D (length n_mu, shared across all temperatures) or 2D
            (shape (n_T, n_mu), an explicit mu for every grid point --
            what a pressure ladder needs; see `pressure.py`).
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
        lambda_cubed_of_T : callable, optional
            A function ``lambda_cubed_of_T(T) -> float`` giving each
            replica's thermal de Broglie wavelength cubed (see
            `ThermoState.lambda_cubed`). Default always returns 1.0,
            correct for reduced LJ units.
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

        Raises
        ------
        ValueError
            If `mus` is 2D and its first dimension doesn't match
            ``len(temperatures)``, or if `mus` is neither 1D nor 2D.
        """
        self.temperatures = list(temperatures)
        mus_arr = np.asarray(mus, dtype=float)
        if mus_arr.ndim == 1:
            self.mu_grid = np.tile(mus_arr, (len(self.temperatures), 1))
        elif mus_arr.ndim == 2:
            if mus_arr.shape[0] != len(self.temperatures):
                raise ValueError("mus.shape[0] must equal len(temperatures)")
            self.mu_grid = mus_arr
        else:
            raise ValueError("mus must be 1D or 2D (n_T, n_mu)")
        self.shape = self.mu_grid.shape
        self.n_workers = n_workers
        self.lambda_cubed_of_T = lambda_cubed_of_T  # also read by MBARSurfaceAnalysis
        self._executor: Optional[ProcessPoolExecutor] = None

        master_rng = np.random.default_rng(seed)
        self.grid: Dict[Tuple[int, int], Replica] = {}
        self.samples: Dict[Tuple[int, int], Dict[str, list]] = {}
        for it, imu in itertools.product(range(self.shape[0]), range(self.shape[1])):
            rng = np.random.default_rng(master_rng.integers(0, 2 ** 62))
            system = system_factory(rng)
            state = ThermoState(self.temperatures[it], self.mu_grid[it, imu], label=(it, imu),
                                 lambda_cubed=lambda_cubed_of_T(self.temperatures[it]))
            self.grid[(it, imu)] = Replica(state, system, rng, max_disp=max_disp,
                                            hmc_enabled=hmc_enabled,
                                            hmc_leapfrog_steps=hmc_leapfrog_steps,
                                            hmc_dt=hmc_dt)
            self.samples[(it, imu)] = {'U': [], 'N': []}
        # drawn after the replicas' seeds, so their streams are unchanged by it
        self.rng = np.random.default_rng(master_rng.integers(0, 2 ** 62))
        self.swap_stats = {t: [0, 0] for t in SWAP_TYPES}  # type -> [attempts, accepts]
        self.convergence_history: List[float] = []
    def __getitem__(self, index: Tuple[int, int]):
        f"""Overload of getitem, returns the replica from the grid at index [i,j]

        Args:
            index (Tuple[int,int]): position on the grid to retrieve

        Returns:
            Replica: Replica[i,j]
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
        Na, Nb = ra.system.n, rb.system.n

        # log of the swap acceptance ratio (see docs/REGC_algorithm.md, Eq. (8)).
        # The Lambda^3 prefactor uses the same lambda_cubed the insert/delete
        # moves use, so it reduces to the paper's (beta_a/beta_b)^(3/2 dN)
        # for a physical Lambda(T) and to 1 for the reduced-units Lambda = 1.
        log_acc = ((beta_a - beta_b) * (Ua - Ub)
                   - (beta_a * mu_a - beta_b * mu_b) * (Na - Nb)
                   + (Na - Nb) * np.log(ra.state.lambda_cubed / rb.state.lambda_cubed))

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
            stats[0] += 1
            stats[1] += int(self._attempt_exchange(key_a, key_b))

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

            if trajectory_dir is not None and cycle % trajectory_every == 0:
                for key, r in self.grid.items():
                    self._save_trajectory_frame(trajectory_dir, key, r)

            if verbose and (cycle % max(1, n_cycles // 10) == 0 or cycle == n_cycles):
                mean_n = np.mean([r.system.n for r in self.grid.values()])
                print(f"cycle {cycle:6d}/{n_cycles} | mean N across grid = {mean_n:6.2f}")
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
           each uniformly, and attempt all of that pattern's disjoint swaps
           (Eq. (8)).
        2. Diffusion at fixed (N, V, T), in parallel over replicas (see
           `Replica.diffuse`).
        3. Every `record_every` steps after `burn_in_steps`, record (U, N)
           of every replica -- i.e. after the last diffusion move, before
           the next GC/RE step.
        4. Every `convergence_every` steps, test convergence of the
           reduced-energy densities rho(u) (see `rho_u_change`); stop if
           converged, otherwise continue up to `n_steps`.

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
            Converged when the largest total-variation change of rho(u)
            over all grid points since the previous check is below this.
            Default is 0.05.
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
        n_at_last_check = None

        def needs_main_process(step: int) -> bool:
            # steps after which the main process must act on the replicas
            return ((trajectory_dir is not None and step % trajectory_every == 0)
                    or (verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps))
                    or (convergence_every is not None and step > burn_in_steps
                        and step % convergence_every == 0))

        def draw_swap() -> Tuple[str, int]:
            swap_type = swap_types[int(self.rng.integers(len(swap_types)))]
            return swap_type, int(self.rng.integers(2))

        # Steps are executed in blocks: a block starts with (at most) one RE
        # move and continues with GC steps until the next RE draw or a step
        # needing the main process. Between RE moves replicas are independent,
        # so each runs its whole block in one call; the random draws and their
        # order are exactly those of a step-by-step loop.
        step = 0
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
                             mode=diffusion, n=diffusion_steps, **diffusion_kwargs)
            step = first + len(flags) - 1
            for key, r in self.grid.items():
                for U, N in r.block_samples:
                    self.samples[key]['U'].append(U)
                    self.samples[key]['N'].append(N)

            if trajectory_dir is not None and step % trajectory_every == 0:
                for key, r in self.grid.items():
                    self._save_trajectory_frame(trajectory_dir, key, r)

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps):
                mean_n = np.mean([r.system.n for r in self.grid.values()])
                print(f"step {step:7d}/{n_steps} | mean N across grid = {mean_n:6.2f}")

            if (convergence_every is not None and step > burn_in_steps
                    and step % convergence_every == 0):
                n_now = len(next(iter(self.samples.values()))['U'])
                if n_at_last_check is not None and n_at_last_check >= 2:
                    change = self.rho_u_change(n_at_last_check, convergence_bins)
                    self.convergence_history.append(change)
                    if verbose:
                        print(f"step {step:7d} | max change of rho(u) = {change:.4f}")
                    if change < convergence_tol:
                        return True
                n_at_last_check = n_now
        return False

    def rho_u_change(self, n_prev: int, bins: int = 30) -> float:
        """
        Largest change of the reduced-energy density rho(u) between checks.

        For each grid point, u = beta*(U - mu*N) + N*ln(Lambda^3) of its
        recorded samples (Eq. (9) of docs/REGC_algorithm.md) is histogrammed
        twice on common bins: using the first `n_prev` samples (the state
        at the previous check) and using all samples. Returns the maximum
        over grid points of the total-variation distance
        0.5 * sum |p_all - p_prev| -- [ZSG]'s criterion, "no change in the
        density of reduced-energy states rho(U) with increasing simulation
        steps".

        Parameters
        ----------
        n_prev : int
            Number of samples per grid point at the previous check.
        bins : int, optional
            Number of histogram bins. Default 30.

        Returns
        -------
        float
            The maximum total-variation change, in [0, 1].
        """
        worst = 0.0
        for key, r in self.grid.items():
            U = np.asarray(self.samples[key]['U'], dtype=float)
            N = np.asarray(self.samples[key]['N'], dtype=float)
            u = r.state.beta * (U - r.state.mu * N) + N * np.log(r.state.lambda_cubed)
            edges = np.histogram_bin_edges(u, bins=bins)
            if edges[0] == edges[-1]:
                continue  # a single repeated value: no change measurable
            p_all, _ = np.histogram(u, bins=edges)
            p_prev, _ = np.histogram(u[:n_prev], bins=edges)
            tv = 0.5 * np.abs(p_all / p_all.sum() - p_prev / p_prev.sum()).sum()
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
                "lambda_cubed": [float(self.lambda_cubed_of_T(T)) for T in self.temperatures]}

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
        atomic_pickle({"fingerprint": self.fingerprint(), "grid": self.grid,
                       "samples": self.samples, "swap_stats": self.swap_stats,
                       "convergence_history": self.convergence_history,
                       "rng_state": self.rng.bit_generator.state, "extra": extra or {}}, path)

    def load_checkpoint(self, path: str) -> dict:
        """
        Restore a state saved by `save_checkpoint` into this simulation.

        The simulation must have been constructed with the same temperature
        ladder, mu grid and Lambda^3(T); the system factory is not called
        again for the restored replicas.

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
            If the checkpoint's temperature ladder, mu grid or Lambda^3
            differ from this simulation's.
        """
        from .io import load_pickle
        st = load_pickle(path)
        mine, theirs = self.fingerprint(), st["fingerprint"]
        for key in mine:
            if not np.allclose(np.asarray(mine[key]), np.asarray(theirs[key]), rtol=1e-12, atol=1e-12):
                raise ValueError(f"checkpoint {path}: '{key}' differs from this simulation's")
        self.grid, self.samples = st["grid"], st["samples"]
        self.swap_stats, self.convergence_history = st["swap_stats"], st["convergence_history"]
        self.rng.bit_generator.state = st["rng_state"]
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
            ``'displace_acc'``, ``'insert_acc'``, ``'delete_acc'``.
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
        return rows
