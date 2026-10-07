"""
Long REGC runs: chunks, checkpoints, convergence, wall-time limits.

`run_with_checkpoints` drives `ReplicaExchangeGCMC.run_regc` in chunks.
After every chunk it writes the samples (``samples.npz``, see
`gcmc_re.io`) and a full checkpoint (``checkpoint.pkl``); it applies
[ZSG]'s convergence criterion (the reduced-energy densities rho(u) stop
changing, `ReplicaExchangeGCMC.rho_u_change`) and can stop cleanly before a
time limit. Called again on the same directory with ``resume=True`` it
continues bit-identically, which is what batch jobs with wall-time limits
need (see ``hpc/``).

`EXIT_RESUBMIT` is the conventional process exit code for "stopped for the
time limit; run me again to continue".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .io import save_samples

#: Exit code for "stopped for the time limit, resubmit to continue" (sysexits EX_TEMPFAIL).
EXIT_RESUBMIT = 75

CHECKPOINT_FILE = "checkpoint.pkl"
SAMPLES_FILE = "samples.npz"
METADATA_FILE = "run_metadata.json"


@dataclass
class RunStatus:
    """
    Outcome of `run_with_checkpoints`.

    Attributes
    ----------
    steps_done : int
        REGC steps completed so far (over all resumptions).
    finished : bool
        True if the run reached its step count or converged; False if it
        stopped for the time limit (continue with ``resume=True``).
    converged : bool
        True if the rho(u) convergence criterion was met.
    rho_u_changes : list of float
        The convergence measure at every check.
    resumed_from : int
        Step the run was resumed from (0 for a fresh start).
    """

    steps_done: int
    finished: bool
    converged: bool
    rho_u_changes: List[float] = field(default_factory=list)
    resumed_from: int = 0


def _check_metadata(path: str, metadata: dict) -> None:
    """
    Refuse to resume a run whose metadata differ from the saved ones.

    Parameters
    ----------
    path : str
        The saved metadata (JSON).
    metadata : dict
        The current run's metadata.

    Raises
    ------
    ValueError
        If a key's value differs (after a JSON round trip).
    """
    with open(path) as f:
        saved = json.load(f)
    current = json.loads(json.dumps(metadata, default=float))
    for key, value in current.items():
        if key in saved and saved[key] != value:
            raise ValueError(f"cannot resume: '{key}' is {value!r} now but {saved[key]!r} in {path}")


def run_with_checkpoints(sim, out_dir: str, n_steps: int, chunk: int = 1000,
                         x0: float = 0.99, diffusion: str = "mc", diffusion_steps: int = 10,
                         record_every: int = 100, convergence_every: Optional[int] = None,
                         convergence_tol: float = 0.05, time_limit: Optional[float] = None,
                         resume: bool = True, overwrite: bool = False,
                         metadata: Optional[dict] = None,
                         progress: Optional[Callable[[str], None]] = print,
                         **diffusion_kwargs) -> RunStatus:
    """
    Run `sim.run_regc` in checkpointed chunks.

    Parameters
    ----------
    sim : ReplicaExchangeGCMC
        The simulation (freshly constructed; on resume its state is replaced
        by the checkpoint's).
    out_dir : str
        Directory for ``samples.npz``, ``checkpoint.pkl`` and
        ``run_metadata.json``.
    n_steps : int
        Total REGC steps of the run (over all resumptions).
    chunk : int, optional
        Steps between checkpoints. Default 1000.
    x0, diffusion, diffusion_steps, record_every
        Passed to `ReplicaExchangeGCMC.run_regc`.
    convergence_every : int, optional
        Check rho(u) convergence every this many steps (rounded up to a
        multiple of `chunk`); also at the end. Default None: no checks.
    convergence_tol : float, optional
        Converged when the largest change of rho(u) between checks is below
        this. Default 0.05.
    time_limit : float, optional
        Seconds: stop (``finished=False``) before starting a chunk that is
        expected to end later than this, measured from the call. Default
        None (no limit).
    resume : bool, optional
        Continue the run saved in `out_dir` if there is one. Default True.
    overwrite : bool, optional
        Start afresh even if `out_dir` holds a run (with ``resume=False``).
        Default False.
    metadata : dict, optional
        JSON-serialisable description of the run (potential, lattice
        constant, protocol, ...). Saved on a fresh start; on resume every key
        must match the saved value.
    progress : callable, optional
        Called with a one-line status after every chunk. Default `print`;
        None for silence.
    **diffusion_kwargs
        Passed to `run_regc` (e.g. ``timestep_fs``, ``taut_fs``).

    Returns
    -------
    RunStatus
        Steps done, whether finished/converged, and the convergence history.

    Raises
    ------
    FileExistsError
        If `out_dir` holds a run and neither `resume` nor `overwrite` is set.
    ValueError
        On resume, if the checkpoint or `metadata` do not match `sim`.
    """
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, CHECKPOINT_FILE)
    meta_path = os.path.join(out_dir, METADATA_FILE)
    metadata = dict(metadata or {})
    done, n_at_last_check, changes, converged = 0, None, [], False

    if os.path.isfile(ckpt) and resume:
        if os.path.isfile(meta_path):
            _check_metadata(meta_path, metadata)
        extra = sim.load_checkpoint(ckpt)
        done, n_at_last_check = extra["steps_done"], extra["n_at_last_check"]
        changes, converged = list(extra["rho_u_changes"]), extra["converged"]
        if progress:
            progress(f"resumed from {ckpt} at step {done}")
    elif os.path.isfile(ckpt) and not overwrite:
        raise FileExistsError(f"{ckpt} exists: resume=True to continue it, or overwrite=True")
    else:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=1, default=float)
    resumed_from = done

    if convergence_every is not None:
        convergence_every = max(chunk, -(-convergence_every // chunk) * chunk)
    t0, last_chunk = time.time(), 0.0
    while done < n_steps and not converged:
        if time_limit is not None and time.time() - t0 + 1.5 * last_chunk > time_limit:
            if progress:
                progress(f"stopping at step {done} for the time limit; resume to continue")
            return RunStatus(done, False, False, changes, resumed_from)
        t_chunk = time.time()
        n = min(chunk, n_steps - done)
        sim.run_regc(n_steps=n, x0=x0, diffusion=diffusion, diffusion_steps=diffusion_steps,
                     record_every=record_every, verbose=False, **diffusion_kwargs)
        done += n
        if convergence_every is not None and (done % convergence_every == 0 or done >= n_steps):
            n_now = len(next(iter(sim.samples.values()))['U'])
            if n_at_last_check is not None and n_at_last_check >= 2:
                changes.append(sim.rho_u_change(n_at_last_check))
                converged = changes[-1] < convergence_tol
            n_at_last_check = n_now
        save_samples(sim, os.path.join(out_dir, SAMPLES_FILE))
        sim.save_checkpoint(ckpt, extra={"steps_done": done, "n_at_last_check": n_at_last_check,
                                         "rho_u_changes": changes, "converged": converged})
        last_chunk = time.time() - t_chunk
        if progress:
            mean_n = sum(r.system.n for r in sim.grid.values()) / len(sim.grid)
            acc = ", ".join(f"{t} {a:.2f}" for t, a in sim.swap_acceptance().items())
            progress(f"{done:8d}/{n_steps} steps  {time.time() - t0:8.0f} s  mean N = {mean_n:7.2f}"
                     f"  swap acc: {acc}" + (f"  rho(u) change {changes[-1]:.3f}" if changes else ""))
    if converged and progress:
        progress("rho(u) converged; stopping")
    return RunStatus(done, True, converged, changes, resumed_from)
