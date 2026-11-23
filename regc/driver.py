"""
Long REGC runs: chunks, checkpoints, convergence, wall-time limits.

`run_with_checkpoints` drives `ReplicaExchangeGCMC.run_regc` in chunks.
After every chunk it writes the samples (``samples.npz``, see
`regc.io`) and a full checkpoint (``checkpoint.pkl``); it applies
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
import shutil
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .engine import NOISE_CONSECUTIVE, NOISE_MIN_NEFF, NOISE_MIN_ROUND_TRIPS
from .io import save_samples

#: Exit code for "stopped for the time limit, resubmit to continue" (sysexits EX_TEMPFAIL).
EXIT_RESUBMIT = 75

CHECKPOINT_FILE = "checkpoint.pkl"
SAMPLES_FILE = "samples.npz"
METADATA_FILE = "run_metadata.json"
MIXING_FILE = "mixing.json"
LADDER_FILE = "ladder.json"
#: Metadata entry recording what an adapted ladder replaced (see `update_metadata`).
LADDER_META_KEY = "ladder_adaptation"


def write_mixing(sim, path: str, step: Optional[int] = None) -> dict:
    """
    Write the replica-exchange mixing statistics of a simulation as JSON.

    Parameters
    ----------
    sim : ReplicaExchangeGCMC
        The simulation.
    path : str
        Output file.
    step : int, optional
        Steps done. If given, the cumulative pair counts are first
        recorded (`snapshot_pair_stats`), so ``chunks`` gains an entry.

    Returns
    -------
    dict
        What was written: acceptance per pattern, per pair (with the worst
        pair), the round trips of every walker along T and mu (and, with
        bias columns, between the physical and the bias columns), and
        ``chunks``: per interval between snapshots, ``step_start``,
        ``step_end``, ``ladder_version`` and ``attempts`` / ``accepted``
        aligned with ``pairs``.
    """
    if step is not None and hasattr(sim, "snapshot_pair_stats"):
        sim.snapshot_pair_stats(step)
    pairs = sim.pair_acceptance()
    worst = min(pairs.items(), key=lambda kv: kv[1][1]) if pairs else None
    trips = sim.round_trip_counts()
    data = {
        "swap_acceptance": sim.swap_acceptance(),
        "pairs": [{"a": list(a), "b": list(b), "attempts": n, "acceptance": acc}
                  for (a, b), (n, acc) in sorted(pairs.items())],
        "worst_pair": None if worst is None else {"a": list(worst[0][0]), "b": list(worst[0][1]),
                                                  "attempts": worst[1][0],
                                                  "acceptance": worst[1][1]},
        "round_trips_T": trips["T"].tolist(), "round_trips_mu": trips["mu"].tolist(),
    }
    if "bias" in trips:   # bias columns: physical -> bias -> physical
        data["round_trips_bias"] = trips["bias"].tolist()
    order = sorted(pairs)
    chunks = sim.pair_acceptance_by_chunk() if hasattr(sim, "pair_acceptance_by_chunk") else []
    data["chunks"] = [{"step_start": c["step_start"], "step_end": c["step_end"],
                       "ladder_version": c.get("ladder_version", 0),
                       "attempts": [c["pairs"].get(k, (0, 0))[0] for k in order],
                       "accepted": [c["pairs"].get(k, (0, 0))[1] for k in order]}
                      for c in chunks]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    return data


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


def _write_json_atomic(path: str, data: dict) -> None:
    """
    Replace a JSON file atomically.

    Parameters
    ----------
    path : str
        The file.
    data : dict
        JSON-able data (numpy scalars allowed).
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=float)
    os.replace(tmp, path)


def update_metadata(path: str, update: dict) -> None:
    """
    Record a moved ladder in the saved run metadata.

    The entries of `update` replace the saved ones; what they replaced (and
    which keys they added) is kept under ``ladder_adaptation``, so a resumed
    run is still compared with the metadata it started from.

    Parameters
    ----------
    path : str
        ``run_metadata.json``.
    update : dict
        The new entries (e.g. `regc.ladder.LadderAdapter.metadata_update`).
    """
    with open(path) as f:
        saved = json.load(f)
    record = saved.setdefault(LADDER_META_KEY, {"original": {}, "added": []})
    for key in update:
        if key in record["original"] or key in record["added"]:
            continue
        if key in saved:
            record["original"][key] = saved[key]
        else:
            record["added"].append(key)
    saved.update(json.loads(json.dumps(update, default=float)))
    _write_json_atomic(path, saved)


def _check_metadata(path: str, metadata: dict) -> None:
    """
    Refuse to resume a run whose metadata differ from the saved ones.

    Entries changed by an adapted ladder (`update_metadata`) are compared
    in their original form.

    Parameters
    ----------
    path : str
        The saved metadata (JSON).
    metadata : dict
        The current run's metadata.

    Raises
    ------
    ValueError
        If a key's value differs (after a JSON round trip), or a key is
        present on only one side.
    """
    with open(path) as f:
        saved = json.load(f)
    adapted = saved.pop(LADDER_META_KEY, None)
    if adapted:
        for key in adapted.get("added", []):
            saved.pop(key, None)
        saved.update(adapted.get("original", {}))
    current = json.loads(json.dumps(metadata, default=float))
    only = sorted(set(saved) ^ set(current))
    if only:
        raise ValueError(f"cannot resume: metadata keys {only} are in only one of the current "
                         f"run and {path}")
    for key, value in current.items():
        if saved[key] != value:
            raise ValueError(f"cannot resume: '{key}' is {value!r} now but {saved[key]!r} in {path}")


def _run_ladder_es(sim, es, out_dir: str, done: int, n_steps: int, chunk: int, run_kwargs: dict,
                   time_limit: Optional[float], t0: float,
                   progress: Optional[Callable[[str], None]], meta_path: str, ckpt: str,
                   extra: dict) -> bool:
    """
    Run the remaining generations of a `regc.ladder.LadderES`, checkpointing after each.

    Parameters
    ----------
    sim : ReplicaExchangeGCMC
        The simulation.
    es : regc.ladder.LadderES
        The evolution strategy.
    out_dir : str
        Run directory (``ladder.json``).
    done, n_steps : int
        Steps done and the run's total (status file).
    chunk : int
        Evaluation length if the ES does not set one.
    run_kwargs : dict
        Passed to `run_regc` in the evaluations.
    time_limit : float or None
        Seconds since `t0` after which no generation starts that would end later.
    t0 : float
        Start of the call (``time.time()``).
    progress : callable or None
        Log lines.
    meta_path, ckpt : str
        ``run_metadata.json`` and the checkpoint.
    extra : dict
        Driver state saved with the checkpoint (besides ``steps_done``).

    Returns
    -------
    bool
        True if it stopped for the time limit (resume to continue).
    """
    last = 0.0
    while es.pending(sim):
        if time_limit is not None and time.time() - t0 + 1.5 * last > time_limit:
            if progress:
                progress(f"stopping in the ladder ES before generation "
                         f"{len(es.history(sim)) + 1} for the time limit; resume to continue")
            return True
        t_gen = time.time()
        rec = es.run_generation(sim, run_kwargs, step=done, eval_steps=chunk, progress=progress)
        if rec["changed"]:
            update_metadata(meta_path, es.metadata_update(sim))
        if hasattr(sim, "status_extra"):
            sim.status_extra.update(es.status_labels(sim))
            sim._status_offset, sim._status_total = done, n_steps
            sim._maybe_write_status(0, 0, "regc", force=True)
        _write_json_atomic(os.path.join(out_dir, LADDER_FILE), es.to_json(sim))
        sim.save_checkpoint(ckpt, extra={"steps_done": done, **extra})
        last = time.time() - t_gen
        if progress:
            progress(es.describe(rec) + f" ({last:.0f} s)")
    return False


def run_with_checkpoints(sim, out_dir: str, n_steps: int, chunk: int = 1000,
                         x0: float = 0.99, diffusion: str = "mc", diffusion_steps: int = 10,
                         record_every: int = 100, convergence_every: Optional[int] = None,
                         convergence_tol: Optional[float] = None, time_limit: Optional[float] = None,
                         resume: bool = True, overwrite: bool = False,
                         metadata: Optional[dict] = None,
                         progress: Optional[Callable[[str], None]] = print,
                         energy_tol: float = 1e-6, convergence_mode: str = "blocks",
                         convergence_bins: Optional[int] = None,
                         convergence_min_neff: float = NOISE_MIN_NEFF,
                         convergence_median_tol: float = 1.3, convergence_discard: float = 0.2,
                         convergence_consecutive: int = NOISE_CONSECUTIVE,
                         convergence_min_round_trips: int = NOISE_MIN_ROUND_TRIPS,
                         adapt_ladder=None, **diffusion_kwargs) -> RunStatus:
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
        Steps between checkpoints; a multiple of `record_every` (and of
        ``trajectory_every`` if given). Default 1000.
    x0, diffusion, diffusion_steps, record_every
        Passed to `ReplicaExchangeGCMC.run_regc`.
    convergence_every : int, optional
        Check rho(u) convergence every this many steps (rounded up to a
        multiple of `chunk`); also at the end. Default None: no checks.
    convergence_tol : float, optional
        ``convergence_mode='blocks'``: converged when rho(u) of the samples
        since the last check and of the interval before it differ (largest
        total-variation distance) by less than this; see
        `ReplicaExchangeGCMC.rho_u_change` for the noise floor. Default 0.05.
        ``'noise'``: the largest ratio of the distance between the two
        halves of all samples to its noise floor
        (`ReplicaExchangeGCMC.rho_u_noise_check`). Default 3.0.
    convergence_mode : str, optional
        ``'blocks'`` (default) or ``'noise'``.
    convergence_bins : int, optional
        Histogram bins. Default 30 (blocks) or 10 (noise).
    convergence_min_neff : float, optional
        ``'noise'``: independent samples every half must hold. Default 50.
    convergence_median_tol : float, optional
        ``'noise'``: the median ratio over the grid must be below this. Default 1.3.
    convergence_discard : float, optional
        ``'noise'``: fraction of the earliest samples left out (equilibration). Default 0.2.
    convergence_consecutive : int, optional
        ``'noise'``: checks in a row the criterion must pass. Default 2.
    convergence_min_round_trips : int, optional
        ``'noise'``: T round trips every walker must have completed. Default 1.
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
    energy_tol : float, optional
        After every chunk each replica's running energy is compared with a
        full recomputation (`ReplicaExchangeGCMC.check_energies`); a drift
        above this, per atom, is resynced with a warning. Default 1e-6.
    adapt_ladder : regc.ladder.LadderAdapter or regc.ladder.LadderES, optional
        `LadderAdapter`: adapt the ladder after every chunk until it is
        frozen (see `regc.ladder`): each change discards the samples so far
        (kept in ``samples_ladder<version>.npz``), restarts the convergence
        checks (skipped until the ladder is frozen) and is logged in
        ``ladder.json``; ``run_metadata.json`` is updated to the current
        ladder. Never after the last chunk. `LadderES`: run its generations
        before the first chunk (each evaluation lasts its ``eval_steps``,
        default `chunk`; these steps do not count toward `n_steps`), with a
        checkpoint and ``ladder.json`` after every generation; the
        production run starts on the frozen ladder. Default None (fixed
        ladder).
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
        If `chunk` is not a multiple of `record_every`; on resume, if the
        checkpoint or `metadata` do not match `sim`.
    """
    if chunk % record_every:
        raise ValueError(f"chunk ({chunk}) must be a multiple of record_every ({record_every}): "
                         f"each chunk counts steps from 1, so other chunks drop samples and "
                         f"space them unevenly")
    traj_every = diffusion_kwargs.get("trajectory_every")
    if traj_every and chunk % traj_every:
        raise ValueError(f"chunk ({chunk}) must be a multiple of trajectory_every ({traj_every})")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, CHECKPOINT_FILE)
    meta_path = os.path.join(out_dir, METADATA_FILE)
    metadata = dict(metadata or {})
    done, n_at_last_check, n_at_check_before, changes, converged = 0, None, None, [], False

    if os.path.isfile(ckpt) and resume:
        if os.path.isfile(meta_path):
            _check_metadata(meta_path, metadata)
        extra = sim.load_checkpoint(ckpt)
        done, n_at_last_check = extra["steps_done"], extra["n_at_last_check"]
        n_at_check_before = extra.get("n_at_check_before")
        changes, converged = list(extra["rho_u_changes"]), extra["converged"]
        if changes and not sim.convergence_history:      # checkpoints written before the fix
            sim.convergence_history = [float(c) for c in changes]
        if progress:
            progress(f"resumed from {ckpt} at step {done}")
        if adapt_ladder is not None and getattr(sim, "ladder_history", None) \
                and hasattr(sim, "status_extra"):
            sim.status_extra.update(adapt_ladder.status_labels(sim))
    elif os.path.isfile(ckpt) and not overwrite:
        raise FileExistsError(f"{ckpt} exists: resume=True to continue it, or overwrite=True")
    else:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=1, default=float)
    resumed_from = done

    mode, convergence_tol, convergence_bins = sim._convergence_defaults(
        convergence_mode, convergence_tol, convergence_bins)
    if convergence_every is not None:
        convergence_every = max(chunk, -(-convergence_every // chunk) * chunk)
        sim.convergence_settings = {"mode": mode, "tol": convergence_tol, "bins": convergence_bins,
                                    "min_neff": float(convergence_min_neff),
                                    "median_tol": float(convergence_median_tol),
                                    "discard": float(convergence_discard),
                                    "consecutive": int(convergence_consecutive),
                                    "min_round_trips": int(convergence_min_round_trips),
                                    "every": int(convergence_every)}
    t0, last_chunk = time.time(), 0.0
    if getattr(adapt_ladder, "method", None) == "es" and adapt_ladder.pending(sim):
        run_kwargs = dict(x0=x0, diffusion=diffusion, diffusion_steps=diffusion_steps,
                          record_every=record_every, **diffusion_kwargs)
        stopped = _run_ladder_es(sim, adapt_ladder, out_dir, done, n_steps, chunk, run_kwargs,
                                 time_limit, t0, progress, meta_path, ckpt,
                                 {"n_at_last_check": n_at_last_check,
                                  "n_at_check_before": n_at_check_before,
                                  "rho_u_changes": changes, "converged": converged})
        if stopped:
            return RunStatus(done, False, False, changes, resumed_from)
    while done < n_steps and not converged:
        if time_limit is not None and time.time() - t0 + 1.5 * last_chunk > time_limit:
            if progress:
                progress(f"stopping at step {done} for the time limit; resume to continue")
            return RunStatus(done, False, False, changes, resumed_from)
        t_chunk = time.time()
        n = min(chunk, n_steps - done)
        sim._status_offset, sim._status_total = done, n_steps  # whole-run progress in status.json
        sim.run_regc(n_steps=n, x0=x0, diffusion=diffusion, diffusion_steps=diffusion_steps,
                     record_every=record_every, verbose=False, **diffusion_kwargs)
        done += n
        drifts = sim.check_energies(energy_tol)
        bad = {k: d for k, (d, resynced) in drifts.items() if resynced}
        if bad:
            k, d = max(bad.items(), key=lambda kv: abs(kv[1]))
            warnings.warn(f"step {done}: the running energy of {len(bad)} replica(s) drifted from a "
                          f"full recomputation (largest {d:.3g} at {k}); resynced. Persistent "
                          f"drift means the calculator's incremental and full energies disagree",
                          stacklevel=2)
        detail = None
        adapting = (adapt_ladder is not None and getattr(adapt_ladder, "method", "model") != "es"
                    and adapt_ladder.pending(sim))
        if adapting:
            pass   # the ladder will still move: no convergence checks yet
        elif (convergence_every is not None and mode == "noise"
                and (done % convergence_every == 0 or done >= n_steps)):
            detail = sim.rho_u_noise_check(convergence_bins, convergence_tol, convergence_min_neff,
                                           convergence_median_tol, convergence_discard,
                                           convergence_consecutive, convergence_min_round_trips)
            detail["step"] = int(done)
            changes.append(detail["max_ratio"])
            converged = detail["converged"]
        elif convergence_every is not None and (done % convergence_every == 0 or done >= n_steps):
            n_now = len(next(iter(sim.samples.values()))['U'])
            if (n_at_check_before is not None and n_at_last_check - n_at_check_before >= 2
                    and n_now - n_at_last_check >= 2):
                # disjoint blocks: since the last check vs the interval before it
                changes.append(sim.rho_u_change(n_at_last_check, bins=convergence_bins,
                                                n_start=n_at_check_before))
                converged = changes[-1] < convergence_tol
                detail = {"step": int(done), "max_change": changes[-1], "converged": converged}
            n_at_check_before, n_at_last_check = n_at_last_check, n_now
        if detail is not None:
            # publish the check: the engine's history (checkpoint) and status.json (monitor)
            sim.convergence_history.append(float(changes[-1]))
            sim.convergence_details.append(detail)
            sim._maybe_write_status(n, n, "regc", force=True, converged=bool(converged))
        save_samples(sim, os.path.join(out_dir, SAMPLES_FILE))
        write_mixing(sim, os.path.join(out_dir, MIXING_FILE), step=done)
        if adapting and done < n_steps:
            version = sim.ladder_version
            rec, reason = adapt_ladder.adapt(sim, done)
            if rec is None:
                if progress:
                    progress(f"ladder not adapted at step {done}: {reason}")
            else:
                if rec["changed"]:
                    # the samples of the old ladder (still on disk) are kept aside
                    shutil.copyfile(os.path.join(out_dir, SAMPLES_FILE),
                                    os.path.join(out_dir, f"samples_ladder{version}.npz"))
                    changes, n_at_last_check, n_at_check_before = [], None, None
                    update_metadata(meta_path, adapt_ladder.metadata_update(sim))
                    if hasattr(sim, "status_extra"):
                        sim.status_extra.update(adapt_ladder.status_labels(sim))
                        sim._maybe_write_status(n, n, "regc", force=True)
                _write_json_atomic(os.path.join(out_dir, LADDER_FILE), adapt_ladder.to_json(sim))
                if progress:
                    progress(adapt_ladder.describe(rec))
        sim.save_checkpoint(ckpt, extra={"steps_done": done, "n_at_last_check": n_at_last_check,
                                         "n_at_check_before": n_at_check_before,
                                         "rho_u_changes": changes, "converged": converged})
        last_chunk = time.time() - t_chunk
        if progress:
            mean_n = sum(r.system.n for r in sim.grid.values()) / len(sim.grid)
            acc = ", ".join(f"{t} {a:.2f}" for t, a in sim.swap_acceptance().items())
            conv = ""
            if changes and mode == "noise" and sim.convergence_details:
                d = sim.convergence_details[-1]
                conv = (f"  rho(u) halves: TV/noise max {d['max_ratio']:.2f} (tol {convergence_tol:g})"
                        f" median {d['median_ratio']:.2f} (tol {convergence_median_tol:g}),"
                        f" min n_eff {d['min_neff']:.1f}"
                        + (f", T round trips {d['min_round_trips_T']}"
                           if d.get("min_round_trips_T") is not None else ""))
            elif changes:
                conv = f"  rho(u) change {changes[-1]:.3f}"
            progress(f"{done:8d}/{n_steps} steps  {time.time() - t0:8.0f} s  mean N = {mean_n:7.2f}"
                     f"  swap acc: {acc}" + conv)
    if converged and progress:
        progress("rho(u) converged; stopping")
    return RunStatus(done, True, converged, changes, resumed_from)
