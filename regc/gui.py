"""
Local web GUI to build, run and monitor regc simulations.

    regc-gui [WORKSPACE] [--port 8060] [--pythonpath DIR] [--python EXE]
    # then open http://127.0.0.1:8060

WORKSPACE (default: the current directory) is searched for config files
(``*.yaml`` / ``*.yml`` with a ``replica_exchange`` section, at most four
directories deep, output directories skipped). The page has three tabs:

* **Build**: edit a config as YAML or through a form of the common keys
  (structure, gas, calculator, reservoir, ladder, sampling, subsurface,
  convergence, records, output), with ladder helpers; *Validate* runs
  `regc.runner.validate_config`, *Save* writes the file (inside the
  workspace only), *Check setup* runs ``regc-run --check`` (builds the
  system, prints the mu ladder and the bare-slab energy).
* **Runs**: every config with its output directory and state; start a run,
  resume it from its checkpoint, start over (``--overwrite``), re-analyse
  it or stop it. Runs are launched exactly as the example ``run_monitor.sh``
  scripts do (detached with ``setsid``, log in ``<output>/run.log``, process
  id in ``<output>/run.pid``, working directory = the config's directory),
  so they survive the GUI and the scripts and the GUI see each other's runs.
* **Monitor**: the `regc.monitor` page for any of the runs, mounted under
  ``/monitor/``.

The runs use the Python interpreter of the GUI (``--python``) and
``PYTHONPATH`` = ``--pythonpath`` (default: the directory holding the
``regc`` package the GUI imported, so GUI and runs use the same code; it can
be changed on the page, e.g. to another git worktree).

The server binds to 127.0.0.1. Requests that change files or start and stop
processes need a per-session token that only the served page knows, so
other web pages open in the same browser cannot use them; requests naming
another host (DNS rebinding) or, for changes, coming from another origin
are refused. On a remote
machine, forward the port (``ssh -L 8060:localhost:8060 host``).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

from . import monitor

#: Directory names never searched for configs.
_SKIP_DIRS = {"trajectories", "configurations", "__pycache__", ".git", "node_modules", "build",
              "dist", ".ipynb_checkpoints"}
#: Files that mark a directory as a run's output directory (not searched).
_RUN_MARKERS = ("status.json", "run_metadata.json", "checkpoint.pkl", "samples.npz", "records.h5")
MAX_DEPTH = 4
STALE_S = monitor.STALE_S

#: Built-in starting points for the builder.
TEMPLATES = {
    "Pd(111) + H, EMT (quick test, ~1 min)": """\
# Quick test: Pd(111) + H with EMT (not physical; exercises the whole pipeline).
structure:
  build: {surface: fcc111, symbol: Pd, size: [2, 2, 3], a: auto}
  n_frozen_layers: 1
  headroom_angstrom: 5.0
gas:
  species: H
  insertion_z_range_above_surface: [1.0, 3.0]
calculator: {type: EMT}
reservoir: {type: diatomic, d0: self}
replica_exchange:
  temperatures_K: [350, 450]
  pressures: [1.0e-2, 1.0, 1.0e+2]
  pressure_unit: atm
  scheme: regc
  n_steps: 400
  x0: 0.9
  diffusion: mc
  diffusion_steps: 2
  record_every: 5
  convergence_every: 200
  max_disp: 0.3
  seed: 1
  n_workers: 1
checkpoint: {chunk: 100}
records: {save: true}
phase_diagram:
  T_min_K: 350
  T_max_K: 450
  n_T: 10
  p_min: 1.0e-2
  p_max: 1.0e+2
  n_p: 15
  observable: N
  coverage_per: surface_atom
output: {directory: output_test_emt}
""",
    "Slab + O from O2, LAMMPS MLIP (skeleton)": """\
# O on an fcc(100) slab from O2 gas with a LAMMPS machine-learned potential.
# Put model.npz next to this file; fill in the potential and the ladder.
structure:
  build: {surface: fcc100, symbol: Pd, size: [4, 4, 4], a: auto}
  n_frozen_layers: 2
  headroom_angstrom: 5.0
gas: {species: O, mass_amu: 15.999}
calculator:
  type: LAMMPS
  lammps_name: grace
  pair_style: grace/runtime
  pair_coeff: "* * model.npz Pd O"
  species: [Pd, O]
  min_distance: 1.0
reservoir: {type: diatomic, d0: self, bond_length_angstrom: 1.2075,
            vibrational_wavenumber_cm: 1580.2, symmetry_number: 2}
replica_exchange:
  temperatures_K: [300, 350, 410, 480, 560, 650, 760, 900]
  pressures: [1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
  pressure_unit: bar
  scheme: regc
  n_steps: 40000
  x0: 0.85
  diffusion: md
  diffusion_steps: 20
  md: {timestep_fs: 1.0, thermostat: bussi, taut_fs: 50.0}
  record_every: 5
  convergence_every: 2000
  convergence_mode: noise
  adsorbed_cutoff: 2.6
  max_disp: 0.3
  seed: 0
  n_workers: auto
checkpoint: {chunk: 500}
records: {save: true, every: 1, positions: float64}
trajectory: {save: true, directory: trajectories, every: 500}
phase_diagram:
  T_min_K: 300
  T_max_K: 900
  n_T: 120
  p_min: 1.0e-14
  p_max: 1.0
  n_p: 150
  pressure_unit: bar
  observable: N_ads
  coverage_per: surface_atom
  most_probable: true
  discard: 0.3
output: {directory: output}
""",
    "Realistic surface (fixed / thermostat / free) + O, LAMMPS MLIP": """\
# O on an fcc(100) slab from O2 gas with a LAMMPS machine-learned potential, on a
# realistic substrate: 2 fixed layers, 2 thermostatted, 2 free (NVE with the adsorbates).
# Put model.npz next to this file; fill in the potential and the ladder.
structure:
  build: {surface: fcc100, symbol: Pd, size: [4, 4], a: auto}   # depth from layers
  layers: {fixed: 2, thermostat: 2, free: 2}
  headroom_angstrom: 5.0
gas: {species: O, mass_amu: 15.999}
calculator:
  type: LAMMPS
  lammps_name: grace
  pair_style: grace/runtime
  pair_coeff: "* * model.npz Pd O"
  species: [Pd, O]
  min_distance: 1.0
reservoir: {type: diatomic, d0: self, bond_length_angstrom: 1.2075,
            vibrational_wavenumber_cm: 1580.2, symmetry_number: 2}
replica_exchange:
  temperatures_K: [300, 350, 410, 480, 560, 650, 760, 900]
  pressures: [1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
  pressure_unit: bar
  scheme: regc
  n_steps: 40000
  x0: 0.85
  diffusion: md
  diffusion_steps: 20
  md: {timestep_fs: 1.0, thermostat: bussi, taut_fs: 50.0}
  record_every: 5
  convergence_every: 2000
  convergence_mode: noise
  adsorbed_cutoff: 2.6
  max_disp: 0.3
  seed: 0
  n_workers: auto
checkpoint: {chunk: 500}
records: {save: true, every: 1, positions: float64}
trajectory: {save: true, directory: trajectories, every: 500}
phase_diagram:
  T_min_K: 300
  T_max_K: 900
  n_T: 120
  p_min: 1.0e-14
  p_max: 1.0
  n_p: 150
  pressure_unit: bar
  observable: N_ads
  coverage_per: surface_atom
  most_probable: true
  discard: 0.3
output: {directory: output}
""",
}


#: Form fields of the builders: (key, label, kind, placeholder) or (section title,).
#: kind: str, num, int, val (number, bool or string), list, bool, or sel:|opt1|opt2.
FORM_FIELDS = [
    ("Structure",),
    ("structure.file", "slab file", "str"),
    ("structure.build.surface", "build: surface", "str", "fcc111, fcc100, bcc110, hcp0001…"),
    ("structure.build.symbol", "build: element", "str"),
    ("structure.build.size", "build: size", "list", "4, 4, 4"),
    ("structure.build.a", "build: a (Å or auto)", "val"),
    ("structure.n_frozen_layers", "frozen layers", "int"),
    ("structure.layers.fixed", "or layers: fixed", "int", "e.g. 2 (with thermostat, free)"),
    ("structure.layers.thermostat", "layers: thermostat", "int"),
    ("structure.layers.free", "layers: free (NVE)", "int"),
    ("structure.headroom_angstrom", "headroom (Å)", "num"),
    ("Gas and reservoir",),
    ("gas.species", "gas species", "val", "O, or [O, Cl]"),
    ("gas.mass_amu", "mass (amu)", "num"),
    ("reservoir.type", "reservoir", "sel:|diatomic|monatomic"),
    ("reservoir.d0", "D0", "val", "self, experimental or eV"),
    ("Calculator",),
    ("calculator.type", "type", "sel:|EMT|LAMMPS|ASE|LJ|PairLJ"),
    ("calculator.lammps_name", "lammps_name", "str"),
    ("calculator.pair_style", "pair_style", "str"),
    ("calculator.pair_coeff", "pair_coeff", "str"),
    ("calculator.species", "species", "list", "Pd, O"),
    ("calculator.potential", "potential file", "str"),
    ("Replica ladder",),
    ("replica_exchange.temperatures_K", "T (K)", "list"),
    ("replica_exchange.pressures", "pressures", "list"),
    ("replica_exchange.pressure_unit", "pressure unit", "sel:|bar|atm|Pa"),
    ("replica_exchange.mu_eV", "or μ (eV)", "list"),
    ("Sampling",),
    ("replica_exchange.scheme", "scheme", "sel:|regc|sweep"),
    ("replica_exchange.n_steps", "n_steps (cap)", "int"),
    ("replica_exchange.x0", "x0 (GC step prob.)", "num"),
    ("replica_exchange.diffusion", "diffusion", "sel:|mc|mc_moves|hmc|md"),
    ("replica_exchange.diffusion_steps", "diffusion steps", "int"),
    ("replica_exchange.record_every", "record every", "int"),
    ("replica_exchange.adsorbed_cutoff", "adsorbed cutoff (Å)", "num"),
    ("replica_exchange.max_disp", "max disp (Å)", "num"),
    ("replica_exchange.seed", "seed", "int"),
    ("replica_exchange.n_workers", "workers", "val", "auto or a number"),
    ("Convergence",),
    ("replica_exchange.convergence_every", "check every (steps)", "int"),
    ("replica_exchange.convergence_mode", "mode", "sel:|noise|blocks"),
    ("replica_exchange.convergence_tol", "tol (max ratio)", "num"),
    ("replica_exchange.convergence_median_tol", "median tol", "num"),
    ("replica_exchange.convergence_min_neff", "min n_eff", "num"),
    ("Subsurface (NCMC)",),
    ("subsurface.depth_angstrom", "depth (Å)", "num", "empty: no subsurface moves"),
    ("subsurface.z_cut_below_surface", "z_c below top (Å)", "num"),
    ("subsurface.ncmc_steps", "λ steps", "int"),
    ("subsurface.kernel", "kernel", "sel:|langevin|hmc"),
    ("subsurface.weight", "weight", "num"),
    ("Output",),
    ("checkpoint.chunk", "checkpoint every", "int"),
    ("records.save", "records.h5", "bool"),
    ("records.every", "configs every n samples", "int"),
    ("trajectory.save", "trajectories", "bool"),
    ("trajectory.every", "traj. every (steps)", "int"),
    ("phase_diagram.observable", "phase-diagram observable", "str", "N, N_ads, N_O"),
    ("phase_diagram.coverage_per", "coverage per", "sel:|surface_atom|area|none"),
    ("output.directory", "output directory", "str"),
]


# =============================================================================
# Workspace: configs, their runs, launching and stopping
# =============================================================================
def _within(path: str, root: str) -> bool:
    """
    Whether `path` lies inside `root` (after resolving links).

    Parameters
    ----------
    path, root : str
        Paths.

    Returns
    -------
    bool
        True if inside (or equal).
    """
    p, r = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([p, r]) == r


def _proc_stat(pid: int) -> Optional[tuple]:
    """
    State and start time of a process, from ``/proc/<pid>/stat`` (Linux).

    Parameters
    ----------
    pid : int
        Process id.

    Returns
    -------
    tuple or None
        ``(state letter, start time in clock ticks after boot)`` (fields 3
        and 22), or None where ``/proc`` is unavailable.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()   # the name may hold spaces
        return fields[0], int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _proc_start_epoch(pid: int) -> Optional[float]:
    """
    When a process started, in seconds since the epoch (Linux).

    Parameters
    ----------
    pid : int
        Process id.

    Returns
    -------
    float or None
        The start time (to about a second), or None if unknown.
    """
    st = _proc_stat(pid)
    if st is None:
        return None
    try:
        with open("/proc/stat") as f:
            btime = next(int(ln.split()[1]) for ln in f if ln.startswith("btime"))
        return btime + st[1] / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError):
        return None


def pid_alive(pid: Optional[int], start: Optional[int] = None) -> bool:
    """
    Whether a process with this id runs (and is the one meant).

    With `start` (the process start time recorded at launch, field 22 of
    ``/proc/<pid>/stat``) the process must have that start time, so a
    reused id is not taken for the run. Without it (old ``run.pid`` files)
    the command line must contain ``regc``, where it can be read.

    Parameters
    ----------
    pid : int or None
        Process id.
    start : int, optional
        Start time in clock ticks after boot. Default None.

    Returns
    -------
    bool
        True if alive (and, where checkable, the same process).
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    st = _proc_stat(pid)
    if st is not None:
        if st[0] == "Z":                  # exited, not reaped yet
            return False
        if start is not None:
            return st[1] == start
    cmdline = f"/proc/{pid}/cmdline"
    if os.path.exists(cmdline):
        try:
            with open(cmdline, "rb") as f:
                return b"regc" in f.read()
        except OSError:
            return False
    return True


def heartbeat_alive(st: Optional[dict], now: Optional[float] = None) -> Optional[bool]:
    """
    Whether the process that wrote a ``status.json`` heartbeat still runs.

    Heartbeats name their process (``pid``, ``host``); on this host that
    process is checked (it must have started before the heartbeat did, so
    a reused id does not count). Other heartbeats (another host, older
    runs) count as alive while fresher than `STALE_S`.

    Parameters
    ----------
    st : dict or None
        The heartbeat.
    now : float, optional
        Current time. Default: `time.time()`.

    Returns
    -------
    bool or None
        True / False, or None if there is no heartbeat.
    """
    if not st:
        return None
    pid = st.get("pid")
    if isinstance(pid, int) and st.get("host") == socket.gethostname():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            pass                          # exists (another user's)
        stat = _proc_stat(pid)
        if stat is not None and stat[0] == "Z":
            return False
        began = _proc_start_epoch(pid)
        return began is None or began <= float(st.get("started") or st.get("time") or 0) + 2.0
    now = time.time() if now is None else now
    return now - float(st.get("time") or 0) < STALE_S


def _read_pid(path: str) -> Optional[int]:
    """
    A process id from a pid file.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    int or None
        The id, or None if missing or unreadable.
    """
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _read_pid_start(out_dir: str, pid: Optional[int]) -> Optional[int]:
    """
    The start time recorded with a run's process id (``run.pid.start``).

    Parameters
    ----------
    out_dir : str
        The output directory.
    pid : int or None
        The id in ``run.pid``.

    Returns
    -------
    int or None
        Start time in clock ticks after boot, or None if not recorded for this id.
    """
    try:
        with open(os.path.join(out_dir, "run.pid.start")) as f:
            p, start = (int(x) for x in f.read().split()[:2])
    except (OSError, ValueError):
        return None
    return start if p == pid else None


def _tail(path: str, n_bytes: int = 4000) -> str:
    """
    The end of a text file.

    Parameters
    ----------
    path : str
        The file.
    n_bytes : int, optional
        How much to read. Default 4000.

    Returns
    -------
    str
        The last `n_bytes` (empty if missing).
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


class Workspace:
    """
    The configs under a directory, their output directories, and the processes running them.

    Parameters
    ----------
    root : str
        Workspace directory.
    python : str, optional
        Interpreter for the runs. Default: this one.
    pythonpath : str, optional
        ``PYTHONPATH`` of the runs. Default: the directory holding the
        imported ``regc`` package.
    """

    def __init__(self, root: str, python: Optional[str] = None, pythonpath: Optional[str] = None):
        """
        Set up the workspace.

        Parameters
        ----------
        root, python, pythonpath
            See the class docstring.
        """
        self.root = os.path.realpath(root)
        self.python = python or sys.executable
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(monitor.__file__)))
        self.pythonpath = pkg if pythonpath is None else pythonpath
        self.runs: List[monitor.Run] = []           # the monitor's list (append-only)
        self._run_ids: Dict[str, int] = {}
        self._status_cache: Dict[str, tuple] = {}   # path -> (mtime, parsed)
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self.tasks: Dict[str, dict] = {}            # --check processes
        self._task_dir = tempfile.mkdtemp(prefix="regc_gui_")

    # ---- paths -------------------------------------------------------------
    def resolve(self, rel: str) -> str:
        """
        An absolute path for a workspace-relative one, refusing anything outside.

        Parameters
        ----------
        rel : str
            Path relative to the workspace (or absolute inside it).

        Returns
        -------
        str
            The absolute path.

        Raises
        ------
        ValueError
            If the path leaves the workspace.
        """
        p = os.path.normpath(os.path.join(self.root, rel))
        if not _within(p, self.root):
            raise ValueError(f"{rel!r} is outside the workspace {self.root}")
        return p

    def rel(self, path: str) -> str:
        """
        A workspace-relative path.

        Parameters
        ----------
        path : str
            Absolute path.

        Returns
        -------
        str
            Relative to the workspace root.
        """
        return os.path.relpath(path, self.root)

    # ---- discovery ---------------------------------------------------------
    def config_files(self) -> List[str]:
        """
        Config files in the workspace.

        Returns
        -------
        list of str
            Absolute paths of ``*.yaml`` / ``*.yml`` files with a
            ``replica_exchange`` section, sorted.
        """
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            depth = os.path.relpath(dirpath, self.root).count(os.sep)
            if dirpath != self.root and (
                    depth >= MAX_DEPTH or any(m in filenames for m in _RUN_MARKERS)):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in _SKIP_DIRS and not d.startswith("."))
            for name in filenames:
                if not name.endswith((".yaml", ".yml")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > 200_000:
                        continue
                    with open(path, errors="replace") as f:
                        if "replica_exchange" not in f.read():
                            continue
                except OSError:
                    continue
                found.append(path)
        return sorted(found)

    @staticmethod
    def output_dir(config_path: str) -> str:
        """
        The output directory a config writes to.

        Parameters
        ----------
        config_path : str
            The config file.

        Returns
        -------
        str
            ``output.directory`` resolved against the config's directory
            (``regc-run`` resolves it against the working directory, which
            the GUI sets to the config's directory).
        """
        import yaml
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            out = (cfg.get("output") or {}).get("directory", "output")
        except Exception:  # noqa: BLE001 - a broken config still gets a row
            out = "output"
        return os.path.normpath(os.path.join(os.path.dirname(config_path), str(out)))

    def monitor_id(self, out_dir: str) -> int:
        """
        The monitor's index of an output directory (added if new).

        Parameters
        ----------
        out_dir : str
            The output directory.

        Returns
        -------
        int
            Index into `runs`.
        """
        with self._lock:
            if out_dir not in self._run_ids:
                r = monitor.Run(out_dir, os.path.join(out_dir, "run.log"))
                r.name = self.rel(out_dir)
                self._run_ids[out_dir] = len(self.runs)
                self.runs.append(r)
            return self._run_ids[out_dir]

    def _status(self, out_dir: str) -> Optional[dict]:
        """
        A run's ``status.json``, cached by modification time.

        Parameters
        ----------
        out_dir : str
            The output directory.

        Returns
        -------
        dict or None
            The heartbeat.
        """
        path = os.path.join(out_dir, monitor.STATUS_FILE)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None
        hit = self._status_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        st = monitor._read_json(path)
        if st is not None:
            self._status_cache[path] = (mtime, st)
        return st

    def _liveness(self, out_dir: str, st: Optional[dict] = None,
                  log: Optional[str] = None) -> tuple:
        """
        Whether a run is going in an output directory, and whether the GUI started it.

        A run counts as going if the process in ``run.pid`` runs (started by
        a GUI), or else if the ``status.json`` heartbeat says so (started by
        a script or ``regc-run``; see `heartbeat_alive`). A heartbeat
        without a checkable process does not count once sampling is done or
        the log recorded the run's end since.

        Parameters
        ----------
        out_dir : str
            The output directory.
        st : dict, optional
            Its heartbeat (read if None).
        log : str, optional
            The end of its ``run.log`` (read if None).

        Returns
        -------
        tuple
            ``(alive, pid or None, external)``.
        """
        pid = _read_pid(os.path.join(out_dir, "run.pid"))
        if pid_alive(pid, _read_pid_start(out_dir, pid)):
            return True, pid, False
        st = self._status(out_dir) if st is None else st
        if not heartbeat_alive(st):
            return False, None, False
        hb_pid = st.get("pid") if isinstance(st.get("pid"), int) else None
        if hb_pid is None or st.get("host") != socket.gethostname():
            checks = (st.get("convergence") or {}).get("checks") or [{}]
            done = (st.get("converged") or checks[-1].get("converged")
                    or (st.get("n_steps") and st.get("step", 0) >= st["n_steps"]))
            log_path = os.path.join(out_dir, "run.log")
            log = _tail(log_path) if log is None else log
            mark = log.rfind("# regc-gui")
            stopped_here = mark >= 0 and log.startswith("stop", log.find(": ", mark) + 2)
            ended = stopped_here or max(log.rfind("Outputs in"), log.rfind("Traceback"),
                                        log.rfind("Killed")) > mark
            try:
                ended = ended and os.path.getmtime(log_path) >= float(st.get("time") or 0) - 1.0
            except OSError:
                ended = False
            if done or ended:
                return False, None, False
        return True, hb_pid, True

    def job(self, out_dir: str) -> dict:
        """
        The state of the run in an output directory.

        Parameters
        ----------
        out_dir : str
            The output directory.

        Returns
        -------
        dict
            ``state`` (new, starting, running, stalled, finished, converged,
            stopped: a checkpoint to resume from, error, ended), ``pid``,
            ``external`` (going, but not started by a GUI: found through its
            heartbeat), ``step``, ``n_steps``, ``convergence`` (the latest
            check), ``subsurface`` (the heartbeat's totals), ``last_line`` of
            the log, ``resumable``.
        """
        st = self._status(out_dir)
        log = _tail(os.path.join(out_dir, "run.log"))
        alive, pid, external = self._liveness(out_dir, st, log)
        lines = [ln for ln in log.splitlines() if ln.strip()]
        last = lines[-1] if lines else ""
        resumable = os.path.isfile(os.path.join(out_dir, "checkpoint.pkl"))
        conv = None
        if st and (st.get("convergence") or {}).get("checks"):
            conv = st["convergence"]["checks"][-1]
        mark, trace = log.rfind("# regc-gui"), log.rfind("Traceback")
        end = log.rfind("Outputs in")
        # "Stopped for the time limit; continue with --resume. Outputs in ..."
        time_limited = end >= 0 and "Stopped for the time limit" in log[log.rfind("\n", 0, end) + 1:end]
        sampled = bool((conv or {}).get("converged")
                       or (st and st.get("n_steps") and st.get("step", 0) >= st["n_steps"]))
        if alive:
            if st is None:
                state = "starting"
            else:
                state = "running" if time.time() - st.get("time", 0) < STALE_S else "stalled"
        elif not os.path.isdir(out_dir):
            state = "new"
        elif end > max(trace, mark) and time_limited and not sampled:
            state = "stopped"
        elif end > max(trace, mark) or sampled:
            # (runs launched elsewhere may log to another file: trust the heartbeat too)
            state = "converged" if (conv or {}).get("converged") else "finished"
        elif trace > mark or last.endswith("Killed"):
            state = "error"
        elif resumable:
            state = "stopped"
        else:
            state = "ended" if os.listdir(out_dir) else "new"
        return {"state": state, "pid": pid if alive else None, "external": alive and external,
                "step": st.get("step") if st else None, "n_steps": st.get("n_steps") if st else None,
                "convergence": conv, "subsurface": (st or {}).get("subsurface"),
                "last_line": last[-300:], "resumable": resumable}

    def overview(self) -> dict:
        """
        Everything the Runs tab shows.

        Returns
        -------
        dict
            ``root``, ``python``, ``pythonpath``, ``configs`` (one entry per
            config file: ``path``, ``output`` (workspace-relative),
            ``output_dir`` (absolute; may lie outside the workspace),
            ``monitor_id``, ``job``),
            ``templates``.
        """
        rows = []
        for path in self.config_files():
            out = self.output_dir(path)
            rows.append({"path": self.rel(path), "output": self.rel(out), "output_dir": out,
                         "monitor_id": self.monitor_id(out), "job": self.job(out)})
        return {"root": self.root, "python": self.python, "pythonpath": self.pythonpath,
                "configs": rows, "templates": list(TEMPLATES)}

    # ---- processes ---------------------------------------------------------
    def _env(self) -> dict:
        """
        The environment of launched processes.

        Returns
        -------
        dict
            This environment with `pythonpath` first on ``PYTHONPATH``, and one
            thread per process for the math libraries unless set already
            (as the example ``run.sh`` scripts do).
        """
        env = os.environ.copy()
        if self.pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(
                [self.pythonpath] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env.setdefault("PYTHONUNBUFFERED", "1")
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, "1")
        return env

    def _cmd(self, config_path: str, *flags: str) -> List[str]:
        """
        The command line of ``regc-run`` with this interpreter.

        Parameters
        ----------
        config_path : str
            The config.
        *flags : str
            Extra arguments.

        Returns
        -------
        list of str
            The command.
        """
        return [self.python, "-c", "import sys; from regc.runner import main; sys.exit(main())",
                os.path.basename(config_path), *flags]

    def start(self, rel_config: str, mode: str = "run", workers: Optional[int] = None) -> dict:
        """
        Launch ``regc-run`` on a config, detached.

        Parameters
        ----------
        rel_config : str
            Workspace-relative config path.
        mode : str, optional
            ``run`` (refused if the output holds a checkpoint), ``resume``,
            ``overwrite`` (start over, replacing the checkpoint and records)
            or ``analyze`` (``--analyze-only``). Default ``run``.
        workers : int, optional
            ``--workers``. Default None: the config's ``n_workers``.

        Returns
        -------
        dict
            ``pid``, ``log``, ``command``.

        Raises
        ------
        ValueError
            On an unknown mode, a run already going (started by a GUI, a
            script or ``regc-run``), or ``run`` over an existing checkpoint.
        """
        path = self.resolve(rel_config)
        if not os.path.isfile(path):
            raise ValueError(f"no config {rel_config}")
        out = self.output_dir(path)
        if not _within(out, self.root):
            raise ValueError(f"output directory {out} is outside the workspace")
        flags = {"run": [], "resume": ["--resume"], "overwrite": ["--overwrite"],
                 "analyze": ["--analyze-only"]}.get(mode)
        if flags is None:
            raise ValueError(f"unknown mode {mode!r}")
        with self._start_lock:            # check -> launch -> run.pid, one request at a time
            alive, pid, external = self._liveness(out)
            if alive:
                who = "started outside the GUI, " if external else ""
                raise ValueError(f"a run is already going in {self.rel(out)} ({who}pid {pid or '?'})")
            if mode == "run" and os.path.isfile(os.path.join(out, "checkpoint.pkl")):
                raise ValueError(f"{self.rel(out)} holds a checkpoint: resume it, or start over "
                                 f"(overwrite)")
            if workers:
                flags = flags + ["--workers", str(int(workers))]
            os.makedirs(out, exist_ok=True)
            log_path = os.path.join(out, "run.log")
            cmd = self._cmd(path, *flags)
            with open(log_path, "ab") as log:
                log.write(f"\n# regc-gui {time.strftime('%Y-%m-%d %H:%M:%S')}: {mode} "
                          f"(PYTHONPATH={self.pythonpath})\n".encode())
                log.flush()
                proc = subprocess.Popen(cmd, cwd=os.path.dirname(path), stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True, env=self._env())
            stat, start_file = _proc_stat(proc.pid), os.path.join(out, "run.pid.start")
            if stat is not None:                  # the start time: against pid reuse
                with open(start_file, "w") as f:
                    f.write(f"{proc.pid} {stat[1]}\n")
            elif os.path.exists(start_file):
                os.remove(start_file)
            with open(os.path.join(out, "run.pid"), "w") as f:
                f.write(str(proc.pid))
        threading.Thread(target=proc.wait, daemon=True).start()   # reap it when it ends
        self.monitor_id(out)
        return {"pid": proc.pid, "log": self.rel(log_path), "command": " ".join(cmd)}

    def stop(self, rel_config: str) -> dict:
        """
        Stop the run of a config (SIGTERM to its process group).

        The checkpoint written every ``checkpoint.chunk`` steps stays, so
        the run can be resumed.

        Parameters
        ----------
        rel_config : str
            Workspace-relative config path.

        Returns
        -------
        dict
            ``stopped`` (the pid).

        Raises
        ------
        ValueError
            If no run is going, or it was not started by a GUI.
        """
        out = self.output_dir(self.resolve(rel_config))
        pid = _read_pid(os.path.join(out, "run.pid"))
        if not pid_alive(pid, _read_pid_start(out, pid)):
            alive, hb_pid, _ = self._liveness(out)
            if alive:
                raise ValueError(f"the run in {self.rel(out)} was started outside the GUI "
                                 f"(pid {hb_pid or '?'}): stop it where it was started")
            raise ValueError(f"no run going in {self.rel(out)}")
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)
        try:
            with open(os.path.join(out, "run.log"), "a") as f:
                f.write(f"\n# regc-gui {time.strftime('%Y-%m-%d %H:%M:%S')}: stop (SIGTERM to {pid})\n")
        except OSError:
            pass
        return {"stopped": pid}

    def check(self, rel_config: str) -> str:
        """
        Start ``regc-run --check`` on a config.

        Parameters
        ----------
        rel_config : str
            Workspace-relative config path.

        Returns
        -------
        str
            Task id for `task`.

        Raises
        ------
        ValueError
            If the config file does not exist.
        """
        path = self.resolve(rel_config)
        if not os.path.isfile(path):
            raise ValueError(f"no config {rel_config} (save it first)")
        tid = secrets.token_hex(6)
        out_path = os.path.join(self._task_dir, tid + ".log")
        with open(out_path, "wb") as out:
            proc = subprocess.Popen(self._cmd(path, "--check"), cwd=os.path.dirname(path),
                                    stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=self._env())
        self.tasks[tid] = {"proc": proc, "log": out_path, "config": rel_config,
                           "started": time.time()}
        return tid

    def task(self, tid: str) -> dict:
        """
        Output of a ``--check`` task.

        Parameters
        ----------
        tid : str
            Task id.

        Returns
        -------
        dict
            ``done``, ``returncode``, ``output``, ``elapsed``.
        """
        t = self.tasks[tid]
        rc = t["proc"].poll()
        return {"done": rc is not None, "returncode": rc, "output": _tail(t["log"], 60_000),
                "elapsed": time.time() - t["started"], "config": t["config"]}


# =============================================================================
# Config helpers for the builder
# =============================================================================
def parse_config(text: str) -> dict:
    """
    Parse and validate config text.

    Parameters
    ----------
    text : str
        YAML.

    Returns
    -------
    dict
        ``config`` (or None), ``ok``, ``error``, ``summary`` (replicas,
        steps, a rough size of the records file).
    """
    import yaml
    from .runner import validate_config
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as err:
        return {"config": None, "ok": False, "error": f"YAML: {err}", "summary": None}
    if not isinstance(cfg, dict):
        return {"config": None, "ok": False, "error": "the config must be a mapping",
                "summary": None}
    try:
        validate_config(cfg)
        ok, error = True, None
    except Exception as err:  # noqa: BLE001 - shown on the page
        ok, error = False, f"{type(err).__name__}: {err}"
    re_cfg = cfg.get("replica_exchange") or {}
    summary = None
    try:
        n_T = len(re_cfg.get("temperatures_K") or [])
        n_mu = len(re_cfg.get("pressures") or re_cfg.get("mu_eV") or [])
        shifts = (re_cfg.get("bias_columns") or {}).get("shifts_kT") or []
        n_mu += len(next(iter(shifts.values()))) if isinstance(shifts, dict) and shifts \
            else len(shifts)
        n_steps = int(re_cfg.get("n_steps") or 0)
        every = int(re_cfg.get("record_every") or 100)
        summary = {"replicas": n_T * n_mu, "n_T": n_T, "n_mu": n_mu, "n_steps": n_steps,
                   "samples_per_replica": n_steps // max(every, 1)}
    except (TypeError, ValueError):
        pass
    return {"config": cfg, "ok": ok, "error": error, "summary": summary}


def dump_config(cfg: dict) -> str:
    """
    YAML text of a config (block style; comments are not kept).

    Parameters
    ----------
    cfg : dict
        The config.

    Returns
    -------
    str
        YAML.
    """
    import yaml

    class _Dumper(yaml.SafeDumper):
        """Short lists of scalars in flow style, everything else in block style."""

    def _repr_list(dumper, data):
        flow = len(data) <= 16 and all(not isinstance(x, (list, dict)) for x in data)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)

    def _repr_float(dumper, value):
        # 1e-14 as 1.0e-14 (PyYAML reads a bare 1e-14 back as a string)
        text = repr(float(value))
        if "e" in text and "." not in text.split("e")[0]:
            m, e = text.split("e")
            text = f"{m}.0e{'+' if not e.startswith('-') else ''}{e.lstrip('+')}"
        elif "e" in text and not text.split("e")[1].startswith(("-", "+")):
            m, e = text.split("e")
            text = f"{m}e+{e}"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)

    _Dumper.add_representer(list, _repr_list)
    _Dumper.add_representer(float, _repr_float)
    return yaml.dump(cfg, Dumper=_Dumper, sort_keys=False, default_flow_style=False, width=100)


# =============================================================================
# HTTP server
# =============================================================================
def make_handler(ws: Workspace, token: str, plotly_js: Optional[str] = None):
    """
    The request handler class of the GUI.

    Parameters
    ----------
    ws : Workspace
        The workspace.
    token : str
        Session token required by every POST (header ``X-Regc-Token``).
    plotly_js : str, optional
        Local ``plotly.min.js`` for the monitor.

    Returns
    -------
    type
        A `BaseHTTPRequestHandler` subclass.

    Raises
    ------
    ValueError
        Inside a request, for a bad save target; caught and answered as an
        error response, never raised to the caller.
    """
    page = PAGE.replace("__TOKEN__", token).replace(
        "__FIELDS__", json.dumps(FORM_FIELDS, ensure_ascii=False)).encode()

    class Handler(BaseHTTPRequestHandler):
        """GET: page, workspace, configs, tasks, ``/monitor/...``; POST: the actions."""

        def log_message(self, *args) -> None:
            """Silence the per-request log."""

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            """
            Send a response.

            Parameters
            ----------
            body : bytes
                Content.
            ctype : str
                Content type.
            code : int, optional
                Status. Default 200.
            """
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any, code: int = 200) -> None:
            """
            Send JSON.

            Parameters
            ----------
            data : object
                Payload.
            code : int, optional
                Status. Default 200.
            """
            self._send(monitor._json_bytes(data), "application/json", code)

        def _foreign(self, post: bool = False) -> bool:
            """
            Refuse (403) a request for another host name, or a POST from another origin.

            A page of another site whose name was rebound to this machine
            (DNS rebinding) sends its own name as ``Host``; this keeps the
            page, and so the token, away from it.

            Parameters
            ----------
            post : bool, optional
                Check ``Origin`` too. Default False.

            Returns
            -------
            bool
                True if refused (the response is sent).
            """
            bound, port = monitor.server_host(self.server)
            if not monitor.host_allowed(self.headers.get("Host"), bound, port):
                self._json({"error": "forbidden host"}, 403)
                return True
            if post and not monitor.origin_allowed(self.headers.get("Origin"), bound, port):
                self._json({"error": "forbidden origin"}, 403)
                return True
            return False

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            """Serve one GET request."""
            if self._foreign():
                return None
            url = urlparse(self.path)
            path, q = unquote(url.path), parse_qs(url.query)
            try:
                if path in ("/", "/index.html"):
                    return self._send(page, "text/html; charset=utf-8")
                if path == "/monitor":
                    self.send_response(301)
                    self.send_header("Location", "/monitor/")
                    return self.end_headers()
                if path.startswith("/monitor/"):
                    body, ctype, code = monitor.route(path[len("/monitor"):], ws.runs, plotly_js)
                    return self._send(body, ctype, code)
                if path == "/api/workspace":
                    return self._json(ws.overview())
                if path == "/api/config":
                    p = ws.resolve(q["path"][0])
                    with open(p) as f:
                        return self._json({"path": ws.rel(p), "text": f.read()})
                if path == "/api/template":
                    return self._json({"text": TEMPLATES[q["name"][0]]})
                if path.startswith("/api/task/"):
                    return self._json(ws.task(path.rsplit("/", 1)[1]))
                if path == "/api/log":
                    out = ws.output_dir(ws.resolve(q["path"][0]))
                    return self._json({"text": _tail(os.path.join(out, "run.log"), 20_000)})
            except (KeyError, ValueError, TypeError, OSError) as err:
                return self._json({"error": f"{type(err).__name__}: {err}"}, 400)
            self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            """Serve one POST request (token required)."""
            if self._foreign(post=True):
                return None
            if not secrets.compare_digest(self.headers.get("X-Regc-Token", ""), token):
                return self._json({"error": "missing or wrong session token"}, 403)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json({"error": "bad JSON"}, 400)
            if not isinstance(data, dict):
                return self._json({"error": "bad request: a JSON object is expected"}, 400)
            path = urlparse(self.path).path
            try:
                if path == "/api/parse":
                    return self._json(parse_config(data.get("text", "")))
                if path == "/api/dump":
                    return self._json({"text": dump_config(data["config"])})
                if path == "/api/save":
                    p = ws.resolve(data["path"])
                    if not p.endswith((".yaml", ".yml")):
                        raise ValueError("config files end in .yaml or .yml")
                    if os.path.exists(p) and not data.get("overwrite"):
                        raise ValueError(f"{ws.rel(p)} exists (tick 'overwrite' to replace it)")
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as f:
                        f.write(data["text"])
                    return self._json({"saved": ws.rel(p)})
                if path == "/api/check":
                    return self._json({"task": ws.check(data["path"])})
                if path == "/api/start":
                    return self._json(ws.start(data["path"], data.get("mode", "run"),
                                               data.get("workers")))
                if path == "/api/stop":
                    return self._json(ws.stop(data["path"]))
                if path == "/api/settings":
                    if "pythonpath" in data:
                        ws.pythonpath = str(data["pythonpath"]).strip()
                    return self._json({"pythonpath": ws.pythonpath})
            except (KeyError, ValueError, TypeError, OSError) as err:
                return self._json({"error": f"{type(err).__name__}: {err}"}, 400)
            self._json({"error": "not found"}, 404)

    return Handler


def serve(root: str = ".", host: str = "127.0.0.1", port: int = 8060,
          python: Optional[str] = None, pythonpath: Optional[str] = None,
          plotly_js: Optional[str] = None, open_browser: bool = False,
          ready: Optional[threading.Event] = None) -> ThreadingHTTPServer:
    """
    Run the GUI server (blocking until interrupted, or until ``shutdown()``).

    Parameters
    ----------
    root : str, optional
        Workspace directory. Default the current one.
    host : str, optional
        Interface. Default ``'127.0.0.1'``.
    port : int, optional
        Port (0: any free one). Default 8060.
    python, pythonpath : str, optional
        Interpreter and ``PYTHONPATH`` of the runs (see `Workspace`).
    plotly_js : str, optional
        Local ``plotly.min.js`` for the monitor.
    open_browser : bool, optional
        Open the page. Default False.
    ready : threading.Event, optional
        Set once the server listens (``ready.server`` then holds it); for tests.

    Returns
    -------
    ThreadingHTTPServer
        The server, after it stopped.
    """
    ws = Workspace(root, python, pythonpath)
    token = secrets.token_urlsafe(24)
    httpd = ThreadingHTTPServer((host, port), make_handler(ws, token, plotly_js))
    httpd.workspace, httpd.token, httpd.regc_host = ws, token, host
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"regc gui: workspace {ws.root}\n  {url}  (Ctrl-C to stop)\n"
          f"  runs use {ws.python} with PYTHONPATH={ws.pythonpath}", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    if ready is not None:
        ready.server = httpd
        ready.set()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd


def main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``regc-gui [WORKSPACE]``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        0.
    """
    ap = argparse.ArgumentParser(prog="regc-gui",
                                 description="Web GUI to build, run and monitor regc simulations.")
    ap.add_argument("workspace", nargs="?", default=".",
                    help="directory searched for configs (default: current directory)")
    ap.add_argument("--port", type=int, default=8060, help="port (default 8060)")
    ap.add_argument("--host", default="127.0.0.1", help="interface (default 127.0.0.1)")
    ap.add_argument("--python", default=None, help="interpreter of the runs (default: this one)")
    ap.add_argument("--pythonpath", default=None,
                    help="PYTHONPATH of the runs (default: where this regc package lives)")
    ap.add_argument("--plotly-js", default=None, help="local plotly.min.js for the monitor")
    ap.add_argument("--open", action="store_true", help="open a browser")
    args = ap.parse_args(argv)
    serve(args.workspace, args.host, args.port, args.python, args.pythonpath, args.plotly_js,
          args.open)
    return 0


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>regc gui</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--accent:#0072b2;
--ok:#009e73;--warn:#e69f00;--bad:#d55e00;--idle:#8a8a8f;--track:#ececea;--field:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1d1d20;--ink:#ececee;--muted:#9a9aa2;
--line:#2e2e33;--accent:#56b4e9;--ok:#2fc39a;--warn:#f0b43c;--bad:#ff7a3d;--idle:#7d7d85;--track:#2a2a2f;--field:#161618}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;gap:14px;align-items:center;padding:10px 20px;border-bottom:1px solid var(--line);
background:var(--panel);position:sticky;top:0;z-index:2;flex-wrap:wrap}
h1{font-size:16px;margin:0}nav button{border:0;background:none;color:var(--muted);font:inherit;
padding:6px 10px;border-radius:6px;cursor:pointer}nav button.on{background:var(--track);color:var(--ink);font-weight:600}
main{padding:16px 20px;max-width:1500px;margin:0 auto}.tab{display:none}.tab.on{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:14px}
h2{font-size:14px;margin:0 0 10px}h3{font-size:12px;margin:12px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.muted,.note{color:var(--muted)}.note{font-size:12px}
input,select,textarea{font:inherit;color:var(--ink);background:var(--field);border:1px solid var(--line);border-radius:6px;padding:4px 7px}
input[type=checkbox]{padding:0}
textarea{width:100%;font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;min-height:560px;resize:vertical;tab-size:2}
button.b{font:inherit;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:4px 10px;cursor:pointer}
button.b:hover{border-color:var(--accent)}button.p{background:var(--accent);border-color:var(--accent);color:#fff}
button.d{color:var(--bad)}button:disabled{opacity:.4;cursor:default}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.build{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1fr);gap:14px}
@media (max-width:1000px){.build{grid-template-columns:1fr}}
.form{display:grid;grid-template-columns:minmax(150px,auto) 1fr;gap:5px 10px;align-items:center}
.form label{font-size:12.5px;color:var(--muted);text-align:right}.form input,.form select{width:100%}
pre{margin:0;max-height:420px;overflow:auto;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap}
.msg{font-size:13px;padding:6px 10px;border-radius:6px;margin-top:8px;white-space:pre-wrap}
.msg.ok{background:color-mix(in srgb,var(--ok) 15%,transparent)}.msg.err{background:color-mix(in srgb,var(--bad) 15%,transparent)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff;background:var(--idle)}
.badge.running,.badge.starting{background:var(--accent)}.badge.finished,.badge.converged{background:var(--ok)}
.badge.stalled,.badge.stopped{background:var(--warn)}.badge.error{background:var(--bad)}
.bar{height:5px;background:var(--track);border-radius:3px;overflow:hidden;width:120px;margin-top:3px}.bar div{height:100%;background:var(--accent)}
.acts{display:flex;gap:4px;flex-wrap:wrap}.acts button{font-size:12px;padding:2px 7px}
iframe{width:100%;height:calc(100vh - 120px);border:1px solid var(--line);border-radius:10px;background:var(--bg)}
.last{font:11.5px ui-monospace,Menlo,monospace;color:var(--muted);max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style></head><body>
<header><h1>regc gui</h1><nav><button data-t="build">Build</button><button data-t="runs" class="on">Runs</button>
<button data-t="monitor">Monitor</button></nav><span style="flex:1"></span>
<span class="note" id="root"></span></header>
<main>
<!-- ============================== BUILD ============================== -->
<section class="tab" id="t-build">
<div class="card"><div class="row">
<select id="src"></select><button class="b" id="load">Load</button>
<span style="flex:1"></span>
<label class="note">save as (workspace-relative)</label><input id="savepath" size="48" placeholder="examples/my_run/my_run.yaml">
<label class="note"><input type="checkbox" id="overwrite"> overwrite</label>
<button class="b" id="validate">Validate</button><button class="b p" id="save">Save</button>
<button class="b" id="check">Check setup</button><button class="b" id="saverun">Save &amp; start</button></div>
<div id="vmsg"></div></div>
<div class="build">
<div class="card"><h2>Config (YAML)</h2>
<div class="row"><button class="b" id="y2f">YAML → form</button><button class="b" id="f2y">Form → YAML</button>
<span class="note">Form → YAML rewrites the text without comments.</span></div>
<textarea id="yaml" spellcheck="false"></textarea></div>
<div class="card"><h2>Form <span class="note">(common keys; an empty field removes the key)</span></h2>
<div id="form"></div>
<h3>Ladder helpers</h3>
<div class="row"><span class="note">T</span><input id="lt0" size="5" value="300"><span class="note">to</span>
<input id="lt1" size="5" value="900"><span class="note">K, n</span><input id="ltn" size="3" value="8">
<button class="b" id="ltgo">geometric T ladder</button></div>
<div class="row"><span class="note">p</span><input id="lp0" size="7" value="1e-14"><span class="note">to</span>
<input id="lp1" size="7" value="1"><span class="note">n</span><input id="lpn" size="3" value="8">
<button class="b" id="lpgo">log p ladder</button></div>
<h3>Check output</h3><pre id="checkout" class="note">Check setup builds the system with the calculator, prints the μ ladder and the bare-slab energy (regc-run --check). Save first.</pre>
</div></div></section>
<!-- ============================== RUNS ============================== -->
<section class="tab on" id="t-runs">
<div class="card"><div class="row"><b>Runs use</b><span class="note" id="pyexe"></span>
<span class="note">PYTHONPATH</span><input id="pypath" size="70"><button class="b" id="setpp">Set</button>
<span style="flex:1"></span><label class="note">workers</label><input id="workers" size="3" placeholder="cfg">
<label class="note"><input type="checkbox" id="auto" checked> refresh every 5 s</label></div>
<div id="rmsg"></div>
<table><thead><tr><th>config</th><th>state</th><th>progress</th><th>convergence</th><th>subsurface</th><th>actions</th></tr></thead>
<tbody id="rows"></tbody></table></div>
<div class="card" id="logcard" hidden><h2>Log <span class="muted" id="logname"></span>
<button class="b" id="logclose" style="float:right">close</button></h2><pre id="logtext"></pre></div>
</section>
<!-- ============================== MONITOR ============================== -->
<section class="tab" id="t-monitor"><div class="row"><span class="note">The regc monitor; pick the run in its header, or with “Monitor” on the Runs tab.</span>
<a href="/monitor/" target="_blank" id="monlink">open in a new tab</a></div><iframe id="mon" src="about:blank"></iframe></section>
</main>
<script>
const TOKEN="__TOKEN__";const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let WS=null,monId=null;
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Regc-Token":TOKEN},body:JSON.stringify(body)});
const d=await r.json();if(!r.ok||d.error)throw new Error(d.error||r.statusText);return d}
function msg(el,text,ok){$(el).innerHTML=text?`<div class="msg ${ok?"ok":"err"}">${esc(text)}</div>`:""}
function tab(t){document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on",b.dataset.t==t));
document.querySelectorAll(".tab").forEach(s=>s.classList.toggle("on",s.id=="t-"+t));
if(t=="monitor"){const src="/monitor/"+(monId!=null?"#run="+monId:"");if(!$("mon").src.endsWith(src)||$("mon").src=="about:blank")$("mon").src=src;$("monlink").href=src}}
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>tab(b.dataset.t));
// ---------------------------------------------------------------- form
const F=__FIELDS__;
function buildForm(){let h="";for(const f of F){if(f.length==1){h+=`<h3 style="grid-column:1/3">${esc(f[0])}</h3>`;continue}
const [k,l,t,ph]=f,id="f_"+k.replace(/\./g,"__");h+=`<label for="${id}">${esc(l)}</label>`;
if(t.startsWith("sel:"))h+=`<select id="${id}">${t.slice(4).split("|").map(o=>`<option>${o}</option>`).join("")}</select>`;
else if(t=="bool")h+=`<select id="${id}"><option></option><option>true</option><option>false</option></select>`;
else h+=`<input id="${id}" placeholder="${esc(ph||"")}">`}$("form").innerHTML=`<div class="form">${h}</div>`}
const get=(o,k)=>k.split(".").reduce((a,x)=>a&&typeof a=="object"?a[x]:undefined,o);
function set(o,k,v){const p=k.split(".");let a=o;for(let i=0;i<p.length-1;i++){if(typeof a[p[i]]!="object"||a[p[i]]===null){if(v===undefined)return;a[p[i]]={}}a=a[p[i]]}
if(v===undefined){delete a[p.at(-1)];let q=o;const chain=[];for(const x of p.slice(0,-1)){chain.push([q,x]);q=q[x]}
for(let i=chain.length-1;i>=0;i--){const [par,x]=chain[i];if(par[x]&&typeof par[x]=="object"&&!Object.keys(par[x]).length)delete par[x];else break}}else a[p.at(-1)]=v}
function scalar(s){s=s.trim();if(s==="")return undefined;if(s=="true")return true;if(s=="false")return false;
if(/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s))return Number(s);return s}
function show(v){if(v===undefined||v===null)return"";if(Array.isArray(v))return v.map(x=>typeof x=="object"?JSON.stringify(x):x).join(", ");
if(typeof v=="object")return JSON.stringify(v);return String(v)}
const SNAP={};   // each field's text at the last YAML -> form sync
function cfgToForm(c){for(const f of F){if(f.length==1)continue;const el=$("f_"+f[0].replace(/\./g,"__"));const v=get(c,f[0]);
if(el.tagName=="SELECT"&&v!==undefined&&![...el.options].some(o=>o.value==String(v)))el.insertAdjacentHTML("beforeend",`<option>${esc(v)}</option>`);
el.value=show(v);SNAP[f[0]]=el.value}}
function formToCfg(c){for(const f of F){if(f.length==1)continue;const [k,,t]=f,raw=$("f_"+k.replace(/\./g,"__")).value;let v;
// only fields edited since the last sync: YAML edits made since then are kept
if(raw===(SNAP[k]??"")||raw===show(get(c,k)))continue;
if(t=="list")v=raw.trim()?raw.split(/[,\s]+/).filter(Boolean).map(scalar):undefined;
else if(t=="int"){v=scalar(raw);if(typeof v=="number")v=Math.round(v)}
else if(t=="str")v=raw.trim()?raw.trim():undefined;else v=scalar(raw);set(c,k,v)}return c}
let parseError="";
async function parse(){const d=await api("/api/parse",{text:$("yaml").value});
if(d.config===null){parseError=d.error;msg("vmsg",d.error,false);return null}
const s=d.summary;const sum=s?`${s.replicas} replicas (${s.n_T} T × ${s.n_mu} μ/p), n_steps ${s.n_steps}, ~${s.samples_per_replica} samples per replica`:"";
msg("vmsg",d.ok?"Valid. "+sum:d.error+(sum?"\n"+sum:""),d.ok);return d.config}
$("y2f").onclick=async()=>{const c=await parse();if(c)cfgToForm(c)};
$("f2y").onclick=async()=>{try{const c=await parse();
if(!c)return msg("vmsg","Form → YAML not applied: the YAML does not parse, fix it first.\n"+parseError,false);
formToCfg(c);$("yaml").value=(await api("/api/dump",{config:c})).text;const c2=await parse();if(c2)cfgToForm(c2)}
catch(e){msg("vmsg",e.message,false)}};
$("validate").onclick=()=>parse().catch(e=>msg("vmsg",e.message,false));
async function save(){try{if(!$("savepath").value.trim())throw new Error("give a path to save to");
const d=await api("/api/save",{path:$("savepath").value.trim(),text:$("yaml").value,overwrite:$("overwrite").checked});
msg("vmsg","Saved "+d.saved,true);await refreshRuns();loadSources();return d.saved}catch(e){msg("vmsg",e.message,false);return null}}
$("save").onclick=save;
$("saverun").onclick=async()=>{const c=await parse();if(!c)return;const p=await save();if(p){try{await start(p,"run");tab("runs")}catch(e){msg("vmsg",e.message,false)}}};
$("check").onclick=async()=>{try{const p=$("savepath").value.trim();if(!p)throw new Error("save the config first");
const {task}=await api("/api/check",{path:p});$("checkout").textContent="running regc-run --check …";
const poll=async()=>{const d=await api("/api/task/"+task);$("checkout").textContent=d.output+(d.done?`\n[exit ${d.returncode}, ${d.elapsed.toFixed(0)} s]`:`\n… ${d.elapsed.toFixed(0)} s`);if(!d.done)setTimeout(poll,1500)};poll()}
catch(e){$("checkout").textContent=e.message}};
$("ltgo").onclick=()=>{const a=+$("lt0").value,b=+$("lt1").value,n=+$("ltn").value;
if(!(a>0&&b>0&&Number.isInteger(n)&&n>0))return msg("vmsg","T ladder: give positive temperatures and a whole number of points.",false);const T=[...Array(n)].map((_,i)=>Math.round(a*Math.pow(b/a,n>1?i/(n-1):0)));
$("f_replica_exchange__temperatures_K").value=T.join(", ")};
$("lpgo").onclick=()=>{const a=Math.log10(+$("lp0").value),b=Math.log10(+$("lp1").value),n=+$("lpn").value;
if(!(isFinite(a)&&isFinite(b)&&Number.isInteger(n)&&n>0))return msg("vmsg","p ladder: give positive pressures and a whole number of points.",false);
$("f_replica_exchange__pressures").value=[...Array(n)].map((_,i)=>{const e=a+(b-a)*(n>1?i/(n-1):0);return Number.isInteger(+e.toFixed(6))?`1.0e${e<0?"":"+"}${Math.round(e)}`:Math.pow(10,e).toPrecision(3)}).join(", ")};
function loadSources(){if(!WS)return;$("src").innerHTML=`<optgroup label="templates">${WS.templates.map(t=>`<option value="t:${esc(t)}">${esc(t)}</option>`).join("")}</optgroup>`+
`<optgroup label="configs in the workspace">${WS.configs.map(c=>`<option value="c:${esc(c.path)}">${esc(c.path)}</option>`).join("")}</optgroup>`}
async function loadSource(v,toBuild=true){try{if(v.startsWith("t:")){$("yaml").value=(await api("/api/template?name="+encodeURIComponent(v.slice(2)))).text;$("savepath").value="";$("overwrite").checked=false}
else{const d=await api("/api/config?path="+encodeURIComponent(v.slice(2)));$("yaml").value=d.text;$("savepath").value=d.path;$("overwrite").checked=false}
const c=await parse();if(c)cfgToForm(c);if(toBuild)tab("build")}catch(e){msg("vmsg",e.message,false)}}
$("load").onclick=()=>loadSource($("src").value);
// ---------------------------------------------------------------- runs
async function start(path,mode){const w=$("workers").value.trim();const d=await api("/api/start",{path,mode,workers:w?+w:null});
msg("rmsg",`started ${mode}: pid ${d.pid}, log ${d.log}`,true);await refreshRuns();return d}
async function act(path,mode){try{
if(mode=="overwrite"&&!confirm(`Start ${path} over? Its checkpoint, samples and records are replaced.`))return;
if(mode=="stop"){if(!confirm(`Stop the run of ${path}? It can be resumed from its last checkpoint.`))return;
const d=await api("/api/stop",{path});msg("rmsg","sent SIGTERM to "+d.stopped,true);return refreshRuns()}
await start(path,mode)}catch(e){msg("rmsg",e.message,false)}}
async function showLog(path){try{const d=await api("/api/log?path="+encodeURIComponent(path));$("logcard").hidden=false;$("logname").textContent=path;
$("logtext").textContent=d.text||"(empty)";$("logtext").scrollTop=1e9}catch(e){msg("rmsg",e.message,false)}}
$("logclose").onclick=()=>{$("logcard").hidden=true;$("logname").textContent=""};
function row(c,i){const j=c.job,live=["running","starting","stalled"].includes(j.state);
const frac=j.step!=null&&j.n_steps?j.step/j.n_steps:0;const cv=j.convergence;const ss=j.subsurface;
const B=(m,l,dis,cls="")=>`<button class="b ${cls}" data-i="${i}" data-m="${m}"${dis?" disabled":""}>${l}</button>`;
return `<tr><td><b>${esc(c.path)}</b><div class="note">→ ${esc(c.output)}</div><div class="last" title="${esc(j.last_line)}">${esc(j.last_line)}</div></td>
<td><span class="badge ${j.state}">${j.state}${j.external?" (external)":""}</span>${j.pid?`<div class="note">pid ${j.pid}</div>`:""}</td>
<td>${j.step!=null?`${j.step.toLocaleString()} / ${(j.n_steps||0).toLocaleString()}<div class="bar"><div style="width:${(100*frac).toFixed(1)}%"></div></div>`:"–"}</td>
<td>${cv&&cv.max_ratio!=null?`max ${cv.max_ratio.toFixed(2)} · med ${cv.median_ratio.toFixed(2)}<div class="note">n_eff ${cv.min_neff.toFixed(1)}${cv.converged?" · converged":""}</div>`:cv&&cv.max_change!=null?`Δρ ${cv.max_change.toFixed(3)}`:"–"}</td>
<td>${ss?`${ss.n_now} now<div class="note">ins ${ss.inserted}/${ss.insert_attempts} · rem ${ss.removed}/${ss.delete_attempts}</div>`:"–"}</td>
<td><div class="acts">${B("edit","Edit")}${B("run","Start",live||j.resumable,"p")}${B("resume","Resume",live||!j.resumable)}
${B("overwrite","Start over",live||!j.resumable)}${B("analyze","Analyze",live)}${B("stop","Stop",!live||j.external,"d")}${B("monitor","Monitor")}${B("log","Log")}</div></td></tr>`}
async function refreshRuns(){try{WS=await api("/api/workspace");$("root").textContent="workspace "+WS.root;$("pyexe").textContent=WS.python;
if(document.activeElement!==$("pypath"))$("pypath").value=WS.pythonpath;
$("rows").innerHTML=WS.configs.length?WS.configs.map(row).join(""):`<tr><td colspan="6" class="note">No configs found in the workspace. Make one on the Build tab.</td></tr>`;
document.querySelectorAll("#rows button").forEach(b=>b.onclick=()=>{const c=WS.configs[+b.dataset.i],m=b.dataset.m;
if(m=="edit")return loadSource("c:"+c.path);if(m=="monitor"){monId=c.monitor_id;$("mon").src="about:blank";return tab("monitor")}
if(m=="log")return showLog(c.path);act(c.path,m)});
if(!$("logcard").hidden&&$("logname").textContent)showLog($("logname").textContent)}catch(e){msg("rmsg",e.message,false)}}
$("setpp").onclick=async()=>{try{const d=await api("/api/settings",{pythonpath:$("pypath").value});msg("rmsg","runs will use PYTHONPATH="+d.pythonpath,true)}catch(e){msg("rmsg",e.message,false)}};
buildForm();
(async()=>{await refreshRuns();loadSources();await loadSource("t:"+WS.templates[0],false);
setInterval(()=>{if($("auto").checked&&$("t-runs").classList.contains("on"))refreshRuns()},5000)})();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
