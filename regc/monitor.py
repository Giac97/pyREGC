"""
Live monitor: a small local web app showing a running (or finished) simulation.

    python -m regc.monitor OUTPUT_DIR [OUTPUT_DIR,LOGFILE ...] [--port 8050]
    regc-monitor output_fe100_h_eam            # then open http://127.0.0.1:8050

It reads, without touching the run, whatever the output directory holds:

* ``status.json``: the heartbeat a run writes every few seconds
  (`ReplicaExchangeGCMC.save_status`; `regc-run` always writes it): progress,
  per-replica sample counts and averages, recent N and U traces, swap
  statistics, round trips, convergence history;
* a log file (``DIR,LOG`` or any ``*.log`` in the directory): progress lines,
  errors, the last lines;
* ``run_metadata.json``, ``mixing.json``, ``samples.npz`` (checkpointed or
  finished runs), ``records.h5`` (once the run is idle) and the plots of a
  finished analysis.

Runs started before the heartbeat existed are still followed through their
log. The server binds to 127.0.0.1; on a cluster, forward the port
(``ssh -L 8050:localhost:8050 node``). The page loads Plotly from a CDN
(the browser needs internet access, the compute node does not).
"""

from __future__ import annotations

import argparse
import functools
import glob
import ipaddress
import json
import math
import os
import re
import socket
import threading
import time
import warnings
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import numpy as np

from .observables import COMPOSITION_TYPES

STATUS_FILE = "status.json"
STALE_S = 120.0          # no update for this long: the run is not progressing
_SERVED = {".png": "image/png", ".svg": "image/svg+xml", ".json": "application/json",
           ".log": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
           ".yaml": "text/plain; charset=utf-8", ".out": "text/plain; charset=utf-8"}
_PROGRESS = [
    (re.compile(r"^step\s+(\d+)/(\d+) \| mean N across grid =\s*([-+\d.eE]+)"), "regc"),
    (re.compile(r"^cycle\s+(\d+)/(\d+) \| mean N across grid =\s*([-+\d.eE]+)"), "sweep"),
    (re.compile(r"^\s*(\d+)/(\d+) steps\s+[\d.]+ s\s+mean N =\s*([-+\d.eE]+)"), "regc"),
]


def _read_json(path: str) -> Optional[dict]:
    """
    Read a JSON file, None if missing or being replaced.

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    dict or None
        Its content.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_log(path: str, tail: int = 40) -> dict:
    """
    Progress, state and last lines of a run's log.

    Parameters
    ----------
    path : str
        Log file.
    tail : int, optional
        Number of last lines kept. Default 40.

    Returns
    -------
    dict
        ``progress`` (list of ``[step, n_steps, mean_N]``), ``tail``,
        ``finished``, ``error``, ``grid`` (the replica-grid line), ``mtime``.
    """
    out = {"path": path, "progress": [], "tail": [], "finished": False, "error": False,
           "grid": None, "mtime": None}
    try:
        out["mtime"] = os.path.getmtime(path)
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return out
    offset = 0
    for line in lines:
        for rx, _ in _PROGRESS:
            m = rx.match(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                if out["progress"] and step < out["progress"][-1][0] - offset and total != \
                        out["progress"][-1][1]:
                    offset = out["progress"][-1][0]  # a new run_regc call (chunks): continue
                out["progress"].append([step + offset, total, float(m.group(3))])
                break
        if line.startswith("replica grid:"):
            out["grid"] = line
        if line.startswith("Traceback") or re.match(r"^\w*Error\b", line):
            out["error"] = True
        if ("Outputs in" in line or line.startswith("WALL") or line.startswith("Done")
                or "rho(u) converged" in line):
            out["finished"] = True
    out["tail"] = lines[-tail:]
    return out


_CACHE: Dict[str, tuple] = {}


def _samples_summary(path: str, trace_length: int = 400) -> Optional[dict]:
    """
    Per-replica counts, means and traces from a saved ``samples.npz``.

    Parameters
    ----------
    path : str
        The file.
    trace_length : int, optional
        Points per trace. Default 400.

    Returns
    -------
    dict or None
        ``keys``, ``n_samples``, ``mean_N``, ``std_N``, ``mean_U``,
        ``traces`` (``N``, ``U``, ``stride``), ``mtime``, and for a
        semi-grand-canonical substrate ``composition`` (as in the status
        heartbeat, without the current values and identity acceptance);
        None if unreadable.
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    hit = _CACHE.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        with np.load(path, allow_pickle=False) as d:
            U = np.asarray(d["U"], dtype=float)
            N = np.asarray(d["N"], dtype=float)
            keys = np.asarray(d["keys"], dtype=int)
            obs_meta = json.loads(str(d["observables_meta"])) if "observables_meta" in d else {}
            obs = {n: np.asarray(d["obs:" + n], dtype=float) for n, m in obs_meta.items()
                   if m.get("type") in ("SGCCounts",) + COMPOSITION_TYPES and "obs:" + n in d}
    except (OSError, ValueError, KeyError):
        return None
    U, N = U.reshape(len(keys), -1), N.reshape(len(keys), -1)
    stride = max(1, int(np.ceil(N.shape[1] / trace_length)))
    out = {"mtime": mtime, "keys": keys.tolist(), "n_samples": int(N.shape[1]),
           "mean_N": N.mean(axis=1).tolist() if N.size else [],
           "std_N": N.std(axis=1).tolist() if N.size else [],
           "mean_U": U.mean(axis=1).tolist() if U.size else [],
           "traces": {"stride": stride, "N": N[:, ::stride].tolist(), "U": U[:, ::stride].tolist()}}
    out["composition"] = _composition_from_samples(obs_meta, obs, keys, stride)
    _CACHE[path] = (mtime, out)
    return out


def _composition_from_samples(obs_meta: dict, obs: dict, keys: np.ndarray,
                              stride: int) -> Optional[dict]:
    """
    The composition block of the page, from the saved samples.

    Parameters
    ----------
    obs_meta : dict
        ``observables_meta`` of ``samples.npz``.
    obs : dict
        ``{name: array (n_keys, n_samples, size)}`` of the composition observables.
    keys : numpy.ndarray
        Grid point of each row, shape (n_keys, 2).
    stride : int
        Trace subsampling.

    Returns
    -------
    dict or None
        Like ``status['composition']``, with ``current_*`` and
        ``identity_acceptance`` grids absent; None without
        semi-grand-canonical counts.
    """
    from .observables import composition_fractions
    f = composition_fractions(obs_meta, obs)
    if f is None or not len(keys):
        return None
    shape = (int(keys[:, 0].max()) + 1, int(keys[:, 1].max()) + 1)

    def grid(values):
        z = [[None] * shape[1] for _ in range(shape[0])]
        for (i, j), v in zip(keys, values):
            z[int(i)][int(j)] = None if not np.isfinite(v) else float(v)
        return z

    def traces(a):
        return [[None if not np.isfinite(x) else float(x) for x in row[::stride]] for row in a]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows
        grids = {"mean_swappable": grid(np.nanmean(f["swappable"], axis=1))}
        if f["top"] is not None:
            grids["mean_top"] = grid(np.nanmean(f["top"], axis=1))
    return {"species": f["species"], "sgc_species": f["sgc_species"], "dmu": f["dmu"],
            "grids": grids, "traces": {"stride": stride, "swappable": traces(f["swappable"]),
                                       "top": None if f["top"] is None else traces(f["top"])}}


def _records_summary(path: str) -> dict:
    """
    Size, row count and potentials of a ``records.h5`` (opened read-only, unlocked).

    Parameters
    ----------
    path : str
        The file.

    Returns
    -------
    dict
        ``size``, and ``rows``, ``samples``, ``configurations``,
        ``potentials`` (with the evaluated count of each) if readable.
    """
    out = {"size": os.path.getsize(path)}
    try:
        import h5py
        with h5py.File(path, "r", locking=False) as f:
            if "samples" not in f:
                return out
            rows = int(f["samples/sample"].shape[0])
            st, sm = f["samples/state"][:], f["samples/sample"][:]
            out.update(rows=rows, samples=len({(a, b, c) for (a, b), c in zip(st, sm)}),
                       configurations=int((f["config/offset"][:] >= 0).sum()),
                       sampling_potential=str(f.attrs.get("sampling_potential", "")),
                       potentials={k: int(np.isfinite(v[:]).sum())
                                   for k, v in f["samples/E"].items()})
    except Exception as err:  # noqa: BLE001 - being written, or h5py missing
        out["note"] = f"not readable now ({type(err).__name__})"
    return out


class Run:
    """
    One monitored simulation: an output directory and an optional log.

    Parameters
    ----------
    directory : str
        The run's output directory (may not exist yet).
    log : str, optional
        Its log file. Default None: the newest ``*.log`` in the directory.
    """

    def __init__(self, directory: str, log: Optional[str] = None):
        """
        Store the paths.

        Parameters
        ----------
        directory, log
            See the class docstring.
        """
        self.directory = os.path.abspath(directory)
        self.log = os.path.abspath(log) if log else None
        self.name = os.path.basename(self.directory.rstrip(os.sep))
        self.history: deque = deque(maxlen=2000)   # (time, step, mean N) seen by this monitor
        self._lock = threading.Lock()

    def _log_path(self) -> Optional[str]:
        """
        The log to follow.

        Returns
        -------
        str or None
            `log`, or the newest ``*.log`` in the directory.
        """
        if self.log:
            return self.log
        logs = sorted(glob.glob(os.path.join(self.directory, "*.log")), key=os.path.getmtime)
        return logs[-1] if logs else None

    def collect(self) -> dict:
        """
        Everything the page shows about this run, read now.

        Returns
        -------
        dict
            ``name``, ``directory``, ``state`` (running; stalled: no
            heartbeat for `STALE_S`; quiet: a run without heartbeat whose log
            has not changed for `STALE_S`, normal between its progress lines;
            finished; error; waiting), ``status`` (the heartbeat), ``log``,
            ``metadata``, ``mixing``, ``samples``, ``records``, ``files``,
            ``images``, ``history``, ``rate`` (steps/s), ``eta_s`` and
            ``composition`` (semi-grand-canonical substrate, else None).
        """
        now = time.time()
        d = self.directory
        status = _read_json(os.path.join(d, STATUS_FILE))
        log_path = self._log_path()
        log = parse_log(log_path) if log_path else None
        files = []
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                files.append({"name": name + ("/" if os.path.isdir(p) else ""),
                              "size": st.st_size, "mtime": st.st_mtime})
        last_update = max([t for t in (status and status.get("time"),
                                       log and log["mtime"]) if t] or [0.0])
        finished = bool((log and log["finished"]) or (status and status.get("converged"))
                        or (status and status["n_steps"] and status["step"] >= status["n_steps"]
                            and (not log or now - last_update > 30)))
        if log and log["error"] and not finished:
            state = "error"
        elif finished:
            state = "finished"
        elif last_update == 0.0:
            state = "waiting"
        elif now - last_update > STALE_S:
            # a heartbeat comes every few seconds; a log line may come every 10 % of the run
            state = "stalled" if status else "quiet"
        else:
            state = "running"
        step = n_steps = mean_n = None
        if status:
            step, n_steps = status["step"], status["n_steps"]
            cur = [x for row in status["replicas"]["current_N"] for x in row if x is not None]
            mean_n = float(np.mean(cur)) if cur else None
        elif log and log["progress"]:
            step, n_steps, mean_n = log["progress"][-1]
        with self._lock:
            if step is not None and (not self.history or self.history[-1][1] != step):
                self.history.append((now, step, mean_n))
            hist = list(self.history)
        rate = eta = None
        if status and status["step"] > 0 and status.get("started"):
            el = status["time"] - status["started"]
            # a resumed run: its own steps only (offset unknown here), so prefer the history
            rate = status["step"] / el if el > 0 else None
        recent = [h for h in hist if now - h[0] < 1800]
        if len(recent) >= 2 and recent[-1][1] > recent[0][1]:
            rate = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])
        if rate and step is not None and n_steps and state in ("running", "quiet"):
            eta = max(0.0, (n_steps - step) / rate)
        samples_path = os.path.join(d, "samples.npz")
        samples = _samples_summary(samples_path) if os.path.isfile(samples_path) else None
        rec_path = (status or {}).get("records", {}) or {}
        rec_path = rec_path.get("path") or os.path.join(d, "records.h5")
        records = None
        if os.path.isfile(rec_path):
            idle = state in ("finished", "stalled", "error")  # not "quiet": may be writing
            records = (_records_summary(rec_path) if idle else
                       {"size": os.path.getsize(rec_path), "note": "being written; summary when idle"})
        images = [f["name"] for f in files if f["name"].endswith((".png", ".svg"))]
        # substrate composition: live from the heartbeat, else from the last checkpoint
        composition = (status or {}).get("composition")
        if composition:
            composition = dict(composition, source="status.json (live)", keys=status["keys"])
        elif samples and samples.get("composition"):
            composition = dict(samples["composition"], keys=samples["keys"],
                               source=f"samples.npz (last checkpoint, {samples['n_samples']} "
                                      f"samples per replica)")
        return {
            "name": self.name, "directory": d, "now": now, "state": state,
            "last_update": last_update or None, "step": step, "n_steps": n_steps,
            "mean_N": mean_n, "rate": rate, "eta_s": eta, "status": status, "log": log,
            "metadata": _read_json(os.path.join(d, "run_metadata.json")),
            "mixing": _read_json(os.path.join(d, "mixing.json")),
            "samples": samples, "records": records, "files": files, "images": images,
            "composition": composition or None,
            "history": [[t, s, m] for t, s, m in hist],
        }


def discover(paths: List[str]) -> List[Run]:
    """
    Runs from command-line arguments.

    Parameters
    ----------
    paths : list of str
        ``DIR`` or ``DIR,LOG``. A directory holding no run but run
        subdirectories (with ``status.json``, ``run_metadata.json`` or
        ``samples.npz``) contributes each of them.

    Returns
    -------
    list of Run
        The runs.
    """
    runs = []
    markers = (STATUS_FILE, "run_metadata.json", "samples.npz", "records.h5")
    for arg in paths:
        directory, _, log = arg.partition(",")
        is_run = (not os.path.isdir(directory) or log
                  or any(os.path.exists(os.path.join(directory, m)) for m in markers)
                  or glob.glob(os.path.join(directory, "*.log")))
        subs = [] if is_run else sorted(
            p for p in glob.glob(os.path.join(directory, "*"))
            if os.path.isdir(p) and any(os.path.exists(os.path.join(p, m)) for m in markers))
        if subs:
            runs.extend(Run(p) for p in subs)
        else:
            runs.append(Run(directory, log or None))
    return runs


def _finite(data):
    """
    A copy of JSON-like data with NaN and infinities replaced by None.

    Parameters
    ----------
    data : object
        Nested dicts, lists, tuples, numpy arrays and scalars.

    Returns
    -------
    object
        The same structure with plain Python values.
    """
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {k: _finite(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_finite(v) for v in data]
    if isinstance(data, np.ndarray):
        return _finite(data.tolist())
    if isinstance(data, np.generic):
        return _finite(data.item())
    return data


def _json_bytes(data) -> bytes:
    """
    JSON with NaN and infinities as null (strings are left alone).

    Parameters
    ----------
    data : object
        Payload.

    Returns
    -------
    bytes
        UTF-8 JSON.
    """
    return json.dumps(_finite(data), allow_nan=False,
                      default=lambda o: _finite(float(o))).encode()


_WILDCARD = ("", "0.0.0.0", "::")


@functools.lru_cache(maxsize=16)
def allowed_hosts(bound: str, port: int) -> frozenset:
    """
    The ``Host`` header values a local server answers (against DNS rebinding).

    Parameters
    ----------
    bound : str
        The interface the server was asked to bind (``--host``).
    port : int
        Its port.

    Returns
    -------
    frozenset of str
        Lower-case ``name:port`` forms: the loopback names, plus the bound
        host if it is a specific one, or this machine's names if it is a
        wildcard address.
    """
    names = {"127.0.0.1", "localhost", "[::1]"}
    if bound in _WILDCARD:
        names |= {socket.gethostname(), socket.getfqdn()}
    else:
        names.add(f"[{bound}]" if ":" in bound else bound)
    forms = {f"{n}:{port}".lower() for n in names}
    if port == 80:
        forms |= {n.lower() for n in names}
    return frozenset(forms)


def host_allowed(host: Optional[str], bound: str, port: int) -> bool:
    """
    Whether a request's ``Host`` header names this server.

    A page served from another name (a DNS-rebinding attack) sends that
    name, so it is refused. With a wildcard bind any IP literal with the
    right port is accepted too (an address cannot be rebound).

    Parameters
    ----------
    host : str or None
        The ``Host`` header.
    bound : str
        The interface the server was asked to bind.
    port : int
        Its port.

    Returns
    -------
    bool
        True if allowed.
    """
    h = (host or "").strip().lower()
    if not h:
        return False
    if h in allowed_hosts(bound, port):
        return True
    if bound in _WILDCARD:
        name, _, p = h.rpartition(":")
        if p == str(port):
            try:
                ipaddress.ip_address(name[1:-1] if name.startswith("[") else name)
                return True
            except ValueError:
                pass
    return False


def origin_allowed(origin: Optional[str], bound: str, port: int) -> bool:
    """
    Whether a request's ``Origin`` header (if any) is this server.

    Parameters
    ----------
    origin : str or None
        The ``Origin`` header (absent for same-origin GETs and non-browser clients).
    bound : str
        The interface the server was asked to bind.
    port : int
        Its port.

    Returns
    -------
    bool
        True if absent or ``http://`` plus an allowed host.
    """
    if origin is None:
        return True
    o = origin.strip()
    return o.lower().startswith("http://") and host_allowed(o[7:], bound, port)


def server_host(server) -> Tuple[str, int]:
    """
    The interface a server was asked to bind and its port.

    Parameters
    ----------
    server : socketserver.BaseServer
        The server (``regc_host`` set by `serve`, else its bound address).

    Returns
    -------
    tuple
        ``(host, port)``.
    """
    return getattr(server, "regc_host", server.server_address[0]), server.server_address[1]


def route(path: str, runs: List[Run], plotly_js: Optional[str] = None) -> Tuple[bytes, str, int]:
    """
    Answer one monitor GET request (also mounted under ``/monitor/`` by `regc.gui`).

    Parameters
    ----------
    path : str
        Unquoted request path without the query: ``/``, ``/plotly.min.js``,
        ``/api/runs``, ``/api/run/<i>`` or ``/file/<i>/<name>``.
    runs : list of Run
        The monitored runs (the list may grow while the server runs).
    plotly_js : str, optional
        Local ``plotly.min.js``. Default None.

    Returns
    -------
    tuple
        ``(body, content type, HTTP status)``.
    """
    if path in ("/", "/index.html"):
        return PAGE.encode(), "text/html; charset=utf-8", 200
    if path == "/plotly.min.js":
        if plotly_js and os.path.isfile(plotly_js):
            with open(plotly_js, "rb") as f:
                return f.read(), "text/javascript", 200
        return b"", "text/javascript", 404
    if path == "/api/runs":
        return _json_bytes([{"id": i, "name": r.name, "directory": r.directory,
                             "log": r._log_path()} for i, r in enumerate(runs)]), \
            "application/json", 200
    m = re.match(r"^/api/run/(\d+)$", path)
    if m and int(m.group(1)) < len(runs):
        try:
            return _json_bytes(runs[int(m.group(1))].collect()), "application/json", 200
        except Exception as err:  # noqa: BLE001 - report, keep serving
            return _json_bytes({"error": f"{type(err).__name__}: {err}"}), "application/json", 500
    m = re.match(r"^/file/(\d+)/([^/]+)$", path)
    if m and int(m.group(1)) < len(runs):
        name = os.path.basename(m.group(2))
        ext = os.path.splitext(name)[1].lower()
        full = os.path.join(runs[int(m.group(1))].directory, name)
        if ext in _SERVED and os.path.isfile(full):
            with open(full, "rb") as f:
                return f.read(), _SERVED[ext], 200
    return b"not found", "text/plain", 404


def make_handler(runs: List[Run], plotly_js: Optional[str] = None):
    """
    The HTTP request handler class for a list of runs.

    Parameters
    ----------
    runs : list of Run
        The monitored runs.
    plotly_js : str, optional
        A local ``plotly.min.js`` served as ``/plotly.min.js`` (for a
        browser without internet access). Default None: the page loads
        Plotly from a CDN.

    Returns
    -------
    type
        A `BaseHTTPRequestHandler` subclass.
    """

    class Handler(BaseHTTPRequestHandler):
        """Routes: ``/``, ``/api/runs``, ``/api/run/<i>``, ``/file/<i>/<name>``."""

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

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            """Serve one request (only for this server's own host names)."""
            if not host_allowed(self.headers.get("Host"), *server_host(self.server)):
                return self._send(b"forbidden host", "text/plain", 403)
            body, ctype, code = route(unquote(self.path.split("?", 1)[0]), runs, plotly_js)
            self._send(body, ctype, code)

    return Handler


def serve(paths: List[str], host: str = "127.0.0.1", port: int = 8050,
          open_browser: bool = False, plotly_js: Optional[str] = None) -> ThreadingHTTPServer:
    """
    Start the monitor (blocking until interrupted).

    Parameters
    ----------
    paths : list of str
        ``DIR`` or ``DIR,LOG`` (see `discover`).
    host : str, optional
        Interface. Default ``'127.0.0.1'`` (this machine only).
    port : int, optional
        Port. Default 8050.
    open_browser : bool, optional
        Open the page in a browser. Default False.
    plotly_js : str, optional
        Local ``plotly.min.js`` (see `make_handler`). Default None.

    Returns
    -------
    ThreadingHTTPServer
        The server (after it was shut down).
    """
    runs = discover(paths)
    httpd = ThreadingHTTPServer((host, port), make_handler(runs, plotly_js))
    httpd.regc_host = host
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"regc monitor: {len(runs)} run(s) at {url}  (Ctrl-C to stop)", flush=True)
    for r in runs:
        print(f"  {r.name}: {r.directory}" + (f"  log {r._log_path()}" if r._log_path() else ""))
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd


def main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``regc-monitor DIR [DIR,LOG ...]``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        0.
    """
    ap = argparse.ArgumentParser(prog="regc-monitor",
                                 description="Live web monitor of regc runs.")
    ap.add_argument("runs", nargs="+",
                    help="output directories (DIR, or DIR,LOGFILE); a directory of runs works too")
    ap.add_argument("--port", type=int, default=8050, help="port (default 8050)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="interface (default 127.0.0.1; 0.0.0.0 exposes it to the network)")
    ap.add_argument("--open", action="store_true", help="open a browser")
    ap.add_argument("--plotly-js", default=None,
                    help="local plotly.min.js, for a browser without internet access")
    args = ap.parse_args(argv)
    serve(args.runs, args.host, args.port, args.open, args.plotly_js)
    return 0


PAGE = r"""<!doctype html>
<html lang="en" data-theme="auto"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>regc monitor</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Fira+Sans:wght@400;500;600;700&family=VT323&display=swap">
<script src="plotly.min.js"></script>
<script>if(typeof Plotly==="undefined")document.write('<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"><\/script>')</script>
<script>if(typeof Plotly==="undefined")document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"><\/script>')</script>
<script>try{const t=localStorage.getItem("regc.theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
<style>
/* ---- tokens: light, dark (auto follows the system), CRT phosphors ------------------------ */
:root{--bg:#f3f5f9;--panel:#ffffff;--panel2:#f7f9fc;--ink:#0f172a;--muted:#55607a;--line:#e1e6ef;
--accent:#0a6cc2;--accent-ink:#ffffff;--ok:#12a150;--warn:#b77900;--bad:#d1432c;--idle:#6f7a90;--track:#e6ebf3;
--shadow:0 1px 2px rgba(15,23,42,.06),0 4px 16px rgba(15,23,42,.05);
--sans:"Fira Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"Fira Code",ui-monospace,Menlo,Consolas,monospace;
--crt:"VT323","Courier New",ui-monospace,monospace;--radius:12px}
:root[data-theme="dark"]{--bg:#0b1020;--panel:#121a2b;--panel2:#162036;--ink:#e6ebf5;--muted:#9aa7c0;--line:#24304a;
--accent:#4cc2ff;--accent-ink:#06121f;--ok:#2fd37a;--warn:#f5b83d;--bad:#ff6b5a;--idle:#7c879c;--track:#1d2840;
--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.25)}
@media (prefers-color-scheme:dark){:root[data-theme="auto"]{--bg:#0b1020;--panel:#121a2b;--panel2:#162036;--ink:#e6ebf5;
--muted:#9aa7c0;--line:#24304a;--accent:#4cc2ff;--accent-ink:#06121f;--ok:#2fd37a;--warn:#f5b83d;--bad:#ff6b5a;--idle:#7c879c;
--track:#1d2840;--shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.25)}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans);transition:background .2s,color .2s}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:6px}
button,select,input{font:inherit;color:var(--ink)}
/* ---- header ------------------------------------------------------------------------------- */
header{position:sticky;top:0;z-index:10;display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;
padding:10px 24px;background:color-mix(in srgb,var(--panel) 88%,transparent);backdrop-filter:blur(10px);
border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;letter-spacing:.01em;white-space:nowrap}
.brand svg{width:26px;height:26px;flex:none}
.sel{appearance:none;background:var(--panel2) url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2388a' stroke-width='2.5'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E") no-repeat right 10px center;
border:1px solid var(--line);border-radius:8px;padding:6px 30px 6px 10px;min-height:36px;cursor:pointer;max-width:min(52vw,420px);
text-overflow:ellipsis;transition:border-color .15s}
.sel:hover{border-color:var(--accent)}
.pill{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:4px 12px 4px 10px;font-size:12px;font-weight:600;
text-transform:uppercase;letter-spacing:.06em;color:#fff;white-space:nowrap}
.pill i{width:8px;height:8px;border-radius:50%;background:currentColor;opacity:.9}
.running{background:var(--ok)}.finished{background:var(--accent);color:var(--accent-ink)}.stalled{background:var(--warn);color:#1a1200}
.error{background:var(--bad)}.waiting,.quiet{background:var(--idle)}
.running i{animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(255,255,255,.7)}50%{box-shadow:0 0 0 5px rgba(255,255,255,0)}}
.updated{color:var(--muted);font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40ch}
.spacer{flex:1}
.toggle{display:inline-flex;align-items:center;gap:8px;cursor:pointer;color:var(--muted);font-size:13px;white-space:nowrap;
min-height:36px;user-select:none}
.toggle input{appearance:none;width:34px;height:20px;border-radius:999px;background:var(--track);position:relative;cursor:pointer;
transition:background .2s;flex:none;margin:0}
.toggle input::after{content:"";position:absolute;top:3px;left:3px;width:14px;height:14px;border-radius:50%;background:#fff;
box-shadow:0 1px 2px rgba(0,0,0,.3);transition:transform .2s}
.toggle input:checked{background:var(--accent)}.toggle input:checked::after{transform:translateX(14px)}
.iconbtn{display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:8px;cursor:pointer;
background:var(--panel2);border:1px solid var(--line);transition:border-color .15s,transform .15s}
.iconbtn:hover{border-color:var(--accent)}.iconbtn svg{width:18px;height:18px}
/* ---- layout ------------------------------------------------------------------------------- */
main{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:16px;padding:18px 24px 28px;max-width:1800px;margin:0 auto}
.card{grid-column:span 12;background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px 12px;
box-shadow:var(--shadow);min-width:0;display:flex;flex-direction:column}
.s8{grid-column:span 8}.s7{grid-column:span 7}.s6{grid-column:span 6}.s5{grid-column:span 5}.s4{grid-column:span 4}
@media (max-width:1180px){.s8,.s7,.s6,.s5,.s4{grid-column:span 6}}
@media (max-width:760px){main{grid-template-columns:1fr;padding:12px}.card,.s8,.s7,.s6,.s5,.s4{grid-column:1/-1}
header{padding:10px 12px}.updated{max-width:none;flex-basis:100%;order:9}}
.card-h{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin:0 0 8px;min-width:0}
.card-h h2{font-size:12px;margin:0;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
.sub{font-size:12px;color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.note{font-size:12.5px;color:var(--muted)}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.plot{width:100%;height:300px}.plot.tall{height:380px}.plot.short{height:200px}
/* ---- progress ----------------------------------------------------------------------------- */
.prog{display:flex;align-items:center;gap:14px;margin:4px 0 14px}
.bar{flex:1;height:10px;background:var(--track);border-radius:999px;overflow:hidden}
.bar>div{height:100%;background:linear-gradient(90deg,var(--accent),color-mix(in srgb,var(--accent) 60%,var(--ok)));
border-radius:999px;transition:width .6s ease}
.pct{font:600 15px var(--mono);font-variant-numeric:tabular-nums;min-width:6ch;text-align:right}
.kpis{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
.kpi b{display:block;font:600 19px/1.25 var(--sans);font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi span{display:block;font-size:12px;color:var(--muted);margin-top:2px;overflow-wrap:anywhere}
.where{margin-top:10px;font:12px var(--mono);color:var(--muted);display:flex;gap:16px;flex-wrap:wrap}
.where span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}
/* ---- tables ------------------------------------------------------------------------------- */
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;white-space:nowrap}
td.num,th.num{text-align:right;white-space:nowrap}td.nw{white-space:nowrap}
td.file{font-family:var(--mono);font-size:12.5px;white-space:nowrap}
tr:hover td{background:var(--panel2)}
.imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
.imgs a{color:inherit;text-decoration:none;min-width:0}.imgs img{width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;
transition:transform .15s}.imgs a:hover img{transform:translateY(-2px)}
.imgs .note{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:4px}
/* ---- log, modern ---------------------------------------------------------------------------- */
.logbar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:10px}
.chip{font:12px var(--mono);color:var(--muted);background:var(--panel2);border:1px solid var(--line);border-radius:6px;
padding:2px 8px;max-width:min(60vw,640px);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.screen{position:relative;border-radius:10px;background:var(--panel2);border:1px solid var(--line)}
pre#log{margin:0;height:380px;overflow:auto;padding:12px 14px;font:12.5px/1.55 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere}
pre#log.nowrap{white-space:pre;overflow-wrap:normal}
.bezel-label,.led{display:none}
.phos{display:none}
/* ---- log, vintage CRT (1960s/70s video terminal) ---------------------------------------------- */
.crt-on{--ph:#41ff7a;--ph-dim:#1f9f4a;--ph-glow:rgba(65,255,122,.45);--glass:#0d1a10,#050906,#010201}
.crt-on.ph-amber{--ph:#ffb238;--ph-dim:#a8701a;--ph-glow:rgba(255,178,56,.42)}
.crt-on.ph-white{--ph:#e9f1ff;--ph-dim:#8995a8;--ph-glow:rgba(205,222,255,.35)}
.crt-on .phos{display:inline-flex}
.crt-on .bezel{padding:22px 24px 34px;border-radius:26px;position:relative;
background:linear-gradient(160deg,#e3dbc6 0%,#cfc6ae 45%,#b9b099 100%);
box-shadow:inset 0 2px 0 rgba(255,255,255,.6),inset 0 -3px 0 rgba(0,0,0,.18),0 10px 30px rgba(0,0,0,.35)}
.crt-on .bezel-label{display:block;position:absolute;left:28px;bottom:9px;font:600 11px/1 var(--sans);letter-spacing:.22em;
color:#6a6352;text-shadow:0 1px 0 rgba(255,255,255,.55)}
.crt-on .led{display:block;position:absolute;right:30px;bottom:11px;width:9px;height:9px;border-radius:50%;
background:var(--ph);box-shadow:0 0 6px var(--ph),0 0 14px var(--ph-glow)}
.crt-on .screen{border:0;border-radius:22px/30px;overflow:hidden;
background:radial-gradient(ellipse at 50% 45%,#0d1a10 0%,#050906 62%,#010201 100%);
box-shadow:inset 0 0 70px rgba(0,0,0,.95),inset 0 0 10px rgba(0,0,0,.9),0 0 0 3px #2c2a24,0 0 0 6px #191814}
.crt-on.ph-amber .screen{background:radial-gradient(ellipse at 50% 45%,#1d1406 0%,#0b0703 62%,#020100 100%)}
.crt-on.ph-white .screen{background:radial-gradient(ellipse at 50% 45%,#141922 0%,#07090d 62%,#010102 100%)}
.crt-on pre#log{font:20px/1.18 var(--crt);color:var(--ph);padding:22px 26px;height:420px;letter-spacing:.02em;
text-shadow:0 0 1px var(--ph),0 0 6px var(--ph-glow),0 0 14px var(--ph-glow);scrollbar-color:var(--ph-dim) transparent;
animation:flicker 4s infinite steps(1)}
.crt-on pre#log::selection{background:var(--ph);color:#000}
.crt-on .screen::before{content:"";position:absolute;inset:0;pointer-events:none;z-index:2;
background:repeating-linear-gradient(to bottom,rgba(0,0,0,.32) 0,rgba(0,0,0,.32) 1px,transparent 2px,transparent 3px)}
.crt-on .screen::after{content:"";position:absolute;inset:0;pointer-events:none;z-index:3;
background:radial-gradient(ellipse at center,transparent 55%,rgba(0,0,0,.55) 100%),
linear-gradient(to bottom,rgba(255,255,255,.05),transparent 30%);mix-blend-mode:normal}
.crt-on .cursor{display:inline-block;width:.6em;height:1em;vertical-align:-.12em;background:var(--ph);box-shadow:0 0 8px var(--ph-glow);
animation:blink 1.05s steps(1) infinite}
.cursor{display:none}
.crt-on .screen.boot pre#log{animation:boot .5s ease-out,flicker 4s infinite steps(1)}
@keyframes blink{50%{opacity:0}}
@keyframes flicker{0%{opacity:1}47%{opacity:.985}48%{opacity:1}83%{opacity:.975}84%{opacity:1}}
@keyframes boot{0%{transform:scaleY(.01);filter:brightness(3)}60%{transform:scaleY(1);filter:brightness(1.6)}100%{filter:none}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
.phos select{min-height:30px;padding:3px 28px 3px 8px}
</style></head><body>
<header>
<div class="brand"><svg viewBox="0 0 32 32" aria-hidden="true"><rect x="2" y="2" width="28" height="28" rx="7" fill="var(--accent)"/>
<g fill="var(--accent-ink)" opacity=".95"><rect x="8" y="8" width="5" height="5" rx="1"/><rect x="15" y="8" width="5" height="5" rx="1" opacity=".6"/>
<rect x="22" y="8" width="3" height="5" rx="1" opacity=".35"/><rect x="8" y="15" width="5" height="5" rx="1" opacity=".6"/>
<rect x="15" y="15" width="5" height="5" rx="1"/><rect x="22" y="15" width="3" height="5" rx="1" opacity=".6"/>
<rect x="8" y="22" width="5" height="3" rx="1" opacity=".35"/><rect x="15" y="22" width="5" height="3" rx="1" opacity=".6"/>
<rect x="22" y="22" width="3" height="3" rx="1"/></g></svg><span>regc monitor</span></div>
<label class="visually-hidden" for="run" style="position:absolute;left:-9999px">run</label>
<select id="run" class="sel" aria-label="run"></select>
<span id="badge" class="pill waiting" role="status"><i></i><span id="badgetxt">…</span></span>
<span id="updated" class="updated"></span><span class="spacer"></span>
<label class="toggle"><input type="checkbox" id="auto" checked> auto-refresh 5 s</label>
<button class="iconbtn" id="theme" type="button" aria-label="colour theme" title="colour theme"></button>
</header>
<main>
<section class="card"><div class="card-h"><h2>Progress</h2><span class="sub" id="stepline"></span></div>
<div class="prog"><div class="bar" role="progressbar" aria-label="run progress"><div id="pbar" style="width:0"></div></div><span class="pct" id="pct">0 %</span></div>
<div class="kpis" id="kpis"></div><div class="where" id="where"></div></section>

<section class="card s7"><div class="card-h"><h2>Replica grid</h2><span class="sub">click a cell to show its traces</span></div>
<div class="row"><select id="gridq" class="sel" aria-label="grid quantity">
<option value="mean_N">mean N</option><option value="current_N">current N</option>
<option value="std_N">std of N</option><option value="n_samples">samples</option>
<option value="mean_U">mean U (eV)</option><option value="current_U">current U (eV)</option>
<option value="mean_N_ads">mean N adsorbed</option><option value="mean_N_sub">mean N subsurface</option>
<option value="current_N_sub">current N subsurface</option><option value="sub_accepted">subsurface moves accepted</option>
<option value="sub_acceptance">subsurface acceptance</option></select></div>
<div id="grid" class="plot tall"></div><div class="note" id="gridnote"></div></section>
<section class="card s5"><div class="card-h"><h2>Traces</h2><span id="tracekey" class="sub"></span></div><div id="traces" class="plot tall"></div></section>

<section class="card s7" id="compcard" hidden><div class="card-h"><h2>Substrate composition</h2><span class="sub">click a cell to show its traces</span></div>
<div class="row"><select id="compq" class="sel" aria-label="composition quantity"></select></div>
<div id="compgrid" class="plot tall"></div><div class="note" id="compsrc"></div></section>
<section class="card s5" id="comptrcard" hidden><div class="card-h"><h2>Composition traces</h2><span id="comptrkey" class="sub"></span></div><div id="comptr" class="plot tall"></div></section>

<section class="card s4"><div class="card-h"><h2>Mean N across the grid</h2></div><div id="hist" class="plot"></div></section>
<section class="card s4"><div class="card-h"><h2>Replica exchange</h2><span class="sub">acceptance per swap pattern</span></div>
<div id="swaps" class="plot short"></div><div class="tablewrap" id="pairs"></div></section>
<section class="card s4"><div class="card-h"><h2>Round trips per walker</h2></div><div id="trips" class="plot"></div></section>

<section class="card s6"><div class="card-h"><h2>Convergence of ρ(u)</h2><span class="sub" id="convsub"></span></div><div id="conv" class="plot"></div></section>
<section class="card s6"><div class="card-h"><h2>Available data</h2></div><div class="tablewrap" id="data"></div></section>

<section class="card" id="imgcard" hidden><div class="card-h"><h2>Plots</h2></div><div class="imgs" id="imgs"></div></section>

<section class="card" id="logcard"><div class="logbar"><div class="card-h" style="margin:0"><h2>Log</h2></div>
<span id="logname" class="chip"></span><span class="spacer"></span>
<label class="toggle"><input type="checkbox" id="follow" checked> follow</label>
<label class="toggle"><input type="checkbox" id="wrap" checked> wrap</label>
<label class="toggle" title="1960s/70s video terminal"><input type="checkbox" id="crt"> vintage</label>
<label class="phos toggle">phosphor <select id="phosphor" class="sel" aria-label="phosphor colour">
<option value="green">P1 green</option><option value="amber">P3 amber</option><option value="white">P4 white</option></select></label></div>
<div class="bezel"><div class="screen" id="screen"><pre id="log" tabindex="0" aria-label="run log"></pre></div>
<span class="bezel-label">REGC&nbsp;&nbsp;MODEL&nbsp;1970&nbsp;&nbsp;VIDEO&nbsp;DISPLAY&nbsp;TERMINAL</span><span class="led" aria-hidden="true"></span></div></section>
</main>
<script>
const $=id=>document.getElementById(id);let runs=[],cur=0,sel=null,last=null;
const store={get(k,d){try{const v=localStorage.getItem("regc."+k);return v==null?d:v}catch(e){return d}},
set(k,v){try{localStorage.setItem("regc."+k,v)}catch(e){}}};
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtT=s=>{if(s==null||!isFinite(s))return"–";s=Math.round(s);const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);
return d?`${d} d ${h} h`:h?`${h} h ${m} min`:m?`${m} min ${s%60} s`:`${s} s`};
const fmtB=b=>b>1e9?(b/1e9).toFixed(2)+" GB":b>1e6?(b/1e6).toFixed(1)+" MB":b>1e3?(b/1e3).toFixed(0)+" kB":b+" B";
const fmtN=(x,d=3)=>x==null||!isFinite(x)?"–":Number(x).toPrecision(d).replace("-","−");
const fmtP=x=>{const a=Math.abs(x);return (a&&(a<1e-2||a>=1e4)?Number(x).toExponential(2).replace("e+","e").replace(/e(-?)0*(\d)/,"e$1$2"):Number(x).toPrecision(3)).replace("-","−")};
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const ord=n=>n+(n%100>=11&&n%100<=13?"th":["th","st","nd","rd"][n%10]||"th");
const tail=(p,n=3)=>{if(!p)return"";const a=String(p).split("/").filter(Boolean);return a.length>n?"…/"+a.slice(-n).join("/"):p};
/* ---- theme ---------------------------------------------------------------------------------- */
const ICON={auto:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 0 0 18z" fill="currentColor"/></svg>',
light:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
dark:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'};
function setTheme(t){document.documentElement.dataset.theme=t;store.set("theme",t);$("theme").innerHTML=ICON[t];
$("theme").title=$("theme").ariaLabel=`colour theme: ${t} (click to change)`;if(last)draw(last)}
$("theme").onclick=()=>{const o=["auto","light","dark"],t=document.documentElement.dataset.theme;setTheme(o[(o.indexOf(t)+1)%3])};
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change",()=>last&&draw(last));
/* ---- log controls --------------------------------------------------------------------------- */
function setCrt(on,boot){$("logcard").classList.toggle("crt-on",on);$("crt").checked=on;store.set("crt",on?"1":"0");
if(on&&boot){const s=$("screen");s.classList.remove("boot");void s.offsetWidth;s.classList.add("boot")}renderLog(last)}
$("crt").onchange=()=>setCrt($("crt").checked,true);
$("phosphor").onchange=()=>{const p=$("phosphor").value;$("logcard").classList.remove("ph-amber","ph-white");if(p!="green")$("logcard").classList.add("ph-"+p);store.set("phosphor",p)};
$("wrap").onchange=()=>{$("log").classList.toggle("nowrap",!$("wrap").checked);store.set("wrap",$("wrap").checked?"1":"0")};
$("follow").onchange=()=>{store.set("follow",$("follow").checked?"1":"0");if($("follow").checked)$("log").scrollTop=1e9};
$("log").addEventListener("scroll",()=>{const l=$("log");const atEnd=l.scrollHeight-l.scrollTop-l.clientHeight<24;
if(!atEnd&&$("follow").checked&&!l._prog){$("follow").checked=false;store.set("follow","0")}});
function renderLog(d){const l=$("log");if(!d){return}const crt=$("logcard").classList.contains("crt-on");
const lines=d.log?d.log.tail:["(no log)"];
let text=lines.join("\n");if(crt){text=`REGC RUN MONITOR  ·  ${new Date().toISOString().slice(0,19).replace("T"," ")} UTC\n`+
`${"=".repeat(64)}\n`+text}
const atEnd=l.scrollHeight-l.scrollTop-l.clientHeight<24;l.innerHTML=esc(text)+(crt?'\n<span class="cursor"></span>':"");
if($("follow").checked||atEnd){l._prog=true;l.scrollTop=l.scrollHeight;setTimeout(()=>l._prog=false,50)}
$("logname").textContent=d.log?tail(d.log.path,3):"";$("logname").title=d.log?d.log.path:""}
/* ---- plotly ----------------------------------------------------------------------------------- */
function layout(extra){const ink=css("--ink"),line=css("--line"),mut=css("--muted");
const ax={gridcolor:line,zerolinecolor:line,linecolor:line,automargin:true,tickfont:{color:mut,size:11},title:{standoff:10,font:{color:mut,size:12}}};
const base={margin:{l:10,r:10,t:30,b:10,pad:2},paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",
font:{color:ink,size:12,family:css("--sans")},hoverlabel:{font:{family:css("--mono")}},
xaxis:{...ax},yaxis:{...ax},legend:{orientation:"h",x:0,y:-0.22,yanchor:"top",font:{size:11,color:mut}},hovermode:"closest"};
const e=extra||{};for(const k of Object.keys(e)){if((k.startsWith("xaxis")||k.startsWith("yaxis"))&&typeof e[k]=="object"){
const t=e[k].title;base[k]={...ax,...e[k],title:typeof t=="string"?{...ax.title,text:t}:{...ax.title,...(t||{})}}}else base[k]=e[k]}
return base}
const cfg={displaylogo:false,responsive:true,modeBarButtonsToRemove:["lasso2d","select2d","autoScale2d"]};
const empty=(id,msg)=>{Plotly.purge(id);$(id).innerHTML=`<p class="note" style="padding:24px 4px">${msg}</p>`};
/* ---- data ------------------------------------------------------------------------------------ */
async function loadRuns(){runs=await (await fetch("api/runs")).json();
$("run").innerHTML=runs.map(r=>`<option value="${r.id}">${esc(r.name)}</option>`).join("");
$("run").onchange=()=>{cur=+$("run").value;sel=null;refresh()};fromHash()}
function fromHash(){const m=location.hash.match(/run=(\d+)/);if(m&&runs.some(r=>r.id==+m[1])){cur=+m[1];$("run").value=cur;sel=null}}
window.addEventListener("hashchange",async()=>{await loadRuns();refresh()});
function axes(st,meta){const lab=st&&st.labels||{};const nT=st?st.shape[0]:0,nM=st?st.shape[1]:0;
const TK=lab.temperatures_K||(meta&&meta.temperatures_K);let y=TK?TK.map(t=>(+Number(t).toPrecision(4))+" K"):st.temperatures.map(t=>"T="+fmtN(t,4));
const P=lab.pressures||(meta&&meta.pressures),u=lab.pressure_unit||(meta&&meta.pressure_unit)||"";
const PG=lab.pressure_grid||(meta&&meta.pressure_grid);const CL=lab.column_labels||(meta&&meta.column_labels);
const shortP=s=>s.replace(/\s*(bar|Pa|atm|torr)\b/i,"").replace("e+0","e").replace("e+","e").replace("e-0","e-");
if(CL&&CL.length==nM){let b=0;const x=CL.map((c,j)=>`${j}: ${c.replace(/mu_/g,"μ_")}`);
const xs=CL.map(c=>{const m=c.match(/,\s*mu_(\w+)\s*-\s*([\d.]+)\s*kT/);if(m){b++;return `B${b} −${+m[2]}kT`}return shortP(c)});
return {x,xs,y,xt:`p (${u||"–"}) · B = bias columns, μ shifted`,yt:"T"}}
let x=[],xs=[];for(let j=0;j<nM;j++){let lab_;if(PG&&PG.length==nT){const c=PG.map(r=>r[j]),lo=Math.min(...c),hi=Math.max(...c);
lab_=`${fmtP(lo)}${hi>lo*(1+1e-9)?"–"+fmtP(hi):""}`}else lab_=P?fmtP(P[j]):`μ=${fmtN(Array.isArray(st.mu_grid[0][j])?st.mu_grid[0][j][0]:st.mu_grid[0][j],4)}`;
x.push(`${j}: ${lab_} ${P||PG?u:""}`);xs.push(lab_)}
return {x,xs,y,xt:PG?`p (${u}), range over T: adapted ladder`:P?`p (${u})`:"μ (eV, lowest T)",yt:"T"}}
function draw(d){last=d;const st=d.status;$("badge").className="pill "+d.state;$("badgetxt").textContent=d.state;
$("updated").textContent=d.last_update?`updated ${fmtT(d.now-d.last_update)} ago`+(d.state=="quiet"?" · no heartbeat (log every 10 %)":""):"no data yet";
$("updated").title=$("updated").textContent;
const frac=d.step&&d.n_steps?d.step/d.n_steps:0;$("pbar").style.width=(100*frac).toFixed(2)+"%";$("pct").textContent=(100*frac).toFixed(1)+" %";
$("stepline").textContent=d.step!=null?`step ${d.step.toLocaleString()} of ${d.n_steps.toLocaleString()}`:"";
const nrep=st?st.shape[0]*st.shape[1]:(d.samples?d.samples.keys.length:null);const ns=st?Math.max(...st.replicas.n_samples.flat().filter(x=>x!=null)):(d.samples?d.samples.n_samples:null);
const k=[["rate",d.rate?(d.rate>=0.1?fmtN(d.rate,3)+" steps/s":fmtN(1/d.rate,3)+" s/step"):"–"],
["remaining",d.state=="running"||d.state=="quiet"?fmtT(d.eta_s):"–"],["elapsed",st&&st.started?fmtT(st.time-st.started):"–"],
["mean N now",fmtN(d.mean_N,3)],["replicas",nrep??"–"],["samples / replica",ns??"–"]];
const ss=st&&st.subsurface;if(ss){const sp=ss.species.join("+");k.push([`subsurface ${sp} now (all replicas)`,`${ss.n_now} in ${ss.replicas_occupied}`],
["subsurface insertions accepted",`${ss.inserted.toLocaleString()} / ${ss.insert_attempts.toLocaleString()}`],
["subsurface removals accepted",`${ss.removed.toLocaleString()} / ${ss.delete_attempts.toLocaleString()}`])}
$("kpis").innerHTML=k.map(([a,b])=>`<div class="kpi" title="${esc(a)}: ${esc(b)}"><b>${esc(b)}</b><span>${esc(a)}</span></div>`).join("");
$("where").innerHTML=`<span title="${esc(d.directory)}">${esc(tail(d.directory,4))}</span>`+(st?"":`<span>no status.json: progress from the log only</span>`);
drawGrid(d);drawTraces(d);drawComp(d);drawHist(d);drawSwaps(d);drawConv(d);drawData(d);renderLog(d);
const imgs=d.images||[];$("imgcard").hidden=!imgs.length;
$("imgs").innerHTML=imgs.map(n=>`<a href="file/${cur}/${encodeURIComponent(n)}" target="_blank" rel="noopener"><img loading="lazy" src="file/${cur}/${encodeURIComponent(n)}?t=${Math.round(d.now)}" alt="${esc(n)}"><div class="note" title="${esc(n)}">${esc(n)}</div></a>`).join("");}
function gridData(d){const st=d.status;if(st)return {st,z:st.replicas[$("gridq").value]};
const s=d.samples;if(!s||!d.metadata)return null;const nT=d.metadata.temperatures_K.length,keys=s.keys,
nM=Math.max(...keys.map(k=>k[1]))+1;const q=$("gridq").value,src={mean_N:s.mean_N,std_N:s.std_N,mean_U:s.mean_U}[q];
if(!src)return null;const z=[...Array(nT)].map(()=>Array(nM).fill(null));keys.forEach((k,i)=>z[k[0]][k[1]]=src[i]);
return {st:{shape:[nT,nM],temperatures:d.metadata.temperatures_K,mu_grid:d.metadata.mu_grid_eV,labels:{}},z}}
function heat(id,z,a,opts){const nM=a.x.length,nT=a.y.length;const cellW=(($(id).clientWidth||600)-140)/Math.max(1,nM);
const big=nM*nT<=160&&cellW>=34;   /* cell values only where they fit */
const text=z.map(r=>r.map(v=>v==null?"":fmtN(v,opts.digits||3)));
const cd=z.map((r,i)=>r.map((_,j)=>a.x[j]));
Plotly.react(id,[{type:"heatmap",z,x:a.x,y:a.y,customdata:cd,colorscale:opts.colorscale||"Viridis",reversescale:!!opts.reverse,zmin:opts.zmin,zmax:opts.zmax,
xgap:1.5,ygap:1.5,text,texttemplate:big?"%{text}":"",textfont:{size:nM>9?9.5:11,family:css("--mono")},
hovertemplate:"%{y} · %{customdata}<br><b>%{z}</b><extra></extra>",colorbar:{thickness:10,outlinewidth:0,tickfont:{color:css("--muted"),size:10}}}],
layout({xaxis:{title:a.xt,type:"category",tickvals:a.x,ticktext:a.xs,tickangle:nM>7?-40:0,showgrid:false},
yaxis:{title:a.yt,type:"category",showgrid:false}}),cfg)}
function drawGrid(d){const g=gridData(d);if(!g||!g.z){empty("grid",g?"Not recorded in this run (adsorbed_cutoff / subsurface section).":"Per-replica data appear with status.json (new runs) or samples.npz (checkpoints, end of run).");return}
const a=axes(g.st,d.metadata);heat("grid",g.z,a,{});$("gridnote").textContent="";
$("grid").removeAllListeners?.("plotly_click");$("grid").on&&$("grid").on("plotly_click",e=>{const p=e.points[0];
sel=[a.y.indexOf(p.y),a.x.indexOf(p.x)];drawTraces(last);drawCompTraces(last)})}
function drawTraces(d){const st=d.status;let tr=st?st.traces:d.samples&&d.samples.traces,keys=st?st.keys:d.samples&&d.samples.keys;
if(!tr){Plotly.purge("traces");$("tracekey").textContent="";return}if(!sel)sel=keys[0];
let i=keys.findIndex(k=>k[0]==sel[0]&&k[1]==sel[1]);if(i<0)i=0;const x=tr.N[i].map((_,j)=>j*tr.stride);
$("tracekey").textContent=`replica (${keys[i][0]}, ${keys[i][1]}) · ${tr.stride>1?"every "+ord(tr.stride)+" sample":"all samples"}`;
Plotly.react("traces",[{x,y:tr.N[i],name:"N",mode:"lines",line:{color:css("--accent"),width:1.6}},
{x,y:tr.U[i],name:"U (eV)",yaxis:"y2",mode:"lines",line:{color:css("--warn"),width:1.2,dash:"dot"}},
...(tr.N_sub&&tr.N_sub[i]&&tr.N_sub[i].length?[{x,y:tr.N_sub[i],name:"N subsurface",mode:"lines",line:{color:css("--ok"),width:1.4}}]:[])],
layout({xaxis:{title:"sample"},yaxis:{title:"N"},yaxis2:{title:"U (eV)",overlaying:"y",side:"right",showgrid:false}}),cfg)}
function compSt(d,c){if(d.status)return d.status;const m=d.metadata||{},z=c.grids.mean_swappable;
return {shape:[z.length,z[0].length],temperatures:m.temperatures_K||z.map((_,i)=>i),mu_grid:m.mu_grid_eV||[z[0]],labels:{}}}
function drawComp(d){const c=d.composition;$("compcard").hidden=$("comptrcard").hidden=!c;if(!c)return;
const sp=c.species,names={mean_swappable:`mean ${sp} fraction, swappable layers`,mean_top:`mean ${sp} fraction, top layer`,
current_swappable:`current ${sp} fraction, swappable layers`,current_top:`current ${sp} fraction, top layer`,
identity_acceptance:"identity-swap acceptance"};
const avail=Object.keys(names).filter(k=>c.grids[k]&&c.grids[k].flat().some(v=>v!=null));const q=$("compq");
if(q.dataset.opts!==avail.join()){const keep=q.value;q.innerHTML=avail.map(k=>`<option value="${k}">${names[k]}</option>`).join("");
q.dataset.opts=avail.join();if(avail.includes(keep))q.value=keep}
const key=q.value||avail[0],z=c.grids[key],st=compSt(d,c),a=axes(st,d.metadata),acc=key=="identity_acceptance";
heat("compgrid",z,a,{zmin:0,zmax:1,colorscale:acc?"Viridis":"RdBu",reverse:!acc,digits:2});
$("compgrid").removeAllListeners?.("plotly_click");$("compgrid").on&&$("compgrid").on("plotly_click",e=>{const p=e.points[0];
sel=[a.y.indexOf(p.y),a.x.indexOf(p.x)];drawTraces(last);drawCompTraces(last)});
const dm=c.sgc_species.slice(1).map((s,i)=>`${s} ${fmtN(c.dmu[i+1],4)}`).join(", ");
$("compsrc").textContent=`${c.sgc_species.join("/")} semi-grand-canonical, Δμ vs ${c.sgc_species[0]}: ${dm} eV · from ${c.source}`;
drawCompTraces(d)}
function drawCompTraces(d){const c=d.composition;if(!c)return;const keys=c.keys,tr=c.traces;if(!sel)sel=keys[0];
let i=keys.findIndex(k=>k[0]==sel[0]&&k[1]==sel[1]);if(i<0)i=0;const ys=tr.swappable[i]||[];const x=ys.map((_,j)=>j*tr.stride);
$("comptrkey").textContent=`replica (${keys[i][0]}, ${keys[i][1]})`;
const t=[{x,y:ys,name:`${c.species}, swappable layers`,mode:"lines",line:{color:css("--accent"),width:1.4}}];
if(tr.top)t.push({x,y:tr.top[i],name:`${c.species}, top layer`,mode:"lines",line:{color:css("--warn"),width:1.4}});
Plotly.react("comptr",t,layout({xaxis:{title:"sample"},yaxis:{title:`${c.species} fraction`,range:[0,1]}}),cfg)}
function drawHist(d){const lp=d.log?d.log.progress:[];const h=d.history||[];
const tr=[];if(lp.length)tr.push({x:lp.map(p=>p[0]),y:lp.map(p=>p[2]),name:"log",mode:"lines+markers",line:{color:css("--accent"),width:2},marker:{size:5}});
if(h.length)tr.push({x:h.map(p=>p[1]),y:h.map(p=>p[2]),name:"status (this monitor)",mode:"markers",marker:{color:css("--warn"),size:5}});
if(!tr.length){empty("hist","No progress yet.");return}
Plotly.react("hist",tr,layout({xaxis:{title:"step"},yaxis:{title:"mean N"},showlegend:tr.length>1}),cfg)}
function drawSwaps(d){const src=d.status||d.mixing;if(!src){empty("swaps","Swap statistics appear with status.json or mixing.json.");$("pairs").innerHTML="";empty("trips","–");return}
const acc=src.swap_acceptance,att=src.swap_attempts||{},t=Object.keys(acc).filter(k=>att[k]==null||att[k]>0);
Plotly.react("swaps",[{type:"bar",x:t,y:t.map(k=>acc[k]),marker:{color:t.map(k=>acc[k]<0.05?css("--bad"):acc[k]<0.15?css("--warn"):css("--accent"))},
text:t.map(k=>fmtN(acc[k],2)),textposition:"outside",cliponaxis:false,textfont:{size:11,color:css("--ink")},
customdata:t.map(k=>att[k]!=null?att[k].toLocaleString()+" attempts":""),hovertemplate:"%{x}: %{y:.3f}<br>%{customdata}<extra></extra>"}],
layout({yaxis:{title:"acceptance",range:[0,1.15],dtick:.25},xaxis:{tickangle:0},bargap:.35}),cfg);
const pairs=(src.pairs||[]).filter(p=>p.attempts>0).sort((a,b)=>a.acceptance-b.acceptance||b.attempts-a.attempts).slice(0,5);
$("pairs").innerHTML=pairs.length?`<table><tr><th>worst pairs</th><th class="num">attempts</th><th class="num">acceptance</th></tr>${pairs.map(p=>`<tr><td class="nw">(${p.a}) ↔ (${p.b})</td><td class="num">${p.attempts}</td><td class="num">${fmtN(p.acceptance,3)}</td></tr>`).join("")}</table>`:"";
const T=src.round_trips_T||[],M=src.round_trips_mu||[],B=src.round_trips_bias||[];const w=T.map((_,i)=>i);
const tr=[{type:"bar",x:w,y:T,name:"along T",marker:{color:css("--accent")}},{type:"bar",x:w,y:M,name:"along μ / p",marker:{color:css("--ok")}}];
if(B.length)tr.push({type:"bar",x:w,y:B,name:"to bias columns",marker:{color:css("--warn")}});
const mx=Math.max(1,...T,...M,...B);
Plotly.react("trips",tr,layout({barmode:"group",bargap:.15,xaxis:{title:"walker"},yaxis:{title:"half round trips",range:[0,mx*1.1],rangemode:"tozero"}}),cfg)}
function drawConv(d){const c=(d.status&&d.status.convergence_history)||[];if(!c.length){$("convsub").textContent="";
empty("conv","No convergence checks yet (replica_exchange.convergence_every).");return}
const cv=(d.status&&d.status.convergence)||{},st=cv.settings||{},noise=st.mode==="noise";const xs=c.map((_,i)=>i+1);
const tr=[{y:c,x:xs,mode:"lines+markers",name:noise?"max TV / noise":"max |Δρ(u)|",line:{color:css("--accent"),width:2}}];
if(st.tol!=null)tr.push({y:c.map(()=>st.tol),x:xs,mode:"lines",name:"tolerance (max)",line:{color:css("--accent"),dash:"dash",width:1}});
if(noise){const ch=cv.checks||[];if(ch.length){tr.push({y:ch.map(x=>x.median_ratio),x:ch.map((_,i)=>i+1),mode:"lines+markers",name:"median TV / noise",line:{color:css("--warn"),width:2}});
if(st.median_tol!=null)tr.push({y:ch.map(()=>st.median_tol),x:ch.map((_,i)=>i+1),mode:"lines",name:"tolerance (median)",line:{color:css("--warn"),dash:"dash",width:1}})}}
const lc=(cv.checks||[]).slice(-1)[0];
$("convsub").textContent=lc&&lc.min_neff!=null?`min n_eff ${lc.min_neff.toFixed(1)}${st.min_neff!=null?" (need "+st.min_neff+")":""}`:"";
Plotly.react("conv",tr,layout({xaxis:{title:"check",dtick:c.length<12?1:undefined},yaxis:{title:noise?"TV / noise floor":"max |Δρ(u)|",rangemode:"tozero"}}),cfg)}
function drawData(d){const desc={"status.json":"live heartbeat","samples.npz":"samples (last checkpoint)",
"checkpoint.pkl":"checkpoint","run_metadata.json":"setup","mixing.json":"swap statistics","records.h5":"per-sample records",
"phase_diagram_data.npz":"analysis results","ladder.json":"ladder history"};
let rows=(d.files||[]).map(f=>`<tr><td class="file">${desc[f.name]?`<b>${esc(f.name)}</b>`:esc(f.name)}</td><td class="note">${desc[f.name]||""}</td><td class="num">${f.name.endsWith("/")?"":fmtB(f.size)}</td><td class="num">${fmtT(d.now-f.mtime)} ago</td></tr>`).join("");
let extra="";const r=d.records;if(r){extra+=`<p class="note">records.h5: ${r.rows!=null?`${r.samples} samples (${r.configurations} configurations), energies: ${Object.entries(r.potentials).map(([k,v])=>k+" "+v).join(", ")}`:esc(r.note||"")}</p>`}
if(d.samples)extra+=`<p class="note">samples.npz: ${d.samples.keys.length} replicas × ${d.samples.n_samples} samples</p>`;
$("data").innerHTML=(rows?`<table><tr><th>file</th><th></th><th class="num">size</th><th class="num">modified</th></tr>${rows}</table>`:'<p class="note">Directory empty or not created yet.</p>')+extra}
async function refresh(){try{const r=await fetch(`api/run/${cur}`);const d=await r.json();if(d.error)throw new Error(d.error);draw(d)}
catch(e){$("badge").className="pill error";$("badgetxt").textContent="monitor error";$("updated").textContent=String(e)}}
$("gridq").onchange=()=>last&&drawGrid(last);$("compq").onchange=()=>last&&drawComp(last);
(async()=>{setTheme(store.get("theme","auto"));$("phosphor").value=store.get("phosphor","green");$("phosphor").onchange();
$("wrap").checked=store.get("wrap","1")=="1";$("wrap").onchange();$("follow").checked=store.get("follow","1")=="1";
setCrt(store.get("crt","0")=="1",false);
if(typeof Plotly==="undefined"){document.body.insertAdjacentHTML("afterbegin",'<p style="margin:0;padding:8px 24px;background:var(--bad);color:#fff">Plotly could not be loaded (from cdn.jsdelivr.net or cdnjs.cloudflare.com): give the browser internet access, or start the monitor with --plotly-js path/to/plotly.min.js.</p>');window.Plotly={react(){},purge(){}}}
await loadRuns();await refresh();setInterval(()=>{if($("auto").checked&&!document.hidden)refresh()},5000)})();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
