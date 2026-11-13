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
import glob
import json
import os
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import unquote

import numpy as np

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
        ``traces`` (``N``, ``U``, ``stride``), ``mtime``; None if unreadable.
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
    except (OSError, ValueError, KeyError):
        return None
    U, N = U.reshape(len(keys), -1), N.reshape(len(keys), -1)
    stride = max(1, int(np.ceil(N.shape[1] / trace_length)))
    out = {"mtime": mtime, "keys": keys.tolist(), "n_samples": int(N.shape[1]),
           "mean_N": N.mean(axis=1).tolist() if N.size else [],
           "std_N": N.std(axis=1).tolist() if N.size else [],
           "mean_U": U.mean(axis=1).tolist() if U.size else [],
           "traces": {"stride": stride, "N": N[:, ::stride].tolist(), "U": U[:, ::stride].tolist()}}
    _CACHE[path] = (mtime, out)
    return out


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
            ``images``, ``history``, ``rate`` (steps/s) and ``eta_s``.
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
        return {
            "name": self.name, "directory": d, "now": now, "state": state,
            "last_update": last_update or None, "step": step, "n_steps": n_steps,
            "mean_N": mean_n, "rate": rate, "eta_s": eta, "status": status, "log": log,
            "metadata": _read_json(os.path.join(d, "run_metadata.json")),
            "mixing": _read_json(os.path.join(d, "mixing.json")),
            "samples": samples, "records": records, "files": files, "images": images,
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

        def _json(self, data, code: int = 200) -> None:
            """
            Send JSON (NaN as null).

            Parameters
            ----------
            data : object
                Payload.
            code : int, optional
                Status. Default 200.
            """
            text = json.dumps(data, default=float).replace("NaN", "null").replace(
                "-Infinity", "null").replace("Infinity", "null")
            self._send(text.encode(), "application/json", code)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            """Serve one request."""
            path = unquote(self.path.split("?", 1)[0])
            if path in ("/", "/index.html"):
                return self._send(PAGE.encode(), "text/html; charset=utf-8")
            if path == "/plotly.min.js":
                if plotly_js and os.path.isfile(plotly_js):
                    with open(plotly_js, "rb") as f:
                        return self._send(f.read(), "text/javascript")
                return self._send(b"", "text/javascript", 404)
            if path == "/api/runs":
                return self._json([{"id": i, "name": r.name, "directory": r.directory,
                                    "log": r._log_path()} for i, r in enumerate(runs)])
            m = re.match(r"^/api/run/(\d+)$", path)
            if m and int(m.group(1)) < len(runs):
                try:
                    return self._json(runs[int(m.group(1))].collect())
                except Exception as err:  # noqa: BLE001 - report, keep serving
                    return self._json({"error": f"{type(err).__name__}: {err}"}, 500)
            m = re.match(r"^/file/(\d+)/([^/]+)$", path)
            if m and int(m.group(1)) < len(runs):
                name = os.path.basename(m.group(2))
                ext = os.path.splitext(name)[1].lower()
                full = os.path.join(runs[int(m.group(1))].directory, name)
                if ext in _SERVED and os.path.isfile(full):
                    with open(full, "rb") as f:
                        return self._send(f.read(), _SERVED[ext])
            self._send(b"not found", "text/plain", 404)

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
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>regc monitor</title>
<script src="plotly.min.js"></script>
<script>if(typeof Plotly==="undefined")document.write('<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"><\/script>')</script>
<script>if(typeof Plotly==="undefined")document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"><\/script>')</script>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--accent:#0072b2;
--ok:#009e73;--warn:#e69f00;--bad:#d55e00;--idle:#8a8a8f;--track:#ececea}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1d1d20;--ink:#ececee;--muted:#9a9aa2;
--line:#2e2e33;--accent:#56b4e9;--ok:#2fc39a;--warn:#f0b43c;--bad:#ff7a3d;--idle:#7d7d85;--track:#2a2a2f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:14px 20px;
border-bottom:1px solid var(--line);background:var(--panel);position:sticky;top:0;z-index:5}
h1{font-size:16px;margin:0 8px 0 0;font-weight:650}select,button{font:inherit;color:var(--ink);
background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px}
.badge{border-radius:999px;padding:2px 10px;font-weight:600;font-size:12px;color:#fff}
.running{background:var(--ok)}.finished{background:var(--accent)}.stalled{background:var(--warn)}
.error{background:var(--bad)}.waiting,.quiet{background:var(--idle)}
.muted{color:var(--muted)}main{padding:16px 20px;display:grid;gap:16px;
grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:13px;margin:0 0 8px;font-weight:650;letter-spacing:.02em;text-transform:uppercase;
color:var(--muted)}.wide{grid-column:1/-1}
.bar{height:10px;background:var(--track);border-radius:6px;overflow:hidden;margin:8px 0}
.bar>div{height:100%;background:var(--accent);transition:width .6s}
.kpis{display:flex;flex-wrap:wrap;gap:22px}.kpi b{display:block;font-size:20px;font-weight:650;
font-variant-numeric:tabular-nums}.kpi span{font-size:12px;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:600}
pre{margin:0;max-height:260px;overflow:auto;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;
white-space:pre-wrap}.plot{width:100%;height:300px}.imgs{display:flex;flex-wrap:wrap;gap:10px}
.imgs a{flex:1 1 220px}.imgs img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.note{font-size:12px;color:var(--muted)}
@media (max-width:520px){main{grid-template-columns:1fr;padding:12px}header{padding:12px}}
</style></head><body>
<header><h1>regc monitor</h1><select id="run"></select><span id="badge" class="badge waiting">…</span>
<span id="updated" class="muted"></span><span style="flex:1"></span>
<label class="muted"><input type="checkbox" id="auto" checked> refresh every 5 s</label></header>
<main>
<section class="card wide"><h2>Progress</h2><div class="bar"><div id="pbar" style="width:0"></div></div>
<div class="kpis" id="kpis"></div><div class="note" id="where"></div></section>
<section class="card"><h2>Replica grid</h2><div class="row"><select id="gridq">
<option value="mean_N">mean N</option><option value="current_N">current N</option>
<option value="std_N">std of N</option><option value="n_samples">samples</option>
<option value="mean_U">mean U (eV)</option><option value="current_U">current U (eV)</option></select>
<span class="note">click a cell to show its traces</span></div><div id="grid" class="plot"></div></section>
<section class="card"><h2>Traces <span id="tracekey" class="muted"></span></h2><div id="traces" class="plot"></div></section>
<section class="card"><h2>Mean N across the grid</h2><div id="hist" class="plot"></div></section>
<section class="card"><h2>Replica exchange</h2><div id="swaps" class="plot" style="height:220px"></div>
<div id="pairs"></div></section>
<section class="card"><h2>Round trips per walker</h2><div id="trips" class="plot"></div></section>
<section class="card"><h2>Convergence: largest change of ρ(u)</h2><div id="conv" class="plot"></div></section>
<section class="card"><h2>Available data</h2><div id="data"></div></section>
<section class="card wide" id="imgcard" hidden><h2>Plots</h2><div class="imgs" id="imgs"></div></section>
<section class="card wide"><h2>Log <span id="logname" class="muted"></span></h2><pre id="log"></pre></section>
</main>
<script>
const $=id=>document.getElementById(id);let runs=[],cur=0,sel=null,last=null;
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmtT=s=>{if(s==null||!isFinite(s))return"–";s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
return h?`${h} h ${m} min`:m?`${m} min ${s%60} s`:`${s} s`};
const fmtB=b=>b>1e9?(b/1e9).toFixed(2)+" GB":b>1e6?(b/1e6).toFixed(1)+" MB":b>1e3?(b/1e3).toFixed(0)+" kB":b+" B";
const fmtN=(x,d=3)=>x==null||!isFinite(x)?"–":Number(x).toPrecision(d).replace("-","−");
const fmtP=x=>{const a=Math.abs(x);return (a&&(a<1e-2||a>=1e4)?Number(x).toExponential(2).replace("e+","e").replace(/e(-?)0*(\d)/,"e$1$2"):Number(x).toPrecision(3)).replace("-","−")};
function layout(extra){const ink=css("--ink"),line=css("--line");return Object.assign({margin:{l:52,r:12,t:8,b:42},
paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:ink,size:11},
xaxis:{gridcolor:line,zerolinecolor:line},yaxis:{gridcolor:line,zerolinecolor:line},
legend:{orientation:"h",y:1.12},hovermode:"closest"},extra||{})}
const cfg={displaylogo:false,responsive:true,modeBarButtonsToRemove:["lasso2d","select2d"]};
async function loadRuns(){runs=await (await fetch("api/runs")).json();
$("run").innerHTML=runs.map(r=>`<option value="${r.id}">${r.name}</option>`).join("");
$("run").onchange=()=>{cur=+$("run").value;sel=null;refresh()};}
function axes(st,meta){const lab=st&&st.labels||{};const nT=st?st.shape[0]:0,nM=st?st.shape[1]:0;
const TK=lab.temperatures_K||(meta&&meta.temperatures_K);let y=TK?TK.map(t=>t+" K"):st.temperatures.map(t=>"T="+fmtN(t,4));
const P=lab.pressures||(meta&&meta.pressures),u=lab.pressure_unit||(meta&&meta.pressure_unit)||"";
let x=[];for(let j=0;j<nM;j++){x.push(P?`${fmtP(P[j])} ${u}`:`μ=${fmtN(Array.isArray(st.mu_grid[0][j])?st.mu_grid[0][j][0]:st.mu_grid[0][j],4)}`)}
return {x,y,xt:P?"pressure":"μ (column, eV at lowest T)",yt:"temperature"}}
function draw(d){last=d;const st=d.status,meta=d.metadata;$("badge").className="badge "+d.state;$("badge").textContent=d.state;
$("updated").textContent=d.last_update?`updated ${fmtT(d.now-d.last_update)} ago`+(d.state=="quiet"?" (no heartbeat: the log reports every 10 % of the run)":""):"no data yet";
const frac=d.step&&d.n_steps?d.step/d.n_steps:0;$("pbar").style.width=(100*frac).toFixed(1)+"%";
const nrep=st?st.shape[0]*st.shape[1]:(d.samples?d.samples.keys.length:null);const ns=st?Math.max(...st.replicas.n_samples.flat().filter(x=>x!=null)):(d.samples?d.samples.n_samples:null);
const k=[["step",d.step!=null?`${d.step.toLocaleString()} / ${d.n_steps.toLocaleString()}`:"–"],["done",(100*frac).toFixed(1)+" %"],
["rate",d.rate?fmtN(d.rate,3)+" steps/s":"–"],["remaining",d.state=="running"||d.state=="quiet"?fmtT(d.eta_s):"–"],
["elapsed",st&&st.started?fmtT(st.time-st.started):"–"],["mean N now",fmtN(d.mean_N,3)],
["replicas",nrep??"–"],["samples / replica",ns??"–"]];
$("kpis").innerHTML=k.map(([a,b])=>`<div class="kpi"><b>${b}</b><span>${a}</span></div>`).join("");
$("where").textContent=d.directory+(d.log?"  ·  log "+d.log.path:"")+(st?"":"  ·  no status.json: progress from the log only (run started without the heartbeat)");
drawGrid(d);drawTraces(d);drawHist(d);drawSwaps(d);drawConv(d);drawData(d);
$("logname").textContent=d.log?d.log.path:"";$("log").textContent=d.log?d.log.tail.join("\n"):"(no log)";
const imgs=d.images||[];$("imgcard").hidden=!imgs.length;
$("imgs").innerHTML=imgs.map(n=>`<a href="file/${cur}/${encodeURIComponent(n)}" target="_blank"><img loading="lazy" src="file/${cur}/${encodeURIComponent(n)}?t=${Math.round(d.now)}" alt="${n}"><div class="note">${n}</div></a>`).join("");}
function gridData(d){const st=d.status;if(st)return {st,z:st.replicas[$("gridq").value]};
const s=d.samples;if(!s||!d.metadata)return null;const nT=d.metadata.temperatures_K.length,keys=s.keys,
nM=Math.max(...keys.map(k=>k[1]))+1;const q=$("gridq").value,src={mean_N:s.mean_N,std_N:s.std_N,mean_U:s.mean_U}[q];
if(!src)return null;const z=[...Array(nT)].map(()=>Array(nM).fill(null));keys.forEach((k,i)=>z[k[0]][k[1]]=src[i]);
return {st:{shape:[nT,nM],temperatures:d.metadata.temperatures_K,mu_grid:d.metadata.mu_grid_eV,labels:{}},z}}
function drawGrid(d){const g=gridData(d);if(!g){Plotly.purge("grid");$("grid").innerHTML='<p class="note">Per-replica data appear with status.json (new runs) or samples.npz (checkpoints, end of run).</p>';return}
const a=axes(g.st,d.metadata);const text=g.z.map(r=>r.map(v=>v==null?"":fmtN(v,3)));
Plotly.react("grid",[{type:"heatmap",z:g.z,x:a.x,y:a.y,colorscale:"Viridis",text,texttemplate:"%{text}",
hovertemplate:"%{y}, %{x}<br>%{z}<extra></extra>",colorbar:{thickness:10}}],
layout({xaxis:{title:a.xt,type:"category",tickangle:-30},yaxis:{title:a.yt,type:"category"},margin:{l:70,r:10,t:8,b:70}}),cfg);
$("grid").on&&($("grid").removeAllListeners?.("plotly_click"),$("grid").on("plotly_click",e=>{const p=e.points[0];
sel=[a.y.indexOf(p.y),a.x.indexOf(p.x)];drawTraces(last)}));}
function drawTraces(d){const st=d.status;let tr=st?st.traces:d.samples&&d.samples.traces,keys=st?st.keys:d.samples&&d.samples.keys;
if(!tr){Plotly.purge("traces");$("tracekey").textContent="";return}if(!sel)sel=keys[0];
let i=keys.findIndex(k=>k[0]==sel[0]&&k[1]==sel[1]);if(i<0)i=0;const x=tr.N[i].map((_,j)=>j*tr.stride);
$("tracekey").textContent=`replica (${keys[i][0]}, ${keys[i][1]}) · ${tr.stride>1?"every "+tr.stride+"th sample":"all samples"}`;
Plotly.react("traces",[{x,y:tr.N[i],name:"N",mode:"lines",line:{color:css("--accent"),width:1.2}},
{x,y:tr.U[i],name:"U (eV)",yaxis:"y2",mode:"lines",line:{color:css("--warn"),width:1,dash:"dot"}}],
layout({xaxis:{title:"sample"},yaxis:{title:"N"},yaxis2:{title:"U (eV)",overlaying:"y",side:"right",showgrid:false},margin:{l:48,r:60,t:8,b:42}}),cfg)}
function drawHist(d){const lp=d.log?d.log.progress:[];const h=d.history||[];
const tr=[];if(lp.length)tr.push({x:lp.map(p=>p[0]),y:lp.map(p=>p[2]),name:"log",mode:"lines+markers",line:{color:css("--accent")}});
if(h.length)tr.push({x:h.map(p=>p[1]),y:h.map(p=>p[2]),name:"status (this monitor)",mode:"markers",marker:{color:css("--warn"),size:5}});
if(!tr.length){Plotly.purge("hist");$("hist").innerHTML='<p class="note">No progress yet.</p>';return}
Plotly.react("hist",tr,layout({xaxis:{title:"step"},yaxis:{title:"mean N"}}),cfg)}
function drawSwaps(d){const src=d.status||d.mixing;if(!src){Plotly.purge("swaps");$("swaps").innerHTML='<p class="note">Swap statistics appear with status.json or mixing.json.</p>';$("pairs").innerHTML="";Plotly.purge("trips");return}
const acc=src.swap_acceptance,att=src.swap_attempts||{},t=Object.keys(acc);
Plotly.react("swaps",[{type:"bar",x:t,y:t.map(k=>acc[k]),marker:{color:css("--accent")},text:t.map(k=>att[k]!=null?att[k]+" att.":""),
hovertemplate:"%{x}: %{y:.3f}<br>%{text}<extra></extra>"}],layout({yaxis:{title:"acceptance",range:[0,1]},margin:{l:52,r:10,t:8,b:30}}),cfg);
const pairs=(src.pairs||[]).filter(p=>p.attempts>0).sort((a,b)=>a.acceptance-b.acceptance).slice(0,6);
$("pairs").innerHTML=pairs.length?`<table><tr><th>worst pairs</th><th>attempts</th><th>acceptance</th></tr>${pairs.map(p=>`<tr><td>(${p.a}) ↔ (${p.b})</td><td>${p.attempts}</td><td>${fmtN(p.acceptance,3)}</td></tr>`).join("")}</table>`:"";
const T=src.round_trips_T||[],M=src.round_trips_mu||[];const w=T.map((_,i)=>i);
Plotly.react("trips",[{type:"bar",x:w,y:T,name:"along T",marker:{color:css("--accent")}},{type:"bar",x:w,y:M,name:"along μ/p",marker:{color:css("--ok")}}],
layout({barmode:"group",xaxis:{title:"walker"},yaxis:{title:"half round trips"}}),cfg)}
function drawConv(d){const c=(d.status&&d.status.convergence_history)||[];if(!c.length){Plotly.purge("conv");
$("conv").innerHTML='<p class="note">No convergence checks yet (replica_exchange.convergence_every).</p>';return}
Plotly.react("conv",[{y:c,x:c.map((_,i)=>i+1),mode:"lines+markers",line:{color:css("--accent")}}],layout({xaxis:{title:"check"},yaxis:{title:"max |Δρ(u)|",rangemode:"tozero"}}),cfg)}
function drawData(d){const desc={"status.json":"live heartbeat","samples.npz":"samples (at the last checkpoint)",
"checkpoint.pkl":"checkpoint","run_metadata.json":"setup","mixing.json":"swap statistics","records.h5":"per-sample records",
"phase_diagram_data.npz":"analysis results"};
let rows=(d.files||[]).map(f=>`<tr><td>${desc[f.name]?`<b>${f.name}</b>`:f.name}</td><td>${desc[f.name]||""}</td><td>${f.name.endsWith("/")?"":fmtB(f.size)}</td><td>${fmtT(d.now-f.mtime)} ago</td></tr>`).join("");
let extra="";const r=d.records;if(r){extra+=`<p class="note">records.h5: ${r.rows!=null?`${r.samples} samples (${r.configurations} configurations), energies: ${Object.entries(r.potentials).map(([k,v])=>k+" "+v).join(", ")}`:r.note||""}</p>`}
if(d.samples)extra+=`<p class="note">samples.npz: ${d.samples.keys.length} replicas × ${d.samples.n_samples} samples</p>`;
$("data").innerHTML=(rows?`<table><tr><th>file</th><th></th><th>size</th><th>modified</th></tr>${rows}</table>`:'<p class="note">Directory empty or not created yet.</p>')+extra}
async function refresh(){try{const r=await fetch(`api/run/${cur}`);const d=await r.json();if(d.error)throw new Error(d.error);draw(d)}
catch(e){$("badge").className="badge error";$("badge").textContent="monitor error";$("updated").textContent=String(e)}}
$("gridq").onchange=()=>last&&drawGrid(last);
(async()=>{if(typeof Plotly==="undefined"){document.body.insertAdjacentHTML("afterbegin",'<p style="padding:8px 20px;background:#d55e00;color:#fff">Plotly could not be loaded (from cdn.jsdelivr.net or cdnjs.cloudflare.com): give the browser internet access, or start the monitor with --plotly-js path/to/plotly.min.js.</p>');window.Plotly={react(){},purge(){}}}
await loadRuns();await refresh();setInterval(()=>{if($("auto").checked)refresh()},5000)})();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
