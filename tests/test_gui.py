"""Tests of the web GUI (regc.gui): HTTP API, token, workspace boundary, launching runs."""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
import yaml

from regc import gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def server(tmp_path):
    """A GUI server on a free port over an empty workspace; yields (base url, token, workspace)."""
    ready = threading.Event()
    th = threading.Thread(target=gui.serve, kwargs=dict(root=str(tmp_path), port=0, pythonpath=ROOT,
                                                       ready=ready), daemon=True)
    th.start()
    assert ready.wait(10)
    httpd = ready.server
    yield f"http://127.0.0.1:{httpd.server_address[1]}", httpd.token, httpd.workspace
    for c in httpd.workspace.overview()["configs"]:      # never leave a run behind
        if c["job"]["pid"]:
            httpd.workspace.stop(c["path"])
    httpd.shutdown()


def call(base, path, body=None, token=None):
    """GET (body None) or POST JSON; returns (status, parsed JSON)."""
    req = urllib.request.Request(base + path)
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("X-Regc-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def quick_config(n_steps=40):
    """The EMT template, shortened."""
    cfg = yaml.safe_load(next(iter(gui.TEMPLATES.values())))
    cfg["replica_exchange"].update(n_steps=n_steps, convergence_every=None, record_every=2)
    del cfg["replica_exchange"]["convergence_every"]
    cfg["checkpoint"] = {"chunk": 20}
    cfg["phase_diagram"].update(n_T=4, n_p=4)
    return cfg


def test_templates_are_valid_configs():
    for name, text in gui.TEMPLATES.items():
        d = gui.parse_config(text)
        assert d["ok"], (name, d["error"])


def test_dump_round_trip_keeps_floats_readable():
    cfg = yaml.safe_load(gui.TEMPLATES["Slab + O from O2, LAMMPS MLIP (skeleton)"])
    text = gui.dump_config(cfg)
    assert "1.0e-14" in text
    assert yaml.safe_load(text) == cfg       # 1e-14 stays a float, not a string


def test_page_and_token(server):
    base, token, _ = server
    with urllib.request.urlopen(base + "/") as r:
        page = r.read().decode()
    assert token in page and "regc gui" in page
    assert call(base, "/api/parse", {"text": "a: 1"})[0] == 403            # no token
    assert call(base, "/api/parse", {"text": "a: 1"}, "wrong")[0] == 403
    code, d = call(base, "/api/parse", {"text": "a: 1"}, token)
    assert code == 200 and not d["ok"]                                    # unknown section


def test_save_refuses_outside_and_overwrite(server, tmp_path):
    base, token, _ = server
    text = yaml.safe_dump(quick_config())
    assert call(base, "/api/save", {"path": "../evil.yaml", "text": text}, token)[0] == 400
    assert call(base, "/api/save", {"path": "x/run.txt", "text": text}, token)[0] == 400
    code, d = call(base, "/api/save", {"path": "x/run.yaml", "text": text}, token)
    assert code == 200 and (tmp_path / "x" / "run.yaml").is_file()
    assert call(base, "/api/save", {"path": "x/run.yaml", "text": text}, token)[0] == 400
    assert call(base, "/api/save", {"path": "x/run.yaml", "text": text, "overwrite": True},
                token)[0] == 200
    code, ws = call(base, "/api/workspace")
    assert [c["path"] for c in ws["configs"]] == [os.path.join("x", "run.yaml")]
    assert ws["configs"][0]["job"]["state"] == "new"
    assert call(base, "/api/config?path=../../etc/passwd")[0] == 400


def wait_state(base, states, timeout=300):
    """Poll the workspace until the only config's state is one of `states`."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = call(base, "/api/workspace")[1]["configs"][0]["job"]
        if job["state"] in states:
            return job
        time.sleep(1)
    raise AssertionError(f"state stayed {job['state']}: {job['last_line']}")


def test_run_monitor_and_resume_guard(server, tmp_path):
    base, token, _ = server
    call(base, "/api/save", {"path": "r/run.yaml", "text": yaml.safe_dump(quick_config())}, token)
    code, d = call(base, "/api/start", {"path": "r/run.yaml", "mode": "run"}, token)
    assert code == 200, d
    job = wait_state(base, {"finished", "converged", "error"})
    assert job["state"] == "finished", (tmp_path / "r" / "output_test_emt" / "run.log").read_text()
    out = tmp_path / "r" / "output_test_emt"
    assert (out / "records.h5").is_file() and (out / "phase_diagram.png").is_file()
    # a second plain start would overwrite the checkpoint: refused
    code, d = call(base, "/api/start", {"path": "r/run.yaml", "mode": "run"}, token)
    assert code == 400 and "checkpoint" in d["error"]
    # the monitor mounted under /monitor/ sees the run
    ws = call(base, "/api/workspace")[1]
    mid = ws["configs"][0]["monitor_id"]
    code, runs = call(base, "/monitor/api/runs")
    assert code == 200 and runs[mid]["directory"] == str(out)
    code, m = call(base, f"/monitor/api/run/{mid}")
    assert code == 200 and m["state"] == "finished" and m["status"]["step"] == 40


def test_stop_and_resume(server, tmp_path):
    base, token, _ = server
    call(base, "/api/save", {"path": "s/run.yaml", "text": yaml.safe_dump(quick_config(100000))},
         token)
    call(base, "/api/start", {"path": "s/run.yaml", "mode": "run"}, token)
    out = tmp_path / "s" / "output_test_emt"
    t0 = time.time()
    while not (out / "checkpoint.pkl").is_file():           # first chunk written
        assert time.time() - t0 < 300, (out / "run.log").read_text()
        time.sleep(1)
    assert call(base, "/api/start", {"path": "s/run.yaml", "mode": "resume"}, token)[0] == 400
    code, d = call(base, "/api/stop", {"path": "s/run.yaml"}, token)
    assert code == 200
    job = wait_state(base, {"stopped", "error", "ended"}, timeout=60)
    assert job["state"] == "stopped" and job["resumable"]
    code, d = call(base, "/api/start", {"path": "s/run.yaml", "mode": "resume"}, token)
    assert code == 200
    wait_state(base, {"running", "starting"}, timeout=60)
    assert call(base, "/api/stop", {"path": "s/run.yaml"}, token)[0] == 200


def test_check_task(server):
    base, token, _ = server
    call(base, "/api/save", {"path": "c/run.yaml", "text": yaml.safe_dump(quick_config())}, token)
    code, d = call(base, "/api/check", {"path": "c/run.yaml"}, token)
    assert code == 200
    t0 = time.time()
    while True:
        t = call(base, f"/api/task/{d['task']}")[1]
        if t["done"]:
            break
        assert time.time() - t0 < 120
        time.sleep(1)
    assert t["returncode"] == 0 and "calculator works" in t["output"], t["output"]


# ---- request origin (DNS rebinding) -----------------------------------------
def raw(base, path, headers, body=None):
    """A request with explicit headers; returns (status, body bytes)."""
    req = urllib.request.Request(base + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_foreign_host_and_origin_refused(server):
    base, token, _ = server
    port = base.rsplit(":", 1)[1]
    code, body = raw(base, "/", {"Host": f"attacker.example:{port}"})
    assert code == 403 and token.encode() not in body
    assert raw(base, "/monitor/api/runs", {"Host": f"attacker.example:{port}"})[0] == 403
    assert raw(base, "/", {"Host": f"127.0.0.1:{int(port) + 1}"})[0] == 403
    for host in (f"localhost:{port}", f"127.0.0.1:{port}"):
        code, body = raw(base, "/", {"Host": host})
        assert code == 200 and token.encode() in body
    post = {"Content-Type": "application/json", "X-Regc-Token": token}
    data = json.dumps({"text": "a: 1"}).encode()
    assert raw(base, "/api/parse", {**post, "Origin": "http://attacker.example"}, data)[0] == 403
    assert raw(base, "/api/parse", {**post, "Origin": "null"}, data)[0] == 403
    assert raw(base, "/api/parse", {**post, "Origin": f"http://localhost:{port}"}, data)[0] == 200
    assert raw(base, "/api/parse", post, data)[0] == 200          # no Origin: not a browser


def test_host_allowed_rules():
    from regc import monitor
    ok = monitor.host_allowed
    assert ok("127.0.0.1:8060", "127.0.0.1", 8060) and ok("LOCALHOST:8060", "127.0.0.1", 8060)
    assert ok("[::1]:8060", "127.0.0.1", 8060)
    assert not ok("evil.example:8060", "127.0.0.1", 8060) and not ok(None, "127.0.0.1", 8060)
    assert not ok("192.168.1.5:8060", "127.0.0.1", 8060)
    assert ok("192.168.1.5:8060", "0.0.0.0", 8060)              # wildcard bind: own addresses
    assert not ok("evil.example:8060", "0.0.0.0", 8060)
    assert ok("myhost.lan:8060", "myhost.lan", 8060)
    assert monitor.origin_allowed(None, "127.0.0.1", 8060)
    assert not monitor.origin_allowed("https://127.0.0.1:8060", "127.0.0.1", 8060)


def test_standalone_monitor_refuses_foreign_host(tmp_path):
    from http.server import ThreadingHTTPServer
    from regc import monitor
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), monitor.make_handler([]))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert raw(base, "/api/runs", {"Host": "attacker.example"})[0] == 403
        assert raw(base, "/api/runs", {})[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- JSON, bad input, environment ------------------------------------------
def test_json_bytes_keeps_text_and_nulls_nonfinite():
    import numpy as np
    from regc import monitor
    data = {"text": "# skip NaN energies; Infinity and -Infinity too", "x": float("nan"),
            "y": [np.inf, 1.0, (-np.inf, 2)], "z": np.float64("nan"), "a": np.arange(2)}
    d = json.loads(monitor._json_bytes(data))
    assert d == {"text": data["text"], "x": None, "y": [None, 1.0, [None, 2]], "z": None, "a": [0, 1]}


def test_config_text_with_nan_served_intact(server, tmp_path):
    base, token, _ = server
    text = yaml.safe_dump(quick_config()) + "# skip NaN energies\n"
    assert call(base, "/api/save", {"path": "n/run.yaml", "text": text}, token)[0] == 200
    code, d = call(base, "/api/config?path=n/run.yaml")
    assert code == 200 and d["text"] == text


def test_bad_input_is_a_400(server):
    base, token, _ = server
    code, d = call(base, "/api/start", {"path": 5}, token)
    assert code == 400 and "TypeError" in d["error"]
    code, _ = raw(base, "/api/parse", {"Content-Type": "application/json", "X-Regc-Token": token},
                  b"[1, 2]")
    assert code == 400


def test_env_prepends_pythonpath_and_limits_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    env = gui.Workspace(str(tmp_path), pythonpath="/the/repo")._env()
    assert env["PYTHONPATH"] == os.pathsep.join(["/the/repo", "/somewhere/else"])
    assert env["OMP_NUM_THREADS"] == "1" and env["OPENBLAS_NUM_THREADS"] == "1"
    assert env["MKL_NUM_THREADS"] == "4"


# ---- run liveness -------------------------------------------------------------
def workspace_with_output(tmp_path, checkpoint=True):
    """A workspace with one config; returns (workspace, its output directory)."""
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "run.yaml").write_text(yaml.safe_dump(quick_config()))
    out = tmp_path / "w" / "output_test_emt"
    out.mkdir()
    if checkpoint:
        (out / "checkpoint.pkl").write_bytes(b"x")
    return gui.Workspace(str(tmp_path), pythonpath=ROOT), out


def write_status(out, **kw):
    st = {"time": time.time(), "started": time.time(), "step": 100, "n_steps": 1000}
    st.update(kw)
    (out / "status.json").write_text(json.dumps(st))


def test_external_run_found_by_fresh_heartbeat(tmp_path):
    ws, out = workspace_with_output(tmp_path)
    write_status(out)                               # an older run: no pid in the heartbeat
    job = ws.job(str(out))
    assert job["state"] == "running" and job["external"]
    for mode in ("run", "resume", "overwrite", "analyze"):
        with pytest.raises(ValueError, match="already going"):
            ws.start("w/run.yaml", mode)
    with pytest.raises(ValueError, match="outside the GUI"):
        ws.stop("w/run.yaml")
    write_status(out, time=time.time() - 2 * gui.STALE_S)           # heartbeat gone stale
    ws._status_cache.clear()
    assert ws.job(str(out))["state"] == "stopped"


def test_heartbeat_pid_checked_on_this_host(tmp_path):
    import socket
    import subprocess
    ws, out = workspace_with_output(tmp_path)
    proc = subprocess.Popen(["sleep", "60"])
    try:
        time.sleep(0.2)
        # alive even with an old heartbeat (stalled); not a reused id
        write_status(out, pid=proc.pid, host=socket.gethostname(), time=time.time() - 1000)
        job = ws.job(str(out))
        assert job["state"] == "stalled" and job["external"] and job["pid"] == proc.pid
        # a heartbeat that began before the process did: the id was reused
        write_status(out, pid=proc.pid, host=socket.gethostname(), started=time.time() - 3600)
        ws._status_cache.clear()
        assert ws.job(str(out))["state"] == "stopped"
    finally:
        proc.kill()
        proc.wait()
    write_status(out, pid=proc.pid, host=socket.gethostname())    # fresh, but the process is gone
    ws._status_cache.clear()
    assert ws.job(str(out))["state"] == "stopped"


def test_pid_file_start_time(tmp_path):
    import subprocess
    proc = subprocess.Popen(["sleep", "60"])
    try:
        time.sleep(0.2)
        start = gui._proc_stat(proc.pid)[1]
        assert gui.pid_alive(proc.pid, start)
        assert not gui.pid_alive(proc.pid, start + 1)      # same id, another process
        assert not gui.pid_alive(proc.pid)                 # old pid file: not a regc command line
        (tmp_path / "run.pid").write_text(str(proc.pid))
        (tmp_path / "run.pid.start").write_text(f"{proc.pid} {start}\n")
        assert gui._read_pid(str(tmp_path / "run.pid")) == proc.pid
        assert gui._read_pid_start(str(tmp_path), proc.pid) == start
        assert gui._read_pid_start(str(tmp_path), proc.pid + 1) is None
    finally:
        proc.kill()
        proc.wait()
    assert not gui.pid_alive(proc.pid, start)


def test_start_is_serialised(tmp_path, monkeypatch):
    ws, out = workspace_with_output(tmp_path, checkpoint=False)
    monkeypatch.setattr(ws, "_cmd", lambda path, *flags: ["sleep", "60"])
    results, barrier = [], threading.Barrier(4)

    def go():
        barrier.wait()
        try:
            results.append(ws.start("w/run.yaml", "run")["pid"])
        except ValueError as err:
            results.append(str(err))
    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pids = [r for r in results if isinstance(r, int)]
    try:
        assert len(pids) == 1 and sum("already going" in str(r) for r in results) == 3
        assert gui._read_pid_start(str(out), pids[0]) is not None
        assert ws.job(str(out))["state"] == "starting" and not ws.job(str(out))["external"]
        ws.stop("w/run.yaml")
        assert "stop (SIGTERM" in (out / "run.log").read_text()
    finally:
        for p in pids:
            try:
                os.kill(p, 9)
            except OSError:
                pass


def test_time_limit_stop_is_resumable(tmp_path):
    ws, out = workspace_with_output(tmp_path)
    write_status(out, time=time.time() - 2 * gui.STALE_S)
    (out / "run.log").write_text("# regc-gui 2026-01-01 00:00:00: run (PYTHONPATH=x)\n"
                                 "step 100/1000\nStopped for the time limit; continue with "
                                 f"--resume. Outputs in {out}/\n")
    job = ws.job(str(out))
    assert job["state"] == "stopped" and job["resumable"]
    with open(out / "run.log", "a") as f:                # resumed and finished
        f.write("# regc-gui 2026-01-01 01:00:00: resume (PYTHONPATH=x)\nPhase diagram. "
                f"Outputs in {out}/\n")
    assert ws.job(str(out))["state"] == "finished"


def test_overview_lists_output_outside_workspace(tmp_path):
    ws_dir, scratch = tmp_path / "ws", tmp_path / "scratch"
    ws_dir.mkdir()
    scratch.mkdir()
    cfg = quick_config()
    cfg["output"] = {"directory": str(scratch / "out")}
    (ws_dir / "run.yaml").write_text(yaml.safe_dump(cfg))
    rows = gui.Workspace(str(ws_dir)).overview()["configs"]
    assert rows[0]["output_dir"] == str(scratch / "out") and rows[0]["job"]["state"] == "new"
