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
