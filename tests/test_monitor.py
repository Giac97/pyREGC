"""
The live monitor (`regc.monitor`) and the status heartbeat it reads
(`ReplicaExchangeGCMC.save_status`).
"""

import json
import threading
import urllib.request

import numpy as np
import pytest

from regc import LJCalculator, LJSystem, ReplicaExchangeGCMC
from regc.monitor import Run, discover, make_handler, parse_log


def _make(rng):
    return LJSystem(5.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def test_status_file_tracks_the_run(tmp_path):
    path = tmp_path / "status.json"
    with ReplicaExchangeGCMC([1.0, 1.2], [[-3.4, -3.0], [-3.6, -3.2]], _make, seed=0) as sim:
        sim.save_status(str(path), interval_s=0.0, extra={"temperatures_K": [100, 120]})
        sim.run_regc(n_steps=300, x0=0.8, diffusion="mc", diffusion_steps=1, record_every=5,
                     verbose=False)
    st = json.loads(path.read_text())
    assert st["step"] == st["n_steps"] == 300
    assert st["shape"] == [2, 2] and st["labels"]["temperatures_K"] == [100, 120]
    n = [len(sim.samples[k]["N"]) for k in sorted(sim.samples)]
    assert np.ravel(st["replicas"]["n_samples"]).tolist() == n
    assert np.ravel(st["replicas"]["mean_N"]) == pytest.approx(
        [np.mean(sim.samples[k]["N"]) for k in sorted(sim.samples)])
    assert len(st["traces"]["N"]) == 4 and len(st["traces"]["N"][0]) == n[0]
    assert set(st["swap_acceptance"]) == set(sim.swap_acceptance())


def test_status_is_whole_run_progress_under_the_driver(tmp_path):
    from regc import run_with_checkpoints
    with ReplicaExchangeGCMC([1.0, 1.2], [-3.2, -3.0], _make, seed=0) as sim:
        sim.save_status(str(tmp_path / "status.json"), interval_s=0.0)
        run_with_checkpoints(sim, str(tmp_path), n_steps=400, chunk=200, x0=0.8, diffusion="mc",
                             diffusion_steps=1, record_every=5)
    st = json.loads((tmp_path / "status.json").read_text())
    assert st["step"] == 400 and st["n_steps"] == 400
    r = Run(str(tmp_path)).collect()
    assert r["state"] == "finished" and r["samples"]["n_samples"] == len(sim.samples[(0, 0)]["N"])
    assert r["mixing"] is not None


def test_parse_log_progress_and_state(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("replica grid: 2 T x 3 p = 6 replicas\n"
                   "step    1500/15000 | mean N across grid =   5.72\n"
                   "step    3000/15000 | mean N across grid =   6.44\n")
    p = parse_log(str(log))
    assert p["progress"] == [[1500, 15000, 5.72], [3000, 15000, 6.44]]
    assert not p["finished"] and not p["error"] and p["grid"].startswith("replica grid")
    log.write_text(log.read_text() + "Traceback (most recent call last):\nValueError: x\n")
    assert parse_log(str(log))["error"]
    run = Run(str(tmp_path / "out"), str(log))
    assert run.collect()["state"] == "error"


def test_server_routes(tmp_path):
    from http.server import ThreadingHTTPServer
    with ReplicaExchangeGCMC([1.0], [-3.2, -3.0], _make, seed=0) as sim:
        sim.save_status(str(tmp_path / "status.json"), interval_s=0.0)
        sim.run_regc(n_steps=50, x0=0.8, diffusion="mc", diffusion_steps=1, record_every=5,
                     verbose=False)
    (tmp_path / "plot.png").write_bytes(b"\x89PNG fake")
    runs = discover([str(tmp_path)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runs))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<title>regc monitor</title>" in page
        assert json.loads(urllib.request.urlopen(base + "/api/runs").read())[0]["id"] == 0
        d = json.loads(urllib.request.urlopen(base + "/api/run/0").read())
        assert d["step"] == 50 and d["images"] == ["plot.png"]
        assert urllib.request.urlopen(base + "/file/0/plot.png").read() == b"\x89PNG fake"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/file/0/..%2Fsecret.png")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/file/0/status.json.tmp")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_discover_expands_a_directory_of_runs(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "status.json").write_text("{}")
    (tmp_path / "notarun").mkdir()
    assert [r.name for r in discover([str(tmp_path)])] == ["a", "b"]
