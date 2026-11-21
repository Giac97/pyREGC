"""Tests of the desktop GUI (regc.qtgui): form helpers, and the window driving a real EMT run."""

import os
import time

import pytest
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from regc import gui, qtgui  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_apply_field_round_trip():
    cfg = yaml.safe_load(gui.TEMPLATES["Slab + O from O2, LAMMPS MLIP (skeleton)"])
    before = yaml.safe_dump(cfg)
    for f in gui.FORM_FIELDS:                      # writing back unchanged text changes nothing
        if len(f) > 1:
            qtgui.apply_field(cfg, f[0], f[2], qtgui.field_text(qtgui._get(cfg, f[0])))
    assert yaml.safe_dump(cfg) == before
    qtgui.apply_field(cfg, "replica_exchange.temperatures_K", "list", "300, 400 500")
    qtgui.apply_field(cfg, "replica_exchange.n_steps", "int", "1e4")
    qtgui.apply_field(cfg, "records.save", "bool", "false")
    qtgui.apply_field(cfg, "subsurface.depth_angstrom", "num", "3")
    qtgui.apply_field(cfg, "trajectory.every", "int", "")
    re_cfg = cfg["replica_exchange"]
    assert re_cfg["temperatures_K"] == [300, 400, 500] and re_cfg["n_steps"] == 10000
    assert cfg["records"]["save"] is False and cfg["subsurface"] == {"depth_angstrom": 3}
    assert "every" not in cfg["trajectory"]
    qtgui.apply_field(cfg, "subsurface.depth_angstrom", "num", "")
    assert "subsurface" not in cfg                 # emptied section removed
    assert gui.parse_config(gui.dump_config(cfg))["ok"]


@pytest.mark.parametrize("name", ["light", "dark"])
def test_theme_contrast(name):
    """Text tokens reach WCAG AA (4.5:1) on every background they are drawn on."""
    t = qtgui.THEMES[name]
    for bg in ("surface", "bg", "selection", "muted"):
        for fg in ("fg", "muted_fg", "success", "warning", "danger", "info", "neutral", "primary"):
            if bg == "muted" and fg not in ("fg", "muted_fg"):
                continue                       # status colours are not drawn on muted
            assert qtgui.contrast(t[fg], t[bg]) >= 4.5, (name, fg, bg, qtgui.contrast(t[fg], t[bg]))
    assert qtgui.contrast(t["on_primary"], t["primary"]) >= 4.5
    assert qtgui.contrast(t["fg"], t["chunk"]) >= 4.5          # progress text on the filled part
    assert qtgui.contrast(t["fg"], t["error_bg"]) >= 4.5
    for c in t["series"]:                                       # chart lines vs plot background
        assert qtgui.contrast(c, t["surface"]) >= 3.0


def test_convergence_text_readable_without_colour():
    c = {"max_ratio": 8.13, "median_ratio": 1.0, "min_neff": 12.0, "converged": False}
    assert qtgui.convergence_text(c) == "max 8.13 > 3 · median 1.00 < 1.3 · n_eff 12.0 ≥ 10"


def test_explain_error_suggests_key():
    d = gui.parse_config(gui.TEMPLATES["Slab + O from O2, LAMMPS MLIP (skeleton)"].replace(
        "  x0: 0.85", "  x0: 0.85\n  temperture_K: 300"))
    text = qtgui.explain_error(d["error"])
    assert "'temperture_K'" in text and "Did you mean 'temperatures_K'?" in text and "allowed" not in text


def test_ladders():
    assert qtgui.geometric_ladder(300, 900, 3) == [300, 520, 900]
    assert qtgui.log_ladder(1e-14, 1.0, 8)[:2] == [1e-14, 1e-12] and qtgui.log_ladder(1e-2, 1, 3)[-1] == 1.0


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait(app, cond, timeout=300):
    t0 = time.time()
    while not cond():
        assert time.time() - t0 < timeout
        app.processEvents()
        time.sleep(0.5)


def test_window_build_run_monitor(app, tmp_path):
    w = qtgui.MainWindow(str(tmp_path), pythonpath=ROOT)
    w.auto.setChecked(False)
    try:
        b = w.build
        # template -> form -> shorter run -> YAML
        b.load("t", next(iter(gui.TEMPLATES)))
        b.fields["replica_exchange.n_steps"][0].setText("40")
        b.fields["replica_exchange.convergence_every"][0].setText("20")
        b.fields["checkpoint.chunk"][0].setText("20")
        b.form_to_yaml()
        cfg = yaml.safe_load(b.yaml.toPlainText())
        assert cfg["replica_exchange"]["n_steps"] == 40 and "Valid" in b.msg.text.text()
        b.save_path.setText("../outside.yaml")
        assert b.save() is None and "outside" in b.msg.text.text()
        b.save_path.setText("t/run.yaml")
        assert b.save() == os.path.join("t", "run.yaml")
        assert b.save() is None                     # exists, overwrite unticked
        # the Runs tab lists it; start it
        assert [r["path"] for r in w.runs.rows] == [os.path.join("t", "run.yaml")]
        w.runs.table.selectRow(0)
        assert w.runs.buttons["run"].isEnabled() and not w.runs.buttons["stop"].isEnabled()
        b.yaml.insertPlainText("# edited\n")
        assert b.dirty and w.windowTitle().startswith("•")
        b.overwrite.setChecked(True)
        assert b.save() and not b.dirty
        assert w.runs.start("t/run.yaml", "run")
        out = tmp_path / "t" / "output_test_emt"

        def done():
            w.refresh()
            return w.runs.rows[0]["job"]["state"] in ("finished", "converged", "error")
        wait(app, done)
        assert w.runs.rows[0]["job"]["state"] in ("finished", "converged"), \
            (out / "run.log").read_text()
        w.runs.table.selectRow(0)
        w.runs.selection_changed()
        assert w.runs.buttons["resume"].isEnabled() and not w.runs.buttons["run"].isEnabled()
        assert "Outputs in" in w.runs.log.toPlainText()
        # monitor: grid, traces, convergence drawn; clicking a cell selects that replica
        w.runs.action("monitor")
        m = w.monitor
        assert m.out_dir == str(out) and m.data["status"]["step"] == 40
        assert m.tiles["step"].value.text() == "40" and len(m.fig.axes) >= 4
        assert m.tiles["state"].value.text() in ("finished", "converged")
        ev = type("E", (), {"inaxes": m.ax_grid, "xdata": 2.1, "ydata": 0.9})()
        m.on_click(ev)
        assert m.sel == (1, 2)
        for i in range(m.grid_q.count()):          # every grid quantity draws (or says absent)
            m.grid_q.setCurrentIndex(i)
    finally:
        w.build.set_dirty(False)        # no unsaved-changes dialog offscreen
        w.close()


def test_apply_field_keeps_yaml_edits_since_sync():
    cfg = {"replica_exchange": {"n_steps": 999}}      # YAML edited after the form was filled
    qtgui.apply_field(cfg, "replica_exchange.n_steps", "int", "400", synced="400")
    assert cfg["replica_exchange"]["n_steps"] == 999
    qtgui.apply_field(cfg, "replica_exchange.n_steps", "int", "500", synced="400")
    assert cfg["replica_exchange"]["n_steps"] == 500


def test_ladders_reject_bad_input():
    for bad in ((0, 900, 3), (300, -1, 3), (300, 900, 0)):
        with pytest.raises(ValueError):
            qtgui.geometric_ladder(*bad)
    with pytest.raises(ValueError):
        qtgui.log_ladder(0.0, 1.0, 3)


def test_window_outside_output_form_and_external_run(app, tmp_path):
    """No run launched: output dirs outside the workspace, Form -> YAML, an external run."""
    ws_dir, scratch = tmp_path / "ws", tmp_path / "scratch"
    for d in (ws_dir / "a", ws_dir / "b", scratch / "out"):
        d.mkdir(parents=True)
    cfg = yaml.safe_load(next(iter(gui.TEMPLATES.values())))
    cfg["output"] = {"directory": str(scratch / "out")}                 # absolute, outside
    (ws_dir / "a" / "run.yaml").write_text(yaml.safe_dump(cfg))
    cfg["output"] = {"directory": "../../elsewhere"}                     # relative, outside
    (ws_dir / "b" / "run.yaml").write_text(yaml.safe_dump(cfg))
    # a run started by a script into the scratch output: fresh heartbeat, a checkpoint
    (scratch / "out" / "checkpoint.pkl").write_bytes(b"x")
    (scratch / "out" / "status.json").write_text(
        '{"time": %f, "started": %f, "step": 10, "n_steps": 400}' % (time.time(), time.time()))
    w = qtgui.MainWindow(str(ws_dir), pythonpath=ROOT)
    w.auto.setChecked(False)
    try:
        by = {r["path"]: r for r in w.runs.rows}
        assert by[os.path.join("a", "run.yaml")]["output_dir"] == str(scratch / "out")
        job = by[os.path.join("a", "run.yaml")]["job"]
        assert job["state"] == "running" and job["external"]
        w.runs.select(os.path.join("a", "run.yaml"))
        w.runs.selection_changed()
        for m in ("run", "resume", "overwrite", "analyze", "stop"):
            assert not w.runs.buttons[m].isEnabled(), m
        with pytest.raises(ValueError, match="outside the workspace"):   # launching stays inside
            w.ws.start(os.path.join("a", "run.yaml"), "overwrite")
        assert w.monitor.run_box.findData(str(scratch / "out")) >= 0
        w.runs.select(os.path.join("b", "run.yaml"))            # ../../elsewhere: no crash
        w.runs.selection_changed()
        assert w.runs.buttons["run"].isEnabled()
        w.runs.action("monitor")
        assert w.monitor.out_dir == str(tmp_path / "elsewhere")
        # Form -> YAML keeps YAML edits made after the form was filled, and can be undone
        b = w.build
        b.load("t", next(iter(gui.TEMPLATES)), ask=False)
        edited = b.yaml.toPlainText().replace("n_steps: 400", "n_steps: 999") + \
            "subsurface:\n  depth_angstrom: 2.0\n"
        b.yaml.setPlainText(edited)
        b.fields["replica_exchange.seed"][0].setText("7")
        b.form_to_yaml()
        new = yaml.safe_load(b.yaml.toPlainText())
        assert new["replica_exchange"]["n_steps"] == 999 and new["replica_exchange"]["seed"] == 7
        assert new["subsurface"] == {"depth_angstrom": 2.0}
        assert b.fields["replica_exchange.n_steps"][0].text() == "999"   # form re-synced
        b.yaml.undo()
        assert b.yaml.toPlainText() == edited
        # YAML that does not parse: left alone, error shown
        b.yaml.setPlainText("replica_exchange: [1,\n")
        b.form_to_yaml()
        assert b.yaml.toPlainText() == "replica_exchange: [1,\n"
        assert "not applied" in b.msg.text.text()
        # ladder from 0 K: an error message, no exception
        b.t0.setText("0")
        b.t_ladder()
        assert "Ladder" in b.msg.text.text()
    finally:
        w.build.set_dirty(False)
        w.close()
