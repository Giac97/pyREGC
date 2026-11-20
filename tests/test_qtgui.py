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
        assert cfg["replica_exchange"]["n_steps"] == 40 and "Valid" in b.msg.text()
        b.save_path.setText("../outside.yaml")
        assert b.save() is None and "outside" in b.msg.text()
        b.save_path.setText("t/run.yaml")
        assert b.save() == os.path.join("t", "run.yaml")
        assert b.save() is None                     # exists, overwrite unticked
        # the Runs tab lists it; start it
        assert [r["path"] for r in w.runs.rows] == [os.path.join("t", "run.yaml")]
        w.runs.table.selectRow(0)
        assert w.runs.buttons["run"].isEnabled() and not w.runs.buttons["stop"].isEnabled()
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
        assert "step" in m.kpi.text() and len(m.fig.axes) >= 4
        ev = type("E", (), {"inaxes": m.ax_grid, "xdata": 2.1, "ydata": 0.9})()
        m.on_click(ev)
        assert m.sel == (1, 2)
        for i in range(m.grid_q.count()):          # every grid quantity draws (or says absent)
            m.grid_q.setCurrentIndex(i)
    finally:
        w.close()
