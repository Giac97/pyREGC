"""
Desktop GUI (Qt) to build, run and monitor regc simulations.

    regc gui [WORKSPACE]          # or: regc-gui [WORKSPACE]; python -m regc.qtgui
    regc gui --web [WORKSPACE]    # the browser version (regc.gui)

A window with three tabs over a workspace directory (default: the current
one; *File → Open workspace* changes it):

* **Build**: the config as YAML next to a form of the common keys, ladder
  helpers, *Validate* (`regc.runner.validate_config`), *Save* (inside the
  workspace only), *Check setup* (``regc-run --check``, output streamed),
  *Save & start*.
* **Runs**: every config in the workspace with the state of its output
  directory, progress, latest convergence check and subsurface totals;
  start, resume, start over, re-analyse, stop, log.
* **Monitor**: progress figures, the replica grid (click a cell for its
  traces), N / N_sub / U traces, the rho(u) convergence checks, swap
  acceptance and round trips, the end of the log.

Runs are launched by `regc.gui.Workspace` exactly as the examples'
``run_monitor.sh`` do (detached, ``<output>/run.log``, ``<output>/run.pid``),
so they keep running when the window closes and scripts, the web GUI and
this window all see the same runs. Needs PySide6 (``pip install PySide6``)
and matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from . import monitor
from .gui import FORM_FIELDS, TEMPLATES, Workspace, dump_config, parse_config
from .gui import _tail as tail_text

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as _err:  # pragma: no cover - reported by main()
    QtCore = QtGui = QtWidgets = None
    _QT_ERROR = _err
else:
    _QT_ERROR = None

REFRESH_MS = 5000
STATE_COLOURS = {"running": "#0072b2", "starting": "#0072b2", "finished": "#009e73",
                 "converged": "#009e73", "stalled": "#e69f00", "stopped": "#e69f00",
                 "error": "#d55e00", "new": "#8a8a8f", "ended": "#8a8a8f"}


# =============================================================================
# Form values (no Qt: also used by the tests)
# =============================================================================
def _get(cfg: dict, key: str):
    """
    A nested value by dotted key.

    Parameters
    ----------
    cfg : dict
        The config.
    key : str
        ``a.b.c``.

    Returns
    -------
    object
        The value, or None if absent.
    """
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set(cfg: dict, key: str, value) -> None:
    """
    Set (or, with None, remove) a nested value by dotted key; empty sections are dropped.

    Parameters
    ----------
    cfg : dict
        The config (changed in place).
    key : str
        ``a.b.c``.
    value : object
        The value; None removes the key.
    """
    parts = key.split(".")
    chain, cur = [], cfg
    for part in parts[:-1]:
        if not isinstance(cur.get(part), dict):
            if value is None:
                return
            cur[part] = {}
        chain.append((cur, part))
        cur = cur[part]
    if value is None:
        cur.pop(parts[-1], None)
        for parent, part in reversed(chain):
            if parent[part] == {}:
                del parent[part]
            else:
                break
    else:
        cur[parts[-1]] = value


def field_text(value) -> str:
    """
    How a config value is shown in a form field.

    Parameters
    ----------
    value : object
        The value.

    Returns
    -------
    str
        Lists comma-separated, mappings as JSON, None as empty.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(json.dumps(x) if isinstance(x, (list, dict)) else field_text(x)
                         for x in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def _scalar(text: str):
    """
    A form entry as a number, bool or string (None if empty).

    Parameters
    ----------
    text : str
        The entry.

    Returns
    -------
    object
        The value.
    """
    t = text.strip()
    if not t:
        return None
    if t in ("true", "false"):
        return t == "true"
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def apply_field(cfg: dict, key: str, kind: str, text: str) -> None:
    """
    Write one form entry into the config (unchanged entries leave the config alone).

    Parameters
    ----------
    cfg : dict
        The config (changed in place).
    key : str
        Dotted key.
    kind : str
        ``str``, ``num``, ``int``, ``val``, ``list``, ``bool`` or ``sel:...``.
    text : str
        The entry; empty removes the key.
    """
    if text == field_text(_get(cfg, key)):
        return                                     # untouched (keeps non-scalar values too)
    if kind == "list":
        value = [_scalar(x) for x in text.replace(",", " ").split()] or None
    elif kind == "str":
        value = text.strip() or None
    else:
        value = _scalar(text)
        if kind == "int" and isinstance(value, float) and value.is_integer():
            value = int(value)
    _set(cfg, key, value)


def geometric_ladder(t0: float, t1: float, n: int) -> List[int]:
    """
    A geometric temperature ladder, rounded to whole kelvin.

    Parameters
    ----------
    t0, t1 : float
        End points (K).
    n : int
        Number of temperatures.

    Returns
    -------
    list of int
        The ladder.
    """
    if n < 2:
        return [int(round(t0))]
    return [int(round(t0 * (t1 / t0) ** (i / (n - 1)))) for i in range(n)]


def log_ladder(p0: float, p1: float, n: int) -> List[float]:
    """
    Log-spaced pressures.

    Parameters
    ----------
    p0, p1 : float
        End points.
    n : int
        Number of pressures.

    Returns
    -------
    list of float
        The ladder (3 significant digits).
    """
    if n < 2:
        return [p0]
    e0, e1 = np.log10(p0), np.log10(p1)
    return [float(f"{10 ** (e0 + (e1 - e0) * i / (n - 1)):.3g}") for i in range(n)]


# =============================================================================
# Qt widgets
# =============================================================================
if QtWidgets is not None:
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.patches
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    def _mono() -> "QtGui.QFont":
        """A fixed-width font."""
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        f.setPointSize(max(9, f.pointSize()))
        return f

    def _button(text: str, slot, tip: str = "") -> "QtWidgets.QPushButton":
        """A push button wired to a slot."""
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(slot)
        if tip:
            b.setToolTip(tip)
        return b

    def _process_env(ws: Workspace) -> "QtCore.QProcessEnvironment":
        """The environment of processes started from the window."""
        env = QtCore.QProcessEnvironment.systemEnvironment()
        if ws.pythonpath:
            env.insert("PYTHONPATH", ws.pythonpath)
        env.insert("PYTHONUNBUFFERED", "1")
        return env

    # -------------------------------------------------------------- Build tab
    class BuildTab(QtWidgets.QWidget):
        """YAML editor, form, ladder helpers, validate / save / check."""

        def __init__(self, main: "MainWindow"):
            """
            Build the tab.

            Parameters
            ----------
            main : MainWindow
                The window (workspace, status bar, other tabs).
            """
            super().__init__()
            self.main = main
            self.fields: Dict[str, tuple] = {}
            self.proc: Optional[QtCore.QProcess] = None

            top = QtWidgets.QHBoxLayout()
            self.source = QtWidgets.QComboBox()
            self.source.setMinimumWidth(320)
            top.addWidget(self.source, 2)
            top.addWidget(_button("Load", self.load_selected, "load the template or config"))
            top.addSpacing(16)
            top.addWidget(QtWidgets.QLabel("Save as"))
            self.save_path = QtWidgets.QLineEdit()
            self.save_path.setPlaceholderText("workspace-relative, e.g. my_run/my_run.yaml")
            top.addWidget(self.save_path, 3)
            top.addWidget(_button("…", self.browse_save, "choose the file"))
            self.overwrite = QtWidgets.QCheckBox("overwrite")
            top.addWidget(self.overwrite)

            acts = QtWidgets.QHBoxLayout()
            acts.addWidget(_button("Validate", self.validate))
            save = _button("Save", self.save)
            save.setDefault(True)
            acts.addWidget(save)
            acts.addWidget(_button("Check setup", self.check,
                                   "regc-run --check: build the system, print the μ ladder"))
            acts.addWidget(_button("Save && start", self.save_and_start))
            acts.addSpacing(16)
            self.msg = QtWidgets.QLabel()
            self.msg.setWordWrap(True)
            self.msg.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            acts.addWidget(self.msg, 1)

            # left: YAML
            self.yaml = QtWidgets.QPlainTextEdit()
            self.yaml.setFont(_mono())
            self.yaml.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.yaml.setTabStopDistance(2 * self.yaml.fontMetrics().horizontalAdvance(" "))
            left = QtWidgets.QWidget()
            ll = QtWidgets.QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            sync = QtWidgets.QHBoxLayout()
            sync.addWidget(_button("YAML → form", self.yaml_to_form))
            sync.addWidget(_button("Form → YAML", self.form_to_yaml,
                                   "rewrites the YAML from the form (comments are lost)"))
            sync.addStretch(1)
            ll.addLayout(sync)
            ll.addWidget(self.yaml)

            # right: form + ladders
            form_w = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(form_w)
            form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            for f in FORM_FIELDS:
                if len(f) == 1:
                    lab = QtWidgets.QLabel(f"<b>{f[0]}</b>")
                    lab.setContentsMargins(0, 8, 0, 0)
                    form.addRow(lab)
                    continue
                key, label, kind = f[:3]
                if kind.startswith("sel:") or kind == "bool":
                    w = QtWidgets.QComboBox()
                    w.addItems(kind[4:].split("|") if kind.startswith("sel:") else ["", "true", "false"])
                else:
                    w = QtWidgets.QLineEdit()
                    if len(f) > 3:
                        w.setPlaceholderText(f[3])
                w.setToolTip(key)
                self.fields[key] = (w, kind)
                form.addRow(label, w)
            lad = QtWidgets.QGridLayout()
            self.t0, self.t1, self.tn = (QtWidgets.QLineEdit(v) for v in ("300", "900", "8"))
            self.p0, self.p1, self.pn = (QtWidgets.QLineEdit(v) for v in ("1e-14", "1", "8"))
            for row, (a, b, n, lab, slot) in enumerate((
                    (self.t0, self.t1, self.tn, "T from / to (K), n", self.t_ladder),
                    (self.p0, self.p1, self.pn, "p from / to, n", self.p_ladder))):
                lad.addWidget(QtWidgets.QLabel(lab), row, 0)
                for col, w in enumerate((a, b, n)):
                    w.setMaximumWidth(80)
                    lad.addWidget(w, row, 1 + col)
                lad.addWidget(_button("fill", slot), row, 4)
            form.addRow(QtWidgets.QLabel("<b>Ladder helpers</b>"))
            form.addRow(lad)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form_w)

            # bottom: check output
            self.out = QtWidgets.QPlainTextEdit()
            self.out.setReadOnly(True)
            self.out.setFont(_mono())
            self.out.setPlaceholderText("Check setup output (regc-run --check) appears here.")

            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            split.addWidget(left)
            split.addWidget(scroll)
            split.setSizes([700, 500])
            vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            vsplit.addWidget(split)
            vsplit.addWidget(self.out)
            vsplit.setSizes([700, 160])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(top)
            lay.addLayout(acts)
            lay.addWidget(vsplit, 1)

        # ---- helpers
        def say(self, text: str, ok: bool = True) -> None:
            """
            Show a message under the toolbar.

            Parameters
            ----------
            text : str
                The message.
            ok : bool, optional
                Green (True) or red. Default True.
            """
            self.msg.setStyleSheet(f"color: {'#009e73' if ok else '#d55e00'}")
            self.msg.setText(text)

        def refresh_sources(self, configs: List[str]) -> None:
            """
            Fill the template / config chooser.

            Parameters
            ----------
            configs : list of str
                Workspace-relative config paths.
            """
            cur = self.source.currentData()
            self.source.blockSignals(True)
            self.source.clear()
            for t in TEMPLATES:
                self.source.addItem(f"template: {t}", ("t", t))
            for c in configs:
                self.source.addItem(c, ("c", c))
            if cur is not None:
                i = self.source.findData(cur)
                if i >= 0:
                    self.source.setCurrentIndex(i)
            self.source.blockSignals(False)

        def load_selected(self) -> None:
            """Load the chosen template or config."""
            data = self.source.currentData()
            if data:
                self.load(*data)

        def load(self, kind: str, name: str) -> None:
            """
            Load a template (``t``) or a workspace config (``c``) into the editor and form.

            Parameters
            ----------
            kind : str
                ``t`` or ``c``.
            name : str
                Template name or workspace-relative path.
            """
            if kind == "t":
                self.yaml.setPlainText(TEMPLATES[name])
                self.save_path.clear()
            else:
                try:
                    with open(self.main.ws.resolve(name)) as f:
                        self.yaml.setPlainText(f.read())
                except (OSError, ValueError) as err:
                    return self.say(str(err), False)
                self.save_path.setText(name)
            self.overwrite.setChecked(False)
            self.yaml_to_form()

        def parse(self) -> Optional[dict]:
            """
            Parse and validate the editor's text; report on the message line.

            Returns
            -------
            dict or None
                The config, or None if it does not parse.
            """
            d = parse_config(self.yaml.toPlainText())
            s = d["summary"]
            summary = (f"{s['replicas']} replicas ({s['n_T']} T × {s['n_mu']} μ/p), "
                       f"n_steps {s['n_steps']}, ~{s['samples_per_replica']} samples per replica"
                       if s else "")
            if d["config"] is None:
                self.say(d["error"], False)
                return None
            self.say(("Valid. " + summary) if d["ok"] else f"{d['error']}  [{summary}]", d["ok"])
            return d["config"]

        def validate(self) -> None:
            """Validate the YAML."""
            self.parse()

        def yaml_to_form(self) -> None:
            """Fill the form from the YAML."""
            cfg = self.parse()
            if cfg is None:
                return
            for key, (w, kind) in self.fields.items():
                text = field_text(_get(cfg, key))
                if isinstance(w, QtWidgets.QComboBox):
                    if w.findText(text) < 0:
                        w.addItem(text)
                    w.setCurrentText(text)
                else:
                    w.setText(text)
                    w.setCursorPosition(0)

        def form_to_yaml(self) -> None:
            """Rewrite the YAML with the form's values."""
            cfg = self.parse() or {}
            for key, (w, kind) in self.fields.items():
                text = w.currentText() if isinstance(w, QtWidgets.QComboBox) else w.text()
                apply_field(cfg, key, kind, text)
            self.yaml.setPlainText(dump_config(cfg))
            self.parse()

        def t_ladder(self) -> None:
            """Fill T with a geometric ladder."""
            try:
                T = geometric_ladder(float(self.t0.text()), float(self.t1.text()), int(self.tn.text()))
            except ValueError as err:
                return self.say(f"ladder: {err}", False)
            self.fields["replica_exchange.temperatures_K"][0].setText(", ".join(map(str, T)))

        def p_ladder(self) -> None:
            """Fill the pressures with a log-spaced ladder."""
            try:
                p = log_ladder(float(self.p0.text()), float(self.p1.text()), int(self.pn.text()))
            except ValueError as err:
                return self.say(f"ladder: {err}", False)
            self.fields["replica_exchange.pressures"][0].setText(", ".join(f"{x:g}" for x in p))

        def browse_save(self) -> None:
            """Choose where to save."""
            start = self.main.ws.resolve(self.save_path.text() or ".")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", start,
                                                            "YAML (*.yaml *.yml)")
            if path:
                self.save_path.setText(self.main.ws.rel(path))
                self.overwrite.setChecked(os.path.exists(path))   # the dialog already asked

        def save(self) -> Optional[str]:
            """
            Save the YAML to the chosen path.

            Returns
            -------
            str or None
                Workspace-relative path, or None on failure.
            """
            rel = self.save_path.text().strip()
            try:
                if not rel:
                    raise ValueError("give a path to save to")
                p = self.main.ws.resolve(rel)
                if not p.endswith((".yaml", ".yml")):
                    raise ValueError("config files end in .yaml or .yml")
                if os.path.exists(p) and not self.overwrite.isChecked():
                    raise ValueError(f"{rel} exists (tick 'overwrite' to replace it)")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(self.yaml.toPlainText())
            except (OSError, ValueError) as err:
                self.say(str(err), False)
                return None
            self.say(f"Saved {self.main.ws.rel(p)}")
            self.main.refresh()
            return self.main.ws.rel(p)

        def save_and_start(self) -> None:
            """Save, then start the run and show the Runs tab."""
            if self.parse() is None:
                return
            rel = self.save()
            if rel and self.main.runs.start(rel, "run"):
                self.main.tabs.setCurrentWidget(self.main.runs)

        def check(self) -> None:
            """Run ``regc-run --check`` on the saved config, streaming its output."""
            rel = self.save_path.text().strip()
            try:
                if not rel or not os.path.isfile(self.main.ws.resolve(rel)):
                    raise ValueError("save the config first")
            except ValueError as err:
                return self.say(str(err), False)
            if self.proc is not None and self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
                return self.say("a check is still running", False)
            path = self.main.ws.resolve(rel)
            self.out.setPlainText(f"$ regc-run {os.path.basename(path)} --check\n")
            self.proc = QtCore.QProcess(self)
            self.proc.setProcessEnvironment(_process_env(self.main.ws))
            self.proc.setWorkingDirectory(os.path.dirname(path))
            self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.proc.readyReadStandardOutput.connect(
                lambda: self._append(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")))
            self.proc.finished.connect(lambda code, _s: self._append(f"\n[exit {code}]\n"))
            self.proc.start(self.main.ws.python, self.main.ws._cmd(path, "--check")[1:])

        def _append(self, text: str) -> None:
            """Append to the check output."""
            self.out.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.out.insertPlainText(text)
            self.out.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # --------------------------------------------------------------- Runs tab
    class RunsTab(QtWidgets.QWidget):
        """Configs, their runs' states, and the run actions."""

        COLUMNS = ["config", "output", "state", "progress", "convergence", "subsurface", "last log line"]

        def __init__(self, main: "MainWindow"):
            """
            Build the tab.

            Parameters
            ----------
            main : MainWindow
                The window.
            """
            super().__init__()
            self.main = main
            self.rows: List[dict] = []
            bar = QtWidgets.QHBoxLayout()
            self.buttons = {}
            for mode, label, tip in (
                    ("edit", "Edit", "open the config on the Build tab"),
                    ("run", "Start", "start a new run (refused over a checkpoint)"),
                    ("resume", "Resume", "continue from checkpoint.pkl (--resume)"),
                    ("overwrite", "Start over", "replace checkpoint, samples and records (--overwrite)"),
                    ("analyze", "Analyze", "re-run the analysis (--analyze-only)"),
                    ("stop", "Stop", "SIGTERM; the last checkpoint stays"),
                    ("monitor", "Monitor", "show it on the Monitor tab")):
                b = _button(label, lambda _=False, m=mode: self.action(m), tip)
                self.buttons[mode] = b
                bar.addWidget(b)
            bar.addStretch(1)
            bar.addWidget(QtWidgets.QLabel("workers"))
            self.workers = QtWidgets.QSpinBox()
            self.workers.setRange(0, 512)
            self.workers.setSpecialValueText("config")
            self.workers.setToolTip("--workers (config: the config's n_workers)")
            bar.addWidget(self.workers)

            self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
            self.table.setHorizontalHeaderLabels(self.COLUMNS)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.verticalHeader().setVisible(False)
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hh.setStretchLastSection(True)
            self.table.itemSelectionChanged.connect(self.selection_changed)
            self.table.doubleClicked.connect(lambda _i: self.action("monitor"))

            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setFont(_mono())
            self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            split.addWidget(self.table)
            split.addWidget(self.log)
            split.setSizes([600, 250])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(bar)
            lay.addWidget(split, 1)
            self.selection_changed()

        def selected(self) -> Optional[dict]:
            """The selected config row, or None."""
            r = self.table.currentRow()
            return self.rows[r] if 0 <= r < len(self.rows) else None

        def update_rows(self, configs: List[dict]) -> None:
            """
            Refill the table, keeping the selection.

            Parameters
            ----------
            configs : list of dict
                `Workspace.overview` rows.
            """
            keep = (self.selected() or {}).get("path")
            self.rows = configs
            self.table.setRowCount(len(configs))
            for i, c in enumerate(configs):
                j = c["job"]
                conv, sub = j.get("convergence"), j.get("subsurface")
                if conv and conv.get("max_ratio") is not None:
                    conv_t = (f"max {conv['max_ratio']:.2f} · med {conv['median_ratio']:.2f} · "
                              f"n_eff {conv['min_neff']:.1f}" + (" ✓" if conv.get("converged") else ""))
                elif conv and conv.get("max_change") is not None:
                    conv_t = f"Δρ {conv['max_change']:.3f}"
                else:
                    conv_t = ""
                sub_t = (f"{sub['n_now']} now · ins {sub['inserted']}/{sub['insert_attempts']} · "
                         f"rem {sub['removed']}/{sub['delete_attempts']}") if sub else ""
                prog = (f"{j['step']:,} / {j['n_steps']:,}" if j.get("step") is not None
                        and j.get("n_steps") else "")
                cells = [c["path"], c["output"], j["state"] + (f" (pid {j['pid']})" if j["pid"] else ""),
                         prog, conv_t, sub_t, j.get("last_line", "")]
                for col, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    if col == 2:
                        item.setForeground(QtGui.QColor(STATE_COLOURS.get(j["state"], "#8a8a8f")))
                        f = item.font()
                        f.setBold(True)
                        item.setFont(f)
                    if col == 6:
                        item.setToolTip(text)
                    self.table.setItem(i, col, item)
                if j.get("step") is not None and j.get("n_steps"):
                    pb = QtWidgets.QProgressBar()
                    pb.setRange(0, int(j["n_steps"]))
                    pb.setValue(min(int(j["step"]), int(j["n_steps"])))
                    pb.setFormat(prog)
                    self.table.setCellWidget(i, 3, pb)
                else:
                    self.table.removeCellWidget(i, 3)
            if keep:
                for i, c in enumerate(configs):
                    if c["path"] == keep:
                        self.table.selectRow(i)
                        break
            self.selection_changed()

        def selection_changed(self) -> None:
            """Enable the actions that fit the selected run, and show its log."""
            c = self.selected()
            j = (c or {}).get("job") or {}
            live = j.get("state") in ("running", "starting", "stalled")
            resumable = bool(j.get("resumable"))
            enabled = {"edit": bool(c), "run": bool(c) and not live and not resumable,
                       "resume": bool(c) and not live and resumable,
                       "overwrite": bool(c) and not live and resumable,
                       "analyze": bool(c) and not live, "stop": bool(c) and live,
                       "monitor": bool(c)}
            for m, b in self.buttons.items():
                b.setEnabled(enabled[m])
            if c:
                out = self.main.ws.resolve(c["output"])
                text = tail_text(os.path.join(out, "run.log"), 20_000)
                bar = self.log.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 2
                if text != self.log.toPlainText():
                    self.log.setPlainText(text or "(no run.log)")
                    if at_end:
                        bar.setValue(bar.maximum())
            else:
                self.log.setPlainText("")

        def start(self, rel: str, mode: str) -> bool:
            """
            Launch a run.

            Parameters
            ----------
            rel : str
                Workspace-relative config.
            mode : str
                ``run``, ``resume``, ``overwrite`` or ``analyze``.

            Returns
            -------
            bool
                True if launched.
            """
            try:
                d = self.main.ws.start(rel, mode, self.workers.value() or None)
            except (ValueError, OSError) as err:
                QtWidgets.QMessageBox.warning(self, "regc", str(err))
                return False
            self.main.statusBar().showMessage(f"{mode}: {rel}, pid {d['pid']}, log {d['log']}", 10000)
            self.main.refresh()
            return True

        def action(self, mode: str) -> None:
            """
            Run an action on the selected config.

            Parameters
            ----------
            mode : str
                ``edit``, ``run``, ``resume``, ``overwrite``, ``analyze``, ``stop`` or ``monitor``.
            """
            c = self.selected()
            if c is None:
                return
            if mode == "edit":
                self.main.build.load("c", c["path"])
                self.main.tabs.setCurrentWidget(self.main.build)
            elif mode == "monitor":
                self.main.monitor.show_run(self.main.ws.resolve(c["output"]))
                self.main.tabs.setCurrentWidget(self.main.monitor)
            elif mode == "stop":
                if QtWidgets.QMessageBox.question(
                        self, "Stop run", f"Stop the run of {c['path']}?\n"
                        f"It can be resumed from its last checkpoint.") != \
                        QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                try:
                    d = self.main.ws.stop(c["path"])
                    self.main.statusBar().showMessage(f"sent SIGTERM to {d['stopped']}", 10000)
                except (ValueError, OSError) as err:
                    QtWidgets.QMessageBox.warning(self, "regc", str(err))
                self.main.refresh()
            else:
                if mode == "overwrite" and QtWidgets.QMessageBox.question(
                        self, "Start over", f"Start {c['path']} over?\nIts checkpoint, samples and "
                        f"records are replaced.") != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                self.start(c["path"], mode)

    # ------------------------------------------------------------ Monitor tab
    class MonitorTab(QtWidgets.QWidget):
        """Native view of one run's heartbeat: KPIs, replica grid, traces, convergence, swaps, log."""

        GRID_QUANTITIES = [("mean_N", "mean N"), ("current_N", "current N"), ("std_N", "std of N"),
                           ("n_samples", "samples"), ("mean_U", "mean U (eV)"),
                           ("current_U", "current U (eV)"), ("mean_N_ads", "mean N adsorbed"),
                           ("mean_N_sub", "mean N subsurface"),
                           ("current_N_sub", "current N subsurface"),
                           ("sub_accepted", "subsurface moves accepted"),
                           ("sub_acceptance", "subsurface acceptance")]

        def __init__(self, main: "MainWindow"):
            """
            Build the tab.

            Parameters
            ----------
            main : MainWindow
                The window.
            """
            super().__init__()
            self.main = main
            self.out_dir: Optional[str] = None
            self.sel = None                   # selected replica (iT, imu)
            self.history: Dict[str, deque] = {}
            self.data: Optional[dict] = None
            self._mtime = None

            top = QtWidgets.QHBoxLayout()
            self.run_box = QtWidgets.QComboBox()
            self.run_box.setMinimumWidth(380)
            self.run_box.activated.connect(lambda _i: self.show_run(self.run_box.currentData()))
            top.addWidget(QtWidgets.QLabel("Run"))
            top.addWidget(self.run_box, 1)
            top.addWidget(_button("Open directory…", self.open_dir, "monitor any output directory"))
            top.addWidget(_button("Refresh", lambda: self.refresh(force=True)))
            self.grid_q = QtWidgets.QComboBox()
            for k, label in self.GRID_QUANTITIES:
                self.grid_q.addItem(label, k)
            self.grid_q.currentIndexChanged.connect(lambda _i: self.draw())
            top.addWidget(QtWidgets.QLabel("grid"))
            top.addWidget(self.grid_q)

            self.kpi = QtWidgets.QLabel("Choose a run.")
            self.kpi.setWordWrap(True)
            self.kpi.setTextFormat(QtCore.Qt.TextFormat.RichText)
            self.kpi.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

            self.fig = Figure(figsize=(10, 7), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setFont(_mono())
            self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            split.addWidget(self.canvas)
            split.addWidget(self.log)
            split.setSizes([700, 150])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(top)
            lay.addWidget(self.kpi)
            lay.addWidget(split, 1)

        def update_runs(self, configs: List[dict]) -> None:
            """
            Fill the run chooser with the workspace's output directories.

            Parameters
            ----------
            configs : list of dict
                `Workspace.overview` rows.
            """
            have = {self.run_box.itemData(i) for i in range(self.run_box.count())}
            for c in configs:
                out = self.main.ws.resolve(c["output"])
                if out not in have:
                    self.run_box.addItem(f"{c['output']}   ({c['path']})", out)
                    have.add(out)

        def open_dir(self) -> None:
            """Monitor an arbitrary output directory."""
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory",
                                                           self.out_dir or self.main.ws.root)
            if d:
                self.show_run(d)

        def show_run(self, out_dir: Optional[str]) -> None:
            """
            Switch to a run.

            Parameters
            ----------
            out_dir : str
                Its output directory.
            """
            if not out_dir:
                return
            out_dir = os.path.abspath(out_dir)
            if self.run_box.findData(out_dir) < 0:
                self.run_box.addItem(out_dir, out_dir)
            self.run_box.setCurrentIndex(self.run_box.findData(out_dir))
            if out_dir != self.out_dir:
                self.out_dir, self.sel, self._mtime = out_dir, None, None
            self.refresh(force=True)

        # ---- data
        def load(self) -> Optional[dict]:
            """
            The run's data: the heartbeat, else the last checkpoint's samples.

            Returns
            -------
            dict or None
                ``status``, ``samples`` (fallback), ``metadata``, ``job``, ``log``.
            """
            d = self.out_dir
            if not d:
                return None
            status = monitor._read_json(os.path.join(d, monitor.STATUS_FILE))
            samples = None
            if status is None and os.path.isfile(os.path.join(d, "samples.npz")):
                samples = monitor._samples_summary(os.path.join(d, "samples.npz"))
            return {"status": status, "samples": samples,
                    "metadata": monitor._read_json(os.path.join(d, "run_metadata.json")),
                    "mixing": monitor._read_json(os.path.join(d, "mixing.json")),
                    "job": self.main.ws.job(d),
                    "log": tail_text(os.path.join(d, "run.log"), 12_000)}

        def refresh(self, force: bool = False) -> None:
            """
            Reload and redraw if the heartbeat changed (or `force`).

            Parameters
            ----------
            force : bool, optional
                Redraw anyway. Default False.
            """
            if not self.out_dir:
                return
            path = os.path.join(self.out_dir, monitor.STATUS_FILE)
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
            if not force and mtime == self._mtime and self.data is not None:
                self.data["job"] = self.main.ws.job(self.out_dir)
                self.draw_kpis()
                return
            self._mtime = mtime
            self.data = self.load()
            st = (self.data or {}).get("status")
            if st:
                h = self.history.setdefault(self.out_dir, deque(maxlen=500))
                if not h or h[-1][1] != st["step"]:
                    h.append((st["time"], st["step"]))
            self.draw()

        # ---- drawing
        def _axes_labels(self, shape) -> tuple:
            """Row (T) and column (p or μ) tick labels."""
            st, meta = self.data.get("status") or {}, self.data.get("metadata") or {}
            lab = st.get("labels") or {}
            T = lab.get("temperatures_K") or meta.get("temperatures_K")
            P = lab.get("pressures") or meta.get("pressures")
            unit = lab.get("pressure_unit") or meta.get("pressure_unit") or ""
            rows = [f"{t:g} K" for t in T] if T else [str(i) for i in range(shape[0])]
            cols = [f"{p:.0e}" for p in P] if P else [str(j) for j in range(shape[1])]
            return rows, cols, (f"p ({unit})" if P else "μ column")

        def grid(self) -> tuple:
            """
            The replica-grid array of the chosen quantity.

            Returns
            -------
            tuple
                ``(z, keys, traces)`` with z an (n_T, n_mu) array (NaN: no
                value), or ``(None, keys, traces)`` if not recorded.
            """
            st, sm = self.data.get("status"), self.data.get("samples")
            q = self.grid_q.currentData()
            if st:
                src = st["replicas"].get(q)
                z = None if src is None else np.array(
                    [[np.nan if v is None else v for v in row] for row in src], dtype=float)
                return z, [tuple(k) for k in st["keys"]], st["traces"]
            if sm:
                keys = [tuple(k) for k in sm["keys"]]
                n_T = max(k[0] for k in keys) + 1
                n_m = max(k[1] for k in keys) + 1
                vals = {"mean_N": sm["mean_N"], "std_N": sm["std_N"], "mean_U": sm["mean_U"],
                        "n_samples": [sm["n_samples"]] * len(keys)}.get(q)
                if vals is None:
                    return None, keys, sm["traces"]
                z = np.full((n_T, n_m), np.nan)
                for k, v in zip(keys, vals):
                    z[k] = v
                return z, keys, sm["traces"]
            return None, [], None

        def draw_kpis(self) -> None:
            """The text line above the figures."""
            d, st = self.data or {}, (self.data or {}).get("status")
            j = d.get("job") or {}
            colour = STATE_COLOURS.get(j.get("state"), "#8a8a8f")
            parts = [f"<b style='color:{colour}'>{j.get('state', '?')}</b>"]
            if st:
                step, n = st["step"], st["n_steps"]
                parts.append(f"step <b>{step:,}</b> / {n:,} ({100 * step / max(n, 1):.1f} %)")
                h = [x for x in self.history.get(self.out_dir, []) if time.time() - x[0] < 1800]
                rate = None
                if len(h) >= 2 and h[-1][1] > h[0][1]:
                    rate = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
                elif st.get("started") and st["time"] > st["started"]:
                    rate = step / (st["time"] - st["started"])
                if rate:
                    parts.append(f"{rate:.3g} steps/s")
                    if j.get("state") in ("running", "stalled"):
                        parts.append(f"≤ {_fmt_dt(max(0, n - step) / rate)} left (cap)")
                if st.get("started"):
                    parts.append(f"elapsed {_fmt_dt(st['time'] - st['started'])}")
                cur = [x for row in st["replicas"]["current_N"] for x in row if x is not None]
                if cur:
                    parts.append(f"mean N now {np.mean(cur):.2f}")
                sub = st.get("subsurface")
                if sub:
                    sp = "+".join(sub["species"])
                    parts.append(f"subsurface {sp}: <b>{sub['n_now']}</b> now in "
                                 f"{sub['replicas_occupied']} replicas · inserted {sub['inserted']}/"
                                 f"{sub['insert_attempts']} · removed {sub['removed']}/"
                                 f"{sub['delete_attempts']}")
                checks = (st.get("convergence") or {}).get("checks") or []
                if checks and checks[-1].get("max_ratio") is not None:
                    c, s = checks[-1], st["convergence"]["settings"] or {}
                    parts.append(f"ρ(u) halves: max {c['max_ratio']:.2f} (tol {s.get('tol', 3):g}), "
                                 f"median {c['median_ratio']:.2f} (tol {s.get('median_tol', 1.3):g}), "
                                 f"n_eff {c['min_neff']:.1f} (need {s.get('min_neff', 10):g})"
                                 + (" — <b>converged</b>" if c.get("converged") else ""))
                age = time.time() - st["time"]
                parts.append(f"heartbeat {_fmt_dt(age)} ago")
            elif d.get("samples"):
                parts.append(f"no status.json: samples.npz ({d['samples']['n_samples']} samples "
                             f"per replica)")
            else:
                parts.append("no data yet")
            self.kpi.setText(" &nbsp;·&nbsp; ".join(parts))

        def draw(self) -> None:
            """Redraw everything."""
            if not self.data:
                return
            self.draw_kpis()
            log = self.data.get("log") or ""
            if log != self.log.toPlainText():
                self.log.setPlainText(log or "(no run.log)")
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            fig = self.fig
            fig.clear()
            gs = fig.add_gridspec(2, 2)
            ax_g, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
            ax_c, ax_s = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
            self.ax_grid = ax_g
            z, keys, traces = self.grid()
            st = self.data.get("status")
            # grid
            if z is not None:
                rows, cols, xl = self._axes_labels(z.shape)
                im = ax_g.imshow(z, origin="lower", aspect="auto", cmap="viridis")
                fig.colorbar(im, ax=ax_g, fraction=0.05, pad=0.02)
                ax_g.set_xticks(range(z.shape[1]), cols, rotation=45, fontsize=7, ha="right")
                ax_g.set_yticks(range(z.shape[0]), rows, fontsize=7)
                ax_g.set_xlabel(xl, fontsize=8)
                if z.size <= 150:
                    finite = z[np.isfinite(z)]
                    mid = np.nanmean(finite) if finite.size else 0
                    for (i, k), v in np.ndenumerate(z):
                        if np.isfinite(v):
                            ax_g.text(k, i, f"{v:.3g}", ha="center", va="center", fontsize=6,
                                      color="white" if v < mid else "black")
                if self.sel is None and keys:
                    self.sel = keys[0]
                if self.sel is not None:
                    ax_g.add_patch(matplotlib.patches.Rectangle(
                        (self.sel[1] - 0.5, self.sel[0] - 0.5), 1, 1, fill=False, lw=2, ec="#d55e00"))
                ax_g.set_title(f"{self.grid_q.currentText()} (click a cell)", fontsize=9)
            else:
                ax_g.text(0.5, 0.5, "not recorded in this run", ha="center", va="center",
                          transform=ax_g.transAxes, color="0.5")
                ax_g.set_axis_off()
            # traces
            if traces and keys and self.sel in keys:
                i = keys.index(self.sel)
                stride = traces.get("stride", 1)
                N = np.asarray(traces["N"][i], dtype=float)
                x = np.arange(len(N)) * stride
                ax_t.plot(x, N, lw=1, color="#0072b2", label="N")
                if traces.get("N_sub") and len(traces["N_sub"][i]):
                    ax_t.plot(x[:len(traces["N_sub"][i])], traces["N_sub"][i], lw=1.2,
                              color="#009e73", label="N subsurface")
                ax_t.set_xlabel("sample", fontsize=8)
                ax_t.set_ylabel("N", fontsize=8)
                ax_u = ax_t.twinx()
                ax_u.plot(x, traces["U"][i], lw=0.8, ls=":", color="#e69f00", label="U")
                ax_u.set_ylabel("U (eV)", fontsize=8)
                ax_u.tick_params(labelsize=7)
                h1, l1 = ax_t.get_legend_handles_labels()
                h2, l2 = ax_u.get_legend_handles_labels()
                ax_t.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
                rows, cols, _ = self._axes_labels((max(k[0] for k in keys) + 1,
                                                   max(k[1] for k in keys) + 1))
                ax_t.set_title(f"replica {rows[self.sel[0]]}, {cols[self.sel[1]]}", fontsize=9)
            else:
                ax_t.set_axis_off()
            ax_t.tick_params(labelsize=7)
            # convergence
            conv = (st or {}).get("convergence") or {}
            checks = conv.get("checks") or []
            hist = (st or {}).get("convergence_history") or []
            s = conv.get("settings") or {}
            if checks and checks[0].get("max_ratio") is not None:
                steps = [c.get("step", i + 1) for i, c in enumerate(checks)]
                ax_c.plot(steps, [c["max_ratio"] for c in checks], "o-", color="#0072b2",
                          label="max TV / noise")
                ax_c.plot(steps, [c["median_ratio"] for c in checks], "s-", color="#e69f00",
                          label="median TV / noise")
                ax_c.axhline(s.get("tol", 3.0), ls="--", color="#0072b2", lw=0.8)
                ax_c.axhline(s.get("median_tol", 1.3), ls=":", color="#e69f00", lw=0.8)
                ax_c.set_xlabel("step", fontsize=8)
                ax_c.set_ylim(bottom=0)
                ax_c.legend(fontsize=7)
                ax_c.set_title(f"ρ(u) convergence (min n_eff {checks[-1]['min_neff']:.1f}, "
                               f"need {s.get('min_neff', 10):g})", fontsize=9)
            elif hist:
                ax_c.plot(range(1, len(hist) + 1), hist, "o-")
                ax_c.set_xlabel("check", fontsize=8)
                ax_c.set_title("ρ(u) change between blocks", fontsize=9)
            else:
                ax_c.text(0.5, 0.5, "no convergence checks yet", ha="center", va="center",
                          transform=ax_c.transAxes, color="0.5")
                ax_c.set_axis_off()
            ax_c.tick_params(labelsize=7)
            # swaps + round trips
            acc = (st or {}).get("swap_acceptance") or (self.data.get("mixing") or {}).get(
                "swap_acceptance") or {}
            if acc:
                names = list(acc)
                ax_s.bar(names, [acc[n] or 0 for n in names], color="#56b4e9")
                ax_s.set_ylim(0, 1)
                ax_s.set_ylabel("swap acceptance", fontsize=8)
                tT = (st or {}).get("round_trips_T")
                if tT:
                    ax_s.set_title(f"swaps; T round trips per walker: median {np.median(tT):.1f}, "
                                   f"{sum(1 for x in tT if x == 0)} of {len(tT)} with none",
                                   fontsize=9)
                else:
                    ax_s.set_title("swap acceptance", fontsize=9)
            else:
                ax_s.set_axis_off()
            ax_s.tick_params(labelsize=7)
            self.canvas.draw_idle()

        def on_click(self, event) -> None:
            """Select the replica under the mouse on the grid."""
            if getattr(self, "ax_grid", None) is None or event.inaxes is not self.ax_grid:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.sel = (int(round(event.ydata)), int(round(event.xdata)))
            self.draw()

    def _fmt_dt(s: float) -> str:
        """A duration as h/min/s."""
        s = float(s)
        if s < 120:
            return f"{s:.0f} s"
        if s < 7200:
            return f"{s / 60:.0f} min"
        return f"{s / 3600:.1f} h"

    # ------------------------------------------------------------ main window
    class MainWindow(QtWidgets.QMainWindow):
        """The window: workspace bar, Build / Runs / Monitor tabs, refresh timer."""

        def __init__(self, root: str = ".", python: Optional[str] = None,
                     pythonpath: Optional[str] = None):
            """
            Build the window.

            Parameters
            ----------
            root : str, optional
                Workspace directory. Default the current one.
            python, pythonpath : str, optional
                Interpreter and ``PYTHONPATH`` of the runs (see `regc.gui.Workspace`).
            """
            super().__init__()
            self.ws = Workspace(root, python, pythonpath)
            self.setWindowTitle(f"regc — {self.ws.root}")
            self.resize(1500, 950)

            bar = QtWidgets.QToolBar("workspace")
            bar.setMovable(False)
            self.addToolBar(bar)
            bar.addWidget(QtWidgets.QLabel(" Workspace "))
            self.root_label = QtWidgets.QLabel(self.ws.root)
            self.root_label.setStyleSheet("font-weight:600")
            bar.addWidget(self.root_label)
            act = QtGui.QAction("Open…", self)
            act.triggered.connect(self.open_workspace)
            bar.addAction(act)
            bar.addSeparator()
            bar.addWidget(QtWidgets.QLabel(" runs use PYTHONPATH "))
            self.pp = QtWidgets.QLineEdit(self.ws.pythonpath)
            self.pp.setMinimumWidth(420)
            self.pp.setToolTip(f"interpreter: {self.ws.python}\nPYTHONPATH of the runs started "
                               f"from here (another git worktree, say); Enter to apply")
            self.pp.editingFinished.connect(self.set_pythonpath)
            bar.addWidget(self.pp)
            bar.addSeparator()
            self.auto = QtWidgets.QCheckBox("refresh every 5 s")
            self.auto.setChecked(True)
            bar.addWidget(self.auto)
            act = QtGui.QAction("Refresh now", self)
            act.triggered.connect(self.refresh)
            bar.addAction(act)

            self.tabs = QtWidgets.QTabWidget()
            self.build, self.runs, self.monitor = BuildTab(self), RunsTab(self), MonitorTab(self)
            self.tabs.addTab(self.runs, "Runs")
            self.tabs.addTab(self.build, "Build")
            self.tabs.addTab(self.monitor, "Monitor")
            self.tabs.currentChanged.connect(lambda _i: self.tick())
            self.setCentralWidget(self.tabs)
            self.statusBar().showMessage(f"interpreter {self.ws.python}")

            menu = self.menuBar().addMenu("&File")
            for text, slot, key in (("Open workspace…", self.open_workspace, "Open"),
                                    ("Refresh", self.refresh, "Refresh"),
                                    (None, None, None),
                                    ("Quit (runs keep going)", self.close, "Quit")):
                if text is None:
                    menu.addSeparator()
                    continue
                act = QtGui.QAction(text, self)
                act.setShortcut(QtGui.QKeySequence(getattr(QtGui.QKeySequence.StandardKey, key)))
                act.triggered.connect(slot)
                menu.addAction(act)

            self.refresh()
            self.build.load("t", next(iter(TEMPLATES)))
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(REFRESH_MS)

        def set_pythonpath(self) -> None:
            """Apply the PYTHONPATH field."""
            if self.pp.text().strip() != self.ws.pythonpath:
                self.ws.pythonpath = self.pp.text().strip()
                self.statusBar().showMessage(f"runs will use PYTHONPATH={self.ws.pythonpath}", 8000)

        def open_workspace(self) -> None:
            """Choose another workspace directory."""
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Workspace", self.ws.root)
            if not d:
                return
            self.ws = Workspace(d, self.ws.python, self.ws.pythonpath)
            self.root_label.setText(self.ws.root)
            self.setWindowTitle(f"regc — {self.ws.root}")
            self.monitor.run_box.clear()
            self.monitor.out_dir = None
            self.refresh()

        def refresh(self) -> None:
            """Rescan the workspace and update the tabs."""
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                ov = self.ws.overview()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            self.runs.update_rows(ov["configs"])
            self.build.refresh_sources([c["path"] for c in ov["configs"]])
            self.monitor.update_runs(ov["configs"])
            if self.tabs.currentWidget() is self.monitor:
                self.monitor.refresh()

        def tick(self) -> None:
            """Timer: refresh what is on screen."""
            if not self.auto.isChecked():
                return
            w = self.tabs.currentWidget()
            if w is self.runs:
                self.runs.update_rows(self.ws.overview()["configs"])
            elif w is self.monitor:
                self.monitor.refresh()


def run(root: str = ".", python: Optional[str] = None, pythonpath: Optional[str] = None,
        argv: Optional[list] = None) -> int:
    """
    Open the window and run the Qt event loop.

    Parameters
    ----------
    root : str, optional
        Workspace directory.
    python, pythonpath : str, optional
        Interpreter and ``PYTHONPATH`` of the runs.
    argv : list of str, optional
        Arguments for QApplication. Default ``sys.argv[:1]``.

    Returns
    -------
    int
        The event loop's exit code.
    """
    if QtWidgets is None:
        raise SystemExit(f"regc gui needs PySide6 ({_QT_ERROR}); pip install PySide6, or use "
                         f"the browser version: regc gui --web")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv or sys.argv[:1])
    app.setApplicationName("regc")
    win = MainWindow(root, python, pythonpath)
    win.show()
    return app.exec()


def main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``regc gui [WORKSPACE]`` / ``regc-gui``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    ap = argparse.ArgumentParser(prog="regc gui",
                                 description="Desktop GUI to build, run and monitor regc simulations.")
    ap.add_argument("workspace", nargs="?", default=".",
                    help="directory searched for configs (default: current directory)")
    ap.add_argument("--python", default=None, help="interpreter of the runs (default: this one)")
    ap.add_argument("--pythonpath", default=None,
                    help="PYTHONPATH of the runs (default: where this regc package lives)")
    ap.add_argument("--web", action="store_true",
                    help="serve the browser version instead (regc.gui; --port, --host)")
    ap.add_argument("--port", type=int, default=8060, help="--web: port (default 8060)")
    ap.add_argument("--host", default="127.0.0.1", help="--web: interface (default 127.0.0.1)")
    args = ap.parse_args(argv)
    if args.web:
        from .gui import serve
        serve(args.workspace, args.host, args.port, args.python, args.pythonpath)
        return 0
    return run(args.workspace, args.python, args.pythonpath)


if __name__ == "__main__":
    raise SystemExit(main())
