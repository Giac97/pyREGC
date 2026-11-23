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


def apply_field(cfg: dict, key: str, kind: str, text: str, synced: Optional[str] = None) -> None:
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
    synced : str, optional
        The entry's text when the form was last filled from the YAML: an
        entry not edited since is left alone, so YAML edits made since
        then are kept. Default None (compare with the config only).
    """
    if text == synced or text == field_text(_get(cfg, key)):
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

    Raises
    ------
    ValueError
        If an end point is not positive or `n` < 1.
    """
    if not (t0 > 0 and t1 > 0) or n < 1:
        raise ValueError("the temperatures must be positive and the number of points at least 1")
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

    Raises
    ------
    ValueError
        If an end point is not positive or `n` < 1.
    """
    if not (p0 > 0 and p1 > 0) or n < 1:
        raise ValueError("the pressures must be positive and the number of points at least 1")
    if n < 2:
        return [p0]
    e0, e1 = np.log10(p0), np.log10(p1)
    return [float(f"{10 ** (e0 + (e1 - e0) * i / (n - 1)):.3g}") for i in range(n)]


# =============================================================================
# Theme: semantic colour tokens (no Qt: also used by the tests)
# =============================================================================
#: Semantic colour tokens. Light: the "Research Lab" palette (navy primary, blue
#: secondary, cool neutrals); dark: lighter, desaturated tonal variants of the
#: same roles. Status colours are darker (light) / lighter (dark) than usual so
#: that as text they reach 4.5:1 on surface, background and selection; they are
#: always paired with an icon and a word, never used alone.
THEMES = {
    "light": {"bg": "#F8FAFC", "surface": "#FFFFFF", "fg": "#0F172A", "muted": "#EEF2F7",
              "muted_fg": "#475569", "border": "#CBD5E1", "grid": "#E2E8F0",
              "primary": "#1E3A5F", "on_primary": "#FFFFFF", "secondary": "#1D4ED8",
              "accent": "#A16207", "danger": "#B91C1C", "on_danger": "#FFFFFF",
              "success": "#15803D", "warning": "#A16207", "info": "#1D4ED8", "neutral": "#475569",
              "focus": "#2563EB", "selection": "#EFF6FF", "chunk": "#BFDBFE",
              "error_bg": "#FEE2E2", "series": ["#1D4ED8", "#15803D", "#A16207", "#B91C1C"]},
    "dark": {"bg": "#0F172A", "surface": "#1E293B", "fg": "#E2E8F0", "muted": "#273449",
             "muted_fg": "#A5B4C8", "border": "#3B4A61", "grid": "#334155",
             "primary": "#93C5FD", "on_primary": "#0F172A", "secondary": "#60A5FA",
             "accent": "#FBBF24", "danger": "#F87171", "on_danger": "#0F172A",
             "success": "#4ADE80", "warning": "#FBBF24", "info": "#93C5FD", "neutral": "#A5B4C8",
             "focus": "#60A5FA", "selection": "#172554", "chunk": "#1E40AF",
             "error_bg": "#4C1D1D", "series": ["#60A5FA", "#4ADE80", "#FBBF24", "#F87171"]},
}

#: Run states: colour token, standard icon, short explanation (tooltip).
STATE_STYLE = {
    "running": ("info", "SP_MediaPlay", "the process is alive and the heartbeat is fresh"),
    "starting": ("info", "SP_BrowserReload", "the process is alive; no heartbeat yet (setup)"),
    "stalled": ("warning", "SP_MessageBoxWarning", "the process is alive but the heartbeat is old"),
    "finished": ("success", "SP_DialogApplyButton", "the run finished and was analysed"),
    "converged": ("success", "SP_DialogApplyButton", "the convergence check passed; analysed"),
    "stopped": ("warning", "SP_MediaPause", "not running; a checkpoint to resume from"),
    "error": ("danger", "SP_MessageBoxCritical", "the run ended with an error (see the log)"),
    "new": ("neutral", "SP_FileIcon", "not run yet"),
    "ended": ("neutral", "SP_FileIcon", "not running; no checkpoint and no result"),
}


def _luminance(colour: str) -> float:
    """
    WCAG relative luminance of a ``#RRGGBB`` colour.

    Parameters
    ----------
    colour : str
        The colour.

    Returns
    -------
    float
        Luminance in [0, 1].
    """
    c = colour.lstrip("#")
    rgb = [int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    """
    WCAG contrast ratio of two colours.

    Parameters
    ----------
    a, b : str
        ``#RRGGBB`` colours.

    Returns
    -------
    float
        Between 1 and 21.
    """
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def convergence_text(conv: Optional[dict], settings: Optional[dict] = None) -> str:
    """
    The latest convergence check in words (pass or fail readable without colour).

    Parameters
    ----------
    conv : dict or None
        A check (``max_ratio``, ``median_ratio``, ``min_neff``, ``converged``)
        or a block check (``max_change``).
    settings : dict, optional
        ``tol``, ``median_tol``, ``min_neff``. Default: 3, 1.3, 10.

    Returns
    -------
    str
        E.g. ``max 8.13 > 3 · median 1.92 > 1.3 · n_eff 9.3 < 10``.
    """
    if not conv:
        return ""
    if conv.get("max_ratio") is None:
        return f"Δρ {conv['max_change']:.3f}" if conv.get("max_change") is not None else ""
    s = settings or {}
    tol, mtol, need = s.get("tol", 3.0), s.get("median_tol", 1.3), s.get("min_neff", 10.0)
    rel = lambda ok: "<" if ok else ">"          # noqa: E731
    return (f"max {conv['max_ratio']:.2f} {rel(conv['max_ratio'] < tol)} {tol:g} · "
            f"median {conv['median_ratio']:.2f} {rel(conv['median_ratio'] < mtol)} {mtol:g} · "
            f"n_eff {conv['min_neff']:.1f} {'≥' if conv['min_neff'] >= need else '<'} {need:g}")


def explain_error(error: str) -> str:
    """
    A config error in words: the cause and how to fix it.

    Unknown keys get the closest allowed key as a suggestion instead of the
    whole list of allowed keys.

    Parameters
    ----------
    error : str
        The validation error (``Type: message``).

    Returns
    -------
    str
        The explanation.
    """
    import difflib
    import re
    m = re.search(r"unknown key\(s\) \[([^\]]*)\] in config section '([^']*)'.*?allowed: \[([^\]]*)\]",
                  error)
    if not m:
        return error.split(": ", 1)[-1] if error.startswith(("ValueError", "KeyError", "TypeError")) \
            else error
    unknown = [k.strip().strip("'\"") for k in m.group(1).split(",")]
    allowed = [k.strip().strip("'\"") for k in m.group(3).split(",")]
    parts = []
    for k in unknown:
        close = difflib.get_close_matches(k, allowed, n=1, cutoff=0.6)
        parts.append(f"Unknown key '{k}' in {m.group(2)}."
                     + (f" Did you mean '{close[0]}'?" if close else " Remove it, or check the manual "
                        "for the section's keys (hover here for the list)."))
    return " ".join(parts)


# =============================================================================
# Qt widgets
# =============================================================================
if QtWidgets is not None:
    import matplotlib
    matplotlib.use("QtAgg")
    import matplotlib.patches
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    SP = QtWidgets.QStyle.StandardPixmap

    def _icon(name: str) -> "QtGui.QIcon":
        """A standard icon of the current style by ``SP_*`` name."""
        return QtWidgets.QApplication.style().standardIcon(getattr(SP, name))

    def _mono() -> "QtGui.QFont":
        """The system fixed-width font."""
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        f.setPointSize(max(10, f.pointSize()))
        return f

    def _tabular(font: "QtGui.QFont") -> "QtGui.QFont":
        """The font with tabular (equal-width) figures, where the font supports it."""
        f = QtGui.QFont(font)
        try:
            f.setFeature(QtGui.QFont.Tag("tnum"), 1)
        except (AttributeError, TypeError):
            pass
        return f

    def _button(text: str, slot, tip: str = "", icon: Optional[str] = None,
                role: Optional[str] = None, shortcut: Optional[str] = None) -> "QtWidgets.QPushButton":
        """
        A push button.

        Parameters
        ----------
        text : str
            Label (a visible label always; icons are decoration beside it).
        slot : callable
            Clicked handler.
        tip : str, optional
            Tooltip (the shortcut is appended).
        icon : str, optional
            ``SP_*`` standard icon.
        role : str, optional
            ``primary`` or ``danger`` (styled by the theme).
        shortcut : str, optional
            Key sequence.
        """
        b = QtWidgets.QPushButton(text)
        b.clicked.connect(slot)
        if icon:
            b.setIcon(_icon(icon))
        if role:
            b.setProperty("role", role)
        if shortcut:
            b.setShortcut(QtGui.QKeySequence(shortcut))
            tip = f"{tip} ({QtGui.QKeySequence(shortcut).toString(QtGui.QKeySequence.SequenceFormat.NativeText)})"
        if tip:
            b.setToolTip(tip.strip())
        b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        return b

    def _process_env(ws: Workspace) -> "QtCore.QProcessEnvironment":
        """The environment of processes started from the window."""
        env = QtCore.QProcessEnvironment()
        for k, v in ws._env().items():      # the runs' environment (PYTHONPATH, threads)
            env.insert(k, v)
        return env

    def _repolish(w: "QtWidgets.QWidget") -> None:
        """Re-apply the style sheet after a dynamic property changed."""
        w.style().unpolish(w)
        w.style().polish(w)

    def apply_theme(app: "QtWidgets.QApplication", t: dict) -> None:
        """
        Fusion style, a palette and a small style sheet from the theme tokens.

        Parameters
        ----------
        app : QApplication
            The application.
        t : dict
            Tokens (`THEMES` entry).
        """
        app.setStyle("Fusion")
        C = QtGui.QColor
        pal = QtGui.QPalette()
        R = QtGui.QPalette.ColorRole
        for role, tok in ((R.Window, "bg"), (R.Base, "surface"), (R.AlternateBase, "bg"),
                          (R.Text, "fg"), (R.WindowText, "fg"), (R.Button, "surface"),
                          (R.ButtonText, "fg"), (R.Highlight, "selection"), (R.HighlightedText, "fg"),
                          (R.ToolTipBase, "surface"), (R.ToolTipText, "fg"),
                          (R.PlaceholderText, "muted_fg"), (R.Link, "secondary"),
                          (R.Mid, "border"), (R.Midlight, "muted"), (R.Light, "surface"),
                          (R.Dark, "border")):
            pal.setColor(role, C(t[tok]))
        G = QtGui.QPalette.ColorGroup
        for role in (R.Text, R.WindowText, R.ButtonText):
            pal.setColor(G.Disabled, role, C(t["muted_fg"]))
        pal.setColor(G.Disabled, R.Button, C(t["muted"]))
        app.setPalette(pal)
        app.setStyleSheet(f"""
QToolTip {{ color: {t['fg']}; background: {t['surface']}; border: 1px solid {t['border']}; padding: 4px; }}
QPushButton {{ padding: 5px 12px; border: 1px solid {t['border']}; border-radius: 6px;
  background: {t['surface']}; min-height: 22px; }}
QPushButton:hover {{ border-color: {t['secondary']}; }}
QPushButton:focus {{ border: 2px solid {t['focus']}; padding: 4px 11px; }}
QPushButton:disabled {{ color: {t['muted_fg']}; background: {t['muted']}; border-color: {t['muted']}; }}
QPushButton[role="primary"] {{ background: {t['primary']}; color: {t['on_primary']}; border-color: {t['primary']};
  font-weight: 600; }}
QPushButton[role="primary"]:disabled {{ color: {t['muted_fg']}; background: {t['muted']}; border-color: {t['muted']}; }}
QPushButton[role="danger"] {{ color: {t['danger']}; border-color: {t['danger']}; }}
QPushButton[role="danger"]:disabled {{ color: {t['muted_fg']}; border-color: {t['muted']}; }}
QLineEdit, QPlainTextEdit {{ border: 1px solid {t['border']}; border-radius: 5px; padding: 3px 6px;
  background: {t['surface']}; selection-background-color: {t['chunk']}; selection-color: {t['fg']}; }}
QLineEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {t['focus']}; padding: 2px 5px; }}
QLineEdit[error="true"] {{ border: 2px solid {t['danger']}; padding: 2px 5px; }}
QGroupBox {{ font-weight: 600; border: 1px solid {t['border']}; border-radius: 8px; margin-top: 14px;
  padding: 10px 8px 6px 8px; background: {t['surface']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {t['fg']}; }}
QHeaderView::section {{ background: {t['muted']}; color: {t['fg']}; font-weight: 600; padding: 5px 6px;
  border: none; border-bottom: 1px solid {t['border']}; border-right: 1px solid {t['border']}; }}
QTableWidget {{ gridline-color: {t['grid']}; background: {t['surface']}; alternate-background-color: {t['bg']};
  border: 1px solid {t['border']}; border-radius: 6px; }}
QTableWidget::item:selected {{ background: {t['selection']}; }}
QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; background: {t['muted']};
  text-align: center; color: {t['fg']}; min-height: 16px; }}
QProgressBar::chunk {{ background: {t['chunk']}; border-radius: 3px; }}
QFrame#tile {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; }}
QLabel#tileLabel, QLabel#muted {{ color: {t['muted_fg']}; }}
QFrame#message {{ border-radius: 6px; }}
QFrame#message[kind="ok"] {{ background: {t['selection']}; border: 1px solid {t['success']}; }}
QFrame#message[kind="error"] {{ background: {t['error_bg']}; border: 1px solid {t['danger']}; }}
QFrame#message[kind="info"] {{ background: {t['muted']}; border: 1px solid {t['border']}; }}
QTabBar::tab {{ padding: 7px 16px; }}
""")

    def style_axes(ax, t: dict) -> None:
        """Matplotlib axes in the theme's colours, with subtle grid lines."""
        ax.set_facecolor(t["surface"])
        for sp in ax.spines.values():
            sp.set_color(t["border"])
        ax.tick_params(colors=t["muted_fg"], labelcolor=t["fg"], labelsize=8)
        ax.xaxis.label.set_color(t["fg"])
        ax.yaxis.label.set_color(t["fg"])
        ax.title.set_color(t["fg"])

    class Message(QtWidgets.QFrame):
        """An inline message: icon + text on a tinted background (not colour alone)."""

        def __init__(self):
            """Build the (hidden) message."""
            super().__init__()
            self.setObjectName("message")
            lay = QtWidgets.QHBoxLayout(self)
            lay.setContentsMargins(8, 4, 8, 4)
            self.icon = QtWidgets.QLabel()
            self.text = QtWidgets.QLabel()
            self.text.setWordWrap(True)
            self.text.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            lay.addWidget(self.icon, 0, QtCore.Qt.AlignmentFlag.AlignTop)
            lay.addWidget(self.text, 1)
            self.hide()

        def show_message(self, text: str, kind: str = "ok") -> None:
            """
            Show a message.

            Parameters
            ----------
            text : str
                The message (empty hides it).
            kind : str, optional
                ``ok``, ``error`` or ``info``. Default ``ok``.
            """
            if not text:
                return self.hide()
            icon = {"ok": "SP_DialogApplyButton", "error": "SP_MessageBoxCritical",
                    "info": "SP_MessageBoxInformation"}[kind]
            self.icon.setPixmap(_icon(icon).pixmap(18, 18))
            self.text.setText(text)
            self.setProperty("kind", kind)
            _repolish(self)
            self.show()

    class StatTile(QtWidgets.QFrame):
        """A small stat tile: caption, value (tabular figures), detail line."""

        def __init__(self, caption: str):
            """
            Build the tile.

            Parameters
            ----------
            caption : str
                What the value is.
            """
            super().__init__()
            self.setObjectName("tile")
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(10, 6, 10, 6)
            lay.setSpacing(1)
            head = QtWidgets.QHBoxLayout()
            self.icon = QtWidgets.QLabel()
            self.label = QtWidgets.QLabel(caption)
            self.label.setObjectName("tileLabel")
            head.addWidget(self.icon)
            head.addWidget(self.label, 1)
            self.value = QtWidgets.QLabel("–")
            f = _tabular(self.value.font())
            f.setPointSizeF(f.pointSizeF() * 1.35)
            f.setWeight(QtGui.QFont.Weight.DemiBold)
            self.value.setFont(f)
            self.detail = QtWidgets.QLabel("")
            self.detail.setObjectName("tileLabel")
            self.detail.setFont(_tabular(self.detail.font()))
            self.detail.setWordWrap(True)
            lay.addLayout(head)
            lay.addWidget(self.value)
            lay.addWidget(self.detail)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                               QtWidgets.QSizePolicy.Policy.Maximum)

        def set(self, value: str, detail: str = "", colour: Optional[str] = None,
                icon: Optional[str] = None, tip: str = "") -> None:
            """
            Update the tile.

            Parameters
            ----------
            value : str
                Main value.
            detail : str, optional
                Second line.
            colour : str, optional
                Value colour (always with text, and an icon where given).
            icon : str, optional
                ``SP_*`` icon beside the caption.
            tip : str, optional
                Tooltip.
            """
            self.value.setText(value)
            self.value.setStyleSheet(f"color: {colour}" if colour else "")
            self.detail.setText(detail)
            self.detail.setVisible(bool(detail))
            self.icon.setPixmap(_icon(icon).pixmap(16, 16) if icon else QtGui.QPixmap())
            self.icon.setVisible(bool(icon))
            self.setToolTip(tip)

    class YamlHighlighter(QtGui.QSyntaxHighlighter):
        """Comments, keys, numbers and literals of YAML, in the theme's colours."""

        def __init__(self, doc: "QtGui.QTextDocument", t: dict):
            """
            Set up the rules.

            Parameters
            ----------
            doc : QTextDocument
                The editor's document.
            t : dict
                Theme tokens.
            """
            super().__init__(doc)
            self.set_theme(t)

        def set_theme(self, t: dict) -> None:
            """Colours from the theme tokens."""
            def fmt(colour, bold=False, italic=False):
                f = QtGui.QTextCharFormat()
                f.setForeground(QtGui.QColor(colour))
                if bold:
                    f.setFontWeight(QtGui.QFont.Weight.DemiBold)
                f.setFontItalic(italic)
                return f
            R = QtCore.QRegularExpression
            self.rules = [
                (R(r"^\s*-?\s*[\w.\-]+(?=\s*:)"), fmt(t["primary"], True)),
                (R(r"(?<![\w.])[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?(?![\w.])"), fmt(t["success"])),
                (R(r"\b(true|false|null|auto|self)\b"), fmt(t["accent"])),
                (R(r"(\"[^\"]*\"|'[^']*')"), fmt(t["accent"])),
            ]
            self.comment = fmt(t["muted_fg"], italic=True)
            self.rehighlight()

        def highlightBlock(self, text: str) -> None:  # noqa: N802 - Qt API
            """Highlight one line."""
            code, hash_at = text, -1
            in_q = None
            for i, ch in enumerate(text):           # '#' outside quotes starts a comment
                if ch in "\"'" and in_q in (None, ch):
                    in_q = None if in_q else ch
                elif ch == "#" and in_q is None and (i == 0 or text[i - 1].isspace()):
                    hash_at = i
                    break
            if hash_at >= 0:
                code = text[:hash_at]
                self.setFormat(hash_at, len(text) - hash_at, self.comment)
            for rx, f in self.rules:
                it = rx.globalMatch(code)
                while it.hasNext():
                    m = it.next()
                    self.setFormat(m.capturedStart(), m.capturedLength(), f)

    # -------------------------------------------------------------- Build tab
    class BuildTab(QtWidgets.QWidget):
        """YAML editor, grouped form, ladder helpers; validate / save / check."""

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
            self.dirty = False
            self._loading = False
            self._synced: Dict[str, str] = {}      # field texts at the last YAML -> form fill
            self._synced_groups: Dict[str, bool] = {}

            # row 1: source and destination
            top = QtWidgets.QHBoxLayout()
            top.addWidget(QtWidgets.QLabel("Start from"))
            self.source = QtWidgets.QComboBox()
            self.source.setMinimumWidth(300)
            self.source.setToolTip("a built-in template or a config of the workspace")
            top.addWidget(self.source, 2)
            top.addWidget(_button("Load", self.load_selected, "load it into the editor",
                                  "SP_DialogOpenButton"))
            top.addSpacing(18)
            lab = QtWidgets.QLabel("Save as")
            top.addWidget(lab)
            self.save_path = QtWidgets.QLineEdit()
            self.save_path.setPlaceholderText("workspace-relative, e.g. my_run/my_run.yaml")
            lab.setBuddy(self.save_path)
            top.addWidget(self.save_path, 3)
            top.addWidget(_button("Browse…", self.browse_save, "choose the file",
                                  "SP_DirOpenIcon"))
            self.overwrite = QtWidgets.QCheckBox("Overwrite")
            self.overwrite.setToolTip("replace the file if it exists")
            top.addWidget(self.overwrite)

            # row 2: actions
            acts = QtWidgets.QHBoxLayout()
            acts.addWidget(_button("Validate", self.validate, "check sections and keys",
                                   "SP_DialogApplyButton", shortcut="Ctrl+Return"))
            self.check_btn = _button("Check setup", self.check,
                                     "regc-run --check: build the system with the calculator, "
                                     "print the μ ladder and the bare-slab energy",
                                     "SP_FileDialogContentsView")
            acts.addWidget(self.check_btn)
            acts.addSpacing(18)
            acts.addWidget(_button("Save", self.save, "write the YAML to “Save as”",
                                   "SP_DialogSaveButton", shortcut="Ctrl+S"))
            acts.addWidget(_button("Save and start", self.save_and_start,
                                   "save, then start the run (Runs tab)", "SP_MediaPlay",
                                   role="primary"))
            self.dirty_label = QtWidgets.QLabel("")
            self.dirty_label.setObjectName("muted")
            acts.addWidget(self.dirty_label)
            acts.addStretch(1)
            self.msg = Message()

            # left: YAML editor
            self.yaml = QtWidgets.QPlainTextEdit()
            self.yaml.setFont(_mono())
            self.yaml.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.yaml.setTabStopDistance(2 * self.yaml.fontMetrics().horizontalAdvance(" "))
            self.yaml.textChanged.connect(self._changed)
            self.highlighter = YamlHighlighter(self.yaml.document(), main.theme)
            left = QtWidgets.QWidget()
            ll = QtWidgets.QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            head = QtWidgets.QHBoxLayout()
            t = QtWidgets.QLabel("<b>Config (YAML)</b>")
            head.addWidget(t)
            head.addStretch(1)
            head.addWidget(_button("Fill form from YAML", self.yaml_to_form,
                                   "update the form from the text", "SP_ArrowRight"))
            ll.addLayout(head)
            ll.addWidget(self.yaml)

            # right: form in groups
            form_w = QtWidgets.QWidget()
            fl = QtWidgets.QVBoxLayout(form_w)
            fl.setContentsMargins(0, 0, 6, 0)
            group = form = None
            self.groups: Dict[str, QtWidgets.QGroupBox] = {}
            for f in FORM_FIELDS:
                if len(f) == 1:
                    group = QtWidgets.QGroupBox(f[0])
                    form = QtWidgets.QFormLayout(group)
                    form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
                    if f[0].startswith("Subsurface"):
                        group.setCheckable(True)
                        group.setToolTip("off: no subsurface section (no NCMC moves)")
                        group.toggled.connect(lambda on, g=group: self._show_group(g, on))
                    self.groups[f[0]] = group
                    fl.addWidget(group)
                    continue
                key, label, kind = f[:3]
                if kind.startswith("sel:") or kind == "bool":
                    w = QtWidgets.QComboBox()
                    w.addItems(kind[4:].split("|") if kind.startswith("sel:") else ["", "true", "false"])
                else:
                    w = QtWidgets.QLineEdit()
                    if len(f) > 3:
                        w.setPlaceholderText(f[3])
                w.setToolTip(f"{key}" + (f" — {f[3]}" if len(f) > 3 else ""))
                lab = QtWidgets.QLabel(label)
                lab.setBuddy(w)
                self.fields[key] = (w, kind)
                form.addRow(lab, w)
            lad_box = QtWidgets.QGroupBox("Ladder helpers")
            lad = QtWidgets.QGridLayout(lad_box)
            self.t0, self.t1, self.tn = (QtWidgets.QLineEdit(v) for v in ("300", "900", "8"))
            self.p0, self.p1, self.pn = (QtWidgets.QLineEdit(v) for v in ("1e-14", "1", "8"))
            for row, (a, b, n, lab, slot, what) in enumerate((
                    (self.t0, self.t1, self.tn, "T (K)", self.t_ladder, "geometric"),
                    (self.p0, self.p1, self.pn, "p", self.p_ladder, "log-spaced"))):
                lad.addWidget(QtWidgets.QLabel(lab), row, 0)
                for col, (w, tip) in enumerate(((a, "from"), (b, "to"), (n, "number of points"))):
                    w.setMaximumWidth(80)
                    w.setToolTip(tip)
                    lad.addWidget(w, row, 1 + col)
                lad.addWidget(_button(f"Fill {what}", slot, f"fill the {lab} field"), row, 4)
            fl.addWidget(lad_box)
            fl.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(form_w)
            right = QtWidgets.QWidget()
            rl = QtWidgets.QVBoxLayout(right)
            rl.setContentsMargins(0, 0, 0, 0)
            rhead = QtWidgets.QHBoxLayout()
            rhead.addWidget(_button("Write form to YAML", self.form_to_yaml,
                                    "rewrite the YAML from the form (its comments are lost)",
                                    "SP_ArrowLeft"))
            rhead.addStretch(1)
            hint = QtWidgets.QLabel("Common keys; an empty field removes the key.")
            hint.setObjectName("muted")
            rhead.addWidget(hint)
            rl.addLayout(rhead)
            rl.addWidget(scroll)

            # bottom: check output
            out_w = QtWidgets.QWidget()
            ol = QtWidgets.QVBoxLayout(out_w)
            ol.setContentsMargins(0, 0, 0, 0)
            oh = QtWidgets.QHBoxLayout()
            oh.addWidget(QtWidgets.QLabel("<b>Check setup output</b>"))
            self.busy = QtWidgets.QProgressBar()
            self.busy.setRange(0, 0)
            self.busy.setMaximumWidth(160)
            self.busy.setTextVisible(False)
            self.busy.hide()
            oh.addWidget(self.busy)
            self.cancel_btn = _button("Cancel", self.cancel_check, "stop the check",
                                      "SP_DialogCancelButton")
            self.cancel_btn.hide()
            oh.addWidget(self.cancel_btn)
            oh.addStretch(1)
            ol.addLayout(oh)
            self.out = QtWidgets.QPlainTextEdit()
            self.out.setReadOnly(True)
            self.out.setFont(_mono())
            self.out.setPlaceholderText("Save the config, then “Check setup” builds the system with "
                                        "the calculator and prints the μ ladder here.")
            ol.addWidget(self.out)

            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            split.addWidget(left)
            split.addWidget(right)
            split.setSizes([720, 520])
            vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            vsplit.addWidget(split)
            vsplit.addWidget(out_w)
            vsplit.setSizes([720, 170])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(top)
            lay.addLayout(acts)
            lay.addWidget(self.msg)
            lay.addWidget(vsplit, 1)

        # ---- state
        def _show_group(self, group: "QtWidgets.QGroupBox", on: bool) -> None:
            """Show or hide a checkable group's fields."""
            for w in group.findChildren(QtWidgets.QWidget, options=QtCore.Qt.FindChildOption.FindDirectChildrenOnly):
                w.setVisible(on)

        def _changed(self) -> None:
            """The text changed: mark unsaved."""
            if not self._loading:
                self.set_dirty(True)

        def set_dirty(self, dirty: bool) -> None:
            """
            Mark the editor as (un)saved.

            Parameters
            ----------
            dirty : bool
                Unsaved changes.
            """
            self.dirty = dirty
            self.dirty_label.setText("● unsaved changes" if dirty else "")
            self.main.update_title()

        def confirm_discard(self) -> bool:
            """
            Ask before unsaved changes are replaced.

            Returns
            -------
            bool
                True to go on (saved or discarded), False to cancel.
            """
            if not self.dirty:
                return True
            B = QtWidgets.QMessageBox.StandardButton
            ans = QtWidgets.QMessageBox.question(
                self, "Unsaved changes", "The config in the editor has unsaved changes.",
                B.Save | B.Discard | B.Cancel, B.Save)
            if ans == B.Save:
                return self.save() is not None
            return ans == B.Discard

        # ---- helpers
        def say(self, text: str, ok: bool = True) -> None:
            """
            Show a message under the toolbar.

            Parameters
            ----------
            text : str
                The message.
            ok : bool, optional
                Success (True) or error. Default True.
            """
            self.msg.show_message(text, "ok" if ok else "error")

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
                self.source.addItem(_icon("SP_FileDialogNewFolder"), f"Template: {t}", f"t:{t}")
            if configs:
                self.source.insertSeparator(self.source.count())
            for c in configs:
                self.source.addItem(_icon("SP_FileIcon"), c, f"c:{c}")
            if cur is not None:
                i = self.source.findData(cur)
                if i >= 0:
                    self.source.setCurrentIndex(i)
            self.source.blockSignals(False)

        def load_selected(self) -> None:
            """Load the chosen template or config."""
            data = self.source.currentData()
            if data:
                self.load(data[0], data[2:])

        def load(self, kind: str, name: str, ask: bool = True) -> bool:
            """
            Load a template (``t``) or a workspace config (``c``) into the editor and form.

            Parameters
            ----------
            kind : str
                ``t`` or ``c``.
            name : str
                Template name or workspace-relative path.
            ask : bool, optional
                Ask before discarding unsaved changes. Default True.

            Returns
            -------
            bool
                True if loaded.
            """
            if ask and not self.confirm_discard():
                return False
            try:
                if kind == "t":
                    text, path = TEMPLATES[name], ""
                else:
                    with open(self.main.ws.resolve(name)) as f:
                        text, path = f.read(), name
            except (OSError, ValueError) as err:
                self.say(f"Could not open {name}: {err}", False)
                return False
            self._loading = True
            self.yaml.setPlainText(text)
            self._loading = False
            self.save_path.setText(path)
            self.overwrite.setChecked(False)
            i = self.source.findData(f"{kind}:{name}")
            if i >= 0:
                self.source.setCurrentIndex(i)
            self.set_dirty(False)
            self.yaml_to_form()
            return True

        def _clear_errors(self) -> None:
            """Remove field and line error marks."""
            for w, _k in self.fields.values():
                if w.property("error"):
                    w.setProperty("error", False)
                    w.setToolTip(w.toolTip().split("\nError: ")[0])
                    _repolish(w)
            self.yaml.setExtraSelections([])

        def _mark_line(self, line: int) -> None:
            """Highlight a YAML line (0-based) and put the cursor there."""
            block = self.yaml.document().findBlockByNumber(max(0, line))
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format.setBackground(QtGui.QColor(self.main.theme["error_bg"]))
            sel.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            cur = QtGui.QTextCursor(block)
            sel.cursor = cur
            self.yaml.setExtraSelections([sel])
            self.yaml.setTextCursor(cur)
            self.yaml.centerCursor()

        def _locate_error(self, error: str) -> str:
            """
            Mark the field or YAML line an error message names.

            Parameters
            ----------
            error : str
                The validation error.

            Returns
            -------
            str
                Where it was marked (for the message), or empty.
            """
            import re
            hits = [k for k in self.fields if k in error]
            for k in hits:
                w = self.fields[k][0]
                w.setProperty("error", True)
                w.setToolTip(w.toolTip() + "\nError: " + error)
                _repolish(w)
            m = re.search(r"unknown key\(s\) \[([^\]]*)\]", error)
            if m:
                key = m.group(1).split(",")[0].strip().strip("'\"")
                for i, line in enumerate(self.yaml.toPlainText().splitlines()):
                    if re.match(rf"^\s*{re.escape(key)}\s*:", line):
                        self._mark_line(i)
                        return f"line {i + 1} of the YAML"
            if hits:
                return "field " + ", ".join(self.fields[k][0].toolTip().split(" —")[0].split("\n")[0]
                                            for k in hits)
            return ""

        def parse(self) -> Optional[dict]:
            """
            Parse and validate the editor's text; report next to the problem.

            Returns
            -------
            dict or None
                The config, or None if it does not parse.
            """
            self._clear_errors()
            d = parse_config(self.yaml.toPlainText())
            s = d["summary"]
            summary = (f"{s['replicas']} replicas ({s['n_T']} T × {s['n_mu']} μ/p), "
                       f"n_steps {s['n_steps']:,}, about {s['samples_per_replica']:,} samples per replica"
                       if s else "")
            if d["config"] is None:
                import re
                m = re.search(r"line (\d+)", d["error"] or "")
                if m:
                    self._mark_line(int(m.group(1)) - 1)
                self.say(f"{d['error']}\nFix the highlighted YAML line and validate again.", False)
                return None
            if d["ok"]:
                self.say(f"Valid. {summary}")
                self.msg.setToolTip("")
            else:
                where = self._locate_error(d["error"])
                self.say(explain_error(d["error"]) + (f"\nMarked: {where}." if where else ""), False)
                self.msg.setToolTip(d["error"])
            return d["config"]

        def validate(self) -> None:
            """Validate the YAML."""
            self.parse()

        def yaml_to_form(self) -> None:
            """Fill the form from the YAML."""
            cfg = self.parse()
            if cfg is not None:
                self._fill_form(cfg)

        def _fill_form(self, cfg: dict) -> None:
            """
            Fill the form from a config and remember what it shows.

            Parameters
            ----------
            cfg : dict
                The config.
            """
            for key, (w, kind) in self.fields.items():
                text = field_text(_get(cfg, key))
                if isinstance(w, QtWidgets.QComboBox):
                    if w.findText(text) < 0:
                        w.addItem(text)
                    w.setCurrentText(text)
                else:
                    w.setText(text)
                    w.setCursorPosition(0)
                self._synced[key] = text
            for title, g in self.groups.items():
                if g.isCheckable():
                    g.setChecked(isinstance(cfg.get("subsurface"), dict))
                    self._synced_groups[title] = g.isChecked()

        def form_to_yaml(self) -> None:
            """
            Rewrite the YAML with the form's values.

            Only the entries edited since the form was last filled from the
            YAML are written, so edits typed into the YAML since then are
            kept. Nothing happens if the YAML does not parse. The editor's
            undo history covers the rewrite.
            """
            cfg = self.parse()
            if cfg is None:
                self.say("“Write form to YAML” not applied: the YAML does not parse (see the "
                         "highlighted line). Fix it, or undo, and try again.\n"
                         + self.msg.text.text(), False)
                return
            for key, (w, kind) in self.fields.items():
                g = w.parentWidget()
                if isinstance(g, QtWidgets.QGroupBox) and g.isCheckable() and not g.isChecked():
                    if self._synced_groups.get(g.title(), True):     # switched off in the form
                        _set(cfg, key.split(".")[0], None)
                    continue
                text = w.currentText() if isinstance(w, QtWidgets.QComboBox) else w.text()
                apply_field(cfg, key, kind, text, self._synced.get(key, ""))
            cur = QtGui.QTextCursor(self.yaml.document())
            cur.beginEditBlock()                    # one undo step, history kept
            cur.select(QtGui.QTextCursor.SelectionType.Document)
            cur.insertText(dump_config(cfg))
            cur.endEditBlock()
            cfg = self.parse()
            if cfg is not None:
                self._fill_form(cfg)

        def t_ladder(self) -> None:
            """Fill T with a geometric ladder."""
            try:
                T = geometric_ladder(float(self.t0.text()), float(self.t1.text()), int(self.tn.text()))
            except ValueError as err:
                return self.say(f"Ladder: {err}. Give numbers: from, to and a whole number of points.",
                                False)
            self.fields["replica_exchange.temperatures_K"][0].setText(", ".join(map(str, T)))
            self.say("T ladder filled in the form; “Write form to YAML” applies it.")

        def p_ladder(self) -> None:
            """Fill the pressures with a log-spaced ladder."""
            try:
                p = log_ladder(float(self.p0.text()), float(self.p1.text()), int(self.pn.text()))
            except ValueError as err:
                return self.say(f"Ladder: {err}. Give numbers: from, to and a whole number of points.",
                                False)
            self.fields["replica_exchange.pressures"][0].setText(", ".join(f"{x:g}" for x in p))
            self.say("Pressure ladder filled in the form; “Write form to YAML” applies it.")

        def browse_save(self) -> None:
            """Choose where to save."""
            try:
                start = self.main.ws.resolve(self.save_path.text() or ".")
            except ValueError:
                start = self.main.ws.root
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
                    self.save_path.setFocus()
                    raise ValueError("Give a path in “Save as” (relative to the workspace).")
                p = self.main.ws.resolve(rel)
                if not p.endswith((".yaml", ".yml")):
                    raise ValueError(f"{rel}: config files end in .yaml or .yml.")
                if os.path.exists(p) and not self.overwrite.isChecked():
                    raise ValueError(f"{rel} exists. Tick “Overwrite” to replace it, or change "
                                     f"“Save as”.")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(self.yaml.toPlainText())
            except (OSError, ValueError) as err:
                self.say(str(err), False)
                return None
            self.set_dirty(False)
            self.say(f"Saved {self.main.ws.rel(p)}.")
            self.main.notify(f"Saved {self.main.ws.rel(p)}")
            self.main.refresh()
            return self.main.ws.rel(p)

        def save_and_start(self) -> None:
            """Save, then start the run and show the Runs tab."""
            if self.parse() is None:
                return
            rel = self.save()
            if rel and self.main.runs.start(rel, "run"):
                self.main.tabs.setCurrentWidget(self.main.runs)
                self.main.runs.select(rel)

        def _check_running(self) -> bool:
            """Whether a check process is running."""
            return (self.proc is not None
                    and self.proc.state() != QtCore.QProcess.ProcessState.NotRunning)

        def check(self) -> None:
            """Run ``regc-run --check`` on the saved config, streaming its output."""
            rel = self.save_path.text().strip()
            try:
                if not rel or not os.path.isfile(self.main.ws.resolve(rel)):
                    raise ValueError("Save the config first: the check runs on the saved file.")
            except ValueError as err:
                return self.say(str(err), False)
            if self.dirty:
                return self.say("The editor has unsaved changes: save them first, so the check "
                                "sees what you see.", False)
            if self._check_running():
                return
            path = self.main.ws.resolve(rel)
            self.out.setPlainText(f"$ regc-run {os.path.basename(path)} --check\n")
            self.proc = QtCore.QProcess(self)
            self.proc.setProcessEnvironment(_process_env(self.main.ws))
            self.proc.setWorkingDirectory(os.path.dirname(path))
            self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.proc.readyReadStandardOutput.connect(
                lambda: self._append(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")))
            self.proc.finished.connect(self._check_done)
            self.busy.show()
            self.cancel_btn.show()
            self.check_btn.setEnabled(False)
            self.msg.show_message(f"Checking {rel}: building the system with the calculator …", "info")
            self.proc.start(self.main.ws.python, self.main.ws._cmd(path, "--check")[1:])

        def _check_done(self, code: int, _status) -> None:
            """The check ended."""
            self.busy.hide()
            self.cancel_btn.hide()
            self.check_btn.setEnabled(True)
            self._append(f"\n[exit {code}]\n")
            if code == 0:
                self.say("Check passed: the system builds and the calculator works (output below).")
            else:
                self.say(f"Check failed (exit {code}). The error is at the end of the output below.",
                         False)

        def cancel_check(self) -> None:
            """Stop the running check."""
            if self._check_running():
                self.proc.kill()

        def _append(self, text: str) -> None:
            """Append to the check output."""
            self.out.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.out.insertPlainText(text)
            self.out.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    # --------------------------------------------------------------- Runs tab
    class RunsTab(QtWidgets.QWidget):
        """Configs, their runs' states, and the run actions."""

        COLUMNS = ["Config", "State", "Progress", "Convergence", "Subsurface", "Output", "Last log line"]

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
            self.by_path: Dict[str, dict] = {}
            bar = QtWidgets.QHBoxLayout()
            self.buttons = {}
            spec = [("edit", "Edit", "open the config on the Build tab", "SP_FileDialogDetailedView",
                     None, "Ctrl+E"),
                    None,
                    ("run", "Start", "start a new run", "SP_MediaPlay", "primary", None),
                    ("resume", "Resume", "continue from the last checkpoint (--resume)",
                     "SP_MediaSeekForward", None, None),
                    ("analyze", "Analyze", "re-run the MBAR analysis and plots (--analyze-only)",
                     "SP_FileDialogContentsView", None, None),
                    ("monitor", "Monitor", "show the run's plots (also: double-click, Enter)",
                     "SP_ComputerIcon", None, None),
                    "stretch",
                    ("overwrite", "Start over", "discard checkpoint, samples and records and start "
                     "again (--overwrite); asks first", "SP_BrowserReload", "danger", None),
                    ("stop", "Stop", "end the run (SIGTERM); it can be resumed from its last "
                     "checkpoint; asks first", "SP_MediaStop", "danger", None)]
            for s in spec:
                if s is None:
                    bar.addSpacing(14)
                    continue
                if s == "stretch":
                    bar.addStretch(1)
                    continue
                mode, label, tip, icon, role, key = s
                b = _button(label, lambda _=False, m=mode: self.action(m), tip, icon, role, key)
                self.buttons[mode] = b
                bar.addWidget(b)

            filt = QtWidgets.QHBoxLayout()
            self.filter = QtWidgets.QLineEdit()
            self.filter.setPlaceholderText("Filter configs (path or state)")
            self.filter.setClearButtonEnabled(True)
            self.filter.textChanged.connect(lambda _t: self.apply_filter())
            filt.addWidget(self.filter, 2)
            self.hide_done = QtWidgets.QCheckBox("Hide finished")
            self.hide_done.toggled.connect(lambda _b: self.apply_filter())
            filt.addWidget(self.hide_done)
            filt.addStretch(1)
            self.count_label = QtWidgets.QLabel("")
            self.count_label.setObjectName("muted")
            filt.addWidget(self.count_label)
            filt.addSpacing(12)
            wl = QtWidgets.QLabel("Workers")
            filt.addWidget(wl)
            self.workers = QtWidgets.QSpinBox()
            self.workers.setRange(0, 512)
            self.workers.setSpecialValueText("from config")
            self.workers.setToolTip("--workers for runs started here (from config: its n_workers)")
            wl.setBuddy(self.workers)
            filt.addWidget(self.workers)

            self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
            self.table.setHorizontalHeaderLabels(self.COLUMNS)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setAlternatingRowColors(True)
            self.table.setShowGrid(False)
            self.table.setWordWrap(False)
            self.table.verticalHeader().setVisible(False)
            self.table.verticalHeader().setDefaultSectionSize(32)
            self.table.setSortingEnabled(True)
            self.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
            hh = self.table.horizontalHeader()
            hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
            hh.setStretchLastSection(True)
            for col, wdt in enumerate((330, 130, 150, 300, 230, 330)):
                self.table.setColumnWidth(col, wdt)
            self.table.itemSelectionChanged.connect(self.selection_changed)
            self.table.doubleClicked.connect(lambda _i: self.action("monitor"))
            QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Return), self.table,
                            activated=lambda: self.action("monitor"))

            # empty state
            self.empty = QtWidgets.QWidget()
            el = QtWidgets.QVBoxLayout(self.empty)
            el.addStretch(1)
            et = QtWidgets.QLabel("<h3>No configs in this workspace yet</h3>"
                                  "Configs are <code>*.yaml</code> files with a <code>replica_exchange</code> "
                                  "section, up to four folders deep.")
            et.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            el.addWidget(et)
            eb = QtWidgets.QHBoxLayout()
            eb.addStretch(1)
            eb.addWidget(_button("New config from a template", self.new_config, "", "SP_FileIcon",
                                 "primary"))
            eb.addWidget(_button("Open another workspace…", lambda: self.main.open_workspace(), "",
                                 "SP_DirOpenIcon"))
            eb.addStretch(1)
            el.addLayout(eb)
            el.addStretch(2)
            self.stack = QtWidgets.QStackedWidget()
            self.stack.addWidget(self.table)
            self.stack.addWidget(self.empty)

            log_w = QtWidgets.QWidget()
            lv = QtWidgets.QVBoxLayout(log_w)
            lv.setContentsMargins(0, 0, 0, 0)
            self.log_title = QtWidgets.QLabel("<b>Log</b>")
            lv.addWidget(self.log_title)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setFont(_mono())
            self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.log.setPlaceholderText("Select a run to see the end of its log.")
            lv.addWidget(self.log)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            split.addWidget(self.stack)
            split.addWidget(log_w)
            split.setSizes([600, 260])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(bar)
            lay.addLayout(filt)
            lay.addWidget(split, 1)
            self.selection_changed()

        def new_config(self) -> None:
            """Go to the Build tab with the first template."""
            if self.main.build.load("t", next(iter(TEMPLATES))):
                self.main.tabs.setCurrentWidget(self.main.build)
                self.main.build.save_path.setFocus()

        def selected(self) -> Optional[dict]:
            """The selected config row, or None."""
            r = self.table.currentRow()
            item = self.table.item(r, 0) if r >= 0 else None
            if item is None or not self.table.selectionModel().hasSelection():
                return None
            return self.by_path.get(item.data(QtCore.Qt.ItemDataRole.UserRole))

        def select(self, path: str) -> None:
            """
            Select a config's row.

            Parameters
            ----------
            path : str
                Workspace-relative config path.
            """
            path = os.path.normpath(path)
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it is not None and it.data(QtCore.Qt.ItemDataRole.UserRole) == path:
                    self.table.selectRow(r)
                    return

        def _item(self, text: str, tip: str = "", sort=None) -> "QtWidgets.QTableWidgetItem":
            """A table cell with a tooltip (full text) and an optional sort key."""
            it = QtWidgets.QTableWidgetItem(text)
            it.setToolTip(tip or text)
            if sort is not None:
                it.setData(QtCore.Qt.ItemDataRole.UserRole + 1, sort)
            return it

        def update_rows(self, configs: List[dict]) -> None:
            """
            Refill the table, keeping the selection, sort order and filter.

            Parameters
            ----------
            configs : list of dict
                `Workspace.overview` rows.
            """
            t = self.main.theme
            keep = (self.selected() or {}).get("path")
            self.rows = configs
            self.by_path = {c["path"]: c for c in configs}
            self.stack.setCurrentWidget(self.table if configs else self.empty)
            self.table.setSortingEnabled(False)
            self.table.setRowCount(len(configs))
            tab_font = _tabular(self.table.font())
            for i, c in enumerate(configs):
                j = c["job"]
                conv, sub = j.get("convergence"), j.get("subsurface")
                name = self._item(c["path"])
                name.setData(QtCore.Qt.ItemDataRole.UserRole, c["path"])
                self.table.setItem(i, 0, name)
                tok, icon, why = STATE_STYLE.get(j["state"], ("neutral", "SP_FileIcon", ""))
                if j.get("external"):
                    why = "started outside the GUI (a script or regc-run); found by its heartbeat"
                label = j["state"] + (" (external)" if j.get("external") else "")
                st = self._item(label, f"{label}: {why}" + (f" (pid {j['pid']})" if j["pid"] else ""))
                st.setIcon(_icon(icon))
                st.setForeground(QtGui.QColor(t[tok]))
                f = st.font()
                f.setWeight(QtGui.QFont.Weight.DemiBold)
                st.setFont(f)
                self.table.setItem(i, 1, st)
                has_prog = j.get("step") is not None and bool(j.get("n_steps"))
                prog = f"{j['step']:,} / {j['n_steps']:,}" if has_prog else "–"
                pit = self._item("" if has_prog else "–", prog)
                pit.setData(QtCore.Qt.ItemDataRole.DisplayRole, "" if has_prog else "–")
                self.table.setItem(i, 2, pit)
                if has_prog:
                    pb = QtWidgets.QProgressBar()
                    pb.setRange(0, int(j["n_steps"]))
                    pb.setValue(min(int(j["step"]), int(j["n_steps"])))
                    pb.setFormat(prog)
                    pb.setFont(tab_font)
                    pb.setToolTip(f"{prog} steps ({100 * j['step'] / j['n_steps']:.1f} %)"
                                  + (" — n_steps is a cap; the convergence check can stop earlier"
                                     if conv else ""))
                    self.table.setCellWidget(i, 2, pb)
                else:
                    self.table.removeCellWidget(i, 2)
                ctext = convergence_text(conv)
                cit = self._item(ctext or "–", "latest ρ(u) check: " + (ctext or "none yet"))
                cit.setFont(tab_font)
                if conv and conv.get("converged"):
                    cit.setIcon(_icon("SP_DialogApplyButton"))
                self.table.setItem(i, 3, cit)
                stext = (f"{sub['n_now']} now · ins {sub['inserted']:,}/{sub['insert_attempts']:,} · "
                         f"rem {sub['removed']:,}/{sub['delete_attempts']:,}") if sub else "–"
                sit = self._item(stext, "subsurface atoms now (all replicas); accepted / attempted "
                                        "NCMC insertions and removals" if sub else "no subsurface moves")
                sit.setFont(tab_font)
                self.table.setItem(i, 4, sit)
                self.table.setItem(i, 5, self._item(c["output"]))
                last = self._item(j.get("last_line", "").strip())
                last.setFont(_mono())
                self.table.setItem(i, 6, last)
            self.table.setSortingEnabled(True)
            self.apply_filter()
            if keep:
                self.select(keep)
            self.selection_changed()

        def apply_filter(self) -> None:
            """Hide rows that do not match the filter."""
            text = self.filter.text().strip().lower()
            shown = 0
            for r in range(self.table.rowCount()):
                c = self.by_path.get(self.table.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole), {})
                state = c.get("job", {}).get("state", "")
                hide = ((text and text not in c.get("path", "").lower() and text not in state)
                        or (self.hide_done.isChecked() and state in ("finished", "converged")))
                self.table.setRowHidden(r, bool(hide))
                shown += not hide
            running = sum(1 for c in self.rows if c["job"]["state"] in ("running", "starting", "stalled"))
            self.count_label.setText(f"{shown} of {len(self.rows)} configs · {running} running")

        def selection_changed(self) -> None:
            """Enable the actions that fit the selected run, and show its log."""
            c = self.selected()
            j = (c or {}).get("job") or {}
            live = j.get("state") in ("running", "starting", "stalled")
            resumable = bool(j.get("resumable"))
            enabled = {"edit": bool(c), "run": bool(c) and not live and not resumable,
                       "resume": bool(c) and not live and resumable,
                       "overwrite": bool(c) and not live and resumable,
                       "analyze": bool(c) and not live,
                       "stop": bool(c) and live and not j.get("external"),
                       "monitor": bool(c)}
            for m, b in self.buttons.items():
                b.setEnabled(enabled[m])
            if c:
                out = c["output_dir"]           # may lie outside the workspace
                self.log_title.setText(f"<b>Log</b> &nbsp;<span style='color:{self.main.theme['muted_fg']}'>"
                                       f"{c['output']}/run.log</span>")
                text = tail_text(os.path.join(out, "run.log"), 20_000)
                bar = self.log.verticalScrollBar()
                at_end = bar.value() >= bar.maximum() - 2
                if text != self.log.toPlainText():
                    self.log.setPlainText(text or "(no run.log yet)")
                    if at_end:
                        bar.setValue(bar.maximum())
            else:
                self.log_title.setText("<b>Log</b>")
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
                QtWidgets.QMessageBox.warning(self, "Could not start", f"{err}")
                return False
            verb = {"run": "Started", "resume": "Resumed", "overwrite": "Started over",
                    "analyze": "Analysing"}[mode]
            self.main.notify(f"{verb} {rel} (pid {d['pid']}); log {d['log']}")
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
            B = QtWidgets.QMessageBox.StandardButton
            if mode == "edit":
                if self.main.build.load("c", c["path"]):
                    self.main.tabs.setCurrentWidget(self.main.build)
            elif mode == "monitor":
                self.main.monitor.show_run(c["output_dir"])
                self.main.tabs.setCurrentWidget(self.main.monitor)
            elif mode == "stop":
                box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "Stop run",
                                            f"Stop the run of {c['path']}?",
                                            B.Cancel, self)
                box.setInformativeText("It can be resumed from its last checkpoint; steps since "
                                       "then are lost.")
                stop = box.addButton("Stop run", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
                box.setDefaultButton(B.Cancel)
                box.exec()
                if box.clickedButton() is not stop:
                    return
                try:
                    d = self.main.ws.stop(c["path"])
                    self.main.notify(f"Sent SIGTERM to {c['path']} (pid {d['stopped']})")
                except (ValueError, OSError) as err:
                    QtWidgets.QMessageBox.warning(self, "Could not stop", str(err))
                self.main.refresh()
            else:
                if mode == "overwrite":
                    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Warning, "Start over",
                                                f"Start {c['path']} over?", B.Cancel, self)
                    box.setInformativeText(f"The checkpoint, samples and records in {c['output']} "
                                           f"are replaced. This cannot be undone.")
                    go = box.addButton("Start over", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
                    box.setDefaultButton(B.Cancel)
                    box.exec()
                    if box.clickedButton() is not go:
                        return
                self.start(c["path"], mode)

    # ------------------------------------------------------------ Monitor tab
    class MonitorTab(QtWidgets.QWidget):
        """One run's heartbeat: stat tiles, replica grid, traces, convergence, swaps, log."""

        GRID_QUANTITIES = [("mean_N", "mean N"), ("current_N", "current N"), ("std_N", "std of N"),
                           ("n_samples", "samples"), ("mean_U", "mean U (eV)"),
                           ("current_U", "current U (eV)"), ("mean_N_ads", "mean N adsorbed"),
                           ("mean_N_sub", "mean N subsurface"),
                           ("current_N_sub", "current N subsurface"),
                           ("sub_accepted", "subsurface moves accepted"),
                           ("sub_acceptance", "subsurface acceptance")]
        TILES = [("state", "State"), ("step", "Step"), ("rate", "Speed"), ("left", "Time left (cap)"),
                 ("N", "Mean N now"), ("sub", "Subsurface"), ("conv", "Convergence"),
                 ("beat", "Heartbeat")]

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
            self.ax_grid = None

            top = QtWidgets.QHBoxLayout()
            rl = QtWidgets.QLabel("Run")
            top.addWidget(rl)
            self.run_box = QtWidgets.QComboBox()
            self.run_box.setMinimumWidth(380)
            self.run_box.activated.connect(lambda _i: self.show_run(self.run_box.currentData()))
            rl.setBuddy(self.run_box)
            top.addWidget(self.run_box, 1)
            top.addWidget(_button("Open folder…", self.open_dir, "monitor any output directory",
                                  "SP_DirOpenIcon"))
            top.addSpacing(14)
            gl = QtWidgets.QLabel("Grid shows")
            top.addWidget(gl)
            self.grid_q = QtWidgets.QComboBox()
            for k, label in self.GRID_QUANTITIES:
                self.grid_q.addItem(label, k)
            self.grid_q.currentIndexChanged.connect(lambda _i: self.draw())
            gl.setBuddy(self.grid_q)
            top.addWidget(self.grid_q)

            tiles = QtWidgets.QGridLayout()
            tiles.setSpacing(8)
            self.tiles: Dict[str, StatTile] = {}
            for i, (k, cap) in enumerate(self.TILES):
                self.tiles[k] = StatTile(cap)
                tiles.addWidget(self.tiles[k], 0, i)
                tiles.setColumnStretch(i, 2 if k in ("conv", "sub") else 1)

            self.fig = Figure(figsize=(10, 7), layout="constrained")
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.mpl_connect("button_press_event", self.on_click)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setToolTip("zoom, pan and save the figure; auto-refresh pauses while "
                                    "zoom or pan is on")
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setFont(_mono())
            self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
            self.log.setPlaceholderText("The end of the run's log appears here.")
            fig_w = QtWidgets.QWidget()
            fv = QtWidgets.QVBoxLayout(fig_w)
            fv.setContentsMargins(0, 0, 0, 0)
            fv.addWidget(self.toolbar)
            fv.addWidget(self.canvas, 1)
            split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            split.addWidget(fig_w)
            split.addWidget(self.log)
            split.setSizes([720, 140])
            lay = QtWidgets.QVBoxLayout(self)
            lay.addLayout(top)
            lay.addLayout(tiles)
            lay.addWidget(split, 1)
            self.draw_empty()

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
                out = c["output_dir"]
                if out not in have:
                    self.run_box.addItem(f"{c['output']}   ({c['path']})", out)
                    have.add(out)
            if self.out_dir is None:
                self.run_box.setCurrentIndex(-1)
                self.run_box.setPlaceholderText("choose a run, or double-click one on the Runs tab")

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
                ``status``, ``samples`` (fallback), ``metadata``, ``mixing``, ``job``, ``log``.
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

            While the plot toolbar's zoom or pan is on, only the tiles update,
            so the view is kept.

            Parameters
            ----------
            force : bool, optional
                Redraw anyway. Default False.
            """
            if not self.out_dir:
                return
            path = os.path.join(self.out_dir, monitor.STATUS_FILE)
            mtime = os.path.getmtime(path) if os.path.exists(path) else None
            if not force and self.data is not None and (mtime == self._mtime or self.toolbar.mode):
                self.data["job"] = self.main.ws.job(self.out_dir)
                self.draw_tiles()
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
        def _labels(self, shape) -> tuple:
            """Row (T) and column (p or μ) tick labels, and the column axis title."""
            st, meta = self.data.get("status") or {}, self.data.get("metadata") or {}
            lab = st.get("labels") or {}
            T = lab.get("temperatures_K") or meta.get("temperatures_K")
            P = lab.get("pressures") or meta.get("pressures")
            PG = lab.get("pressure_grid") or meta.get("pressure_grid")
            unit = lab.get("pressure_unit") or meta.get("pressure_unit") or ""
            rows = [f"{t:.4g} K" for t in T] if T else [str(i) for i in range(shape[0])]
            CL = lab.get("column_labels") or meta.get("column_labels")
            if CL and len(CL) == shape[1]:   # bias columns: pressure, then the mu shift
                return rows, [c.replace("mu_", "μ_") for c in CL], \
                    f"pressure ({unit}); bias columns: shifted μ"
            if PG and len(PG) == shape[0]:   # an adapted ladder: pressures differ per row
                cols = []
                for j in range(shape[1]):
                    lo, hi = min(r[j] for r in PG), max(r[j] for r in PG)
                    cols.append(f"{lo:.0e}" + (f"–{hi:.0e}" if hi > lo * (1 + 1e-9) else ""))
                return rows, cols, f"pressure ({unit}), range over T"
            cols = [f"{p:.0e}" for p in P] if P else [str(j) for j in range(shape[1])]
            return rows, cols, (f"pressure ({unit})" if P else "μ column")

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

        def draw_tiles(self) -> None:
            """The stat tiles."""
            t = self.main.theme
            d = self.data or {}
            st = d.get("status")
            j = d.get("job") or {}
            T = self.tiles
            state = j.get("state", "–")
            tok, icon, why = STATE_STYLE.get(state, ("neutral", "SP_FileIcon", ""))
            T["state"].set(state, f"pid {j['pid']}" if j.get("pid") else "", t[tok], icon, why)
            for k in ("step", "rate", "left", "N", "sub", "conv", "beat"):
                T[k].set("–")
            if not st:
                if d.get("samples"):
                    T["step"].set("no heartbeat", f"samples.npz: {d['samples']['n_samples']:,} "
                                                  f"samples per replica")
                return
            step, n = st["step"], st["n_steps"]
            T["step"].set(f"{step:,}", f"of {n:,} ({100 * step / max(n, 1):.1f} %)")
            h = [x for x in self.history.get(self.out_dir, []) if time.time() - x[0] < 1800]
            rate = None
            if len(h) >= 2 and h[-1][1] > h[0][1]:
                rate = (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])
            elif st.get("started") and st["time"] > st["started"]:
                rate = step / (st["time"] - st["started"])
            if rate:
                T["rate"].set(f"{rate:.3g} steps/s", f"{1 / rate:.2f} s per step")
                if state in ("running", "stalled"):
                    T["left"].set(f"≤ {_fmt_dt(max(0, n - step) / rate)}",
                                  "to the n_steps cap" + ("; the check can stop earlier"
                                                          if (st.get("convergence") or {}).get("settings")
                                                          else ""))
            if st.get("started"):
                T["beat"].set(f"{_fmt_dt(time.time() - st['time'])} ago",
                              f"elapsed {_fmt_dt(st['time'] - st['started'])}")
            cur = [x for row in st["replicas"]["current_N"] for x in row if x is not None]
            if cur:
                T["N"].set(f"{np.mean(cur):.2f}", f"range {min(cur):g}–{max(cur):g} over "
                                                  f"{len(cur)} replicas")
            sub = st.get("subsurface")
            if sub:
                sp = "+".join(sub["species"])
                T["sub"].set(f"{sub['n_now']} {sp} now", f"in {sub['replicas_occupied']} replicas · "
                             f"inserted {sub['inserted']:,}/{sub['insert_attempts']:,} · removed "
                             f"{sub['removed']:,}/{sub['delete_attempts']:,}",
                             tip="atoms in the subsurface region across all replicas; accepted / "
                                 "attempted NCMC insertions and removals")
            conv = st.get("convergence") or {}
            checks = conv.get("checks") or []
            if checks:
                c = checks[-1]
                ok = bool(c.get("converged"))
                T["conv"].set("converged" if ok else "not yet",
                              convergence_text(c, conv.get("settings")),
                              t["success"] if ok else t["fg"],
                              "SP_DialogApplyButton" if ok else "SP_MessageBoxInformation",
                              "latest ρ(u) check: both halves of every replica within sampling noise"
                              " when max and median ratios are below their tolerances")

        def draw_empty(self) -> None:
            """The empty state: no run chosen."""
            t = self.main.theme
            self.fig.clear()
            self.fig.set_facecolor(t["bg"])
            self.fig.text(0.5, 0.55, "No run selected", ha="center", fontsize=14, color=t["fg"],
                          weight="bold")
            self.fig.text(0.5, 0.47, "Choose one in “Run” above, or double-click a run on the "
                          "Runs tab.", ha="center", fontsize=10, color=t["muted_fg"])
            self.ax_grid = None
            self.canvas.draw_idle()
            for tile in self.tiles.values():
                tile.set("–")

        def draw(self) -> None:
            """Redraw everything."""
            if not self.data:
                return self.draw_empty()
            t = self.main.theme
            self.draw_tiles()
            log = self.data.get("log") or ""
            if log != self.log.toPlainText():
                self.log.setPlainText(log or "(no run.log)")
                self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
            fig = self.fig
            fig.clear()
            fig.set_facecolor(t["bg"])
            gs = fig.add_gridspec(2, 2)
            ax_g, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
            ax_c, ax_s = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
            for ax in (ax_g, ax_t, ax_c, ax_s):
                style_axes(ax, t)
            self.ax_grid = ax_g
            z, keys, traces = self.grid()
            st = self.data.get("status")
            c1, c2, c3, c4 = t["series"]

            def empty(ax, text):
                ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
                        color=t["muted_fg"], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            # replica grid (heat map with numeric colour bar and printed values)
            if z is not None and np.isfinite(z).any():
                rows, cols, xl = self._labels(z.shape)
                cmap = matplotlib.colormaps["viridis"]
                im = ax_g.imshow(z, origin="lower", aspect="auto", cmap=cmap)
                cb = fig.colorbar(im, ax=ax_g, fraction=0.05, pad=0.02)
                cb.ax.tick_params(colors=t["muted_fg"], labelcolor=t["fg"], labelsize=8)
                cb.outline.set_edgecolor(t["border"])
                ax_g.set_xticks(range(z.shape[1]), cols, rotation=40, ha="right")
                ax_g.set_yticks(range(z.shape[0]), rows)
                ax_g.set_xlabel(xl)
                if z.size <= 150:
                    for (i, k), v in np.ndenumerate(z):
                        if np.isfinite(v):
                            r, g, b, _ = cmap(im.norm(v))
                            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            ax_g.text(k, i, f"{v:.3g}", ha="center", va="center", fontsize=7,
                                      color="black" if lum > 0.45 else "white")
                if self.sel is None and keys:
                    self.sel = keys[0]
                if self.sel is not None:
                    ax_g.add_patch(matplotlib.patches.Rectangle(
                        (self.sel[1] - 0.5, self.sel[0] - 0.5), 1, 1, fill=False, lw=2.2,
                        ec=t["danger"]))

                def fmt(x, y, z=z, rows=rows, cols=cols):
                    i, k = int(round(y)), int(round(x))
                    if 0 <= i < z.shape[0] and 0 <= k < z.shape[1]:
                        return f"{rows[i]}, {cols[k]}: {self.grid_q.currentText()} = {z[i, k]:.4g}"
                    return ""
                ax_g.format_coord = fmt
                ax_g.set_title(f"{self.grid_q.currentText()} per replica (click a cell)", fontsize=10)
            else:
                empty(ax_g, f"“{self.grid_q.currentText()}” is not recorded in this run")
            # traces of the selected replica (line styles + colours + legend)
            if traces and keys and self.sel in keys:
                i = keys.index(self.sel)
                stride = traces.get("stride", 1)
                N = np.asarray(traces["N"][i], dtype=float)
                x = np.arange(len(N)) * stride
                ax_t.plot(x, N, lw=1.3, color=c1, label="N (left)")
                if traces.get("N_sub") and len(traces["N_sub"][i]):
                    ax_t.plot(x[:len(traces["N_sub"][i])], traces["N_sub"][i], lw=1.6, ls="--",
                              color=c2, label="N subsurface (left)")
                ax_t.set_xlabel("sample index")
                ax_t.set_ylabel("atoms")
                ax_t.grid(True, color=t["grid"], lw=0.6)
                ax_u = ax_t.twinx()
                ax_u.plot(x, traces["U"][i], lw=0.9, ls=":", color=c3, label="U (right)")
                ax_u.set_ylabel("U (eV)", color=t["fg"])
                ax_u.tick_params(colors=t["muted_fg"], labelcolor=t["fg"], labelsize=8)
                for sp in ax_u.spines.values():
                    sp.set_color(t["border"])
                h1, l1 = ax_t.get_legend_handles_labels()
                h2, l2 = ax_u.get_legend_handles_labels()
                leg = ax_t.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left",
                                  facecolor=t["surface"], edgecolor=t["border"], labelcolor=t["fg"])
                leg.set_zorder(5)
                rows, cols, _ = self._labels((max(k[0] for k in keys) + 1,
                                              max(k[1] for k in keys) + 1))
                ax_t.set_title(f"Replica {rows[self.sel[0]]}, {cols[self.sel[1]]}"
                               + (f" (every {stride}th sample)" if stride > 1 else ""), fontsize=10)
            else:
                empty(ax_t, "No traces yet")
            # convergence checks against their tolerances (direct labels)
            conv = (st or {}).get("convergence") or {}
            checks = conv.get("checks") or []
            hist = (st or {}).get("convergence_history") or []
            s = conv.get("settings") or {}
            if checks and checks[0].get("max_ratio") is not None:
                steps = [c.get("step", i + 1) for i, c in enumerate(checks)]
                tol, mtol = s.get("tol", 3.0), s.get("median_tol", 1.3)
                ax_c.plot(steps, [c["max_ratio"] for c in checks], "o-", color=c1, lw=1.5,
                          label="max over replicas")
                ax_c.plot(steps, [c["median_ratio"] for c in checks], "s--", color=c3, lw=1.5,
                          label="median")
                ax_c.axhline(tol, ls="-", color=c1, lw=0.8, alpha=0.6)
                ax_c.axhline(mtol, ls="--", color=c3, lw=0.8, alpha=0.6)
                x_end = ax_c.get_xlim()[1]
                ax_c.text(x_end, tol, f" max tol {tol:g}", va="bottom", ha="right", fontsize=8,
                          color=t["fg"])
                ax_c.text(x_end, mtol, f" median tol {mtol:g}", va="bottom", ha="right", fontsize=8,
                          color=t["fg"])
                ax_c.set_xlabel("step")
                ax_c.set_ylabel("TV distance / noise floor")
                ax_c.set_ylim(bottom=0)
                ax_c.grid(True, color=t["grid"], lw=0.6)
                ax_c.legend(fontsize=8, loc="upper left", facecolor=t["surface"],
                            edgecolor=t["border"], labelcolor=t["fg"])
                ax_c.set_title(f"ρ(u) halves vs noise (min n_eff {checks[-1]['min_neff']:.1f}, "
                               f"need {s.get('min_neff', 10):g})", fontsize=10)
            elif hist:
                ax_c.plot(range(1, len(hist) + 1), hist, "o-", color=c1)
                ax_c.set_xlabel("check")
                ax_c.set_ylabel("largest Δρ(u)")
                ax_c.grid(True, color=t["grid"], lw=0.6)
                ax_c.set_title("ρ(u) change between blocks", fontsize=10)
            else:
                empty(ax_c, "No convergence checks yet\n(replica_exchange.convergence_every)")
            # swaps (bar labels with values)
            acc = (st or {}).get("swap_acceptance") or (self.data.get("mixing") or {}).get(
                "swap_acceptance") or {}
            if acc:
                names = list(acc)
                vals = [acc[n] or 0 for n in names]
                bars = ax_s.bar(names, vals, color=c1, alpha=0.85, edgecolor=c1, lw=1)
                ax_s.bar_label(bars, [f"{v:.2f}" for v in vals], fontsize=8, color=t["fg"], padding=2)
                ax_s.set_ylim(0, 1)
                ax_s.set_ylabel("acceptance")
                ax_s.grid(True, axis="y", color=t["grid"], lw=0.6)
                ax_s.set_axisbelow(True)
                tT = (st or {}).get("round_trips_T")
                ax_s.set_title("Replica swaps" + (f" · T round trips: median {np.median(tT):.1f}, "
                                                  f"{sum(1 for x in tT if x == 0)} of {len(tT)} "
                                                  f"walkers none" if tT else ""), fontsize=10)
            else:
                empty(ax_s, "No swap statistics yet")
            self.canvas.draw_idle()

        def on_click(self, event) -> None:
            """Select the replica under the mouse on the grid."""
            if self.toolbar.mode or self.ax_grid is None or event.inaxes is not self.ax_grid:
                return
            if event.xdata is None or event.ydata is None:
                return
            self.sel = (int(round(event.ydata)), int(round(event.xdata)))
            self.draw()

    def _fmt_dt(s: float) -> str:
        """A duration as s, min or h."""
        s = float(s)
        if s < 120:
            return f"{s:.0f} s"
        if s < 7200:
            return f"{s / 60:.0f} min"
        return f"{s / 3600:.1f} h"

    # ------------------------------------------------------------ main window
    class MainWindow(QtWidgets.QMainWindow):
        """The window: workspace bar, Runs / Build / Monitor tabs, theme, refresh timer."""

        def __init__(self, root: str = ".", python: Optional[str] = None,
                     pythonpath: Optional[str] = None, theme: str = "auto"):
            """
            Build the window.

            Parameters
            ----------
            root : str, optional
                Workspace directory. Default the current one.
            python, pythonpath : str, optional
                Interpreter and ``PYTHONPATH`` of the runs (see `regc.gui.Workspace`).
            theme : str, optional
                ``auto`` (follow the system), ``light`` or ``dark``. Default ``auto``.
            """
            super().__init__()
            self.ws = Workspace(root, python, pythonpath)
            self.theme_mode = theme
            self.theme = THEMES[self._resolve_theme()]
            apply_theme(QtWidgets.QApplication.instance(), self.theme)
            self.resize(1560, 980)
            self.setWindowIcon(_icon("SP_ComputerIcon"))

            bar = QtWidgets.QToolBar("Workspace")
            bar.setMovable(False)
            bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.addToolBar(bar)
            act = QtGui.QAction(_icon("SP_DirOpenIcon"), "Open workspace…", self)
            act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
            act.triggered.connect(self.open_workspace)
            bar.addAction(act)
            self.root_label = QtWidgets.QLabel()
            self.root_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            bar.addWidget(self.root_label)
            bar.addSeparator()
            ref = QtGui.QAction(_icon("SP_BrowserReload"), "Refresh", self)
            ref.setShortcut(QtGui.QKeySequence.StandardKey.Refresh)
            ref.triggered.connect(self.refresh)
            bar.addAction(ref)
            self.auto = QtWidgets.QCheckBox("Auto-refresh (5 s)")
            self.auto.setChecked(True)
            bar.addWidget(self.auto)
            spacer = QtWidgets.QWidget()
            spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                 QtWidgets.QSizePolicy.Policy.Preferred)
            bar.addWidget(spacer)
            ppl = QtWidgets.QLabel("Code for new runs (PYTHONPATH) ")
            ppl.setObjectName("muted")
            bar.addWidget(ppl)
            self.pp = QtWidgets.QLineEdit(self.ws.pythonpath)
            self.pp.setMinimumWidth(380)
            self.pp.setToolTip(f"interpreter: {self.ws.python}\nPYTHONPATH of the runs started from "
                               f"here, e.g. another git worktree. Press Enter to apply.")
            self.pp.editingFinished.connect(self.set_pythonpath)
            ppl.setBuddy(self.pp)
            bar.addWidget(self.pp)

            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setDocumentMode(True)
            self.runs, self.build, self.monitor = RunsTab(self), BuildTab(self), MonitorTab(self)
            self.tabs.addTab(self.runs, _icon("SP_FileDialogListView"), "&Runs")
            self.tabs.addTab(self.build, _icon("SP_FileDialogDetailedView"), "&Build")
            self.tabs.addTab(self.monitor, _icon("SP_ComputerIcon"), "&Monitor")
            for i in range(3):
                QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self,
                                activated=lambda i=i: self.tabs.setCurrentIndex(i))
                self.tabs.setTabToolTip(i, f"Ctrl+{i + 1}")
            self.tabs.currentChanged.connect(lambda _i: self.tick())
            self.setCentralWidget(self.tabs)
            self.statusBar().showMessage(f"Interpreter {self.ws.python}", 5000)

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
            view = self.menuBar().addMenu("&View")
            group = QtGui.QActionGroup(self)
            for mode, text in (("auto", "Theme: follow system"), ("light", "Theme: light"),
                               ("dark", "Theme: dark")):
                act = QtGui.QAction(text, self, checkable=True)
                act.setChecked(mode == theme)
                act.triggered.connect(lambda _c=False, m=mode: self.set_theme(m))
                group.addAction(act)
                view.addAction(act)
            hints = QtGui.QGuiApplication.styleHints()
            if hasattr(hints, "colorSchemeChanged"):
                hints.colorSchemeChanged.connect(lambda _s: self.theme_mode == "auto"
                                                 and self.set_theme("auto"))

            self.update_title()
            self.refresh()
            self.build.load("t", next(iter(TEMPLATES)), ask=False)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.tick)
            self.timer.start(REFRESH_MS)

        def _resolve_theme(self) -> str:
            """``light`` or ``dark`` for the theme mode."""
            if self.theme_mode in ("light", "dark"):
                return self.theme_mode
            try:
                dark = (QtGui.QGuiApplication.styleHints().colorScheme()
                        == QtCore.Qt.ColorScheme.Dark)
            except AttributeError:
                dark = False
            return "dark" if dark else "light"

        def set_theme(self, mode: str) -> None:
            """
            Switch the theme.

            Parameters
            ----------
            mode : str
                ``auto``, ``light`` or ``dark``.
            """
            self.theme_mode = mode
            self.theme = THEMES[self._resolve_theme()]
            apply_theme(QtWidgets.QApplication.instance(), self.theme)
            self.build.highlighter.set_theme(self.theme)
            self.refresh()
            self.monitor.draw()

        def update_title(self) -> None:
            """Window title: workspace, unsaved marker."""
            dirty = getattr(self, "build", None) is not None and self.build.dirty
            self.setWindowTitle(f"{'• ' if dirty else ''}regc — {os.path.basename(self.ws.root)}")
            if hasattr(self, "root_label"):
                self.root_label.setText(f" {self.ws.root} ")

        def notify(self, text: str) -> None:
            """
            A short confirmation in the status bar (5 s).

            Parameters
            ----------
            text : str
                The message.
            """
            self.statusBar().showMessage(text, 5000)

        def set_pythonpath(self) -> None:
            """Apply the PYTHONPATH field."""
            if self.pp.text().strip() != self.ws.pythonpath:
                self.ws.pythonpath = self.pp.text().strip()
                self.notify(f"New runs will use PYTHONPATH={self.ws.pythonpath}")

        def open_workspace(self) -> None:
            """Choose another workspace directory."""
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Workspace", self.ws.root)
            if not d or not self.build.confirm_discard():
                return
            self.ws = Workspace(d, self.ws.python, self.ws.pythonpath)
            self.update_title()
            self.monitor.run_box.clear()
            self.monitor.out_dir, self.monitor.data = None, None
            self.monitor.draw_empty()
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

        def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
            """Ask about unsaved changes; runs keep going."""
            if self.build.confirm_discard():
                event.accept()
            else:
                event.ignore()


def run(root: str = ".", python: Optional[str] = None, pythonpath: Optional[str] = None,
        argv: Optional[list] = None, theme: str = "auto") -> int:
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
    theme : str, optional
        ``auto`` (follow the system), ``light`` or ``dark``. Default ``auto``.

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
    win = MainWindow(root, python, pythonpath, theme)
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
    ap.add_argument("--theme", choices=("auto", "light", "dark"), default="auto",
                    help="colour theme (default: follow the system)")
    ap.add_argument("--web", action="store_true",
                    help="serve the browser version instead (regc.gui; --port, --host)")
    ap.add_argument("--port", type=int, default=8060, help="--web: port (default 8060)")
    ap.add_argument("--host", default="127.0.0.1", help="--web: interface (default 127.0.0.1)")
    args = ap.parse_args(argv)
    if args.web:
        from .gui import serve
        serve(args.workspace, args.host, args.port, args.python, args.pythonpath)
        return 0
    return run(args.workspace, args.python, args.pythonpath, theme=args.theme)


if __name__ == "__main__":
    raise SystemExit(main())
