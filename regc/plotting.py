"""Plotting helpers. Requires matplotlib, imported lazily -- importing this
module never requires it; only calling `plot_phase_diagram` does."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def plot_phase_diagram(T_grid: np.ndarray, p_grid: np.ndarray, theta: np.ndarray,
                        coexistence: Optional[List[Tuple[float, float]]] = None,
                        save_path: str = "phase_diagram.png",
                        colorbar_label: str = "coverage (particles / area)",
                        xlabel: str = "pressure p", ylabel: str = "temperature T",
                        title: str = "Reweighted surface coverage $\\theta(T, p)$") -> None:
    """
    Plot and save a coverage(T, p) heatmap.

    With the (optional) coexistence line overlaid. Requires matplotlib;
    does nothing but print a note if it isn't installed.

    Plots against log10(p) directly, with tick labels reformatted as
    10^x, rather than plotting p on a matplotlib log-scale axis:
    pcolormesh infers cell edges by linearly averaging adjacent coordinate
    values regardless of the axis's display scale, and for log-spaced p
    that linear average is completely dominated by the larger neighbor
    whenever p spans more than a couple of decades -- visually collapsing
    every lower-pressure column into an imperceptible sliver even though
    the underlying data is fine. Working in log10(p) directly sidesteps
    this: the coordinates are then genuinely evenly spaced, so the (still
    linear) edge-averaging is exactly right.

    Parameters
    ----------
    T_grid : numpy.ndarray
        Temperature grid, length n_T.
    p_grid : numpy.ndarray
        Pressure grid, length n_p.
    theta : numpy.ndarray
        Coverage array, shape (n_T, n_p) (e.g. from
        `MBARSurfaceAnalysis.phase_diagram_Tp`).
    coexistence : list of tuple of float, optional
        ``(T, p)`` coexistence-line points to overlay (e.g. from
        `MBARSurfaceAnalysis.coexistence_curve_Tp`). Default is None (no
        overlay).
    save_path : str, optional
        Path to save the figure to. Default is ``"phase_diagram.png"``.
    colorbar_label : str, optional
        Colorbar label. Default ``"coverage (particles / area)"``.
    xlabel, ylabel : str, optional
        Axis labels. Defaults ``"pressure p"`` and ``"temperature T"``.
    title : str, optional
        Plot title. Default: reweighted surface coverage theta(T, p).

    Returns
    -------
    None
        Saves the figure to `save_path` and returns nothing.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib).")
        return

    log_p = np.log10(p_grid)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    mesh = ax.pcolormesh(log_p, T_grid, theta, shading='auto', cmap='viridis')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"$10^{{{x:.0f}}}$"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=colorbar_label)

    if coexistence:
        Tc, pc = zip(*coexistence)
        ax.plot(np.log10(pc), Tc, 'o-', color='white', markeredgecolor='black',
                 label='condensation line')
        ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved phase diagram to {save_path}")


def _cell_edges(c: np.ndarray) -> np.ndarray:
    """
    Edges of the cells centred on a 1-D grid (midpoints, extended at the ends).

    Parameters
    ----------
    c : numpy.ndarray
        Cell centres, length n (n >= 2).

    Returns
    -------
    numpy.ndarray
        Length n + 1.
    """
    mid = 0.5 * (c[1:] + c[:-1])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def _label_boundaries(x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> list:
    """
    Line segments on the cell edges between neighbouring cells of different label.

    Parameters
    ----------
    x, y : numpy.ndarray
        Cell centres along the columns (length n_x) and rows (length n_y).
    labels : numpy.ndarray
        Shape (n_y, n_x).

    Returns
    -------
    list
        ``[[(x0, y0), (x1, y1)], ...]`` for a LineCollection.
    """
    xe, ye = _cell_edges(np.asarray(x, dtype=float)), _cell_edges(np.asarray(y, dtype=float))
    lab = np.asarray(labels)
    segs = []
    for i, j in zip(*np.nonzero(lab[:, 1:] != lab[:, :-1])):   # vertical edges
        segs.append([(xe[j + 1], ye[i]), (xe[j + 1], ye[i + 1])])
    for i, j in zip(*np.nonzero(lab[1:, :] != lab[:-1, :])):   # horizontal edges
        segs.append([(xe[j], ye[i + 1]), (xe[j + 1], ye[i + 1])])
    return segs


def plot_map(T_grid: np.ndarray, grid: np.ndarray, values: np.ndarray, save_path: str,
             log_x: bool = True, valid: Optional[np.ndarray] = None,
             points: Optional[np.ndarray] = None, discrete: bool = False,
             log_color: bool = False, colorbar_label: str = "", xlabel: str = "pressure p",
             ylabel: str = "temperature T", title: str = "", cmap: str = "viridis",
             invalid_label: str = "too few effective samples",
             vmin: Optional[float] = None, vmax: Optional[float] = None,
             contour_levels: Optional[Sequence[float]] = None, contour_label: str = "",
             points_label: str = "$C_V$ peak", figsize: Tuple[float, float] = (6.0, 4.0),
             dpi: int = 300) -> None:
    """
    Plot a (T, p) or (T, mu) map, with unreliable cells shown as gaps.

    Cells where `valid` is False are drawn grey and hatched (not
    interpolated or hidden, and distinct from any colour of the map), and
    named in a legend placed outside the axes so it never covers data.
    Integer label maps (``discrete=True``) get one colour per value up to
    20 values; beyond that the colours are continuous but the boundaries
    between values are outlined. On a logarithmic colour scale, values
    at or below zero are not hidden: they take the colour below the scale
    (black) and get their own legend entry.

    Parameters
    ----------
    T_grid : numpy.ndarray
        Temperatures (y axis), length n_T.
    grid : numpy.ndarray
        Pressures or chemical potentials (x axis), length n_y.
    values : numpy.ndarray
        Shape (n_T, n_y).
    save_path : str
        Output image.
    log_x : bool, optional
        Plot log10 of `grid` (pressures). Default True.
    valid : numpy.ndarray, optional
        Boolean mask of trustworthy cells (e.g. effective sample size above
        a threshold). Default None (all valid).
    points : numpy.ndarray, optional
        ``(T, x)`` points to overlay (e.g. heat-capacity ridges, x in the
        units of `grid`). Default None.
    discrete : bool, optional
        Integer labels (e.g. most probable N). Default False.
    log_color : bool, optional
        Logarithmic colour scale (for quantities spanning decades, like
        C_V). Default False.
    colorbar_label, xlabel, ylabel, title : str, optional
        Labels.
    cmap : str, optional
        Matplotlib colormap (perceptually uniform by default). Default
        ``'viridis'``.
    invalid_label : str, optional
        Legend text for the hatched cells.
    vmin, vmax : float, optional
        Colour range. Default None: the data range (for `log_color`, from
        the 2nd percentile of the positive values; the colorbar arrow
        marks anything below).
    contour_levels : sequence of float, optional
        Draw these contour levels of `values` in white (e.g. an effective
        sample size threshold). Default None.
    contour_label : str, optional
        Legend text for the contours.
    points_label : str, optional
        Legend text for `points`. Default ``'$C_V$ peak'``.
    figsize : tuple of float, optional
        Figure size in inches. Default (6.0, 4.0).
    dpi : int, optional
        Resolution of the saved image. Default 300.

    Returns
    -------
    None
        Saves the figure.
    """
    try:
        import warnings

        import matplotlib as mpl
        import matplotlib.collections  # noqa: F401  (mpl.collections)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib import patheffects
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib).")
        return
    x = np.log10(grid) if log_x else np.asarray(grid, dtype=float)
    raw = np.asarray(values, dtype=float)
    bad = ~np.isfinite(raw)
    if valid is not None:
        bad |= ~np.asarray(valid, bool)
    vals = np.ma.masked_where(bad, raw)
    cm = mpl.colormaps[cmap]
    norm, extend, handles, outline = None, "neither", [], None
    if discrete and vals.count():
        levels = np.unique(vals.compressed()).astype(int)
        if len(levels) <= 20:
            cm = mpl.colors.ListedColormap(mpl.colormaps[cmap](np.linspace(0, 1, len(levels))))
            norm = mpl.colors.BoundaryNorm(np.concatenate([levels - 0.5, [levels[-1] + 0.5]]), cm.N)
        else:
            warnings.warn(f"{len(levels)} distinct labels: continuous colours, boundaries outlined",
                          stacklevel=2)
            norm = mpl.colors.BoundaryNorm(np.arange(levels[0] - 0.5, levels[-1] + 1.0), 256)
            outline = levels[:-1] + 0.5
    elif log_color and vals.count():
        pos = vals.compressed()
        pos = pos[pos > 0]
        if pos.size:
            lo = float(vmin) if vmin is not None else float(np.percentile(pos, 2))
            hi = float(vmax) if vmax is not None else float(pos.max())
            norm = mpl.colors.LogNorm(vmin=lo, vmax=hi)
            nonpos = np.ma.filled(vals <= 0, False)
            if nonpos.any():  # shown, not hidden: below the scale (black)
                vals = np.ma.where(nonpos, lo / 10.0, vals)
                handles.append(Patch(facecolor="black", label="value ≤ 0"))
            extend = "min" if np.ma.min(vals) < lo else "neither"
            cm = cm.with_extremes(under="black")
    elif vmin is not None or vmax is not None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cm = cm.with_extremes(bad="#bdbdbd")
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    mesh = ax.pcolormesh(x, T_grid, vals, shading="auto", cmap=cm, norm=norm)
    if bad.any():  # hatch as well as grey: grey alone can match a colour of the map
        ax.contourf(x, T_grid, bad.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["////"])
        handles.append(Patch(facecolor="#bdbdbd", hatch="////", edgecolor="0.35",
                             label=invalid_label))
    if outline is not None:  # the cell edges between different labels: the phase boundaries
        ax.add_collection(mpl.collections.LineCollection(
            _label_boundaries(x, np.asarray(T_grid, dtype=float), raw), colors="k",
            linewidths=0.5))
    if contour_levels is not None:  # white with a black outline: visible on any colour
        stroke = [patheffects.withStroke(linewidth=2.2, foreground="black")]
        cs = ax.contour(x, T_grid, raw, levels=list(contour_levels), colors="white",
                        linewidths=1.0)
        cs.set(path_effects=stroke)
        handles.append(Line2D([], [], color="white", lw=1.0, path_effects=stroke,
                              label=contour_label or f"contour {list(contour_levels)}"))
    if log_x:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=8))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"$10^{{{v:.0f}}}$"))
    cbar = fig.colorbar(mesh, ax=ax, label=colorbar_label, extend=extend)
    if discrete and outline is None and isinstance(norm, mpl.colors.BoundaryNorm):
        cbar.set_ticks(levels)
    elif outline is not None:
        cbar.set_ticks(mticker.MaxNLocator(integer=True).tick_values(levels[0], levels[-1]))
        cbar.ax.minorticks_off()
    if points is not None and len(points):
        pts = np.asarray(points, dtype=float)
        ax.plot(np.log10(pts[:, 1]) if log_x else pts[:, 1], pts[:, 0], "o", ms=2.5,
                color="white", markeredgecolor="black", markeredgewidth=0.4, linestyle="none")
        handles.insert(0, Line2D([], [], marker="o", color="white", markeredgecolor="black",
                                 linestyle="none", label=points_label))
    if handles:  # outside the axes: a legend must never cover (masked) data
        fig.legend(handles=handles, loc="outside lower center", ncol=min(3, len(handles)),
                   fontsize=8, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {save_path}")


#: Okabe-Ito colours and line styles: colour plus a redundant non-colour cue.
LINE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)))


def plot_radial_distribution(curves: dict, save_path: str,
                             ylabel: str = "$n(r)$ (Å$^{-1}$)", xlabel: str = "$r$ (Å)",
                             title: str = "", xlim: Optional[Tuple[float, float]] = None,
                             figsize: Tuple[float, float] = (5.0, 3.2), dpi: int = 300) -> None:
    """
    Plot reweighted radial distributions at several states.

    Each curve gets a colour-blind-safe colour and its own line style, so
    the curves stay distinguishable in greyscale.

    Parameters
    ----------
    curves : dict
        ``{label: (r, values)}``, e.g. from
        `MBARSurfaceAnalysis.radial_distribution`.
    save_path : str
        Output image.
    ylabel, xlabel, title : str, optional
        Labels.
    xlim : tuple of float, optional
        x range. Default None: from the first nonzero value to the end.
    figsize : tuple of float, optional
        Figure size in inches. Default (5.0, 3.2).
    dpi : int, optional
        Resolution of the saved image. Default 300.

    Returns
    -------
    None
        Saves the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping plot (pip install matplotlib).")
        return
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    first = np.inf
    for k, (label, (r, v)) in enumerate(curves.items()):
        ax.plot(r, v, color=LINE_COLORS[k % len(LINE_COLORS)],
                ls=LINE_STYLES[k % len(LINE_STYLES)], label=label, lw=1.4)
        nz = np.flatnonzero(np.asarray(v) > 0)
        if nz.size:
            first = min(first, float(r[nz[0]]))
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif np.isfinite(first):
        ax.set_xlim(max(0.0, np.floor(first * 2) / 2 - 0.5), float(np.max(r)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {save_path}")
