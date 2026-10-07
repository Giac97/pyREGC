"""Plotting helpers. Requires matplotlib, imported lazily -- importing this
module never requires it; only calling `plot_phase_diagram` does."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def plot_phase_diagram(T_grid: np.ndarray, p_grid: np.ndarray, theta: np.ndarray,
                        coexistence: Optional[List[Tuple[float, float]]] = None,
                        save_path: str = "phase_diagram.png") -> None:
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
    ax.set_xlabel('pressure p')
    ax.set_ylabel('temperature T')
    ax.set_title('Reweighted surface coverage $\\theta(T, p)$')
    fig.colorbar(mesh, ax=ax, label='coverage (particles / area)')

    if coexistence:
        Tc, pc = zip(*coexistence)
        ax.plot(np.log10(pc), Tc, 'o-', color='white', markeredgecolor='black',
                 label='condensation line')
        ax.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved phase diagram to {save_path}")
