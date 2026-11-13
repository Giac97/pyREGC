"""
Figures of the regc manual (docs/manual/regc_manual.md).

Every quantitative figure is computed with the package itself, so the
manual shows what the code does:

* physics figures (gas reservoirs, swap patterns, ideal-gas validation,
  convergence criterion) run in seconds;
* the LJ-surface figures need a run of examples/14_observables_heat_capacity.py
  (``--run-dir``; about 7 min on 12 cores with its defaults). Without one
  they are skipped.

Conventions (scientific-visualization guidance): Okabe-Ito colours with a
second, non-colour cue (line style or marker) for categories; perceptually
uniform sequential maps for ordered quantities; masked cells shown, not
hidden; true minus signs; 300 dpi PNGs sized for a 16 cm text width.

Usage:
    python docs/manual/make_figures.py --run-dir path/to/output_14_observables

Requires: ase, pymbar, scipy, matplotlib.
"""

from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from regc.plotting import LINE_COLORS, LINE_STYLES

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
MINUS = "−"

mpl.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "figure.constrained_layout.use": True,
})


def num(x, fmt="g"):
    """A number with a true minus sign, for text set outside mathtext."""
    return format(x, fmt).replace("-", MINUS)


def save(fig, name):
    """Write a figure as PNG (for the Markdown/PDF manual) and close it."""
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


# -----------------------------------------------------------------------------
# 1. Gas reservoirs: Delta mu of the exchanged atom vs T, with JANAF checks
# -----------------------------------------------------------------------------
def fig_reservoirs():
    """Figure 1: Delta mu = mu_X2 / 2 against T at four pressures, with JANAF points."""
    from regc import (H2_BOND_LENGTH_ANGSTROM, H2_MASS_AMU, H2_VIBRATIONAL_WAVENUMBER_CM,
                      diatomic_gas_mu_eV)
    janaf = {"H2": {300: -0.3186, 500: -0.6065, 1000: -1.4206},
             "O2": {300: -0.5479, 500: -0.9906, 1000: -2.1992}}
    gases = {"H2": (H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM, H2_VIBRATIONAL_WAVENUMBER_CM, 1),
             "O2": (31.998, 1.2075, 1580.2, 3)}
    T = np.linspace(100, 1200, 200)
    pressures = [1e-10, 1e-5, 1.0, 1e5]
    labels = ["$10^{-10}$ bar", "$10^{-5}$ bar", "1 bar", "$10^{5}$ bar"]
    colors = mpl.colormaps["viridis"](np.linspace(0.0, 0.85, len(pressures)))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, (gas, (m, r, w, g)) in zip(axes, gases.items()):
        for k, p_bar in enumerate(pressures):
            mu = [0.5 * diatomic_gas_mu_eV(t, p_bar * 1e5, m, r, w, 2, g) for t in T]
            ax.plot(T, mu, color=colors[k], ls=LINE_STYLES[k], lw=1.4)
            ax.annotate(labels[k], (T[-1], mu[-1]), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=colors[k])
        Tj = sorted(janaf[gas])
        ax.plot(Tj, [0.5 * janaf[gas][t] for t in Tj], "o", mfc="white", mec="black", ms=4.5,
                label="NIST-JANAF, 1 bar")
        ax.axhline(0, color="0.75", lw=0.6)
        ax.set_xlabel("$T$ (K)")
        ax.set_xlim(100, 1500)
        ax.set_xticks([200, 400, 600, 800, 1000, 1200])
        ax.set_title(f"{gas[0]} atoms from {gas[0]}$_2$"
                     + (" (triplet, $g_\\mathrm{el}=3$)" if g == 3 else ""))
    axes[0].set_ylabel(r"$\Delta\mu=\frac{1}{2}\mu_{X_2}(T,p)$ (eV)")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, "reservoir_mu.png")


# -----------------------------------------------------------------------------
# 2. Swap patterns on a replica grid ([ZSG] Fig. 2), from ReplicaExchangeGCMC.swap_pairs
# -----------------------------------------------------------------------------
def fig_swap_patterns():
    """Figure 2: the four swap patterns, drawn from ReplicaExchangeGCMC.swap_pairs."""
    from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC
    n_T, n_mu = 4, 5
    sim = ReplicaExchangeGCMC(list(np.linspace(1, 2, n_T)), list(np.linspace(-3, -1, n_mu)),
                              lambda rng: BulkSystem(4.0, LJCalculator()), seed=0)
    titles = {"T": "$T$ swaps", "mu": r"$\mu$ swaps", "cross_left": "diagonal (left)",
              "cross_right": "diagonal (right)"}
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.0))
    for ax, t in zip(axes, titles):
        for parity in (0, 1):
            for i, (a, b) in enumerate(sim.swap_pairs(t, parity)):
                ax.plot([a[1], b[1]], [a[0], b[0]], ls=LINE_STYLES[parity],
                        color=LINE_COLORS[parity], lw=1.8,
                        label=f"parity {parity}" if i == 0 else None)
        mu_i, T_i = np.meshgrid(range(n_mu), range(n_T))
        ax.plot(mu_i, T_i, "o", color="0.25", ms=4, zorder=3)
        ax.set_title(titles[t])
        ax.set_xticks(range(n_mu))
        ax.set_yticks(range(n_T))
        ax.set_xlabel(r"$\mu$ index")
        ax.set_xlim(-0.5, n_mu - 0.5)
        ax.set_ylim(-0.5, n_T - 0.5)
        ax.set_aspect("equal")
    axes[0].set_ylabel("$T$ index")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2,
               frameon=False)
    save(fig, "swap_patterns.png")


# -----------------------------------------------------------------------------
# 3. Validation: an ideal gas, where <N> = V exp(beta mu) / Lambda^3 exactly
# -----------------------------------------------------------------------------
def _ideal(rng):
    from regc import BulkSystem, LJCalculator
    return BulkSystem(3.0, LJCalculator(1.0, 0.0, 1.4))  # epsilon = 0: U = 0 always


def fig_ideal_gas():
    """Figure 3: <N> and P(N) of an ideal gas against the exact results."""
    from matplotlib.lines import Line2D
    from scipy.stats import poisson

    from regc import MBARSurfaceAnalysis, ReplicaExchangeGCMC
    T, mus = [1.0, 1.5], np.linspace(-3.0, 0.0, 6)
    with ReplicaExchangeGCMC(T, list(mus), _ideal, seed=7) as sim:
        sim.run(n_cycles=1500, sweeps_per_cycle=60, record_every=1, verbose=False)
    a = MBARSurfaceAnalysis(sim)
    V = 27.0
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.8))
    bmu = np.linspace(-3.4, 0.4, 80)
    ax.plot(bmu, V * np.exp(bmu), color="black", lw=1)
    for k, t in enumerate(T):
        col = LINE_COLORS[k]
        direct = [np.mean(sim.samples[(k, j)]["N"]) for j in range(len(mus))]
        ax.plot(mus / t, direct, "o", color=col, mfc="white", ms=5)
        ax.plot(bmu[::8], [a.mean_N(t, b * t)[0] for b in bmu[::8]], "x", color=col, ms=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\beta\mu$")
    ax.set_ylabel(r"$\langle N\rangle$")
    ax.set_title(r"$\langle N\rangle = V e^{\beta\mu}/\Lambda^3$ (reduced units, $\Lambda^3=1$)")
    handles = [Line2D([], [], color="black", lw=1, label="exact"),
               Line2D([], [], marker="o", mfc="white", color="0.3", ls="none",
                      label="replica averages"),
               Line2D([], [], marker="x", color="0.3", ls="none", label=r"MBAR, other $\mu$"),
               Line2D([], [], color=LINE_COLORS[0], lw=3, label="$T = 1.0$"),
               Line2D([], [], color=LINE_COLORS[1], lw=3, label="$T = 1.5$")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left")
    N = np.array(sim.samples[(0, 4)]["N"])
    lam = V * np.exp(mus[4] / T[0])
    k = np.arange(N.min(), N.max() + 1)
    pmf = poisson.pmf(k, lam)
    bx.bar(k, np.bincount(N, minlength=N.max() + 1)[k] / N.size, color=LINE_COLORS[0], alpha=0.55,
           width=0.85, label=f"sampled ({N.size} samples)")
    bx.plot(k, pmf, "o-", color="black", ms=3, lw=0.8, label=f"Poisson, mean {lam:.1f}")
    bx.set_xlabel("$N$")
    bx.set_ylabel("$P(N)$")
    bx.set_ylim(0, 1.45 * pmf.max())
    bx.set_title(f"$T = 1.0$, $\\mu = {mus[4]:.1f}$")
    bx.legend(frameon=False, loc="upper right", fontsize=7)
    cv = a.heat_capacity([T[0]], [mus[4]])[0, 0]
    print(f"ideal gas: C_V/k_B = {cv:.2f}, 15/4 <N> = {3.75 * a.mean_N(T[0], mus[4])[0]:.2f}")
    save(fig, "ideal_gas.png")


# -----------------------------------------------------------------------------
# 4. The convergence criterion: nested vs disjoint blocks
# -----------------------------------------------------------------------------
def fig_convergence():
    """Figure 4: the old nested and the new disjoint-block rho(u) statistics."""
    from regc import BulkSystem, LJCalculator, ReplicaExchangeGCMC
    sim = ReplicaExchangeGCMC([1.0], [-1.0], lambda rng: BulkSystem(4.0, LJCalculator()), seed=0)
    key = next(iter(sim.samples))
    rng = np.random.default_rng(1)
    n_block, n_checks, bins = 4000, 20, 10

    def curves(drift):
        t = np.arange(n_block * n_checks)
        U = rng.normal(size=t.size) + drift * t / t.size
        nested, disjoint = [], []
        for k in range(2, n_checks):
            n_prev = k * n_block
            u = U[:n_prev + n_block]
            edges = np.histogram_bin_edges(u, bins=30)
            p_all = np.histogram(u, edges)[0] / u.size
            p_prev = np.histogram(u[:n_prev], edges)[0] / n_prev
            nested.append(0.5 * np.abs(p_all - p_prev).sum())   # the old criterion
            sim.samples[key] = {"U": list(u), "N": [0] * u.size}
            disjoint.append(sim.rho_u_change(n_prev, bins=bins, n_start=n_prev - n_block))
        return np.arange(2, n_checks), np.array(nested), np.array(disjoint)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    series = [(0.0, "stationary"), (10.0, r"drifting ($+10\sigma$ over the run)")]
    for i, (drift, name) in enumerate(series):
        k, nested, disjoint = curves(drift)
        ax.plot(k, nested, ":", color=LINE_COLORS[i], lw=1.4,
                label=f"{name}: first $n$ vs all (old)")
        ax.plot(k, disjoint, "-", color=LINE_COLORS[i], lw=1.4,
                label=f"{name}: block $k$ vs $k-1$")
    ax.axhline(0.05, color="0.35", lw=0.8, ls="--")
    ax.text(n_checks - 1, 0.056, "tolerance 0.05", ha="right", va="bottom", fontsize=7, color="0.3")
    floor = np.sqrt(bins / (np.pi * n_block))
    ax.axhline(floor, color="0.55", lw=0.8, ls="-.")
    ax.text(n_checks - 1, floor * 0.92, r"noise floor $\sqrt{\mathrm{bins}/\pi n}$",
            ha="right", va="top", fontsize=7, color="0.4")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"check $k$ ({n_block} samples per block, {bins} bins)")
    ax.set_ylabel(r"total-variation change of $\rho(u)$")
    fig.legend(loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    save(fig, "convergence_criterion.png")


# -----------------------------------------------------------------------------
# 5. LJ surface run (examples/14): maps, isotherm, RDFs, structures
# -----------------------------------------------------------------------------
def _ex14():
    """The examples/14 module (system, reservoir, ladder)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ex14", os.path.join(HERE, "..", "..", "examples", "14_observables_heat_capacity.py"))
    ex14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex14)
    return ex14


def _analysis(run_dir):
    """MBAR analysis and sampled mu ladder of an examples/14 run (20 % discarded)."""
    from regc import MBARSurfaceAnalysis, load_samples
    s = load_samples(os.path.join(run_dir, "samples.npz"), discard=0.2)
    a = MBARSurfaceAnalysis(s, reservoir=_ex14().RESERVOIR, pressure_unit="atm")
    return a, np.unique(np.asarray(s.mu_grid)[0]), s.n_samples


def fig_cv_ladder(coarse_dir, dense_dir, min_ess=20.0):
    """Figure 7: C_V peaks on the sampled mu are an overlap artefact; a denser ladder removes them."""
    from regc import heat_capacity_ridges, kelvin_to_eV
    T_iso = kelvin_to_eV(290.0)
    (ac, lad_c, _), (ad, lad_d, nd) = _analysis(coarse_dir), _analysis(dense_dir)
    mu_line = np.linspace(lad_d[0], lad_d[-1], 201)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
    for k, (a, lad, name) in enumerate([(ac, lad_c, f"0.15 eV ladder"), (ad, lad_d, f"0.04 eV ladder")]):
        cv = a.heat_capacity([T_iso], mu_line)[0]
        ax.plot(mu_line, cv, color=LINE_COLORS[k], ls=LINE_STYLES[k], lw=1.3, label=name)
        inside = lad[(lad >= mu_line[0]) & (lad <= mu_line[-1])]
        ax.plot(inside, np.full(inside.size, 0.97 - 0.05 * k), "|", color=LINE_COLORS[k], ms=7,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel(r"$\mu_B$ (eV)")
    ax.set_ylabel("$C_V/k_B$")
    ax.set_title("290 K; ticks: sampled $\\mu_B$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.9))
    T_K = np.linspace(200, 650, 46)
    T = np.array([kelvin_to_eV(t) for t in T_K])
    mu = np.linspace(lad_d[0], lad_d[-1], 41)
    cvm = ad.heat_capacity(T, mu)
    ess = ad.effective_sample_size(T, mu)
    r = heat_capacity_ridges(T_K, mu, cvm, valid=ess >= min_ess, prominence=0.3, ladder=lad_d)
    norm = mpl.colors.LogNorm(vmin=np.percentile(cvm[cvm > 0], 2), vmax=cvm.max())
    mesh = bx.pcolormesh(mu, T_K, cvm, shading="auto", cmap="magma", norm=norm)
    cb = fig.colorbar(mesh, ax=bx, label="$C_V/k_B$", extend="min")
    cb.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    cb.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False))
    pts, on = r["points"], r["on_ladder"]
    bx.plot(pts[~on, 1], pts[~on, 0], "o", ms=2.5, color="white", mec="black", mew=0.4, ls="none",
            label="$C_V$ peak")
    bx.plot(pts[on, 1], pts[on, 0], "x", ms=3, color="0.6", ls="none", label="peak on a sampled $\\mu_B$")
    bx.set_xlabel(r"$\mu_B$ (eV)")
    bx.set_ylabel("$T$ (K)")
    bx.set_title(f"0.04 eV ladder ({nd} samples per replica)")
    fig.legend(*bx.get_legend_handles_labels(), loc="outside lower right", ncol=2, fontsize=7,
               frameon=False)
    save(fig, "lj_cv_ladder.png")
    print(f"dense run: {len(pts)} C_V peaks, {on.mean():.0%} on the ladder; min ESS {ess.min():.0f}")


def fig_lj_surface(run_dir, min_ess=20.0, iso_dir=None):
    """Figures 5-12 from the samples and configurations of an examples/14 run."""
    import importlib.util

    from regc import (MBARSurfaceAnalysis, export_phase_configurations, heat_capacity_ridges,
                      kelvin_to_eV, load_samples, plot_map)
    spec = importlib.util.spec_from_file_location(
        "ex14", os.path.join(HERE, "..", "..", "examples", "14_observables_heat_capacity.py"))
    ex14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex14)
    s = load_samples(os.path.join(run_dir, "samples.npz"), discard=0.2)
    a = MBARSurfaceAnalysis(s, reservoir=ex14.RESERVOIR, pressure_unit="atm")
    print(f"LJ surface: {s.n_samples} samples per state, {len(s.samples)} states")
    T_K = np.linspace(200, 650, 46)
    T = np.array([kelvin_to_eV(t) for t in T_K])
    mu = np.linspace(ex14.MUS_EV[0], ex14.MUS_EV[-1], 61)
    ess = a.effective_sample_size(T, mu)
    valid = ess >= min_ess
    most = a.most_probable_N(T, mu)
    cv = a.heat_capacity(T, mu)
    ladder = np.unique(np.asarray(s.mu_grid)[0])
    ridges = heat_capacity_ridges(T_K, mu, cv, valid=valid, prominence=0.3, ladder=ladder)
    print(f"coarse run: {len(ridges['points'])} C_V peaks, "
          f"{ridges['on_ladder'].mean():.0%} on the sampled mu")
    kw = dict(log_x=False, xlabel=r"$\mu_B$ (eV)", ylabel="$T$ (K)", figsize=(6.0, 3.9),
              invalid_label=f"fewer than {min_ess:g} effective samples")
    os.makedirs(FIG, exist_ok=True)
    plot_map(T_K, mu, most, os.path.join(FIG, "lj_most_probable.png"), valid=valid,
             discrete=True, colorbar_label="most probable $N_B$", title="Stable phase (MBAR)",
             **kw)
    plot_map(T_K, mu, cv, os.path.join(FIG, "lj_heat_capacity.png"), valid=valid,
             points=ridges["points"], log_color=True, cmap="magma", colorbar_label="$C_V/k_B$",
             title="Heat capacity, Eq. (9)", **kw)
    cn = a.mean_coordination("cn_BA", T, mu)
    plot_map(T_K, mu, cn, os.path.join(FIG, "lj_coordination.png"), valid=valid & np.isfinite(cn),
             cmap="cividis", colorbar_label="A neighbours per B atom ($r<2.8$ Å)",
             title="Mean B–A coordination",
             **dict(kw, invalid_label=f"no B atoms, or fewer than {min_ess:g} effective samples"))
    plot_map(T_K, mu, ess, os.path.join(FIG, "lj_ess.png"), log_color=True, cmap="viridis",
             vmin=1.0, contour_levels=[min_ess], contour_label=f"{min_ess:g} effective samples",
             colorbar_label="Kish effective samples", title="Effective sample size, Eq. (8)", **kw)

    # one isotherm: <N> and C_V stacked on a shared mu axis, P(N) beside them
    T_iso_K = 290.0
    T_iso = kelvin_to_eV(T_iso_K)
    a_iso = a
    mu_line = np.linspace(-2.1, -1.0, 221)
    if iso_dir is not None:  # the denser ladder: better overlap for P(N) across a step
        a, lad_d, _ = _analysis(iso_dir)
        mu_line = np.linspace(lad_d[0], lad_d[-1], 201)
    nbar = a.phase_diagram([T_iso], mu_line)[0]
    cv_line = a.heat_capacity([T_iso], mu_line)[0]
    window = (mu_line > -1.7) & (mu_line < -1.3)
    peak = float(mu_line[window][np.argmax(cv_line[window])])
    picks = [round(peak + d, 3) for d in (-0.06, -0.01, 0.01, 0.06)]
    fig, ax = plt.subplot_mosaic([["N", "P"], ["C", "P"]], figsize=(6.3, 3.6))
    ax["C"].sharex(ax["N"])
    ax["N"].plot(mu_line, nbar, color="black", lw=1.2)
    ax["N"].set_ylabel(r"$\langle N_B\rangle$")
    ax["N"].tick_params(labelbottom=False)
    ax["N"].set_title(f"Isotherm at {T_iso_K:.0f} K")
    ax["C"].plot(mu_line, cv_line, color="black", lw=1.0)
    ax["C"].set_ylabel("$C_V/k_B$")
    ax["C"].set_xlabel(r"$\mu_B$ (eV)")
    for k, m in enumerate(picks):
        for name in ("N", "C"):
            ax[name].axvline(m, color=LINE_COLORS[k], ls=LINE_STYLES[k], lw=1.0)
        c, p = a.pN(T_iso, m)
        ax["P"].step(c, p, where="mid", color=LINE_COLORS[k], ls=LINE_STYLES[k], lw=1.3,
                     label=rf"$\mu_B = {num(m, '.3f')}$ eV")
    lo, hi = int(a.mean_N(T_iso, picks[0])[0]) - 6, int(a.mean_N(T_iso, picks[-1])[0]) + 6
    ax["P"].set_xlim(max(lo, 0), hi)
    ax["P"].set_xlabel("$N_B$")
    ax["P"].set_ylabel("$P(N_B)$ (MBAR)")
    ax["P"].set_title(r"$P(N_B)$ at the marked $\mu_B$")
    ax["P"].legend(frameon=False, fontsize=7.5)
    save(fig, "lj_isotherm.png")
    print(f"isotherm: C_V peak at {peak:.3f} eV; <N> {a.mean_N(T_iso, picks[0])[0]:.1f} -> "
          f"{a.mean_N(T_iso, picks[-1])[0]:.1f}")
    a = a_iso

    # RDFs along the isotherm: colour + line style, x from the first shell
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharex=True)
    for k, m in enumerate([-1.75, -1.4, -1.0]):
        n = a.mean_N(T_iso, m)[0]
        for ax_, name in zip(axes, ("rdf_BA", "rdf_BB")):
            r, g = a.radial_distribution(name, T_iso, m)
            ax_.plot(r, g, color=LINE_COLORS[k], ls=LINE_STYLES[k], lw=1.3,
                     label=rf"$\mu_B = {num(m)}$ eV, $\langle N_B\rangle = {n:.1f}$")
    axes[0].set_title("B–A: A neighbours of a B atom")
    axes[1].set_title("B–B: B neighbours of a B atom")
    for ax_ in axes:
        ax_.set_xlabel("$r$ (Å)")
        ax_.set_xlim(0.9, 5.0)
        ax_.set_ylim(bottom=0)
    axes[0].set_ylabel(r"$n(r)$ (Å$^{-1}$)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3,
               frameon=False, fontsize=7.5)
    save(fig, "lj_rdf.png")

    # structures of the phases: export, then draw four of them
    entries = export_phase_configurations(
        a, os.path.join(run_dir, "configurations"), T, mu, os.path.join(FIG, "lj_phases.extxyz"),
        min_ess=min_ess, temperature_K=True, min_cells=6,
        tile_path=os.path.join(FIG, "lj_phases_tiled.extxyz"))
    _draw_phases(entries, os.path.join(FIG, "lj_phases.extxyz"))
    i_max = np.unravel_index(ess.argmax(), ess.shape)
    print(f"{len(entries)} phases exported; C_V peaks: {len(ridges['points'])}; ESS min "
          f"{ess.min():.0f}, max {ess.max():.0f} at {T_K[i_max[0]]:.0f} K, mu = {mu[i_max[1]]:.2f}")


def _draw_phases(entries, path):
    """Figure 11: top and side views of four exported phase structures."""
    from ase import Atoms
    from ase.visualize.plot import plot_atoms

    from regc.trajectory import read_extxyz_frames
    frames = list(read_extxyz_frames(path))
    by_n = {}
    for e, fr in zip(entries, frames):
        if e["phase_N"] not in by_n or e["P_phase"] > by_n[e["phase_N"]][0]["P_phase"]:
            by_n[e["phase_N"]] = (e, fr)
    wanted = [n for n in sorted(by_n) if n > 0]
    if len(wanted) > 4:
        wanted = [wanted[int(round(i))] for i in np.linspace(0, len(wanted) - 1, 4)]
    fig, axes = plt.subplots(2, len(wanted), figsize=(1.6 * len(wanted), 2.9),
                             gridspec_kw={"height_ratios": [1.6, 0.6], "hspace": 0.02},
                             layout=None)
    axes = np.atleast_2d(axes).reshape(2, -1)
    for col, n in enumerate(wanted):
        e, fr = by_n[n]
        atoms = Atoms(symbols=["Cu" if s == "A" else "O" for s in fr["species"]],
                      positions=fr["positions"], cell=fr["lattice"], pbc=fr["pbc"])
        atoms.wrap()
        plot_atoms(atoms, axes[0, col], rotation="0x,0y,0z", radii=0.6)
        plot_atoms(atoms, axes[1, col], rotation="-90x,0y,0z", radii=0.6, show_unit_cell=0)
        axes[0, col].set_title(f"$N_B = {n}$\n{e['T_K']:.0f} K, $\\mu_B = {num(e['mu'], '.2f')}$ eV"
                               f"\n$P(N_B) = {e['P_phase']:.2f}$", fontsize=7.5)
    for r, label in enumerate(("top view", "side view")):
        axes[r, 0].text(-0.06, 0.5, label, rotation=90, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
    save(fig, "lj_phases.png")


def fig_reliability(coarse_dir, dense_dir, n_boot=60):
    """Figure: MBAR neighbour overlap per replica, and C_V at 290 K with bootstrap bands."""
    from regc import heat_capacity_ridges, kelvin_to_eV
    from regc.diagnostics import (OVERLAP_THRESHOLD, bootstrap, neighbour_overlap,
                                  replica_inefficiencies, ridge_support)
    runs = []
    for name, d in (("0.15 eV ladder", coarse_dir), ("0.04 eV ladder", dense_dir)):
        a, lad, n = _analysis(d)
        ov = neighbour_overlap(a)
        n_ind = np.array([v["n_independent"] for v in replica_inefficiencies(a._source).values()])
        runs.append((name, a, lad, ov, n_ind))
        print(f"{name}: {len(ov['gaps'])}/{len(ov['pairs'])} pairs below {OVERLAP_THRESHOLD}; "
              f"independent samples per replica median {np.median(n_ind):.1f}")
    fig = plt.figure(figsize=(6.6, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35])
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1.0)
    T_ladder = np.asarray(_ex14().TEMPERATURES_K)
    for r, (name, a, lad, ov, _) in enumerate(runs):
        ax = fig.add_subplot(gs[r, 0])
        m = np.clip(ov["min_map"], 1e-4, 1.0)
        mesh = ax.pcolormesh(np.arange(len(lad)), np.arange(len(T_ladder)), m, cmap="viridis",
                             norm=norm, shading="nearest")
        gap = ov["min_map"] < OVERLAP_THRESHOLD
        ax.contourf(np.arange(len(lad)), np.arange(len(T_ladder)), gap.astype(float),
                    levels=[0.5, 1.5], colors="none", hatches=["////"])
        ax.set_yticks(range(len(T_ladder)))
        ax.set_yticklabels([f"{t:.0f}" for t in T_ladder], fontsize=6.5)
        ax.set_xticks(range(0, len(lad), 2))
        ax.set_xticklabels([num(v, ".2f") for v in lad[::2]], fontsize=6.5, rotation=45)
        ax.set_ylabel("$T$ (K)", fontsize=8)
        ax.set_title(f"{name}: {len(ov['gaps'])}/{len(ov['pairs'])} pairs < {OVERLAP_THRESHOLD}",
                     fontsize=8)
        if r == 1:
            ax.set_xlabel(r"sampled $\mu_B$ (eV)", fontsize=8)
    cb = fig.colorbar(mesh, ax=fig.axes[:2], fraction=0.08, pad=0.02,
                      label="smallest neighbour overlap")
    cb.ax.axhline(OVERLAP_THRESHOLD, color="white", lw=1.2)
    bx = fig.add_subplot(gs[:, 1])
    T_iso = kelvin_to_eV(290.0)
    lad_d = runs[1][2]
    mu_line = np.linspace(lad_d[0], lad_d[-1], 121)
    for k, (name, a, lad, _, _) in enumerate(runs):
        b = bootstrap(a, lambda an: an.heat_capacity([T_iso], mu_line)[0], n_boot=n_boot, seed=k)
        lo, hi = b.interval(0.90)
        bx.fill_between(mu_line, lo, hi, color=LINE_COLORS[k], alpha=0.25, lw=0)
        bx.plot(mu_line, b.estimate, color=LINE_COLORS[k], ls=LINE_STYLES[k], lw=1.3,
                label=f"{name} (90 % band)")
        r = heat_capacity_ridges([290.0], mu_line, b.estimate[None], prominence=0.3, ladder=lad)
        sup = ridge_support(r, b.replicates[:, None, :])
        for (_, m), sp, on in zip(r["points"], sup, r["on_ladder"]):
            y = b.estimate[np.argmin(np.abs(mu_line - m))]
            bx.plot(m, y, "v" if on else "o", color=LINE_COLORS[k], mec="black", mew=0.4, ms=4.5)
            bx.annotate(f"{sp:.2f}", (m, y), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=6.5, color=LINE_COLORS[k])
            print(f"  {name}: peak at {m:+.3f} eV, support {sp:.2f}, on ladder {bool(on)}")
    bx.set_xlabel(r"$\mu_B$ (eV)")
    bx.set_ylabel("$C_V/k_B$ at 290 K")
    bx.set_ylim(bottom=0)
    from matplotlib.lines import Line2D
    handles, labels = bx.get_legend_handles_labels()
    handles += [Line2D([], [], marker="v", color="0.5", mec="black", ls="none"),
                Line2D([], [], marker="o", color="0.5", mec="black", ls="none")]
    labels += ["peak on a sampled $\\mu_B$", "peak between samples"]
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7, frameon=False)
    bx.set_title("numbers: bootstrap support of each peak", fontsize=8)
    save(fig, "lj_reliability.png")


def fig_reweighting(npz_path, min_ess=20.0):
    """Figure: Fe(100)-H, the EAM run reweighted to EAM' against a direct EAM' run."""
    d = np.load(npz_path)
    T_K, p = d["T_K"], d["p_bar"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axs[0]
    for k, t in enumerate((T_K[0], T_K[-1])):
        i = int(np.argmin(np.abs(T_K - t)))
        ax.plot(p, d["theta_A"][i], color="0.55", ls=":", lw=1.1,
                label="EAM (sampled)" if k == 0 else None)
        ax.fill_between(p, d["lo_Bw"][i], d["hi_Bw"][i], color=LINE_COLORS[k], alpha=0.25, lw=0)
        ax.plot(p, d["theta_Bw"][i], color=LINE_COLORS[k], ls="--", lw=1.3,
                label="EAM′ reweighted" if k == 0 else None)
        ax.plot(p, d["theta_Bd"][i], color=LINE_COLORS[k], ls="-", lw=0.9,
                label="EAM′ direct run" if k == 0 else None)
        ax.text(p[-1] * 1.4, d["theta_Bd"][i][-1], f"{t:.0f} K", fontsize=7, va="center",
                color=LINE_COLORS[k])
    ax.set_xscale("log")
    ax.set_xlabel("$p_{H_2}$ (bar)")
    ax.set_ylabel("H per surface Fe")
    ax.set_title("(a) coverage")
    ax.fill_between([], [], [], color="0.6", alpha=0.3, lw=0, label="90 % bootstrap band (reweighted)")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False,
               fontsize=7.5)
    sig = np.sqrt(d["std_Bw"] ** 2 + d["std_Bd"] ** 2)
    z = (d["theta_Bw"] - d["theta_Bd"]) / np.where(sig > 0, sig, np.nan)
    bad = d["ess_Bw"] < min_ess
    zz = np.ma.masked_where(bad, z)
    m = axs[1].pcolormesh(p, T_K, zz, shading="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    if bad.any():
        axs[1].contourf(p, T_K, bad.astype(float), levels=[0.5, 1.5], colors="none",
                        hatches=["////"])
    fig.colorbar(m, ax=axs[1], label="difference / combined σ", extend="both")
    axs[1].set_title("(b) reweighted − direct")
    ratio = d["ess_Bw"] / d["ess_A"]
    m2 = axs[2].pcolormesh(p, T_K, ratio, shading="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(m2, ax=axs[2], label="ESS(EAM′) / ESS(EAM)")
    axs[2].set_title("(c) overlap A → EAM′")
    for a in axs[1:]:
        a.set_xscale("log")
        a.set_xlabel("$p_{H_2}$ (bar)")
        a.set_ylabel("$T$ (K)")
    save(fig, "feh_reweighting.png")
    ok = ~bad
    print(f"|z| < 2 in {np.mean(np.abs(z[ok]) < 2):.0%} of cells with ESS >= {min_ess:g}; "
          f"ESS ratio median {np.median(ratio):.2f}")


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run-dir", default=None, help="output directory of examples/14")
    ap.add_argument("--dense-run-dir", default=None,
                    help="examples/14 run with a denser mu ladder (--mu-min -1.65 --mu-max -1.25)")
    ap.add_argument("--reweight-npz", default=None,
                    help="compare.npz of the Fe-H EAM -> EAM' reweighting benchmark")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset: reservoirs swap ideal convergence lj reliability")
    args = ap.parse_args()
    todo = args.only or ["reservoirs", "swap", "ideal", "convergence", "lj"]
    if args.reweight_npz:
        fig_reweighting(args.reweight_npz)
    if "reservoirs" in todo:
        fig_reservoirs()
    if "swap" in todo:
        fig_swap_patterns()
    if "ideal" in todo:
        fig_ideal_gas()
    if "convergence" in todo:
        fig_convergence()
    if "reliability" in todo and args.dense_run_dir and "lj" not in todo:
        fig_reliability(args.run_dir, args.dense_run_dir)
    if "lj" in todo:
        if args.run_dir and os.path.isfile(os.path.join(args.run_dir, "samples.npz")):
            fig_lj_surface(args.run_dir, iso_dir=args.dense_run_dir)
            if args.dense_run_dir:
                fig_cv_ladder(args.run_dir, args.dense_run_dir)
                fig_reliability(args.run_dir, args.dense_run_dir)
        else:
            print("skipping the LJ-surface figures: pass --run-dir of examples/14")


if __name__ == "__main__":
    main()
