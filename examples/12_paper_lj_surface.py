"""
Reproduce the Lennard-Jones surface of Zhou, Scheffler & Ghiringhelli,
PRB 100, 174106 (2019), Sec. III A, with the [ZSG] REGC scheme.

System (paper Sec. III A and Appendix 1):
  * A18: frozen two-layer fcc(111) slab, hexagonal 3x3 surface cell with
    lattice-vector length 11.489 A (A-A nearest neighbours 3.830 A); no A-A
    interaction.
  * B gas: LJ with eps_AB = 0.66 eV, sigma_AB = 1.91 A, eps_BB = 0.01 eV,
    sigma_BB = 1.2 A (cutoff 2.5 sigma, shifted -- the paper gives none).
  * System volume V = the "surface zone": 48 A above the topmost A atoms,
    all x, y. B lives only there (the top A layer is put at z = 0, the
    box's hard wall). Letting B leave the zone is wrong for this system:
    B (sigma_AB = 1.91 A) slips through the hollows of the top layer and
    most of it ends up absorbed between the two A layers.
  * 10 temperatures 200..650 K (step 50) x 16 chemical potentials
    -2.4..-0.9 eV (step 0.1) = 160 replicas; x0 = 0.99; ten MC moves after
    each GC/RE attempt; 1.2e5 REGC steps; samples every 100 steps.
  * mu reference: atomic ideal gas, beta mu = ln(beta p Lambda^3). The mass
    of B is not stated in the paper; m_B = 16.48 amu is what reproduces the
    Fig. 3 caption's "(200 K, -0.9 eV) is exactly 2.03e-17 atm".
  * MC displacement amplitude: not stated; 0.5 A here.

What the paper reports, and what this script checks:
  * RE acceptance "about 25%".
  * MBAR@REGC phase diagram (Fig. 3a): stable phases include N_B = 0
    (pristine), 18 (first monolayer, all hollow sites), 45 (second
    monolayer), 59 (thicker second layer), 85 (third layer).
  * aiAT@REGC (Fig. 3b, lowest sampled energy per N_B, Eq. (13) with
    vibrations neglected) coincides with MBAR@REGC up to N_B = 18 and
    differs above it (higher coverages shifted to higher T / lower p).

Usage:
    python 12_paper_lj_surface.py                     # full paper run (~80 min on 12 cores)
    python 12_paper_lj_surface.py --steps 20000       # shorter run
    python 12_paper_lj_surface.py --resume            # continue an interrupted run
    python 12_paper_lj_surface.py --analyze-only      # re-analyze saved samples

Requires: pip install ase pymbar matplotlib
"""

import argparse
import os

import numpy as np
from ase.build import fcc111

from gcmc_re import (MBARSurfaceAnalysis, MonatomicIdealGasReservoir, PairLJCalculator,
                     ReplicaExchangeGCMC, kelvin_to_eV, load_samples, run_with_checkpoints,
                     slab_geometry_from_atoms)
from gcmc_re.io import load_pickle

LATTICE_VECTOR = 11.489                      # A, 3x3 hexagonal surface cell
PAIR_PARAMS = {("A", "B"): (0.66, 1.91), ("B", "B"): (0.01, 1.2)}   # (eps eV, sigma A)
ZONE_HEIGHT = 48.0
TEMPERATURES_K = np.arange(200.0, 651.0, 50.0)
MUS_EV = np.round(np.arange(-2.4, -0.85, 0.1), 2)
PAPER_PHASES = (0, 18, 45, 59, 85)
RESERVOIR = MonatomicIdealGasReservoir(mass_amu=16.48)   # B gas; mu <-> p, Lambda^3

# A18: all layers frozen, top layer at z = 0, so the box (0, 48] IS the paper's zone
_slab = fcc111("Cu", size=(3, 3, 2), a=LATTICE_VECTOR / 3 * np.sqrt(2), vacuum=None, orthogonal=False)
GEOMETRY = slab_geometry_from_atoms(_slab, n_frozen_layers=None, headroom=ZONE_HEIGHT)
GEOMETRY.frozen_species = np.array(["A"] * len(GEOMETRY.frozen_positions))


def make_system(rng: np.random.Generator):
    """One replica's system: the A18 slab with its own calculator."""
    return GEOMETRY.make_system(PairLJCalculator(PAIR_PARAMS), exchange_species="B")


def simulate(args) -> None:
    """Run (or resume) the paper's protocol with checkpoints every --chunk steps."""
    T_eV = [kelvin_to_eV(T) for T in TEMPERATURES_K]
    with ReplicaExchangeGCMC(T_eV, list(MUS_EV), make_system, seed=args.seed, max_disp=0.5,
                             n_workers=args.workers,
                             lambda_cubed_of_T=RESERVOIR.lambda_cubed_of_T()) as sim:
        run_with_checkpoints(sim, args.out, n_steps=args.steps, chunk=args.chunk, x0=0.99,
                             diffusion="mc_moves", diffusion_steps=10, record_every=100,
                             resume=args.resume, overwrite=not args.resume,
                             metadata={"system": "ZSG LJ surface", "mass_B_amu": RESERVOIR.mass_amu},
                             progress=lambda line: print(line, flush=True))


def swap_acceptance(out: str) -> str:
    """Swap acceptance from the checkpoint (or an older run's stats.json)."""
    ckpt = os.path.join(out, "checkpoint.pkl")
    if os.path.isfile(ckpt):
        stats = load_pickle(ckpt)["swap_stats"]
        acc = {t: a / max(1, n) for t, (n, a) in stats.items()}
    elif os.path.isfile(os.path.join(out, "stats.json")):
        import json
        with open(os.path.join(out, "stats.json")) as f:
            acc = json.load(f)["swap_acceptance"]
    else:
        return "unknown"
    return ", ".join(f"{t} {a:.2f}" for t, a in acc.items())


def analyze(args) -> None:
    """MBAR@REGC vs aiAT@REGC phase diagrams, in (T, mu) and (T, p) (cf. [ZSG] Fig. 3)."""
    samples = load_samples(os.path.join(args.out, "samples.npz"), discard=args.discard,
                           temperatures=[kelvin_to_eV(T) for T in TEMPERATURES_K], mu_grid=MUS_EV,
                           lambda_cubed_of_T=RESERVOIR.lambda_cubed_of_T())
    print(f"{samples.n_samples} samples per state (first {args.discard:.0%} discarded); "
          f"swap acceptance: {swap_acceptance(args.out)}   (paper: about 25%)")
    analysis = MBARSurfaceAnalysis(samples, reservoir=RESERVOIR, pressure_unit="atm")
    T_fine_K = np.linspace(200.0, 650.0, 46)
    mu_fine = np.linspace(-2.4, -0.9, 61)
    mbar_map = analysis.most_probable_N(np.array([kelvin_to_eV(T) for T in T_fine_K]), mu_fine)
    aiat_map = np.broadcast_to(analysis.lowest_energy_N(mu_fine), mbar_map.shape)

    def stability_table(m, label):
        vals, counts = np.unique(m, return_counts=True)
        order = np.argsort(-counts)
        print(f"\n{label}: stable N_B and share of the (T, mu) window")
        print("  " + ", ".join(f"{vals[i]} ({counts[i] / m.size:.1%})" for i in order[:15]))
        return set(vals.tolist())

    s_mbar = stability_table(mbar_map, "MBAR@REGC")
    s_aiat = stability_table(aiat_map, "aiAT@REGC")
    print("\nPaper's highlighted phases N_B = 0, 18, 45, 59, 85:")
    for n in PAPER_PHASES:
        print(f"  N_B = {n:2d}: MBAR {'yes' if n in s_mbar else 'no '}   "
              f"aiAT {'yes' if n in s_aiat else 'no '}")
    low = (mbar_map <= 18) | (aiat_map <= 18)

    def agreement(mask):
        return f"{np.mean(mbar_map[mask] == aiat_map[mask]):.1%}" if mask.any() else "n/a"

    print(f"\nWhere either diagram has N_B <= 18, they agree on {agreement(low)} of the points "
          f"(paper: 'almost coincide'); above 18 they agree on {agreement(~low)}.")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plot_Tmu(plt, mbar_map, aiat_map, T_fine_K, mu_fine, args.out)
    plot_Tp(plt, analysis, args.out)


def plot_Tmu(plt, mbar_map, aiat_map, T_K, mu, out_dir) -> None:
    """Most stable N_B on the sampled (T, mu) window, with B-gas isobars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    vmax = max(mbar_map.max(), aiat_map.max())
    logp = np.log10([[RESERVOIR.pressure_pascal(T, m) / 101325.0 for m in mu] for T in T_K])
    for ax, m, title in zip(axes, (mbar_map, aiat_map), ("MBAR@REGC", "aiAT@REGC")):
        mesh = ax.pcolormesh(mu, T_K, m, shading="nearest", cmap="nipy_spectral", vmin=0, vmax=vmax)
        cs = ax.contour(mu, T_K, logp, levels=np.arange(-40, 11, 5), colors="w", linewidths=0.6)
        ax.clabel(cs, fmt=lambda v: f"1e{int(v)} atm", fontsize=7)
        ax.set_title(f"{title}: most stable N_B")
        ax.set_xlabel("mu (eV)")
    axes[0].set_ylabel("T (K)")
    fig.colorbar(mesh, ax=axes, label="N_B")
    path = os.path.join(out_dir, "paper_lj_phase_diagrams.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path} (white lines: B-gas isobars)")


def plot_Tp(plt, analysis, out_dir) -> None:
    """[ZSG] Fig. 3: most stable N_B on a (T, p) plane, MBAR (a) and aiAT (b)."""
    T_K = np.linspace(50.0, 650.0, 61)
    logp = np.linspace(-20.0, 5.0, 101)
    T_eV = np.array([kelvin_to_eV(T) for T in T_K])
    mbar_map = analysis.most_probable_N_Tp(T_eV, 10.0 ** logp)
    aiat_map = analysis.lowest_energy_N_Tp(T_eV, 10.0 ** logp)
    mu = np.array([[analysis.mu_from_p(t, 10.0 ** lp) for lp in logp] for t in T_eV])
    # outside the sampled (T, mu) window MBAR extrapolates: hatch it
    outside = (T_K[:, None] < TEMPERATURES_K.min()) | (mu < MUS_EV.min()) | (mu > MUS_EV.max())
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 10), sharex=True)
    vmax = max(85, mbar_map.max())
    for ax, m, label in zip(axes, (mbar_map, aiat_map), ("(a) MBAR@REGC", "(b) aiAT@REGC")):
        mesh = ax.pcolormesh(T_K, logp, m.T, shading="nearest", cmap="gnuplot", vmin=0, vmax=vmax)
        ax.contourf(T_K, logp, outside.T.astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["///"])
        ax.contour(T_K, logp, outside.T.astype(float), levels=[0.5], colors="w", linewidths=0.8)
        for (T, mu_c), c in (((650, -0.9), "cyan"), ((650, -2.4), "lime"), ((200, -0.9), "deeppink")):
            ax.plot(T, np.log10(RESERVOIR.pressure_pascal(T, mu_c) / 101325.0), "*", ms=14,
                    color=c, mec="k")
        ax.set_ylim(-20, 5)
        ax.set_ylabel("log10 pressure (atm)")
        ax.set_title(label, loc="left")
        fig.colorbar(mesh, ax=ax, label="N_B")
    axes[1].set_xlabel("Temperature (K)")
    fig.suptitle("Most stable N_B; hatched = outside sampled (T, mu) window")
    path = os.path.join(out_dir, "paper_lj_Tp_phase_diagrams.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path} (compare [ZSG] Fig. 3)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue the run saved in --out")
    ap.add_argument("--discard", type=float, default=0.1,
                    help="fraction of samples discarded as equilibration (default 0.1)")
    ap.add_argument("--out", default="output_12_paper_lj")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        simulate(args)
    analyze(args)
