"""
Structural observables, the heat capacity and the structures behind a phase
diagram, on the Lennard-Jones surface of [ZSG] PRB 100, 174106, Sec. III A
(the system of examples/12_paper_lj_surface.py).

What it shows:

* Observables recorded with every sample and reweighted by MBAR like N
  (`regc.observables`): the mean B-A coordination (how many substrate
  atoms an adsorbate touches: 3 in a hollow, fewer in the second layer),
  the distribution of B-B coordination numbers, and the B-B and B-A
  radial distributions.
* The heat capacity C_V(T, p) = (<E^2> - <E>^2) / k_B T^2 of [ZSG] PRL
  128, 246101 (2022), Eq. (1), with its ridges (peaks along T at fixed p
  and along p at fixed T) overlaid on the most-probable-N phase map: a
  change of N across a ridge is a first-order transition, a change without
  one a smooth crossover.
* The configurations behind the diagram: the run stores every sample's
  structure (``configurations_dir``), and `export_phase_configurations`
  writes, for each phase region, the most probable recorded structure with
  that N (``phases.extxyz``, one frame per phase) and all of them side by
  side (``phases_tiled.extxyz``, a visual phase diagram), ready for OVITO,
  ASE, VMD or pySNOW. The comment line of each frame gives the state, the
  phase, its probability there and where the sample came from.

Cells reached by reweighting far from any sampled state (few effective
samples, `MBARSurfaceAnalysis.effective_sample_size`) are drawn grey and
skipped by the export and the ridge search.

Usage:
    python 14_observables_heat_capacity.py                   # ~15 min on 8 cores
    python 14_observables_heat_capacity.py --steps 2000      # quick look
    python 14_observables_heat_capacity.py --analyze-only    # re-analyse saved samples

Requires: pip install ase pymbar scipy matplotlib
"""

import argparse
import os

import numpy as np
from ase.build import fcc111

from regc.diagnostics import (OVERLAP_THRESHOLD, bootstrap, neighbour_overlap,
                              replica_inefficiencies, ridge_support)
from regc import (CoordinationHistogram, CoordinationNumber, MBARSurfaceAnalysis,
                  MonatomicIdealGasReservoir, PairLJCalculator, RadialDistribution,
                  ReplicaExchangeGCMC, export_phase_configurations, heat_capacity_ridges,
                  kelvin_to_eV, load_samples, plot_map, plot_radial_distribution,
                  run_with_checkpoints, slab_geometry_from_atoms)

LATTICE_VECTOR = 11.489
PAIR_PARAMS = {("A", "B"): (0.66, 1.91), ("B", "B"): (0.01, 1.2)}
RESERVOIR = MonatomicIdealGasReservoir(mass_amu=16.48)
TEMPERATURES_K = np.array([200.0, 290.0, 380.0, 470.0, 560.0, 650.0])
MUS_EV = np.round(np.arange(-2.4, -0.85, 0.15), 2)

_slab = fcc111("Cu", size=(3, 3, 2), a=LATTICE_VECTOR / 3 * np.sqrt(2), vacuum=None, orthogonal=False)
GEOMETRY = slab_geometry_from_atoms(_slab, n_frozen_layers=None, headroom=48.0)
GEOMETRY.frozen_species = np.array(["A"] * len(GEOMETRY.frozen_positions))

# A-B pair minimum at 2^(1/6) * 1.91 = 2.14 A; B-B at 1.35 A
OBSERVABLES = [
    CoordinationNumber("cn_BA", r_cut=2.8, center="mobile", neighbor="substrate"),
    CoordinationHistogram("cn_BB", r_cut=1.8, max_cn=8, center="mobile", neighbor="mobile"),
    RadialDistribution("rdf_BB", r_max=4.9, n_bins=98, a="mobile", b="mobile"),
    RadialDistribution("rdf_BA", r_max=4.9, n_bins=98, a="mobile", b="substrate"),
]


def mu_ladder(args):
    """The sampled chemical potentials: MUS_EV, or --mu-min/--mu-max/--mu-step."""
    if args.mu_min is None:
        return MUS_EV
    return np.round(np.arange(args.mu_min, args.mu_max + 0.5 * args.mu_step, args.mu_step), 4)


def make_system(rng):
    """One replica: the frozen A18 slab with B gas in the 48 A zone above it."""
    return GEOMETRY.make_system(PairLJCalculator(PAIR_PARAMS), exchange_species="B")


def simulate(args):
    """REGC with observables recorded and every sample's configuration stored."""
    T_eV = [kelvin_to_eV(T) for T in TEMPERATURES_K]
    with ReplicaExchangeGCMC(T_eV, list(mu_ladder(args)), make_system, seed=args.seed, max_disp=0.5,
                             n_workers=args.workers, observables=OBSERVABLES,
                             configurations_dir=os.path.join(args.out, "configurations"),
                             lambda_cubed_of_T=RESERVOIR.lambda_cubed_of_T()) as sim:
        run_with_checkpoints(sim, args.out, n_steps=args.steps, chunk=args.chunk, x0=0.99,
                             diffusion="mc_moves", diffusion_steps=10, record_every=50,
                             resume=args.resume, overwrite=not args.resume,
                             metadata={"system": "ZSG LJ surface, example 14"},
                             progress=lambda line: print(line, flush=True))


def analyze(args):
    """Maps of N, <CN>, C_V with its ridges; RDFs; the structures of every phase."""
    samples = load_samples(os.path.join(args.out, "samples.npz"), discard=args.discard)
    a = MBARSurfaceAnalysis(samples, reservoir=RESERVOIR, pressure_unit="atm")
    T_K = np.linspace(TEMPERATURES_K[0], TEMPERATURES_K[-1], 46)
    T = np.array([kelvin_to_eV(t) for t in T_K])
    # (T, mu) maps: the ladder covers every cell (a (T, p) window would reach outside it,
    # the pressure of a given mu changing by decades with T)
    ladder = mu_ladder(args)
    mu = np.linspace(ladder[0], ladder[-1], 61)
    ess = a.effective_sample_size(T, mu)
    valid = ess >= args.min_ess
    print(f"{samples.n_samples} samples per state; effective samples on the mesh: "
          f"min {ess.min():.0f}, median {np.median(ess):.0f}; "
          f"{(~valid).mean():.0%} of cells below {args.min_ess:g}")
    # how far to trust the maps: correlation and overlap (regc.diagnostics)
    ineff = replica_inefficiencies(samples)
    n_ind = np.array([v["n_independent"] for v in ineff.values()])
    ov = neighbour_overlap(a)
    print(f"independent samples per replica: min {n_ind.min():.0f}, median {np.median(n_ind):.0f}; "
          f"{len(ov['gaps'])} of {len(ov['pairs'])} neighbouring replica pairs overlap less than "
          f"{OVERLAP_THRESHOLD}")
    kw = dict(log_x=False, xlabel="$\\mu_B$ (eV)", ylabel="temperature (K)",
              invalid_label=f"effective samples < {args.min_ess:g}")

    most = a.most_probable_N(T, mu)
    cv = a.heat_capacity(T, mu)
    ridges = heat_capacity_ridges(T_K, mu, cv, valid=valid, ladder=ladder)
    if args.bootstrap:
        b = bootstrap(a, lambda an: an.heat_capacity(T, mu), n_boot=args.bootstrap, seed=args.seed)
        support = ridge_support(ridges, b.replicates)
        print(f"C_V peaks: {len(support)}, supported by >= 95 % of {args.bootstrap} resamples: "
              f"{int(np.sum(support >= 0.95))}, of these on the sampled mu: "
              f"{int(np.sum((support >= 0.95) & ridges['on_ladder']))} (an overlap artefact "
              f"is reproducible: the bootstrap cannot detect it)")
    plot_map(T_K, mu, most, os.path.join(args.out, "most_probable_N.png"), valid=valid,
             discrete=True, points=ridges["points"], colorbar_label="most probable $N_B$",
             title="Stable phase and $C_V$ peaks", **kw)
    plot_map(T_K, mu, cv, os.path.join(args.out, "heat_capacity.png"), valid=valid,
             points=ridges["points"], log_color=True, cmap="magma", colorbar_label="$C_V/k_B$",
             title="Heat capacity ([ZSG] PRL 2022, Eq. 1)", **kw)
    cn = a.mean_coordination("cn_BA", T, mu)
    plot_map(T_K, mu, cn, os.path.join(args.out, "coordination_BA.png"),
             valid=valid & np.isfinite(cn), cmap="cividis",
             colorbar_label="mean A neighbours of a B atom (r < 2.8 A)",
             title="B-A coordination", **kw)

    # RDFs and B-B coordination at three states along the 290 K isotherm
    T_iso = kelvin_to_eV(290.0)
    curves_bb, curves_ba = {}, {}
    for m in (-2.2, -1.6, -1.0):
        p_atm = RESERVOIR.pressure_pascal(290.0, m) / 101325.0
        label = f"290 K, mu = {m} eV ({p_atm:.0e} atm), <N> = {a.mean_N(T_iso, m)[0]:.0f}"
        curves_bb[label] = a.radial_distribution("rdf_BB", T_iso, m)
        curves_ba[label] = a.radial_distribution("rdf_BA", T_iso, m)
        dist = a.coordination_distribution("cn_BB", T_iso, m)
        print(f"{label}: P(B-B coordination = 0..8) = {np.round(dist, 2)}")
    plot_radial_distribution(curves_bb, os.path.join(args.out, "rdf_BB.png"),
                             ylabel="B neighbours of a B atom per A", title="B-B (MBAR)")
    plot_radial_distribution(curves_ba, os.path.join(args.out, "rdf_BA.png"),
                             ylabel="A neighbours of a B atom per A", title="B-A (MBAR)")

    entries = export_phase_configurations(
        a, os.path.join(args.out, "configurations"), T, mu,
        os.path.join(args.out, "phases.extxyz"), min_ess=args.min_ess,
        temperature_K=True, min_cells=args.min_cells,
        tile_path=os.path.join(args.out, "phases_tiled.extxyz"))
    print(f"\n{len(entries)} phase regions exported to {args.out}/phases.extxyz:")
    for e in entries:
        print(f"  N_B = {e['phase_N']:3d} at {e['T_K']:5.0f} K, mu = {e['mu']:+.2f} eV "
              f"(P = {e['P_phase']:.2f}, from replica T{e['source_T_index']}_mu"
              f"{e['source_mu_index']} sample {e['source_sample']})")
    print(f"C_V peaks: {len(ridges['points'])} points (white dots in the maps)")


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="output_14_observables")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--discard", type=float, default=0.2)
    ap.add_argument("--min-ess", type=float, default=20.0)
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="block-bootstrap resamples for the C_V peaks (e.g. 60); 0: off")
    ap.add_argument("--mu-min", type=float, default=None,
                    help="denser mu ladder: first mu (eV); default the 11-point -2.4..-0.9 ladder")
    ap.add_argument("--mu-max", type=float, default=None, help="last mu (eV)")
    ap.add_argument("--mu-step", type=float, default=0.04, help="mu spacing (eV)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        simulate(args)
    analyze(args)


if __name__ == "__main__":
    main()
