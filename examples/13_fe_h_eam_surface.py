"""
H on Fe(110) with an Fe-H potential: REGC sampling and a (T, p_H2) surface
phase diagram, following the protocol Zhou, Scheffler & Ghiringhelli, PRB 100,
174106 (2019) used for their ab initio runs.

Potential: an Fe-H EAM in LAMMPS format (``*.eam.fs`` or ``*.eam.alloy``)
via --potential -- e.g. ``examples/FeH_Wen.eam.fs`` (M. Wen, Comput. Mater.
Sci. 197, 110640 (2021): a = 2.855 A, H2 r0 = 0.739 A, D0 = 4.46 eV, so the
default --d0 self is appropriate) -- or any LAMMPS pair style via
--pair-style/--pair-coeff, e.g. a
GRACE-FS model (needs a LAMMPS built with GRACE):
    --pair-style grace/fs --pair-coeff "* * model.yaml Fe H" --min-distance 0.6
--min-distance adds a hard core keeping GCMC insertions out of unphysical,
off-training-set configurations of machine-learned potentials.

Setup (constants below or CLI flags):
  * Fe(110), 3x4 orthogonal cell, 6 layers; the bottom 3 frozen, the top 3
    displaceable; lattice constant from the potential itself.
  * H lives in z in (0, top + HEADROOM], z = 0 being the top frozen layer:
    subsurface sites of the relaxing layers are reachable, the frozen block
    is not.
  * H2 reservoir: D0 and the isolated-H energy from the same potential
    (--d0 experimental: D0 = 4.478 eV, for potentials with a poor H2).
  * Grid: temperatures x H2 pressures; REGC with x0 = 0.9 and Bussi NVT MD
    (0.5 fs, 40 steps per REGC step), or exact HMC (--diffusion hmc).
  * Engine: LAMMPS (default, if its Python module is installed) or ASE's
    pure-Python EAM (--backend ase, ~1000x slower).

Usage (nothing heavy happens without --run):
    python 13_fe_h_eam_surface.py --potential FeH_Wen.eam.fs --check
    python 13_fe_h_eam_surface.py --potential FeH_Wen.eam.fs --run --workers 12
    python 13_fe_h_eam_surface.py --analyze-only
--run continues a saved run with --resume, and stops cleanly before
--time-limit seconds (exit code 75); see hpc/ for the Slurm setup.

Requires: pip install ase pymbar matplotlib (and the LAMMPS Python module)
"""

import argparse
import functools
import os
import time

import numpy as np

from gcmc_re import (EXIT_RESUBMIT, H2_D0_EXPERIMENTAL_EV, ASECalculatorAdapter, DiatomicReservoir,
                     LammpsCalculator, MBARSurfaceAnalysis, ReplicaExchangeGCMC, cubic_lattice_constant,
                     eam_pair_commands, insertion_energy_scan, kelvin_to_eV, lammps_ase_calculator,
                     load_samples, run_with_checkpoints, slab_geometry_from_atoms)
from gcmc_re.replica import Replica
from gcmc_re.state import ThermoState

# ---- system --------------------------------------------------------------------
SPECIES = ["Fe", "H"]          # LAMMPS atom types 1, 2 (element order in pair_coeff)
SURFACE_SIZE = (3, 4)          # orthogonal Fe(110) cell; second number must be even
N_LAYERS, N_FROZEN_LAYERS = 6, 3
HEADROOM = 5.0                 # A of space for H above the top Fe layer
H_MASS_AMU = 1.008

# ---- thermodynamic grid -----------------------------------------------------------
TEMPERATURES_K = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
PRESSURES_BAR = list(np.logspace(-8, 4, 7))     # H2 pressure ladder

# ---- REGC protocol ---------------------------------------------------------------
X0 = 0.9
DIFFUSION_STEPS = 40           # MD steps or HMC trajectories per REGC step
MD = {"timestep_fs": 0.5, "thermostat": "bussi", "taut_fs": 20.0}
HMC_LEAPFROG_STEPS, HMC_DT = 10, 0.5 / 10.18     # 0.5 fs in the (A, amu, eV) time unit
MAX_DISP = 0.3                 # A, single-atom MC moves (not used by md/hmc diffusion)
RECORD_EVERY, CONVERGENCE_EVERY = 10, 2000


# =============================================================================
# Potential, geometry, reservoir
# =============================================================================
def pair_commands(args) -> list:
    """LAMMPS pair lines: explicit (e.g. grace/fs), or inferred from the EAM file."""
    if args.pair_style:
        return [f"pair_style {args.pair_style}", f"pair_coeff {args.pair_coeff}"]
    return eam_pair_commands(args.potential, SPECIES)


def ase_calculator(args):
    """The simulation's potential as an ASE calculator, for the setup scans."""
    if args.backend == "lammps":
        return lammps_ase_calculator(pair_commands(args), SPECIES)
    from ase.calculators.eam import EAM
    return EAM(potential=args.potential)


def system_factory(args, geometry):
    """``factory(rng) -> SlabSystem``: a fresh calculator for every replica."""
    lmp_cmds = pair_commands(args) if args.backend == "lammps" else None

    def make(rng):
        if lmp_cmds is not None:
            calc = LammpsCalculator(lmp_cmds, SPECIES, masses={"H": H_MASS_AMU},
                                    min_distance=args.min_distance)
        else:
            from ase.calculators.eam import EAM
            calc = ASECalculatorAdapter(functools.partial(EAM, potential=args.potential),
                                        cell=[*np.diag(geometry.cell_xy), geometry.Lz],
                                        pbc=(True, True, False))
        return geometry.make_system(calc, "H", mass_of_species={"H": H_MASS_AMU})
    return make


def setup(args):
    """Lattice constant, slab geometry and H2 reservoir, all from the potential."""
    from ase.build import bcc110
    calc = ase_calculator(args)
    a = cubic_lattice_constant(calc, "Fe", "bcc")
    slab = bcc110("Fe", size=(*SURFACE_SIZE, N_LAYERS), a=a, vacuum=None, orthogonal=True)
    geometry = slab_geometry_from_atoms(slab, n_frozen_layers=N_FROZEN_LAYERS, headroom=HEADROOM)
    reservoir = DiatomicReservoir.from_calculator(calc, "H", d0=args.d0)
    return a, geometry, reservoir


def metadata(args, a, reservoir) -> dict:
    """What a resumed run must agree with (checked by run_with_checkpoints)."""
    return {"pair_commands": pair_commands(args) if args.backend == "lammps" else None,
            "potential": os.path.abspath(args.potential) if args.potential else None,
            "lattice_constant": a, "D0_eV": reservoir.D0_eV, "e_atom_eV": reservoir.e_atom_eV,
            "temperatures_K": TEMPERATURES_K, "pressures_bar": [float(p) for p in PRESSURES_BAR],
            "x0": X0, "diffusion": args.diffusion, "diffusion_steps": DIFFUSION_STEPS,
            "record_every": RECORD_EVERY, "n_surface": None}


def diffusion_kwargs(args) -> dict:
    """Keyword arguments of the diffusion step."""
    return dict(MD) if args.diffusion == "md" else {}


def build_simulation(args, geometry, reservoir) -> ReplicaExchangeGCMC:
    """The replica grid: one replica per (T, p_H2)."""
    return ReplicaExchangeGCMC(
        [kelvin_to_eV(T) for T in TEMPERATURES_K],
        reservoir.mu_grid(TEMPERATURES_K, PRESSURES_BAR, unit="bar"),
        system_factory(args, geometry), seed=args.seed, max_disp=MAX_DISP,
        n_workers=args.workers, lambda_cubed_of_T=reservoir.lambda_cubed_of_T(),
        hmc_leapfrog_steps=HMC_LEAPFROG_STEPS, hmc_dt=HMC_DT)


# =============================================================================
# --check
# =============================================================================
def check(args) -> None:
    """Sanity checks of the potential and setup, plus a timing estimate."""
    t0 = time.time()
    a, geometry, reservoir = setup(args)
    info = reservoir.calculator_info
    print(f"bcc Fe lattice constant (potential): {a:.4f} A   (experiment 2.8665 A)")
    print(f"Fe(110) slab: {geometry.summary()}")
    print(f"H2 from the potential: r0 = {info['r0_angstrom']:.3f} A (exp 0.741), "
          f"D0 = {info['D0_eV']:.3f} eV (exp {H2_D0_EXPERIMENTAL_EV}), ZPE = {info['zpe_eV']:.3f} eV")
    print(f"isolated H energy e_atom = {reservoir.e_atom_eV:.4f} eV (zero of the GCMC energy scale)")
    if abs(info["r0_angstrom"] - 0.741) > 0.1 or abs(info["D0_eV"] - H2_D0_EXPERIMENTAL_EV) > 0.5:
        print("  WARNING: the potential's H2 is far from experiment; consider --d0 experimental")
    print(f"D0 used for the reservoir ({info['d0_source']}): {reservoir.D0_eV:.3f} eV")

    make = system_factory(args, geometry)
    bare = make(np.random.default_rng(0))
    half_h2 = reservoir.e_atom_eV - 0.5 * reservoir.D0_eV   # E(1/2 H2) on the calculator's scale
    top = bare.substrate_positions[np.argmax(bare.substrate_positions[:, 2])]
    ax, ay = np.diag(geometry.cell_xy) / np.array(SURFACE_SIZE)
    sites = {"on top (0, 0)": (0, 0), "offset (1/2, 1/2)": (0.5, 0.5),
             "offset (1/2, 0)": (0.5, 0), "offset (0, 1/2)": (0, 0.5)}
    dz = np.linspace(-1.5, 2.5, 41)
    scan = insertion_energy_scan(bare, [(top[0] + fx * ax, top[1] + fy * ay)
                                        for fx, fy in sites.values()], geometry.top_z + dz)
    print("H binding on unrelaxed Fe(110), height scan -1.5..+2.5 A around the top layer:")
    for name, row in zip(sites, scan):
        j = int(np.argmin(row))
        edge = "  [AT SCAN EDGE]" if j in (0, len(dz) - 1) else ""
        print(f"  {name:18s}: min at {dz[j]:+.2f} A, adsorption energy vs 1/2 H2 = "
              f"{row[j] - half_h2:+.3f} eV{edge}")

    mus = reservoir.mu_grid(TEMPERATURES_K, PRESSURES_BAR, unit="bar")
    print("\nmu_H grid (eV): rows T (K), columns p(H2) (bar)")
    print("        " + " ".join(f"{p:9.1e}" for p in PRESSURES_BAR))
    for T, row in zip(TEMPERATURES_K, mus):
        print(f"{T:6.0f}  " + " ".join(f"{m:9.3f}" for m in row))

    i_T = len(TEMPERATURES_K) // 2
    T = kelvin_to_eV(TEMPERATURES_K[i_T])
    rep = Replica(ThermoState(T, mus[i_T, -1], (0, 0),
                              lambda_cubed=reservoir.lambda_cubed_of_T()(T)),
                  make(np.random.default_rng(1)), np.random.default_rng(2), max_disp=MAX_DISP,
                  hmc_leapfrog_steps=HMC_LEAPFROG_STEPS, hmc_dt=HMC_DT)
    rep.regc_local_step(gc=True, mode=args.diffusion, n=DIFFUSION_STEPS, **diffusion_kwargs(args))
    n_probe = 3 if args.backend == "ase" else 30
    t1 = time.time()
    for _ in range(n_probe):
        rep.regc_local_step(gc=True, mode=args.diffusion, n=DIFFUSION_STEPS, **diffusion_kwargs(args))
    per_step = (time.time() - t1) / n_probe
    n_rep = len(TEMPERATURES_K) * len(PRESSURES_BAR)
    total = per_step * n_rep * args.steps
    print(f"\ntiming ({args.backend}): {per_step * 1e3:.1f} ms per REGC step per replica "
          f"({args.diffusion}, {DIFFUSION_STEPS} steps, N_H = {rep.system.n}); {n_rep} replicas x "
          f"{args.steps} steps = {total / 3600:.2f} CPU-hours, ~{total / 60 / max(1, args.workers):.0f} "
          f"min on {args.workers} workers (grows with H coverage; excludes process-pool overhead)")
    print(f"energy bookkeeping exact: {abs(rep.energy - rep.system.total_energy()) < 1e-6}")
    print(f"(check took {time.time() - t0:.0f} s)")


# =============================================================================
# --run and analysis
# =============================================================================
def simulate(args) -> bool:
    """Run or resume; True when finished, False when stopped for --time-limit."""
    a, geometry, reservoir = setup(args)
    meta = metadata(args, a, reservoir)
    meta["n_surface"] = geometry.n_surface
    with build_simulation(args, geometry, reservoir) as sim:
        status = run_with_checkpoints(
            sim, args.out, n_steps=args.steps, chunk=args.chunk, x0=X0, diffusion=args.diffusion,
            diffusion_steps=DIFFUSION_STEPS, record_every=RECORD_EVERY,
            convergence_every=CONVERGENCE_EVERY, convergence_tol=args.convergence_tol,
            time_limit=args.time_limit, resume=args.resume, overwrite=args.overwrite,
            metadata=meta, progress=lambda line: print(line, flush=True), **diffusion_kwargs(args))
    return status.finished


def analyze(args) -> None:
    """MBAR (T, p) phase diagram from the saved samples."""
    import json
    with open(os.path.join(args.out, "run_metadata.json")) as f:
        meta = json.load(f)
    reservoir = DiatomicReservoir.hydrogen(D0_eV=meta["D0_eV"], e_atom_eV=meta["e_atom_eV"])
    samples = load_samples(os.path.join(args.out, "samples.npz"), discard=args.discard)
    analysis = MBARSurfaceAnalysis(samples, reservoir=reservoir, pressure_unit="bar")
    T_K = np.linspace(min(meta["temperatures_K"]), max(meta["temperatures_K"]), 26)
    logp = np.linspace(np.log10(min(meta["pressures_bar"])), np.log10(max(meta["pressures_bar"])), 49)
    T_eV = np.array([kelvin_to_eV(T) for T in T_K])
    coverage = analysis.phase_diagram_Tp(T_eV, 10 ** logp, area=meta["n_surface"])  # H per surface Fe
    most_prob = analysis.most_probable_N_Tp(T_eV, 10 ** logp)
    np.savez(os.path.join(args.out, "phase_diagram.npz"), T_K=T_K, log10_p_bar=logp,
             coverage_ML=coverage, most_probable_N_H=most_prob)
    print(f"{samples.n_samples} samples per state; coverage {coverage.min():.3f}-{coverage.max():.3f} "
          f"ML (H per surface Fe); most probable N_H: {sorted(set(most_prob.ravel().tolist()))}")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, data, label, cmap in ((axes[0], coverage, "coverage (H per surface Fe)", "viridis"),
                                  (axes[1], most_prob, "most stable N_H", "gnuplot")):
        mesh = ax.pcolormesh(T_K, logp, data.T, shading="nearest", cmap=cmap)
        fig.colorbar(mesh, ax=ax, label=label)
        ax.set_xlabel("Temperature (K)")
        ax.set_title(f"MBAR@REGC: {label}")
    axes[0].set_ylabel("log10 p(H2) (bar)")
    path = os.path.join(args.out, "fe110_h_Tp_phase_diagram.png")
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def parse_args():
    """Command line."""
    ap = argparse.ArgumentParser(description="H on Fe(110): REGC (T, p_H2) phase diagram")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="sanity checks + timing, no simulation")
    mode.add_argument("--run", action="store_true", help="run (or resume) the simulation, then analyze")
    mode.add_argument("--analyze-only", action="store_true", help="analyze saved samples")
    ap.add_argument("--potential", help="Fe-H EAM file (*.eam.fs / *.eam.alloy)")
    ap.add_argument("--pair-style", help="LAMMPS pair_style instead of an EAM file, e.g. grace/fs")
    ap.add_argument("--pair-coeff", help="matching pair_coeff arguments, e.g. '* * model.yaml Fe H'")
    ap.add_argument("--min-distance", type=float, default=None,
                    help="hard core (A) for machine-learned potentials, e.g. 0.6")
    ap.add_argument("--backend", choices=("lammps", "ase"), default=None,
                    help="LAMMPS (default if installed) or ASE's pure-Python EAM")
    ap.add_argument("--d0", choices=("self", "experimental"), default="self",
                    help="H2 dissociation energy: from the potential (default) or experiment")
    ap.add_argument("--diffusion", choices=("md", "hmc"), default="md")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--chunk", type=int, default=1000, help="steps between checkpoints")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue the run saved in --out")
    ap.add_argument("--overwrite", action="store_true", help="start afresh even if --out holds a run")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds: stop cleanly (exit code 75) before exceeding this")
    ap.add_argument("--convergence-tol", type=float, default=0.05)
    ap.add_argument("--discard", type=float, default=0.2, help="equilibration fraction for the analysis")
    ap.add_argument("--out", default="output_13_fe_h_eam")
    args = ap.parse_args()
    if args.backend is None:
        try:
            import lammps  # noqa: F401
            args.backend = "lammps"
        except ImportError:
            args.backend = "ase"
    if not args.analyze_only:
        if args.pair_style or args.pair_coeff:
            if not (args.pair_style and args.pair_coeff):
                ap.error("--pair-style and --pair-coeff go together")
            if args.backend != "lammps":
                ap.error("--pair-style/--pair-coeff need --backend lammps")
        elif args.potential is None or not os.path.isfile(args.potential):
            ap.error(f"--check/--run need --potential (an existing EAM file) or "
                     f"--pair-style/--pair-coeff (got {args.potential!r})")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.check:
        check(args)
    else:
        if args.run and not simulate(args):
            raise SystemExit(EXIT_RESUBMIT)
        analyze(args)
