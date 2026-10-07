"""
Surface adsorption replica-exchange GCMC + MBAR reweighting into a
temperature-pressure surface phase diagram.

Uses the fast LJCalculator + a continuum wall so this runs in about a
minute; for a real Fe(-H) system swap `calculator` for
ASECalculatorAdapter(EAM(...)) or an MLIP (see examples/03_frozen_substrate.py
and the README) -- MBARSurfaceAnalysis below doesn't change at all, since it
only ever reads back recorded (U, N).

Requires: pip install pymbar matplotlib
"""

import numpy as np

from gcmc_re import (
    LJCalculator,
    MBARSurfaceAnalysis,
    ReplicaExchangeGCMC,
    SlabSystem,
    mu_grid_from_pressures,
    p0_reduced_units,
    plot_phase_diagram,
)

Lx = Ly = 8.0
Lz = 10.0
WALL_EPSILON = 5.0


def make_slab(rng: np.random.Generator) -> SlabSystem:
    calc = LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5)
    return SlabSystem(Lx, Ly, Lz, calc, exchange_species="X",
                         wall_epsilon=WALL_EPSILON, wall_sigma=1.0)


if __name__ == "__main__":
    temperatures = list(np.geomspace(0.60, 0.90, 4))
    pressures = list(np.geomspace(2e-4, 1e1, 7))
    mu_grid = mu_grid_from_pressures(temperatures, pressures, p0_of_T=p0_reduced_units)

    with ReplicaExchangeGCMC(temperatures, mu_grid, make_slab, seed=7,
                              max_disp=0.2, n_workers=8) as sim:
        sim.run(n_cycles=400, sweeps_per_cycle=120, exchange_every=1,
                 record_every=2, burn_in_cycles=100)

    analysis = MBARSurfaceAnalysis(sim, p0_of_T=p0_reduced_units)

    print("\nReweighted isotherm at T = 0.75 (not in the T-ladder):")
    p_fine = np.geomspace(2e-4, 3e-2, 12)
    theta, dtheta = analysis.isotherm_p(T=0.75, p_grid=p_fine, area=Lx * Ly)
    for p, th, dth in zip(p_fine, theta, dtheta):
        print(f"  p={p:9.2e}   coverage={th:.4f} +- {dth:.4f}")

    T_fine = np.linspace(min(temperatures), max(temperatures), 15)
    p_fine_grid = np.geomspace(min(pressures), max(pressures), 25)
    theta_grid = analysis.phase_diagram_Tp(T_fine, p_fine_grid, area=Lx * Ly)

    print("\nSearching for a 2D condensation line across the T-ladder:")
    coexistence = analysis.coexistence_curve_Tp(
        T_grid=temperatures, p_bracket=(min(pressures), max(pressures)), n_bins=60)
    if not coexistence:
        print("  No bimodal P(N) found on this ladder -- needs pressure points "
              "concentrated in the steep part of the isotherm above, not spread "
              "evenly across the whole range. See the README's 'Extending' notes.")

    plot_phase_diagram(T_fine, p_fine_grid, theta_grid, coexistence=coexistence,
                        save_path="surface_phase_diagram.png")
