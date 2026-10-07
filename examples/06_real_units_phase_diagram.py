"""
The surface phase diagram from examples/05_emt_metal_surface.py, but
properly referenced against a REAL H2 gas reservoir at (T, p) -- not just
scanned over an abstract mu_H, and not just converted via the naive
reduced-units p0(T) = T convention from pressure.py (which assumes an
*atomic* ideal gas reservoir; here the real reservoir is diatomic H2).

Read this alongside examples/05_emt_metal_surface.py's caveats about what
EMT actually predicts for H. The reservoir referencing needs two
calculator-consistent energies: H2's dissociation energy D0 and the
calculator's own isolated-H energy e_atom (the zero of every GCMC energy
difference; +3.21 eV for EMT). Leaving out e_atom would shift every mu_H
down by 3.21 eV and make ordinary pressures look like they give exactly zero
coverage.

Requires: pip install ase pymbar matplotlib
"""

import numpy as np
from ase.build import fcc111
from ase.calculators.emt import EMT

from gcmc_re import (
    ASECalculatorAdapter,
    DiatomicReservoir,
    MBARSurfaceAnalysis,
    ReplicaExchangeGCMC,
    SlabSystem,
    eV_to_kelvin,
    kelvin_to_eV,
)

METAL = "Au"

# =============================================================================
# 1. Self-consistent H2 dissociation energy, AS EMT SEES IT.
#
# GCMC here inserts/deletes single H atoms, but a real experiment controls
# H2 gas pressure: mu_H = (1/2) mu_H2(p, T). That conversion only makes
# sense once mu_H2 is referenced to "two separated ground-state H atoms" as
# its zero of energy -- which means subtracting H2's dissociation energy D0.
# Since EMT is only an approximation, D0 must come from EMT ITSELF (a bond-
# length scan + harmonic zero-point correction, both done with EMT), not
# from the real experimental value (4.478 eV) -- using the experimental
# number would silently introduce an offset error, because EMT's own
# isolated-atom energy has no absolute physical meaning to begin with. This
# is the same self-consistency point examples/03 and 05 make about
# adsorption energies, just applied to the gas-phase reference instead.
# =============================================================================
reservoir = DiatomicReservoir.from_calculator(EMT(), "H")   # D0 and e_atom from EMT itself
print("Self-consistent H2 dissociation energy, as EMT sees it:")
for k, v in reservoir.calculator_info.items():
    print(f"  {k:28s} = {v:.4f}" if isinstance(v, float) else f"  {k:28s} = {v}")
print("  (for comparison, real H2: r0 = 0.741 A, D0 = 4.478 eV)")

# =============================================================================
# 2. What does a REAL H2 reservoir's pressure translate to, in mu_H?
#
# The rotational/vibrational partition function uses REAL H2 spectroscopy
# (that's real, calculator-independent physics of the gas in the cylinder);
# D0 and e_atom come from EMT (step 1), for the reason above:
#     mu_H = (1/2) [mu_H2(T, p) - D0] + e_atom
# =============================================================================
print("\nWhat real H2 pressure gives which mu_H (using EMT's own D0 and e_atom):")
for T_K in (350.0, 450.0):
    for p_bar in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        mu_H = reservoir.mu_eV(T_K, p_bar * 1e5)
        print(f"  T={T_K:5.0f} K   p={p_bar:10.1e} bar   ->   mu_H = {mu_H:7.3f} eV")

# and the reverse: which pressures does the simulated mu_H window (step 3) cover?
for mu_H in (-0.2, 0.8):
    p_bar = reservoir.pressure_pascal(400.0, mu_H) / 1e5
    print(f"\nmu_H = {mu_H:+.1f} eV at 400 K  <->  p(H2) = {p_bar:.3e} bar")
print("So the simulated mu_H window contains the whole lab-relevant 1e-4 to 1e6 atm range.")

# =============================================================================
# 3. The actual simulation -- identical setup to examples/05_emt_metal_surface.py,
#    on a mu_H grid. Step 2 shows how that axis maps to real pressure:
#    -0.2 to 0.8 eV spans ~1e-19 to ~2e7 bar at 400 K, so the realistic
#    1e-4 to 1e6 atm window (mu_H ~ +0.3 to +0.75 eV) lies inside it;
#    examples/07 plots the same samples on a pressure axis.
# =============================================================================
slab = fcc111(METAL, size=(4, 4, 3), vacuum=8.0, orthogonal=True, periodic=False)
Lx, Ly, Lz = slab.cell[0, 0], slab.cell[1, 1], slab.cell[2, 2]
top_z = slab.positions[:, 2].max()
frozen_positions = slab.positions.copy()
frozen_species = np.array(slab.get_chemical_symbols())


def make_system(rng: np.random.Generator) -> SlabSystem:
    calc = ASECalculatorAdapter(EMT, cell=[Lx, Ly, Lz], pbc=(True, True, False))
    return SlabSystem(
        Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
        frozen_positions=frozen_positions, frozen_species=frozen_species,
        insertion_z_range=(top_z - 0.5, top_z + 5.0),
    )


temperatures_K = [350.0, 450.0]
temperatures_eV = [kelvin_to_eV(T) for T in temperatures_K]
mu_H_eV = [-0.2, 0.0, 0.2, 0.5, 0.8]


lambda_cubed_of_T = reservoir.lambda_cubed_of_T()   # H atoms


if __name__ == "__main__":
    with ReplicaExchangeGCMC(temperatures_eV, mu_H_eV, make_system, seed=11,
                              max_disp=0.3, n_workers=1,
                              lambda_cubed_of_T=lambda_cubed_of_T) as sim:
        sim.run(n_cycles=25, sweeps_per_cycle=20, exchange_every=1,
                 record_every=1, burn_in_cycles=5)

    analysis = MBARSurfaceAnalysis(sim)

    # =========================================================================
    # 4. The phase diagram itself: coverage(T, mu_H), reweighted onto a fine
    #    mesh -- real T in Kelvin, real mu_H in eV, real coverage in
    #    particles/Angstrom^2. examples/07 shows it against p(H2) instead.
    # =========================================================================
    T_fine_eV = np.linspace(min(temperatures_eV), max(temperatures_eV), 12)
    mu_fine_eV = np.linspace(min(mu_H_eV), max(mu_H_eV), 30)
    area = Lx * Ly
    theta_grid = analysis.phase_diagram(T_fine_eV, mu_fine_eV, area=area)
    T_fine_K = np.array([eV_to_kelvin(T_eV) for T_eV in T_fine_eV])

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 5))
        mesh = ax.pcolormesh(mu_fine_eV, T_fine_K, theta_grid, shading="auto", cmap="viridis")
        ax.set_xlabel("mu_H (eV)")
        ax.set_ylabel("temperature (K)")
        ax.set_title(f"{METAL}(111) + H via EMT: coverage(T, mu_H)\n"
                      f"(1e-4 to 1e6 atm H2 corresponds to mu_H ~ +0.3 to +0.75 eV)")
        fig.colorbar(mesh, ax=ax, label="coverage (H / Angstrom^2)")
        fig.tight_layout()
        fig.savefig("real_units_phase_diagram.png", dpi=150)
        print("\nSaved real_units_phase_diagram.png")
    except ImportError:
        print("\nmatplotlib not installed -- skipping plot (pip install matplotlib).")
