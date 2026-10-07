"""
The real (T, p) surface phase diagram for Au(111) + H via EMT -- the same
simulation setup as examples/05/06 (rerun here, so this script is
self-contained) but now with the actual diatomic H2 reservoir wired in as
MBARSurfaceAnalysis's mu(T, p) conversion, instead of the simple atomic-gas
p0(T) convention.

This uses the general (T, mu) -> (T, p) conversion: MBARSurfaceAnalysis accepts a general
`mu_of_Tp`/`p_of_Tmu` pair that overrides the built-in simple conversion,
and every method built on top of it (`isotherm_p`, `phase_diagram_Tp`,
`locate_coexistence_p`, ...) uses the override automatically, unchanged.

Read examples/05 and 06 first for the caveats about what EMT+H actually
predicts. The pressure axis below is the p_of_Tmu image of the simulated
mu_H window (-0.2 to 0.8 eV), i.e. roughly 1e-22 to 1e8 bar: it contains
the lab-relevant 1e-4 to 1e6 atm range, so the coverage in that range is
interpolated by MBAR between simulated states, not extrapolated. Both the
dissociation energy D0 and the isolated-atom energy e_atom must come from
EMT itself (see gcmc_re/reservoir.py).

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
    ThermalWavelengthCubed,
    eV_to_kelvin,
    kelvin_to_eV,
    plot_phase_diagram,
)

METAL = "Au"

# =============================================================================
# Identical simulation setup to examples/05 and 06 -- see those for the
# caveats about EMT+H and Au's real-world hydrogen chemistry.
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


lambda_cubed_of_T = ThermalWavelengthCubed(1.008)   # H atoms


if __name__ == "__main__":
    with ReplicaExchangeGCMC(temperatures_eV, mu_H_eV, make_system, seed=11,
                              max_disp=0.3, n_workers=1,
                              lambda_cubed_of_T=lambda_cubed_of_T) as sim:
        sim.run(n_cycles=25, sweeps_per_cycle=20, exchange_every=1,
                 record_every=1, burn_in_cycles=5)

    # =========================================================================
    # Self-consistent D0 (see examples/06 for why it must come from EMT
    # itself, not an experimental table), then wire the real H2 reservoir in
    # as MBARSurfaceAnalysis's mu(T, p) conversion.
    # =========================================================================
    reservoir = DiatomicReservoir.from_calculator(EMT(), "H")
    print(f"Self-consistent D0 (EMT's own H2): {reservoir.D0_eV:.4f} eV; "
          f"EMT isolated-H energy (zero of the mu scale): {reservoir.e_atom_eV:.4f} eV")
    analysis = MBARSurfaceAnalysis(sim, reservoir=reservoir, pressure_unit="bar")

    # =========================================================================
    # The actual (T, p) phase diagram. phase_diagram_Tp is UNCHANGED from
    # examples/04 -- it just calls mu_from_p under the hood, which now uses
    # the real H2 reservoir instead of the simple atomic-gas convention.
    # The pressure range below isn't picked for a nice-looking plot: it's
    # exactly the p_of_Tmu(T, mu) span for the mu_H window this simulation
    # actually explored (-0.2 to 0.8 eV) at both temperatures.
    # =========================================================================
    T_fine_eV = np.linspace(min(temperatures_eV), max(temperatures_eV), 12)
    p_bounds_bar = [analysis.p_from_mu(T_eV, mu)
                    for T_eV in (min(temperatures_eV), max(temperatures_eV))
                    for mu in (min(mu_H_eV), max(mu_H_eV))]
    p_grid_bar = np.geomspace(min(p_bounds_bar) * 0.1, max(p_bounds_bar) * 10, 40)
    area = Lx * Ly

    theta_grid = analysis.phase_diagram_Tp(T_fine_eV, p_grid_bar, area=area)
    T_fine_K = np.array([eV_to_kelvin(T_eV) for T_eV in T_fine_eV])

    plot_phase_diagram(T_fine_K, p_grid_bar, theta_grid,
                        save_path="real_Tp_phase_diagram.png")
    print(f"\nPressure axis spans {p_grid_bar.min():.2e} to {p_grid_bar.max():.2e} bar --")
    print("the image of the simulated mu_H window; it contains the 1e-4 to 1e6 atm range.")
