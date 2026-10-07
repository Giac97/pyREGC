"""
A real multi-element ASE calculator (EMT) driving GCMC: H gas adsorbing
onto a rigid Au(111) slab, in REAL physical units (eV, Angstrom, Kelvin) --
not the reduced Lennard-Jones units used in the other examples.

Read the two caveats below before treating any number this script prints
as physically meaningful -- they're not a formality, they're based on
actually probing what EMT predicts for H near this surface (see the
"What EMT actually predicts here" section).

Requires: pip install ase
"""

import numpy as np
from ase.build import fcc111
from ase.calculators.emt import EMT

from gcmc_re import (
    ASECalculatorAdapter,
    MBARSurfaceAnalysis,
    ReplicaExchangeGCMC,
    SlabSystem,
    kelvin_to_eV,
    ThermalWavelengthCubed,
)

# =============================================================================
# Two caveats to keep in mind
# =============================================================================
#
# 1. EMT's own docs are explicit about this: "Supports the following
#    standard EMT metals: Al, Cu, Ag, Au, Ni, Pd and Pt. In addition, the
#    following elements are supported. They are NOT well described by EMT,
#    and the parameters are not for any serious use: H, C, N, O." Au is a
#    well-described EMT metal; H is not -- regardless of which metal it's
#    near. Swapping METAL below to "Pd" doesn't fix this; it's an H-specific
#    caveat, not an Au-specific one.
#
# 2. Separately, real-world chemistry: gold is a poor choice if you actually
#    want to see interesting hydrogen-metal chemistry, EMT caveat aside --
#    Au barely binds H2 in reality (weak, largely non-dissociative
#    physisorption), whereas Pd is the textbook example of strong
#    dissociative H2 chemisorption and bulk hydride formation, and Fe-H
#    chemistry (hydrogen embrittlement) is what motivated the EAM potentials
#    linked in the README. If you want genuinely interesting adsorption
#    physics AND trustworthy energetics, use a real Fe-H or Pd-H EAM/MLIP
#    (examples/03_frozen_substrate.py shows the same setup pattern), not
#    EMT+H on any metal.
#
# What EMT actually predicts here (checked directly, not assumed): scanning
# a single H atom's height above this Au(111) slab, EMT gives a steeply
# repulsive wall below about 1.8 Angstrom, then a shallow secondary minimum
# around 2.1-2.2 Angstrom that implies roughly a 3 eV binding energy
# relative to an isolated H atom -- implausibly strong for real hydrogen
# chemisorption, and a concrete illustration of caveat 1. Which is exactly
# why this script is a mechanism demo (does the engine correctly drive a
# real multi-element ASE calculator, in real units, with a rigid substrate?)
# and not a claim about real Au-H physics.
METAL = "Au"

# =============================================================================
# Build a small rigid Au(111) slab (frozen substrate) with ase.build.fcc111.
# orthogonal=True gives a rectangular cell -- required, since SlabSystem's
# geometry assumes axis-aligned periodicity in x, y. periodic=False gives
# pbc=(True, True, False): periodic in x, y, open in z, matching the slab
# geometry this package expects.
# =============================================================================
slab = fcc111(METAL, size=(4, 4, 3), vacuum=8.0, orthogonal=True, periodic=False)
Lx, Ly, Lz = slab.cell[0, 0], slab.cell[1, 1], slab.cell[2, 2]
top_z = slab.positions[:, 2].max()
frozen_positions = slab.positions.copy()
frozen_species = np.array(slab.get_chemical_symbols())


def make_system(rng: np.random.Generator) -> SlabSystem:
    # Pass the EMT class itself (a zero-arg factory), not EMT() (a live
    # instance): each replica gets its own fresh calculator, lazily
    # constructed on first use, and -- unlike a pre-built instance -- this
    # stays picklable even after use, which n_workers > 1 needs (see
    # ASECalculatorAdapter's docstring for why a used EMT() breaks pickling).
    calc = ASECalculatorAdapter(EMT, cell=[Lx, Ly, Lz], pbc=(True, True, False))
    return SlabSystem(
        Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
        frozen_positions=frozen_positions, frozen_species=frozen_species,
        # Restrict insertion attempts to a thin region just above the
        # surface instead of the whole (mostly-vacuum) box: EMT's favorable
        # window here is only ~0.3-0.5 Angstrom thick out of several
        # Angstrom of headroom, so unrestricted insertion would waste
        # nearly every attempt. volume shrinks to match, so the ensemble
        # stays exact (see SlabSystem's docstring).
        insertion_z_range=(top_z - 0.5, top_z + 5.0),
    )


# =============================================================================
# Real units: EMT reports energies in eV, so temperature must be in eV too
# (ThermoState.beta = 1/temperature), and the historical Lambda=1 convention
# baked into Replica's insertion/deletion formulas is wrong here -- H's
# actual thermal de Broglie wavelength depends on its mass and T. Both are
# handled by gcmc_re.units.
# =============================================================================
temperatures_K = [350.0, 450.0]
temperatures_eV = [kelvin_to_eV(T) for T in temperatures_K]
mu_H_eV = [-0.2, 0.0, 0.2, 0.5, 0.8]  # found by a quick empirical scan to span
                                       # empty -> covered


lambda_cubed_of_T = ThermalWavelengthCubed(1.008)   # H atoms


if __name__ == "__main__":
    with ReplicaExchangeGCMC(temperatures_eV, mu_H_eV, make_system, seed=11,
                              max_disp=0.3, n_workers=1,
                              lambda_cubed_of_T=lambda_cubed_of_T) as sim:
        sim.run(n_cycles=25, sweeps_per_cycle=20, exchange_every=1,
                 record_every=1, burn_in_cycles=5)

        print("\nFinal grid summary (T_kelvin is the real temperature; 'mu' is mu_H in eV):")
        for row in sim.summary():
            row["T_kelvin"] = temperatures_K[row["T_index"]]
            del row["T"]  # was in eV and redundant with T_kelvin; drop to avoid confusion
            print(row)

    # MBARSurfaceAnalysis doesn't change at all for a real-units Calculator --
    # it only ever reads back the recorded (U, N), same as every other example.
    analysis = MBARSurfaceAnalysis(sim)
    print("\nReweighted <N(H)> vs mu_H at T = 400 K (interpolated between the two "
          "simulated temperatures):")
    T_400K_eV = kelvin_to_eV(400.0)
    for mu in np.linspace(min(mu_H_eV), max(mu_H_eV), 8):
        mean_n, sigma_n = analysis.mean_N(T_400K_eV, mu)
        print(f"  mu_H={mu:5.2f} eV   <N>={mean_n:.3f} +- {sigma_n:.3f}")
