"""
A rigid substrate (e.g. a truncated Fe slab) with a mobile GCMC-exchangeable
species (e.g. H) adsorbing onto it -- the setup you'd use for a real Fe + H2
problem, just with a toy LJ calculator standing in for a real EAM/MLIP so
this runs with no extra dependencies beyond ASE.

For the real thing, swap `ASECalculatorAdapter(ASELennardJones(...))` below
for `ASECalculatorAdapter(EAM(potential='Fe_H.eam.fs'))` (see the README for
where to find real Fe-H EAM potentials) or an MLIP's ASE wrapper -- nothing
else in this script changes.

Requires: pip install ase
"""

import numpy as np
from ase.calculators.lj import LennardJones as ASELennardJones

from gcmc_re import ASECalculatorAdapter, Replica, SlabSystem, ThermoState

rng = np.random.default_rng(3)

# Toy "substrate": a small square lattice of frozen atoms at z=0, standing in
# for a truncated Fe slab. In a real setup you'd build this with e.g.
# ase.build.bcc110(...) and hand its .positions / .get_chemical_symbols() in.
nx, ny, a = 4, 4, 1.2
xs, ys = np.meshgrid(np.arange(nx) * a, np.arange(ny) * a)
frozen_positions = np.stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)], axis=1)
frozen_species = np.array(["Fe"] * frozen_positions.shape[0])

Lx = Ly = nx * a
Lz = 8.0
calc = ASECalculatorAdapter(
    ASELennardJones(sigma=1.0, epsilon=1.0, rc=2.5, smooth=False, shift=True),
    cell=[Lx, Ly, Lz], pbc=(True, True, False),
)

# wall_epsilon=0: the substrate's own interaction with the mobile species
# supplies the attraction/repulsion instead of an artificial continuum wall.
system = SlabSystem(Lx, Ly, Lz, calc, exchange_species="H", wall_epsilon=0.0,
                       frozen_positions=frozen_positions, frozen_species=frozen_species)

state = ThermoState(temperature=1.2, mu=-2.0, label=(0, 0))
replica = Replica(state, system, rng, max_disp=0.3)

print(f"initial: {system.n} mobile atoms, {system.frozen_positions.shape[0]} frozen atoms")
print(f"initial energy (Fe-Fe self-energy of the substrate lattice): {replica.energy:.4f}")

replica.sweep(400, p_displace=0.4, p_insert=0.3, p_delete=0.3)

pos, sp = system.full_configuration()
counts = {s: int((sp == s).sum()) for s in set(sp.tolist())}
print(f"\nafter 400 moves: {system.n} mobile atoms")
print(f"full_configuration species counts: {counts}")
print(f"energy bookkeeping still exact after 400 moves: "
      f"{abs(replica.energy - system.total_energy()) < 1e-6} "
      f"({replica.energy:.4f} vs {system.total_energy():.4f})")
print(f"acceptance rates: {replica.acceptance_rates()}")
