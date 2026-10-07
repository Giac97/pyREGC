"""
A short bulk LJ fluid replica-exchange GCMC run on the parallel engine.

n_workers > 1 uses a ProcessPoolExecutor across replicas -- see
gcmc_re/engine.py's module docstring for what does and doesn't parallelize.
Speedup requires actual spare CPU cores and a per-replica cost that
dominates the pool's own overhead; worthwhile with an EAM/MLIP calculator,
not necessarily with the fast native LJCalculator used here.

Requires: only numpy (gcmc_re's core).
"""

import numpy as np

from gcmc_re import LJCalculator, BulkSystem, ReplicaExchangeGCMC


def make_bulk(rng: np.random.Generator) -> BulkSystem:
    return BulkSystem(box_length=7.0, calculator=LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


if __name__ == "__main__":
    temperatures = list(np.linspace(0.9, 1.3, 4))
    mus = list(np.linspace(-3.8, -3.2, 4))

    with ReplicaExchangeGCMC(temperatures, mus, make_bulk, seed=7,
                              max_disp=0.2, n_workers=1) as sim:
        sim.run(n_cycles=150, sweeps_per_cycle=150, exchange_every=1,
                 record_every=5, burn_in_cycles=20)

        print("\nFinal grid summary:")
        for row in sim.summary():
            print(row)
