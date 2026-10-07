"""
Hybrid Monte Carlo (HMC) as an alternative to single-particle random MC
displacement: collective, force-directed proposals instead of a blind
random walk, with an exact Metropolis correction so correctness never
depends on getting the integrator or timestep right (only efficiency does).

This compares plain MC-augmented GCMC against HMC-augmented GCMC on the
same bulk LJ system, at conditions chosen so both actually equilibrate
within a short run (checked, not assumed -- see the printed first-half vs
second-half comparison). If HMC's forces or leapfrog integration were
wrong, this would show up as a genuinely different equilibrium, not just
noisier numbers -- that's what to look for if you extend this to a new
Calculator's forces() implementation.

See Replica's docstring for the full design rationale (why HMC and not
just "run some MD"), and its current limitation (forces don't include any
external_energy field, e.g. a continuum wall -- fine for this bulk demo,
worth knowing before enabling HMC on a SlabSystem with wall_epsilon > 0).

Requires: only numpy (gcmc_re's core) -- no ASE needed for this one, since
it uses the native LJCalculator throughout.
"""

import numpy as np

from gcmc_re import LJCalculator, BulkSystem, ReplicaExchangeGCMC

T = 1.0
MU = -3.6  # chosen so the system equilibrates quickly at this box size/T


def make_system(rng: np.random.Generator) -> BulkSystem:
    return BulkSystem(7.0, LJCalculator(sigma=1.0, epsilon=1.0, r_cut=2.5))


def run(hmc_enabled: bool, hmc_dt: float, seed: int, n_cycles: int = 2000, sweeps: int = 30):
    # n_cycles: a few hundred cycles give only ~40 strongly correlated samples,
    # and MC vs HMC <N> then differ by ~20% from noise alone. At 2000 cycles a
    # single run's <N> is good to about +-0.4 (averaged over 3 seeds: MC
    # 12.49 +- 0.26, HMC 12.69 +- 0.22). Takes ~1.5 min.
    with ReplicaExchangeGCMC([T], [MU], make_system, seed=seed, max_disp=0.2,
                              hmc_enabled=hmc_enabled, hmc_dt=hmc_dt, n_workers=1) as sim:
        sim.run(n_cycles=n_cycles, sweeps_per_cycle=sweeps, record_every=2,
                 burn_in_cycles=n_cycles // 3, verbose=False)
        Ns = np.array(sim.samples[(0, 0)]["N"])
        half = len(Ns) // 2
        replica = sim.grid[(0, 0)]
        return Ns.mean(), Ns[:half].mean(), Ns[half:].mean(), replica.acceptance_rates()


if __name__ == "__main__":
    print("Plain MC (single-particle random displacement):")
    N_mc, first_mc, second_mc, acc_mc = run(hmc_enabled=False, hmc_dt=0.0, seed=1)
    print(f"  <N>={N_mc:.2f}  (1st half {first_mc:.2f}, 2nd half {second_mc:.2f} -- "
          f"agreement here is the equilibration check, not a formality)")
    print(f"  displace acceptance: {acc_mc['displace']:.1%}")

    print("\nHMC (conservative timestep):")
    N_hmc, first_hmc, second_hmc, acc_hmc = run(hmc_enabled=True, hmc_dt=0.01, seed=2)
    print(f"  <N>={N_hmc:.2f}  (1st half {first_hmc:.2f}, 2nd half {second_hmc:.2f})")
    print(f"  HMC acceptance: {acc_hmc['hmc']:.1%}")

    print("\nHMC (4x larger, sloppier timestep -- correctness should be unaffected,")
    print("only acceptance/efficiency):")
    N_hmc_sloppy, first_s, second_s, acc_s = run(hmc_enabled=True, hmc_dt=0.04, seed=3)
    print(f"  <N>={N_hmc_sloppy:.2f}  (1st half {first_s:.2f}, 2nd half {second_s:.2f})")
    print(f"  HMC acceptance: {acc_s['hmc']:.1%}")

    print(f"\nAll three <N> should agree within statistical noise: "
          f"{N_mc:.2f}, {N_hmc:.2f}, {N_hmc_sloppy:.2f}")
