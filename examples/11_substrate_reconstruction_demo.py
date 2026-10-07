"""
The displaceable substrate: atoms that enter the energy evaluation and CAN
move (unlike `frozen_positions`/`frozen_species`), but are never inserted
or deleted by GCMC (unlike the mobile species) -- what a reconstructing
surface needs.

This demonstrates the core physics on the simplest case where the correct
answer is known exactly: two substrate atoms, interacting via a real
(ASE) Lennard-Jones potential, started well away from that potential's
known equilibrium separation (2^(1/6)*sigma). If substrate dynamics are
implemented correctly, they should relax toward that separation under real
forces -- via joint HMC (substrate + mobile species evolving together
under one trajectory) and, independently, via plain single-particle MC
substrate moves. Both are shown below; both should land close to the same
answer, by two different (and independently validated -- see
tests/test_substrate.py) mechanisms.

A genuinely interesting demo -- adsorbate-induced reconstruction, where a
GCMC-controlled gas coverage measurably shifts the substrate's own
equilibrium geometry -- needs more careful (T, mu, mass, timestep) tuning
than fits here: a first attempt at combining both, with too-favorable mu,
gave overwhelming gas coverage that disrupted the substrate dynamics
entirely rather than gently perturbing them. That tuning is real,
unfinished work, not a solved problem -- this script deliberately keeps
the two mechanisms (substrate relaxation, GCMC coverage) demonstrated
separately, each validated on its own, rather than force a combined result
that hasn't been properly calibrated.

Requires: pip install ase (for the reference LennardJones calculator used
here; this all works identically with EAM or an MLIP for a real surface).
"""

import numpy as np

from gcmc_re import ASECalculatorAdapter, BulkSystem, ReplicaExchangeGCMC, ThermoState, Replica

SIGMA, EPSILON, R_CUT = 1.0, 1.0, 2.5
R_NATURAL = 2 ** (1.0 / 6.0) * SIGMA  # the LJ potential's known equilibrium separation
R_START = 1.8 * SIGMA                 # deliberately far from it


def make_stretched_substrate():
    from ase.calculators.lj import LennardJones as ASELennardJones

    calc = ASECalculatorAdapter(
        ASELennardJones(sigma=SIGMA, epsilon=EPSILON, rc=R_CUT, smooth=False, shift=True),
        cell=[10.0, 10.0, 10.0], pbc=(True, True, True))
    system = BulkSystem(10.0, calc)
    system.substrate_positions = np.array([[5.0, 5.0, 5.0], [5.0 + R_START, 5.0, 5.0]])
    system.substrate_species = np.array(["X", "X"])
    system.mass_of_species = {"X": 10.0}  # heavier than a typical adsorbate, as real substrates are
    return system


def separation(system) -> float:
    return float(np.linalg.norm(system.substrate_positions[1] - system.substrate_positions[0]))


if __name__ == "__main__":
    print(f"LJ equilibrium separation: {R_NATURAL:.4f}   starting separation: {R_START:.4f}\n")

    # --- Mechanism 1: joint HMC (substrate moves together with the force field) ---
    system_hmc = make_stretched_substrate()
    state = ThermoState(temperature=0.05, mu=-100.0, label=(0, 0))  # low T: relax, don't diffuse
    rng = np.random.default_rng(7)
    replica_hmc = Replica(state, system_hmc, rng, hmc_enabled=True,
                            hmc_leapfrog_steps=15, hmc_dt=0.02)
    print("Mechanism 1: joint HMC")
    for step in [0, 50, 100, 150, 200]:
        if step > 0:
            for _ in range(50):
                replica_hmc._move_hmc()
        print(f"  after {step:4d} HMC trials: separation = {separation(system_hmc):.4f}")
    print(f"  HMC acceptance: {replica_hmc.acceptance_rates()['hmc']:.1%}")

    # --- Mechanism 2: plain single-particle MC substrate moves ---
    system_mc = make_stretched_substrate()
    rng2 = np.random.default_rng(8)
    replica_mc = Replica(state, system_mc, rng2, max_disp=0.1, hmc_enabled=False)
    print("\nMechanism 2: plain MC substrate displacement")
    checkpoints = [0, 300, 600, 900, 1500]
    print(f"  after {0:4d} MC trials: separation = {separation(system_mc):.4f}")
    for prev, curr in zip(checkpoints[:-1], checkpoints[1:]):
        for _ in range(curr - prev):
            replica_mc._move_displace_substrate()
        print(f"  after {curr:4d} MC trials: separation = {separation(system_mc):.4f}")
    print(f"  displace_substrate acceptance: {replica_mc.acceptance_rates()['displace_substrate']:.1%}")

    print(f"\nBoth mechanisms should land close to {R_NATURAL:.4f} -- "
          f"HMC: {separation(system_hmc):.4f}, plain MC: {separation(system_mc):.4f}")

    # --- This also works through the standard engine, with p_displace_substrate ---
    print("\nSame mechanism through the standard ReplicaExchangeGCMC.run() interface:")
    with ReplicaExchangeGCMC([0.05], [-100.0], lambda rng: make_stretched_substrate(),
                              seed=1, max_disp=0.1, n_workers=1) as sim:
        sim.run(n_cycles=50, sweeps_per_cycle=30, record_every=10, verbose=False,
                 p_displace=0.0, p_insert=0.0, p_delete=0.0, p_displace_substrate=1.0)
    final_system = sim.grid[(0, 0)].system
    print(f"  final separation: {separation(final_system):.4f}")
