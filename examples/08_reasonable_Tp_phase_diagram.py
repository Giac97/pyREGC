"""
Pd(111) + H via EMT on a realistic (T, p) window, using the config-driven
pipeline (gcmc_re.runner) programmatically -- built as an inline dict here
rather than loaded from a file (see examples/09_config_driven_run.py +
pd111_emt_example.yaml for the file-based version of the same thing).

Read examples/05, 06 and 07 first. The runner maps p(H2) to mu_H with
EMT's own H2 dissociation energy D0 AND EMT's own isolated-H energy e_atom
(+3.21 eV, the zero of every GCMC energy difference; see
gcmc_re/reservoir.py). With both, 1e-4 to 1e6 atm at 350-450 K corresponds
to mu_H ~ +0.3 to +0.75 eV, inside the simulated mu_H grid.

What to expect: large H uptake across the whole window -- on this small
2x2 Pd(111) cell, roughly 10-25 H atoms, i.e. several H per surface Pd,
far beyond any real Pd-H surface phase. That is EMT's H description, not
the pipeline: EMT binds H about 3.1 eV below an isolated atom (vs ~2.5 eV
for half an EMT H2 molecule), and its H parameters are documented as "not
for any serious use". Use a real Pd-H EAM or an MLIP for physical results.

The demo is deliberately small-scale (8-atom Pd slab, 80 cycles) to keep
runtime reasonable, and at this length <N> is still rising at the end of
the run, i.e. NOT equilibrated. For a production run, scale up
replica_exchange.n_cycles/sweeps_per_cycle (or use scheme: regc with a
convergence check), watch coverage plateau before trusting any result, and
use n_workers > 1 on a multi-core machine.

Requires: pip install ase pymbar matplotlib
"""

import os

from gcmc_re.runner import run_from_config

HERE = os.path.dirname(os.path.abspath(__file__))

config = {
    "structure": {"file": os.path.join(HERE, "data", "pd111_slab.xyz")},
    "gas": {
        "species": "H",
        "insertion_z_range_above_surface": [1.0, 3.0],  # Angstrom above the top Pd layer
    },
    "calculator": {"type": "EMT"},
    "replica_exchange": {
        "temperatures_K": [350.0, 450.0],
        "mu_eV": [-0.5, 0.0, 0.3, 0.6, 0.9, 1.2],  # brackets mu_H ~ 0.3-0.75 eV of 1e-4..1e6 atm
        "n_cycles": 80,
        "sweeps_per_cycle": 15,
        "burn_in_cycles": 20,
        "record_every": 3,
        "seed": 7,
        "n_workers": 1,  # bump this on a multi-core machine
    },
    "trajectory": {"save": True, "directory": "trajectories", "every": 15},
    "diatomic_reservoir": {},  # D0 and e_atom computed self-consistently via EMT (printed below)
    "phase_diagram": {
        "T_min_K": 350, "T_max_K": 450, "n_T": 4,
        "p_min_atm": 1.0e-4, "p_max_atm": 1.0e6, "n_p": 8,
    },
    "output": {"directory": os.path.join(HERE, "output_08_pd_emt")},
}

if __name__ == "__main__":
    result = run_from_config(config)

    theta = result["theta_grid"]
    print(f"\nCoverage range across the whole (T, p) grid: "
          f"{theta.min():.3e} to {theta.max():.3e} particles/Angstrom^2")
    print(f"\nSelf-consistent D0 (EMT's own H2): {result['D0_eV']:.4f} eV; "
          f"EMT isolated-H energy (zero of the mu scale): {result['e_atom_eV']:.4f} eV")
    print("Outputs (plot, raw .npz, trajectories) are in", config["output"]["directory"])
