"""
Config-file-driven end-to-end runner: read a structure file, run replica-
exchange GCMC (with optional trajectory saving), and produce a (T, p)
surface phase diagram -- all from one config file (YAML or JSON).

See examples/09_config_driven_run.py (the CLI wrapper) and
examples/pd111_emt_example.yaml (an annotated example config) for how this
is meant to be used; `run_from_config` below is the whole pipeline in one
function, callable directly if you'd rather build the config dict in code.

A YAML gotcha worth knowing before writing your own config: PyYAML (using
the YAML 1.1 spec) only recognizes scientific notation as a float when the
exponent has an EXPLICIT sign -- `1.0e-4` parses as the float 0.0001, but
`1.0e6` (no `+`) silently parses as the STRING "1.0e6". `run_from_config`
defensively re-parses the numeric fields most likely to hit this (pressures,
D0) using Python's own `float()`, which has no such sign requirement, so a
bare `1.0e6` there is silently corrected rather than causing a confusing
downstream failure -- but only for those specific fields; a genuinely
invalid value (not number-shaped at all) still raises a clear error. Other
numeric fields you add yourself don't get this protection, so it's worth
just always including the sign regardless.

Requires: pip install ase pymbar matplotlib, plus pyyaml if using a
.yaml/.yml config (a .json config needs no extra dependency -- json is in
the standard library).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a JSON or YAML config file.

    Format is inferred from the extension.

    Parameters
    ----------
    path : str
        Path to the config file. ``.yaml``/``.yml`` is parsed as YAML
        (requires ``pip install pyyaml``); anything else is parsed as
        JSON.

    Returns
    -------
    dict
        The parsed config. A relative ``structure.file`` is resolved
        against the config file's own directory, so the config works from
        any working directory.
    """
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml  # pip install pyyaml
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    struct = config.get("structure", {})
    if "file" in struct and not os.path.isabs(struct["file"]):
        struct["file"] = os.path.join(os.path.dirname(os.path.abspath(path)), struct["file"])
    return config


def _as_float(value: Any, field_name: str) -> float:
    """
    Coerce a config value to float, with a clear error message.

    Rather than a confusing downstream one, if it came through as a
    string -- the classic symptom of the YAML unsigned-exponent gotcha
    documented in this module's docstring.

    Parameters
    ----------
    value : Any
        The config value to coerce.
    field_name : str
        The config field's name (dotted path), used only in the error
        message.

    Returns
    -------
    float
        `value`, coerced to float.

    Raises
    ------
    ValueError
        If `value` can't be coerced to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Config field {field_name!r} = {value!r} isn't a valid number. If you wrote "
            f"this in scientific notation in a YAML file, PyYAML needs an explicit sign on "
            f"the exponent (1.0e+6, not 1.0e6) or it silently parses as a string -- see this "
            f"module's docstring."
        )


def _lookup_mass_amu(symbol: str) -> float:
    """
    Look up a real element's atomic mass via ASE's data tables.

    Parameters
    ----------
    symbol : str
        A real element symbol (e.g. ``'H'``).

    Returns
    -------
    float
        The atomic mass, in atomic mass units.
    """
    from ase.data import atomic_masses, atomic_numbers
    return float(atomic_masses[atomic_numbers[symbol]])


def _raw_ase_calculator_factory(calc_cfg: Dict[str, Any]) -> Optional[Callable[[], Any]]:
    """
    Build a factory for a fresh, raw ASE calculator instance.

    Not wrapped in our `ASECalculatorAdapter` -- what
    `diatomic_dissociation_energy_eV` needs for a self-consistent D0.

    Parameters
    ----------
    calc_cfg : dict
        The config's ``calculator`` section, with a ``'type'`` key.

    Returns
    -------
    callable or None
        A zero-argument factory for the raw ASE calculator class, or None
        for calculator types (like our own native `LJCalculator`) that
        have no such raw ASE equivalent, in which case
        `diatomic_reservoir.D0_eV` must be given explicitly in the config.
    """
    calc_type = calc_cfg.get("type", "EMT").upper()
    if calc_type == "EMT":
        from ase.calculators.emt import EMT
        return EMT
    return None


def _build_calculator(calc_cfg: Dict[str, Any], cell, pbc):
    """
    Build the wrapped `Calculator` used to drive the simulation.

    Either a real ASE calculator via `ASECalculatorAdapter`, or, for quick
    pipeline testing, the fast native `LJCalculator`.

    Parameters
    ----------
    calc_cfg : dict
        The config's ``calculator`` section, with a ``'type'`` key
        (``'EMT'`` or ``'LJ'``) and any type-specific parameters
        (``lj_sigma``, ``lj_epsilon``, ``lj_r_cut`` for ``'LJ'``).
    cell : sequence of float
        The (Lx, Ly, Lz) cell dimensions, passed through to
        `ASECalculatorAdapter` if `calc_cfg`'s type is ``'EMT'``.
    pbc : tuple of bool
        Periodicity along each axis, passed through to
        `ASECalculatorAdapter` if `calc_cfg`'s type is ``'EMT'``.

    Returns
    -------
    Calculator
        The constructed calculator (see `calculators.py`).

    Raises
    ------
    ValueError
        If ``calc_cfg['type']`` is neither ``'EMT'`` nor ``'LJ'``.
    """
    calc_type = calc_cfg.get("type", "EMT").upper()
    if calc_type == "EMT":
        from ase.calculators.emt import EMT
        from gcmc_re import ASECalculatorAdapter
        return ASECalculatorAdapter(EMT, cell=cell, pbc=pbc)
    elif calc_type == "LJ":
        from gcmc_re import LJCalculator
        return LJCalculator(sigma=calc_cfg.get("lj_sigma", 1.0),
                             epsilon=calc_cfg.get("lj_epsilon", 0.1),
                             r_cut=calc_cfg.get("lj_r_cut", 3.0))
    else:
        raise ValueError(f"Unknown calculator.type '{calc_cfg.get('type')}' -- use 'EMT' or 'LJ'.")


def run_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Run the whole config-driven pipeline: structure to phase diagram.

    See this module's docstring, and ``examples/pd111_emt_example.yaml``
    for the config schema. Writes the phase diagram plot (and, by default,
    the raw grid as .npz) to ``config['output']['directory']``.

    Parameters
    ----------
    config : dict
        The parsed config (see `load_config`), with top-level sections
        ``'structure'``, ``'gas'``, ``'calculator'`` (optional),
        ``'replica_exchange'``, ``'diatomic_reservoir'`` (optional),
        ``'phase_diagram'``, ``'output'`` (optional), and ``'trajectory'``
        (optional).
    verbose : bool, optional
        If True, print progress messages. Default is True.

    Returns
    -------
    dict
        Keys ``'sim'`` (the `ReplicaExchangeGCMC` instance), ``'analysis'``
        (the `MBARSurfaceAnalysis` instance), ``'theta_grid'`` (the
        computed coverage surface), ``'T_grid_K'``, ``'p_grid_atm'`` (the
        grids `theta_grid` was evaluated on), ``'D0_eV'`` (the
        dissociation energy used), and ``'e_atom_eV'`` (the calculator's
        isolated-atom energy, the zero of the mu scale; from
        ``diatomic_reservoir.e_atom_eV`` if given, else evaluated with the
        configured calculator) -- for further use in a notebook or REPL.

    Raises
    ------
    ValueError
        If the structure file's cell is non-orthogonal, if
        ``diatomic_reservoir.D0_eV`` is missing and the calculator type
        has no automatic self-consistent D0, or if any numeric field
        can't be coerced (see `_as_float`).
    """
    from ase.io import read as ase_read

    from gcmc_re import (
        DiatomicReservoir,
        MBARSurfaceAnalysis,
        ReplicaExchangeGCMC,
        SlabSystem,
        ThermalWavelengthCubed,
        diatomic_dissociation_energy_eV,
        kelvin_to_eV,
        plot_phase_diagram,
    )

    # ---- structure -----------------------------------------------------
    struct_cfg = config["structure"]
    atoms = ase_read(struct_cfg["file"])
    cell = np.array(atoms.cell[:], dtype=float)
    off_diag = cell - np.diag(np.diag(cell))
    if not np.allclose(off_diag, 0.0, atol=1e-6):
        raise ValueError(
            "gcmc_re's slab geometry needs an orthogonal (diagonal) cell, but "
            f"{struct_cfg['file']!r} has a non-orthogonal one. Rebuild it as an "
            "orthogonal supercell first -- e.g. ase.build.fcc111(..., orthogonal=True) "
            "for an fcc(111) surface -- before pointing this runner at it."
        )
    Lx, Ly, Lz = float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])
    Lz += float(struct_cfg.get("extra_vacuum_angstrom", 0.0))
    frozen_positions = atoms.positions.copy()
    frozen_species = np.array(atoms.get_chemical_symbols())
    top_z = float(frozen_positions[:, 2].max()) if len(atoms) else 0.0

    # ---- gas species -----------------------------------------------------
    gas_cfg = config["gas"]
    gas_species = gas_cfg["species"]
    gas_mass_amu = gas_cfg.get("mass_amu") or _lookup_mass_amu(gas_species)
    z_lo_rel, z_hi_rel = gas_cfg.get("insertion_z_range_above_surface", [1.0, 3.0])
    insertion_z_range = (top_z + z_lo_rel, top_z + z_hi_rel)

    # ---- calculator + system factory ---------------------------------------
    calc_cfg = config.get("calculator", {"type": "EMT"})
    if calc_cfg.get("type", "EMT").upper() == "LJ" and len(frozen_positions) > 0:
        print(
            "WARNING: calculator.type is 'LJ' (the fast native LJCalculator), which only "
            "ever evaluates interactions between MOBILE particles -- it has no notion of a "
            f"frozen substrate, so the {len(frozen_positions)} atom(s) read from "
            f"{struct_cfg['file']!r} will be silently ignored: gas atoms will feel no "
            "attraction or repulsion from them at all. 'LJ' is only meaningful here for "
            "testing the pipeline's mechanics (expect no real adsorption behavior); use "
            "calculator.type: EMT for physically meaningful results with a real structure."
        )

    def make_system(rng: np.random.Generator) -> SlabSystem:
        calc = _build_calculator(calc_cfg, cell=[Lx, Ly, Lz], pbc=(True, True, False))
        return SlabSystem(Lx, Ly, Lz, calc, exchange_species=gas_species, wall_epsilon=0.0,
                             frozen_positions=frozen_positions, frozen_species=frozen_species,
                             insertion_z_range=insertion_z_range)

    # ---- replica exchange ----------------------------------------------
    re_cfg = config["replica_exchange"]
    temperatures_K = list(re_cfg["temperatures_K"])
    temperatures_eV = [kelvin_to_eV(T) for T in temperatures_K]
    lambda_cubed_of_T = ThermalWavelengthCubed(gas_mass_amu)

    out_cfg = config.get("output", {})
    out_dir = out_cfg.get("directory", "output")
    os.makedirs(out_dir, exist_ok=True)

    traj_cfg = config.get("trajectory", {})
    traj_dir = (os.path.join(out_dir, traj_cfg.get("directory", "trajectories"))
                if traj_cfg.get("save") else None)

    with ReplicaExchangeGCMC(temperatures_eV, re_cfg["mu_eV"], make_system,
                              seed=re_cfg.get("seed", 0),
                              max_disp=re_cfg.get("max_disp", 0.3),
                              n_workers=re_cfg.get("n_workers", 1),
                              lambda_cubed_of_T=lambda_cubed_of_T) as sim:
        scheme = re_cfg.get("scheme", "sweep")
        if scheme == "regc":  # [ZSG] Fig. 1 loop, see ReplicaExchangeGCMC.run_regc
            sim.run_regc(n_steps=re_cfg["n_steps"],
                         x0=re_cfg.get("x0", 0.99),
                         diffusion=re_cfg.get("diffusion", "mc"),
                         diffusion_steps=re_cfg.get("diffusion_steps", 10),
                         record_every=re_cfg.get("record_every", 100),
                         burn_in_steps=re_cfg.get("burn_in_steps", 0),
                         convergence_every=re_cfg.get("convergence_every"),
                         convergence_tol=re_cfg.get("convergence_tol", 0.05),
                         verbose=verbose,
                         trajectory_dir=traj_dir,
                         trajectory_every=traj_cfg.get("every"),
                         **re_cfg.get("md", {}))
        elif scheme == "sweep":
            sim.run(n_cycles=re_cfg["n_cycles"],
                     sweeps_per_cycle=re_cfg.get("sweeps_per_cycle", 100),
                     exchange_every=re_cfg.get("exchange_every", 1),
                     record_every=re_cfg.get("record_every", 5),
                     burn_in_cycles=re_cfg.get("burn_in_cycles", 0),
                     verbose=verbose,
                     trajectory_dir=traj_dir,
                     trajectory_every=traj_cfg.get("every"),
                     diagonal_swaps=re_cfg.get("diagonal_swaps", True))
        else:
            raise ValueError(f"replica_exchange.scheme must be 'sweep' or 'regc', got {scheme!r}")

    # ---- gas-phase reservoir -----------------------------------------------
    diat_cfg = config.get("diatomic_reservoir", {})
    D0 = diat_cfg.get("D0_eV")
    if D0 is not None:
        D0 = _as_float(D0, "diatomic_reservoir.D0_eV")
    if D0 is None:
        raw_calc_factory = _raw_ase_calculator_factory(calc_cfg)
        if raw_calc_factory is None:
            raise ValueError(
                "diatomic_reservoir.D0_eV must be given explicitly in the config when "
                "calculator.type isn't one with an automatic self-consistent D0 wired up "
                "here (only EMT currently has one)."
            )
        D0 = diatomic_dissociation_energy_eV(raw_calc_factory(), gas_species,
                                              mass_amu=gas_mass_amu)["D0_eV"]
        if verbose:
            print(f"Self-consistent D0 ({calc_cfg.get('type', 'EMT')}): {D0:.4f} eV")
    # Zero of the GCMC energy scale: the calculator's energy of one isolated gas
    # atom (EMT's H: +3.21 eV; LJCalculator: 0). The reservoir mu_atom must be on
    # that same scale -- see reservoir.py's module docstring.
    e_atom = diat_cfg.get("e_atom_eV")
    if e_atom is not None:
        e_atom = _as_float(e_atom, "diatomic_reservoir.e_atom_eV")
    else:
        probe = SlabSystem(Lx, Ly, Lz, _build_calculator(calc_cfg, cell=[Lx, Ly, Lz],
                                                          pbc=(True, True, False)),
                             exchange_species=gas_species, positions=[[Lx / 2, Ly / 2, Lz / 2]])
        e_atom = float(probe.total_energy())
    if verbose:
        print(f"Isolated {gas_species} atom energy (calculator's zero for mu): {e_atom:.4f} eV")
    reservoir = DiatomicReservoir(
        D0_eV=D0, e_atom_eV=e_atom,
        mass_molecule_amu=diat_cfg.get("mass_molecule_amu") or 2.0 * gas_mass_amu,
        bond_length_angstrom=diat_cfg.get("bond_length_angstrom", 0.741),
        vib_wavenumber_cm=diat_cfg.get("vibrational_wavenumber_cm", 4401.2),
        symmetry_number=diat_cfg.get("symmetry_number", 2))
    analysis = MBARSurfaceAnalysis(sim, reservoir=reservoir, pressure_unit="Pa")

    # ---- phase diagram -----------------------------------------------------
    pd_cfg = config["phase_diagram"]
    T_grid_K = np.linspace(pd_cfg["T_min_K"], pd_cfg["T_max_K"], pd_cfg.get("n_T", 15))
    T_grid_eV = np.array([kelvin_to_eV(T) for T in T_grid_K])
    p_min_atm = _as_float(pd_cfg["p_min_atm"], "phase_diagram.p_min_atm")
    p_max_atm = _as_float(pd_cfg["p_max_atm"], "phase_diagram.p_max_atm")
    p_grid_atm = np.geomspace(p_min_atm, p_max_atm, pd_cfg.get("n_p", 40))
    area = Lx * Ly

    theta_grid = analysis.phase_diagram_Tp(T_grid_eV, p_grid_atm * 101325.0, area=area)

    plot_path = os.path.join(out_dir, out_cfg.get("plot_filename", "phase_diagram.png"))
    plot_phase_diagram(T_grid_K, p_grid_atm, theta_grid, save_path=plot_path)

    if out_cfg.get("save_raw_data", True):
        np.savez(os.path.join(out_dir, "phase_diagram_data.npz"),
                 T_K=T_grid_K, p_atm=p_grid_atm, theta=theta_grid)

    if verbose:
        print(f"Done. Outputs in {out_dir}/")

    return {"sim": sim, "analysis": analysis, "theta_grid": theta_grid,
            "T_grid_K": T_grid_K, "p_grid_atm": p_grid_atm, "D0_eV": D0,
            "e_atom_eV": e_atom}



def main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``gcmc-re-run config.yaml``.

    Runs the whole structure -> REGC -> MBAR -> (T, p) phase diagram
    pipeline of `run_from_config` for one config file.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (without the program name). Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status (0 on success).
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="gcmc-re-run",
        description="Structure -> replica-exchange GCMC -> MBAR -> (T, p) phase diagram, "
                    "from a YAML/JSON config (see examples/pd111_emt_example.yaml).")
    parser.add_argument("config", help="config file (.yaml/.yml or .json)")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    run_from_config(load_config(args.config), verbose=not args.quiet)
    return 0
