"""
Config-file-driven end-to-end runner: structure -> REGC -> MBAR -> (T, p)
surface phase diagram, all from one YAML or JSON file.

``gcmc-re-run config.yaml`` (or `run_from_config` on a dict) does what
``examples/13_fe_h_eam_surface.py`` does in Python:

1. **structure** -- read any ASE-readable slab (``structure.file``) or build
   one with an ``ase.build`` surface function (``structure.build``, lattice
   constant optionally fitted to the potential, ``a: auto``); lay it out
   with `slab_geometry_from_atoms` (bottom ``n_frozen_layers`` rigid, the
   rest displaceable, ``headroom_angstrom`` of space above). Orthogonal and
   oblique (e.g. hexagonal fcc(111)) cells both work.
2. **calculator** -- ``EMT``, ``LJ``, ``PairLJ``, ``LAMMPS`` (any pair style:
   an EAM file, GRACE, ACE, ...; native MD) or ``ASE`` (any ASE calculator
   class by import path, e.g. ``ase.calculators.eam.EAM``).
3. **reservoir** -- a `DiatomicReservoir` (H2, O2, ...; D0 and the
   isolated-atom energy from the potential itself, from experiment, or
   given) or a `MonatomicIdealGasReservoir`; one per species for a gas
   feed of several species (``gas.composition``).
4. **replica_exchange** -- a (T x p) ladder (``pressures``, mu from the
   reservoir) or an explicit (T x mu) one (``mu_eV``); the [ZSG] REGC
   scheme (``scheme: regc``, MC/HMC/MD diffusion) or the sweep scheme;
   optional ``adsorbed_cutoff`` to record the adsorbed count ``N_ads``.
5. **checkpoint** (regc only) -- run in chunks with `run_with_checkpoints`
   (``checkpoint.pkl``/``samples.npz``), resumable (``--resume``) and
   stopping cleanly before ``--time-limit`` (exit code 75, `EXIT_RESUBMIT`).
6. **phase_diagram** -- MBAR-reweighted coverage (of ``N``, ``N_ads`` or one
   species ``N_<sp>``) on a (T, p) mesh, per area or per surface atom, and
   optionally the most probable N; re-runnable on saved samples with
   ``--analyze-only``.

The run always writes ``samples.npz`` and ``run_metadata.json`` (everything
the analysis needs: ladder, reservoirs, area) to ``output.directory``.

See ``examples/pd111_emt_example.yaml`` (small, EMT) and
``examples/fe100_h_eam_example.yaml`` (LAMMPS EAM, the setup of example 13)
for annotated configs covering every key. Unknown keys raise an error, so
typos do not pass silently. Configs written for earlier versions of this
runner (``diatomic_reservoir``, ``p_min_atm``, scheme inferred from
``n_cycles``/``n_steps``) still work.

Relative paths -- ``structure.file``, ``calculator.potential``, files named
in ``calculator.pair_coeff``/``pair_commands`` and string values of
``calculator.kwargs`` -- are resolved against the config file's directory
when such a file exists there. ``output.directory`` is relative to the
working directory.

A YAML gotcha: PyYAML (YAML 1.1) reads ``1.0e-4`` as a float but ``1.0e6``
(no sign on the exponent) as the STRING ``"1.0e6"``. Every numeric field
here is coerced with Python's ``float()``, so this self-corrects, but
writing the sign (``1.0e+6``) is safest.

Requires: pip install ase pymbar matplotlib, plus pyyaml for .yaml/.yml
configs and the LAMMPS Python module for ``calculator.type: LAMMPS``.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: One time unit of (Angstrom, amu, eV), in fs: HMC timesteps given in fs are divided by it.
FS_PER_INTERNAL_TIME = 10.1805

_SECTIONS = {
    "structure": {"file", "build", "n_frozen_layers", "headroom_angstrom", "layer_tol"},
    "gas": {"species", "mass_amu", "insertion_z_range_above_surface", "composition"},
    "reservoir": {"type", "d0", "D0_eV", "e_atom_eV", "bond_length_angstrom",
                  "vibrational_wavenumber_cm", "symmetry_number", "mass_molecule_amu", "mass_amu"},
    "trajectory": {"save", "directory", "every"},
    "checkpoint": {"chunk", "time_limit_s"},
    "phase_diagram": {"T_min_K", "T_max_K", "n_T", "p_min", "p_max", "n_p", "pressure_unit",
                      "p_min_atm", "p_max_atm", "observable", "coverage_per", "most_probable",
                      "discard"},
    "output": {"directory", "plot_filename", "save_raw_data"},
}
_TOP_LEVEL = set(_SECTIONS) | {"calculator", "replica_exchange", "diatomic_reservoir"}
_CALCULATOR_KEYS = {
    "EMT": set(),
    "LJ": {"lj_sigma", "lj_epsilon", "lj_r_cut"},
    "PAIRLJ": {"pair_params", "r_cut_factor"},
    "LAMMPS": {"species", "potential", "pair_style", "pair_coeff", "pair_commands",
               "min_distance", "z_padding"},
    "ASE": {"class", "kwargs"},
}
_RE_COMMON = {"temperatures_K", "mu_eV", "pressures", "pressure_unit", "scheme", "seed",
              "max_disp", "n_workers", "adsorbed_cutoff", "hmc", "record_every"}
_RE_SCHEME = {
    "regc": {"n_steps", "x0", "diffusion", "diffusion_steps", "burn_in_steps",
             "convergence_every", "convergence_tol", "swap_types", "displace_substrate", "md"},
    "sweep": {"n_cycles", "sweeps_per_cycle", "exchange_every", "burn_in_cycles",
              "diagonal_swaps", "p_displace", "p_insert", "p_delete", "p_displace_substrate"},
}
_HMC_KEYS = {"enabled", "leapfrog_steps", "timestep_fs", "dt"}
_MD_KEYS = {"timestep_fs", "thermostat", "taut_fs", "friction_per_fs"}


# =============================================================================
# Config file handling
# =============================================================================
def _resolve_existing(value: str, base: str) -> str:
    """
    Make a relative path absolute if it names a file under `base`.

    Parameters
    ----------
    value : str
        A path, or any other string.
    base : str
        Directory to resolve against.

    Returns
    -------
    str
        ``base/value`` if `value` is relative and that file exists, else
        `value` unchanged.
    """
    if isinstance(value, str) and value and not os.path.isabs(value):
        candidate = os.path.join(base, value)
        if os.path.exists(candidate):
            return candidate
    return value


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
        against the config file's own directory, and so are the potential
        files of the ``calculator`` section when they exist there (see the
        module docstring), so the config works from any working directory.
    """
    with open(path) as f:
        text = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml  # pip install pyyaml
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    base = os.path.dirname(os.path.abspath(path))
    struct = config.get("structure", {})
    if "file" in struct and not os.path.isabs(struct["file"]):
        struct["file"] = os.path.join(base, struct["file"])
    calc = config.get("calculator")
    if isinstance(calc, dict):
        if "potential" in calc:
            calc["potential"] = _resolve_existing(calc["potential"], base)
        if isinstance(calc.get("pair_coeff"), str):
            calc["pair_coeff"] = " ".join(_resolve_existing(t, base)
                                          for t in calc["pair_coeff"].split())
        if isinstance(calc.get("pair_commands"), list):
            calc["pair_commands"] = [" ".join(_resolve_existing(t, base) for t in str(c).split())
                                     for c in calc["pair_commands"]]
        if isinstance(calc.get("kwargs"), dict):
            calc["kwargs"] = {k: _resolve_existing(v, base) for k, v in calc["kwargs"].items()}
    return config


def _as_float(value: Any, field_name: str) -> float:
    """
    Coerce a config value to float, with a clear error message.

    Rather than a confusing downstream one, if it came through as a
    string -- the classic symptom of the YAML unsigned-exponent gotcha
    documented in this module's docstring (which ``float()`` corrects).

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


def _floats(values: Any, field_name: str) -> List[float]:
    """
    Coerce a list of config values to floats (see `_as_float`).

    Parameters
    ----------
    values : sequence
        The config values.
    field_name : str
        The config field's name, for error messages.

    Returns
    -------
    list of float
        The coerced values.

    Raises
    ------
    ValueError
        If `values` is not a list.
    """
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Config field {field_name!r} must be a list, got {values!r}")
    return [_as_float(v, f"{field_name}[{i}]") for i, v in enumerate(values)]


def _check_keys(section: Any, allowed: set, name: str) -> None:
    """
    Reject unknown keys in a config section.

    Parameters
    ----------
    section : dict
        The section.
    allowed : set of str
        Its allowed keys.
    name : str
        The section's name, for the error message.

    Raises
    ------
    ValueError
        If `section` is not a mapping or has a key not in `allowed`.
    """
    if not isinstance(section, dict):
        raise ValueError(f"config section {name!r} must be a mapping, got {section!r}")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in config section {name!r}; "
                         f"allowed: {sorted(allowed)}")


def _scheme(re_cfg: Dict[str, Any]) -> str:
    """
    The sampling scheme of the ``replica_exchange`` section.

    Parameters
    ----------
    re_cfg : dict
        The section.

    Returns
    -------
    str
        ``'regc'`` or ``'sweep'``: ``scheme`` if given, else inferred
        (``n_steps`` -> regc, otherwise sweep, the old default).

    Raises
    ------
    ValueError
        If ``scheme`` is neither.
    """
    scheme = re_cfg.get("scheme", "regc" if "n_steps" in re_cfg else "sweep")
    if scheme not in _RE_SCHEME:
        raise ValueError(f"replica_exchange.scheme must be 'sweep' or 'regc', got {scheme!r}")
    return scheme


def validate_config(config: Dict[str, Any]) -> None:
    """
    Check a config's sections and keys before anything expensive runs.

    Parameters
    ----------
    config : dict
        The parsed config.

    Raises
    ------
    ValueError
        On an unknown section or key, a missing required section, or
        keys of the other sampling scheme.
    """
    _check_keys(config, _TOP_LEVEL, "<top level>")
    for name in ("structure", "gas", "replica_exchange"):
        if name not in config:
            raise ValueError(f"config needs a {name!r} section")
    for name, allowed in _SECTIONS.items():
        if name in config and name != "reservoir":
            _check_keys(config[name], allowed, name)
    for name in ("reservoir", "diatomic_reservoir"):
        if name in config:
            for sp, sec in _per_species_reservoir_cfg(config, _species_list(config)).items():
                _check_keys(sec, _SECTIONS["reservoir"], f"{name}.{sp}" if sp else name)
    calc = config.get("calculator", {"type": "EMT"})
    ctype = str(calc.get("type", "EMT")).upper()
    if ctype not in _CALCULATOR_KEYS:
        raise ValueError(f"unknown calculator.type {calc.get('type')!r}; "
                         f"use one of EMT, LJ, PairLJ, LAMMPS, ASE")
    _check_keys(calc, _CALCULATOR_KEYS[ctype] | {"type"}, "calculator")
    re_cfg = config["replica_exchange"]
    scheme = _scheme(re_cfg)
    _check_keys(re_cfg, _RE_COMMON | _RE_SCHEME[scheme], f"replica_exchange (scheme {scheme})")
    if "hmc" in re_cfg:
        _check_keys(re_cfg["hmc"], _HMC_KEYS, "replica_exchange.hmc")
    if "md" in re_cfg:
        _check_keys(re_cfg["md"], _MD_KEYS, "replica_exchange.md")
    if ("mu_eV" in re_cfg) == ("pressures" in re_cfg):
        raise ValueError("replica_exchange needs exactly one of 'mu_eV' (explicit chemical "
                         "potentials) or 'pressures' (gas-pressure ladder via the reservoir)")
    if "checkpoint" in config and scheme != "regc":
        raise ValueError("the checkpoint section needs replica_exchange.scheme: regc")
    if "build" in config["structure"] and "file" in config["structure"]:
        raise ValueError("structure needs one of 'file' or 'build', not both")
    if "build" not in config["structure"] and "file" not in config["structure"]:
        raise ValueError("structure needs 'file' (an ASE-readable slab) or 'build'")


# =============================================================================
# Species, calculators
# =============================================================================
def _species_list(config: Dict[str, Any]) -> List[str]:
    """
    The exchanged species of a config, as a list.

    Parameters
    ----------
    config : dict
        The parsed config.

    Returns
    -------
    list of str
        ``gas.species`` (a string or a list) as a list.
    """
    sp = config["gas"]["species"]
    return [sp] if isinstance(sp, str) else [str(s) for s in sp]


def _gas_masses(gas_cfg: Dict[str, Any], species: Sequence[str]) -> Dict[str, float]:
    """
    Mass of every exchanged species.

    Parameters
    ----------
    gas_cfg : dict
        The ``gas`` section; ``mass_amu`` is a number (one species) or a
        ``{species: mass}`` mapping, optional.
    species : sequence of str
        The exchanged species.

    Returns
    -------
    dict
        ``{species: mass_amu}``, ASE's tables for those not given.
    """
    from .units import default_mass_amu
    given = gas_cfg.get("mass_amu")
    if given is not None and not isinstance(given, dict):
        given = {species[0]: given}
    given = given or {}
    return {s: (_as_float(given[s], f"gas.mass_amu.{s}") if given.get(s) is not None
                else default_mass_amu(s)) for s in species}


class _CalculatorSpec:
    """
    The ``calculator`` section: builds simulation and setup calculators.

    Parameters
    ----------
    cfg : dict
        The ``calculator`` section.
    species : sequence of str
        Every species the simulation contains (slab, then gas), the
        default LAMMPS atom-type order.
    masses : dict
        Masses (amu) of the exchanged species, for LAMMPS.
    """

    def __init__(self, cfg: Dict[str, Any], species: Sequence[str], masses: Dict[str, float]):
        """
        Parse the section.

        Parameters
        ----------
        cfg : dict
            The ``calculator`` section.
        species : sequence of str
            Every species in the simulation (default LAMMPS type order).
        masses : dict
            Masses (amu) of the exchanged species.

        Raises
        ------
        ValueError
            If a LAMMPS calculator has no potential definition, or
            ``pair_style``/``pair_coeff`` without ``species``.
        """
        self.cfg = dict(cfg)
        self.type = str(cfg.get("type", "EMT")).upper()
        self.masses = dict(masses)
        if self.type == "LAMMPS":
            from .lammps_backend import eam_pair_commands
            explicit = cfg.get("pair_commands") or cfg.get("pair_style")
            if explicit and "species" not in cfg:
                raise ValueError("calculator.species (LAMMPS atom-type order, matching "
                                 "pair_coeff) is required with pair_style/pair_commands")
            self.lammps_species = [str(s) for s in cfg.get("species", species)]
            if cfg.get("pair_commands"):
                self.pair_commands = [str(c) for c in cfg["pair_commands"]]
            elif cfg.get("pair_style"):
                if not cfg.get("pair_coeff"):
                    raise ValueError("calculator.pair_style needs calculator.pair_coeff")
                self.pair_commands = [f"pair_style {cfg['pair_style']}",
                                      f"pair_coeff {cfg['pair_coeff']}"]
            elif cfg.get("potential"):
                self.pair_commands = eam_pair_commands(cfg["potential"], self.lammps_species)
            else:
                raise ValueError("calculator.type LAMMPS needs 'potential' (an *.eam.fs / "
                                 "*.eam.alloy file), 'pair_style' + 'pair_coeff', or "
                                 "'pair_commands'")

    def _ase_factory(self) -> Optional[Callable[[], Any]]:
        """
        Zero-argument factory of the raw ASE calculator, if there is one.

        Returns
        -------
        callable or None
            EMT's class, a `functools.partial` of the configured ASE class,
            or None (LJ, PairLJ, LAMMPS).

        Raises
        ------
        ValueError
            If ``calculator.class`` is missing or not a dotted import path.
        """
        if self.type == "EMT":
            from ase.calculators.emt import EMT
            return EMT
        if self.type == "ASE":
            path = self.cfg.get("class")
            if not path or "." not in str(path):
                raise ValueError("calculator.type ASE needs 'class', a dotted import path "
                                 "such as ase.calculators.eam.EAM")
            module, name = str(path).rsplit(".", 1)
            return functools.partial(getattr(importlib.import_module(module), name),
                                     **dict(self.cfg.get("kwargs") or {}))
        return None

    def ase_calculator(self):
        """
        The potential as an ASE calculator, for setup work.

        Lattice-constant fits and the reservoir's molecule
        (`DiatomicReservoir.from_calculator`) use it.

        Returns
        -------
        ase.calculators.calculator.Calculator or None
            None for the native LJ calculators, which have no ASE
            counterpart.
        """
        if self.type == "LAMMPS":
            from .lammps_backend import lammps_ase_calculator
            return lammps_ase_calculator(self.pair_commands, self.lammps_species)
        factory = self._ase_factory()
        return factory() if factory is not None else None

    def make(self, cell: np.ndarray):
        """
        A fresh simulation `Calculator` (each system needs its own).

        Parameters
        ----------
        cell : numpy.ndarray
            3x3 cell matrix (rows a1, a2, a3) for ASE-backed calculators;
            periodic in x and y, open in z.

        Returns
        -------
        Calculator
            The calculator.
        """
        if self.type == "LJ":
            from .calculators import LJCalculator
            return LJCalculator(sigma=_as_float(self.cfg.get("lj_sigma", 1.0), "calculator.lj_sigma"),
                                epsilon=_as_float(self.cfg.get("lj_epsilon", 0.1),
                                                  "calculator.lj_epsilon"),
                                r_cut=_as_float(self.cfg.get("lj_r_cut", 3.0), "calculator.lj_r_cut"))
        if self.type == "PAIRLJ":
            from .calculators import PairLJCalculator
            params = {(str(a), str(b)): (_as_float(e, "calculator.pair_params"),
                                         _as_float(s, "calculator.pair_params"))
                      for a, b, e, s in self.cfg["pair_params"]}
            return PairLJCalculator(params, r_cut_factor=_as_float(
                self.cfg.get("r_cut_factor", 2.5), "calculator.r_cut_factor"))
        if self.type == "LAMMPS":
            from .lammps_backend import LammpsCalculator
            md = self.cfg.get("min_distance")
            return LammpsCalculator(self.pair_commands, self.lammps_species, masses=self.masses,
                                    min_distance=None if md is None else _as_float(
                                        md, "calculator.min_distance"),
                                    z_padding=_as_float(self.cfg.get("z_padding", 5.0),
                                                        "calculator.z_padding"))
        from .calculators import ASECalculatorAdapter
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell - np.diag(np.diag(cell)), 0.0):
            cell = np.diag(cell)
        return ASECalculatorAdapter(self._ase_factory(), cell=cell, pbc=(True, True, False))

    def isolated_atom_energy(self, species: str) -> float:
        """
        The calculator's energy of one isolated atom (the zero of mu).

        Parameters
        ----------
        species : str
            The atom's species.

        Returns
        -------
        float
            Energy, in eV (0 for pair potentials).
        """
        from .systems import SlabSystem
        L = 30.0
        system = SlabSystem(L, L, L, self.make(np.diag([L, L, L])), exchange_species=species,
                            positions=[[L / 2, L / 2, L / 2]])
        return float(system.total_energy())


# =============================================================================
# Structure
# =============================================================================
def _build_structure(struct_cfg: Dict[str, Any], calc_spec_of: Callable[[List[str]], _CalculatorSpec]
                     ) -> Tuple[Any, _CalculatorSpec, Optional[float]]:
    """
    Read or build the slab, and the calculator spec that goes with it.

    Parameters
    ----------
    struct_cfg : dict
        The ``structure`` section.
    calc_spec_of : callable
        ``calc_spec_of(slab_species) -> _CalculatorSpec``.

    Returns
    -------
    tuple
        ``(atoms, calc_spec, lattice_constant)``; the lattice constant is
        None for a structure read from a file.

    Raises
    ------
    ValueError
        If ``build`` lacks ``surface``/``symbol``, ``a: auto`` is asked for
        a non-cubic crystal or a potential with no ASE counterpart, or
        ``ase.build`` has no such surface function.
    """
    if "file" in struct_cfg:
        from ase.io import read as ase_read
        atoms = ase_read(struct_cfg["file"])
        slab_species = list(dict.fromkeys(atoms.get_chemical_symbols()))
        return atoms, calc_spec_of(slab_species), None

    import ase.build
    build = dict(struct_cfg["build"])
    for key in ("surface", "symbol"):
        if key not in build:
            raise ValueError(f"structure.build needs {key!r} (e.g. surface: fcc111, symbol: Pd)")
    surface, symbol = str(build.pop("surface")), str(build.pop("symbol"))
    builder = getattr(ase.build, surface, None)
    if builder is None:
        raise ValueError(f"ase.build has no surface function {surface!r}")
    calc_spec = calc_spec_of([symbol])
    a = build.pop("a", None)
    a_range = build.pop("a_range", None)
    if a == "auto":
        from .structures import cubic_lattice_constant
        crystal = surface[:3]
        if crystal not in ("fcc", "bcc"):
            raise ValueError(f"structure.build.a: auto needs an fcc/bcc surface, got {surface!r}")
        ase_calc = calc_spec.ase_calculator()
        if ase_calc is None:
            raise ValueError("structure.build.a: auto needs a calculator with an ASE "
                             "counterpart (EMT, LAMMPS, ASE); give 'a' explicitly")
        a_range = tuple(_floats(a_range, "structure.build.a_range")) if a_range else (2.3, 5.5)
        a = cubic_lattice_constant(ase_calc, symbol, crystal, a_range=a_range, n_coarse=97)
    elif a is not None:
        a = _as_float(a, "structure.build.a")
    if "size" in build:
        build["size"] = tuple(int(n) for n in build["size"])
    atoms = builder(symbol, a=a, **build) if a is not None else builder(symbol, **build)
    return atoms, calc_spec, a


# =============================================================================
# Reservoirs and the gas feed
# =============================================================================
def _per_species_reservoir_cfg(config: Dict[str, Any], species: Sequence[str]
                               ) -> Dict[str, Dict[str, Any]]:
    """
    The reservoir section(s), keyed by species.

    Parameters
    ----------
    config : dict
        The parsed config (``reservoir``, or the legacy
        ``diatomic_reservoir``, flat for one species or keyed by species).
    species : sequence of str
        The exchanged species.

    Returns
    -------
    dict
        ``{species: section}`` (an empty section where none is given);
        with a flat section for one species, the key is that species.

    Raises
    ------
    ValueError
        If several species are exchanged but the section is not keyed by
        species.
    """
    legacy = "reservoir" not in config and "diatomic_reservoir" in config
    sec = config.get("reservoir", config.get("diatomic_reservoir")) or {}
    if legacy:
        sec = dict(sec, type="diatomic")
    keyed = bool(sec) and all(k in species and isinstance(v, dict) for k, v in sec.items())
    if keyed:
        out = {s: dict(sec.get(s, {})) for s in species}
        if legacy:
            for v in out.values():
                v["type"] = "diatomic"
        return out
    if len(species) > 1 and sec:
        raise ValueError(f"with several exchanged species {list(species)}, the reservoir "
                         f"section must be keyed by species (reservoir: {{{species[0]}: ..., ...}})")
    return {s: dict(sec) for s in species}


def _build_reservoir(sp: str, cfg: Dict[str, Any], mass_amu: float, calc_spec: _CalculatorSpec,
                     ase_calc_getter: Callable[[], Any]):
    """
    The gas reservoir of one exchanged species.

    Parameters
    ----------
    sp : str
        The species.
    cfg : dict
        Its reservoir section.
    mass_amu : float
        Its atomic mass.
    calc_spec : _CalculatorSpec
        The calculator (for the isolated-atom energy).
    ase_calc_getter : callable
        Returns the potential as an ASE calculator (or None); called only
        when the potential's own molecule is needed.

    Returns
    -------
    Reservoir
        A `DiatomicReservoir` or `MonatomicIdealGasReservoir`.

    Raises
    ------
    ValueError
        If the type is unknown, or the potential's own D0 is asked for
        with a calculator that has no ASE counterpart (give ``D0_eV``).
    """
    from .reservoir import (H2_BOND_LENGTH_ANGSTROM, H2_D0_EXPERIMENTAL_EV,
                            H2_VIBRATIONAL_WAVENUMBER_CM, DiatomicReservoir,
                            MonatomicIdealGasReservoir)
    rtype = str(cfg.get("type", "diatomic")).lower()
    e_atom = cfg.get("e_atom_eV")
    e_atom = None if e_atom is None else _as_float(e_atom, f"reservoir.{sp}.e_atom_eV")
    if rtype == "monatomic":
        m = _as_float(cfg.get("mass_amu", mass_amu), f"reservoir.{sp}.mass_amu")
        return MonatomicIdealGasReservoir(m, e_atom if e_atom is not None
                                          else calc_spec.isolated_atom_energy(sp))
    if rtype != "diatomic":
        raise ValueError(f"reservoir.type must be 'diatomic' or 'monatomic', got {rtype!r}")

    d0 = cfg.get("D0_eV", cfg.get("d0", "self"))
    if d0 is None:
        d0 = "self"
    if d0 not in ("self", "experimental"):
        d0 = _as_float(d0, f"reservoir.{sp}.D0_eV")
    bond = cfg.get("bond_length_angstrom")
    vib = cfg.get("vibrational_wavenumber_cm")
    bond = None if bond is None else _as_float(bond, f"reservoir.{sp}.bond_length_angstrom")
    vib = None if vib is None else _as_float(vib, f"reservoir.{sp}.vibrational_wavenumber_cm")
    m_mol = cfg.get("mass_molecule_amu")
    m_mol = 2.0 * mass_amu if m_mol is None else _as_float(m_mol, f"reservoir.{sp}.mass_molecule_amu")
    sym = int(cfg.get("symmetry_number", 2))

    ase_calc = ase_calc_getter() if (d0 == "self" or e_atom is None) else None
    if d0 == "self" and ase_calc is None:
        raise ValueError(
            f"reservoir D0_eV for {sp} must be given explicitly (e.g. D0_eV: 4.478, or "
            f"d0: experimental for H) when calculator.type is "
            f"{calc_spec.type} -- it has no ASE counterpart to evaluate the molecule with.")
    if ase_calc is not None:
        res = DiatomicReservoir.from_calculator(ase_calc, sp, d0=d0, bond_length_angstrom=bond,
                                                vib_wavenumber_cm=vib, mass_molecule_amu=m_mol,
                                                symmetry_number=sym)
        return res if e_atom is None else dataclasses.replace(res, e_atom_eV=e_atom)
    if d0 == "experimental":
        if sp != "H":
            raise ValueError("d0: experimental is only tabulated for H2; give D0_eV")
        d0 = H2_D0_EXPERIMENTAL_EV
    if sp == "H":
        bond = bond or H2_BOND_LENGTH_ANGSTROM
        vib = vib or H2_VIBRATIONAL_WAVENUMBER_CM
    if bond is None or vib is None:
        raise ValueError(f"reservoir.{sp} needs bond_length_angstrom and "
                         f"vibrational_wavenumber_cm for {sp}2")
    if e_atom is None:
        e_atom = calc_spec.isolated_atom_energy(sp)
    return DiatomicReservoir(float(d0), e_atom, m_mol, bond, vib, sym)


def _reservoir_to_dict(res) -> Dict[str, Any]:
    """
    A reservoir as a JSON-able dict (for ``run_metadata.json``).

    Parameters
    ----------
    res : Reservoir
        A `DiatomicReservoir` or `MonatomicIdealGasReservoir`.

    Returns
    -------
    dict
        Its type and parameters (diagnostics dropped).
    """
    from .reservoir import DiatomicReservoir
    d = {f.name: getattr(res, f.name) for f in dataclasses.fields(res)
         if f.name != "calculator_info"}
    d["type"] = "diatomic" if isinstance(res, DiatomicReservoir) else "monatomic"
    return d


def _reservoir_from_dict(d: Dict[str, Any]):
    """
    Rebuild a reservoir written by `_reservoir_to_dict`.

    Parameters
    ----------
    d : dict
        The dict.

    Returns
    -------
    Reservoir
        The reservoir.
    """
    from .reservoir import DiatomicReservoir, MonatomicIdealGasReservoir
    d = dict(d)
    cls = DiatomicReservoir if d.pop("type") == "diatomic" else MonatomicIdealGasReservoir
    return cls(**d)


class GasFeed:
    """
    One or several gas reservoirs at a fixed feed composition.

    The chemical potentials at total pressure p are those of every
    species' reservoir at its partial pressure x_s * p: a single number
    for one species, one per species (in `species` order) for several --
    what `ReplicaExchangeGCMC`'s ``mus`` and `MBARSurfaceAnalysis`'s
    ``mu_of_Tp`` take.

    Parameters
    ----------
    species : sequence of str
        The exchanged species.
    reservoirs : sequence of Reservoir
        One per species.
    fractions : sequence of float
        Feed mole fractions, one per species.
    """

    def __init__(self, species: Sequence[str], reservoirs: Sequence[Any], fractions: Sequence[float]):
        """
        Store the reservoirs.

        Parameters
        ----------
        species : sequence of str
            The exchanged species.
        reservoirs : sequence of Reservoir
            One per species.
        fractions : sequence of float
            Feed mole fractions, one per species.
        """
        self.species = list(species)
        self.reservoirs = list(reservoirs)
        self.fractions = [float(x) for x in fractions]

    def mu_at_T_eV(self, T_eV: float, p: float, unit: str = "Pa"):
        """
        Chemical potential(s) at temperature k_B*T and total pressure p.

        Parameters
        ----------
        T_eV : float
            k_B * T, in eV.
        p : float
            Total pressure, in `unit`.
        unit : str, optional
            Pressure unit. Default ``'Pa'``.

        Returns
        -------
        float or numpy.ndarray
            mu (eV) for one species, an array of one per species otherwise.
        """
        mus = [r.mu_at_T_eV(T_eV, x * p, unit) for r, x in zip(self.reservoirs, self.fractions)]
        return mus[0] if len(mus) == 1 else np.array(mus)

    def mu_grid(self, temperatures_K: Sequence[float], pressures: Sequence[float],
                unit: str = "Pa") -> np.ndarray:
        """
        Chemical potentials of a (temperature x pressure) replica ladder.

        Parameters
        ----------
        temperatures_K : sequence of float
            Temperatures, in Kelvin.
        pressures : sequence of float
            Total pressures, in `unit`.
        unit : str, optional
            Pressure unit. Default ``'Pa'``.

        Returns
        -------
        numpy.ndarray
            Shape (n_T, n_p) for one species, (n_T, n_p, n_species)
            otherwise.
        """
        grids = [r.mu_grid(temperatures_K, [x * p for p in pressures], unit=unit)
                 for r, x in zip(self.reservoirs, self.fractions)]
        return grids[0] if len(grids) == 1 else np.stack(grids, axis=-1)

    def lambda_cubed_of_T(self):
        """
        Lambda^3(T) of the exchanged particle(s).

        Returns
        -------
        ThermalWavelengthCubed or SpeciesLambdaCubed
            For `ReplicaExchangeGCMC`'s ``lambda_cubed_of_T``.
        """
        from .units import SpeciesLambdaCubed
        lams = [r.lambda_cubed_of_T() for r in self.reservoirs]
        return lams[0] if len(lams) == 1 else SpeciesLambdaCubed(lams)


def _feed_fractions(gas_cfg: Dict[str, Any], species: Sequence[str]) -> List[float]:
    """
    Feed mole fractions from ``gas.composition``.

    Parameters
    ----------
    gas_cfg : dict
        The ``gas`` section.
    species : sequence of str
        The exchanged species.

    Returns
    -------
    list of float
        One fraction per species (1.0 for a single species), normalised to
        sum to 1.

    Raises
    ------
    ValueError
        If several species are exchanged without a composition covering
        each of them.
    """
    comp = gas_cfg.get("composition")
    if comp is None:
        if len(species) > 1:
            raise ValueError("several exchanged species with a pressure ladder or a (T, p) "
                             "phase diagram need gas.composition (feed mole fractions, "
                             "e.g. {O: 0.999, Cl: 0.001})")
        return [1.0]
    missing = [s for s in species if s not in comp]
    if missing:
        raise ValueError(f"gas.composition lacks {missing}")
    x = np.array([_as_float(comp[s], f"gas.composition.{s}") for s in species])
    return list(x / x.sum())


# =============================================================================
# Setup
# =============================================================================
def setup_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Everything a run needs, built from a config but nothing sampled yet.

    Parameters
    ----------
    config : dict
        The parsed config (see the module docstring).
    verbose : bool, optional
        Print the setup's key numbers. Default True.

    Returns
    -------
    dict
        ``geometry`` (`SlabGeometry`), ``lattice_constant`` (A, or None),
        ``calc_spec``, ``species``, ``masses``, ``feed`` (`GasFeed`, or None
        when the run needs no reservoir), ``temperatures_K``, ``mu_grid``,
        ``pressures``/``pressure_unit`` (None for an explicit mu ladder),
        ``lambda_cubed_of_T``, ``insertion_z_range``, ``system_factory``
        and ``metadata`` (JSON-able; what a resumed run must agree with,
        and what `analyze_from_config` reads).

    Raises
    ------
    ValueError
        On an invalid config (see `validate_config`) or an insertion
        z-range outside the box.
    """
    from .structures import slab_geometry_from_atoms
    from .units import ThermalWavelengthCubed

    validate_config(config)
    species = _species_list(config)
    if "phase_diagram" in config:  # fail on a bad mesh now, not after the sampling
        _phase_diagram_grid(config["phase_diagram"],
                            config["replica_exchange"].get("pressure_unit"))
    gas_cfg = config["gas"]
    masses = _gas_masses(gas_cfg, species)
    calc_cfg = config.get("calculator", {"type": "EMT"})

    def calc_spec_of(slab_species):
        return _CalculatorSpec(calc_cfg, list(dict.fromkeys(list(slab_species) + species)), masses)

    # ---- structure -----------------------------------------------------------
    struct_cfg = config["structure"]
    atoms, calc_spec, a = _build_structure(struct_cfg, calc_spec_of)
    nfl = struct_cfg.get("n_frozen_layers")
    geometry = slab_geometry_from_atoms(
        atoms, n_frozen_layers=None if nfl is None else int(nfl),
        headroom=_as_float(struct_cfg.get("headroom_angstrom", 5.0), "structure.headroom_angstrom"),
        layer_tol=_as_float(struct_cfg.get("layer_tol", 1e-3), "structure.layer_tol"))
    cell = np.zeros((3, 3))
    cell[:2, :2], cell[2, 2] = geometry.cell_xy, geometry.Lz

    zr = gas_cfg.get("insertion_z_range_above_surface")
    insertion_z_range = None
    if zr is not None:
        lo, hi = _floats(zr, "gas.insertion_z_range_above_surface")
        insertion_z_range = (geometry.top_z + lo, geometry.top_z + hi)
        if not 0.0 <= insertion_z_range[0] < insertion_z_range[1] <= geometry.Lz + 1e-9:
            raise ValueError(
                f"gas.insertion_z_range_above_surface {zr} gives z in {insertion_z_range}, "
                f"outside the box (0, {geometry.Lz:.3f}] (the top frozen layer is at z = 0, "
                f"the top atom at {geometry.top_z:.3f} A): keep it within "
                f"[{-geometry.top_z:.3f}, {geometry.Lz - geometry.top_z:.3f}] or raise "
                f"structure.headroom_angstrom")
        insertion_z_range = (insertion_z_range[0], min(insertion_z_range[1], geometry.Lz))

    if calc_spec.type == "LJ" and (len(geometry.frozen_positions) or len(geometry.substrate_positions)):
        print("WARNING: calculator.type is 'LJ' (the native LJCalculator), which only evaluates "
              "interactions between MOBILE particles: the slab atoms are ignored, so gas atoms "
              "feel nothing from them. Use it to test the pipeline's mechanics only; PairLJ "
              "(which sees the slab), EMT, LAMMPS or ASE for meaningful results.")

    # ---- reservoir(s) and the replica ladder ------------------------------------
    re_cfg = config["replica_exchange"]
    temperatures_K = _floats(re_cfg["temperatures_K"], "replica_exchange.temperatures_K")
    pressures = (_floats(re_cfg["pressures"], "replica_exchange.pressures")
                 if "pressures" in re_cfg else None)
    pressure_unit = str(re_cfg.get("pressure_unit", "bar"))
    needs_reservoir = (pressures is not None or "phase_diagram" in config
                       or "reservoir" in config or "diatomic_reservoir" in config)
    feed = None
    if needs_reservoir:
        cache = {}

        def ase_calc():
            if "c" not in cache:
                cache["c"] = calc_spec.ase_calculator()
            return cache["c"]
        res_cfgs = _per_species_reservoir_cfg(config, species)
        reservoirs = [_build_reservoir(s, res_cfgs[s], masses[s], calc_spec, ase_calc)
                      for s in species]
        feed = GasFeed(species, reservoirs, _feed_fractions(gas_cfg, species))
    if pressures is not None:
        mu_grid = feed.mu_grid(temperatures_K, pressures, unit=pressure_unit)
    else:
        mu_grid = np.array(re_cfg["mu_eV"], dtype=object)
        mu_grid = np.vectorize(lambda v: _as_float(v, "replica_exchange.mu_eV"), otypes=[float])(mu_grid)
        pressure_unit = None
    if feed is not None:
        lambda_cubed_of_T = feed.lambda_cubed_of_T()
    elif len(species) == 1:
        lambda_cubed_of_T = ThermalWavelengthCubed(masses[species[0]])
    else:
        from .units import SpeciesLambdaCubed
        lambda_cubed_of_T = SpeciesLambdaCubed([ThermalWavelengthCubed(masses[s]) for s in species])

    exchange = species[0] if len(species) == 1 else list(species)

    def system_factory(rng):
        return geometry.make_system(calc_spec.make(cell), exchange, mass_of_species=masses,
                                    insertion_z_range=insertion_z_range)

    scheme = _scheme(re_cfg)
    sampling = {"scheme": scheme, "record_every": re_cfg.get("record_every"),
                "adsorbed_cutoff": re_cfg.get("adsorbed_cutoff"), "max_disp": re_cfg.get("max_disp")}
    if scheme == "regc":
        sampling.update({k: re_cfg.get(k) for k in ("x0", "diffusion", "diffusion_steps",
                                                     "displace_substrate", "md", "hmc")})
    metadata = {
        "species": species, "masses_amu": masses,
        "temperatures_K": temperatures_K, "mu_grid_eV": np.asarray(mu_grid).tolist(),
        "pressures": pressures, "pressure_unit": pressure_unit,
        "composition": dict(zip(species, feed.fractions)) if feed else None,
        "reservoirs": ({s: _reservoir_to_dict(r) for s, r in zip(species, feed.reservoirs)}
                       if feed else None),
        "structure": struct_cfg, "calculator": calc_cfg, "lattice_constant_angstrom": a,
        "n_surface": geometry.n_surface, "area_angstrom2": geometry.area,
        "insertion_z_range": insertion_z_range, "sampling": sampling,
    }

    if verbose:
        if a is not None:
            print(f"lattice constant from the potential: {a:.4f} A")
        print(f"slab: {geometry.summary()}")
        if insertion_z_range is not None:
            print(f"insertions in z in ({insertion_z_range[0]:.2f}, {insertion_z_range[1]:.2f}] A")
        if feed is not None:
            for s, r in zip(species, feed.reservoirs):
                d0 = ""
                if hasattr(r, "D0_eV"):
                    src = (r.calculator_info or {}).get("d0_source", "given")
                    d0 = f"D0 = {r.D0_eV:.4f} eV ({src}), "
                print(f"{s} reservoir ({type(r).__name__}): {d0}e_atom = {r.e_atom_eV:.4f} eV")
        n_mu = np.shape(mu_grid)[-1 if len(species) == 1 else -2]
        print(f"replica grid: {len(temperatures_K)} T x {n_mu} "
              f"{'p' if pressures is not None else 'mu'} = {len(temperatures_K) * n_mu} replicas")

    return {"geometry": geometry, "lattice_constant": a, "calc_spec": calc_spec,
            "species": species, "masses": masses, "feed": feed,
            "temperatures_K": temperatures_K, "mu_grid": mu_grid, "pressures": pressures,
            "pressure_unit": pressure_unit, "lambda_cubed_of_T": lambda_cubed_of_T,
            "insertion_z_range": insertion_z_range, "system_factory": system_factory,
            "metadata": metadata}


# =============================================================================
# Sampling
# =============================================================================
def _hmc_settings(re_cfg: Dict[str, Any]) -> Tuple[bool, int, float]:
    """
    HMC options of the ``replica_exchange.hmc`` block.

    Parameters
    ----------
    re_cfg : dict
        The ``replica_exchange`` section.

    Returns
    -------
    tuple
        ``(enabled, leapfrog_steps, dt)``; ``dt`` in the internal time unit
        (``timestep_fs`` / `FS_PER_INTERNAL_TIME`, or ``dt`` as given, for
        reduced units). ``enabled`` (sweep scheme only) replaces MC
        displacements by HMC; the regc scheme uses HMC through
        ``diffusion: hmc``.
    """
    hmc = re_cfg.get("hmc") or {}
    if "timestep_fs" in hmc:
        dt = _as_float(hmc["timestep_fs"], "replica_exchange.hmc.timestep_fs") / FS_PER_INTERNAL_TIME
    else:
        dt = _as_float(hmc.get("dt", 0.005), "replica_exchange.hmc.dt")
    return bool(hmc.get("enabled", False)), int(hmc.get("leapfrog_steps", 10)), dt


def _write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Write a JSON file (numbers as floats).

    Parameters
    ----------
    path : str
        Output file.
    data : dict
        JSON-able data (numpy scalars allowed).
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)


def run_from_config(config: Dict[str, Any], verbose: bool = True, resume: bool = False,
                    overwrite: bool = False, time_limit: Optional[float] = None,
                    n_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the whole config-driven pipeline: structure to phase diagram.

    See this module's docstring and ``examples/pd111_emt_example.yaml``
    for the config schema. Writes ``samples.npz``, ``run_metadata.json``
    (and, with a ``checkpoint`` section, ``checkpoint.pkl``), trajectories
    if asked for, and -- with a ``phase_diagram`` section -- the phase
    diagram plot(s) and raw grid (.npz), all to ``output.directory``.

    Parameters
    ----------
    config : dict
        The parsed config (see `load_config`).
    verbose : bool, optional
        If True, print progress messages. Default is True.
    resume : bool, optional
        With a ``checkpoint`` section: continue the run saved in the
        output directory, if there is one. Default False.
    overwrite : bool, optional
        With a ``checkpoint`` section: start afresh even if the output
        directory holds a run. Default False.
    time_limit : float, optional
        With a ``checkpoint`` section: seconds after which to stop cleanly
        (overrides ``checkpoint.time_limit_s``). Default None.
    n_workers : int, optional
        Worker processes (overrides ``replica_exchange.n_workers``).
        Default None.

    Returns
    -------
    dict
        ``'sim'`` (the `ReplicaExchangeGCMC`), ``'setup'`` (see
        `setup_from_config`), ``'status'`` (`RunStatus` of a checkpointed
        run, else None), ``'finished'`` (False if a checkpointed run
        stopped for its time limit; no analysis is done then),
        ``'D0_eV'``/``'e_atom_eV'`` (of the single species' reservoir, or
        dicts by species; None without a reservoir), plus everything
        `analyze_from_config` returns when the config has a
        ``phase_diagram`` section and the run finished.

    Raises
    ------
    ValueError
        On an invalid config (see `validate_config`), or ``burn_in_steps``
        with a checkpointed run of several chunks.
    FileExistsError
        With a ``checkpoint`` section, if the output directory holds a run
        and neither `resume` nor `overwrite` is set.
    """
    from .driver import METADATA_FILE, SAMPLES_FILE, run_with_checkpoints
    from .engine import ReplicaExchangeGCMC
    from .io import save_samples
    from .units import kelvin_to_eV

    setup = setup_from_config(config, verbose=verbose)
    re_cfg = config["replica_exchange"]
    scheme = _scheme(re_cfg)
    out_cfg = config.get("output", {})
    out_dir = out_cfg.get("directory", "output")
    os.makedirs(out_dir, exist_ok=True)
    traj_cfg = config.get("trajectory", {})
    traj_dir = (os.path.join(out_dir, traj_cfg.get("directory", "trajectories"))
                if traj_cfg.get("save") else None)
    traj_every = traj_cfg.get("every")

    workers = n_workers if n_workers is not None else re_cfg.get("n_workers", 1)
    workers = (os.cpu_count() or 1) if workers == "auto" else int(workers)
    hmc_enabled, leapfrog, hmc_dt = _hmc_settings(re_cfg)
    cutoff = re_cfg.get("adsorbed_cutoff")
    status, finished = None, True

    with ReplicaExchangeGCMC([kelvin_to_eV(T) for T in setup["temperatures_K"]], setup["mu_grid"],
                             setup["system_factory"], seed=int(re_cfg.get("seed", 0)),
                             max_disp=_as_float(re_cfg.get("max_disp", 0.3), "replica_exchange.max_disp"),
                             n_workers=workers, lambda_cubed_of_T=setup["lambda_cubed_of_T"],
                             hmc_enabled=hmc_enabled and scheme == "sweep",
                             hmc_leapfrog_steps=leapfrog, hmc_dt=hmc_dt,
                             adsorbed_cutoff=None if cutoff is None else _as_float(
                                 cutoff, "replica_exchange.adsorbed_cutoff")) as sim:
        if scheme == "regc":
            diffusion = re_cfg.get("diffusion", "mc")
            kwargs = dict(re_cfg.get("md") or {}) if diffusion == "md" else {}
            if re_cfg.get("displace_substrate"):
                kwargs["displace_substrate"] = True
            if "swap_types" in re_cfg:
                kwargs["swap_types"] = tuple(re_cfg["swap_types"])
            common = dict(n_steps=int(re_cfg["n_steps"]), x0=_as_float(re_cfg.get("x0", 0.99),
                                                                        "replica_exchange.x0"),
                          diffusion=diffusion, diffusion_steps=int(re_cfg.get("diffusion_steps", 10)),
                          record_every=int(re_cfg.get("record_every", 100)),
                          convergence_every=re_cfg.get("convergence_every"),
                          convergence_tol=_as_float(re_cfg.get("convergence_tol", 0.05),
                                                    "replica_exchange.convergence_tol"))
            burn_in = int(re_cfg.get("burn_in_steps", 0))
            if "checkpoint" in config:
                ck = config["checkpoint"] or {}
                chunk = int(ck.get("chunk", common["n_steps"]))
                if burn_in and chunk < common["n_steps"]:
                    raise ValueError("replica_exchange.burn_in_steps would apply to every "
                                     "checkpoint chunk; drop it and discard equilibration "
                                     "in the analysis instead (phase_diagram.discard)")
                if burn_in:
                    kwargs["burn_in_steps"] = burn_in
                if traj_dir is not None:
                    kwargs.update(trajectory_dir=traj_dir, trajectory_every=traj_every)
                limit = time_limit if time_limit is not None else ck.get("time_limit_s")
                status = run_with_checkpoints(
                    sim, out_dir, chunk=chunk, time_limit=None if limit is None else float(limit),
                    resume=resume, overwrite=overwrite, metadata=setup["metadata"],
                    progress=(lambda line: print(line, flush=True)) if verbose else None,
                    **common, **kwargs)
                finished = status.finished
            else:
                sim.run_regc(**common, burn_in_steps=burn_in, verbose=verbose,
                             trajectory_dir=traj_dir, trajectory_every=traj_every, **kwargs)
                _write_json(os.path.join(out_dir, METADATA_FILE), setup["metadata"])
        else:
            sim.run(n_cycles=int(re_cfg["n_cycles"]),
                    sweeps_per_cycle=int(re_cfg.get("sweeps_per_cycle", 100)),
                    exchange_every=int(re_cfg.get("exchange_every", 1)),
                    record_every=int(re_cfg.get("record_every", 5)),
                    burn_in_cycles=int(re_cfg.get("burn_in_cycles", 0)),
                    verbose=verbose, trajectory_dir=traj_dir, trajectory_every=traj_every,
                    p_displace=_as_float(re_cfg.get("p_displace", 0.5), "replica_exchange.p_displace"),
                    p_insert=_as_float(re_cfg.get("p_insert", 0.25), "replica_exchange.p_insert"),
                    p_delete=_as_float(re_cfg.get("p_delete", 0.25), "replica_exchange.p_delete"),
                    p_displace_substrate=_as_float(re_cfg.get("p_displace_substrate", 0.0),
                                                   "replica_exchange.p_displace_substrate"),
                    diagonal_swaps=bool(re_cfg.get("diagonal_swaps", True)))
            _write_json(os.path.join(out_dir, METADATA_FILE), setup["metadata"])
        if status is None:
            save_samples(sim, os.path.join(out_dir, SAMPLES_FILE))

    feed = setup["feed"]
    if feed is None:
        D0 = e_atom = None
    else:
        D0 = {s: getattr(r, "D0_eV", None) for s, r in zip(feed.species, feed.reservoirs)}
        e_atom = {s: r.e_atom_eV for s, r in zip(feed.species, feed.reservoirs)}
        if len(feed.species) == 1:
            D0, e_atom = D0[feed.species[0]], e_atom[feed.species[0]]
    result = {"sim": sim, "setup": setup, "status": status, "finished": finished,
              "D0_eV": D0, "e_atom_eV": e_atom}
    if not finished:
        if verbose:
            print(f"Stopped for the time limit; continue with --resume. Outputs in {out_dir}/")
        return result
    if "phase_diagram" in config:
        result.update(analyze_from_config(config, verbose=verbose))
    elif verbose:
        print(f"Done (no phase_diagram section: sampling only). Outputs in {out_dir}/")
    return result


# =============================================================================
# Analysis
# =============================================================================
_COVERAGE_LABEL = {"area": "particles / A$^2$", "surface_atom": "particles per surface atom",
                   "none": "particles per cell"}


def _phase_diagram_grid(pd_cfg: Dict[str, Any], ladder_unit: Optional[str]
                        ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    The (T, p) mesh of the ``phase_diagram`` section.

    Parameters
    ----------
    pd_cfg : dict
        The ``phase_diagram`` section.
    ladder_unit : str or None
        Pressure unit of the replica ladder, the default unit (else
        ``'bar'``; ``'atm'`` for the legacy ``p_min_atm``/``p_max_atm``).

    Returns
    -------
    tuple
        ``(T_grid_K, p_grid, pressure_unit)``.

    Raises
    ------
    ValueError
        If a bound is missing or not a number, the unit is unknown, or
        ``coverage_per`` is not one of area, surface_atom, none.
    """
    from .units import PRESSURE_UNITS
    legacy_atm = "p_min_atm" in pd_cfg or "p_max_atm" in pd_cfg
    unit = str(pd_cfg.get("pressure_unit", "atm" if legacy_atm else (ladder_unit or "bar")))
    if unit not in PRESSURE_UNITS:
        raise ValueError(f"phase_diagram.pressure_unit must be one of {list(PRESSURE_UNITS)}, "
                         f"got {unit!r}")
    bounds = []
    for name in ("p_min", "p_max"):
        key = name if name in pd_cfg else f"{name}_atm"
        if key not in pd_cfg:
            raise ValueError(f"phase_diagram needs {name!r}")
        bounds.append(_as_float(pd_cfg[key], f"phase_diagram.{key}"))
    for key in ("T_min_K", "T_max_K"):
        if key not in pd_cfg:
            raise ValueError(f"phase_diagram needs {key!r}")
    T_grid_K = np.linspace(_as_float(pd_cfg["T_min_K"], "phase_diagram.T_min_K"),
                           _as_float(pd_cfg["T_max_K"], "phase_diagram.T_max_K"),
                           int(pd_cfg.get("n_T", 15)))
    if pd_cfg.get("coverage_per", "area") not in _COVERAGE_LABEL:
        raise ValueError(f"phase_diagram.coverage_per must be area, surface_atom or none, "
                         f"got {pd_cfg['coverage_per']!r}")
    return T_grid_K, np.geomspace(bounds[0], bounds[1], int(pd_cfg.get("n_p", 40))), unit


def analyze_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    MBAR (T, p) phase diagram from the samples a run saved.

    Reads ``samples.npz`` and ``run_metadata.json`` from
    ``output.directory`` (no calculator is needed: the reservoirs are
    rebuilt from the metadata), so a finished run can be re-analysed with
    other ``phase_diagram`` settings (``gcmc-re-run --analyze-only``).

    Parameters
    ----------
    config : dict
        The parsed config; its ``phase_diagram`` and ``output`` sections
        are used.
    verbose : bool, optional
        Print a summary. Default True.

    Returns
    -------
    dict
        ``'analysis'`` (`MBARSurfaceAnalysis`), ``'theta_grid'`` (shape
        (n_T, n_p)), ``'T_grid_K'``, ``'p_grid'`` (in ``'pressure_unit'``),
        ``'p_grid_atm'``, ``'coverage_per'``, ``'observable'`` and, with
        ``most_probable: true``, ``'most_probable_N'``.

    Raises
    ------
    ValueError
        If the config has no ``phase_diagram`` section, the run had no
        reservoir, or ``coverage_per`` is unknown.
    FileNotFoundError
        If the output directory holds no saved run.
    """
    from .analysis import MBARSurfaceAnalysis
    from .driver import METADATA_FILE, SAMPLES_FILE
    from .io import load_samples
    from .plotting import plot_phase_diagram
    from .units import kelvin_to_eV, pascal_per

    if "phase_diagram" not in config:
        raise ValueError("config has no phase_diagram section")
    pd_cfg = config["phase_diagram"]
    _check_keys(pd_cfg, _SECTIONS["phase_diagram"], "phase_diagram")
    out_cfg = config.get("output", {})
    out_dir = out_cfg.get("directory", "output")
    meta_path, samples_path = (os.path.join(out_dir, METADATA_FILE),
                               os.path.join(out_dir, SAMPLES_FILE))
    for path in (meta_path, samples_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found: run the simulation first")
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("reservoirs"):
        raise ValueError("the run had no gas reservoir, so pressures are undefined; add a "
                         "reservoir section (or a pressure ladder) to the config and rerun")
    species = meta["species"]
    feed = GasFeed(species, [_reservoir_from_dict(meta["reservoirs"][s]) for s in species],
                   [meta["composition"][s] for s in species])

    T_grid_K, p_grid, unit = _phase_diagram_grid(pd_cfg, meta.get("pressure_unit"))
    observable = str(pd_cfg.get("observable", "N"))
    per = str(pd_cfg.get("coverage_per", "area"))
    norm = {"area": meta["area_angstrom2"], "surface_atom": meta["n_surface"], "none": 1.0}[per]

    samples = load_samples(samples_path, discard=_as_float(pd_cfg.get("discard", 0.0),
                                                           "phase_diagram.discard"))
    analysis = MBARSurfaceAnalysis(samples, mu_of_Tp=functools.partial(feed.mu_at_T_eV, unit=unit))
    T_grid_eV = np.array([kelvin_to_eV(T) for T in T_grid_K])
    theta = analysis.phase_diagram_Tp(T_grid_eV, p_grid, area=norm, observable=observable)

    plot_name = out_cfg.get("plot_filename", "phase_diagram.png")
    xlabel, ylabel = f"pressure ({unit})", "temperature (K)"
    plot_phase_diagram(T_grid_K, p_grid, theta, save_path=os.path.join(out_dir, plot_name),
                       colorbar_label=f"<{observable}>: {_COVERAGE_LABEL[per]}",
                       xlabel=xlabel, ylabel=ylabel)
    result = {"analysis": analysis, "theta_grid": theta, "T_grid_K": T_grid_K, "p_grid": p_grid,
              "pressure_unit": unit, "p_grid_atm": p_grid * pascal_per(unit) / pascal_per("atm"),
              "coverage_per": per, "observable": observable}
    raw = dict(T_K=T_grid_K, p=p_grid, pressure_unit=unit, p_atm=result["p_grid_atm"],
               theta=theta, observable=observable, coverage_per=per)
    if pd_cfg.get("most_probable"):
        most = analysis.most_probable_N_Tp(T_grid_eV, p_grid, observable=observable)
        result["most_probable_N"] = most
        raw["most_probable_N"] = most
        root, ext = os.path.splitext(plot_name)
        plot_phase_diagram(T_grid_K, p_grid, most,
                           save_path=os.path.join(out_dir, f"{root}_most_probable{ext}"),
                           colorbar_label=f"most probable {observable}", xlabel=xlabel,
                           ylabel=ylabel, title="Most stable phase (MBAR)")
    if out_cfg.get("save_raw_data", True):
        np.savez(os.path.join(out_dir, "phase_diagram_data.npz"), **raw)
    if verbose:
        print(f"{samples.n_samples} samples per state; <{observable}> per {per}: "
              f"{theta.min():.4g} to {theta.max():.4g}. Outputs in {out_dir}/")
    return result


# =============================================================================
# Command line
# =============================================================================
def check_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the setup and print it, with the mu ladder, without sampling.

    Parameters
    ----------
    config : dict
        The parsed config.

    Returns
    -------
    dict
        The setup (see `setup_from_config`).
    """
    setup = setup_from_config(config, verbose=True)
    mu = np.asarray(setup["mu_grid"])
    cols = setup["pressures"]
    label = f"p ({setup['pressure_unit']})" if cols is not None else "ladder index"
    if mu.ndim == 1:
        mu = np.tile(mu, (len(setup["temperatures_K"]), 1))
    cols = cols if cols is not None else list(range(mu.shape[1]))
    for k, sp in enumerate(setup["species"]):
        grid = mu if mu.ndim == 2 else mu[..., k]
        print(f"\nmu_{sp} (eV): rows T (K), columns {label}")
        print("        " + " ".join(f"{c:9.2e}" if isinstance(c, float) else f"{c:9d}" for c in cols))
        for T, row in zip(setup["temperatures_K"], grid):
            print(f"{T:6.0f}  " + " ".join(f"{m:9.3f}" for m in row))
    bare = setup["system_factory"](np.random.default_rng(0))
    print(f"\nbare-slab energy: {bare.total_energy():.4f} eV "
          f"({setup['calc_spec'].type} calculator works)")
    return setup


def main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``gcmc-re-run config.yaml``.

    Runs the structure -> REGC -> MBAR -> (T, p) phase diagram pipeline of
    `run_from_config` for one config file; ``--check`` only builds and
    prints the setup, ``--analyze-only`` re-analyses saved samples.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (without the program name). Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 on success, `EXIT_RESUBMIT` (75) if a
        checkpointed run stopped for its time limit.
    """
    import argparse

    from .driver import EXIT_RESUBMIT
    parser = argparse.ArgumentParser(
        prog="gcmc-re-run",
        description="Structure -> replica-exchange GCMC -> MBAR -> (T, p) phase diagram, "
                    "from a YAML/JSON config (see examples/pd111_emt_example.yaml).")
    parser.add_argument("config", help="config file (.yaml/.yml or .json)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="build and print the setup (slab, reservoir, mu ladder); no sampling")
    mode.add_argument("--analyze-only", action="store_true",
                      help="phase diagram from the samples saved in output.directory")
    parser.add_argument("--resume", action="store_true",
                        help="continue the checkpointed run saved in output.directory")
    parser.add_argument("--overwrite", action="store_true",
                        help="start afresh even if output.directory holds a checkpointed run")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="seconds: stop a checkpointed run cleanly (exit code 75) before this")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (overrides replica_exchange.n_workers)")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.check:
        check_config(config)
        return 0
    if args.analyze_only:
        analyze_from_config(config, verbose=not args.quiet)
        return 0
    result = run_from_config(config, verbose=not args.quiet, resume=args.resume,
                             overwrite=args.overwrite, time_limit=args.time_limit,
                             n_workers=args.workers)
    return 0 if result["finished"] else EXIT_RESUBMIT
