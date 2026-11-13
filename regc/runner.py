"""
Config-file-driven end-to-end runner: structure -> REGC -> MBAR -> (T, p)
surface phase diagram, all from one YAML or JSON file.

``regc-run config.yaml`` (or `run_from_config` on a dict) does what
``examples/13_fe_h_eam_surface.py`` does in Python:

1. **structure** -- a slab or a cluster. A slab is read from any
   ASE-readable file (``structure.file``) or built with an ``ase.build``
   surface function (``structure.build``, lattice constant optionally
   fitted to the potential, ``a: auto``), and laid out with
   `slab_geometry_from_atoms` (bottom ``n_frozen_layers`` rigid, the rest
   displaceable, ``headroom_angstrom`` of space above); orthogonal and
   oblique (e.g. hexagonal fcc(111)) cells both work. A cluster is built
   with an ``ase.cluster`` function (``structure.cluster``, e.g.
   ``Icosahedron``) or read from a file with ``geometry: cluster``, and laid
   out with `cluster_geometry_from_atoms`: centred in a non-periodic cubic
   box, ``frozen`` atoms rigid (default: the central one), insertions in
   the box or a shell (``gas.insertion_shell_around_cluster``).
2. **calculator** -- ``EMT``, ``LJ``, ``PairLJ``, ``LAMMPS`` (any pair style:
   an EAM file, GRACE, ACE, ...; native MD; ``lammps_name`` picks the LAMMPS
   library, e.g. ``grace`` for liblammps_grace.so) or ``ASE`` (any ASE calculator
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
   species ``N_<sp>``) on a (T, p) mesh, per area, per surface atom or per
   slab/cluster atom, and
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
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: One time unit of (Angstrom, amu, eV), in fs: HMC timesteps given in fs are divided by it.
FS_PER_INTERNAL_TIME = 10.1805


def _available_cpus() -> int:
    """
    CPUs this process may run on (respects Slurm/cgroup affinity, unlike os.cpu_count).

    Returns
    -------
    int
        The number of usable CPUs, at least 1.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1

_SECTIONS = {
    "structure": {"file", "build", "cluster", "geometry", "n_frozen_layers", "headroom_angstrom",
                  "layer_tol", "frozen", "full_coordination"},
    "gas": {"species", "mass_amu", "insertion_z_range_above_surface",
            "insertion_shell_around_cluster", "composition"},
    "reservoir": {"type", "d0", "D0_eV", "e_atom_eV", "bond_length_angstrom",
                  "vibrational_wavenumber_cm", "symmetry_number", "electronic_degeneracy",
                  "mass_molecule_amu", "mass_amu"},
    "trajectory": {"save", "directory", "every"},
    "configurations": {"save", "directory", "every"},
    "records": {"save", "file", "every", "potential", "positions"},
    "checkpoint": {"chunk", "time_limit_s"},
    "phase_diagram": {"T_min_K", "T_max_K", "n_T", "p_min", "p_max", "n_p", "pressure_unit",
                      "p_min_atm", "p_max_atm", "observable", "coverage_per", "most_probable",
                      "discard", "min_ess", "heat_capacity", "rdf_states",
                      "export_configurations", "bootstrap", "merge", "suggest_ladder",
                      "potentials"},
    "output": {"directory", "plot_filename", "save_raw_data"},
}
_TOP_LEVEL = set(_SECTIONS) | {"calculator", "replica_exchange", "diatomic_reservoir",
                                "observables"}
_EXPORT_KEYS = {"mode", "n_T", "n_p", "min_cells", "tile"}
_BOOTSTRAP_KEYS = {"n", "seed", "level", "min_agreement", "min_support"}
_CALCULATOR_KEYS = {
    "EMT": set(),
    "LJ": {"lj_sigma", "lj_epsilon", "lj_r_cut"},
    "PAIRLJ": {"pair_params", "r_cut_factor"},
    "LAMMPS": {"species", "potential", "pair_style", "pair_coeff", "pair_commands",
               "min_distance", "z_padding", "lammps_name"},
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


def _observables_from_config(config: Dict[str, Any]) -> list:
    """
    The structural observables of the ``observables`` section.

    Parameters
    ----------
    config : dict
        The parsed config; ``observables`` is a list of
        ``{type: coordination | coordination_histogram | rdf, name: ..., ...}``.

    Returns
    -------
    list of regc.observables.Observable
        The observables (empty without the section).

    Raises
    ------
    ValueError
        If the section is not a list, or an entry is invalid.
    """
    from .observables import check_observables, observable_from_config
    entries = config.get("observables") or []
    if not isinstance(entries, list):
        raise ValueError("observables must be a list of {type: ..., name: ..., ...} entries")
    out = []
    for e in entries:
        e = dict(e)
        if "z_range" in e and e["z_range"] is not None:
            e["z_range"] = tuple(_floats(e["z_range"], f"observables.{e.get('name')}.z_range"))
        out.append(observable_from_config(e))
    return check_observables(out)


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
    if "observables" in config:
        _observables_from_config(config)
    export = (config.get("phase_diagram") or {}).get("export_configurations")
    if isinstance(export, dict):
        _check_keys(export, _EXPORT_KEYS, "phase_diagram.export_configurations")
    boot = (config.get("phase_diagram") or {}).get("bootstrap")
    if isinstance(boot, dict):
        _check_keys(boot, _BOOTSTRAP_KEYS, "phase_diagram.bootstrap")
    if "md" in re_cfg:
        _check_keys(re_cfg["md"], _MD_KEYS, "replica_exchange.md")
        if scheme != "regc" or re_cfg.get("diffusion", "mc") != "md":
            raise ValueError("replica_exchange.md is used only with scheme: regc and "
                             "diffusion: md; remove it or set diffusion: md")
    if scheme == "regc" and (re_cfg.get("hmc") or {}).get("enabled"):
        raise ValueError("replica_exchange.hmc.enabled applies to scheme: sweep; with "
                         "scheme: regc use diffusion: hmc (the hmc block still sets "
                         "leapfrog_steps and the time step)")
    gas = config.get("gas") or {}
    species = set(_species_list(config)) if "species" in gas else set()
    for key in ("mass_amu", "composition"):
        if isinstance(gas.get(key), dict):
            foreign = sorted(set(map(str, gas[key])) - species)
            if foreign:
                raise ValueError(f"gas.{key} has entries {foreign} that are not exchanged "
                                 f"species {sorted(species)}")
    if ("mu_eV" in re_cfg) == ("pressures" in re_cfg):
        raise ValueError("replica_exchange needs exactly one of 'mu_eV' (explicit chemical "
                         "potentials) or 'pressures' (gas-pressure ladder via the reservoir)")
    if "checkpoint" in config and scheme != "regc":
        raise ValueError("the checkpoint section needs replica_exchange.scheme: regc")
    struct = config["structure"]
    sources = [k for k in ("file", "build", "cluster") if k in struct]
    if len(sources) != 1:
        raise ValueError("structure needs exactly one of 'file' (an ASE-readable slab or "
                         "cluster), 'build' (an ase.build surface) or 'cluster' (an ase.cluster "
                         f"particle); got {sources or 'none'}")
    kind = _geometry_kind(struct)
    if kind == "cluster" and "build" in struct:
        raise ValueError("structure.build makes a slab; use structure.cluster for a cluster")
    wrong = ({"n_frozen_layers", "layer_tol"} if kind == "cluster"
             else {"frozen", "full_coordination"}) & set(struct)
    wrong_gas = {"insertion_z_range_above_surface" if kind == "cluster"
                 else "insertion_shell_around_cluster"} & set(config["gas"])
    if wrong or wrong_gas:
        keys = sorted(wrong) + [f"gas.{k}" for k in sorted(wrong_gas)]
        raise ValueError(f"{', '.join(keys)} not used with a {kind} (structure.geometry)")
    if (kind == "cluster" and "phase_diagram" in config
            and config["phase_diagram"].get("coverage_per") == "area"):
        raise ValueError("phase_diagram.coverage_per: area needs a slab; for a cluster use "
                         "surface_atom, atom or none")


def _geometry_kind(struct_cfg: Dict[str, Any]) -> str:
    """
    Whether the ``structure`` section describes a slab or a cluster.

    Parameters
    ----------
    struct_cfg : dict
        The ``structure`` section.

    Returns
    -------
    str
        ``'cluster'`` with a ``cluster`` block or ``geometry: cluster``,
        else ``'slab'``.

    Raises
    ------
    ValueError
        If ``geometry`` is neither slab nor cluster, or contradicts a
        ``cluster`` block.
    """
    kind = str(struct_cfg.get("geometry", "cluster" if "cluster" in struct_cfg else "slab")).lower()
    if kind not in ("slab", "cluster"):
        raise ValueError(f"structure.geometry must be slab or cluster, got {kind!r}")
    if kind == "slab" and "cluster" in struct_cfg:
        raise ValueError("structure.cluster builds a cluster: drop geometry: slab")
    return kind


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
            name = cfg.get("lammps_name")   # None: REGC_LAMMPS_NAME, else the default library
            self.lammps_name = None if name is None else str(name)
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
            return lammps_ase_calculator(self.pair_commands, self.lammps_species,
                                         lammps_name=self.lammps_name)
        factory = self._ase_factory()
        return factory() if factory is not None else None

    def make(self, cell: np.ndarray, pbc: Tuple[bool, bool, bool] = (True, True, False)):
        """
        A fresh simulation `Calculator` (each system needs its own).

        Parameters
        ----------
        cell : numpy.ndarray
            3x3 cell matrix (rows a1, a2, a3) for ASE-backed calculators.
        pbc : tuple of bool, optional
            Their periodicity. Default ``(True, True, False)``: a slab
            (a cluster has none).

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
                                                        "calculator.z_padding"),
                                    lammps_name=self.lammps_name)
        from .calculators import ASECalculatorAdapter
        cell = np.asarray(cell, dtype=float)
        if np.allclose(cell - np.diag(np.diag(cell)), 0.0):
            cell = np.diag(cell)
        return ASECalculatorAdapter(self._ase_factory(), cell=cell, pbc=tuple(pbc))

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
    Read or build the slab or cluster, and the calculator spec that goes with it.

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
        If ``build``/``cluster`` lacks its shape or ``symbol``, ``a: auto``
        is asked for a non-cubic crystal or a potential with no ASE
        counterpart, or ``ase.build``/``ase.cluster`` has no such function.
    """
    if "file" in struct_cfg:
        from ase.io import read as ase_read
        atoms = ase_read(struct_cfg["file"])
        slab_species = list(dict.fromkeys(atoms.get_chemical_symbols()))
        return atoms, calc_spec_of(slab_species), None
    if "cluster" in struct_cfg:
        return _build_cluster(dict(struct_cfg["cluster"]), calc_spec_of)

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
    a = _lattice_constant(build.pop("a", None), build.pop("a_range", None), surface[:3], symbol,
                          calc_spec, "structure.build")
    if "size" in build:
        build["size"] = tuple(int(n) for n in build["size"])
    atoms = builder(symbol, a=a, **build) if a is not None else builder(symbol, **build)
    return atoms, calc_spec, a


#: Lattice of each ase.cluster function, for ``a: auto``.
_CLUSTER_CRYSTAL = {"Icosahedron": "fcc", "Octahedron": "fcc", "Decahedron": "fcc",
                    "FaceCenteredCubic": "fcc", "BodyCenteredCubic": "bcc"}


def _build_cluster(cl: Dict[str, Any], calc_spec_of: Callable[[List[str]], _CalculatorSpec]
                   ) -> Tuple[Any, _CalculatorSpec, Optional[float]]:
    """
    Build a cluster with an ``ase.cluster`` function.

    Parameters
    ----------
    cl : dict
        The ``structure.cluster`` block: ``shape`` (the function, e.g.
        ``Icosahedron``), ``symbol``, ``a`` (``auto``, a number, or absent
        for ASE's default), ``a_range``, and the function's own keywords
        (``noshells``, ``length``, ``cutoff``, ``p``, ``q``, ``r``, ...).
    calc_spec_of : callable
        ``calc_spec_of(cluster_species) -> _CalculatorSpec``.

    Returns
    -------
    tuple
        ``(atoms, calc_spec, lattice_constant)``.

    Raises
    ------
    ValueError
        If ``shape``/``symbol`` is missing or ``ase.cluster`` has no such
        function, or ``a: auto`` cannot be fitted.
    """
    import ase.cluster
    for key in ("shape", "symbol"):
        if key not in cl:
            raise ValueError(f"structure.cluster needs {key!r} (e.g. shape: Icosahedron, "
                             "symbol: Pt, noshells: 3)")
    shape, symbol = str(cl.pop("shape")), str(cl.pop("symbol"))
    builder = getattr(ase.cluster, shape, None)
    if builder is None or not callable(builder):
        raise ValueError(f"ase.cluster has no function {shape!r} (e.g. Icosahedron, Octahedron, "
                         "Decahedron)")
    calc_spec = calc_spec_of([symbol])
    a = _lattice_constant(cl.pop("a", None), cl.pop("a_range", None),
                          _CLUSTER_CRYSTAL.get(shape, shape), symbol, calc_spec,
                          "structure.cluster")
    atoms = builder(symbol, latticeconstant=a, **cl) if a is not None else builder(symbol, **cl)
    return atoms, calc_spec, a


def _lattice_constant(a: Any, a_range: Any, crystal: str, symbol: str,
                      calc_spec: _CalculatorSpec, where: str) -> Optional[float]:
    """
    The lattice constant of a ``build``/``cluster`` block.

    Parameters
    ----------
    a : 'auto', number or None
        The block's ``a``.
    a_range : sequence of float or None
        Scan range for ``a: auto``. Default: 0.8-1.2 times the experimental
        lattice constant (`cubic_lattice_constant`).
    crystal : str
        ``'fcc'`` or ``'bcc'`` (anything else refuses ``auto``).
    symbol : str
        The element.
    calc_spec : _CalculatorSpec
        The potential.
    where : str
        Config path, for messages.

    Returns
    -------
    float or None
        The lattice constant (A), or None to use ASE's default.

    Raises
    ------
    ValueError
        If ``auto`` is asked for another crystal or a potential with no
        ASE counterpart.
    """
    if a is None:
        return None
    if a != "auto":
        return _as_float(a, f"{where}.a")
    from .structures import cubic_lattice_constant
    if crystal not in ("fcc", "bcc"):
        raise ValueError(f"{where}.a: auto needs an fcc or bcc lattice, got {crystal!r}")
    ase_calc = calc_spec.ase_calculator()
    if ase_calc is None:
        raise ValueError(f"{where}.a: auto needs a calculator with an ASE counterpart "
                         "(EMT, LAMMPS, ASE); give 'a' explicitly")
    a_range = tuple(_floats(a_range, f"{where}.a_range")) if a_range else None  # +-20% of expt.
    return cubic_lattice_constant(ase_calc, symbol, crystal, a_range=a_range, n_coarse=97)


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
                            MonatomicIdealGasReservoir, default_electronic_degeneracy)
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
    g_el = cfg.get("electronic_degeneracy")
    g_el = default_electronic_degeneracy(sp) if g_el is None else int(g_el)

    ase_calc = ase_calc_getter() if (d0 == "self" or e_atom is None) else None
    if d0 == "self" and ase_calc is None:
        raise ValueError(
            f"reservoir D0_eV for {sp} must be given explicitly (e.g. D0_eV: 4.478, or "
            f"d0: experimental for H) when calculator.type is "
            f"{calc_spec.type} -- it has no ASE counterpart to evaluate the molecule with.")
    if ase_calc is not None:
        res = DiatomicReservoir.from_calculator(ase_calc, sp, d0=d0, bond_length_angstrom=bond,
                                                vib_wavenumber_cm=vib, mass_molecule_amu=m_mol,
                                                symmetry_number=sym, electronic_degeneracy=g_el)
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
    return DiatomicReservoir(float(d0), e_atom, m_mol, bond, vib, sym, g_el)


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
        ``geometry`` (`SlabGeometry` or `ClusterGeometry`), ``lattice_constant`` (A, or None),
        ``calc_spec``, ``species``, ``masses``, ``feed`` (`GasFeed`, or None
        when the run needs no reservoir), ``temperatures_K``, ``mu_grid``,
        ``pressures``/``pressure_unit`` (None for an explicit mu ladder),
        ``lambda_cubed_of_T``, ``insertion_z_range`` (slab) or
        ``insertion_shell`` (cluster), ``system_factory``
        and ``metadata`` (JSON-able; what a resumed run must agree with,
        and what `analyze_from_config` reads).

    Raises
    ------
    ValueError
        On an invalid config (see `validate_config`) or an insertion
        z-range outside the box.
    """
    from .structures import cluster_geometry_from_atoms, slab_geometry_from_atoms
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
    kind = _geometry_kind(struct_cfg)
    atoms, calc_spec, a = _build_structure(struct_cfg, calc_spec_of)
    insertion_z_range = insertion_shell = None
    if kind == "cluster":
        frozen = struct_cfg.get("frozen", "center")
        geometry = cluster_geometry_from_atoms(
            atoms, headroom=_as_float(struct_cfg.get("headroom_angstrom", 6.0),
                                      "structure.headroom_angstrom"),
            frozen=frozen if isinstance(frozen, str) else int(frozen),
            full_coordination=int(struct_cfg.get("full_coordination", 12)))
        cell, pbc = np.diag([geometry.box] * 3), (False, False, False)
        shell = gas_cfg.get("insertion_shell_around_cluster")
        if shell is not None:
            insertion_shell = geometry.shell(*_floats(shell, "gas.insertion_shell_around_cluster"))
        region = {"insertion_shell": insertion_shell}
    else:
        nfl = struct_cfg.get("n_frozen_layers")
        geometry = slab_geometry_from_atoms(
            atoms, n_frozen_layers=None if nfl is None else int(nfl),
            headroom=_as_float(struct_cfg.get("headroom_angstrom", 5.0),
                               "structure.headroom_angstrom"),
            layer_tol=_as_float(struct_cfg.get("layer_tol", 1e-3), "structure.layer_tol"))
        cell, pbc = np.zeros((3, 3)), (True, True, False)
        cell[:2, :2], cell[2, 2] = geometry.cell_xy, geometry.Lz
        region = {"insertion_z_range": None}

    zr = gas_cfg.get("insertion_z_range_above_surface")
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
        region = {"insertion_z_range": insertion_z_range}

    if calc_spec.type == "LJ" and (len(geometry.frozen_positions) or len(geometry.substrate_positions)):
        warnings.warn("calculator.type is 'LJ' (the native LJCalculator), which only evaluates "
                      "interactions between MOBILE particles: the slab atoms are ignored, so gas "
                      "atoms feel nothing from them. Use it to test the pipeline's mechanics "
                      "only; PairLJ (which sees the slab), EMT, LAMMPS or ASE for meaningful "
                      "results.", stacklevel=2)

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
        return geometry.make_system(calc_spec.make(cell, pbc), exchange, mass_of_species=masses,
                                    **region)

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
        "geometry": kind, "n_surface": geometry.n_surface, "area_angstrom2": geometry.area,
        "n_atoms": len(geometry.frozen_positions) + len(geometry.substrate_positions),
        "insertion_z_range": insertion_z_range, "insertion_shell": insertion_shell,
        "sampling": sampling,
    }
    if config.get("observables"):
        metadata["observables"] = config["observables"]

    if verbose:
        if a is not None:
            print(f"lattice constant from the potential: {a:.4f} A")
        print(f"{kind}: {geometry.summary()}")
        if insertion_z_range is not None:
            print(f"insertions in z in ({insertion_z_range[0]:.2f}, {insertion_z_range[1]:.2f}] A")
        if insertion_shell is not None:
            print(f"insertions in the shell {insertion_shell[0]:.2f} <= r < "
                  f"{insertion_shell[1]:.2f} A around the cluster centre")
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
            "insertion_z_range": insertion_z_range, "insertion_shell": insertion_shell,
            "system_factory": system_factory,
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
    from .driver import (METADATA_FILE, MIXING_FILE, SAMPLES_FILE, run_with_checkpoints,
                         write_mixing)
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
    conf_cfg = config.get("configurations") or {}
    conf_dir = (os.path.join(out_dir, conf_cfg.get("directory", "configurations"))
                if conf_cfg.get("save") else None)
    rec_cfg = config.get("records") or {}
    rec_path = _records_path(config, out_dir) if rec_cfg.get("save") else None
    if rec_path is not None and not resume and os.path.exists(rec_path):
        from .driver import CHECKPOINT_FILE
        guarded = ("checkpoint" in config and not overwrite
                   and os.path.isfile(os.path.join(out_dir, CHECKPOINT_FILE)))
        if not guarded:  # (a guarded run stops below: resume or overwrite it)
            os.remove(rec_path)  # a fresh run: do not append to an old run's records

    workers = n_workers if n_workers is not None else re_cfg.get("n_workers", 1)
    workers = _available_cpus() if workers == "auto" else int(workers)
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
                                 cutoff, "replica_exchange.adsorbed_cutoff"),
                             observables=_observables_from_config(config),
                             configurations_dir=conf_dir,
                             configurations_every=int(conf_cfg.get("every", 1)),
                             records_path=rec_path,
                             records_potential=str(rec_cfg.get("potential", "sampling")),
                             records_every=int(rec_cfg.get("every", 1))) as sim:
        sim.save_status(os.path.join(out_dir, STATUS_FILE), extra={
            "temperatures_K": [float(T) for T in setup["temperatures_K"]],
            "pressures": None if setup["pressures"] is None else [float(p) for p in setup["pressures"]],
            "pressure_unit": setup["pressure_unit"], "species": list(setup["species"])})
        if rec_path is not None and rec_cfg.get("positions"):
            sim.save_records(rec_path, str(rec_cfg.get("potential", "sampling")),
                             int(rec_cfg.get("every", 1)), str(rec_cfg["positions"]))
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
                write_mixing(sim, os.path.join(out_dir, MIXING_FILE))
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
            write_mixing(sim, os.path.join(out_dir, MIXING_FILE))
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


def _records_path(config: Dict[str, Any], out_dir: Optional[str] = None) -> str:
    """
    Path of a run's per-sample records file.

    Parameters
    ----------
    config : dict
        The parsed config (``records.file``, default ``records.h5``).
    out_dir : str, optional
        Output directory. Default None: ``output.directory``.

    Returns
    -------
    str
        The path.
    """
    if out_dir is None:
        out_dir = (config.get("output") or {}).get("directory", "output")
    return os.path.join(out_dir, (config.get("records") or {}).get("file", "records.h5"))


def reevaluate_from_config(config: Dict[str, Any], potential: Dict[str, Any], name: str,
                           every: int = 1, n_workers: int = 1, verbose: bool = True) -> int:
    """
    Re-evaluate a run's recorded configurations with another potential.

    The run's config with its ``calculator`` section (and any other
    top-level section given in `potential`, e.g. ``reservoir``) replaced
    builds the same geometry with the new potential; its energies go to
    ``E/<name>`` in the records file, and its gas reservoirs (whose
    reference energies depend on the potential) to the file's attribute
    ``reservoirs:<name>``, for the pressure axes of the reweighted phase
    diagram (``phase_diagram.potentials``).

    Parameters
    ----------
    config : dict
        The run's parsed config (``records: {save: true}`` must have been on).
    potential : dict
        Top-level sections to replace, at least ``calculator``.
    name : str
        Name of the new potential.
    every : int, optional
        Evaluate every this many stored samples (by sample index). Default 1.
    n_workers : int, optional
        Worker processes. Default 1.
    verbose : bool, optional
        Print progress. Default True.

    Returns
    -------
    int
        Configurations evaluated.

    Raises
    ------
    FileNotFoundError
        If the run has no records file.
    ValueError
        If `potential` has no calculator, or the new setup's cell differs
        from the recorded one (fix the lattice constant in ``structure``).
    """
    import copy

    from .records import Records, reevaluate
    from .trajectory import lattice_of
    path = _records_path(config)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found: run with records: {{save: true}}")
    if "calculator" not in potential:
        raise ValueError("the potential file needs a calculator section")
    cfg = copy.deepcopy(config)
    cfg.update(copy.deepcopy(potential))
    cfg.pop("phase_diagram", None)
    setup = setup_from_config(cfg, verbose=False)
    system = setup["system_factory"](np.random.default_rng(0))
    rec = Records(path)
    try:
        cell = np.asarray(rec.attrs["cell"], dtype=float)
    finally:
        rec.close()
    if not np.allclose(lattice_of(system), cell, atol=1e-8):
        raise ValueError(f"the cell built with potential {name!r} differs from the recorded one "
                         f"({np.diag(lattice_of(system))} vs {np.diag(cell)} A): set "
                         f"structure.lattice_constant explicitly so both use the same geometry")
    every = max(1, int(every))
    progress = ((lambda d, t: print(f"{name}: {d}/{t} configurations", flush=True))
                if verbose else None)
    n = reevaluate(path, setup["system_factory"], name,
                   subset=None if every == 1 else (lambda key, s: s % every == 0),
                   n_workers=n_workers, progress=progress)
    feed = setup["feed"]
    if feed is not None:
        import h5py
        with h5py.File(path, "a") as f:
            f.attrs["reservoirs:" + name] = json.dumps(
                {s: _reservoir_to_dict(r) for s, r in zip(feed.species, feed.reservoirs)})
    if verbose:
        print(f"{n} configurations evaluated with {name!r} in {path}")
    return n


def _potential_maps(analysis, samples, config: Dict[str, Any], meta: Dict[str, Any], names,
                    T_grid_K: np.ndarray, T_grid_eV: np.ndarray, p_grid: np.ndarray, unit: str,
                    norm: float, per: str, observable: str, theta_A: np.ndarray, min_ess: float,
                    out_dir: str,
                    root: str, ext: str, verbose: bool) -> Dict[str, Any]:
    """
    Phase diagrams reweighted to other potentials (``phase_diagram.potentials``).

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The sampling potential's analysis.
    samples : SampleSet
        Its samples (energies under the other potentials are attached).
    config, meta : dict
        The config and the run metadata.
    names : sequence of str
        Potentials (re-evaluated with ``regc-reevaluate``).
    T_grid_K, T_grid_eV, p_grid : numpy.ndarray
        The mesh.
    unit : str
        Pressure unit.
    norm : float
        Coverage normalisation.
    per : str
        Its name (``coverage_per``).
    observable : str
        Mapped observable.
    theta_A : numpy.ndarray
        The sampling potential's map.
    min_ess : float
        Mask threshold.
    out_dir, root, ext : str
        Output location and file name parts.
    verbose : bool
        Print a summary.

    Returns
    -------
    dict
        Raw arrays ``theta_<name>``, ``ess_<name>``.
    """
    from .records import Records, attach_energies
    from .plotting import plot_map
    path = _records_path(config, out_dir)
    attach_energies(samples, path, names)
    rec = Records(path)
    try:
        stored = {n: rec.attrs.get("reservoirs:" + n) for n in names}
    finally:
        rec.close()
    raw = {}
    xlabel, ylabel = f"pressure ({unit})", "temperature (K)"
    for name in names:
        kw = {}
        if stored[name] is not None:
            res = json.loads(stored[name])
            feed = GasFeed(meta["species"], [_reservoir_from_dict(res[s]) for s in meta["species"]],
                           [meta["composition"][s] for s in meta["species"]])
            kw["mu_of_Tp"] = functools.partial(feed.mu_at_T_eV, unit=unit)
        B = analysis.with_potential(name, **kw)
        theta = B.phase_diagram_Tp(T_grid_eV, p_grid, area=norm, observable=observable)
        ess = B.effective_sample_size(T_grid_eV, p_grid, pressure=True)
        valid = ess >= max(min_ess, 1.0)
        raw[f"theta_{name}"], raw[f"ess_{name}"] = theta, ess
        tag = f"{root}_{name}{ext}"
        plot_map(T_grid_K, p_grid, theta, os.path.join(out_dir, tag), valid=valid,
                 colorbar_label=f"<{observable}> (per {per})", xlabel=xlabel, ylabel=ylabel,
                 title=f"Coverage reweighted to {name}", invalid_label=f"effective samples < {min_ess:g}")
        plot_map(T_grid_K, p_grid, theta - theta_A,
                 os.path.join(out_dir, f"{root}_{name}_difference{ext}"), valid=valid,
                 cmap="RdBu_r", colorbar_label=f"<{observable}> ({name}) - <{observable}> (sampling)",
                 xlabel=xlabel, ylabel=ylabel, title=f"{name} minus sampling potential",
                 invalid_label=f"effective samples < {min_ess:g}")
        plot_map(T_grid_K, p_grid, ess, os.path.join(out_dir, f"{root}_{name}_ess{ext}"),
                 log_color=True, cmap="viridis", colorbar_label="Kish effective samples",
                 xlabel=xlabel, ylabel=ylabel, title=f"Overlap of the samples with {name}")
        if verbose:
            print(f"{name}: effective samples {np.nanmin(ess):.3g} to {np.nanmax(ess):.3g} "
                  f"({100 * valid.mean():.0f} % of cells >= {max(min_ess, 1.0):g})")
    return raw


# =============================================================================
# Analysis
# =============================================================================
STATUS_FILE = "status.json"  # heartbeat for regc.monitor

_COVERAGE_LABEL = {"area": "particles / A$^2$", "surface_atom": "particles per surface atom",
                   "atom": "particles per slab/cluster atom", "none": "particles per cell"}


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
        ``coverage_per`` is not one of area, surface_atom, atom, none.
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
        raise ValueError(f"phase_diagram.coverage_per must be area, surface_atom, atom or none, "
                         f"got {pd_cfg['coverage_per']!r}")
    return T_grid_K, np.geomspace(bounds[0], bounds[1], int(pd_cfg.get("n_p", 40))), unit


def _reliability(analysis, samples, meta: Dict[str, Any], out_dir: str,
                 verbose: bool) -> Dict[str, Any]:
    """
    Correlation, overlap and mixing diagnostics of a run (no resampling).

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    samples : SampleSet
        Its samples.
    meta : dict
        The run metadata.
    out_dir : str
        Output directory (``mixing.json`` is read from it if present).
    verbose : bool
        Print warnings and a summary.

    Returns
    -------
    dict
        Arrays for ``phase_diagram_data.npz``: ``n_independent`` and
        ``statistical_inefficiency`` (replica grid), ``overlap_min``
        (replica grid, smallest MBAR overlap with a neighbour).
    """
    from .diagnostics import OVERLAP_THRESHOLD, neighbour_overlap, replica_inefficiencies
    from .driver import MIXING_FILE
    ineff = replica_inefficiencies(samples, analysis.lambda_cubed_of_T)
    merged = any(len(k) != 2 for k in ineff)
    ov = neighbour_overlap(analysis, pressure=merged and bool(meta.get("pressures")))
    if merged:   # several runs: one value per state, in the analysis's state order
        keys = analysis.states
        g_map = np.array([ineff[k]["g"] for k in keys])
        n_ind = np.array([ineff[k]["n_independent"] for k in keys])
        ov_min = np.array([ov["min_map"].get(k, np.nan) for k in keys])
        out = {"state_keys": np.array(keys), "statistical_inefficiency": g_map,
               "n_independent": n_ind, "overlap_min": ov_min}
    else:
        n_T = 1 + max(k[0] for k in ineff)
        n_mu = 1 + max(k[1] for k in ineff)
        g_map, n_ind = np.full((n_T, n_mu), np.nan), np.full((n_T, n_mu), np.nan)
        for k, v in ineff.items():
            g_map[k], n_ind[k] = v["g"], v["n_independent"]
        ov_min = ov["min_map"]
        out = {"statistical_inefficiency": g_map, "n_independent": n_ind, "overlap_min": ov_min}
    if verbose:
        print(f"correlation: statistical inefficiency median {np.nanmedian(g_map):.1f}, "
              f"independent samples per replica min {np.nanmin(n_ind):.0f}, "
              f"median {np.nanmedian(n_ind):.0f}")
        if np.nanmin(n_ind) < 20:
            print("  WARNING: some replicas hold fewer than 20 independent samples; averages "
                  "and especially C_V are unreliable there, and bootstrap intervals too narrow")
        print(f"overlap: {len(ov['gaps'])} of {len(ov['pairs'])} neighbouring replica pairs below "
              f"{OVERLAP_THRESHOLD} (smallest {np.nanmin(ov_min):.2g})")
        if ov["gaps"]:
            print("  WARNING: between poorly overlapping replicas MBAR rests on few samples; "
                  "C_V is underestimated there and peaks appear on the sampled states. "
                  "Add replicas in the gaps: phase_diagram.suggest_ladder writes the fill "
                  "runs, phase_diagram.merge analyses them together with this one")
    mix_path = os.path.join(out_dir, MIXING_FILE)
    if os.path.isfile(mix_path):
        with open(mix_path) as f:
            mix = json.load(f)
        trips = np.asarray(mix.get("round_trips_T", []))
        out["round_trips_T"] = trips
        if verbose:
            w = mix.get("worst_pair")
            if w:
                print(f"mixing: worst swap pair {tuple(w['a'])}-{tuple(w['b'])} accepted "
                      f"{w['acceptance']:.2f} of {w['attempts']} attempts; T round trips per "
                      f"walker: median {np.median(trips) if trips.size else 0:.0f}, "
                      f"{int(np.sum(trips == 0))} walkers with none")
    return out


def _merge_runs(config: Dict[str, Any], out_dir: str, meta: Dict[str, Any], samples, others,
                discard: float, verbose: bool):
    """
    Merge the samples of other runs of the same system into the analysis.

    Parameters
    ----------
    config : dict
        The parsed config (for the configurations directory name).
    out_dir : str
        This run's output directory.
    meta : dict
        This run's metadata.
    samples : SampleSet
        This run's samples (equilibration discarded).
    others : str or list of str
        Output directories of the other runs.
    discard : float
        Fraction of each run's samples dropped as equilibration.
    verbose : bool
        Print a summary.

    Returns
    -------
    tuple
        ``(merged SampleSet, configuration directory of every run)``.

    Raises
    ------
    FileNotFoundError
        If another run has no samples or metadata.
    ValueError
        If another run is not the same system (structure, calculator,
        species, reservoirs).
    """
    from .driver import METADATA_FILE, SAMPLES_FILE
    from .io import load_samples, merge_samples
    others = [others] if isinstance(others, str) else list(others)
    conf_name = (config.get("configurations") or {}).get("directory", "configurations")
    sets, dirs, labels = [samples], [os.path.join(out_dir, conf_name)], [out_dir]
    for other in others:
        mpath, spath = os.path.join(other, METADATA_FILE), os.path.join(other, SAMPLES_FILE)
        for path in (mpath, spath):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"phase_diagram.merge: {path} not found")
        with open(mpath) as f:
            om = json.load(f)
        for key in ("species", "structure", "calculator", "reservoirs", "lattice_constant_angstrom",
                    "masses_amu", "geometry"):
            if om.get(key) != meta.get(key):
                raise ValueError(f"phase_diagram.merge: {other} differs in '{key}'; only runs of "
                                 f"the same system and potential can be merged")
        sets.append(load_samples(spath, discard=discard))
        dirs.append(os.path.join(other, conf_name))
        labels.append(other)
    merged = merge_samples(*sets, labels=labels)
    if verbose:
        print(f"merged {len(sets)} runs: {len(merged.samples)} states")
    return merged, dirs


def _write_ladder_suggestion(analysis, meta: Dict[str, Any], cfg, out_dir: str, unit: str,
                             verbose: bool) -> None:
    """
    Write the fill runs proposed by `suggest_ladder` as config snippets.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    meta : dict
        Run metadata (pressure ladder or mu ladder, pressure unit).
    cfg : bool or dict
        ``true``, or ``{target_overlap: 0.1, max_insert: 4}``.
    out_dir : str
        Output directory (writes ``ladder_suggestion.yaml``).
    unit : str
        The analysis's pressure unit (that of ``phase_diagram``): the
        proposed pressures are written in it.
    verbose : bool
        Print a summary.
    """
    from .diagnostics import suggest_ladder
    cfg = {} if cfg is True else dict(cfg)
    pressure = bool(meta.get("pressures"))
    sug = suggest_ladder(analysis, target_overlap=float(cfg.get("target_overlap", 0.1)),
                         pressure=pressure, max_insert=int(cfg.get("max_insert", 4)))
    text = sug.to_config(kelvin=True, pressure_unit=unit)
    path = os.path.join(out_dir, "ladder_suggestion.yaml")
    with open(path, "w") as f:
        f.write("# Fill runs proposed by regc.diagnostics.suggest_ladder: run each with this "
                "config's other sections\n# (configurations: save), then add their output "
                "directories to phase_diagram.merge.\n" + (text or "# no gaps: nothing to fill")
                + "\n")
    if verbose:
        print(f"ladder: {len(sug.gaps)} gaps below {cfg.get('target_overlap', 0.1)}; "
              f"{sum(len(r['T']) * len(r['y']) for r in sug.fill_runs)} new replicas proposed "
              f"in {path}")


def _bootstrap_maps(analysis, T_grid_eV: np.ndarray, p_grid: np.ndarray, norm: float,
                    observable: str, with_cv: bool, cfg: Dict[str, Any],
                    verbose: bool) -> Dict[str, Any]:
    """
    Block-bootstrap the coverage, most-probable and C_V maps in one pass.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    T_grid_eV, p_grid : numpy.ndarray
        The mesh.
    norm : float
        Coverage normalisation.
    observable : str
        The phase-diagram observable.
    with_cv : bool
        Also resample C_V.
    cfg : dict
        ``n`` (default 100), ``seed`` (0), ``level`` (0.95),
        ``min_agreement`` (0.8), ``min_support`` (0.95).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        BootstrapResults ``'theta'``, ``'most'``, (``'cv'``), plus the
        settings.
    """
    from .diagnostics import BootstrapResult, bootstrap
    n = int(cfg.get("n", 100))

    def stat(a):
        maps = [a.phase_diagram_Tp(T_grid_eV, p_grid, area=norm, observable=observable),
                a.most_probable_N_Tp(T_grid_eV, p_grid, observable=observable).astype(float)]
        if with_cv:
            maps.append(a.heat_capacity(T_grid_eV, p_grid, pressure=True))
        return np.stack(maps)

    if verbose:
        print(f"bootstrap: {n} block resamples (MBAR re-solved for each) ...", flush=True)
    b = bootstrap(analysis, stat, n_boot=n, seed=int(cfg.get("seed", 0)))
    parts = {"theta": 0, "most": 1, "cv": 2}
    out = {name: BootstrapResult(b.estimate[i], b.replicates[:, i], b.block_lengths)
           for name, i in parts.items() if i < b.estimate.shape[0]}
    out.update(n=n, level=float(cfg.get("level", 0.95)),
               min_agreement=float(cfg.get("min_agreement", 0.8)),
               min_support=float(cfg.get("min_support", 0.95)))
    return out


def analyze_from_config(config: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    MBAR (T, p) phase diagram from the samples a run saved.

    Reads ``samples.npz`` and ``run_metadata.json`` from
    ``output.directory`` (no calculator is needed: the reservoirs are
    rebuilt from the metadata), so a finished run can be re-analysed with
    other ``phase_diagram`` settings (``regc-run --analyze-only``).

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
    from .analysis import export_phase_configurations, heat_capacity_ridges
    from .diagnostics import ridge_support
    from .plotting import plot_map, plot_radial_distribution
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
    per = str(pd_cfg.get("coverage_per",
                         "surface_atom" if meta.get("geometry") == "cluster" else "area"))
    norm = {"area": meta.get("area_angstrom2"), "surface_atom": meta.get("n_surface"),
            "atom": meta.get("n_atoms"), "none": 1.0}[per]
    if norm is None:
        raise ValueError(f"coverage_per: {per} is not available for this run "
                         + ("(a cluster has no area; use surface_atom, atom or none)"
                            if per == "area" else "(run_metadata.json predates it; rerun setup)"))

    discard = _as_float(pd_cfg.get("discard", 0.2), "phase_diagram.discard")
    samples = load_samples(samples_path, discard=discard)
    conf_dirs = None
    if pd_cfg.get("merge"):
        samples, conf_dirs = _merge_runs(config, out_dir, meta, samples, pd_cfg["merge"],
                                         discard, verbose)
    analysis = MBARSurfaceAnalysis(samples, mu_of_Tp=functools.partial(feed.mu_at_T_eV, unit=unit))
    T_grid_eV = np.array([kelvin_to_eV(T) for T in T_grid_K])
    theta = analysis.phase_diagram_Tp(T_grid_eV, p_grid, area=norm, observable=observable)

    plot_name = out_cfg.get("plot_filename", "phase_diagram.png")
    root, ext = os.path.splitext(plot_name)
    xlabel, ylabel = f"pressure ({unit})", "temperature (K)"
    ess = analysis.effective_sample_size(T_grid_eV, p_grid, pressure=True)
    min_ess = _as_float(pd_cfg.get("min_ess", 0.0), "phase_diagram.min_ess")
    valid = ess >= min_ess
    invalid = f"effective samples < {min_ess:g}"
    # -- reliability: correlation, overlap, mixing (always), bootstrap (on request) --------
    diag = _reliability(analysis, samples, meta, out_dir, verbose)
    if pd_cfg.get("suggest_ladder"):
        _write_ladder_suggestion(analysis, meta, pd_cfg["suggest_ladder"], out_dir, unit, verbose)
    boot_cfg = pd_cfg.get("bootstrap")
    boot = None
    if boot_cfg:
        boot_cfg = {} if boot_cfg is True else dict(boot_cfg)
        boot = _bootstrap_maps(analysis, T_grid_eV, p_grid, norm, observable,
                               bool(pd_cfg.get("heat_capacity")), boot_cfg, verbose)
    plot_map(T_grid_K, p_grid, theta, os.path.join(out_dir, plot_name), valid=valid,
             colorbar_label=f"<{observable}>: {_COVERAGE_LABEL[per]}", xlabel=xlabel,
             ylabel=ylabel, title="Reweighted coverage (MBAR)", invalid_label=invalid)
    result = {"analysis": analysis, "theta_grid": theta, "T_grid_K": T_grid_K, "p_grid": p_grid,
              "pressure_unit": unit, "p_grid_atm": p_grid * pascal_per(unit) / pascal_per("atm"),
              "coverage_per": per, "observable": observable, "effective_samples": ess}
    raw = dict(T_K=T_grid_K, p=p_grid, pressure_unit=unit, p_atm=result["p_grid_atm"],
               theta=theta, observable=observable, coverage_per=per, effective_samples=ess,
               min_ess=min_ess)
    raw.update(diag)
    if boot is not None:
        lo, hi = boot["theta"].interval(boot["level"])
        raw.update(theta_ci_low=lo, theta_ci_high=hi, bootstrap_level=boot["level"],
                   bootstrap_n=boot["n"])
        result["theta_interval"] = (lo, hi)
        plot_map(T_grid_K, p_grid, 0.5 * (hi - lo), os.path.join(out_dir, f"{root}_uncertainty{ext}"),
                 valid=valid, cmap="viridis", xlabel=xlabel, ylabel=ylabel,
                 colorbar_label=f"half-width of the {100 * boot['level']:.0f} % interval of <{observable}>",
                 title=f"Bootstrap uncertainty ({boot['n']} block resamples)", invalid_label=invalid)
    if pd_cfg.get("most_probable"):
        most = analysis.most_probable_N_Tp(T_grid_eV, p_grid, observable=observable)
        result["most_probable_N"] = most
        raw["most_probable_N"] = most
        robust, label_note = valid, invalid
        if boot is not None:
            agree = boot["most"].agreement()
            raw["most_probable_agreement"] = agree
            robust = valid & (agree >= boot["min_agreement"])
            label_note = (f"{invalid}, or label kept in < {100 * boot['min_agreement']:.0f} % "
                          f"of resamples")
        plot_map(T_grid_K, p_grid, most, os.path.join(out_dir, f"{root}_most_probable{ext}"),
                 valid=robust, discrete=True, colorbar_label=f"most probable {observable}",
                 xlabel=xlabel, ylabel=ylabel, title="Most stable phase (MBAR)",
                 invalid_label=label_note)
    if pd_cfg.get("heat_capacity"):
        cv = analysis.heat_capacity(T_grid_eV, p_grid, pressure=True)
        ladder_p = meta.get("pressures")
        ladder_p = (np.asarray(ladder_p, dtype=float) * pascal_per(meta.get("pressure_unit") or unit)
                    / pascal_per(unit)) if ladder_p else None
        ridges = heat_capacity_ridges(T_grid_K, p_grid, cv, valid=valid, ladder=ladder_p)
        keep = np.ones(len(ridges["points"]), bool)
        if "on_ladder" in ridges:
            keep &= ~ridges["on_ladder"]
            raw["heat_capacity_ridges_on_ladder"] = ridges["on_ladder"]
        if boot is not None and "cv" in boot:
            support = ridge_support(ridges, boot["cv"].replicates)
            keep &= support >= boot["min_support"]
            raw["heat_capacity_ridges_support"] = support
            lo, hi = boot["cv"].interval(boot["level"])
            raw.update(heat_capacity_ci_low=lo, heat_capacity_ci_high=hi)
        result.update(heat_capacity=cv, heat_capacity_ridges=ridges["points"],
                      heat_capacity_ridges_kept=ridges["points"][keep])
        raw.update(heat_capacity_kB=cv, heat_capacity_ridges_T_K_p=ridges["points"],
                   heat_capacity_ridges_kept=keep)
        kept_label = "$C_V$ peak" + (" (bootstrap-supported" if boot is not None else "") + (
            ", off the ladder" if "on_ladder" in ridges else "") + (")" if boot is not None else "")
        plot_map(T_grid_K, p_grid, cv, os.path.join(out_dir, f"{root}_heat_capacity{ext}"),
                 valid=valid, points=ridges["points"][keep], points_label=kept_label,
                 log_color=True, cmap="magma", colorbar_label="$C_V/k_B$", xlabel=xlabel,
                 ylabel=ylabel, title="Heat capacity (MBAR, [ZSG] Eq. 1)", invalid_label=invalid)
        if verbose:
            print(f"C_V: {len(ridges['points'])} peaks, {int(keep.sum())} kept"
                  + (f" ({int(ridges['on_ladder'].sum())} on the sampled pressures)"
                     if "on_ladder" in ridges else ""))
    for name, m in analysis.observables_meta.items():
        kind = m.get("type")
        if kind == "CoordinationNumber":
            cn = analysis.mean_coordination(name, T_grid_eV, p_grid, pressure=True)
            result[f"cn_{name}"] = raw[f"cn_{name}"] = cn
            plot_map(T_grid_K, p_grid, cn, os.path.join(out_dir, f"{root}_cn_{name}{ext}"),
                     valid=valid & np.isfinite(cn), cmap="cividis",
                     colorbar_label=f"mean coordination ({m['center']} - {m['neighbor']}, "
                                    f"r < {m['r_cut']:g} A)", xlabel=xlabel, ylabel=ylabel,
                     title=f"<CN> {name}", invalid_label=invalid + " or no centre atoms")
        elif kind == "CoordinationHistogram":
            h = analysis.observable_map(name, T_grid_eV, p_grid, pressure=True)
            total = h.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                raw[f"cn_hist_{name}"] = np.where(total > 0, h / total, np.nan)
        elif kind == "RadialDistribution" and pd_cfg.get("rdf_states"):
            curves = {}
            for T_K, p in pd_cfg["rdf_states"]:
                T_eV = kelvin_to_eV(_as_float(T_K, "phase_diagram.rdf_states"))
                p = _as_float(p, "phase_diagram.rdf_states")
                r, g = analysis.radial_distribution(name, T_eV, analysis.mu_from_p(T_eV, p))
                curves[f"{float(T_K):.0f} K, {p:.3g} {unit}"] = (r, g)
            raw[f"rdf_{name}_r"] = next(iter(curves.values()))[0]
            raw[f"rdf_{name}"] = np.array([g for _, g in curves.values()])
            raw[f"rdf_{name}_states"] = np.array(pd_cfg["rdf_states"], dtype=float)
            plot_radial_distribution(curves, os.path.join(out_dir, f"{root}_rdf_{name}{ext}"),
                                     ylabel=f"n(r) per {m['a']} atom (1/A)",
                                     title=f"{m['a']} - {m['b']} (MBAR)")
    export = pd_cfg.get("export_configurations")
    if export:
        export = {"mode": export} if isinstance(export, str) else dict(export)
        conf_cfg = config.get("configurations") or {}
        conf_dir = conf_dirs or os.path.join(out_dir, conf_cfg.get("directory", "configurations"))
        mode = export.get("mode", "phases")
        if mode == "grid":
            Tg = np.linspace(T_grid_K[0], T_grid_K[-1], int(export.get("n_T", 5)))
            pg = np.logspace(np.log10(p_grid[0]), np.log10(p_grid[-1]), int(export.get("n_p", 6)))
        else:
            Tg, pg = T_grid_K, p_grid
        tile = os.path.join(out_dir, f"configurations_{mode}_tiled.extxyz") \
            if export.get("tile", True) else None
        try:
            entries = export_phase_configurations(
                analysis, conf_dir, np.array([kelvin_to_eV(T) for T in Tg]), pg,
                os.path.join(out_dir, f"configurations_{mode}.extxyz"), pressure=True, mode=mode,
                observable=observable, min_ess=min_ess, temperature_K=True, area=norm,
                tile_path=tile, min_cells=int(export.get("min_cells", 1)))
        except (FileNotFoundError, ValueError) as err:
            warnings.warn(f"configurations not exported: {err}", stacklevel=2)
            entries = []
        result["exported_configurations"] = entries
        if verbose and entries:
            print(f"wrote {len(entries)} configurations ({mode}) to "
                  f"{out_dir}/configurations_{mode}.extxyz")
    potentials = pd_cfg.get("potentials")
    if potentials:
        if pd_cfg.get("merge"):
            raise ValueError("phase_diagram.potentials with merge is not supported yet: "
                             "reweight each run on its own")
        names = [potentials] if isinstance(potentials, str) else list(potentials)
        raw.update(_potential_maps(analysis, samples, config, meta, names, T_grid_K, T_grid_eV,
                                   p_grid, unit, norm, per, observable, theta, min_ess, out_dir, root,
                                   ext, verbose))
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
    Command-line entry point: ``regc-run config.yaml``.

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
        prog="regc-run",
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


def reevaluate_main(argv: Optional[list] = None) -> int:
    """
    Command-line entry point: ``regc-reevaluate config.yaml potential.yaml --name B``.

    Re-evaluates the configurations a run recorded (``records: {save:
    true}``) with the potential described in ``potential.yaml`` (a
    ``calculator`` section, plus any other top-level section to replace);
    then list ``B`` under ``phase_diagram.potentials`` and run
    ``regc-run --analyze-only`` for the reweighted phase diagram.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        0.
    """
    import argparse
    parser = argparse.ArgumentParser(
        prog="regc-reevaluate",
        description="Energies of a run's recorded configurations under another potential, "
                    "for MBAR reweighting to it (phase_diagram.potentials).")
    parser.add_argument("config", help="the run's config file")
    parser.add_argument("potential", help="YAML/JSON with the new calculator section")
    parser.add_argument("--name", required=True, help="name of the new potential")
    parser.add_argument("--every", type=int, default=1,
                        help="evaluate every k-th stored sample (default 1: all)")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    args = parser.parse_args(argv)
    reevaluate_from_config(load_config(args.config), load_config(args.potential), args.name,
                           every=args.every, n_workers=args.workers, verbose=not args.quiet)
    return 0
