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
   displaceable, ``headroom_angstrom`` of space above; or ``layers: {fixed,
   thermostat, free}``, the three-region substrate whose free top layers MD
   integrates without a thermostat, see `surface_slab`); orthogonal and
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
   feed of several species (``gas.composition``). Or a ``molecules``
   section: gases of molecules (CO, O2, CH4, ...) relaxed with the
   potential (`MolecularGas`), whose mixture fixes the elemental chemical
   potentials (`MolecularGasFeed`), each optionally exchanged as a
   molecule (`regc.molecules.MoleculeMove`), plus ``gas.atomic_exchange``
   for species also exchanged atom by atom.
   A ``subsurface`` section (slab) adds NCMC insertion and removal in
   interstitial sites below the surface (`regc.subsurface`) and confines
   the other moves to the gas region above it. A ``site_exchange`` list adds
   further NCMC channels, each with its own species, z-range and sites
   (``trilateration``: vacancies and hollows of a metal surface, found from
   the configuration), by default as an overlay of the gas moves.
4. **replica_exchange** -- a (T x p) ladder (``pressures``, mu from the
   reservoir) or an explicit (T x mu) one (``mu_eV``); the [ZSG] REGC
   scheme (``scheme: regc``, MC/HMC/MD diffusion) or the sweep scheme;
   optional ``adsorbed_cutoff`` to record the adsorbed count ``N_ads``.
5. **checkpoint** (regc only) -- run in chunks with `run_with_checkpoints`
   (``checkpoint.pkl``/``samples.npz``), resumable (``--resume``) and
   stopping cleanly before ``--time-limit`` (exit code 75, `EXIT_RESUBMIT`).
   With ``replica_exchange.adapt_ladder`` the (T, p) ladder is moved after
   the first chunks from the measured swap rates, then frozen (`regc.ladder`).
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

from .observables import COMPOSITION_TYPES, composition_layers

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
    "structure": {"file", "build", "cluster", "geometry", "n_frozen_layers", "layers",
                  "exchangeable_top_layers",
                  "headroom_angstrom",
                  "layer_tol", "frozen", "full_coordination", "alloy", "alloy_seed", "alloy_per_layer"},
    "sgc": {"species", "dmu_eV", "attempts_per_step", "bulk_composition", "ensemble"},
    "subsurface": {"z_cut_below_surface", "depth_angstrom", "species", "delta_angstrom",
                   "ncmc_steps", "kernel", "kernel_steps", "leapfrog_steps", "timestep_fs",
                   "friction_per_fs", "max_disp", "weight", "site_min_distance",
                   "freeze_switched"},
    "gas": {"species", "mass_amu", "insertion_z_range_above_surface",
            "insertion_shell_around_cluster", "composition", "atomic_exchange"},
    "reservoir": {"type", "d0", "D0_eV", "e_atom_eV", "bond_length_angstrom",
                  "vibrational_wavenumber_cm", "symmetry_number", "electronic_degeneracy",
                  "mass_molecule_amu", "mass_amu", "crystal", "lattice_constant_angstrom",
                  "supercell", "e_bulk_eV", "mean_wavenumber_cm"},
    "trajectory": {"save", "directory", "every"},
    "configurations": {"save", "directory", "every"},
    "records": {"save", "file", "every", "potential", "positions"},
    "checkpoint": {"chunk", "time_limit_s"},
    "phase_diagram": {"T_min_K", "T_max_K", "n_T", "p_min", "p_max", "n_p", "pressure_unit",
                      "p_min_atm", "p_max_atm", "observable", "coverage_per", "most_probable",
                      "discard", "min_ess", "heat_capacity", "rdf_states",
                      "export_configurations", "bootstrap", "merge", "suggest_ladder",
                      "potentials", "extra_observables"},
    "output": {"directory", "plot_filename", "save_raw_data"},
}
_TOP_LEVEL = set(_SECTIONS) | {"calculator", "replica_exchange", "diatomic_reservoir",
                                "observables", "molecules", "site_exchange"}
#: Keys of one entry of the ``site_exchange`` list (one NCMC site-exchange channel).
_SITE_EXCHANGE_KEYS = {"species", "z_range_above_surface", "sites", "bond_length_angstrom",
                       "min_distance_angstrom", "merge_angstrom", "delta_angstrom",
                       "defensive_fraction", "partition_gas_zone", "ncmc_steps", "kernel",
                       "kernel_steps", "leapfrog_steps", "timestep_fs", "friction_per_fs",
                       "max_disp", "weight", "freeze_switched"}
#: Keys of one entry of the ``molecules`` section (a gas, and optionally an insertion move).
_MOLECULE_KEYS = {"template", "fraction", "insert", "anchor", "shell_angstrom",
                  "bond_sigma_angstrom", "angle_sigma_deg", "weight", "symmetry_number",
                  "spin_multiplicity", "vibrational_wavenumbers_cm", "energy_eV",
                  "vibrations"}
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
             "convergence_every", "convergence_tol", "convergence_mode", "convergence_bins",
             "convergence_min_neff", "convergence_median_tol", "convergence_discard",
             "convergence_consecutive", "convergence_min_round_trips", "swap_types",
             "displace_substrate",
             "md", "adapt_ladder"},
    "sweep": {"n_cycles", "sweeps_per_cycle", "exchange_every", "burn_in_cycles",
              "diagonal_swaps", "p_displace", "p_insert", "p_delete", "p_displace_substrate"},
}
_HMC_KEYS = {"enabled", "leapfrog_steps", "timestep_fs", "dt"}
#: Keys of ``replica_exchange.adapt_ladder`` and their defaults (`regc.ladder.LadderAdapter`).
_ADAPT_LADDER_DEFAULTS = {"chunks": 3, "trust": 1.5, "per_row": True, "min_gap_ratio": 0.25,
                          "max_gap_ratio": 4.0, "min_attempts": 20, "max_evals": 20000,
                          "max_time_s": None, "min_samples": 5}
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
    for sec in (config.get("molecules") or {}).values():
        if isinstance(sec, dict) and isinstance(sec.get("template"), str):
            sec["template"] = _resolve_existing(sec["template"], base)
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


def _observables_from_config(config: Dict[str, Any], geometry: Any = None) -> list:
    """
    The structural observables of the ``observables`` section.

    Parameters
    ----------
    config : dict
        The parsed config; ``observables`` is a list of
        ``{type: coordination | coordination_histogram | rdf | layer_composition,
        name: ..., ...}``.
    geometry : SlabGeometry, optional
        The structure, for ``layer_composition`` with ``layer_z: auto`` (its
        layer heights). Default None (validation only: a placeholder).

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
        if str(e.get("type", "")).lower() == "layer_composition":
            if e.get("layer_z", "auto") == "auto":
                # the slab's own layers; validation (no geometry yet) uses a placeholder
                layers = getattr(geometry, "layer_z", None)
                e["layer_z"] = [0.0] if layers is None else [float(z) for z in layers]
            else:
                e["layer_z"] = _floats(e["layer_z"], f"observables.{e.get('name')}.layer_z")
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
    if "molecules" in config:
        _validate_molecules(config)
    if "subsurface" in config and _geometry_kind(config["structure"]) != "slab":
        raise ValueError("the subsurface section (NCMC insertion in subsurface sites) needs a slab")
    if "site_exchange" in config:
        _validate_site_exchange(config)
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
    if scheme == "regc":
        _adapt_ladder_settings(config)
    if "sgc" in config:
        sgc = config["sgc"] or {}
        if scheme != "regc":
            raise ValueError("the sgc section (identity swaps) needs replica_exchange.scheme: regc")
        ens = str(sgc.get("ensemble", "semigrand")).lower()
        if ens not in ("semigrand", "canonical"):
            raise ValueError(f"sgc.ensemble must be 'semigrand' or 'canonical', got {ens!r}")
        if "species" not in sgc:
            raise ValueError("sgc needs 'species' (reference first, e.g. [Pd, Pt])")
        if ens == "semigrand" and "dmu_eV" not in sgc:
            raise ValueError("sgc needs 'dmu_eV' (e.g. {Pt: -0.5}: mu_Pt - mu_Pd), or "
                             "ensemble: canonical (pair swaps at fixed composition)")
        if ens == "canonical" and ("dmu_eV" in sgc or "bulk_composition" in sgc):
            raise ValueError("sgc.ensemble: canonical keeps the composition fixed: "
                             "drop dmu_eV and bulk_composition")
    struct = config["structure"]
    sources = [k for k in ("file", "build", "cluster") if k in struct]
    if len(sources) != 1:
        raise ValueError("structure needs exactly one of 'file' (an ASE-readable slab or "
                         "cluster), 'build' (an ase.build surface) or 'cluster' (an ase.cluster "
                         f"particle); got {sources or 'none'}")
    kind = _geometry_kind(struct)
    if kind == "cluster" and "build" in struct:
        raise ValueError("structure.build makes a slab; use structure.cluster for a cluster")
    wrong = ({"n_frozen_layers", "layers", "layer_tol"} if kind == "cluster"
             else {"frozen", "full_coordination"}) & set(struct)
    if kind == "slab" and "layers" in struct:
        from .structures import substrate_layers
        if "n_frozen_layers" in struct:
            raise ValueError("structure.layers already sets the fixed (frozen) layers: "
                             "drop structure.n_frozen_layers")
        n_layers = sum(substrate_layers(struct["layers"]))
        size = (struct.get("build") or {}).get("size")
        if size is not None and len(size) == 3 and int(size[2]) != n_layers:
            raise ValueError(f"structure.build.size has {size[2]} layers but structure.layers "
                             f"adds up to {n_layers}; give size as [nx, ny] and the layers "
                             f"set the depth")
        re_cfg = config.get("replica_exchange") or {}
        if _scheme(re_cfg) != "regc" or re_cfg.get("diffusion", "mc") != "md":
            warnings.warn("structure.layers: the thermostat / free split acts only on MD "
                          "diffusion (scheme: regc, diffusion: md); other moves see fixed vs "
                          "displaceable layers only", stacklevel=2)
    wrong_gas = {"insertion_z_range_above_surface" if kind == "cluster"
                 else "insertion_shell_around_cluster"} & set(config["gas"])
    if wrong or wrong_gas:
        keys = sorted(wrong) + [f"gas.{k}" for k in sorted(wrong_gas)]
        raise ValueError(f"{', '.join(keys)} not used with a {kind} (structure.geometry)")
    if (kind == "cluster" and "phase_diagram" in config
            and config["phase_diagram"].get("coverage_per") == "area"):
        raise ValueError("phase_diagram.coverage_per: area needs a slab; for a cluster use "
                         "surface_atom, atom or none")


def _adapt_ladder_settings(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    The ``replica_exchange.adapt_ladder`` settings, checked and with defaults.

    Parameters
    ----------
    config : dict
        The parsed config.

    Returns
    -------
    dict or None
        The settings (`_ADAPT_LADDER_DEFAULTS` keys), or None when the key is
        absent or false.

    Raises
    ------
    ValueError
        On an unknown key or a bad value, or outside a checkpointed regc run
        with a pressure ladder of at least 3 temperatures or 3 pressures.
    """
    re_cfg = config["replica_exchange"]
    cfg = re_cfg.get("adapt_ladder")
    if cfg is None or cfg is False:
        return None
    cfg = {} if cfg is True else cfg
    _check_keys(cfg, set(_ADAPT_LADDER_DEFAULTS), "replica_exchange.adapt_ladder")
    out = dict(_ADAPT_LADDER_DEFAULTS, **cfg)
    where = "replica_exchange.adapt_ladder"
    for key in ("chunks", "min_attempts", "max_evals", "min_samples"):
        v = _as_float(out[key], f"{where}.{key}")
        if v != int(v) or v < (2 if key == "min_samples" else 1):
            raise ValueError(f"{where}.{key} must be an integer >= "
                             f"{2 if key == 'min_samples' else 1}, got {out[key]!r}")
        out[key] = int(v)
    for key in ("trust", "min_gap_ratio", "max_gap_ratio"):
        out[key] = _as_float(out[key], f"{where}.{key}")
    if out["max_time_s"] is not None:
        out["max_time_s"] = _as_float(out["max_time_s"], f"{where}.max_time_s")
        if out["max_time_s"] <= 0:
            raise ValueError(f"{where}.max_time_s must be positive")
    if not isinstance(out["per_row"], bool):
        raise ValueError(f"{where}.per_row must be true or false, got {out['per_row']!r}")
    if not out["trust"] > 1.0:
        raise ValueError(f"{where}.trust must be > 1, got {out['trust']}")
    if not 0.0 < out["min_gap_ratio"] <= 1.0 <= out["max_gap_ratio"]:
        raise ValueError(f"{where}: need 0 < min_gap_ratio <= 1 <= max_gap_ratio, got "
                         f"{out['min_gap_ratio']}, {out['max_gap_ratio']}")
    if "pressures" not in re_cfg:
        raise ValueError(f"{where} needs a pressure ladder (replica_exchange.pressures with a "
                         f"gas reservoir): the ladder is moved in (T, log10 p)")
    if "checkpoint" not in config:
        raise ValueError(f"{where} needs a checkpoint section: the ladder is adapted between "
                         f"checkpoint chunks")
    n_steps = re_cfg.get("n_steps")
    chunk = (config["checkpoint"] or {}).get("chunk", n_steps)
    if n_steps is not None and chunk is not None and int(chunk) >= int(n_steps):
        raise ValueError(f"{where} needs checkpoint.chunk ({chunk}) < n_steps ({n_steps}): the "
                         f"ladder is adapted after a chunk, before the next one")
    n_T, n_p = len(re_cfg.get("temperatures_K") or []), len(re_cfg.get("pressures") or [])
    if n_T < 3 and n_p < 3:
        raise ValueError(f"{where} needs at least 3 temperatures or 3 pressures: the corners "
                         f"of the ladder stay fixed")
    return out


def _validate_site_exchange(config: Dict[str, Any]) -> None:
    """
    Check the ``site_exchange`` list against the rest of the config.

    Parameters
    ----------
    config : dict
        The parsed config.

    Raises
    ------
    ValueError
        If the section is not a list of mappings, an entry has unknown or
        missing keys, an unknown ``sites`` method or a bad z-range, a
        species is not exchanged, or the structure is not a slab.
    """
    from .subsurface import SITE_METHODS
    entries = config["site_exchange"]
    if _geometry_kind(config["structure"]) != "slab":
        raise ValueError("the site_exchange section (NCMC site channels) needs a slab")
    if not isinstance(entries, list) or not entries:
        raise ValueError("site_exchange must be a non-empty list of channels, e.g. "
                         "[{species: [Pd], z_range_above_surface: [-1.0, 2.5]}]")
    species = set(_species_list(config))
    for k, sec in enumerate(entries):
        where = f"site_exchange[{k}]"
        _check_keys(sec, _SITE_EXCHANGE_KEYS, where)
        for key in ("species", "z_range_above_surface"):
            if key not in sec:
                raise ValueError(f"{where} needs {key!r}")
        sp = [sec["species"]] if isinstance(sec["species"], str) else [str(x) for x in sec["species"]]
        if not sp:
            raise ValueError(f"{where}.species is empty")
        foreign = sorted(set(sp) - species)
        if foreign:
            raise ValueError(f"{where}.species {foreign} are not exchanged species {sorted(species)}")
        zr = _floats(sec["z_range_above_surface"], f"{where}.z_range_above_surface")
        if len(zr) != 2 or not zr[0] < zr[1]:
            raise ValueError(f"{where}.z_range_above_surface must be [lo, hi] with lo < hi, "
                             f"got {sec['z_range_above_surface']}")
        method = str(sec.get("sites", "trilateration")).lower()
        if method not in SITE_METHODS:
            raise ValueError(f"{where}.sites must be one of {list(SITE_METHODS)}, got "
                             f"{sec['sites']!r}")
        for key in ("partition_gas_zone", "freeze_switched"):
            if key in sec:
                _subsurface_flag(sec[key], f"{where}.{key}")


def _validate_molecules(config: Dict[str, Any]) -> None:
    """
    Check the ``molecules`` section against the rest of the config.

    Parameters
    ----------
    config : dict
        The parsed config.

    Raises
    ------
    ValueError
        On unknown keys, a missing template, an unknown ``vibrations``
        model, a reservoir section or
        ``gas.composition`` next to it (the molecules are the reservoir), or
        atomically exchanged species that are not exchanged.
    """
    mols = config["molecules"]
    if not isinstance(mols, dict) or not mols:
        raise ValueError("molecules must map names to gas definitions, e.g. "
                         "{CO: {template: CO}}")
    for name, sec in mols.items():
        if not isinstance(sec, dict):
            raise ValueError(f"molecules.{name} must be a mapping (at least template: ...)")
        _check_keys(sec, _MOLECULE_KEYS, f"molecules.{name}")
        if "template" not in sec:
            raise ValueError(f"molecules.{name} needs a template (an ase.build.molecule name "
                             f"such as CO, or a structure file)")
        if "vibrations" in sec:
            from .reservoir import VIBRATION_MODELS
            if str(sec["vibrations"]).strip().lower() not in VIBRATION_MODELS:
                raise ValueError(f"molecules.{name}.vibrations must be one of "
                                 f"{list(VIBRATION_MODELS)}, not {sec['vibrations']!r}")
    for key in ("reservoir", "diatomic_reservoir"):
        if key in config:
            raise ValueError(f"with a molecules section the gases are the reservoir: drop {key}")
    gas = config["gas"]
    if "composition" in gas:
        raise ValueError("with a molecules section give each gas's mole fraction as "
                         "molecules.<name>.fraction instead of gas.composition")
    species = set(_species_list(config))
    atomic = gas.get("atomic_exchange") or []
    atomic = [atomic] if isinstance(atomic, str) else [str(a) for a in atomic]
    if set(atomic) - species:
        raise ValueError(f"gas.atomic_exchange {sorted(set(atomic) - species)} are not "
                         f"exchanged species {sorted(species)}")


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
    alloy = _alloy_fractions(struct_cfg)
    if "file" in struct_cfg:
        from ase.io import read as ase_read
        atoms = ase_read(struct_cfg["file"])
        if alloy:
            _substitute(atoms, alloy, int(struct_cfg.get("alloy_seed", 0)),
                        bool(struct_cfg.get("alloy_per_layer", False)))
        slab_species = list(dict.fromkeys(atoms.get_chemical_symbols()))
        return atoms, calc_spec_of(slab_species + list(alloy)), None
    if "cluster" in struct_cfg:
        atoms, calc_spec, a = _build_cluster(dict(struct_cfg["cluster"]), calc_spec_of, alloy)
    else:
        import ase.build
        build = dict(struct_cfg["build"])
        for key in ("surface", "symbol"):
            if key not in build:
                raise ValueError(f"structure.build needs {key!r} (e.g. surface: fcc111, symbol: Pd)")
        surface, symbol = str(build.pop("surface")), str(build.pop("symbol"))
        builder = getattr(ase.build, surface, None)
        if builder is None:
            raise ValueError(f"ase.build has no surface function {surface!r}")
        calc_spec = calc_spec_of([symbol] + list(alloy))
        a = _alloy_lattice_constant(build.pop("a", None), build.pop("a_range", None), surface[:3],
                                    symbol, alloy, calc_spec, "structure.build")
        if "size" in build:
            build["size"] = tuple(int(n) for n in build["size"])
            if "layers" in struct_cfg and len(build["size"]) == 2:   # the layers set the depth
                from .structures import substrate_layers
                build["size"] += (sum(substrate_layers(struct_cfg["layers"])),)
        atoms = builder(symbol, a=a, **build) if a is not None else builder(symbol, **build)
    if alloy:
        if struct_cfg.get("alloy_per_layer") and "cluster" in struct_cfg:
            raise ValueError("structure.alloy_per_layer needs a slab (layers along z)")
        _substitute(atoms, alloy, int(struct_cfg.get("alloy_seed", 0)),
                    bool(struct_cfg.get("alloy_per_layer", False)),
                    _as_float(struct_cfg.get("layer_tol", 1e-3), "structure.layer_tol"))
    return atoms, calc_spec, a


def _alloy_fractions(struct_cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    The ``structure.alloy`` block: ``{species: fraction}`` substituted at random.

    Parameters
    ----------
    struct_cfg : dict
        The ``structure`` section.

    Returns
    -------
    dict
        ``{species: fraction}`` (empty without the block).

    Raises
    ------
    ValueError
        If it is not a mapping, or the fractions are not in (0, 1) with a sum below 1.
    """
    alloy = struct_cfg.get("alloy") or {}
    if not isinstance(alloy, dict):
        raise ValueError("structure.alloy must map species to fractions, e.g. {Pt: 0.5}")
    out = {str(k): _as_float(v, f"structure.alloy.{k}") for k, v in alloy.items()}
    if any(not 0.0 < f < 1.0 for f in out.values()) or sum(out.values()) >= 1.0:
        raise ValueError(f"structure.alloy fractions must lie in (0, 1) and sum below 1, got {out}")
    return out


def _substitute(atoms, alloy: Dict[str, float], seed: int, per_layer: bool = False,
                layer_tol: float = 1e-3) -> None:
    """
    Random substitutional alloy: replace exact numbers of atoms, in place.

    ``round(fraction * n)`` atoms per species, drawn uniformly without
    replacement, so the composition is exact and uniform in space. With
    `per_layer`, n counts the atoms of each z layer and every layer gets the
    exact composition (a slab whose frozen layers stand for the bulk);
    otherwise n counts the whole structure (frozen and displaceable alike).

    Parameters
    ----------
    atoms : ase.Atoms
        The structure (all one host element, typically).
    alloy : dict
        ``{species: fraction}``.
    seed : int
        Random seed of the draw.
    per_layer : bool, optional
        Exact composition in every layer. Default False.
    layer_tol : float, optional
        z tolerance (A) grouping atoms into layers. Default 1e-3.

    Raises
    ------
    ValueError
        If `per_layer` and a fraction does not give a whole number of atoms
        in some layer.
    """
    rng = np.random.default_rng(seed)
    symbols = atoms.get_chemical_symbols()
    if per_layer:
        z = np.round(atoms.positions[:, 2] / layer_tol)
        groups = [np.flatnonzero(z == v) for v in np.unique(z)]
    else:
        groups = [np.arange(len(symbols))]
    for idx in groups:
        order = idx[rng.permutation(len(idx))]
        start = 0
        for sp, frac in alloy.items():
            exact = frac * len(idx)
            n = int(round(exact))
            if per_layer and abs(exact - n) > 1e-9:
                raise ValueError(f"structure.alloy {sp}: {frac} of a {len(idx)}-atom layer is "
                                 f"not a whole number of atoms")
            for i in order[start:start + n]:
                symbols[i] = sp
            start += n
    atoms.set_chemical_symbols(symbols)


def _alloy_lattice_constant(a: Any, a_range: Any, crystal: str, symbol: str,
                            alloy: Dict[str, float], calc_spec: _CalculatorSpec,
                            where: str) -> Optional[float]:
    """
    `_lattice_constant`, with Vegard's law for ``a: auto`` on an alloy.

    Parameters
    ----------
    a, a_range, crystal, symbol, calc_spec, where
        As for `_lattice_constant` (`symbol` the host).
    alloy : dict
        ``{species: fraction}`` substituted into the host (may be empty).

    Returns
    -------
    float or None
        The host's lattice constant, or for ``a: auto`` on an alloy the
        composition-weighted mean of every element's own fitted one.
    """
    a_host = _lattice_constant(a, a_range if not alloy else None, crystal, symbol, calc_spec, where)
    if a != "auto" or not alloy:
        return a_host
    x_host = 1.0 - sum(alloy.values())
    return x_host * a_host + sum(
        f * _lattice_constant("auto", None, crystal, sp, calc_spec, where) for sp, f in alloy.items())


#: Lattice of each ase.cluster function, for ``a: auto``.
_CLUSTER_CRYSTAL = {"Icosahedron": "fcc", "Octahedron": "fcc", "Decahedron": "fcc",
                    "FaceCenteredCubic": "fcc", "BodyCenteredCubic": "bcc"}


def _build_cluster(cl: Dict[str, Any], calc_spec_of: Callable[[List[str]], _CalculatorSpec],
                   alloy: Optional[Dict[str, float]] = None
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
    alloy : dict, optional
        ``{species: fraction}`` the caller substitutes afterwards: it sets
        the calculator's species and, with ``a: auto``, Vegard's law.

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
    alloy = alloy or {}
    calc_spec = calc_spec_of([symbol] + list(alloy))
    a = _alloy_lattice_constant(cl.pop("a", None), cl.pop("a_range", None),
                                _CLUSTER_CRYSTAL.get(shape, shape), symbol, alloy, calc_spec,
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
    if rtype == "bulk":
        return _build_bulk_reservoir(sp, cfg, mass_amu, calc_spec, ase_calc_getter)
    e_atom = cfg.get("e_atom_eV")
    e_atom = None if e_atom is None else _as_float(e_atom, f"reservoir.{sp}.e_atom_eV")
    if rtype == "monatomic":
        m = _as_float(cfg.get("mass_amu", mass_amu), f"reservoir.{sp}.mass_amu")
        return MonatomicIdealGasReservoir(m, e_atom if e_atom is not None
                                          else calc_spec.isolated_atom_energy(sp))
    if rtype != "diatomic":
        raise ValueError(f"reservoir.type must be 'diatomic', 'monatomic' or 'bulk', got {rtype!r}")

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


def _build_bulk_reservoir(sp: str, cfg: Dict[str, Any], mass_amu: float,
                          calc_spec: _CalculatorSpec, ase_calc_getter: Callable[[], Any]):
    """
    A bulk-crystal reservoir (``reservoir.<sp>.type: bulk``) for exchanged slab atoms.

    ``e_bulk_eV`` and ``mean_wavenumber_cm`` may be given; otherwise both
    come from the potential (`BulkSolidReservoir.from_calculator`) at
    ``lattice_constant_angstrom`` (default: the slab's, set by the runner).

    Parameters
    ----------
    sp : str
        The element.
    cfg : dict
        Its reservoir section.
    mass_amu : float
        Its atomic mass.
    calc_spec : _CalculatorSpec
        The calculator.
    ase_calc_getter : callable
        Returns the potential as an ASE calculator (or None).

    Returns
    -------
    BulkSolidReservoir
        The reservoir.

    Raises
    ------
    ValueError
        If only one of the two numbers is given, or the potential is needed
        but has no ASE counterpart.
    """
    from .reservoir import BulkSolidReservoir
    e_bulk, wn = cfg.get("e_bulk_eV"), cfg.get("mean_wavenumber_cm")
    if (e_bulk is None) != (wn is None):
        raise ValueError(f"reservoir.{sp}: give both e_bulk_eV and mean_wavenumber_cm, or "
                         f"neither (then both come from the potential)")
    if e_bulk is not None:
        return BulkSolidReservoir(float(mass_amu), _as_float(e_bulk, f"reservoir.{sp}.e_bulk_eV"),
                                  _as_float(wn, f"reservoir.{sp}.mean_wavenumber_cm"))
    ase_calc = ase_calc_getter()
    if ase_calc is None:
        raise ValueError(f"reservoir.{sp} (bulk): the calculator has no ASE counterpart; give "
                         f"e_bulk_eV and mean_wavenumber_cm")
    a = cfg.get("lattice_constant_angstrom")
    return BulkSolidReservoir.from_calculator(
        ase_calc, sp, crystal=str(cfg.get("crystal", "fcc")),
        a=None if a is None else _as_float(a, f"reservoir.{sp}.lattice_constant_angstrom"),
        supercell=int(cfg.get("supercell", 2)), mass_amu=float(mass_amu))


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
    from .reservoir import BulkSolidReservoir, DiatomicReservoir
    d = {f.name: getattr(res, f.name) for f in dataclasses.fields(res)
         if f.name != "calculator_info"}
    d["type"] = ("diatomic" if isinstance(res, DiatomicReservoir) else
                 "bulk" if isinstance(res, BulkSolidReservoir) else "monatomic")
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
    from .reservoir import BulkSolidReservoir, DiatomicReservoir, MonatomicIdealGasReservoir
    d = dict(d)
    cls = {"diatomic": DiatomicReservoir, "bulk": BulkSolidReservoir}.get(
        d.pop("type"), MonatomicIdealGasReservoir)
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


def _feed_fractions(gas_cfg: Dict[str, Any], species: Sequence[str],
                    solid: Sequence[str] = ()) -> List[float]:
    """
    Feed mole fractions from ``gas.composition``.

    Parameters
    ----------
    gas_cfg : dict
        The ``gas`` section.
    species : sequence of str
        The exchanged species.
    solid : sequence of str, optional
        Species with a bulk-solid reservoir: no gas fraction (their mu does
        not depend on the pressure); they get 1.0, the gases are
        normalised among themselves. Default none.

    Returns
    -------
    list of float
        One fraction per species (1.0 for a single gas species), the gas
        fractions normalised to sum to 1.

    Raises
    ------
    ValueError
        If several species are exchanged without a composition covering
        each of them.
    """
    comp = gas_cfg.get("composition")
    gases = [s for s in species if s not in solid]
    if comp is None and len(gases) <= 1:
        return [1.0] * len(species)
    if comp is None:
        if len(species) > 1:
            raise ValueError("several exchanged species with a pressure ladder or a (T, p) "
                             "phase diagram need gas.composition (feed mole fractions, "
                             "e.g. {O: 0.999, Cl: 0.001})")
        return [1.0]
    missing = [s for s in gases if s not in comp]
    if missing:
        raise ValueError(f"gas.composition lacks {missing}")
    x = {s: _as_float(comp[s], f"gas.composition.{s}") for s in gases}
    total = sum(x.values())
    return [x[s] / total if s in x else 1.0 for s in species]


def _build_molecular_feed(config: Dict[str, Any], species: Sequence[str],
                          masses: Dict[str, float], calc_spec: "_CalculatorSpec",
                          ase_calc_getter: Callable[[], Any], base: str = "."
                          ) -> Tuple[Any, list, Tuple[str, ...]]:
    """
    The gases of the ``molecules`` section, their insertion moves and the atomic species.

    Parameters
    ----------
    config : dict
        The parsed config.
    species : sequence of str
        The exchanged elements (``gas.species``).
    masses : dict
        Their atomic masses (``gas.mass_amu``); the gas molecules use them
        too (Lambda, moments of inertia, Hessian frequencies).
    calc_spec : _CalculatorSpec
        The calculator.
    ase_calc_getter : callable
        Returns the potential as an ASE calculator (or None).
    base : str, optional
        Directory that relative template files are resolved against.
        Default ``'.'``.

    Returns
    -------
    tuple
        ``(feed, moves, atomic_exchange)``: a `MolecularGasFeed`, the
        `MoleculeMove` list and the atomically exchanged species.

    Raises
    ------
    ValueError
        If an exchanged species is in no gas, a move's chemical potential
        is not fixed by the gases, or a gas has to be described without an
        ASE calculator but lacks ``energy_eV`` or vibrations.
    """
    from .molecules import MoleculeMove
    from .reservoir import SPIN_MULTIPLICITIES, SYMMETRY_NUMBERS, MolecularGas, MolecularGasFeed
    from .units import default_mass_amu

    gases, fractions, moves = [], [], []
    for name, sec in config["molecules"].items():
        tmpl = str(sec["template"])
        path = _resolve_existing(tmpl, base)
        if os.path.isfile(path):
            import ase.io
            template = ase.io.read(path)
        else:
            template = tmpl
        vib = sec.get("vibrational_wavenumbers_cm")
        vib = None if vib is None else _floats(vib, f"molecules.{name}.vibrational_wavenumbers_cm")
        sym, spin = sec.get("symmetry_number"), sec.get("spin_multiplicity")
        vibrations = str(sec.get("vibrations", "classical"))
        calc = ase_calc_getter() if "energy_eV" not in sec else None
        if calc is not None:
            gas = MolecularGas.from_calculator(calc, template, name=str(name),
                                               symmetry_number=None if sym is None else int(sym),
                                               spin_multiplicity=None if spin is None else int(spin),
                                               vib_wavenumbers_cm=vib, masses_amu=masses,
                                               vibrations=vibrations)
        else:
            from ase.build import molecule
            atoms = molecule(template) if isinstance(template, str) else template
            if "energy_eV" not in sec or (vib is None and len(atoms) > 1):
                raise ValueError(f"molecules.{name}: calculator.type {calc_spec.type} has no ASE "
                                 f"counterpart to relax the molecule with; give energy_eV (and "
                                 f"vibrational_wavenumbers_cm)")
            if sym is None and len(atoms) > 1 and str(name) not in SYMMETRY_NUMBERS:
                raise ValueError(f"molecules.{name}: give symmetry_number")
            gas = MolecularGas(str(name), atoms.get_chemical_symbols(), atoms.get_positions(),
                               _as_float(sec["energy_eV"], f"molecules.{name}.energy_eV"),
                               tuple(vib or ()),
                               int(sym) if sym is not None else SYMMETRY_NUMBERS.get(str(name), 1),
                               int(spin) if spin is not None
                               else SPIN_MULTIPLICITIES.get(str(name), 1),
                               tuple(masses.get(s, default_mass_amu(s))
                                     for s in atoms.get_chemical_symbols()),
                               vibrations)
        gases.append(gas)
        fractions.append(_as_float(sec.get("fraction", 1.0), f"molecules.{name}.fraction"))
        if sec.get("insert", True) and len(gas.symbols) > 1:
            shell = sec.get("shell_angstrom")
            moves.append(MoleculeMove(
                gas.name, gas.symbols, gas.positions_angstrom, anchor=sec.get("anchor"),
                shell=None if shell is None else tuple(_floats(shell, f"molecules.{name}.shell_angstrom")),
                bond_sigma=_as_float(sec.get("bond_sigma_angstrom", 0.05),
                                     f"molecules.{name}.bond_sigma_angstrom"),
                angle_sigma_deg=_as_float(sec.get("angle_sigma_deg", 10.0),
                                          f"molecules.{name}.angle_sigma_deg"),
                weight=_as_float(sec.get("weight", 1.0), f"molecules.{name}.weight")))
    feed = MolecularGasFeed(species, gases, fractions, masses)
    covered = set().union(*(g.composition for g in gases))
    missing = [s for s in species if s not in covered]
    if missing:
        raise ValueError(f"exchanged species {missing} are in no gas of the molecules section, "
                         f"so their chemical potential is undefined")
    atomic = config["gas"].get("atomic_exchange") or []
    atomic = tuple([atomic] if isinstance(atomic, str) else [str(a) for a in atomic])
    feed.check_moves({**{f"atomic {a}": {a: 1} for a in atomic},
                      **{f"molecule {m.name}": m.composition for m in moves}})
    if not moves and not atomic:
        raise ValueError("no GC moves: set insert: true on a molecule or list "
                         "gas.atomic_exchange")
    return feed, moves, atomic


def _subsurface_settings(config: Dict[str, Any], geometry: Any, species: Sequence[str],
                         feed: Any, molecule_moves: Sequence, atomic_exchange: Optional[Sequence[str]]):
    """
    The ``subsurface`` section as `regc.subsurface.Subsurface` settings.

    Default species: every exchanged species without a ``molecules``
    section; with one, the atomically exchanged species and the element of
    every homonuclear diatomic gas (O2, H2: their subsurface attempts are
    atomic), not the elements of heteronuclear or larger molecules, which
    are exchanged only above the surface.

    Parameters
    ----------
    config : dict
        The parsed config.
    geometry : SlabGeometry
        The slab (``top_z``).
    species : sequence of str
        The exchanged species.
    feed : GasFeed or MolecularGasFeed or None
        The gas feed.
    molecule_moves : sequence of MoleculeMove
        The molecular moves.
    atomic_exchange : sequence of str or None
        The atomically exchanged species (None without molecules).

    Returns
    -------
    Subsurface or None
        The settings; None without the section.

    Raises
    ------
    ValueError
        If no species qualifies, a species is not exchanged, or its
        chemical potential is not fixed by the gases.
    """
    from .subsurface import Subsurface
    sec = config.get("subsurface")
    if sec is None:
        return None
    sec = sec or {}
    if "species" in sec:
        sub_species = [sec["species"]] if isinstance(sec["species"], str) else [
            str(x) for x in sec["species"]]
    elif "molecules" not in config:
        sub_species = list(species)
    else:
        sub_species = list(atomic_exchange or [])
        for g in feed.gases:
            comp = g.composition
            if len(comp) == 1 and sum(comp.values()) == 2:     # O2, H2, N2
                sub_species += list(comp)
        sub_species = list(dict.fromkeys(sub_species))
    if not sub_species:
        raise ValueError("subsurface: no species qualifies (heteronuclear and larger molecules "
                         "are exchanged only above the surface); list subsurface.species")
    foreign = sorted(set(sub_species) - set(species))
    if foreign:
        raise ValueError(f"subsurface.species {foreign} are not exchanged species {list(species)}")
    if feed is not None and hasattr(feed, "check_moves"):
        feed.check_moves({f"subsurface {e}": {e: 1} for e in sub_species})
    z_cut = geometry.top_z - _as_float(sec.get("z_cut_below_surface", 0.0),
                                       "subsurface.z_cut_below_surface")
    depth = _as_float(sec.get("depth_angstrom", 4.0), "subsurface.depth_angstrom")
    return Subsurface(
        z_cut=z_cut, z_min=z_cut - depth, species=tuple(sub_species),
        delta=_as_float(sec.get("delta_angstrom", 0.3), "subsurface.delta_angstrom"),
        ncmc_steps=int(sec.get("ncmc_steps", 10)), kernel=str(sec.get("kernel", "hmc")).lower(),
        kernel_steps=int(sec.get("kernel_steps", 1)),
        leapfrog_steps=int(sec.get("leapfrog_steps", 5)),
        dt=_as_float(sec.get("timestep_fs", 1.0), "subsurface.timestep_fs") / FS_PER_INTERNAL_TIME,
        friction=_as_float(sec.get("friction_per_fs", 0.05), "subsurface.friction_per_fs")
        * FS_PER_INTERNAL_TIME,
        max_disp=_as_float(sec.get("max_disp", 0.1), "subsurface.max_disp"),
        weight=_as_float(sec.get("weight", 1.0), "subsurface.weight"),
        site_min_distance=_as_float(sec.get("site_min_distance", 1.0),
                                    "subsurface.site_min_distance"),
        freeze_switched=_subsurface_flag(sec.get("freeze_switched", False),
                                         "subsurface.freeze_switched"))


def _site_exchange_settings(config: Dict[str, Any], geometry: Any, species: Sequence[str],
                            feed: Any, bond_length: Optional[float]) -> list:
    """
    The ``site_exchange`` list as `regc.subsurface.Subsurface` channels.

    Each entry's region is ``top_z + z_range_above_surface``; defaults
    differ from ``subsurface`` where the channel is meant for metal atoms:
    trilateration sites, ``defensive_fraction`` 0.05, an overlay
    (``partition_gas_zone: false``) and ``freeze_switched: true``.

    Parameters
    ----------
    config : dict
        The parsed (validated) config.
    geometry : SlabGeometry
        The slab (``top_z``).
    species : sequence of str
        The exchanged species.
    feed : GasFeed or MolecularGasFeed or None
        The gas feed.
    bond_length : float or None
        Default trilateration R0 (A): the nearest-neighbour distance of the
        slab's crystal.

    Returns
    -------
    list of Subsurface
        The channels (empty without the section).

    Raises
    ------
    ValueError
        If a channel's species' chemical potential is not fixed by the
        gases, a trilateration channel has no bond length, or a setting is
        invalid.
    """
    from .subsurface import Subsurface
    out = []
    for k, sec in enumerate(config.get("site_exchange") or []):
        where = f"site_exchange[{k}]"
        sp = [sec["species"]] if isinstance(sec["species"], str) else [str(x) for x in sec["species"]]
        if feed is not None and hasattr(feed, "check_moves"):
            feed.check_moves({f"site exchange {e}": {e: 1} for e in sp})
        lo, hi = _floats(sec["z_range_above_surface"], f"{where}.z_range_above_surface")
        method = str(sec.get("sites", "trilateration")).lower()
        r0 = sec.get("bond_length_angstrom", bond_length if method == "trilateration" else None)
        if method == "trilateration" and r0 is None:
            raise ValueError(f"{where}: give bond_length_angstrom (the slab's crystal is not "
                             f"known, so its nearest-neighbour distance can't be the default)")
        r0 = None if r0 is None else _as_float(r0, f"{where}.bond_length_angstrom")
        min_d = sec.get("min_distance_angstrom", 0.6 * r0 if method == "trilateration" else 1.0)
        merge = sec.get("merge_angstrom")
        out.append(Subsurface(
            z_cut=geometry.top_z + hi, z_min=geometry.top_z + lo, species=tuple(sp),
            delta=_as_float(sec.get("delta_angstrom", 0.3), f"{where}.delta_angstrom"),
            ncmc_steps=int(sec.get("ncmc_steps", 10)), kernel=str(sec.get("kernel", "hmc")).lower(),
            kernel_steps=int(sec.get("kernel_steps", 1)),
            leapfrog_steps=int(sec.get("leapfrog_steps", 5)),
            dt=_as_float(sec.get("timestep_fs", 1.0), f"{where}.timestep_fs") / FS_PER_INTERNAL_TIME,
            friction=_as_float(sec.get("friction_per_fs", 0.05), f"{where}.friction_per_fs")
            * FS_PER_INTERNAL_TIME,
            max_disp=_as_float(sec.get("max_disp", 0.1), f"{where}.max_disp"),
            weight=_as_float(sec.get("weight", 1.0), f"{where}.weight"),
            site_min_distance=_as_float(min_d, f"{where}.min_distance_angstrom"),
            freeze_switched=_subsurface_flag(sec.get("freeze_switched", True),
                                             f"{where}.freeze_switched"),
            site_method=method, bond_length=r0,
            site_merge=None if merge is None else _as_float(merge, f"{where}.merge_angstrom"),
            defensive_fraction=_as_float(sec.get("defensive_fraction", 0.05),
                                         f"{where}.defensive_fraction"),
            partition_gas_zone=_subsurface_flag(sec.get("partition_gas_zone", False),
                                                f"{where}.partition_gas_zone")))
    return out


def _slab_bond_length(struct_cfg: Dict[str, Any], a: Optional[float],
                      geometry: Any) -> Optional[float]:
    """
    Nearest-neighbour distance of the slab's bulk crystal (default trilateration R0).

    From the lattice constant of a built slab (``structure.build``: fcc
    a / sqrt 2, bcc a sqrt 3 / 2, hcp a, diamond a sqrt 3 / 4; the ASE
    reference constant if ``a`` was not given); otherwise the shortest
    distance between the slab's frozen and substrate atoms.

    Parameters
    ----------
    struct_cfg : dict
        The ``structure`` section.
    a : float or None
        The lattice constant the runner used (None if ASE's default or a file).
    geometry : SlabGeometry
        The slab.

    Returns
    -------
    float or None
        The distance (A), or None if it can't be told.
    """
    build = struct_cfg.get("build")
    if isinstance(build, dict) and "surface" in build:
        crystal, symbol = str(build["surface"])[:3], str(build.get("symbol", ""))
        if crystal == "dia":
            crystal = "diamond"
        if a is None:
            from .structures import reference_lattice_constant
            a = reference_lattice_constant(symbol, crystal)
        factor = {"fcc": 1.0 / np.sqrt(2.0), "bcc": np.sqrt(3.0) / 2.0, "hcp": 1.0,
                  "diamond": np.sqrt(3.0) / 4.0}.get(crystal)
        if a is not None and factor is not None:
            return float(a) * float(factor)
    metal = np.vstack([geometry.frozen_positions, geometry.substrate_positions]).reshape(-1, 3)
    if len(metal) < 2:
        return None
    A = np.asarray(geometry.cell_xy, dtype=float)
    shifts = np.array([[i * A[0, 0] + j * A[1, 0], i * A[0, 1] + j * A[1, 1], 0.0]
                       for i in (-1, 0, 1) for j in (-1, 0, 1)])
    d = np.linalg.norm(metal[:, None, None, :] + shifts[None, :, None, :] - metal[None, None],
                       axis=-1)
    d = d[d > 1e-6]
    return float(d.min()) if d.size else None


def _subsurface_flag(value: Any, where: str) -> bool:
    """
    A YAML boolean, strictly.

    Parameters
    ----------
    value : Any
        The value.
    where : str
        The key, for the message.

    Returns
    -------
    bool
        The flag.

    Raises
    ------
    ValueError
        If the value is not true or false.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"{where} must be true or false, got {value!r}")


def _feed_to_meta(feed) -> Dict[str, Any]:
    """
    The metadata entries describing a gas feed.

    Parameters
    ----------
    feed : GasFeed or MolecularGasFeed
        The feed.

    Returns
    -------
    dict
        ``composition``, ``reservoirs`` and, for molecules, ``gas_feed``.
    """
    if hasattr(feed, "gases"):
        return {"composition": {g.name: x for g, x in zip(feed.gases, feed.fractions)},
                "reservoirs": {g.name: g.to_dict() for g in feed.gases},
                "gas_feed": feed.to_dict()}
    return {"composition": dict(zip(feed.species, feed.fractions)),
            "reservoirs": {s: _reservoir_to_dict(r) for s, r in zip(feed.species, feed.reservoirs)}}


def _feed_from_meta(meta: Dict[str, Any], stored: Optional[Dict[str, Any]] = None):
    """
    Rebuild the gas feed of a run from its metadata.

    Parameters
    ----------
    meta : dict
        ``run_metadata.json``.
    stored : dict, optional
        Reservoirs stored for another potential (`reevaluate_from_config`):
        a feed dict, or ``{species: reservoir dict}``. Default None: the
        run's own.

    Returns
    -------
    GasFeed or MolecularGasFeed
        The feed.
    """
    from .reservoir import MolecularGasFeed
    if stored is not None:
        if stored.get("type") == "molecular_feed":
            return MolecularGasFeed.from_dict(stored)
        return GasFeed(meta["species"], [_reservoir_from_dict(stored[s]) for s in meta["species"]],
                       [meta["composition"][s] for s in meta["species"]])
    if meta.get("gas_feed"):
        return MolecularGasFeed.from_dict(meta["gas_feed"])
    return GasFeed(meta["species"], [_reservoir_from_dict(meta["reservoirs"][s])
                                     for s in meta["species"]],
                   [meta["composition"][s] for s in meta["species"]])


def _feed_record(feed) -> Dict[str, Any]:
    """
    The reservoirs of a feed as stored with a re-evaluated potential.

    Parameters
    ----------
    feed : GasFeed or MolecularGasFeed
        The feed.

    Returns
    -------
    dict
        What `_feed_from_meta` takes as ``stored``.
    """
    if hasattr(feed, "gases"):
        return feed.to_dict()
    return {s: _reservoir_to_dict(r) for s, r in zip(feed.species, feed.reservoirs)}


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
        ``insertion_shell`` (cluster), ``system_factory``, ``subsurface``
        (`Subsurface` or None) and ``site_exchange`` (list of `Subsurface`
        channels), and ``metadata`` (JSON-able; what a resumed run must agree with,
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

    # species the slab can turn into by SGC swaps, even if absent at the start
    sgc_species = [str(x) for x in (config.get("sgc") or {}).get("species", [])]

    def calc_spec_of(slab_species):
        return _CalculatorSpec(calc_cfg, list(dict.fromkeys(list(slab_species) + sgc_species
                                                            + species)), masses)

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
        nfl, n_th, ppl = struct_cfg.get("n_frozen_layers"), None, 1
        layer_tol = _as_float(struct_cfg.get("layer_tol", 1e-3), "structure.layer_tol")
        if "layers" in struct_cfg:
            from .structures import layer_planes, substrate_layers
            nfl, n_th, n_free = substrate_layers(struct_cfg["layers"])
            try:     # open / stepped surfaces: several atomic planes per layer
                ppl = layer_planes(atoms, nfl + n_th + n_free, layer_tol)
            except ValueError as err:
                raise ValueError(f"structure.layers adds up to {nfl + n_th + n_free} layers, "
                                 f"which does not match the slab: {err}") from None
        geometry = slab_geometry_from_atoms(
            atoms, n_frozen_layers=None if nfl is None else int(nfl),
            headroom=_as_float(struct_cfg.get("headroom_angstrom", 5.0),
                               "structure.headroom_angstrom"),
            layer_tol=layer_tol, n_thermostat_layers=n_th, planes_per_layer=ppl,
            n_exchangeable_layers=int(struct_cfg.get("exchangeable_top_layers", 0)))
        if geometry.mobile_species is not None:
            foreign = sorted(set(geometry.mobile_species) - set(species))
            if foreign:
                raise ValueError(f"structure.exchangeable_top_layers: the top layers hold "
                                 f"{foreign}, which are not exchanged species {list(species)} "
                                 f"(add them to gas.species with a reservoir, e.g. type: bulk)")
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
    molecule_moves, atomic_exchange = [], None
    cache = {}

    def ase_calc():
        if "c" not in cache:
            cache["c"] = calc_spec.ase_calculator()
        return cache["c"]
    if "molecules" in config:
        feed, molecule_moves, atomic_exchange = _build_molecular_feed(
            config, species, masses, calc_spec, ase_calc)
    elif needs_reservoir:
        res_cfgs = _per_species_reservoir_cfg(config, species)
        solid = [s for s in species if str(res_cfgs[s].get("type", "")).lower() == "bulk"]
        for s in solid:     # the bulk crystal at the slab's lattice constant
            res_cfgs[s].setdefault("lattice_constant_angstrom", a)
        reservoirs = [_build_reservoir(s, res_cfgs[s], masses[s], calc_spec, ase_calc)
                      for s in species]
        feed = GasFeed(species, reservoirs, _feed_fractions(gas_cfg, species, solid))
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
    sgc = _sgc_settings(config, geometry)
    subsurface = _subsurface_settings(config, geometry, species, feed, molecule_moves,
                                      atomic_exchange)
    site_exchange = (_site_exchange_settings(config, geometry, species, feed,
                                             _slab_bond_length(struct_cfg, a, geometry))
                     if "site_exchange" in config else [])
    channels = ([subsurface] if subsurface is not None else []) + site_exchange

    def system_factory(rng):
        system = geometry.make_system(calc_spec.make(cell, pbc), exchange, mass_of_species=masses,
                                      **region)
        if sgc:
            system.set_sgc_species(sgc["species"], canonical=sgc["canonical"])
        if atomic_exchange is not None:
            system.set_molecules(molecule_moves, atomic_exchange=atomic_exchange)
        if channels:
            system.set_subsurface(channels)
        return system

    scheme = _scheme(re_cfg)
    sampling = {"scheme": scheme, "record_every": re_cfg.get("record_every"),
                "adsorbed_cutoff": re_cfg.get("adsorbed_cutoff"), "max_disp": re_cfg.get("max_disp")}
    if scheme == "regc":
        sampling.update({k: re_cfg.get(k) for k in ("x0", "diffusion", "diffusion_steps",
                                                     "displace_substrate", "md", "hmc")})
        adapt = _adapt_ladder_settings(config)
        if adapt is not None:   # only then: older runs' metadata stay unchanged
            sampling["adapt_ladder"] = adapt
    metadata = {
        "species": species, "masses_amu": masses,
        "temperatures_K": temperatures_K, "mu_grid_eV": np.asarray(mu_grid).tolist(),
        "pressures": pressures, "pressure_unit": pressure_unit,
        "composition": None, "reservoirs": None,
        "structure": struct_cfg, "calculator": calc_cfg, "lattice_constant_angstrom": a,
        "geometry": kind, "n_surface": geometry.n_surface, "area_angstrom2": geometry.area,
        "n_atoms": len(geometry.frozen_positions) + len(geometry.substrate_positions),
        "initial_mobile_counts": (
            {} if getattr(geometry, "mobile_species", None) is None else
            {str(sp): int(np.sum(geometry.mobile_species == sp))
             for sp in dict.fromkeys(geometry.mobile_species)}),
        "insertion_z_range": insertion_z_range, "insertion_shell": insertion_shell,
        "sampling": sampling,
    }
    if feed is not None:
        metadata.update(_feed_to_meta(feed))
    if subsurface is not None:
        metadata["subsurface"] = subsurface.metadata()
    if site_exchange:
        metadata["site_exchange"] = [c.metadata() for c in site_exchange]
    if atomic_exchange is not None:
        metadata["molecules"] = [m.metadata() for m in molecule_moves]
        metadata["atomic_exchange"] = list(atomic_exchange)
    if config.get("observables"):
        metadata["observables"] = config["observables"]
    if sgc:
        metadata["sgc"] = dict(sgc)

    if verbose:
        if a is not None:
            print(f"lattice constant from the potential: {a:.4f} A")
        print(f"{kind}: {geometry.summary()}")
        if insertion_z_range is not None:
            print(f"insertions in z in ({insertion_z_range[0]:.2f}, {insertion_z_range[1]:.2f}] A")
        if insertion_shell is not None:
            print(f"insertions in the shell {insertion_shell[0]:.2f} <= r < "
                  f"{insertion_shell[1]:.2f} A around the cluster centre")
        if feed is not None and hasattr(feed, "gases"):
            for g, x in zip(feed.gases, feed.fractions):
                print(f"gas {g.name} (x = {x:.4g}): E = {g.energy_eV:.4f} eV, vibrations "
                      f"{[round(w) for w in g.vib_wavenumbers_cm]} cm^-1, sigma = "
                      f"{g.symmetry_number}, spin multiplicity {g.spin_multiplicity}, "
                      f"{g.vibrations} vibrations")
            for m in molecule_moves:
                print(f"molecule move {m.name}: anchor {m.anchor} + {list(m.partners)}, shell "
                      f"[{m.shell[0]:.2f}, {m.shell[1]:.2f}] A, weight {m.weight:g}")
            print(f"atomic exchange: {list(atomic_exchange) or 'none'}")
        elif feed is not None:
            for s, r in zip(species, feed.reservoirs):
                d0 = ""
                if hasattr(r, "D0_eV"):
                    src = (r.calculator_info or {}).get("d0_source", "given")
                    d0 = f"D0 = {r.D0_eV:.4f} eV ({src}), "
                if hasattr(r, "e_bulk_eV"):
                    print(f"{s} reservoir ({type(r).__name__}): e_bulk = {r.e_bulk_eV:.4f} eV/atom, "
                          f"mean phonon {r.mean_wavenumber_cm:.1f} cm^-1; mu = "
                          + ", ".join(f"{r.mu_eV(T):.4f}" for T in (temperatures_K[0],
                                                                    temperatures_K[-1]))
                          + f" eV at {temperatures_K[0]:g}, {temperatures_K[-1]:g} K")
                else:
                    print(f"{s} reservoir ({type(r).__name__}): {d0}e_atom = {r.e_atom_eV:.4f} eV")
        if sgc and sgc["canonical"]:
            print(f"canonical substrate alloy {sgc['species']}: {sgc['n_swappable']} "
                  f"swappable atoms {dict(zip(sgc['species'], sgc['counts']))} (fixed), "
                  f"{sgc['attempts_per_step']} pair-swap attempts per step")
        elif sgc:
            others = sgc["species"][1:]
            field = "{" + ", ".join(f"{sp}: {d:+.4f}" for sp, d in zip(others, sgc["dmu"][1:])) + "} eV"
            if any(sgc["log_ratio"]):
                field += " + kT x {" + ", ".join(
                    f"{sp}: {r:+.4f}" for sp, r in zip(others, sgc["log_ratio"][1:])) + "}"
            print(f"semi-grand-canonical substrate {sgc['species']}: {sgc['n_swappable']} "
                  f"swappable atoms {dict(zip(sgc['species'], sgc['counts']))}, "
                  f"dmu vs {sgc['species'][0]} = {field}, "
                  f"{sgc['attempts_per_step']} identity attempts per step")
        if subsurface is not None:
            probe = geometry.make_system(calc_spec.make(cell, pbc), exchange,
                                         mass_of_species=masses, **region)
            n_sites = len(subsurface.sites(probe))
            print(f"subsurface NCMC: {list(subsurface.species)} in {subsurface.z_min:.2f} <= z < "
                  f"{subsurface.z_cut:.2f} A ({n_sites} interstitial sites), gas moves above; "
                  f"{subsurface.ncmc_steps} lambda steps, {subsurface.kernel} kernel, "
                  f"weight {subsurface.weight:g}")
        for k, c in enumerate(site_exchange):
            probe = geometry.make_system(calc_spec.make(cell, pbc), exchange,
                                         mass_of_species=masses, **region)
            r0 = "" if c.bond_length is None else f", R0 = {c.bond_length:.3f} A"
            print(f"site exchange {k}: {list(c.species)} in {c.z_min:.2f} <= z < {c.z_cut:.2f} A, "
                  f"{len(c.sites(probe))} {c.site_method} sites now{r0}, delta {c.delta:g} A, "
                  f"eps {c.defensive_fraction:g}, "
                  f"{'partitions the gas zone' if c.partition_gas_zone else 'overlay'}; "
                  f"{c.ncmc_steps} lambda steps, {c.kernel} kernel"
                  f"{', switched atom held' if c.freeze_switched else ''}, weight {c.weight:g}")
        n_mu = np.shape(mu_grid)[-1 if len(species) == 1 else -2]
        print(f"replica grid: {len(temperatures_K)} T x {n_mu} "
              f"{'p' if pressures is not None else 'mu'} = {len(temperatures_K) * n_mu} replicas")
        adapt = sampling.get("adapt_ladder")
        if adapt:
            print(f"adaptive ladder: after each of the first {adapt['chunks']} chunks (with >= "
                  f"{adapt['min_attempts']} swap attempts per pair), "
                  f"{'one p ladder per T' if adapt['per_row'] else 'one shared p ladder'}, gaps "
                  f"x{adapt['trust']:g} at most per step and within [{adapt['min_gap_ratio']:g}, "
                  f"{adapt['max_gap_ratio']:g}] x the initial ones; samples discarded at each "
                  f"change, then frozen")

    return {"geometry": geometry, "lattice_constant": a, "calc_spec": calc_spec,
            "species": species, "masses": masses, "feed": feed,
            "temperatures_K": temperatures_K, "mu_grid": mu_grid, "pressures": pressures,
            "pressure_unit": pressure_unit, "lambda_cubed_of_T": lambda_cubed_of_T,
            "insertion_z_range": insertion_z_range, "insertion_shell": insertion_shell,
            "system_factory": system_factory, "sgc": sgc,
            "molecules": molecule_moves, "atomic_exchange": atomic_exchange,
            "subsurface": subsurface, "site_exchange": site_exchange,
            "metadata": metadata}


def _sgc_settings(config: Dict[str, Any], geometry: Any) -> Optional[Dict[str, Any]]:
    """
    The ``sgc`` section: a semi-grand-canonical (identity-swapping) substrate.

    Parameters
    ----------
    config : dict
        The parsed config; ``sgc`` has ``species`` (reference first),
        ``dmu_eV`` (``{species: mu_species - mu_reference}``, eV; the
        reference may be omitted), ``attempts_per_step`` (default 1) and
        optionally ``bulk_composition`` (``{species: x}`` of the bulk
        reservoir: ``dmu_eV`` is then its energy part dE/dN at that
        composition and the field at T is ``dmu_eV + kT ln(x_k / x_ref)``,
        ideal mixing), or ``ensemble: canonical`` instead of the field:
        pair swaps at fixed composition (a free nanoparticle).
    geometry : SlabGeometry or ClusterGeometry
        The structure (its displaceable substrate is what swaps).

    Returns
    -------
    dict or None
        ``species``, ``dmu`` (list, reference 0; zeros when canonical),
        ``canonical`` (bool), ``log_ratio``
        (ln(x_k / x_ref), zeros without ``bulk_composition``),
        ``bulk_composition``, ``attempts_per_step``, ``counts`` (initial
        swappable atoms per species) and ``n_swappable``; None without the
        section.

    Raises
    ------
    ValueError
        If a species has no dmu, the reference's dmu is not 0, or the
        substrate holds none of the species.
    """
    sec = config.get("sgc")
    if not sec:
        return None
    species = [str(s) for s in sec["species"]]
    if len(species) < 2 or len(set(species)) != len(species):
        raise ValueError(f"sgc.species must be two or more distinct species, got {species}")
    canonical = str(sec.get("ensemble", "semigrand")).lower() == "canonical"
    given = {sp: 0.0 for sp in species[1:]} if canonical else sec["dmu_eV"]
    if not isinstance(given, dict):
        if len(species) != 2:
            raise ValueError("sgc.dmu_eV must map species to values with more than two species")
        given = {species[1]: given}
    unknown = sorted(set(map(str, given)) - set(species))
    if unknown:
        raise ValueError(f"sgc.dmu_eV has entries {unknown} not in sgc.species {species}")
    given = {str(k): _as_float(v, f"sgc.dmu_eV.{k}") for k, v in given.items()}
    if given.get(species[0], 0.0) != 0.0:
        raise ValueError(f"sgc.dmu_eV.{species[0]} must be 0 (the reference) or omitted")
    missing = [sp for sp in species[1:] if sp not in given]
    if missing:
        raise ValueError(f"sgc.dmu_eV needs a value for {missing}")
    sub = np.asarray(geometry.substrate_species).astype(str)
    counts = [int(np.count_nonzero(sub == sp)) for sp in species]
    if sum(counts) == 0:
        raise ValueError(f"no displaceable substrate atom is one of sgc.species {species}: "
                         f"identity swaps only act on non-frozen atoms")
    attempts = int(sec.get("attempts_per_step", 1))
    if attempts < 1:
        raise ValueError("sgc.attempts_per_step must be at least 1")
    log_ratio = [0.0] * len(species)
    bulk = sec.get("bulk_composition")
    if bulk:
        if not isinstance(bulk, dict) or set(map(str, bulk)) != set(species):
            raise ValueError(f"sgc.bulk_composition must give a fraction for each of {species}")
        x = {str(k): _as_float(v, f"sgc.bulk_composition.{k}") for k, v in bulk.items()}
        if any(v <= 0 for v in x.values()) or abs(sum(x.values()) - 1.0) > 1e-6:
            raise ValueError(f"sgc.bulk_composition fractions must be positive and sum to 1, got {x}")
        log_ratio = [float(np.log(x[sp] / x[species[0]])) for sp in species]
    return {"species": species, "dmu": [0.0] + [given[sp] for sp in species[1:]],
            "canonical": canonical,
            "log_ratio": log_ratio, "bulk_composition": bulk or None,
            "attempts_per_step": attempts, "counts": counts, "n_swappable": int(sum(counts))}


# =============================================================================
# Sampling
# =============================================================================
def _convergence_settings(re_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    The rho(u) convergence keys of ``replica_exchange``.

    ``convergence_mode`` defaults to ``noise`` in configs (the noise-aware
    criterion, `ReplicaExchangeGCMC.rho_u_noise_check`); ``blocks`` is the
    original fixed-block test. In ``noise`` mode ``convergence_tol`` bounds
    the largest replica's ratio to the noise floor (default 3.0) and
    ``convergence_median_tol`` the median one (default 1.3), so a value
    below 1 can practically never be met; a warning says so.

    Parameters
    ----------
    re_cfg : dict
        The ``replica_exchange`` section.

    Returns
    -------
    dict
        ``convergence_mode``, ``convergence_tol``, ``convergence_bins``,
        ``convergence_min_neff``, ``convergence_median_tol``, ``convergence_discard``,
        ``convergence_consecutive``, ``convergence_min_round_trips`` for
        `run_with_checkpoints` / `run_regc`.

    Raises
    ------
    ValueError
        On an unknown mode.
    """
    from .engine import NOISE_CONSECUTIVE, NOISE_MIN_NEFF, NOISE_MIN_ROUND_TRIPS
    mode = str(re_cfg.get("convergence_mode", "noise")).lower()
    if mode not in ("noise", "blocks"):
        raise ValueError(f"replica_exchange.convergence_mode must be noise or blocks, got {mode!r}")
    tol = re_cfg.get("convergence_tol")
    tol = None if tol is None else _as_float(tol, "replica_exchange.convergence_tol")
    if mode == "noise" and tol is not None and tol < 1.0 and re_cfg.get("convergence_every"):
        warnings.warn(f"replica_exchange.convergence_tol = {tol} with convergence_mode: noise is a "
                      f"multiple of the sampling-noise floor (default 3 for the largest replica); "
                      f"below 1 the run will "
                      f"practically never stop early. Use convergence_mode: blocks for the old "
                      f"absolute tolerance", stacklevel=2)
    bins = re_cfg.get("convergence_bins")
    return {"convergence_mode": mode, "convergence_tol": tol,
            "convergence_bins": None if bins is None else int(bins),
            "convergence_min_neff": _as_float(re_cfg.get("convergence_min_neff", NOISE_MIN_NEFF),
                                              "replica_exchange.convergence_min_neff"),
            "convergence_consecutive": int(_as_float(
                re_cfg.get("convergence_consecutive", NOISE_CONSECUTIVE),
                "replica_exchange.convergence_consecutive")),
            "convergence_min_round_trips": int(_as_float(
                re_cfg.get("convergence_min_round_trips", NOISE_MIN_ROUND_TRIPS),
                "replica_exchange.convergence_min_round_trips")),
            "convergence_median_tol": _as_float(re_cfg.get("convergence_median_tol", 1.3),
                                                "replica_exchange.convergence_median_tol"),
            "convergence_discard": _as_float(re_cfg.get("convergence_discard", 0.2),
                                             "replica_exchange.convergence_discard")}


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
                             observables=_observables_from_config(config, setup["geometry"]),
                             sgc_dmu=setup["sgc"]["dmu"] if setup["sgc"] else None,
                             sgc_attempts=setup["sgc"]["attempts_per_step"] if setup["sgc"] else 1,
                             sgc_log_ratio=setup["sgc"]["log_ratio"] if setup["sgc"] else None,
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
                          **_convergence_settings(re_cfg))
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
                    adapt_ladder=_ladder_adapter(setup, sim), **common, **kwargs)
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
    if feed is None or hasattr(feed, "gases"):
        D0 = e_atom = None
    else:
        D0 = {s: getattr(r, "D0_eV", None) for s, r in zip(feed.species, feed.reservoirs)}
        e_atom = {s: getattr(r, "e_atom_eV", None) for s, r in zip(feed.species, feed.reservoirs)}
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


def _ladder_adapter(setup: Dict[str, Any], sim) -> Any:
    """
    The `LadderAdapter` of a run with ``replica_exchange.adapt_ladder``.

    Parameters
    ----------
    setup : dict
        From `setup_from_config`.
    sim : ReplicaExchangeGCMC
        The simulation (its Lambda^3(T)).

    Returns
    -------
    regc.ladder.LadderAdapter or None
        None without ``adapt_ladder``.
    """
    from .ladder import LadderAdapter
    from .units import kelvin_to_eV
    settings = setup["metadata"]["sampling"].get("adapt_ladder")
    if not settings:
        return None
    feed, unit = setup["feed"], setup["pressure_unit"]

    def mu_of(T, y):
        return feed.mu_at_T_eV(T, 10.0 ** y, unit=unit)
    return LadderAdapter(mu_of, sim.lambda_cubed_of_T,
                         [kelvin_to_eV(T) for T in setup["temperatures_K"]],
                         np.log10(np.asarray(setup["pressures"], dtype=float)),
                         kelvin_per_unit=1.0 / kelvin_to_eV(1.0), pressure_unit=unit, **settings)


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
            f.attrs["reservoirs:" + name] = json.dumps(_feed_record(feed))
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
            feed = _feed_from_meta(meta, json.loads(stored[name]))
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
                    "masses_amu", "geometry", "sgc"):
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
        ``most_probable: true``, ``'most_probable_N'``. Each of
        ``phase_diagram.extra_observables`` (e.g. ``[N_Pd]``) adds
        ``'theta_<name>'``, and for an exchangeable slab species
        (``structure.exchangeable_top_layers``) ``'removed_<species>'``: the
        atoms gone to the reservoir, (N_start - <N>) per normalisation unit.

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
    feed = _feed_from_meta(meta)

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
    extra = pd_cfg.get("extra_observables") or []
    extra = [extra] if isinstance(extra, str) else [str(x) for x in extra]
    start = meta.get("initial_mobile_counts") or {}
    for obs in extra:
        th = analysis.phase_diagram_Tp(T_grid_eV, p_grid, area=norm, observable=obs)
        result[f"theta_{obs}"] = raw[f"theta_{obs}"] = th
        plot_map(T_grid_K, p_grid, th, os.path.join(out_dir, f"{root}_{obs}{ext}"), valid=valid,
                 colorbar_label=f"<{obs}>: {_COVERAGE_LABEL[per]}", xlabel=xlabel,
                 ylabel=ylabel, title=f"Reweighted <{obs}> (MBAR)", invalid_label=invalid)
        sp = obs[2:] if obs.startswith("N_") else None
        if sp in start:      # an exchangeable slab species: how much went to its reservoir
            removed = start[sp] / norm - th
            result[f"removed_{sp}"] = raw[f"removed_{sp}"] = removed
            plot_map(T_grid_K, p_grid, removed, os.path.join(out_dir, f"{root}_{sp}_removed{ext}"),
                     valid=valid, cmap="viridis",
                     colorbar_label=f"{sp} removed to the reservoir: {_COVERAGE_LABEL[per]}",
                     xlabel=xlabel, ylabel=ylabel,
                     title=f"{sp} removed from the surface (start: {start[sp]} atoms)",
                     invalid_label=invalid)
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
        if meta.get("pressure_grid") is not None:   # an adapted ladder
            pg = np.asarray(meta["pressure_grid"], dtype=float)
            # per-row pressures: no single sampled pressure list to flag ridges against
            ladder_p = pg[0].tolist() if np.allclose(pg, pg[:1], rtol=1e-12) else None
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
        elif kind == "SubsurfaceCounts":
            h = analysis.observable_map(name, T_grid_eV, p_grid, pressure=True)
            for j, sp in enumerate(m["species"]):
                sub_map = h[:, :, j] / norm
                result[f"subsurface_{sp}"] = raw[f"subsurface_{sp}"] = sub_map
                plot_map(T_grid_K, p_grid, sub_map, os.path.join(out_dir, f"{root}_subsurface_{sp}{ext}"),
                         valid=valid, cmap="viridis",
                         colorbar_label=f"subsurface {sp} per {per.replace('_', ' ')}",
                         xlabel=xlabel, ylabel=ylabel,
                         title=f"Subsurface {sp} ({m['z_min']:.2f} <= z < {m['z_cut']:.2f} A)",
                         invalid_label=invalid)
        elif kind in COMPOSITION_TYPES + ("SGCCounts",):
            # fractions as ratios of reweighted counts: <N_k> / <N>
            sp_list = list(m["species"])
            h = analysis.observable_map(name, T_grid_eV, p_grid, pressure=True)
            if kind in COMPOSITION_TYPES:
                h = h.reshape(h.shape[:2] + (composition_layers(m), len(sp_list)))
                where = "top layer" if kind == "LayerComposition" else "surface"
            else:
                h = h[:, :, None, :]
                where = "swappable substrate"
            total = h.sum(axis=-1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(total > 0, h / total, np.nan)   # (n_T, n_p, n_layers, n_species)
            key = "layer_fraction" if kind in COMPOSITION_TYPES else "sgc_fraction"
            raw[f"{key}_{name}"] = frac
            raw[f"{key}_{name}_species"] = np.array(sp_list)
            result[f"{key}_{name}"] = frac
            for j, sp in enumerate(sp_list[1:] if len(sp_list) > 1 else sp_list):
                j = sp_list.index(sp)
                top = frac[:, :, -1, j]
                plot_map(T_grid_K, p_grid, top,
                         os.path.join(out_dir, f"{root}_{name}_{sp}{ext}"),
                         valid=valid & np.isfinite(top), cmap="coolwarm",
                         vmin=float(np.nanmin(np.where(valid, top, np.nan))),
                         vmax=float(np.nanmax(np.where(valid, top, np.nan))),
                         colorbar_label=f"{sp} fraction, {where}", xlabel=xlabel, ylabel=ylabel,
                         title=f"{sp} fraction in the {where} (MBAR)", invalid_label=invalid)
                if verbose:
                    print(f"{sp} fraction in the {where}: {np.nanmin(top):.3f} to "
                          f"{np.nanmax(top):.3f}")
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
