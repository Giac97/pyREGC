"""
LAMMPS as the energy/force engine *and* the MD engine.

`LammpsCalculator` runs any LAMMPS pair style -- EAM (`eam/fs`,
`eam/alloy`), GRACE-FS (`grace/fs`), ACE (`pace`), ... -- given only its
``pair_style`` / ``pair_coeff`` lines and the species order those lines use.
It plugs into `ParticleSystem` like every other `Calculator`:

* energies and forces come from ``run 0`` on a persistent LAMMPS instance
  (all atom categories -- frozen, displaceable substrate, mobile -- are
  loaded), so GCMC insertion/removal and MC/HMC moves work unchanged
  through the generic clone-and-recompute path;
* `run_md` runs a whole thermostatted MD segment natively in LAMMPS
  (``fix nve`` + ``fix temp/csvr`` -- the Bussi thermostat -- and
  ``fix wall/reflect`` at the system's hard z-walls, frozen atoms excluded),
  which `Replica.diffuse('md')` uses automatically instead of ASE's MD.

Units are LAMMPS ``metal`` (eV, Angstrom, ps, amu): the same as this
package's real-units convention. Requires the LAMMPS Python module and a
LAMMPS shared library built with the pair style you use (e.g. ML-PACE for
``grace/fs``, a grace-runtime build for ``grace/runtime``). Which library is
loaded is chosen with ``lammps_name`` (``"grace"`` -> ``liblammps_grace.so``)
or the environment variable ``REGC_LAMMPS_NAME``; see `open_lammps`.

Pickling (process-pool workers) drops the LAMMPS handle; it is rebuilt
lazily in the receiving process.
"""

from __future__ import annotations

import ctypes
import os
import warnings
from typing import Optional, Sequence

import numpy as np

from .calculators import Calculator

KB_EV_PER_K = 8.617333262e-5
# LAMMPS metal units: m v^2 in amu (A/ps)^2 -> eV
MVV2E_METAL = 1.66053906660e-27 * 1.0e4 / 1.602176634e-19

# LAMMPS instances shared by all LammpsCalculators of this process with the same
# pair commands, species, masses and box: a worker process then loads the
# potential (an EAM table, a GRACE model, ...) once, not every time a replica is
# shipped to it. Sharing is safe: every evaluation reloads all atoms, and a
# process evaluates one configuration at a time.
_INSTANCES: dict = {}

# Environment variable naming the LAMMPS shared library to load when a calculator
# does not name one itself: REGC_LAMMPS_NAME=grace loads liblammps_grace.so.
LAMMPS_NAME_ENV = "REGC_LAMMPS_NAME"

# The LAMMPS library this process has loaded (None: none yet). The LAMMPS Python
# module loads libraries with RTLD_GLOBAL, so a second, different build loaded into
# the same process binds part of its calls to the first one's symbols: the two
# must never be mixed in one process.
_LOADED_LIBRARY: Optional[str] = None

# What each LAMMPS instance currently holds (keyed by id(lmp)): atom count, types, box,
# and whether it has been set up since its last change of fixes (MD). With the same atoms
# only the positions are sent and the energy comes from `run 1 pre no` (one step, no
# integrator), which skips LAMMPS's setup -- most of the cost of an energy call -- while
# still refreshing ghost atoms and rebuilding neighbour lists when atoms moved too far.
_LOADED: dict = {}
_ENERGY_CACHE_SIZE = 2   # the configurations before and after the last trial move
# REGC_LAMMPS_FAST=0 turns the fast path, the energy cache and the MD end-energy hand-off
# off (every evaluation is a full reload + setup): for checks and benchmarks.
FAST_ENV = "REGC_LAMMPS_FAST"


def _fast() -> bool:
    """
    Whether the fast evaluation path is enabled.

    Returns
    -------
    bool
        False only if the environment variable ``REGC_LAMMPS_FAST`` is ``'0'``.
    """
    return os.environ.get(FAST_ENV, "1") != "0"



def _preload_h5py() -> None:
    """
    Import h5py (if installed) before the LAMMPS library is loaded.

    A conda LAMMPS can link another HDF5 than h5py's; whichever loads
    first wins, and h5py imported after it fails ("Not a datatype"). Loading
    h5py first keeps per-sample records (`regc.records`) usable in LAMMPS
    runs; LAMMPS itself works with either.
    """
    try:
        import h5py  # noqa: F401
    except Exception:  # not installed (or broken): nothing to protect
        pass


def resolve_lammps_name(lammps_name: Optional[str] = None) -> str:
    """
    The LAMMPS library ("machine") name to load.

    Parameters
    ----------
    lammps_name : str, optional
        An explicit name: ``"grace"`` loads ``liblammps_grace.so``, ``""``
        the default ``liblammps.so``. Default None: the value of the
        environment variable ``REGC_LAMMPS_NAME``, else ``""``.

    Returns
    -------
    str
        The name, as LAMMPS's ``lammps(name=...)`` takes it.
    """
    if lammps_name is None:
        lammps_name = os.environ.get(LAMMPS_NAME_ENV, "")
    return str(lammps_name).strip()


def open_lammps(lammps_name: Optional[str] = None,
                cmdargs: Sequence[str] = ("-log", "none", "-screen", "none", "-nocite")):
    """
    Start a LAMMPS instance from a chosen shared library.

    The LAMMPS Python module loads ``liblammps_<name>.so`` (``liblammps.so``
    for an empty name) from its own directory or, failing that, from the
    dynamic loader's search path (the environment's ``lib/``,
    ``LD_LIBRARY_PATH``). Several LAMMPS builds can so live in one
    environment, e.g. conda-forge's as the default and a grace-runtime build
    as ``liblammps_grace.so`` (``hpc/build_lammps.sh``).

    A process can use only one LAMMPS library: asking for a second, different
    one raises instead of loading it (two builds in one process share
    symbols and corrupt each other's memory). Compare builds in separate
    processes.

    The module refuses a library of another LAMMPS version than its own.
    The C library interface regc uses (commands, atom creation and
    gathering, thermo output) is stable across versions, so when a named
    library has a different version this function warns and turns the
    module's version check off for the rest of the process.

    Parameters
    ----------
    lammps_name : str, optional
        Library name (see `resolve_lammps_name`). Default None
        (``REGC_LAMMPS_NAME``, else the default library).
    cmdargs : sequence of str, optional
        LAMMPS command-line arguments. Default: no log file, no screen
        output, no citation reminder.

    Returns
    -------
    lammps.lammps
        The instance.

    Raises
    ------
    OSError
        If the library cannot be found or loaded.
    RuntimeError
        If this process already uses another LAMMPS library.
    """
    global _LOADED_LIBRARY
    _preload_h5py()
    import lammps as lammps_module
    name = resolve_lammps_name(lammps_name)
    if _LOADED_LIBRARY is not None and name != _LOADED_LIBRARY:
        raise RuntimeError(
            f"this process already uses {_library_file(_LOADED_LIBRARY)}; loading "
            f"{_library_file(name)} as well would mix two LAMMPS builds in one process, which "
            "corrupts memory. Use one lammps_name per process (worker processes inherit "
            "REGC_LAMMPS_NAME); compare builds in separate processes.")
    lmp = _start_lammps(lammps_module, name, list(cmdargs))
    _LOADED_LIBRARY = name
    return lmp


def _start_lammps(lammps_module, name: str, cmdargs: list):
    """
    `lammps.lammps(name, cmdargs)`, tolerating a library/module version mismatch.

    Parameters
    ----------
    lammps_module : module
        The imported ``lammps`` package.
    name : str
        Library name.
    cmdargs : list of str
        Command-line arguments.

    Returns
    -------
    lammps.lammps
        The instance.
    """
    try:
        return lammps_module.lammps(name=name, cmdargs=cmdargs)
    except AttributeError as err:
        if "shared library is version" not in str(err):
            raise
        warnings.warn(f"{err} (library {_library_file(name)}); regc runs it anyway and skips "
                      "the LAMMPS Python module's version check from now on",
                      RuntimeWarning, stacklevel=3)
        lammps_module.__version__ = 0   # the module's own 'in-place build' value: no check
        return lammps_module.lammps(name=name, cmdargs=cmdargs)


def _library_file(name: str) -> str:
    """
    File name of a LAMMPS shared library.

    Parameters
    ----------
    name : str
        Library name, possibly empty.

    Returns
    -------
    str
        ``liblammps_<name>.so`` or ``liblammps.so``.
    """
    return f"liblammps_{name}.so" if name else "liblammps.so"


def _f(x) -> str:
    """
    Format a number as LAMMPS input text.

    Parameters
    ----------
    x : float
        Any real number (NumPy scalars included).

    Returns
    -------
    str
        The plain Python float repr (full precision; never ``np.float64(...)``).
    """
    return repr(float(x))


def eam_pair_commands(potential_file: str, species: Sequence[str]) -> list:
    """
    LAMMPS pair commands for an EAM file, inferring the style from its name.

    Parameters
    ----------
    potential_file : str
        A ``*.eam.fs`` (Finnis-Sinclair) or ``*.eam.alloy`` (setfl) file.
    species : sequence of str
        Element names in LAMMPS type order, as they appear in the file.

    Returns
    -------
    list of str
        ``['pair_style eam/fs', 'pair_coeff * * /abs/path Fe H']`` or the
        ``eam/alloy`` equivalent.

    Raises
    ------
    ValueError
        If the extension is neither ``.fs`` nor ``.alloy``.
    """
    import os
    ext = os.path.splitext(potential_file)[1]
    style = {".fs": "eam/fs", ".alloy": "eam/alloy"}.get(ext)
    if style is None:
        raise ValueError(f"cannot infer a multi-element EAM style from {potential_file!r} "
                         f"(expected *.eam.fs or *.eam.alloy)")
    return [f"pair_style {style}",
            f"pair_coeff * * {os.path.abspath(potential_file)} {' '.join(species)}"]


def lammps_ase_calculator(pair_commands: Sequence[str], species: Sequence[str],
                          lammps_name: Optional[str] = None):
    """
    An ASE calculator running the same LAMMPS potential.

    For setup work done through ASE (lattice-constant scans, the
    reservoir's molecule, ...), so it uses exactly the potential the
    simulation's `LammpsCalculator` uses.

    Parameters
    ----------
    pair_commands : sequence of str
        LAMMPS ``pair_style``/``pair_coeff`` lines.
    species : sequence of str
        Species in LAMMPS type order.
    lammps_name : str, optional
        LAMMPS library to load (``"grace"`` -> ``liblammps_grace.so``).
        Default None: ``REGC_LAMMPS_NAME``, else the default library.

    Returns
    -------
    ase.calculators.lammpslib.LAMMPSlib
        The calculator.
    """
    from ase.calculators.lammpslib import LAMMPSlib
    name = resolve_lammps_name(lammps_name)
    open_lammps(name).close()   # fails early if missing; relaxes a version mismatch (see open_lammps)
    return LAMMPSlib(lmpcmds=list(pair_commands),
                     atom_types={s: i + 1 for i, s in enumerate(species)},
                     lammps_name=name, log_file=None, keep_alive=True)


class LammpsCalculator(Calculator):
    """
    Energies, forces and native MD from LAMMPS, for any pair style.

    Parameters
    ----------
    pair_commands : sequence of str
        The LAMMPS commands defining the interaction, e.g.
        ``["pair_style eam/fs", "pair_coeff * * Fe-H.eam.fs Fe H"]`` or
        ``["pair_style grace/fs", "pair_coeff * * model.yaml Fe H"]`` or, with
        a grace-runtime build (``lammps_name="grace"``),
        ``["pair_style grace/runtime", "pair_coeff * * kokkos.npz Fe H"]``.
    species : sequence of str
        Species in LAMMPS atom-type order (type 1 = species[0], ...); must
        match the element order in `pair_commands`.
    masses : dict, optional
        Mass (amu) per species; defaults to ASE's tables.
    min_distance : float, optional
        Hard core: the energy is +inf if any mobile atom is closer than
        this to any other atom (an exact modification of the potential).
        Guards GCMC insertions against unphysical, off-training-set
        configurations of machine-learned potentials. Choose it below any
        physically reachable separation. Default None (off).
    z_padding : float, optional
        Extra space (A) beyond the outermost atoms/walls along every
        non-periodic direction of the LAMMPS box (z for a slab, all three
        for a cluster). Default 5.0.
    lammps_name : str, optional
        Which LAMMPS shared library to load: ``"grace"`` loads
        ``liblammps_grace.so``, ``""`` the default ``liblammps.so``.
        Default None: the environment variable ``REGC_LAMMPS_NAME`` of the
        process that evaluates (worker processes inherit it), else the
        default library. See `open_lammps`.
    """

    supports_incremental = False
    supports_forces = True

    def __init__(self, pair_commands: Sequence[str], species: Sequence[str],
                 masses: Optional[dict] = None, min_distance: Optional[float] = None,
                 z_padding: float = 5.0, lammps_name: Optional[str] = None):
        """
        Store the potential definition; LAMMPS starts lazily on first use.

        Parameters
        ----------
        pair_commands : sequence of str
            LAMMPS ``pair_style``/``pair_coeff`` lines.
        species : sequence of str
            Species in LAMMPS atom-type order.
        masses : dict, optional
            Mass (amu) per species. Default None (ASE tables).
        min_distance : float, optional
            Hard-core distance (A). Default None (off).
        z_padding : float, optional
            Extra box height (A) beyond the atoms/walls. Default 5.0.
        lammps_name : str, optional
            LAMMPS library name. Default None (``REGC_LAMMPS_NAME``, else
            the default library).
        """
        self.pair_commands = list(pair_commands)
        self.species = list(species)
        self.type_of = {s: i + 1 for i, s in enumerate(self.species)}
        self.masses = dict(masses) if masses else {}
        self.min_distance = min_distance
        self.z_padding = z_padding
        self.lammps_name = lammps_name
        self._lmp = None
        self._box_key = None

    # -- LAMMPS instance ---------------------------------------------------------
    def __getstate__(self) -> dict:
        """
        Picklable state, without the LAMMPS handle.

        Returns
        -------
        dict
            The instance dict with the (unpicklable) LAMMPS handle cleared;
            it is rebuilt -- or taken from the process's cache -- on first use.
        """
        state = self.__dict__.copy()
        state['_lmp'] = None  # not picklable; rebuilt on first use
        state['_box_key'] = None
        state.pop('_run0', None)
        return state

    def _mass(self, s: str) -> float:
        """
        Mass of a species.

        Parameters
        ----------
        s : str
            Species label.

        Returns
        -------
        float
            Mass in amu: from `masses` if given, else ASE's tables.
        """
        if s in self.masses:
            return float(self.masses[s])
        from .units import default_mass_amu
        return default_mass_amu(s)

    @staticmethod
    def _cell_matrix(system) -> np.ndarray:
        """
        A system's cell as a 3x3 matrix.

        Parameters
        ----------
        system : ParticleSystem
            The system (``cell`` either (Lx, Ly, Lz) or a 3x3 matrix).

        Returns
        -------
        numpy.ndarray
            The 3x3 cell matrix (rows a1, a2, a3).
        """
        cell = system.cell
        if np.ndim(cell) == 1:
            return np.diag(np.asarray(cell, dtype=float))
        return np.asarray(cell, dtype=float)

    @staticmethod
    def _pbc(system) -> tuple:
        """
        A system's periodicity.

        Parameters
        ----------
        system : ParticleSystem
            The system.

        Returns
        -------
        tuple of bool
            ``system.pbc``; ``(True, True, False)`` (a slab) if it has none.
        """
        return tuple(bool(p) for p in getattr(system, 'pbc', (True, True, False)))

    @staticmethod
    def _walls(system) -> list:
        """
        A system's hard walls, per axis.

        Parameters
        ----------
        system : ParticleSystem
            The system.

        Returns
        -------
        list
            Three entries, ``(lo, hi)`` or None (`ParticleSystem.md_wall_bounds`).
        """
        if hasattr(system, 'md_wall_bounds'):
            return list(system.md_wall_bounds())
        z = system.md_z_bounds() if hasattr(system, 'md_z_bounds') else None
        return [None, None, z]

    def _setup(self, system, lo: np.ndarray, hi: np.ndarray) -> None:
        """
        Attach a LAMMPS instance with the right box (cached per process).

        Periodic axes span the cell (``0`` to the cell's length); the
        non-periodic ones span ``lo`` to ``hi`` (atoms and walls plus
        `z_padding`).

        Parameters
        ----------
        system : ParticleSystem
            Supplies the cell and periodicity.
        lo, hi : numpy.ndarray
            Box extent (3,) along the non-periodic axes; entries of
            periodic axes are ignored.

        Raises
        ------
        ValueError
            If the cell is not in LAMMPS's restricted-triclinic form, or is
            tilted along a non-periodic axis.
        """
        H = self._cell_matrix(system)
        if abs(H[0, 1]) > 1e-12 or abs(H[0, 2]) > 1e-12 or abs(H[1, 2]) > 1e-12:
            raise ValueError("LammpsCalculator needs a1 along x and a2 in the xy plane "
                             "(LAMMPS's restricted-triclinic convention)")
        pbc = self._pbc(system)
        xy, xz, yz = H[1, 0], H[2, 0], H[2, 1]
        for tilt, axes, label in ((xy, (0, 1), "xy"), (xz, (0, 2), "xz"), (yz, (1, 2), "yz")):
            if abs(tilt) > 1e-12 and not all(pbc[k] for k in axes):
                raise ValueError(f"LammpsCalculator: the cell is tilted in {label}, along a "
                                 "non-periodic axis")
        box_lo = np.array([0.0 if pbc[k] else float(lo[k]) for k in range(3)])
        box_hi = np.array([H[k, k] if pbc[k] else float(hi[k]) for k in range(3)])
        name = resolve_lammps_name(getattr(self, 'lammps_name', None))  # absent in old pickles
        key = (name, tuple(self.pair_commands), tuple(self.species),
               tuple(sorted((s, self._mass(s)) for s in self.species)),
               tuple(np.round(H.ravel(), 10)), tuple(np.round(box_lo, 10)),
               tuple(np.round(box_hi, 10)), pbc)
        if key in _INSTANCES:
            self._lmp, self._H, self._Hinv, self._lo, self._hi = _INSTANCES[key]
            return
        lmp = open_lammps(name)
        bnd = " ".join("p" if p else "f" for p in pbc)
        bounds = " ".join(f"{_f(a)} {_f(b)}" for a, b in zip(box_lo, box_hi))
        cmds = ["units metal", "atom_style atomic", "atom_modify map array sort 0 0.0",
                f"boundary {bnd}"]
        if max(abs(xy), abs(xz), abs(yz)) > 1e-12:
            cmds.append(f"region box prism {bounds} {_f(xy)} {_f(xz)} {_f(yz)} units box")
        else:
            cmds.append(f"region box block {bounds} units box")
        cmds.append(f"create_box {len(self.species)} box")
        cmds += [f"mass {i + 1} {_f(self._mass(s))}" for i, s in enumerate(self.species)]
        cmds += self.pair_commands
        cmds += ["neighbor 1.0 bin", "neigh_modify every 1 delay 0 check yes",
                 "thermo_style custom step pe", "thermo 0"]
        lmp.commands_list(cmds)
        _LOADED.pop(id(lmp), None)
        self._lmp = lmp
        self._H = H
        self._Hinv = np.linalg.inv(H)
        self._lo, self._hi = box_lo, box_hi
        _INSTANCES[key] = (lmp, H, self._Hinv, self._lo, self._hi)

    def _load(self, system) -> int:
        """
        Replace LAMMPS's atoms by `system`'s full configuration.

        Positions are wrapped into the cell along periodic axes; along the
        others the box grows (a new instance) if an atom or wall lies
        outside it. Atom ids follow `full_configuration`'s order (frozen
        1..n_frozen, then substrate, then mobile). When the instance already
        holds the same atoms (count, types, box), only the positions are
        sent; the run command to use next is left in ``self._run0``:
        ``run 1 pre no`` (one step without an integrator: no setup, but ghost
        atoms refreshed and neighbour lists rebuilt when needed), or a full
        ``run 0`` after any change of atoms or fixes.

        Parameters
        ----------
        system : ParticleSystem
            The configuration to load.

        Returns
        -------
        int
            The number of frozen atoms.

        Raises
        ------
        RuntimeError
            If LAMMPS dropped atoms (one outside the box).
        """
        pos, sp = system.full_configuration()
        pbc = self._pbc(system)
        walls = self._walls(system)
        lo, hi = np.zeros(3), np.zeros(3)
        for k in range(3):
            if pbc[k]:
                continue
            vals = [0.0] if not len(pos) else [pos[:, k].min(), pos[:, k].max()]
            if walls[k] is not None:
                vals += list(walls[k])
            lo[k], hi[k] = min(vals), max(vals)
        key = (tuple(np.round(self._cell_matrix(system).ravel(), 10)), pbc)
        if (self._lmp is None or self._box_key != key
                or any(not pbc[k] and (lo[k] < self._lo[k] or hi[k] > self._hi[k])
                       for k in range(3))):
            self._setup(system, lo - self.z_padding, hi + self.z_padding)
            self._box_key = key
        lmp = self._lmp
        n = len(pos)
        x = pos
        if n:
            frac = pos @ self._Hinv
            for k in range(3):
                if pbc[k]:
                    frac[:, k] %= 1.0
            x = frac @ self._H
        types = tuple(self.type_of[str(s)] for s in sp)
        state = _LOADED.get(id(lmp))
        if (_fast() and state is not None and state["n"] == n and n > 0
                and state["types"] == types and state["box"] == key):
            # same atoms: only move them
            lmp.scatter_atoms("x", 1, 3, (ctypes.c_double * (3 * n))(*x.ravel()))
            if state["valid"]:
                # one step without integrator: LAMMPS refreshes the ghost atoms, or
                # rebuilds the neighbour lists itself if an atom moved more than half
                # the skin (neigh_modify check yes); `run 0 pre no` would skip the
                # ghost update and give wrong energies
                self._run0 = "run 1 pre no post no"
            else:
                self._run0 = "run 0 post no"
                state["valid"] = True
            return system.frozen_positions.shape[0]
        lmp.command("delete_atoms group all compress yes")
        if n:
            lmp.create_atoms(n, list(range(1, n + 1)), list(types), x.ravel().tolist())
            if lmp.get_natoms() != n:
                _LOADED.pop(id(lmp), None)
                raise RuntimeError(f"LAMMPS kept {lmp.get_natoms()} of {n} atoms "
                                   "(an atom outside the box?)")
        _LOADED[id(lmp)] = {"n": n, "types": types, "box": key, "valid": True}
        self._run0 = "run 0 post no"
        return system.frozen_positions.shape[0]

    def _gather(self, name: str, n: int) -> np.ndarray:
        """
        Gather a per-atom 3-vector property, ordered by atom id.

        Parameters
        ----------
        name : str
            LAMMPS property (``'x'`` or ``'f'``).
        n : int
            Number of atoms.

        Returns
        -------
        numpy.ndarray
            Shape (n, 3).
        """
        data = self._lmp.gather_atoms(name, 1, 3)
        return np.ctypeslib.as_array(data, shape=(n * 3,)).reshape(n, 3).copy()

    # -- Calculator interface ------------------------------------------------------
    def _violates_hard_core(self, system) -> bool:
        """
        Whether a mobile particle is closer than `min_distance` to any atom.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        bool
            False if the hard core is off.
        """
        if self.min_distance is None or system.n == 0:
            return False
        others = np.vstack([system.frozen_positions, system.substrate_positions, system.positions])
        n_other = others.shape[0] - system.n
        for i in range(system.n):
            d = system._min_image(others - system.positions[i])
            r2 = np.einsum('ij,ij->i', d, d)
            r2[n_other + i] = np.inf
            if r2.min() < self.min_distance ** 2:
                return True
        return False

    def total_energy(self, system) -> float:
        """
        Potential energy of the full configuration (``run 0``).

        The energies of the last two configurations are cached (exact
        match of positions, species and cell), so the energy before a
        trial move -- the configuration evaluated at the previous move --
        costs nothing; see also `_load` for the cheap re-evaluation.

        Parameters
        ----------
        system : ParticleSystem
            The configuration (frozen + substrate + mobile atoms).

        Returns
        -------
        float
            The LAMMPS potential energy in eV (0 for an empty system;
            +inf if the optional hard core is violated).
        """
        pos, sp = system.full_configuration()
        if pos.shape[0] == 0:
            return 0.0
        if self._violates_hard_core(system):
            return np.inf
        key = (np.ascontiguousarray(pos, dtype=float).tobytes(), tuple(map(str, sp)),
               np.asarray(system.cell, dtype=float).tobytes())
        cache = self.__dict__.setdefault("_energy_cache", {})
        if key in cache and _fast():   # e.g. the energy before a trial move
            return cache[key]
        self._load(system)
        self._lmp.command(self._run0)
        energy = float(self._lmp.get_thermo("pe"))
        cache[key] = energy
        while len(cache) > _ENERGY_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        return energy

    def forces(self, system) -> np.ndarray:
        """
        Forces on substrate + mobile atoms (frozen rows dropped).

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Shape (n_substrate + n_mobile, 3), in eV/A.
        """
        n_tot = system.full_configuration()[0].shape[0]
        n_fr = system.frozen_positions.shape[0]
        if n_tot - n_fr == 0:
            return np.zeros((0, 3))
        self._load(system)
        self._lmp.command(self._run0)
        return self._gather("f", n_tot)[n_fr:]

    def energy_after_md(self, system) -> float:
        """
        Potential energy of the configuration `run_md` just produced.

        Parameters
        ----------
        system : ParticleSystem
            The system after its positions were updated from `run_md`.

        Returns
        -------
        float
            The LAMMPS potential energy of the segment's last step (no
            re-evaluation), +inf if the optional hard core is violated.
        """
        if not _fast():
            return self.total_energy(system)
        if self._violates_hard_core(system):
            return np.inf
        return self.last_md_energy

    # -- native MD -----------------------------------------------------------------
    def run_md(self, system, n_steps: int, temperature_eV: float, timestep_fs: float,
               taut_fs: float, seed: int, thermostat: str = "bussi",
               friction_per_fs: Optional[float] = None) -> np.ndarray:
        """
        One NVT MD segment inside LAMMPS.

        Fresh Maxwell-Boltzmann velocities at T for the moving atoms, drawn
        here (not with ``velocity create``, which rescales the draw to
        exactly T and fails for a single atom), so the start is an exact
        canonical momentum sample; ``fix nve`` + ``fix temp/csvr`` (Bussi;
        or ``langevin``), ``fix wall/reflect`` at the system's hard walls
        (`md_wall_bounds`: the z-walls of a slab, the six faces of a
        cluster box), frozen atoms excluded from integration. The Bussi
        thermostat's temperature counts 3N degrees of freedom (``extra/dof
        0``): frozen atoms and walls break momentum conservation, and the
        ASE path (`Replica._diffuse_md`) counts the same.

        Parameters
        ----------
        system : ParticleSystem
            The system to propagate (not modified here).
        n_steps : int
            Number of MD steps.
        temperature_eV : float
            k_B T in eV.
        timestep_fs, taut_fs : float
            Time step and Bussi thermostat time constant, in fs.
        seed : int
            Random seed (velocities and thermostat).
        thermostat : str, optional
            ``'bussi'`` (default) or ``'langevin'``.
        friction_per_fs : float, optional
            Langevin friction in 1/fs (damping time 1/friction), as in the
            ASE path. Default None: damping time `taut_fs`.

        Returns
        -------
        numpy.ndarray
            New positions of the substrate + mobile atoms, shape
            (n_substrate + n_mobile, 3), unwrapped along periodic axes.

        Raises
        ------
        ValueError
            If `thermostat` is unknown.
        """
        if thermostat not in ("bussi", "langevin"):
            raise ValueError(f"thermostat must be 'bussi' or 'langevin', got {thermostat!r}")
        n_fr = self._load(system)
        _, species = system.full_configuration()
        n_tot = len(species)
        lmp = self._lmp
        T_K = temperature_eV / KB_EV_PER_K
        seed = int(seed) % 900000000 + 1
        # Maxwell-Boltzmann velocities in metal units (A/ps): v = sqrt(kT / (m mvv2e))
        masses = np.array([self._mass(str(sp)) for sp in species[n_fr:]])
        sigma = np.sqrt(temperature_eV / (masses * MVV2E_METAL))
        v = np.zeros((n_tot, 3))
        v[n_fr:] = np.random.default_rng(seed).standard_normal((n_tot - n_fr, 3)) * sigma[:, None]
        lmp.scatter_atoms("v", 1, 3, (ctypes.c_double * (3 * n_tot))(*v.ravel()))
        cmds = []
        if n_fr:
            cmds += [f"group frozen id 1:{n_fr}", "group mobile subtract all frozen"]
        else:
            cmds += ["group mobile union all"]
        cmds += [f"timestep {_f(timestep_fs / 1000.0)}", "fix gcmc_nve mobile nve"]
        if thermostat == "bussi":
            cmds += ["compute gcmc_temp mobile temp", "compute_modify gcmc_temp extra/dof 0",
                     f"fix gcmc_thermo mobile temp/csvr {_f(T_K)} {_f(T_K)} "
                     f"{_f(taut_fs / 1000.0)} {seed + 1}",
                     "fix_modify gcmc_thermo temp gcmc_temp"]
        else:
            damp_fs = taut_fs if friction_per_fs is None else 1.0 / friction_per_fs
            cmds.append(f"fix gcmc_thermo mobile langevin {_f(T_K)} {_f(T_K)} "
                        f"{_f(damp_fs / 1000.0)} {seed + 1}")
        pbc = self._pbc(system)
        faces = [f"{ax}lo {_f(b[0])} {ax}hi {_f(b[1])}"
                 for ax, b, p in zip("xyz", self._walls(system), pbc) if b is not None and not p]
        if faces:
            cmds.append("fix gcmc_walls mobile wall/reflect " + " ".join(faces))
        lmp.commands_list(cmds)
        lmp.command(f"run {int(n_steps)} post no")
        x = self._gather("x", n_tot)
        # the potential energy of the end point (thermo is evaluated on the last step):
        # spares a full re-evaluation after the segment (see `energy_after_md`)
        self.last_md_energy = float(lmp.get_thermo("pe"))
        state = _LOADED.get(id(lmp))
        if state is not None:   # fixes were added and removed: the next run needs a setup
            state["valid"] = False
        lmp.commands_list(["unfix gcmc_nve", "unfix gcmc_thermo"]
                          + (["unfix gcmc_walls"] if faces else [])
                          + (["uncompute gcmc_temp"] if thermostat == "bussi" else [])
                          + ["group mobile delete"] + (["group frozen delete"] if n_fr else []))
        return x[n_fr:]
