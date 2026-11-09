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

import os
import warnings
from typing import Optional, Sequence

import numpy as np

from .calculators import Calculator

KB_EV_PER_K = 8.617333262e-5

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
        Extra space (A) below the lowest and above the highest atom/wall
        in the non-periodic z direction of the LAMMPS box. Default 5.0.
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

    def _setup(self, system, z_lo: float, z_hi: float) -> None:
        """
        Attach a LAMMPS instance with the right box (cached per process).

        Parameters
        ----------
        system : ParticleSystem
            Supplies the cell and periodicity.
        z_lo, z_hi : float
            Box extent along z.

        Raises
        ------
        ValueError
            If the cell is not in LAMMPS's restricted-triclinic form.
        """
        H = self._cell_matrix(system)
        if abs(H[0, 1]) > 1e-12 or abs(H[0, 2]) > 1e-12 or abs(H[1, 2]) > 1e-12:
            raise ValueError("LammpsCalculator needs a1 along x and a2 in the xy plane "
                             "(LAMMPS's restricted-triclinic convention)")
        Lx, xy, Ly = H[0, 0], H[1, 0], H[1, 1]
        name = resolve_lammps_name(getattr(self, 'lammps_name', None))  # absent in old pickles
        key = (name, tuple(self.pair_commands), tuple(self.species),
               tuple(sorted((s, self._mass(s)) for s in self.species)),
               tuple(np.round(H.ravel(), 10)), round(z_lo, 10), round(z_hi, 10),
               tuple(getattr(system, 'pbc', (True, True, False))))
        if key in _INSTANCES:
            self._lmp, self._H, self._Hinv, self._z_range = _INSTANCES[key]
            return
        lmp = open_lammps(name)
        pbc = tuple(getattr(system, 'pbc', (True, True, False)))
        bnd = " ".join("p" if p else "f" for p in pbc)
        tilt = abs(xy) > 1e-12
        cmds = ["units metal", "atom_style atomic", "atom_modify map array sort 0 0.0",
                f"boundary {bnd}"]
        if tilt:
            cmds.append(f"region box prism 0 {_f(Lx)} 0 {_f(Ly)} {_f(z_lo)} {_f(z_hi)} {_f(xy)} 0 0 units box")
        else:
            cmds.append(f"region box block 0 {_f(Lx)} 0 {_f(Ly)} {_f(z_lo)} {_f(z_hi)} units box")
        cmds.append(f"create_box {len(self.species)} box")
        cmds += [f"mass {i + 1} {_f(self._mass(s))}" for i, s in enumerate(self.species)]
        cmds += self.pair_commands
        cmds += ["neighbor 1.0 bin", "neigh_modify every 1 delay 0 check yes",
                 "thermo_style custom step pe", "thermo 0"]
        lmp.commands_list(cmds)
        self._lmp = lmp
        self._H = H
        self._Hinv = np.linalg.inv(H[:2, :2])
        self._z_range = (z_lo, z_hi)
        _INSTANCES[key] = (lmp, H, self._Hinv, self._z_range)

    def _load(self, system) -> int:
        """
        Replace LAMMPS's atoms by `system`'s full configuration.

        Positions are wrapped into the periodic cell; atom ids follow
        `full_configuration`'s order (frozen 1..n_frozen, then substrate,
        then mobile).

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
        walls = system.md_z_bounds() if hasattr(system, 'md_z_bounds') else None
        z_min = min(pos[:, 2].min() if len(pos) else 0.0, walls[0] if walls else 0.0)
        z_max = max(pos[:, 2].max() if len(pos) else 0.0, walls[1] if walls else 0.0)
        key = (tuple(np.round(self._cell_matrix(system).ravel(), 10)),)
        if (self._lmp is None or self._box_key != key
                or z_min < self._z_range[0] or z_max > self._z_range[1]):
            self._setup(system, z_min - self.z_padding, z_max + self.z_padding)
            self._box_key = key
        lmp = self._lmp
        lmp.command("delete_atoms group all compress yes")
        n = len(pos)
        if n:
            x = pos.copy()
            frac = x[:, :2] @ self._Hinv
            pbc = getattr(system, 'pbc', (True, True, False))
            for k in (0, 1):
                if pbc[k]:
                    frac[:, k] %= 1.0
            x[:, :2] = frac @ self._H[:2, :2]
            types = [self.type_of[str(s)] for s in sp]
            lmp.create_atoms(n, list(range(1, n + 1)), types, x.ravel().tolist())
            if lmp.get_natoms() != n:
                raise RuntimeError(f"LAMMPS kept {lmp.get_natoms()} of {n} atoms "
                                   "(an atom outside the box?)")
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
        if system.full_configuration()[0].shape[0] == 0:
            return 0.0
        if self._violates_hard_core(system):
            return np.inf
        self._load(system)
        self._lmp.command("run 0 post no")
        return float(self._lmp.get_thermo("pe"))

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
        self._lmp.command("run 0 post no")
        return self._gather("f", n_tot)[n_fr:]

    # -- native MD -----------------------------------------------------------------
    def run_md(self, system, n_steps: int, temperature_eV: float, timestep_fs: float,
               taut_fs: float, seed: int, thermostat: str = "bussi") -> np.ndarray:
        """
        One NVT MD segment inside LAMMPS.

        Fresh Maxwell-Boltzmann velocities at T for the moving atoms
        (exact: momenta and positions are independent in the canonical
        ensemble), ``fix nve`` + ``fix temp/csvr`` (Bussi; or ``langevin``),
        ``fix wall/reflect`` at the system's hard z-walls, frozen atoms
        excluded from integration.

        Parameters
        ----------
        system : ParticleSystem
            The system to propagate (not modified here).
        n_steps : int
            Number of MD steps.
        temperature_eV : float
            k_B T in eV.
        timestep_fs, taut_fs : float
            Time step and thermostat time constant, in fs.
        seed : int
            Random seed (velocities and thermostat).
        thermostat : str, optional
            ``'bussi'`` (default) or ``'langevin'``.

        Returns
        -------
        numpy.ndarray
            New positions of the substrate + mobile atoms, shape
            (n_substrate + n_mobile, 3), unwrapped in x, y.

        Raises
        ------
        ValueError
            If `thermostat` is unknown.
        """
        n_fr = self._load(system)
        n_tot = system.full_configuration()[0].shape[0]
        lmp = self._lmp
        T_K = temperature_eV / KB_EV_PER_K
        seed = int(seed) % 900000000 + 1
        cmds = []
        if n_fr:
            cmds += [f"group frozen id 1:{n_fr}", "group mobile subtract all frozen"]
        else:
            cmds += ["group mobile union all"]
        cmds += [f"velocity mobile create {_f(T_K)} {seed} dist gaussian mom no rot no",
                 f"timestep {_f(timestep_fs / 1000.0)}", "fix gcmc_nve mobile nve"]
        if thermostat == "bussi":
            cmds.append(f"fix gcmc_thermo mobile temp/csvr {_f(T_K)} {_f(T_K)} "
                        f"{_f(taut_fs / 1000.0)} {seed + 1}")
        elif thermostat == "langevin":
            cmds.append(f"fix gcmc_thermo mobile langevin {_f(T_K)} {_f(T_K)} "
                        f"{_f(taut_fs / 1000.0)} {seed + 1}")
        else:
            raise ValueError(f"thermostat must be 'bussi' or 'langevin', got {thermostat!r}")
        walls = system.md_z_bounds() if hasattr(system, 'md_z_bounds') else None
        if walls is not None:
            cmds.append(f"fix gcmc_walls mobile wall/reflect zlo {_f(walls[0])} zhi {_f(walls[1])}")
        lmp.commands_list(cmds)
        lmp.command(f"run {int(n_steps)} post no")
        x = self._gather("x", n_tot)
        lmp.commands_list(["unfix gcmc_nve", "unfix gcmc_thermo"]
                          + (["unfix gcmc_walls"] if walls is not None else [])
                          + ["group mobile delete"] + (["group frozen delete"] if n_fr else []))
        return x[n_fr:]
