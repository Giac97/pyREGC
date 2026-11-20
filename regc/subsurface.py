"""
Grand-canonical insertion and removal in subsurface sites with nonequilibrium
candidate Monte Carlo (NCMC).

An atom inserted at a random point inside a metal is almost never accepted:
the lattice has no room for it until it relaxes. Following the algorithm in
docs/REGC_algorithm.md Sec. 11, a slab is split at a fixed height z_c:

* **gas region** (z >= z_c): the usual instantaneous insertions and removals
  (atomic and molecular). `ParticleSystem.set_subsurface` raises the lower
  bound of the exchange zone to z_c, so the existing moves act there only;
* **subsurface region** (z_min <= z < z_c): one channel per species. An
  insertion picks an interstitial site s_k -- a vertex of the Voronoi
  tessellation of the metal atoms -- uniformly among the K sites in the
  region, places the atom at y = s_k + eta (eta uniform in a ball of radius
  delta), and switches it on with NCMC: lambda goes 0 -> 1 in T steps on
  U_lambda = (1 - lambda) U_N + lambda U_{N+1}, the configuration (atom,
  neighbours, displaceable substrate) relaxing between steps under a kernel
  in detailed balance at fixed lambda (HMC or MC sweeps). The protocol work
  W replaces the energy change in the acceptance

      insert:  min(1, z_s e^{-beta W} / (N_sub' q(y; x)))
      remove:  min(1, N_sub q(y; x') e^{-beta W} / z_s)

  with z_s = e^{beta mu_s} / Lambda_s^3, N_sub the number of atoms of the
  species in the region (after the insertion, before the removal), and
  q(y; x) = (1/K) sum_{k: |y - s_k| < delta} 1 / V_delta the proposal
  density, which depends on the metal atoms of x (the sites move with
  them). A removal switches a random subsurface atom off (lambda 1 -> 0, the
  reversed schedule) and then deletes it. Proposals or end points outside
  the region are rejected, which keeps the moves reversible.

Uniform site choice (instead of weighting sites by their Voronoi volume)
keeps q exactly computable; the site weights only affect efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

#: Sites closer than this (A) are merged (degenerate Voronoi vertices of a lattice).
SITE_MERGE_TOL = 0.25


@dataclass
class Subsurface:
    """
    Settings of the subsurface NCMC channels of a slab.

    Parameters
    ----------
    z_cut : float
        z_c, the fixed height separating the subsurface (z < z_c) from the
        gas region (z >= z_c), in the slab's coordinates (A).
    z_min : float
        Lower bound of the subsurface region (A).
    species : sequence of str
        Exchanged species that get a subsurface channel.
    delta : float, optional
        Radius of the insertion ball around a site (A). Default 0.3.
    ncmc_steps : int, optional
        Number T of lambda steps of the switching protocol. Default 10.
    kernel : str, optional
        Propagation at fixed lambda between steps: ``'hmc'`` (one HMC
        trajectory of all displaceable atoms, full momentum refresh) or
        ``'mc'`` (single-atom Metropolis displacements). Default ``'hmc'``.
    kernel_steps : int, optional
        HMC trajectories, or MC displacement attempts per displaceable
        atom, per lambda step. Default 1.
    leapfrog_steps : int, optional
        Leapfrog steps per HMC trajectory. Default 5.
    dt : float, optional
        HMC time step in internal units (A sqrt(amu/eV); 1 fs = 0.0982).
        Default 0.0982 (1 fs).
    max_disp : float, optional
        MC kernel displacement per axis (A). Default 0.1.
    weight : float, optional
        Relative frequency of each subsurface channel among the GC
        channels (an atomic gas channel has weight 1). Default 1.
    site_min_distance : float, optional
        Sites closer than this to a metal atom are dropped (A). Default 1.0.

    Raises
    ------
    ValueError
        On an empty or inverted region, no species, or bad parameters.
    """

    z_cut: float
    z_min: float
    species: Tuple[str, ...]
    delta: float = 0.3
    ncmc_steps: int = 10
    kernel: str = "hmc"
    kernel_steps: int = 1
    leapfrog_steps: int = 5
    dt: float = 0.0982
    max_disp: float = 0.1
    weight: float = 1.0
    site_min_distance: float = 1.0

    def __post_init__(self):
        """
        Normalise and check the settings.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        self.species = tuple(str(s) for s in self.species)
        self.z_cut, self.z_min = float(self.z_cut), float(self.z_min)
        if not self.z_min < self.z_cut:
            raise ValueError(f"subsurface region needs z_min < z_cut, got {self.z_min}, {self.z_cut}")
        if not self.species:
            raise ValueError("subsurface needs at least one species")
        if self.kernel not in ("hmc", "mc"):
            raise ValueError(f"subsurface kernel must be 'hmc' or 'mc', got {self.kernel!r}")
        if (self.delta <= 0 or self.ncmc_steps < 1 or self.kernel_steps < 0 or self.weight <= 0
                or self.leapfrog_steps < 1 or self.dt <= 0 or self.max_disp <= 0):
            raise ValueError("subsurface: delta, dt, max_disp, weight must be positive and "
                             "ncmc_steps, leapfrog_steps >= 1, kernel_steps >= 0")

    @property
    def v_delta(self) -> float:
        """float: Volume of the insertion ball, 4/3 pi delta^3."""
        return 4.0 / 3.0 * math.pi * self.delta ** 3

    def signature(self) -> tuple:
        """
        Hashable description, for checking that every replica uses the same settings.

        Returns
        -------
        tuple
            Every setting.
        """
        return (round(self.z_cut, 9), round(self.z_min, 9), self.species, self.delta,
                self.ncmc_steps, self.kernel, self.kernel_steps, self.leapfrog_steps, self.dt,
                self.max_disp, self.weight, self.site_min_distance)

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            The settings.
        """
        return {"z_cut": self.z_cut, "z_min": self.z_min, "species": list(self.species),
                "delta": self.delta, "ncmc_steps": self.ncmc_steps, "kernel": self.kernel,
                "kernel_steps": self.kernel_steps, "leapfrog_steps": self.leapfrog_steps,
                "dt": self.dt, "max_disp": self.max_disp, "weight": self.weight,
                "site_min_distance": self.site_min_distance}

    # -- region and sites --------------------------------------------------------------------
    def in_region(self, positions: np.ndarray) -> np.ndarray:
        """
        Whether positions lie in the subsurface region, z_min <= z < z_cut.

        Parameters
        ----------
        positions : numpy.ndarray
            Positions, shape (n, 3).

        Returns
        -------
        numpy.ndarray
            Boolean array, shape (n,).
        """
        z = np.asarray(positions, dtype=float).reshape(-1, 3)[:, 2]
        return (z >= self.z_min) & (z < self.z_cut)

    def sites(self, system) -> np.ndarray:
        """
        Interstitial sites of the metal: Voronoi vertices of its atoms in the region.

        The substrate and frozen atoms, with their periodic images in x and
        y, are tessellated; the vertices in the region and at least
        `site_min_distance` from every metal atom are kept, wrapped into
        the primary cell, and vertices closer than `SITE_MERGE_TOL`
        (minimum image) merged in a deterministic order. A deterministic function of the metal atoms,
        as the proposal density requires.

        Parameters
        ----------
        system : SlabSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Site positions, shape (K, 3) (K may be 0).
        """
        from scipy.spatial import Voronoi
        metal = np.vstack([system.frozen_positions, system.substrate_positions])
        if len(metal) < 5:
            return np.zeros((0, 3))
        A = np.asarray(system._A, dtype=float)               # rows: a1, a2 (xy)
        shifts = [i * A[0] + j * A[1] for i in (-1, 0, 1) for j in (-1, 0, 1)]
        pts = np.vstack([metal + np.array([s[0], s[1], 0.0]) for s in shifts])
        vert = Voronoi(pts).vertices
        # around the primary cell (a margin, so vertices on its edges are not lost to
        # round-off), in the region; wrapped into the cell, duplicates merged below
        frac = vert[:, :2] @ np.linalg.inv(A)
        keep = np.all((frac >= -0.05) & (frac < 1.05), axis=1) & self.in_region(vert)
        vert, frac = vert[keep], frac[keep]
        if len(vert) == 0:
            return np.zeros((0, 3))
        vert = np.column_stack([(frac - np.floor(frac)) @ A, vert[:, 2]])
        d = np.linalg.norm(system._min_image(vert[:, None, :] - metal[None, :, :]), axis=-1)
        vert = vert[d.min(axis=1) >= self.site_min_distance]
        vert = vert[np.lexsort((vert[:, 0], vert[:, 1], vert[:, 2]))]
        kept = []
        for v in vert:
            if not kept or np.min(np.linalg.norm(system._min_image(np.array(kept) - v),
                                                 axis=-1)) > SITE_MERGE_TOL:
                kept.append(v)
        return np.array(kept).reshape(-1, 3)

    def log_q(self, y: np.ndarray, sites: np.ndarray, system) -> float:
        """
        Log density of proposing position y from these sites.

        Parameters
        ----------
        y : numpy.ndarray
            The (3,) position.
        sites : numpy.ndarray
            The sites, shape (K, 3).
        system : SlabSystem
            For the minimum image.

        Returns
        -------
        float
            log[(1/K) sum_{k: |y - s_k| < delta} 1/V_delta]; -inf if no
            site is within delta or y is outside the region.
        """
        if len(sites) == 0 or not self.in_region(y[None, :])[0]:
            return -math.inf
        n = int(np.count_nonzero(np.linalg.norm(system._min_image(sites - y), axis=-1)
                                 < self.delta))
        return -math.inf if n == 0 else math.log(n / (len(sites) * self.v_delta))

    def propose(self, sites: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Draw an insertion point: a uniform site, then uniform in its delta-ball.

        Parameters
        ----------
        sites : numpy.ndarray
            The sites, shape (K >= 1, 3).
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            The (3,) point (not wrapped).
        """
        s = sites[int(rng.integers(len(sites)))]
        u = rng.standard_normal(3)
        r = self.delta * rng.random() ** (1.0 / 3.0)
        return s + r * u / np.linalg.norm(u)


# =============================================================================
# The alchemical energy U_lambda and the NCMC protocol
# =============================================================================
class _Alchemy:
    """
    U_lambda = (1 - lambda) U_without + lambda U_with for one mobile atom.

    U_with is the full energy (calculator + external field); U_without the
    energy with atom `i` removed. A hard boundary (infinite external energy)
    of atom `i` forbids the configuration at every lambda.

    Parameters
    ----------
    system : ParticleSystem
        The configuration, atom `i` present.
    i : int
        Index of the switched atom in the mobile arrays.
    """

    def __init__(self, system, i: int):
        """
        Store the system and the switched atom.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.
        i : int
            The switched atom.
        """
        self.system = system
        self.i = int(i)

    def energies(self) -> Tuple[float, float]:
        """
        U_with and U_without of the current configuration.

        Returns
        -------
        tuple of float
            ``(U_with, U_without)``; ``(inf, inf)`` if atom i is outside a
            hard boundary.
        """
        s, i = self.system, self.i
        if not np.isfinite(s.external_energy(s.positions[i])):
            return np.inf, np.inf
        u_with = s.total_energy()
        pos, sp = s.positions, s.species
        s.positions, s.species = np.delete(pos, i, axis=0), np.delete(sp, i)
        try:
            u_without = s.total_energy()
        finally:
            s.positions, s.species = pos, sp
        return float(u_with), float(u_without)

    def forces(self, lam: float) -> np.ndarray:
        """
        Forces of U_lambda on the substrate and mobile atoms.

        Parameters
        ----------
        lam : float
            The coupling.

        Returns
        -------
        numpy.ndarray
            Shape (n_substrate + n_mobile, 3), substrate rows first.
        """
        s, i = self.system, self.i
        f_with = s.calculator.forces(s)
        pos, sp = s.positions, s.species
        s.positions, s.species = np.delete(pos, i, axis=0), np.delete(sp, i)
        try:
            f_without = s.calculator.forces(s)
        finally:
            s.positions, s.species = pos, sp
        f_without = np.insert(f_without, s.n_substrate + i, 0.0, axis=0)
        return (1.0 - lam) * f_without + lam * f_with


def _kernel_hmc(alch: _Alchemy, lam: float, energies: Tuple[float, float], settings: Subsurface,
                beta: float, rng: np.random.Generator) -> Tuple[float, float]:
    """
    One HMC trajectory at fixed lambda (full momentum refresh; exact Metropolis).

    Parameters
    ----------
    alch : _Alchemy
        The switched system.
    lam : float
        The coupling.
    energies : tuple of float
        ``(U_with, U_without)`` of the current configuration.
    settings : Subsurface
        Leapfrog steps and time step.
    beta : float
        Inverse temperature.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple of float
        ``(U_with, U_without)`` of the configuration after the move.
    """
    s = alch.system
    n_sub = s.n_substrate
    masses = np.concatenate([s.substrate_masses(), s.masses()])
    v = rng.standard_normal((len(masses), 3)) * np.sqrt(1.0 / (beta * masses))[:, None]
    u0 = (1.0 - lam) * energies[1] + lam * energies[0]
    ke0 = 0.5 * np.sum(masses[:, None] * v ** 2)
    old_sub, old_mob = s.substrate_positions.copy(), s.positions.copy()
    pos = np.vstack([old_sub, old_mob])
    f = alch.forces(lam)
    dt = settings.dt
    for _ in range(settings.leapfrog_steps):
        v = v + 0.5 * dt * f / masses[:, None]
        pos = pos + dt * v
        s.substrate_positions, s.positions = pos[:n_sub], pos[n_sub:]
        f = alch.forces(lam)
        v = v + 0.5 * dt * f / masses[:, None]
    new = alch.energies()
    u1 = (1.0 - lam) * new[1] + lam * new[0]
    dh = (u1 - u0) + (0.5 * np.sum(masses[:, None] * v ** 2) - ke0)
    if np.isfinite(dh) and (dh <= 0.0 or rng.random() < math.exp(-beta * dh)):
        for k in range(n_sub):
            s.apply_move_substrate(k, pos[k])
        for k in range(s.n):
            s.apply_move(k, pos[n_sub + k])
        return new
    s.substrate_positions, s.positions = old_sub, old_mob
    return energies


def _kernel_mc(alch: _Alchemy, lam: float, energies: Tuple[float, float], settings: Subsurface,
               beta: float, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Single-atom Metropolis displacements of the displaceable atoms at fixed lambda.

    Parameters
    ----------
    alch : _Alchemy
        The switched system.
    lam : float
        The coupling.
    energies : tuple of float
        ``(U_with, U_without)`` of the current configuration.
    settings : Subsurface
        Displacement size and attempts per atom.
    beta : float
        Inverse temperature.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple of float
        ``(U_with, U_without)`` of the configuration after the moves.
    """
    s = alch.system
    n_sub = s.n_substrate
    for _ in range(max(1, (n_sub + s.n))):
        k = int(rng.integers(n_sub + s.n))
        step = (rng.random(3) - 0.5) * 2.0 * settings.max_disp
        if k < n_sub:
            old = s.substrate_positions[k].copy()
            s.substrate_positions[k] = old + step
        else:
            old = s.positions[k - n_sub].copy()
            s.positions[k - n_sub] = old + step
        new = alch.energies()
        du = ((1.0 - lam) * (new[1] - energies[1]) + lam * (new[0] - energies[0]))
        if np.isfinite(du) and (du <= 0.0 or rng.random() < math.exp(-beta * du)):
            if k < n_sub:
                s.apply_move_substrate(k, s.substrate_positions[k])
            else:
                s.apply_move(k - n_sub, s.positions[k - n_sub])
            energies = new
        elif k < n_sub:
            s.substrate_positions[k] = old
        else:
            s.positions[k - n_sub] = old
    return energies


def ncmc(system, i: int, insert: bool, settings: Subsurface, beta: float,
         rng: np.random.Generator) -> Tuple[float, Tuple[float, float]]:
    """
    Switch mobile atom `i` on (insert) or off with the NCMC protocol.

    lambda_t = t / T (or 1 - t / T); for t = 1..T the work gains
    (lambda_t - lambda_{t-1}) (U_with - U_without) at the current
    configuration, and for t < T the configuration is propagated at
    lambda_t (`kernel_steps` applications of the kernel). The reversed
    schedule uses the kernels in the reversed order, as detailed balance of
    the pair of moves requires.

    Parameters
    ----------
    system : ParticleSystem
        The configuration, atom `i` present (modified in place).
    i : int
        Index of the switched atom in the mobile arrays.
    insert : bool
        lambda 0 -> 1 (True) or 1 -> 0.
    settings : Subsurface
        Protocol and kernel.
    beta : float
        Inverse temperature.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    tuple
        ``(W, (U_with, U_without))``: the protocol work and the two
        energies of the final configuration (W is inf for a forbidden
        configuration).
    """
    alch = _Alchemy(system, i)
    kernel = _kernel_hmc if settings.kernel == "hmc" else _kernel_mc
    T = settings.ncmc_steps
    lams = np.linspace(0.0, 1.0, T + 1) if insert else np.linspace(1.0, 0.0, T + 1)
    energies = alch.energies()
    work = 0.0
    for t in range(1, T + 1):
        if not np.all(np.isfinite(energies)):
            return math.inf, energies
        work += (lams[t] - lams[t - 1]) * (energies[0] - energies[1])
        if t < T:
            for _ in range(settings.kernel_steps):
                energies = kernel(alch, float(lams[t]), energies, settings, beta, rng)
    return float(work), energies
