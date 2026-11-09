"""
Building simulation cells from a potential and an ASE structure.

* `cubic_lattice_constant` -- the equilibrium lattice constant of an element
  *as a given potential sees it* (energy scan + parabola), so a slab built
  with it is not strained.
* `slab_geometry_from_atoms` / `SlabGeometry` -- turn any ASE slab
  (``ase.build.fcc111``, ``bcc110``, ..., orthogonal or hexagonal cell) into
  the layout the package uses: the frozen block at z <= 0 (its top layer at
  z = 0, the box's hard wall), displaceable substrate layers above it, and a
  headroom for the exchanged species; `SlabGeometry.make_system` then builds
  the `SlabSystem` for any calculator.
* `cluster_geometry_from_atoms` / `ClusterGeometry` -- the same for a
  nanoparticle (``ase.cluster.Icosahedron``, ``Octahedron``, ... or a file):
  centred in a cubic box with no periodicity, its central atom (or core)
  frozen, surface atoms counted by coordination; `ClusterGeometry.make_system`
  builds the `ClusterSystem`.
* `insertion_energy_scan` -- the energy of inserting one particle along
  vertical lines above chosen lateral points (adsorption-site scans).

Requires ASE (imported lazily).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .calculators import Calculator
from .systems import ClusterSystem, SlabSystem


def reference_lattice_constant(symbol: str, crystal: str) -> Optional[float]:
    """
    The experimental cubic lattice constant of an element, from ASE's tables.

    Parameters
    ----------
    symbol : str
        Element symbol.
    crystal : {'bcc', 'fcc', 'sc'}
        Crystal structure.

    Returns
    -------
    float or None
        ``ase.data.reference_states`` lattice constant (A) when the
        element's reference structure is `crystal`, else None.
    """
    from ase.data import atomic_numbers, reference_states
    ref = reference_states[atomic_numbers[symbol]] if symbol in atomic_numbers else None
    if ref and ref.get("symmetry") == crystal and "a" in ref:
        return float(ref["a"])
    return None


def cubic_lattice_constant(ase_calculator, symbol: str, crystal: str = "bcc",
                           a_range: Optional[Tuple[float, float]] = None, n_coarse: int = 61,
                           n_fine: int = 21) -> float:
    """
    Equilibrium lattice constant of a cubic crystal for a given potential.

    A coarse energy scan of the conventional cubic cell locates the minimum,
    a finer scan around it is fitted with a parabola.

    Parameters
    ----------
    ase_calculator : ase.calculators.calculator.Calculator
        The potential (e.g. ASE's EAM, or
        `regc.lammps_backend.lammps_ase_calculator`).
    symbol : str
        Element symbol.
    crystal : {'bcc', 'fcc', 'sc'}, optional
        Crystal structure. Default ``'bcc'``.
    a_range : tuple of float, optional
        Range (A) of the coarse scan. Default: 0.8 to 1.2 times the
        element's experimental lattice constant (`reference_lattice_constant`),
        or (2.3, 5.5) if ASE has none for this crystal. Keep the range near
        the real value for machine-learned potentials: far below it they
        can have unphysical energy holes, which a wide scan finds.
    n_coarse, n_fine : int, optional
        Points of the coarse and fine scans. Defaults 61 and 21.

    Returns
    -------
    float
        The lattice constant, in Angstrom.

    Raises
    ------
    ValueError
        If the energy has no minimum inside `a_range` (wrong potential or
        range) or is not convex around it.
    """
    from ase.build import bulk

    def energy(a: float) -> float:
        atoms = bulk(symbol, crystal, a=a, cubic=True)
        atoms.calc = ase_calculator
        return float(atoms.get_potential_energy())

    if a_range is None:
        a_ref = reference_lattice_constant(symbol, crystal)
        a_range = (0.8 * a_ref, 1.2 * a_ref) if a_ref else (2.3, 5.5)
    coarse = np.linspace(a_range[0], a_range[1], n_coarse)
    e_coarse = np.array([energy(a) for a in coarse])
    i = int(np.argmin(e_coarse))
    if i in (0, len(coarse) - 1):
        raise ValueError(f"{crystal} {symbol} energy has no minimum for a in "
                         f"[{coarse[0]}, {coarse[-1]}] A (lowest at the edge, a = {coarse[i]:.3f}); "
                         f"wrong potential, element or a_range?")
    fine = np.linspace(coarse[i - 1], coarse[i + 1], n_fine)
    c = np.polyfit(fine, [energy(a) for a in fine], 2)
    if c[0] <= 0:
        raise ValueError(f"{crystal} {symbol} energy is not convex around its minimum")
    return float(-c[1] / (2 * c[0]))


@dataclass
class SlabGeometry:
    """
    A slab laid out for `SlabSystem`: frozen block at z <= 0, substrate above.

    Attributes
    ----------
    cell_xy : numpy.ndarray
        2x2 in-plane cell (rows a1, a2).
    Lz : float
        Height of the box the exchanged species lives in, (0, Lz].
    top_z : float
        z of the topmost slab atom.
    frozen_positions, substrate_positions : numpy.ndarray
        Positions of the rigid and the displaceable slab atoms, (M, 3).
    frozen_species, substrate_species : numpy.ndarray
        Their species labels.
    layer_z : numpy.ndarray
        z of every atomic layer, ascending (after the shift).
    n_surface : int
        Number of atoms in the top layer (for coverages per surface atom).
    """

    cell_xy: np.ndarray
    Lz: float
    top_z: float
    frozen_positions: np.ndarray
    frozen_species: np.ndarray
    substrate_positions: np.ndarray
    substrate_species: np.ndarray
    layer_z: np.ndarray = field(repr=False)
    n_surface: int = 0

    @property
    def orthogonal(self) -> bool:
        """bool: Whether the in-plane cell is rectangular."""
        return abs(self.cell_xy[0, 1]) < 1e-12 and abs(self.cell_xy[1, 0]) < 1e-12

    @property
    def area(self) -> float:
        """float: Surface-cell area, in A^2."""
        return float(abs(np.linalg.det(self.cell_xy)))

    def make_system(self, calculator: Calculator, exchange_species: Union[str, Sequence[str]],
                    mass_of_species: Optional[dict] = None,
                    insertion_z_range: Optional[Tuple[float, float]] = None) -> SlabSystem:
        """
        Build the `SlabSystem` for this geometry.

        Parameters
        ----------
        calculator : Calculator
            The energy backend (each system needs its own instance).
        exchange_species : str or sequence of str
            Species GCMC inserts and deletes (several, e.g. ``['O', 'Cl']``,
            each with its own reservoir; see `ParticleSystem`).
        mass_of_species : dict, optional
            Masses (amu) by species, for MD/HMC. Default None (ASE tables).
        insertion_z_range : tuple of float, optional
            Restrict insertions to this z-range (see `SlabSystem`); default
            None, the whole box (0, Lz].

        Returns
        -------
        SlabSystem
            An empty (no mobile particles) system.
        """
        kw = dict(exchange_species=exchange_species,
                  frozen_positions=self.frozen_positions, frozen_species=self.frozen_species,
                  substrate_positions=self.substrate_positions,
                  substrate_species=self.substrate_species, mass_of_species=mass_of_species,
                  insertion_z_range=insertion_z_range)
        if self.orthogonal:
            return SlabSystem(self.cell_xy[0, 0], self.cell_xy[1, 1], self.Lz, calculator, **kw)
        return SlabSystem(None, None, self.Lz, calculator, cell_xy=self.cell_xy, **kw)

    def summary(self) -> str:
        """
        One-line description.

        Returns
        -------
        str
            Atom counts, cell and box height.
        """
        a1, a2 = (np.linalg.norm(v) for v in self.cell_xy)
        return (f"{len(self.frozen_positions)} frozen + {len(self.substrate_positions)} displaceable "
                f"slab atoms, {self.n_surface} per surface layer, cell |a1| = {a1:.3f} A, "
                f"|a2| = {a2:.3f} A ({'rectangular' if self.orthogonal else 'oblique'}), "
                f"mobile species in z in (0, {self.Lz:.2f}] A")


def slab_geometry_from_atoms(atoms, n_frozen_layers: Optional[int] = None, headroom: float = 5.0,
                             layer_tol: float = 1e-3) -> SlabGeometry:
    """
    Lay out an ASE slab for `SlabSystem`.

    Layers are identified by z. The slab is shifted so the top *frozen*
    layer is at z = 0: the box's hard wall then keeps the exchanged species
    out of the frozen block, while it can still reach the displaceable
    layers above (subsurface sites included).

    Parameters
    ----------
    atoms : ase.Atoms
        The slab, with a1 along x and a2 in the xy plane (as ``ase.build``
        surface functions give), periodic in x and y.
    n_frozen_layers : int, optional
        Number of bottom layers kept rigid; the rest are displaceable
        substrate. Default None: all layers frozen (a rigid surface).
    headroom : float, optional
        Space (A) for the exchanged species above the topmost atom.
        Default 5.0.
    layer_tol : float, optional
        z tolerance (A) for grouping atoms into layers. Default 1e-3.

    Returns
    -------
    SlabGeometry
        The geometry.

    Raises
    ------
    ValueError
        If the cell is not in the restricted (a1 along x, a2 in xy) form, or
        `n_frozen_layers` is out of range.
    """
    cell = np.array(atoms.cell[:], dtype=float)
    if abs(cell[0, 1]) > 1e-8 or abs(cell[0, 2]) > 1e-8 or abs(cell[1, 2]) > 1e-8:
        raise ValueError("the slab cell must have a1 along x and a2 in the xy plane")
    pos = atoms.positions.copy()
    species = np.array(atoms.get_chemical_symbols())
    z_round = np.round(pos[:, 2] / layer_tol) * layer_tol
    layers = np.unique(z_round)
    layer_of = np.searchsorted(layers, z_round)
    n_layers = len(layers)
    if n_frozen_layers is None:
        n_frozen_layers = n_layers
    if not 0 < n_frozen_layers <= n_layers:
        raise ValueError(f"n_frozen_layers must be in 1..{n_layers}, got {n_frozen_layers}")
    # exact z of the top frozen layer's highest atom (not the rounded layer key)
    shift = float(pos[layer_of == n_frozen_layers - 1, 2].max())
    pos[:, 2] -= shift
    frozen = layer_of < n_frozen_layers
    top_z = float(pos[:, 2].max())
    return SlabGeometry(cell_xy=cell[:2, :2].copy(), Lz=top_z + headroom, top_z=top_z,
                        frozen_positions=pos[frozen], frozen_species=species[frozen],
                        substrate_positions=pos[~frozen], substrate_species=species[~frozen],
                        layer_z=layers - shift, n_surface=int(np.sum(layer_of == n_layers - 1)))


@dataclass
class ClusterGeometry:
    """
    A cluster laid out for `ClusterSystem`: centred in a cubic, non-periodic box.

    Attributes
    ----------
    box : float
        Edge of the cubic box, in A.
    center : numpy.ndarray
        The box centre (3,), where the cluster's centroid sits.
    radius : float
        Largest distance of a cluster atom from `center`, in A.
    frozen_positions, substrate_positions : numpy.ndarray
        Positions of the rigid and the displaceable cluster atoms, (M, 3).
    frozen_species, substrate_species : numpy.ndarray
        Their species labels.
    coordination : numpy.ndarray
        Coordination number of every cluster atom (frozen first, then
        substrate, as in `full_configuration`).
    n_surface : int
        Number of surface atoms (coordination below full), for coverages
        per surface atom.
    """

    box: float
    center: np.ndarray
    radius: float
    frozen_positions: np.ndarray
    frozen_species: np.ndarray
    substrate_positions: np.ndarray
    substrate_species: np.ndarray
    coordination: np.ndarray = field(repr=False)
    n_surface: int = 0

    @property
    def n_atoms(self) -> int:
        """int: Number of cluster atoms (frozen + displaceable)."""
        return len(self.frozen_positions) + len(self.substrate_positions)

    @property
    def area(self) -> None:
        """None: a cluster has no surface-cell area (coverages per atom instead)."""
        return None

    def shell(self, d_in: float, d_out: float) -> Tuple[float, float]:
        """
        Insertion-shell radii from distances relative to the cluster's surface.

        Parameters
        ----------
        d_in, d_out : float
            Inner and outer distance from `radius` (A); negative values
            reach into the cluster.

        Returns
        -------
        tuple of float
            ``(max(0, radius + d_in), radius + d_out)``, for
            ``ClusterSystem(insertion_shell=...)``.

        Raises
        ------
        ValueError
            If the shell is empty or does not fit in the box.
        """
        r_in, r_out = max(0.0, self.radius + d_in), self.radius + d_out
        if not r_in < r_out:
            raise ValueError(f"empty insertion shell: r_in = {r_in:.3f}, r_out = {r_out:.3f} A")
        if r_out > 0.5 * self.box + 1e-9:
            raise ValueError(f"insertion shell r_out = {r_out:.3f} A exceeds half the box "
                             f"({0.5 * self.box:.3f} A): raise the headroom")
        return (r_in, r_out)

    def make_system(self, calculator: Calculator, exchange_species: Union[str, Sequence[str]],
                    mass_of_species: Optional[dict] = None,
                    insertion_shell: Optional[Tuple[float, float]] = None) -> ClusterSystem:
        """
        Build the `ClusterSystem` for this geometry.

        Parameters
        ----------
        calculator : Calculator
            The energy backend (each system needs its own instance).
        exchange_species : str or sequence of str
            Species GCMC inserts and deletes.
        mass_of_species : dict, optional
            Masses (amu) by species, for MD/HMC. Default None (ASE tables).
        insertion_shell : tuple of float, optional
            ``(r_in, r_out)`` around the centre (see `shell`). Default None:
            the whole box.

        Returns
        -------
        ClusterSystem
            An empty (no mobile particles) system.
        """
        return ClusterSystem(self.box, calculator, exchange_species=exchange_species,
                             frozen_positions=self.frozen_positions,
                             frozen_species=self.frozen_species,
                             substrate_positions=self.substrate_positions,
                             substrate_species=self.substrate_species,
                             insertion_shell=insertion_shell, center=self.center,
                             mass_of_species=mass_of_species)

    def summary(self) -> str:
        """
        One-line description.

        Returns
        -------
        str
            Atom counts, radius and box.
        """
        species = ", ".join(f"{n} {s}" for s, n in zip(*np.unique(
            np.concatenate([self.frozen_species, self.substrate_species]), return_counts=True)))
        return (f"{self.n_atoms} atoms ({species}): {len(self.frozen_positions)} frozen + "
                f"{len(self.substrate_positions)} displaceable, {self.n_surface} on the surface, "
                f"radius {self.radius:.2f} A, in a {self.box:.2f} A cubic box (no periodicity)")


def coordination_numbers(positions: np.ndarray, cutoff: Optional[float] = None) -> np.ndarray:
    """
    Coordination number of every atom of a finite (non-periodic) structure.

    Parameters
    ----------
    positions : numpy.ndarray
        Atom positions, (N, 3).
    cutoff : float, optional
        Neighbour cutoff, in A. Default: halfway between the fcc first and
        second neighbour shells, ``(1 + sqrt 2)/2`` times the median
        nearest-neighbour distance (12 neighbours inside an fcc or
        icosahedral particle; 14 inside a bcc one).

    Returns
    -------
    numpy.ndarray
        Integer coordination numbers, (N,).
    """
    pos = np.asarray(positions, dtype=float)
    if len(pos) < 2:
        return np.zeros(len(pos), dtype=int)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    if cutoff is None:
        cutoff = 0.5 * (1.0 + np.sqrt(2.0)) * float(np.median(d.min(axis=1)))
    return np.count_nonzero(d < cutoff, axis=1)


def cluster_geometry_from_atoms(atoms, headroom: float = 6.0,
                                frozen: Union[str, int] = "center",
                                full_coordination: int = 12,
                                cn_cutoff: Optional[float] = None) -> ClusterGeometry:
    """
    Lay out a cluster for `ClusterSystem`.

    The cluster is translated so its centroid is at the centre of a cubic
    box of edge ``2 (radius + headroom)``: a spherical insertion shell up
    to ``headroom`` beyond the outermost atom then fits inside the box.

    Parameters
    ----------
    atoms : ase.Atoms
        The cluster (its cell and periodicity are ignored).
    headroom : float, optional
        Space (A) between the outermost atom and the box walls along the
        axes. Default 6.0.
    frozen : {'center', 'core', 'none'} or int, optional
        Which atoms are rigid: the one nearest the centroid (default; pins
        the particle in place while it relaxes), every fully coordinated
        atom (``coordination >= full_coordination``), none, or the given
        number of atoms nearest the centroid.
    full_coordination : int, optional
        Coordination of an interior atom: 12 for fcc, hcp and icosahedral
        particles, 14 for bcc with the default cutoff. Atoms below it are
        surface atoms. Default 12.
    cn_cutoff : float, optional
        Neighbour cutoff for the coordination numbers (see
        `coordination_numbers`). Default None (automatic).

    Returns
    -------
    ClusterGeometry
        The geometry.

    Raises
    ------
    ValueError
        If `frozen` is not understood, or the cluster has no atoms.
    """
    pos = np.array(atoms.positions, dtype=float)
    species = np.array(atoms.get_chemical_symbols())
    if len(pos) == 0:
        raise ValueError("the cluster has no atoms")
    centroid = pos.mean(axis=0)
    dist = np.linalg.norm(pos - centroid, axis=1)
    radius = float(dist.max())
    box = 2.0 * (radius + float(headroom))
    center = np.full(3, 0.5 * box)
    pos = pos - centroid + center
    cn = coordination_numbers(pos, cn_cutoff)
    if isinstance(frozen, str):
        key = frozen.lower()
        if key == "center":
            is_frozen = np.zeros(len(pos), dtype=bool)
            is_frozen[int(np.argmin(dist))] = True
        elif key == "core":
            is_frozen = cn >= full_coordination
        elif key == "none":
            is_frozen = np.zeros(len(pos), dtype=bool)
        else:
            raise ValueError(f"frozen must be 'center', 'core', 'none' or a number, got {frozen!r}")
    else:
        n_fr = int(frozen)
        if not 0 <= n_fr <= len(pos):
            raise ValueError(f"frozen must be in 0..{len(pos)}, got {n_fr}")
        is_frozen = np.zeros(len(pos), dtype=bool)
        is_frozen[np.argsort(dist, kind="stable")[:n_fr]] = True
    order = np.concatenate([np.flatnonzero(is_frozen), np.flatnonzero(~is_frozen)])
    return ClusterGeometry(box=box, center=center, radius=radius,
                           frozen_positions=pos[is_frozen], frozen_species=species[is_frozen],
                           substrate_positions=pos[~is_frozen],
                           substrate_species=species[~is_frozen],
                           coordination=cn[order],
                           n_surface=int(np.count_nonzero(cn < full_coordination)))


def insertion_energy_scan(system, xy_points: Sequence[Sequence[float]],
                          z_values: Sequence[float]) -> np.ndarray:
    """
    Energy of inserting one particle along vertical lines.

    For each lateral point and height, the energy change of inserting one
    `system.exchange_species` particle there (the system is unchanged).
    With `xy_points` above adsorption sites, the minimum over z of each row
    is the site's unrelaxed binding energy relative to the calculator's
    zero (subtract a reservoir reference, e.g. 1/2 E(H2), for adsorption
    energies).

    Parameters
    ----------
    system : ParticleSystem
        The system (typically a bare slab).
    xy_points : sequence of (float, float)
        Lateral positions, in A.
    z_values : sequence of float
        Heights, in A (positions outside the box give +inf).

    Returns
    -------
    numpy.ndarray
        Shape (len(xy_points), len(z_values)): insertion energies in eV.
        Points where the calculator returns NaN (e.g. exactly on top of an
        atom, r = 0) are reported as +inf, so ``argmin`` finds real minima.
    """
    out = np.empty((len(xy_points), len(z_values)))
    for i, (x, y) in enumerate(xy_points):
        for j, z in enumerate(z_values):
            out[i, j] = system.generic_insert_dU(np.array([x, y, z], dtype=float))
    return np.where(np.isnan(out), np.inf, out)
