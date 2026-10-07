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
* `insertion_energy_scan` -- the energy of inserting one particle along
  vertical lines above chosen lateral points (adsorption-site scans).

Requires ASE (imported lazily).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .calculators import Calculator
from .systems import SlabSystem


def cubic_lattice_constant(ase_calculator, symbol: str, crystal: str = "bcc",
                           a_range: Tuple[float, float] = (2.3, 3.5), n_coarse: int = 61,
                           n_fine: int = 21) -> float:
    """
    Equilibrium lattice constant of a cubic crystal for a given potential.

    A coarse energy scan of the conventional cubic cell locates the minimum,
    a finer scan around it is fitted with a parabola.

    Parameters
    ----------
    ase_calculator : ase.calculators.calculator.Calculator
        The potential (e.g. ASE's EAM, or
        `gcmc_re.lammps_backend.lammps_ase_calculator`).
    symbol : str
        Element symbol.
    crystal : {'bcc', 'fcc', 'sc'}, optional
        Crystal structure. Default ``'bcc'``.
    a_range : tuple of float, optional
        Range (A) of the coarse scan. Default (2.3, 3.5).
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

    def make_system(self, calculator: Calculator, exchange_species: str,
                    mass_of_species: Optional[dict] = None,
                    insertion_z_range: Optional[Tuple[float, float]] = None) -> SlabSystem:
        """
        Build the `SlabSystem` for this geometry.

        Parameters
        ----------
        calculator : Calculator
            The energy backend (each system needs its own instance).
        exchange_species : str
            Species GCMC inserts and deletes.
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
