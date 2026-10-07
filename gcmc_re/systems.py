"""
Geometry, boundary conditions, and species bookkeeping.

`ParticleSystem` is the geometry base class: it owns positions and three
particle categories, delegating all physics to a pluggable `Calculator`
(see `calculators.py`):

  * mobile (`positions`/`species`) -- GCMC-exchangeable: displaced,
    inserted, and deleted.
  * substrate (`substrate_positions`/`substrate_species`) -- a fixed
    inventory that enters the energy evaluation and CAN be displaced (MC
    or, more usefully, jointly with HMC -- see `replica.py`), but is never
    inserted or deleted. This is what a reconstructing surface needs: real
    atoms that can relax/rearrange, without being part of the grand
    canonical exchange.
  * frozen (`frozen_positions`/`frozen_species`) -- fully rigid: enters
    the energy evaluation but never moves at all, by any mechanism. Use
    this for whatever part of a substrate you don't need to let relax
    (e.g. deep, bulk-truncated layers far from the surface).

Two concrete geometries (both work with any `Calculator`):
  * `BulkSystem` -- bulk, fully-periodic cube.
  * `SlabSystem` -- periodic in x, y only (rectangular or general, e.g.
    hexagonal, surface cell); hard walls at z = 0 and z = Lz; a continuum
    9-3 wall OR an explicit substrate (rigid, displaceable, or both).

`LJSystem` and `SlabLJSystem` are the historical names of these two
classes and remain available as aliases.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .calculators import Calculator


class ParticleSystem:
    """
    Geometry, boundary conditions, and species bookkeeping.

    Owns positions and species for all three particle categories
    described in this module's docstring (mobile, substrate, frozen). All
    physics is delegated to `calculator`.
    """

    def __init__(self, calculator: Calculator, exchange_species: str = 'X',
                 positions: Optional[np.ndarray] = None,
                 species: Optional[np.ndarray] = None,
                 frozen_positions: Optional[np.ndarray] = None,
                 frozen_species: Optional[np.ndarray] = None,
                 substrate_positions: Optional[np.ndarray] = None,
                 substrate_species: Optional[np.ndarray] = None,
                 mass_of_species: Optional[dict] = None):
        """
        Construct a particle system.

        Parameters
        ----------
        calculator : Calculator
            The energy-evaluation backend all physics is delegated to.
        exchange_species : str, optional
            The species label given to newly inserted mobile particles
            (and used as the default `species` for any `positions` given
            without an explicit `species` array). Default is ``'X'``.
        positions : numpy.ndarray, optional
            Initial mobile (GCMC-exchangeable) particle positions, shape
            (N, 3). Default is None (no mobile particles).
        species : numpy.ndarray, optional
            Species label for each mobile particle in `positions`, shape
            (N,). Default is None, which fills every entry with
            `exchange_species`.
        frozen_positions : numpy.ndarray, optional
            Positions of fully rigid atoms that enter the energy
            evaluation but never move, shape (M, 3). Default is None (no
            frozen atoms).
        frozen_species : numpy.ndarray, optional
            Species label for each frozen atom, shape (M,). Default is
            None (empty).
        substrate_positions : numpy.ndarray, optional
            Positions of displaceable-but-non-exchangeable substrate
            atoms, shape (K, 3). Default is None (no substrate atoms).
        substrate_species : numpy.ndarray, optional
            Species label for each substrate atom, shape (K,). Default is
            None (empty).
        mass_of_species : dict, optional
            Mapping from species label to mass (in atomic mass units),
            used by `masses()`/`substrate_masses()` for HMC. Species not
            present in this dict fall back to `units.default_mass_amu`.
            Default is None (empty; every species falls back).
        """
        self.calculator = calculator
        self.exchange_species = exchange_species
        self.positions = (np.zeros((0, 3)) if positions is None
                           else np.array(positions, dtype=float).reshape(-1, 3))
        self.species = (np.array([exchange_species] * self.positions.shape[0])
                         if species is None else np.asarray(species))
        self.frozen_positions = (np.zeros((0, 3)) if frozen_positions is None
                                  else np.array(frozen_positions, dtype=float).reshape(-1, 3))
        self.frozen_species = (np.array([], dtype='<U2') if frozen_species is None
                                else np.asarray(frozen_species))
        self.substrate_positions = (np.zeros((0, 3)) if substrate_positions is None
                                     else np.array(substrate_positions, dtype=float).reshape(-1, 3))
        self.substrate_species = (np.array([], dtype='<U2') if substrate_species is None
                                   else np.asarray(substrate_species))
        self.mass_of_species = dict(mass_of_species) if mass_of_species else {}

    @property
    def n(self) -> int:
        """
        int: The number of mobile (GCMC-exchangeable) particles.
        """
        return self.positions.shape[0]

    @property
    def n_substrate(self) -> int:
        """
        int: The number of displaceable substrate atoms.
        """
        return self.substrate_positions.shape[0]

    def _mass_lookup(self, species_array: np.ndarray) -> np.ndarray:
        """
        Look up a mass for each entry of a species array.

        Parameters
        ----------
        species_array : numpy.ndarray
            Array of species labels to look up masses for.

        Returns
        -------
        numpy.ndarray
            The mass (in atomic mass units) for each entry of
            `species_array`, from `self.mass_of_species` if present, else
            `units.default_mass_amu`.
        """
        from .units import default_mass_amu
        return np.array([self.mass_of_species.get(s, default_mass_amu(s)) for s in species_array])

    def masses(self) -> np.ndarray:
        """
        Get the per-mobile-particle mass array, for HMC.

        Returns
        -------
        numpy.ndarray
            Array of shape (n,), in `system.positions` order. Looked up
            from `mass_of_species` if given for that particle's species;
            otherwise a best-effort real atomic mass (via ASE, if
            installed) or 1.0 (reduced units) as a fallback -- see
            `units.default_mass_amu`.
        """
        return self._mass_lookup(self.species)

    def substrate_masses(self) -> np.ndarray:
        """
        Get the per-substrate-atom mass array, for HMC.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_substrate,), in `system.substrate_positions`
            order, using the same lookup as `masses()`. Substrate atoms
            are typically much heavier than the mobile species, which
            `mass_of_species` should reflect for realistic HMC dynamics
            (e.g. ``{'Fe': 55.845, 'H': 1.008}``).
        """
        return self._mass_lookup(self.substrate_species)

    # -- geometry hooks (subclasses must provide) --------------------------
    def _min_image(self, dr: np.ndarray) -> np.ndarray:
        """
        Apply the minimum-image convention to a displacement vector.

        Parameters
        ----------
        dr : numpy.ndarray
            One or more displacement vectors, shape (..., 3).

        Returns
        -------
        numpy.ndarray
            `dr`, wrapped into the periodic images nearest the origin
            along each periodic axis, same shape as `dr`.

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this
            with their own periodicity.
        """
        raise NotImplementedError

    @property
    def min_image_radius(self) -> float:
        """
        float: Largest interaction cutoff for which the minimum-image
        convention (`_min_image`) counts every periodic image: half the
        smallest lattice-plane spacing. ``inf`` on the base class (no
        periodicity known).
        """
        return np.inf

    @property
    def volume(self) -> float:
        """
        float: The accessible volume used in the GCMC insertion/deletion
        acceptance formulas.

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    @property
    def cell(self) -> Tuple[float, float, float]:
        """
        tuple of float: (Lx, Ly, Lz)-style orthorhombic cell dimensions,
        for anything that needs to write this system's geometry out (e.g.
        trajectory saving in `engine.py`).

        Not required for the simulation itself -- only for that optional
        feature -- so subclasses that skip it just can't use it.

        Raises
        ------
        NotImplementedError
            On the base class, and on any subclass that doesn't define a
            cell.
        """
        raise NotImplementedError(
            "This geometry doesn't define a cell -- trajectory saving isn't available for it."
        )

    @property
    def pbc(self) -> Tuple[bool, bool, bool]:
        """
        tuple of bool: Periodicity along each cell axis, paired with
        `cell` for anything that writes this system's geometry out.

        Subclasses must override this alongside `cell` -- there is no
        sensible shared default (a bulk cube and a slab have genuinely
        different periodicity).

        Raises
        ------
        NotImplementedError
            On the base class, and on any subclass that doesn't define
            periodicity.
        """
        raise NotImplementedError(
            "This geometry doesn't define pbc -- trajectory saving isn't available for it."
        )

    def external_energy(self, pos: np.ndarray) -> float:
        """
        Evaluate any one-body external field at a position.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position to evaluate the field at.

        Returns
        -------
        float
            The field energy at `pos` (e.g. from a continuum wall).
            Default (on the base class) is always 0.0, i.e. no field.
            ``numpy.inf`` marks a forbidden region -- moves there are
            rejected for free by the Metropolis criterion.
        """
        return 0.0

    def random_insertion_position(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a random trial position for a GCMC insertion move.

        Parameters
        ----------
        rng : numpy.random.Generator
            The random number generator to draw from.

        Returns
        -------
        numpy.ndarray
            A (3,) Cartesian position, uniformly distributed over the
            geometry's accessible insertion volume.

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    def apply_move(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th mobile particle.

        Parameters
        ----------
        i : int
            Index of the mobile particle (into `self.positions`) to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to, wrapped through the
            geometry's own boundary conditions before being stored.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    def apply_move_substrate(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th substrate atom.

        Through the same geometry-specific wrapping `apply_move` uses.
        Only needed if you actually have substrate atoms and want to move
        them (MC or HMC).

        Parameters
        ----------
        i : int
            Index of the substrate atom (into `self.substrate_positions`)
            to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    def apply_insert(self, pos: np.ndarray) -> None:
        """
        Commit the insertion of a new mobile particle.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position of the new particle, wrapped
            through the geometry's own boundary conditions before being
            stored. Its species is `self.exchange_species`.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    def apply_delete(self, i: int) -> None:
        """
        Commit the deletion of a mobile particle.

        Parameters
        ----------
        i : int
            Index of the mobile particle (into `self.positions` and
            `self.species`) to remove.

        Returns
        -------
        None
        """
        self.positions = np.delete(self.positions, i, axis=0)
        self.species = np.delete(self.species, i)

    def exchange_zone_indices(self) -> np.ndarray:
        """
        Indices of the mobile particles inside the insertion/deletion zone.

        GCMC deletion may only pick particles from the same region that
        `random_insertion_position` samples (whose size is `volume`), and
        the acceptance rules use this count, not `n`, for detailed balance.
        The base-class zone is the whole system.

        Returns
        -------
        numpy.ndarray
            Integer indices into `self.positions`.
        """
        return np.arange(self.n)

    def md_z_bounds(self) -> Optional[Tuple[float, float]]:
        """
        Hard z-boundaries that MD must enforce by reflection, if any.

        MD integrates the calculator's forces only, so it cannot see the
        infinite `external_energy` outside the box; `Replica._diffuse_md`
        reflects positions/velocities at these bounds instead. The base
        class has none (fully periodic).

        Returns
        -------
        tuple of float or None
            ``(z_min, z_max)``, or None.
        """
        return None

    def full_configuration(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble the full configuration for a Calculator to evaluate.

        Combines the frozen atoms, the displaceable substrate, and the
        mobile particles (in that order, skipping any empty category).
        `Calculator.forces()` relies on this exact order to know which
        rows of its own output are frozen (and so can be dropped) -- see
        `ASECalculatorAdapter.forces`.

        Returns
        -------
        tuple of numpy.ndarray
            ``(positions, species)``: the concatenated positions, shape
            (n_frozen + n_substrate + n_mobile, 3), and species, shape
            (n_frozen + n_substrate + n_mobile,).
        """
        position_parts = [p for p in (self.frozen_positions, self.substrate_positions, self.positions)
                           if p.shape[0] > 0]
        species_parts = [s for s in (self.frozen_species, self.substrate_species, self.species)
                          if s.shape[0] > 0]
        if not position_parts:
            return np.zeros((0, 3)), np.array([], dtype='<U2')
        return np.vstack(position_parts), np.concatenate(species_parts)

    # -- fast incremental path: only valid if calculator.supports_incremental
    def pair_energy_of(self, pos: np.ndarray, exclude: Optional[int] = None) -> float:
        """
        Compute the energy of a single point with the rest of the system.

        Only usable when `self.calculator.supports_incremental` is True.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position to evaluate.
        exclude : int, optional
            If given, the index of a mobile particle to exclude from the
            sum (used when `pos` is a trial position for that particle
            itself). Default is None.

        Returns
        -------
        float
            The calculator's pairwise energy of `pos` with the rest of
            the mobile particles, plus any `external_energy` at `pos`
            (or, if that field is infinite there, just that infinity,
            without evaluating the calculator at all).

        Raises
        ------
        NotImplementedError
            If `self.calculator.supports_incremental` is False -- use the
            generic ``generic_*_dU`` methods instead.
        """
        if not self.calculator.supports_incremental:
            raise NotImplementedError(
                "This calculator has no fast single-particle energy; "
                "Replica uses the generic_*_dU path for it automatically."
            )
        ext = self.external_energy(pos)
        if not np.isfinite(ext):
            return ext
        return self.calculator.pair_energy_of(self, pos, exclude=exclude) + ext

    def total_energy(self) -> float:
        """
        Compute the total energy of the current configuration.

        Returns
        -------
        float
            The calculator's total energy for the full configuration,
            plus the sum of `external_energy` over every mobile and
            substrate particle (frozen atoms don't call `external_energy`,
            since they never move and so can never enter a forbidden
            region).
        """
        ext = (sum(self.external_energy(self.positions[i]) for i in range(self.n))
               if self.n else 0.0)
        ext += (sum(self.external_energy(self.substrate_positions[i]) for i in range(self.n_substrate))
                if self.n_substrate else 0.0)
        return self.calculator.total_energy(self) + ext

    # -- generic path: correct for ANY calculator (used automatically when
    #    calculator.supports_incremental is False) --------------------------
    def generic_displace_dU(self, i: int, new_pos: np.ndarray) -> float:
        """
        Compute the energy change of displacing a mobile particle.

        Via clone-mutate-recompute: correct for any calculator, regardless
        of whether it supports the fast incremental path.

        Parameters
        ----------
        i : int
            Index of the mobile particle (into `self.positions`) to
            displace.
        new_pos : numpy.ndarray
            The (3,) Cartesian trial position.

        Returns
        -------
        float
            The total energy change (calculator + external field) of
            moving particle `i` to `new_pos`. ``numpy.inf`` if `new_pos`
            is in a forbidden region (`external_energy` returns
            non-finite there), without evaluating the calculator at all.
        """
        ext_old = self.external_energy(self.positions[i])
        ext_new = self.external_energy(new_pos)
        if not np.isfinite(ext_new):
            return np.inf
        u0 = self.calculator.total_energy(self)
        saved = self.positions[i].copy()
        self.positions[i] = new_pos
        u1 = self.calculator.total_energy(self)
        self.positions[i] = saved
        return (u1 - u0) + (ext_new - ext_old)

    def generic_displace_substrate_dU(self, i: int, new_pos: np.ndarray) -> float:
        """
        Compute the energy change of displacing a substrate atom.

        Like `generic_displace_dU`, but for the i-th substrate atom.

        Parameters
        ----------
        i : int
            Index of the substrate atom (into `self.substrate_positions`)
            to displace.
        new_pos : numpy.ndarray
            The (3,) Cartesian trial position.

        Returns
        -------
        float
            The total energy change (calculator + external field) of
            moving substrate atom `i` to `new_pos`. ``numpy.inf`` if
            `new_pos` is in a forbidden region.
        """
        ext_old = self.external_energy(self.substrate_positions[i])
        ext_new = self.external_energy(new_pos)
        if not np.isfinite(ext_new):
            return np.inf
        u0 = self.calculator.total_energy(self)
        saved = self.substrate_positions[i].copy()
        self.substrate_positions[i] = new_pos
        u1 = self.calculator.total_energy(self)
        self.substrate_positions[i] = saved
        return (u1 - u0) + (ext_new - ext_old)

    def generic_insert_dU(self, pos: np.ndarray) -> float:
        """
        Compute the energy change of inserting a new mobile particle.

        Via clone-mutate-recompute: correct for any calculator.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian trial position for the new particle
            (species `self.exchange_species`).

        Returns
        -------
        float
            The total energy change (calculator + external field) of
            adding a particle at `pos`. ``numpy.inf`` if `pos` is in a
            forbidden region, without evaluating the calculator at all.
        """
        ext = self.external_energy(pos)
        if not np.isfinite(ext):
            return np.inf
        u0 = self.calculator.total_energy(self)
        self.positions = np.vstack([self.positions, pos])
        self.species = np.append(self.species, self.exchange_species)
        u1 = self.calculator.total_energy(self)
        self.positions = self.positions[:-1]
        self.species = self.species[:-1]
        return (u1 - u0) + ext

    def generic_delete_dU(self, i: int) -> float:
        """
        Compute the energy change of deleting a mobile particle.

        Via clone-mutate-recompute: correct for any calculator.

        Parameters
        ----------
        i : int
            Index of the mobile particle (into `self.positions`) to
            remove.

        Returns
        -------
        float
            The total energy change (calculator + external field) of
            removing particle `i`.
        """
        pos = self.positions[i].copy()
        sp = self.species[i]
        ext = self.external_energy(pos)
        u0 = self.calculator.total_energy(self)
        self.positions = np.delete(self.positions, i, axis=0)
        self.species = np.delete(self.species, i)
        u1 = self.calculator.total_energy(self)
        self.positions = np.insert(self.positions, i, pos, axis=0)
        self.species = np.insert(self.species, i, sp)
        return (u1 - u0) - ext


# =============================================================================
# BulkSystem: bulk, fully-periodic cube
# =============================================================================
class BulkSystem(ParticleSystem):
    """Bulk, fully-periodic cubic box."""

    def __init__(self, box_length: float, calculator: Calculator,
                 positions: Optional[np.ndarray] = None,
                 substrate_positions: Optional[np.ndarray] = None,
                 substrate_species: Optional[np.ndarray] = None,
                 mass_of_species: Optional[dict] = None):
        """
        Construct a bulk, fully-periodic cubic system.

        Parameters
        ----------
        box_length : float
            The cube's edge length, L (periodic along all three axes).
        calculator : Calculator
            The energy-evaluation backend.
        positions : numpy.ndarray, optional
            Initial mobile particle positions, shape (N, 3). Default is
            None (no mobile particles).
        substrate_positions : numpy.ndarray, optional
            Positions of displaceable substrate atoms, shape (K, 3).
            Default is None (no substrate atoms).
        substrate_species : numpy.ndarray, optional
            Species label for each substrate atom, shape (K,). Default is
            None (empty).
        mass_of_species : dict, optional
            Mapping from species label to mass (in atomic mass units), for
            HMC. Default is None (empty).
        """
        super().__init__(calculator, exchange_species='X', positions=positions,
                          substrate_positions=substrate_positions,
                          substrate_species=substrate_species,
                          mass_of_species=mass_of_species)
        self.L = float(box_length)

    def _min_image(self, dr: np.ndarray) -> np.ndarray:
        """
        Apply the minimum-image convention along all three axes.

        Parameters
        ----------
        dr : numpy.ndarray
            One or more displacement vectors, shape (..., 3).

        Returns
        -------
        numpy.ndarray
            `dr`, wrapped into (-L/2, L/2] along each axis.
        """
        return dr - self.L * np.round(dr / self.L)

    def _wrap(self, pos: np.ndarray) -> np.ndarray:
        """
        Wrap a position into the primary cell [0, L).

        Parameters
        ----------
        pos : numpy.ndarray
            A (3,) Cartesian position, or an array of them.

        Returns
        -------
        numpy.ndarray
            `pos`, wrapped into [0, L) along each axis.
        """
        return pos % self.L

    @property
    def min_image_radius(self) -> float:
        """float: Half the box length, L/2."""
        return 0.5 * self.L

    @property
    def volume(self) -> float:
        """
        float: The cube's volume, L^3.
        """
        return self.L ** 3

    @property
    def cell(self) -> Tuple[float, float, float]:
        """
        tuple of float: (L, L, L), the cube's edge lengths.
        """
        return (self.L, self.L, self.L)

    @property
    def pbc(self) -> Tuple[bool, bool, bool]:
        """
        tuple of bool: ``(True, True, True)`` -- periodic along all three
        axes.
        """
        return (True, True, True)

    def random_insertion_position(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a uniformly-distributed random position in the box.

        Parameters
        ----------
        rng : numpy.random.Generator
            The random number generator to draw from.

        Returns
        -------
        numpy.ndarray
            A (3,) Cartesian position, uniform over [0, L)^3.
        """
        return rng.random(3) * self.L

    def apply_move(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th mobile particle.

        Parameters
        ----------
        i : int
            Index of the mobile particle to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to; wrapped into the
            primary cell before being stored.

        Returns
        -------
        None
        """
        self.positions[i] = self._wrap(new_pos)

    def apply_move_substrate(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th substrate atom.

        Parameters
        ----------
        i : int
            Index of the substrate atom to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to; wrapped into the
            primary cell before being stored.

        Returns
        -------
        None
        """
        self.substrate_positions[i] = self._wrap(new_pos)

    def apply_insert(self, pos: np.ndarray) -> None:
        """
        Commit the insertion of a new mobile particle.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position of the new particle; wrapped into
            the primary cell before being stored. Its species is
            `self.exchange_species`.

        Returns
        -------
        None
        """
        self.positions = np.vstack([self.positions, self._wrap(pos)])
        self.species = np.append(self.species, self.exchange_species)


# =============================================================================
# SlabSystem: surface-adsorption geometry, with an optional rigid substrate
# (e.g. a truncated Fe slab) for use with EAM/MLIP calculators
# =============================================================================
class SlabSystem(ParticleSystem):
    """Periodic in x, y only; particles live between z=0 and z=Lz.

    `wall_epsilon > 0` adds a continuum 9-3 wall -- use this for a fast
    LJ-only demo with no explicit substrate atoms. For a real substrate
    (e.g. Fe or Au), instead pass `frozen_positions`/`frozen_species`
    (rigid) and/or `substrate_positions`/`substrate_species` (displaceable
    -- what a reconstructing surface needs) and leave `wall_epsilon = 0`:
    the substrate's own interaction with the mobile species, through
    whatever Calculator you plug in, supplies the attraction/repulsion
    instead of an artificial continuum field. Don't use both at once --
    that would double-count the substrate.

    `insertion_z_range=(z_lo, z_hi)` restricts random insertion attempts to
    a sub-slab (and correspondingly narrows the `volume` used in the
    insertion/deletion acceptance formulas to match, and restricting
    deletion to the particles inside that sub-slab, see
    `exchange_zone_indices` -- together these keep the ensemble exact) instead of sampling z uniformly across the whole box. This
    matters whenever most of the box is somewhere insertion attempts would
    essentially never be accepted anyway -- e.g. deep inside a solid
    substrate, or in a large vacuum region far from a real (rather than
    continuum-wall) surface -- since without it, GCMC would waste nearly
    every insertion attempt on that dead volume. Defaults to the full
    (0, Lz) range, unchanged from previous behavior.
    """

    def __init__(self, Lx: float, Ly: float, Lz: float, calculator: Calculator,
                 exchange_species: str = 'X', wall_epsilon: float = 0.0,
                 wall_sigma: float = 1.0, wall_cut: Optional[float] = None,
                 frozen_positions: Optional[np.ndarray] = None,
                 frozen_species: Optional[np.ndarray] = None,
                 substrate_positions: Optional[np.ndarray] = None,
                 substrate_species: Optional[np.ndarray] = None,
                 positions: Optional[np.ndarray] = None,
                 species: Optional[np.ndarray] = None,
                 insertion_z_range: Optional[Tuple[float, float]] = None,
                 mass_of_species: Optional[dict] = None,
                 cell_xy: Optional[np.ndarray] = None):
        """
        Construct a surface-adsorption slab system.

        Parameters
        ----------
        Lx, Ly, Lz : float
            The cell dimensions. Periodic along x and y; particles live
            between z=0 and z=Lz (open, not periodic, along z). Pass
            ``Lx = Ly = None`` together with `cell_xy` for a non-orthogonal
            (e.g. hexagonal) surface cell.
        calculator : Calculator
            The energy-evaluation backend.
        exchange_species : str, optional
            The species label for newly inserted mobile particles.
            Default is ``'X'``.
        wall_epsilon : float, optional
            Well depth of an optional continuum 9-3 wall at z=0 (a fast
            LJ-only stand-in for an explicit substrate). Default is 0.0
            (no wall). Must be 0 if `frozen_positions` or
            `substrate_positions` are given (see class docstring).
        wall_sigma : float, optional
            Length scale of the continuum wall. Default is 1.0.
        wall_cut : float, optional
            Cutoff distance beyond which the continuum wall's energy is
            zero. Default is None, which uses ``3.0 * wall_sigma``.
        frozen_positions : numpy.ndarray, optional
            Positions of fully rigid substrate atoms, shape (M, 3).
            Default is None (none).
        frozen_species : numpy.ndarray, optional
            Species label for each frozen atom, shape (M,). Default is
            None (empty).
        substrate_positions : numpy.ndarray, optional
            Positions of displaceable substrate atoms, shape (K, 3).
            Default is None (none).
        substrate_species : numpy.ndarray, optional
            Species label for each substrate atom, shape (K,). Default is
            None (empty).
        positions : numpy.ndarray, optional
            Initial mobile particle positions, shape (N, 3). Default is
            None (none).
        species : numpy.ndarray, optional
            Species label for each mobile particle, shape (N,). Default is
            None, which fills every entry with `exchange_species`.
        insertion_z_range : tuple of float, optional
            (z_lo, z_hi) sub-range of [0, Lz] to restrict random insertion
            attempts (and the `volume` used in the insertion/deletion
            acceptance formulas) to. Default is None, which uses the full
            (0, Lz) range.
        mass_of_species : dict, optional
            Mapping from species label to mass (in atomic mass units), for
            HMC. Default is None (empty).
        cell_xy : array_like, optional
            2x2 matrix whose rows are the in-plane lattice vectors a1, a2
            (z components zero), e.g. a hexagonal fcc(111) supercell. When
            given, `Lx` and `Ly` must be None. Default is None (the
            rectangular Lx x Ly cell).

        Raises
        ------
        ValueError
            If `wall_epsilon` > 0 and either `frozen_positions` or
            `substrate_positions` is also given (double-counts the
            substrate), or if `insertion_z_range` doesn't satisfy
            ``0 <= z_lo < z_hi <= Lz``.
        """
        super().__init__(calculator, exchange_species, positions, species,
                          frozen_positions, frozen_species,
                          substrate_positions, substrate_species, mass_of_species)
        if cell_xy is None:
            if Lx is None or Ly is None:
                raise ValueError("give Lx and Ly, or cell_xy (with Lx = Ly = None)")
            self._A = np.diag([float(Lx), float(Ly)])
            self._orthogonal = True
        else:
            if Lx is not None or Ly is not None:
                raise ValueError("with cell_xy, pass Lx = Ly = None (the cell is fully "
                                 "given by cell_xy)")
            self._A = np.array(cell_xy, dtype=float).reshape(2, 2)
            self._orthogonal = bool(abs(self._A[0, 1]) < 1e-12 and abs(self._A[1, 0]) < 1e-12)
        self._Ainv = np.linalg.inv(self._A)
        a1, a2 = self._A
        l_min = min(np.linalg.norm(i * a1 + j * a2)
                    for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0))
        self._half_lmin2 = (0.5 * l_min) ** 2
        self._SHIFTS = np.array([[i, j] for i in (-1.0, 0.0, 1.0) for j in (-1.0, 0.0, 1.0)])
        # Lx, Ly: lattice-vector lengths (the cell edges when orthogonal)
        self.Lx, self.Ly = (float(np.linalg.norm(self._A[0])), float(np.linalg.norm(self._A[1])))
        self.Lz = float(Lz)
        self.wall_epsilon = wall_epsilon
        self.wall_sigma = wall_sigma
        self.wall_cut = wall_cut if wall_cut is not None else 3.0 * wall_sigma
        if wall_epsilon > 0 and (self.frozen_positions.shape[0] > 0
                                  or self.substrate_positions.shape[0] > 0):
            raise ValueError("Don't combine a continuum wall with an explicit "
                              "substrate (rigid or displaceable) -- pick one "
                              "(see class docstring).")
        self.insertion_z_range = (insertion_z_range if insertion_z_range is not None
                                   else (0.0, self.Lz))
        z_lo, z_hi = self.insertion_z_range
        if not (0.0 <= z_lo < z_hi <= self.Lz):
            raise ValueError(f"insertion_z_range={self.insertion_z_range} must satisfy "
                              f"0 <= z_lo < z_hi <= Lz={self.Lz}")

    def _min_image(self, dr: np.ndarray) -> np.ndarray:
        """
        Apply the minimum-image convention along x and y only.

        Parameters
        ----------
        dr : numpy.ndarray
            One or more displacement vectors, shape (..., 3).

        Returns
        -------
        numpy.ndarray
            `dr`, wrapped into (-Lx/2, Lx/2] and (-Ly/2, Ly/2] along x and
            y; the z component is left untouched, since z is not periodic.
        """
        dr = np.array(dr, dtype=float, copy=True)
        if self._orthogonal:
            dr[..., 0] -= self.Lx * np.round(dr[..., 0] / self.Lx)
            dr[..., 1] -= self.Ly * np.round(dr[..., 1] / self.Ly)
            return dr
        # non-orthogonal: rounding fractional coordinates is not always the
        # shortest image (e.g. 60-degree hexagonal cells), so also try the 8
        # neighbouring images and keep the shortest
        frac = dr[..., :2] @ self._Ainv
        frac -= np.round(frac)
        best = frac @ self._A
        best_r2 = np.einsum('...i,...i->...', best, best)
        # a vector shorter than half the shortest lattice vector is already the
        # minimum image (any other image is at least |L_min| - |d| > |d| long)
        unsure = best_r2 > self._half_lmin2
        if np.any(unsure):
            cand = (frac[unsure][:, None, :] + self._SHIFTS) @ self._A  # (M, 9, 2)
            c_r2 = np.einsum('mki,mki->mk', cand, cand)
            best[unsure] = cand[np.arange(cand.shape[0]), np.argmin(c_r2, axis=1)]
        dr[..., :2] = best
        return dr

    def _wrap(self, pos: np.ndarray) -> np.ndarray:
        """
        Wrap the x and y components of a position into the primary cell.

        Parameters
        ----------
        pos : numpy.ndarray
            A (3,) Cartesian position.

        Returns
        -------
        numpy.ndarray
            `pos`, with x wrapped into [0, Lx) and y into [0, Ly); z is
            intentionally left unwrapped -- see `external_energy`.
        """
        wrapped = np.array(pos, dtype=float, copy=True)
        if self._orthogonal:
            wrapped[0] %= self.Lx
            wrapped[1] %= self.Ly
        else:
            wrapped[:2] = ((wrapped[:2] @ self._Ainv) % 1.0) @ self._A
        return wrapped  # z is intentionally left unwrapped -- see external_energy

    def md_z_bounds(self) -> Optional[Tuple[float, float]]:
        """
        Hard z-walls of `external_energy`, for MD reflection.

        Returns
        -------
        tuple of float
            ``(0, Lz)``.
        """
        return (0.0, self.Lz)

    def exchange_zone_indices(self) -> np.ndarray:
        """
        Indices of the mobile particles with ``z_lo <= z < z_hi``.

        The same region `random_insertion_position` samples (x, y are
        always inside, being periodic). Particles that have diffused out of
        `insertion_z_range` stay in the system but can't be deleted until
        they diffuse back in.

        Returns
        -------
        numpy.ndarray
            Integer indices into `self.positions`.
        """
        z_lo, z_hi = self.insertion_z_range
        z = self.positions[:, 2]
        return np.flatnonzero((z >= z_lo) & (z < z_hi))

    @property
    def volume(self) -> float:
        """
        float: The accessible volume, ``area * (z_hi - z_lo)`` over
        `insertion_z_range`.
        """
        z_lo, z_hi = self.insertion_z_range
        return self.area * (z_hi - z_lo)

    @property
    def min_image_radius(self) -> float:
        """
        float: Half the smallest in-plane lattice-plane spacing,
        ``min(area/|a1|, area/|a2|) / 2`` (``min(Lx, Ly)/2`` if orthogonal).
        """
        return 0.5 * self.area / max(np.linalg.norm(self._A[0]), np.linalg.norm(self._A[1]))

    @property
    def area(self) -> float:
        """
        float: The surface-cell area, ``|det(cell_xy)|`` (``Lx * Ly`` if
        orthogonal).
        """
        return float(abs(np.linalg.det(self._A)))

    @property
    def cell(self) -> Tuple[float, float, float]:
        """
        tuple or list: (Lx, Ly, Lz) for an orthogonal cell, else the 3x3
        cell matrix (rows a1, a2, (0, 0, Lz)).
        """
        if self._orthogonal:
            return (self.Lx, self.Ly, self.Lz)
        a1, a2 = self._A
        return [[a1[0], a1[1], 0.0], [a2[0], a2[1], 0.0], [0.0, 0.0, self.Lz]]

    @property
    def pbc(self) -> Tuple[bool, bool, bool]:
        """
        tuple of bool: ``(True, True, False)`` -- periodic in x, y; open
        in z.
        """
        return (True, True, False)

    def external_energy(self, pos: np.ndarray) -> float:
        """
        Evaluate the hard z-boundary and optional continuum wall.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position to evaluate.

        Returns
        -------
        float
            ``numpy.inf`` if `pos`'s z coordinate is outside (0, Lz].
            Otherwise, the continuum 9-3 wall's energy at that z (0.0 if
            `wall_epsilon` is 0 or z is beyond `wall_cut`).
        """
        z = pos[2]
        if z <= 0.0 or z > self.Lz:
            return np.inf
        if self.wall_epsilon == 0.0 or z > self.wall_cut:
            return 0.0
        sr3 = (self.wall_sigma / z) ** 3
        sr9 = sr3 ** 3
        return self.wall_epsilon * ((2.0 / 15.0) * sr9 - sr3)

    def random_insertion_position(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a uniformly-distributed random position within
        `insertion_z_range`.

        Parameters
        ----------
        rng : numpy.random.Generator
            The random number generator to draw from.

        Returns
        -------
        numpy.ndarray
            A (3,) Cartesian position, uniform over the surface cell
            (``[0, Lx) x [0, Ly)`` if orthogonal) times ``[z_lo, z_hi)``.
        """
        z_lo, z_hi = self.insertion_z_range
        xy = np.array([rng.random(), rng.random()]) @ self._A
        return np.array([xy[0], xy[1], z_lo + rng.random() * (z_hi - z_lo)])

    def apply_move(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th mobile particle.

        Parameters
        ----------
        i : int
            Index of the mobile particle to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to; x, y wrapped into
            the primary cell, z left as given.

        Returns
        -------
        None
        """
        self.positions[i] = self._wrap(new_pos)

    def apply_move_substrate(self, i: int, new_pos: np.ndarray) -> None:
        """
        Commit a displacement of the i-th substrate atom.

        Parameters
        ----------
        i : int
            Index of the substrate atom to move.
        new_pos : numpy.ndarray
            The (3,) Cartesian position to move it to; x, y wrapped into
            the primary cell, z left as given.

        Returns
        -------
        None
        """
        self.substrate_positions[i] = self._wrap(new_pos)

    def apply_insert(self, pos: np.ndarray) -> None:
        """
        Commit the insertion of a new mobile particle.

        Parameters
        ----------
        pos : numpy.ndarray
            The (3,) Cartesian position of the new particle; x, y wrapped
            into the primary cell. Its species is `self.exchange_species`.

        Returns
        -------
        None
        """
        self.positions = np.vstack([self.positions, self._wrap(pos)])
        self.species = np.append(self.species, self.exchange_species)

    def coverage(self) -> float:
        """
        Compute the mobile-particle areal coverage.

        Returns
        -------
        float
            The number of mobile particles per unit area, ``n / (Lx *
            Ly)``.
        """
        return self.n / self.area


#: Historical name of `BulkSystem` (kept for backward compatibility).
LJSystem = BulkSystem
#: Historical name of `SlabSystem` (kept for backward compatibility).
SlabLJSystem = SlabSystem
