"""
Structural observables recorded with every REGC sample, next to U and N.

MBAR reweights any function of the configuration with the same weights as
N (docs/REGC_algorithm.md, Sec. 5), so an observable recorded at the
sampled (T, mu) grid points is available at every (T, mu) or (T, p) of
the phase diagram. Pass a list of observables to `ReplicaExchangeGCMC`
(``observables=``); each one is evaluated on the replica's configuration
whenever a sample is recorded (in the worker processes of a parallel run,
so they must be picklable), stored under ``samples[key]['obs:<name>']``,
written to ``samples.npz`` by `regc.io.save_samples`, and analysed with
`regc.analysis.MBARSurfaceAnalysis` (`observable_mean`, `observable_map`,
`mean_coordination`, `coordination_distribution`, `radial_distribution`).

Every observable returns a fixed-length float vector per sample. Ratios
(the mean coordination of a set of atoms whose size fluctuates, the RDF
per centre atom) are recorded as numerator and denominator separately and
divided only after reweighting, i.e. <bonds>/<centres>: averaging
per-sample ratios would weight a configuration with one centre atom as
much as one with fifty, and is undefined when there are none.

Atom selections (`center`, `neighbor`, `a`, `b`) are strings:

* ``'mobile'``: the exchanged (GCMC) particles;
* ``'substrate'``: the substrate atoms, frozen and displaceable;
* ``'all'``: every atom;
* a species label, e.g. ``'H'`` or ``'Si'``: those atoms in any group;
* ``'mobile:H'``, ``'substrate:Si'``: a species within a group.

``z_range=(z_min, z_max)`` further restricts the centre atoms to a slab of
absolute z (Angstrom; in a slab built by
`regc.structures.slab_geometry_from_atoms` the top frozen layer is at z = 0
and ``geometry.top_z`` is the height of the topmost slab atom), e.g. the 3 A
"surface region" of [ZSG] PRL 128, 246101 (2022). Distances use the
system's minimum-image convention (periodic in x, y for a slab, in x, y, z
for bulk, none for a cluster), so cutoffs must stay below
``system.min_image_radius``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Group codes of the atoms of `ParticleSystem.full_configuration`, in its order.
GROUP_FROZEN, GROUP_SUBSTRATE, GROUP_MOBILE = 0, 1, 2
_GROUPS = {"frozen": (GROUP_FROZEN,), "substrate": (GROUP_FROZEN, GROUP_SUBSTRATE),
           "mobile": (GROUP_MOBILE,), "all": (GROUP_FROZEN, GROUP_SUBSTRATE, GROUP_MOBILE)}
#: Sample keys of the engine; observable names may not clash with them.
RESERVED_NAMES = {"U", "N", "N_ads", "N_s"}


def configuration_groups(system) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Positions, species and group code of every atom of a system.

    Parameters
    ----------
    system : ParticleSystem
        The system.

    Returns
    -------
    tuple of numpy.ndarray
        ``(positions, species, groups)`` in `full_configuration` order
        (frozen, displaceable substrate, mobile); groups are
        `GROUP_FROZEN`, `GROUP_SUBSTRATE` or `GROUP_MOBILE`.
    """
    positions, species = system.full_configuration()
    n_fr = system.frozen_positions.shape[0]
    n_sub = system.substrate_positions.shape[0]
    groups = np.concatenate([np.full(n_fr, GROUP_FROZEN), np.full(n_sub, GROUP_SUBSTRATE),
                             np.full(positions.shape[0] - n_fr - n_sub, GROUP_MOBILE)])
    return np.asarray(positions, dtype=float).reshape(-1, 3), np.asarray(species).astype(str), groups


def select(selector: str, species: np.ndarray, groups: np.ndarray,
           positions: Optional[np.ndarray] = None,
           z_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Indices of the atoms an atom-selection string picks.

    Parameters
    ----------
    selector : str
        ``'mobile'``, ``'substrate'``, ``'frozen'``, ``'all'``, a species
        label, or ``'<group>:<species>'`` (see the module docstring).
    species, groups : numpy.ndarray
        Per-atom species labels and group codes (`configuration_groups`).
    positions : numpy.ndarray, optional
        Per-atom positions; needed with `z_range`. Default None.
    z_range : tuple of float, optional
        Keep only atoms with ``z_min <= z < z_max``. Default None.

    Returns
    -------
    numpy.ndarray
        Integer indices into the full configuration.
    """
    group, _, sp = str(selector).partition(":")
    if group in _GROUPS:
        mask = np.isin(groups, _GROUPS[group])
        if sp:
            mask &= species == sp
    else:  # a bare species label
        mask = species == selector
    if z_range is not None:
        z_lo, z_hi = z_range
        z = positions[:, 2]
        mask &= (z >= (-np.inf if z_lo is None else z_lo)) & (z < (np.inf if z_hi is None else z_hi))
    return np.flatnonzero(mask)


def _check_selector(selector: str) -> str:
    """
    Validate an atom-selection string.

    Parameters
    ----------
    selector : str
        The selection.

    Returns
    -------
    str
        `selector`, unchanged.

    Raises
    ------
    ValueError
        If it is empty or names an unknown group.
    """
    selector = str(selector)
    group, colon, sp = selector.partition(":")
    if not selector or (colon and (group not in _GROUPS or not sp)):
        raise ValueError(f"bad atom selection {selector!r}: use 'mobile', 'substrate', 'all', a "
                         f"species label, or '<group>:<species>' with group in {sorted(_GROUPS)}")
    return selector


def pair_distances(system, idx_a: np.ndarray, idx_b: np.ndarray, positions: np.ndarray,
                   r_max: float) -> np.ndarray:
    """
    Minimum-image distances between two atom sets (self-pairs excluded).

    Parameters
    ----------
    system : ParticleSystem
        Supplies the minimum-image convention.
    idx_a, idx_b : numpy.ndarray
        Indices into `positions`.
    positions : numpy.ndarray
        All positions, shape (n, 3).
    r_max : float
        Largest distance needed; must not exceed the system's
        minimum-image radius.

    Returns
    -------
    numpy.ndarray
        Shape (len(idx_a), len(idx_b)); ``inf`` where an atom meets itself.

    Raises
    ------
    ValueError
        If `r_max` exceeds ``system.min_image_radius`` (images beyond the
        nearest one would be missed).
    """
    radius = system.min_image_radius
    if r_max > radius + 1e-12:
        raise ValueError(f"observable cutoff {r_max} A exceeds the cell's minimum-image radius "
                         f"{radius:.3f} A: nearer periodic images would be missed")
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.zeros((len(idx_a), len(idx_b)))
    dr = system._min_image(positions[idx_b][None, :, :] - positions[idx_a][:, None, :])
    r = np.sqrt(np.einsum("ijk,ijk->ij", dr, dr))
    r[idx_a[:, None] == idx_b[None, :]] = np.inf
    return r


class Observable:
    """
    Base class: a named, fixed-length vector function of a configuration.

    Subclasses set `name`, implement `labels` (one per component) and
    `__call__` (system -> numpy array of that length), and describe
    themselves in `metadata` (JSON-able; stored in ``samples.npz`` so the
    analysis can interpret the components without the original objects).
    """

    name: str = "observable"

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            One label per component.

        Raises
        ------
        NotImplementedError
            On the base class.
        """
        raise NotImplementedError

    @property
    def size(self) -> int:
        """int: Number of components."""
        return len(self.labels())

    def metadata(self) -> dict:
        """
        JSON-able description (type, parameters, component labels).

        Returns
        -------
        dict
            At least ``'type'`` and ``'labels'``.
        """
        return {"type": type(self).__name__, "labels": self.labels()}

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            Float vector of length `size`.

        Raises
        ------
        NotImplementedError
            On the base class.
        """
        raise NotImplementedError


class CoordinationNumber(Observable):
    """
    Mean coordination number of a set of atoms.

    Records ``[bonds, centers]``: the number of (centre, neighbour) pairs
    closer than `r_cut` and the number of centre atoms. The reweighted
    mean coordination is <bonds> / <centers>
    (`MBARSurfaceAnalysis.mean_coordination`).

    Parameters
    ----------
    name : str
        Observable name (sample key ``'obs:<name>'``).
    r_cut : float
        Bond cutoff, in Angstrom (e.g. the first minimum of the
        corresponding RDF; [ZSG] use 2.0 A for D-Si).
    center : str, optional
        Atoms whose coordination is counted. Default ``'mobile'``.
    neighbor : str, optional
        Atoms counted as neighbours. Default ``'substrate'``.
    z_range : tuple of float, optional
        Restrict the centre atoms to ``z_min <= z < z_max``. Default None.
    """

    def __init__(self, name: str, r_cut: float, center: str = "mobile",
                 neighbor: str = "substrate", z_range: Optional[Tuple[float, float]] = None):
        """
        Store the definition.

        Parameters
        ----------
        name, r_cut, center, neighbor, z_range
            See the class docstring.
        """
        self.name = str(name)
        self.r_cut = float(r_cut)
        self.center = _check_selector(center)
        self.neighbor = _check_selector(neighbor)
        self.z_range = None if z_range is None else tuple(z_range)

    def labels(self) -> List[str]:
        """
        Component names.

        Returns
        -------
        list of str
            ``['bonds', 'centers']``.
        """
        return ["bonds", "centers"]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels and parameters.
        """
        return dict(super().metadata(), r_cut=self.r_cut, center=self.center,
                    neighbor=self.neighbor, z_range=self.z_range)

    def coordination(self, system) -> np.ndarray:
        """
        Coordination number of every centre atom.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Integer array, one entry per centre atom.
        """
        pos, sp, grp = configuration_groups(system)
        a = select(self.center, sp, grp, pos, self.z_range)
        b = select(self.neighbor, sp, grp)
        r = pair_distances(system, a, b, pos, self.r_cut)
        return np.count_nonzero(r < self.r_cut, axis=1)

    def __call__(self, system) -> np.ndarray:
        """
        ``[bonds, centers]`` of a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Shape (2,).
        """
        cn = self.coordination(system)
        return np.array([cn.sum(), cn.size], dtype=float)


class CoordinationHistogram(CoordinationNumber):
    """
    Distribution of the coordination numbers of a set of atoms.

    Records how many centre atoms have coordination 0, 1, ..., `max_cn`
    (the last bin collects ``>= max_cn``): [ZSG]'s coordination histogram
    H_coord of the top-layer Si atoms, a structural phase descriptor.
    `MBARSurfaceAnalysis.coordination_distribution` normalises the
    reweighted counts to a probability per centre atom.

    Parameters
    ----------
    name : str
        Observable name.
    r_cut : float
        Bond cutoff, in Angstrom.
    max_cn : int, optional
        Largest coordination with its own bin. Default 12.
    center, neighbor, z_range
        As in `CoordinationNumber` (defaults ``'substrate'`` and ``'all'``:
        the substrate atoms' bonds to anything).
    """

    def __init__(self, name: str, r_cut: float, max_cn: int = 12, center: str = "substrate",
                 neighbor: str = "all", z_range: Optional[Tuple[float, float]] = None):
        """
        Store the definition.

        Parameters
        ----------
        name, r_cut, max_cn, center, neighbor, z_range
            See the class docstring.
        """
        super().__init__(name, r_cut, center, neighbor, z_range)
        self.max_cn = int(max_cn)

    def labels(self) -> List[str]:
        """
        Component names.

        Returns
        -------
        list of str
            ``['cn0', 'cn1', ..., 'cn<max_cn>']``.
        """
        return [f"cn{k}" for k in range(self.max_cn + 1)]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels and parameters.
        """
        return dict(super().metadata(), max_cn=self.max_cn)

    def __call__(self, system) -> np.ndarray:
        """
        Coordination counts of a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Shape (max_cn + 1,).
        """
        cn = np.minimum(self.coordination(system), self.max_cn)
        return np.bincount(cn, minlength=self.max_cn + 1).astype(float)


class RadialDistribution(Observable):
    """
    Pair-distance histogram between two atom sets (for an RDF).

    Records the number of (a, b) pairs per distance bin (self-pairs
    excluded), followed by ``n_a``, ``n_b`` and the system volume.
    `MBARSurfaceAnalysis.radial_distribution` turns the reweighted
    histogram into

    * ``'per_center'``: n(r) = <H(r)> / (<n_a> dr), neighbours of a centre
      per unit distance -- its integral up to r_c is the mean coordination
      (the natural form for surfaces and clusters, where no uniform
      reference density exists);
    * ``'density'``: n(r) / (4 pi r^2), the local density of b around a;
    * ``'g'``: that density over <n_b> / V, the bulk-normalised g(r)
      (meaningful for bulk systems only: V includes a slab's vacuum).

    Parameters
    ----------
    name : str
        Observable name.
    r_max : float
        Largest distance, in Angstrom (below the minimum-image radius).
    n_bins : int, optional
        Number of distance bins. Default 100.
    a, b : str, optional
        The two atom sets. Default ``'mobile'`` and ``'substrate'``.
    z_range : tuple of float, optional
        Restrict the a atoms to ``z_min <= z < z_max``. Default None.
    """

    def __init__(self, name: str, r_max: float, n_bins: int = 100, a: str = "mobile",
                 b: str = "substrate", z_range: Optional[Tuple[float, float]] = None):
        """
        Store the definition.

        Parameters
        ----------
        name, r_max, n_bins, a, b, z_range
            See the class docstring.
        """
        self.name = str(name)
        self.r_max = float(r_max)
        self.n_bins = int(n_bins)
        self.a = _check_selector(a)
        self.b = _check_selector(b)
        self.z_range = None if z_range is None else tuple(z_range)

    @property
    def edges(self) -> np.ndarray:
        """numpy.ndarray: The n_bins + 1 bin edges, in Angstrom."""
        return np.linspace(0.0, self.r_max, self.n_bins + 1)

    def labels(self) -> List[str]:
        """
        Component names.

        Returns
        -------
        list of str
            ``['h0', ..., 'h<n_bins-1>', 'n_a', 'n_b', 'volume']``.
        """
        return [f"h{k}" for k in range(self.n_bins)] + ["n_a", "n_b", "volume"]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels and parameters (incl. the bin edges).
        """
        return dict(super().metadata(), r_max=self.r_max, n_bins=self.n_bins, a=self.a, b=self.b,
                    z_range=self.z_range, edges=self.edges.tolist())

    def __call__(self, system) -> np.ndarray:
        """
        Histogram, n_a, n_b and volume of a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Shape (n_bins + 3,).
        """
        pos, sp, grp = configuration_groups(system)
        a = select(self.a, sp, grp, pos, self.z_range)
        b = select(self.b, sp, grp)
        r = pair_distances(system, a, b, pos, self.r_max)
        h, _ = np.histogram(r[np.isfinite(r)], bins=self.edges)
        try:
            volume = float(system.volume)
        except (AttributeError, NotImplementedError, TypeError):
            volume = np.nan
        return np.concatenate([h.astype(float), [len(a), len(b), volume]])


class FunctionObservable(Observable):
    """
    Wrap a user function ``f(system) -> float or array`` as an observable.

    The function must be picklable (a module-level function, not a lambda)
    to run in parallel workers, and must always return the same number of
    components.

    Parameters
    ----------
    name : str
        Observable name.
    func : callable
        ``func(system)`` returning a float or a 1-D array.
    labels : sequence of str, optional
        Component labels. Default None: ``['value']`` (one component).
    """

    def __init__(self, name: str, func: Callable, labels: Optional[Sequence[str]] = None):
        """
        Store the function.

        Parameters
        ----------
        name, func, labels
            See the class docstring.
        """
        self.name = str(name)
        self.func = func
        self._labels = list(labels) if labels is not None else ["value"]

    def labels(self) -> List[str]:
        """
        Component names.

        Returns
        -------
        list of str
            As given (default ``['value']``).
        """
        return list(self._labels)

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels and the function's qualified name.
        """
        return dict(super().metadata(), function=getattr(self.func, "__qualname__", repr(self.func)))

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate the wrapped function.

        Parameters
        ----------
        system : ParticleSystem
            The configuration.

        Returns
        -------
        numpy.ndarray
            Float vector of length `size`.

        Raises
        ------
        ValueError
            If the function returns the wrong number of components.
        """
        value = np.atleast_1d(np.asarray(self.func(system), dtype=float)).ravel()
        if value.size != self.size:
            raise ValueError(f"observable {self.name!r} returned {value.size} values, "
                             f"expected {self.size} ({self._labels})")
        return value


#: Name under which the engine records `SGCCounts` (reserved for it).
SGC_COUNTS_NAME = "sgc_counts"


class SGCCounts(Observable):
    """
    Counts of the swappable substrate atoms of each semi-grand-canonical species.

    Recorded automatically by `ReplicaExchangeGCMC` when the substrate is
    semi-grand-canonical (`ParticleSystem.sgc_species`), under the reserved
    name `SGC_COUNTS_NAME`. Its metadata carries the field ``dmu`` (one
    value per species, reference first and 0), which
    `regc.analysis.MBARSurfaceAnalysis` needs to put the
    -beta * sum_k dmu_k N_k term into the reduced potential: without it
    MBAR would reweight with the wrong ensemble.

    Parameters
    ----------
    species : sequence of str
        The semi-grand-canonical species, reference first.
    dmu : sequence of float
        mu_k - mu_ref per species (energy units of the run, eV for real
        potentials), the same for every grid point: the energy part of
        the field.
    log_ratio : sequence of float, optional
        Ideal-mixing part, ln(x_k / x_ref) of the bulk reservoir: the field
        at T is dmu + T * log_ratio. State-independent in units of kT, so
        the analysis does not need it; recorded for reference. Default
        None (zeros).
    """

    def __init__(self, species: Sequence[str], dmu: Sequence[float],
                 log_ratio: Optional[Sequence[float]] = None):
        """
        Set up the counter.

        Parameters
        ----------
        species : sequence of str
            The semi-grand-canonical species, reference first.
        dmu : sequence of float
            mu_k - mu_ref per species (energy part).
        log_ratio : sequence of float, optional
            ln(x_k / x_ref) per species. Default None (zeros).

        Raises
        ------
        ValueError
            If `dmu` does not have one value per species.
        """
        self.name = SGC_COUNTS_NAME
        self.species = [str(s) for s in species]
        self.dmu = [float(x) for x in dmu]
        if len(self.dmu) != len(self.species):
            raise ValueError(f"dmu needs one value per species {self.species}, got {self.dmu}")
        self.log_ratio = ([0.0] * len(self.species) if log_ratio is None
                          else [float(x) for x in log_ratio])

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            The species.
        """
        return list(self.species)

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels, species, the field ``dmu`` and, if not zero,
            ``log_ratio``.
        """
        meta = dict(super().metadata(), species=self.species, dmu=self.dmu)
        if any(self.log_ratio):
            meta["log_ratio"] = self.log_ratio
        return meta

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            `ParticleSystem.sgc_counts` as floats.
        """
        return np.asarray(system.sgc_counts(), dtype=float)


#: Reserved name under which the engine records the molecule counts.
MOLECULE_COUNTS_NAME = "molecules"
#: Reserved name under which the engine records the subsurface atom counts.
SUBSURFACE_COUNTS_NAME = "subsurface"


class SubsurfaceCounts(Observable):
    """
    Number of mobile atoms of each subsurface species inside the subsurface region.

    Recorded automatically by `ReplicaExchangeGCMC` when the slab has
    subsurface NCMC channels (`SlabSystem.set_subsurface`), under the reserved
    name `SUBSURFACE_COUNTS_NAME`: the atoms with z_min <= z < z_cut,
    whichever move put them there (subsurface insertions, or diffusion from
    the surface). Reweighted like N, it gives subsurface coverage maps next
    to the total adsorbed coverage.

    Parameters
    ----------
    species : sequence of str
        The subsurface species.
    z_min, z_cut : float
        The region (A).
    """

    def __init__(self, species: Sequence[str], z_min: float, z_cut: float):
        """
        Set up the counter.

        Parameters
        ----------
        species : sequence of str
            The subsurface species.
        z_min, z_cut : float
            The region (A).
        """
        self.name = SUBSURFACE_COUNTS_NAME
        self.species = [str(s) for s in species]
        self.z_min, self.z_cut = float(z_min), float(z_cut)

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            The species.
        """
        return list(self.species)

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels, species and the region.
        """
        return dict(super().metadata(), species=self.species, z_min=self.z_min, z_cut=self.z_cut)

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            One count per species, as floats.
        """
        z = system.positions[:, 2] if system.n else np.zeros(0)
        inside = (z >= self.z_min) & (z < self.z_cut)
        return np.array([np.count_nonzero(inside & (system.species == s)) for s in self.species],
                        dtype=float)


class MoleculeCounts(Observable):
    """
    Number of intact molecules of each exchanged molecule type.

    Recorded automatically by `ReplicaExchangeGCMC` when molecules are
    exchanged (`ParticleSystem.set_molecules`), under the reserved name
    `MOLECULE_COUNTS_NAME`. A molecule is a unit as the deletion move
    defines it (`regc.molecules.MoleculeMove.count_units`): atoms with the
    molecule's composition, all partners within the shell around the
    anchor -- here anywhere in the system, not only in the exchange zone.
    Comparing it with the element counts tells intact from dissociated
    adsorbates (CO against C + O).

    Parameters
    ----------
    molecules : sequence of MoleculeMove
        The exchanged molecules.
    """

    def __init__(self, molecules: Sequence):
        """
        Set up the counter.

        Parameters
        ----------
        molecules : sequence of MoleculeMove
            The exchanged molecules.
        """
        self.name = MOLECULE_COUNTS_NAME
        self.molecules = list(molecules)

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            The molecule names.
        """
        return [m.name for m in self.molecules]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels and every molecule's definition.
        """
        return dict(super().metadata(), molecules=[m.metadata() for m in self.molecules])

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            One count per molecule, as floats.
        """
        zone = np.ones(system.n, dtype=bool)
        return np.array([m.count_units(system.positions, system.species, zone,
                                       system._min_image, zone_only=False)
                         for m in self.molecules], dtype=float)


class LayerComposition(Observable):
    """
    Number of atoms of each species in each atomic layer of a slab.

    Layers are z-intervals between the midpoints of the slab's initial
    layer heights (`regc.structures.SlabGeometry.layer_z`); the lowest and
    the highest are open-ended, so an adatom counts in the top layer.
    Only slab atoms (frozen and displaceable substrate) are counted, never
    the exchanged gas. Recorded as counts -- surface fractions are ratios
    of reweighted counts, <N_Pt,top> / <N_top> -- in the order
    (layer, species), layer 0 the lowest.

    Parameters
    ----------
    name : str
        Observable name.
    species : sequence of str
        The species to count.
    layer_z : sequence of float
        z of every layer, ascending (the config runner fills it in from
        the slab geometry for ``layer_z: auto``).
    """

    def __init__(self, name: str, species: Sequence[str], layer_z: Sequence[float]):
        """
        Set up the layers.

        Parameters
        ----------
        name : str
            Observable name.
        species : sequence of str
            The species to count.
        layer_z : sequence of float
            z of every layer (any order).

        Raises
        ------
        ValueError
            Without species or layer heights.
        """
        self.name = name
        self.species = [str(s) for s in species]
        self.layer_z = [float(z) for z in np.sort(np.asarray(layer_z, dtype=float))]
        if not self.species or not self.layer_z:
            raise ValueError("layer_composition needs species and at least one layer height")
        mid = 0.5 * (np.asarray(self.layer_z[1:]) + np.asarray(self.layer_z[:-1]))
        self._edges = mid

    @property
    def n_layers(self) -> int:
        """int: Number of layers."""
        return len(self.layer_z)

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            ``'L<layer>:<species>'``, layer-major.
        """
        return [f"L{i}:{s}" for i in range(self.n_layers) for s in self.species]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels, species and layer heights.
        """
        return dict(super().metadata(), species=self.species, layer_z=self.layer_z)

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            The counts, length n_layers * n_species.
        """
        pos, sp, groups = configuration_groups(system)
        slab = groups != GROUP_MOBILE
        layer = np.searchsorted(self._edges, pos[slab, 2])
        sp = sp[slab]
        out = np.zeros((self.n_layers, len(self.species)))
        for j, s in enumerate(self.species):
            out[:, j] = np.bincount(layer[sp == s], minlength=self.n_layers)[:self.n_layers]
        return out.ravel()


class ShellComposition(LayerComposition):
    """
    Number of atoms of each species in the core and on the surface of a cluster.

    The cluster's atoms (frozen and displaceable substrate, never the
    exchanged gas) are split by their coordination among themselves,
    recomputed on every configuration: an atom with fewer than
    `cn_surface` neighbours closer than `r_cut` is a surface atom, the
    others are core atoms. Recorded as counts in the order (shell,
    species), shell 0 the core and shell 1 the surface -- the same layout
    as `LayerComposition`, the surface playing the top layer, so the
    analysis and the monitor treat both alike.

    Parameters
    ----------
    name : str
        Observable name.
    species : sequence of str
        The species to count.
    r_cut : float, optional
        Metal-metal bond cutoff in Angstrom, between the first and second
        neighbour shells. Default 3.3 (Au, Pt, Pd, Rh, Ir).
    cn_surface : int, optional
        Atoms with fewer neighbours are surface atoms. Default 10 (a
        bulk atom has 12, a (100) facet atom 8, a (111) facet atom 9).
    """

    SHELLS = ("core", "surface")

    def __init__(self, name: str, species: Sequence[str], r_cut: float = 3.3,
                 cn_surface: int = 10):
        """
        Store the definition.

        Parameters
        ----------
        name, species, r_cut, cn_surface
            See the class docstring.

        Raises
        ------
        ValueError
            Without species, or on a non-positive cutoff.
        """
        self.name = name
        self.species = [str(s) for s in species]
        self.r_cut = float(r_cut)
        self.cn_surface = int(cn_surface)
        if not self.species:
            raise ValueError("shell_composition needs species")
        if self.r_cut <= 0:
            raise ValueError(f"shell_composition r_cut must be positive, got {r_cut}")

    @property
    def n_layers(self) -> int:
        """int: Number of shells (2: core, surface)."""
        return len(self.SHELLS)

    def labels(self) -> List[str]:
        """
        Names of the components.

        Returns
        -------
        list of str
            ``'<shell>:<species>'``, shell-major (core first).
        """
        return [f"{sh}:{s}" for sh in self.SHELLS for s in self.species]

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Type, labels, species, shells, cutoff and surface threshold.
        """
        return {"type": type(self).__name__, "labels": self.labels(), "species": self.species,
                "shells": list(self.SHELLS), "r_cut": self.r_cut, "cn_surface": self.cn_surface}

    def __call__(self, system) -> np.ndarray:
        """
        Evaluate on a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The replica's current configuration.

        Returns
        -------
        numpy.ndarray
            The counts, length 2 * n_species.
        """
        pos, sp, groups = configuration_groups(system)
        metal = groups != GROUP_MOBILE
        p, sp = pos[metal], sp[metal]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        cn = np.count_nonzero(d < self.r_cut, axis=1) - 1      # minus the atom itself
        shell = (cn < self.cn_surface).astype(int)              # 0 core, 1 surface
        out = np.zeros((2, len(self.species)))
        for j, s in enumerate(self.species):
            out[:, j] = np.bincount(shell[sp == s], minlength=2)[:2]
        return out.ravel()


#: metadata types of the per-layer (or per-shell) composition observables
COMPOSITION_TYPES = ("LayerComposition", "ShellComposition")


def composition_layers(meta: dict) -> int:
    """
    Number of layers (or shells) of a composition observable, from its metadata.

    Parameters
    ----------
    meta : dict
        Metadata of a `LayerComposition` or `ShellComposition`.

    Returns
    -------
    int
        Its layer (shell) count; the last one is the top layer (surface).
    """
    return len(meta["layer_z"]) if "layer_z" in meta else len(meta["shells"])


def composition_fractions(observables_meta: dict, values: Dict[str, np.ndarray],
                          species: Optional[str] = None) -> Optional[dict]:
    """
    Per-sample substrate fractions of one species, for monitoring.

    Parameters
    ----------
    observables_meta : dict
        ``{name: metadata}`` of the recorded observables (needs an
        `SGCCounts` entry).
    values : dict
        ``{name: array}``: the recorded values of those observables, each
        shape (..., size) with the samples along the leading axes.
    species : str, optional
        The species whose fraction to give. Default: the first
        non-reference semi-grand-canonical species.

    Returns
    -------
    dict or None
        ``species`` (the one shown), ``sgc_species``, ``dmu``, ``swappable``
        (its fraction among the swappable substrate atoms, shape of the
        leading axes) and ``top`` (its fraction in the top layer of the
        first `LayerComposition` -- or on the surface of the first
        `ShellComposition` -- observable that counts it, or None); None
        without semi-grand-canonical counts.
    """
    sgc = next((n for n, m in observables_meta.items() if m.get("type") == "SGCCounts"
                and n in values), None)
    if sgc is None:
        return None
    meta = observables_meta[sgc]
    sgc_species = list(meta["species"])
    show = species or sgc_species[min(1, len(sgc_species) - 1)]
    c = np.asarray(values[sgc], dtype=float)
    total = c.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        swappable = np.where(total > 0, c[..., sgc_species.index(show)] / total, np.nan)
    top = None
    for n, m in observables_meta.items():
        if m.get("type") in COMPOSITION_TYPES and n in values and show in m["species"]:
            S, L = len(m["species"]), composition_layers(m)
            a = np.asarray(values[n], dtype=float)
            a = a.reshape(a.shape[:-1] + (L, S))[..., -1, :]
            t = a.sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                top = np.where(t > 0, a[..., m["species"].index(show)] / t, np.nan)
            break
    return {"species": show, "sgc_species": sgc_species, "dmu": list(meta["dmu"]),
            "swappable": swappable, "top": top}


def check_observables(observables: Optional[Sequence[Observable]]) -> List[Observable]:
    """
    Validate a list of observables (unique, non-reserved names).

    Parameters
    ----------
    observables : sequence of Observable, optional
        The observables (None: none).

    Returns
    -------
    list of Observable
        The observables, as a list.

    Raises
    ------
    ValueError
        On a duplicate or reserved name, or a name with ':'.
    """
    obs = list(observables or [])
    names = [o.name for o in obs]
    for o, n in zip(obs, names):
        if n == SGC_COUNTS_NAME and isinstance(o, SGCCounts):
            continue  # the engine's own record of the semi-grand-canonical counts
        if n == MOLECULE_COUNTS_NAME and isinstance(o, MoleculeCounts):
            continue  # the engine's own record of the molecule counts
        if n == SUBSURFACE_COUNTS_NAME and isinstance(o, SubsurfaceCounts):
            continue  # the engine's own record of the subsurface counts
        if (n in RESERVED_NAMES or n in (SGC_COUNTS_NAME, MOLECULE_COUNTS_NAME,
                                         SUBSURFACE_COUNTS_NAME) or ":" in n or not n):
            raise ValueError(f"observable name {n!r} is reserved or invalid "
                             f"(reserved: {sorted(RESERVED_NAMES)}; no ':')")
    if len(set(names)) != len(names):
        raise ValueError(f"observable names must be unique, got {names}")
    return obs


def evaluate_observables(observables: Sequence[Observable], system) -> Dict[str, list]:
    """
    Evaluate every observable on a configuration.

    Parameters
    ----------
    observables : sequence of Observable
        The observables.
    system : ParticleSystem
        The configuration.

    Returns
    -------
    dict
        ``{name: list of float}``.
    """
    return {o.name: np.asarray(o(system), dtype=float).tolist() for o in observables}


_TYPES = {"coordination": CoordinationNumber, "coordination_number": CoordinationNumber,
          "coordination_histogram": CoordinationHistogram, "rdf": RadialDistribution,
          "radial_distribution": RadialDistribution, "layer_composition": LayerComposition,
          "shell_composition": ShellComposition}


def observable_from_config(cfg: dict) -> Observable:
    """
    Build an observable from a config-file entry.

    Parameters
    ----------
    cfg : dict
        ``{type: coordination | coordination_histogram | rdf |
        layer_composition | shell_composition, name: ..., ...}`` with the class's keyword
        arguments (``z_range`` as a two-element list).

    Returns
    -------
    Observable
        The observable.

    Raises
    ------
    ValueError
        On an unknown type, or missing/unknown keys.
    """
    cfg = dict(cfg)
    kind = str(cfg.pop("type", "")).lower()
    if kind not in _TYPES:
        raise ValueError(f"observable type must be one of {sorted(set(_TYPES))}, got {kind!r}")
    cls = _TYPES[kind]
    if "name" not in cfg:
        raise ValueError(f"observable of type {kind!r} needs a 'name'")
    try:
        return cls(**cfg)
    except TypeError as err:
        raise ValueError(f"observable {cfg.get('name')!r} ({kind}): {err}") from None
