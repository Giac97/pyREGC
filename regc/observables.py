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
    for n in names:
        if n in RESERVED_NAMES or ":" in n or not n:
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
          "radial_distribution": RadialDistribution}


def observable_from_config(cfg: dict) -> Observable:
    """
    Build an observable from a config-file entry.

    Parameters
    ----------
    cfg : dict
        ``{type: coordination | coordination_histogram | rdf, name: ...,
        ...}`` with the class's keyword arguments (``z_range`` as a
        two-element list).

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
