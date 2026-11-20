"""
Molecular insertion and deletion moves on top of atomic bookkeeping.

The simulation state stays a set of atoms with elemental chemical
potentials mu_e (`ThermoState.mu`, one per exchanged species): the target
distribution is the grand-canonical one of the atoms,

    pi({r}) ~ prod_e z_e^{N_e} exp(-beta U),      z_e = exp(beta mu_e) / Lambda_e^3,

(unlabelled, i.e. per set of positions), and replica exchange, MBAR and the
sample files are unchanged. A `MoleculeMove` is a *proposal* that inserts or
deletes several atoms at once, in the shape of a molecule:

* insertion: an anchor atom (C of CO or CH4, O of H2O, ...) uniformly in
  the exchange zone (density 1 / V), then each partner atom grown from the
  relaxed gas-phase template with a density q_j that can be evaluated --
  a truncated Gaussian bond length in the shell [r_min, r_max] around the
  anchor, and a direction that is uniform (first partner), at the template
  angle to the first (second partner) or von Mises-Fisher around the
  template direction in the frame of the first two (the rest);
* deletion: a uniform pick among the M "units" of the configuration -- sets
  of atoms with the molecule's composition whose anchor is in the zone and
  whose partners all lie in the shell around it.

With Q(u) = sum over the valid (anchor, partner-to-role) assignments of the
unit u of (1 / V) prod_j q_j, the acceptance probabilities are

    insert:  min(1, prod_e z_e^{nu_e} exp(-beta dU) / (M_after Q(u)))
    delete:  min(1, prod_e z_e^{-nu_e} exp(-beta dU) M_before Q(u))

(docs/REGC_algorithm.md, Sec. 10). For one atom (no partners) this is the
usual V z exp(-beta dU) / (N + 1). Since the partners' densities are
continuous and positive over the whole shell, the moves stay exact while
HMC/MD flex, break and re-form the bonds; fragments outside the shell are
simply not deletable as a molecule (atomic moves may still exchange them).

Molecules must be star-shaped: every partner bonded to the anchor (CO, O2,
N2, CO2, H2O, NH3, CH4, ...).
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_4PI = math.log(4.0 * math.pi)
LOG_2PI = math.log(2.0 * math.pi)
#: Above this many candidate units a configuration is considered pathological.
MAX_UNITS = 1_000_000


# =============================================================================
# One-dimensional and spherical densities
# =============================================================================
def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    Parameters
    ----------
    x : float
        Argument.

    Returns
    -------
    float
        Phi(x).
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def truncnorm_logpdf(x: float, mean: float, sd: float, lo: float, hi: float) -> float:
    """
    Log density of a normal distribution truncated to [lo, hi].

    Parameters
    ----------
    x : float
        Point.
    mean, sd : float
        Mean and standard deviation of the untruncated normal.
    lo, hi : float
        Support.

    Returns
    -------
    float
        The log density; -inf outside [lo, hi].
    """
    if not lo <= x <= hi:
        return -math.inf
    mass = _norm_cdf((hi - mean) / sd) - _norm_cdf((lo - mean) / sd)
    return (-0.5 * ((x - mean) / sd) ** 2 - math.log(sd * math.sqrt(2.0 * math.pi))
            - math.log(mass))


def truncnorm_sample(rng: np.random.Generator, mean: float, sd: float, lo: float,
                     hi: float) -> float:
    """
    Draw from a normal distribution truncated to [lo, hi].

    Exact rejection sampling from the untruncated normal: efficient when
    the mean lies in (or near) the support, as it does for every use here.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    mean, sd : float
        Mean and standard deviation of the untruncated normal.
    lo, hi : float
        Support.

    Returns
    -------
    float
        The sample.

    Raises
    ------
    RuntimeError
        If the support holds a negligible part of the normal (a
        misconfigured width).
    """
    for _ in range(100_000):
        x = mean + sd * rng.standard_normal()
        if lo <= x <= hi:
            return float(x)
    raise RuntimeError(f"truncated normal ({mean}, {sd}) on [{lo}, {hi}]: support too unlikely")


def vmf_logpdf(d: np.ndarray, mu: np.ndarray, kappa: float) -> float:
    """
    Log density of the von Mises-Fisher distribution on the unit sphere.

    Parameters
    ----------
    d : numpy.ndarray
        Unit vector, shape (3,).
    mu : numpy.ndarray
        Mean direction (unit), shape (3,).
    kappa : float
        Concentration, > 0.

    Returns
    -------
    float
        log [kappa / (4 pi sinh kappa) exp(kappa mu . d)], per steradian.
    """
    return (math.log(kappa) - LOG_2PI - math.log1p(-math.exp(-2.0 * kappa))
            + kappa * (float(np.dot(mu, d)) - 1.0))


def vmf_sample(rng: np.random.Generator, mu: np.ndarray, kappa: float) -> np.ndarray:
    """
    Draw a unit vector from the von Mises-Fisher distribution.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    mu : numpy.ndarray
        Mean direction (unit), shape (3,).
    kappa : float
        Concentration, > 0.

    Returns
    -------
    numpy.ndarray
        Unit vector, shape (3,).
    """
    xi = rng.random()
    w = 1.0 + math.log(xi + (1.0 - xi) * math.exp(-2.0 * kappa)) / kappa
    w = min(1.0, max(-1.0, w))
    a, b = _perpendicular_basis(mu)
    phi = 2.0 * math.pi * rng.random()
    return w * mu + math.sqrt(max(0.0, 1.0 - w * w)) * (math.cos(phi) * a + math.sin(phi) * b)


def _perpendicular_basis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two unit vectors completing the unit vector `v` to an orthonormal basis.

    Parameters
    ----------
    v : numpy.ndarray
        Unit vector, shape (3,).

    Returns
    -------
    tuple of numpy.ndarray
        ``(a, b)`` with a, b, v orthonormal and right-handed (a x b = v).
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(v, helper)
    a /= np.linalg.norm(a)
    return a, np.cross(v, a)


def uniform_direction(rng: np.random.Generator) -> np.ndarray:
    """
    Draw a unit vector uniformly on the sphere.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Unit vector, shape (3,).
    """
    u = rng.standard_normal(3)
    return u / np.linalg.norm(u)


def _frame(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Orthonormal frame spanned by two directions (Gram-Schmidt).

    Parameters
    ----------
    d1, d2 : numpy.ndarray
        Unit vectors, shape (3,); d2 should not be parallel to d1 (if it
        is, an arbitrary perpendicular is used).

    Returns
    -------
    numpy.ndarray
        Rows e1 = d1, e2 (in the d1-d2 plane, towards d2), e3 = e1 x e2.
    """
    e2 = d2 - np.dot(d2, d1) * d1
    n = np.linalg.norm(e2)
    e2 = _perpendicular_basis(d1)[0] if n < 1e-10 else e2 / n
    return np.array([d1, e2, np.cross(d1, e2)])


def _logsumexp(values: Sequence[float]) -> float:
    """
    log(sum(exp(values))), stable, -inf for an empty or all -inf input.

    Parameters
    ----------
    values : sequence of float
        Log terms.

    Returns
    -------
    float
        The log of the sum.
    """
    vals = [v for v in values if v > -math.inf]
    if not vals:
        return -math.inf
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


# =============================================================================
# The move
# =============================================================================
class MoleculeMove:
    """
    Insertion/deletion of a star-shaped molecule as a block of atoms.

    Parameters
    ----------
    name : str
        Label (e.g. ``'CO'``), used in move statistics and observables.
    symbols : sequence of str
        Element of every atom of the gas-phase template.
    positions : array_like
        Template positions, shape (n_atoms, 3), Angstrom (relaxed with the
        simulation's potential, ideally).
    anchor : str, optional
        Element of the anchor atom. Default None: the element with the
        fewest atoms in the template (alphabetically first on a tie), e.g.
        C in CO, CO2 and CH4, N in NO, O in H2O and O2.
    shell : tuple of float, optional
        ``(r_min, r_max)`` in Angstrom: partners lie at these distances
        from the anchor, both in proposals and in the definition of a
        deletable unit. Default None: 0.75 times the shortest and 1.25
        times the longest anchor-partner distance of the template.
    bond_sigma : float, optional
        Width of the Gaussian bond-length proposal, Angstrom. Default 0.05.
    angle_sigma_deg : float, optional
        Angular width of the direction proposals (second partner's angle,
        von Mises-Fisher concentration 1 / sigma^2 for the others), in
        degrees. Default 10.
    weight : float, optional
        Relative frequency among the GC move channels (each atomically
        exchanged species is a channel of weight 1). Default 1.

    Raises
    ------
    ValueError
        On a single-atom template, an anchor element not in the template,
        a molecule that is not star-shaped within the shell, or bad widths.
    """

    def __init__(self, name: str, symbols: Sequence[str], positions, anchor: Optional[str] = None,
                 shell: Optional[Tuple[float, float]] = None, bond_sigma: float = 0.05,
                 angle_sigma_deg: float = 10.0, weight: float = 1.0):
        """
        Build the move from a gas-phase template.

        Parameters
        ----------
        name, symbols, positions, anchor, shell, bond_sigma, angle_sigma_deg, weight
            See the class docstring.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        symbols = [str(s) for s in symbols]
        pos = np.asarray(positions, dtype=float).reshape(-1, 3)
        if len(symbols) != len(pos):
            raise ValueError(f"molecule {name}: {len(symbols)} symbols for {len(pos)} positions")
        if len(symbols) < 2:
            raise ValueError(f"molecule {name} has one atom: exchange it atomically instead")
        counts = Counter(symbols)
        if anchor is None:
            fewest = min(counts.values())
            anchor = min(s for s in counts if counts[s] == fewest)
        anchor = str(anchor)
        if anchor not in counts:
            raise ValueError(f"molecule {name}: anchor {anchor!r} is not one of its elements "
                             f"{sorted(counts)}")
        if bond_sigma <= 0 or angle_sigma_deg <= 0 or weight <= 0:
            raise ValueError(f"molecule {name}: bond_sigma, angle_sigma_deg and weight must be "
                             f"positive")
        # the anchor atom: the one of its element closest to the centre of the others
        cand = [i for i, s in enumerate(symbols) if s == anchor]
        ia = min(cand, key=lambda i: np.max(np.linalg.norm(pos - pos[i], axis=1)))
        others = [i for i in range(len(symbols)) if i != ia]
        vecs = pos[others] - pos[ia]
        dist = np.linalg.norm(vecs, axis=1)
        if shell is None:
            if dist.max() > 1.5 * dist.min():
                raise ValueError(
                    f"molecule {name} is not star-shaped around {anchor}: anchor-partner "
                    f"distances {np.round(dist, 3).tolist()} A are not all bond lengths (only "
                    f"molecules whose atoms are all bonded to one anchor atom have a default "
                    f"shell; give shell explicitly to override)")
            shell = (0.75 * float(dist.min()), 1.25 * float(dist.max()))
        r_min, r_max = float(shell[0]), float(shell[1])
        if not 0.0 < r_min < r_max:
            raise ValueError(f"molecule {name}: shell must satisfy 0 < r_min < r_max, got {shell}")
        if np.any(dist <= r_min) or np.any(dist >= r_max):
            raise ValueError(
                f"molecule {name} is not star-shaped around {anchor} within the shell "
                f"[{r_min:.3f}, {r_max:.3f}] A: anchor-partner distances "
                f"{np.round(dist, 3).tolist()} (only molecules whose atoms are all bonded to one "
                f"anchor atom are supported)")
        order = self._role_order(vecs)
        self.name = str(name)
        self.anchor = anchor
        self.partners: Tuple[str, ...] = tuple(symbols[others[j]] for j in order)
        self.template = vecs[order]                       # role vectors from the anchor
        self.r0 = np.linalg.norm(self.template, axis=1)
        self.shell = (r_min, r_max)
        self.bond_sigma = float(bond_sigma)
        self.angle_sigma = math.radians(float(angle_sigma_deg))
        self.kappa = 1.0 / self.angle_sigma ** 2
        self.weight = float(weight)
        self.composition: Dict[str, int] = dict(Counter((anchor,) + self.partners))
        self._setup_angles()

    # -- construction helpers ---------------------------------------------------------
    @staticmethod
    def _role_order(vecs: np.ndarray) -> List[int]:
        """
        Order the partners so that the first two are not collinear (if possible).

        Parameters
        ----------
        vecs : numpy.ndarray
            Anchor-partner vectors of the template, shape (k, 3).

        Returns
        -------
        list of int
            A permutation of range(k).
        """
        k = len(vecs)
        order = list(range(k))
        if k >= 3:
            u = vecs / np.linalg.norm(vecs, axis=1)[:, None]
            for j in range(1, k):
                if np.linalg.norm(np.cross(u[0], u[j])) > 1e-3:
                    order[1], order[j] = order[j], order[1]
                    break
        return order

    def _setup_angles(self) -> None:
        """
        Precompute the template's angular data for the direction proposals.

        Returns
        -------
        None
        """
        k = len(self.partners)
        t = self.template / self.r0[:, None]
        self._c12 = float(np.dot(t[0], t[1])) if k >= 2 else 0.0
        theta = math.acos(min(1.0, max(-1.0, self._c12)))
        self._s12 = max(self.angle_sigma * math.sin(theta), 0.5 * self.angle_sigma ** 2)
        self._coords = None
        self._chiral = None
        if k >= 3:
            f = _frame(t[0], t[1])
            self._coords = t @ f.T                       # role directions in the template frame
            out = [j for j in range(2, k) if abs(self._coords[j, 2]) > 1e-6]
            self._chiral = out[0] if out else None

    # -- properties --------------------------------------------------------------------
    @property
    def symbols(self) -> Tuple[str, ...]:
        """tuple of str: Elements in proposal order, anchor first then the partner roles."""
        return (self.anchor,) + self.partners

    @property
    def n_atoms(self) -> int:
        """int: Atoms per molecule."""
        return 1 + len(self.partners)

    def signature(self) -> tuple:
        """
        Hashable description, for checking that every replica uses the same move.

        Returns
        -------
        tuple
            Name, symbols, rounded template, shell, widths and weight.
        """
        return (self.name, self.symbols, tuple(np.round(self.template, 6).ravel()), self.shell,
                self.bond_sigma, self.angle_sigma, self.weight)

    def metadata(self) -> dict:
        """
        JSON-able description.

        Returns
        -------
        dict
            Name, anchor, partners, template vectors, shell, widths, weight.
        """
        return {"name": self.name, "anchor": self.anchor, "partners": list(self.partners),
                "template_angstrom": self.template.tolist(), "shell_angstrom": list(self.shell),
                "bond_sigma_angstrom": self.bond_sigma,
                "angle_sigma_deg": math.degrees(self.angle_sigma), "weight": self.weight}

    # -- directions ----------------------------------------------------------------------
    def _target(self, j: int, frame: np.ndarray, hand: float) -> np.ndarray:
        """
        Template direction of role j in an actual frame.

        Parameters
        ----------
        j : int
            Role index (>= 2).
        frame : numpy.ndarray
            Rows e1, e2, e3 from the first two actual directions.
        hand : float
            +1 or -1: the sign of the out-of-plane component.

        Returns
        -------
        numpy.ndarray
            Unit vector.
        """
        c = self._coords[j]
        return c[0] * frame[0] + c[1] * frame[1] + hand * c[2] * frame[2]

    def _handedness(self, d_chiral: np.ndarray, frame: np.ndarray) -> float:
        """
        Which mirror image the actual chiral-role partner is closest to.

        Parameters
        ----------
        d_chiral : numpy.ndarray
            Actual direction of the first out-of-plane role.
        frame : numpy.ndarray
            Rows e1, e2, e3 from the first two actual directions.

        Returns
        -------
        float
            +1 if it lies on the template's side of the e1-e2 plane, else -1.
        """
        side = float(np.dot(d_chiral, frame[2])) * self._coords[self._chiral, 2]
        return 1.0 if side >= 0.0 else -1.0

    def propose(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw partner positions relative to the anchor.

        Parameters
        ----------
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        numpy.ndarray
            Vectors from the anchor to each partner role, shape (k, 3);
            their density is exp(`log_q_roles`).
        """
        k = len(self.partners)
        r_min, r_max = self.shell
        dirs = [uniform_direction(rng)]
        if k >= 2:
            u = truncnorm_sample(rng, self._c12, self._s12, -1.0, 1.0)
            a, b = _perpendicular_basis(dirs[0])
            phi = 2.0 * math.pi * rng.random()
            dirs.append(u * dirs[0] + math.sqrt(max(0.0, 1.0 - u * u))
                        * (math.cos(phi) * a + math.sin(phi) * b))
        if k >= 3:
            frame = _frame(dirs[0], dirs[1])
            hand = 1.0
            for j in range(2, k):
                if j == self._chiral:
                    mu = self._target(j, frame, 1.0 if rng.random() < 0.5 else -1.0)
                    d = vmf_sample(rng, mu, self.kappa)
                    hand = self._handedness(d, frame)
                else:
                    d = vmf_sample(rng, self._target(j, frame, hand), self.kappa)
                dirs.append(d)
        radii = [truncnorm_sample(rng, self.r0[j], self.bond_sigma, r_min, r_max)
                 for j in range(k)]
        return np.array([r * d for r, d in zip(radii, dirs)])

    def log_q_roles(self, vecs: np.ndarray) -> float:
        """
        Log proposal density of partner vectors in role order.

        Parameters
        ----------
        vecs : numpy.ndarray
            Anchor-to-partner vectors (minimum image), shape (k, 3), the
            j-th assigned to role j.

        Returns
        -------
        float
            log prod_j q_j (per Angstrom^3 per partner); -inf if a partner
            lies outside the shell.
        """
        k = len(self.partners)
        r_min, r_max = self.shell
        r = np.linalg.norm(vecs, axis=1)
        if np.any(r < r_min) or np.any(r > r_max):
            return -math.inf
        out = 0.0
        for j in range(k):
            out += truncnorm_logpdf(float(r[j]), self.r0[j], self.bond_sigma, r_min, r_max)
        out -= 2.0 * float(np.sum(np.log(r)))
        d = vecs / r[:, None]
        out -= LOG_4PI
        if k >= 2:
            u = float(np.clip(np.dot(d[0], d[1]), -1.0, 1.0))
            out += truncnorm_logpdf(u, self._c12, self._s12, -1.0, 1.0) - LOG_2PI
        if k >= 3:
            frame = _frame(d[0], d[1])
            hand = self._handedness(d[self._chiral], frame) if self._chiral is not None else 1.0
            for j in range(2, k):
                if j == self._chiral:
                    out += _logsumexp([math.log(0.5) + vmf_logpdf(d[j], self._target(j, frame, h),
                                                                  self.kappa)
                                       for h in (1.0, -1.0)])
                else:
                    out += vmf_logpdf(d[j], self._target(j, frame, hand), self.kappa)
        return out

    # -- units -----------------------------------------------------------------------------
    def log_unit_density(self, positions: np.ndarray, species: np.ndarray, in_zone: np.ndarray,
                         unit: Sequence[int], min_image: Callable[[np.ndarray], np.ndarray],
                         volume: float) -> float:
        """
        Log of the forward proposal density Q(u) of a unit of atoms.

        Sums (1 / V) prod_j q_j over every way the insertion could have
        produced the unit: every atom of the anchor element inside the
        zone as the anchor, every assignment of the others to the partner
        roles of their element.

        Parameters
        ----------
        positions : numpy.ndarray
            All mobile positions, shape (n, 3).
        species : numpy.ndarray
            Their elements, shape (n,).
        in_zone : numpy.ndarray
            Whether each lies in the exchange zone, shape (n,).
        unit : sequence of int
            Indices of the unit's atoms.
        min_image : callable
            The geometry's minimum-image map of displacement vectors.
        volume : float
            Volume of the exchange zone (the anchor's proposal region).

        Returns
        -------
        float
            log Q(u); -inf if no assignment is valid.
        """
        unit = [int(i) for i in unit]
        roles_of: Dict[str, List[int]] = {}
        for j, s in enumerate(self.partners):
            roles_of.setdefault(s, []).append(j)
        elems = list(roles_of)
        terms = []
        for a in unit:
            if species[a] != self.anchor or not in_zone[a]:
                continue
            rest = [i for i in unit if i != a]
            groups = {e: [i for i in rest if species[i] == e] for e in elems}
            if any(len(groups[e]) != len(roles_of[e]) for e in elems) or \
                    sum(len(g) for g in groups.values()) != len(rest):
                continue
            vec = {i: min_image(positions[i] - positions[a]) for i in rest}
            for perms in itertools.product(*[itertools.permutations(groups[e]) for e in elems]):
                vecs = np.empty((len(self.partners), 3))
                for e, perm in zip(elems, perms):
                    for j, i in zip(roles_of[e], perm):
                        vecs[j] = vec[i]
                terms.append(-math.log(volume) + self.log_q_roles(vecs))
        return _logsumexp(terms)

    def _candidates(self, positions: np.ndarray, species: np.ndarray, in_zone: np.ndarray,
                    min_image: Callable[[np.ndarray], np.ndarray], zone_only: bool = True
                    ) -> List[Tuple[int, Dict[str, np.ndarray]]]:
        """
        Per possible anchor, the atoms of each partner element inside its shell.

        Parameters
        ----------
        positions, species, in_zone, min_image
            As in `log_unit_density`.
        zone_only : bool, optional
            Only anchors inside the zone (deletable units). Default True.

        Returns
        -------
        list of tuple
            ``(anchor index, {element: candidate indices})`` for anchors
            with enough candidates of every partner element.
        """
        need = Counter(self.partners)
        anchors = np.flatnonzero(species == self.anchor)
        if zone_only:
            anchors = anchors[np.asarray(in_zone, dtype=bool)[anchors]]
        r_min, r_max = self.shell
        out = []
        for a in anchors:
            r = np.linalg.norm(min_image(positions - positions[a]), axis=-1)
            near = (r >= r_min) & (r <= r_max)
            near[a] = False
            cand = {e: np.flatnonzero(near & (species == e)) for e in need}
            if all(len(cand[e]) >= n for e, n in need.items()):
                out.append((int(a), cand))
        return out

    def _needs_dedup(self) -> bool:
        """
        Whether one unit can be reached from two anchors (anchor element among the partners).

        Returns
        -------
        bool
            True for e.g. O2, False for CO, CO2, CH4.
        """
        return self.anchor in self.partners

    def _enumerate(self, cands) -> List[Tuple[int, ...]]:
        """
        Every distinct unit, as sorted index tuples.

        Parameters
        ----------
        cands : list
            Output of `_candidates`.

        Returns
        -------
        list of tuple of int
            The units.

        Raises
        ------
        RuntimeError
            Beyond `MAX_UNITS` candidate units (shrink the shell).
        """
        need = Counter(self.partners)
        units = set()
        for a, cand in cands:
            for combo in itertools.product(*[itertools.combinations(cand[e], n)
                                             for e, n in need.items()]):
                units.add(tuple(sorted((a,) + tuple(int(i) for c in combo for i in c))))
                if len(units) > MAX_UNITS:
                    raise RuntimeError(f"molecule {self.name}: more than {MAX_UNITS} candidate "
                                       f"units; shrink its shell")
        return sorted(units)

    def count_units(self, positions: np.ndarray, species: np.ndarray, in_zone: np.ndarray,
                    min_image: Callable[[np.ndarray], np.ndarray], zone_only: bool = True) -> int:
        """
        Number M of units (deletable molecules) in a configuration.

        Parameters
        ----------
        positions, species, in_zone, min_image
            As in `log_unit_density`.
        zone_only : bool, optional
            Count only units with the anchor in the exchange zone (what the
            deletion move picks from). Default True.

        Returns
        -------
        int
            M.
        """
        cands = self._candidates(positions, species, in_zone, min_image, zone_only)
        if self._needs_dedup():
            return len(self._enumerate(cands))
        need = Counter(self.partners)
        return int(sum(math.prod(math.comb(len(c[e]), n) for e, n in need.items())
                       for _, c in cands))

    def random_unit(self, positions: np.ndarray, species: np.ndarray, in_zone: np.ndarray,
                    min_image: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator
                    ) -> Tuple[Optional[Tuple[int, ...]], int]:
        """
        Pick a deletable unit uniformly at random.

        Parameters
        ----------
        positions, species, in_zone, min_image
            As in `log_unit_density`.
        rng : numpy.random.Generator
            Random number generator.

        Returns
        -------
        tuple
            ``(unit, M)``: the unit's indices (None if there is none) and
            the number of units it was drawn from.
        """
        cands = self._candidates(positions, species, in_zone, min_image)
        if self._needs_dedup():
            units = self._enumerate(cands)
            if not units:
                return None, 0
            return units[int(rng.integers(len(units)))], len(units)
        need = Counter(self.partners)
        counts = np.array([math.prod(math.comb(len(c[e]), n) for e, n in need.items())
                           for _, c in cands], dtype=float)
        total = int(counts.sum())
        if total == 0:
            return None, 0
        a, cand = cands[int(rng.choice(len(cands), p=counts / counts.sum()))]
        picked = [a]
        for e, n in need.items():
            picked += [int(i) for i in rng.choice(cand[e], size=n, replace=False)]
        return tuple(sorted(picked)), total


def molecule_moves_from_gases(gases: Sequence, **kwargs) -> List[MoleculeMove]:
    """
    One `MoleculeMove` per polyatomic gas, from their relaxed templates.

    Parameters
    ----------
    gases : sequence of regc.reservoir.MolecularGas
        The gases (monatomic ones are skipped).
    **kwargs
        Passed to every `MoleculeMove` (anchor, shell, widths, weight).

    Returns
    -------
    list of MoleculeMove
        The moves, named after the gases.
    """
    return [MoleculeMove(g.name, g.symbols, g.positions_angstrom, **kwargs)
            for g in gases if len(g.symbols) > 1]
