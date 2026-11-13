"""
How far to trust a REGC result: correlation, error bars, overlap, mixing.

MBAR turns every recorded sample into a smooth map, but a map is only as
good as the samples behind it. This module answers four questions:

1. **How many independent samples does each replica hold?**
   `statistical_inefficiency` and `replica_inefficiencies`: successive REGC
   samples are correlated; a replica with n samples and statistical
   inefficiency g carries about n/g independent ones.
2. **How large is the statistical error of a map?** `bootstrap`: a moving-
   block bootstrap within each replica (blocks longer than the
   correlation, so correlation is kept), MBAR re-solved for every
   resample, any statistic recomputed. Percentile intervals for <N>,
   C_V, observables; for an integer map (the most probable N), the
   fraction of resamples that agree with it.
3. **Do neighbouring replicas overlap?** `neighbour_overlap`: the MBAR
   overlap between grid neighbours. A gap leaves the states between two
   replicas resting on a few samples; there, variances such as C_V are
   underestimated and spurious peaks appear on the sampled values.
4. **Is a C_V peak real?** `ridge_support`: the fraction of bootstrap
   resamples in which a detected peak still rises above its flanks.

Mixing of the replica exchange itself (per-pair swap acceptance, round
trips of each configuration across the ladder) is recorded by the engine:
`ReplicaExchangeGCMC.pair_acceptance` and `round_trip_counts`.

Requires pymbar (and scipy for the ridges); imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. Correlation
# =============================================================================
def statistical_inefficiency(x, c: float = 5.0) -> float:
    """
    Statistical inefficiency g = 1 + 2 sum_t rho(t) of a time series.

    The autocorrelation rho(t) comes from an FFT; the sum is truncated with
    Sokal's automatic window (the smallest M with M >= c * g(M)), which
    keeps the noise of the long-time tail out of the estimate. n samples
    with inefficiency g carry about n / g independent ones.

    Parameters
    ----------
    x : array_like
        The series (one value per recorded sample, in recording order).
    c : float, optional
        Window constant. Default 5.

    Returns
    -------
    float
        g >= 1 (1 for a constant or too short series).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    x = x - x.mean()
    var = float(x @ x) / n
    if var <= 0.0:
        return 1.0
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (var * n)
    g = 2.0 * np.cumsum(acf) - 1.0
    window = np.arange(n) < c * g
    m = int(np.argmin(window)) if not window.all() else n - 1
    return float(max(1.0, g[m]))


def replica_inefficiencies(source, lambda_cubed_of_T: Optional[Callable] = None) -> Dict:
    """
    Statistical inefficiency of N, U and the reduced potential, per replica.

    Parameters
    ----------
    source : ReplicaExchangeGCMC or SampleSet
        Anything with ``samples``, ``temperatures`` and ``mu_grid``.
    lambda_cubed_of_T : callable, optional
        Lambda^3(T) for the reduced potential. Default: the source's.

    Returns
    -------
    dict
        ``{key: {'n': samples, 'g_N': ..., 'g_U': ..., 'g_u': ..., 'g': max,
        'n_independent': n / g}}``.
    """
    from .io import state_of
    lam = lambda_cubed_of_T or getattr(source, "lambda_cubed_of_T", lambda T: 1.0)
    out = {}
    for key, v in source.samples.items():
        T, mu = state_of(source, key)
        U = np.asarray(v["U"], dtype=float)
        if np.ndim(mu) == 1:  # several species: u summed over species
            Ns = np.asarray(v["N_s"], dtype=float).reshape(len(U), -1)
            N = Ns.sum(axis=1)
            lam3 = np.broadcast_to(np.asarray(lam(T), dtype=float), (Ns.shape[1],))
            u = U / T + Ns @ (np.log(lam3) - np.asarray(mu) / T)
        else:
            N = np.asarray(v["N"], dtype=float)
            u = (U - mu * N) / T + N * np.log(float(lam(T)))
        g = {name: statistical_inefficiency(series)
             for name, series in (("g_N", N), ("g_U", U), ("g_u", u))}
        g_max = max(g.values())
        out[key] = dict(g, n=len(U), g=g_max, n_independent=len(U) / g_max)
    return out


# =============================================================================
# 2. Bootstrap error bars
# =============================================================================
@dataclass
class BootstrapResult:
    """
    A statistic of the full data set and its bootstrap replicates.

    Attributes
    ----------
    estimate : numpy.ndarray
        The statistic on the full data.
    replicates : numpy.ndarray
        Shape (n_boot,) + estimate.shape.
    block_lengths : dict
        ``{key: block length}`` used per replica.
    """

    estimate: np.ndarray
    replicates: np.ndarray
    block_lengths: Dict

    def interval(self, level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Percentile bootstrap interval.

        Parameters
        ----------
        level : float, optional
            Coverage. Default 0.95.

        Returns
        -------
        tuple of numpy.ndarray
            ``(low, high)``, each shaped like `estimate`.
        """
        a = 50.0 * (1.0 - level)
        return (np.nanpercentile(self.replicates, a, axis=0),
                np.nanpercentile(self.replicates, 100.0 - a, axis=0))

    def std(self) -> np.ndarray:
        """
        Bootstrap standard error.

        Returns
        -------
        numpy.ndarray
            Shaped like `estimate`.
        """
        return np.nanstd(self.replicates, axis=0, ddof=1)

    def agreement(self) -> np.ndarray:
        """
        Fraction of replicates equal to the estimate (for integer maps).

        For the most probable N: 1 where every resample gives the same
        phase, low where the label is decided by noise.

        Returns
        -------
        numpy.ndarray
            Shaped like `estimate`, in [0, 1].
        """
        return np.mean(self.replicates == self.estimate[None], axis=0)


def _block_resample(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """
    Indices of one moving-block bootstrap resample of a length-n series.

    Parameters
    ----------
    n : int
        Series length.
    block : int
        Block length (1: the ordinary bootstrap).
    rng : numpy.random.Generator
        Random stream.

    Returns
    -------
    numpy.ndarray
        n indices: ceil(n / block) blocks of consecutive indices, starts
        uniform in [0, n - block], concatenated and cut to n.
    """
    block = int(min(max(1, block), n))
    starts = rng.integers(0, n - block + 1, size=-(-n // block))
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def bootstrap(analysis, statistic: Callable, n_boot: int = 100, seed: int = 0,
              block_lengths: Optional[Dict] = None,
              progress: Optional[Callable[[int], None]] = None) -> BootstrapResult:
    """
    Block-bootstrap error bars for any statistic of an MBAR analysis.

    Each replica's samples are resampled in blocks of consecutive samples,
    about twice its statistical inefficiency long, so correlation inside
    a block is kept; MBAR is re-solved on every resample (warm-started
    from the full data) and `statistic` recomputed. The resamples are drawn
    independently per replica, which ignores the (weak) correlation that
    replica exchange creates between replicas.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis of the full data.
    statistic : callable
        ``statistic(analysis) -> array`` (e.g. ``lambda a:
        a.heat_capacity(T, p, pressure=True)``).
    n_boot : int, optional
        Number of resamples. Default 100 (enough for 90 % intervals; use
        a few hundred for 95 % intervals you will quote).
    seed : int, optional
        Random seed. Default 0.
    block_lengths : dict, optional
        ``{key: block length}``. Default: ``ceil(2 g)`` per replica from
        `replica_inefficiencies`, at most a quarter of the samples.
    progress : callable, optional
        Called with the resample index. Default None.

    Returns
    -------
    BootstrapResult
        Estimate, replicates and the block lengths used.
    """
    src = analysis._source
    if block_lengths is None:
        ineff = replica_inefficiencies(src, analysis.lambda_cubed_of_T)
        block_lengths = {k: int(min(max(1, np.ceil(2.0 * v["g"])), max(1, v["n"] // 4)))
                         for k, v in ineff.items()}
    rng = np.random.default_rng(seed)
    estimate = np.asarray(statistic(analysis))
    reps = np.empty((n_boot,) + estimate.shape, dtype=float)
    for b in range(n_boot):
        idx = {k: _block_resample(len(v["U"]), block_lengths[k], rng) for k, v in src.samples.items()}
        reps[b] = np.asarray(statistic(analysis.resampled(idx)), dtype=float)
        if progress is not None:
            progress(b)
    return BootstrapResult(estimate, reps, block_lengths)


# =============================================================================
# 3. Overlap between neighbouring replicas
# =============================================================================
#: Smallest MBAR overlap between neighbouring states commonly considered
#: adequate (Klimovich, Shirts & Mobley, J. Comput.-Aided Mol. Des. 29, 397
#: (2015)): below it, estimates between the two states rest on few samples.
OVERLAP_THRESHOLD = 0.03


def _state_coordinates(analysis, pressure: bool) -> Dict:
    """
    (T, y) of every sampled state: y = mu, or log10 p with ``pressure``.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    pressure : bool
        Use log10 of the reservoir pressure (via ``p_from_mu``) for y.

    Returns
    -------
    dict
        ``{key: (T, y)}``.
    """
    out = {}
    for key, beta, mu in zip(analysis.states, analysis.betas, analysis.state_mus):
        T = 1.0 / beta
        y = np.log10(analysis.p_from_mu(T, mu)) if pressure else float(np.ravel(mu)[0])
        out[key] = (float(T), float(y))
    return out


def ladder_pairs(analysis, pressure: bool = False, decimals: int = 6) -> Dict[str, list]:
    """
    Neighbouring sampled states, along y at fixed T and along T at fixed y.

    Works for one rectangular grid and for merged runs alike: states with
    the same T (to `decimals`) form a row, ordered by y; states with the
    same y form a column, ordered by T; neighbours are consecutive.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    pressure : bool, optional
        Order by log10 p instead of mu (pressure ladders, whose mu differ
        from row to row). Default False.
    decimals : int, optional
        Rounding used to group equal T or y. Default 6.

    Returns
    -------
    dict
        ``{'y': [(key_a, key_b), ...], 'T': [...]}``.
    """
    coords = _state_coordinates(analysis, pressure)
    rows, cols = {}, {}
    for key, (T, y) in coords.items():
        rows.setdefault(round(T, decimals), []).append(key)
        cols.setdefault(round(y, decimals), []).append(key)
    pairs = {"y": [], "T": []}
    for axis, groups, coord in (("y", rows, 1), ("T", cols, 0)):
        for keys in groups.values():
            keys = sorted(keys, key=lambda k: coords[k][coord])
            pairs[axis] += list(zip(keys[:-1], keys[1:]))
    return pairs


def neighbour_overlap(analysis, pressure: bool = False) -> Dict:
    """
    MBAR overlap between grid neighbours, and the worst one of every state.

    The overlap O_ab = sum_n w_n(a) w_n(b) N_b (pymbar's overlap matrix)
    is the probability that a sample drawn at a is reweighted to b; for
    equal sample counts it is symmetric, and its average over a and b is
    used here.

    For one grid, neighbours are along T, mu and both diagonals; for
    merged runs (keys ``(run, it, imu)``), consecutive states along y at
    fixed T and along T at fixed y (`ladder_pairs`).

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis.
    pressure : bool, optional
        Merged runs only: order states by log10 p instead of mu. Default
        False.

    Returns
    -------
    dict
        ``'pairs'``: ``{(key_a, key_b): overlap}``; ``'min_map'``: for a
        grid, array (n_T, n_mu), the smallest overlap of each state with
        any neighbour (nan for a lone state), for merged runs ``{key:
        overlap}``; ``'gaps'``: the pairs below `OVERLAP_THRESHOLD`.
    """
    O = np.asarray(analysis.mbar.compute_overlap()["matrix"], dtype=float)
    index = {key: i for i, key in enumerate(analysis.states)}
    if any(len(k) != 2 for k in index):   # merged runs: no single grid
        pairs = {}
        for axis_pairs in ladder_pairs(analysis, pressure).values():
            for a, b in axis_pairs:
                i, j = index[a], index[b]
                pairs[(a, b)] = 0.5 * (O[i, j] + O[j, i])
        min_map = {}
        for (a, b), o in pairs.items():
            for k in (a, b):
                min_map[k] = min(min_map.get(k, np.inf), o)
        return {"pairs": pairs, "min_map": min_map,
                "gaps": {p: o for p, o in pairs.items() if o < OVERLAP_THRESHOLD}}
    n_T = 1 + max(k[0] for k in index)
    n_mu = 1 + max(k[1] for k in index)
    pairs = {}
    for (it, im), i in index.items():
        for d in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = (it + d[0], im + d[1])
            if other in index:
                j = index[other]
                pairs[((it, im), other)] = 0.5 * (O[i, j] + O[j, i])
    min_map = np.full((n_T, n_mu), np.nan)
    for (a, b), o in pairs.items():
        for k in (a, b):
            min_map[k] = o if np.isnan(min_map[k]) else min(min_map[k], o)
    gaps = {p: o for p, o in pairs.items() if o < OVERLAP_THRESHOLD}
    return {"pairs": pairs, "min_map": min_map, "gaps": gaps}


# =============================================================================
# 4. Support of heat-capacity peaks
# =============================================================================
def ridge_support(ridges: dict, cv_replicates: np.ndarray) -> np.ndarray:
    """
    Fraction of bootstrap resamples in which each C_V peak survives.

    A peak found by `regc.analysis.heat_capacity_ridges` survives a
    resample if C_V at the peak still exceeds C_V at both of its flanking
    minima (the bases of its prominence) along the axis where it was found.

    Parameters
    ----------
    ridges : dict
        From `heat_capacity_ridges` (with its ``'along_y_bases'`` and
        ``'along_T_bases'``).
    cv_replicates : numpy.ndarray
        Bootstrap C_V maps, shape (n_boot, n_T, n_y).

    Returns
    -------
    numpy.ndarray
        One support value per point of ``ridges['points']`` (the larger of
        its supports along y and along T).
    """
    reps = np.asarray(cv_replicates, dtype=float)
    support = {}
    for (i, j), (lo, hi) in zip(ridges["along_y"], ridges["along_y_bases"]):
        line = reps[:, i, :]
        s = float(np.mean(line[:, j] > np.maximum(line[:, lo], line[:, hi])))
        support[(int(i), int(j))] = max(support.get((int(i), int(j)), 0.0), s)
    for (i, j), (lo, hi) in zip(ridges["along_T"], ridges["along_T_bases"]):
        line = reps[:, :, j]
        s = float(np.mean(line[:, i] > np.maximum(line[:, lo], line[:, hi])))
        support[(int(i), int(j))] = max(support.get((int(i), int(j)), 0.0), s)
    return np.array([support.get(tuple(c), 0.0) for c in ridges["cells"]])


# =============================================================================
# 5. Filling the gaps of a ladder
# =============================================================================
@dataclass
class LadderSuggestion:
    """
    New states proposed where neighbouring replicas overlap too little.

    Attributes
    ----------
    gaps : list of dict
        One entry per neighbouring pair below the target: ``a``, ``b`` (the
        keys), ``axis`` (``'y'`` or ``'T'``), ``overlap`` and ``inserted``
        (how many points that pair needs; each interval of the ladder gets
        the largest number any of its pairs needs).
    new_y : numpy.ndarray
        New y values (mu, or log10 p with ``pressure``), sorted.
    new_T : numpy.ndarray
        New temperatures, sorted.
    T_ladder : numpy.ndarray
        The temperatures of the analysed states.
    y_ladder : numpy.ndarray
        Their y values.
    pressure : bool
        Whether y is log10 p.
    """

    gaps: list
    new_y: np.ndarray
    new_T: np.ndarray
    T_ladder: np.ndarray
    y_ladder: np.ndarray
    pressure: bool

    @property
    def fill_runs(self) -> List[dict]:
        """
        Rectangular grids that add the new states.

        The engine samples a (temperature x y) grid, so the new y columns
        are run at every temperature, and the new temperatures at every y
        (old and new).

        Returns
        -------
        list of dict
            ``{'T': array, 'y': array}`` per run (none if nothing to fill).
        """
        runs = []
        if self.new_y.size:
            runs.append({"T": np.array(self.T_ladder), "y": np.array(self.new_y)})
        if self.new_T.size:
            runs.append({"T": np.array(self.new_T),
                         "y": np.unique(np.concatenate([self.y_ladder, self.new_y]))})
        return runs

    def to_config(self, kelvin: bool = True, pressure_unit: str = "bar") -> str:
        """
        The fill runs as ``replica_exchange`` snippets for a config file.

        Parameters
        ----------
        kelvin : bool, optional
            Temperatures were k_B T in eV: write them in K. Default True.
        pressure_unit : str, optional
            Unit of the analysis's pressures (``pressure=True``). Default
            ``'bar'``.

        Returns
        -------
        str
            YAML text, one block per fill run (empty if none).
        """
        from .units import eV_to_kelvin
        blocks = []
        for k, run in enumerate(self.fill_runs):
            T = [float(eV_to_kelvin(t)) if kelvin else float(t) for t in run["T"]]
            lines = [f"# fill run {k + 1}: {len(run['T'])} x {len(run['y'])} replicas",
                     "replica_exchange:",
                     "  temperatures" + ("_K" if kelvin else "") + ": ["
                     + ", ".join(f"{t:.4g}" for t in T) + "]"]
            if self.pressure:
                lines += ["  pressures: [" + ", ".join(f"{10 ** y:.4g}" for y in run["y"]) + "]",
                          f"  pressure_unit: {pressure_unit}"]
            else:
                lines.append("  mu_eV: [" + ", ".join(f"{y:.4f}" for y in run["y"]) + "]")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def suggest_ladder(analysis, target_overlap: float = 0.1, pressure: bool = False,
                   max_insert: int = 4) -> LadderSuggestion:
    """
    Propose new states where neighbouring replicas overlap too little.

    For every neighbouring pair (`ladder_pairs`) with MBAR overlap below
    `target_overlap`, points are inserted evenly between them. The gap is
    cut into ceil(ln o / ln target) pieces, o being the pair's overlap:
    exact if the overlap decays like exp(-a d) with the distance d, and
    conservative for the faster, Gaussian exp(-a d^2) decay of broad energy
    distributions (Poisson-like particle numbers decay more slowly than
    that, and a Gaussian rule left gaps open in tests). Check the result by
    asking again after merging the fill run.
    Each interval between consecutive ladder values gets one set of
    evenly spaced points, as many as its worst pair needs, shared by every
    row (or column): rectangular fill runs
    (`LadderSuggestion.fill_runs`) whose samples are combined with the
    original ones by `regc.io.merge_samples`. Start them from equilibrated
    structures with `system_factory_from_configurations`.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis of the existing run(s).
    target_overlap : float, optional
        Overlap to reach between neighbours. Default 0.1 (above the 0.03
        floor of `OVERLAP_THRESHOLD`, with margin).
    pressure : bool, optional
        Work in log10 p (pressure ladders; needs a reservoir in the
        analysis). Default False (mu).
    max_insert : int, optional
        Most points inserted in one gap. Default 4.

    Returns
    -------
    LadderSuggestion
        The gaps and the proposed states.

    Raises
    ------
    ValueError
        If `target_overlap` is not in (0, 1).
    """
    if not 0.0 < target_overlap < 1.0:
        raise ValueError(f"target_overlap must be in (0, 1), got {target_overlap}")
    O = np.asarray(analysis.mbar.compute_overlap()["matrix"], dtype=float)
    index = {key: i for i, key in enumerate(analysis.states)}
    coords = _state_coordinates(analysis, pressure)
    # every interval between two consecutive ladder values gets ONE set of evenly spaced
    # points (as many as its worst pair needs), shared by all rows / columns
    gaps, need = [], {"y": {}, "T": {}}
    for axis, pairs in ladder_pairs(analysis, pressure).items():
        c = 1 if axis == "y" else 0
        for a, b in pairs:
            o = 0.5 * (O[index[a], index[b]] + O[index[b], index[a]])
            if o >= target_overlap:
                continue
            pieces = np.ceil(np.log(max(o, 1e-300)) / np.log(target_overlap))
            k = int(min(max_insert, max(1, pieces - 1)))
            interval = (round(coords[a][c], 9), round(coords[b][c], 9))
            need[axis][interval] = max(need[axis].get(interval, 0), k)
            gaps.append({"a": a, "b": b, "axis": axis, "overlap": float(o), "inserted": k})
    T_all = np.unique(np.round([t for t, _ in coords.values()], 12))
    y_all = np.unique(np.round([y for _, y in coords.values()], 12))

    def points(axis):
        pts = [lo + (hi - lo) * m / (k + 1) for (lo, hi), k in need[axis].items()
               for m in range(1, k + 1)]
        return np.unique(np.round(pts, 9))
    return LadderSuggestion(gaps, points("y"), points("T"), T_all, y_all, pressure)


def system_factory_from_configurations(make_system: Callable, configurations_dir, analysis,
                                       temperatures, mus) -> Callable:
    """
    A system factory that starts each new replica from a stored structure.

    For a fill run (or any new run), every new grid point (T_i, mu_ij) gets
    the last stored configuration of the nearest sampled state, nearest in
    (beta, beta mu) scaled by the ladder's spacings, instead of an empty
    or arbitrary start: the new replicas start close to equilibrium.

    The engine builds its systems in grid order, (0, 0), (0, 1), ...; the
    returned factory hands out the configurations in that order, so build
    exactly one `ReplicaExchangeGCMC` with it.

    Parameters
    ----------
    make_system : callable
        ``make_system(rng)``: an empty system of the right geometry and
        calculator (the run's usual factory).
    configurations_dir : str or sequence of str
        Stored configurations of the analysed run(s) (one per merged run).
    analysis : MBARSurfaceAnalysis
        The analysis of the existing run(s).
    temperatures : sequence of float
        Temperatures of the new run (k_B T, as in the analysis).
    mus : array_like
        Its mu grid, shape (n_T, n_mu) (or (n_mu,), shared by all T).

    Returns
    -------
    callable
        ``factory(rng) -> ParticleSystem``.
    """
    from .analysis import _config_location
    from .trajectory import load_configurations
    T_new = np.asarray(temperatures, dtype=float)
    mu_new = np.asarray(mus, dtype=float)
    if mu_new.ndim == 1:
        mu_new = np.tile(mu_new, (T_new.size, 1))
    b_old = analysis.betas
    mu_old = np.asarray(analysis.state_mus, dtype=float)
    bm_old = analysis.betas * (mu_old[:, 0] if mu_old.ndim == 2 else mu_old)
    sb = max(np.ptp(b_old), 1e-12)
    sm = max(np.ptp(bm_old), 1e-12)
    order = [(i, j) for i in range(T_new.size) for j in range(mu_new.shape[1])]
    starts = []
    for i, j in order:
        b, bm = 1.0 / T_new[i], np.ravel(mu_new[i, j])[0] / T_new[i]
        d = ((b_old - b) / sb) ** 2 + ((bm_old - bm) / sm) ** 2
        for k in np.argsort(d):   # nearest state with a stored configuration
            frames = load_configurations(*_config_location(configurations_dir, analysis.states[k]))
            if frames:
                starts.append(frames[max(frames)])
                break
        else:
            starts.append(None)
    it = iter(starts)

    def factory(rng):
        system = make_system(rng)
        fr = next(it, None)
        if fr is None:
            return system
        groups = np.asarray(fr["arrays"]["group"])
        pos, sp = np.asarray(fr["positions"]), np.asarray(fr["species"]).astype(str)
        if int(np.sum(groups == 1)) == system.substrate_positions.shape[0]:
            system.substrate_positions = pos[groups == 1].copy()
        system.positions = pos[groups == 2].copy()
        system.species = sp[groups == 2].copy()
        return system
    return factory
