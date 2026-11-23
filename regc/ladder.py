"""
Adaptive replica ladder: move the (T, p) states of a running grid so that
replicas travel more easily between its ends, from measured swap rates.

The ladder is a temperature ladder T_l shared by each row and a pressure
coordinate y = log10 p per state (every species' mu follows from the gas
feed at (T, p)). The corners T_0, T_last and the first and last pressure
of every row stay fixed; the interior moves, keeping its order.

Acceptance model of a neighbouring pair (i, j), calibrated on the swap rate
a_ij measured at the current ladder (per-chunk pair counts,
`ReplicaExchangeGCMC.pair_acceptance_by_chunk`):

    a = 2 Phi(-s / 2),   s^2 = kappa_ij dth^T (C_i + C_j) dth,

dth = theta_i - theta_j, theta = (beta, beta mu_s - ln Lambda_s^3, ...) and
C the covariance of (U, -N_s, ...) at a state (the Gaussian swap
statistics; kappa = 1 for exactly Gaussian fluctuations). kappa_ij makes
the model reproduce the measured a_ij at the current ladder; a moved state
takes C interpolated from the sampled states (`CovarianceField`).

The loss is ln R_T + ln R_mu, the effective resistances of the swap graph
(conductance of a pair: predicted acceptance x its attempt rate) between
the coldest and hottest rows and between the lowest and highest pressure
columns. Every gap may change by at most a factor `trust` per adaptation
and must stay within [min_gap_ratio, max_gap_ratio] x its initial value:
without this coverage bound the loss games itself (one test run pulled the
whole coldest row onto its low-pressure corner).

`propose_ladder` is the pure optimisation; `LadderAdapter` applies it to a
running simulation between chunks (`regc.driver.run_with_checkpoints`).
Reweighting (MBAR) predictions of the acceptances were tried instead of the
measured rates and are unreliable near phase transitions (effective sample
sizes of about one), so they are not used. Needs scipy (Powell).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Range of the calibration factor kappa (outside it, the model cannot explain the rate).
KAPPA_RANGE = (1e-3, 1e3)
#: Smallest acceptance in a conductance (keeps the swap graph connected).
ACCEPT_FLOOR = 1e-12

Key = Tuple[int, int]
Pair = Tuple[Key, Key]


def state_parameters(T: float, y: float, mu_of: Callable, lambda_cubed_of_T: Callable
                     ) -> np.ndarray:
    """
    Conjugate parameters theta = (beta, beta mu_s - ln Lambda_s^3, ...) of one state.

    Parameters
    ----------
    T : float
        Temperature (engine units: k_B T in eV for real runs).
    y : float
        log10 of the pressure.
    mu_of : callable
        ``mu_of(T, y)``: the chemical potential(s) at (T, 10**y).
    lambda_cubed_of_T : callable
        Lambda^3(T), one value or one per species.

    Returns
    -------
    numpy.ndarray
        Shape (1 + n_species,).
    """
    beta = 1.0 / T
    mu = np.atleast_1d(np.asarray(mu_of(T, y), dtype=float))
    lam3 = np.broadcast_to(np.asarray(lambda_cubed_of_T(T), dtype=float), mu.shape)
    return np.concatenate([[beta], beta * mu - np.log(lam3)])


def state_covariances(samples: Dict[Key, Dict[str, list]], shape: Tuple[int, int],
                      n_species: int, min_samples: int = 2) -> np.ndarray:
    """
    Covariance of (U, -N_s, ...) of the recorded samples of every state.

    Parameters
    ----------
    samples : dict
        ``{(l, m): {'U': [...], 'N': [...], 'N_s': [...]}}`` (`ReplicaExchangeGCMC.samples`).
    shape : tuple of int
        Grid shape (n_T, n_mu).
    n_species : int
        Exchanged species ('N' is used for one, 'N_s' for several).
    min_samples : int, optional
        Fewest samples a state needs. Default 2.

    Returns
    -------
    numpy.ndarray
        Shape (n_T, n_mu, 1 + n_species, 1 + n_species).

    Raises
    ------
    ValueError
        If a state holds fewer than `min_samples` samples.
    """
    d = 1 + n_species
    out = np.zeros(tuple(shape) + (d, d))
    for l in range(shape[0]):
        for m in range(shape[1]):
            v = samples[(l, m)]
            U = np.asarray(v["U"], dtype=float)
            if U.size < max(2, min_samples):
                raise ValueError(f"state {(l, m)} has {U.size} samples, fewer than "
                                 f"{max(2, min_samples)}")
            N = (np.asarray(v["N"], dtype=float)[:, None] if n_species == 1
                 else np.asarray(v["N_s"], dtype=float).reshape(-1, n_species))
            X = np.concatenate([U[:, None], -N], axis=1)
            out[l, m] = np.atleast_2d(np.cov(X.T))
    return out


class CovarianceField:
    """
    Covariance C(T, y) interpolated from the states of a (possibly non-product) grid.

    Rows are interpolated linearly in ln T, and within each row linearly in
    y; outside the sampled range the value at the edge is used. At a
    sampled state the sampled covariance is returned exactly.

    Parameters
    ----------
    temperatures : array_like
        The sampled ladder, shape (n_T,), monotonic.
    log10p : array_like
        log10 p of every sampled state, shape (n_T, n_mu), monotonic along rows.
    covariances : numpy.ndarray
        Shape (n_T, n_mu, d, d).
    """

    def __init__(self, temperatures, log10p, covariances: np.ndarray):
        """
        Store the sampled grid, sorted to increasing ln T and y.

        Parameters
        ----------
        temperatures : array_like
            The sampled ladder, shape (n_T,).
        log10p : array_like
            log10 p of every sampled state, shape (n_T, n_mu).
        covariances : numpy.ndarray
            Shape (n_T, n_mu, d, d).
        """
        lnT = np.log(np.asarray(temperatures, dtype=float))
        Y = np.asarray(log10p, dtype=float)
        C = np.asarray(covariances, dtype=float)
        order = np.argsort(lnT)
        self.lnT, Y, C = lnT[order], Y[order], C[order]
        self.Y = np.empty_like(Y)
        self.C = np.empty_like(C)
        for l in range(Y.shape[0]):
            o = np.argsort(Y[l])
            self.Y[l], self.C[l] = Y[l, o], C[l, o]

    def _in_row(self, l: int, y: np.ndarray) -> np.ndarray:
        """
        C along one sampled row at the y values `y` (linear, clamped).

        Parameters
        ----------
        l : int
            Sampled row.
        y : numpy.ndarray
            Query values, shape (K,).

        Returns
        -------
        numpy.ndarray
            Shape (K, d, d).
        """
        n = self.Y.shape[1]
        if n == 1:
            return np.repeat(self.C[l, :1], len(y), axis=0)
        f = np.interp(y, self.Y[l], np.arange(n))
        m0 = np.minimum(f.astype(int), n - 2)
        w = (f - m0)[:, None, None]
        return (1.0 - w) * self.C[l, m0] + w * self.C[l, m0 + 1]

    def __call__(self, T, y) -> np.ndarray:
        """
        Interpolated covariances.

        Parameters
        ----------
        T : array_like
            Temperatures, shape (K,).
        y : array_like
            log10 p, shape (K,).

        Returns
        -------
        numpy.ndarray
            Shape (K, d, d).
        """
        lnT = np.log(np.atleast_1d(np.asarray(T, dtype=float)))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        n = len(self.lnT)
        if n == 1:
            return self._in_row(0, y)
        f = np.interp(lnT, self.lnT, np.arange(n))
        l0 = np.minimum(f.astype(int), n - 2)
        w = f - l0
        out = np.empty((len(y),) + self.C.shape[2:])
        for l in np.unique(l0):
            sel = l0 == l
            ww = w[sel][:, None, None]
            out[sel] = (1.0 - ww) * self._in_row(l, y[sel]) + ww * self._in_row(l + 1, y[sel])
        return out


def window_counts(intervals: Sequence[dict], version: int) -> Dict[Pair, Tuple[int, int]]:
    """
    Swap attempts and accepts of every pair, summed over the intervals of one ladder.

    Parameters
    ----------
    intervals : sequence of dict
        From `ReplicaExchangeGCMC.pair_acceptance_by_chunk` (``pairs`` and
        ``ladder_version`` per interval; a missing version counts as 0).
    version : int
        The ladder version to sum.

    Returns
    -------
    dict
        ``{(key_a, key_b): (attempts, accepts)}``.
    """
    out: Dict[Pair, List[int]] = {}
    for c in intervals:
        if int(c.get("ladder_version", 0)) != int(version):
            continue
        for pair, (att, acc) in c["pairs"].items():
            tot = out.setdefault(tuple(map(tuple, pair)), [0, 0])
            tot[0] += int(att)
            tot[1] += int(acc)
    return {k: (v[0], v[1]) for k, v in out.items()}


def effective_resistance(n_nodes: int, ia: np.ndarray, ib: np.ndarray, conductance: np.ndarray,
                         source: Sequence[int], sink: Sequence[int]) -> float:
    """
    Effective resistance of a graph between two sets of nodes, each held at one potential.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    ia, ib : numpy.ndarray
        Endpoints of every edge.
    conductance : numpy.ndarray
        Conductance of every edge (positive).
    source, sink : sequence of int
        The two node sets (disjoint).

    Returns
    -------
    float
        1 / (current from `source` to `sink` at unit potential difference).
    """
    L = np.zeros((n_nodes, n_nodes))
    np.add.at(L, (ia, ia), conductance)
    np.add.at(L, (ib, ib), conductance)
    np.add.at(L, (ia, ib), -conductance)
    np.add.at(L, (ib, ia), -conductance)
    fixed = np.zeros(n_nodes, bool)
    v = np.zeros(n_nodes)
    fixed[list(source)] = True
    fixed[list(sink)] = True
    v[list(source)] = 1.0
    free = ~fixed
    if free.any():
        v[free] = np.linalg.solve(L[np.ix_(free, free)], -L[np.ix_(free, fixed)] @ v[fixed])
    current = float((L @ v)[list(source)].sum())
    return 1.0 / current if current > 0 else float("inf")


def _erfc(x: np.ndarray) -> np.ndarray:
    """
    Elementwise complementary error function.

    Parameters
    ----------
    x : numpy.ndarray
        Arguments.

    Returns
    -------
    numpy.ndarray
        erfc(x).
    """
    return np.array([math.erfc(v) for v in np.ravel(x)]).reshape(np.shape(x))


class AcceptanceModel:
    """
    Swap acceptances of a ladder, calibrated on the rates measured at the current one.

    Parameters
    ----------
    temperatures : array_like
        Current ladder, shape (n_T,).
    log10p : array_like
        Current log10 p of every state, shape (n_T, n_mu).
    covariances : numpy.ndarray
        Covariance of (U, -N_s, ...) at every current state, shape (n_T, n_mu, d, d).
    pairs : sequence of tuple
        Neighbouring pairs ``((l, m), (l', m'))`` that were attempted.
    attempts, accepted : array_like
        Swap attempts and accepts of every pair in the calibration window.
    mu_of : callable
        ``mu_of(T, y)``: chemical potential(s) at (T, 10**y).
    lambda_cubed_of_T : callable
        Lambda^3(T).

    Raises
    ------
    ValueError
        If a pair was never attempted or the arrays do not match.
    """

    def __init__(self, temperatures, log10p, covariances: np.ndarray, pairs: Sequence[Pair],
                 attempts, accepted, mu_of: Callable, lambda_cubed_of_T: Callable):
        """
        Calibrate kappa of every pair on its measured acceptance.

        Parameters
        ----------
        temperatures : array_like
            Current ladder, shape (n_T,).
        log10p : array_like
            Current log10 p of every state, shape (n_T, n_mu).
        covariances : numpy.ndarray
            Shape (n_T, n_mu, d, d).
        pairs : sequence of tuple
            Attempted neighbouring pairs.
        attempts, accepted : array_like
            Window counts per pair.
        mu_of : callable
            ``mu_of(T, y)``.
        lambda_cubed_of_T : callable
            Lambda^3(T).

        Raises
        ------
        ValueError
            If a pair was never attempted or the arrays do not match.
        """
        self.T0 = np.asarray(temperatures, dtype=float)
        self.Y0 = np.asarray(log10p, dtype=float)
        self.shape = self.Y0.shape
        if self.T0.shape != (self.shape[0],):
            raise ValueError(f"{len(self.T0)} temperatures for a log10p grid of shape {self.shape}")
        self.mu_of, self.lambda_cubed_of_T = mu_of, lambda_cubed_of_T
        self.pairs = [tuple(map(tuple, p)) for p in pairs]
        self.attempts = np.asarray(attempts, dtype=float)
        self.accepted = np.asarray(accepted, dtype=float)
        if not (len(self.pairs) == len(self.attempts) == len(self.accepted)):
            raise ValueError("pairs, attempts and accepted must have the same length")
        if len(self.pairs) == 0 or np.any(self.attempts <= 0):
            raise ValueError("every pair needs swap attempts to be calibrated")
        nm = self.shape[1]
        self.ia = np.array([a[0] * nm + a[1] for a, _ in self.pairs])
        self.ib = np.array([b[0] * nm + b[1] for _, b in self.pairs])
        self.rate = self.attempts / self.attempts.sum()
        self.measured = self.accepted / self.attempts
        self.target = np.clip(self.measured, 0.5 / self.attempts, 1.0 - 1e-6)
        inv = NormalDist().inv_cdf
        s_meas = np.array([-2.0 * inv(a / 2.0) for a in self.target])
        self.field = CovarianceField(self.T0, self.Y0, covariances)
        C = np.asarray(covariances, dtype=float).reshape((-1,) + covariances.shape[2:])
        s2 = self._s2(self.thetas(self.T0, self.Y0), C)
        self.kappa = np.clip(s_meas ** 2 / np.maximum(s2, 1e-300), *KAPPA_RANGE)

    def thetas(self, T, Y) -> np.ndarray:
        """
        theta of every state of a ladder, flat in grid order.

        Parameters
        ----------
        T : array_like
            Ladder, shape (n_T,).
        Y : array_like
            log10 p, shape (n_T, n_mu).

        Returns
        -------
        numpy.ndarray
            Shape (n_T * n_mu, d).
        """
        return np.array([state_parameters(T[l], Y[l, m], self.mu_of, self.lambda_cubed_of_T)
                         for l in range(self.shape[0]) for m in range(self.shape[1])])

    def _s2(self, th: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        dth^T (C_i + C_j) dth of every pair.

        Parameters
        ----------
        th : numpy.ndarray
            theta per state, shape (K, d).
        C : numpy.ndarray
            Covariance per state, shape (K, d, d).

        Returns
        -------
        numpy.ndarray
            One value per pair.
        """
        d = th[self.ia] - th[self.ib]
        return np.einsum("pi,pij,pj->p", d, C[self.ia] + C[self.ib], d)

    def acceptance(self, T, Y) -> np.ndarray:
        """
        Predicted acceptance of every pair on a ladder.

        Parameters
        ----------
        T : array_like
            Ladder, shape (n_T,).
        Y : array_like
            log10 p, shape (n_T, n_mu).

        Returns
        -------
        numpy.ndarray
            One value per pair (`pairs` order).
        """
        T = np.asarray(T, dtype=float)
        Y = np.asarray(Y, dtype=float)
        C = self.field(np.repeat(T, self.shape[1]), Y.ravel())
        s = np.sqrt(np.maximum(self.kappa * self._s2(self.thetas(T, Y), C), 0.0))
        return _erfc(s / (2.0 * math.sqrt(2.0)))

    def resistances(self, acceptance: np.ndarray) -> Tuple[float, float]:
        """
        Effective resistances R_T (coldest to hottest row) and R_mu (first to last column).

        Parameters
        ----------
        acceptance : numpy.ndarray
            Acceptance of every pair.

        Returns
        -------
        tuple of float
            ``(R_T, R_mu)``; nan along an axis with a single state.
        """
        nT, nm = self.shape
        g = np.maximum(acceptance, ACCEPT_FLOOR) * self.rate
        n = nT * nm
        R_T = (effective_resistance(n, self.ia, self.ib, g, range(nm),
                                    range((nT - 1) * nm, n)) if nT > 1 else float("nan"))
        R_mu = (effective_resistance(n, self.ia, self.ib, g, range(0, n, nm),
                                     range(nm - 1, n, nm)) if nm > 1 else float("nan"))
        return R_T, R_mu


def _loss(R: Tuple[float, float]) -> float:
    """
    ln R_T + ln R_mu, leaving out an axis with a single state.

    Parameters
    ----------
    R : tuple of float
        ``(R_T, R_mu)``.

    Returns
    -------
    float
        The loss.
    """
    return float(sum(math.log(r) for r in R if np.isfinite(r) and r > 0))


def _project(raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Gap fractions within bounds summing to one per row: clip(raw * e^t, lo, hi), t by bisection.

    Parameters
    ----------
    raw : numpy.ndarray
        Positive raw gaps, shape (R, G).
    lo, hi : numpy.ndarray
        Bounds on the fractions, shape (R, G), with sum(lo) <= 1 <= sum(hi) per row.

    Returns
    -------
    numpy.ndarray
        Fractions, shape (R, G).
    """
    lr = np.log(raw)
    a = np.min(np.log(lo) - lr, axis=1)
    b = np.max(np.log(hi) - lr, axis=1)
    for _ in range(64):
        mid = 0.5 * (a + b)
        low = np.clip(np.exp(lr + mid[:, None]), lo, hi).sum(axis=1) < 1.0
        a = np.where(low, mid, a)
        b = np.where(low, b, mid)
    f = np.clip(np.exp(lr + 0.5 * (a + b)[:, None]), lo, hi)
    return f / f.sum(axis=1, keepdims=True)


def _fractions(x: np.ndarray, what: str) -> np.ndarray:
    """
    Gaps of monotonic rows as fractions of each row's span.

    Parameters
    ----------
    x : numpy.ndarray
        Shape (R, n), each row strictly monotonic.
    what : str
        Name for the error message.

    Returns
    -------
    numpy.ndarray
        Shape (R, n - 1), positive, rows summing to 1.

    Raises
    ------
    ValueError
        If a row is not strictly monotonic.
    """
    g = np.diff(x, axis=1)
    span = x[:, -1:] - x[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = g / span
    if not np.all(f > 0):
        raise ValueError(f"the {what} must be strictly monotonic")
    return f


@dataclass
class LadderProposal:
    """
    A proposed ladder and what the model predicts for it.

    Attributes
    ----------
    temperatures : numpy.ndarray
        New ladder, shape (n_T,).
    log10p : numpy.ndarray
        New log10 p of every state, shape (n_T, n_mu).
    mu_grid : numpy.ndarray
        Chemical potentials, shape (n_T, n_mu) or (n_T, n_mu, n_species).
    resistance_before, resistance_after : tuple of float
        ``(R_T, R_mu)`` of the current and the proposed ladder (model).
    pairs : list of tuple
        The calibrated pairs.
    attempts, accepted : numpy.ndarray
        Window counts per pair.
    predicted_before, predicted_after : numpy.ndarray
        Model acceptance per pair at the current (the clipped measured
        rate, by calibration) and at the proposed ladder.
    kappa : numpy.ndarray
        Calibration factor per pair.
    n_evals : int
        Loss evaluations of the optimiser.
    seconds : float
        Wall time of the optimisation.
    improved : bool
        Whether the proposal lowers the loss (if not, it is the current ladder).
    message : str
        Optimiser status.
    """

    temperatures: np.ndarray
    log10p: np.ndarray
    mu_grid: np.ndarray
    resistance_before: Tuple[float, float]
    resistance_after: Tuple[float, float]
    pairs: List[Pair]
    attempts: np.ndarray
    accepted: np.ndarray
    predicted_before: np.ndarray
    predicted_after: np.ndarray
    kappa: np.ndarray
    n_evals: int
    seconds: float
    improved: bool
    message: str

    def to_dict(self) -> dict:
        """
        JSON-able summary (what ``ladder.json`` stores per adaptation).

        Returns
        -------
        dict
            The fields as lists and floats; ``pairs`` as ``[[l, m], [l', m']]``.
        """
        return {"temperatures": self.temperatures.tolist(), "log10p": self.log10p.tolist(),
                "mu_grid": np.asarray(self.mu_grid).tolist(),
                "resistance_before": [float(r) for r in self.resistance_before],
                "resistance_after": [float(r) for r in self.resistance_after],
                "pairs": [[list(a), list(b)] for a, b in self.pairs],
                "attempts": [int(x) for x in self.attempts],
                "accepted": [int(x) for x in self.accepted],
                "measured": [float(c / max(a, 1)) for a, c in zip(self.attempts, self.accepted)],
                "predicted_before": [float(x) for x in self.predicted_before],
                "predicted_after": [float(x) for x in self.predicted_after],
                "kappa": [float(x) for x in self.kappa], "n_evals": int(self.n_evals),
                "seconds": float(self.seconds), "improved": bool(self.improved),
                "message": self.message}


class _Stop(Exception):
    """The optimiser's evaluation or time budget is used up."""


class _Budget:
    """
    Count loss evaluations against an evaluation and wall-time budget.

    Parameters
    ----------
    max_evals : int
        Most evaluations.
    max_time_s : float or None
        Wall-time budget (None: none).
    """

    def __init__(self, max_evals: int, max_time_s: Optional[float]):
        """
        Start the clock.

        Parameters
        ----------
        max_evals : int
            Most evaluations.
        max_time_s : float or None
            Wall-time budget.
        """
        self.max_evals, self.max_time_s = int(max_evals), max_time_s
        self.n, self.t0 = 0, time.time()

    def charge(self) -> None:
        """
        Count one evaluation.

        Raises
        ------
        _Stop
            If the budget is used up.
        """
        if self.n >= self.max_evals or (self.max_time_s is not None
                                        and time.time() - self.t0 > self.max_time_s):
            raise _Stop()
        self.n += 1


def propose_ladder(temperatures, log10p, covariances: np.ndarray, pairs: Sequence[Pair],
                   attempts, accepted, mu_of: Callable, lambda_cubed_of_T: Callable,
                   initial_temperatures=None, initial_log10p=None, trust: float = 1.5,
                   per_row: bool = True, min_gap_ratio: float = 0.25, max_gap_ratio: float = 4.0,
                   max_evals: int = 20000, max_time_s: Optional[float] = None) -> LadderProposal:
    """
    Optimise the ladder for the swap-graph resistances, within trust and coverage bounds.

    Parametrisation: the gaps of ln T, and of y = log10 p along every row
    (``per_row``) or along one row shared by all (product grid), as fractions
    of the fixed span; corners T_0, T_last and every row's first and last y
    are fixed, so order is kept. Each gap stays within [1/trust, trust] x
    its current value and [min_gap_ratio, max_gap_ratio] x its initial value
    (bounds enforced exactly by projection). Minimises ln R_T + ln R_mu of
    `AcceptanceModel` with Powell's method from the current ladder.

    Parameters
    ----------
    temperatures : array_like
        Current ladder, shape (n_T,).
    log10p : array_like
        Current log10 p, shape (n_T, n_mu) (or (n_mu,), shared by every row).
    covariances : numpy.ndarray
        Covariance of (U, -N_s, ...) at every current state, shape (n_T, n_mu, d, d).
    pairs : sequence of tuple
        Attempted neighbouring pairs ``((l, m), (l', m'))``.
    attempts, accepted : array_like
        Swap attempts and accepts of every pair in the calibration window.
    mu_of : callable
        ``mu_of(T, y)``: chemical potential(s) at (T, 10**y).
    lambda_cubed_of_T : callable
        Lambda^3(T).
    initial_temperatures, initial_log10p : array_like, optional
        The ladder the run started from (coverage bounds). Default: the current one.
    trust : float, optional
        Largest factor by which one gap changes. Default 1.5.
    per_row : bool, optional
        One pressure ladder per row (True) or one shared by all rows. Default True.
    min_gap_ratio, max_gap_ratio : float, optional
        Bounds of every gap relative to its initial value. Default 0.25, 4.
    max_evals : int, optional
        Most loss evaluations. Default 20000 (Powell usually converges
        before: about 10000 on an 8 x 8 grid, per row).
    max_time_s : float, optional
        Wall-time budget of the optimisation (makes the result timing
        dependent). Default None (no limit).

    Returns
    -------
    LadderProposal
        The proposal (the current ladder if nothing better was found).

    Raises
    ------
    ValueError
        On inconsistent shapes, a non-monotonic ladder, a product mode with
        different rows, moved corners, or bad bounds.
    """
    from scipy.optimize import minimize

    if trust < 1.0 or not 0.0 < min_gap_ratio <= 1.0 <= max_gap_ratio:
        raise ValueError("need trust >= 1 and 0 < min_gap_ratio <= 1 <= max_gap_ratio")
    T = np.asarray(temperatures, dtype=float)
    nT = len(T)
    Y = np.asarray(log10p, dtype=float)
    Y = np.tile(Y, (nT, 1)) if Y.ndim == 1 else Y
    nm = Y.shape[1]
    T_init = T if initial_temperatures is None else np.asarray(initial_temperatures, dtype=float)
    Y_init = Y if initial_log10p is None else np.asarray(initial_log10p, dtype=float)
    Y_init = np.tile(Y_init, (nT, 1)) if Y_init.ndim == 1 else Y_init
    if Y.shape != (nT, nm) or T_init.shape != T.shape or Y_init.shape != Y.shape:
        raise ValueError(f"inconsistent ladder shapes: T {T.shape}, log10p {Y.shape}, initial "
                         f"T {T_init.shape}, initial log10p {Y_init.shape}")
    if not (np.allclose(T[[0, -1]], T_init[[0, -1]], rtol=1e-9)
            and np.allclose(Y[:, [0, -1]], Y_init[:, [0, -1]], rtol=0, atol=1e-9)):
        raise ValueError("the corners of the ladder (T_0, T_last, every row's first and last "
                         "pressure) must stay those of the initial ladder")
    if not per_row and not np.allclose(Y, Y[:1], rtol=0, atol=1e-12):
        raise ValueError("per_row=False needs one pressure ladder shared by every row")
    model = AcceptanceModel(T, Y, covariances, pairs, attempts, accepted, mu_of,
                            lambda_cubed_of_T)
    lnT, lnT_init = np.log(T)[None], np.log(T_init)[None]
    rows = nT if per_row else 1

    def bounds(cur, init):
        lo = np.maximum(cur / trust, init * min_gap_ratio)
        hi = np.minimum(cur * trust, init * max_gap_ratio)
        return np.minimum(lo, cur), np.maximum(hi, cur)

    blocks = []    # (current fractions, lo, hi) of the T ladder and the p rows
    if nT > 2:
        fT = _fractions(lnT, "temperature ladder")
        blocks.append(("T", fT) + bounds(fT, _fractions(lnT_init, "initial temperature ladder")))
    if nm > 2:
        fY = _fractions(Y[:rows], "pressure ladder of every row")
        blocks.append(("y", fY) + bounds(fY, _fractions(Y_init[:rows], "initial pressure ladder")))
    sizes = [f.shape[0] * (f.shape[1] - 1) for _, f, _, _ in blocks]

    def unpack(z):
        Tn, Yn = T.copy(), Y.copy()
        k = 0
        for (name, f0, lo, hi), n in zip(blocks, sizes):
            R, G = f0.shape
            raw = np.exp(np.concatenate([np.zeros((R, 1)), z[k:k + n].reshape(R, G - 1)], 1))
            k += n
            f = _project(raw, lo, hi)
            if name == "T":
                x = lnT[0, 0] + (lnT[0, -1] - lnT[0, 0]) * np.concatenate([[0.0], np.cumsum(f[0])])
                x[-1] = lnT[0, -1]
                Tn = np.exp(x)
                Tn[[0, -1]] = T[[0, -1]]
            else:
                span = Y[:rows, -1:] - Y[:rows, :1]
                y = Y[:rows, :1] + span * np.concatenate([np.zeros((R, 1)), np.cumsum(f, 1)], 1)
                y[:, -1] = Y[:rows, -1]
                Yn = y if per_row else np.tile(y, (nT, 1))
        return Tn, Yn

    z0 = np.concatenate([np.log(f[:, 1:] / f[:, :1]).ravel() for _, f, _, _ in blocks]) \
        if blocks else np.zeros(0)
    acc0 = model.acceptance(T, Y)
    R0 = model.resistances(acc0)
    loss0 = _loss(R0)
    best = {"loss": loss0, "z": z0}
    budget = _Budget(max_evals, max_time_s)

    def objective(z):
        budget.charge()
        val = _loss(model.resistances(model.acceptance(*unpack(z))))
        if val < best["loss"]:
            best["loss"], best["z"] = val, np.array(z, dtype=float)
        return val

    message = "nothing to optimise (no interior gap)"
    if z0.size:
        try:
            res = minimize(objective, z0, method="Powell",
                           options={"maxfev": int(max_evals), "xtol": 1e-3, "ftol": 1e-6})
            message = str(res.message)
        except _Stop:
            message = "evaluation or time budget used up"
    improved = best["loss"] < loss0 - 1e-9
    Tn, Yn = unpack(best["z"]) if improved else (T.copy(), Y.copy())
    acc = model.acceptance(Tn, Yn)
    mu = np.array([[np.asarray(mu_of(Tn[l], Yn[l, m]), dtype=float) for m in range(nm)]
                   for l in range(nT)])
    return LadderProposal(Tn, Yn, mu, R0, model.resistances(acc), model.pairs,
                          model.attempts.astype(int), model.accepted.astype(int), acc0, acc,
                          model.kappa, budget.n, time.time() - budget.t0, bool(improved),
                          message)


class LadderAdapter:
    """
    Adapt the ladder of a running `ReplicaExchangeGCMC` between chunks.

    At each call of `adapt` (the driver calls it after every chunk while
    `pending`): sum the per-chunk swap counts of the current ladder
    (`window_counts`); if every pair has `min_attempts`, estimate the
    covariances from the samples recorded since the last ladder change,
    run `propose_ladder` and apply the result with
    `ReplicaExchangeGCMC.set_ladder` (which discards the samples of the old
    states). After `chunks` adaptations the ladder is frozen.

    Parameters
    ----------
    mu_of : callable
        ``mu_of(T, y)``: chemical potential(s) at engine temperature T and
        pressure 10**y (e.g. a gas feed's ``mu_at_T_eV``).
    lambda_cubed_of_T : callable
        Lambda^3(T) (the simulation's).
    temperatures : array_like
        The initial (configured) ladder, engine units.
    log10p : array_like
        Its log10 pressures, shape (n_mu,) or (n_T, n_mu).
    chunks : int, optional
        Adaptations before the ladder is frozen. Default 3.
    trust : float, optional
        Largest factor by which a gap changes per adaptation. Default 1.5.
    per_row : bool, optional
        One pressure ladder per temperature (True) or one shared. Default True.
    min_gap_ratio, max_gap_ratio : float, optional
        Bounds of every gap relative to its initial value. Default 0.25, 4.
    min_attempts : int, optional
        Swap attempts every pair needs in the window. Default 20.
    max_evals : int, optional
        Loss evaluations per optimisation. Default 20000.
    max_time_s : float, optional
        Wall-time budget per optimisation. Default None.
    min_samples : int, optional
        Samples every state needs in the window. Default 5.
    kelvin_per_unit : float, optional
        Engine temperature unit in kelvin (1 / k_B in eV/K for real runs),
        for the reports. Default None (engine units only).
    pressure_unit : str, optional
        Unit of 10**y, for the reports. Default None.

    Raises
    ------
    ValueError
        On invalid settings or a ladder with nothing to adapt.
    """

    def __init__(self, mu_of: Callable, lambda_cubed_of_T: Callable, temperatures, log10p,
                 chunks: int = 3, trust: float = 1.5, per_row: bool = True,
                 min_gap_ratio: float = 0.25, max_gap_ratio: float = 4.0, min_attempts: int = 20,
                 max_evals: int = 20000, max_time_s: Optional[float] = None, min_samples: int = 5,
                 kelvin_per_unit: Optional[float] = None, pressure_unit: Optional[str] = None):
        """
        Store the settings and the initial ladder.

        Parameters
        ----------
        mu_of : callable
            ``mu_of(T, y)``.
        lambda_cubed_of_T : callable
            Lambda^3(T).
        temperatures : array_like
            Initial ladder.
        log10p : array_like
            Initial log10 pressures.
        chunks : int, optional
            Adaptations before freezing. Default 3.
        trust : float, optional
            Trust factor. Default 1.5.
        per_row : bool, optional
            Per-row pressure ladders. Default True.
        min_gap_ratio, max_gap_ratio : float, optional
            Coverage bounds. Default 0.25, 4.
        min_attempts : int, optional
            Attempts per pair. Default 20.
        max_evals : int, optional
            Evaluations per optimisation. Default 20000.
        max_time_s : float, optional
            Time per optimisation. Default None.
        min_samples : int, optional
            Samples per state. Default 5.
        kelvin_per_unit : float, optional
            Kelvin per engine temperature unit. Default None.
        pressure_unit : str, optional
            Pressure unit. Default None.

        Raises
        ------
        ValueError
            On invalid settings or a ladder with nothing to adapt.
        """
        self.mu_of, self.lambda_cubed_of_T = mu_of, lambda_cubed_of_T
        self.initial_temperatures = np.asarray(temperatures, dtype=float)
        nT = len(self.initial_temperatures)
        Y = np.asarray(log10p, dtype=float)
        self.initial_log10p = np.tile(Y, (nT, 1)) if Y.ndim == 1 else Y
        if self.initial_log10p.shape[0] != nT:
            raise ValueError(f"log10p has {self.initial_log10p.shape[0]} rows for {nT} "
                             f"temperatures")
        _fractions(np.log(self.initial_temperatures)[None], "temperature ladder")
        if self.initial_log10p.shape[1] > 1:
            _fractions(self.initial_log10p, "pressure ladder")
        if nT <= 2 and self.initial_log10p.shape[1] <= 2:
            raise ValueError("an adaptive ladder needs at least 3 temperatures or 3 pressures "
                             "(the corners are fixed)")
        if int(chunks) < 1 or int(min_attempts) < 1 or int(max_evals) < 1 or int(min_samples) < 2:
            raise ValueError("chunks, min_attempts and max_evals must be >= 1, min_samples >= 2")
        if not float(trust) > 1.0:
            raise ValueError(f"trust must be > 1, got {trust}")
        if not 0.0 < float(min_gap_ratio) <= 1.0 <= float(max_gap_ratio):
            raise ValueError(f"need 0 < min_gap_ratio <= 1 <= max_gap_ratio, got "
                             f"{min_gap_ratio}, {max_gap_ratio}")
        if max_time_s is not None and not float(max_time_s) > 0:
            raise ValueError(f"max_time_s must be positive, got {max_time_s}")
        self.chunks, self.trust, self.per_row = int(chunks), float(trust), bool(per_row)
        self.min_gap_ratio, self.max_gap_ratio = float(min_gap_ratio), float(max_gap_ratio)
        self.min_attempts, self.max_evals = int(min_attempts), int(max_evals)
        self.max_time_s = None if max_time_s is None else float(max_time_s)
        self.min_samples = int(min_samples)
        self.kelvin_per_unit, self.pressure_unit = kelvin_per_unit, pressure_unit

    def settings(self) -> dict:
        """
        The adaptation settings.

        Returns
        -------
        dict
            JSON-able.
        """
        return {"chunks": self.chunks, "trust": self.trust, "per_row": self.per_row,
                "min_gap_ratio": self.min_gap_ratio, "max_gap_ratio": self.max_gap_ratio,
                "min_attempts": self.min_attempts, "max_evals": self.max_evals,
                "max_time_s": self.max_time_s, "min_samples": self.min_samples}

    @staticmethod
    def history(sim) -> List[dict]:
        """
        The adaptations made so far (kept by the simulation, so in its checkpoint).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        list of dict
            One record per adaptation.
        """
        return getattr(sim, "ladder_history", [])

    def pending(self, sim) -> bool:
        """
        Whether adaptations remain (the ladder is not frozen yet).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        bool
            True while fewer than `chunks` adaptations were made.
        """
        return len(self.history(sim)) < self.chunks

    def current_log10p(self, sim) -> np.ndarray:
        """
        log10 p of every current state.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        numpy.ndarray
            Shape (n_T, n_mu).
        """
        for rec in reversed(self.history(sim)):
            if rec.get("log10p") is not None:
                return np.asarray(rec["log10p"], dtype=float)
        return self.initial_log10p.copy()

    def propose(self, sim) -> Tuple[Optional[LadderProposal], str]:
        """
        A ladder proposal from the current window, if the window suffices.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation (its per-chunk pair counts and current samples).

        Returns
        -------
        tuple
            ``(proposal, '')``, or ``(None, reason)`` when the window holds
            too few swap attempts or samples.

        Raises
        ------
        ValueError
            If the simulation's grid does not match the ladder.
        """
        Y = self.current_log10p(sim)
        if tuple(sim.shape) != Y.shape:
            raise ValueError(f"the simulation's grid {tuple(sim.shape)} is not the ladder's "
                             f"{Y.shape}")
        counts = window_counts(sim.pair_acceptance_by_chunk(), getattr(sim, "ladder_version", 0))
        if not counts:
            return None, "no swap attempts recorded on this ladder yet"
        pairs = sorted(counts)
        att = np.array([counts[p][0] for p in pairs])
        if att.min() < self.min_attempts:
            worst = pairs[int(np.argmin(att))]
            return None, (f"pair {worst[0]}-{worst[1]} has {int(att.min())} swap attempts "
                          f"on this ladder, fewer than min_attempts = {self.min_attempts}")
        try:
            C = state_covariances(sim.samples, sim.shape, sim.n_species, self.min_samples)
        except ValueError as err:
            return None, str(err)
        prop = propose_ladder(np.asarray(sim.temperatures, dtype=float), Y, C, pairs, att,
                              [counts[p][1] for p in pairs], self.mu_of, self.lambda_cubed_of_T,
                              self.initial_temperatures, self.initial_log10p, self.trust,
                              self.per_row, self.min_gap_ratio, self.max_gap_ratio,
                              self.max_evals, self.max_time_s)
        return prop, ""

    def apply(self, sim, proposal: LadderProposal, step: Optional[int] = None) -> dict:
        """
        Apply a proposal to the simulation and record it.

        An improving proposal goes through `ReplicaExchangeGCMC.set_ladder`
        (new states, samples discarded); otherwise the ladder is kept and
        only the record is added.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.
        proposal : LadderProposal
            From `propose`.
        step : int, optional
            Steps done (recorded). Default None.

        Returns
        -------
        dict
            The record (also appended to ``sim.ladder_history``).
        """
        rec = proposal.to_dict()
        rec.update({"adaptation": len(self.history(sim)) + 1, "step": step,
                    "changed": proposal.improved, "samples_discarded": 0})
        if self.kelvin_per_unit is not None:
            rec["temperatures_K"] = (proposal.temperatures * self.kelvin_per_unit).tolist()
        rec["pressure_grid"] = (10.0 ** proposal.log10p).tolist()
        if proposal.improved:
            rec["samples_discarded"] = int(max((len(v["U"]) for v in sim.samples.values()),
                                               default=0))
            sim.set_ladder(proposal.temperatures, proposal.mu_grid, info=rec, step=step)
            rec["version"] = int(sim.ladder_version)
        else:
            rec["version"] = int(getattr(sim, "ladder_version", 0))
            sim.ladder_history.append(rec)
        return rec

    def adapt(self, sim, step: Optional[int] = None) -> Tuple[Optional[dict], str]:
        """
        Propose and apply a new ladder if the window suffices.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.
        step : int, optional
            Steps done. Default None.

        Returns
        -------
        tuple
            ``(record, '')`` or ``(None, reason)``.
        """
        if not self.pending(sim):
            return None, "the ladder is frozen"
        prop, reason = self.propose(sim)
        if prop is None:
            return None, reason
        return self.apply(sim, prop, step), ""

    def describe(self, rec: dict) -> str:
        """
        One line summarising an adaptation.

        Parameters
        ----------
        rec : dict
            A record from `apply`.

        Returns
        -------
        str
            The line.
        """
        (rt0, rm0), (rt1, rm1) = rec["resistance_before"], rec["resistance_after"]

        def ratio(a, b):
            return f"{b / a:.2f}x" if np.isfinite(a) and a > 0 else "-"
        frozen = "; ladder frozen" if rec["adaptation"] >= self.chunks else ""
        if not rec["changed"]:
            return (f"ladder {rec['adaptation']}/{self.chunks} at step {rec['step']}: no "
                    f"improvement found, kept ({rec['n_evals']} evaluations){frozen}")
        T = rec.get("temperatures_K") or rec["temperatures"]
        unit = " K" if rec.get("temperatures_K") else ""
        return (f"ladder {rec['adaptation']}/{self.chunks} at step {rec['step']}: model R_T "
                f"{ratio(rt0, rt1)}, R_mu {ratio(rm0, rm1)}; T = ["
                + ", ".join(f"{t:.4g}" for t in T) + f"]{unit}; {rec['n_evals']} evaluations, "
                f"{rec['seconds']:.1f} s; {rec['samples_discarded']} samples per state "
                f"discarded{frozen}")

    def to_json(self, sim) -> dict:
        """
        Everything ``ladder.json`` holds: settings, initial ladder, adaptations.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        dict
            JSON-able.
        """
        init = {"temperatures": self.initial_temperatures.tolist(),
                "log10p": self.initial_log10p.tolist(),
                "pressure_grid": (10.0 ** self.initial_log10p).tolist()}
        if self.kelvin_per_unit is not None:
            init["temperatures_K"] = (self.initial_temperatures * self.kelvin_per_unit).tolist()
        return {"settings": self.settings(), "pressure_unit": self.pressure_unit,
                "initial": init, "version": int(getattr(sim, "ladder_version", 0)),
                "frozen": not self.pending(sim), "history": list(self.history(sim))}

    def metadata_update(self, sim) -> dict:
        """
        The run-metadata entries describing the current ladder.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        dict
            ``temperatures_K`` (or ``temperatures`` without `kelvin_per_unit`),
            ``mu_grid_eV``, ``pressure_grid`` (n_T, n_mu) and ``ladder_version``.
        """
        T = np.asarray(sim.temperatures, dtype=float)
        out = {"mu_grid_eV": np.asarray(sim.mu_grid, dtype=float).tolist(),
               "pressure_grid": (10.0 ** self.current_log10p(sim)).tolist(),
               "ladder_version": int(getattr(sim, "ladder_version", 0))}
        if self.kelvin_per_unit is not None:
            out["temperatures_K"] = (T * self.kelvin_per_unit).tolist()
        else:
            out["temperatures"] = T.tolist()
        return out

    def status_labels(self, sim) -> dict:
        """
        The monitor labels of the current ladder (``status.json``).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        dict
            ``temperatures_K`` (if known) and ``pressure_grid``.
        """
        out = {"pressure_grid": (10.0 ** self.current_log10p(sim)).tolist()}
        if self.kelvin_per_unit is not None:
            out["temperatures_K"] = [float(T) * self.kelvin_per_unit for T in sim.temperatures]
        return out
