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

`LadderES` is the model-free alternative (Kowaguchi et al., J. Chem. Inf.
Model. 62, 6544 (2022), "ES method 1", on the 2D grid): before the run,
each generation mutates the parent ladder's gap fractions into a few
candidates, runs every candidate (and the parent) for a short stretch from
the same configurations, scores the measured swap rates
(`fitness_uniform`, `fitness_resistance`) and keeps the best ladder.
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


class _LadderRecords:
    """
    What the ladder optimisers share: the history kept by the simulation and the outputs.

    Subclasses set ``method``, ``initial_temperatures``, ``initial_log10p``,
    ``kelvin_per_unit``, ``pressure_unit`` and define ``settings`` and ``pending``.
    """

    method = ""

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
        return {"method": self.method, "settings": self.settings(),
                "pressure_unit": self.pressure_unit,
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


class LadderAdapter(_LadderRecords):
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

    method = "model"

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


# ---- evolution strategy (model free) ---------------------------------------------------------
#: Fitness functions of `LadderES`.
ES_FITNESS = ("uniform", "resistance")
#: Swap patterns, as in `regc.engine.SWAP_TYPES`.
_PAIR_TYPES = ("T", "mu", "cross_left", "cross_right")
#: Added to the run seed for the ES's own random stream.
_ES_STREAM = 2022


def pair_type(pair: Pair) -> str:
    """
    The swap pattern a neighbouring pair belongs to.

    Parameters
    ----------
    pair : tuple
        ``((l, m), (l', m'))`` in either order.

    Returns
    -------
    str
        ``'T'`` ((l, m)-(l+1, m)), ``'mu'`` ((l, m)-(l, m+1)), ``'cross_left'``
        ((l, m+1)-(l+1, m)) or ``'cross_right'`` ((l, m)-(l+1, m+1)).

    Raises
    ------
    ValueError
        If the two states are not grid neighbours.
    """
    a, b = sorted(tuple(map(int, k)) for k in pair)
    dl, dm = b[0] - a[0], b[1] - a[1]
    kind = {(1, 0): "T", (0, 1): "mu", (1, -1): "cross_left", (1, 1): "cross_right"}.get((dl, dm))
    if kind is None:
        raise ValueError(f"{pair} is not a pair of grid neighbours")
    return kind


def acceptance_by_type(counts: Dict[Pair, Tuple[int, int]]) -> Dict[str, List[float]]:
    """
    Measured acceptance of every attempted pair, grouped by swap pattern.

    Parameters
    ----------
    counts : dict
        ``{pair: (attempts, accepts)}``.

    Returns
    -------
    dict
        ``{type: [acceptance, ...]}`` for the patterns with attempted pairs
        (pairs without attempts are left out), in sorted pair order.
    """
    out: Dict[str, List[float]] = {}
    for pair in sorted(counts):
        att, acc = counts[pair]
        if att > 0:
            out.setdefault(pair_type(pair), []).append(acc / att)
    return {t: out[t] for t in _PAIR_TYPES if t in out}


def fitness_uniform(counts: Dict[Pair, Tuple[int, int]]) -> float:
    """
    Kowaguchi et al.'s loss (their Eq. 6) per swap pattern: spread of the measured acceptances.

    sum over the patterns present (T, mu, cross_left, cross_right) of
    sum_i (R_i - mean R)^2 over the pattern's pairs, R_i the measured
    acceptance of pair i. Pairs without attempts carry no information and
    are left out (every candidate of a generation sees the same attempted
    pairs: the swap schedule does not depend on the ladder). Lower is
    better; 0 when the acceptance of each pattern is even. Equal but low
    acceptances also score 0: the fixed corners keep the ladder from
    collapsing, but `fitness_resistance` does not have this blind spot.

    Parameters
    ----------
    counts : dict
        ``{pair: (attempts, accepts)}``.

    Returns
    -------
    float
        The fitness.
    """
    return float(sum(np.sum((np.asarray(r) - np.mean(r)) ** 2)
                     for r in acceptance_by_type(counts).values()))


def measured_resistances(shape: Tuple[int, int], counts: Dict[Pair, Tuple[int, int]]
                         ) -> Tuple[float, float]:
    """
    Effective resistances R_T and R_mu of the swap graph with measured conductances.

    As `AcceptanceModel.resistances`, with the measured rates: an edge per
    pair with conductance max(acceptance, `ACCEPT_FLOOR`) x its share of the
    attempts. Every T and mu neighbour pair is an edge, so the graph stays
    connected: one that was not attempted counts as one attempt at the
    floor acceptance.

    Parameters
    ----------
    shape : tuple of int
        Grid shape (n_T, n_mu).
    counts : dict
        ``{pair: (attempts, accepts)}``.

    Returns
    -------
    tuple of float
        ``(R_T, R_mu)``; nan along an axis with a single state.
    """
    nT, nm = int(shape[0]), int(shape[1])
    edges: Dict[Pair, Tuple[int, int]] = {}
    for l in range(nT):
        for m in range(nm):
            for b in ((l + 1, m), (l, m + 1)):
                if b[0] < nT and b[1] < nm:
                    edges[((l, m), b)] = (0, 0)
    for pair, (att, acc) in counts.items():
        a, b = sorted(tuple(map(int, k)) for k in pair)
        edges[(a, b)] = (int(att), int(acc))
    pairs = sorted(edges)
    att = np.array([edges[p][0] for p in pairs], dtype=float)
    acc = np.array([edges[p][1] for p in pairs], dtype=float)
    rate = np.maximum(att, 1.0) / max(att.sum(), 1.0)
    g = np.maximum(acc / np.maximum(att, 1.0), ACCEPT_FLOOR) * rate
    ia = np.array([a[0] * nm + a[1] for a, _ in pairs])
    ib = np.array([b[0] * nm + b[1] for _, b in pairs])
    n = nT * nm
    R_T = (effective_resistance(n, ia, ib, g, range(nm), range((nT - 1) * nm, n))
           if nT > 1 else float("nan"))
    R_mu = (effective_resistance(n, ia, ib, g, range(0, n, nm), range(nm - 1, n, nm))
            if nm > 1 else float("nan"))
    return R_T, R_mu


def fitness_resistance(shape: Tuple[int, int], counts: Dict[Pair, Tuple[int, int]]) -> float:
    """
    ln R_T + ln R_mu of the swap graph with the measured conductances (lower is better).

    Parameters
    ----------
    shape : tuple of int
        Grid shape (n_T, n_mu).
    counts : dict
        ``{pair: (attempts, accepts)}``.

    Returns
    -------
    float
        The fitness (`measured_resistances`; an axis with one state is left out).
    """
    return _loss(measured_resistances(shape, counts))


def fitness_noise(shape: Tuple[int, int], counts: Dict[Pair, Tuple[int, int]],
                  n_boot: int = 200, seed: int = 0) -> Tuple[float, float]:
    """
    Standard errors of `fitness_uniform` and `fitness_resistance` from the binomial swap noise.

    Parametric bootstrap: every pair's accepts are redrawn as
    Binomial(attempts, p), p its measured acceptance clipped to
    [0.5 / attempts, 1 - 0.5 / attempts] (so a pair that never or always
    swapped still has some spread), and both fitnesses recomputed. Only the
    noise of a finite run is covered, not a drift of the system.

    Parameters
    ----------
    shape : tuple of int
        Grid shape (n_T, n_mu).
    counts : dict
        ``{pair: (attempts, accepts)}``.
    n_boot : int, optional
        Resamples. Default 200.
    seed : int, optional
        Seed of the resampling (it does not touch any run stream). Default 0.

    Returns
    -------
    tuple of float
        ``(se_uniform, se_resistance)``: standard deviations over the resamples.
    """
    pairs = [p for p in sorted(counts) if counts[p][0] > 0]
    if not pairs:
        return 0.0, 0.0
    att = np.array([counts[p][0] for p in pairs])
    p_hat = np.clip(np.array([counts[p][1] for p in pairs]) / att, 0.5 / att, 1.0 - 0.5 / att)
    draws = np.random.default_rng(seed).binomial(att, p_hat, size=(int(n_boot), len(pairs)))
    uni, res = [], []
    for row in draws:
        c = {p: (int(a), int(k)) for p, a, k in zip(pairs, att, row)}
        uni.append(fitness_uniform(c))
        res.append(fitness_resistance(shape, c))
    return float(np.std(uni, ddof=1)), float(np.std(res, ddof=1))


def select_survivor(parent_fitness: Optional[float], fitnesses: Sequence[float]) -> Optional[int]:
    """
    ES method 1 selection: the best candidate replaces the parent if it is strictly better.

    Parameters
    ----------
    parent_fitness : float or None
        The parent's fitness (None or nan: any finite candidate wins).
    fitnesses : sequence of float
        The candidates' fitnesses (nan counts as worst).

    Returns
    -------
    int or None
        Index of the surviving candidate, or None if the parent survives.
    """
    f = np.array([np.inf if not np.isfinite(x) else x for x in fitnesses], dtype=float)
    if f.size == 0 or not np.isfinite(f.min()):
        return None
    best = int(np.argmin(f))
    if parent_fitness is None or not np.isfinite(parent_fitness) or f[best] < parent_fitness:
        return best
    return None


class LadderES(_LadderRecords):
    """
    Optimise the ladder by an evolution strategy on measured swap rates, before the run.

    Kowaguchi, Endo, Brumby, Nomura, Yasuoka, J. Chem. Inf. Model. 62, 6544
    (2022), ES method 1, on the (T, p) grid. Each generation (`run_generation`):

    1. `population` candidates are mutated from the parent (`mutate`):
       Gaussian noise of width `sigma` on the log gap fractions of the ln T
       ladder and of every row's log10 p ladder (one shared ladder with
       ``per_row=False``), renormalised, so the corners stay and the order
       is kept.
    2. Every candidate, and the parent (`reevaluate_parent`), runs
       `eval_steps` REGC steps from the same snapshot of the configurations
       (`ReplicaExchangeGCMC.snapshot_state`); its per-pair swap counts give
       both fitnesses (`fitness_uniform`, `fitness_resistance`), and its
       samples are discarded.
    3. The best candidate replaces the parent if its fitness (`fitness`) is
       lower than the parent's (`select_survivor`). The run continues from
       the survivor's configurations at the end of its evaluation
       (equilibrated to its ladder), or the parent's.

    After `generations` generations the ladder is frozen. The history is
    kept in ``sim.ladder_history`` (one record per generation, so it is in
    the checkpoint), with the random state of the ES after the generation.

    Parameters
    ----------
    mu_of : callable
        ``mu_of(T, y)``: chemical potential(s) at engine temperature T and
        pressure 10**y.
    lambda_cubed_of_T : callable
        Lambda^3(T) (only kept for symmetry with `LadderAdapter`).
    temperatures : array_like
        The initial (configured) ladder, engine units.
    log10p : array_like
        Its log10 pressures, shape (n_mu,) or (n_T, n_mu).
    generations : int, optional
        Generations before the ladder is frozen. Default 3.
    population : int, optional
        Candidates per generation. Default 4.
    sigma : float, optional
        Width of the noise on the log gap fractions. Default 0.3.
    eval_steps : int, optional
        REGC steps per evaluation. Default None (the driver uses its chunk).
    fitness : str, optional
        ``'uniform'`` (default) or ``'resistance'``: what selection uses
        (both are recorded).
    per_row : bool, optional
        One pressure ladder per temperature (True) or one shared. Default True.
    reevaluate_parent : bool, optional
        Run the parent again in every generation, so a lucky noisy estimate
        does not survive for ever. Default True.
    min_gap_fraction : float, optional
        Smallest gap of a candidate, as a fraction of its block's span.
        Default 0 (none).
    warmup_steps : int, optional
        REGC steps on the initial ladder before generation 1 (samples and
        swap counts discarded, configurations kept), so the evaluations do
        not measure the drift of a system still equilibrating. Default 0.
    min_attempts_per_pair : int, optional
        Extend every evaluation in blocks of its evaluation length until
        every pair of the swap patterns has this many attempts (the swap
        schedule does not depend on the ladder, so every evaluation of a
        generation gets the same length). Default None (one block).
    eval_steps_max : int, optional
        Longest evaluation with `min_attempts_per_pair`. Default None
        (10 x the evaluation length).
    seed : int, optional
        Seed of the ES's random stream (the run seed). Default 0.
    kelvin_per_unit : float, optional
        Kelvin per engine temperature unit, for the reports. Default None.
    pressure_unit : str, optional
        Unit of 10**y, for the reports. Default None.

    Raises
    ------
    ValueError
        On invalid settings or a ladder with nothing to optimise.
    """

    method = "es"

    def __init__(self, mu_of: Callable, lambda_cubed_of_T: Callable, temperatures, log10p,
                 generations: int = 3, population: int = 4, sigma: float = 0.3,
                 eval_steps: Optional[int] = None, fitness: str = "uniform",
                 per_row: bool = True, reevaluate_parent: bool = True,
                 min_gap_fraction: float = 0.0, warmup_steps: int = 0,
                 min_attempts_per_pair: Optional[int] = None,
                 eval_steps_max: Optional[int] = None, seed: int = 0,
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
        generations : int, optional
            Generations. Default 3.
        population : int, optional
            Candidates per generation. Default 4.
        sigma : float, optional
            Mutation width. Default 0.3.
        eval_steps : int, optional
            Steps per evaluation. Default None.
        fitness : str, optional
            Selection fitness. Default ``'uniform'``.
        per_row : bool, optional
            Per-row pressure ladders. Default True.
        reevaluate_parent : bool, optional
            Re-run the parent every generation. Default True.
        min_gap_fraction : float, optional
            Smallest gap fraction. Default 0.
        warmup_steps : int, optional
            Steps on the initial ladder before generation 1. Default 0.
        min_attempts_per_pair : int, optional
            Attempts every pair needs per evaluation. Default None.
        eval_steps_max : int, optional
            Longest evaluation. Default None (10 x the evaluation length).
        seed : int, optional
            Seed. Default 0.
        kelvin_per_unit : float, optional
            Kelvin per engine temperature unit. Default None.
        pressure_unit : str, optional
            Pressure unit. Default None.

        Raises
        ------
        ValueError
            On invalid settings or a ladder with nothing to optimise.
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
        nm = self.initial_log10p.shape[1]
        if nm > 1:
            _fractions(self.initial_log10p, "pressure ladder")
        if nT <= 2 and nm <= 2:
            raise ValueError("an optimised ladder needs at least 3 temperatures or 3 pressures "
                             "(the corners are fixed)")
        if not per_row and not np.allclose(self.initial_log10p, self.initial_log10p[:1],
                                           rtol=0, atol=1e-12):
            raise ValueError("per_row=False needs one pressure ladder shared by every row")
        if int(generations) < 1 or int(population) < 1:
            raise ValueError("generations and population must be >= 1")
        if eval_steps is not None and int(eval_steps) < 1:
            raise ValueError(f"eval_steps must be >= 1, got {eval_steps}")
        if not float(sigma) > 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if fitness not in ES_FITNESS:
            raise ValueError(f"fitness must be one of {ES_FITNESS}, got {fitness!r}")
        most_gaps = max(n - 1 for n in (nT, nm) if n > 2)
        if not 0.0 <= float(min_gap_fraction) < 1.0 / most_gaps:
            raise ValueError(f"min_gap_fraction must be in [0, 1/{most_gaps}) (the most gaps "
                             f"of one ladder), got {min_gap_fraction}")
        if int(warmup_steps) < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if min_attempts_per_pair is not None and int(min_attempts_per_pair) < 1:
            raise ValueError(f"min_attempts_per_pair must be >= 1, got {min_attempts_per_pair}")
        if eval_steps_max is not None and (int(eval_steps_max) < 1 or (
                eval_steps is not None and int(eval_steps_max) < int(eval_steps))):
            raise ValueError(f"eval_steps_max must be >= eval_steps, got {eval_steps_max}")
        self.warmup_steps = int(warmup_steps)
        self.min_attempts_per_pair = (None if min_attempts_per_pair is None
                                      else int(min_attempts_per_pair))
        self.eval_steps_max = None if eval_steps_max is None else int(eval_steps_max)
        self.generations, self.population = int(generations), int(population)
        self.sigma, self.fitness = float(sigma), str(fitness)
        self.eval_steps = None if eval_steps is None else int(eval_steps)
        self.per_row, self.reevaluate_parent = bool(per_row), bool(reevaluate_parent)
        self.min_gap_fraction, self.seed = float(min_gap_fraction), int(seed)
        self.kelvin_per_unit, self.pressure_unit = kelvin_per_unit, pressure_unit

    def settings(self) -> dict:
        """
        The ES settings.

        Returns
        -------
        dict
            JSON-able.
        """
        return {"method": "es", "generations": self.generations, "population": self.population,
                "sigma": self.sigma, "eval_steps": self.eval_steps, "fitness": self.fitness,
                "per_row": self.per_row, "reevaluate_parent": self.reevaluate_parent,
                "min_gap_fraction": self.min_gap_fraction, "warmup_steps": self.warmup_steps,
                "min_attempts_per_pair": self.min_attempts_per_pair,
                "eval_steps_max": self.eval_steps_max, "seed": self.seed}

    def n_evaluations(self) -> int:
        """
        Evaluations of the whole ES.

        Returns
        -------
        int
            generations x population, plus the parent's (every generation
            with `reevaluate_parent`, else once).
        """
        return self.generations * self.population + (self.generations if self.reevaluate_parent
                                                     else 1)

    def pending(self, sim) -> bool:
        """
        Whether generations remain (the ladder is not frozen yet).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation.

        Returns
        -------
        bool
            True while fewer than `generations` generations were run.
        """
        return len(self.history(sim)) < self.generations

    def rng(self, sim) -> np.random.Generator:
        """
        The ES's random stream where the last generation left it (or fresh from the seed).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation (its ladder history).

        Returns
        -------
        numpy.random.Generator
            The stream.
        """
        rng = np.random.default_rng(np.random.SeedSequence([abs(self.seed), _ES_STREAM]))
        hist = self.history(sim)
        if hist and hist[-1].get("rng_state") is not None:
            rng.bit_generator.state = hist[-1]["rng_state"]
        return rng

    def _perturb(self, f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Mutate gap fractions: Gaussian noise on their logs, renormalised per row.

        Parameters
        ----------
        f : numpy.ndarray
            Gap fractions, shape (R, G), rows summing to 1.
        rng : numpy.random.Generator
            Random stream.

        Returns
        -------
        numpy.ndarray
            New fractions, positive, rows summing to 1 (each >= `min_gap_fraction`).
        """
        z = np.log(f) + self.sigma * rng.standard_normal(f.shape)
        raw = np.exp(z - z.max(axis=1, keepdims=True))
        if self.min_gap_fraction > 0.0:
            return _project(raw, np.full(f.shape, self.min_gap_fraction), np.ones(f.shape))
        return raw / raw.sum(axis=1, keepdims=True)

    def mutate(self, temperatures, log10p, rng: np.random.Generator
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One candidate ladder from a parent.

        Parameters
        ----------
        temperatures : array_like
            Parent ladder, shape (n_T,).
        log10p : array_like
            Parent log10 p, shape (n_T, n_mu).
        rng : numpy.random.Generator
            Random stream (the T block draws first, then the pressure rows).

        Returns
        -------
        tuple of numpy.ndarray
            ``(temperatures, log10p)``: corners and every row's first and
            last pressure unchanged, order kept.

        Raises
        ------
        ValueError
            If the parent is not strictly monotonic or, with
            ``per_row=False``, its rows differ.
        """
        T = np.asarray(temperatures, dtype=float)
        Y = np.asarray(log10p, dtype=float)
        nT, nm = Y.shape
        Tn, Yn = T.copy(), Y.copy()
        if nT > 2:
            lnT = np.log(T)
            f = self._perturb(_fractions(lnT[None], "temperature ladder"), rng)[0]
            x = lnT[0] + (lnT[-1] - lnT[0]) * np.concatenate([[0.0], np.cumsum(f)])
            Tn = np.exp(x)
            Tn[[0, -1]] = T[[0, -1]]
        if nm > 2:
            if not self.per_row and not np.allclose(Y, Y[:1], rtol=0, atol=1e-12):
                raise ValueError("per_row=False needs one pressure ladder shared by every row")
            rows = nT if self.per_row else 1
            f = self._perturb(_fractions(Y[:rows], "pressure ladder of every row"), rng)
            span = Y[:rows, -1:] - Y[:rows, :1]
            y = Y[:rows, :1] + span * np.concatenate([np.zeros((rows, 1)), np.cumsum(f, 1)], 1)
            y[:, [0, -1]] = Y[:rows, [0, -1]]
            Yn = y if self.per_row else np.tile(y, (nT, 1))
        return Tn, Yn

    def mu_grid(self, temperatures, log10p) -> np.ndarray:
        """
        Chemical potentials of a ladder.

        Parameters
        ----------
        temperatures : array_like
            Ladder, shape (n_T,).
        log10p : array_like
            log10 p, shape (n_T, n_mu).

        Returns
        -------
        numpy.ndarray
            Shape (n_T, n_mu) or (n_T, n_mu, n_species).
        """
        T = np.asarray(temperatures, dtype=float)
        Y = np.asarray(log10p, dtype=float)
        return np.array([[np.asarray(self.mu_of(T[l], Y[l, m]), dtype=float)
                          for m in range(Y.shape[1])] for l in range(len(T))])

    def _evaluation(self, label: str, T: np.ndarray, Y: np.ndarray,
                    counts: Dict[Pair, Tuple[int, int]], shape: Tuple[int, int],
                    seconds: float, steps: Optional[int] = None) -> dict:
        """
        The record of one evaluation.

        Parameters
        ----------
        label : str
            ``'parent'`` or ``'candidate k'``.
        T, Y : numpy.ndarray
            The evaluated ladder.
        counts : dict
            Its measured ``{pair: (attempts, accepts)}``.
        shape : tuple of int
            Grid shape.
        seconds : float
            Wall time of the evaluation.
        steps : int, optional
            REGC steps it ran. Default None.

        Returns
        -------
        dict
            JSON-able: ladder, counts, acceptance per pattern, both
            fitnesses and their standard errors (`fitness_noise`).
        """
        pairs = sorted(counts)
        R_T, R_mu = measured_resistances(shape, counts)
        by_type = acceptance_by_type(counts)
        rec = {"label": label, "temperatures": T.tolist(), "log10p": Y.tolist(),
               "pressure_grid": (10.0 ** Y).tolist(),
               "pairs": [[list(a), list(b)] for a, b in pairs],
               "attempts": [int(counts[p][0]) for p in pairs],
               "accepted": [int(counts[p][1]) for p in pairs],
               "acceptance_by_type": {t: float(np.mean(v)) for t, v in by_type.items()},
               "min_acceptance_by_type": {t: float(np.min(v)) for t, v in by_type.items()},
               "fitness_uniform": fitness_uniform(counts),
               "fitness_resistance": _loss((R_T, R_mu)),
               "R_T": float(R_T), "R_mu": float(R_mu), "seconds": float(seconds),
               "steps": None if steps is None else int(steps), "survived": False}
        attempted = [a for a in rec["attempts"] if a > 0]
        rec["min_attempts"] = int(min(attempted)) if attempted else 0
        rec["fitness_uniform_se"], rec["fitness_resistance_se"] = fitness_noise(shape, counts)
        if self.kelvin_per_unit is not None:
            rec["temperatures_K"] = (T * self.kelvin_per_unit).tolist()
        rec["fitness"] = rec["fitness_" + self.fitness]
        rec["fitness_se"] = rec[f"fitness_{self.fitness}_se"]
        return rec

    def describe_evaluation(self, generation: int, rec: dict) -> str:
        """
        One line summarising an evaluation.

        Parameters
        ----------
        generation : int
            Generation number (from 1).
        rec : dict
            From the generation record's ``evaluations``.

        Returns
        -------
        str
            The line.
        """
        short = {"T": "T", "mu": "mu", "cross_left": "xl", "cross_right": "xr"}
        acc = " ".join(f"{short[t]} {a:.2f}" for t, a in rec["acceptance_by_type"].items())
        steps = "" if rec.get("steps") is None else f"{rec['steps']} steps, "
        return (f"ladder ES {generation}/{self.generations} {rec['label']}: fitness uniform "
                f"{rec['fitness_uniform']:.4g} +- {rec['fitness_uniform_se']:.2g}, resistance "
                f"{rec['fitness_resistance']:.4g} +- {rec['fitness_resistance_se']:.2g} "
                f"(R_T {rec['R_T']:.3g}, R_mu {rec['R_mu']:.3g}); acceptance {acc}; "
                f">= {rec['min_attempts']} attempts per pair; {steps}{rec['seconds']:.1f} s")

    def run_generation(self, sim, run_kwargs: Optional[dict] = None, step: Optional[int] = None,
                       eval_steps: Optional[int] = None,
                       progress: Optional[Callable[[str], None]] = None,
                       evaluate: Optional[Callable] = None) -> dict:
        """
        Run one generation on a simulation and continue it from the survivor.

        Generation 1 starts with `warmup_steps` on the initial ladder (record
        ``warmup``). Each evaluation runs the evaluation length, repeated
        while a pair has fewer than `min_attempts_per_pair` attempts (up to
        `eval_steps_max`).

        Parameters
        ----------
        sim : ReplicaExchangeGCMC
            The simulation; it ends on the survivor's ladder (`set_ladder`
            if it changed) with the survivor's end-of-evaluation
            configurations, and its samples, swap and pair counts as before
            the generation. The record is appended to ``sim.ladder_history``.
        run_kwargs : dict, optional
            Passed to `ReplicaExchangeGCMC.run_regc` for every evaluation
            (``x0``, ``diffusion``, ...; not ``n_steps``, ``burn_in_steps``
            or ``verbose``). Default None.
        step : int, optional
            Steps done (recorded). Default None.
        eval_steps : int, optional
            Steps per evaluation if `eval_steps` was not set. Default None.
        progress : callable, optional
            Called with one line per evaluation. Default None.
        evaluate : callable, optional
            ``evaluate(temperatures, log10p, mu_grid) -> counts`` (or
            ``(counts, steps)``) replacing the simulation (tests); the
            configurations are then left alone and there is no warm-up.
            Default None.

        Returns
        -------
        dict
            The generation record: ``evaluations`` (parent first if
            evaluated, then the candidates), ``survivor`` (its index in
            ``evaluations``, or that of the parent), ``changed``, the new
            parent's ladder (``temperatures``, ``log10p``, ``mu_grid``,
            ``pressure_grid``) and ``fitness``, and ``rng_state``.

        Raises
        ------
        ValueError
            If the ladder is frozen, or no evaluation length is known.
        """
        if not self.pending(sim):
            raise ValueError("the ladder is frozen: all generations were run")
        steps = self.eval_steps if self.eval_steps is not None else eval_steps
        if evaluate is None and (steps is None or int(steps) < 1):
            raise ValueError("no evaluation length: set eval_steps")
        hist = self.history(sim)
        g = len(hist) + 1
        rng = self.rng(sim)
        T0 = np.asarray(sim.temperatures, dtype=float)
        Y0 = self.current_log10p(sim)
        if tuple(sim.shape) != Y0.shape:
            raise ValueError(f"the simulation's grid {tuple(sim.shape)} is not the ladder's "
                             f"{Y0.shape}")
        mu0 = np.asarray(sim.mu_grid, dtype=float)
        stored = hist[-1]["fitness"] if hist else None
        ladders = [self.mutate(T0, Y0, rng) for _ in range(self.population)]
        rng_state = rng.bit_generator.state
        jobs = ([("parent", T0, Y0, mu0)] if self.reevaluate_parent or stored is None else [])
        jobs += [(f"candidate {k + 1}", T, Y, self.mu_grid(T, Y))
                 for k, (T, Y) in enumerate(ladders)]

        base, warmup = None, None
        if evaluate is None:
            kwargs = {k: v for k, v in (run_kwargs or {}).items()
                      if k not in ("trajectory_dir", "trajectory_every", "burn_in_steps",
                                   "n_steps", "verbose")}
            kwargs.setdefault("record_every", int(steps))
            types = kwargs.get("swap_types", _PAIR_TYPES)
            all_pairs = {p for t in types for par in (0, 1) for p in sim.swap_pairs(t, par)}
            cap = self.eval_steps_max if self.eval_steps_max is not None else 10 * int(steps)

            def run_counted(n):
                # n steps with nothing recorded (burn-in covers them); swap counts of the run
                before = {k: tuple(v) for k, v in sim.pair_stats.items()}
                sim.run_regc(n_steps=int(n), burn_in_steps=int(n), verbose=False, **kwargs)
                return {k: (int(v[0]) - before.get(k, (0, 0))[0],
                            int(v[1]) - before.get(k, (0, 0))[1])
                        for k, v in sim.pair_stats.items()}

            def evaluate(T, Y, mu):
                sim.restore_state(base)
                sim.assign_states(T, mu)
                counts, done = {}, 0
                while True:
                    n = min(int(steps), cap - done)
                    for k, (a, c) in run_counted(n).items():
                        a0, c0 = counts.get(k, (0, 0))
                        counts[k] = (a0 + a, c0 + c)
                    done += n
                    if (self.min_attempts_per_pair is None or done >= cap
                            or min(counts.get(p, (0, 0))[0] for p in all_pairs)
                            >= self.min_attempts_per_pair):
                        return counts, done

            if g == 1 and self.warmup_steps > 0:
                # equilibrate on the initial ladder; keep only the configurations
                t0 = time.time()
                pre = sim.snapshot_state()
                n_before = float(np.mean([r.system.n for r in sim.grid.values()]))
                counts = run_counted(self.warmup_steps)
                end = sim.snapshot_state()
                sim.restore_state(pre)
                sim.restore_state(end, configurations_only=True)
                by_type = acceptance_by_type(counts)
                warmup = {"steps": self.warmup_steps, "seconds": time.time() - t0,
                          "mean_N_before": n_before,
                          "mean_N_after": float(np.mean([r.system.n for r in sim.grid.values()])),
                          "acceptance_by_type": {t: float(np.mean(v))
                                                 for t, v in by_type.items()}}
                if progress:
                    progress(f"ladder ES warm-up: {self.warmup_steps} steps on the initial "
                             f"ladder, mean N {n_before:.2f} -> {warmup['mean_N_after']:.2f}, "
                             f"acceptance " + " ".join(f"{t} {a:.2f}" for t, a in
                                                      warmup["acceptance_by_type"].items())
                             + f"; {warmup['seconds']:.1f} s")
            base = sim.snapshot_state()

        def score(x):
            return x if np.isfinite(x) else np.inf
        has_parent = jobs[0][0] == "parent"
        evaluations, ends, best = [], {}, None   # end states: the parent's and the best's
        status_path = getattr(sim, "status_path", None)
        sim.status_path = None   # the heartbeat would show the trial runs
        try:
            for i, (label, T, Y, mu) in enumerate(jobs):
                t0 = time.time()
                out = evaluate(T, Y, mu)
                counts, n_run = out if isinstance(out, tuple) else (out, None)
                rec = self._evaluation(label, T, Y, counts, tuple(sim.shape), time.time() - t0,
                                       n_run)
                evaluations.append(rec)
                is_best = label != "parent" and (
                    best is None or score(rec["fitness"]) < score(evaluations[best]["fitness"]))
                best = i if is_best else best
                if base is not None and (label == "parent" or is_best):
                    ends[i] = sim.snapshot_state()
                    for j in [j for j in ends if j != best and not (has_parent and j == 0)]:
                        del ends[j]
                if progress:
                    progress(self.describe_evaluation(g, rec))
        finally:
            sim.status_path = status_path
            if base is not None:
                sim.restore_state(base)   # samples, counts and ladder as before the generation
        offset = 1 if has_parent else 0
        parent_fit = evaluations[0]["fitness"] if has_parent else stored
        pick = select_survivor(parent_fit, [e["fitness"] for e in evaluations[offset:]])
        changed = pick is not None
        win = pick + offset if changed else (0 if has_parent else None)
        if win is not None:
            evaluations[win]["survived"] = True
        T1, Y1 = (jobs[win][1], jobs[win][2]) if changed else (T0, Y0)
        mu1 = jobs[win][3] if changed else mu0
        if base is not None and win is not None:
            sim.restore_state(ends[win], configurations_only=True)
        rec = {"generation": g, "step": step, "eval_steps": None if steps is None else int(steps),
               "evaluations": evaluations, "survivor": win, "changed": changed,
               "parent_fitness": None if parent_fit is None else float(parent_fit),
               "fitness": float(evaluations[win]["fitness"]) if win is not None else stored,
               "temperatures": T1.tolist(), "log10p": Y1.tolist(),
               "mu_grid": np.asarray(mu1).tolist(), "pressure_grid": (10.0 ** Y1).tolist(),
               "fitness_se": (float(evaluations[win]["fitness_se"]) if win is not None
                              else (hist[-1].get("fitness_se") if hist else None)),
               "parent_fitness_se": (float(evaluations[0]["fitness_se"]) if has_parent
                                     else (hist[-1].get("fitness_se") if hist else None)),
               "warmup": warmup, "rng_state": rng_state, "samples_discarded": 0}
        if self.kelvin_per_unit is not None:
            rec["temperatures_K"] = (T1 * self.kelvin_per_unit).tolist()
        if changed:
            rec["samples_discarded"] = int(max((len(v["U"]) for v in sim.samples.values()),
                                               default=0))
            sim.set_ladder(T1, mu1, info=rec, step=step)
            rec["version"] = int(sim.ladder_version)
        else:
            sim.assign_states(T1, mu1)
            sim.reset_round_trips()     # the survivor's trial round trips do not count
            rec["version"] = int(getattr(sim, "ladder_version", 0))
            sim.ladder_history.append(rec)
        return rec

    def describe(self, rec: dict) -> str:
        """
        One line summarising a generation.

        Parameters
        ----------
        rec : dict
            A record from `run_generation`.

        Returns
        -------
        str
            The line.
        """
        g, frozen = rec["generation"], ("; ladder frozen" if rec["generation"] >= self.generations
                                        else "")
        def pm(f, se):
            return "-" if f is None else f"{f:.4g}" + ("" if se is None else f" +- {se:.2g}")
        parent = pm(rec["parent_fitness"], rec.get("parent_fitness_se"))
        if not rec["changed"]:
            return (f"ladder ES generation {g}/{self.generations}: parent kept ({self.fitness} "
                    f"fitness {parent}, no candidate lower){frozen}")
        T = rec.get("temperatures_K") or rec["temperatures"]
        unit = " K" if rec.get("temperatures_K") else ""
        win = rec["evaluations"][rec["survivor"]]
        return (f"ladder ES generation {g}/{self.generations}: {win['label']} survives "
                f"({self.fitness} fitness {pm(win['fitness'], win.get('fitness_se'))} < "
                f"parent {parent}); T = ["
                + ", ".join(f"{t:.4g}" for t in T) + f"]{unit}{frozen}")
