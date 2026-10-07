"""
MBARSurfaceAnalysis: reweight every recorded (U, N) sample from a finished
ReplicaExchangeGCMC run into a smooth coverage(T, mu) / coverage(T, p)
surface phase diagram -- including at (T, mu)/(T, p) values that were never
directly simulated.

Requires: pip install pymbar (imported lazily, only when you instantiate
MBARSurfaceAnalysis -- importing this module itself never requires pymbar).

This only ever consumes `sim.samples[key]['U'/'N']`, `sim.temperatures`,
`sim.mu_grid` (and `sim.lambda_cubed_of_T`, if present), so it works completely unchanged regardless of which
Calculator produced those energies -- LJ, EAM, or an MLIP.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .pressure import invert_mu_of_Tp, mu_from_pressure, p0_reduced_units, pressure_from_mu

if TYPE_CHECKING:
    from .engine import ReplicaExchangeGCMC


def _mbar_free_energies_lbfgs(u_kn: np.ndarray, N_k: np.ndarray) -> np.ndarray:
    """
    MBAR free energies by direct minimization of the convex MBAR objective.

    F(f) = sum_n log sum_k N_k exp(f_k - u_kn) - sum_k N_k f_k, whose
    stationarity condition is exactly the MBAR self-consistent equation.
    L-BFGS gets there in seconds even for ~100+ states with poor overlap,
    where pymbar's adaptive iteration can take many minutes; the result is
    passed to pymbar as `initial_f_k`, so pymbar still does (and checks)
    the final solve.

    Parameters
    ----------
    u_kn : numpy.ndarray
        Reduced potentials, shape (K, N).
    N_k : numpy.ndarray
        Samples per state, shape (K,).

    Returns
    -------
    numpy.ndarray
        f_k with f_0 = 0.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    log_N = np.log(np.asarray(N_k, dtype=float))
    sampled = np.asarray(N_k) > 0
    N_tot = float(np.sum(N_k))

    def objective(g):
        f = np.concatenate([[0.0], g])
        a = (log_N + f)[:, None] - u_kn
        a[~sampled] = -np.inf
        log_den = logsumexp(a, axis=0)
        w = np.exp(a - log_den[None, :])  # posterior weight of state k for sample n
        value = (np.sum(log_den) - np.dot(np.asarray(N_k, dtype=float), f)) / N_tot
        grad = (w.sum(axis=1) - np.asarray(N_k, dtype=float)) / N_tot
        return value, grad[1:]

    g0 = np.zeros(u_kn.shape[0] - 1)
    res = minimize(objective, g0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 20000, "gtol": 1e-12, "ftol": 1e-15})
    return np.concatenate([[0.0], res.x])


class MBARSurfaceAnalysis:
    """Reweight recorded (U, N) samples from a finished
    `ReplicaExchangeGCMC` run into a smooth coverage(T, mu) / coverage(T, p)
    surface phase diagram, at (T, mu)/(T, p) values that were never
    directly simulated."""

    def __init__(self, sim: "ReplicaExchangeGCMC", solver_protocol: str = "robust",
                 p0_of_T: Callable[[float], float] = p0_reduced_units,
                 mu_of_Tp: Optional[Callable[[float, float], float]] = None,
                 p_of_Tmu: Optional[Callable[[float, float], float]] = None,
                 lambda_cubed_of_T: Optional[Callable[[float], float]] = None,
                 reservoir=None, pressure_unit: str = "Pa"):
        """
        Fit an MBAR reweighting model to a finished simulation's samples.

        The simplest way to get pressures right in real units is to pass the
        gas reservoir the simulation's mu grid came from (`reservoir`, a
        `gcmc_re.reservoir.Reservoir`): it sets `mu_of_Tp`/`p_of_Tmu`, with
        pressures in `pressure_unit`. The lower-level options below remain
        for other conventions.

        `p0_of_T` sets the simple atomic-ideal-gas convention
        mu = T*ln(p/p0(T)) (the default; unchanged from before). For a
        reservoir that doesn't fit that closed form -- e.g. a diatomic gas
        via `reservoir.py`, which adds a rotational/vibrational entropy
        term and a dissociation-energy offset -- pass `mu_of_Tp(T, p) ->
        mu` instead; it overrides `p0_of_T` entirely. `p_of_Tmu(T, mu) ->
        p` is its inverse, used by `p_from_mu` and everything built on it
        (`locate_coexistence_p`, `coexistence_curve_Tp`); if you have a
        closed-form inverse (like
        `pressure_from_atomic_mu_reservoir_pascal`) supplying it is exact
        and cheap. If you only have `mu_of_Tp`, `p_from_mu` falls back to
        numerically inverting it (`invert_mu_of_Tp`) -- robust even when
        the root sits at an extreme pressure, just more expensive per call.

        Parameters
        ----------
        sim : ReplicaExchangeGCMC or gcmc_re.io.SampleSet
            The finished simulation (or its saved samples, `load_samples`);
            must have recorded samples (`sim.samples[key]['U'/'N']`) for
            every grid point.
        solver_protocol : str, optional
            The pymbar solver protocol to use. Default is ``"robust"``.
        p0_of_T : callable, optional
            A function ``p0_of_T(T) -> float`` giving the reference
            pressure at temperature T, for the default simple
            ideal-gas mu-pressure relationship. Default is
            `p0_reduced_units`. Ignored if `mu_of_Tp` is given.
        mu_of_Tp : callable, optional
            A function ``mu_of_Tp(T, p) -> mu`` overriding `p0_of_T`
            entirely, for a reservoir with a non-simple mu-pressure
            relationship. Default is None.
        p_of_Tmu : callable, optional
            A function ``p_of_Tmu(T, mu) -> p``, the inverse of
            `mu_of_Tp`. Default is None, which falls back to numerically
            inverting `mu_of_Tp` (or, with neither given, the closed-form
            inverse of `p0_of_T`'s relationship).
        lambda_cubed_of_T : callable, optional
            A function ``lambda_cubed_of_T(T) -> float`` giving the thermal
            de Broglie wavelength cubed used during sampling. It enters the
            reduced potential as ``N * ln(Lambda^3(T))`` (see `_u_of_state`).
            Default is None, which uses ``sim.lambda_cubed_of_T`` if the
            simulation has one, else Lambda^3 = 1 (reduced units).
        reservoir : gcmc_re.reservoir.Reservoir, optional
            Sets `mu_of_Tp` and `p_of_Tmu` (unless given explicitly) from
            `Reservoir.analysis_functions`. Default None.
        pressure_unit : str, optional
            Pressure unit for `reservoir` (``'Pa'``, ``'bar'``, ``'atm'``).
            Default ``'Pa'``.

        Raises
        ------
        ValueError
            If any grid point in `sim` has no recorded samples.
        """
        from pymbar import MBAR  # local import: no hard dependency for the rest of the package
        if reservoir is not None:
            res_mu, res_p = reservoir.analysis_functions(pressure_unit)
            mu_of_Tp = mu_of_Tp or res_mu
            p_of_Tmu = p_of_Tmu or res_p
        self._MBAR_cls = MBAR
        self.p0_of_T = p0_of_T
        self.mu_of_Tp = mu_of_Tp
        self.p_of_Tmu = p_of_Tmu
        if lambda_cubed_of_T is None:
            lambda_cubed_of_T = getattr(sim, "lambda_cubed_of_T", lambda T: 1.0)
        self.lambda_cubed_of_T = lambda_cubed_of_T

        states = sorted(sim.samples.keys())
        self.states = states
        self.betas = np.array([1.0 / sim.temperatures[it] for it, imu in states])
        self.state_mus = np.array([sim.mu_grid[it, imu] for it, imu in states])

        Nk, U_chunks, N_chunks = [], [], []
        for key in states:
            U = np.asarray(sim.samples[key]['U'], dtype=float)
            N = np.asarray(sim.samples[key]['N'], dtype=float)
            if U.shape[0] == 0:
                raise ValueError(
                    f"No recorded samples for state {key}. Run the simulation with "
                    f"record_every set (and cycles beyond burn_in_cycles) before analyzing."
                )
            Nk.append(U.shape[0])
            U_chunks.append(U)
            N_chunks.append(N)

        self.Nk = np.array(Nk)
        self.U_all = np.concatenate(U_chunks)
        self.N_all = np.concatenate(N_chunks)

        K, Ntot = len(states), self.U_all.shape[0]
        u_kn = np.empty((K, Ntot))
        for k in range(K):
            u_kn[k, :] = self._u_of_state(1.0 / self.betas[k], self.state_mus[k])
        self.u_kn = u_kn

        self.mbar = MBAR(u_kn, self.Nk, solver_protocol=solver_protocol,
                         initial_f_k=_mbar_free_energies_lbfgs(u_kn, self.Nk))
        # log of the MBAR mixture denominator, sum_k N_k exp(f_k - u_k(n)),
        # fixed once the f_k are solved: reweighting to any (T, mu) is then
        # one vector operation (see `_log_weights`)
        a = np.log(self.Nk)[:, None] + np.asarray(self.mbar.f_k)[:, None] - u_kn
        a_max = a.max(axis=0)
        self._log_den = a_max + np.log(np.exp(a - a_max).sum(axis=0))

    def _u_of_state(self, T: float, mu: float) -> np.ndarray:
        """
        Compute the reduced potential of every sample at a given (T, mu).

        Parameters
        ----------
        T : float
            Temperature to evaluate the reduced potential at.
        mu : float
            Chemical potential to evaluate the reduced potential at.

        Returns
        -------
        numpy.ndarray
            The reduced potential u(T, mu) = beta*U - beta*mu*N
            + N*ln(Lambda^3(T)) for every recorded sample across all
            simulated states. The Lambda^3 term is the part of the
            grand-canonical weight V^N / (N! Lambda^(3N)) that depends on
            the state (V^N / N! is the same at every state and cancels);
            it vanishes for the reduced-units Lambda^3 = 1.
        """
        beta = 1.0 / T
        return (beta * self.U_all - beta * mu * self.N_all
                + self.N_all * np.log(self.lambda_cubed_of_T(T)))

    def mean_N(self, T: float, mu: float) -> Tuple[float, float]:
        """
        Compute the reweighted mean particle number at arbitrary (T, mu).

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        mu : float
            Chemical potential to reweight to.

        Returns
        -------
        tuple of float
            ``(mean, std_error)``: the MBAR-reweighted mean of N, and its
            standard error, at (T, mu).
        """
        result = self.mbar.compute_expectations(self.N_all, u_kn=self._u_of_state(T, mu))
        # pymbar names its output array 'mu' as in "the estimated mean" --
        # unrelated to our chemical potential mu, just an unlucky key name.
        return float(result['mu'][0]), float(result['sigma'][0])

    def isotherm(self, T: float, mu_grid: np.ndarray, area: float = 1.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one adsorption isotherm: coverage vs mu at fixed T.

        Parameters
        ----------
        T : float
            Temperature of the isotherm.
        mu_grid : numpy.ndarray
            Chemical potentials to evaluate the isotherm at.
        area : float, optional
            Surface area to divide N by, giving areal coverage rather than
            raw particle count. Default is 1.0 (i.e. N itself).

        Returns
        -------
        tuple of numpy.ndarray
            ``(means, sigmas)``, each of shape matching `mu_grid`: the
            reweighted mean coverage and its standard error at each mu.
        """
        means, sigmas = [], []
        for mu in mu_grid:
            m, s = self.mean_N(T, mu)
            means.append(m / area)
            sigmas.append(s / area)
        return np.array(means), np.array(sigmas)

    def phase_diagram(self, T_grid: np.ndarray, mu_grid: np.ndarray, area: float = 1.0
                       ) -> np.ndarray:
        """
        Compute a reweighted coverage(T, mu) surface on a fine mesh.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures to evaluate the surface at.
        mu_grid : numpy.ndarray
            Chemical potentials to evaluate the surface at.
        area : float, optional
            Surface area to divide N by. Default is 1.0.

        Returns
        -------
        numpy.ndarray
            Array of shape (len(T_grid), len(mu_grid)): the reweighted
            mean coverage at each (T, mu) point.
        """
        theta = np.zeros((len(T_grid), len(mu_grid)))
        for i, T in enumerate(T_grid):
            theta[i, :], _ = self.isotherm(T, mu_grid, area=area)
        return theta

    # -- pressure-native wrappers -------------------------------------------
    # Default: mu = T*ln(p/p0(T)) (a simple atomic ideal gas). Overridden
    # entirely by `mu_of_Tp`/`p_of_Tmu` if given at construction (e.g. for a
    # diatomic reservoir) -- see this class's docstring.
    def mu_from_p(self, T: float, p: float) -> float:
        """
        Convert a (temperature, pressure) pair to a chemical potential.

        Parameters
        ----------
        T : float
            Temperature.
        p : float
            Pressure.

        Returns
        -------
        float
            The chemical potential at (T, p), via `mu_of_Tp` if given at
            construction, otherwise `pressure.mu_from_pressure` with
            `p0_of_T`.
        """
        if self.mu_of_Tp is not None:
            return self.mu_of_Tp(T, p)
        return mu_from_pressure(T, p, self.p0_of_T)

    def p_from_mu(self, T: float, mu: float,
                  p_bracket: Tuple[float, float] = (1e-40, 1e100)) -> float:
        """
        Convert a (temperature, chemical potential) pair to a pressure.

        Parameters
        ----------
        T : float
            Temperature.
        mu : float
            Chemical potential.
        p_bracket : tuple of float, optional
            Initial pressure bracket for numerical inversion, used only
            when `p_of_Tmu` is not given but `mu_of_Tp` is (see
            `pressure.invert_mu_of_Tp`). Default is ``(1e-40, 1e100)``.

        Returns
        -------
        float
            The pressure at (T, mu), via `p_of_Tmu` if given at
            construction, else by numerically inverting `mu_of_Tp` if that
            was given, else `pressure.pressure_from_mu` with `p0_of_T`.
        """
        if self.p_of_Tmu is not None:
            return self.p_of_Tmu(T, mu)
        if self.mu_of_Tp is not None:
            return invert_mu_of_Tp(self.mu_of_Tp, T, mu, p_bracket=p_bracket)
        return pressure_from_mu(T, mu, self.p0_of_T)

    def mean_N_p(self, T: float, p: float) -> Tuple[float, float]:
        """
        Compute the reweighted mean particle number at arbitrary (T, p).

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        p : float
            Pressure to reweight to (converted to mu via `mu_from_p`).

        Returns
        -------
        tuple of float
            ``(mean, std_error)``: the MBAR-reweighted mean of N, and its
            standard error, at (T, p).
        """
        return self.mean_N(T, self.mu_from_p(T, p))

    def isotherm_p(self, T: float, p_grid: np.ndarray, area: float = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coverage vs pressure at fixed T -- the isotherm as it's
        normally plotted.

        `p_grid` need not include, or be limited to, pressures that were
        actually part of the simulated ladder.

        Parameters
        ----------
        T : float
            Temperature of the isotherm.
        p_grid : numpy.ndarray
            Pressures to evaluate the isotherm at.
        area : float, optional
            Surface area to divide N by. Default is 1.0.

        Returns
        -------
        tuple of numpy.ndarray
            ``(means, sigmas)``, each of shape matching `p_grid`: the
            reweighted mean coverage and its standard error at each
            pressure.
        """
        mu_grid = np.array([self.mu_from_p(T, p) for p in p_grid])
        return self.isotherm(T, mu_grid, area=area)

    def phase_diagram_Tp(self, T_grid: np.ndarray, p_grid: np.ndarray, area: float = 1.0
                          ) -> np.ndarray:
        """
        Compute the reweighted coverage(T, p) surface phase diagram.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures to evaluate the surface at.
        p_grid : numpy.ndarray
            Pressures to evaluate the surface at.
        area : float, optional
            Surface area to divide N by. Default is 1.0.

        Returns
        -------
        numpy.ndarray
            Array of shape (len(T_grid), len(p_grid)): the reweighted mean
            coverage at each (T, p) point -- the temperature-pressure
            surface phase diagram for the number of adsorbed particles.
        """
        theta = np.zeros((len(T_grid), len(p_grid)))
        for i, T in enumerate(T_grid):
            theta[i, :], _ = self.isotherm_p(T, p_grid, area=area)
        return theta

    def _log_weights(self, T: float, mu: float) -> np.ndarray:
        """
        Normalized MBAR log-weights of every sample at an arbitrary (T, mu).

        Parameters
        ----------
        T, mu : float
            Target state.

        Returns
        -------
        numpy.ndarray
            log w_n with sum_n w_n = 1; <A>(T, mu) = sum_n w_n A_n.
        """
        lw = -self._u_of_state(T, mu) - self._log_den
        lw -= lw.max()
        return lw - np.log(np.exp(lw).sum())

    def most_probable_N(self, T_grid: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
        """
        The most stable phase, labelled by N, on a (T, mu) mesh.

        [ZSG] Sec. II B: with phases defined by the number of particles,
        f_N = -kT ln P(N), so the stable phase is argmax_N P(N).

        Parameters
        ----------
        T_grid, mu_grid : numpy.ndarray
            The mesh axes.

        Returns
        -------
        numpy.ndarray
            Integer array of shape (len(T_grid), len(mu_grid)).
        """
        N_int = np.rint(self.N_all).astype(int)
        out = np.empty((len(T_grid), len(mu_grid)), dtype=int)
        for i, T in enumerate(T_grid):
            for j, mu in enumerate(mu_grid):
                pN = np.bincount(N_int, weights=np.exp(self._log_weights(T, mu)))
                out[i, j] = int(np.argmax(pN))
        return out

    def most_probable_N_Tp(self, T_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
        """
        `most_probable_N` on a (T, p) mesh ([ZSG] Fig. 3).

        Each (T, p) is converted with `mu_from_p` (so `mu_of_Tp`/`p0_of_T`
        given at construction set the reservoir).

        Parameters
        ----------
        T_grid, p_grid : numpy.ndarray
            The mesh axes (p in the units `mu_from_p` expects).

        Returns
        -------
        numpy.ndarray
            Integer array of shape (len(T_grid), len(p_grid)).
        """
        out = np.empty((len(T_grid), len(p_grid)), dtype=int)
        for i, T in enumerate(T_grid):
            mus = np.array([self.mu_from_p(T, p) for p in p_grid])
            out[i, :] = self.most_probable_N(np.array([T]), mus)[0]
        return out

    def lowest_energy_per_N(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lowest sampled energy for every observed particle number.

        Returns
        -------
        tuple of numpy.ndarray
            ``(N_values, E_min)``: the distinct N in the samples and the
            lowest potential energy sampled with that N.
        """
        N_int = np.rint(self.N_all).astype(int)
        Ns = np.unique(N_int)
        return Ns, np.array([self.U_all[N_int == n].min() for n in Ns])

    def lowest_energy_N(self, mu_values: Sequence[float]) -> np.ndarray:
        """
        Phase diagram from lowest energies only ([ZSG] "aiAT@REGC", Eq. 13).

        For each mu, the N minimising E_min(N) - mu*N, with E_min(N) the
        lowest sampled energy at that N (`lowest_energy_per_N`): the
        ab initio atomistic-thermodynamics construction with all vibrational
        (and other entropic) contributions neglected. Compare with
        `most_probable_N` to see their effect. Independent of T at fixed mu.

        Parameters
        ----------
        mu_values : sequence of float
            Chemical potentials.

        Returns
        -------
        numpy.ndarray
            Integer array of the stable N, one per mu.
        """
        Ns, E = self.lowest_energy_per_N()
        mu = np.asarray(mu_values, dtype=float)
        return Ns[np.argmin(E[None, :] - mu[:, None] * Ns[None, :], axis=1)]

    def lowest_energy_N_Tp(self, T_grid: np.ndarray, p_grid: np.ndarray) -> np.ndarray:
        """
        `lowest_energy_N` on a (T, p) mesh.

        Parameters
        ----------
        T_grid, p_grid : numpy.ndarray
            The mesh axes (p in the units `mu_from_p` expects).

        Returns
        -------
        numpy.ndarray
            Integer array of shape (len(T_grid), len(p_grid)).
        """
        return np.array([self.lowest_energy_N([self.mu_from_p(T, p) for p in p_grid])
                         for T in T_grid])

    def pN(self, T: float, mu: float, n_bins: Optional[int] = None
           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reweighted probability mass function of N at (T, mu).

        A bimodal P(N) is the signature of two-phase coexistence (dilute
        film vs. condensed islands/layer) near a 2D condensation
        transition.

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        mu : float
            Chemical potential to reweight to.
        n_bins : int, optional
            Number of histogram bins. Default is None, which uses one bin
            per integer N from 0 to the observed maximum.

        Returns
        -------
        tuple of numpy.ndarray
            ``(centers, probs)``: the bin centers (in N) and the
            reweighted probability mass in each bin.
        """
        n_max = int(self.N_all.max())
        edges = (np.arange(-0.5, n_max + 1.5, 1.0) if n_bins is None
                 else np.linspace(-0.5, n_max + 0.5, n_bins + 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        w = np.exp(self._log_weights(T, mu))
        probs, _ = np.histogram(self.N_all, bins=edges, weights=w)
        return centers, probs

    def pN_p(self, T: float, p: float, n_bins: Optional[int] = None
             ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the reweighted P(N) at (T, p).

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        p : float
            Pressure to reweight to (converted to mu via `mu_from_p`).
        n_bins : int, optional
            Number of histogram bins. Default is None (see `pN`).

        Returns
        -------
        tuple of numpy.ndarray
            ``(centers, probs)`` -- see `pN`.
        """
        return self.pN(T, self.mu_from_p(T, p), n_bins=n_bins)

    def locate_coexistence_mu(self, T: float, mu_bracket: Tuple[float, float],
                               n_iter: int = 25, n_bins: Optional[int] = None) -> float:
        """
        Locate the approximate 2D-condensation coexistence mu at fixed T.

        Via the equal-weight rule: bisect mu until reweighted P(N) has
        equal integrated probability in its low-N ('dilute') and high-N
        ('condensed') basins, split at the valley between the two peaks.

        Caveats (this is a diagnostic tool, not a black box): it assumes
        P(N) genuinely becomes bimodal somewhere in `mu_bracket` -- i.e.
        the potential/conditions you chose actually support 2D
        condensation -- and that the replica ladder has enough mu-points
        near the transition for MBAR to have good sample overlap there. If
        P(N) never shows two peaks across the bracket, this raises rather
        than returning a misleading number.

        Parameters
        ----------
        T : float
            Temperature at which to locate coexistence.
        mu_bracket : tuple of float
            (mu_lo, mu_hi) bracket to bisect within; must contain the
            transition, with a bimodal P(N) detected at both ends.
        n_iter : int, optional
            Number of bisection iterations. Default is 25.
        n_bins : int, optional
            Number of histogram bins for `pN`. Default is None.

        Returns
        -------
        float
            The coexistence chemical potential.

        Raises
        ------
        RuntimeError
            If a bimodal P(N) crossing can't be bracketed within
            `mu_bracket`.
        """
        lo_mu, hi_mu = mu_bracket

        def imbalance(mu: float) -> Optional[float]:
            centers, probs = self.pN(T, mu, n_bins=n_bins)
            is_max = (probs[1:-1] > probs[:-2]) & (probs[1:-1] > probs[2:])
            maxima_idx = np.where(is_max)[0] + 1
            if len(maxima_idx) < 2:
                return None
            top2 = maxima_idx[np.argsort(probs[maxima_idx])[-2:]]
            i_lo, i_hi = sorted(top2)
            valley = i_lo + int(np.argmin(probs[i_lo:i_hi + 1]))
            return float(probs[:valley + 1].sum() - probs[valley + 1:].sum())

        f_lo, f_hi = imbalance(lo_mu), imbalance(hi_mu)
        if f_lo is None or f_hi is None or f_lo * f_hi > 0:
            raise RuntimeError(
                "Could not bracket a bimodal P(N) crossing in this mu range -- either "
                "this system doesn't show 2D condensation here, or the replica ladder "
                "needs more mu-points near the transition for good MBAR overlap."
            )
        for _ in range(n_iter):
            mid = 0.5 * (lo_mu + hi_mu)
            f_mid = imbalance(mid)
            if f_mid is None:
                lo_mu, hi_mu = (mid, hi_mu) if f_hi is not None else (lo_mu, mid)
                continue
            if f_lo * f_mid <= 0:
                hi_mu, f_hi = mid, f_mid
            else:
                lo_mu, f_lo = mid, f_mid
        return 0.5 * (lo_mu + hi_mu)

    def locate_coexistence_p(self, T: float, p_bracket: Tuple[float, float],
                              n_iter: int = 25, n_bins: Optional[int] = None) -> float:
        """
        Locate the 2D-condensation coexistence pressure at fixed T.

        `locate_coexistence_mu`, expressed in pressure: one point on the
        2D condensation line in the (T, p) phase diagram.

        Parameters
        ----------
        T : float
            Temperature at which to locate coexistence.
        p_bracket : tuple of float
            (p_lo, p_hi) bracket to search within (converted to a mu
            bracket via `mu_from_p`).
        n_iter : int, optional
            Number of bisection iterations. Default is 25.
        n_bins : int, optional
            Number of histogram bins for `pN`. Default is None.

        Returns
        -------
        float
            The coexistence pressure.

        Raises
        ------
        RuntimeError
            If a bimodal P(N) crossing can't be bracketed within
            `p_bracket` (via `locate_coexistence_mu`).
        """
        mu_bracket = (self.mu_from_p(T, p_bracket[0]), self.mu_from_p(T, p_bracket[1]))
        mu_coex = self.locate_coexistence_mu(T, mu_bracket, n_iter=n_iter, n_bins=n_bins)
        return self.p_from_mu(T, mu_coex)

    def coexistence_curve_Tp(self, T_grid: np.ndarray, p_bracket, n_iter: int = 25,
                              n_bins: Optional[int] = None, verbose: bool = True
                              ) -> List[Tuple[float, float]]:
        """
        Trace the 2D condensation line p_coex(T) across a temperature grid.

        The phase boundary on the (T, p) diagram.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures to trace the coexistence curve at.
        p_bracket : tuple of float or callable
            Either a fixed (p_lo, p_hi) tuple used at every T, or a
            callable ``T -> (p_lo, p_hi)`` for a bracket that needs to
            move with T.
        n_iter : int, optional
            Number of bisection iterations per T. Default is 25.
        n_bins : int, optional
            Number of histogram bins for `pN`. Default is None.
        verbose : bool, optional
            If True, print a message for any T where no bimodal crossing
            is found. Default is True.

        Returns
        -------
        list of tuple of float
            ``(T, p_coex)`` pairs, one per T in `T_grid` where a
            coexistence pressure was found. Skips (and reports, if
            `verbose`) any T where no bimodal crossing is found, rather
            than raising.
        """
        curve = []
        for T in T_grid:
            bracket = p_bracket(T) if callable(p_bracket) else p_bracket
            try:
                p_coex = self.locate_coexistence_p(T, bracket, n_iter=n_iter, n_bins=n_bins)
                curve.append((T, p_coex))
            except RuntimeError as e:
                if verbose:
                    print(f"  [T={T:.3f}] no coexistence found in bracket {bracket}: {e}")
        return curve
