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

Several exchanged species (a run whose `mu_grid` has a trailing species axis,
see `ReplicaExchangeGCMC`): the reduced potential becomes
beta*U - sum_s (beta*mu_s - ln Lambda_s^3) N_s, built from the per-species
counts ``'N_s'``. Wherever a mu is taken it is then a vector (one entry per
species, e.g. (mu_O, mu_Cl)), a "mu grid" is a path of such vectors, shape
(n_mu, n_species), and `mu_of_Tp(T, p)` returns a vector (e.g. a gas feed of
fixed composition). Observables ``'N_<species>'`` give each species' count.
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from .pressure import invert_mu_of_Tp, mu_from_pressure, p0_reduced_units, pressure_from_mu

if TYPE_CHECKING:
    from .engine import ReplicaExchangeGCMC


def _mbar_free_energies_newton(u_kn: np.ndarray, N_k: np.ndarray, tol: float = 1e-10,
                               max_iter: int = 500, max_step: float = 5.0,
                               f0: Optional[np.ndarray] = None,
                               stall_tol: float = 1e-6) -> np.ndarray:
    """
    MBAR free energies by damped Newton minimization of the convex MBAR objective.

    F(f) = sum_n log sum_k N_k exp(f_k - u_kn) - sum_k N_k f_k, whose
    stationarity condition is exactly the MBAR self-consistent equation.
    Its Hessian, diag(sum_n W_kn) - W W^T (W_kn the posterior weight of
    state k for sample n), is only K x K, so Newton costs barely more per
    step than a gradient method and converges in tens of iterations even
    for ~100+ states with poor overlap and f_k spanning thousands of kT,
    where pymbar's adaptive iteration (or L-BFGS) can take many minutes.
    The start is each state's mean reduced potential over its own samples,
    steps are capped at `max_step` (kT) and Armijo-backtracked, and a tiny
    Levenberg shift keeps the Hessian invertible for states that carry no
    posterior weight yet. The result is passed to pymbar as `initial_f_k`,
    so pymbar still does (and checks) the final solve.

    Parameters
    ----------
    u_kn : numpy.ndarray
        Reduced potentials, shape (K, N).
    N_k : numpy.ndarray
        Samples per state, shape (K,); every state must have samples, in
        the order of the columns of `u_kn`.
    tol : float, optional
        Convergence threshold on the largest gradient component
        (normalized by the total sample count). Default 1e-10.
    max_iter : int, optional
        Maximum number of Newton steps. Default 500.
    max_step : float, optional
        Largest change of any f_k in one step. Default 5.0.
    f0 : numpy.ndarray, optional
        Starting point (e.g. the solution for a closely related data set,
        as in a bootstrap). Default None: each state's mean reduced
        potential over its own samples.
    stall_tol : float, optional
        Stop when the line search accepts only a step below 1e-6 of the
        Newton step (the objective is flat to round-off along it) and the
        gradient is already below this. The objective is then flat to
        round-off: `tol` is out of reach in double precision, and without
        this test the solve would spend `max_iter` steps of ~25 objective
        evaluations each without moving (minutes for 10^5 samples).
        Default 1e-6 (an imbalance of 1e-6 of the samples).

    Returns
    -------
    numpy.ndarray
        f_k with f_0 = 0.
    """
    N_k = np.asarray(N_k, dtype=float)
    if N_k.size == 1:   # one state: f_0 = 0 by convention, nothing to solve
        return np.zeros(1)
    log_N = np.log(N_k)
    N_tot = N_k.sum()

    def evaluate(f):
        a = (log_N + f)[:, None] - u_kn
        a_max = a.max(axis=0)
        W = np.exp(a - a_max)
        den = W.sum(axis=0)
        W /= den  # posterior weight of state k for sample n
        return (np.sum(a_max + np.log(den)) - N_k @ f) / N_tot, W

    if f0 is not None:
        f = np.asarray(f0, dtype=float) - float(f0[0])
    else:
        own = np.repeat(np.arange(len(N_k)), N_k.astype(int))
        f = np.array([u_kn[k, own == k].mean() for k in range(len(N_k))])
        f -= f[0]
    F, W = evaluate(f)
    for _ in range(max_iter):
        w_k = W.sum(axis=1)
        g = (w_k - N_k)[1:] / N_tot
        if np.abs(g).max() < tol:
            break
        H = (np.diag(w_k) - W @ W.T)[1:, 1:] / N_tot
        H[np.diag_indices_from(H)] += 1e-10 + 1e-8 * np.diag(H).max()
        step = np.concatenate([[0.0], np.linalg.solve(H, -g)])
        t = min(1.0, max_step / np.abs(step).max())
        while True:
            F_new, W_new = evaluate(f + t * step)
            if F_new <= F + 1e-4 * t * (g @ step[1:]) or t < 1e-12:
                break
            t *= 0.5
        f, F, W = f + t * step, F_new, W_new
        if t < 1e-6 and np.abs(g).max() < stall_tol:
            # the objective is flat to round-off along the Newton direction: only
            # steps ~1e-7 of the full one pass (by noise), so nothing more is gained
            break
    return f


def _subset_samples(src, indices: dict, lambda_cubed_of_T):
    """
    A `SampleSet` with the given samples of every state (thinned or resampled).

    Parameters
    ----------
    src : ReplicaExchangeGCMC or SampleSet
        The samples.
    indices : dict
        ``{state_key: integer index array}`` (repetition allowed).
    lambda_cubed_of_T : callable
        Lambda^3(T) to store with the set.

    Returns
    -------
    SampleSet
        Same ladder, grid and metadata, with the selected samples.
    """
    from .io import SampleSet
    samples = {key: {name: [vals[i] for i in indices[key]] for name, vals in v.items()}
               for key, v in src.samples.items()}
    mu_grid = getattr(src, "mu_grid", None)
    first = getattr(src, "first_index", 0)
    return SampleSet(list(src.temperatures), None if mu_grid is None else np.array(mu_grid),
                     samples, lambda_cubed_of_T, getattr(src, "exchange_species", None),
                     getattr(src, "observables_meta", None), getattr(src, "n_substrate", None),
                     dict(first) if isinstance(first, dict) else int(first),
                     getattr(src, "state_table", None), getattr(src, "run_labels", None))


def _merged_initial_f_k(u_kn: np.ndarray, N_k: np.ndarray, runs: Sequence[int]) -> np.ndarray:
    """
    Starting free energies for merged runs: each run solved alone, then offset.

    A cold start on the union of several runs can need hundreds of Newton
    steps (free energies spanning thousands of kT, steps capped). Each run
    alone is a small, well-started problem; what is left is one constant
    per run, itself an MBAR problem with one "state" per run (each run's
    mixture density), solved exactly by the same Newton method.

    Parameters
    ----------
    u_kn : numpy.ndarray
        Reduced potentials of all states, shape (K, N).
    N_k : numpy.ndarray
        Samples per state.
    runs : sequence of int
        The run of every state.

    Returns
    -------
    numpy.ndarray
        f_k, shape (K,).
    """
    runs = np.asarray(runs)
    own = np.repeat(np.arange(len(N_k)), np.asarray(N_k, dtype=int))
    f0 = np.zeros(len(N_k))
    groups = [np.flatnonzero(runs == r) for r in np.unique(runs)]
    for ks in groups:
        cols = np.isin(own, ks)
        f0[ks] = _mbar_free_energies_newton(u_kn[np.ix_(ks, cols)], N_k[ks])
    # each run as one state: u_rn = -ln sum_{k in r} N_k exp(f0_k - u_kn) + ln N_r
    N_r = np.array([N_k[ks].sum() for ks in groups], dtype=float)
    u_rn = np.empty((len(groups), u_kn.shape[1]))
    for i, ks in enumerate(groups):
        a = np.log(N_k[ks])[:, None] + f0[ks][:, None] - u_kn[ks]
        a_max = a.max(axis=0)
        u_rn[i] = -(a_max + np.log(np.exp(a - a_max).sum(axis=0))) + np.log(N_r[i])
    c = _mbar_free_energies_newton(u_rn, N_r)
    for i, ks in enumerate(groups):
        f0[ks] += c[i]
    return f0 - f0[0]


class _SolvedMBAR:
    """
    The parts of pymbar's MBAR the analysis uses, from already-solved f_k.

    Bootstrap resamples are solved by warm-started Newton (agreeing with
    pymbar to round-off) and need no further pymbar machinery, whose setup
    would dominate the cost. Uncertainties come from the bootstrap itself,
    so the analytical ``'sigma'`` is not computed (nan).

    Parameters
    ----------
    f_k : numpy.ndarray
        Free energies, shape (K,).
    u_kn : numpy.ndarray
        Reduced potentials, shape (K, N).
    N_k : numpy.ndarray
        Samples per state.
    """

    def __init__(self, f_k: np.ndarray, u_kn: np.ndarray, N_k: np.ndarray):
        """
        Store the solution.

        Parameters
        ----------
        f_k, u_kn, N_k
            See the class docstring.
        """
        self.f_k = np.asarray(f_k, dtype=float)
        self._u_kn = u_kn
        self._N_k = np.asarray(N_k, dtype=float)
        a = np.log(self._N_k)[:, None] + self.f_k[:, None] - u_kn
        a_max = a.max(axis=0)
        self._log_den = a_max + np.log(np.exp(a - a_max).sum(axis=0))

    def compute_expectations(self, A: np.ndarray, u_kn: np.ndarray) -> dict:
        """
        <A> at the state of reduced potential `u_kn`.

        Parameters
        ----------
        A : numpy.ndarray
            Observable per sample.
        u_kn : numpy.ndarray
            Reduced potential of every sample at the target state.

        Returns
        -------
        dict
            ``{'mu': [mean], 'sigma': [nan]}`` (pymbar's keys).
        """
        lw = -np.asarray(u_kn).ravel() - self._log_den
        w = np.exp(lw - lw.max())
        return {"mu": np.array([float(w @ A / w.sum())]), "sigma": np.array([np.nan])}

    def compute_overlap(self) -> dict:
        """
        The overlap matrix O_ij = sum_n W_ni W_nj N_j of the sampled states.

        Returns
        -------
        dict
            ``{'matrix': O}``.
        """
        lw = self.f_k[:, None] - self._u_kn - self._log_den[None, :]
        W = np.exp(lw)                      # each row sums to 1 over the samples
        return {"matrix": (W @ W.T) * self._N_k[None, :]}


class _TargetMBAR(_SolvedMBAR):
    """
    `_SolvedMBAR` for target weights under another potential (`with_potential`).

    Expectations use the sampled states' mixture denominator (infinite for
    samples without an energy under the target potential, i.e. zero weight);
    the overlap matrix is that of the sampled states.

    Parameters
    ----------
    parent : MBAR or _SolvedMBAR
        The sampling analysis's solved MBAR.
    log_den : numpy.ndarray
        Log denominator per sample.
    """

    def __init__(self, parent, log_den: np.ndarray):
        """
        Store the parent and the denominator.

        Parameters
        ----------
        parent, log_den
            See the class docstring.
        """
        self._parent = parent
        self.f_k = np.asarray(parent.f_k, dtype=float)
        self._log_den = log_den

    def compute_overlap(self) -> dict:
        """
        The sampled states' overlap matrix (the parent's).

        Returns
        -------
        dict
            ``{'matrix': O}``.
        """
        return self._parent.compute_overlap()


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
                 reservoir=None, pressure_unit: Optional[str] = None, initial_f_k=None,
                 use_pymbar: Optional[bool] = None, subsample=None):
        """
        Fit an MBAR reweighting model to a finished simulation's samples.

        The simplest way to get pressures right in real units is to pass the
        gas reservoir the simulation's mu grid came from (`reservoir`, a
        `regc.reservoir.Reservoir`): it sets `mu_of_Tp`/`p_of_Tmu`, with
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
        sim : ReplicaExchangeGCMC or regc.io.SampleSet
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
        reservoir : regc.reservoir.Reservoir, optional
            Sets `mu_of_Tp` and `p_of_Tmu` (unless given explicitly) from
            `Reservoir.analysis_functions`. Default None.
        pressure_unit : str, optional
            Pressure unit for `reservoir` (``'Pa'``, ``'bar'``, ``'atm'``).
            Default None: Pa, with a FutureWarning (give it explicitly).
        initial_f_k : array_like, optional
            Starting free energies for the MBAR solver (e.g. those of the
            full data set when analysing a bootstrap resample). Default
            None (a Newton estimate).
        use_pymbar : bool, optional
            Finish (and check) the solve with pymbar, which also provides
            analytical uncertainties. False keeps the Newton solution only
            (it agrees with pymbar to round-off), for the many solves of a
            bootstrap. Default None: True for one run, False for merged runs
            (`regc.io.merge_samples`), whose duplicated or barely overlapping
            states can stall pymbar's solver for many minutes; use the
            bootstrap for their uncertainties.
        subsample : {None, 'g'} or int, optional
            Thin every replica before the analysis: ``'g'`` keeps every
            g-th sample, g being the replica's statistical inefficiency
            (`regc.diagnostics.replica_inefficiencies`), so the kept
            samples are roughly independent and pymbar's analytical
            uncertainties (``mean_N``'s std error) become meaningful; an
            integer keeps every that-many-th sample. Thinning discards
            information: the estimates get noisier, so prefer the full data
            with `regc.diagnostics.bootstrap` error bars for maps, and use
            this for quick, honest single-number errors. Default None.

        Raises
        ------
        ValueError
            If any grid point in `sim` has no recorded samples, or several
            species are exchanged but per-species counts were not recorded.
        """
        from pymbar import MBAR  # local import: no hard dependency for the rest of the package
        if reservoir is not None:
            res_mu, res_p = reservoir.analysis_functions(pressure_unit)  # None: Pa + warning
            mu_of_Tp = mu_of_Tp or res_mu
            p_of_Tmu = p_of_Tmu or res_p
        self._MBAR_cls = MBAR
        self._kept = None
        if subsample is not None:
            lam = lambda_cubed_of_T or getattr(sim, "lambda_cubed_of_T", lambda T: 1.0)
            if subsample == "g":
                from .diagnostics import replica_inefficiencies
                ineff = replica_inefficiencies(sim, lam)
                stride = {k: max(1, int(np.ceil(v["g"]))) for k, v in ineff.items()}
            else:
                stride = {k: max(1, int(subsample)) for k in sim.samples}
            self._kept = {k: np.arange(0, len(v["U"]), stride[k]) for k, v in sim.samples.items()}
            sim = _subset_samples(sim, self._kept, lam)
        # what `resampled` needs to build the same analysis on resampled data
        self._source = sim
        self._options = dict(solver_protocol=solver_protocol, p0_of_T=p0_of_T, mu_of_Tp=mu_of_Tp,
                             p_of_Tmu=p_of_Tmu, lambda_cubed_of_T=lambda_cubed_of_T)
        self.p0_of_T = p0_of_T
        self.mu_of_Tp = mu_of_Tp
        self.p_of_Tmu = p_of_Tmu
        if lambda_cubed_of_T is None:
            lambda_cubed_of_T = getattr(sim, "lambda_cubed_of_T", lambda T: 1.0)
        self.lambda_cubed_of_T = lambda_cubed_of_T

        states = sorted(sim.samples.keys())
        self.states = states
        from .io import state_of
        thermo = [state_of(sim, key) for key in states]
        self.betas = np.array([1.0 / T for T, _ in thermo])
        self.state_mus = np.array([mu for _, mu in thermo])
        self.n_species = self.state_mus.shape[1] if self.state_mus.ndim == 2 else 1
        self.state_table = getattr(sim, "state_table", None)
        self.run_labels = getattr(sim, "run_labels", None)
        self.exchange_species = getattr(sim, "exchange_species", None)
        if self.n_species > 1 and self.mu_of_Tp is None:
            # the simple mu = T ln(p/p0) convention has one mu per p
            self.mu_of_Tp = self._no_species_pressure

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
        # counts per exchanged species, shape (n_samples, n_species)
        if self.n_species == 1:
            self.Ns_all = self.N_all[:, None]
        else:
            if not all('N_s' in sim.samples[key] for key in states):
                raise ValueError("several exchanged species but no 'N_s' samples")
            self.Ns_all = np.concatenate([np.asarray(sim.samples[key]['N_s'], dtype=float)
                                          .reshape(-1, self.n_species) for key in states])
        # adsorbed count: only an observable reweighted with the same weights
        # -- the reduced potential (and so MBAR) is always in terms of N
        self.N_ads_all = (np.concatenate([np.asarray(sim.samples[key]['N_ads'], dtype=float)
                                          for key in states])
                          if all('N_ads' in sim.samples[key] for key in states) else None)

        # recorded structural observables (regc.observables), shape (n_samples, size) each
        self.observables_meta = dict(getattr(sim, "observables_meta", None) or {})
        self.obs_all = {}
        for name in self.observables_meta:
            if not all('obs:' + name in sim.samples[key] for key in states):
                continue
            self.obs_all[name] = np.concatenate(
                [np.asarray(sim.samples[key]['obs:' + name], dtype=float).reshape(
                    len(sim.samples[key]['U']), -1) for key in states])
        self.n_substrate = getattr(sim, "n_substrate", None)
        self.first_index = getattr(sim, "first_index", 0)
        self.potential = None        # the sampling potential (see `with_potential`)
        self._evaluated = None       # mask of samples with an energy under `potential`
        self._offsets = np.concatenate([[0], np.cumsum(self.Nk)])

        K, Ntot = len(states), self.U_all.shape[0]
        u_kn = np.empty((K, Ntot))
        for k in range(K):
            u_kn[k, :] = self._u_of_state(1.0 / self.betas[k], self.state_mus[k])
        self.u_kn = u_kn

        if initial_f_k is None and self.state_table is not None:
            initial_f_k = _merged_initial_f_k(u_kn, self.Nk, [k[0] for k in states])
        f_newton = _mbar_free_energies_newton(u_kn, self.Nk, f0=initial_f_k)
        if use_pymbar is None:
            use_pymbar = self.state_table is None
        self.mbar = (MBAR(u_kn, self.Nk, solver_protocol=solver_protocol, initial_f_k=f_newton)
                     if use_pymbar else _SolvedMBAR(f_newton, u_kn, self.Nk))
        # log of the MBAR mixture denominator, sum_k N_k exp(f_k - u_k(n)),
        # fixed once the f_k are solved: reweighting to any (T, mu) is then
        # one vector operation (see `_log_weights`, `_reweight`)
        a = np.log(self.Nk)[:, None] + np.asarray(self.mbar.f_k)[:, None] - u_kn
        a_max = a.max(axis=0)
        self._log_den = a_max + np.log(np.exp(a - a_max).sum(axis=0))

    def resampled(self, indices: dict) -> "MBARSurfaceAnalysis":
        """
        The same analysis on a resampled data set (for the bootstrap).

        Parameters
        ----------
        indices : dict
            ``{state_key: integer index array}``: the samples of each state
            to keep, with repetition (see `regc.diagnostics.bootstrap`).

        Returns
        -------
        MBARSurfaceAnalysis
            Built with the options of this one, the MBAR solve warm-started
            from its free energies.
        """
        if self.potential is not None:
            return self._parent.resampled(indices).with_potential(self.potential,
                                                                  **self._potential_options)
        sset = _subset_samples(self._source, indices, self.lambda_cubed_of_T)
        opts = dict(self._options, lambda_cubed_of_T=self.lambda_cubed_of_T)
        return MBARSurfaceAnalysis(sset, initial_f_k=self.mbar.f_k, use_pymbar=False, **opts)

    def with_potential(self, name: str, reservoir=None, pressure_unit: Optional[str] = None,
                       mu_of_Tp: Optional[Callable] = None,
                       p_of_Tmu: Optional[Callable] = None) -> "MBARSurfaceAnalysis":
        """
        The same analysis reweighted to another potential.

        The samples were drawn with potential A (the sampling one); their
        energies under B (``samples[key]['E:<name>']``, from
        `regc.records.reevaluate` and `regc.records.attach_energies`)
        give B's weight at any (T, mu),

            w_n^B(T, mu) ~ exp(-beta U_B(n) + beta mu N_n - N_n ln Lambda^3)
                           / sum_k N_k exp(f_k - u_k^A(n)),

        the denominator being A's MBAR mixture over the sampled states
        (Shirts & Chodera 2008, J. Chem. Phys. 129, 124105, Sec. on
        reweighting to unsampled states). Every map of the returned
        analysis (coverage, C_V, observables, effective sample size,
        configurations) is then B's. The effective sample size measures
        how well A's samples cover B's distribution: where it is low, B
        favours configurations A rarely visited, and only a run with B
        itself can say what happens there.

        Samples without a B energy (re-evaluation of a subset) get zero
        weight; choose the subset independently of the configurations
        (every k-th sample) so the estimates stay unbiased.

        Chemical potentials keep their meaning only if both potentials
        place the reservoir's reference (e.g. the isolated molecule) at the
        same energy: if B binds the gas molecule differently, give B's own
        `reservoir` (or ``mu_of_Tp``/``p_of_Tmu``) for the pressure axes.

        Parameters
        ----------
        name : str
            Potential name (the samples hold ``'E:<name>'``).
        reservoir : regc.reservoir.Reservoir, optional
            B's gas reservoir (its pressure <-> mu mapping). Default None:
            keep this analysis's mapping.
        pressure_unit : str, optional
            Pressure unit for `reservoir`. Default None (Pa with a warning).
        mu_of_Tp, p_of_Tmu : callable, optional
            Explicit mapping for B (override `reservoir`). Default None.

        Returns
        -------
        MBARSurfaceAnalysis
            A copy sharing A's free energies, with U = U_B. Its ``mbar``
            gives no analytical uncertainties (use
            `regc.diagnostics.bootstrap`, which resamples and reweights
            again); its overlap matrix is A's (the sampled states).

        Raises
        ------
        ValueError
            If the samples hold no ``'E:<name>'`` energies, or none are finite.
        """
        import copy
        if self.potential is not None:
            raise ValueError(f"this analysis is already reweighted to {self.potential!r}; "
                             f"call with_potential on the sampling analysis")
        col = "E:" + name
        src = self._source
        missing = [k for k in self.states if col not in src.samples[k]]
        if missing:
            have = sorted({c[2:] for v in src.samples.values() for c in v if c.startswith("E:")})
            raise ValueError(f"no {col!r} energies for {len(missing)} states (e.g. {missing[0]}); "
                             f"available: {have}. Re-evaluate the records "
                             f"(regc.records.reevaluate) and attach them "
                             f"(regc.records.attach_energies)")
        U_B = np.concatenate([np.asarray(src.samples[k][col], dtype=float) for k in self.states])
        ok = np.isfinite(U_B)
        if not ok.any():
            raise ValueError(f"no sample has a finite {col!r} energy")
        new = copy.copy(self)
        new.potential = name
        new._evaluated = ok
        new._parent = self
        new.U_all = np.where(ok, U_B, U_B[ok].mean())  # placeholders get zero weight below
        new._log_den = np.where(ok, self._log_den, np.inf)
        new.mbar = _TargetMBAR(self.mbar, new._log_den)
        if reservoir is not None:
            res_mu, res_p = reservoir.analysis_functions(pressure_unit)
            mu_of_Tp, p_of_Tmu = mu_of_Tp or res_mu, p_of_Tmu or res_p
        if mu_of_Tp is not None:
            new.mu_of_Tp, new.p_of_Tmu = mu_of_Tp, p_of_Tmu
        new._potential_options = dict(reservoir=reservoir, pressure_unit=pressure_unit,
                                      mu_of_Tp=mu_of_Tp, p_of_Tmu=p_of_Tmu)
        return new

    def grand_potential_shift(self, T: float, mu) -> Tuple[float, float]:
        """
        Omega_B - Omega_A at one (T, mu), by exponential averaging.

        For an analysis from `with_potential`:
        beta (Omega_B - Omega_A) = -ln < exp(-beta (U_B - U_A)) >_A(T, mu)
        (Zwanzig's free-energy perturbation, in the grand ensemble), with
        the A average MBAR-reweighted to (T, mu) over the evaluated
        samples. Reliable only while the effective sample size stays large.

        Parameters
        ----------
        T : float
            Temperature.
        mu : float or array_like
            Chemical potential (as in this analysis).

        Returns
        -------
        tuple of float
            ``(dOmega, ess)``: the shift in energy units, and the Kish
            effective sample size of B's weights there.

        Raises
        ------
        ValueError
            On the sampling analysis itself.
        """
        if self.potential is None:
            raise ValueError("grand_potential_shift needs an analysis from with_potential")
        parent, ok = self._parent, self._evaluated
        lw_A = parent._log_weights(T, mu)[ok]
        a = lw_A - (self.U_all[ok] - parent.U_all[ok]) / T
        a_max = a.max()
        d_beta_omega = -(a_max + np.log(np.exp(a - a_max).sum())
                         - np.log(np.exp(lw_A - lw_A.max()).sum()) - lw_A.max())
        w = np.exp(self._log_weights(T, mu))
        return float(T * d_beta_omega), float(1.0 / (w @ w))

    def _no_species_pressure(self, T: float, p: float):
        """
        Placeholder `mu_of_Tp` for several species without a reservoir.

        Parameters
        ----------
        T, p : float
            Temperature and pressure.

        Returns
        -------
        numpy.ndarray
            Never returns.

        Raises
        ------
        ValueError
            Always: several species need ``mu_of_Tp`` (or ``reservoir``)
            returning one mu per species.
        """
        raise ValueError("several exchanged species: pass mu_of_Tp(T, p) returning one mu "
                         "per species (e.g. a gas feed of fixed composition)")

    def _log_lambda3(self, T: float) -> np.ndarray:
        """
        ln Lambda^3 of every exchanged species at one temperature.

        Parameters
        ----------
        T : float
            Temperature.

        Returns
        -------
        numpy.ndarray
            Shape (n_species,).
        """
        lam3 = np.asarray(self.lambda_cubed_of_T(T), dtype=float)
        return np.log(np.broadcast_to(lam3, (self.n_species,)))

    def _u_of_state(self, T: float, mu) -> np.ndarray:
        """
        Compute the reduced potential of every sample at a given (T, mu).

        Parameters
        ----------
        T : float
            Temperature to evaluate the reduced potential at.
        mu : float or array_like
            Chemical potential to evaluate the reduced potential at (one
            per species with several exchanged species).

        Returns
        -------
        numpy.ndarray
            The reduced potential u(T, mu) = beta*U - beta*mu*N
            + N*ln(Lambda^3(T)) for every recorded sample across all
            simulated states. The Lambda^3 term is the part of the
            grand-canonical weight V^N / (N! Lambda^(3N)) that depends on
            the state (V^N / N! is the same at every state and cancels);
            it vanishes for the reduced-units Lambda^3 = 1. With several
            species, the mu and Lambda^3 terms are summed over species.
        """
        beta = 1.0 / T
        if self.n_species == 1:
            return (beta * self.U_all - beta * mu * self.N_all
                    + self.N_all * np.log(self.lambda_cubed_of_T(T)))
        mu = np.broadcast_to(np.asarray(mu, dtype=float), (self.n_species,))
        return beta * self.U_all + self.Ns_all @ (self._log_lambda3(T) - beta * mu)

    def mean_N(self, T: float, mu) -> Tuple[float, float]:
        """
        Compute the reweighted mean particle number at arbitrary (T, mu).

        With several exchanged species, N is the total (see
        `mean_N_species` for each species).

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        mu : float or array_like
            Chemical potential to reweight to (one per species with
            several exchanged species).

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

    def mean_N_species(self, T: float, mu) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reweighted mean count of each exchanged species at arbitrary (T, mu).

        Parameters
        ----------
        T : float
            Temperature to reweight to.
        mu : array_like
            One chemical potential per species (a float for one species).

        Returns
        -------
        tuple of numpy.ndarray
            ``(means, std_errors)``, each of shape (n_species,), in the
            order of `exchange_species`.
        """
        u = self._u_of_state(T, mu)
        means, sigmas = [], []
        for s in range(self.n_species):
            result = self.mbar.compute_expectations(self.Ns_all[:, s], u_kn=u)
            means.append(float(result['mu'][0]))
            sigmas.append(float(result['sigma'][0]))
        return np.array(means), np.array(sigmas)

    def isotherm(self, T: float, mu_grid: np.ndarray, area: float = 1.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute one adsorption isotherm: coverage vs mu at fixed T.

        Parameters
        ----------
        T : float
            Temperature of the isotherm.
        mu_grid : numpy.ndarray
            Chemical potentials to evaluate the isotherm at (rows of one
            mu per species, shape (n_mu, n_species), with several).
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

    def _observable(self, name: str) -> np.ndarray:
        """
        Per-sample particle count to reweight.

        Parameters
        ----------
        name : str
            ``'N'``, ``'N_ads'``, or ``'N_<species>'`` (e.g. ``'N_Cl'``)
            for one exchanged species' count.

        Returns
        -------
        numpy.ndarray
            `N_all`, `N_ads_all` or a column of `Ns_all`, shape
            (n_samples,).

        Raises
        ------
        ValueError
            If `name` is unknown, or ``'N_ads'`` was not recorded.
        """
        if name == "N":
            return self.N_all
        if name == "N_ads":
            if self.N_ads_all is None:
                raise ValueError("no 'N_ads' samples: run the simulation with adsorbed_cutoff=")
            return self.N_ads_all
        species = list(self.exchange_species or ())
        if name.startswith("N_") and name[2:] in species:
            return self.Ns_all[:, species.index(name[2:])]
        base, _, comp = name.partition("[")
        if base in self.obs_all:
            values = self.obs_all[base]
            labels = self.observables_meta[base].get("labels", [])
            if comp:
                comp = comp.rstrip("]")
                k = labels.index(comp) if comp in labels else int(comp)
                return values[:, k]
            if values.shape[1] == 1:
                return values[:, 0]
            raise ValueError(f"observable {base!r} has {values.shape[1]} components {labels}: "
                             f"pick one as '{base}[<label>]', or use observable_map / "
                             f"mean_coordination / radial_distribution")
        choices = (["N", "N_ads"] + [f"N_{sp}" for sp in species]
                   + [f"{o}[<component>]" for o in self.obs_all])
        raise ValueError(f"unknown observable {name!r}; choose one of {choices}")

    def _Tmu_mesh(self, T_grid: np.ndarray, mu_grid: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (T, mu) meshes for a temperature axis and a mu axis (or mu path).

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures, length n_T.
        mu_grid : numpy.ndarray
            Chemical potentials, length n_mu; with several species a path
            of mu vectors, shape (n_mu, n_species).

        Returns
        -------
        tuple of numpy.ndarray
            ``(T_mesh, mu_mesh)``, shapes (n_T, n_mu) and (n_T, n_mu) or
            (n_T, n_mu, n_species).
        """
        if self.n_species == 1:
            return np.meshgrid(T_grid, mu_grid, indexing="ij")
        path = np.asarray(mu_grid, dtype=float).reshape(-1, self.n_species)
        T_mesh = np.broadcast_to(np.asarray(T_grid, dtype=float)[:, None], (len(T_grid), len(path)))
        return T_mesh, np.broadcast_to(path[None], T_mesh.shape + (self.n_species,))

    def phase_diagram(self, T_grid: np.ndarray, mu_grid: np.ndarray, area: float = 1.0,
                      observable: str = "N") -> np.ndarray:
        """
        Compute a reweighted coverage(T, mu) surface on a fine mesh.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures to evaluate the surface at.
        mu_grid : numpy.ndarray
            Chemical potentials to evaluate the surface at; with several
            species a path of mu vectors, shape (n_mu, n_species).
        area : float, optional
            Surface area to divide N by. Default is 1.0.
        observable : str, optional
            ``'N'`` (all mobile particles, the default), ``'N_ads'``
            (adsorbed ones only; needs a run with ``adsorbed_cutoff``) or
            ``'N_<species>'``.

        Returns
        -------
        numpy.ndarray
            Array of shape (len(T_grid), len(mu_grid)): the reweighted
            mean coverage at each (T, mu) point.
        """
        T_mesh, mu_mesh = self._Tmu_mesh(T_grid, mu_grid)
        return self._reweight(T_mesh, mu_mesh, self._observable(observable)) / area

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

        Raises
        ------
        ValueError
            With several exchanged species and no `p_of_Tmu`.
        """
        if self.p_of_Tmu is not None:
            return self.p_of_Tmu(T, mu)
        if self.n_species > 1:
            raise ValueError("several exchanged species: pass p_of_Tmu(T, mu_vector) -> p")
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

    def phase_diagram_Tp(self, T_grid: np.ndarray, p_grid: np.ndarray, area: float = 1.0,
                         observable: str = "N") -> np.ndarray:
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
        observable : str, optional
            ``'N'`` (all mobile particles, the default) or ``'N_ads'``
            (adsorbed ones only; needs a run with ``adsorbed_cutoff``).

        Returns
        -------
        numpy.ndarray
            Array of shape (len(T_grid), len(p_grid)): the reweighted mean
            coverage at each (T, p) point -- the temperature-pressure
            surface phase diagram for the number of adsorbed particles.
        """
        return self._reweight(*self._Tp_mesh(T_grid, p_grid), self._observable(observable)) / area

    def _log_weights(self, T: float, mu: float) -> np.ndarray:
        """
        Normalized MBAR log-weights of every sample at an arbitrary (T, mu).

        Parameters
        ----------
        T : float
            Target temperature.
        mu : float or array_like
            Target chemical potential (one per species with several).

        Returns
        -------
        numpy.ndarray
            log w_n with sum_n w_n = 1; <A>(T, mu) = sum_n w_n A_n.
        """
        lw = -self._u_of_state(T, mu) - self._log_den
        lw -= lw.max()
        return lw - np.log(np.exp(lw).sum())

    def _reweight(self, T: np.ndarray, mu: np.ndarray, A: np.ndarray,
                  max_elements: int = 2 ** 24) -> np.ndarray:
        """
        MBAR expectations of sample observables at many (T, mu) at once.

        The reduced potential is linear in (U, N) -- u = beta*U + c*N with
        c = -beta*mu + ln Lambda^3(T), a sum over species with several --
        so the log-weights of every sample
        at a whole block of target states are one rank-2 outer product
        (`_log_weights`, batched). Blocks are sized to keep the weight
        matrix under `max_elements` entries.

        Parameters
        ----------
        T, mu : numpy.ndarray
            Target states, any (common) shape S; with several species
            `mu` has shape S + (n_species,).
        A : numpy.ndarray
            Observable per sample, shape (n_samples,) or (n_samples, m).
        max_elements : int, optional
            Largest weight-matrix block. Default 2**24 (128 MB of float64).

        Returns
        -------
        numpy.ndarray
            <A> at every target state, shape S (or S + (m,)).
        """
        T = np.asarray(T, dtype=float)
        A = np.asarray(A, dtype=float)
        out = np.empty((T.size,) + A.shape[1:])
        for sl, w in self._weight_blocks(T, mu, max_elements):
            out[sl] = w @ A
        return out.reshape(T.shape + A.shape[1:])

    def _weight_blocks(self, T: np.ndarray, mu: np.ndarray, max_elements: int = 2 ** 24):
        """
        Normalized MBAR weights of every sample, in blocks of target states.

        Parameters
        ----------
        T, mu : numpy.ndarray
            Target states, common shape S (`mu` with a trailing species
            axis with several species).
        max_elements : int, optional
            Largest weight-matrix block. Default 2**24.

        Yields
        ------
        tuple
            ``(slice, w)``: a slice of the flattened target states and
            their weights, shape (n_block, n_samples), each row summing to 1.
        """
        T = np.asarray(T, dtype=float)
        S = self.n_species
        mu = np.broadcast_to(np.asarray(mu, dtype=float), T.shape + ((S,) if S > 1 else ())
                             ).reshape(-1, S)
        T_flat = T.ravel()
        T_unique, inverse = np.unique(T_flat, return_inverse=True)
        log_lam3 = np.array([self._log_lambda3(t) for t in T_unique])[inverse.ravel()]
        beta = 1.0 / T_flat
        c = -beta[:, None] * mu + log_lam3  # (n_states, n_species)
        block = max(1, max_elements // self.U_all.size)
        for i in range(0, T_flat.size, block):
            sl = slice(i, i + block)
            cN = np.outer(c[sl, 0], self.N_all) if S == 1 else c[sl] @ self.Ns_all.T
            lw = -(np.outer(beta[sl], self.U_all) + cN) - self._log_den
            lw -= lw.max(axis=1, keepdims=True)
            w = np.exp(lw, out=lw)
            w /= w.sum(axis=1, keepdims=True)
            yield sl, w

    # -- meshes ------------------------------------------------------------------------
    def _mesh(self, T_grid: np.ndarray, grid: np.ndarray, pressure: bool
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (T, mu) meshes for a (T, p) or (T, mu) grid.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures.
        grid : numpy.ndarray
            Pressures (``pressure=True``, in the analysis's pressure unit)
            or chemical potentials.
        pressure : bool
            Which kind `grid` is.

        Returns
        -------
        tuple of numpy.ndarray
            ``(T_mesh, mu_mesh)``.
        """
        T_grid = np.atleast_1d(np.asarray(T_grid, dtype=float))
        if pressure:
            return self._Tp_mesh(T_grid, np.atleast_1d(np.asarray(grid, dtype=float)))
        grid = np.asarray(grid, dtype=float)
        if self.n_species == 1:
            grid = np.atleast_1d(grid)
        return self._Tmu_mesh(T_grid, grid)

    # -- reliability --------------------------------------------------------------------
    def effective_sample_size(self, T_grid: np.ndarray, grid: np.ndarray,
                              pressure: bool = False) -> np.ndarray:
        """
        Kish effective number of samples, 1 / sum_n w_n^2, on a mesh.

        It counts how many samples effectively carry the reweighted
        average at a target state: about the number of (independent)
        samples of a simulated state there, and ~1 where one sample
        dominates (extrapolation beyond the sampled ladder, or a state
        between poorly overlapping replicas). Mask cells below a few tens
        before trusting a map; correlated samples make the true number
        smaller still (by the statistical inefficiency).

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures.
        grid : numpy.ndarray
            Pressures (``pressure=True``) or chemical potentials.
        pressure : bool, optional
            Whether `grid` holds pressures. Default False.

        Returns
        -------
        numpy.ndarray
            Shape (len(T_grid), len(grid)).
        """
        T_mesh, mu_mesh = self._mesh(T_grid, grid, pressure)
        out = np.empty(T_mesh.size)
        for sl, w in self._weight_blocks(T_mesh, mu_mesh):
            out[sl] = 1.0 / np.einsum("ij,ij->i", w, w)
        return out.reshape(T_mesh.shape)

    # -- structural observables -----------------------------------------------------------
    def _obs(self, name: str, kind: Optional[str] = None) -> np.ndarray:
        """
        The per-sample values of a recorded observable.

        Parameters
        ----------
        name : str
            Observable name.
        kind : str, optional
            Required metadata type (e.g. ``'RadialDistribution'``). Default
            None (any).

        Returns
        -------
        numpy.ndarray
            Shape (n_samples, size).

        Raises
        ------
        ValueError
            If it was not recorded, or is of another kind.
        """
        if name not in self.obs_all:
            raise ValueError(f"no observable {name!r} recorded; recorded: {sorted(self.obs_all)} "
                             f"(pass observables= to ReplicaExchangeGCMC)")
        got = self.observables_meta[name].get("type")
        if kind is not None and got not in (kind if isinstance(kind, tuple) else (kind,)):
            raise ValueError(f"observable {name!r} is a {got}, not a {kind}")
        return self.obs_all[name]

    def observable_map(self, name: str, T_grid: np.ndarray, grid: np.ndarray,
                       pressure: bool = False) -> np.ndarray:
        """
        Reweighted mean of every component of a recorded observable on a mesh.

        Parameters
        ----------
        name : str
            Observable name (``observables=`` of the run).
        T_grid : numpy.ndarray
            Temperatures.
        grid : numpy.ndarray
            Pressures (``pressure=True``) or chemical potentials.
        pressure : bool, optional
            Whether `grid` holds pressures. Default False.

        Returns
        -------
        numpy.ndarray
            Shape (len(T_grid), len(grid), size).
        """
        return self._reweight(*self._mesh(T_grid, grid, pressure), self._obs(name))

    def observable_mean(self, name: str, T: float, mu) -> np.ndarray:
        """
        Reweighted mean of every component of a recorded observable at one (T, mu).

        Parameters
        ----------
        name : str
            Observable name.
        T : float
            Temperature.
        mu : float or array_like
            Chemical potential (one per species with several).

        Returns
        -------
        numpy.ndarray
            Shape (size,).
        """
        return self._reweight(np.array([T]), np.asarray(mu, dtype=float)[None], self._obs(name))[0]

    def mean_coordination(self, name: str, T_grid: np.ndarray, grid: np.ndarray,
                          pressure: bool = False) -> np.ndarray:
        """
        Reweighted mean coordination number, <bonds> / <centers>, on a mesh.

        Parameters
        ----------
        name : str
            A `regc.observables.CoordinationNumber` observable.
        T_grid : numpy.ndarray
            Temperatures.
        grid : numpy.ndarray
            Pressures (``pressure=True``) or chemical potentials.
        pressure : bool, optional
            Whether `grid` holds pressures. Default False.

        Returns
        -------
        numpy.ndarray
            Shape (len(T_grid), len(grid)); NaN where there are (almost)
            no centre atoms (<centers> < 1e-6).
        """
        self._obs(name, "CoordinationNumber")
        m = self.observable_map(name, T_grid, grid, pressure)
        bonds, centers = m[..., 0], m[..., 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(centers > 1e-6, bonds / centers, np.nan)

    def coordination_distribution(self, name: str, T: float, mu) -> np.ndarray:
        """
        Probability that a centre atom has coordination 0, 1, ..., max_cn.

        Parameters
        ----------
        name : str
            A `regc.observables.CoordinationHistogram` observable.
        T : float
            Temperature.
        mu : float or array_like
            Chemical potential.

        Returns
        -------
        numpy.ndarray
            Shape (max_cn + 1,), summing to 1 (all zeros without centres);
            the last entry is ``P(CN >= max_cn)``.
        """
        self._obs(name, "CoordinationHistogram")
        h = self.observable_mean(name, T, mu)
        total = h.sum()
        return h / total if total > 0 else h

    def radial_distribution(self, name: str, T: float, mu, normalize: str = "per_center"
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reweighted radial distribution at one (T, mu).

        Parameters
        ----------
        name : str
            A `regc.observables.RadialDistribution` observable.
        T : float
            Temperature.
        mu : float or array_like
            Chemical potential.
        normalize : {'per_center', 'density', 'g'}, optional
            ``'per_center'``: n(r) = <H(r)> / (<n_a> dr), neighbours of a
            centre per unit distance (its integral to r_c is the mean
            coordination; for surfaces and clusters). ``'density'``: n(r)
            / (4 pi r^2), local density of b around a (1/A^3). ``'g'``:
            density / (<n_b> / V), the bulk g(r) (V includes any vacuum,
            so only meaningful in bulk). Default ``'per_center'``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(r, values)``: bin centres (A) and the distribution (zeros if
            there are no a atoms).

        Raises
        ------
        ValueError
            On an unknown `normalize`.
        """
        self._obs(name, "RadialDistribution")
        meta = self.observables_meta[name]
        edges = np.asarray(meta["edges"], dtype=float)
        n_bins = len(edges) - 1
        m = self.observable_mean(name, T, mu)
        h, n_a, n_b, volume = m[:n_bins], m[n_bins], m[n_bins + 1], m[n_bins + 2]
        r = 0.5 * (edges[1:] + edges[:-1])
        dr = np.diff(edges)
        per_center = h / (n_a * dr) if n_a > 0 else np.zeros(n_bins)
        if normalize == "per_center":
            return r, per_center
        shell = 4.0 * np.pi * r ** 2
        if normalize == "density":
            return r, per_center / shell
        if normalize == "g":
            rho = n_b / volume if (volume > 0 and n_b > 0) else np.nan
            return r, per_center / shell / rho
        raise ValueError(f"normalize must be 'per_center', 'density' or 'g', got {normalize!r}")

    # -- heat capacity ([ZSG] PRL 128, 246101 (2022), Eq. (1)) ----------------------------
    def heat_capacity(self, T_grid: np.ndarray, grid: np.ndarray, pressure: bool = False,
                      kinetic: bool = True) -> np.ndarray:
        """
        Heat capacity C_V = (<E^2> - <E>^2) / (k_B T^2) on a mesh, in units of k_B.

        [ZSG] PRL 128, 246101, Eq. (1): E is the total energy, potential
        plus kinetic, and the averages are MBAR-reweighted to every (T, mu)
        or (T, p). Ridges of C_V along T at fixed p, or along p at fixed T,
        mark first-order phase boundaries (`heat_capacity_ridges`); a
        change of coverage without a ridge is a smooth crossover.

        The kinetic energy of the n moving atoms (mobile + displaceable
        substrate) is classical and independent of the configuration given
        n: <K | n> = (3/2) n k_B T and Var(K | n) = (3/2) n (k_B T)^2. So,
        with x = U + (3/2) k_B T n,

            Var(E) = Var(x) + (3/2) (k_B T)^2 <n>,

        computed from the recorded U and N alone (MC diffusion never
        samples momenta, and it does not need to).

        In the grand ensemble (momenta included, Lambda^3 proportional to
        T^(-3/2)) this fluctuation formula is exactly d<E>/dT at fixed
        beta*mu -- not at fixed mu, p or N; its ridges still locate the
        transitions, which is what it is for. It is a variance, so it needs
        more samples than <N> to converge: check `effective_sample_size`
        and compare halves of the run.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures (k_B T, in the energy unit of U).
        grid : numpy.ndarray
            Pressures (``pressure=True``) or chemical potentials.
        pressure : bool, optional
            Whether `grid` holds pressures. Default False.
        kinetic : bool, optional
            Include the kinetic energy (as [ZSG]). Default True; False
            gives the configurational part Var(U) / (k_B T)^2.

        Returns
        -------
        numpy.ndarray
            C_V / k_B, shape (len(T_grid), len(grid)).
        """
        T_mesh, mu_mesh = self._mesh(T_grid, grid, pressure)
        U = self.U_all - self.U_all.mean()  # shift: avoid cancellation in <U^2> - <U>^2
        if kinetic:
            if self.n_substrate is None:
                warnings.warn("the number of displaceable substrate atoms is unknown (samples "
                              "from an older run): their constant (3/2) k_B each is left out "
                              "of the kinetic heat capacity")
            n = self.N_all + float(self.n_substrate or 0)
        else:
            n = np.zeros_like(self.N_all)
        A = np.column_stack([U, U ** 2, n, n ** 2, U * n])
        m = self._reweight(T_mesh, mu_mesh, A)
        kT = T_mesh
        var_U = m[..., 1] - m[..., 0] ** 2
        var_n = m[..., 3] - m[..., 2] ** 2
        cov_Un = m[..., 4] - m[..., 0] * m[..., 2]
        var_E = var_U + 3.0 * kT * cov_Un + (1.5 * kT) ** 2 * var_n + 1.5 * kT ** 2 * m[..., 2]
        return var_E / kT ** 2

    # -- configurations behind the phase diagram ------------------------------------------
    def sample_location(self, index: int) -> Tuple[Tuple[int, int], int]:
        """
        Grid point and run sample index of a sample (in the analysis order).

        Parameters
        ----------
        index : int
            Index into `U_all`.

        Returns
        -------
        tuple
            ``(key, sample)``: the grid point it was recorded at and its
            index there in the run's numbering (as written with the stored
            configurations, `regc.trajectory.load_configurations`).
        """
        k = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return self.states[k], int(self.run_indices(k)[int(index - self._offsets[k])])

    def run_indices(self, k: int) -> np.ndarray:
        """
        Run sample indices of the samples of state `k` (in the analysis order).

        Accounts for discarded equilibration (``first_index``) and for
        thinning (``subsample``).

        Parameters
        ----------
        k : int
            Index into `states`.

        Returns
        -------
        numpy.ndarray
            One run sample index per sample of that state.
        """
        from .io import first_index_of
        key = self.states[k]
        local = (np.asarray(self._kept[key]) if self._kept is not None
                 else np.arange(self.Nk[k]))
        return local + first_index_of(self._source if self._kept is None else self, key)

    def representative_samples(self, T: np.ndarray, mu: np.ndarray, target_N: np.ndarray,
                               observable: str = "N", candidates: Optional[np.ndarray] = None
                               ) -> np.ndarray:
        """
        The most probable recorded sample with a given N at each target state.

        At each (T, mu) the sample of largest MBAR weight among those whose
        `observable` equals the target value (or, if none has it, the
        closest value): a typical configuration of the phase shown there.

        Parameters
        ----------
        T, mu : numpy.ndarray
            Target states, common shape S (`mu` with a species axis with
            several species).
        target_N : numpy.ndarray
            Target values of `observable`, shape S (e.g. the most probable N).
        observable : str, optional
            ``'N'``, ``'N_ads'``, ... Default ``'N'``.
        candidates : numpy.ndarray, optional
            Boolean mask over all samples allowed to be picked (e.g. those
            whose configuration was stored). Default None (all).

        Returns
        -------
        numpy.ndarray
            Integer indices into `U_all`, shape S (see `sample_location`).

        Raises
        ------
        ValueError
            If `candidates` allows no sample.
        """
        T = np.asarray(T, dtype=float)
        target = np.asarray(target_N, dtype=float).ravel()
        values = np.rint(self._observable(observable))
        allowed = np.ones(values.size, bool) if candidates is None else np.asarray(candidates, bool)
        if self._evaluated is not None:
            allowed = allowed & self._evaluated
        out = np.empty(T.size, int)
        for sl, w in self._weight_blocks(T, mu):
            for row, i in enumerate(range(sl.start, sl.start + w.shape[0])):
                pool = allowed.copy()
                if not pool.any():
                    raise ValueError("no candidate samples")
                miss = np.abs(values - target[i])
                pool &= miss == miss[pool].min()
                out[i] = int(np.flatnonzero(pool)[np.argmax(w[row, pool])])
        return out.reshape(T.shape)

    def _most_probable_N_at(self, T: np.ndarray, mu: np.ndarray,
                            observable: str = "N") -> np.ndarray:
        """
        argmax_N P(N) at every (T, mu) of two same-shaped arrays.

        Parameters
        ----------
        T, mu : numpy.ndarray
            Target states, common shape S.
        observable : str, optional
            ``'N'`` (all mobile particles, the default) or ``'N_ads'``
            (adsorbed ones only; needs a run with ``adsorbed_cutoff``).

        Returns
        -------
        numpy.ndarray
            Integer array of shape S.
        """
        N_int = np.rint(self._observable(observable)).astype(int)
        one_hot = np.zeros((N_int.size, N_int.max() + 1))
        one_hot[np.arange(N_int.size), N_int] = 1.0
        return np.argmax(self._reweight(T, mu, one_hot), axis=-1)

    def _Tp_mesh(self, T_grid: np.ndarray, p_grid: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (T, mu) meshes, shape (len(T_grid), len(p_grid)), for a (T, p) mesh.

        Parameters
        ----------
        T_grid, p_grid : numpy.ndarray
            The mesh axes (p in the units `mu_from_p` expects).

        Returns
        -------
        tuple of numpy.ndarray
            ``(T_mesh, mu_mesh)``, each mu via `mu_from_p` (mu_mesh with a
            trailing species axis with several species).
        """
        shape = (len(T_grid), len(p_grid))
        mu_mesh = np.array([[self.mu_from_p(T, p) for p in p_grid] for T in T_grid],
                           dtype=float).reshape(shape + ((self.n_species,) if self.n_species > 1 else ()))
        T_mesh = np.broadcast_to(np.asarray(T_grid, dtype=float)[:, None], shape)
        return T_mesh, mu_mesh

    def most_probable_N(self, T_grid: np.ndarray, mu_grid: np.ndarray,
                        observable: str = "N") -> np.ndarray:
        """
        The most stable phase, labelled by N, on a (T, mu) mesh.

        [ZSG] Sec. II B: with phases defined by the number of particles,
        f_N = -kT ln P(N), so the stable phase is argmax_N P(N).

        Parameters
        ----------
        T_grid, mu_grid : numpy.ndarray
            The mesh axes (`mu_grid` a path of mu vectors, shape (n_mu,
            n_species), with several species).
        observable : str, optional
            ``'N'`` (all mobile particles, the default), ``'N_ads'``
            (adsorbed ones only; needs a run with ``adsorbed_cutoff``) or
            ``'N_<species>'``.

        Returns
        -------
        numpy.ndarray
            Integer array of shape (len(T_grid), len(mu_grid)).
        """
        T_mesh, mu_mesh = self._Tmu_mesh(T_grid, mu_grid)
        return self._most_probable_N_at(T_mesh, mu_mesh, observable)

    def most_probable_composition(self, T_grid: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
        """
        The most stable phase, labelled by its composition (N_1, ..., N_S).

        `most_probable_N` for several exchanged species: the argmax of the
        reweighted joint P(N_1, ..., N_S), e.g. telling an O-rich from a
        Cl-rich phase of the same total coverage.

        Parameters
        ----------
        T_grid : numpy.ndarray
            Temperatures, length n_T.
        mu_grid : numpy.ndarray
            Path of mu vectors, shape (n_mu, n_species) (length n_mu for
            one species).

        Returns
        -------
        numpy.ndarray
            Integer array of shape (n_T, n_mu, n_species).
        """
        T_mesh, mu_mesh = self._Tmu_mesh(T_grid, mu_grid)
        comps, label = np.unique(np.rint(self.Ns_all).astype(int), axis=0, return_inverse=True)
        one_hot = np.zeros((label.size, comps.shape[0]))
        one_hot[np.arange(label.size), label.ravel()] = 1.0
        return comps[np.argmax(self._reweight(T_mesh, mu_mesh, one_hot), axis=-1)]

    def most_probable_N_Tp(self, T_grid: np.ndarray, p_grid: np.ndarray,
                           observable: str = "N") -> np.ndarray:
        """
        `most_probable_N` on a (T, p) mesh ([ZSG] Fig. 3).

        Each (T, p) is converted with `mu_from_p` (so `mu_of_Tp`/`p0_of_T`
        given at construction set the reservoir).

        Parameters
        ----------
        T_grid, p_grid : numpy.ndarray
            The mesh axes (p in the units `mu_from_p` expects).
        observable : str, optional
            ``'N'`` (all mobile particles, the default) or ``'N_ads'``
            (adsorbed ones only; needs a run with ``adsorbed_cutoff``).

        Returns
        -------
        numpy.ndarray
            Integer array of shape (len(T_grid), len(p_grid)).
        """
        return self._most_probable_N_at(*self._Tp_mesh(T_grid, p_grid), observable)

    def lowest_energy_per_N(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lowest sampled energy for every observed particle number.

        Returns
        -------
        tuple of numpy.ndarray
            ``(N_values, E_min)``: the distinct N in the samples and the
            lowest potential energy sampled with that N. With several
            species, `N_values` holds the distinct compositions, shape
            (n_compositions, n_species).
        """
        ok = slice(None) if self._evaluated is None else self._evaluated
        U = self.U_all[ok]
        if self.n_species > 1:
            comps, label = np.unique(np.rint(self.Ns_all[ok]).astype(int), axis=0,
                                     return_inverse=True)
            E = np.full(comps.shape[0], np.inf)
            np.minimum.at(E, label.ravel(), U)
            return comps, E
        N_int = np.rint(self.N_all[ok]).astype(int)
        Ns = np.unique(N_int)
        return Ns, np.array([U[N_int == n].min() for n in Ns])

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
            Chemical potentials (mu vectors, shape (n, n_species), with
            several species).

        Returns
        -------
        numpy.ndarray
            Integer array of the stable N, one per mu (with several
            species, the stable composition, shape (n, n_species)).
        """
        Ns, E = self.lowest_energy_per_N()
        mu = np.asarray(mu_values, dtype=float)
        if self.n_species > 1:
            mu = mu.reshape(-1, self.n_species)
            return Ns[np.argmin(E[None, :] - mu @ Ns.T, axis=1)]
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
            Integer array of shape (len(T_grid), len(p_grid)) (plus a
            trailing species axis with several species).
        """
        return np.array([self.lowest_energy_N([self.mu_from_p(T, p) for p in p_grid])
                         for T in T_grid])

    def pN(self, T: float, mu, n_bins: Optional[int] = None, observable: str = "N"
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
        mu : float or array_like
            Chemical potential to reweight to (one per species with
            several).
        n_bins : int, optional
            Number of histogram bins. Default is None, which uses one bin
            per integer N from 0 to the observed maximum.
        observable : str, optional
            Count to histogram: ``'N'`` (default), ``'N_ads'`` or
            ``'N_<species>'`` (see `phase_diagram`).

        Returns
        -------
        tuple of numpy.ndarray
            ``(centers, probs)``: the bin centers (in N) and the
            reweighted probability mass in each bin.
        """
        N = self._observable(observable)
        n_max = int(N.max())
        edges = (np.arange(-0.5, n_max + 1.5, 1.0) if n_bins is None
                 else np.linspace(-0.5, n_max + 0.5, n_bins + 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        w = np.exp(self._log_weights(T, mu))
        probs, _ = np.histogram(N, bins=edges, weights=w)
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
        ('condensed') basins. The basins are split at one fixed N, midway
        between the valleys of the bimodal P(N) at the two bracket ends, so
        the bisection stays well defined where P(N) is unimodal.

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
        ValueError
            With several exchanged species (no single mu to bisect; scan
            `pN` along a mu path instead).
        """
        if self.n_species > 1:
            raise ValueError("locate_coexistence_mu bisects one mu; with several species, "
                             "scan pN(T, mu_vector) along a path instead")
        lo_mu, hi_mu = mu_bracket

        def valley_N(mu: float) -> Optional[float]:
            # N at the valley between the two highest peaks of P(N), or None if unimodal
            centers, probs = self.pN(T, mu, n_bins=n_bins)
            is_max = (probs[1:-1] > probs[:-2]) & (probs[1:-1] > probs[2:])
            maxima_idx = np.where(is_max)[0] + 1
            if len(maxima_idx) < 2:
                return None
            top2 = maxima_idx[np.argsort(probs[maxima_idx])[-2:]]
            i_lo, i_hi = sorted(top2)
            return float(centers[i_lo + int(np.argmin(probs[i_lo:i_hi + 1]))])

        v_lo, v_hi = valley_N(lo_mu), valley_N(hi_mu)
        if v_lo is None or v_hi is None:
            raise RuntimeError(
                "Could not bracket a bimodal P(N) crossing in this mu range -- either "
                "this system doesn't show 2D condensation here, or the replica ladder "
                "needs more mu-points near the transition for good MBAR overlap."
            )
        # one fixed dilute/condensed split for the whole bisection, so the imbalance is
        # defined (and decreasing in mu) even where P(N) is unimodal
        split = 0.5 * (v_lo + v_hi)

        def imbalance(mu: float) -> float:
            centers, probs = self.pN(T, mu, n_bins=n_bins)
            return float(probs[centers <= split].sum() - probs[centers > split].sum())

        f_lo, f_hi = imbalance(lo_mu), imbalance(hi_mu)
        if f_lo * f_hi > 0:
            raise RuntimeError(
                "Could not bracket a bimodal P(N) crossing in this mu range -- either "
                "this system doesn't show 2D condensation here, or the replica ladder "
                "needs more mu-points near the transition for good MBAR overlap."
            )
        for _ in range(n_iter):
            mid = 0.5 * (lo_mu + hi_mu)
            f_mid = imbalance(mid)
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


# =============================================================================
# Heat-capacity ridges, phase regions and the configurations behind a diagram
# =============================================================================
def heat_capacity_ridges(T_grid: np.ndarray, grid: np.ndarray, cv: np.ndarray,
                         valid: Optional[np.ndarray] = None, prominence: float = 0.2,
                         ladder: Optional[Sequence[float]] = None,
                         ladder_tol: Optional[float] = None) -> dict:
    """
    Local maxima of C_V along each axis: candidate phase-boundary points.

    [ZSG] PRL 128, 246101: a phase boundary connects the (T, p) points
    where C_V(p) at fixed T, or C_V(T) at fixed p, has a peak; a ridge
    that fades out ends at a critical point.

    Parameters
    ----------
    T_grid : numpy.ndarray
        Temperatures, length n_T.
    grid : numpy.ndarray
        Pressures or chemical potentials, length n_y.
    cv : numpy.ndarray
        Heat capacity, shape (n_T, n_y) (`MBARSurfaceAnalysis.heat_capacity`).
    valid : numpy.ndarray, optional
        Boolean mask of trustworthy cells (e.g. effective sample size above
        a threshold); peaks in other cells are dropped. Default None (all).
    prominence : float, optional
        Minimum relative height of a peak over the deeper of its two
        flanking minima: a peak counts if C_V there exceeds that minimum by
        this fraction (found on ln C_V, so it is the same for a small bump
        on a large background and for one near C_V = 0). Default 0.2
        (20 %); raise it for noisy maps from short runs.
    ladder : sequence of float, optional
        The sampled values along `grid` (the replica ladder's p or mu).
        C_V is a variance: between replicas whose samples barely overlap,
        MBAR rests on few effective samples and underestimates it, so
        spurious peaks appear *on* the sampled values. With `ladder` given,
        each point is flagged (``'on_ladder'``) if it lies within
        `ladder_tol` of a sampled value, and a warning is issued when most
        do. Real boundaries move with T; artefacts stay on the ladder. For
        a pressure grid, give the ladder's pressures. Default None.
    ladder_tol : float, optional
        Distance counting as "on" the ladder. Default: a quarter of the
        smallest ladder spacing (in log10 for a positive ladder spanning
        more than a decade, i.e. pressures).

    Returns
    -------
    dict
        ``'along_y'``: (i_T, j_y) index pairs of peaks of C_V(y) at fixed
        T; ``'along_T'``: the same for C_V(T) at fixed y; ``'points'``:
        the union as ``(T, y)`` values, shape (n, 2), and ``'cells'`` the
        same as index pairs; ``'along_y_bases'``, ``'along_T_bases'``: the
        indices of the two flanking minima of each peak along its axis
        (for `regc.diagnostics.ridge_support`); with `ladder`, also
        ``'on_ladder'``.
    """
    from scipy.signal import find_peaks
    cv = np.asarray(cv, dtype=float)
    valid = np.isfinite(cv) if valid is None else (np.asarray(valid, bool) & np.isfinite(cv))
    if not valid.any():
        return {"along_y": np.zeros((0, 2), int), "along_T": np.zeros((0, 2), int),
                "points": np.zeros((0, 2))}
    positive = cv[valid & (cv > 0)]
    floor = positive.min() if positive.size else 1.0
    filled = np.log(np.where(valid & (cv > 0), cv, floor))
    min_prom = np.log1p(prominence)

    def peaks(axis):
        out, bases = [], []
        lines = filled if axis == 1 else filled.T
        for i, line in enumerate(lines):
            idx, props = find_peaks(line, prominence=min_prom)
            for j, lo, hi in zip(idx, props["left_bases"], props["right_bases"]):
                cell = (i, j) if axis == 1 else (j, i)
                if valid[cell]:
                    out.append(cell)
                    bases.append((lo, hi))
        return (np.array(out, dtype=int).reshape(-1, 2),
                np.array(bases, dtype=int).reshape(-1, 2))

    (along_y, bases_y), (along_T, bases_T) = peaks(1), peaks(0)
    both = np.unique(np.vstack([along_y, along_T]), axis=0) if (len(along_y) or len(along_T)) \
        else np.zeros((0, 2), int)
    T_grid, grid = np.asarray(T_grid, dtype=float), np.asarray(grid, dtype=float)
    points = (np.column_stack([T_grid[both[:, 0]], grid[both[:, 1]]]) if len(both)
              else np.zeros((0, 2)))
    out = {"along_y": along_y, "along_T": along_T, "points": points, "cells": both,
           "along_y_bases": bases_y, "along_T_bases": bases_T}
    if ladder is not None:
        lad = np.unique(np.asarray(ladder, dtype=float))
        log = bool(lad.min() > 0 and lad.max() / lad.min() > 10.0)
        f = np.log10 if log else (lambda v: v)
        tol = ladder_tol if ladder_tol is not None else (
            0.25 * float(np.diff(f(lad)).min()) if lad.size > 1 else np.inf)
        on = (np.abs(f(points[:, 1])[:, None] - f(lad)[None, :]).min(axis=1) <= tol
              if len(points) else np.zeros(0, bool))
        out["on_ladder"] = on
        if len(points) and on.mean() > 0.5:
            warnings.warn(f"{on.mean():.0%} of the C_V peaks lie on sampled values of the "
                          f"ladder: likely an overlap artefact (C_V is underestimated between "
                          f"replicas), not phase boundaries; use a denser ladder there",
                          stacklevel=2)
    return out


def phase_regions(labels: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
    """
    Connected regions of equal phase label on a (T, y) map.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer map, e.g. the most probable N, shape (n_T, n_y).

    Returns
    -------
    tuple
        ``(region_map, regions)``: an integer map numbering the regions
        0, 1, ... (4-connectivity), and one dict per region with
        ``'label'`` (the phase label) and ``'cells'`` (array of (i, j)).
    """
    from scipy import ndimage
    labels = np.asarray(labels)
    region_map = np.full(labels.shape, -1, int)
    regions = []
    for value in np.unique(labels):
        lab, n = ndimage.label(labels == value)
        for k in range(1, n + 1):
            cells = np.argwhere(lab == k)
            region_map[lab == k] = len(regions)
            regions.append({"label": int(value), "cells": cells})
    return region_map, regions


def _config_location(configurations_dir, key: tuple) -> Tuple[str, Tuple[int, int]]:
    """
    Directory and grid key of a state's stored configurations.

    Parameters
    ----------
    configurations_dir : str or sequence of str
        One directory, or one per merged run.
    key : tuple
        ``(it, imu)``, or ``(run, it, imu)`` for merged runs.

    Returns
    -------
    tuple
        ``(directory, (it, imu))``.
    """
    if len(key) == 3:
        dirs = [configurations_dir] if isinstance(configurations_dir, str) else configurations_dir
        return dirs[key[0]], tuple(key[1:])
    return configurations_dir, tuple(key)


def _stored_samples(directory: str, key: Tuple[int, int]) -> set:
    """
    Sample indices with a stored configuration at one grid point.

    Parameters
    ----------
    directory : str
        The configurations directory.
    key : tuple of int
        Grid point.

    Returns
    -------
    set of int
        The ``sample=`` values of the frames (comment lines only are read).
    """
    import os
    import re

    from .trajectory import configuration_path
    path = configuration_path(directory, key)
    if not os.path.isfile(path):
        return set()
    found = set()
    with open(path) as f:
        while True:
            line = f.readline()
            if not line.strip():
                return found
            n = int(line.split()[0])
            m = re.search(r"\bsample=(\d+)", f.readline())
            if m:
                found.add(int(m.group(1)))
            for _ in range(n):
                f.readline()


def _tile(frames: List[dict], positions_in_grid: List[Tuple[int, int]], gap: float = 4.0) -> dict:
    """
    Put several configurations side by side in one frame (a visual phase diagram).

    Slabs are wrapped into their cell; clusters are centred in their tile.
    Column = y index, row = T index.

    Parameters
    ----------
    frames : list of dict
        Frames as from `regc.trajectory.read_extxyz_frames`.
    positions_in_grid : list of tuple of int
        ``(row, col)`` of each frame.
    gap : float, optional
        Space between tiles, in Angstrom. Default 4.0.

    Returns
    -------
    dict
        One frame (``positions``, ``species``, ``lattice``, ``pbc``,
        ``arrays`` with ``tile_row``/``tile_col``/``group``).
    """
    shifted, extents = [], []
    for fr in frames:
        pos = np.array(fr["positions"], dtype=float)
        lat = np.asarray(fr["lattice"], dtype=float)
        if fr["pbc"][0] and fr["pbc"][1]:
            frac = pos @ np.linalg.inv(lat)
            frac[:, :2] %= 1.0
            pos = frac @ lat
            ext = np.array([np.abs(lat[:, 0]).sum(), np.abs(lat[:, 1]).sum()])
        else:
            pos -= pos.min(axis=0) if len(pos) else 0.0
            ext = np.ptp(pos[:, :2], axis=0) + 1.0 if len(pos) else np.ones(2)
        shifted.append(pos)
        extents.append(ext)
    w, h = np.max(extents, axis=0) + gap
    out_pos, out_sp, rows, cols, groups = [], [], [], [], []
    for pos, fr, (r, c) in zip(shifted, frames, positions_in_grid):
        out_pos.append(pos + np.array([c * w, r * h, 0.0]))
        out_sp.append(np.asarray(fr["species"]))
        rows.append(np.full(len(pos), r))
        cols.append(np.full(len(pos), c))
        groups.append(np.asarray(fr["arrays"].get("group", np.full(len(pos), -1))))
    pos = np.concatenate(out_pos)
    lo = pos.min(axis=0) - gap / 2
    pos -= lo
    return {"positions": pos, "species": np.concatenate(out_sp),
            "lattice": np.diag(pos.max(axis=0) + gap / 2), "pbc": [False, False, False],
            "arrays": {"tile_row": np.concatenate(rows).astype(int),
                       "tile_col": np.concatenate(cols).astype(int),
                       "group": np.concatenate(groups).astype(int)}}


def export_phase_configurations(analysis: MBARSurfaceAnalysis, configurations_dir: str,
                                T_grid: np.ndarray, grid: np.ndarray, path: str,
                                pressure: bool = False, mode: str = "phases",
                                observable: str = "N", min_ess: float = 0.0,
                                temperature_K: bool = False, area: Optional[float] = None,
                                tile_path: Optional[str] = None, min_cells: int = 1
                                ) -> List[dict]:
    """
    Write the configurations visible in a phase diagram as extended XYZ.

    The run must have stored configurations (``configurations_dir`` of
    `ReplicaExchangeGCMC`, ``configurations: save`` in a config file). On
    the (T, y) mesh, the most probable value of `observable` labels the
    phase of every cell. Then:

    * ``mode='phases'``: one frame per connected phase region, taken at
      the cell of the region where that phase is most probable;
    * ``mode='grid'``: one frame per mesh cell (use a coarse mesh).

    Each frame is the recorded sample with the largest MBAR weight at that
    cell among those with the phase's value of `observable` (and a stored
    configuration): a typical structure of the phase, as pySNOW shows when
    a point of the diagram is picked. The comment line carries the cell's
    T, p or mu, the phase label, the probability of that label there, the
    sample's weight, its origin (grid point, sample index, energy) and the
    effective sample size; per-atom ``group`` is 0 frozen, 1 substrate,
    2 mobile.

    Parameters
    ----------
    analysis : MBARSurfaceAnalysis
        The analysis of the run.
    configurations_dir : str or sequence of str
        Where the run stored ``configs_T{it}_mu{imu}.extxyz``; for merged
        runs (`regc.io.merge_samples`), one directory per run, in the order
        the runs were merged.
    T_grid : numpy.ndarray
        Temperatures (k_B T, as given to the analysis).
    grid : numpy.ndarray
        Pressures (``pressure=True``) or chemical potentials.
    path : str
        Output ``.extxyz`` file (multi-frame).
    pressure : bool, optional
        Whether `grid` holds pressures. Default False.
    mode : {'phases', 'grid'}, optional
        See above. Default ``'phases'``.
    observable : str, optional
        Phase label: ``'N'`` (default), ``'N_ads'``, ``'N_<species>'``.
    min_ess : float, optional
        Ignore cells whose effective sample size is below this (their
        phase is extrapolated). Default 0.0.
    temperature_K : bool, optional
        Also write ``T_K`` (temperatures given as k_B T in eV). Default False.
    area : float, optional
        If given, also write ``coverage`` = label / area. Default None.
    tile_path : str, optional
        If given, also write all exported frames side by side in one frame
        (rows: T, columns: p or mu), a visual phase diagram. Default None.
    min_cells : int, optional
        ``mode='phases'``: skip regions smaller than this many mesh cells
        (where the label changes almost continuously, as in a fluid, every
        cell is its own region). Default 1.

    Returns
    -------
    list of dict
        One entry per exported frame: its comment-line values.

    Raises
    ------
    FileNotFoundError
        If no configurations were stored.
    ValueError
        On an unknown `mode`, or if no cell passes `min_ess`.
    """
    from .trajectory import format_extxyz_frame, load_configurations
    from .units import eV_to_kelvin

    if mode not in ("phases", "grid"):
        raise ValueError(f"mode must be 'phases' or 'grid', got {mode!r}")
    T_grid = np.atleast_1d(np.asarray(T_grid, dtype=float))
    grid_arr = np.atleast_1d(np.asarray(grid, dtype=float))
    T_mesh, mu_mesh = analysis._mesh(T_grid, grid_arr, pressure)

    # which samples have a stored configuration
    stored = np.zeros(analysis.U_all.size, bool)
    for k, key in enumerate(analysis.states):
        have = _stored_samples(*_config_location(configurations_dir, key))
        if have:
            idx = analysis.run_indices(k)
            stored[analysis._offsets[k]:analysis._offsets[k + 1]] = np.isin(idx, list(have))
    if not stored.any():
        raise FileNotFoundError(f"no stored configurations in {configurations_dir} (run with "
                                f"configurations_dir= / configurations: save)")

    values = np.rint(analysis._observable(observable)).astype(int)
    one_hot = np.zeros((values.size, values.max() + 1))
    one_hot[np.arange(values.size), values] = 1.0
    prob = analysis._reweight(T_mesh, mu_mesh, one_hot)       # P(label) per cell
    label_map = np.argmax(prob, axis=-1)
    ess = analysis.effective_sample_size(T_grid, grid_arr, pressure)
    ok = ess >= min_ess
    if not ok.any():
        raise ValueError(f"no cell has an effective sample size >= {min_ess}")

    if mode == "phases":
        _, regions = phase_regions(np.where(ok, label_map, -1))
        cells = []
        for reg in regions:
            if reg["label"] < 0 or len(reg["cells"]) < min_cells:
                continue
            c = reg["cells"]
            best = c[np.argmax(prob[c[:, 0], c[:, 1], reg["label"]])]
            cells.append(tuple(best))
    else:
        cells = [tuple(c) for c in np.argwhere(ok)]

    picks = analysis.representative_samples(
        np.array([T_mesh[c] for c in cells]), np.array([mu_mesh[c] for c in cells]),
        np.array([label_map[c] for c in cells]), observable=observable, candidates=stored)

    cache, frames, entries = {}, [], []
    for (i, j), n in zip(cells, picks):
        key, sample = analysis.sample_location(int(n))
        if key not in cache:
            cache[key] = load_configurations(*_config_location(configurations_dir, key))
        fr = cache[key][sample]
        T = float(T_grid[i])
        w = np.exp(analysis._log_weights(T, mu_mesh[i, j]))
        info = {"T": T}
        if temperature_K:
            info["T_K"] = float(eV_to_kelvin(T))
        if pressure:
            info["p"] = float(grid_arr[j])
        if analysis.n_species == 1:
            info["mu"] = float(mu_mesh[i, j])
        label = int(label_map[i, j])
        info.update({"phase_" + observable: label, "P_phase": float(prob[i, j, label]),
                     "ESS": float(ess[i, j]), "weight": float(w[n]), "source_T_index": key[-2],
                     "source_mu_index": key[-1], "source_sample": sample,
                     "U": float(fr["info"].get("U", np.nan)), "N": int(fr["info"].get("N", -1)),
                     "T_index": int(i), "y_index": int(j)})
        if len(key) == 3:
            info["source_run"] = (analysis.run_labels[key[0]] if analysis.run_labels
                                  else int(key[0]))
        if area:
            info["coverage"] = label / float(area)
        frames.append((fr, info))
        entries.append(info)

    with open(path, "w") as f:
        for fr, info in frames:
            f.write(format_extxyz_frame(fr["positions"], fr["species"], fr["lattice"], fr["pbc"],
                                        info, fr["arrays"]))
    if tile_path is not None:
        t = _tile([fr for fr, _ in frames], [(info["T_index"], info["y_index"]) for _, info in frames])
        with open(tile_path, "w") as f:
            f.write(format_extxyz_frame(t["positions"], t["species"], t["lattice"], t["pbc"],
                                        {"frames": len(frames), "rows": "T", "columns":
                                         "p" if pressure else "mu"}, t["arrays"]))
    return entries
