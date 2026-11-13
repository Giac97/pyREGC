"""
The package's Newton MBAR solver (`regc.analysis._mbar_free_energies_newton`):
exactness on a solvable case, and the stall exit that stops it from
spending its whole iteration budget once the objective is flat to
round-off (a bootstrap resample took 185 s instead of 1 s without it).
"""

import numpy as np

import regc.analysis as an


def _harmonic_states(K=6, n=3000, seed=0):
    """K harmonic states u_k(x) = (x - k)^2 / 2: f_k are all equal (0)."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(k, 1.0, n) for k in range(K)])
    u = 0.5 * (x[None, :] - np.arange(K)[:, None]) ** 2
    return u, np.full(K, n)


def test_newton_matches_the_exact_free_energies():
    u, Nk = _harmonic_states()
    f = an._mbar_free_energies_newton(u, Nk)
    assert np.abs(f).max() < 0.05  # exact: 0; sampling noise ~ 1/sqrt(n)
    # self-consistency to round-off
    a = np.log(Nk)[:, None] + f[:, None] - u
    W = np.exp(a - a.max(axis=0))
    W /= W.sum(axis=0)
    assert np.abs(W.sum(axis=1) - Nk).max() < 1e-4  # samples, of 18000


def test_stall_exit_when_the_target_is_below_round_off(monkeypatch):
    """With a gradient target below double-precision round-off (here forced
    with tol=1e-30; in a real bootstrap resample of 38400 samples, 1e-10
    was out of reach) every Newton step past convergence backtracks ~25
    times and is accepted only by noise. The stall exit stops there, with
    the same answer; without it the solve uses its whole iteration budget."""
    u, Nk = _harmonic_states()
    calls = {"n": 0}
    real_exp = np.exp

    def counting_exp(x, *args, **kwargs):
        if np.ndim(x) == 2:
            calls["n"] += 1
        return real_exp(x, *args, **kwargs)

    monkeypatch.setattr(an.np, "exp", counting_exp)
    f0 = 0.3 * np.arange(len(Nk))
    f_fast = an._mbar_free_energies_newton(u, Nk, f0=f0, tol=1e-30)
    fast = calls["n"]
    calls["n"] = 0
    f_slow = an._mbar_free_energies_newton(u, Nk, f0=f0, tol=1e-30, stall_tol=0.0, max_iter=40)
    slow = calls["n"]
    assert np.abs(f_fast - f_slow).max() < 1e-9
    assert fast < 40 and slow > 40 * 5
