"""One grid slot: a fixed (T, mu) plus the GCMC move set and bookkeeping."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .observables import evaluate_observables
from .state import ThermoState
from .systems import ParticleSystem
from .trajectory import snapshot


class Replica:
    """Note the design choice: the ThermoState stays attached to the grid
    slot; it is the *configuration* (`self.system`) that gets swapped between
    slots during replica exchange (see `engine.py`). Each slot keeps its own
    RNG stream so move statistics and pseudo-random sequences stay
    reproducible per grid point regardless of how sweeps are scheduled
    (serially or across a process pool).

    `hmc_enabled=True` replaces the single-particle random MC displacement
    move with Hybrid Monte Carlo (HMC): resample momenta from Maxwell-
    Boltzmann, run a short leapfrog trajectory for ALL mobile particles at
    once, and accept/reject the whole trajectory on the total (kinetic +
    potential) energy change. This is exact regardless of integrator error
    or timestep -- the Metropolis step, not the integrator, carries the
    correctness guarantee, so a too-large `hmc_dt` costs you acceptance
    rate (efficiency), never correctness. Collective, force-directed moves
    like this explore configuration space far more efficiently than
    single-particle random displacement whenever particles need to move
    together, which is the whole motivation (surface reconstruction is the
    eventual target; a dense/structured fluid benefits too). Needs a
    Calculator with `supports_forces = True` (LJCalculator and
    ASECalculatorAdapter both qualify); raises immediately at construction
    otherwise, rather than failing confusingly on the first move.

    Current scope/limitation worth knowing: `hmc_dt`'s momentum kicks use
    only `calculator.forces()`, i.e. the interatomic contribution -- not
    any `external_energy` field (a continuum wall, or a hard z-boundary).
    Correctness is unaffected (the final accept/reject step still checks
    the true total energy including that field, so a trajectory that ends
    somewhere forbidden is still always rejected), but a trajectory isn't
    steered away from such a boundary the way real forces would, so
    acceptance efficiency near one may be poor. Fine for a bulk system with
    no external field (the common case); something to watch for a
    wall-bounded or hard-boundary geometry."""

    def __init__(self, state: ThermoState, system: ParticleSystem, rng: np.random.Generator,
                 max_disp: float = 0.2, hmc_enabled: bool = False,
                 hmc_leapfrog_steps: int = 10, hmc_dt: float = 0.005):
        """
        Construct a replica-exchange grid slot.

        Parameters
        ----------
        state : ThermoState
            The (temperature, chemical potential) label for this grid
            slot.
        system : ParticleSystem
            The initial configuration to attach to this slot.
        rng : numpy.random.Generator
            This slot's own random number generator stream, kept separate
            per grid point so move statistics and pseudo-random sequences
            stay reproducible regardless of how sweeps are scheduled
            (serially or across a process pool).
        max_disp : float, optional
            Maximum single-particle displacement (per axis) for plain MC
            displacement moves (`_move_displace`, `_move_displace_substrate`).
            Default is 0.2.
        hmc_enabled : bool, optional
            If True, replace the single-particle random MC displacement
            move with Hybrid Monte Carlo (see this class's docstring for
            the full rationale). Default is False.
        hmc_leapfrog_steps : int, optional
            Number of leapfrog integration steps per HMC trial. Default
            is 10.
        hmc_dt : float, optional
            Leapfrog integration timestep for HMC. Default is 0.005.

        Raises
        ------
        ValueError
            If `hmc_enabled` is True but `system.calculator` doesn't have
            `supports_forces = True`.
        """
        self.state = state
        self.system = system
        self.rng = rng
        self.max_disp = max_disp
        self.hmc_enabled = hmc_enabled
        self.hmc_leapfrog_steps = hmc_leapfrog_steps
        self.hmc_dt = hmc_dt
        if hmc_enabled and not system.calculator.supports_forces:
            raise ValueError(
                f"hmc_enabled=True needs a Calculator with supports_forces=True; "
                f"{type(system.calculator).__name__} doesn't provide forces(). "
                f"LJCalculator and ASECalculatorAdapter both do; pass "
                f"hmc_enabled=False to use plain single-particle MC displacement instead."
            )
        self.energy = system.total_energy()

        self.n_attempts = {'displace': 0, 'insert': 0, 'delete': 0, 'hmc': 0,
                            'displace_substrate': 0, 'identity': 0}
        self.n_accepts = {'displace': 0, 'insert': 0, 'delete': 0, 'hmc': 0,
                           'displace_substrate': 0, 'identity': 0}
        # semi-grand-canonical identity swaps per REGC step (set by the engine)
        self.identity_attempts = 0
        self.exchange_attempts = 0
        self.exchange_accepts = 0

    def _move_displace(self) -> None:
        """
        Attempt one single-particle random MC displacement of a mobile
        particle.

        Picks a random mobile particle, proposes a uniform random
        displacement of up to `max_disp` per axis, and accepts or rejects
        it via the Metropolis criterion. Updates `self.energy` and the
        `'displace'` entries of `n_attempts`/`n_accepts` in place. A no-op
        if there are no mobile particles.

        Returns
        -------
        None
        """
        sysm = self.system
        if sysm.n == 0:
            return
        i = int(self.rng.integers(sysm.n))
        old_pos = sysm.positions[i].copy()
        disp = (self.rng.random(3) - 0.5) * 2.0 * self.max_disp
        new_pos = old_pos + disp
        if sysm.calculator.supports_incremental:
            u_old = sysm.pair_energy_of(old_pos, exclude=i)
            u_new = sysm.pair_energy_of(new_pos, exclude=i)
            du = u_new - u_old
        else:
            du = sysm.generic_displace_dU(i, new_pos)

        self.n_attempts['displace'] += 1
        if du <= 0.0 or self.rng.random() < np.exp(-self.state.beta * du):
            sysm.apply_move(i, new_pos)
            self.energy += du
            self.n_accepts['displace'] += 1

    def _move_displace_substrate(self) -> None:
        """
        Attempt one single-particle random MC displacement of a substrate
        atom.

        Single-particle random MC displacement for one substrate atom
        (not mobile) -- always via the generic clone-and-recompute path,
        since substrate atoms are meant for real ASE calculators
        (supports_incremental=False) rather than the fast native
        LJCalculator, which doesn't account for substrate atoms at all
        (see this module's HMC docstring for the equivalent limitation
        there). Updates `self.energy` and the `'displace_substrate'`
        entries of `n_attempts`/`n_accepts` in place. A no-op if there are
        no substrate atoms.

        Returns
        -------
        None
        """
        sysm = self.system
        n_sub = sysm.n_substrate
        if n_sub == 0:
            return
        i = int(self.rng.integers(n_sub))
        old_pos = sysm.substrate_positions[i].copy()
        disp = (self.rng.random(3) - 0.5) * 2.0 * self.max_disp
        new_pos = old_pos + disp
        du = sysm.generic_displace_substrate_dU(i, new_pos)

        self.n_attempts['displace_substrate'] += 1
        if du <= 0.0 or self.rng.random() < np.exp(-self.state.beta * du):
            sysm.apply_move_substrate(i, new_pos)
            self.energy += du
            self.n_accepts['displace_substrate'] += 1

    def _move_hmc(self) -> None:
        """
        Attempt one Hybrid Monte Carlo trajectory.

        Jointly propagates substrate atoms (if any) alongside the mobile
        ones, in a single trajectory/single accept-reject decision -- the
        right way to capture correlated relaxation between an adsorbate
        and a substrate that responds to it (adsorbate-induced
        reconstruction), since both then move under the same force field,
        at the same time, rather than as independent, uncoordinated
        proposals. When there are no substrate atoms this reduces to
        exactly the mobile-only HMC move.

        Only meaningful with a calculator whose `forces()` accounts for
        substrate atoms (`ASECalculatorAdapter` does, via the full
        configuration); `LJCalculator` does not (documented limitation --
        it only ever sees `system.positions`), so combining substrate
        atoms with `LJCalculator` + HMC raises a clear error rather than
        silently giving wrong physics. Updates `self.energy` and the
        `'hmc'` entries of `n_attempts`/`n_accepts` in place.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If substrate atoms are present but `calc.forces()` doesn't
            return one row per substrate + mobile particle (i.e. the
            calculator doesn't account for substrate atoms).
        """
        sysm = self.system
        calc = sysm.calculator
        n_mobile = sysm.n
        n_sub = sysm.n_substrate
        n_total = n_mobile + n_sub
        self.n_attempts['hmc'] += 1
        if n_total == 0:
            return

        masses = np.concatenate([sysm.substrate_masses(), sysm.masses()])
        v = self.rng.normal(0.0, 1.0, size=(n_total, 3)) * np.sqrt(self.state.temperature / masses)[:, None]
        KE_old = 0.5 * np.sum(masses[:, None] * v ** 2)
        U_old = self.energy

        saved_substrate = sysm.substrate_positions.copy()
        saved_mobile = sysm.positions.copy()
        pos = np.vstack([sysm.substrate_positions, sysm.positions])
        F = calc.forces(sysm)
        if F.shape[0] != n_total:
            raise ValueError(
                f"hmc_enabled with substrate atoms present needs a Calculator whose "
                f"forces() accounts for them (should return {n_total} rows: "
                f"{n_sub} substrate + {n_mobile} mobile), but "
                f"{type(calc).__name__}.forces() returned {F.shape[0]} row(s) instead. "
                f"LJCalculator doesn't support substrate atoms (a documented "
                f"limitation -- it only ever sees system.positions); use "
                f"ASECalculatorAdapter with a real potential for substrate "
                f"reconstruction, or set hmc_enabled=False and use "
                f"p_displace_substrate for plain MC substrate moves instead."
            )
        dt = self.hmc_dt
        for _ in range(self.hmc_leapfrog_steps):
            v = v + 0.5 * dt * F / masses[:, None]
            pos = pos + dt * v
            sysm.substrate_positions = pos[:n_sub]
            sysm.positions = pos[n_sub:]  # temporary, for force/energy evaluation only
            F = calc.forces(sysm)
            v = v + 0.5 * dt * F / masses[:, None]

        U_new = sysm.total_energy()  # includes external_energy, not just the calculator's part
        KE_new = 0.5 * np.sum(masses[:, None] * v ** 2)
        dH = (U_new - U_old) + (KE_new - KE_old)

        if dH <= 0.0 or self.rng.random() < np.exp(-self.state.beta * dH):
            for i in range(n_sub):
                sysm.apply_move_substrate(i, pos[i])  # commits through each geometry's own wrapping
            for i in range(n_mobile):
                sysm.apply_move(i, pos[n_sub + i])
            self.energy = U_new
            self.n_accepts['hmc'] += 1
        else:
            sysm.substrate_positions = saved_substrate
            sysm.positions = saved_mobile

    def _move_insert(self) -> None:
        """
        Attempt one GCMC particle insertion.

        Proposes a random trial position (via
        `system.random_insertion_position`), and accepts or rejects it via
        the grand-canonical insertion acceptance formula. Updates
        `self.energy` and the `'insert'` entries of
        `n_attempts`/`n_accepts` in place.

        With several exchanged species, see `_move_insert_species`.

        Returns
        -------
        None
        """
        sysm = self.system
        if getattr(sysm, 'molecules', ()) or getattr(sysm, 'subsurface', None) is not None:
            self._move_gc_channel(insert=True)
            return
        if sysm.n_exchange_species > 1:
            self._move_insert_species()
            return
        pos = sysm.random_insertion_position(self.rng)
        du = (sysm.pair_energy_of(pos) if sysm.calculator.supports_incremental
              else sysm.generic_insert_dU(pos))
        # n counts only particles in the exchange zone (the region `pos` was
        # drawn from), matching the reverse deletion move's candidate set
        beta, mu, n = self.state.beta, self.state.mu, len(sysm.exchange_zone_indices())
        log_arg = (beta * mu - beta * du
                   + np.log(sysm.volume / ((n + 1) * self.state.lambda_cubed)))

        self.n_attempts['insert'] += 1
        if log_arg >= 0.0 or self.rng.random() < np.exp(log_arg):
            sysm.apply_insert(pos)
            self.energy += du
            self.n_accepts['insert'] += 1

    def _move_delete(self) -> None:
        """
        Attempt one GCMC particle deletion.

        Picks a random mobile particle from the exchange zone (see
        `ParticleSystem.exchange_zone_indices`), and accepts or rejects
        removing it via the grand-canonical deletion acceptance formula,
        with N the number of particles in that zone. Updates
        `self.energy` and the `'delete'` entries of
        `n_attempts`/`n_accepts` in place. A no-op if there are no mobile
        particles in the exchange zone.

        With several exchanged species, see `_move_delete_species`.

        Returns
        -------
        None
        """
        sysm = self.system
        if getattr(sysm, 'molecules', ()) or getattr(sysm, 'subsurface', None) is not None:
            self._move_gc_channel(insert=False)
            return
        if sysm.n_exchange_species > 1:
            self._move_delete_species()
            return
        candidates = sysm.exchange_zone_indices()
        n = len(candidates)
        if n == 0:
            return
        i = int(candidates[self.rng.integers(n)])
        pos = sysm.positions[i]
        du = (-sysm.pair_energy_of(pos, exclude=i) if sysm.calculator.supports_incremental
              else sysm.generic_delete_dU(i))
        beta, mu = self.state.beta, self.state.mu
        log_arg = (-beta * mu - beta * du
                   + np.log(n * self.state.lambda_cubed / sysm.volume))

        self.n_attempts['delete'] += 1
        if log_arg >= 0.0 or self.rng.random() < np.exp(log_arg):
            sysm.apply_delete(i)
            self.energy += du
            self.n_accepts['delete'] += 1

    def _count(self, move: str, species: str, accepted: bool) -> None:
        """
        Count one attempt of a per-species move, overall and per species.

        Parameters
        ----------
        move : str
            ``'insert'`` or ``'delete'``.
        species : str
            The species moved.
        accepted : bool
            Whether the attempt was accepted.

        Returns
        -------
        None
        """
        key = f"{move}_{species}"
        self.n_attempts[move] += 1
        self.n_attempts[key] = self.n_attempts.get(key, 0) + 1
        if accepted:
            self.n_accepts[move] += 1
            self.n_accepts[key] = self.n_accepts.get(key, 0) + 1

    def _move_insert_species(self) -> None:
        """
        Attempt one insertion when several species are exchanged.

        Draws the species s uniformly among `system.exchange_species_list`,
        then a position as in `_move_insert`, and accepts with
        min(1, V / ((N_s + 1) Lambda_s^3) exp(beta (mu_s - dU))), N_s the
        number of s particles in the exchange zone (docs/REGC_algorithm.md,
        Sec. 8). Counted under ``'insert'`` and ``'insert_<s>'``.

        Returns
        -------
        None
        """
        self._insert_species_index(int(self.rng.integers(self.system.n_exchange_species)))

    def _insert_species_index(self, s: int) -> None:
        """
        Attempt one single-atom insertion of exchanged species number `s`.

        The body of `_move_insert_species` after the species draw.

        Parameters
        ----------
        s : int
            Index into `system.exchange_species_list`.

        Returns
        -------
        None
        """
        sysm = self.system
        sp = sysm.exchange_species_list[s]
        pos = sysm.random_insertion_position(self.rng)
        du = (sysm.pair_energy_of(pos, species=sp) if sysm.calculator.supports_incremental
              else sysm.generic_insert_dU(pos, species=sp))
        beta, mu, lam3 = self.state.beta, self.state.mu_vec[s], self.state.lambda_cubed_vec[s]
        n = len(sysm.exchange_zone_indices(sp))
        log_arg = beta * mu - beta * du + np.log(sysm.volume / ((n + 1) * lam3))
        accept = bool(log_arg >= 0.0 or self.rng.random() < np.exp(log_arg))
        if accept:
            sysm.apply_insert(pos, species=sp)
            self.energy += du
        self._count('insert', sp, accept)

    def _move_delete_species(self) -> None:
        """
        Attempt one deletion when several species are exchanged.

        Draws the species s uniformly, then one s particle of the exchange
        zone, and accepts with min(1, N_s Lambda_s^3 / V exp(-beta (mu_s +
        dU))) -- the reverse of `_move_insert_species`. A no-op if the zone
        holds no s particle. Counted under ``'delete'`` and
        ``'delete_<s>'``.

        Returns
        -------
        None
        """
        self._delete_species_index(int(self.rng.integers(self.system.n_exchange_species)))

    def _delete_species_index(self, s: int) -> None:
        """
        Attempt one single-atom deletion of exchanged species number `s`.

        The body of `_move_delete_species` after the species draw.

        Parameters
        ----------
        s : int
            Index into `system.exchange_species_list`.

        Returns
        -------
        None
        """
        sysm = self.system
        sp = sysm.exchange_species_list[s]
        candidates = sysm.exchange_zone_indices(sp)
        n = len(candidates)
        if n == 0:
            return
        i = int(candidates[self.rng.integers(n)])
        du = (-sysm.pair_energy_of(sysm.positions[i], exclude=i)
              if sysm.calculator.supports_incremental else sysm.generic_delete_dU(i))
        beta, mu, lam3 = self.state.beta, self.state.mu_vec[s], self.state.lambda_cubed_vec[s]
        log_arg = -beta * mu - beta * du + np.log(n * lam3 / sysm.volume)
        accept = bool(log_arg >= 0.0 or self.rng.random() < np.exp(log_arg))
        if accept:
            sysm.apply_delete(i)
            self.energy += du
        self._count('delete', sp, accept)

    # -- molecular insertion / deletion (regc.molecules) ---------------------------------
    def _move_gc_channel(self, insert: bool) -> None:
        """
        One GC attempt when molecules are exchanged: pick a channel, then move.

        The channels are every atomically exchanged species
        (`ParticleSystem.atomic_exchange`, weight 1 each), every molecule
        (weight `MoleculeMove.weight`) and, with subsurface sites
        (`SlabSystem.set_subsurface`), every subsurface species (weight
        ``Subsurface.weight``, NCMC moves); the choice does not depend on
        the configuration and is the same for insertions and deletions, so
        each channel satisfies detailed balance on its own.

        Parameters
        ----------
        insert : bool
            Insertion (True) or deletion (False).

        Returns
        -------
        None
        """
        sysm = self.system
        atomic = tuple(getattr(sysm, 'atomic_exchange', sysm.exchange_species_list))
        mols = tuple(getattr(sysm, 'molecules', ()))
        sub = getattr(sysm, 'subsurface', None)
        sub_species = tuple(sub.species) if sub is not None else ()
        w = np.array([1.0] * len(atomic) + [m.weight for m in mols]
                     + [sub.weight] * len(sub_species) if sub is not None
                     else [1.0] * len(atomic) + [m.weight for m in mols])
        c = int(self.rng.choice(len(w), p=w / w.sum()))
        if c < len(atomic):
            s = sysm.exchange_species_list.index(atomic[c])
            (self._insert_species_index if insert else self._delete_species_index)(s)
        elif c < len(atomic) + len(mols):
            m = mols[c - len(atomic)]
            (self._move_insert_molecule if insert else self._move_delete_molecule)(m)
        else:
            sp = sub_species[c - len(atomic) - len(mols)]
            (self._move_insert_subsurface if insert else self._move_delete_subsurface)(sp)

    def _log_activity(self, mol) -> float:
        """
        log prod_e z_e^{nu_e} of a molecule, z_e = exp(beta mu_e) / Lambda_e^3.

        Parameters
        ----------
        mol : MoleculeMove
            The molecule.

        Returns
        -------
        float
            The log activity of its atoms at this replica's state.
        """
        species = self.system.exchange_species_list
        mu, lam3, beta = self.state.mu_vec, self.state.lambda_cubed_vec, self.state.beta
        return float(sum(n * (beta * mu[species.index(e)] - np.log(lam3[species.index(e)]))
                         for e, n in mol.composition.items()))

    def _move_insert_molecule(self, mol) -> None:
        """
        Attempt to insert one molecule (`regc.molecules`).

        Anchor uniform in the exchange zone, partners from
        `MoleculeMove.propose`; accepted with min(1, prod_e z_e^{nu_e}
        exp(-beta dU) / (M' Q)), M' the number of deletable units after the
        insertion and Q the proposal density of the inserted unit
        (docs/REGC_algorithm.md, Sec. 10). Counted under ``'insert'`` and
        ``'insert_mol:<name>'``.

        Parameters
        ----------
        mol : MoleculeMove
            The molecule.

        Returns
        -------
        None
        """
        sysm = self.system
        anchor = sysm.random_insertion_position(self.rng)
        rel = mol.propose(self.rng)
        pos = np.array([sysm.wrap_position(p) for p in np.vstack([anchor, anchor + rel])])
        species = list(mol.symbols)
        du = sysm.generic_insert_many_dU(pos, species)
        accept = False
        if np.isfinite(du):
            P = np.vstack([sysm.positions, pos])
            S = np.append(sysm.species, species)
            Z = sysm.in_exchange_zone(P)
            new = range(sysm.n, sysm.n + len(pos))
            log_q = mol.log_unit_density(P, S, Z, new, sysm._min_image, sysm.volume)
            m_after = mol.count_units(P, S, Z, sysm._min_image)
            if np.isfinite(log_q) and m_after > 0:
                log_arg = (self._log_activity(mol) - self.state.beta * du
                           - np.log(m_after) - log_q)
                accept = bool(log_arg >= 0.0 or self.rng.random() < np.exp(log_arg))
        if accept:
            sysm.apply_insert_many(pos, species)
            self.energy += du
        self._count('insert', f"mol:{mol.name}", accept)

    def _move_delete_molecule(self, mol) -> None:
        """
        Attempt to delete one molecule (`regc.molecules`).

        Picks one of the M deletable units uniformly and accepts with
        min(1, prod_e z_e^{-nu_e} exp(-beta dU) M Q), the reverse of
        `_move_insert_molecule`. A no-op without units. Counted under
        ``'delete'`` and ``'delete_mol:<name>'``.

        Parameters
        ----------
        mol : MoleculeMove
            The molecule.

        Returns
        -------
        None
        """
        sysm = self.system
        P, S = sysm.positions, sysm.species
        Z = sysm.in_exchange_zone(P)
        unit, m_before = mol.random_unit(P, S, Z, sysm._min_image, self.rng)
        if unit is None:
            return
        du = sysm.generic_delete_many_dU(unit)
        log_q = mol.log_unit_density(P, S, Z, unit, sysm._min_image, sysm.volume)
        log_arg = (-self._log_activity(mol) - self.state.beta * du + np.log(m_before) + log_q)
        accept = bool(np.isfinite(log_arg)
                      and (log_arg >= 0.0 or self.rng.random() < np.exp(log_arg)))
        if accept:
            sysm.apply_delete_many(unit)
            self.energy += du
        self._count('delete', f"mol:{mol.name}", accept)

    # -- subsurface insertion / removal with NCMC (regc.subsurface) ----------------------
    def _subsurface_count(self, sp: str) -> int:
        """
        Number of mobile atoms of species `sp` in the subsurface region.

        Parameters
        ----------
        sp : str
            The species.

        Returns
        -------
        int
            The count.
        """
        sysm = self.system
        sub = sysm.subsurface
        return int(np.count_nonzero(sub.in_region(sysm.positions) & (sysm.species == sp)))

    def _log_z(self, sp: str) -> float:
        """
        log z_s = beta mu_s - ln Lambda_s^3 of an exchanged species.

        Parameters
        ----------
        sp : str
            The species.

        Returns
        -------
        float
            The log activity.
        """
        k = self.system.exchange_species_list.index(sp)
        return float(self.state.beta * self.state.mu_vec[k] - np.log(self.state.lambda_cubed_vec[k]))

    def _save_state(self) -> tuple:
        """
        Copies of everything an NCMC move may change.

        Returns
        -------
        tuple
            Mobile positions and species, substrate positions.
        """
        sysm = self.system
        return sysm.positions.copy(), sysm.species.copy(), sysm.substrate_positions.copy()

    def _restore_state(self, saved: tuple) -> None:
        """
        Undo an NCMC move.

        Parameters
        ----------
        saved : tuple
            Output of `_save_state`.

        Returns
        -------
        None
        """
        sysm = self.system
        sysm.positions, sysm.species, sysm.substrate_positions = (saved[0].copy(), saved[1].copy(),
                                                                  saved[2].copy())

    def _move_insert_subsurface(self, sp: str) -> None:
        """
        Attempt to insert an atom of species `sp` at a subsurface site, with NCMC.

        A site of the current metal configuration is drawn uniformly, the
        atom placed uniformly in its delta-ball and switched on (lambda 0 ->
        1) while the system relaxes (`regc.subsurface.ncmc`); accepted with
        min(1, z_s exp(-beta W) / (N_sub' q(y; x))), N_sub' the atoms of the
        species in the region at the end (docs/REGC_algorithm.md, Sec. 11).
        Rejected if the proposal or the final position of the atom is
        outside the region. Counted under ``'insert'`` and
        ``'insert_sub:<sp>'``.

        Parameters
        ----------
        sp : str
            The species.

        Returns
        -------
        None
        """
        from .subsurface import ncmc
        sysm, sub = self.system, self.system.subsurface
        sites = sub.sites(sysm)
        accept = False
        if len(sites):
            y = sysm.wrap_position(sub.propose(sites, self.rng))
            log_q = sub.log_q(y, sites, sysm)
            if np.isfinite(log_q) and np.isfinite(sysm.external_energy(y)):
                saved = self._save_state()
                sysm.positions = np.vstack([sysm.positions, y])
                sysm.species = np.append(sysm.species, sp)
                i = sysm.n - 1
                work, (u_with, _) = ncmc(sysm, i, True, sub, self.state.beta, self.rng)
                ok = np.isfinite(work) and sub.in_region(sysm.positions[i:i + 1])[0]
                if ok:
                    log_arg = (self._log_z(sp) - self.state.beta * work
                               - np.log(self._subsurface_count(sp)) - log_q)
                    accept = bool(log_arg >= 0.0 or self.rng.random() < np.exp(log_arg))
                if accept:
                    self.energy = u_with
                else:
                    self._restore_state(saved)
        self._count('insert', f"sub:{sp}", accept)

    def _move_delete_subsurface(self, sp: str) -> None:
        """
        Attempt to remove a subsurface atom of species `sp`, with NCMC.

        Picks one of the N_sub atoms of the species in the region
        uniformly, switches it off (lambda 1 -> 0) while the system relaxes,
        and deletes it; accepted with min(1, N_sub q(y; x') exp(-beta W) /
        z_s), y its final position and x' the configuration without it --
        the reverse of `_move_insert_subsurface`. A no-op without
        subsurface atoms. Counted under ``'delete'`` and
        ``'delete_sub:<sp>'``.

        Parameters
        ----------
        sp : str
            The species.

        Returns
        -------
        None
        """
        from .subsurface import ncmc
        sysm, sub = self.system, self.system.subsurface
        cand = np.flatnonzero(sub.in_region(sysm.positions) & (sysm.species == sp))
        if len(cand) == 0:
            return
        n_before = len(cand)
        i = int(cand[self.rng.integers(n_before)])
        saved = self._save_state()
        work, (_, u_without) = ncmc(sysm, i, False, sub, self.state.beta, self.rng)
        accept = False
        if np.isfinite(work):
            y = sysm.positions[i].copy()
            sysm.positions = np.delete(sysm.positions, i, axis=0)
            sysm.species = np.delete(sysm.species, i)
            log_q = sub.log_q(y, sub.sites(sysm), sysm)
            if np.isfinite(log_q):
                log_arg = (np.log(n_before) + log_q - self._log_z(sp) - self.state.beta * work)
                accept = bool(log_arg >= 0.0 or self.rng.random() < np.exp(log_arg))
        if accept:
            self.energy = u_without
        else:
            self._restore_state(saved)
        self._count('delete', f"sub:{sp}", accept)

    def _move_identity(self) -> None:
        """
        Attempt one semi-grand-canonical identity swap of a substrate atom.

        Picks a swappable substrate atom uniformly (`system.sgc_candidates`,
        a set the move does not change) and a new species uniformly among
        the other `system.sgc_species`: a symmetric proposal. Accepts with
        min(1, exp(-beta * dU + beta * (dmu_new - dmu_old))), dmu from
        `state.sgc_dmu` -- the Metropolis rule for the weight
        exp(-beta U + beta sum_k dmu_k N_k) at fixed total substrate count
        (docs/REGC_algorithm.md, Sec. 9). Counted under ``'identity'``. A
        no-op when the substrate is not semi-grand-canonical.

        Returns
        -------
        None
        """
        sysm = self.system
        species = getattr(sysm, 'sgc_species', ())
        if not species or self.state.sgc_dmu is None:
            return
        if getattr(sysm, 'sgc_canonical', False):
            self._move_pair_swap()
            return
        candidates = sysm.sgc_candidates()
        if len(candidates) == 0:
            return
        i = int(candidates[self.rng.integers(len(candidates))])
        old = str(sysm.substrate_species[i])
        k_old = species.index(old)
        k_new = int(self.rng.integers(len(species) - 1))
        k_new += k_new >= k_old            # uniform among the other species
        new = species[k_new]
        du = sysm.generic_identity_dU(i, new)
        dmu = self.state.sgc_dmu
        beta = self.state.beta
        log_arg = -beta * du + beta * (dmu[k_new] - dmu[k_old])
        self.n_attempts['identity'] = self.n_attempts.get('identity', 0) + 1
        if np.isfinite(log_arg) and (log_arg >= 0.0 or self.rng.random() < np.exp(log_arg)):
            sysm.apply_identity(i, new)
            self.energy += du
            self.n_accepts['identity'] = self.n_accepts.get('identity', 0) + 1

    def _move_pair_swap(self) -> None:
        """
        Attempt one canonical identity move: exchange the species of two atoms.

        Picks a swappable substrate atom i uniformly, then j uniformly among
        the swappable atoms of a different species, and exchanges their
        species. Every species count is unchanged, so the reverse move (j
        then i, or i then j) has the same probability: the proposal is
        symmetric and the acceptance is min(1, exp(-beta * dU)). Counted
        under ``'identity'``; a no-op when all swappable atoms share one
        species.

        Returns
        -------
        None
        """
        sysm = self.system
        candidates = sysm.sgc_candidates()
        if len(candidates) < 2:
            return
        i = int(candidates[self.rng.integers(len(candidates))])
        others = candidates[sysm.substrate_species[candidates] != sysm.substrate_species[i]]
        if len(others) == 0:
            return
        j = int(others[self.rng.integers(len(others))])
        du = sysm.generic_pair_swap_dU(i, j)
        log_arg = -self.state.beta * du
        self.n_attempts['identity'] = self.n_attempts.get('identity', 0) + 1
        if np.isfinite(log_arg) and (log_arg >= 0.0 or self.rng.random() < np.exp(log_arg)):
            sysm.apply_pair_swap(i, j)
            self.energy += du
            self.n_accepts['identity'] = self.n_accepts.get('identity', 0) + 1

    # -- [ZSG] Fig. 1 building blocks (used by ReplicaExchangeGCMC.run_regc) --
    def gc_attempt(self) -> None:
        """
        Make one particle insertion/removal attempt ([ZSG] Sec. II A 1).

        Removal if a uniform random number is below 0.5, insertion
        otherwise -- equal numbers of insertion and removal attempts, as
        detailed balance requires.

        Returns
        -------
        None
        """
        if self.rng.random() < 0.5:
            self._move_delete()
        else:
            self._move_insert()

    def regc_local_step(self, gc: bool, mode: str, n: int, **kwargs) -> "Replica":
        """
        The per-replica part of one REGC step: optional GC attempt, the
        `identity_attempts` semi-grand-canonical identity swaps (if the
        substrate is semi-grand-canonical), then diffusion.

        Parameters
        ----------
        gc : bool
            Whether to make a `gc_attempt` first (this step drew the GC
            branch rather than replica exchange).
        mode, n, **kwargs
            Passed to `diffuse`.

        Returns
        -------
        Replica
            `self`.
        """
        if gc:
            self.gc_attempt()
        for _ in range(getattr(self, 'identity_attempts', 0)):
            self._move_identity()
        self.diffuse(mode, n, **kwargs)
        return self

    def regc_local_block(self, gc_flags, first_step: int, record_every: int,
                         burn_in_steps: int, mode: str, n: int,
                         adsorbed_cutoff: Optional[float] = None, observables=None,
                         record_configurations: bool = False, **kwargs) -> "Replica":
        """
        Several consecutive REGC steps with no replica exchange in between.

        Between two RE moves the replicas are independent, so the engine
        hands each one a whole block of steps in a single (possibly remote)
        call instead of one call per step. The result is identical to
        calling `regc_local_step` once per step.

        Parameters
        ----------
        gc_flags : sequence of bool
            One entry per step: whether that step starts with a GC attempt.
        first_step : int
            1-based index of the block's first step in the whole run.
        record_every, burn_in_steps : int
            Record (energy, N) after step k if ``k > burn_in_steps`` and
            ``k % record_every == 0``, as `ReplicaExchangeGCMC.run_regc` does.
        adsorbed_cutoff : float, optional
            If given, each sample also carries
            ``system.n_adsorbed(adsorbed_cutoff)``; else None. Default None.
        observables : sequence of regc.observables.Observable, optional
            Evaluated on the configuration of every recorded sample.
            Default None.
        record_configurations : bool, optional
            If True, each sample also carries a copy of the configuration
            (`ParticleSystem.full_configuration` plus the group sizes).
            Default False.
        mode, n, **kwargs
            Passed to `diffuse`.

        Returns
        -------
        Replica
            `self`, with the samples of this block in `self.block_samples`
            as a list of ``(energy, N, N_ads, N_s, obs, config)``; ``N_s``
            is `system.species_counts()` with several exchanged species,
            else None; ``obs`` is ``{name: values}`` (empty without
            observables); ``config`` is ``(positions, species, n_frozen,
            n_substrate)`` or None.
        """
        self.block_samples = []
        for k, gc in enumerate(gc_flags):
            self.regc_local_step(gc, mode, n, **kwargs)
            step = first_step + k
            if step > burn_in_steps and step % record_every == 0:
                n_ads = (None if adsorbed_cutoff is None
                         else self.system.n_adsorbed(adsorbed_cutoff))
                n_s = (self.system.species_counts().tolist()
                       if self.system.n_exchange_species > 1 else None)
                obs = evaluate_observables(observables, self.system) if observables else {}
                self.block_samples.append((self.energy, self.system.n, n_ads, n_s, obs,
                                           snapshot(self.system) if record_configurations
                                           else None))
        return self

    def diffuse(self, mode: str = 'mc', n: int = 10, displace_substrate: bool = False,
                **md_kwargs) -> "Replica":
        """
        Sample the canonical (fixed N, V, T) ensemble ([ZSG] Sec. II A 3).

        Parameters
        ----------
        mode : str, optional
            ``'mc_moves'``: `n` single-atom displacement attempts (Eq. (3)).
            ``'mc'``: `n` MC cycles, each of as many single-atom
            displacement attempts (Eq. (3)) as there are movable atoms, so
            each is attempted once on average. ``'hmc'``: `n` HMC
            trajectories (`_move_hmc`; needs forces). ``'md'``: `n` steps of
            ASE NVT molecular dynamics (`_diffuse_md`). Default is ``'mc'``.
        n : int, optional
            Number of cycles / trajectories / MD steps. Default is 10.
        displace_substrate : bool, optional
            For ``'mc'``: also count displaceable substrate atoms as
            movable (each attempt picks uniformly among mobile + substrate
            atoms). HMC and MD always move substrate atoms. Default False.
        **md_kwargs
            Passed to `_diffuse_md` for ``'md'``.

        Returns
        -------
        Replica
            `self`.

        Raises
        ------
        ValueError
            If `mode` is unknown, or ``'hmc'`` is requested with a
            calculator without forces.
        """
        if mode == 'mc_moves':
            # n single displacement attempts, each of a uniformly chosen
            # movable atom ([ZSG] Sec. III A: "ten sequential MC moves")
            n_mob = self.system.n
            n_mov = n_mob + (self.system.n_substrate if displace_substrate else 0)
            if n_mov:
                for _ in range(n):
                    if self.rng.integers(n_mov) < n_mob:
                        self._move_displace()
                    else:
                        self._move_displace_substrate()
        elif mode == 'mc':
            for _ in range(n):
                n_mob = self.system.n
                n_mov = n_mob + (self.system.n_substrate if displace_substrate else 0)
                for _ in range(n_mov):
                    if self.rng.integers(n_mov) < n_mob:
                        self._move_displace()
                    else:
                        self._move_displace_substrate()
        elif mode == 'hmc':
            if not self.system.calculator.supports_forces:
                raise ValueError("diffusion='hmc' needs a Calculator with supports_forces=True")
            for _ in range(n):
                self._move_hmc()
        elif mode == 'md':
            self._diffuse_md(n, **md_kwargs)
        else:
            raise ValueError(f"diffusion mode must be 'mc_moves', 'mc', 'hmc' or 'md', got {mode!r}")
        return self

    def _diffuse_md(self, n_steps: int, timestep_fs: float = 1.0, thermostat: str = 'bussi',
                    taut_fs: float = 20.0, friction_per_fs: float = 0.01) -> None:
        """
        Run `n_steps` of NVT molecular dynamics with ASE.

        Velocities are redrawn from Maxwell-Boltzmann at this slot's T at
        the start of every segment (positions and momenta are independent
        in the canonical ensemble, so this is exact, and it handles freshly
        inserted particles and configurations just swapped in from another
        T). Frozen atoms are held by `FixAtoms`; displaceable substrate and
        mobile atoms move. With a thermostat region
        (`ParticleSystem.set_thermostat_region`: the fixed / thermostat /
        free slab) only that part of the substrate is thermostatted and the
        rest moves NVE (`_md_partial_thermostat`). Hard walls (`system.md_wall_bounds()`: the z-walls
        of a slab, all six faces of a cluster box) are
        enforced by specular reflection after every step, like the
        reflecting boundaries [ZSG] apply via PLUMED. A segment ending at
        non-finite positions or energy is discarded (``md_reverted``).

        Unlike MC/HMC, MD has no accept/reject step: the sampled ensemble
        carries the integrator's time-step error ([ZSG] use 1 fs with a
        Bussi thermostat, tau = 20 fs).

        Units: `self.state.temperature` must be k_B*T in eV (as for every
        real-units Calculator in this package); positions in Angstrom,
        masses in amu (from `ParticleSystem.masses()`).

        Parameters
        ----------
        n_steps : int
            Number of MD steps.
        timestep_fs : float, optional
            Time step in fs. Default is 1.0.
        thermostat : str, optional
            ``'bussi'`` (stochastic velocity rescaling, as in [ZSG]) or
            ``'langevin'``. Default is ``'bussi'``.
        taut_fs : float, optional
            Bussi relaxation time in fs. Default is 20.0.
        friction_per_fs : float or None, optional
            Langevin friction in 1/fs; None means ``1 / taut_fs`` (the
            LAMMPS backend's convention). Default is 0.01.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the calculator is not an `ASECalculatorAdapter`, the system
            has a continuum wall (MD would not feel it), or the thermostat
            is unknown.
        """
        sysm = self.system
        calc = sysm.calculator
        if friction_per_fs is None:      # as LAMMPS: Langevin damping time = taut_fs
            friction_per_fs = 1.0 / taut_fs
        if hasattr(calc, 'run_md'):  # e.g. LammpsCalculator: the whole segment natively
            self._diffuse_md_native(n_steps, timestep_fs, thermostat, taut_fs, friction_per_fs)
            return

        from ase import Atoms, units
        from ase.constraints import FixAtoms
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

        if not hasattr(calc, 'ase_calculator'):
            raise ValueError("diffusion='md' runs ASE molecular dynamics and needs an "
                             "ASECalculatorAdapter; use 'mc' or 'hmc' for other calculators.")
        if getattr(sysm, 'wall_epsilon', 0.0) > 0.0:
            raise ValueError("diffusion='md' integrates calculator forces only and would not "
                             "feel the continuum wall (wall_epsilon > 0); use 'mc' or 'hmc'.")
        n_fr = sysm.frozen_positions.shape[0]
        n_sub, n_mob = sysm.n_substrate, sysm.n
        if n_sub + n_mob == 0 or n_steps <= 0:
            return

        positions, species = sysm.full_configuration()
        atoms = Atoms(symbols=list(species), positions=positions, cell=calc.cell, pbc=calc.pbc)
        atoms.set_masses(np.concatenate([sysm._mass_lookup(sysm.frozen_species),
                                         sysm.substrate_masses(), sysm.masses()]))
        if n_fr:
            atoms.set_constraint(FixAtoms(indices=list(range(n_fr))))
        atoms.calc = calc.ase_calculator

        T_K = self.state.temperature / units.kB
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_K, rng=self.rng)
        dt = timestep_fs * units.fs
        walls = [(k, b) for k, b in enumerate(sysm.md_wall_bounds()) if b is not None]
        mask = getattr(sysm, 'thermostat_mask', None)
        if mask is not None:
            if thermostat not in ('bussi', 'langevin'):
                raise ValueError(f"thermostat must be 'bussi' or 'langevin', got {thermostat!r}")
            self._md_partial_thermostat(atoms, n_fr, n_fr + np.flatnonzero(mask), n_steps,
                                        dt, walls, thermostat, timestep_fs / taut_fs,
                                        friction_per_fs * timestep_fs)
            self._end_md_segment(atoms.get_positions()[n_fr:], sysm.total_energy)
            return
        if thermostat == 'bussi':
            from ase.md.bussi import Bussi
            dyn = Bussi(atoms, dt, temperature_K=T_K, taut=taut_fs * units.fs, rng=self.rng)
        elif thermostat == 'langevin':
            from ase.md.langevin import Langevin
            dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction_per_fs / units.fs,
                           fixcm=False, rng=self.rng)
        else:
            raise ValueError(f"thermostat must be 'bussi' or 'langevin', got {thermostat!r}")

        if walls:
            def reflect() -> None:
                pos, vel = atoms.get_positions(), atoms.get_velocities()
                hit = False
                for k, (lo, hi) in walls:
                    x, v = pos[n_fr:, k], vel[n_fr:, k]
                    low, high = x <= lo, x > hi
                    if low.any() or high.any():
                        x[low] = 2.0 * lo - x[low]
                        x[high] = 2.0 * hi - x[high]
                        v[low | high] *= -1.0
                        hit = True
                if hit:
                    atoms.set_positions(pos, apply_constraint=False)
                    atoms.set_velocities(vel)

            dyn.attach(reflect, interval=1)
        dyn.run(n_steps)

        self._end_md_segment(atoms.get_positions()[n_fr:], sysm.total_energy)

    def _end_md_segment(self, new: np.ndarray, energy_of) -> None:
        """
        Take the end point of an MD segment, or discard it if it is not finite.

        Parameters
        ----------
        new : numpy.ndarray
            End positions of the substrate then the mobile atoms, (n_sub + n, 3).
        energy_of : callable
            Returns the energy of the system once the moves are applied.

        Returns
        -------
        None
            The positions and `self.energy` are updated in place; a segment
            that ends at non-finite positions or energy (e.g. inside a hard
            core of the calculator) is discarded and counted in
            ``n_attempts['md_reverted']``.
        """
        sysm = self.system
        n_sub = sysm.n_substrate
        saved_sub, saved_mob = sysm.substrate_positions.copy(), sysm.positions.copy()
        energy = np.nan
        if np.all(np.isfinite(new)):
            for i in range(n_sub):
                sysm.apply_move_substrate(i, new[i])   # wraps x, y through the geometry
            for i in range(sysm.n):
                sysm.apply_move(i, new[n_sub + i])
            energy = energy_of()
        if np.isfinite(energy):
            self.energy = energy
        else:
            sysm.substrate_positions, sysm.positions = saved_sub, saved_mob
            self.n_attempts['md_reverted'] = self.n_attempts.get('md_reverted', 0) + 1

    def _md_partial_thermostat(self, atoms, n_fr: int, bath: np.ndarray, n_steps: int,
                               dt: float, walls: list, thermostat: str, dt_over_tau: float,
                               gamma_dt: float) -> None:
        """
        Velocity Verlet for every moving atom, thermostat on the `bath` atoms only.

        Each step is thermostat(dt/2) - velocity Verlet(dt) - thermostat(dt/2),
        the thermostat acting on the bath atoms alone: Bussi stochastic
        velocity rescaling with their 3 * len(bath) degrees of freedom, or a
        Langevin (Ornstein-Uhlenbeck) velocity update. The other moving
        atoms (atoms ``n_fr`` onwards outside `bath`) are integrated NVE.
        Hard walls reflect positions and velocities after every drift.
        `atoms` is updated in place (positions and velocities).

        Parameters
        ----------
        atoms : ase.Atoms
            The full configuration (frozen first, held by `FixAtoms`) with
            a calculator and Maxwell-Boltzmann velocities already set.
        n_fr : int
            Number of frozen atoms (the first rows).
        bath : numpy.ndarray
            Indices (into `atoms`) of the thermostatted atoms.
        n_steps : int
            Number of steps.
        dt : float
            Time step, in ASE time units.
        walls : list
            ``[(axis, (lo, hi)), ...]`` hard walls (`md_wall_bounds`).
        thermostat : str
            ``'bussi'`` or ``'langevin'``.
        dt_over_tau : float
            Time step over the Bussi time constant.
        gamma_dt : float
            Langevin friction times the time step.

        Returns
        -------
        None
        """
        kT = self.state.temperature
        m = atoms.get_masses()[:, None]
        x, v = atoms.get_positions(), atoms.get_velocities()
        f = atoms.get_forces()
        mb = m[bath]
        ndof = 3 * len(bath)
        c_half = np.exp(-0.5 * dt_over_tau)                  # Bussi, per half step
        c1 = np.exp(-0.5 * gamma_dt)                         # Langevin, per half step
        c2 = np.sqrt((1.0 - c1 * c1) * kT / mb)

        def bath_step() -> None:
            if thermostat == 'langevin':
                v[bath] = c1 * v[bath] + c2 * self.rng.standard_normal((len(bath), 3))
                return
            kin = 0.5 * float(np.sum(mb * v[bath] ** 2))
            if kin <= 0.0:
                return
            s = (1.0 - c_half) * (0.5 * ndof * kT) / kin / ndof
            r1 = self.rng.standard_normal()
            chi2 = 2.0 * self.rng.standard_gamma(0.5 * (ndof - 1)) if ndof > 1 else 0.0
            alpha2 = c_half + s * (chi2 + r1 * r1) + 2.0 * r1 * np.sqrt(c_half * s)
            v[bath] *= np.sqrt(max(alpha2, 0.0))

        mov = slice(n_fr, None)
        for _ in range(int(n_steps)):
            bath_step()
            v[mov] += 0.5 * dt * f[mov] / m[mov]
            x[mov] += dt * v[mov]
            for k, (lo, hi) in walls:
                xk, vk = x[mov, k], v[mov, k]
                low, high = xk <= lo, xk > hi
                if low.any() or high.any():
                    xk[low] = 2.0 * lo - xk[low]
                    xk[high] = 2.0 * hi - xk[high]
                    vk[low | high] *= -1.0
                    x[mov, k], v[mov, k] = xk, vk
            atoms.set_positions(x, apply_constraint=False)
            f = atoms.get_forces()
            v[mov] += 0.5 * dt * f[mov] / m[mov]
            bath_step()
        atoms.set_velocities(v)

    def _diffuse_md_native(self, n_steps: int, timestep_fs: float, thermostat: str,
                           taut_fs: float, friction_per_fs: float = 0.01) -> None:
        """
        MD segment run by the calculator itself (`calculator.run_md`).

        Same physics as the ASE path of `_diffuse_md` (fresh Maxwell-
        Boltzmann velocities, thermostat, reflecting walls, frozen atoms
        fixed), but the whole segment runs inside the engine (LAMMPS). If
        the end point has infinite energy (e.g. a hard core of the
        calculator), the segment is discarded and counted in
        ``n_attempts['md_reverted']``.

        Parameters
        ----------
        n_steps : int
            Number of MD steps.
        timestep_fs : float
            Time step, in fs.
        thermostat : str
            ``'bussi'`` or ``'langevin'``.
        taut_fs : float
            Bussi thermostat time constant, in fs.
        friction_per_fs : float, optional
            Langevin friction, in 1/fs (as in the ASE path). Default 0.01.
        """
        sysm = self.system
        n_sub = sysm.n_substrate
        if n_sub + sysm.n == 0 or n_steps <= 0:
            return
        seed = int(self.rng.integers(1, 2 ** 31 - 1))
        new = sysm.calculator.run_md(sysm, n_steps, self.state.temperature, timestep_fs,
                                     taut_fs, seed, thermostat=thermostat,
                                     friction_per_fs=friction_per_fs)
        if hasattr(sysm.calculator, "energy_after_md"):   # the segment's own end energy
            def energy_of() -> float:
                return sysm.calculator.energy_after_md(sysm) + sysm.external_energy_total()
        else:
            energy_of = sysm.total_energy
        self._end_md_segment(np.asarray(new, dtype=float), energy_of)

    def sweep(self, n_steps: int, p_displace: float = 0.5,
              p_insert: float = 0.25, p_delete: float = 0.25,
              p_displace_substrate: float = 0.0) -> "Replica":
        """
        Run a batch of randomly-chosen Monte Carlo moves.

        `p_displace_substrate` (default 0, i.e. off) adds an independent
        single-particle MC move for substrate atoms, on top of whatever
        substrate motion already happens via HMC's joint trajectory when
        `hmc_enabled=True` -- the two aren't mutually exclusive. With no
        substrate atoms present, this is a no-op regardless of the
        probability given (see `_move_displace_substrate`).

        Parameters
        ----------
        n_steps : int
            Number of moves to attempt in this sweep.
        p_displace : float, optional
            Relative probability of a displacement move (single-particle
            MC, or the joint HMC trajectory if `self.hmc_enabled` is
            True). Default is 0.5.
        p_insert : float, optional
            Relative probability of a GCMC insertion move. Default is
            0.25.
        p_delete : float, optional
            Relative probability of a GCMC deletion move; must equal
            `p_insert`. Default is 0.25.
        p_displace_substrate : float, optional
            Relative probability of an independent substrate displacement
            move. Default is 0.0 (off).

        Returns
        -------
        Replica
            `self`, returned so a process-pool worker can send the
            mutated replica back to the main process.

        Raises
        ------
        ValueError
            If `p_insert` differs from `p_delete` (the acceptance rules
            assume equal attempt probabilities).
        """
        if not np.isclose(p_insert, p_delete):
            raise ValueError(f"p_insert ({p_insert}) must equal p_delete ({p_delete}): the "
                             f"insertion/deletion acceptance rules assume equal attempt "
                             f"probabilities (detailed balance)")
        probs = np.array([p_displace, p_insert, p_delete, p_displace_substrate], dtype=float)
        probs /= probs.sum()
        choices = self.rng.choice(4, size=n_steps, p=probs)
        move_fn = self._move_hmc if self.hmc_enabled else self._move_displace
        for c in choices:
            if c == 0:
                move_fn()
            elif c == 1:
                self._move_insert()
            elif c == 2:
                self._move_delete()
            else:
                self._move_displace_substrate()
        return self  # returned so a process-pool worker can send the mutated replica back

    def check_energy(self, tol: float = 1e-6) -> "Replica":
        """
        Compare the running energy with a full recomputation, and resync it.

        MC moves update the energy incrementally (``energy += dU``); this
        catches any drift (round-off, or a calculator whose incremental and
        full energies disagree). A drift larger than `tol` per atom is
        stored in `last_energy_drift` and the energy is reset to the
        recomputed value; a smaller one is only recorded, so runs whose
        energies agree to round-off stay bit-identical.

        Parameters
        ----------
        tol : float, optional
            Tolerance per atom (substrate + mobile), in energy units.
            Default 1e-6.

        Returns
        -------
        Replica
            `self` (for process-pool workers), with ``last_energy_drift``
            (running minus recomputed energy) and ``energy_resynced``.
        """
        exact = self.system.total_energy()
        drift = float(self.energy - exact) if np.isfinite(exact) else np.inf
        n_atoms = max(1, self.system.n + self.system.n_substrate)
        self.last_energy_drift = drift
        self.energy_resynced = bool(abs(drift) > tol * n_atoms)
        if self.energy_resynced and np.isfinite(exact):
            self.energy = exact
        return self

    def acceptance_rates(self) -> Dict[str, float]:
        """
        Compute the acceptance rate of each move type attempted so far.

        Returns
        -------
        dict of str to float
            One entry per move type (``'displace'``, ``'insert'``,
            ``'delete'``, ``'hmc'``, ``'displace_substrate'``, and with
            several exchanged species ``'insert_<s>'``/``'delete_<s>'``): the
            fraction of attempts of that type that were accepted (0.0 for
            a move type never attempted).
        """
        return {k: self.n_accepts.get(k, 0) / max(1, self.n_attempts[k]) for k in self.n_attempts}
