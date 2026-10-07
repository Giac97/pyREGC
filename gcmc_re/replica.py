"""One grid slot: a fixed (T, mu) plus the GCMC move set and bookkeeping."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .state import ThermoState
from .systems import ParticleSystem


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
                            'displace_substrate': 0}
        self.n_accepts = {'displace': 0, 'insert': 0, 'delete': 0, 'hmc': 0,
                           'displace_substrate': 0}
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

        Returns
        -------
        None
        """
        sysm = self.system
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

        Returns
        -------
        None
        """
        sysm = self.system
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
        The per-replica part of one REGC step: optional GC attempt, then
        diffusion.

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
        self.diffuse(mode, n, **kwargs)
        return self

    def regc_local_block(self, gc_flags, first_step: int, record_every: int,
                         burn_in_steps: int, mode: str, n: int, **kwargs) -> "Replica":
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
        mode, n, **kwargs
            Passed to `diffuse`.

        Returns
        -------
        Replica
            `self`, with the samples of this block in `self.block_samples`
            as a list of ``(energy, N)``.
        """
        self.block_samples = []
        for k, gc in enumerate(gc_flags):
            self.regc_local_step(gc, mode, n, **kwargs)
            step = first_step + k
            if step > burn_in_steps and step % record_every == 0:
                self.block_samples.append((self.energy, self.system.n))
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
        mobile atoms move. Hard z-walls (`system.md_z_bounds()`) are
        enforced by specular reflection after every step, like the
        reflecting boundaries [ZSG] apply via PLUMED.

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
        friction_per_fs : float, optional
            Langevin friction in 1/fs. Default is 0.01.

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
        if hasattr(calc, 'run_md'):  # e.g. LammpsCalculator: the whole segment natively
            self._diffuse_md_native(n_steps, timestep_fs, thermostat, taut_fs)
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
        if thermostat == 'bussi':
            from ase.md.bussi import Bussi
            dyn = Bussi(atoms, dt, temperature_K=T_K, taut=taut_fs * units.fs, rng=self.rng)
        elif thermostat == 'langevin':
            from ase.md.langevin import Langevin
            dyn = Langevin(atoms, dt, temperature_K=T_K, friction=friction_per_fs / units.fs,
                           fixcm=False, rng=self.rng)
        else:
            raise ValueError(f"thermostat must be 'bussi' or 'langevin', got {thermostat!r}")

        bounds = sysm.md_z_bounds()
        if bounds is not None:
            z_min, z_max = bounds

            def reflect() -> None:
                pos, vel = atoms.get_positions(), atoms.get_velocities()
                z, vz = pos[n_fr:, 2], vel[n_fr:, 2]
                low, high = z <= z_min, z > z_max
                if low.any() or high.any():
                    z[low] = 2.0 * z_min - z[low]
                    z[high] = 2.0 * z_max - z[high]
                    vz[low | high] *= -1.0
                    atoms.set_positions(pos, apply_constraint=False)
                    atoms.set_velocities(vel)

            dyn.attach(reflect, interval=1)
        dyn.run(n_steps)

        new = atoms.get_positions()[n_fr:]
        for i in range(n_sub):
            sysm.apply_move_substrate(i, new[i])  # wraps x, y through the geometry
        for i in range(n_mob):
            sysm.apply_move(i, new[n_sub + i])
        self.energy = sysm.total_energy()

    def _diffuse_md_native(self, n_steps: int, timestep_fs: float, thermostat: str,
                           taut_fs: float) -> None:
        """
        MD segment run by the calculator itself (`calculator.run_md`).

        Same physics as the ASE path of `_diffuse_md` (fresh Maxwell-
        Boltzmann velocities, thermostat, reflecting z-walls, frozen atoms
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
            Thermostat time constant, in fs.
        """
        sysm = self.system
        n_sub = sysm.n_substrate
        if n_sub + sysm.n == 0 or n_steps <= 0:
            return
        seed = int(self.rng.integers(1, 2 ** 31 - 1))
        new = sysm.calculator.run_md(sysm, n_steps, self.state.temperature, timestep_fs,
                                     taut_fs, seed, thermostat=thermostat)
        saved_sub, saved_mob = sysm.substrate_positions.copy(), sysm.positions.copy()
        for i in range(n_sub):
            sysm.apply_move_substrate(i, new[i])
        for i in range(sysm.n):
            sysm.apply_move(i, new[n_sub + i])
        energy = sysm.total_energy()
        if np.isfinite(energy):
            self.energy = energy
        else:
            sysm.substrate_positions, sysm.positions = saved_sub, saved_mob
            self.n_attempts['md_reverted'] = self.n_attempts.get('md_reverted', 0) + 1

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
            Relative probability of a GCMC deletion move. Default is
            0.25.
        p_displace_substrate : float, optional
            Relative probability of an independent substrate displacement
            move. Default is 0.0 (off).

        Returns
        -------
        Replica
            `self`, returned so a process-pool worker can send the
            mutated replica back to the main process.
        """
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

    def acceptance_rates(self) -> Dict[str, float]:
        """
        Compute the acceptance rate of each move type attempted so far.

        Returns
        -------
        dict of str to float
            One entry per move type (``'displace'``, ``'insert'``,
            ``'delete'``, ``'hmc'``, ``'displace_substrate'``): the
            fraction of attempts of that type that were accepted (0.0 for
            a move type never attempted).
        """
        return {k: self.n_accepts.get(k, 0) / max(1, self.n_attempts[k]) for k in self.n_attempts}
