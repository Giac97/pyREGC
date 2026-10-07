"""
Pluggable energy-evaluation backends.

`Calculator` is the interface `ParticleSystem` delegates all physics to.
`LJCalculator` is the fast, incremental pairwise Lennard-Jones potential;
`PairLJCalculator` is its multi-species version that also sees frozen and
substrate atoms.
`ASECalculatorAdapter` wraps any ASE-compatible Calculator (EAM, an MLIP
such as MACE/NequIP/Allegro, ASE's own EMT/LennardJones, ...) so it can
drive this GCMC engine -- see its docstring for why that needs a different
(generic, always-correct-but-slower) move implementation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .systems import ParticleSystem


class Calculator:
    """Energy-evaluation backend. `ParticleSystem` delegates all physics to
    one of these; everything else (geometry, boundaries, GCMC bookkeeping)
    is completely agnostic to what potential is used.

    `supports_incremental = True` means the calculator also implements
    `pair_energy_of(system, pos, exclude=None)`, the energy of a single
    point's interaction with the rest of the system -- unlocking the fast
    incremental single-particle GCMC path. Leave it False (the default) for
    any potential where that isn't available or safe to hand-derive; Replica
    then uses the generic clone-and-recompute path automatically, which is
    always correct.

    `supports_forces = True` means the calculator also implements
    `forces(system)`, returning the force (eV/Angstrom, or the matching
    reduced-unit force, per system) on each MOBILE particle, in
    `system.positions` order -- what `Replica`'s HMC move needs (see
    `replica.py`). Leave it False for any potential with no cheap/available
    force evaluation; HMC then just isn't available for that calculator."""

    supports_incremental = False
    supports_forces = False

    def total_energy(self, system: "ParticleSystem") -> float:
        """
        Compute the total interatomic energy for a configuration.

        Parameters
        ----------
        system : ParticleSystem
            The system whose current configuration (positions, species,
            and any frozen/substrate atoms) should be evaluated.

        Returns
        -------
        float
            The calculator's own contribution to the total energy (i.e.
            not including any `ParticleSystem.external_energy` field,
            which `ParticleSystem.total_energy` adds separately).

        Raises
        ------
        NotImplementedError
            Always, on the base class -- subclasses must override this.
        """
        raise NotImplementedError

    def forces(self, system: "ParticleSystem") -> np.ndarray:
        """
        Compute forces on the movable particles, for HMC.

        Parameters
        ----------
        system : ParticleSystem
            The system whose current configuration should be evaluated.

        Returns
        -------
        numpy.ndarray
            The force on each movable particle, in an array-specific
            order documented by the concrete subclass (see
            `LJCalculator.forces` and `ASECalculatorAdapter.forces`).

        Raises
        ------
        NotImplementedError
            Always, on the base class, and on any subclass with
            `supports_forces = False` (the default) that doesn't
            override this method.
        """
        raise NotImplementedError


def _check_min_image(system: "ParticleSystem", r_cut: float) -> None:
    """
    Refuse a cutoff the minimum-image convention cannot honour.

    Beyond `system.min_image_radius` a particle can interact with more than
    one periodic image of another, which `_min_image`-based sums silently
    miss (LAMMPS, for instance, counts them all).

    Parameters
    ----------
    system : ParticleSystem
        The system (its `min_image_radius`, if it has one).
    r_cut : float
        The interaction cutoff to check.

    Raises
    ------
    ValueError
        If ``r_cut > system.min_image_radius``.
    """
    if r_cut > getattr(system, 'min_image_radius', np.inf) + 1e-12:
        raise ValueError(
            f"interaction cutoff {r_cut:.4g} exceeds the minimum-image radius "
            f"{system.min_image_radius:.4g} of this cell (half its smallest lattice-plane "
            f"spacing): periodic images beyond the nearest would be missed. Use a larger "
            f"cell, a shorter cutoff, or a calculator that counts all images (LAMMPS).")


class LJCalculator(Calculator):
    """The original fast, incremental pairwise Lennard-Jones potential."""

    supports_incremental = True
    supports_forces = True

    def __init__(self, sigma: float = 1.0, epsilon: float = 1.0, r_cut: float = 2.5):
        """
        Construct a Lennard-Jones calculator.

        Parameters
        ----------
        sigma : float, optional
            The Lennard-Jones length parameter (the separation at which
            the potential crosses zero). Default is 1.0.
        epsilon : float, optional
            The Lennard-Jones energy parameter (the well depth). Default
            is 1.0.
        r_cut : float, optional
            The cutoff radius beyond which interactions are ignored. The
            potential is shifted so that U(r_cut) = 0, avoiding an energy
            discontinuity at the cutoff. Default is 2.5.
        """
        self.sigma, self.epsilon, self.r_cut = sigma, epsilon, r_cut
        self.r_cut2 = r_cut * r_cut
        sr6 = (sigma / r_cut) ** 6
        self.e_shift = 4.0 * epsilon * (sr6 ** 2 - sr6)  # U(r_cut) = 0

    def pair_energy_of(self, system: "ParticleSystem", pos: np.ndarray,
                        exclude: Optional[int] = None) -> float:
        """
        Compute the pairwise energy of a single point with the system.

        Parameters
        ----------
        system : ParticleSystem
            The system supplying the other particles and the minimum-image
            convention (`system._min_image`).
        pos : numpy.ndarray
            The (3,) Cartesian position of the point to evaluate.
        exclude : int, optional
            If given, the index of a particle in `system.positions` to
            exclude from the sum -- used when `pos` is (a trial position
            for) that particle itself, so it doesn't interact with its own
            unmoved copy. Default is None (no exclusion).

        Returns
        -------
        float
            The sum of pairwise Lennard-Jones interactions between `pos`
            and every other (non-excluded) mobile particle within
            `r_cut`, using the minimum-image convention.
        """
        if system.n == 0:
            return 0.0
        _check_min_image(system, self.r_cut)
        others = system.positions
        if exclude is not None:
            mask = np.ones(system.n, dtype=bool)
            mask[exclude] = False
            others = others[mask]
            if others.shape[0] == 0:
                return 0.0
        dr = system._min_image(others - pos)
        r2 = np.einsum('ij,ij->i', dr, dr)
        inside = r2 < self.r_cut2
        if not np.any(inside):
            return 0.0
        r2 = r2[inside]
        sr6 = (self.sigma ** 2 / r2) ** 3
        u = 4.0 * self.epsilon * (sr6 ** 2 - sr6) - self.e_shift
        return float(np.sum(u))

    def total_energy(self, system: "ParticleSystem") -> float:
        """
        Compute the total pairwise Lennard-Jones energy of the system.

        Parameters
        ----------
        system : ParticleSystem
            The system whose mobile particles (`system.positions`) should
            be evaluated. Frozen and substrate atoms are NOT included --
            a documented limitation of `LJCalculator` (see the class
            docstring); use `ASECalculatorAdapter` for a potential that
            accounts for them.

        Returns
        -------
        float
            The sum of pairwise Lennard-Jones interactions over every
            pair of mobile particles within `r_cut`.
        """
        if system.n < 2:
            return 0.0
        return sum(self.pair_energy_of(system, system.positions[i], exclude=i)
                   for i in range(system.n)) / 2.0

    def forces(self, system: "ParticleSystem") -> np.ndarray:
        """
        Compute forces on every mobile particle, fully vectorized.

        Every pair at once (an (N, N, 3) displacement tensor,
        minimum-imaged), no per-particle Python loop -- validated to match
        a per-particle loop exactly, and about 10x faster, which matters
        since HMC calls this every leapfrog half-step.

        Parameters
        ----------
        system : ParticleSystem
            The system whose mobile particles (`system.positions`) should
            be evaluated. As with `total_energy`, frozen and substrate
            atoms are not included.

        Returns
        -------
        numpy.ndarray
            Array of shape (system.n, 3): the net Lennard-Jones force on
            each mobile particle, in `system.positions` order.
        """
        n = system.n
        if n < 2:
            return np.zeros((n, 3))
        _check_min_image(system, self.r_cut)
        pos = system.positions
        dr = system._min_image(pos[:, None, :] - pos[None, :, :])  # (N,N,3): r_i - r_j
        r2 = np.einsum('ijk,ijk->ij', dr, dr)
        np.fill_diagonal(r2, np.inf)  # exclude self-interaction
        inside = r2 < self.r_cut2
        r2_safe = np.where(inside, r2, 1.0)
        s = np.where(inside, (self.sigma ** 2) / r2_safe, 0.0)
        coeff = np.where(inside, (24.0 * self.epsilon / r2_safe) * (2 * s ** 6 - s ** 3), 0.0)
        return np.einsum('ij,ijk->ik', coeff, dr)  # sum over j for each i


class PairLJCalculator(Calculator):
    """
    Multi-species, shifted 12-6 Lennard-Jones with per-pair parameters.

    Unlike `LJCalculator`, this one sees every atom category -- frozen,
    displaceable substrate and mobile -- so a rigid LJ surface (e.g. the
    A18 fcc(111) slab of Zhou, Scheffler & Ghiringhelli, PRB 100, 174106)
    is evaluated natively, on the fast incremental path. Pairs of species
    with no entry in `pair_params` do not interact (the paper defines no
    A-A interaction, for instance).

    Each pair (a, b) has phi(r) = 4 eps [(sig/r)^12 - (sig/r)^6] - phi(r_c)
    for r < r_c = r_cut_factor * sig, and 0 beyond (shifted so phi(r_c) = 0).
    """

    supports_incremental = True
    supports_forces = True

    def __init__(self, pair_params: dict, r_cut_factor: float = 2.5):
        """
        Construct the calculator.

        Parameters
        ----------
        pair_params : dict
            ``{(species_a, species_b): (epsilon, sigma)}``; order of the two
            species does not matter.
        r_cut_factor : float, optional
            Cutoff of each pair in units of its sigma. Default is 2.5.
        """
        self.r_cut_factor = r_cut_factor
        self.params = {}
        for (a, b), (eps, sig) in pair_params.items():
            rc = r_cut_factor * sig
            sr6 = (sig / rc) ** 6
            entry = (float(eps), float(sig), rc * rc, 4.0 * eps * (sr6 ** 2 - sr6))
            self.params[(a, b)] = entry
            self.params[(b, a)] = entry
        self.max_r_cut = r_cut_factor * max(sig for _, sig in pair_params.values())
        # integer species codes + (code, code) lookup tables for the fast path;
        # code len(species) is "any other species" (interacts with nothing)
        names = sorted({x for pair in pair_params for x in pair})
        self._code = {name: i for i, name in enumerate(names)}
        n = len(names) + 1
        self._eps = np.zeros((n, n)); self._sig2 = np.ones((n, n))
        self._rc2 = np.zeros((n, n)); self._shift = np.zeros((n, n))
        for (a, b), (eps, sig, rc2, shift) in self.params.items():
            i, j = self._code[a], self._code[b]
            self._eps[i, j], self._sig2[i, j], self._rc2[i, j], self._shift[i, j] = (
                eps, sig * sig, rc2, shift)
        self._codes_cache: dict = {}

    def __getstate__(self) -> dict:
        """
        Picklable state, without the species-code cache.

        Returns
        -------
        dict
            The instance dict with the cache emptied (it holds references to
            arrays and is rebuilt on demand).
        """
        state = self.__dict__.copy()
        state['_codes_cache'] = {}  # holds array references; rebuilt on demand
        return state

    def _codes(self, species: np.ndarray) -> np.ndarray:
        """
        Integer codes of a species array, cached by array identity.

        Species arrays are replaced (never mutated in place) when particles
        are inserted or deleted, so identity is a safe cache key.

        Parameters
        ----------
        species : numpy.ndarray
            Species labels.

        Returns
        -------
        numpy.ndarray
            One integer code per entry.
        """
        hit = self._codes_cache.get(id(species))
        if hit is not None and hit[0] is species:
            return hit[1]
        other = len(self._code)
        codes = np.array([self._code.get(str(x), other) for x in species], dtype=np.intp)
        if len(self._codes_cache) > 16:
            self._codes_cache.clear()
        self._codes_cache[id(species)] = (species, codes)
        return codes

    def pair_energy_of(self, system: "ParticleSystem", pos: np.ndarray,
                       exclude: Optional[int] = None) -> float:
        """
        Energy of a (trial) mobile particle at `pos` with everything else.

        Parameters
        ----------
        system : ParticleSystem
            The system.
        pos : numpy.ndarray
            The (3,) position.
        exclude : int, optional
            Index of the mobile particle `pos` belongs to (its species is
            used, and it is left out of the sum). None means a new particle
            of `system.exchange_species`.

        Returns
        -------
        float
            Its interaction energy with frozen, substrate and other mobile
            atoms.
        """
        _check_min_image(system, self.max_r_cut)
        sp = system.exchange_species if exclude is None else system.species[exclude]
        c = self._code.get(str(sp), len(self._code))
        mob, mob_c = system.positions, self._codes(system.species)
        if exclude is not None:
            mob = np.delete(mob, exclude, axis=0)
            mob_c = np.delete(mob_c, exclude)
        others = np.vstack([system.frozen_positions, system.substrate_positions, mob])
        if others.shape[0] == 0:
            return 0.0
        oc = np.concatenate([self._codes(system.frozen_species),
                             self._codes(system.substrate_species), mob_c])
        dr = system._min_image(others - pos)
        r2 = np.einsum('ij,ij->i', dr, dr)
        sel = r2 < self._rc2[c, oc]
        if not np.any(sel):
            return 0.0
        ocs = oc[sel]
        sr6 = (self._sig2[c, ocs] / r2[sel]) ** 3
        return float(np.sum(4.0 * self._eps[c, ocs] * (sr6 * sr6 - sr6) - self._shift[c, ocs]))

    def _pair_matrices(self, system: "ParticleSystem"):
        """
        Pairwise displacements and per-pair parameters of the full system.

        Parameters
        ----------
        system : ParticleSystem
            The system (all atom categories).

        Returns
        -------
        tuple
            ``(dr, r2, eps, sig2, inside, shift)`` over all atom pairs of
            `system.full_configuration()`, with `inside` False on the
            diagonal, beyond the cutoff, and for undefined species pairs.
        """
        _check_min_image(system, self.max_r_cut)
        pos, sp = system.full_configuration()
        sp = sp.astype(str)
        n = pos.shape[0]
        dr = system._min_image(pos[:, None, :] - pos[None, :, :])
        r2 = np.einsum('ijk,ijk->ij', dr, dr)
        eps = np.zeros((n, n)); sig2 = np.ones((n, n)); rc2 = np.zeros((n, n))
        shift = np.zeros((n, n))
        species = np.unique(sp)
        for a in species:
            for b in species:
                par = self.params.get((a, b))
                if par is None:
                    continue
                m = np.outer(sp == a, sp == b)
                eps[m], sig2[m], rc2[m], shift[m] = par[0], par[1] ** 2, par[2], par[3]
        inside = r2 < rc2
        np.fill_diagonal(inside, False)
        return dr, r2, eps, sig2, inside, shift

    def total_energy(self, system: "ParticleSystem") -> float:
        """
        Total pair energy of every interacting pair in the system.

        Parameters
        ----------
        system : ParticleSystem
            The system (frozen + substrate + mobile atoms).

        Returns
        -------
        float
            Sum over pairs i < j (including frozen-frozen pairs if their
            species pair is defined; constant, since they never move).
        """
        if system.full_configuration()[0].shape[0] < 2:
            return 0.0
        _, r2, eps, sig2, inside, shift = self._pair_matrices(system)
        sr6 = np.where(inside, (sig2 / np.where(inside, r2, 1.0)) ** 3, 0.0)
        u = np.where(inside, 4.0 * eps * (sr6 * sr6 - sr6) - shift, 0.0)
        return float(np.sum(u) / 2.0)

    def forces(self, system: "ParticleSystem") -> np.ndarray:
        """
        Forces on the substrate and mobile atoms.

        Parameters
        ----------
        system : ParticleSystem
            The system.

        Returns
        -------
        numpy.ndarray
            Shape (n_substrate + n_mobile, 3), substrate rows first -- the
            layout `Replica._move_hmc` expects. Frozen atoms act as sources
            only.
        """
        n_fr = system.frozen_positions.shape[0]
        n_mov = system.n_substrate + system.n
        if n_mov == 0:
            return np.zeros((0, 3))
        if n_fr + n_mov < 2:
            return np.zeros((n_mov, 3))
        dr, r2, eps, sig2, inside, _ = self._pair_matrices(system)
        r2s = np.where(inside, r2, 1.0)
        s = np.where(inside, sig2 / r2s, 0.0)
        coeff = np.where(inside, (24.0 * eps / r2s) * (2.0 * s ** 6 - s ** 3), 0.0)
        return np.einsum('ij,ijk->ik', coeff, dr)[n_fr:]


class ASECalculatorAdapter(Calculator):
    """Wraps any ASE-compatible Calculator -- EAM, an MLIP (MACE, NequIP,
    Allegro, CHGNet, ...), ASE's own EMT/LennardJones, even DFT if you're
    patient -- so it can drive this GCMC engine. `supports_incremental` stays
    False: for a general potential there is no safe shortcut, so every trial
    move goes through `ParticleSystem.generic_*_dU` (clone, mutate, evaluate
    total energy twice). This is correct for ANY potential, many-body ones
    included; it is simply not the fastest possible implementation.

    Why many-body potentials need this instead of the fast path: the fast
    LJ path works because a *pairwise* potential's single-particle energy is
    a cheap, clean sum over neighbors. EAM is explicitly NOT pairwise:
    E = sum_i F(rho_i) + (1/2) sum_i,j!=i phi(r_ij), where rho_i is a local
    electron density contributed to by every neighbor -- moving one atom
    changes the embedding energy of all its neighbors too, not just its own
    energy. Most MLIPs are even less local: a single forward pass gives the
    whole configuration's energy, with no cheap way to extract "just the
    change from one atom." Cloning the configuration and diffing two full
    evaluations is the one formulation that is always correct regardless of
    the potential's structure.

    Example -- a tabulated EAM potential (LAMMPS/DYNAMO format):
        from ase.calculators.eam import EAM
        calc = ASECalculatorAdapter(EAM(potential='Fe_H.eam.fs'),
                                     cell=[Lx, Ly, Lz], pbc=(True, True, False))

    Example -- a trained MLIP via its ASE wrapper:
        from mace.calculators import MACECalculator
        calc = ASECalculatorAdapter(
            MACECalculator(model_paths='fe_h_model.model', device='cuda'),
            cell=[Lx, Ly, Lz], pbc=(True, True, False))

    Both plug into `ParticleSystem`/`SlabSystem` exactly like `LJCalculator`
    -- nothing else in this package needs to know which one is in use.

    Parallelism (`n_workers > 1` in `ReplicaExchangeGCMC`) pickles whole
    `Replica` objects to send them to worker processes, which means pickling
    whatever calculator each one holds. Many ASE calculators become
    unpicklable once they've actually been used: EMT, for instance, builds a
    `collections.defaultdict` with a lambda default factory the first time
    `initialize()` runs (i.e. the first energy evaluation), and lambdas
    can't be pickled by the standard library's pickler -- so a live,
    already-used EMT instance fails with a cryptic
    `AttributeError: Can't pickle local object 'EMT.initialize.<locals>.<lambda>'`
    deep inside `concurrent.futures`, not any obvious line of your own code.

    To make this safe, pass a zero-argument FACTORY instead of a live
    instance whenever you want `n_workers > 1`: the class itself, if it
    needs no constructor arguments (`ASECalculatorAdapter(EMT, cell=...,
    pbc=...)`), or `functools.partial` for one that does
    (`ASECalculatorAdapter(functools.partial(EAM, potential='Fe_H.eam.fs'),
    cell=..., pbc=...)`). The adapter then constructs its own fresh
    calculator instance lazily, the first time it's actually needed --
    separately in each worker process, so nothing initialized (and
    therefore potentially unpicklable) ever needs to cross a process
    boundary. Passing a live instance still works exactly as before for
    `n_workers=1`; trying to pickle one (i.e. actually using it with
    `n_workers > 1`) raises a clear `TypeError` pointing you at this fix,
    rather than the cryptic pickling traceback.
    """

    supports_incremental = False
    supports_forces = True

    def __init__(self, ase_calculator_or_factory, cell: Sequence[float],
                 pbc: Tuple[bool, bool, bool] = (True, True, False)):
        """
        Wrap an ASE calculator (or a factory for one) as a `Calculator`.

        Parameters
        ----------
        ase_calculator_or_factory : ase.calculators.calculator.Calculator or callable
            Either a live, already-constructed ASE calculator instance
            (fine for single-process use, i.e. `n_workers=1`), or a
            zero-argument factory -- a class needing no constructor
            arguments, or a `functools.partial` for one that does -- which
            is lazily called to build the calculator the first time it's
            actually needed. A factory is required for `n_workers > 1`;
            see this class's docstring for why.
        cell : sequence of float
            The (Lx, Ly, Lz) orthorhombic cell dimensions to build the ASE
            `Atoms` object with on every evaluation.
        pbc : tuple of bool, optional
            Periodicity along each of the three cell axes, passed to the
            ASE `Atoms` constructor. Default is ``(True, True, False)``
            (periodic in x, y; open in z, matching `SlabSystem`'s
            geometry).
        """
        from ase.calculators.calculator import Calculator as ASEBaseCalculator

        if isinstance(ase_calculator_or_factory, ASEBaseCalculator):
            self._calc_factory = None
            self._calc_instance = ase_calculator_or_factory
        else:
            self._calc_factory = ase_calculator_or_factory  # a class, function, or partial
            self._calc_instance = None
        self.cell = np.array(cell, dtype=float)
        self.pbc = pbc

    @property
    def ase_calculator(self):
        """
        The underlying, lazily-constructed ASE calculator instance.

        Returns
        -------
        ase.calculators.calculator.Calculator
            The live ASE calculator, built from the factory on first
            access if that's how this adapter was constructed (and cached
            for subsequent accesses); otherwise the instance passed at
            construction, directly.
        """
        if self._calc_instance is None:
            self._calc_instance = self._calc_factory()
        return self._calc_instance

    def __getstate__(self) -> dict:
        """
        Return the picklable state of this adapter, for multiprocessing.

        Returns
        -------
        dict
            The adapter's state with `_calc_instance` cleared to None, so
            a live (and possibly already-used, and therefore possibly
            unpicklable) ASE calculator instance is never included in the
            pickled data; only the factory (if one was given) is kept, so
            each worker process can lazily rebuild its own fresh instance.

        Raises
        ------
        TypeError
            If this adapter was built from a live calculator instance
            rather than a factory, since there is then no safe way to
            reconstruct the calculator in another process. See this
            class's docstring for the fix.
        """
        if self._calc_factory is None:
            raise TypeError(
                "This ASECalculatorAdapter holds a live calculator instance, not a "
                "factory, so it can't be safely pickled for multiprocessing (many ASE "
                "calculators -- EMT included -- become unpicklable once actually used). "
                "Build it with a zero-argument factory instead: the class itself if it "
                "needs no constructor arguments (ASECalculatorAdapter(EMT, cell=..., "
                "pbc=...)), or functools.partial for one that does "
                "(ASECalculatorAdapter(functools.partial(EAM, potential='Fe_H.eam.fs'), "
                "cell=..., pbc=...)). See this class's docstring for more."
            )
        return {"_calc_factory": self._calc_factory, "_calc_instance": None,
                "cell": self.cell, "pbc": self.pbc}

    def __setstate__(self, state: dict) -> None:
        """
        Restore this adapter's state after unpickling.

        Parameters
        ----------
        state : dict
            The state dict produced by `__getstate__` (with
            ``_calc_instance`` set to None), to be applied via
            ``self.__dict__.update(state)``.

        Returns
        -------
        None
        """
        self.__dict__.update(state)

    def total_energy(self, system: "ParticleSystem") -> float:
        """
        Compute the total energy of the full configuration via ASE.

        Parameters
        ----------
        system : ParticleSystem
            The system whose full configuration (frozen + substrate +
            mobile particles, via `system.full_configuration()`) should
            be evaluated.

        Returns
        -------
        float
            The potential energy reported by the wrapped ASE calculator
            for the full configuration, or 0.0 if the configuration is
            empty.
        """
        from ase import Atoms  # local import: no hard ASE dependency otherwise
        positions, species = system.full_configuration()
        if positions.shape[0] == 0:
            return 0.0
        atoms = Atoms(symbols=list(species), positions=positions,
                       cell=self.cell, pbc=self.pbc)
        atoms.calc = self.ase_calculator
        return float(atoms.get_potential_energy())

    def forces(self, system: "ParticleSystem") -> np.ndarray:
        """
        Compute forces on the substrate + mobile particles via ASE.

        Both substrate and mobile particles can move; only frozen atoms
        are excluded. `full_configuration()` always puts frozen atoms
        first (see `ParticleSystem`), so dropping the first
        `frozen_positions.shape[0]` rows leaves [substrate; mobile], in
        that order -- what `Replica`'s HMC move expects.

        Parameters
        ----------
        system : ParticleSystem
            The system whose full configuration should be evaluated.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_substrate + n_mobile, 3): the force on each
            substrate atom (first) and mobile particle (last), as reported
            by the wrapped ASE calculator's `get_forces()`.
        """
        from ase import Atoms
        positions, species = system.full_configuration()
        n_frozen = system.frozen_positions.shape[0]
        n_movable = positions.shape[0] - n_frozen
        if n_movable == 0:
            return np.zeros((0, 3))
        atoms = Atoms(symbols=list(species), positions=positions,
                       cell=self.cell, pbc=self.pbc)
        atoms.calc = self.ase_calculator
        all_forces = atoms.get_forces()
        return all_forces[n_frozen:]
