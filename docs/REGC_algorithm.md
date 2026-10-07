# REGC algorithm as implemented in `gcmc_re`

This document states the replica-exchange grand-canonical (REGC) Monte Carlo algorithm
that `gcmc_re` implements, with every acceptance rule written out, and relates it to

> Y. Zhou, M. Scheffler, L. M. Ghiringhelli, *Determining surface phase diagrams including
> anharmonic effects*, Phys. Rev. B **100**, 174106 (2019) — referred to below as **[ZSG]**.

Equations here are numbered (1), (2), …; equations of the paper are cited as [ZSG Eq. n].

---

## 0. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $T,\ \beta = 1/k_BT$ | temperature (energy units, $k_B=1$) | `ThermoState.temperature`, `.beta` |
| $\mu$ | chemical potential of the exchanged species | `ThermoState.mu` |
| $\Lambda(T)$ | thermal de Broglie wavelength, $h/\sqrt{2\pi m k_B T}$ | `ThermoState.lambda_cubed` $=\Lambda^3$ |
| $E(\mathbf R)$ | potential energy of configuration $\mathbf R$ (calculator + external field) | `Replica.energy`, `ParticleSystem.total_energy()` |
| $N$ | number of mobile (exchangeable) particles | `ParticleSystem.n` |
| $V_\text{box}$ | region where mobile particles may live | slab: $L_xL_y\,(0, L_z]$ |
| $V_\text{ex}$ | **exchange zone**: region where insertions are placed and deletions are drawn from | `ParticleSystem.volume` |
| $N_\text{ex}$ | number of mobile particles currently inside $V_\text{ex}$ | `len(exchange_zone_indices())` |
| $(l,m)$ | grid index: $T_l$, $\mu_m$ | `sim.grid[(it, imu)]` |

**One $\Lambda^3(T)$ convention, used everywhere.** $\Lambda^3$ enters three places: the
insertion/deletion rules (§2), the replica-exchange rule (§3) and the MBAR reduced potential
(§5). All three read the same function, `lambda_cubed_of_T`, passed to `ReplicaExchangeGCMC`
(and picked up from `sim` by `MBARSurfaceAnalysis`):

* **Reduced LJ units** (default): $\Lambda^3 \equiv 1$ at every $T$.
* **Real units**: $\Lambda^3(T) = \left(h^2/2\pi m k_B T\right)^{3/2}$
  (`units.thermal_wavelength_cubed_angstrom3`), which must be valid at *any* $T$, not only at
  grid temperatures, because MBAR evaluates it off-grid.

---

## 1. Target ensemble

Each grid point $(l,m)$ samples the grand-canonical density [ZSG Eq. 1]

$$
\mathcal N_{\mu,V,T}(\mathbf R; N) \;\propto\;
\frac{e^{\beta\mu N}\, V_\text{box}^{N}}{\Lambda^{3N}(T)\, N!}\;
e^{-\beta E(\mathbf R;N)}
\qquad (1)
$$

with positions in $V_\text{box}$ (normalized coordinates). The extended ensemble is the product
over all $L\times M$ grid points [ZSG Eq. 4]:

$$
Q_\text{ext} = \prod_{l=1}^{L}\prod_{m=1}^{M} \;\sum_{N}
\frac{e^{\beta_l \mu_m N}}{\Lambda^{3N}(T_l)\,N!}\int_{V_\text{box}^N} d\mathbf R\;
e^{-\beta_l E(\mathbf R;N)} .
\qquad (2)
$$

---

## 2. Local moves (one grid point, fixed $T,\mu$)

A *sweep* (`Replica.sweep`) performs `sweeps_per_cycle` moves. Each move is drawn
independently with relative probabilities `p_displace : p_insert : p_delete :
p_displace_substrate` (renormalized). `p_insert = p_delete` is needed for detailed balance and
is the default.

### 2.1 Displacement (`_move_displace`) [ZSG Eq. 7]

Pick a mobile particle $i$ uniformly and propose $\mathbf r_i' = \mathbf r_i + \boldsymbol\delta$,
with $\boldsymbol\delta$ uniform in $[-\delta_\text{max},\delta_\text{max}]^3$:

$$
P_\text{acc}(\mathbf r_i\to\mathbf r_i') = \min\!\left[1,\; e^{-\beta\,\Delta E}\right],
\qquad \Delta E = E(\mathbf r_i',\mathbf r^{N-1}) - E(\mathbf r_i,\mathbf r^{N-1}).
\qquad (3)
$$

A move outside $(0,L_z]$ has $E=+\infty$ and is always rejected. Particles may move freely in
and out of $V_\text{ex}$.

`_move_displace_substrate` applies the same rule (3) to a displaceable substrate atom.

### 2.2 Diffusion step of the REGC loop (`Replica.diffuse`) [ZSG §II A 3]

In the [ZSG] loop (§4), each step ends with canonical (fixed $N,V,T$) sampling, run in parallel
over replicas. Three modes are available:

* **`'mc_moves'`**: $n$ single displacement attempts of rule (3), each of a uniformly chosen
  movable atom. This is [ZSG]'s LJ protocol ("ten sequential MC moves").
* **`'mc'`**: $n$ MC cycles. One cycle is $N_\text{mov}$ attempts of rule (3), where
  $N_\text{mov}$ is the number of mobile atoms (plus substrate atoms if
  `displace_substrate=True`), so on average each atom is attempted once per cycle.
* **`'hmc'`**: $n$ Hybrid MC trajectories. Draw momenta
  $\mathbf p \sim \text{Maxwell–Boltzmann}(T)$, integrate $n_\text{LF}$ leapfrog steps of size
  $\Delta t$ for all mobile and substrate atoms together, and accept with
  $\min[1, e^{-\beta\Delta H}]$, $H = E + \sum_j p_j^2/2m_j$. The result is exact for any
  $\Delta t$. This mode is an extension beyond [ZSG].
* **`'md'`**: $n$ steps of NVT molecular dynamics with ASE (`Replica._diffuse_md`), as in the
  paper's ab initio REGCMD:
  * The thermostat is Bussi stochastic velocity rescaling (default; [ZSG] use $\tau = 20$ fs and
    a 1 fs time step) or Langevin.
  * Velocities are redrawn from Maxwell–Boltzmann at $T_l$ at the start of each segment. This is
    exact because $\mathbf p$ and $\mathbf R$ are independent in the canonical ensemble, and it
    handles freshly inserted atoms and configurations just swapped in from another $T$.
  * Frozen atoms are held by `FixAtoms`.
  * The hard walls of $V_\text{box}$ are enforced by specular reflection after every step
    ($z\to 2z_\text{wall}-z$, $v_z\to -v_z$), the analogue of the reflecting boundaries that
    [ZSG] impose with PLUMED.
  * After the segment, $E$ is recomputed with the system's calculator.
  * With `LammpsCalculator` (`gcmc_re/lammps_backend.py`) the whole segment runs natively in
    LAMMPS: `velocity create`, `fix nve` + `fix temp/csvr` (Bussi), `fix wall/reflect`, with
    frozen atoms excluded. Any LAMMPS pair style works this way (EAM, GRACE-FS `grace/fs`,
    ACE, ...). It is about $10^3\times$ faster than ASE for an EAM slab.
  * MD has no accept/reject step, so it carries the integrator's time-step bias. The
    requirements are an `ASECalculatorAdapter` and $T$ given as $k_BT$ in eV.

The mixed sweep scheme `run` (§4.2) does not use `diffuse`. It interleaves the single moves of
§2.1 (or HMC, with `hmc_enabled=True`) with GC moves.

### 2.3 Insertion (`_move_insert`) [ZSG Eq. 3]

Draw $\mathbf r_\text{new}$ uniformly in $V_\text{ex}$ (`random_insertion_position`) and accept with

$$
P_\text{acc}(N\to N+1) = \min\!\left[1,\;
\frac{V_\text{ex}}{\Lambda^3(T)\,(N_\text{ex}+1)}\;
e^{\beta\left[\mu - (E_{N+1}-E_N)\right]}\right].
\qquad (4)
$$

### 2.4 Deletion (`_move_delete`) [ZSG Eq. 2]

Pick particle $i$ uniformly **among the $N_\text{ex}$ particles inside $V_\text{ex}$** (no-op if
$N_\text{ex}=0$) and accept with

$$
P_\text{acc}(N\to N-1) = \min\!\left[1,\;
\frac{\Lambda^3(T)\,N_\text{ex}}{V_\text{ex}}\;
e^{-\beta\left[\mu + (E_{N-1}-E_N)\right]}\right].
\qquad (5)
$$

### 2.5 Why $V_\text{ex}$ and $N_\text{ex}$ (fix 3)

Treat the particles as an unlabelled set. Then the density (1) is
$\pi(\mathbf R) \propto e^{\beta\mu N}\Lambda^{-3N}e^{-\beta E}$ per unit volume element
(the $N!$ cancels against the labelling). Take a configuration with $N_\text{ex}$ particles
inside $V_\text{ex}$:

* the forward move (insertion at $\mathbf r_\text{new}\in V_\text{ex}$) has proposal density $1/V_\text{ex}$;
* the reverse move (deleting exactly that particle) has proposal probability
  $1/(N_\text{ex}+1)$, because the candidates are the zone particles *after* insertion.

Detailed balance $\pi_\text{old}\,q_\text{fwd}\,P_\text{acc}^\text{fwd} = \pi_\text{new}\,q_\text{rev}\,P_\text{acc}^\text{rev}$ then gives

$$
\frac{P_\text{acc}^\text{fwd}}{P_\text{acc}^\text{rev}}
= \frac{\pi_\text{new}}{\pi_\text{old}}\,\frac{q_\text{rev}}{q_\text{fwd}}
= \frac{e^{\beta\mu}}{\Lambda^3}\,e^{-\beta\Delta E}\;\frac{V_\text{ex}}{N_\text{ex}+1},
\qquad (6)
$$

which is exactly (4)/(5). Two consequences:

* The stationary distribution is the full grand-canonical ensemble (1) over **all of
  $V_\text{box}$**, not only $V_\text{ex}$. For example, an ideal gas has
  $\langle N\rangle = e^{\beta\mu}V_\text{box}/\Lambda^3$. The recorded $N$ is the total $N$,
  and that is the $N$ to use in §3 and §5.
* Using $V_\text{ex}$ with the *total* $N$, or deleting particles that sit outside $V_\text{ex}$,
  breaks (6). The previous code did both. Insertion into a sub-region also matches [ZSG],
  which restricts "insertion (and removal)" to a 48 Å surface zone.

Test: `tests/test_paper_consistency.py::test_deletion_restricted_to_insertion_zone_ideal_gas`.

---

## 3. Replica exchange (`ReplicaExchangeGCMC._attempt_exchange`)

Replica $a$ sits at $(\beta_a,\mu_a)$ with configuration $\mathbf R_i$ ($N_a$, $E_a$), and
replica $b$ at $(\beta_b,\mu_b)$ with $\mathbf R_j$ ($N_b$, $E_b$). The swap exchanges the
*configurations*; the thermodynamic states stay on their grid slots. Detailed balance in the
extended ensemble [ZSG Eq. 5]:

$$
\mathcal N(\beta_a,\mu_a,\mathbf R_i)\,\mathcal N(\beta_b,\mu_b,\mathbf R_j)\,P_{\to}
= \mathcal N(\beta_a,\mu_a,\mathbf R_j)\,\mathcal N(\beta_b,\mu_b,\mathbf R_i)\,P_{\leftarrow}.
\qquad (7)
$$

Insert (1). The factors $V^N$ and $N!$ appear once on each side and cancel. What remains is

$$
P_\text{acc}^\text{swap} = \min\!\left[1,\;
\left(\frac{\Lambda^3(T_a)}{\Lambda^3(T_b)}\right)^{N_a-N_b}
\exp\!\Big\{(\beta_a-\beta_b)(E_a-E_b) \;-\; (\beta_a\mu_a-\beta_b\mu_b)(N_a-N_b)\Big\}\right].
\qquad (8)
$$

### 3.1 Relation to [ZSG Eq. 6]

* **Λ prefactor (fix 1).** For a physical $\Lambda\propto\beta^{1/2}$,
  $(\Lambda_a^3/\Lambda_b^3)^{N_a-N_b} = (\beta_a/\beta_b)^{\frac32(N_a-N_b)}$, which is the
  paper's factor. The code computes the prefactor from `state.lambda_cubed`, the same value used
  in (4)/(5), rather than hard-coding $(\beta_a/\beta_b)^{3/2}$. In reduced units
  ($\Lambda^3\equiv1$) the prefactor is 1. The previous code always applied the $\beta$ factor,
  which was inconsistent with the $\Lambda^3=1$ insert/delete moves.
* **Sign of the $\mu$ term.** As printed, [ZSG Eq. 6] has
  $+(\beta_l\mu_m-\beta_k\mu_o)(N_{l,m}-N_{k,o})$. Working out (7) from (1) gives the minus
  sign in (8). Sanity check: at equal $\beta$ with $\mu_a>\mu_b$ and $N_a>N_b$, moving the
  larger configuration to the lower $\mu$ must be *penalized*, and only the minus sign does
  that. The code uses (8).

Tests: `test_exchange_acceptance_uses_same_lambda_as_gcmc` (rule (8) exactly, both Λ
conventions) and `test_replica_exchange_ideal_gas_density` (end-to-end ideal-gas check).

### 3.2 Swap patterns ([ZSG] Fig. 2, `ReplicaExchangeGCMC.swap_pairs`)

Each pattern is a set of **disjoint** pairs, so all of its swaps can be attempted at once. Each
attempt uses (8). With offset $o\in\{0,1\}$:

| Type | Pairs | Fig. 2 |
|---|---|---|
| `'T'` | $(l,m)\leftrightarrow(l+1,m)$, $l = o, o+2, \dots$; all $m$ | (a) |
| `'mu'` | $(l,m)\leftrightarrow(l,m+1)$, $m = o, o+2, \dots$; all $l$ | (b) |
| `'cross_left'` | $(l,m+1)\leftrightarrow(l+1,m)$, $l = o, o+2, \dots$; all $m$ | (c) |
| `'cross_right'` | $(l,m)\leftrightarrow(l+1,m+1)$, $l = o, o+2, \dots$; all $m$ | (d) |

Over all types and both offsets, every replica is connected to all of its 3–8 neighbours
([ZSG]: "between three and eight neighbors"). Fig. 2 draws only offset 0 (pairs
$T_1T_2$, $T_3T_4$). With that offset alone the grid would split into disconnected $2\times M$
blocks, since $T_2$ would never swap with $T_3$. The code therefore draws the offset at random
as well.

**Detailed balance.** The choice of pattern and offset does not depend on the current
configurations. The pairs in a pattern are disjoint, and each pair's swap satisfies (7). So
every RE move satisfies detailed balance in the extended ensemble (2).

---

## 4. Simulation loop

### 4.1 [ZSG] Fig. 1 scheme: `ReplicaExchangeGCMC.run_regc`

```
repeat for step = 1, 2, … , n_steps:
    draw rand ~ U[0,1)                         (one draw, shared by all replicas)
    if rand < x0:                              particle insertion/removal
        every replica: one GC attempt          removal if y < 1/2, else insertion (Eqs. 4-5)
    else:                                      replica exchange
        draw type ∈ {T, mu, cross_left, cross_right} and offset ∈ {0,1}, uniformly
        attempt every pair of that pattern     (Eq. 8)
    every replica, in parallel: diffusion at fixed N,V,T_l  (MC / HMC / ASE-MD, §2.2)
    if step > burn_in and step % record_every == 0:
        record (E, N) for every replica        (after diffusion, before next GC/RE)
    if step % convergence_every == 0 and ρ(u) unchanged (Eq. 15): stop
```

**Execution in blocks.** Between two RE moves the replicas are independent. `run_regc`
therefore draws the GC/RE branches in order and hands each replica a whole block of steps in
one call (`Replica.regc_local_block`), with samples recorded inside the block. A block ends
before the next RE move or after a step that needs the main process (trajectory frame,
progress line, convergence check). The random draws and their order are exactly those of a
step-by-step loop, so results are bit-identical (tested). With $x_0 = 0.99$ a block is about
100 steps long, which makes process-pool parallelism worthwhile.

**Which branch $x_0$ selects.** The paper is internally inconsistent here:

* Fig. 1 labels `rand ≤ x0` → *replica exchange* and *no* → *particle insertion/removal*.
* The text of §II A says "the system has a probability $x_0$ to attempt exchanging a particle
  with the reservoir and probability $(1-x_0)$ to perform a RE move".
* The parameters agree with the **text**. For the LJ surface, $x_0 = 0.99$ together with
  "configuration swaps were attempted every 100 REGC steps" ($1-x_0 = 0.01$). For the Si
  clusters $x_0 = 0.9$.

`run_regc` follows the text: **$x_0 = P(\text{GC attempt})$**, default 0.99. To use Fig. 1's
convention, pass `x0 = 1 - x0_fig`.

**Convergence** ([ZSG] §III A: "no change in the density of reduced-energy states $\rho(U)$
with increasing simulation steps"). Every `convergence_every` steps, for each grid point $k$, the
reduced energies $u$ of Eq. (9) of all samples recorded so far are histogrammed on common bins.
The same is done for the samples that existed at the previous check. The run stops when

$$
\max_k\; \tfrac12 \sum_{\text{bins } b} \left|\rho_k^\text{now}(b) - \rho_k^\text{prev}(b)\right|
\;<\; \texttt{convergence\_tol},
\qquad (15)
$$

that is, when the total-variation change of every $\rho_k(u)$ is small. `n_steps` is a cap.
`run_regc` returns whether the criterion was met, and `convergence_history` stores the values.
`swap_acceptance()` reports the acceptance per pattern. [ZSG] tune the $(T,\mu)$ spacing to
reach about 25%.

### 4.2 Mixed sweep scheme: `ReplicaExchangeGCMC.run`

```
for cycle = 1 … n_cycles:
    every replica, in parallel: sweeps_per_cycle moves drawn from
        {displace|HMC, insert, delete, displace_substrate}
    if cycle % exchange_every == 0:
        for type in (T, mu, cross_left, cross_right):   # diagonals: diagonal_swaps=True
            for offset in (0, 1): attempt all pairs of that pattern
    record (E, N) every record_every cycles after burn-in
```

This scheme mixes GC moves into the sweep and visits the swap patterns in a fixed order.
Each move satisfies detailed balance, and a fixed-order composition of such moves still leaves
(2) invariant (global balance). Both loops therefore sample the same ensemble. §4.1 reproduces
the paper's protocol; §4.2 is usually more efficient for cheap force fields.

The YAML runner selects between them with `replica_exchange.scheme: sweep | regc`. The
`regc` scheme reads `n_steps`, `x0`, `diffusion`, `diffusion_steps`, `record_every`,
`burn_in_steps`, `convergence_every`, `convergence_tol`, and an optional `md:` block
(`timestep_fs`, `thermostat`, `taut_fs`, `friction_per_fs`).

---

## 5. MBAR reweighting (`analysis.MBARSurfaceAnalysis`) [ZSG §II B]

### 5.1 Reduced potential (fix 2)

MBAR needs, for every sample $n$ and every state $k=(T,\mu)$, the reduced potential
$u_k(n) = -\ln q_k(n)$ up to a sample-independent constant that may depend on $k$. From (1), the
factors $V^N/N!$ are the same at every state, so they cancel between states, but
$\Lambda^{-3N}(T)$ does not. Hence

$$
u(\mathbf R; \mu, \beta) = \beta\left[E(\mathbf R) - \mu N(\mathbf R)\right]
\;+\; N(\mathbf R)\,\ln\Lambda^3(T).
\qquad (9)
$$

[ZSG Eq. 9] omits the last term. This is consistent only if $\Lambda$ is absorbed into $\mu$
or is $T$-independent. Otherwise it contradicts the $\Lambda$ factor in their own Eq. 6. The
code now uses (9) both to build `u_kn` and for off-grid states (`_u_of_state`), evaluating
$\Lambda^3$ through `lambda_cubed_of_T` at arbitrary $T$. In reduced units the extra term
vanishes.

Test: `test_mbar_includes_lambda_cubed_term` (exact Poisson ideal-gas samples with a
$T$-dependent $\Lambda$, reweighted to an unsampled $(T,\mu)$).

### 5.2 Estimators

With $q(\mathbf R;\mu,\beta)=e^{-u}$, MBAR solves self-consistently for the normalizations
$\hat c_{l,m}$ [ZSG Eq. 10]:

$$
\hat c_{l,m} = \sum_{n=1}^{N_\text{tot}}
\frac{q(\mathbf R_n;\mu_m,\beta_l)}{\sum_{l',m'} \Lambda_{l',m'}\,\hat c_{l',m'}^{-1}\,q(\mathbf R_n;\mu_{m'},\beta_{l'})},
\qquad (10)
$$

where $\Lambda_{l',m'}$ is the **number of samples** recorded at state $(l',m')$ (the paper's
notation; not the wavelength). An observable $A$ at any $(\mu,\beta)$ is then

$$
\langle A\rangle_{\mu,\beta} =
\frac{\sum_n A(\mathbf R_n)\, q(\mathbf R_n;\mu,\beta)\,/\,D_n}{\sum_n q(\mathbf R_n;\mu,\beta)\,/\,D_n},
\qquad D_n = \sum_{l',m'} \Lambda_{l',m'}\,\hat c_{l',m'}^{-1}\,q(\mathbf R_n;\mu_{m'},\beta_{l'}).
\qquad (11)
$$

`mean_N` uses $A=N$. For phases defined by an indicator $\chi_i$ (e.g. $N=N_i$), the
probability is $p_i=\langle\chi_i\rangle$ and the phase free energy is [ZSG Eq. 8]

$$
f_i(\mu,T) = -k_BT\,\ln p_i(\mu,T),
\qquad (12)
$$

with the stable phase at $(\mu,T)$ being $\arg\min_i f_i = \arg\max_i p_i$.

---

## 6. From $\mu$ to pressure

For a monatomic ideal-gas reservoir, [ZSG §III A] gives

$$
\beta\mu = \ln\!\left(\beta p\,\Lambda^3\right)
\quad\Longleftrightarrow\quad
\mu = k_BT\ln\frac{p}{p_0(T)},\qquad p_0(T)=\frac{k_BT}{\Lambda^3(T)}.
\qquad (13)
$$

This uses the **same** $\Lambda^3(T)$ as §2–§5. `pressure.p0_ideal_gas_wavelength` implements
it. `pressure.p0_reduced_units` ($p_0=T$) is the $\Lambda^3\equiv1$ case.

For an atom exchanged with a diatomic reservoir (e.g. H from H₂), `reservoir.py` uses the
standard atomistic-thermodynamics form (cf. [ZSG Eq. 15]):

$$
\mu_\text{H}(T,p) = \tfrac12\left[\mu_{\text{H}_2}(T,p) - D_0\right] + e_\text{atom},\qquad
\mu_{\text{H}_2} = k_BT\ln\frac{p\,\Lambda_{\text{H}_2}^3}{k_BT} - k_BT\ln\!\left(q_\text{rot}\,q_\text{vib}\right),
\qquad (14)
$$

with $D_0$ and $e_\text{atom}$ computed with the same calculator that evaluates the
adsorbed atoms. $e_\text{atom}$ is the calculator's energy of one isolated atom. All GCMC energy
differences are measured on the calculator's scale, so an atom far from the surface costs
$E_{N+1}-E_N = e_\text{atom}$ in (4). Equivalently, $\mu_\text{H}$ is half the H₂ chemical
potential measured from the calculator's own $E_{\text{H}_2}^\text{min} + \text{ZPE}$. This is
the role $E_\text{DFT}$ plays in [ZSG Eq. 15]. For EMT's H, $e_\text{atom} = 3.21$ eV, while
the native `LJCalculator` has $e_\text{atom}=0$. Omitting it shifts every $\mu_\text{H}$ by
$-e_\text{atom}$. In (4), (5), (8) and (9), the
$\Lambda$ is that of the **exchanged species** (the H atom), since that is the particle whose
number fluctuates.

---

## 7. The LJ surface of [ZSG] §III A (`examples/12_paper_lj_surface.py`)

| Item | [ZSG] | Here |
|---|---|---|
| Surface | frozen A$_{18}$, 2-layer fcc(111), hexagonal $3\times3$ cell, lattice vectors 11.489 Å long | `SlabSystem(cell_xy=...)`, exact minimum image for non-orthogonal cells |
| Interactions | LJ 12-6: $\epsilon_{AB}=0.66$ eV, $\sigma_{AB}=1.91$ Å; $\epsilon_{BB}=0.01$ eV, $\sigma_{BB}=1.2$ Å; no A–A | `PairLJCalculator` (sees frozen atoms); cutoff $2.5\sigma$, shifted (not stated in [ZSG]) |
| System volume | the 48 Å "surface zone" above the topmost A atoms, all $x,y$ | box $=$ zone: top A layer at $z=0$ (hard wall), $L_z = 48$ Å; B confined to it |
| Grid | $T = 200\dots650$ K (step 50), $\mu = -2.4\dots-0.9$ eV (step 0.1): 160 replicas | same |
| Protocol | $x_0=0.99$; 10 MC moves per step; $1.2\times10^5$ steps; sample every 100 steps | `run_regc(x0=0.99, diffusion='mc_moves', diffusion_steps=10, record_every=100)` |
| Mass of B | not stated | $m_B = 16.48$ amu, from the Fig. 3 caption: (200 K, $-0.9$ eV) $\equiv 2.03\times10^{-17}$ atm via Eq. (13) |
| Displacement amplitude | not stated | 0.5 Å |

Confining B to the zone matters. If B may leave it (only insertion restricted, §2.5), B
($\sigma_{AB}=1.91$ Å) slips through the hollows of the top layer (A–A distance 3.83 Å). Most
of it then ends up absorbed between the two A layers, which roughly doubles the coverage and
erases the paper's layer plateaus. With B confined, single replicas give the following:

* (200 K, $-0.9$ eV): 18 B in the top-layer hollows ($z\approx0$), about 27 in a second layer
  ($z\approx0.75$–1.25 Å), and a few above, $N_B\approx 61$. [ZSG] give 59.
* (650 K, $-2.4$ eV): pristine.

The script compares two phase diagrams, both labelling phases by $N_B$:

* **MBAR@REGC:** `MBARSurfaceAnalysis.most_probable_N`, i.e. $\arg\max_N P(N)$ from Eq. (12).
* **aiAT@REGC:** the lowest sampled energy per $N_B$ in [ZSG Eq. 13], with vibrations neglected.

It reports how these diagrams relate to the paper's results:

* RE acceptance; [ZSG] report about 25%.
* Whether the paper's phases $N_B = 0, 18, 45, 59, 85$ appear.
* Agreement between the two diagrams at $N_B\le 18$; [ZSG] say they "almost coincide".

---

## 7. Summary of changes relative to [ZSG]

| # | Item | Status |
|---|---|---|
| 1 | Swap prefactor hard-coded as $(\beta_a/\beta_b)^{3/2\,\Delta N}$, wrong when $\Lambda^3\equiv1$ | **Fixed:** Eq. (8) with `lambda_cubed` (`engine.py`) |
| 2 | MBAR reduced potential missing $N\ln\Lambda^3(T)$ | **Fixed:** Eq. (9) (`analysis.py`); `lambda_cubed_of_T` in runner and examples 05–07 made valid at any $T$ |
| 3 | Deletion drew from all particles and used total $N$ with restricted $V_\text{ex}$ | **Fixed:** Eqs. (4)–(5) with $N_\text{ex}$ (`systems.exchange_zone_indices`, `replica.py`) |
| — | Sign of $\mu$ term in [ZSG Eq. 6] | Paper typo; code follows the derivation (8) |
| 4 | Diagonal swaps; one random swap type per RE move ([ZSG] Fig. 2) | **Implemented:** `swap_pairs`, `_exchange_round` (§3.2); random offset added for connectivity |
| 5 | Fig. 1 loop: $x_0$ GC vs RE → diffusion → sample → convergence | **Implemented:** `run_regc` (§4.1); $x_0 = P(\text{GC})$ as in the text (Fig. 1's labels are swapped) |
| 6 | Canonical diffusion via thermostatted MD | **Implemented:** `diffuse('md')` with ASE Bussi/Langevin NVT (§2.2) |
| 7 | Convergence by unchanged $\rho(U)$ | **Implemented:** Eq. (15) |
| — | Mixed sweep scheme `run` | Kept as an alternative (§4.2) |
| 8 | Reservoir $\mu_\text{H}$ missing the calculator's isolated-atom energy | **Fixed:** Eq. (14), `e_atom_eV` in `reservoir.py` and the runner |
