# REGC algorithm as implemented in `regc`

This document states the replica-exchange grand-canonical (REGC) Monte Carlo algorithm
that `regc` implements, with every acceptance rule written out, and relates it to

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
    a 1 fs time step) or Langevin (friction `friction_per_fs`, default 0.01/fs). Its target
    kinetic energy counts $3N$ degrees of freedom of the moving atoms: frozen atoms and the walls
    break momentum conservation, so no centre-of-mass degrees of freedom are removed. The native
    LAMMPS path (`LammpsCalculator.run_md`) does the same (`compute_modify extra/dof 0`).
  * Velocities are redrawn from Maxwell–Boltzmann at $T_l$ at the start of each segment. This is
    exact because $\mathbf p$ and $\mathbf R$ are independent in the canonical ensemble, and it
    handles freshly inserted atoms and configurations just swapped in from another $T$.
  * Frozen atoms are held by `FixAtoms`.
  * The hard walls of $V_\text{box}$ are enforced by specular reflection after every step
    ($z\to 2z_\text{wall}-z$, $v_z\to -v_z$), the analogue of the reflecting boundaries that
    [ZSG] impose with PLUMED.
  * After the segment, $E$ is recomputed with the system's calculator.
  * With `LammpsCalculator` (`regc/lammps_backend.py`) the whole segment runs natively in
    LAMMPS: Maxwell–Boltzmann velocities drawn in Python and scattered into LAMMPS (not
    `velocity create`, which rescales the draw to exactly $T$ and fails for one atom), `fix nve`
    + `fix temp/csvr` (Bussi, with `extra/dof 0`), `fix wall/reflect`, with frozen atoms
    excluded. Any LAMMPS pair style works this way (EAM, GRACE-FS `grace/fs`,
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

**Patterns without pairs.** On a grid with a single row or column, most (pattern, offset)
combinations have no pairs (on an $n_T\times 1$ grid, six of eight). With
`skip_empty_swaps=True`, the default for new runs, `run_regc` draws one combination uniformly
among those that have pairs, so no RE step is wasted. The draw still ignores the
configurations, so detailed balance is unchanged. A run resumed from a checkpoint keeps the
draw it started with (the previous behaviour for checkpoints written before this option), so
its continuation is bit-identical.

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
reduced energies $u$ of Eq. (9) are histogrammed on common bins for two disjoint blocks: the
samples recorded since the previous check ("new") and those recorded in the interval before it
("old"). The run stops when

$$
\max_k\; \tfrac12 \sum_{\text{bins } b} \left|\rho_k^\text{new}(b) - \rho_k^\text{old}(b)\right|
\;<\; \texttt{convergence\_tol},
\qquad (15)
$$

that is, when the total-variation distance between consecutive blocks is small for every
$\rho_k(u)$. The first verdict needs three checks. The blocks must be disjoint: comparing the
samples at the previous check with *all* samples (which contain them) gives a distance that
shrinks like $1/k$ after $k$ equal checks whatever the data do, so a drifting run would pass.
Two blocks of $n$ samples from one distribution still differ by roughly
$\sqrt{\text{bins}/\pi n}$ from sampling noise, so `convergence_tol` must lie above that
floor (fewer bins, or longer blocks). `n_steps` is a cap.
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
(2) invariant (global balance). Both loops therefore sample the same ensemble with MC or HMC
diffusion; with `'md'` both carry the finite-time-step bias of §2.2. §4.1 reproduces
the paper's protocol; §4.2 is usually more efficient for cheap force fields.

The YAML runner selects between them with `replica_exchange.scheme: sweep | regc`. The
`regc` scheme reads `n_steps`, `x0`, `diffusion` (`mc`, `mc_moves`, `hmc`, `md`),
`diffusion_steps`, `record_every`, `burn_in_steps`, `convergence_every`, `convergence_tol`,
`swap_types`, `displace_substrate`, an optional `md:` block (`timestep_fs`, `thermostat`,
`taut_fs`, `friction_per_fs`) and an `hmc:` block (`leapfrog_steps`, `timestep_fs`). With a
`checkpoint:` section it runs through `run_with_checkpoints` (resumable, wall-time aware).

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

### 5.3 Structural observables (`regc.observables`)

Eq. (10) holds for any function $A(\mathbf R)$ of the configuration, so an
observable recorded with every sample (`observables=` of the engine)
needs no change to the MBAR machinery. Where it is a ratio whose
denominator fluctuates, the numerator and denominator are recorded
separately and divided after reweighting. The mean coordination is
$\langle n_\text{bonds}\rangle/\langle n_\text{centres}\rangle$. The
per-centre radial distribution is
$n(r)=\langle H(r)\rangle/(\langle n_a\rangle\,\Delta r)$, with
$\int_0^{r_c}n(r)\,dr$ the mean coordination. Distances use the system's
minimum-image convention.

### 5.4 Heat capacity ([ZSG] PRL 128, 246101 (2022), Eq. 1; `heat_capacity`)

$$
C_V(T,\mu)=\frac{\langle E^2\rangle-\langle E\rangle^2}{k_BT^2},\qquad E=U+K,
\qquad (20)
$$

with both averages from Eq. (10). The momenta are not sampled by MC, and
they need not be. For $n$ moving atoms (the mobile ones plus any
displaceable substrate) the kinetic energy is independent of the
configuration given $n$, with $\langle K\,|\,n\rangle=\tfrac32 nk_BT$ and
$\mathrm{Var}(K\,|\,n)=\tfrac32 n(k_BT)^2$. So, with
$x=U+\tfrac32k_BT\,n$,

$$
\mathrm{Var}(E)=\mathrm{Var}(x)+\tfrac32(k_BT)^2\langle n\rangle .
\qquad (21)
$$

The phase-space weight of the grand ensemble with momenta integrated out
is $e^{\beta\mu N}\Lambda^{-3N}e^{-\beta U}/N!$, with
$\Lambda^{-3}\propto T^{3/2}$. Differentiating $\langle E\rangle$ with
respect to $T$ at fixed $\beta\mu$ gives exactly Eq. (20)–(21):

$$
\left.\frac{\partial\langle E\rangle}{\partial T}\right|_{\beta\mu}
=\frac{\mathrm{Var}(U)}{k_BT^2}+\frac{3\,\mathrm{Cov}(U,n)}{T}+\tfrac94 k_B\,\mathrm{Var}(n)
+\tfrac32 k_B\langle n\rangle .
$$

It is therefore not the heat capacity at fixed $p$ or fixed $N$. Its
ridges, the local maxima along $p$ at fixed $T$ and along $T$ at fixed
$p$ (`heat_capacity_ridges`), are what [ZSG] use to draw phase
boundaries. Two checks of the implementation:
- The MBAR estimator satisfies this identity exactly; a test compares it
  with a finite difference of the reweighted $\langle E\rangle$.
- An ideal gas gives $C_V/k_B=\tfrac{15}{4}\langle N\rangle$.

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

## 8. Several exchanged species (an extension beyond [ZSG])

Two gas reservoirs act at once in many surface problems (O$_2$ + a Cl promoter on Ag, O$_2$ + CO on
Pd). `ParticleSystem(exchange_species=['O', 'Cl'])` exchanges $S$ species $s = 1,\dots,S$, each
with its own $\mu_s$ and $\Lambda_s(T)$. The target ensemble (1) becomes

$$
\mathcal N(\beta,\boldsymbol\mu,\mathbf R) \propto
\prod_{s}\frac{V^{N_s}}{N_s!\,\Lambda_s^{3N_s}(T)}\;
e^{-\beta\left[E(\mathbf R) - \sum_s \mu_s N_s\right]},
\qquad (16)
$$

with $N_s$ the number of mobile particles of species $s$ (`ParticleSystem.species_counts`).
Particles of different species are distinguishable; particles of the same species are not.

**GC moves** (`Replica._move_insert_species`, `_move_delete_species`). Draw $s$ uniformly. An
insertion places one $s$ particle uniformly in $V_\text{ex}$; a deletion removes one of the
$N_{\text{ex},s}$ particles of species $s$ inside $V_\text{ex}$. The acceptance rules are
(4)–(5) with $\mu\to\mu_s$, $\Lambda\to\Lambda_s$, $N_\text{ex}\to N_{\text{ex},s}$:

$$
P_\text{acc}(N_s\to N_s+1) = \min\!\left[1,\;
\frac{V_\text{ex}}{\Lambda_s^3(T)\,(N_{\text{ex},s}+1)}\;
e^{\beta\left[\mu_s - (E_{N+1}-E_N)\right]}\right].
\qquad (17)
$$

The $1/S$ species-choice probability is the same in both directions and cancels. Counting
$N_\text{ex}$ over all species instead of species $s$ is a classic error that biases the
composition; `tests/test_multispecies.py` catches it.

**Replica exchange.** Inserting (16) into (7) gives (8) summed over species:

$$
P_\text{acc}^\text{swap} = \min\!\left[1,\;
\exp\!\Big\{(\beta_a-\beta_b)(E_a-E_b)
- \sum_s(\beta_a\mu_{a,s}-\beta_b\mu_{b,s})(N_{a,s}-N_{b,s})
+ \sum_s (N_{a,s}-N_{b,s})\ln\frac{\Lambda_s^3(T_a)}{\Lambda_s^3(T_b)}\Big\}\right].
\qquad (18)
$$

**The ladder stays 2D.** Each grid point carries a vector $\boldsymbol\mu = (\mu_1,\dots,\mu_S)$,
so `mus` has shape $(n_\mu, S)$ or $(n_T, n_\mu, S)$. The second ladder axis is therefore a path
in $\boldsymbol\mu$ space. A natural choice is a gas feed of fixed composition at increasing
total pressure: $\boldsymbol\mu(T,p) = (\mu_{\text{O}}(T, x_\text{O}p), \mu_\text{Cl}(T, x_\text{Cl}p))$.
All four swap patterns of §3.2 and the $\rho(u)$ convergence check work unchanged with (18) and (19).

**MBAR.** The reduced potential (9) becomes

$$
u(\mathbf R; \boldsymbol\mu, \beta) = \beta E(\mathbf R)
- \sum_s \left[\beta\mu_s - \ln\Lambda_s^3(T)\right] N_s(\mathbf R),
\qquad (19)
$$

built from the per-species counts the engine records as `samples[key]['N_s']`.
`MBARSurfaceAnalysis` reweights to any $(T,\boldsymbol\mu)$, including points *off* the
simulated path, as far as the sampled $(N_1,\dots,N_S)$ distributions overlap there. Several
quantities take a $\mu$ vector or a path of $\mu$ vectors:
* `mean_N_species`;
* `phase_diagram(..., observable='N_Cl')`;
* `most_probable_composition`, the argmax of the joint $P(N_1,\dots,N_S)$;
* `lowest_energy_N`, the aiAT construction over compositions;
* `pN(..., observable=...)`.

For $(T,p)$ output, pass `mu_of_Tp(T, p)` returning the vector (e.g. the fixed-feed path).
`locate_coexistence_mu` bisects a scalar $\mu$ and is disabled for $S>1$; scan `pN` along the
path instead.

**Exact checks** (`tests/test_multispecies.py`):
* Ideal gas, per species: $\langle N_s\rangle = V e^{\beta\mu_s}/\Lambda_s^3$ at every grid
  point, with a steeply $T$-dependent $\Lambda_\text{Cl}$ so that dropping the $\Lambda$ term of
  (18) shows up as a ~50% error. Also checked after MBAR reweighting to an unsampled
  $(T,\boldsymbol\mu)$.
* Two labels on *one* interacting particle type (`LJCalculator` ignores species). The grand
  partition function only depends on $z_1 + z_2$ with $z_s = e^{\beta\mu_s}/\Lambda_s^3$, so
  $\langle N\rangle$ and $\langle E\rangle$ must equal those of a single species with
  $\mu = kT\ln(z_1+z_2)$. In addition, each particle carries label $s$ with probability
  $z_s/(z_1+z_2)$.
* With one species, every code path is the original one, bit for bit.

---

## 9. Semi-grand-canonical substrate (an extension beyond [ZSG])

An alloy surface (PdPt, PtNi, ...) under a gas changes its own composition: an adsorbate that
binds one metal more strongly pulls that metal to the surface. `ParticleSystem.set_sgc_species`
(config section `sgc`) makes the displaceable substrate semi-grand-canonical: its atoms of species
$k \in \{1,\dots,K\}$ change identity among themselves at fixed total
$M = \sum_k N_k$, under chemical-potential differences $\Delta\mu_k = \mu_k - \mu_1$
(`ThermoState.sgc_dmu`, reference $k=1$, $\Delta\mu_1 = 0$). Frozen atoms keep their species. The
gas stays grand canonical, so (16) becomes

$$
\mathcal N(\beta,\boldsymbol\mu,\Delta\boldsymbol\mu,\mathbf R,\boldsymbol\sigma) \propto
\prod_{s}\frac{V^{N_s}}{N_s!\,\Lambda_s^{3N_s}(T)}\;
e^{-\beta\left[E(\mathbf R,\boldsymbol\sigma) - \sum_s \mu_s N_s
- \sum_k \Delta\mu_k N_k(\boldsymbol\sigma)\right]},
\qquad (22)
$$

with $\boldsymbol\sigma$ the substrate identities and $N_k(\boldsymbol\sigma)$ the swappable atoms
of species $k$ (`ParticleSystem.sgc_counts`). Derived from a grand-canonical substrate, (22) is
missing the factor $\prod_k \Lambda_k^{-3N_k}$ and the term $\mu_1 M$. Both are constant here:
$M$ is fixed, and $\Lambda_k^{-3N_k}$ differs between identities only by
$(m_k/m_1)^{3N_k/2}$, which is independent of $T$. $\Delta\mu_k$ is therefore *configurational*,
i.e. mass-free, the quantity that sets alloy phase equilibria. For a random bulk alloy at
$x = 1/2$ the ideal-mixing term $kT\ln[x/(1-x)]$ vanishes, so the $\Delta\mu$ of a 50-50
reservoir is $\partial E/\partial N_k$ there and is independent of $T$ (regular solution;
computed as the energy change of single swaps in a relaxed random 50-50 bulk cell). Away from $x = 1/2$ the random-alloy reservoir has

$$
\Delta\mu_k(T) = \frac{\partial E}{\partial N_k}\Big|_{\mathbf x}
+ kT\ln\frac{x_k}{x_1}
\qquad (22a)
$$

(config `sgc.bulk_composition`, engine `sgc_log_ratio`). The second term is the same at every
grid point in units of $kT$, so $\beta\Delta\mu_k = \beta\,\partial E/\partial N_k + \ln(x_k/x_1)$
contributes a state-independent $-\ln(x_k/x_1) N_k$ to the reduced potential. That term cancels
in (24) and in MBAR, so it only enters the identity move (23).

**Identity move** (`Replica._move_identity`, `sgc.attempts_per_step` per replica per step). Pick a
swappable atom uniformly from a set the move does not change, and a new species $k'$ uniformly
among the other $K-1$. The proposal is symmetric, so

$$
P_\text{acc}(k\to k') = \min\!\left[1,\; e^{-\beta\left[E' - E - (\Delta\mu_{k'} - \Delta\mu_k)\right]}\right].
\qquad (23)
$$

**Replica exchange.** Inserting (22) into (7) adds one term to (18):

$$
\ln P_\text{acc}^\text{swap} \mathrel{+}= -\sum_k (\beta_a\Delta\mu_{a,k} - \beta_b\Delta\mu_{b,k})
(N_{a,k} - N_{b,k}).
\qquad (24)
$$

The term has no $\Lambda$ factor. Writing it with full $\mu_k$ and $\Lambda_k$, as for a gas,
gives the same result, because $\sum_k (N_{a,k} - N_{b,k}) = 0$. Shifting every $\mu_k$ of a
replica by a constant (its reference $\mu_1$) then adds
$-(\beta_a c_a - \beta_b c_b)\cdot 0$. The ratio $\Lambda_k(T_a)/\Lambda_k(T_b) =
(T_b/T_a)^{1/2}$ is the same for every species, so the $\Lambda$ term reduces to
$\tfrac32\ln(T_b/T_a)\cdot 0$. Both cancellations need equal $M$ in every replica, and
`ReplicaExchangeGCMC` checks this. `ReplicaExchangeGCMC.swap_log_acceptance` implements
(18) + (24).

**MBAR.** The reduced potential (19) gains $-\beta\sum_k \Delta\mu_k N_k$. The engine records the
counts as the observable `sgc_counts` (`regc.observables.SGCCounts`). Its metadata carries
$\Delta\boldsymbol\mu$, so the analysis can rebuild (22) from `samples.npz` alone. With
$\Delta\boldsymbol\mu$ equal at every grid point and independent of $T$, the term is an energy:
`MBARSurfaceAnalysis` reweights with $E_\text{eff} = E - \sum_k\Delta\mu_k N_k$ (`U_all`; the
physical energy is kept in `U_phys_all`). Every $(T,p)$ map, and the heat capacity (20)–(21)
taken at fixed $\Delta\boldsymbol\mu$, then holds in the ensemble (22). Surface compositions
are ratios of reweighted counts, $\langle N_{k,\text{top}}\rangle/\langle N_\text{top}\rangle$
(`regc.observables.LayerComposition`).

**Tests** (`tests/test_sgc.py`). Swappable atoms with site energies only are independent two-state
systems, $x_k(T) = [1 + e^{(h_k - h_1 - \Delta\mu_k)/T}]^{-1}$. The test checks this on a ladder
in $T$, at the grid points and reweighted between them. It uses a large $\Delta\mu$ and
swap-dominated sampling, so dropping the term (24) biases the composition visibly and fails the
test.

## 10. Molecular insertion and deletion (an extension beyond [ZSG])

CO, CH$_4$ or H$_2$O cannot be exchanged atom by atom: a pure CO gas fixes only
$\mu_\text{C} + \mu_\text{O}$, not each, and an isolated C atom is almost never accepted.
`regc.molecules` adds **molecular moves without changing the ensemble**: the state is still a set
of atoms with elemental chemical potentials $\mu_e$, sampled from (16), and replica exchange (18),
MBAR and the sample files are those of §8. A molecule is only a way of *proposing* several atoms
at once (`ParticleSystem.set_molecules`, config section `molecules`).

**Target, unlabelled.** For a set $S$ of mobile atoms (the $N_e!$ of (16) absorbed by counting
each set once),

$$
\pi(S) \propto \prod_e z_e^{N_e}\, e^{-\beta E(S)}, \qquad z_e = \frac{e^{\beta\mu_e}}{\Lambda_e^3(T)} .
\qquad (25)
$$

**Units.** A molecule type has an anchor element (C in CO, CO$_2$, CH$_4$; O in O$_2$, H$_2$O),
partner roles $j = 1,\dots,k$ with their elements, a relaxed gas-phase template $\mathbf t_j$
(anchor to partner $j$) and a shell $[r_\text{min}, r_\text{max}]$. A *unit* of a configuration is a
set of atoms with the molecule's composition, its anchor inside $V_\text{ex}$, every partner within
the shell of the anchor (minimum image). $M(S)$ is their number
(`MoleculeMove.count_units`). Only star-shaped molecules (all partners bonded to the anchor) fit
this definition.

**Proposal.** An insertion draws the anchor uniformly in $V_\text{ex}$ (density $1/V_\text{ex}$) and
then each partner from a density $q_j$ that can be evaluated (`MoleculeMove.propose`,
`log_q_roles`): the distance $r_j$ from a normal around $|\mathbf t_j|$ truncated to the shell
(density $f_j(r)/r^2$ per volume); the direction uniform for $j = 1$; for $j = 2$ with
$u = \hat{\mathbf d}_2\cdot\hat{\mathbf d}_1$ from a normal around $\hat{\mathbf t}_1\cdot\hat{\mathbf t}_2$ truncated to
$[-1, 1]$ and a uniform azimuth; for $j \ge 3$ von Mises-Fisher around the template direction expressed
in the frame of $\hat{\mathbf d}_1, \hat{\mathbf d}_2$. The first out-of-plane role draws either mirror
image with probability ½, so its density is the ½–½ mixture; later roles take the handedness of the
actual position of that role. A deletion picks one of the $M(S)$ units uniformly.

The same unit $u$ can come out of several insertions: every atom of the anchor element inside
$V_\text{ex}$ as the anchor (both O of O$_2$), every assignment of the other atoms to the roles of their
element (4! for the H of CH$_4$). Its proposal density is the sum

$$
Q(u) = \sum_{\text{anchor } a \in u}\;\sum_{\sigma}\; \frac{1}{V_\text{ex}}
\prod_{j=1}^{k} q_j\big(\mathbf r_{\sigma(j)} - \mathbf r_a \,\big|\, \text{roles } < j\big)
\qquad (26)
$$

(`MoleculeMove.log_unit_density`). Every $q_j$ is continuous and positive over the shell, so $Q(u) > 0$
for every unit whatever HMC or MD has done to its bonds.

**Acceptance.** With $\nu_e$ the molecule's composition, (25) and the proposal ratio give

$$
P_\text{acc}(S\to S\cup u) = \min\!\left[1,\;
\frac{\prod_e z_e^{\nu_e}\; e^{-\beta\,(E_{S\cup u}-E_S)}}{M(S\cup u)\; Q(u)}\right],
\qquad
P_\text{acc}(S\to S\setminus u) = \min\!\left[1,\;
\frac{M(S)\,Q(u)\; e^{-\beta\,(E_{S\setminus u}-E_S)}}{\prod_e z_e^{\nu_e}}\right]
\qquad (27)
$$

(`Replica._move_insert_molecule`, `_move_delete_molecule`). For a single atom ($k = 0$,
$Q = 1/V_\text{ex}$, $M = N_\text{ex}+1$) this is (17). GC attempts pick a channel -- every atomically
exchanged species (weight 1) and every molecule (weight `weight`) -- independently of the
configuration and alike for insertions and deletions, so each channel is in detailed balance on its
own and they can be combined freely.

**Elemental chemical potentials from gases** (`reservoir.MolecularGas`, `MolecularGasFeed`). A gas
$g$ of composition $\boldsymbol\nu_g$ at partial pressure $x_g p$ has

$$
\mu_g = E_g + \text{ZPE}_g + k_BT\ln\frac{x_g p\,\Lambda_g^3}{k_BT} - k_BT\ln\!\big(g_\text{el}\, q_\text{rot}\, q_\text{vib}\big),
\qquad
\boldsymbol\nu_g\cdot\boldsymbol\mu = \mu_g \;\;\forall g,
\qquad (28)
$$

$E_g$ the energy of the molecule relaxed with the simulation potential, a classical rigid rotor
(linear or not, symmetry number $\sigma$), harmonic vibrations from the potential's Hessian, $q_\text{vib}$
counted from $v = 0$. For a diatomic, (28) is (14) times two. The linear system is solved with the
pseudo-inverse (minimum norm). The gas compositions must be linearly independent: CO, O$_2$ and CO$_2$
together have common $\mu_e$ only at gas-phase equilibrium. One gas per independent composition is the
*constrained equilibrium* of atomistic thermodynamics (Reuter and Scheffler, PRB 68, 045407 (2003)):
the surface is in equilibrium with every gas, the gases not with each other. With fewer gases than
elements (pure CO) the individual $\mu_e$ are not fixed. Only the combinations that the moves use
enter (27), and those are unique if every move's composition is a combination of the gases'
(`MolecularGasFeed.check_moves`: a pure CO feed allows CO moves, not atomic C).

**Tests** (`tests/test_molecules.py`). The densities integrate to 1. In an ideal gas the molecule
number follows $P(n) \propto (\prod_e z_e^{\nu_e} V^{\sum\nu})^n / \prod_e N_e!$ for CO, O$_2$ (anchor
among the partners) and CH$_4$ (24 orderings and the mirror mixture). With atomic moves added, $N_\text{C}$
and $N_\text{O}$ are independent Poisson variables. In an interacting gas, atomic and atomic + molecular
exchange give the same averages. The running energy matches a recomputation with EMT.

## 11. Subsurface insertion with nonequilibrium candidate Monte Carlo (an extension beyond [ZSG])

An atom inserted at a random point inside a metal is essentially never accepted: the lattice has no
room until it relaxes around the new atom. `regc.subsurface` (config section `subsurface`,
`SlabSystem.set_subsurface`) splits a slab at a fixed height $z_c$:

* **gas region** $z \ge z_c$: the instantaneous moves of §2.3–2.4, §8 and §10 (atoms and molecules).
  The exchange zone's lower bound is raised to $z_c$, so they act only there;
* **subsurface region** $z_\text{min} \le z < z_c$: one channel per species, with site-biased
  insertion and nonequilibrium candidate Monte Carlo (NCMC, Nilmeier et al., PNAS 108, E1009 (2011)).

**Sites and proposal.** The interstitial sites $\{\mathbf s_k\}_{k=1}^{K}$ of a configuration
$\mathbf x$ are the vertices of the Voronoi tessellation of its metal atoms (frozen and displaceable,
with their periodic images) that lie in the region, at least `site_min_distance` from every metal
atom, merged within 0.25 Å (`Subsurface.sites`). For fcc they are the octahedral and tetrahedral
holes. An insertion draws a site uniformly and a point uniformly in its ball of radius $\delta$:

$$
q(\mathbf y;\mathbf x) = \frac{1}{K(\mathbf x)} \sum_{k:\,|\mathbf y-\mathbf s_k|<\delta} \frac{1}{V_\delta},
\qquad V_\delta = \tfrac43\pi\delta^3 .
\qquad (29)
$$

The sites move with the metal atoms, so $q$ depends on $\mathbf x$. Weighting the sites by the volume of
their Voronoi cells, as in the original algorithm, would only change the efficiency, and the uniform
choice keeps (29) exactly computable.

**NCMC.** With $U_\lambda = (1-\lambda)\,U_N + \lambda\,U_{N+1}$ ($U_N$: the energy without the switched
atom; a hard boundary forbids it at every $\lambda$) and a schedule $\lambda_t = t/T$, the protocol
(`subsurface.ncmc`) accumulates the work

$$
W = \sum_{t=1}^{T}\big[U_{\lambda_t}(\mathbf x_{t-1}) - U_{\lambda_{t-1}}(\mathbf x_{t-1})\big],
\qquad \mathbf x_t = K_{\lambda_t}(\mathbf x_{t-1}) \;\; (t < T),
\qquad (30)
$$

where $K_\lambda$ is a kernel in detailed balance with $e^{-\beta U_\lambda}$: an HMC trajectory of
every displaceable atom (substrate and mobile, the switched one included, with full momentum
refresh and a Metropolis test), or single-atom Metropolis displacements. A removal runs the reversed
schedule $\lambda_t = 1 - t/T$, so its kernels are those of the insertion in the reverse order.

The third kernel is Langevin dynamics with the palindromic BAOAB splitting (V R O R V; `kernel:
langevin`). Velocities are drawn from Maxwell–Boltzmann at the start of the switch, carried through
every $\lambda$ step and discarded at the end. A discrete Langevin step is not in detailed balance,
but for this splitting the total work

$$
W = W_\text{prot} + W_\text{shad}, \qquad
W_\text{shad} = \sum_\text{steps}\big[(K_1 - K_0) + (K_3 - K_2) + U_\lambda(\mathbf x') - U_\lambda(\mathbf x)\big]
\qquad (30a)
$$

obeys the Crooks fluctuation theorem exactly at any time step (Sivak, Chodera, Crooks, PRX 3, 011007
(2013)). Here $W_\text{prot}$ is (30), $K_0 \dots K_3$ are the kinetic energies around the two kicks,
and the heat exchanged in the O substep is excluded. Using this $W$ in (31) makes the moves exact.
The friction carries the strain energy released by the relaxing lattice into the bath as heat, so
it does not count against acceptance. GCNCMC (Melling et al., JCTC 19, 1050 (2023)) uses Langevin
propagation in the same way. Thermostatted MD with velocity rescaling has no such work accounting
and is not offered as a kernel.

**Acceptance.** With $z_s = e^{\beta\mu_s}/\Lambda_s^3$ and $N_\text{sub}$ the number of atoms of
species $s$ in the region,

$$
P_\text{acc}^\text{ins} = \min\!\left[1, \frac{z_s\, e^{-\beta W}}{N_\text{sub}(\mathbf x')\; q(\mathbf y;\mathbf x)}\right],
\qquad
P_\text{acc}^\text{rem} = \min\!\left[1, \frac{N_\text{sub}(\mathbf x)\; q(\mathbf y;\mathbf x')\; e^{-\beta W}}{z_s}\right]
\qquad (31)
$$

(`Replica._move_insert_subsurface`, `_move_delete_subsurface`). For an insertion, $\mathbf y$ is the
proposed point and $\mathbf x'$ the final configuration. For a removal, $\mathbf y$ is the switched
atom's final position and $\mathbf x'$ the configuration without it. The atom counts are taken in
the configuration where the reverse move would pick, because the propagation can carry other atoms
across the region's boundaries. A proposal outside the region, or a switched atom that ends outside
it, is rejected, which keeps the pair of moves reversible. Summing the kernels' detailed balance
along the path turns the ratio of path probabilities into $e^{-\beta(W - \Delta U)}$, and (31)
follows. With $T = 1$ (no propagation) $W = \Delta U$, and (31) is an instantaneous site-biased
insertion. For one atom and $q = 1/V_\text{ex}$ it reduces to (17).

**Channels.** A GC attempt picks a channel independently of the configuration and alike for insertions
and deletions: an atomic gas channel per atomically exchanged species (weight 1), a molecule (its
`weight`), or a subsurface species (`subsurface.weight`). Each channel is in detailed balance on its own.
Molecules are exchanged only in the gas region. The default subsurface species are the exchanged
species without a `molecules` section. With one, they are the atomically exchanged species and the
elements of homonuclear diatomic gases (O$_2$, H$_2$: their subsurface attempts are atomic, with
$\mu_\text{O} = \mu_{\text{O}_2}/2$). Heteronuclear and larger molecules (CO, CH$_4$) stay above the
surface.

**Tests** (`tests/test_subsurface.py`). For fcc(100) the sites are exactly the octahedral and
tetrahedral holes. The proposal density integrates to the probability of landing in the region. In a
one-body potential well $f(z)$ the mean number of atoms in the subsurface and gas regions equals
$zA\int e^{-\beta f}dz$ for $T = 1$ and for the HMC, Langevin and MC kernels. At zero friction the
shadow work equals the integrator's energy error. The running energy stays exact with a
displaceable substrate.

## 12. Summary of changes relative to [ZSG]

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
| 9 | Several gas reservoirs at once (e.g. O + Cl) | **Implemented:** Eqs. (16)–(19), §8 |
| 10 | Semi-grand-canonical substrate (identity swaps) with a gas | **Implemented:** Eqs. (22)–(24), §9 |
| 11 | Molecular insertion/deletion (CO, O$_2$, CH$_4$, ...) with elemental $\mu$ from gas mixtures | **Implemented:** Eqs. (25)–(28), §10 |
| 12 | Subsurface insertion/removal: site-biased proposal + NCMC | **Implemented:** Eqs. (29)–(31), §11 |
