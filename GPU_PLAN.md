# Plan: running regc on a single GPU

Ideas and work items for using one NVIDIA GPU, with enough detail to pick
each item up cold. Nothing here is implemented yet (2026-11-13).

## Goal and summary

Make a REGC run, and the work around it (re-evaluation, analysis), faster
on one GPU, without changing the statistics it produces.

- **A GPU helps only if many replicas are evaluated at once.** A replica
  has 40–70 atoms. On a GPU a single energy of that size is bound by
  kernel-launch latency, not arithmetic. The parallel work is across
  replicas: between replica-exchange (RE) moves they are independent, and
  in a REGC step every replica makes its GC attempt at the same moment.
  32 replicas × 60 atoms is about 2000 atoms per call.
- **For EAM the CPU is probably faster.** Measured with the LAMMPS fast
  path (60-atom Fe(100)+H, Wen EAM, one core of an Intel Core Ultra 5
  225U): 0.21 ms per energy, 0.27 ms per MC move, 2.2 ms per GC attempt,
  6.3 ms per 40-step MD segment. GPU latency per call is of the same
  order, so expect no gain.
- **For MLIPs (GRACE, MACE) it is promising.** Each evaluation does far
  more arithmetic, and the models accept batches of structures natively.
  This has to be measured (G0) before anything is built.

## Constraints

- **The statistics must not change.** Every item below must keep
  detailed balance. Test it the way the CPU code is tested: exact
  ideal-gas ⟨N⟩, and the same results as the CPU path on a small LJ/EAM
  case within error bars (bit-identity is not expected; see below).
- **Keep the CPU path.** GPU code is optional, behind a calculator or
  backend choice; `import regc` must keep working without CUDA.
- **Measure before porting** (Scientific Agent Skills,
  `optimize-for-gpu`): representative sizes, warm-up excluded,
  synchronized timings, end-to-end time per REGC step and per
  independent sample, not per kernel.
- **Hardware.** The development workstation has no usable NVIDIA GPU
  (Intel Arrow Lake graphics only). All GPU work needs a GPU node.

## Where the parallel work is

| Level | Parallel? | Used now |
|---|---|---|
| Steps of one replica | no (Markov chain) | – |
| Replicas between RE moves | yes, independent | worker processes (`n_workers`), one replica per call |
| GC attempts within one REGC step | yes, every replica attempts at once | in workers, separately |
| Candidates within one move | yes, if the move has several (multiple-try, CBMC) | no such moves yet |
| Re-evaluation of stored configurations | yes, fully independent | worker processes |
| MBAR (K × N matrices), bootstrap resamples | yes | NumPy, one core |

## Work items

### G0. Measure — S (do first)

- On a GPU node, time with a GRACE model through `grace/runtime` (a
  1L model, as in `examples/pd111_o_grace`) and with the Wen EAM:
  1. one 60-atom energy on CPU (LAMMPS fast path) and on GPU (LAMMPS
     Kokkos, `grace/runtime`);
  2. one evaluation of a stack of 8, 32 and 128 replicas (G2 geometry,
     built by hand) on the GPU, with per-replica energies from
     `compute pe/atom` summed per group;
  3. host↔device copy and `run 1 pre no post no` overhead per call;
  4. the same with `n_workers` processes sharing the GPU (G1).
- **Decision rule.** The GPU is worth it when its batched time per
  configuration is below (CPU time per evaluation) / (cores available
  for workers). Record GPU model, LAMMPS build, package versions and
  batch sizes with the numbers.
- **Output.** A short table in this file, and a go/no-go per potential.

### G1. Share one GPU among the existing workers — S

- **What.** No change to regc. Each worker process opens a LAMMPS built
  with Kokkos/CUDA (`-k on g 1 -sf kk`, via `lammps_name` and launch
  arguments), all on the same GPU. Run with the NVIDIA Multi-Process
  Service (MPS) so kernels from different processes run concurrently.
- **Check.** The fast path (`scatter_atoms` + `run 1 pre no post no`)
  with Kokkos: does it trigger full host↔device syncs every call? Does
  `energy_after_md` still hold?
- **Limits.** One CUDA context per process (roughly 0.3–0.5 GB each) caps
  the number of workers; every call stays tiny and latency-bound.
- **Expected.** Small to moderate gain for GRACE, none for EAM. Cheapest
  possible test; it also tells whether G2 is needed.
- **Touches.** `lammps_backend.py` (launch arguments, a `gpu` option),
  `hpc/` scripts (start/stop MPS), docs.

### G2. Stacked replicas in one LAMMPS instance — M

- **What.** All replicas live in one LAMMPS box, stacked along z with
  gaps larger than the potential's cutoff (plus the receptive field for
  message-passing MLIPs). Slabs can share the box because every replica
  has the same in-plane cell. Clusters can be placed apart in x, y and z.
  - Each replica's atoms form a LAMMPS group;
    `compute pe/atom` + `compute reduce` per group gives every replica's
    energy from one evaluation.
  - **Lockstep MC/GC.** Every replica draws its trial move with its own
    RNG; all moves are applied; one evaluation; each replica
    accepts or rejects from its own ΔU; rejected moves are reverted (the
    reverted positions are scattered back at the next evaluation, no
    extra call).
  - **MD diffusion.** One `run` of the whole stack with one Bussi
    thermostat per group at that replica's T (`fix temp/csvr` per group)
    and per-replica walls (`fix wall/reflect` or `wall/region` per
    group); velocities drawn per replica in Python as now.
- **Swaps.** Replica exchange moves configurations between states, i.e.
  between stack slots. Either swap the state labels of slots (cheap: the
  thermostat target and μ move, atoms stay) or swap coordinates.
  Swapping labels is the natural choice.
- **Exactness.** Replicas are independent between swaps; lockstep only
  reorders independent computations. The RE decision needs only (β, μ, U,
  N) and is unchanged.
- **Touches.** New `regc/stack.py` (or a `StackedLammps` backend),
  `engine.py` (a lockstep driver for the regc scheme alongside the
  per-replica worker path), `replica.py` (split "propose" from
  "accept/revert").
- **Risks.** LAMMPS bookkeeping when N changes (see G3); MLIPs whose
  receptive field needs large gaps; per-group fixes multiplying with
  132 replicas (fine in LAMMPS, but check the overhead).

### G3. Fixed atom count: parked atoms — M (inside G2)

- **What.** Give every replica slots for its largest possible number of
  mobile atoms. An unused slot is "parked": its atom sits far from
  everything, each parked atom in its own spot, beyond the cutoff of all
  others. A parked atom contributes the isolated-atom energy, a constant
  (0 for EAM; the reference energy E₀ for an MLIP). Subtract
  (number parked) × E₀ per replica.
- **Effect.** Insertion = move a parked atom to the trial position;
  deletion = park it. LAMMPS never creates or deletes atoms, and the
  existing fast path (copy positions, `run 1`) is all that runs.
- **Check.** That the isolated-atom energy is really constant (no
  dependence on the parking position or on long-range terms). Grow the
  number of slots (rare full rebuild) when a replica runs out.

### G4. Batched Python calculator, without LAMMPS — M–L

- **What.** A calculator interface `energies(configs) -> array` for a
  list of configurations. GRACE (TensorFlow/TensorPotential) and MACE
  (PyTorch) evaluate a batch of graphs natively. The engine's lockstep
  driver (G2) calls it once per step for all replicas, in the main
  process, with replica state kept in arrays.
- **Why.** Best GPU use and no LAMMPS bookkeeping. It is also the natural
  interface for G5 and G6.
- **MD** then needs forces from the same model and a batched integrator.
  Either keep MD in LAMMPS (G2) or write a small batched velocity-Verlet
  with a Bussi thermostat (NVIDIA Warp or the model's framework).
- **Touches.** `calculators.py` (interface), new `batched.py`,
  `engine.py`.

### G5. Spend the spare GPU capacity on sampling — M (after G2 or G4)

A batched GPU evaluates k candidates nearly as fast as one. Ways to use
that:

- **Multiple-try / configurational-bias insertion**:
  k candidate positions per replica in the same batch, choose one with
  Rosenbluth weights, accept with the CBMC rule; the reverse (deletion)
  move needs k − 1 decoys. This raises GC acceptance at high coverage,
  where most insertions fail now.
- **Speculative MC:** evaluate the next d proposals of a replica at once,
  each assuming the earlier ones are rejected; keep results up to the
  first acceptance. Exact; pays off when acceptance is low.
- **More replicas at almost no extra wall time** while the GPU is not
  saturated. The Fe–H benchmark (`MANUAL.md` §9.6.1) shows the need: 26
  of 94 neighbouring pairs overlap below 0.03.
- **Exactness.** Multiple-try acceptance must include the reverse-move
  weights. Test on the ideal gas and an LJ fluid against the
  single-try path.

### G6. Re-evaluation of records on the GPU — S–M

- **What.** `regc.records.reevaluate` already has thousands of
  independent stored configurations. Evaluate them in batches of hundreds
  with a batched MLIP (G4 interface) or a stacked LAMMPS (G2).
- **Why first among the builds.** Embarrassingly parallel, no change to
  sampling, and it serves the potential hierarchy (re-evaluation with
  fine-tuned MLIPs, distillation). For reference: 48 000 Fe–H
  configurations took 77 s with EAM on 6 CPU workers; with an MLIP that
  becomes hours on CPU.

### G7. Analysis on the GPU — S

- **What.** CuPy (a drop-in for NumPy) for the K × N exponentials and
  matrix products in `_weight_blocks`, `_mbar_free_energies_newton` and
  the bootstrap loop.
- **Priority: low.** After the November 2026 Newton fix a 40-resample
  bootstrap of 38 400 samples takes about a minute. Worth it for merged
  sets of 100+ states, 10⁵–10⁶ samples, or hundreds of resamples.
- **Keep** the NumPy path as default; select by an option.

## Exactness and reproducibility on a GPU

- **Lockstep is exact.** Each replica keeps its own random stream, so
  lockstep runs make the same decisions as the CPU run up to energy
  differences.
- **Energies are not bit-reproducible.** Atomic additions in different
  orders, and float32 in MLIPs, make the same configuration give
  slightly different energies. Consequences:
  - the running energy must be updated with ΔU from one consistent
    scheme (both energies from the same kind of evaluation);
  - `check_energies` needs a tolerance matching the GPU's noise;
  - noise of ~10⁻⁶ eV is negligible against kT ≈ 0.03 eV, but measure
    it (G0) rather than assume it.
- **Tests to add with the first GPU code:** ideal-gas ⟨N⟩ through the
  lockstep driver; LJ/EAM lockstep vs per-replica path within bootstrap
  errors; energy noise per configuration on the GPU.

## Order and decision points

1. **G0** (measure) on a GPU node with a GRACE model and the EAM.
2. **G1** (shared GPU, no code). If it already beats the CPU workers for
   GRACE, decide whether more is needed.
3. **G6** whenever convenient: independent of sampling, immediately
   useful for re-evaluating with other potentials.
4. If G0 shows batching wins: **G2 + G3** (stacked LAMMPS, parked atoms);
   **G4** only if LAMMPS bookkeeping becomes the bottleneck or a model
   has no LAMMPS interface.
5. **G5** on top of G2/G4.
6. **G7** when analyses grow (merged runs, large bootstraps).

## Open questions

- Which GPU nodes are available (model, memory, MPS allowed?), and is the
  GRACE LAMMPS build there built with Kokkos/CUDA?
- Does `grace/runtime` give per-atom energies (needed for G2's per-group
  energies)? If not, G2 needs one evaluation per replica group or G4.
- Receptive field of the GRACE models: sets the gap between stacked
  replicas and the parking distance.
- Does lockstep (G2/G4) interact with a future asynchronous replica
  pool? They pull in opposite directions: lockstep batches across
  replicas, an asynchronous pool lets replicas run at their own pace.
  Decide per hardware: GPU → lockstep, many CPU nodes → asynchronous.
