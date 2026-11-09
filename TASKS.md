# Tasks

Planned work on regc, with enough detail to pick each item up cold.

## Molecular exchange (e.g. CO on Pt55)

**Goal:** exchange whole molecules with a molecular gas, for example CO
adsorbing on a cluster or a surface. The gas would be CO, not O₂ dissociating
into O atoms.

**Why the current code cannot do it:**

- GC moves insert and delete one atom (`Replica._move_insert` /
  `_move_delete`, `ParticleSystem.apply_insert` / `apply_delete`).
- Exchanging C and O separately does not work. A lone C atom costs several
  eV, so its insertions are essentially never accepted. A CO gas also fixes
  only μ_C + μ_O, not each one separately.
- The reservoirs exchange atoms: `MonatomicIdealGasReservoir`, and
  `DiatomicReservoir` (X atoms from an X₂ gas, ½(μ_X₂ − D0) + e_atom).

**What is needed:**

1. **Molecule bookkeeping in `ParticleSystem`.**
   - Each mobile atom gets a molecule id; a molecule type is a rigid
     template (species and internal coordinates, e.g. C–O at the model's
     own bond length).
   - `n` and the recorded N count molecules. The per-species counts
     (`species_counts`) and `n_adsorbed` work per molecule: a molecule is
     adsorbed if any of its atoms is within the cutoff.
   - `full_configuration` is unchanged: the calculator still sees atoms.
2. **Moves.**
   - *Insertion:* a random centre of mass in the insertion zone, a random
     orientation (uniform on SO(3)), the template's atoms placed rigidly.
     Acceptance: exp(β(μ_mol − ΔU)) · V / ((N_mol + 1) Λ³_mol), with
     μ_mol carrying the internal (rotational, vibrational) free energy, so
     the inserted molecule is a rigid rotor at its equilibrium geometry.
   - *Deletion:* pick a molecule in the zone (by its centre of mass), remove
     all its atoms. Detailed balance needs the same orientation measure as
     the insertion.
   - *Rigid translation and rotation* MC moves for molecules, alongside
     the atom displacements.
   - *MD and HMC* move atoms freely. That is physically right, but a
     molecule can then dissociate or swap atoms with another one.
     - Options: (a) only intact molecules (bond within a tolerance) are
       deletable, so the zone count stays consistent; (b) LAMMPS
       `fix rigid/small` for molecules during MD.
     - Decide and test; (a) is simpler and keeps the ensemble exact as long
       as the deletable set is defined by the configuration alone.
3. **Reservoir.** `MolecularReservoir` for whole molecules:
   - μ_CO(T, p) = E_CO + ZPE + kT ln(p Λ³ / kT) − kT ln(q_rot q_vib), for
     a heteronuclear diatomic with symmetry number 1. `diatomic_gas_mu_eV`
     already computes most of it; it needs the molecule's energy as the
     zero and no ½ or D0.
   - E_CO comes from the potential (`from_calculator`: relax CO in a box),
     like `d0: self` for the atomic reservoirs.
   - Polyatomic molecules later: q_rot from the three moments of inertia,
     several vibrational modes.
4. **Several molecular species, and molecules alongside atoms** (CO + O
   from O₂): the multi-species machinery (`gas.composition`,
   `SpeciesLambdaCubed`) should carry over, with Λ³ of each molecule.
5. **Runner.**
   - `gas.molecules`: a name, or a template with species and geometry, e.g.
     `{CO: {atoms: [C, O], bond_length_angstrom: 1.128}}`.
   - `reservoir.type: molecular`.
   - `phase_diagram`: CO per surface atom.
6. **Analysis.** MBAR on N_mol needs no change; the coverage labels and
   docs need "molecules per surface atom".
7. **Tests.**
   - An ideal molecular gas, exact: ⟨N⟩ = exp(βμ) V/Λ³ with a zero
     potential, including the orientation measure.
   - Detailed balance of rigid rotations.
   - The CO reservoir against a hand-computed μ(T, p).
   - A small CO-on-Pt13 run with EMT (EMT has C and O; unphysical, but fast).
8. **Example.** `examples/pt55_co_grace/` mirroring
   `examples/pt55_o_grace/`.

**Open questions:**

- Keep molecules rigid during insertion only (then free in MD), or always?
  Free is more physical; rigid is simpler.
- Count a dissociated CO as one molecule, as C + O, or forbid dissociation?
  This matters only where the potential allows CO dissociation, which is
  rare on Pt.
