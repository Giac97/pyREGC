"""
Validate that ASECalculatorAdapter (the generic, correct-for-any-potential
path) exactly matches the fast native LJCalculator path, on the same
configuration. This is the check to rerun -- or copy the pattern from --
whenever you plug in a new Calculator: it's cheap insurance that the
generic clone-and-recompute dU is behaving correctly before you trust it
for something expensive like EAM or an MLIP.

Requires: pip install ase
"""

import numpy as np
from ase.calculators.lj import LennardJones as ASELennardJones

from gcmc_re import ASECalculatorAdapter, LJCalculator, BulkSystem

rng = np.random.default_rng(0)
L = 6.0
sigma, epsilon, r_cut = 1.0, 1.0, 2.5
positions = rng.random((30, 3)) * L

# native fast path
sys_fast = BulkSystem(L, LJCalculator(sigma, epsilon, r_cut), positions=positions.copy())

# generic path, wrapping ASE's own LennardJones with matching settings
# (shift=True reproduces this package's U(r_cut)=0 convention exactly)
ase_calc = ASECalculatorAdapter(
    ASELennardJones(sigma=sigma, epsilon=epsilon, rc=r_cut, smooth=False, shift=True),
    cell=[L, L, L], pbc=(True, True, True),
)
sys_generic = BulkSystem(L, ase_calc, positions=positions.copy())

# 1. total energy must match exactly
e_fast = sys_fast.total_energy()
e_generic = sys_generic.total_energy()
print(f"total_energy   fast={e_fast:.10f}  generic={e_generic:.10f}  "
      f"diff={abs(e_fast - e_generic):.2e}")
assert abs(e_fast - e_generic) < 1e-8

# 2. per-move dU must match exactly, for all three move types
i = 5
new_pos = positions[i] + np.array([0.3, -0.2, 0.1])
du_fast = (sys_fast.pair_energy_of(new_pos, exclude=i)
           - sys_fast.pair_energy_of(positions[i], exclude=i))
du_generic = sys_generic.generic_displace_dU(i, new_pos)
print(f"displace dU    fast={du_fast:.10f}  generic={du_generic:.10f}  "
      f"diff={abs(du_fast - du_generic):.2e}")
assert abs(du_fast - du_generic) < 1e-8

trial_pos = rng.random(3) * L
du_fast_ins = sys_fast.pair_energy_of(trial_pos)
du_generic_ins = sys_generic.generic_insert_dU(trial_pos)
print(f"insert dU      fast={du_fast_ins:.10f}  generic={du_generic_ins:.10f}  "
      f"diff={abs(du_fast_ins - du_generic_ins):.2e}")
assert abs(du_fast_ins - du_generic_ins) < 1e-8

j = 10
du_fast_del = -sys_fast.pair_energy_of(positions[j], exclude=j)
du_generic_del = sys_generic.generic_delete_dU(j)
print(f"delete dU      fast={du_fast_del:.10f}  generic={du_generic_del:.10f}  "
      f"diff={abs(du_fast_del - du_generic_del):.2e}")
assert abs(du_fast_del - du_generic_del) < 1e-8

print("\nAll checks passed: the generic path is safe to trust for a new Calculator.")
