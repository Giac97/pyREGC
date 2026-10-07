"""
Check that this environment can run gcmc_re's Fe-H example on the cluster.

    python hpc/check_env.py            # EAM styles
    python hpc/check_env.py --grace    # also require pair_style grace/fs

Exit code 0 if everything needed is present, 1 otherwise.
"""

import argparse
import importlib
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--grace", action="store_true", help="also require pair_style grace/fs")
args = ap.parse_args()

ok = True


def report(name: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"  [{'ok' if good else 'MISSING'}] {name}{('  ' + detail) if detail else ''}")


print("Python packages:")
for mod in ("numpy", "scipy", "ase", "pymbar", "matplotlib", "gcmc_re"):
    try:
        m = importlib.import_module(mod)
        report(mod, True, getattr(m, "__version__", ""))
    except ImportError as e:
        report(mod, False, str(e))

print("LAMMPS Python module and styles:")
try:
    from lammps import lammps
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    report("lammps", True, f"version {lmp.version()}")
    needed = [("pair", "eam/fs"), ("pair", "eam/alloy"), ("fix", "nve"), ("fix", "temp/csvr"),
              ("fix", "wall/reflect"), ("compute", "temp")]
    if args.grace:
        needed.append(("pair", "grace/fs"))
    for cat, name in needed:
        report(f"{cat} {name}", lmp.has_style(cat, name))
    lmp.close()
except Exception as e:  # noqa: BLE001 -- any failure means LAMMPS is unusable
    report("lammps", False, str(e))

print("Threading (should all be 1 inside jobs):")
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    print(f"  {var}={os.environ.get(var, '(unset)')}")

print("environment OK" if ok else "environment INCOMPLETE")
sys.exit(0 if ok else 1)
