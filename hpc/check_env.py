"""
Check that this environment can run regc's Fe-H example on the cluster.

    python hpc/check_env.py                                  # EAM styles, default LAMMPS
    python hpc/check_env.py --grace                          # also require pair_style grace/fs
    python hpc/check_env.py --lammps-name grace --grace-runtime
        # liblammps_grace.so (hpc/build_lammps.sh), also require grace/runtime + pace/runtime

Exit code 0 if everything needed is present, 1 otherwise.
"""

import argparse
import importlib
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--grace", action="store_true", help="also require pair_style grace/fs")
ap.add_argument("--grace-runtime", action="store_true",
                help="also require pair_style grace/runtime and pace/runtime (grace-runtime build)")
ap.add_argument("--lammps-name", default=None,
                help="LAMMPS library to check: NAME -> liblammps_NAME.so (default: "
                     "$REGC_LAMMPS_NAME, else liblammps.so)")
args = ap.parse_args()

ok = True


def report(name: str, good: bool, detail: str = "") -> None:
    global ok
    ok &= good
    print(f"  [{'ok' if good else 'MISSING'}] {name}{('  ' + detail) if detail else ''}")


print("Python packages:")
for mod in ("numpy", "scipy", "ase", "pymbar", "matplotlib", "regc"):
    try:
        m = importlib.import_module(mod)
        report(mod, True, getattr(m, "__version__", ""))
    except ImportError as e:
        report(mod, False, str(e))

print("LAMMPS Python module and styles:")
try:
    import warnings

    import lammps as lammps_module
    from regc.lammps_backend import open_lammps, resolve_lammps_name
    name = resolve_lammps_name(args.lammps_name)
    module_version = lammps_module.__version__
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lmp = open_lammps(name)
    report("lammps", True, f"library {lmp.lib._name}, version {lmp.version()} "
                           f"(Python module {module_version})")
    for w in caught:
        print(f"  [note] {w.message}")
    needed = [("pair", "eam/fs"), ("pair", "eam/alloy"), ("fix", "nve"), ("fix", "temp/csvr"),
              ("fix", "wall/reflect"), ("compute", "temp")]
    if args.grace:
        needed.append(("pair", "grace/fs"))
    if args.grace_runtime:
        needed += [("pair", "grace/runtime"), ("pair", "pace/runtime")]
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
