"""
Deprecated alias of `regc`, the package's former name.

``import gcmc_re`` and ``from gcmc_re.engine import ...`` keep working and give
the very same objects as ``regc``. This keeps old scripts, notebooks and,
above all, pickled checkpoints (``checkpoint.pkl`` names its classes as
``gcmc_re.engine.ReplicaExchangeGCMC`` and so on) loadable. New code should
import `regc`.
"""

import importlib
import sys
import warnings

import regc
from regc import *  # noqa: F401,F403
from regc import __all__, __version__  # noqa: F401

warnings.warn("the package 'gcmc_re' is now called 'regc'; import regc instead",
              DeprecationWarning, stacklevel=2)

_SUBMODULES = ("analysis", "calculators", "driver", "engine", "io", "lammps_backend",
               "plotting", "pressure", "replica", "reservoir", "runner", "state",
               "structures", "systems", "trajectory", "units")

for _name in _SUBMODULES:
    try:
        _module = importlib.import_module(f"regc.{_name}")
    except ImportError:  # an optional dependency is missing; regc.<name> fails the same way
        continue
    sys.modules[f"{__name__}.{_name}"] = _module
    setattr(sys.modules[__name__], _name, _module)
