"""
The ``regc`` command: one entry point for the tools.

    regc gui [WORKSPACE]            desktop GUI (build, run, monitor); --web for the browser one
    regc run CONFIG [--resume ...]  same as regc-run
    regc monitor DIR [...]          same as regc-monitor
    regc reevaluate CONFIG POT ...  same as regc-reevaluate
"""

from __future__ import annotations

import sys
from typing import Optional

COMMANDS = {
    "gui": ("regc.qtgui", "main", "desktop GUI to build, run and monitor simulations"),
    "run": ("regc.runner", "main", "run a config (regc-run)"),
    "monitor": ("regc.monitor", "main", "live web monitor of output directories (regc-monitor)"),
    "reevaluate": ("regc.runner", "reevaluate_main",
                   "re-evaluate recorded configurations with another potential (regc-reevaluate)"),
}


def main(argv: Optional[list] = None) -> int:
    """
    Dispatch ``regc <command> [arguments]``.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Default None: ``sys.argv[1:]``.

    Returns
    -------
    int
        The command's exit code (2 for an unknown command).
    """
    import importlib
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print("usage: regc <command> [arguments]\n\ncommands:")
        for name, (_, _, desc) in COMMANDS.items():
            print(f"  {name:11s} {desc}")
        print("\n'regc <command> -h' shows the command's options.")
        return 0 if argv else 2
    if argv[0] not in COMMANDS:
        print(f"regc: unknown command {argv[0]!r} (one of {', '.join(COMMANDS)})", file=sys.stderr)
        return 2
    module, func, _ = COMMANDS[argv[0]]
    return int(getattr(importlib.import_module(module), func)(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
