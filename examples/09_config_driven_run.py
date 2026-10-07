"""
Run the whole structure -> replica-exchange GCMC -> MBAR -> (T, p) phase
diagram pipeline from a single config file (JSON or YAML).

Usage:
    python 09_config_driven_run.py pd111_emt_example.yaml
    gcmc-re-run pd111_emt_example.yaml          # the same, as installed by pip

See pd111_emt_example.yaml for an annotated example config, and
gcmc_re/runner.py's module docstring for the full schema. Read that
example's own comments before running it: EMT's H parameters give large,
unphysical H uptake, so this demonstrates the pipeline, not real Pd-H
physics. A relative `structure.file` is resolved against the config file's
directory, so this runs from any working directory.
"""

from gcmc_re.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
