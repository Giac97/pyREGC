"""
Run the whole structure -> replica-exchange GCMC -> MBAR -> (T, p) phase
diagram pipeline from a single config file (JSON or YAML).

Usage:
    python 09_config_driven_run.py pd111_emt_example.yaml
    gcmc-re-run pd111_emt_example.yaml                  # the same, as installed by pip
    gcmc-re-run pd111_emt_example.yaml --check          # setup + mu ladder only
    gcmc-re-run pd111_emt_example.yaml --analyze-only   # re-analyse saved samples
    gcmc-re-run fe100_h_eam_example.yaml --resume --time-limit 85000   # batch jobs

See pd111_emt_example.yaml (EMT, every key annotated) and
fe100_h_eam_example.yaml (LAMMPS EAM, the setup of example 13, with
checkpoints), and gcmc_re/runner.py's module docstring. Read the Pd
example's own comments before running it: EMT's H parameters give large,
unphysical H uptake, so this demonstrates the pipeline, not real Pd-H
physics. Relative paths (structure file, potential files) are resolved
against the config file's directory, so this runs from any working
directory.
"""

from gcmc_re.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
