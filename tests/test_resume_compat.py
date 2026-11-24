"""Resuming runs saved before metadata keys were added (v0.6.0 compatibility)."""
import json

import pytest

from regc.driver import _check_metadata


def _write(tmp_path, meta):
    p = tmp_path / "run_metadata.json"
    p.write_text(json.dumps(meta))
    return str(p)


def test_new_top_level_key_with_default_is_accepted(tmp_path):
    old = {"species": ["Ar"], "n_atoms": 10}
    _check_metadata(_write(tmp_path, old), {**old, "initial_mobile_counts": {}})


def test_new_top_level_key_with_a_real_value_is_refused(tmp_path):
    old = {"species": ["Ar"], "n_atoms": 10}
    with pytest.raises(ValueError, match="initial_mobile_counts"):
        _check_metadata(_write(tmp_path, old), {**old, "initial_mobile_counts": {"Pd": 25}})


def test_new_subsurface_fields_with_defaults_are_accepted(tmp_path):
    sub_old = {"z_cut": 1.0, "z_min": -3.0, "species": ["O"], "ncmc_steps": 10}
    sub_new = {**sub_old, "freeze_switched": False, "site_method": "voronoi",
               "bond_length": None, "site_merge": 0.25, "defensive_fraction": 0.0,
               "partition_gas_zone": True}
    _check_metadata(_write(tmp_path, {"subsurface": sub_old}), {"subsurface": sub_new})
    with pytest.raises(ValueError, match="subsurface"):
        _check_metadata(_write(tmp_path, {"subsurface": sub_old}),
                        {"subsurface": {**sub_new, "freeze_switched": True}})
