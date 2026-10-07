"""units.py against textbook reference values."""

import pytest

from gcmc_re import kelvin_to_eV, thermal_wavelength_angstrom


def test_kelvin_to_eV_matches_textbook_value():
    # the standard "kT ~ 25.85 meV at 300 K" fact
    assert kelvin_to_eV(300.0) == pytest.approx(0.025851, abs=2e-6)


def test_heavier_atom_has_shorter_wavelength():
    lam_H = thermal_wavelength_angstrom(300.0, mass_amu=1.008)
    lam_Au = thermal_wavelength_angstrom(300.0, mass_amu=196.97)
    assert lam_Au < lam_H  # heavier -> shorter thermal wavelength, same T
    assert 0.5 < lam_H < 2.0  # H at room T is close to (coincidentally) 1 Angstrom
