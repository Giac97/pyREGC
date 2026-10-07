"""reservoir.py: self-consistent D0 calculation and the mu_atom <-> p round-trip."""

import pytest

ase = pytest.importorskip("ase", reason="pip install ase to run this test")
from ase.calculators.lj import LennardJones as ASELennardJones  # noqa: E402

from gcmc_re import (
    H2_BOND_LENGTH_ANGSTROM,
    H2_MASS_AMU,
    H2_VIBRATIONAL_WAVENUMBER_CM,
    diatomic_dissociation_energy_eV,
    mu_atomic_from_diatomic_reservoir_eV,
    pressure_from_atomic_mu_reservoir_pascal,
)


@pytest.mark.parametrize("e_atom", [0.0, 3.21])
def test_mu_p_round_trip(e_atom):
    T_K, D0 = 400.0, 4.92
    mu_target = -1.5
    p = pressure_from_atomic_mu_reservoir_pascal(
        T_K, mu_target, D0, H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM, H2_VIBRATIONAL_WAVENUMBER_CM,
        e_atom_eV=e_atom)
    mu_back = mu_atomic_from_diatomic_reservoir_eV(
        T_K, p, D0, H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM, H2_VIBRATIONAL_WAVENUMBER_CM,
        e_atom_eV=e_atom)
    assert mu_back == pytest.approx(mu_target, rel=1e-10)


def test_mu_atom_is_on_the_calculators_energy_scale():
    """GCMC sees an isolated atom at energy e_atom (not 0), so mu_atom must be
    half the molecule's chemical potential measured from the calculator's own
    molecular energy: mu_atom = (1/2) [mu_mol_rel + E_min + ZPE]. Equivalent
    to (1/2)(mu_mol_rel - D0) + e_atom; dropping e_atom shifts it by e_atom."""
    from ase.calculators.emt import EMT

    from gcmc_re import diatomic_gas_mu_eV

    info = diatomic_dissociation_energy_eV(EMT(), "H")
    assert info["e_atom_eV"] == pytest.approx(3.21, abs=0.01)  # EMT's H is far from 0
    T_K, p = 400.0, 1.0e5
    mu_mol_rel = diatomic_gas_mu_eV(T_K, p, H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM,
                                    H2_VIBRATIONAL_WAVENUMBER_CM)
    expected = 0.5 * (mu_mol_rel + info["e_min_eV"] + info["zpe_eV"])
    mu_atom = mu_atomic_from_diatomic_reservoir_eV(
        T_K, p, info["D0_eV"], H2_MASS_AMU, H2_BOND_LENGTH_ANGSTROM,
        H2_VIBRATIONAL_WAVENUMBER_CM, e_atom_eV=info["e_atom_eV"])
    assert mu_atom == pytest.approx(expected, abs=1e-10)


def test_diatomic_dissociation_energy_sane_for_lj():
    """A pairwise LJ 'molecule' should have D0 close to the well depth
    epsilon minus its zero-point energy -- sanity check the machinery
    against a potential whose answer we know analytically, independent of
    EMT's own (disclaimed) H parameters."""
    sigma, epsilon = 1.2, 0.8
    calc = ASELennardJones(sigma=sigma, epsilon=epsilon, rc=6.0 * sigma, smooth=False, shift=True)
    result = diatomic_dissociation_energy_eV(calc, "H", mass_amu=1.008,
                                              r_scan=[x * 0.02 + 1.0 for x in range(60)])
    # LJ minimum sits at r = 2^(1/6) sigma, with well depth exactly epsilon
    r_min_expected = 2 ** (1.0 / 6.0) * sigma
    assert result["r0_angstrom"] == pytest.approx(r_min_expected, abs=0.01)
    assert result["De_eV"] == pytest.approx(epsilon, abs=1e-3)
    assert 0.0 < result["zpe_eV"] < result["De_eV"]
    assert result["D0_eV"] == pytest.approx(result["De_eV"] - result["zpe_eV"], abs=1e-8)


def test_d0_independent_of_scan_grid():
    """The minimum is refined off the scan grid, so different scans agree."""
    from ase.calculators.emt import EMT

    a = diatomic_dissociation_energy_eV(EMT(), "H", r_scan=[x * 0.0356 + 0.4 for x in range(60)])
    b = diatomic_dissociation_energy_eV(EMT(), "H", r_scan=[x * 0.0329 + 0.4 for x in range(80)])
    assert a["r0_angstrom"] == pytest.approx(b["r0_angstrom"], abs=1e-5)
    assert a["D0_eV"] == pytest.approx(b["D0_eV"], abs=1e-5)
