import pytest
from saxs_single_bead.form_factors import form_factor

def test_form_factor_values_from_paper():
    assert form_factor('ALA',0.25) == pytest.approx(8.239,rel=0.01)
    assert form_factor('GLU',0.25) == pytest.approx(18.694,rel=0.01)
    assert form_factor('LEU',0.25) == pytest.approx(3.893,rel=0.01)
