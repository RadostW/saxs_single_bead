from saxs_single_bead.form_factors import _single_letter_dict
from saxs_single_bead.form_factors import _three_letter_dict

def test_dicts_coincide():
    assert _single_letter_dict['A'] == _three_letter_dict['ALA']
    
    assert _single_letter_dict['C'] == _three_letter_dict['CYS']
    assert _single_letter_dict['D'] == _three_letter_dict['ASP']
    assert _single_letter_dict['E'] == _three_letter_dict['GLU']
    assert _single_letter_dict['F'] == _three_letter_dict['PHE']
    assert _single_letter_dict['G'] == _three_letter_dict['GLY']
    assert _single_letter_dict['H'] == _three_letter_dict['HIS']
    assert _single_letter_dict['I'] == _three_letter_dict['ILE']

    assert _single_letter_dict['K'] == _three_letter_dict['LYS']
    assert _single_letter_dict['L'] == _three_letter_dict['LEU']
    assert _single_letter_dict['M'] == _three_letter_dict['MET']
    assert _single_letter_dict['N'] == _three_letter_dict['ASN']

    assert _single_letter_dict['P'] == _three_letter_dict['PRO']
    assert _single_letter_dict['Q'] == _three_letter_dict['GLN']
    assert _single_letter_dict['R'] == _three_letter_dict['ARG']
    assert _single_letter_dict['S'] == _three_letter_dict['SER']
    assert _single_letter_dict['T'] == _three_letter_dict['THR']

    assert _single_letter_dict['V'] == _three_letter_dict['VAL']
    assert _single_letter_dict['W'] == _three_letter_dict['TRP']
    assert _single_letter_dict['Y'] == _three_letter_dict['TYR']
                
