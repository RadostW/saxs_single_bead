# form_factors.py
# raw data and helper functions for working with tabulated form factors

import numpy as np
import json


"""
Source: 
Accurate optimization of amino acid form factors for computing 
small-angle X-ray scattering intensity of atomistic protein structures
D. Tong, S. Yang and L. Lu
J. Appl. Cryst. (2016)

Supplemental data, tabulated form factors for single bead approximation located at C_alpha sites.
"""
from saxs_single_bead.form_factors_raw import _raw_data

"""
Source: ibid.

Supplemental data, tabulated form factors for two bead approximation located at C_alpha and COE sites.
COE - centre of electrons (like centre of mass but you count only protons in the nucleus)
"""
from saxs_single_bead.form_factors_raw import _raw_data_two_bead


def _raw_form_factor(residue_id, q):
    """
    Returns linearly interpolated form factor for a given `residue_id`.

    Parameters
    ----------
    residue_id: int
        Residue id
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """
    if residue_id not in range(0, 20):
        raise IndexError("Wrong residue id value (has to be 0-19)")
    return np.interp(q, _raw_data[0, :], _raw_data[residue_id + 1, :])

def _raw_form_factor_two_bead(residue_id, q):
    """
    Returns linearly interpolated form factor for a given `residue_id`.

    Parameters
    ----------
    residue_id: int
        Residue id
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """
    if residue_id == -1:
        return np.interp(q, _raw_data[0, :], _raw_data_two_bead[1, :])
        
    if residue_id not in range(0, 20):
        raise IndexError("Wrong residue id value (has to be 0-19) or -1 for backbone")        
        
    return np.interp(q, _raw_data[0, :], _raw_data_two_bead[residue_id + 2, :])


_three_letter_dict = {
    "ALA": 0,
    "CYS": 1,
    "ASP": 2,
    "GLU": 3,
    "PHE": 4,
    "GLY": 5,
    "HIS": 6,
    "ILE": 7,
    "LYS": 8,
    "LEU": 9,
    "MET": 10,
    "ASN": 11,
    "PRO": 12,
    "GLN": 13,
    "ARG": 14,
    "SER": 15,
    "THR": 16,
    "VAL": 17,
    "TRP": 18,
    "TYR": 19,
}
_single_letter_dict = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}


def form_factor(residue_name, q):
    """
    Returns linearly interpolated form factor for a given `residue_name`.

    Parameters
    ----------
    residue_name: string
        Name of the residue. Either in single letter format or three letter format.
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """

    if len(residue_name) == 3:
        return _raw_form_factor(_three_letter_dict[residue_name], q)
    elif len(residue_name) == 1:
        return _raw_form_factor(_single_letter_dict[residue_name], q)
    else:
        raise IndexError("Wrong length of residue name (has to be 3 or 1)")
        
        
def form_factor_two_bead(residue_name, q):
    """
    Returns linearly interpolated form factor for a given `residue_name`.

    Parameters
    ----------
    residue_name: string
        Name of the residue. Either in single letter format or three letter format.
    q: float
        Scattering vector, units: Angstrom^(-1)

    Returns
    -------
    float:
        Form factor value.
    """

    if residue_name == 'BB':
        return _raw_form_factor_two_bead(-1, q)
    if len(residue_name) == 3:
        return _raw_form_factor_two_bead(_three_letter_dict[residue_name], q)
    elif len(residue_name) == 1:
        return _raw_form_factor_two_bead(_single_letter_dict[residue_name], q)
    else:
        raise IndexError("Wrong length of residue name (has to be 3 or 1)")
