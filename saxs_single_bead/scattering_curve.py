import saxs_single_bead.form_factors
import numpy as np


def scattering_curve(
    residue_codes, residue_locations, minimal_q=0.0, maximal_q=0.5, points=20
):
    """
    Computes scattering curve from `residue_codes` and `residue_locations` `N` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G")
    residue_locations: np.array(float)
        Rectangular array with size `N` by `3` of locations of `C_alpha` atoms (one per residue)
    minimal_q: float, optional
        Minimal scattering vector, default `0.0`, units: Angstrom^(-1)
    maximal_q: float, optional
        Maximal scattering vector, default `0.5`, units: Angstrom^(-1)
    points: int, optional
        Number of points int the plot, default `20.`

    Returns
    -------
    (np.array(float),np.array(float))
        A tuple of numpy arrays containing values of `q` and `I(q)` respectively.
    """
    distance_matrix = np.sqrt(
        np.sum(
            (residue_locations[np.newaxis, :, :] - residue_locations[:, np.newaxis, :])
            ** 2,
            axis=-1,
        )
    )

    q_values = np.linspace(minimal_q, maximal_q, points)
    I_values = np.zeros_like(q_values)

    for i, q in enumerate(q_values):
        form_factors = np.array(
            [
                saxs_single_bead.form_factors.form_factor(code, q)
                for code in residue_codes
            ]
        )
        I_values[i] = np.sum(
            form_factors[:, np.newaxis]
            * form_factors[np.newaxis, :]
            * np.sinc(distance_matrix * q / np.pi),
            axis=(0, 1),
        )

    return (q_values, I_values)


def scattering_curve_ensemble(
    residue_codes, residue_locations, minimal_q=0.0, maximal_q=0.5, points=20
):
    """
    Computes average scattering curve from `residue_codes` and `residue_locations` `M` by `N` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G")
    residue_locations: np.array(float)
        Rectangular array with size `M` by `N` by `3` of locations of `C_alpha` atoms (one per residue)
    minimal_q: float, optional
        Minimal scattering vector, default `0.0`, units: Angstrom^(-1)
    maximal_q: float, optional
        Maximal scattering vector, default `0.5`, units: Angstrom^(-1)
    points: int, optional
        Number of points int the plot, default `20.`

    Returns
    -------
    (np.array(float),np.array(float))
        A tuple of numpy arrays containing values of `q` and `I(q)` respectively.
    """
    distance_matrices = np.sqrt(
        np.sum(
            (
                residue_locations[:, np.newaxis, :, :]
                - residue_locations[:, :, np.newaxis, :]
            )
            ** 2,
            axis=-1,
        )
    )

    q_values = np.linspace(minimal_q, maximal_q, points)
    I_values = np.zeros_like(q_values)

    for i, q in enumerate(q_values):
        form_factors = np.array(
            [
                saxs_single_bead.form_factors.form_factor(code, q)
                for code in residue_codes
            ]
        )

        I_values[i] = np.sum(
            form_factors[:, np.newaxis]
            * form_factors[np.newaxis, :]
            * np.mean(np.sinc(distance_matrices * q / np.pi), axis=0),
            axis=(0, 1),
        )

    return (q_values, I_values)
