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
        Rank 3 array with size `M` by `N` by `3` of locations of `C_alpha` atoms (one per residue)
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


def scattering_curve_two_bead(
    residue_codes, residue_locations, minimal_q=0.0, maximal_q=0.5, points=20
):
    """
    Computes scattering curve from `residue_codes` and `residue_locations` `N` by `2` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G")
    residue_locations: np.array(float)
        Array with shape `N` by `2` by `3` of locations of `C_alpha` atoms and `COE` (one per residue)
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

    residue_codes = (["BB"] * len(residue_codes)) + residue_codes
    residue_locations = np.vstack(
        [residue_locations[:, 0], residue_locations[:, 1]]
    )  # combine bead lists to one long list

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
                saxs_single_bead.form_factors.form_factor_two_bead(code, q)
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


def scattering_curve_two_bead_ensemble(
    residue_codes, residue_locations, minimal_q=0.0, maximal_q=0.5, points=20
):
    """
    Computes average scattering curve from `residue_codes` and `residue_locations` `M` by `N` by `2` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G")
    residue_locations: np.array(float)
        Rank 3 array with size `M` by `N` by `2` by `3` of locations of `C_alpha` atoms and `COE`
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

    residue_codes = (["BB"] * len(residue_codes)) + residue_codes
    (M, N, _, _) = residue_locations.shape

    residue_locations = np.transpose(residue_locations, axes=(0, 2, 1, 3)).reshape(
        M, 2 * N, 3
    )  # combine bead lists to one long list

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
                saxs_single_bead.form_factors.form_factor_two_bead(code, q)
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


def _form_factor_one_two(code, q, model_type):
    if model_type == 1:
        return saxs_single_bead.form_factors.form_factor(code, q)
    elif model_type == 2:
        return saxs_single_bead.form_factors.form_factor_two_bead(code, q)
    else:
        raise ValueError(
            f"{model_type} is not a valid model type. '1' or '2' are allowed."
        )


def scattering_curve_one_two_blend(
    residue_codes,
    residue_model,
    residue_locations,
    minimal_q=0.0,
    maximal_q=0.5,
    points=20,
):
    """
    Computes scattering curve from `residue_codes` and `residue_locations` `N` by `2` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G") or "BB"
        for two bead backbone locaitons.
    residue_model: list(int)
        List describing which model is used `1` for single bead and `2` for two bead.
    residue_locations: np.array(float)
        Array with shape `N` by `3` of locations of `C_alpha` atoms and `COE` (one per residue).
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
                _form_factor_one_two(code, q, model_type)
                for (code, model_type) in zip(residue_codes, residue_model)
            ]
        )
        I_values[i] = np.sum(
            form_factors[:, np.newaxis]
            * form_factors[np.newaxis, :]
            * np.sinc(distance_matrix * q / np.pi),
            axis=(0, 1),
        )

    return (q_values, I_values)


def scattering_curve_one_two_blend_ensemble(
    residue_codes,
    residue_model,
    residue_locations,
    minimal_q=0.0,
    maximal_q=0.5,
    points=20,
):
    """
    Computes average scattering curve from `residue_codes` and `residue_locations` `M` by `N` by `2` by `3` array.


    Parameters
    ----------
    residue_codes: list(string)
        List of residues of length `N`. Can be 3 letter codes (such as "GLY") or single letter codes (such as "G") or "BB"
        for two bead backbone locaitons.
    residue_model: list(int)
        List describing which model is used `1` for single bead and `2` for two bead.
    residue_locations: np.array(float)
        Rank 3 array with size `M` by `N` by `2` by `3` of locations of `C_alpha` atoms and `COE`
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
                _form_factor_one_two(code, q, model_type)
                for (code, model_type) in zip(residue_codes, residue_model)
            ]
        )

        I_values[i] = np.sum(
            form_factors[:, np.newaxis]
            * form_factors[np.newaxis, :]
            * np.mean(np.sinc(distance_matrices * q / np.pi), axis=0),
            axis=(0, 1),
        )

    return (q_values, I_values)
