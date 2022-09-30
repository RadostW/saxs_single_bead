import numpy as np


def _get_spherical():
    """
    Return vector distributed uniformly on a sphere
    """
    x = 2.0 * np.random.uniform() - 1.0
    phi = 2.0 * np.random.uniform() - 1.0
    scale = np.sqrt(1.0 - x * x)
    y = scale * np.sin(np.pi * phi)
    z = scale * np.cos(np.pi * phi)
    return np.array([x, y, z])


def _normalize(vector):
    """
    Return normalized vector
    """
    return vector / np.sqrt(np.sum(vector ** 2, axis=-1))


def replace_bead(
    conglomerate, conglomerate_attachment_point, locations, sizes, bead_id
):
    """
    Replace single bead in chain defined by locations by a conglomerte

    Parameters
    ----------
    conglomarate : np.array
        `N` by `3` array of locations of beads within the conglomerate
    attachment_point : np.array or tuple(np.array)
        vector of length `3` describing attachment point of the conglomerate to the chain or tuple of two such attachment points
    locations : np.array
        `M` by `3` array of locations of beads in chain
    sizes : np.array
        vector of length `M` of sizes of beads in chain
    bead_id : int
        index of bead to be replaced

    Returns
    -------
    np.array
        locations of chain with bead replaced with conglomerate
    """
    assert isinstance(bead_id, int)

    locations_rest = np.delete(locations, bead_id, axis=0)
    if bead_id == 0:
        chain_attachment_point = (locations[0] * sizes[1] + locations[1] * sizes[0]) / (
            sizes[0] + sizes[1]
        )
        chain_attachment_vector = locations[1] - locations[0]
        chain_attachment_vector = _normalize(chain_attachment_vector)
        
    elif bead_id == -1 or bead_id == (len(sizes) - 1):
        chain_attachment_point = (locations[-1] * sizes[-2] + locations[-2] * sizes[-1]) / (
            sizes[-1] + sizes[-2]
        )
        chain_attachment_vector = locations[-1] - locations[-2]
        chain_attachment_vector = _normalize(chain_attachment_vector)
    else:
        raise NotImplementedError

    # Conglomerate direction vector
    conglomerate_centre = np.mean(conglomerate, axis=0)
    conglomerate_extent = np.sqrt(
        np.sum((conglomerate_centre - conglomerate_attachment_point) ** 2, axis=-1)
    )
    conglomerate_direction_vector = _normalize(
        conglomerate_centre - conglomerate_attachment_point
    )

    # Align centre of mass
    centre_of_mass = np.mean(conglomerate, axis=0)
    conglomerate_centered = conglomerate - centre_of_mass

    # Align conglomerate direction vector with x-axis
    # TODO replace _get_spherical() with something faster
    tmp_a = _normalize(np.cross(_get_spherical(), conglomerate_direction_vector))
    rotation_a = np.transpose(
        np.array(
            [
                conglomerate_direction_vector,
                tmp_a,
                np.cross(conglomerate_direction_vector, tmp_a),
            ]
        )
    )
    conglomerate_axis_aligned = conglomerate_centered @ rotation_a

    # Align x direction vector with chain direction vector
    tmp_b = _normalize(np.cross(_get_spherical(), chain_attachment_vector))
    rotation_b = np.transpose(
        np.array(
            [
                chain_attachment_vector,
                tmp_b,
                np.cross(chain_attachment_vector, tmp_b),
            ]
        )
    )
    conglomerate_direction_aligned = conglomerate_axis_aligned @ np.transpose(
        rotation_b
    )

    # Shift conglomerate centre to right location
    conglomerate_shifted = (
        conglomerate_direction_aligned
        + chain_attachment_point
        + chain_attachment_vector * conglomerate_extent
    )

    return np.vstack((locations_rest, conglomerate_shifted))


if __name__ == "__main__":

    import sarw_spheres
    import matplotlib.pyplot as plt

    sizes = np.array([30.0] + [1.0] * 60)
    beads = sarw_spheres.generateChain(sizes)
    beads = beads + np.array([5.0, 7.0, 10.0])
    domain = _normalize(np.array([1.0, 2.0, 3.0])) * np.linspace(
        0.0, 60.0, num=30
    ).reshape(-1, 1)

    beads_with_domain = replace_bead(domain, domain[-1], beads, sizes, bead_id=0)

    ax = plt.axes(projection="3d")
    (x, y, z) = (
        beads_with_domain[:, 0],
        beads_with_domain[:, 1],
        beads_with_domain[:, 2],
    )
    ax.scatter(x, y, z, c=z, cmap="viridis", linewidth=1.0)
    plt.show()
