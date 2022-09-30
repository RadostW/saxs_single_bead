import numpy as np


def read_c_alpha_pdb(filename):
    """
    Reads a .pdb file into np.array

    Parameters
    ----------
    filename: string
        Path to file to be read

    Returns
    -------
    np.array
        Numpy array of shape `N` by `3` containing locations of C_alpha
        atoms (in Angstroms)
    """

    with open(filename, encoding="utf-8") as f:
        contents = f.read()

    lines = contents.splitlines()

    residues = list()
    for line in lines:
        if line[0:4] == "ATOM":
            if "CA" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residues.append([x, y, z])
        elif line[0:3] == "END":
            break

    return np.array(residues)


if __name__ == "__main__":
    shape = read_c_alpha_pdb("test.pdb")
