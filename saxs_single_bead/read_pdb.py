import numpy as np
import sys


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

    residues = dict()
    for line in lines:
        if line[0:4] == "ATOM":
            if "CA" in line:
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if resid in residues:
                    print(f"Warning: multiple locations for residue {resid}", file=sys.stderr)
                residues[resid] = [x, y, z]
        elif line[0:3] == "END":
            break

    residues_list = list()
    for i in sorted(residues.keys()):
        residues_list.append(residues[i])

    return np.array(residues_list)


if __name__ == "__main__":
    shape = read_c_alpha_pdb("test.pdb")
