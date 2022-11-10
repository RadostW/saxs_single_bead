import numpy as np
import sys

_masses_dict = {" H": 1, " C": 12, " N": 14, " O": 16, " S": 32}

_electrons_dict = {" H": 1, " C": 6, " N": 7, " O": 8, " S": 16}


def element_mass(element_name):
    return _masses_dict[element_name]

def element_electrons(element_name):
    return _electrons_dict[element_name]

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
                    print(
                        f"Warning: multiple locations for residue {resid}",
                        file=sys.stderr,
                    )
                residues[resid] = [x, y, z]
        elif line[0:6] == "ENDMDL":
            # print("Warning: loaded one model of possibly many",file=sys.stderr)
            break
        elif line[0:3] == "END":
            break

    residues_list = list()
    for i in sorted(residues.keys()):
        residues_list.append(residues[i])

    return np.array(residues_list)


def residue_coe(residue_dict):
    """
    Determines centre of electrons of a residue described by `residue_dict`

    Parameters
    ----------
    residue_dict: dict(list())
        Dict containing length `4` lists containing atom locations and masses [x,y,z,m]

    Returns
    -------
    np.array
         Numpy array of length `3` with location of centre of mass

    """
            
    total_mass = 0.0
    accumulated_location = np.array([0.0, 0.0, 0.0])
    for (k, v) in residue_dict.items():
        total_mass += v[3]
        accumulated_location += v[3] * np.array(v[:3])

    return accumulated_location / total_mass


def read_backbone_and_coe_pdb(filename):
    """
    Reads a .pdb file into np.array

    Parameters
    ----------
    filename: string
        Path to file to be read

    Returns
    -------
    np.array
        Numpy array of shape `N` by `2` by `3` containing locations of
        backbone centres and COE of sidechain of each residue (in Angstroms)
    """

    with open(filename, encoding="utf-8") as f:
        contents = f.read()

    lines = contents.splitlines()
    
    backboneroles = [' N  ',' CA ',' C  ',' O  ']

    ca_atoms = dict()
    c_residues = dict()
    
    sidechain_atoms = dict()
    backbone_atoms = dict()
    
    for line in lines:
        if line[0:4] == "ATOM":
            resid = int(line[22:26])
            atomid = int(line[6:11])
            atomrole = line[12:16] # for example "CA "
            elementid = line[76:78]  # two letter code, right justified
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            if ' CA ' == atomrole:
                if resid in ca_atoms:
                    print(
                        f"Warning: multiple locations for residue {resid}",
                        file=sys.stderr,
                    )
                ca_atoms[resid] = [x, y, z]

            if resid not in sidechain_atoms:
                sidechain_atoms[resid] = dict()
            if resid not in backbone_atoms:
                backbone_atoms[resid] = dict()


            if atomrole not in backboneroles:
                if atomid in sidechain_atoms[resid]:
                    print(f"Warning: multiple locations for atom {atomid}", file=sys.stderr)
                sidechain_atoms[resid][atomid] = [x, y, z, element_electrons(elementid)]
            else: # atom in backbone
                if atomid in backbone_atoms[resid]:
                    print(f"Warning: multiple locations for atom {atomid}", file=sys.stderr)
                backbone_atoms[resid][atomid] = [x, y, z, element_electrons(elementid)]
                
        elif line[0:6] == "ENDMDL":
            print("Warning: loaded one model of possibly many", file=sys.stderr)
            break
        elif line[0:3] == "END":
            break

    bb_residues_list = list()
    for i in sorted(ca_atoms.keys()):
        bb_residues_list.append(residue_coe(backbone_atoms[i]))
        
    coe_residues_list = list()
    for i in sorted(ca_atoms.keys()):
        if len(sidechain_atoms[i]) != 0: #all but glycine        
            coe_residues_list.append(residue_coe(sidechain_atoms[i]))
        else:
            coe_residues_list.append(ca_atoms[i]) #glycines sidechain is hydrogen atom on C_alpha

    return np.transpose(np.array([bb_residues_list, coe_residues_list]), axes=(1, 0, 2))


if __name__ == "__main__":
    shape = read_c_alpha_pdb("test.pdb")
