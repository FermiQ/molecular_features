"""Functions for openbabel.

Specific functions for molecular features.
"""

from openbabel import openbabel


def ob_build_xyz_string_from_list(atoms, coords):
    """
    Make a xyz string from atom and coordinate list.

    Args:
        atoms (list): Atom list of type ['H','C','H',...].
        coords (array): Coordinate list of shape (N,3).

    Returns:
        str: XYZ string for a xyz file.
    """
    xyz_str = str(int(len(atoms))) + "\n"
    for i in range(len(atoms)):
        xyz_str = xyz_str + "\n" + atoms[i] + " " + str(coords[i][0]) + " " + str(coords[i][1]) + " " + str(coords[i][2])
    return xyz_str


def ob_readXYZs(filename):
    """
    Ready stacked xyz's from file.

    Args:
        filename (str): Filepath.

    Returns:
        tuple: [elements,coords]
        
        - elements (list): Coordinate list of shape (Molecules,Atoms,3).
        - coords (list): Molecule list of shape (Molecules,Atoms).
    """
    infile = open(filename, "r")
    coords = [[]]
    elements = [[]]
    for line in infile.readlines():
        if len(line.split()) == 1 and len(coords[-1]) != 0:
            coords.append([])
            elements.append([])
        elif len(line.split()) == 4:
            elements[-1].append(line.split()[0].capitalize())
            coords[-1].append([float(line.split()[1]), float(line.split()[2]), float(line.split()[3])])
    infile.close()
    return elements, coords


def ob_get_bond_table_from_coordinates(atoms, coords):
    """
    Get bond order information by reading a xyz string.
    
    The order of atoms in the list should be the same as output.
    But output is completely generated from OBMol.

    Args:
        atoms (list): Atom list of type ['H','C','H',...].
        coords (array): Coordinate list of shape (N,3).

    Returns:
        tuple: ob_ats,ob_proton,bonds,ob_coord
        
        - ob_ats (list): Atom list of type ['H','Car','O2',...] of OBType i.e. with aromatic and state info.
        - ob_proton (list): Atomic Number of atoms as list.
        - bonds (list): Bond information of shape [i,j,order].
        - ob_coord (list): Coordinates as list.
    """
    ob_conversion = openbabel.OBConversion()
    # ob_conversion.SetInAndOutFormats("xyz", "pdb")
    ob_conversion.SetInFormat("xyz")

    # Make xy string
    xyz_str = str(int(len(atoms))) + "\n"
    for i in range(len(atoms)):
        xyz_str = xyz_str + "\n" + atoms[i] + " " + str(coords[i][0]) + " " + str(coords[i][1]) + " " + str(
            coords[i][2])

    mol = openbabel.OBMol()
    ob_conversion.ReadString(mol, xyz_str)
    # print(xyz_str)

    bonds = []
    for i in range(mol.NumBonds()):
        bnd = mol.GetBondById(i)
        # bnd = mol.GetBond(i)
        if bnd is not None:
            bonds.append([bnd.GetBeginAtomIdx() - 1, bnd.GetEndAtomIdx() - 1, bnd.GetBondOrder()])
    ob_ats = []
    ob_coord = []
    ob_proton = []
    for i in range(mol.NumAtoms()):
        ats = mol.GetAtomById(i)
        # ats = mol.GetAtom(i+1)
        ob_ats.append(ats.GetType())
        ob_proton.append(ats.GetAtomicNum())
        ob_coord.append([ats.GetVector().GetX(), ats.GetVector().GetY(), ats.GetVector().GetZ()])

    # outMDL = ob_conversion.WriteString(mol)
    return ob_ats, ob_proton, bonds, ob_coord

# a,c = ob_readXYZs("E:\\Benutzer\\Patrick\\PostDoc\\Projects ML\\dynamical_disorder\\data\\DPEPO\\dump_4984000.xyz")
# ou1,ou2,ou3,ou4 =ob_get_bond_table_from_coordinates(a[0],c[0])
