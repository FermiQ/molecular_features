"""Functions for molecular properties from rdkit.

Note: All functions are supposed to work out of the box without any dependencies, i.e. do not depend on each other.
"""

import rdkit
import rdkit.Chem.AllChem
import numpy as np

from molreps.methods.geo_npy import coordinates_to_distancematrix,define_adjacency_from_distance


def rdkit_mol_from_atoms_bonds(atoms,bonds,sani=False):
    """
    Convert an atom list and bond list to a rdkit mol class.

    Args:
        atoms (list): List of atoms (N,).
        bonds (list, np.array): Bond list matching atom index. Shape (N,3) or (N,2).
        sani (bool, optional): Whether to sanitize molecule. Defaults to False.

    Returns:
        rdkit.Chem.Mol: Rdkit Mol object. Molecule generated.

    """
    bond_names =  {'AROMATIC': rdkit.Chem.rdchem.BondType.AROMATIC, 'DATIVE': rdkit.Chem.rdchem.BondType.DATIVE, 'DATIVEL': rdkit.Chem.rdchem.BondType.DATIVEL, 'DATIVEONE': rdkit.Chem.rdchem.BondType.DATIVEONE, 'DATIVER': rdkit.Chem.rdchem.BondType.DATIVER, 'DOUBLE': rdkit.Chem.rdchem.BondType.DOUBLE, 'FIVEANDAHALF': rdkit.Chem.rdchem.BondType.FIVEANDAHALF, 'FOURANDAHALF': rdkit.Chem.rdchem.BondType.FOURANDAHALF, 'HEXTUPLE': rdkit.Chem.rdchem.BondType.HEXTUPLE, 'HYDROGEN': rdkit.Chem.rdchem.BondType.HYDROGEN, 'IONIC': rdkit.Chem.rdchem.BondType.IONIC, 'ONEANDAHALF': rdkit.Chem.rdchem.BondType.ONEANDAHALF, 'OTHER': rdkit.Chem.rdchem.BondType.OTHER, 'QUADRUPLE': rdkit.Chem.rdchem.BondType.QUADRUPLE, 'QUINTUPLE': rdkit.Chem.rdchem.BondType.QUINTUPLE, 'SINGLE': rdkit.Chem.rdchem.BondType.SINGLE, 'THREEANDAHALF': rdkit.Chem.rdchem.BondType.THREEANDAHALF, 'THREECENTER': rdkit.Chem.rdchem.BondType.THREECENTER, 'TRIPLE': rdkit.Chem.rdchem.BondType.TRIPLE, 'TWOANDAHALF': rdkit.Chem.rdchem.BondType.TWOANDAHALF, 'UNSPECIFIED': rdkit.Chem.rdchem.BondType.UNSPECIFIED, 'ZERO': rdkit.Chem.rdchem.BondType.ZERO}
    bond_vals = {0: rdkit.Chem.rdchem.BondType.UNSPECIFIED, 1: rdkit.Chem.rdchem.BondType.SINGLE, 2: rdkit.Chem.rdchem.BondType.DOUBLE, 3: rdkit.Chem.rdchem.BondType.TRIPLE, 4: rdkit.Chem.rdchem.BondType.QUADRUPLE, 5: rdkit.Chem.rdchem.BondType.QUINTUPLE, 6: rdkit.Chem.rdchem.BondType.HEXTUPLE, 7: rdkit.Chem.rdchem.BondType.ONEANDAHALF, 8: rdkit.Chem.rdchem.BondType.TWOANDAHALF, 9: rdkit.Chem.rdchem.BondType.THREEANDAHALF, 10: rdkit.Chem.rdchem.BondType.FOURANDAHALF, 11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF, 12: rdkit.Chem.rdchem.BondType.AROMATIC, 13: rdkit.Chem.rdchem.BondType.IONIC, 14: rdkit.Chem.rdchem.BondType.HYDROGEN, 15: rdkit.Chem.rdchem.BondType.THREECENTER, 16: rdkit.Chem.rdchem.BondType.DATIVEONE, 17: rdkit.Chem.rdchem.BondType.DATIVE, 18: rdkit.Chem.rdchem.BondType.DATIVEL, 19: rdkit.Chem.rdchem.BondType.DATIVER, 20: rdkit.Chem.rdchem.BondType.OTHER, 21: rdkit.Chem.rdchem.BondType.ZERO}
    
    mol = rdkit.Chem.RWMol()
    for atm in atoms:
        mol.AddAtom(rdkit.Chem.Atom(atm))
    
    for i in range(len(bonds)):
        if(not mol.GetBondBetweenAtoms(int(bonds[i][0]),int(bonds[i][1])) and int(bonds[i][0]) != int(bonds[i][1])):
            if(len(bonds[i]) == 3):
                bi = bonds[i][2]
                if(isinstance(bi,str)):
                    bond_type = bond_names[bi]
                elif(isinstance(bi,int)):
                    bond_type = bond_vals[bi]  
                else:
                    bond_type = bi #or directly rdkit.Chem.rdchem.BondType
                mol.AddBond(int(bonds[i][0]), int(bonds[i][1]), bond_type)
            else:
                mol.AddBond(int(bonds[i][0]), int(bonds[i][1]))
    
    mol = mol.GetMol()
    
    if(sani == True):
        rdkit.Chem.SanitizeMol(mol)
    
    return mol


def rdkit_add_conformer(mol,coords,assignID = False):
    """
    Add a confromer to a mol object.

    Args:
        mol (rdkit.Chem.Mol): Mol object to add conformer.
        coords (array): Array of coordinates of shape (N,3).
        assignID (bool,int): To assing conformer iD. Default is False.

    Returns:
        rdkit.Chem.Mol: Mol Object with added conformer.

    """
    conf = rdkit.Chem.Conformer(len(coords))
    for i in range(len(coords)):
        conf.SetAtomPosition(i,[coords[i,0],coords[i,1],coords[i,2]])
    mol.AddConformer(conf,assignId=assignID)
    return mol



def rdkit_make_mol_from_structure(atoms,bondlist,coordinates=None):
    """
    Make a mol object with 

    Args:
        atoms (list): List of atoms (N,).
        bondlist (list, np.array): Bond list matching atom index. Shape (N,3) or (N,2).
                                   For (N,3) last entry can give bond order.
        coordinates (np.array, optional): Coordinates of Atoms of shape (N,3). Defaults to None.

    Returns:
        rdkit.Chem.Mol: Mol Object with added conformer.
    """
    mol = rdkit_mol_from_atoms_bonds(atoms,bondlist)
    if(coordinates is not None):
        rdkit_add_conformer(mol,coordinates)
    return mol
    


def rdkit_atom_list(mol,key,method,args={}):
    """
     Make a list of atoms with atomic information from rdkit.mol.

    Args:
        mol (rdkit.Chem.Mol): Mol object to get information from.
        key (str): Key of property to put in list.
        method (func): Class member method for rdkit.Chem.rdchem.Atom.
        args (dict, optional): Optinal arguments for class method. Defaults to {}.

    Returns:
        list: Atomlist that can be used in a networkx graph of shape [(i, {key: property})]

    """
        #lenats = len(mol.GetAtoms())
    G = []
    for i,atm in enumerate(mol.GetAtoms()):
        attr = {key: method(atm,**args)}
        G.append((i, attr))
    return G




def rdkit_bond_list(mol,key,method,args={},trafo=int):
    """
    Make a list of bonds with bond-type information from rdkit.mol.

    Args:
        mol (rdkit.Chem.Mol): Mol object to get information from.
        key (str, optional): Key of property to put in list.
        method (func): Class member method for rdkit.Chem.rdchem.Bond.
        args (dict, optional): Optinal arguments for class method. Defaults to {}.
        trafo (func): Casting or trafo funciton. Default is int.

    Returns:
        list: Bondlist that can be used in a networkx graph of shape [(i,j, {key: property})]

    """
    #lenats = len(mol.GetAtoms())
    #out = np.zeros((lenats,lenats))
    bonds = list(mol.GetBonds())
    G=[]
    for x in bonds:
        attr = {key : trafo(method(x,**args))}
        G.append((x.GetBeginAtomIdx(), x.GetEndAtomIdx(), attr))
        G.append((x.GetEndAtomIdx(),x.GetBeginAtomIdx(), attr) )
    return G


def rdkit_bond_distance_list(mol,key,conf_selection=0,methods='ETKDG',seed =0xf00d, max_distance =np.inf,max_partners=np.inf,bonds_only=True):
    """
    Generate bond list with distance information from rdkit.mol.

    Args:
        mol (rdkit.Chem.Mol): Mol object to get information from.
        key (str, optional): Key of property to put in list.
        conf_selection (int, optional): Select a conformer. Defaults to 0.
        methods (string, optional): Method to generate conformer if none exist. Defaults to 'ETKDG'.
        seed (TYPE, optional): Random seed. Defaults to 0xf00d.
        max_distance (value, optional): Maximum distance to allow bonds. Defaults to np.inf.
        max_partners (value, optional): Maximum nieghbour to allow bonds. Defaults to np.inf.
        bonds_only (TYPE, optional): Only take actual chemical bonds as bonds. Defaults to True.

    Returns:
        list: Bondlist that can be used in a networkx graph of shape [(i,j, {key: distance})].

    """
    if(not len(mol.GetConformers()) > 0 ):
        retval = rdkit.Chem.AllChem.EmbedMolecule(mol,randomSeed=seed)
    dist_mat = coordinates_to_distancematrix(np.array(mol.GetConformers()[conf_selection].GetPositions()))
    adj_dist, idx_dist = define_adjacency_from_distance(dist_mat,max_distance,max_partners,exclusive=True)
    if(bonds_only == True):
        bonds = rdkit_bond_list(mol,key,rdkit.Chem.rdchem.Bond.GetBondType)
        for x in bonds:
            # Replace Bond-type by distance
            x[2][key] = dist_mat[x[0],x[1]]
        return bonds
    else:    
        out_list = []
        for i in range(len(idx_dist)):
            out_list.append((idx_dist[i][0],idx_dist[i][1],{ key : dist_mat[idx_dist[i][0],idx_dist[i][1]]}))    
    return out_list
