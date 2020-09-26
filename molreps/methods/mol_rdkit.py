"""
Functions for molecular properties from rdkit.

Note: All functions are supposed to work out of the box without any dependencies, i.e. do not depend on each other.

@author: Patrick Reiser,
"""

import rdkit
import rdkit.Chem.AllChem


def rdkit_mol_from_atoms_bonds(atoms,bonds,sani=False):
    """
    Convert an atom list and bond list to a rdkit mol class.

    Args:
        atoms (list): List of atoms.
        bonds (list, np.array): Bond list matching atom index.
        sani (bool, optional): Whether to sanitize molecule. Defaults to False.

    Returns:
        mol (rdkit.Chem.Mol): Rdkit Mol object.

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
                    bond_type = bi
                mol.AddBond(int(bonds[i][0]), int(bonds[i][1]), bond_type)
            else:
                mol.AddBond(int(bonds[i][0]), int(bonds[i][1]))
    
    mol = mol.GetMol()
    
    if(sani == True):
        rdkit.Chem.SanitizeMol(mol)
    
    return mol


def rdkit_networkx_add_proton(G,mol,attr_name):
    pass