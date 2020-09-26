"""
Main graph generator for making molecular graphs.

It uses networkx as graph interface and a mol object from rdkit, ase, pymatgen or similar.

@author: Patrick Reiser,
"""

#Necessary
import networkx as nx
import numpy as np

#Rdkit and methods
try:
    import rdkit
    import rdkit.Chem.AllChem
    MOLGRAPH_RDKIT_AVAILABLE = True
except:
    print("Warning: Rdkit not found for mol class.")



class MolGraph(nx.Graph):
    """Molecular Graph which inherits from networkx graph."""
    
    def __init__(self,**kwargs):
        super(MolGraph, self).__init__(**kwargs)
        
        # Determine available mol libs
        self.mol_libs_avail = []
        if(MOLGRAPH_RDKIT_AVAILABLE == True):
            self.mol_libs_avail.append("rdkit")
        
        # Main mol object to use.
        self.mol = None
        
    
    def mol_from_smiles(self,in_smile):
        """
        Generate mol object from a smile string.

        Args:
            in_smile (str): smile.

        Returns:
            MolObject: Representation of the molecule.

        """ 
        m = rdkit.Chem.MolFromSmiles(in_smile)
        m = rdkit.Chem.AddHs(m)
        self.mol = m
        return self.mol
    
    def mol_from_structure(self,atoms,bondtab):
        pass
    
    def mol_from_geometry(self,atoms,coordinates):
        pass
    
    def conformation(self,mol_lib='rdkit',methods='ETKDG',conf_selection=None):
        pass
    
    
    def _proton_atom(self,molgraph):
        pass
        
    def _label_atom(self,molgraph):
        pass
    
    def make(self,nodes={'proton' : {'class': "proton_atom" } ,
                         'label' : {'class' : "label_atom" } ,
                         },
                  edges = {},
                  state = {}):
        pass
    
    def to_tensor(self):
        pass
        
    
