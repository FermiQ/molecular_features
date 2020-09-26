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
    
    
    def _make_state(self,key,propy,args):
        pass
    
    def _make_edges(self,key,propy,args):
        if(propy=="bond_type"):
            lenats = len(self.mol.GetAtoms())
            out = np.zeros((lenats,lenats))
            bonds = list(self.mol.GetBonds())
            for x in bonds:
                attr = {key : int(x.GetBondType())}
                self.add_edge(x.GetBeginAtomIdx(), x.GetEndAtomIdx(), **attr) 
                self.add_edge(x.GetEndAtomIdx(),x.GetBeginAtomIdx(), **attr) 

        
    def _make_nodes(self,key,propy,args):
        if(propy=="proton"):
            #lenats = len(self.mol.GetAtoms())
            for i,atm in enumerate(self.mol.GetAtoms()):
                attr = {key: atm.GetAtomicNum()}
                self.add_node(i, **attr)
    
    def make(self,nodes={'proton' : "proton" ,
                         'atom' : "atom_label"
                         },
                  edges = {'bond' : 'bond_type',
                           'distance' : {'class' : 'distance' , 'args' : {'bonds_onyly' : True, 'max_distance' : np.inf , 'max_partners' : np.inf}}
                           },
                  state = {'size' : 'num_atoms' 
                           }):
        
        for key,value in nodes.items():
            if(isinstance(value,str) == True):
                self._make_nodes(key,value,None)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else None
                    self._make_nodes(key,value['class'],args)
                else:
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
                
        for key,value in edges.items():
            if(isinstance(value,str) == True):
                self._make_edges(key,value,None)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else None
                    self._make_edges(key,value['class'],args)
                else:
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
        
        for key,value in state.items():
            if(isinstance(value,str) == True):
                self._make_state(key,value,None)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else None
                    self._make_state(key,value['class'],args)
                else:
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
            
    
    def to_tensor(self):
        pass
        
    
test = MolGraph()
test.mol_from_smiles("CCCO")
test.make()
nx.draw(test)