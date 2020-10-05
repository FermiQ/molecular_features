"""
Main graph generator for making molecular graphs.

It uses networkx as graph interface and a mol object from rdkit, ase, pymatgen or similar.

@author: Patrick Reiser,
"""

#Necessary
import networkx as nx

#python
from molreps.methods.props_py import element_list_to_value,get_atom_property_dicts

#Numpy
import numpy as np
from molreps.methods.geo_npy import coordinates_to_distancematrix,define_adjacency_from_distance
from molreps.methods.geo_npy import coulombmatrix_to_inversedistance_proton,coordinates_from_distancematrix,invert_distance

#Rdkit
try:
    import rdkit
    import rdkit.Chem.AllChem
    MOLGRAPH_RDKIT_AVAILABLE = True
    from molreps.methods.mol_rdkit import rdkit_atom_list,rdkit_bond_list
    from molreps.methods.mol_rdkit import rdkit_mol_from_atoms_bonds,rdkit_add_conformer
except:
    print("Warning: Rdkit not found for mol class.")
    MOLGRAPH_RDKIT_AVAILABLE = False

#OPenbabel
try:
    import openbabel
    MOLGRAPH_OPENBABEL_AVAILABLE = True
    from molreps.methods.mol_pybel import ob_get_bond_table_from_coordinates
except:
    print("Warning: Openbabel not found for mol class.")
    MOLGRAPH_OPENBABEL_AVAILABLE = False



class MolInterface():
    
    _nodes_implemented = ['proton','symbol']
    _edges_implemented = ['bond','distance']
    _state_implemented = ['size']
    
    def __init__(self,**kwargs):
        # Determine available mol libs
        self.mol_libs_avail = []
        if(MOLGRAPH_RDKIT_AVAILABLE == True):
            self.mol_libs_avail.append("rdkit")
        if(MOLGRAPH_OPENBABEL_AVAILABLE == True):
            self.mol_libs_avail.append("openbabel")
        
        # Main mol object to use.
        self.mol_libs_use = "rdkit"
        self.mol = None
    
    ###########################################################################
    
    # Conformere management with different backends
    def _has_conformere(self):
        if(self.mol_libs_use == "rdkit"):
            conf_info =  len(self.mol.GetConformers()) > 0 
            return conf_info
    
    def _get_conformere(self,conf_selection=0):
        if(self._has_conformere()):
            if(self.mol_libs_use == "rdkit"):
                return np.array(self.mol.GetConformers()[conf_selection].GetPositions())

    
    def conformation(self,mol_lib='rdkit',methods='ETKDG',conf_selection=0):
        if(self.mol_libs_use == "rdkit"):
            seed =0xf00d 
            retval = rdkit.Chem.AllChem.EmbedMolecule(self.mol,randomSeed=seed)
            out = self.mol.GetConformers()[conf_selection].GetPositions()
            return np.array(out)
       
    ###########################################################################
        
    # Make mol class with different backends
    def mol_from_smiles(self,in_smile):
        
        if(self.mol_libs_use == "rdkit"):
            m = rdkit.Chem.MolFromSmiles(in_smile)
            m = rdkit.Chem.AddHs(m)
            self.mol = m
            return self.mol
    
    def mol_from_structure(self,atoms,bondlist,coordinates=None):

        if(self.mol_libs_use == "rdkit"):
            self.mol = rdkit_mol_from_atoms_bonds(atoms,bondlist)
            if(coordinates is not None):
                rdkit_add_conformer(self.mol,coordinates)
            return self.mol
    
    
    def mol_from_geometry(self,atoms,coordinates,backend='openbabel'):

        if("openbabel" in self.mol_libs_avail and backend=='openbabel'):
            _,_,bonds,_ = ob_get_bond_table_from_coordinates
            return self.mol_from_structure(atoms,bonds,coordinates)
        else:
            print("Will be implemented soon")
    
    
    def mol_from_coulombmat(self,coulmat,unit_conversion=1):

        # Does not require mol backend inference, just self.mol_from_geometry
        invd,pr = coulombmatrix_to_inversedistance_proton(coulmat,unit_conversion)
        dist = invert_distance(invd)
        cords = coordinates_from_distancematrix(dist)
        ats = element_list_to_value(pr,get_atom_property_dicts("FromProton"))
        return self.mol_from_geometry(ats,cords)
    
    ###########################################################################
    
    # Property calculation with different mol backends
    def _find_nodes_proton(self,key):
        if(self.mol_libs_use == "rdkit"):
            return rdkit_atom_list(self.mol,key,rdkit.Chem.rdchem.Atom.GetAtomicNum)

    def _find_nodes_atomlabel(self,key):
        if(self.mol_libs_use == "rdkit"):
            return rdkit_atom_list(self.mol,key,rdkit.Chem.rdchem.Atom.GetSymbol)
    
    def _find_edges_bond_type(self,key):
        if(self.mol_libs_use == "rdkit"):
            return rdkit_bond_list(self.mol,key,rdkit.Chem.rdchem.Bond.GetBondType)
    
    def _find_edges_distance(self,key,bonds_only=True,max_distance=None,max_partners=None):
        #No check mol-lib necessary, indirect via _make_edges_bond_type() and conformation()
        if(not self._has_conformere()):
            self.conformation()
        dist_mat = coordinates_to_distancematrix(self._get_conformere())
        adj_dist, idx_dist = define_adjacency_from_distance(dist_mat,max_distance,max_partners)
        if(bonds_only == True):
            bonds = self._find_edges_bond_type(key) # use distance key
            for x in bonds:
                # Replace Bond-type by distance
                x[2][key] = dist_mat[x[0],x[1]]
            return bonds
        else:    
            out_list = []
            for i in range(len(idx_dist)):
                out_list.append((idx_dist[i][0],idx_dist[i][1],{ key : dist_mat[idx_dist[i][0],idx_dist[i][1]]}))    
        return out_list

    def _find_state_size(self,key):
        if(self.mol_libs_use == "rdkit"):
            return {key: len(self.mol.GetAtoms())}   
        



class MolGraph(nx.Graph):
    """Molecular Graph which inherits from networkx graph."""
    
    
    def __init__(self,**kwargs):
        super(MolGraph, self).__init__(**kwargs)
          
        self.mol = MolInterface()
        # State Variable
        self._graph_state = {}
    

    ###########################################################################
        
    # Make mol interface with different backends
    def mol_from_smiles(self,in_smile):
        """
        Generate mol object from a smile string.

        Args:
            in_smile (str): smile.

        Returns:
            MolInterface: Representation of the molecule.

        """ 
        return self.mol.mol_from_smiles(in_smile)
    
    def mol_from_structure(self,atoms,bondlist,coordinates=None):
        """
        Generate a mol object from a given strucutre. 

        Args:
            atoms (list): Atomlist e.g. ['C','H',"H"].
            bondlist (list):    Bondlist of shape (N,3). With a bond as (i,j,type)
                                The bondtype is directly transferred to backend.
            coordinates (array, optional): Coordinates for a given conformere of shape (N,3). Defaults to None.

        Returns:
            MolInterface: Representation of the molecule.

        """
        return self.mol.mol_from_structure(atoms,bondlist,coordinates=None)
    
    
    def mol_from_geometry(self,atoms,coordinates,backend='openbabel'):
        """
        Generate a mol class from coordinates and atom Type.
        
        Note here the bond order and type has to be determined. This is not 
        strictly unique and can differ in the method used,

        Args:
            atoms (list): Atomlist e.g. ['C','H',"H"].
            coordinates (arry): List of shape (N,3)
            backend (str): If a specific mol lib should be used.

        Returns:
             MolInterface: Representation of the molecule.

        """
        return self.mol.mol_from_geometry(atoms,coordinates,backend='openbabel')
    
    
    def mol_from_coulombmat(self,coulmat,unit_conversion=1):
        """
        Map coulombmatrix to mol class. 
        
        This is however a slow and difficult problem as there is not strictly a unique mol.

        Args:
            coulmat (np.array): Coulombmatrix of shape (N,N).
            unit_conversion (TYPE, optional): If distance is not in Angström (often the case). Defaults to 1.

        Returns:
            MolInterface: Representation of the molecule.

        """
        return self.mol.mol_from_coulombmat(coulmat,unit_conversion=1)
        
   
    ###########################################################################
    

    # Conformere management with different backends
    def _has_conformere(self):
        return self.mol._has_conformere()
    
    def _get_conformere(self,conf_selection=0):
        return self.mol._get_conformere(conf_selection=0)
    
    def conformation(self,methods='ETKDG',conf_selection=0,mol_lib='rdkit'):
        return self.mol.conformation(mol_lib,methods,conf_selection)
    
    ###########################################################################
    
    # Check for identifier
    def _make_edges(self,key,propy,args):
        if(propy=="bond"):
            self.add_edges_from(self.mol._find_edges_bond_type(key))
        elif(propy=="distance"):
            self.add_edges_from(self.mol._find_edges_distance(key,**args))
        elif(propy=="inverse_distance"):
            pass
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._nodes_implemented)
            
    def _make_nodes(self,key,propy,args):
        if(propy=="proton"):
            self.add_nodes_from(self.mol._find_nodes_proton(key))
        elif(propy=="symbol"):
            self.add_nodes_from(self.mol._find_nodes_atomlabel(key))
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._edges_implemented)
          
    def _make_state(self,key,propy,args):
        if(propy=="size"):
            self._graph_state.update(self.mol._find_state_size(key))
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._state_implemented)
    
    ###########################################################################
    
    def make(self,nodes={'proton' : "proton" ,
                         'symbol' : "symbol"
                         },
                  edges = {'bond' : 'bond',
                           'distance' : {'class':'distance', 'args':{'bonds_only':True,'max_distance':None,'max_partners':None}}
                           },
                  state = {'size' : 'size' 
                           }):
        """
        Construct graph from mol instance.
        
        The input is a dictionary of properties to calculate. The dict-key 
        can be chosen freely and will be graph attributes. 
        The identifier is a string for built-in function e.g. 'proton'. Or if args have to be provided:
        key : {'class': identifier, 'args':{ args_dict }}
        Otherwise you can provide a costum method via the the identifier dict of the form:
        key : {'class': function/class, 'args':{ args_dict }}
        The callable object of 'class' must accept as first argument this instance. Then key, and then args.
        Info: This matches tf.keras identifier scheme.
        
        Args:
            nodes (dict, optional): Properties for nodes. Defaults to {'proton' : "proton" , ... }                   
            edges (dict, optional): Properties for edges. Defaults to {'bond' : 'bond', ... }                      
            state (dict, optional): Properties for graph state. Defaults to {'size' : 'size', ... }                   

        Raises:
            AttributeError: If mol not found.
            ValueError: If identifier dict is incorrect.
            TypeError: If property info is incorrect.

        Returns:
            None.

        """
        if(self.mol == None):
            raise AttributeError("Initialize Molecule before making graph") 
        
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
                    #Custom function/class here
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
                    #Custom function/class here
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
                    #Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
            
    ###########################################################################
    
    def to_graph_tensors(self,
                         nodes = {'proton' : np.array },
                         edges = {'bond' : np.array, 'distance' : np.array},
                         state = {'size' : np.array},
                         default_nodes = {'proton' : np.array(0)},
                         default_edges = {'bond' : np.array(0), 'distance' : np.array(0)},
                         default_state = {'size' : np.array(0)},
                         out_tensor = np.array
                         ):
        """
        Convert the nx graph into a dict of tensors which can be directly used for GCN.
        
        The desired attributes must be given with a suitable converison function plus default value. 
        Here, one can add also the type of tensor or one-Hot mappings etc. and its default/zero state
        if the attributes is not specified for a specific node/edge.
        We can change this also to a default zero padding if this is a better way.

        Args:
            nodes (dict, optional): Nodes properties. Defaults to {'proton' : np.array }.
            edges (dict, optional): Edge properties. Defaults to {'bond' : np.array, 'distance' : np.array}.
            state (dict, optional): State Properties. Defaults to {'size' : np.array}.
            default_nodes (dict, optional): Zero Nodes properties. Defaults to {'proton' : np.array(0) }.
            default_edges (dict, optional): Zero Edge properties. Defaults to {'bond' : np.array(0), 'distance' : np.array(0)}.
            default_state (dict, optional): Zero State Properties. Defaults to {'size' : np.array(0)}.
            out_tensor (func) : Final Function for each node/edge/state. Default is np.array.

        Returns:
            dict: Graph tensors as dictionary.

        """
        outn = []
        oute = []
        outs = []
        outA = nx.to_numpy_array(self)
        outei = []
        
        node_idx = np.array(list(self.nodes),dtype=np.int)
        edge_idx = np.array(list(self.edges),dtype=np.int)
        
        for i in node_idx:
            current_node = []
            for key,trafo in nodes.items():
                if key in self.nodes[i]:
                    current_node.append(trafo(self.nodes[i][key]))
                else: 
                    current_node.append(default_nodes[key])
            outn.append(current_node)               
        outn = out_tensor(outn)
        
        for i in edge_idx:
            current_edge = []
            for key,trafo in edges.items():
                if key in self.edges[i]: 
                    current_edge.append(trafo(self.edges[i][key]))
                else: 
                    current_edge.append(default_edges[key])
            oute.append(current_edge)
        oute = out_tensor(oute)
        
        for key,trafo in state.items():
            if key in self._graph_state:
                outs.append(trafo(self._graph_state[key]))
            else:
                outs.append(default_state[key])
        outs = out_tensor(outs)
        
        # Make directed
        outei = np.concatenate([np.array(edge_idx),np.flip(np.array(edge_idx),axis=-1)],axis=0)
        oute = np.concatenate([oute,oute],axis=0)
        
        #Need some sorting for e.g. GCN
        sorts = np.argsort(outei[:,0],axis=0)
        outei = outei[sorts]
        oute = oute[sorts]
        
        return {"nodes" : outn,
                "edges" :oute,
                "state" :outs,
                "adjacency" :outA,
                "indices" :outei}
        
    
test = MolGraph()
test.mol_from_smiles("C=O")
test.make()
nx.draw(test)
out = test.to_graph_tensors()