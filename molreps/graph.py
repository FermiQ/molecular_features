"""
Main graph generator for making molecular graphs.

It uses networkx as graph interface and a mol object from rdkit, ase, pymatgen or similar.
"""

#Necessary
import networkx as nx
import numpy as np


#Rdkit
try:
    import rdkit
    import rdkit.Chem.Descriptors
    import rdkit.Chem.AllChem
    MOLGRAPH_RDKIT_AVAILABLE = True
    from molreps.methods.mol_rdkit import rdkit_atom_list,rdkit_bond_list,rdkit_bond_distance_list
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


###############################################################################
   

if MOLGRAPH_RDKIT_AVAILABLE == True:
    
    def rdkit_get_property_atoms(mol,key,prop,**kwargs):
    
        atom_fun_dict={
            "proton" : rdkit.Chem.rdchem.Atom.GetAtomicNum,
            "symbol" : rdkit.Chem.rdchem.Atom.GetSymbol,
            "num_Hs" : rdkit.Chem.rdchem.Atom.GetNumExplicitHs,
            "aromatic" : rdkit.Chem.rdchem.Atom.GetIsAromatic,
            "degree" : rdkit.Chem.rdchem.Atom.GetTotalDegree,
            "valence" : rdkit.Chem.rdchem.Atom.GetTotalValence,
            "mass" : rdkit.Chem.rdchem.Atom.GetMass,
            "in_ring" : rdkit.Chem.rdchem.Atom.IsInRing,
            "hybridization" : rdkit.Chem.rdchem.Atom.GetHybridization,
            }
        if(prop in atom_fun_dict):
            return rdkit_atom_list(mol,key,atom_fun_dict[prop])
        else:
            raise NotImplementedError("Property",prop,"is not predefined, use costum function.")
        
    
    def rdkit_get_property_bonds(mol,key,prop,**kwargs):
        
        bond_fun_dict = {
            "bond" : rdkit.Chem.rdchem.Bond.GetBondType,
            "is_aromatic" : rdkit.Chem.rdchem.Bond.GetIsAromatic,
            "is_conjugated" : rdkit.Chem.rdchem.Bond.GetIsConjugated,
            "in_ring" : rdkit.Chem.rdchem.Bond.IsInRing
            }
        
        if(prop in bond_fun_dict):
            return rdkit_bond_list(mol,key,bond_fun_dict[prop])
        elif(prop == "distance"):
            return rdkit_bond_distance_list(mol,key,**kwargs)    
        else:
            raise NotImplementedError("Property",prop,"is not predefined, use costum function.")
        
    
    def rdkit_get_property_molstate(mol,key,prop,**kwargs):
        state_fun_dict = {
            "mol_weight" : rdkit.Chem.Descriptors.ExactMolWt
            }
        if(prop in state_fun_dict):
            return {key: state_fun_dict[prop](mol)}
        elif(prop == "size"):
            return {key: mol.GetNumAtoms()}
        else:
            raise NotImplementedError("Property",prop,"is not predefined, use costum function.")
    
    
    
    

###############################################################################

# Main class to make graph
class MolGraph(nx.Graph):
    """Molecular Graph which inherits from networkx graph."""
    
    _mols_implemented = {'rdkit': {'nodes': ["proton" ,"symbol","num_Hs","aromatic","degree","valence","mass","in_ring","hybridization"] ,
                                   'edges': ["bond","is_aromatic","is_conjugated","in_ring","distance"] ,
                                   'state': ["mol_weight","size"]}
                         }

    
    def __init__(self,mol = None, **kwargs):
        super(MolGraph, self).__init__(**kwargs)
          
        self.mol = mol
        # State Variable
        self._graph_state = {}
        self.mol_type = None
        if(isinstance(mol,rdkit.Chem.Mol)):
            self.mol_type = "rdkit"
        
    
    ###########################################################################
    
    # Check for identifier
    def _make_edges(self,key,propy,**args):
        if(self.mol_type == "rdkit"):
            self.add_edges_from(rdkit_get_property_bonds(self.mol,key=key,prop=propy,**args))
        else:
            raise ValueError("Property identifier is not implemented for mol type",self.mol_type)
        
    def _make_nodes(self,key,propy,**args):
        if(self.mol_type == "rdkit"):
            self.add_nodes_from(rdkit_get_property_atoms(self.mol,key=key,prop=propy,**args))
        else:
            raise ValueError("Property identifier is not implemented for mol type",self.mol_type)   
          
    def _make_state(self,key,propy,**args):
        if(self.mol_type == "rdkit"):
            self._graph_state.update(rdkit_get_property_molstate(self.mol,key=key,prop=propy,**args))
        else:
            raise ValueError("Property identifier is not implemented for mol type",self.mol_type)   
    
    ###########################################################################
    
    def make(self,nodes={'proton' : "proton" ,
                         'symbol' : "symbol",
                         "num _Hs" : "num_Hs",
                         "aromatic" : "aromatic",
                         "degree" : "degree",
                         "valence" : "valence",
                         "mass" : "mass",
                         "in_ring" : "in_ring",
                         "hybridization" : "hybridization"
                         },
                  edges = {'bond' : 'bond',
                           'distance' : {'class':'distance', 'args':{'bonds_only':True,'max_distance':np.inf ,'max_partners': np.inf}},
                           "is_aromatic" : "is_aromatic",
                           "is_conjugated" : "is_conjugated",
                           "in_ring" : "in_ring",
                           },
                  state = {'size' : 'size',
                           "mol_weight" : "mol_weight"
                           }):
        """
        Construct graph from mol instance.
        
        The input is a dictionary of properties to calculate. The dict-key 
        can be chosen freely and will be graph attributes. 
        The identifier is a string for built-in function e.g. 'proton'. Or if args have to be provided:
        key : {'class': identifier, 'args':{ args_dict }}
        Otherwise you can provide a costum method via the the identifier dict of the form:
        key : {'class': function/class, 'args':{ args_dict }}
        The callable object of 'class' must accept as mol and key arguments this instance. Then key,mol and then additiona args.
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
                self._make_nodes(key,value)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else {}
                    self._make_nodes(key,value['class'],**args)
                else:
                    #Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
                
        for key,value in edges.items():
            if(isinstance(value,str) == True):
                self._make_edges(key,value)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else {}
                    self._make_edges(key,value['class'],**args)
                else:
                    #Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
        
        for key,value in state.items():
            if(isinstance(value,str) == True):
                self._make_state(key,value)
            elif(isinstance(value,dict) == True): 
                if('class' not in value):
                    raise ValueError(" 'class' method must be defined in",value)
                if(isinstance(value['class'],str) == True):
                    args = value['args'] if 'args' in value else {}
                    self._make_state(key,value['class'],**args)
                else:
                    #Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self,key,**args)
            else:
                raise TypeError("Method must be a dict of {'class' : callable function/class or identifier, 'args' : {'value' : 0} }, with optinal args but got", value, "instead")
            
    ###########################################################################
    
    def to_tensor(self,
                         nodes = ['proton' ],
                         edges = ['bond' , 'distance' ],
                         state = ['size' ],
                         trafo_nodes = {},
                         trafo_edges = {},
                         trafo_state = {},
                         default_nodes = {},
                         default_edges = {},
                         default_state = {},
                         out_tensor = np.array
                         ):
        """
        Convert the nx graph into a dict of tensors which can be directly used for GCN.
        
        The desired attributes must be given with a suitable converison function plus default value. 
        Here, one can add also the type of tensor or one-Hot mappings etc. and its default/zero state
        if the attributes is not specified for a specific node/edge.

        Args:
            nodes (list, optional): Nodes properties. Defaults to ['proton'].
            edges (list, optional): Edge properties. Defaults to ['bond' ,'distance' ].
            state (list, optional): State Properties. Defaults to ['size' ].
            trafo_nodes (dict,optinal): Transformation function for nodes. Defaults to np.array if no entry found.
            trafo_edges (dict,optinal): Transformation function for edges. Defaults to np.array if no entry found.
            trafo_state (dict,optinal): Transformation function for state. Defaults to np.array if no entry found.
            default_nodes (dict, optional): Zero Nodes properties. Defaults to np.array(0) if no entry found.
            default_edges (dict, optional): Zero Edge properties. Defaults to np.array(0) if no entry found.
            default_state (dict, optional): Zero State Properties. Defaults to np.array(0) if no entry found.
            out_tensor (func) : Final Function for each node/edge/state. Default is np.array if no entry found.

        Returns:
            dict: Graph tensors as dictionary.

        """
        for x in nodes:
            if x not in trafo_nodes:
                trafo_nodes[x] = np.array
        for x in edges:
            if x not in trafo_edges:
                trafo_edges[x] = np.array
        for x in state:
            if x not in trafo_state:
                trafo_state[x] = np.array
        for x in nodes:
            if x not in default_nodes:
                default_nodes[x] = np.array(0)
        for x in edges:
            if x not in default_edges:
                default_edges[x] = np.array(0)
        for x in state:
            if x not in default_state:
                default_state[x] = np.array(0)
        
        outn = []
        oute = []
        outs = []
        outA = nx.to_numpy_array(self)
        outei = []
        
        node_idx = np.array(list(self.nodes),dtype=np.int)
        edge_idx = np.array(list(self.edges),dtype=np.int)
        
        for i in node_idx:
            current_node = []
            for key in nodes:
                if key in self.nodes[i]:
                    current_node.append(trafo_nodes[key](self.nodes[i][key]))
                else: 
                    current_node.append(default_nodes[key])
            outn.append(current_node)               
        outn = out_tensor(outn)
        
        for i in edge_idx:
            current_edge = []
            for key in edges:
                if key in self.edges[i]: 
                    current_edge.append(trafo_edges[key](self.edges[i][key]))
                else: 
                    current_edge.append(default_edges[key])
            oute.append(current_edge)
        oute = out_tensor(oute)
        
        for key in state:
            if key in self._graph_state:
                outs.append(trafo_state[key](self._graph_state[key]))
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
    
    
        
m = rdkit.Chem.MolFromSmiles("CC=O")    
test = MolGraph(m)
test.make()
nx.draw(test,with_labels=True)
out = test.to_tensor()