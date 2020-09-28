"""
Main graph generator for making molecular graphs.

It uses networkx as graph interface and a mol object from rdkit, ase, pymatgen or similar.

@author: Patrick Reiser,
"""

#Necessary
import networkx as nx
import numpy as np

#General
from molreps.methods.geo_npy import coordinates_to_distancematrix,define_adjacency_from_distance

#Rdkit and methods
try:
    import rdkit
    import rdkit.Chem.AllChem
    MOLGRAPH_RDKIT_AVAILABLE = True
    from molreps.methods.mol_rdkit import rdkit_proton_list,rdkit_atomlabel_list,rdkit_bond_type_list
except:
    print("Warning: Rdkit not found for mol class.")



class MolGraph(nx.Graph):
    """Molecular Graph which inherits from networkx graph."""
    
    _nodes_implemented = ['proton','atom_symbol']
    _edges_implemented = ['bond_type','distance']
    _state_implemented = ['num_atoms']
    
    
    def __init__(self,**kwargs):
        super(MolGraph, self).__init__(**kwargs)
        
        # Determine available mol libs
        self.mol_libs_avail = []
        if(MOLGRAPH_RDKIT_AVAILABLE == True):
            self.mol_libs_avail.append("rdkit")
        
        # Main mol object to use.
        self.mol = None
        
        # State Variable
        self._graph_state = {}
    

    ###########################################################################
        
    # Make mol class with different backends
    def mol_from_smiles(self,in_smile):
        """
        Generate mol object from a smile string.

        Args:
            in_smile (str): smile.

        Returns:
            MolObject: Representation of the molecule.

        """ 
        # choose rdkit etc here
        m = rdkit.Chem.MolFromSmiles(in_smile)
        m = rdkit.Chem.AddHs(m)
        self.mol = m
        return self.mol
    
    def mol_from_structure(self,atoms,bondtab,coordinates=None):
        pass
    
    def mol_from_geometry(self,atoms,coordinates):
        pass
    
    ###########################################################################

    # Conformere management with different backends
    def _has_conformere(self):
        # choose rdkit etc here
        conf_info =  len(self.mol.GetConformers()) > 0 
        return conf_info
    
    def _get_conformere(self,conf_selection=0):
        if(self._has_conformere()):
            # choose rdkit etc here
            return np.array(self.mol.GetConformers()[conf_selection].GetPositions())

    
    def conformation(self,mol_lib='rdkit',methods='ETKDG',conf_selection=0):
        # choose rdkit etc here
        seed =0xf00d 
        retval = rdkit.Chem.AllChem.EmbedMolecule(self.mol,randomSeed=seed)
        out = self.mol.GetConformers()[conf_selection].GetPositions()
        return np.array(out)
       
    
    
    ###########################################################################
    
    # Property calculation with different mol backends
    def _find_nodes_proton(self,key):
        # choose rdkit etc here
        return rdkit_proton_list(self.mol,key)

    def _find_nodes_atomlabel(self,key):
        # choose rdkit etc here
        return rdkit_atomlabel_list(self.mol,key)
    
    def _find_edges_bond_type(self,key):
        # choose rdkit etc here
        return rdkit_bond_type_list(self.mol,key)
    
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
        return {key: len(self.mol.GetAtoms())}
    
    ###########################################################################
    
    # Check for identifier
    def _make_edges(self,key,propy,args):
        if(propy=="bond_type"):
            self.add_edges_from(self._find_edges_bond_type(key))
        elif(propy=="distance"):
            self.add_edges_from(self._find_edges_distance(key,**args))
        elif(propy=="inverse_distance"):
            pass
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._nodes_implemented)
            
    def _make_nodes(self,key,propy,args):
        if(propy=="proton"):
            self.add_nodes_from(self._find_nodes_proton(key))
        elif(propy=="atom_symbol"):
            self.add_nodes_from(self._find_nodes_atomlabel(key))
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._edges_implemented)
          
    def _make_state(self,key,propy,args):
        if(propy=="num_atoms"):
            self._graph_state.update(self._find_state_size(key))
        else:
            raise ValueError("Property identifier",propy,"is not implemented. Choose",self._state_implemented)
    
    ###########################################################################
    
    def make(self,nodes={'proton' : "proton" ,
                         'atom' : "atom_symbol"
                         },
                  edges = {'bond' : 'bond_type',
                           'distance' : {'class':'distance', 'args':{'bonds_only':True,'max_distance':None,'max_partners':None}}
                           },
                  state = {'size' : 'num_atoms' 
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
            edges (dict, optional): Properties for edges. Defaults to {'bond' : 'bond_type', ... }                      
            state (dict, optional): Properties for graph state. Defaults to {'size' : 'num_atoms', ... }                   

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
                         nodes = {'proton' : [np.array,np.array(0)] },
                         edges = {'bond' : [np.array,np.array(0)], 'distance' : [np.array,np.array(0)]},
                         state = {'size' : [np.array,np.array(0)]},
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
            outn.append([trafo[0](self.nodes[i][key]) if key in self.nodes[i] else trafo[1] for key,trafo in nodes.items()])
        outn = out_tensor(outn)
        
        for i in edge_idx:
            oute.append([trafo[0](self.edges[i][key]) if key in self.edges[i] else trafo[1] for key,trafo in edges.items()])
        oute = out_tensor(oute)
        
        outs = [trafo[0](self._graph_state[key]) if key in self._graph_state else trafo[1] for key,trafo in state.items()]
        outs = out_tensor(outs)
        
        # Make directed
        outei = np.concatenate([np.array(edge_idx),np.flip(np.array(edge_idx),axis=-1)],axis=0)
        oute = np.concatenate([oute,oute],axis=0)
        
        #Need some sorting also
        
        
        return {"nodes" : outn,
                "edges" :oute,
                "state" :outs,
                "adjacency" :outA,
                "indices" :outei}
        
    
test = MolGraph()
test.mol_from_smiles("CCCO")
test.make()
nx.draw(test)
out = test.to_graph_tensors()