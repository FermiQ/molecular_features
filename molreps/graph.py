"""
Main graph generator for making molecular graphs.

It uses networkx as graph interface and a mol object from rdkit, ase, pymatgen or similar.
"""

# Necessary
import networkx as nx
import numpy as np
from molreps.methods.geo_npy import add_edges_reverse_indices
# Rdkit
try:
    import rdkit
    import rdkit.Chem.Descriptors
    import rdkit.Chem.AllChem

    MOLGRAPH_RDKIT_AVAILABLE = True
    from molreps.methods.mol_rdkit import rdkit_atom_list, rdkit_bond_list, rdkit_bond_distance_list
    from molreps.methods.mol_rdkit import rdkit_mol_from_atoms_bonds, rdkit_add_conformer
except ModuleNotFoundError:
    print("Warning: Rdkit not found for mol class.")
    MOLGRAPH_RDKIT_AVAILABLE = False

# openbabel
try:
    from openbabel import openbabel

    MOLGRAPH_OPENBABEL_AVAILABLE = True
    from molreps.methods.mol_pybel import ob_get_bond_table_from_coordinates
except ModuleNotFoundError:
    print("Warning: Openbabel not found for mol class.")
    MOLGRAPH_OPENBABEL_AVAILABLE = False

if MOLGRAPH_RDKIT_AVAILABLE:

    def rdkit_get_property_atoms(mol, key, prop, **kwargs):

        atom_fun_dict = {
            "AtomicNum": rdkit.Chem.rdchem.Atom.GetAtomicNum,
            "Symbol": rdkit.Chem.rdchem.Atom.GetSymbol,
            "NumExplicitHs": rdkit.Chem.rdchem.Atom.GetNumExplicitHs,
            "NumImplicitHs": rdkit.Chem.rdchem.Atom.GetNumImplicitHs,
            "IsAromatic": rdkit.Chem.rdchem.Atom.GetIsAromatic,
            "TotalDegree": rdkit.Chem.rdchem.Atom.GetTotalDegree,
            "TotalValence": rdkit.Chem.rdchem.Atom.GetTotalValence,
            "Mass": rdkit.Chem.rdchem.Atom.GetMass,
            "IsInRing": rdkit.Chem.rdchem.Atom.IsInRing,
            "Hybridization": rdkit.Chem.rdchem.Atom.GetHybridization,
            "ChiralTag": rdkit.Chem.rdchem.Atom.GetChiralTag,
            "FormalCharge": rdkit.Chem.rdchem.Atom.GetFormalCharge,
            "ImplicitValence": rdkit.Chem.rdchem.Atom.GetImplicitValence,
            "NumRadicalElectrons": rdkit.Chem.rdchem.Atom.GetNumRadicalElectrons,
        }
        if prop in atom_fun_dict:
            return rdkit_atom_list(mol, key, atom_fun_dict[prop])
        else:
            raise NotImplementedError("Property", prop, "is not predefined, use custom function.")


    def rdkit_get_property_bonds(mol, key, prop, **kwargs):

        bond_fun_dict = {
            "BondType": rdkit.Chem.rdchem.Bond.GetBondType,
            "IsAromatic": rdkit.Chem.rdchem.Bond.GetIsAromatic,
            "IsConjugated": rdkit.Chem.rdchem.Bond.GetIsConjugated,
            "IsInRing": rdkit.Chem.rdchem.Bond.IsInRing,
            "Stereo": rdkit.Chem.rdchem.Bond.GetStereo
        }
        if prop in bond_fun_dict:
            return rdkit_bond_list(mol, key, bond_fun_dict[prop])
        elif prop == "Distance":
            return rdkit_bond_distance_list(mol, key, **kwargs)
        else:
            raise NotImplementedError("Property", prop, "is not predefined, use custom function.")


    def rdkit_get_property_molstate(mol, key, prop, **kwargs):
        state_fun_dict = {
            "ExactMolWt": rdkit.Chem.Descriptors.ExactMolWt
        }
        if prop in state_fun_dict:
            return {key: state_fun_dict[prop](mol)}
        elif prop == "NumAtoms":
            return {key: mol.GetNumAtoms()}
        else:
            raise NotImplementedError("Property", prop, "is not predefined, use custom function.")


# Main class to make graph
class MolGraph(nx.Graph):
    """Molecular Graph which inherits from networkx graph."""

    _mols_implemented = {'rdkit': {
        'nodes': ["AtomicNum", "Symbol", "NumExplicitHs","NumImplicitHs","IsAromatic","TotalDegree",
            "TotalValence","Mass", "IsInRing","Hybridization", "ChiralTag", "FormalCharge",
            "ImplicitValence", "NumRadicalElectrons"],
        'edges': ["BondType","IsAromatic","IsConjugated","IsInRing","Stereo","Distance"],
        'state': ["NumAtoms", "ExactMolWt"]}
    }

    def __init__(self, mol=None, **kwargs):
        super(MolGraph, self).__init__(**kwargs)

        self.mol = mol
        # State Variable
        self._graph_state = {}
        self.mol_type = None
        if isinstance(mol, rdkit.Chem.Mol):
            self.mol_type = "rdkit"

    # Check for identifier
    def _make_edges(self, key, propy, **args):
        if self.mol_type == "rdkit":
            self.add_edges_from(rdkit_get_property_bonds(self.mol, key=key, prop=propy, **args))
        else:
            raise ValueError("Property identifier is not implemented for mol type", self.mol_type)

    def _make_nodes(self, key, propy, **args):
        if self.mol_type == "rdkit":
            self.add_nodes_from(rdkit_get_property_atoms(self.mol, key=key, prop=propy, **args))
        else:
            raise ValueError("Property identifier is not implemented for mol type", self.mol_type)

    def _make_state(self, key, propy, **args):
        if self.mol_type == "rdkit":
            self._graph_state.update(rdkit_get_property_molstate(self.mol, key=key, prop=propy, **args))
        else:
            raise ValueError("Property identifier is not implemented for mol type", self.mol_type)

    def make(self,
             nodes=None,
             edges=None,
             state=None
             ):
        """
        Construct graph from mol instance.
        
        The input is a dictionary of properties to calculate. The dict-key 
        can be chosen freely and will be graph attributes. 
        The identifier is a string for built-in function e.g. 'proton'. Or if args have to be provided:
        key : {'class': identifier, 'args':{ args_dict }}
        Otherwise you can provide a custom method via the the identifier dict of the form:
        key : {'class': function/class, 'args':{ args_dict }}
        The callable object of 'class' must accept as first argument this instance.
        Then key=key and then additional args from 'args':{ args_dict }.
        
        Args:
            nodes (dict, optional): Properties for nodes. Defaults to {'proton' : "proton" }
            edges (dict, optional): Properties for edges. Defaults to
                {'bond': 'bond'} or {'distance': {'class': 'distance', 'args': {}}
            state (dict, optional): Properties for graph state. Defaults to {'size' : 'size'}

        Raises:
            AttributeError: If mol not found.
            ValueError: If identifier dict is incorrect.
            TypeError: If property info is incorrect.

        Returns:
            self: This instance.
        """
        # Set defaults if None
        if self.mol is None:
            raise AttributeError("Initialize Molecule before making graph")
        if nodes is None:
            nodes = [self._mols_implemented[self.mol_type]['nodes'][0]]
        if edges is None:
            edges = [self._mols_implemented[self.mol_type]['edges'][0]]
        if state is None:
            state = [self._mols_implemented[self.mol_type]['state'][0]]

        # Make default keys if only list is inserted
        if isinstance(nodes, list) or isinstance(nodes, tuple):
            nodes_dict = {}
            for x in nodes:
                if isinstance(x, str):
                    nodes_dict.update({x: x})
                elif isinstance(x, dict):
                    nodes_dict.update({x['class']: x})
                else:
                    raise ValueError(
                        "Method must be single string or class dict but got", x)
            nodes = nodes_dict
        if isinstance(edges, list) or isinstance(edges, tuple):
            edges_dict = {}
            for x in edges:
                if isinstance(x, str):
                    edges_dict.update({x: x})
                elif isinstance(x, dict):
                    edges_dict.update({x['class']: x})
                else:
                    raise ValueError(
                        "Method must be single string or class dict serialized, but got", x)
            edges = edges_dict
        if isinstance(state, list) or isinstance(state, tuple):
            state_dict = {}
            for x in state:
                if isinstance(x, str):
                    state_dict.update({x: x})
                elif isinstance(x, dict):
                    state_dict.update({x['class']: x})
                else:
                    raise ValueError(
                        "Method must be single string or class dict but got", x)
            state = state_dict

        for key, value in nodes.items():
            if isinstance(value, str):
                self._make_nodes(key, value)
            elif isinstance(value, dict):
                if 'class' not in value:
                    raise ValueError(" 'class' method must be defined in", value)
                if isinstance(value['class'], str):
                    args = value['args'] if 'args' in value else {}
                    self._make_nodes(key, value['class'], **args)
                else:
                    # Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self, key=key, **args)
            else:
                raise TypeError(
                    "Method must be a dict of {'class' : callable function/class or identifier, \
                    'args' : {'value' : 0} }, with optional args but got",
                    value, "instead")

        for key, value in edges.items():
            if isinstance(value, str):
                self._make_edges(key, value)
            elif isinstance(value, dict):
                if 'class' not in value:
                    raise ValueError(" 'class' method must be defined in", value)
                if isinstance(value['class'], str):
                    args = value['args'] if 'args' in value else {}
                    self._make_edges(key, value['class'], **args)
                else:
                    # Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self, key=key, **args)
            else:
                raise TypeError(
                    "Method must be a dict of {'class' : callable function/class or identifier, \
                    'args' : {'value' : 0} }, with optinal args but got",
                    value, "instead")

        for key, value in state.items():
            if isinstance(value, str):
                self._make_state(key, value)
            elif isinstance(value, dict):
                if 'class' not in value:
                    raise ValueError(" 'class' method must be defined in", value)
                if isinstance(value['class'], str):
                    args = value['args'] if 'args' in value else {}
                    self._make_state(key, value['class'], **args)
                else:
                    # Custom function/class here
                    args = value['args'] if 'args' in value else {}
                    value['class'](self, key=key, **args)
            else:
                raise TypeError(
                    "Method must be a dict of {'class' : callable function/class or identifier, \
                    'args' : {'value' : 0} }, with optinal args but got",
                    value, "instead")

        return self

    def to_tensor(self,
                  nodes=None,
                  edges=None,
                  state=None,
                  trafo_nodes=None,
                  trafo_edges=None,
                  trafo_state=None,
                  default_nodes=None,
                  default_edges=None,
                  default_state=None,
                  out_tensor=np.array
                  ):
        """
        Convert the nx graph into a dict of tensors which can be directly used for GCN.
        
        The desired attributes must be given with a suitable conversion function plus default value.
        Here, one can add also the type of tensor or one-Hot mappings etc. and its default/zero state,
        if the attributes is not specified for a specific node/edge. The properties are always mapped to numpy arrays
        and then converted to out_tensor.

        Args:
            nodes (list, optional): Nodes properties. Defaults to ['proton'].
            edges (list, optional): Edge properties. Defaults to ['bond'].
            state (list, optional): State Properties. Defaults to ['size'].
            trafo_nodes (dict, optional): Transformation function for nodes. Defaults to np.array.
            trafo_edges (dict, optional): Transformation function for edges. Defaults to np.array.
            trafo_state (dict, optional): Transformation function for state. Defaults to np.array.
            default_nodes (dict, optional): Zero Nodes properties. Defaults to np.array(0).
            default_edges (dict, optional): Zero Edge properties. Defaults to np.array(0).
            default_state (dict, optional): Zero State Properties. Defaults to np.array(0).
            out_tensor (func) : Final Function for each node/edge/state. Default is np.array.

        Returns:
            dict: Graph tensors as dictionary.

        """
        if nodes is None:
            nodes = [self._mols_implemented[self.mol_type]['nodes'][0]]
        if edges is None:
            edges = [self._mols_implemented[self.mol_type]['edges'][0]]
        if state is None:
            state = [self._mols_implemented[self.mol_type]['state'][0]]

        if trafo_nodes is None:
            trafo_nodes = {}
        if trafo_edges is None:
            trafo_edges = {}
        if trafo_state is None:
            trafo_state = {}
        if default_nodes is None:
            default_nodes = {}
        if default_edges is None:
            default_edges = {}
        if default_state is None:
            default_state = {}

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
                default_nodes[x] = np.array(0.0)
        for x in edges:
            if x not in default_edges:
                default_edges[x] = np.array(0.0)
        for x in state:
            if x not in default_state:
                default_state[x] = np.array(0.0)

        outn = []
        oute = []
        outs = []
        out_a = nx.to_numpy_array(self)

        node_idx = np.array(list(self.nodes), dtype=np.int)
        edge_idx = np.array(list(self.edges), dtype=np.int)

        for i in node_idx:
            current_node = []
            for key in nodes:
                if key in self.nodes[i]:
                    current_node.append(trafo_nodes[key](self.nodes[i][key]))
                else:
                    current_node.append(default_nodes[key])
            outn.append(current_node)
        outn = np.array(outn)

        for i in edge_idx:
            current_edge = []
            for key in edges:
                if key in self.edges[i]:
                    current_edge.append(trafo_edges[key](self.edges[i][key]))
                else:
                    current_edge.append(default_edges[key])
            oute.append(current_edge)
        oute = np.array(oute)

        for key in state:
            if key in self._graph_state:
                outs.append(trafo_state[key](self._graph_state[key]))
            else:
                outs.append(default_state[key])
        outs = np.array(outs)

        # Make un-directed and sort edges and edge_index
        outei, oute = add_edges_reverse_indices(edge_idx,oute)

        return {"nodes": out_tensor(outn),
                "edges": out_tensor(oute),
                "state": out_tensor(outs),
                "adjacency": out_tensor(out_a),
                "indices": out_tensor(outei)}

# m = rdkit.Chem.MolFromSmiles("CC=O")
# test = MolGraph(m)
# test.make()
# nx.draw(test, with_labels=True)
# out = test.to_tensor()
