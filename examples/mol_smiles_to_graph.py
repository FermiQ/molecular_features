import numpy as np
import rdkit
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from molreps.graph import MolGraph
import networkx as nx


# Start with a smiles code from database
smile = "C#CC1(COC(=O)c2noc(C)n2)CC1"
m = rdkit.Chem.MolFromSmiles(smile)
m = rdkit.Chem.AddHs(m) # add H's to the molecule
Draw.ShowMol(m)

# If no coordinates are known, do embedding with rdkit
AllChem.EmbedMolecule(m)
AllChem.MMFFOptimizeMolecule(m,maxIters=200)

# Make mol graph
mgraph = MolGraph(m).make(nodes={"chiral": "chiral","in_ring" : "in_ring"})
print(mgraph._mols_implemented)
nx.draw(mgraph,with_labels=True)
graph_tensors = mgraph.to_tensor()