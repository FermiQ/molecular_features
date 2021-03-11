[![Documentation Status](https://readthedocs.org/projects/molreps/badge/?version=latest)](https://molreps.readthedocs.io/en/latest/?badge=latest)

# Molecular Descriptors

Molecular features for machine learning.

# Table of Contents
* [General](#general)
* [Installation](#installation)
* [Documentation](#documentation)
* [Implementation details](#implementation-details)
* [Usage](#usage)
* [Examples](#examples)
* [Citing](#citing)
* [References](#references)


 

<a name="general"></a>
# General

This is a collection of methods to generate molecular features for machine learning, including common feautre representations like coulomb matrix etc. 
Also a graph generator for graph neural networks is found in molreps. This repo is currently under construction and can be easily expandend following this recommended style:
* In molreps the main classes are listed
* In methods individual functions are collected that ideally should have limited dependencies, so that they can be further used out of the box.
* Use a google-style doc string documentation for each function or class.
* Methods are sorted by their dependecies in modules.


<a name="installation"></a>
# Installation

Clone repository https://github.com/aimat-lab/molecular_features and install with editable mode:

```bash
pip install -e ./molecular_features
```

or latest release via:
```bash
pip install molreps
```
<a name="documentation"></a>
# Documentation

Auto-documentation is generated at: https://molreps.readthedocs.io/en/latest/index.html .

<a name="implementation-details"></a>
# Implementation details

Since there are many chemcial libraries in use, their dependencies should be flexible. In order to install molreps, external dependecies are not required.
The methods are therefore sorted in modules that require specifically numpy, rdkit etc. However, further external libraries have to be installed manually for certain modules.
That include following optional dependencies:

- networkx
- rdkit
- openpapel
- ase

<a name="usage"></a>
# Usage

## Representations
Simple moleculear representations can be generated from `molreps.descriptors`.

```python
from molreps.descriptors import coulomb_matrix
atoms = ['C','C']
coords = [[0,0,0],[1,0,0]]
cm = coulomb_matrix(atoms,coords)
```

However, also individual function can be used from `molreps.methods`. Like in this case the back-direction.

```python
from molreps.methods.geo_npy import geometry_from_coulombmat
atom_cord = geometry_from_coulombmat(cm)
```

## Graph
For many ML models a graph representation of the molecule is required. The module `MolGraph` from `molreps.graph`
inherits from networkx's `nx.Graph` and can generate a molecular graph based on a mol-object provided by a cheminformatics package like rdkit, openbabel, ase etc. 
This is a flexible way to use functionalities from both networkx and packages like rdkit. First create a mol object.

```python
import rdkit
m = rdkit.Chem.MolFromSmiles("CC1=CC=CC=C1")
m = rdkit.Chem.AddHs(m)
```

The mol object is passed to the MolGraph class constructor but can be further accessed. 

```python
import networkx as nx
import numpy as np
from molreps.graph import MolGraph
mgraph = MolGraph(m)
mgraph.mol  # Access the mol object.
```

The networkx graph is generated by `make()`, where the features and keys can be specified. There are pre-defined features
that can be assigned by an identifier like `'key': 'identifier'` or if further arguments are required by
`'key' : {'class':'identifier', 'args':{'arg1': value1,'arg2': value2 }}`. In the latter case also a custom function or class can be 
provided like `'key' : {'class': my_fun, 'args':{'arg1': value1,'arg2': value2 }}`. A dictionary of predifined identifiers is listed in `print(MolGraph._mols_implemented)`.

```python
mgraph.make()
mgraph.make(nodes = {"proton" : 'proton'},
            edges = {"bond" : 'bond',
                     "distance" : {'class':'distance', 'args':{'bonds_only':True}}},
            state = {"mol_weight" : 'mol_weight'}
            )
```
Note, a custom function must accept `key`,`mol` as arguments and return a list of tuples such as `[(i, {key: property})]`for atoms and `[((i,j, {key: property}))]` for bonds such that it can be read by 
`add_nodes_from()` or `add_edges_from()`, respectively. Then the generated graph can be viewed and treated as a networkx graph, like plotting `nx.draw(mgraph,with_labels=True)`.
Finnaly, a closed form tensor is collected from selected features defined by the key-attribute. 
For each key an additional function to process the features and a default value can be optionally provided but defaults to `np.array`.
A default value has to be added, if a single node or edge is missing a key, to generate a closed form tensor.

```python
mgraph.to_tensor()
graph_tensors= mgraph.to_tensor(nodes = ["proton"],
                                edges = ["bond" ],
                                state = ["mol_weight"],
                                out_tensor = np.array)
```

The returned dictionary containing the feature tensors can be passed to graph models.


<a name="examples"></a>
# Examples

<a name="citing"></a>
# Citing

<a name="references"></a>
# References
