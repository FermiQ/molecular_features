# Molecular Descriptors

Molecular features for machine learning.

# Table of Contents
* [General](#general)
* [Installation](#installation)
* [Implementation details](#implementation-details)
* [Tests](#tests)
 

<a name="general"></a>
# General

This is a collection of tools to generate molecular features for machine learning, including common feautre representations like coulomb matrix etc. 
Also a graph generator for graph neural networks is located in molreps. This repo is built and can be easily expandend following the style:
* In molreps the main classes are listed
* In methods individual functions are collected that ideally should not depend on each other, so that they can be further used out of the box.
* We use a clear google-style doc string documentation for each function or class.
* Methods are sorted by their dependecies in modules.


<a name="installation"></a>
# Installation

Clone repository and install with editable mode:

```bash
pip install -e ./molecular_features
```

<a name="implementation-details"></a>
# Implementation details

Since there are many chemcial libraries in use, their dependencies should be focused, in order to run molreps, one must not install all external dependecies.
The methods are therefore sorted in modules that require specifically numpy, rdkit etc.

<a name="tests"></a>
# Tests

If added a new function, please supply a short test script in [test](/test) to demonstrate usage. Ideally without additional datasets but only python .py scripts.
