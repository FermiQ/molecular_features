.. _intro:
   :maxdepth: 3

Introduction
============


This is a collection of methods to generate molecular features for machine learning, including common feautre representations like coulomb matrix etc. 
Also a graph generator for graph neural networks is found in molreps. This repo is built and can be easily expandend following this recommended style:

- In molreps the main classes are listed
- In methods individual functions are collected that ideally should have limited dependencies, so that they can be further used out of the box.
- Use a google-style doc string documentation for each function or class.
- Methods are sorted by their dependecies in modules.