.. ProtFlow documentation master file, created by
   sphinx-quickstart on Wed Jun  1 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ProtFlow's documentation!
====================================

Introduction
------------

ProtFlow is a python package designed to streamline protein design workflows and enhance your productivity. ProtFlow is essentially a python wrapper around common protein design tools that allows their seamless integration into automated pipelines. ProtFlow uses JobStarters to run protein design tools on different platforms such as slurm based computing clusters or local machines. If you want a JobStarter tailored to your specific computing system please contact us!

Key Features
^^^^^^^^^^^^

- **Protein Design Toolbox**: Wraps around protein design tools and facilitates input and output management into pandas DataFrames.
- **Integrated Protein Metrics**: ProtFlow implements several common metrics used in protein design such as RMSD and TMscore.
- **Integrated Plotting**: Quick and easy to use plotting functions for quick check-up of your design success.

Contents:
---------

.. toctree::
   :caption: Contents:

   protflow

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
