.. ProtFlow documentation master file, created by
   sphinx-quickstart on Wed Jun  1 2024.

.. _home:

Welcome to ProtFlow's documentation!
====================================

ProtFlow is a python package to automate protein design workflows and enhance your productivity. ProtFlow is essentially a python wrapper around common protein design tools that allows their seamless integration into automated pipelines. ProtFlow uses JobStarters to run protein design tools on different platforms such as slurm based computing clusters or local machines. If you want a JobStarter tailored to your specific computing system please contact us!

Key Features
^^^^^^^^^^^^

- **The Poses class**: Provides a container for your proteins and stores all information collected during your design pipeline in pandas DataFrames.
- **Runners**: Implement protein design tools. Want to include your favorite tool? Just implement a Runner class and ProtFlow will take care of the rest.
- **JobStarters**: Handle execution of design tools on your computing system. You can tailor JobStarters to your system and swiftly plug them into any ProtFlow pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: ProtFlow Tools:

   tools/index

.. toctree::
   :maxdepth: 2
   :caption: ProtFlow API docs:

   protflow
