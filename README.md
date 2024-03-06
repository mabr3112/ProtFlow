# ProtSLURM
A Python package for running protein design tools on SLURM clusters.

# Installation
First, clone the repository

>>> git clone git@github.com:mabr3112/ProtSLURM.git

Create a conda environment for ProtSLURM:

>>> conda create -n protslurm python=3.11

Then, install in development mode (Package currently under development and changes a lot.)
Install in the protslurm environment to limit interference with other systems.

>>> conda activate protslurm
>>> pip install -e .

# The Configuration File
You will need to link protslurm's runners to the individual python environments you have set up on your system.
By Default, ProtSLURM looks for environment paths in the file 'config.py'.
Add the paths of your tool's python evironments and their scripts into this file how it is described in the module's docstring.