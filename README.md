[![Run tests](https://github.com/mabr3112/ProtFlow/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/mabr3112/ProtFlow/actions/workflows/pytest.yaml)

# ProtFlow
A Python package for running protein design tools on SLURM clusters.

# Installation
First, clone the repository

```
git clone https://github.com/mabr3112/ProtFlow.git
```

Create a conda environment for ProtFlow:

```
conda create -n protflow python=3.11
```

Then, install in development mode (Package currently under development and changes a lot.)
Install in the protflow environment to limit interference with other systems.

```
cd ProtFlow
conda activate protflow
pip install -e .
```

# The Configuration File
You will need to link protflow's runners to the individual python environments you have set up on your system.
By Default, ProtFlow looks for environment paths in the file 'config.py'.
Add the paths of your tool's python evironments and their scripts into this file how it is described in the module's docstring.

# Overview
![protflow_organigramm drawio (1)](https://github.com/TecnomaLaser/ProtFlow/assets/45593003/3842712a-2399-4e3c-9c90-1525ad6b6690)

# Documentation
ProtFlow is documented with read-the-docs and sphinx. The documentation can be found here:
https://protflow.readthedocs.io/en/latest/
