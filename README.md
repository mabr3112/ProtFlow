[![Run tests](https://github.com/mabr3112/ProtFlow/actions/workflows/pytest.yaml/badge.svg?branch=master)](https://github.com/mabr3112/ProtFlow/actions/workflows/pytest.yaml)

# ProtFlow
A Python package for running protein design tools on SLURM clusters.

# Installation
First, clone the repository

```
git clone https://github.com/mabr3112/ProtFlow.git
```

ProtFlow requires python >= 3.11. You can either install it into an existing environment, or you can create a new one with like this:

```
conda create -n protflow python=3.11
conda activate protflow
```

Install the package in development mode (we are currently under active development).

```
cd ProtFlow
pip install -e .
```

After the install, instantiate your config.py file. This can be done with the command below, which will copy the config.py file from protflow/config_template.py

```
protflow-init-config
```

# The Configuration File
You will need to link protflow's runners to the individual python environments you have set up on your system.
By Default, ProtFlow looks for environment paths in the file 'config.py' which we instantiated during the installation above.
Add the paths of your tool's python evironments and their scripts into this file how it is described in the module's docstring.
If you try to initialize a runner for which you have not set paths in your config.py file, protflow will throw an error directing you to where you will need to set the required paths.

By default ``protflow-init-config`` will set up config.py in ~/.config/protflow/config.py.
If you want to link your protflow installation to a different config.py file elswhere, there is a simple cli-tool to achieve this, ``protflow-set-config``.
As an example, linking your protflow install to a different config.py can come in handy when you want to share one config.py system wide on a computing clusters with multiple users.

Simply provide the path to the target config.py to the tool:
```
protflow-set-config /path/to/your/confit.py
```

If you want to unset this custom override, supply the --unset flag:
```
protflow-set-config --unset
```

Finally, to check which config.py file your protflow is currently using, simply use:
```
protflow-check-config
```


# Overview
![protflow_organigramm drawio (1)](https://github.com/TecnomaLaser/ProtFlow/assets/45593003/3842712a-2399-4e3c-9c90-1525ad6b6690)

# Documentation
ProtFlow is documented with read-the-docs and sphinx. The documentation can be found here:
https://protflow.readthedocs.io/en/latest/
