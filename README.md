# ProtSLURM
A Python package for running protein design tools on SLURM clusters.


# The Configuration File
You will need to link protslurm's runners to the individual python environments you have set up on your system.
By Default, ProtSLURM looks for environment paths in the file 'config.py'.
Add the paths of your tool's python evironments and their scripts into this file how it is described in the module's docstring.