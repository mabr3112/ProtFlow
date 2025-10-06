'''
This module contains all paths to tools integrated in ProtFlow. PRE_CMD are commands that should be run before the runner is executed (e.g. if import of a specific module is necessary for the environment to work)
'''
# path to root directory of ProtFlow:
PROTFLOW_DIR = "" # "/path/to/ProtFlow/"

# protflow environment python
PROTFLOW_ENV = "" # "/path/to/package_manager/envs/protflow/bin/python3"

# auxiliary runners directory path
#TODO: @Markus Implement checkup if AUXILIARY_RUNNER_SCRIPTS_DIR points to the correct directory (Think about pointing to protflow directory)!
AUXILIARY_RUNNER_SCRIPTS_DIR = "" # "/path/to/ProtFlow/protflow/tools/runners_auxiliary_scripts/"

# protein_generator.py
PROTEIN_GENERATOR_SCRIPT_PATH = ""

# ligandmpnn.py
LIGANDMPNN_SCRIPT_PATH = "" # "/path/to/LigandMPNN/run.py"
LIGANDMPNN_PYTHON_PATH = "" # "/path/to/anaconda3/envs/ligandmpnn_env/bin/python3"
LIGANDMPNN_PRE_CMD = "" # "echo 'this will be printed before running ligandmpnn'"

# rosetta.py
ROSETTA_BIN_PATH = "" # "/path/to/Rosetta/main/source/bin/"
ROSETTA_PRE_CMD = ""

# attnpacker.py
ATTNPACKER_PYTHON_PATH = "" # "/path/to/anaconda3/envs/attnpacker/bin/python3"
ATTNPACKER_DIR_PATH = "" # "/path/to/AttnPacker/"
ATTNPACKER_PRE_CMD = ""


# rfdiffusion.py
RFDIFFUSION_SCRIPT_PATH = "" # "/path/to/RFdiffusion/scripts/run_inference.py"
RFDIFFUSION_PYTHON_PATH = "" # "/path/to/miniconda3/envs/SE3nv/bin/python"
RFDIFFUSION_PRE_CMD = ""

# esmfold.py
ESMFOLD_PYTHON_PATH = "" # "/path/to/miniconda3/envs/esm/bin/python"
ESMFOLD_PRE_CMD = ""

# af2
COLABFOLD_SCRIPT_PATH = "" # "/path/to/localcolabfold/colabfold-conda/bin/colabfold_batch"
COLABFOLD_PRE_CMD = ""

# fpocket
FPOCKET_PATH = "" # "/path/to/anaconda3/envs/protslurm/bin/fpocket"

# dssp
DSSP_PATH = "" # "/path/to/mambaforge/envs/dssp/bin/mkdssp"

# boltz
BOLTZ_PATH = "" # /path/to/CONDA/envs/boltz_env/bin/boltz
BOLTZ_PYTHON = "" # /path/to/conda/envs/boltz_env/bin/python
BOLTZ_PRE_CMD = ""

# gromacs
GROMACS_PATH = "" #/path/to/gromacs/bin/

# PLACER
PLACER_SCRIPT_PATH = "" # "/path/to/PLACER/run_PLACER.py"
PLACER_PYTHON_PATH = "" # "/path/to/mambaforge/envs/placer/bin/python"
PLACER_PRE_CMD = ""

# ESM
ESM_PYTHON_PATH = "" # "/path/to/package_manager/envs/your_esm_env/bin/python"
ESM_PRE_CMD = ""

# protein generator
PROTEIN_GENERATOR_PYTHON_PATH = ""
PROTEIN_GENERATOR_SCRIPT_PATH = ""
