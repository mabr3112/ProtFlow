'''Script to test various runners
'''
import logging
import os
import shutil
import pandas as pd


# import jobstarters
from protflow.jobstarters import SbatchArrayJobstarter
from protflow.jobstarters import LocalJobStarter


# customs
from protflow.poses import Poses

# import runners
#from protflow.runners.protein_generator import ProteinGenerator
from protflow.tools.ligandmpnn import LigandMPNN
from protflow.tools.rosetta import Rosetta
from protflow.tools.rfdiffusion import RFdiffusion
from protflow.tools.attnpacker import AttnPacker
from protflow.tools.esmfold import ESMFold
from protflow.tools.colabfold import Colabfold

# import metrics
from protflow.metrics.protparam import ProtParam
from protflow.metrics.tmscore import TMalign
from protflow.metrics.tmscore import TMscore

# import config
from protflow import config
from protflow.utils.plotting import sequence_logo


#TODO: @Adrian: For Attnpacker, ligandmpnn and AF please write Tutorials as Jupyter Notebooks in 'examples' Folder.
#TODO: @Adrian: Please write tests for LigandMPNN running +10 sequences with/ and without pose_options (to test non_batch and batch run)


def check_runner(name:str, runner, poses_options:dict, runner_options:dict=None, config=None, check:str=None):

    if check == name or check == None:
        if runner == None:
            return "Not set up"
        logging.info(f"Running test for {name}...")
        work_dir = f"out_{name}"
        if os.path.isdir(work_dir): shutil.rmtree(work_dir)
        if config == None or all([os.path.isfile(conf) or os.path.isdir(conf) for conf in config]):
            try:
                logging.info(f"Initializing poses for {name}...")
                poses = Poses(**poses_options, work_dir=work_dir, storage_format="json")
                try:
                    logging.info(f"Running runner {name} with options {runner_options}...")
                    runner.run(**runner_options, poses=poses, prefix=f"test_{name}")
                    logging.info(f"Runner {name} passed!")
                    return "Working"
                except Exception as e:
                    logging.warning(f"Runner {name} failed! Error message: \n{e}")
                    return "Failed"
            except Exception as e:
                logging.warning(f"Loading poses for runner {name} failed! Error message: \n{e}")
                return "Loading poses failed!"
        else:
            logging.warning(f"Runner {name} config set up incorrectly!")
            return "Config fail"
    else:
        logging.info(f"Runner {name} not checked.")
        return "Not checked"
    

def main(args):

    '''run tests'''

    js_dict = {
        "slurm_gpu_jobstarter": SbatchArrayJobstarter(max_cores=10, gpus=1),
        "local_jobstarter": LocalJobStarter()
    }

    # set jobstarter
    if args.jobstarter:
        jobstarter = js_dict[args.jobstarter]
    else:
        jobstarter = None

    runner_dict = {
        "ESMFold": {
            "runner": ESMFold() if config.ESMFOLD_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/esmfold/", "glob_suffix": "*.fasta"},
            "runner_options": {"jobstarter": jobstarter or js_dict['slurm_gpu_jobstarter']},
            "config": [config.ESMFOLD_PYTHON_PATH]
        },
        "Rosetta": {
            "runner": Rosetta() if config.ROSETTA_BIN_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "rosetta_application": "rosetta_scripts.linuxgccrelease",
                "nstruct": 2,
                "options": "-parser:protocol input_files/rosettascripts/empty.xml -beta",
                "jobstarter": jobstarter
            },
            "config": [config.ROSETTA_BIN_PATH]
        },
        "AttnPacker": {
            "runner": AttnPacker() if config.ATTNPACKER_DIR_PATH and config.ATTNPACKER_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {"overwrite": True, "jobstarter": jobstarter},
            "config": [config.ATTNPACKER_DIR_PATH, config.ATTNPACKER_PYTHON_PATH]
        },
        "LigandMPNN": {
            "runner": LigandMPNN() if config.LIGANDMPNN_SCRIPT_PATH and config.LIGANDMPNN_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "model_type": "ligand_mpnn",
                "nseq": 2,
                "jobstarter": jobstarter
            },
            "config": [config.LIGANDMPNN_SCRIPT_PATH, config.LIGANDMPNN_PYTHON_PATH]
        },
        "RFdiffusion": {
            "runner": RFdiffusion() if config.RFDIFFUSION_SCRIPT_PATH and config.RFDIFFUSION_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/rfdiffusion/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "options": "diffuser.T=50 potentials.guide_scale=5 'contigmap.contigs=[Q1-21/0 20/A1-5/10-50/B1-5/10-50/C1-5/10-50/D1-5/20]' contigmap.length=200-200 'contigmap.inpaint_seq=[A1/A2/A4/A5/B1/B2/B4/B5/C1/C2/C4/C5/D1/D2/D4/D5]' potentials.substrate=LIG",
                "jobstarter": jobstarter
            },
            "config": [config.RFDIFFUSION_SCRIPT_PATH, config.RFDIFFUSION_PYTHON_PATH]
        },
        "Colabfold": {
            "runner": Colabfold() if config.COLABFOLD_SCRIPT_PATH else None,
            "poses_options": {"poses": "input_files/fastas/", "glob_suffix": "*.fasta"},
            "runner_options": {"jobstarter": jobstarter},
            "config": [config.COLABFOLD_SCRIPT_PATH]
        },
        "TMscore": {
            "runner": TMscore(),
            "poses_options": {"poses": pd.read_json("input_files/pose_df/pose_df.json")},
            "runner_options": {
                "ref_col": "reference",
                "jobstarter": jobstarter
            },
            "config": [config.RFDIFFUSION_SCRIPT_PATH, config.RFDIFFUSION_PYTHON_PATH]
        },
        "TMalign": {
            "runner": TMalign(),
            "poses_options": {"poses": pd.read_json("input_files/pose_df/pose_df.json")},
            "runner_options": {
                "ref_col": "reference",
                "jobstarter": jobstarter
            },
            "config": None
        },
        "ProtParam": {
            "runner": ProtParam(),
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "pH": 6,
                "jobstarter": jobstarter
            },
            "config": None
        }
    }
    test = LigandMPNN()

    if not config.AUXILIARY_RUNNER_SCRIPTS_DIR or not os.path.isdir(config.AUXILIARY_RUNNER_SCRIPTS_DIR):
        logging.warning(f"AUXILIARY_RUNNER_SCRIPTS_DIR was not properly set in config.py!")
        runner_dict['AUXILIARY_RUNNER_SCRIPTS_DIR'] = "NOT SET UP CORRECTLY!"

    if args.runner and not args.runner in runner_dict:
        raise KeyError(f"Runner must be one of {[i for i in runner_dict]}!")
    
    for runner in runner_dict:
        runner_dict[runner] = check_runner(name=runner, runner=runner_dict[runner]["runner"], poses_options=runner_dict[runner]["poses_options"], runner_options=runner_dict[runner]["runner_options"], config=runner_dict[runner]["config"], check=args.runner)

    print(runner_dict)
    df = pd.DataFrame(runner_dict, index=[0])
    df.to_csv('test_results.csv')

if __name__ == "__main__":


    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename="log.txt")
    import argparse
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--jobstarter", type=str, default=None, help="define jobstarter. can be either slurm_gpu_jobstarter or local_jobstarter, if None default slurm is used.")
    argparser.add_argument("--runner", type=str, default=None, help="Only test runner named <runner>. See runner_dict for available runners.")
    args = argparser.parse_args()
    #run main
    main(args)
