'''Script to test various runners
'''
import logging
import os
import shutil
from protslurm.jobstarters import SbatchArrayJobstarter
from protslurm.jobstarters import LocalJobStarter

import protslurm.config
# customs
from protslurm.poses import Poses
#from protslurm.runners.protein_generator import ProteinGenerator
from protslurm.tools.ligandmpnn import LigandMPNN
from protslurm.tools.rosetta import Rosetta
from protslurm.tools.rfdiffusion import RFdiffusion
from protslurm.tools.attnpacker import AttnPacker
from protslurm.tools.esmfold import ESMFold
from protslurm.tools.metrics.protparam import ProtParam
from protslurm.tools.metrics.tmscore import TMalign
from protslurm.tools.metrics.tmscore import TMscore
from protslurm.utils.plotting import sequence_logo



#TODO: Print Test output: Which Runners are Implemented, Which runners succeeded (all if the test runs through).
#TODO: @Adrian: For Attnpacker, ligandmpnn and AF please write Tutorials as Jupyter Notebooks in 'examples' Folder.
#TODO: @Adrian: Please write tests for LigandMPNN running +10 sequences with/ and without pose_options (to test non_batch and batch run)


def check_runner(name:str, runner, poses_options:dict, runner_options:dict=None, config=None, check:str=None):

    if check == runner or check == None:
        if runner == None:
            return "Not set up"
        logging.info(f"Running test for {name}...")
        work_dir = f"out_{name}"
        if os.path.isdir(work_dir): shutil.rmtree(work_dir)
        if config == None or all([os.path.isfile(conf) or os.path.isdir(conf) for conf in config]):
            try:
                logging.info(f"Initializing poses for {name}...")
                poses = Poses(**poses_options, work_dir=work_dir, storage_format="json")
            except:
                return "Loading poses failed!"
            try:
                logging.info(f"Running runner {name}...")
                runner.run(**runner_options, poses=poses, prefix="test")
                logging.info(f"Runner {name} passed!")
                return "Passed"
            except:
                logging.warning(f"Runner {name} failed!")
                return "Failed"
        else:
            logging.warning(f"Runner {name} config set up incorrectly!")
            return "Config fail"
    else:
        logging.info(f"Runner {name} not checked.")
        return "Not checked"
    

def main(args):

    '''.'''

    js_dict = {
        "slurm_gpu_jobstarter": SbatchArrayJobstarter(max_cores=10, gpus=1, options="-c1"),
        "local_jobstarter": LocalJobStarter()
    }

    if args.jobstarter:
        jobstarter = js_dict[args.jobstarter]
    else:
        jobstarter = None

    runner_dict = {
        "ESMFold": {
            "runner": ESMFold() if protslurm.config.ESMFOLD_PYTHON_PATH and protslurm.config.ESMFOLD_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/fastas/", "glob_suffix": "*.fasta"},
            "runner_options": {"jobstarter": jobstarter},
            "config": [protslurm.config.ESMFOLD_PYTHON_PATH, protslurm.config.ESMFOLD_PYTHON_PATH]
        },
        "Rosetta": {
            "runner": Rosetta() if protslurm.config.ROSETTA_BIN_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "rosetta_application": "rosetta_scripts.linuxgccrelease",
                "nstruct": 5,
                "options": "-parser:protocol input_files/rosettascripts/empty.xml -beta",
                "jobstarter": jobstarter
            },
            "config": [protslurm.config.ROSETTA_BIN_PATH]
        },
        "AttnPacker": {
            "runner": AttnPacker() if protslurm.config.ATTNPACKER_DIR_PATH and protslurm.config.ATTNPACKER_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {"overwrite": True, "jobstarter": jobstarter},
            "config": [protslurm.config.ATTNPACKER_DIR_PATH, protslurm.config.ATTNPACKER_PYTHON_PATH]
        },
        "LigandMPNN": {
            "runner": LigandMPNN() if protslurm.config.LIGANDMPNN_SCRIPT_PATH and protslurm.config.LIGANDMPNN_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/pdbs/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "model_type": "ligand_mpnn",
                "nseq": 2,
                "jobstarter": jobstarter
            },
            "config": [protslurm.config.LIGANDMPNN_SCRIPT_PATH, protslurm.config.LIGANDMPNN_PYTHON_PATH]
        },
        "RFdiffusion": {
            "runner": RFdiffusion() if protslurm.config.RFDIFFUSION_SCRIPT_PATH and protslurm.config.RFDIFFUSION_PYTHON_PATH else None,
            "poses_options": {"poses": "input_files/rfdiffusion/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "options": "diffuser.T=50 potentials.guide_scale=5 'contigmap.contigs=[Q1-21/0 20/A1-5/10-50/B1-5/10-50/C1-5/10-50/D1-5/20]' contigmap.length=200-200 'contigmap.inpaint_seq=[A1/A2/A4/A5/B1/B2/B4/B5/C1/C2/C4/C5/D1/D2/D4/D5]' potentials.substrate=LIG",
                "jobstarter": jobstarter
            },
            "config": [protslurm.config.RFDIFFUSION_SCRIPT_PATH, protslurm.config.RFDIFFUSION_PYTHON_PATH]
        },
        "TMscore": {
            "runner": TMscore(),
            "poses_options": {"poses": "input_files/rfdiffusion/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "jobstarter": jobstarter
            },
            "config": [protslurm.config.RFDIFFUSION_SCRIPT_PATH, protslurm.config.RFDIFFUSION_PYTHON_PATH]
        },
        "TMalign": {
            "runner": TMalign(),
            "poses_options": {"poses": "input_files/rfdiffusion/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "jobstarter": jobstarter
            },
            "config": None
        },
        "ProtParam": {
            "runner": ProtParam(),
            "poses_options": {"poses": "input_files/rfdiffusion/", "glob_suffix": "*.pdb"},
            "runner_options": {
                "pH": 6,
                "jobstarter": jobstarter
            },
            "config": None
        }
    }

    if protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR == "" or not os.path.isdir(protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR):
        logging.warning(f"AUXILIARY_RUNNER_SCRIPTS_DIR was not properly set in config.py!")
        runner_dict['AUXILIARY_RUNNER_SCRIPTS_DIR'] = "NOT SET UP CORRECTLY!"

    if args.runner and not args.runner in runner_dict:
        raise KeyError(f"Runner must be one of {[i for i in runner_dict]}!")
    


    for runner in runner_dict:
        runner_dict[runner] = check_runner(name=runner, runner=runner_dict[runner]["runner"], poses_options=runner_dict[runner]["poses_options"], runner_options=runner_dict[runner]["runner_options"], config=runner_dict[runner]["config"], check=args.runner)

    print(runner_dict)

'''



####################### Sequence Logo #######################
    
    if not args.runner or args.runner == "SequenceLogo":
        logging.info("Running test for sequence logo generation...")
        try:
            out_dir = "output_seqlogo"
            if os.path.isdir(out_dir): shutil.rmtree(out_dir)

            proteins = Poses(
                poses="input_files/esmfold/",
                glob_suffix="*.fasta",
                work_dir=out_dir,
                storage_format="json"
            )

            sequence_logo(dataframe=proteins.df, input_col="poses", save_path=os.path.join(out_dir, "seq.logo"), refseq="AS", title=None, resnums=[1,2], units="probability")

            logging.info("ESMFold passed!")
            runner_dict['SequenceLogo'] = "Passed"
        except:
            runner_dict['SequenceLogo'] = "Failed"


####################### TMalign #######################

    logging.info("Running test for TMalign...")
    out_dir = "output_TMalign"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="csv",
        jobstarter=LocalJobStarter()
    )
    proteins.df['ref_col'] = proteins.df['poses']
    proteins = TMalign().run(poses=proteins, prefix="test", ref_col="ref_col", overwrite=True, jobstarter=jobstarter)


    logging.info("TMalign passed!")


####################### TMscore #######################

    logging.info("Running test for TMscore...")
    out_dir = "output_TMscore"
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)

    proteins = Poses(
        poses="input_files/rosettascripts/",
        glob_suffix="*.pdb",
        work_dir=out_dir,
        storage_format="csv",
        jobstarter=LocalJobStarter()
    )

    proteins = TMscore().run(poses=proteins, prefix="test", ref_col="poses", overwrite=True, jobstarter=jobstarter)


    logging.info("TMalign passed!")


'''

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