'''Module to handle AttnPacker within ProtConductor'''
# general imports
import os
import logging

# dependencies
import pandas as pd

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.tools
from protslurm.poses import Poses
from protslurm.runners import Runner, RunnerOutput
from protslurm.jobstarters import JobStarter

class AttnPacker(Runner):
    '''Class to run AttnPacker and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.ATTNPACKER_DIR_PATH, python_path:str=protslurm.config.ATTNPACKER_PYTHON_PATH, jobstarter:str=None) -> None:
        '''sbatch_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under ATTNPACKER_DIR_PATH.")
        if not python_path:
            raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under ATTNPACKER_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "attnpacker.py"
        self.jobstarter = jobstarter
        self.index_layers = 1

    def __str__(self):
        return "attnpacker.py"

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter, options:str=None, pose_options:str=None, overwrite:bool=False) -> Poses:
        '''Runs attnpacker.py on acluster'''
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # setup attnpacker specific dirs:
        pdb_dir = os.path.join(work_dir, 'output_pdbs')
        if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip LigandMPNN.
        scorefile = "attnpacker_scores.csv"
        if not overwrite and os.path.isfile(scorefilepath := os.path.join(work_dir, scorefile)):
            return RunnerOutput(poses=poses, results=pd.read_csv(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()
        elif overwrite and os.path.isfile(scorefilepath := os.path.join(work_dir, scorefile)): 
            os.remove(scorefilepath)

        # parse options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # write attpacker cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df["poses"].to_list(), pose_options)]

        # run:
        logging.info(f"Starting attnpacker.py on {len(poses)} poses with {len(jobstarter.max_cores)} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="attnpacker",
            wait=True,
            output_path=f"{work_dir}/"
        )

        logging.info(f"attnpacker.py finished, collecting scores.")
        scores = pd.read_csv(scorefilepath)
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        '''Writes Command to run ligandmpnn.py'''
        pdb_dir = os.path.join(output_dir, "output_pdbs")

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/run_attnpacker.py --attnpacker_dir {self.script_path} --output_dir {pdb_dir} --input_pdb {pose_path} --scorefile {output_dir}/attnpacker_scores.csv {opts} {flags}"
