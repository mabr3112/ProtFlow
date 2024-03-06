'''Module to handle AttnPacker within ProtSLURM'''
# general imports
import os
import logging
from glob import glob
import shutil

# dependencies
import pandas as pd
import Bio
from Bio import SeqIO

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.tools
from protslurm.runners import Runner
from protslurm.runners import RunnerOutput


class AttnPacker(Runner):
    '''Class to run AttnPacker and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.ATTNPACKER_DIR_PATH, python_path:str=protslurm.config.ATTNPACKER_PYTHON_PATH, jobstarter_options:str=None) -> None:
        '''sbatch_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under ATTNPACKER_DIR_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under ATTNPACKER_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "attnpacker.py"
        self.jobstarter_options = jobstarter_options
        self.index_layers = 1

    def __str__(self):
        return "attnpacker.py"

    def run(self, poses:protslurm.poses.Poses, output_dir:str, prefix:str, options:str=None, pose_options:str=None, overwrite:bool=False, jobstarter:protslurm.jobstarters.JobStarter=None) -> RunnerOutput:
        '''Runs attnpacker.py on acluster'''

        # setup output_dir
        work_dir = os.path.abspath(output_dir)
        if not os.path.isdir(work_dir): os.makedirs(work_dir, exist_ok=True)
        pdb_dir = os.path.join(work_dir, 'output_pdbs')
        if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip LigandMPNN.
        scorefile = "attnpacker_scores.csv"
        if overwrite == False and os.path.isfile(scorefilepath := os.path.join(work_dir, scorefile)):
            return RunnerOutput(poses=poses, results=pd.read_csv(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()
        elif overwrite == True and os.path.isfile(scorefilepath := os.path.join(work_dir, scorefile)): os.remove(scorefilepath)

        # parse options and pose_options:
        pose_options = self.create_pose_options(poses.df, pose_options)

        # write attpacker cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df["poses"].to_list(), pose_options)]

        # run
        jobstarter = jobstarter or poses.default_jobstarter
        jobstarter_options = self.jobstarter_options or f"-c1 -e {work_dir}/attnpacker_err.log -o {work_dir}/attnpacker_out.log"
        jobstarter.start(cmds=cmds,
                         options=jobstarter_options,
                         jobname="attnpacker",
                         wait=True,
                         output_path=f"{output_dir}/"
        )

        scores = pd.read_csv(scorefilepath)

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def create_pose_options(self, df:pd.DataFrame, pose_options:list or str=None) -> list:
        '''Checks if pose_options are of the same length as poses, if pose_options are provided, '''

        def check_if_column_in_poses_df(df:pd.DataFrame, column:str):
            if not column in [col for col in df.columns]: raise ValueError(f"Could not find {column} in poses dataframe! Are you sure you provided the right column name?")
            return

        poses = df['poses'].to_list()


        if isinstance(pose_options, str):
            check_if_column_in_poses_df(df, pose_options)
            pose_options = df[pose_options].to_list()
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            pose_options = ["" for x in poses]

        if len(poses) != len(pose_options):
            raise ValueError(f"Arguments <poses> and <pose_options> for RFdiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
        
        return pose_options

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        '''Writes Command to run ligandmpnn.py'''

        pdb_dir = os.path.join(output_dir, "output_pdbs")

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {protslurm.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/run_attnpacker.py --attnpacker_dir {self.script_path} --output_dir {pdb_dir} --input_pdb {pose_path} --scorefile {output_dir}/attnpacker_scores.csv {opts} {flags}"


