'''Module to handle AttnPacker within ProtConductor'''
# general imports
import os
import logging

# dependencies
import pandas as pd

# custom
import protflow.config
import protflow.jobstarters
import protflow.tools
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput
from protflow.jobstarters import JobStarter

class AttnPacker(Runner):
    '''Class to run AttnPacker and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protflow.config.ATTNPACKER_DIR_PATH, python_path:str=protflow.config.ATTNPACKER_PYTHON_PATH, jobstarter:str=None) -> None:
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

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter=None, options:str=None, pose_options:str=None, overwrite:bool=False) -> Poses:
        '''Runs attnpacker.py on acluster'''
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # setup attnpacker specific dirs:
        pdb_dir = os.path.join(work_dir, 'output_pdbs')
        if not os.path.isdir(pdb_dir): os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip attnpacker.
        scorefile = os.path.join(work_dir, f"attnpacker_scores.{poses.storage_format}")

        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            if overwrite:
                logging.warning(f"Removing previously generated scorefile at {scorefile}")
                os.remove(scorefile)
            else:
                logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
                return RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()

        # parse options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # write attpacker cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df["poses"].to_list(), pose_options)]

        # run:
        logging.info(f"Starting attnpacker.py on {len(poses)} poses with {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="attnpacker",
            wait=True,
            output_path=f"{work_dir}/"
        )

        logging.info(f"{self} finished, collecting scores.")
        # TODO: this is not done gracefully, too lazy to fix atm
        scores = pd.read_csv(scorefile)
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        '''Writes Command to run attnpacker'''

        # check if interfering options were set
        forbidden_options = ['--attnpacker_dir', '--output_dir', '--input_pdb', '--scorefile']
        if (options and any(_ in options for _ in forbidden_options)) or (pose_options and any(_ in pose_options for _ in forbidden_options)):
            raise KeyError(f"Options and pose_options must not contain '--attnpacker_dir', '--output_dir', '--input_pdb' or '--scorefile'!")

        pdb_dir = os.path.join(output_dir, "output_pdbs")
        if options:
            options = options + f" --attnpacker_dir {self.script_path} --output_dir {pdb_dir} --input_pdb {pose_path} --scorefile {output_dir}/attnpacker_scores.csv"
        else:
            options = f"--attnpacker_dir {self.script_path} --output_dir {pdb_dir} --input_pdb {pose_path} --scorefile {output_dir}/attnpacker_scores.csv"

        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options)

        return f"{self.python_path} {protflow.config.AUXILIARY_RUNNER_SCRIPTS_DIR}/run_attnpacker.py {protflow.runners.options_flags_to_string(opts, flags, sep='--')}"
