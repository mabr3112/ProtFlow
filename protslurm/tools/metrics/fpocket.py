'''Module to run fpocket within protflow package'''
# imports
import os

import pandas as pd

# dependencies

# custom
from protslurm.jobstarters import JobStarter
from protslurm.poses import Poses
from protslurm.runners import Runner, RunnerOutput
from protslurm.config import FPOCKET_PATH

class FPocket(Runner):
    """Implements FPocket. Installation see: [...] """
    # class attributes
    index_layers = 0

    def __init__(self, fpocket_path: str = FPOCKET_PATH, jobstarter: JobStarter = None):
        if not fpocket_path:
            raise ValueError(f"No path was set for {self}. Set the path in the config.py file under FPOCKET_PATH!")
        self.jobstarter = jobstarter

    def __str__(self):
        return "fpocket"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter, options: str|list = None, pose_options: str|list = None, overwrite: bool = False) -> Poses:
        '''Implements writing of commands for fpocket and collecting scores into poses integratable DataFrames.'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for present outputs
        scorefile = os.path.join(work_dir, f"{prefix}_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        # prep options:
        options_l = self.prep_fpocket_options(options, pose_options)

        # compile cmds
        cmds = [self.write_fpocket_cmd(pose, opt) for pose, opt in zip(poses.poses_list(), options_l)]

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"fpocket_{prefix}",
        )

        # collect outputs and write scorefile
        scores = collect_fpocket_scores(work_dir)
        self.save_runner_scorefile(scores, scorefile)

        # itegrate and return
        outputs = RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()
        return outputs

    def prep_fpocket_options(self, options: str, pose_options: str|list[str]) -> str:
        '''Preps options from opts and pose_opts for fpocket run.'''
        # remove forbidden options

        # merge options and pose_options, with pose_options priority and return
        return NotImplemented

    def write_fpocket_cmd(self, pose: str, opt: str) -> str:
        '''Writes command that runs fpocket on 'pose' with commandline options specified in 'opt'.'''
        return NotImplemented

def collect_fpocket_scores(output_dir: str) -> pd.DataFrame:
    '''Collects fpocket scores from an output directory for one- or multiple fpocket runs.'''
    return NotImplemented
