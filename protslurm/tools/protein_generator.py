'''Module to handle protein_generator within ProtSLURM'''
# general imports
import os
import logging
from glob import glob
import numpy as np

# dependencies
import pandas as pd

# custom
import protslurm.config
import protslurm.jobstarters
import protslurm.tools
from protslurm.runners import Runner, RunnerOutput
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter

class ProteinGenerator(Runner):
    '''Class to run protein_generator and collect it's outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.PROTEIN_GENERATOR_SCRIPT_PATH, python_path:str=protslurm.config.PROTEIN_GENERATOR_PYTHON_PATH, jobstarter:JobStarter=None) -> None:
        '''sbatch_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path:
            raise ValueError(f"No path is set for {self}. Set the path in the config.py file under PROTEIN_GENERATOR_SCRIPT_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "protein_generator.py"
        self.jobstarter = jobstarter
        self.index_layers = 1

    def __str__(self):
        return "protein_generator.py"

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter, options:str=None, pose_options:str=None, overwrite:bool=False) -> RunnerOutput:
        '''Runs protein_generator.py on acluster'''
        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        
        # setup protein_generator specific directories:
        if not os.path.isdir((pdb_dir := f"{work_dir}/output_pdbs/")):
            os.makedirs(pdb_dir, exist_ok=True)

        # Look for output-file in pdb-dir. If output is present and correct, then skip protein_generator.
        scorefile = os.path.join(work_dir, f"protein_generator_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # parse_options and pose_options:
        pose_options = self.prep_pose_options(poses, pose_options)

        # write protein generator cmds:
        cmds = [self.write_cmd(pose, output_dir=pdb_dir, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses, pose_options)]

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="protein_generator",
            wait=True,
            output_path=f"{pdb_dir}/"
        )

        # collect scores
        scores = self.collect_scores(scores_dir=pdb_dir)
    
        # write scorefile
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def safecheck_pose_options(self, pose_options: list, poses:list) -> list:
        '''Checks if pose_options are of the same length as poses, if now pose_options are provided, '''
        # safety check (pose_options must have the same length as poses)
        if isinstance(pose_options, list):
            if len(poses) != len(pose_options):
                raise ValueError(f"Arguments <poses> and <pose_options> for RFDiffusion must be of the same length. There might be an error with your pose_options argument!\nlen(poses) = {poses}\nlen(pose_options) = {len(pose_options)}")
            return pose_options
        if pose_options is None:
            # make sure an empty list is passed as pose_options!
            return ["" for x in poses]
        raise TypeError(f"Unsupported type for pose_options: {type(pose_options)}. pose_options must be of type [list, None]")

    def write_cmd(self, pose_path:str, output_dir:str, options:str, pose_options:str):
        '''Writes Command to run protein_generator.py'''
        # parse description
        desc = pose_path.rsplit("/", maxsplit=1)[-1].lsplit(".", maxsplit=1)[0]

        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --".join(flags)

        return f"{self.python_path} {self.script_path} --out {output_dir}/{desc} {opts} {flags}"

    def collect_scores(self, scores_dir: str) -> pd.DataFrame:
        '''collects scores from protein_generator output'''
        # read .pdb files
        pl = glob(f"{scores_dir}/*.pdb")
        if not pl:
            raise FileNotFoundError(f"No .pdb files were found in the output directory of protein_generator {scores_dir}. protein_generator might have crashed (check output log), or path might be wrong!")

        # parse .trb-files into DataFrames
        df = pd.concat([self.parse_trbfile(p.replace(".pdb", ".trb")) for p in pl], axis=0).reset_index(drop=True)

        return df

    def parse_trbfile(self, trbfile: str) -> pd.DataFrame:
        '''Reads protein_generator output .trb file and parses the scores into a pandas DataFrame.'''
        trb = np.load(trbfile, allow_pickle=True)

        # expand collected data if needed:
        data_dict = {
            "description": trbfile.split("/")[-1].replace(".trb", ""),
            "location": trbfile.replace("trb", "pdb"),
            "lddt": [sum(trb["lddt"]) / len(trb["lddt"])],
            "perres_lddt": [trb["lddt"]],
            "sequence": trb["args"]["sequence"],
            "contigs": trb["args"]["contigs"],
            "inpaint_str": [trb["inpaint_str"].numpy().tolist()],
            "inpaint_seq": [trb["inpaint_seq"].numpy().tolist()]
        }
        return pd.DataFrame(data_dict)
