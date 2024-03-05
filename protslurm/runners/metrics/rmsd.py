'''Runner Module to calculate RMSDs'''
# import general
import json
import os

# import dependencies
import pandas as pd
import protslurm

# import customs
from protslurm.config import PROTSLURM_PYTHON as protslurm_python
from protslurm.config import AUXILIARY_RUNNER_SCRIPTS_DIR as script_dir
from protslurm.runners.runners import Runner, RunnerOutput
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter

class BackboneRMSD(Runner):
    '''Class handling the calculation of Full-atom RMSDs as a runner.
    By default calculates only CA backbone RMSD.
    Uses BioPython for RMSD calculation'''
    def __init__(self, atoms:list=["CA"], chains:list[str]=None, overwrite:bool=False, jobstarter_options:str=None):
        self.set_atoms(atoms)
        self.set_chains(chains)
        self.set_jobstarter_options(jobstarter_options)
        self.overwrite = overwrite

    ########################## Input ################################################
    def set_atoms(self, atoms:list[str]) -> None:
        '''Method to set the backbone atoms (list of atom names) to calculate RMSD over.'''
        if atoms == "all":
            self.atoms = "all"
        if not isinstance(atoms, list) or not all((isinstance(atom, str) for atom in atoms)):
            raise TypeError(f"Atoms needs to be a list, atom names (list elements) must be string.")
        self.atoms = atoms

    def set_chains(self, chains:list[str]) -> None:
        '''Method to set the chains (list of chain names) to calculate RMSD over.'''
        if chains is None:
            self.chains = None
        if not isinstance(chains, list) or not all((isinstance(chain, str) for chain in chains)):
            raise TypeError(f"Chains needs to be a list, chain names (list elements) must be string.")
        self.chains = chains

    def set_jobstarter_options(self, options: str) -> None:
        '''Sets Options for Jobstarter.'''
        self.jobstarter_options = options
    ########################## Calculations ################################################
    def calc_rmsd(self, poses:Poses, prefix:str, ref_col:str, jobstarter: JobStarter) -> None:
        '''Calculates RMSD as specified.'''
        # if self.atoms is all, calculate Allatom RMSD.

        # prep variables
        work_dir = f"{poses.work_dir}/{prefix}_rmsd/"
        scorefile = f"{poses.work_dir}/{work_dir}/{prefix}_rmsd.json"

        # check if RMSD was calculated if overwrite was not set.
        if os.path.isdir(work_dir): # check if dir exists
            if os.path.isfile(scorefile) and not self.overwrite: # return precalculated RMSDs
                output = RunnerOutput(
                    poses = poses,
                    results = pd.read_json(scorefile),
                    prefix = prefix
                )
                return output.return_poses()

            # if no outputs present, setup work_dir:
            os.makedirs(work_dir)

        # split poses into number of max_cores lists
        num_json_files = jobstarter.max_cores
        pose_dict = {row["poses"]: row[ref_col] for row in poses}
        pose_sublists = protslurm.jobstarters.split_list(poses.poses_list(), num_json_files)

        # setup inputs to calc_rmsd.py
        json_files = []
        cmds = []
        scorefiles = []
        for i, sublist in enumerate(pose_sublists, start=1):
            # create json dict:
            json_dict = {pose: pose_dict[pose] for pose in sublist}

            # write to file
            json_file = f"rmsd_input_{str(i)}"
            with open(json_file, "w", encoding="UTF-8") as f:
                json.dump(json_dict, f)
            json_files.append(json_file)

            # write scorefile and cmd
            scorefiles.append((sf := f"{work_dir}/rmsd_input_{str(i)}_scores.json"))
            cmds.append(f"{protslurm_python} {script_dir}/calc_rmsd.py --input_json {json_file} --out_path --output_path {sf}")

        # add options to cmds:
        if self.atoms:
            cmds = [cmd + f" --atoms {self.atoms}" for cmd in cmds]
        if self.chains:
            cmds = [cmd + f" --chains {self.chains}" for cmd in cmds]

        # run commands
        jobstarter.start(
            cmds = cmds,
            options = None,
            jobname = "backbone_rmsd",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        output_df = pd.concat([pd.read_json(sf) for sf in scorefiles], ignore_index=True)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = output_df,
            prefix = prefix,
        )
        return output.return_poses()

    def calc_all_atom_rmsd(self) -> None:
        raise NotImplementedError

class MotifRMSD(Runner):
    '''Class handling'''
    raise NotImplementedError