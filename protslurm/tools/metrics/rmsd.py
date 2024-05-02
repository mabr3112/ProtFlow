'''Runner Module to calculate RMSDs'''
# import general
import json
import os
from typing import Any

# import dependencies
import pandas as pd
import protslurm

# import customs
from protslurm.config import PROTSLURM_PYTHON as protslurm_python
from protslurm.config import AUXILIARY_RUNNER_SCRIPTS_DIR as script_dir
from protslurm.residues import ResidueSelection
from protslurm.runners import Runner, RunnerOutput, col_in_df
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter, split_list

class BackboneRMSD(Runner):
    '''Class handling the calculation of Full-atom RMSDs as a runner.
    By default calculates only CA backbone RMSD.
    Uses BioPython for RMSD calculation'''
    def __init__(self, ref_col: str = None, atoms: list = ["CA"], chains: list[str] = None, overwrite: bool = False, jobstarter: str =None): # pylint: disable=W0102
        self.set_ref_col(ref_col)
        self.set_atoms(atoms)
        self.set_chains(chains)
        self.set_jobstarter(jobstarter)
        self.overwrite = overwrite

    ########################## Input ################################################
    def set_ref_col(self, ref_col: str) -> None:
        '''Sets default ref_col for calc_rmsd() method.'''
        self.ref_col = ref_col

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

    def set_jobstarter(self, jobstarter: str) -> None:
        '''Sets Jobstarter for BackboneRMSD runner.'''
        self.jobstarter = jobstarter

    ########################## Calculations ################################################
    def calc_rmsd(self, poses: Poses, prefix: str, ref_col: str = None, jobstarter: JobStarter = None) -> None:
        '''Calculates RMSD as specified.'''
        # if self.atoms is all, calculate Allatom RMSD.

        # prep variables
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        ref_col = ref_col or self.ref_col
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
            jobname = "backbone_rmsd",
            output_path = work_dir
        )

        # collect individual DataFrames into one
        output_df = pd.concat([pd.read_json(sf) for sf in scorefiles], ignore_index=True).reset_index()
        output_df.to_json(scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(
            poses = poses,
            results = output_df,
            prefix = prefix,
        )
        return output.return_poses()

    def calc_all_atom_rmsd(self) -> None:
        '''Method to calculate all-atom RMSD between poses'''
        raise NotImplementedError

class MotifRMSD(Runner):
    '''Class handling'''
    def __init__(self, ref_col: str = None, target_motif: str = None, ref_motif: str = None, jobstarter: JobStarter = None):
        self.set_jobstarter(jobstarter)
        self.set_ref_col(ref_col)
        self.set_target_motif(target_motif)
        self.set_ref_motif(ref_motif)

    def __str__(self):
        return "Heavyatom motif rmsd calculator"

    def set_ref_col(self, col: str) -> None:
        '''Sets reference col for .cal_rmsd() method.'''
        self.ref_col = col

    def set_target_motif(self, motif: str) -> None:
        '''Method to set target motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.target_motif = motif

    def set_ref_motif(self, motif: str) -> None:
        '''Method to set reference motif. :motif: has to be string and should be a column name in poses.df that will be passed to the .run() function'''
        self.ref_motif = motif

    def set_jobstarter(self, jobstarter: str) -> None:
        '''Sets Jobstarter for MotifRMSD runner.'''
        self.jobstarter = jobstarter

    ################################################# Calcs ################################################

    def run(self, poses, prefix, jobstarter):
        raise NotImplementedError

    def calc_rmsd(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, ref_col: str = None, ref_motif: Any = None, target_motif: Any = None, overwrite: bool = False):
        '''Method to run Motif_rmsd calculation.'''
        # prep inputs
        ref_col = ref_col or self.ref_col
        ref_motif = ref_motif or self.ref_motif
        target_motif = target_motif or self.target_motif

        # setup runner
        script_path = f"{script_dir}/calc_heavyatom_rmsd_batch.py"
        work_dir, jobstarter = self.generic_run_setup(
            poses = poses,
            prefix = prefix,
            jobstarters = [jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # check if script exists
        if not os.path.isfile(script_path):
            raise ValueError(f"Cannot find script 'calc_heavyatom_rmsd_batch.py' at specified directory: '{script_dir}'. Set path to '/PATH/protslurm/tools/runners_auxiliary_scripts/' for variable AUXILIARY_RUNNER_SCRIPTS_DIR in config.py file.")

        # check if outputs are present
        scorefile = f"{work_dir}/{prefix}_rmsds.json"
        if os.path.isfile(scorefile) and not overwrite:
            outputs = RunnerOutput(
                poses = poses,
                results = pd.read_json(scorefile),
                prefix = prefix
            )
            return outputs.return_poses()

        # setup full input dict, batch later
        input_dict = self.setup_input_dict(
            poses = poses,
            ref_col = ref_col,
            ref_motif = ref_motif,
            target_motif = target_motif
        )

        # split input_dict into subdicts
        split_sublists = split_list(list(input_dict.keys()), n_sublists=jobstarter.max_cores)
        subdicts = [{target: input_dict[target] for target in sublist} for sublist in split_sublists]

        # write n=max_cores input_json files for add_chains_batch.py
        json_files = []
        output_files = []
        for i, subdict in enumerate(subdicts, start=1):
            # setup input_json file for every batch
            opts_json_p = f"{work_dir}/rmsd_input_{str(i).zfill(4)}.json"
            with open(opts_json_p, 'w', encoding="UTF-8") as f:
                json.dump(subdict, f)
            json_files.append(opts_json_p)
            output_files.append(f"{work_dir}/rmsd_output_{str(i).zfill(4)}.json")

        # start add_chains_batch.py
        cmds = [f"{protslurm_python} {script_path} --input_json {json_f} --output_path {output_path}" for json_f, output_path in zip(json_files, output_files)]
        jobstarter.start(
            cmds = cmds,
            jobname = f"heavyatom_rmsd_{prefix}",
            wait = True,
            output_path = work_dir
        )

        # collect outputs
        rmsd_df = pd.concat([pd.read_json(output_path) for output_path in output_files]).reset_index()
        rmsd_df.to_json(scorefile)

        outputs = RunnerOutput(
            poses = poses,
            results = rmsd_df,
            prefix = prefix
        )

        return outputs.return_poses()

    def setup_input_dict(self, poses: Poses, ref_col: str, ref_motif: Any = None, target_motif: Any = None) -> dict:
        '''Sets up dictionary that can be written down as .json file and used as an input to 'calc_heavyatom_rmsd_batch.py' '''
        def setup_ref_col(ref_col: Any, poses: Poses) -> list:
            col_in_df(poses.df, ref_col)
            return poses.df[ref_col].to_list()

        def setup_motif(motif: Any, poses: Poses) -> list:
            if isinstance(motif, str):
                # if motif points to column in DataFrame, get residues.
                col_in_df(poses.df, motif)
                return [residue_selection.to_string() if isinstance(residue_selection, ResidueSelection) else residue_selection for residue_selection in poses.df[motif].to_list()]
            elif isinstance(motif, ResidueSelection):
                return [motif for _ in poses]
            raise TypeError(f"Unsupportet parameter type for motif: {type(motif)}. Either provide a string that points to a column in poses.df containing the motifs, or pass a ResidueSelection object.")

        # use class default if parameters were not set and setup parameters:
        ref_l = setup_ref_col(ref_col or self.ref_col, poses)
        ref_motif_l = setup_motif(ref_motif or self.ref_motif, poses)
        target_motif_l = setup_motif(target_motif or self.target_motif, poses)

        # construct rmsd_input_dict:
        rmsd_input_dict = {pose: {} for pose in poses.poses_list()}
        for pose, ref, ref_motif_, target_motif_ in zip(poses.poses_list(), ref_l, ref_motif_l, target_motif_l):
            rmsd_input_dict[pose]["ref_pdb"] = ref
            rmsd_input_dict[pose]["target_motif"] = target_motif_
            rmsd_input_dict[pose]["reference_motif"] = ref_motif_

        return rmsd_input_dict
