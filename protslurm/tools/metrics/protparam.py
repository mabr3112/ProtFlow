'''Runner Module to calculate protparams'''
# import general
import os
from typing import Any

# import dependencies
import pandas as pd
import numpy as np
import protslurm

# import customs
from protslurm.config import PROTSLURM_ENV
from protslurm.config import AUXILIARY_RUNNER_SCRIPTS_DIR as script_dir
from protslurm.runners import Runner, RunnerOutput
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter
from protslurm.utils.biopython_tools import get_sequence_from_pose, load_sequence_from_fasta, load_structure_from_pdbfile



class ProtParam(Runner):
    '''
    Class handling the calculation of protparams from sequence using the BioPython Bio.SeqUtils.ProtParam module
    '''
    def __init__(self, jobstarter: str = None, default_python=os.path.join(PROTSLURM_ENV, "python3")): # pylint: disable=W0102
        self.jobstarter = jobstarter
        self.python = self.search_path(default_python, "PROTSLURM_ENV")


    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, seq_col: str = None, pH: float = 7, overwrite=False, jobstarter: JobStarter = None) -> None:
        '''
        Calculates protparam sequence features like molecular weight, isoelectric point, molar extinction coefficient etc.
            <poses>                 input poses
            <prefix>                prefix for run
            <seq_col>               if provided, run protparam on sequences in <seq_col> instead of poses.
            <pH>                    pH to determine protein total charge
            <jobstarter>            define jobstarter (since protparam is quite fast, it is recommended to run it locally)
        '''
        
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        scorefile = os.path.join(work_dir, f"{prefix}_protparam.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()


        if not seq_col:
            # check poses file extension
            pose_type = poses.determine_pose_type()
            if len(pose_type) > 1:
                raise TypeError(f"Poses must be of a single type, not {pose_type}!")
            if not pose_type[0] in [".fa", ".fasta", ".pdb"]:
                raise TypeError(f"Poses must be of type '.fa', '.fasta' or '.pdb', not {pose_type}!")
            elif pose_type[0] in [".fa", ".fasta"]:
                # directly use fasta files as input
                # TODO: this assumes that it is a single entry fasta file (as it should be!)
                seqs = [load_sequence_from_fasta(fasta=pose, return_multiple_entries=False).seq for pose in poses.df['poses'].to_list()]     
            elif pose_type[0] == ".pdb":
                # extract sequences from pdbs
                seqs = [get_sequence_from_pose(load_structure_from_pdbfile(path_to_pdb=pose)) for pose in poses.df['poses'].to_list()]
        else:
            # if not running on poses but on arbitrary sequences, get the sequences from the dataframe
            seqs = poses.df[seq_col].to_list()

        names = poses.df['poses_description'].to_list()

        input_df = pd.DataFrame({"name": names, "sequence": seqs})

        num_json_files = jobstarter.max_cores
        if num_json_files > len(input_df.index):
            num_json_files = len(input_df.index)

        json_files = []
        # create multiple input dataframes to run in parallel
        if num_json_files > 1:
            for i, df in enumerate(np.array_split(input_df, num_json_files)):
                name = os.path.join(work_dir, f"input_{i}.json")
                df.to_json(name)
                json_files.append(name)
        else:
            name = os.path.join(work_dir, f"input_1.json")
            input_df.to_json(name)
            json_files.append(name)
        
        # write commands
        cmds = []
        for json in json_files:
            cmds.append(f"{self.python} {script_dir}/run_protparam.py --input_json {json} --output_path {os.path.splitext(json)[0]}_out.json --pH {pH}")
        
        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "protparam",
            output_path = work_dir
        )
        
        # collect scores
        scores = []
        for json in json_files:
            scores.append(pd.read_json(f"{os.path.splitext(json)[0]}_out.json"))
        
        scores = pd.concat(scores)
        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
        return output.return_poses()