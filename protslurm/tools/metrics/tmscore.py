'''Runner Module to calculate protparams'''
# import general
import os
from typing import Any

# import dependencies
import pandas as pd
import glob
import protslurm

# import customs
from protslurm.config import PROTSLURM_PYTHON as protslurm_python
from protslurm.runners import Runner, RunnerOutput, col_in_df
from protslurm.poses import Poses, get_format
from protslurm.jobstarters import JobStarter



class TMalign(Runner):
    '''
    Class handling the calculation of TM scores to compare to protein structures (sequence-length independent). See https://zhanggroup.org/TM-align/, https://zhanggroup.org/TM-score/ or 10.1093/nar/gki524 for more information.
    '''
    def __init__(self, jobstarter: str = None): # pylint: disable=W0102
        self.jobstarter = jobstarter


    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str, options:str=None, pose_options:str=None, overwrite:bool=False, superimpose:bool=True, jobstarter: JobStarter = None) -> None:
        '''
        Calculates the TMscore between poses and a reference structure.
            <poses>                 input poses
            <prefix>                prefix for run
            <ref_col>               column containing paths to pdb used as reference for TM score calculation
            <options>               cmd-line options for TMalign or TMscore ()
            <superimpose>           superimpose structures before calculating TM score? If False, will run TMscore instead of TMalign
            <overwrite>             if previously generated scorefile is found, read it in or run calculation again?
            <jobstarter>            define jobstarter (since protparam is quite fast, it is recommended to run it locally)
        '''
        
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        scorefile = os.path.join(work_dir, f"{prefix}_TM.{poses.storage_format}")
        if scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite):
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()


        # check if reference column exists in poses.df
        col_in_df(poses.df, ref_col)

        # prepare pose options
        pose_options = self.prep_pose_options(poses, pose_options)

        cmds = []
        for pose, ref, pose_opts in zip(poses.df['poses'].to_list(), poses.df[ref_col].to_list(), pose_options):
            cmds.append(self.write_cmd(pose_path=pose, ref_path=ref, superimpose=superimpose, output_dir=work_dir, options=options, pose_options=pose_opts))


        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)
        
        # create batch commands
        cmd_sublists = protslurm.jobstarters.split_list(cmds, n_sublists=num_cmds)
        cmds = []
        for sublist in cmd_sublists:
            cmds.append("; ".join(sublist))
    
        # run command
        jobstarter.start(
            cmds = cmds,
            jobname = "TM",
            output_path = work_dir
        )
        
        scores = self.collect_scores(output_dir=work_dir)

        scores = scores.merge(poses.df[['poses', 'poses_description']], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
        return output.return_poses()
    
    
    def write_cmd(self, pose_path:str, ref_path:str, output_dir:str, options:str=None, pose_options:str=None, superimpose:bool=True):
        '''Writes Command to run ligandmpnn.py'''
        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""
        
        # parse options
        opts, flags = protslurm.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""
        
        # check if structures should be superimposed before calculating TM score and use the corresponding application
        if superimpose == True:
            application = "TMalign"
        else:
            application = "TMscore"

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.tmout")

        # compile command
        run_string = f"{application} {pose_path} {ref_path} {opts} {flags} > {scorefile}"

        return run_string
    
    def collect_scores(self, output_dir:str):

        def extract_scores(score_path:str) -> pd.Series:
            '''
            extract TM scores from scorefile, return a Series
            '''

            TM_scores = {}
            with open(score_path, 'r') as f:
                for line in f:
                    if line.startswith("TM-score") and "Chain_1" in line:
                        # TM score normalized by length of the pose structure
                        TM_scores['TM_score_pose'] = float(line.split()[1])
                    elif line.startswith("TM-score") and "Chain_2" in line:
                        # TM score normalized by length of the reference (this is what should be used)
                        TM_scores['TM_score_ref'] = float(line.split()[1])
                    elif line.startswith("TM-score") and "average" in line:
                        # if -a flag was provided to TMalign, a TM score normalized by the average length of pose and reference will be calculated
                        TM_scores['TM_score_average'] = float(line.split()[1])
                    elif line.startswith("TM-score"):
                        # if TMscore was used instead of TMalign, the output only contains a single score
                        TM_scores['TM_score_ref'] = float(line.split()[2])
            TM_scores['description'] = os.path.splitext(os.path.basename(score_path))[0]
            
            # check if scores were present in scorefile
            if not any('TM_score' in key for key in TM_scores):
                raise RuntimeError(f"Could not find any TM scores in {score_path}!")
            
            return pd.Series(TM_scores)


        # collect scorefiles
        scorefiles = glob.glob(os.path.join(output_dir, "*.tmout"))

        scores = [extract_scores(file) for file in scorefiles]
        scores = pd.DataFrame(scores).reset_index(drop=True)

        return scores




        