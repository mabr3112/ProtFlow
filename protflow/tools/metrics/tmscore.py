'''Runner Module to calculate protparams'''
# import general
import os
import glob

# import dependencies
import pandas as pd

# import customs
import protflow
from protflow.poses import Poses
from protflow.runners import Runner, RunnerOutput, col_in_df
from protflow.jobstarters import JobStarter
from protflow.config import PROTSLURM_ENV
from protflow.utils.metrics import calc_sc_tm

class TMalign(Runner):
    '''
    Class handling the calculation of TM scores to compare to protein structures (sequence-length independent). Structures will be superimposed before calculation. See https://zhanggroup.org/TM-align/ or 10.1093/nar/gki524 for more information.
    '''
    def __init__(self, jobstarter: str = None, application: str = None):
        self.jobstarter = jobstarter
        self.name = "tmscore.py"
        self.index_layers = 0
        self.application = application or os.path.join(PROTSLURM_ENV, "TMalign")

    def __str__(self):
        return "TMalign"

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str, sc_tm_score: bool = True, options: str = None, pose_options: str = None, overwrite: bool = False, jobstarter: JobStarter = None) -> None: # pylint: disable=W0237
        '''
        Calculates the TMscore between poses and a reference structure. It is recommended to use TM_score_ref, as this is normalized by length of the reference structure. Also returns a self consistency score 
        which indicates how many poses with the same reference pose are above the <selfconsistency_tm_cutoff>, indicating the designability of the reference pose. 
            <poses>                     input poses
            <prefix>                    prefix for run
            <ref_col>                   column containing paths to pdb used as reference for TM score calculation
            <sc_tm_score>               if True, calculates sc-TM score (highest TM-Score) for each backbone in ref_col and adds it into the column {prefix}_sc_tm
            <options>                   cmd-line options for TMalign
            <pose_options>              name of poses.df column containing options for TMalign
            <overwrite>                 if previously generated scorefile is found, read it in or run calculation again?
            <jobstarter>                define jobstarter (since protparam is quite fast, it is recommended to run it locally)
        '''
        # setup runner and files
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        scorefile = os.path.join(work_dir, f"{prefix}_TM.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
            if sc_tm_score:
                output.df = calc_sc_tm(input_df=output.df, name=f"{prefix}_sc_tm", ref_col=ref_col, tm_col=f"{prefix}_TM_score_ref")
                print([x for x in output.df.columns if prefix in x])
            return output


        # prepare pose options
        pose_options = self.prep_pose_options(poses, pose_options)

        # prepare references:
        ref_l = self.prep_ref(ref=ref_col, poses=poses)

        cmds = []
        for pose, ref, pose_opts in zip(poses.df['poses'].to_list(), ref_l, pose_options):
            cmds.append(self.write_cmd(pose_path=pose, ref_path=ref, output_dir=work_dir, options=options, pose_options=pose_opts))

        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)

        # create batch commands
        cmd_sublists = protflow.jobstarters.split_list(cmds, n_sublists=num_cmds)
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

        scores = scores.merge(poses.df[['poses', 'poses_description', ref_col]], left_on="description", right_on="poses_description").drop('poses_description', axis=1)
        scores = scores.rename(columns={"poses": "location"})

        #if selfconsistency_tm_cutoff:
        #    dfs = []
        #    for ref, df in scores.groupby(ref_col, sort=False):
        #        above_cutoff = (df['TM_score_ref'] > selfconsistency_tm_cutoff).sum()
        #        df['self_consistency_score'] = above_cutoff / len(df.index)
        #        dfs.append(df)
        #    scores = pd.concat(dfs).reset_index(drop=True)

        # write output scorefile
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        # create standardised output for poses class:
        output = RunnerOutput(poses=poses, results=scores, prefix=prefix).return_poses()
        if sc_tm_score:
            output.df = calc_sc_tm(input_df=output.df, name=f"{prefix}_sc_tm", ref_col=ref_col, tm_col=f"{prefix}_TM_score_ref")
            print([x for x in output.df.columns if prefix in x])
        return output

    def prep_ref(self, ref: str, poses: Poses) -> list[str]:
        '''Preps ref_col parameter for TMalign:
        If ref points to a .pdb file, return list of .pdb-files as ref_l.
        If ref points to a column in the Poses DataFrame, then return the column's entries as a list.'''
        if not isinstance(ref, str):
            raise ValueError(f"Parameter :ref: must be string and either refer to a .pdb file or to a column in poses.df!")
        if ref.endswith(".pdb"):
            return [ref for _ in poses]

        # check if reference column exists in poses.df
        col_in_df(poses.df, ref)
        return poses.df[ref].to_list()

    def write_cmd(self, pose_path: str, ref_path: str, output_dir: str, options: str = None, pose_options: str = None) -> str:
        '''Writes Command to run ligandmpnn.py'''
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""

        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.tmout")

        # compile command
        run_string = f"{self.application} {pose_path} {ref_path} {opts} {flags} > {scorefile}"

        return run_string

    def collect_scores(self, output_dir:str) -> pd.DataFrame:
        '''Collects scores of TMalign runs.'''
        def extract_scores(score_path:str) -> pd.Series:
            '''
            extract TM scores from scorefile, return a Series
            '''

            tm_scores = {}
            with open(score_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    if line.startswith("Aligned length"):
                        tm_scores['num_aligned_res'] = int(line.split()[2].replace(',', ''))
                        tm_scores['RMSD'] = float(line.split()[4].replace(',', ''))
                        tm_scores['n_identical/n_aligned'] = float(line.split()[6])
                        continue
                    elif line.startswith("TM-score") and "Chain_1" in line:
                        # TM score normalized by length of the pose structure
                        tm_scores['TM_score_pose'] = float(line.split()[1])
                        continue
                    elif line.startswith("TM-score") and "Chain_2" in line:
                        # TM score normalized by length of the reference (this is what should be used)
                        tm_scores['TM_score_ref'] = float(line.split()[1])
                        continue
                    elif line.startswith("TM-score") and "average" in line:
                        # if -a flag was provided to TMalign, a TM score normalized by the average length of pose and reference will be calculated
                        tm_scores['TM_score_average'] = float(line.split()[1])

            tm_scores['description'] = os.path.splitext(os.path.basename(score_path))[0]

            # check if scores were present in scorefile
            if not any('TM_score' in key for key in tm_scores):
                raise RuntimeError(f"Could not find any TM scores in {score_path}!")
            return pd.Series(tm_scores)

        # collect scorefiles
        scorefiles = glob.glob(os.path.join(output_dir, "*.tmout"))

        scores = [extract_scores(file) for file in scorefiles]
        scores = pd.DataFrame(scores).reset_index(drop=True)

        return scores

class TMscore(Runner):
    '''
    Class handling the calculation of TM scores to compare to protein structures (sequence-length independent). Structures will NOT be superimposed before calculation. See https://zhanggroup.org/TM-score/ or 10.1093/nar/gki524 for more information.
    '''
    def __init__(self, jobstarter: str = None, application: str = None):
        self.jobstarter = jobstarter
        self.name = "tmscore.py"
        self.index_layers = 0
        self.application = application or os.path.join(PROTSLURM_ENV, "TMscore")

    def __str__(self):
        return self.name

    ########################## Calculations ################################################
    def run(self, poses: Poses, prefix: str, ref_col: str, options: str = None, pose_options: str = None, overwrite: bool = False, jobstarter: JobStarter = None) -> None: # pylint: disable=W0237
        '''
        Calculates the TMscore between poses and a reference structure.
            <poses>                 input poses
            <prefix>                prefix for run
            <ref_col>               column containing paths to pdb used as reference for TM score calculation
            <options>               cmd-line options for TMscore
            <pose_option>           name of poses.df column containing options for TMscore
            <overwrite>             if previously generated scorefile is found, read it in or run calculation again?
            <jobstarter>            previously defined jobstarter
        '''

        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )
        scorefile = os.path.join(work_dir, f"{prefix}_TM.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix)
            return output.return_poses()

        # check if reference column exists in poses.df
        col_in_df(poses.df, ref_col)

        # prepare pose options
        pose_options = self.prep_pose_options(poses, pose_options)

        cmds = []
        for pose, ref, pose_opts in zip(poses.df['poses'].to_list(), poses.df[ref_col].to_list(), pose_options):
            cmds.append(self.write_cmd(pose_path=pose, ref_path=ref, output_dir=work_dir, options=options, pose_options=pose_opts))

        num_cmds = jobstarter.max_cores
        if num_cmds > len(poses.df.index):
            num_cmds = len(poses.df.index)

        # create batch commands
        cmd_sublists = protflow.jobstarters.split_list(cmds, n_sublists=num_cmds)
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

    def write_cmd(self, pose_path: str, ref_path: str, output_dir: str, options: str = None, pose_options: str = None ) -> str:
        '''Writes Command to run TM-Score'''
        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options, sep="-")
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()])
        flags = " -" + " -".join(flags) if flags else ""

        # parse options
        opts, flags = protflow.runners.parse_generic_options(options, pose_options)
        opts = " ".join([f"-{key}={value}" for key, value in opts.items()]) if opts else ""
        flags = " -" + " -".join(flags) if flags else ""

        # define scorefile names
        scorefile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pose_path))[0]}.tmout")

        # compile command
        run_string = f"{self.application} {pose_path} {ref_path} {opts} {flags} > {scorefile}"

        return run_string

    def collect_scores(self, output_dir: str) -> pd.DataFrame:
        '''Collects scores of TMAlign or TMScore runs.'''
        def extract_scores(score_path:str) -> pd.Series:
            '''
            extract TM scores from scorefile, return a Series
            '''

            tm_scores = {}
            with open(score_path, 'r', encoding="UTF-8") as f:
                for line in f:
                    # extract scores
                    if line.startswith("TM-score"):
                        tm_scores['TM_score_ref'] = float(line.split()[2])
                    elif line.startswith("MaxSub-score"):
                        tm_scores['MaxSub_score'] = float(line.split()[1])
                    elif line.startswith("GDT-TS-score"):
                        tm_scores['GDT-TS_score'] = float(line.split()[1])
                    elif line.startswith("GDT-HA-score"):
                        tm_scores['GDT-HA_score'] = float(line.split()[1])
            tm_scores['description'] = os.path.splitext(os.path.basename(score_path))[0]

            # check if scores were present in scorefile
            if len(list(tm_scores)) < 2:
                raise RuntimeError(f"Could not find any TM scores in {score_path}!")
            return pd.Series(tm_scores)

        # collect scorefiles
        scorefiles = glob.glob(os.path.join(output_dir, "*.tmout"))

        scores = [extract_scores(file) for file in scorefiles]
        scores = pd.DataFrame(scores).reset_index(drop=True)

        return scores
