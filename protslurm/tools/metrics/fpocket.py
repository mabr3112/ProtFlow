'''Module to run fpocket within protflow package'''
# imports
import glob
import os
import shutil

# dependencies
import pandas as pd

# custom
from protslurm.jobstarters import JobStarter
from protslurm.poses import Poses
from protslurm.runners import Runner, RunnerOutput, options_flags_to_string, parse_generic_options
from protslurm.config import FPOCKET_PATH

class FPocket(Runner):
    """Implements FPocket. Installation see: [...] """
    # class attributes
    index_layers = 0

    def __init__(self, fpocket_path: str = FPOCKET_PATH, jobstarter: JobStarter = None):
        if not fpocket_path:
            raise ValueError(f"No path was set for {self}. Set the path in the config.py file under FPOCKET_PATH!")
        self.jobstarter = jobstarter
        self.script_path = fpocket_path

    def __str__(self):
        return "fpocket"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, options: str|list = None, pose_options: str|list = None, return_full_scores: bool = False, overwrite: bool = False) -> Poses:
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
        options_l = self.prep_fpocket_options(poses, options, pose_options)

        # move poses to input_dir (fpocket runs in the directory of the input pdb file)
        work_poses = []
        for pose in poses.poses_list():
            new_p = f"{work_dir}/{pose.split('/')[-1]}"
            if not os.path.isfile(new_p):
                shutil.copy(pose, new_p)
            work_poses.append(new_p)

        # compile cmds
        cmds = [f"{self.script_path} --file {pose} {options}" for pose, options in zip(work_poses, options_l)]

        # start
        jobstarter.start(
            cmds = cmds,
            jobname = f"fpocket_{prefix}",
            output_path = work_dir
        )

        # collect outputs and write scorefile
        scores = collect_fpocket_scores(work_dir, return_full_scores=return_full_scores)
        scores["location"] = [_get_fpocket_input_location(description, cmds) for description in scores["description"].to_list()]
        scores["pocket_location"] = scores["pocket_location"].fillna(scores["location"])
        print("inside run()", scores["pocket_location"].head(30).to_list())
        self.save_runner_scorefile(scores, scorefile)

        # itegrate and return
        outputs = RunnerOutput(poses, scores, prefix, index_layers=self.index_layers).return_poses()
        return outputs

    def prep_fpocket_options(self, poses: Poses, options: str, pose_options: str|list[str]) -> list[str]:
        '''Preps options from opts and pose_opts for fpocket run.'''
        forbidden_options = ["file", "pocket_descr_stdout", "write_mode"]
        pose_options = self.prep_pose_options(poses, pose_options)

        # Iterate through pose options, overwrite options and remove options that are not allowed.
        options_l = []
        for pose_opt in pose_options:
            opts, flags = parse_generic_options(options, pose_opt)
            for opt in forbidden_options:
                opts.pop(opt, None)
            options_l.append(options_flags_to_string(opts,flags))

        # merge options and pose_options, with pose_options priority and return
        return options_l

def get_outfile_name(outdir: str) -> str:
    '''Collects name of the output file.'''
    f = [x.strip() for x in outdir.split("/") if x][-1].replace("_out", "_info.txt")
    return f"{outdir}/{f}"

def collect_fpocket_scores(output_dir: str, return_full_scores: bool = False) -> pd.DataFrame:
    '''Collects scores from an fpocket output directory.'''
    # collect output_dirs
    output_dirs = glob.glob(f"{output_dir}/*_out")

    # extract individual scores and merge into DF:
    out_df = pd.concat([collect_fpocket_output(get_outfile_name(out_dir), return_full_scores=return_full_scores) for out_dir in output_dirs]).reset_index(drop=True)
    return out_df

def collect_fpocket_output(output_file: str, return_full_scores: bool = False) -> pd.DataFrame:
    '''Collects output of fpocket.'''
    # instantiate output_dict
    file_scores = parse_fpocket_outfile(output_file)

    if file_scores.empty:
        return pd.DataFrame.from_dict({"description": [output_file.split("/")[-1].replace("_info.txt", "")]})

    # integrate all scores if option is set:
    top_df = file_scores.head(1)
    new_cols = ["top_" + col.lower().replace(" ", "_") for col in top_df.columns]
    top_df = top_df.rename(columns=dict(zip(top_df.columns, new_cols)))
    top_df = top_df.reset_index().rename(columns={"index": "pocket"})

    # collect description and integrate into top_df
    top_df["description"] = output_file.split("/")[-1].replace("_info.txt", "")
    if return_full_scores:
        top_df["all_pocket_scores"] = file_scores

    # rename pocket_location column back.
    top_df = top_df.rename(columns={"top_pocket_location": "pocket_location"})

    return top_df.reset_index(drop=True)

def parse_fpocket_outfile(output_file: str) -> pd.DataFrame:
    '''Collects output of fpocket.'''
    def parse_pocket_line(pocket_line: str) -> tuple[str,float]:
        '''Parses singular line '''
        # split along colon between column: value
        col, val = pocket_line.split(":")
        return col.strip(), float(val[val.index("\t")+1:])

    # read out file and split along "Pocket"
    with open(output_file, 'r', encoding="UTF-8") as f:
        pocket_split = [x.strip() for x in f.read().split("Pocket") if x]

    # create empty pocket dict to populate
    pocket_dict = {}
    for raw_str in pocket_split:
        line_split = [x.strip() for x in raw_str.split("\n") if x]
        pocket_nr = line_split[0].split()[0].strip()
        pocket_dict[f"pocket_{pocket_nr}"] = {col: val for (col, val) in [parse_pocket_line(line) for line in line_split[1:]]}

    out_df = pd.DataFrame.from_dict(pocket_dict).T
    if out_df.empty:
        return out_df
    out_df["pocket_location"] = output_file.replace("info.txt", "out.pdb")
    return out_df.sort_values("Druggability Score", ascending=False)

def _get_fpocket_input_location(description: str, cmds: list[str]) -> str:
    '''Looks ad a pose_description and tries to find the pose in a list of commands that was used as input to generate the description.
    This is an internal function for location mapping'''
    # first get the cmd that contains 'description'
    cmd = [cmd for cmd in cmds if f"/{description}.pdb" in cmd][0]

    # extract location of input pdb:
    return [substr for substr in cmd.split(" ") if f"/{description}.pdb" in substr][0]
