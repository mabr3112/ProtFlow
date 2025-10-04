# general imports
import json
import os
import logging
from glob import glob
import shutil
import re

# dependencies
import pandas as pd

# custom
from protflow import require_config, load_config_path
from protflow.poses import Poses, description_from_path
from protflow.jobstarters import JobStarter, split_list
from protflow.runners import Runner, RunnerOutput, prepend_cmd

class Frame2SeqScore(Runner):
    def __init__(self, python_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter = None) -> None:
        # setup config
        config = require_config()
        self.python_path = python_path or load_config_path(config, "FRAME2SEQ_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "FRAME2SEQ_PRE_CMD", is_pre_cmd=True)
        self.script_path = os.path.join(load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR"), "run_frame2seq.py")

        # setup runner
        self.name = "frame2seqscore.py"
        self.index_layers = 0
        self.jobstarter = jobstarter

    def __str__(self):
        return "frame2seqscore.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, chain: str = "A", options: dict = None, pose_options: list|str = None, preserve_original_output: bool = False, overwrite: bool = False) -> Poses:

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = os.path.join(work_dir, f"frame2seq_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()

        # parse pose_options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # make sure pose options are dict
        if not all(pose_opt is None or isinstance(pose_opt, dict) for pose_opt in pose_options):
            raise TypeError("Pose options must be None or a dictionary!")
        
        # merge options and pose_options
        pose_options = self.merge_opts_and_pose_opts(poses, options, pose_options, chain)

        mandatory_opts = ["chain_id"]
        forbidden_opts = ["save_indiv_neg_pll", "save_indiv_seqs"]
        for pose, pose_opt in zip(poses.poses_list(), pose_options):
            if missing := [mandatory for mandatory in mandatory_opts if not mandatory in pose_opt]:
                raise KeyError(f"Mandatory options {missing} are missing (at least) for pose {pose}!")
            # remove opts important for output handling
            for opt in forbidden_opts:
                pose_opt.pop(opt, None)

        # create input dicts
        input_dicts = []
        for pose, pose_opt in zip(poses.poses_list(), pose_options):
            input_dicts.append({os.path.abspath(pose): pose_opt})
        
        # set up batches
        input_sublists = split_list(input_dicts, n_sublists=min([jobstarter.max_cores, len(input_dicts)]))

        # set up input dict
        input_dicts = [{pose: pose_opt for d in sublist for pose, pose_opt in d.items()} for sublist in input_sublists]

        # write input dicts
        in_jsons = []
        for i, input_dict in enumerate(input_dicts):
            in_json = os.path.join(work_dir, f"in_{str(i)}.json")
            with open(in_json, "w") as f:
                json.dump(input_dict, f, indent=4)  
            in_jsons.append(in_json)

        # write cmds:
        cmds = [self.write_cmd(in_json, work_dir) for in_json in in_jsons]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="frame2seqscore",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            preserve_original_output=preserve_original_output,
        )

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    
    def merge_opts_and_pose_opts(self, poses:Poses, options:dict, pose_options:list, chain: str):

        if not options:
            options = {}
        if chain:
            options["chain_id"] = chain

        for i, pose_opt in enumerate(pose_options):
            if pose_opt is None:
                pose_opt = {}
            # merge pose_opt into options without overwriting pose_options
            merged = {**options, **pose_opt}
            pose_options[i] = merged

        return pose_options

    def write_cmd(self, input_json: str, work_dir: str):
        # write command and return.
        return f"{self.python_path} {self.script_path} --input_json {input_json} --output_dir {work_dir} --method score"

def collect_scores(work_dir: str, preserve_original_output: bool = False) -> pd.DataFrame:

    results_dir = os.path.join(work_dir, "frame2seq_outputs")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"Could not find frame2seq_outputs directory at {results_dir}")
    
    in_jsons = glob(os.path.join(work_dir, "in_*.json"))
    poses = []
    chains = []
    for in_json in in_jsons:
        # import input json
        with open(in_json, "r") as jf:
            input_dict = json.load(jf)
        for key in input_dict.keys():
            poses.append(key)
            chains.append(input_dict[key]["chain_id"])

    scores_dir = os.path.join(results_dir, "scores")

    mean_logs = []
    perres_logs = []
    for pose, chain in zip(poses, chains):
        name = description_from_path(pose)
        score = pd.read_csv(os.path.join(scores_dir, f"{os.path.splitext(name)[0]}_{chain}_seq0.csv")) # TODO: cannot get multiple fasta input to work atm, output might look different then
        mean_logs.append(score["Negative pseudo-log-likelihood"].mean())
        perres_logs.append(score["Negative pseudo-log-likelihood"].to_list())
    
    scores = pd.DataFrame({"location": poses, "description": [description_from_path(pose) for pose in poses], "score": mean_logs, "per_res_neg_log_likelihood": perres_logs})

    # delete files
    if not preserve_original_output:
        shutil.rmtree(results_dir)

    return scores
    

    
