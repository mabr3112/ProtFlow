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
from protflow.residues import ResidueSelection
from protflow.poses import Poses, col_in_df, description_from_path
from protflow.jobstarters import JobStarter, split_list
from protflow.runners import Runner, RunnerOutput, col_in_df, prepend_cmd

class Frame2SeqDesign(Runner):
    def __init__(self, python_path: str|None = None, pre_cmd: str|None = None, jobstarter: JobStarter = None) -> None:
        # setup config
        config = require_config()
        self.python_path = python_path or load_config_path(config, "FRAME2SEQ_PYTHON_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, "FRAME2SEQ_PRE_CMD", is_pre_cmd=True)
        self.script_path = os.path.join(load_config_path(config, "AUXILIARY_RUNNER_SCRIPTS_DIR"), "run_frame2seq.py")

        # setup runner
        self.name = "frame2seqdesign.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "frame2seqdesign.py"

    def run(self, poses: Poses, prefix: str, jobstarter: JobStarter = None, num_samples: int = 1, chain: str = "A", temperature: float = 1, options: dict = None, pose_options: list|str = None, fixed_res_col: str = None, preserve_original_output: bool = False, overwrite: bool = False) -> Poses:

        self.index_layers = 1

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
        pose_options = self.merge_opts_and_pose_opts(poses, options, pose_options, num_samples, chain, temperature, fixed_res_col)

        mandatory_opts = ["chain_id", "temperature", "num_samples"]
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
            jobname="frame2seqdesign",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        scores = collect_scores(
            work_dir=work_dir,
            preserve_original_output=preserve_original_output,
        )

        if len(scores.index) < len(poses.df.index) * num_samples:
            raise RuntimeError("Number of output poses is smaller than number of input poses * nseq. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")
        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def merge_opts_and_pose_opts(self, poses:Poses, options:dict, pose_options:list, num_samples:int, chain: str, temperature: float, fixed_res_col: str = None):

        if not options:
            options = {}
        if chain:
            options["chain_id"] = chain
        if temperature:
            options["temperature"] = temperature
        if num_samples:
            options["num_samples"] = num_samples

        for i, pose_opt in enumerate(pose_options):
            if pose_opt is None:
                pose_opt = {}
            # merge pose_opt into options without overwriting pose_options
            merged = {**options, **pose_opt}
            pose_options[i] = merged

        if fixed_res_col:
            col_in_df(poses.df, fixed_res_col)
            fixed_residues = poses.df[fixed_res_col].to_list()
            for i, (pose_opt, fixed_res) in enumerate(zip(pose_options, fixed_residues)):
                # convert fixed residues to list (without chain information)
                if isinstance(fixed_res, ResidueSelection):
                    fixed_res = fixed_res.to_dict()[pose_opt["chain_id"]]
                if isinstance(fixed_res, dict):
                    fixed_res = fixed_res[pose_opt["chain_id"]]
                if not isinstance(fixed_res, list) or any(isinstance(x, str) for x in fixed_res):
                    raise KeyError(f"<fixed_res_col> must contain either a ResidueSelection, a dict or a 0-indexed list of residue positions without chain indicators!")
                pose_opt["fixed_positions"] = fixed_res
                pose_options[i] = pose_opt # update pose_options

        return pose_options

    def write_cmd(self, input_json: str, work_dir: str):
        # write command and return.
        return f"{self.python_path} {self.script_path} --input_json {input_json} --output_dir {work_dir} --method design"

def collect_scores(work_dir: str, preserve_original_output: bool = False) -> pd.DataFrame:
    def parse_frame2seq_fasta(fasta:str, out_dir:str):

        def parse_line(s: str):
            result = {}
            for part in s.split():
                if "=" in part:
                    key, value = part.split("=", 1)
                    result[key] = value

            result["recovery"] = float(result["recovery"][:-1]) / 100
            return result
        
        def extract_ints(s: str):
            return [int(x) for x in re.findall(r"\d+", s)][0]
    
        # import fasta
        with open(fasta, "r") as fa:
            data = fa.readlines()

        header = data[0]
        seq = data[1]

        # parse information from header
        header_dict = parse_line(header)

        # create output file name
        suffix = extract_ints(fasta.split("_")[-1])
        suffix = suffix + 1
        new_name = f"{header_dict[">pdbid"]}_{str(suffix).zfill(4)}"
        filename = os.path.abspath(os.path.join(out_dir, f"{new_name}.fasta"))
        
        # create fasta file
        data = f">{new_name}\n{seq}"
        with open(filename, "w+") as fa:
            fa.write(data)

        header_dict["location"] = filename
        header_dict["description"] = description_from_path(filename)
        header_dict["seq"] = seq
        header_dict.pop(">pdbid")
        return header_dict


    results_dir = os.path.join(work_dir, "frame2seq_outputs")
    if not os.path.isdir(results_dir):
        raise RuntimeError(f"Could not find frame2seq_outputs directory at {results_dir}")
    
    seqs_dir = os.path.join(results_dir, "seqs")
    scores_dir = os.path.join(results_dir, "scores")

    os.makedirs(updated_seqs := os.path.join(work_dir, "frame2seqs_fasta"), exist_ok=True)

    scores = []
    fastas = glob(os.path.join(seqs_dir, "*.fasta"))
    for fasta in fastas:
        if not fasta.endswith("seqs.fasta"): # ignore collected results
            name = os.path.splitext(os.path.basename(fasta))[0]
            perreslogprobs = pd.read_csv(os.path.join(scores_dir, f"{name}.csv"))
            data_dict = parse_frame2seq_fasta(fasta, updated_seqs)
            data_dict["per_res_neg_log_likelihood"] = perreslogprobs["Negative pseudo-log-likelihood"].to_list()
        scores.append(data_dict)
    
    scores = pd.DataFrame(scores)

    # delete files
    if not preserve_original_output:
        shutil.rmtree(results_dir)

    return scores
    

    
