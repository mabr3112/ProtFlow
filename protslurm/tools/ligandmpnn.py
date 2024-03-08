'''Module to handle LigandMPNN within ProtSLURM'''
# general imports
import json
import os
import logging
from glob import glob
import shutil

# dependencies
import pandas as pd
import Bio
import Bio.SeqIO

# custom
import protslurm.config
from protslurm.residues import ResidueSelection
import protslurm.tools
import protslurm.runners
from protslurm.poses import Poses
from protslurm.jobstarters import JobStarter
from protslurm.runners import Runner, RunnerOutput, expand_options_flags, parse_generic_options, col_in_df, options_flags_to_string

if not protslurm.config.LIGANDMPNN_SCRIPT_PATH:
    raise ValueError(f"No path is set for ligandmpnn run.py. Set the path in the config.py file under LIGANDMPNN_SCRIPT_PATH")

LIGANDMPNN_DIR = protslurm.config.LIGANDMPNN_SCRIPT_PATH.rsplit("/", maxsplit=1)[0]
LIGANDMPNN_CHECKPOINT_DICT = {
    "protein_mpnn": f"{LIGANDMPNN_DIR}/model_params/proteinmpnn_v_48_020.pt",
    "ligand_mpnn": f"{LIGANDMPNN_DIR}/model_params/ligandmpnn_v_32_010_25.pt",
    "per_residue_label_membrane_mpnn": f"{LIGANDMPNN_DIR}/model_params/per_residue_label_membrane_mpnn_v_48_020.pt",
    "global_label_membrane_mpnn": f"{LIGANDMPNN_DIR}/model_params/global_label_membrane_mpnn_v_48_020.pt",
    "soluble_mpnn": f"{LIGANDMPNN_DIR}/model_params/solublempnn_v_48_020.pt"
}

class LigandMPNN(Runner):
    '''Class to run LigandMPNN and collect its outputs into a DataFrame'''
    def __init__(self, script_path:str=protslurm.config.LIGANDMPNN_SCRIPT_PATH, python_path:str=protslurm.config.LIGANDMPNN_PYTHON_PATH, jobstarter:JobStarter=None) -> None:
        '''jobstarter_options are set automatically, but can also be manually set. Manual setting is not recommended.'''
        if not script_path: raise ValueError(f"No path is set for {self}. Set the path in the config.py file under LIGANDMPNN_SCRIPT_PATH.")
        if not python_path: raise ValueError(f"No python path is set for {self}. Set the path in the config.py file under LIGANDMPNN_PYTHON_PATH.")
        self.script_path = script_path
        self.python_path = python_path
        self.name = "ligandmpnn.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "ligandmpnn.py"

    def run(self, poses:Poses, prefix:str, jobstarter:JobStarter=None, nseq:int=None, model_type:str=None, options:str=None, pose_options:object=None, fixed_res_col:str=None, design_res_col:str=None, pose_opt_cols:dict=None, return_seq_threaded_pdbs_as_pose:bool=False, preserve_original_output:bool=True, overwrite:bool=False) -> RunnerOutput:
        '''Runs ligandmpnn.py on acluster.
        Default model_type is ligand_mpnn.'''
        # run in batch mode if pose_options are not set:
        pose_opt_cols = pose_opt_cols or {}
        run_batch = self.check_for_batch_run(pose_options, pose_opt_cols)
        if run_batch:
            logging.info(f"Setting up ligandmpnn for batched design.")

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        # Look for output-file in pdb-dir. If output is present and correct, skip LigandMPNN.
        scorefile = "ligandmpnn_scores.json"
        scorefilepath = os.path.join(work_dir, scorefile)
        if overwrite is False and os.path.isfile(scorefilepath):
            return RunnerOutput(poses=poses, results=pd.read_json(scorefilepath), prefix=prefix, index_layers=self.index_layers).return_poses()

        # integrate redesigned and fixed residue parameters into pose_opt_cols:
        pose_opt_cols["fixed_residues"] = fixed_res_col
        pose_opt_cols["redesigned_residues"] = design_res_col

        # parse pose_opt_cols into pose_options format.
        pose_opt_cols_options = self.parse_pose_opt_cols(pose_opt_cols, output_dir=work_dir)

        # parse pose_options
        pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)

        # combine pose_options and pose_opt_cols_options (priority goes to pose_opt_cols_options):
        pose_options = [options_flags_to_string(*parse_generic_options(pose_opt, pose_opt_cols_opt, sep="--"), sep="--") for pose_opt, pose_opt_cols_opt in zip(pose_options, pose_opt_cols_options)]

        # write ligandmpnn cmds:
        cmds = [self.write_cmd(pose, output_dir=work_dir, model=model_type, nseq=nseq, options=options, pose_options=pose_opts) for pose, pose_opts in zip(poses.df['poses'].to_list(), pose_options)]

        # batch_run setup:
        if run_batch:
            cmds = self.setup_batch_run(cmds, num_batches=jobstarter.max_cores, output_dir=work_dir)

        # run
        jobstarter.start(
            cmds=cmds,
            jobname="ligandmpnn",
            wait=True,
            output_path=f"{work_dir}/"
        )

        # collect scores
        scores = self.collect_scores(
            work_dir=work_dir,
            scorefile=scorefilepath,
            return_seq_threaded_pdbs_as_pose=return_seq_threaded_pdbs_as_pose,
            preserve_original_output=preserve_original_output
        )

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def check_for_batch_run(self, pose_options: str, pose_opt_cols):
        '''Checks if ligandmpnn can be run in batch mode'''
        return pose_options is None and self.multi_cols_only(pose_opt_cols)

    def multi_cols_only(self, pose_opt_cols:dict) -> bool:
        '''checks if only multi_res cols are in pose_opt_cols dict. Only _multi arguments can be used for ligandmpnn_batch runs.'''
        multi_cols = ["omit_AA_per_residue", "bias_AA_per_residue", "redesigned_residues", "fixed_residues"]
        return True if pose_opt_cols is None else all((col in multi_cols for col in pose_opt_cols))

    def setup_batch_run(self, cmds:list[str], num_batches:int, output_dir:str) -> list[str]:
        '''Concatenates cmds for MPNN into batches so that MPNN does not have to be loaded individually for each pdb file.'''
        multi_cols = {
            "omit_AA_per_residue": "omit_AA_per_residue_multi",
            "bias_AA_per_residue": "bias_AA_per_residue_multi", 
            "redesigned_residues": "redesigned_residues_multi",
            "fixed_residues": "fixed_residues_multi", 
            "pdb_path": "pdb_path_multi"
        }
        # setup json directory
        json_dir = f"{output_dir}/input_json_files/"
        if not os.path.isdir(json_dir):
            os.makedirs(json_dir, exist_ok=True)

        # split cmds list into n=num_batches sublists
        cmd_sublists = protslurm.jobstarters.split_list(cmds, n_sublists=num_batches)
        print(cmd_sublists)

        # concatenate cmds: parse _multi arguments into .json files and keep all other arguments in options.
        batch_cmds = []
        for i, cmd_list in enumerate(cmd_sublists, start=1):
            full_cmd_list = [cmd.split(" ") for cmd in cmd_list]
            opts_flags_list = [expand_options_flags(" ".join(cmd_split[2:])) for cmd_split in full_cmd_list]
            opts_list = [x[0] for x in opts_flags_list] # expand_options_flags() returns (options, flags)

            # take first cmd for general options and flags
            full_opts_flags = opts_flags_list[0]
            cmd_start = " ".join(full_cmd_list[0][:2]) # keep /path/to/python3 /path/to/run.py

            # extract lists for _multi options
            for col, multi_col in multi_cols.items():
                # if col does not exist in options, skip:
                if col not in opts_list[0]:
                    continue

                # extract pdb-file to argument mapping as dictionary:
                col_dict = {opts["pdb_path"]: opts[col] for opts in opts_list}

                # write col_dict to json
                col_json_path = f"{json_dir}/{col}_{i}.json"
                with open(col_json_path, 'w', encoding="UTF-8") as f:
                    json.dump(col_dict, f)

                # remove single option from full_opts_flags and set cmd_json file as _multi option:
                del full_opts_flags[0][col]
                full_opts_flags[0][multi_col] = col_json_path

            # reassemble command and put into batch_cmds
            batch_cmd = f"{cmd_start} {options_flags_to_string(*full_opts_flags, sep='--')}"
            batch_cmds.append(batch_cmd)

        return batch_cmds

    def parse_pose_opt_cols(self, poses:Poses, output_dir:str, pose_opt_cols:dict=None) -> list[dict]:
        '''Parses pose_opt_cols into pose_options formatted strings to later combine with pose_options.'''
        # return list of empty strings if pose_opts_col is None.
        if pose_opt_cols is None:
            return ["" for _ in poses]

        # setup output_dir for .json files
        if any([key in ["bias_AA_per_residue", "omit_AA_per_residue"] for key in pose_opt_cols]):
            json_dir = f"{output_dir}/input_json_files/"
            if not os.path.isdir(json_dir):
                os.makedirs(json_dir, exist_ok=True)

        # check if fixed_residues and redesigned_residues were set properly (gets checked in LigandMPNN too, so maybe this is redundant.)
        if "fixed_residues" in pose_opt_cols and "redesigned_residues" in pose_opt_cols:
            raise ValueError(f"Cannot define both <fixed_res_column> and <design_res_column>!")

        # check if all specified columns exist in poses.df:
        for col in list(pose_opt_cols.values()):
            col_in_df(poses.df, col)

        # parse pose_options
        pose_options = []
        for pose in poses:
            opts = []
            for mpnn_arg, mpnn_arg_col in pose_opt_cols.values():
                # arguments that must be written into .json files:
                if mpnn_arg in ["bias_AA_per_residue", "omit_AA_per_residue"]:
                    output_path = f"{json_dir}/{mpnn_arg}_{pose['poses_description']}.json"
                    opts.append(f"--{mpnn_arg}={write_to_json(pose[mpnn_arg_col], output_path)}")

                # arguments that can be parsed as residues (from ResidueSelection objects):
                elif mpnn_arg in ["redesigned_residues", "fixed_residues", "transmembrane_buried", "transmembrane_interface"]:
                    opts.append(f"--{mpnn_arg}={parse_residues(pose[mpnn_arg_col])}")

                # all other arguments:
                else:
                    opts.append(f"--{mpnn_arg}={pose[mpnn_arg_col]}")
            pose_options.append(" ".join(opts))
        return pose_options

    def write_cmd(self, pose_path:str, output_dir:str, model:str, nseq:int, options:str, pose_options:str):
        '''Writes Command to run ligandmpnn.py
        default model: ligand_mpnn
        default number of sequences: 1'''
        # check if specified model is correct.
        available_models = ["protein_mpnn", "ligand_mpnn", "soluble_mpnn", "global_label_membrane_mpnn", "per_residue_label_membrane_mpnn"]
        if model not in available_models:
            raise ValueError(f"{model} must be one of {available_models}!")

        # parse options
        opts, flags = parse_generic_options(options, pose_options)

        # safetychecks:
        if "model_type" not in opts:
            opts["model_type"] = model or "ligand_mpnn"
        if "number_of_batches" not in opts:
            opts["number_of_batches"] = nseq or "1"
        # define model_checkpoint option:
        if f"checkpoint_{model}" not in opts:
            model_checkpoint_options = f"--checkpoint_{model}={LIGANDMPNN_CHECKPOINT_DICT[model]}"

        # safety
        logging.info(f"Setting parse_atoms_with_zero_occupancy to 1 to ensure that the run does not crash.")
        if "parse_atoms_with_zero_occupancy" not in opts:
            opts["parse_atoms_with_zero_occupancy"] = "1"
        elif opts["parse_atoms_with_zero_occupancy"] != "1":
            opts["parse_atoms_with_zero_occupancy"] = "1"

        # convert to string
        options = options_flags_to_string(opts, flags, sep="--")

        # write command and return.
        return f"{self.python_path} {self.script_path} {model_checkpoint_options} --out_folder {output_dir}/ --pdb_path {pose_path} {options}"

    def collect_scores(self, work_dir:str, scorefile:str, return_seq_threaded_pdbs_as_pose:bool, preserve_original_output:bool=True) -> pd.DataFrame:
        '''collects scores from ligandmpnn output'''

        def mpnn_fastaparser(fasta_path):
            '''reads in ligandmpnn multi-sequence fasta, renames sequences and returns them'''
            records = list(Bio.SeqIO.parse(fasta_path, "fasta"))
            #maxlength = len(str(len(records)))

            # Set continuous numerating for the names of mpnn output sequences:
            name = records[0].name.replace(",", "")
            records[0].name = name
            for i, x in enumerate(records[1:]):
                setattr(x, "name", f"{name}_{str(i+1).zfill(4)}")

            return records

        def convert_ligandmpnn_seqs_to_dict(seqs):
            '''
            Takes already parsed list of fastas as input <seqs>. Fastas can be parsed with the function mpnn_fastaparser(file).
            Should be put into list.
            Converts mpnn fastas into a dictionary:
            {
                "col_1": [vals]
                    ...
                "col_n": [vals]
            }
            '''
            # Define cols and initiate them as empty lists:
            seqs_dict = dict()
            cols = ["mpnn_origin", "seed", "description", "sequence", "T", "id", "seq_rec", "overall_confidence", "ligand_confidence"]
            for col in cols:
                seqs_dict[col] = []

            # Read scores of each sequence in each file and append them to the corresponding columns:
            for seq in seqs:
                for f in seq[1:]:
                    seqs_dict["mpnn_origin"].append(seq[0].name)
                    seqs_dict["sequence"].append(str(f.seq))
                    seqs_dict["description"].append(f.name)
                    d = {k: float(v) for k, v in [x.split("=") for x in f.description.split(", ")[1:]]}
                    for k, v in d.items():
                        seqs_dict[k].append(v)
            return seqs_dict

        def write_mpnn_fastas(seqs_dict:dict) -> pd.DataFrame:
            seqs_dict["location"] = list()
            for d, s in zip(seqs_dict["description"], seqs_dict["sequence"]):
                seqs_dict["location"].append((fa_file := f"{seq_dir}/{d}.fa"))
                with open(fa_file, 'w', encoding="UTF-8") as f:
                    f.write(f">{d}\n{s}")
            return pd.DataFrame(seqs_dict)

        def rename_mpnn_pdb(pdb):
            '''changes single digit file extension to 4 digit file extension'''
            filename, extension = os.path.splitext(pdb)[0].rsplit('_', 1)
            filename = f"{filename}_{extension.zfill(4)}.pdb"
            shutil.move(pdb, filename)
            return


        # read .pdb files
        seq_dir = os.path.join(work_dir, 'seqs')
        pdb_dir = os.path.join(work_dir, 'backbones')
        fl = glob(f"{seq_dir}/*.fa")
        pl = glob(f"{pdb_dir}/*.pdb")
        if not fl: raise FileNotFoundError(f"No .fa files were found in the output directory of LigandMPNN {seq_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")
        if not pl: raise FileNotFoundError(f"No .pdb files were found in the output directory of LigandMPNN {pdb_dir}. LigandMPNN might have crashed (check output log), or path might be wrong!")

        seqs = [mpnn_fastaparser(fasta) for fasta in fl]
        seqs_dict = convert_ligandmpnn_seqs_to_dict(seqs)

        original_seqs_dir = os.path.join(seq_dir, 'original_seqs')
        logging.info(f"Copying original .fa files into directory {original_seqs_dir}")
        os.makedirs(original_seqs_dir, exist_ok=True)
        _ = [shutil.move(fasta, os.path.join(original_seqs_dir, os.path.basename(fasta))) for fasta in fl]

        original_pdbs_dir = os.path.join(pdb_dir, 'original_backbones')
        logging.info(f"Copying original .pdb files into directory {original_pdbs_dir}")
        os.makedirs(original_pdbs_dir, exist_ok=True)
        _ = [shutil.copy(pdb, os.path.join(original_pdbs_dir, os.path.basename(pdb))) for pdb in pl]
        _ = [rename_mpnn_pdb(pdb) for pdb in pl]

        # Write new .fa files by iterating through "description" and "sequence" keys of the seqs_dict
        logging.info(f"Writing new fastafiles at original location {seq_dir}.")
        scores = write_mpnn_fastas(seqs_dict)

        if return_seq_threaded_pdbs_as_pose:
            #replace .fa with sequence threaded pdb files as poses
            scores['location'] = [os.path.join(pdb_dir, f"{os.path.splitext(os.path.basename(series['location']))[0]}.pdb") for _, series in scores.iterrows()]

        logging.info(f"Saving scores of {self} at {scorefile}")

        scores.to_json(scorefile)

        if not preserve_original_output:
            if os.path.isdir(original_seqs_dir):
                logging.info(f"Deleting original .fa files at {original_seqs_dir}!")
                shutil.rmtree(original_seqs_dir)
            if os.path.isdir(original_pdbs_dir):
                logging.info(f"Deleting original .pdb files at {original_pdbs_dir}!")
                shutil.rmtree(original_pdbs_dir)

        return scores

def parse_residues(residues:object) -> str:
    '''parses residues from either ResidueSelection object or list or mpnn_formatted string into mpnn_formatted string.'''
    # ResidueSelection should have to_mpnn function.
    if isinstance(residues, ResidueSelection):
        return residues.to_string(delim=" ")

    # strings:
    if isinstance(residues, str):
        if len(residues.split(",")) > 1:
            return " ".join(residues.split(","))
        return residues
    raise ValueError(f"Residues must be of type str or ResidueSelection. Type: {type(residues)}")

def write_to_json(input_dict: dict, output_path:str) -> str:
    '''Writes json serializable :input_dict: into file and returns path to file.'''
    with open(output_path, 'w', encoding="UTF-8") as f:
        json.dump(input_dict, f)
    return output_path
