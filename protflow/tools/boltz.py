"""Module Docstring is missing!"""
# imports
import re
import os
import logging
import shutil
import glob
import json

# dependencies
from Bio import SeqIO
import pandas as pd
from .. import config, jobstarters
from ..poses import Poses
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..jobstarters import JobStarter

# config file: 
# BOLTZ_SCRIPT_PATH = "/home/az3556/boltz/src/boltz/main.py"
# BOLTZ_PYTHON_PATH = "/home/az3556/anaconda3/envs/protflow/bin/python3.11" #needs python 3.9 or higher
# BOLTZ_PRE_CMD = ""

class Boltz(Runner):
    def __init__(self, script_path: str = config.BOLTZ_SCRIPT_PATH, python_path: str = config.BOLTZ_PYTHON_PATH, pre_cmd: str = config.BOLTZ_PRE_CMD, jobstarter: str = None) -> None: 
        if not script_path:
            raise ValueError(f"No path is set for main.py. Set the path in config.py under BOLTZ_SCRIPT_PATH.")

        self.script_path = script_path
        self.python_path = python_path
        self.name = "boltz.py"
        self.pre_cmd = pre_cmd
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "boltz.py"

    def run(self, 
        poses: Poses, 
        prefix: str, 
        jobstarter: JobStarter = None, 
        overwrite: bool = False,
        **kwargs
    ) -> Poses:

        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")

        scorefile = os.path.join(work_dir, f"boltz_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses without rerunning calculations.")
            return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

        if overwrite:
            for folder in ["boltz_preds", "input_files", "output_files"]:
                dir_path = os.path.join(work_dir, folder)
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)

        os.makedirs(output_dir := os.path.join(work_dir, "boltz_preds"), exist_ok=True)
        os.makedirs(input_files := os.path.join(work_dir, "input_files"), exist_ok=True)
        os.makedirs(output_files := os.path.join(work_dir, "output_files"), exist_ok=True)

        file_path = poses.df['poses'].iloc[0] 
        logging.info(f"Expected FASTA file: {file_path}")
        input_paths = []
        for file_path in poses.df['poses']:
            logging.info(f"Processing input file: {file_path}")
            processed_file = create_input_files(input_files, file_path)
            input_paths.append(processed_file)

        input_path = input_files  
        logging.info(f"input_files contents: {os.listdir(input_files)}")
        cmds = [self.write_cmd(
            input_path=input_path, 
            output_dir=output_dir,
            **kwargs
        )]

        if self.pre_cmd:
            cmds = prepend_cmd(cmds=cmds, pre_cmd=self.pre_cmd)

        logging.info(f"Starting Boltz predictions of {len(poses)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(cmds=cmds, jobname="boltz_prediction", wait=True, output_path=work_dir)

        logging.info("Renaming output files.")
        rename_boltz_outputs(output_dir=output_dir)

        logging.info("Copying structure files to output_files folder.")
        copy_structure_files(work_dir)

        scores = collect_scores(working_dir=work_dir, input_files=input_paths)
        if isinstance(scores, dict):
            scores = pd.DataFrame(scores)
        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)
        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than input poses. Some runs might have crashed!")
        #print(f"scores: {scores}")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()

    def parse_input(self, input_path: str, input_files: str):
        """Parses input YAML or FASTA, without converting between formats."""
        fasta_dirs = []
        if os.path.isdir(input_path): 
            logging.info(f"Scanning directory: {input_path}")
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                fasta_dirs.extend(self.parse_input(file_path, input_files)) 
            return fasta_dirs
        if input_path.endswith(".yaml") or input_path.endswith(".yml"):
            logging.info(f"Detected YAML input: {input_path}")
            fasta_dirs.extend(self.handle_yaml(input_path, input_files))
        elif input_path.endswith(".fasta") or input_path.endswith(".fa"):
            logging.info(f"Detected FASTA input: {input_path}")
            fasta_dirs.append(input_path)
        else:
            raise ValueError("Unsupported input format. Provide a FASTA or YAML file.")
        return fasta_dirs

    def handle_yaml(self, yaml_file, input_files):
        """Handles YAML file without converting to FASTA."""
        os.makedirs(input_files, exist_ok=True)

        with open(yaml_file, "r", encoding="UTF-8") as file:
            yaml_data = yaml.safe_load(file)

        fasta_files = []
        for entry in yaml_data.get("entries", []):
            fasta_files.append(entry)

        return fasta_files

    def write_cmd(self, input_path: str, output_dir: str, **kwargs) -> str:
        valid_options = ["output_format", "cache", "checkpoint", "devices", "accelerator", "recycling_steps", "sampling_steps", "diffusion_samples", "step_scale", "output_format", "num_workers", "override", "use_msa_server", "msa_server_url", "msa_pairing_strategy", "write_full_pae", "write_full_pde"]
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in valid_options and value not in [None, False]}
        opts = " ".join([f"--{key} {value}" if not isinstance(value, bool) else f"--{key}" for key, value in filtered_kwargs.items()])
        return f"boltz predict {input_path} --out_dir {output_dir} {opts}"

def collect_scores(working_dir: str, input_files: list) -> pd.DataFrame:
    scores_dict = {
        "confidence_score": {},
        "ptm": {},
        "iptm": {},
        "ligand_iptm": {},
        "protein_iptm": {},
        "complex_plddt": {},
        "complex_iplddt": {},
        "complex_pde": {},
        "complex_ipde": {},
        "chains_ptm": {},
        "pair_chains_iptm": {},
        "description": {},
        "location": {}
    }
    index = 0
    processed_structures = set()
    for input_file in input_files:
        input_base_name = os.path.basename(input_file).split('.')[0].replace('confidence_', '')
        json_files = glob.glob(os.path.join(working_dir, '**', '**', '**', f'confidence_{input_base_name}_*.json'), recursive=True)
        structure_files = glob.glob(os.path.join(working_dir, "output_files", "*.pdb"), recursive=True) + \
                          glob.glob(os.path.join(working_dir, "output_files", "*.mmcif"), recursive=True)
        for json_file in json_files:
            try:
                json_base_name = os.path.basename(json_file).replace('confidence_', '').replace('.json', '').strip()
                if json_base_name in processed_structures:
                    #print(f"Skipping {json_base_name}, already processed.")
                    continue 
                matching_structure = None
                for structure_file in structure_files:
                    output_base_name = os.path.splitext(os.path.basename(structure_file))[0].strip() 
                    #print(f"json_base_name = {json_base_name}, output_base_name = {output_base_name}")
                    if json_base_name == output_base_name and json_base_name not in processed_structures:
                        matching_structure = structure_file
                        processed_structures.add(json_base_name)
                        #print(f"Processed structures so far: {processed_structures}")
                        break
                if not matching_structure:
                    print(f"Warning: No matching structure found for {json_file}")
                    continue
                with open(json_file, "r", encoding="UTF-8") as f:
                    score_data = json.load(f)
                scores_dict["confidence_score"][str(index)] = score_data.get("confidence_score", 0)
                scores_dict["ptm"][str(index)] = score_data.get("ptm", 0)
                scores_dict["iptm"][str(index)] = score_data.get("iptm", 0)
                scores_dict["ligand_iptm"][str(index)] = score_data.get("ligand_iptm", 0)
                scores_dict["protein_iptm"][str(index)] = score_data.get("protein_iptm", 0)
                scores_dict["complex_plddt"][str(index)] = score_data.get("complex_plddt", 0)
                scores_dict["complex_iplddt"][str(index)] = score_data.get("complex_iplddt", 0)
                scores_dict["complex_pde"][str(index)] = score_data.get("complex_pde", 0)
                scores_dict["complex_ipde"][str(index)] = score_data.get("complex_ipde", 0)
                scores_dict["chains_ptm"][str(index)] = score_data.get("chains_ptm", 0)
                scores_dict["pair_chains_iptm"][str(index)] = score_data.get("pair_chains_iptm", {})
                scores_dict["description"][str(index)] = output_base_name
                scores_dict["location"][str(index)] = os.path.abspath(matching_structure)
                index += 1
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
    scores = pd.DataFrame(scores_dict)
    return scores

def copy_structure_files(work_dir: str):
    source_dir = os.path.join(work_dir, "boltz_preds")
    destination_dir = os.path.join(work_dir, "output_files")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    os.makedirs(destination_dir, exist_ok=True)
    structure_files = glob.glob(os.path.join(source_dir, "**", "**", "*.pdb"), recursive=True) + \
                      glob.glob(os.path.join(source_dir, "**", "**", "*.mmcif"), recursive=True)
    if not structure_files:
        logging.warning("No structure files found in boltz_preds.")
    for file_path in structure_files:
        file_name = os.path.basename(file_path)  
        dest_path = os.path.join(destination_dir, file_name)
        shutil.copy(file_path, dest_path)
        logging.info(f"Copied {file_path} to {dest_path}")

VALID_ENTITY_TYPES = {"protein", "dna", "rna", "smiles", "ccd"}
def create_input_files(input_files: str, file_path: str) -> str:
    file_extension = os.path.splitext(file_path)[1]
    os.makedirs(input_files, exist_ok=True)
    if file_extension in [".fasta", ".fa"]:
        input_filename = os.path.join(input_files, os.path.basename(file_path))
        with open(file_path, "r", encoding="UTF-8") as original_fasta_file:
            records = list(SeqIO.parse(original_fasta_file, "fasta"))
        with open(input_filename, "w", encoding="UTF-8") as fasta_file:
            for seq_record in records:
                header_parts = seq_record.description.split("|")
                chain_id = "A"
                for part in header_parts:
                    match = re.search(r"Chain (\w)", part)
                    if match:
                        chain_id = match.group(1)
                        break
                    elif len(part) == 1 and part.isalpha():
                        chain_id = part
                        break
                entity_type = next((part for part in header_parts if part.lower() in VALID_ENTITY_TYPES), "protein")
                msa_path = next((part for part in header_parts if "." in part and "/" in part), None)
                new_header = "|".join(filter(None, [chain_id, entity_type, msa_path]))
                fasta_file.write(f">{new_header}\n{seq_record.seq}\n")
        return input_filename
    elif file_extension in [".yaml", ".yml"]:
        input_filename = os.path.join(input_files, os.path.basename(file_path))
        shutil.copy(file_path, input_filename)
        return input_filename
    else:
        raise ValueError("Unsupported file format. Please provide a .fasta or .yaml file.")
    
def rename_boltz_outputs(output_dir: str):
    boltz_results_files_pattern = os.path.join(output_dir, "**", "**", "**", "*") 
    boltz_results_files = glob.glob(boltz_results_files_pattern, recursive=True)
    if not boltz_results_files:
        print(f"No files found within {output_dir}")
        return
    extensions = ['.json', '.npz', '.pdb', '.mmcif']
    for file_path in boltz_results_files:
        if any(file_path.endswith(ext) for ext in extensions) and 'model_' in file_path:
            file_name = os.path.basename(file_path)
            base_name, file_extension = os.path.splitext(file_name)
            model_match = re.search(r'_model_(\d+)', base_name)
            if model_match:
                model_number = model_match.group(1)
                new_model_number = f"{int(model_number)+1:04d}"  
                new_name = base_name.replace(f'_model_{model_number}', f'_{new_model_number}')
                new_path = os.path.join(os.path.dirname(file_path), new_name + file_extension)
                if os.path.exists(new_path):
                    #print(f"Skipping renaming: {new_path} already exists.")
                    continue  
                try:
                    #print(f"Renaming {file_path} to {new_path}")
                    os.rename(file_path, new_path)
                except Exception as e:
                    print(f"Error renaming {file_path}: {e}")
