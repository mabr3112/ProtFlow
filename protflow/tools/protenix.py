# general imports
import os
import logging
from glob import glob
import shutil
import json
import subprocess
from random import randint
from pathlib import Path

# dependencies
import pandas as pd

# custom
from .. import require_config, load_config_path, runners
from ..runners import Runner, RunnerOutput, prepend_cmd
from ..poses import Poses, col_in_df, description_from_path
from ..jobstarters import JobStarter, split_list
from ..utils.biopython_tools import load_sequence_from_fasta
from ..utils.openbabel_tools import openbabel_fileconverter

# TODO: implement class ProtenixMSA(Runner), class ProtenixMT(Runner), class ProtenixPrep(Runner):


class ProtenixPred(Runner):
    def __init__(
            self,
            bin_path: str|None = None,
            pre_cmd: str|None = None,
            jobstarter: str = None
        ) -> None:

        # setup configs
        config = require_config()
        self.bin_path = bin_path or load_config_path(config, path_var="PROTENIX_BIN_PATH")
        self.pre_cmd = pre_cmd or load_config_path(config, path_var="PROTENIX_PRE_CMD", is_pre_cmd=True)

        # runner setups
        self.name = "protenix.py"
        self.index_layers = 1
        self.jobstarter = jobstarter

    def __str__(self):
        return "colabfold.py"

    def run(self, poses: Poses, prefix: str, nstruct: int = 1, json_column: str = None, num_copies: int = 1, msa_paired: str = None, msa_unpaired: str = None, 
                          templates: str = None, modifications: str | list | dict = None, ligands: str | list | dict = None, ions: str | list | dict = None, additional_entities: str | list | dict = None, 
                          covalent_bonds: str | list | dict = None, constraints: str | list | dict = None, options: str = None, pose_options: str = None,
            jobstarter: JobStarter = None, overwrite: bool = False, return_top_n_models: int = 1, convert_cif_to_pdb: bool = True, 
            seeds: list | int | str = "random", chain_sep: str = ":") -> Poses:

        # setup runner
        work_dir, jobstarter = self.generic_run_setup(
            poses=poses,
            prefix=prefix,
            jobstarters=[jobstarter, self.jobstarter, poses.default_jobstarter]
        )

        logging.info(f"Running {self} in {work_dir} on {len(poses.df.index)} poses.")
        print("starting")
        # Look for output-file in pdb-dir. If output is present and correct, then skip Colabfold.
        scorefile = os.path.join(work_dir, f"protenix_scores.{poses.storage_format}")
        if (scores := self.check_for_existing_scorefile(scorefile=scorefile, overwrite=overwrite)) is not None:
            logging.info(f"Found existing scorefile at {scorefile}. Returning {len(scores.index)} poses from previous run without running calculations.")
            output = RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers)
            return output.return_poses()
        
        if overwrite:
            if os.path.isdir(json_dir := os.path.join(work_dir, "input_json")):
                shutil.rmtree(json_dir)
            if os.path.isdir(preds_dir := os.path.join(work_dir, "protenix_preds")):
                shutil.rmtree(preds_dir)
            if os.path.isdir(pdb_dir := os.path.join(work_dir, "output_predictions")):
                shutil.rmtree(pdb_dir)

        # setup af3-specific directories:
        os.makedirs(json_dir := os.path.join(work_dir, "input_jsons"), exist_ok=True)
        os.makedirs(preds_dir := os.path.join(work_dir, "protenix_preds"), exist_ok=True)

        # create input json files
        if json_column:
            in_jsons = []
            col_in_df(poses.df, json_column)

            # import jsons
            dicts = [read_json(path) for path in poses.df[json_column].to_list()]
            
            # make sure each json has the correct pose name assigned
            for d, name in zip(dicts, poses.df["poses_description"].to_list()):
                d["name"] = name

        else:
            in_jsons = self.create_input_dicts(poses, work_dir, num_copies, msa_paired, msa_unpaired, templates, modifications, ligands, ions, additional_entities, covalent_bonds, constraints, chain_sep)

        if seeds:
            if isinstance(seeds, int):
                seeds = [seeds]
            if seeds == "random":
                seeds = [randint(1, 10000) for _ in range(nstruct)]
            if not isinstance(seeds, list) or not len(seeds) == nstruct:
                raise ValueError(f"Number of seeds must be equal to nstruct. Seeds: {seeds}, nstruct: {nstruct}")
        else:
            seeds = list(range(0, nstruct))
        seeds = [str(seed) for seed in seeds]

        if pose_options:
            # prepare pose options
            pose_options = self.prep_pose_options(poses=poses, pose_options=pose_options)
            in_jsons = [write_json(in_dict, os.path.join(json_dir, f"batch_{i}.json")) for i, in_dict in enumerate(in_jsons)]

            cmds = [self.write_cmd(in_json=in_json, output_dir=preds_dir, seeds=seeds, options=options, pose_options=pose_opt) for in_json, pose_opt in zip([in_jsons, pose_options])]
        
        else:
            # setup input-fastas in batches (to speed up prediction times.), but only if no pose_options are provided!
            num_batches = min(len(poses.df.index), jobstarter.max_cores)

            # split dicts into batches
            dicts_batches = split_list(in_jsons, n_sublists=num_batches)

            # write dicts to json
            in_jsons = [write_json(batch, os.path.join(json_dir, f"batch_{i}.json")) for i, batch in enumerate(dicts_batches)]

            cmds = [self.write_cmd(in_json=in_json, output_dir=preds_dir, seeds=seeds, options=options, pose_options=None) for in_json in in_jsons]

        # prepend pre-cmd if defined:
        if self.pre_cmd:
            cmds = prepend_cmd(cmds = cmds, pre_cmd=self.pre_cmd)

        # run
        logging.info(f"Starting Protenix predictions of {len(poses.df.index)} sequences on {jobstarter.max_cores} cores.")
        jobstarter.start(
            cmds=cmds,
            jobname="ProtenixPred",
            wait=True,
            output_path=work_dir
        )

        # collect scores
        logging.info("Predictions finished, starting to collect scores.")
        scores = collect_scores(work_dir=work_dir, convert_cif_to_pdb=convert_cif_to_pdb, return_top_n_models=return_top_n_models)

        if len(scores.index) < len(poses.df.index):
            raise RuntimeError("Number of output poses is smaller than number of input poses. Some runs might have crashed!")

        logging.info(f"Saving scores of {self} at {scorefile}")
        self.save_runner_scorefile(scores=scores, scorefile=scorefile)

        logging.info(f"{self} finished. Returning {len(scores.index)} poses.")

        return RunnerOutput(poses=poses, results=scores, prefix=prefix, index_layers=self.index_layers).return_poses()
    
    def write_cmd(self, in_json: str, output_dir: str, seeds: list, options: str = None, pose_options: str = None):

        # parse options
        opts, flags = runners.parse_generic_options(options=options, pose_options=pose_options, sep="--")
        opts = " ".join([f"--{key} {value}" for key, value in opts.items()])
        flags = " --" + " --".join(flags) if flags else ""
        return f"{self.bin_path} pred -i {in_json} -o {output_dir} -s {','.join(seeds)} {opts} {flags}"
    
    def create_input_dicts(self, poses: Poses, work_dir: str, num_copies: int, msa_paired: str = None, msa_unpaired: str = None, 
                            templates: str = None, modifications: str | list | dict = None, ligands: str | list | dict = None, ions: str | list | dict = None, additional_entities: str | list | dict = None, 
                            covalent_bonds: str | list | dict = None, constraints: str | list | dict = None, chain_sep: str = ":") -> list:
            
        def add_msa_template_modifications(term: str, in_type:str, pose_dict: dict, pose_row: pd.Series) -> dict:

            if not term:
                return pose_dict
            
            # check if pose-specific msa is provided
            if term in pose_row:
                term = pose_row[term]

            # check if msa exists
            if not in_type == "modifications" and not os.path.isfile(term):
                raise ValueError(f"Could not detect msa or template at {term} for pose {pose_row['poses_description']}.")

            if in_type == "modifications":
                if isinstance(term, dict):
                    term = [term]
                if not isinstance(term, list):
                    raise ValueError("Modifications must be specified via dictionary/list of dictionaries or a poses dataframe column containing dictionaries!")
            
            # check number of chains in dict
            num_prot_chains = sum(1 for d in pose_dict["sequences"] if "proteinChain" in d)
            num_rna_chains = sum(1 for d in pose_dict["sequences"] if "rnaSequence" in d)

            if num_prot_chains + num_rna_chains > 1:
                raise ValueError("MSAs and templates cannot be specified via options if multiple sequences are present in the input structures.\n" \
                "Create jsons containing MSA/template paths manually and specify <json_col>.")

            pose_dict[in_type] = term

            return pose_dict
        
        def add_additional_entities(pose_dict: dict, pose_row: pd.Series, ligands: str | list | dict = None, ions: str | list | dict = None, 
                additional_entities: str | list | dict = None):
            
            if ligands:
                pose_dict["sequences"] = pose_dict["sequences"] +  identify_entities(pose_row, ligands, "ligand")
            if ions:
                pose_dict["sequences"] = pose_dict["sequences"] +  identify_entities(pose_row, ions, "ions")
            if additional_entities:
                additional_entities = check_entity(additional_entities)
                pose_dict["sequences"] = pose_dict["sequences"] +  additional_entities
            
            return pose_dict
        
        def identify_entities(pose_row: pd.Series, entity: str | list | dict, entity_type: str):
            if isinstance(entity, str):
                if entity in pose_row:
                    entity = pose_row[entity]
                    entity = identify_entities(pose_row, entity, entity_type)
                else:
                    entity = {entity_type: {entity_type: entity, "count": 1}}
            if isinstance(entity, list):
                entity = [identify_entities(pose_row, ent, entity_type) for ent in entity]
                check_entity(entity)
            if isinstance(entity, dict):
                entity = check_entity(entity)
            else:
                raise ValueError("Ligands and ions must be specified via dictionary, a list of dictionaries or SMILES/sdf paths/ion CCD code strings OR a poses dataframe column containing these.")

            return entity

        def check_entity(entity: dict | list):

            if isinstance(entity, dict):
                mandatory_keys = ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]
                if not any(key in entity for key in mandatory_keys):
                    raise ValueError(f"Input entity type must be one of {mandatory_keys}. Affected entity: {entity}")
                return [entity]
                
            if isinstance(entity, list):
                for ent in entity:
                    check_entity(ent)
                
                return entity
            
            else:
                raise ValueError(f"Additional entities must be provided as a dict or list of dicts format, not {type(entity)}! Affected entity is {entity}")

        def identify_bonds(pose_row: pd.Series, bonds: str | list | dict):
            if isinstance(bonds, str) and bonds in pose_row:
                bonds = pose_row[bonds]
            if isinstance(bonds, list):
                if not all(isinstance(bond, dict) for bond in bonds):
                    raise ValueError(f"Restraints must be specified via dictionary. Wrong restraint: {bonds}")
            if isinstance(bonds, dict):
                bonds = [bonds]
            else:
                raise ValueError("Covalent bonds must be specified via dict, a list of dicts or a poses dataframe column containing these.")

            return bonds
        
        def create_dict_from_fa(name: str, path: str, sep:str=":"):
            seqs = str(load_sequence_from_fasta(path, return_multiple_entries=False).seq)
            protchains = [{"proteinChain": {"sequence": seq, "count": 1}} for seq in seqs.split(sep)]
            in_dict = {"name": name, "sequences": protchains}
            return in_dict

        # determine pose type to create input dictionaries
        input_type = poses.determine_pose_type(pose_col="poses")
        if input_type == [".fa"] or input_type == [".fasta"]:
            # create input dict for each pose
            poses.df["temp_protenix_in_dict_col"] = poses.df.apply(lambda row: create_dict_from_fa(name=row["poses_description"], path=row["poses"], sep=chain_sep), axis=1)
            
            # create a list
            dicts = poses.df["temp_protenix_in_dict_col"].to_list()

            # drop temp col
            poses.df.drop(["temp_protenix_in_dict_col"], axis=1, inplace=True)

        elif input_type == [".pdb"] or input_type == [".cif"]:
            # create temp folders
            os.makedirs(temp_jsons := os.path.join(work_dir, "temp_jsons"), exist_ok=True)
            os.makedirs(temp_pdbs := os.path.join(work_dir, "temp_pdbs"), exist_ok=True)

            # copy all poses into same dir
            for pose in poses.poses_list():
                shutil.copy(pose, temp_pdbs)

            # create input dict using protenix structure to json function
            json_from_structure(protenix_path=str(self.bin_path), in_dir=temp_pdbs, out_dir=temp_jsons)

            # gather files
            jsons = glob(os.path.join(temp_jsons, "*.json"))

            # merge with original df
            temp_df = pd.DataFrame({"temp_protenix_json_paths": jsons, "poses_description": [description_from_path(j) for j in jsons]})
            poses.df = poses.df.merge(temp_df, how="left", on="poses_description") # left to preserve order

            # create dicts
            dicts = [read_json(json_path, strip_list=True) for json_path in poses.df["temp_protenix_json_paths"].to_list()]
            
            # delete temp
            shutil.rmtree(temp_jsons)
            shutil.rmtree(temp_pdbs)
            poses.df.drop(["temp_protenix_json_paths"], axis=1, inplace=True)

        else:
            raise ValueError(f"Invalid input pose format: {input_type}")
        
        records = []
        for in_dict, (_, row) in zip(dicts, poses.df.iterrows()):
            in_dict["name"] = row["poses_description"]

            # update number of copies
            for seq_rec in in_dict["sequences"]:
                for chain_type in seq_rec:
                    seq_rec[chain_type]["count"] = seq_rec[chain_type]["count"] * num_copies

            # add templates, msas and modifications
            in_dict = add_msa_template_modifications(msa_paired, "pairedMsaPath", in_dict, row)
            in_dict = add_msa_template_modifications(msa_unpaired, "unpairedMsaPath", in_dict, row)
            in_dict = add_msa_template_modifications(templates, "templatesPath", in_dict, row)
            in_dict = add_msa_template_modifications(modifications, "modifications", in_dict, row)

            # add ions, ligands, additional sequences
            in_dict = add_additional_entities(in_dict, row, ligands, ions, additional_entities)

            if covalent_bonds:
                in_dict["covalent_bonds"] = identify_bonds(row, covalent_bonds)
            if constraints:
                # check for pose-specific constraint
                if isinstance(constraints, str) and constraints in row:
                    constraints = row[constraints]
                # unwrap constraints
                if isinstance(constraints, dict) and "constraint" in constraints:
                    constraints = constraints["constraint"]
                if not isinstance(constraints, dict):
                    raise ValueError("Constraints must be specified using a dict or a dataframe column containing dicts.")
                in_dict["constraint"] = constraints

            records.append(in_dict)

        return records


def json_from_structure(protenix_path: str, in_dir: str, out_dir:str, altloc:str=None, assembly_id:str=None, include_discont_poly_poly_bonds:bool=False) -> str:

    cmd = f"{protenix_path} json -i {in_dir} -o {out_dir}"

    if altloc:
        cmd = cmd + f" --altloc {altloc}"
    if assembly_id:
        cmd = cmd + f" --assembly_id {assembly_id}"
    if include_discont_poly_poly_bonds:
        cmd = cmd + " --include_discont_poly_poly_bonds"

    try:
        # Run the command
        subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)

    except subprocess.CalledProcessError as e:
        # This triggers if the protenix command returns an error code
        print(f"Error when creating input dictionary from {in_dir}: {e.stderr}")


def collect_scores(work_dir: str, convert_cif_to_pdb: bool = True, return_top_n_models: int = 1) -> pd.DataFrame:

    def cif_to_pdb(input_cif: str, output_format: str, output:str):
        openbabel_fileconverter(input_file=input_cif, output_format=output_format, output_file=output)
        return output

    # collect all output directories, ignore mmseqs dirs
    out_dirs = [d for d in glob(os.path.join(work_dir, "protenix_preds", "*")) if os.path.isdir(d) and not os.path.basename(d) == "ERR"]

    scores = []
    for out_dir in out_dirs:
        counter = 1
        name = Path(out_dir).name
        seeds_dirs = [d for d in glob(os.path.join(out_dir, "seed_*"))]
        for seed_dir in seeds_dirs:
            seed = int(seed_dir.split("_")[-1])
            jsons = glob(os.path.join(seed_dir, "predictions", "*.json"))
            seed_df = []
            for j in jsons:
                ser = pd.read_json(j, typ="series")
                sample = description_from_path(j).split("_")[-1] # extract sample number
                ser["sample"] = sample
                ser["temp_location"] = os.path.join(seed_dir, "predictions", f"{name}_sample_{sample}.cif")
                seed_df. append(ser)
            seed_df = pd.DataFrame(seed_df)
            seed_df.sort_values("ranking_score", ascending=False, inplace=True)
            seed_df = seed_df.head(return_top_n_models)
            seed_df["seed"] = seed
            seed_df["input"] = name
            seed_df["temp_counter"] = range(counter, counter + len(seed_df)) # assign a temp counter for later renaming
            scores.append(seed_df)
            counter += len(seed_df) # update counter
    
    scores = pd.concat(scores)
    scores.reset_index(drop=True, inplace=True)

    os.makedirs(out_dir := os.path.join(work_dir, "output_predictions"), exist_ok=True)

    # convert to pdb, otherwise rename files (without renaming, wrong number of index layers would be added)
    if convert_cif_to_pdb:
        scores["location"] = scores.apply(lambda row: cif_to_pdb(input_cif=row["temp_location"], output_format="pdb", output=os.path.abspath(os.path.join(out_dir, f"{row['input']}_{str(row['temp_counter']).zfill(4)}.pdb"))), axis=1)
    else:
        scores["location"] = scores.apply(lambda row: shutil.copy(row["temp_location"], os.path.abspath(os.path.join(out_dir, f"{row['input']}_{str(row['temp_counter']).zfill(4)}.pdb"))), axis=1)
    
    # create description col
    scores["description"] = scores.apply(lambda row: description_from_path(row["location"]), axis=1)

    # drop temp cols
    scores.drop([col for col in scores.columns if col.startswith("temp_")], axis=1, inplace=True)
    return scores

def read_json(path, strip_list=False):
    with open(path, 'r', encoding="UTF-8") as j:
        data = json.load(j)

    if strip_list:
        data = data[0]

    return data

def write_json(data, path):
    with open(path, 'w', encoding="UTF-8") as j:
        json.dump(data, j, indent=2)
    return path
