import json
import os
import io
import pandas as pd
import numpy as np
from dataclasses import asdict
from biotite.structure.io.pdbx import CIFFile, CIFColumn, CIFData
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    StructurePredictionInput)
from esm.utils.structure.input_builder import (
    PocketConditioning, 
    CovalentBond, 
    DistogramConditioning,
    ProteinInput,
    RNAInput, 
    LigandInput, 
    DNAInput, 
    Modification)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

def parse_seq_entry(seq):
    # Map the expected keys to their corresponding classes
    PARSERS = {
        "protein": ProteinInput,
        "dna": DNAInput,
        "rna": RNAInput,
        "ligand": LigandInput
    }

    # create modification class for each modification
    for value in seq.values():
        if "modifications" in value:
            value["modifications"] = [Modification(**moddict) for moddict in value["modifications"]]
            
    # Find the matching key and initialize the class
    for key, parser_class in PARSERS.items():
        if key in seq:
            return parser_class(**seq[key])
            
    raise KeyError("Unknown sequence entry!")

def parse_cst_entries(csts: list):
    cst_dict = {"distogram_conditioning": [], "pocket": [], "covalent_bonds": []}
    for cst in csts:
        if "distogramconditioning" in cst:
            cst_dict["distogram_conditioning"].append(DistogramConditioning(**cst["distogramconditioning"]))
        elif "pocketconditioning" in cst:
            cst_dict["pocket"].append(PocketConditioning(**cst["pocketconditioning"]))
        elif "covalentbond" in cst:
            cst_dict["covalent_bonds"].append(CovalentBond(**cst["covalentbond"]))
        else:
            raise KeyError("Unknown constraint entry!")
    
    if len(cst_dict["pocket"]) > 1:
        raise KeyError("Only single Pocket constraint allowed!")
    elif len(cst_dict["pocket"]) == 1:
        cst_dict["pocket"] = cst_dict["pocket"][0]

    for key, value in cst_dict.items():
        if not value:
            cst_dict[key] = None

    return cst_dict

def add_occupancy_to_cif_string(cif_string: str) -> str:
    """
    Default version of ESMFold2 to_mmcif does not write occupancies,
    which prevents parsing of cif files with biopython.
    Takes an mmCIF string, adds a default occupancy column (1.00) 
    to the atom_site loop, and returns the fixed mmCIF string.
    """
    # 1. Read the generated CIF string back into a Biotite CIFFile object
    cif_file = CIFFile.read(io.StringIO(cif_string))
    
    # 2. Iterate through data blocks (usually just one)
    for block_name, block in cif_file.items():
        if "atom_site" in block:
            atom_site = block["atom_site"]
            
            # 3. Find the number of atoms using an existing column
            if "id" in atom_site:
                n_atoms = len(atom_site["id"].as_array())
                
                # 4. Create an array of "1.00" strings and append as the occupancy column
                occupancies = np.full(n_atoms, "1.00", dtype=np.str_)
                atom_site["occupancy"] = CIFColumn(
                    data=CIFData(array=occupancies, dtype=np.str_)
                )
                
    # 5. Export back to a string
    output = io.StringIO()
    cif_file.write(output)
    return output.getvalue()


def main(args):

    model = ESMFold2Model.from_pretrained(args.model).cuda().eval()

    with open(args.input_json, 'r', encoding="UTF-8") as j:
        poses = json.load(j)

    data = []
    for pose in poses:
        with open(poses[pose]["pose_path"], 'r', encoding="UTF-8") as j:
            pose_dict = json.load(j)

        # set constraints to empty list if not defined
        pose_dict.setdefault("constraints", [])

        # extract options
        pose_opts = poses[pose]["options"]

        # parse sequences (protein, dna, rna, ligand)
        seqs = [parse_seq_entry(seq) for seq in pose_dict["sequences"]]

        # parse constraints (covalent bonds, pocket, distogram)
        csts = parse_cst_entries(pose_dict["constraints"])

        # fold
        spi = StructurePredictionInput(sequences=seqs, **csts)
        result = ESMFold2InputBuilder().fold(model=model, input=spi, **pose_opts)

        # if num_diffusion_samples > 1, a list of results is returned
        if not isinstance(result, list):
            result = [result]

        for i, out in enumerate(result):
            # define filename
            description = f"{pose}_{str(i+1).zfill(4)}"
            filename = description + ".cif"
            location = os.path.abspath(os.path.join(args.output_dir, filename))

            # write output cif
            with open(location, "w", encoding="UTF-8") as f:
                raw = out.complex.to_mmcif()
                # esmfold2 to_mmcif does not create occupancy column, which prevents cif parsing with biopython
                added_occupancies = add_occupancy_to_cif_string(raw)
                f.write(added_occupancies)

            # collect data in dict
            result_dict = asdict(out)
            json_ready_dict = {}
            for key, value in result_dict.items():
                if key == "plddt" and value is not None:
                    # FIX 1 & 2: Use .item() for the mean, and .cpu().tolist() for the array
                    json_ready_dict["mean_per_res_plddt"] = value.mean().item()
                    json_ready_dict["per_res_plddt"] = value.cpu().tolist()
                if key in ["ptm", "iptm"] and value is not None:
                    json_ready_dict[key] = float(value)
                else:
                    # skip all other attributes as they inflate output json
                    # TODO: add flag to include additional data
                    continue

            json_ready_dict.update({"description": description, "location": location})

            data.append(json_ready_dict)

    data = pd.DataFrame(data).reset_index()
    name = str(os.path.basename(args.input_json)).replace(".json", "_out.json")
    
    data.to_json(os.path.join(args.output_dir, name))
            

if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--model", type=str, default="biohub/ESMFold2", help="Name of ESMFold2 model (e.g. 'biohub/ESMFold2')")
    argparser.add_argument("--input_json", type=str, help="Path to input json.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")


    args = argparser.parse_args()


    main(args)
