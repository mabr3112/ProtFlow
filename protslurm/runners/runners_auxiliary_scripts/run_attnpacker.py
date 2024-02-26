# coding: utf-8

import os
import sys
import csv



def main(args):

    code_root = os.path.dirname(os.path.dirname(os.getcwd()))
    if code_root not in sys.path:
        print(f"Added {code_root} to python path")
        sys.path.append(code_root)
    # Faster Inference on CPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'

    sys.path.append(args.attnpacker_dir)
    import torch
    from protein_learning.models.inference_utils import (Inference, make_predicted_protein, default)
    import protein_learning.common.protein_constants as pc
    import time
    from protein_learning.protein_utils.sidechains.project_sidechains import project_onto_rotamers

    def project_coords_to_rotamers(
        protein,
        steric_clash_weight=1.,
        optim_repeats=2,
        max_optim_iters=100,
        steric_loss_kwargs=None,
        device="cpu",
        angle_wt=0
    ):
        projected_coords, _ = project_onto_rotamers(
            atom_coords = protein.atom_coords.unsqueeze(0),
            sequence = protein.seq_encoding.unsqueeze(0),
            atom_mask = protein.atom_masks.unsqueeze(0),
            steric_clash_weight=steric_clash_weight,
            optim_repeats = optim_repeats,
            steric_loss_kwargs = default(
                steric_loss_kwargs,
                dict(
                    hbond_allowance = 0.6,
                    global_allowance = 0.05,
                    global_tol_frac = 0.95,
                    top_k = 32 # number of neighboring atoms to consider in steric calculations
                )
            ),
            # set this to smaller value to trade off accuracy and speed.
            # use >= 500 for highest accuracy and ~50 for speed.
            max_optim_iters = max_optim_iters,
            torsion_deviation_loss_wt = angle_wt,
        )
        return projected_coords.squeeze(0)

    model_weights = os.path.join(args.attnpacker_dir, f"weights/AttnPackerPTM_V2")

    RESOURCE_ROOT = model_weights
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Wrapper class for inference
    runner = Inference(RESOURCE_ROOT, use_design_variant = False)
    runner = runner.to(DEVICE)

    os.makedirs(args.output_dir, exist_ok=True)
    pdb_path = args.input_pdb
    start = time.time()
    prediction = runner.infer(
        pdb_path=pdb_path,
        #Boolean Tensor indicating which residues to design
        seq_mask=None,
        #Whether to format output (process into logits, seq labels, etc. or return raw node and pair output)
        format=True,
        #Chunk inference by successively packing smaller crops of size chunk_size
        #This allows packing of arbitrarily long proteins
        chunk_size = 500,
    )
    print(f"Ran Inference on {runner.device} in time {round(time.time()-start,2)} seconds")


    predicted_protein = make_predicted_protein(model_out = prediction['model_out'], seq = prediction['seq'])

    # save optimized packing to the path below
    pp_pdb_out_name = str(predicted_protein.name) + "_" + "1".zfill(4)
    pp_pdb_out_path = os.path.join(args.output_dir, f"{pp_pdb_out_name}.pdb")
    #Please read the doc string for more details
    projected_coords = project_coords_to_rotamers(predicted_protein,device=DEVICE)
    # write new pdb using optimized coordinates, write plddt values in bfactor column
    predicted_protein.to_pdb(pp_pdb_out_path, coords=projected_coords, beta=prediction["pred_plddt"].squeeze())
    sc_plddts = ",".join([str(round(float(i)*100, 2)) for i in prediction["pred_plddt"]])

    print(type(prediction))

    out_dict = {'description': pp_pdb_out_name, 'location': pp_pdb_out_path, 'attnpacker_sc_plddts': sc_plddts}
    
    field_names = list(out_dict.keys())
    
    # Opening the CSV file in append mode
    file_exists = os.path.isfile(args.scorefile)
    with open(args.scorefile, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        if not file_exists: writer.writeheader()
        # Writing the dictionary as a new row
        writer.writerow(out_dict)






if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--attnpacker_dir", type=str, help="Path to AttnPacker directory")
    argparser.add_argument("--input_pdb", type=str, default=".", help="Path to folder containing input pdb files")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    argparser.add_argument("--scorefile", type=str, default="attnpacker_scores.csv", help="Path to output folder")

    args = argparser.parse_args()


    main(args)