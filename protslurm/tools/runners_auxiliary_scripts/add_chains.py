'''Python Script to add chains into proteins as .pdb files.'''
# imports
import json


def main(args) -> None:
    "YEEEESS"
    # parse inputs
    if args.input_json:
        with open(args.input_json, 'r', encoding="UTF-8") as f:
            poses_dict = json.loads(f.read())
    elif (args.target and args.reference_refeerence):
        poses_dict = {args.target, args.reference_refeerence}
    
    # copy chains into poses and save
    for target, reference in poses_dict.items():

        
    # end


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # inputs
    argparser.add_argument("--target", type=str, help="Path to target PDB to copy into.")
    argparser.add_argument("--reference", type=str, help="Path to reference PDB to copy from.")
    argparser.add_argument("--input_json", type=str, help="Path to json mapping of multiple PDBs (for batch runs). Dict has to be {target: reference, ...}")

    # options

    # outputs
    argparser.add_argument("--inplace", type=str, default="False")
    argparser.add_argument("--output_dir", type=str, required=True, help="Directory where to write the output .pdb files to.")
    arguments = argparser.parse_args()

    # arguments safetycheck:
    if arguments.input_json and (arguments.target or arguments.reference):
        raise ValueError(f"If --input_json is specified, target and reference are not allowed!")
    elif not (arguments.target or arguments.reference):
        raise ValueError(f"Both --target and --reference MUST be specified for RMSD calculation if --input_json is not given!")
    
    # check 'inplace' bool:
    if arguments.inplace.lower() in ["1", "true", "yes"]:
        arguments.inplace = True
    else:
        arguments.inplace = False
    
    main(arguments)
