import pandas as pd

from protslurm.utils.biopython_tools import determine_protparams

def main(args):

    in_df = pd.read_json(args.input_json)
    out_df = []
    for i, series in in_df.iterrows():
        params = determine_protparams(seq=series['sequence'], pH=args.pH)
        params['description'] = series['name']
        out_df.append(params)
    out_df = pd.concat(out_df).reset_index(drop=True)
    out_df.to_json(args.output_path)
    
if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # options
    argparser.add_argument("--input_json", type=str, help=".json formatted input file. should contain one column called 'name' and one column called 'sequence'.")
    argparser.add_argument("--output_path", type=str, help="path were output .json file is saved.")
    argparser.add_argument("--pH", type=float, default=7, help="pH for charge calculation")

    args = argparser.parse_args()



    main(args)