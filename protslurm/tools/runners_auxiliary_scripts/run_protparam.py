import pandas as pd

from protslurm.utils.biopython_tools import determine_protparams

def main(args):

    in_df = pd.read_json(args.input_json)
    out_df = []
    for i, series in in_df.iterrows():
        params = determine_protparams(seq=series['seq'], pH=args.pH)
        params['poses_description'] = series['name']
        out_df.append(params)
    out_df = pd.concat(out_df)
    out_df.to_json(args.output_path)
    
if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # options
    argparser.add_argument("--input_json", type=str, help="")
    argparser.add_argument("--output_path", type=str, help="")
    argparser.add_argument("--pH", type=float, help=".json formatted file that contains a dictionary pointing to target and reference pdbs in the following way: {'target': 'reference'}")

    args = argparser.parse_args()



    main(args)