from frame2seq import Frame2seqRunner
import json
import os

def main(args):

    # import input json
    with open(args.input_json, "r") as jf:
        input_dict = json.load(jf)


    # change dir, frame2seq generates output in dir where it was initialized
    os.chdir(args.output_dir) 

    # initialize frame2seq
    runner = Frame2seqRunner()

    if args.method == "score":
        for pose, pose_opts in input_dict.items():
            runner.score(pdb_file=pose, save_indiv_neg_pll=True, **pose_opts)


    elif args.method == "design":
        for pose, pose_opts in input_dict.items():
            runner.design(pdb_file=pose, save_indiv_seqs=True, save_indiv_neg_pll=True, **pose_opts)

    else:
        raise KeyError(f"<method> must be either 'score' or 'design', not {args.method}!")


if __name__ == "__main__":

    import argparse

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--attnpacker_dir", type=str, help="Path to AttnPacker directory")
    argparser.add_argument("--input_json", type=str, default=".", help="Path to folder containing input pdb files")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to output folder")
    argparser.add_argument("--method", type=str, required=True, help="Must be either 'score' or 'design'.")

    args = argparser.parse_args()


    main(args)