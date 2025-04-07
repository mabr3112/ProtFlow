'''Script to predict amidases'''
# imports
import os
import time
import logging
import argparse
from glob import glob

# dependencies
import pandas as pd
import protflow
from protflow.tools.alphafold3 import AlphaFold3
from protflow.runners import Runner
from protflow.poses import Poses

def predict_per_day(model: Runner, files: list, predictions_per_day: int, output_dir: str, smiles: str, samples: int, options: str = None) -> None:
    """Predicts """
    # split list of fasta files into input batches
    split_fl = protflow.jobstarters.split_list(files, element_length=predictions_per_day)
    logging.info(f"Split {len(files)} input files into {len(split_fl)} batches of size {predictions_per_day}.")

    # setup scores aggregation list:
    predictions_dir = os.path.join(output_dir, "predictions")
    scores_list = []
    for i, fl in enumerate(split_fl, start=1):
        start_time = time.time()
        logging.info(f"Starting prediction of batch {i}.")

        # read in batch
        batch = protflow.poses.Poses(
            poses=fl,
            work_dir=os.path.join(output_dir, f"batch_{i}")
        )

        # setup options
        standard_options = "--flash_attention_implementation xla --cuda_compute_7x 1"
        standard_options = standard_options + " " + options if options else standard_options

        # predict batch
        model.run(
            poses = batch,
            prefix = "af3",
            additional_entities = {"ligand": {"id": "Z", "smiles": smiles}},
            nstruct = samples,
            options = standard_options,
            single_sequence_mode = False
        )

        # save scores and poses
        scores_list.append(batch.df)
        batch.save_scores()
        batch.save_poses(predictions_dir)
        logging.info(f"Prediction of batch {i} finished. Structures can be inspected at {predictions_dir}")

        # only 1 batch per day max!
        while time.time() - start_time < 24*3600:
            time.sleep(60)

    # collect and combine scores
    logging.info("All predictions finished, collecting and combining scores of all batches.")
    all_scores = pd.concat(scores_list, axis=0, ignore_index=True).reset_index(drop=True)
    all_scores.to_json(os.path.join(output_dir, "prediction_scores.df.json"))
    return None

def predict(model: Runner, files: list, output_dir: str, smiles: str, samples: int, options: str = None) -> None:
    '''Predicts protein-ligand interactions all at once.'''
    # instantiate poses
    proteins = protflow.poses.Poses(
        poses = files,
        work_dir = output_dir
    )

    # setup options
    standard_options = "--flash_attention_implementation xla --cuda_compute_7x 1"
    standard_options = standard_options + " " + options if options else standard_options

    # run
    model.run(
        poses = proteins,
        prefix = "af3",
        additional_entities={"ligand": {"id": "Z", "smiles": smiles}},
        nstruct = samples,
        options = standard_options,
        single_sequence_mode = False
    )

    return None

def main(args):
    '''do stuff'''
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more details
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("log.txt"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info(f"Running predict_protein_ligand.py on {args.input_dir} with ligand from {args.ligand_file}")

    # setup jobstarters
    if args.jobstarter == "sbatch":
        jst = protflow.jobstarters.SbatchArrayJobstarter(max_cores=args.num_workers, gpus=1)
    elif args.jobstarter == "local":
        jst = protflow.jobstarters.LocalJobStarter(max_cores=args.num_workers)
    else:
        raise ValueError(f"Unsupported options for --jobstarter: {args.jobstarter}. Allowed options: {{sbatch, local}}")

    # setup af3 runner
    af3 = AlphaFold3(jobstarter=jst)

    # load ligand
    with open(args.ligand_file, 'r', encoding="UTF-8") as f:
        smiles_raw = f.read()

    # load fasta files from input directory.
    pl = glob(os.path.join(args.input_dir, "*.fa"))
    if not pl:
        raise FileNotFoundError(f"No input files found at specified directory --input_dir {args.input_dir}")

    if args.predictions_per_day > 0:
        predict_per_day(
            model = af3,
            files = pl,
            predictions_per_day = args.perdictions_per_day,
            output_dir = args.output_dir,
            smiles = smiles_raw,
            samples = args.samples,
            options = args.prediction_options
        )
    else:
        predict(
            model = af3,
            files = pl,
            output_dir = args.output_dir,
            smiles = smiles_raw,
            samples = args.samples,
            options = args.prediction_options
        )
    logging.info("Done.")

if __name__ == "__main__":
    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required options
    argparser.add_argument("--input_dir", required=True, help="Directory that contains .fa files that should be predicted")
    argparser.add_argument("--ligand_file", required=True, help="File containing SMILES string for ligand to be co-folded with every protein.")
    argparser.add_argument("--output_dir", required=True, help="Path to directory where predicted structures shall be stored")
    argparser.add_argument("--jobstarter", type=str, default="sbatch", help="{sbatch, local} Specify which jobstarter class to use for batch downloads.")

    argparser.add_argument("--samples", type=int, default=5, help="Number of samples that should be predicted for each input pose using Af3.")
    argparser.add_argument("--prediction_options", type=str, help="Additional options you would like to pass to the prediction network.")
    argparser.add_argument("--predictions_per_day", type=int, default=0, help="Number of predictions you would like to run per day. Limited to 50 by default to be polite to the MMSeqs2 server.")
    argparser.add_argument("--num_workers", type=int, default=10, help="Number of processes to run in parallel")

    arguments = argparser.parse_args()

    main(arguments)
