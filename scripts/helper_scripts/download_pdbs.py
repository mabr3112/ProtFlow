'''Script to download pdbs from RCSB when given file with list of pdb_ids.'''
# imports
import os
import json
import logging
import requests

def main(args):
    '''do stuff'''
    os.makedirs(args.output_dir, exist_ok=True)
    # setup logging
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for more details
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "log.txt")),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    logging.info(f"Running download_pdbs.py on {args.input_path}")

    # load list of pdb ids
    with open(args.input_path, 'r', encoding="UTF-8") as f:
        if args.input_path.endswith(".json"):
            pdb_list = json.load(f)

        elif args.input_path.endswith(".csv"):
            pdb_list = [pdb_id.strip() for pdb_id in f.read().split(",")]

        else:
            raise ValueError(f"File --input_path must be .json or .csv file and should contain list of pdb_ids. --input_path: {args.input_path}")

    # check correct format and files
    if not pdb_list:
        raise ValueError(f"No pdb_ids specified in input file: {args.input_path}")
    if not isinstance(pdb_list, list):
        raise ValueError(f"pdb_list not loaded as list. Check formatting of input file: {args.input_path}")

    # Download each PDB file
    failed_downloads = []
    logging.info(f"Starting to download {len(pdb_list)} files from RCSB in {args.format} format.")
    for pdb_id in pdb_list:
        # define file to download
        file_path = os.path.join(args.output_dir, f"{pdb_id}.{args.format}")

        # skip download, if file is already there:
        if os.path.isfile(file_path):
            continue

        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.{args.format}"
        try:
            pdb_response = requests.get(pdb_url, timeout=90)
            pdb_response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Error downloading {pdb_id}.{args.format}: {e}")
            failed_downloads.append(pdb_id)
            continue

        # write pdb_id.format file
        with open(file_path, "w", encoding="UTF-8") as f:
            f.write(pdb_response.text)

    # log failed downloads
    if failed_downloads:
        with open(os.path.join(args.output_dir, "failed_downloads.csv"), 'w', encoding="UTF-8") as f:
            f.write(",".join(failed_downloads))

    # log
    logging.info(f"Downloaded {len(pdb_list) - len(failed_downloads)} of {len(pdb_list)} pdb files from RCSB in {args.format} format.")
    logging.info("Done.")

if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, required=True, help="Path to file that holds [json serialized / or csv] list of pdb_ids.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to directory where pdb files should be stored.")
    argparser.add_argument("--format", type=str, default="cif", help="{cif, pdb, fa} File format of downloads.")
    arguments = argparser.parse_args()

    if arguments.format not in {"cif", "pdb", "fa"}:
        raise ValueError(f"Unsupported download format specified by --format: {arguments.format}. Allowed formats: {{cif, pdb, fa}}")

    main(arguments)
