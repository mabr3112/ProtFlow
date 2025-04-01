'''Script that downloads all hits for an advanced RCSB PDB query in batches. Usage see below.'''
# imports
import os
import sys
import json
import time
import logging
import requests

# dependencies
import pandas as pd

# custom
import protflow

def compile_pagination(n_rows: int, rows_per_page: int = 500) -> list[tuple[int, int]]:
    '''compiles pagination for query chaining.'''
    # Calculate how many full pages we have and leftover rows
    full_pages, last_page_rows = divmod(n_rows, rows_per_page)

    pages = [
        (start, rows_per_page)
        for start in range(0, full_pages * rows_per_page, rows_per_page)
    ]

    # Only add a last page if there's a remainder
    if last_page_rows > 0:
        pages.append((full_pages * rows_per_page, last_page_rows))

    return pages

def main(args):
    '''does stuff'''
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
    logging.info(f"Running batch_download_rcsb_query.py on {args.query_file}. Downloading files to {args.output_dir}")

    # read query from file:
    if not os.path.isfile(args.query_file):
        raise FileNotFoundError(args.query_file)

    ############ collect query hits ###################
    logging.info(f"Loading RCSB advanced query from file {args.query_file}")
    with open(args.query_file, 'r', encoding="UTF-8") as f:
        query_dict = json.load(f)

    # RCSB Search API endpoint
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    ############################# INQUIRY QUERY ###########################
    # Send the POST request with the JSON payload
    try:
        response = requests.post(url, json=query_dict, timeout=900)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"RCSB query failed: {e}") from e

    # process result
    if response.status_code == 200: # successful request
        results = response.json()
    else: # request has hit an error
        raise RuntimeError(response.status_code, response.text)

    # if no hits, finish script
    if not results:
        logging.info(f"Query yielded no hits. Query file: {args.query_file}")
        sys.exit(0)

    ########################### COLLECTION QUERY ###############################
    # get results with proper pagination
    num_results = results["total_count"]
    pages = compile_pagination(n_rows=num_results, rows_per_page=500)
    logging.info(f"Retrieving {num_results} pdb_ids from query in {len(pages)} batches with 500 pdb_ids per batch.")

    pdb_ids = []
    for i, page in enumerate(pages):
        query_dict["request_options"]["paginate"]["start"] = page[0]
        query_dict["request_options"]["paginate"]["rows"] = page[1]

        # Send the POST request with the JSON payload
        try:
            response = requests.post(url, json=query_dict, timeout=900)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"RCSB query failed: {e}") from e

        # process result
        if response.status_code == 200: # successful request
            results = response.json()
        else: # request has hit an error
            raise RuntimeError(response.status_code, response.text)

        # aggregate
        pdb_ids.extend([result["identifier"] for result in results.get("result_set", [])])

        # let's be polite
        time.sleep(0.1)

    ############## download pdbs into organized folder structure ################
    # count number of hits and prepare download
    logging.info(f"Found {len(pdb_ids)} entries.")

    if args.pdb_ids_only:
        outf = os.path.join(args.output_dir, "pdb_id_list.json")
        logging.info(f"--pdb_ids_only specified. Storing list of pdb_ids as .json formatted file at {outf}")
        with open(outf, 'w', encoding="UTF-8") as f:
            json.dump(pdb_ids, f)
        logging.info("Done.")
        sys.exit(0)

    # split pdb_ids into batches
    pdb_batches = protflow.jobstarters.split_list(pdb_ids, args.batch_size)

    # create output directory
    batch_input_dir = os.path.join(args.output_dir, "batch_input_files")
    os.makedirs(batch_input_dir, exist_ok=True)

    # store index of which file is where in .csv format for accessibility (relative and absolute)
    cmds = []
    index_dict = {}
    executable = f"python {protflow.config.PROTFLOW_DIR}/scripts/helper_scripts/download_pdbs.py"
    for i, pdb_batch in enumerate(pdb_batches):
        # write pdb_batches to file
        batch_file = os.path.join(batch_input_dir, f"pdb_ids_{i+1}.csv")
        with open(batch_file, 'w', encoding="UTF-8") as f:
            f.write(",".join(pdb_batch))

        # setup batch output dir:
        batch_fn = f"downloads/batch_{str(i+1).zfill(4)}"
        batch_download_dir = os.path.join(args.output_dir, batch_fn)

        # aggregate download command to cmd-list
        cmds.append(f'{executable} --input_path "{batch_file}" --output_dir "{batch_download_dir}" --format {args.format}')

        # add batch_filename -> pdb_ids mapping into index
        index_dict[batch_fn] = pdb_batch

    # setup jobstarter
    if args.jobstarter == "sbatch":
        jst = protflow.jobstarters.SbatchArrayJobstarter(max_cores=args.num_workers)
    elif args.jobstarter == "local":
        jst = protflow.jobstarters.LocalJobStarter(max_cores=args.num_workers)
    else:
        raise ValueError(f"Unsupported options for --jobstarter: {args.jobstarter}. Allowed options: {{sbatch, local}}")

    # run cmds
    jst.start(
        cmds=cmds,
        jobname="rcsb_query_download",
        wait=True,
        output_path=args.output_dir
    )

    ############### store pdb locations in .csv file for later easier retrieval #################
    # aggregate locations into DataFrame
    inverted_index = {pdb_id: batch_dir for batch_dir, pdb_ids in index_dict.items() for pdb_id in pdb_ids} # standard dict inversion
    index_dict_full = {
        "pdb_id": list(inverted_index.keys()),
        "relative_path": list(inverted_index.values()),
        "absolute_path": [os.path.join(args.output_dir, relpath) for relpath in list(inverted_index.values())]
    }

    # store DataFrame in output_dir
    index_fn = os.path.join(args.output_dir, "download_index.df.csv")
    logging.info(f"Storing index of which pdb-file is where at {index_fn}")
    index_df = pd.DataFrame(index_dict_full)
    index_df = index_df.set_index("pdb_id", drop=True)
    index_df.to_csv(index_fn)

    # finish script
    logging.info("Done.")

if __name__ == "__main__":
    import argparse

    # setup args
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--query_file", type=str, required=True, help="Path to the file that contains the query (copied from RCSB advanced query page)")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to root directory of where the files should be stored.")
    argparser.add_argument("--batch_size", type=int, default=1000, help="Size of the batches in which pdb files should be grouped.")
    argparser.add_argument("--format", type=str, default="cif", help="{cif, pdb, fa} File format of downloads.")
    argparser.add_argument("--jobstarter", type=str, default="sbatch", help="{sbatch, local} Specify which jobstarter class to use for batch downloads.")
    argparser.add_argument("--num_workers", type=int, default=32, help="Number of parallel processes to start for download.")
    argparser.add_argument("--pdb_ids_only", action="store_true", help="Set this flag, if you would only like to create a list of pdb_ids that match for the query. Ideal for testing.")
    arguments = argparser.parse_args()

    # check arguments (either input_json or input_pdb + reference_pdb)
    if arguments.format not in {"cif", "pdb", "fa"}:
        raise ValueError(f"Unsupported download format specified by --format: {arguments.format}. Allowed formats: {{cif, pdb, fa}}")
    if arguments.jobstarter not in {"sbatch", "local"}:
        raise ValueError(f"Unsupported options for --jobstarter: {arguments.jobstarter}. Allowed options: {{sbatch, local}}")

    # run
    main(arguments)
