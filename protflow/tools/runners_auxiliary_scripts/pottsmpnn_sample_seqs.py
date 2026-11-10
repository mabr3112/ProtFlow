"""
Code in this file was originally written by or adapted from:

    1) Dauparas et al. "Robust deep learning-based protein sequence design using ProteinMPNN"
    Science, 2022. doi: 10.1126/science.add2187 
        - Code: https://github.com/dauparas/ProteinMPNN

    2) Birnbaum and Keating "Beyond native sequence recovery: Improved modeling of the
    sequence-energy landscape of protein structures"
    bioRxiv, 2026. doi:10.64898/2026.01.14.699067
       - Code: https://github.com/KeatingLab/PottsMPNN

ProtFlow note
-------------
This internal copy uses exact sequence-key ownership during optimization so
PDB names that are prefixes of other PDB names cannot select each other's
sequences.
"""

from tqdm import tqdm
import os
import sys
import torch
import json
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
from itertools import accumulate
from omegaconf import OmegaConf

# PottsMPNN commands run from the configured PottsMPNN checkout.
sys.path.insert(0, os.getcwd())

from run_utils import get_etab, optimize_sequence, tied_optimize_sequence, string_to_int, process_configs, cat_neighbors_nodes, rewrite_pdb_sequences, chain_to_partition_map, inter_partition_contact_mask
from potts_mpnn_utils import PottsMPNN, nlcpl
from data_utils import tied_featurize, parse_PDB, parse_PDB_seq_only, loss_nll
import etab_utils as etab_utils

def str_split(string, tok):
    if not string:
        return []
    return string.split(tok)

def index_external_sequence_keys(sequence_keys, pdb_names):
    """Group external sequence keys under the most specific input PDB key."""
    unique_pdb_names = list(dict.fromkeys(pdb_names))
    sequence_keys_by_pdb = {pdb_name: [] for pdb_name in unique_pdb_names}
    pdb_name_set = set(unique_pdb_names)

    # Exact PDB names take precedence over the optional numeric sample suffix.
    for sequence_key in sequence_keys:
        if sequence_key in pdb_name_set:
            sequence_keys_by_pdb[sequence_key].append(sequence_key)
            continue
        pdb_name, separator, sample_key = sequence_key.rpartition("_")
        if separator and sample_key.isdigit() and pdb_name in pdb_name_set:
            sequence_keys_by_pdb[pdb_name].append(sequence_key)
    return sequence_keys_by_pdb


def get_decoding_order(decoding_orders, pdb_name, sample_key):
    """Return a stored decoding order for one PDB and optional sample key."""
    stored_orders = decoding_orders.get(pdb_name)
    if sample_key is None:
        return stored_orders if not isinstance(stored_orders, dict) else None
    return stored_orders.get(sample_key) if isinstance(stored_orders, dict) else None


def store_decoding_order(decoding_orders, pdb_name, sample_key, decoding_order):
    """Store a decoding order with consistent string sample keys."""
    if sample_key is None:
        decoding_orders[pdb_name] = decoding_order
        return

    stored_orders = decoding_orders.get(pdb_name)
    if not isinstance(stored_orders, dict):
        stored_orders = {}
        decoding_orders[pdb_name] = stored_orders
    stored_orders[sample_key] = decoding_order


def sample_seqs(args):
    
    # Load experiment configuration (OmegaConf file)
    cfg = OmegaConf.load(args.config)
    cfg.model.vocab = 22 if 'msa' in cfg.model.check_path else 21
    if cfg.inference.temperature == 0: cfg.inference.temperature = 1e-6
    if cfg.inference.optimization_temperature == 0: cfg.inference.optimization_temperature = 1e-6
    if cfg.inference.optimize_pdb or cfg.inference.optimize_fasta: cfg.inference.num_samples = 1
    if cfg.inference.optimization_mode == "none": cfg.inference.optimization_mode = ""

    print("Loading model checkpoint...")
    checkpoint = torch.load(cfg.model.check_path, map_location='cpu', weights_only=False) 

    model = PottsMPNN(
        ca_only=False, 
        num_letters=cfg.model.vocab, 
        vocab=cfg.model.vocab, 
        node_features=cfg.model.hidden_dim, 
        edge_features=cfg.model.hidden_dim, 
        hidden_dim=cfg.model.hidden_dim, 
        potts_dim=cfg.model.potts_dim, 
        num_encoder_layers=cfg.model.num_layers, 
        num_decoder_layers=cfg.model.num_layers, 
        k_neighbors=cfg.model.num_edges, 
        augment_eps=cfg.inference.noise
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model = model.to(cfg.dev)
    pad = (0, 2, 0, 2) # Pad for 'X' and '-' tokens

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    print("Model loaded successfully.")

    # Read list of PDBs to process and derive chain mapping
    with open(cfg.input_list, 'r') as f:
        pdb_lines = f.readlines()
    pdb_count = defaultdict(int)
    for pdb in pdb_lines:
        pdb_name = pdb.strip().split('|')[0]
        pdb_count[pdb_name] += 1
    pdb_list = []
    chain_dict = {}
    chain_suffixes = []
    for pdb in pdb_lines:
        pdb = pdb.strip()
        pdb_info = pdb.split('|')
        if pdb_count[pdb_info[0]] > 1:
            assert len(pdb_info) > 1 # If duplicate pdb entries in list, there must be other identifying info provided
            chain_suffixes.append('|' + '|'.join(pdb_info[1:]))
        else:
            chain_suffixes.append('')
        pdb_list.append(pdb_info[0]) # pdb name
        if len(pdb_info) > 1: # Get designed and fixed chain info
            assert len(pdb_info) == 3
            chain_dict[pdb_info[0] + chain_suffixes[-1]] = [str_split(pdb_info[1], ':'), str_split(pdb_info[2], ':')]
        else: # Set chain_dict to empty list, which means all chains are designed
            chain_dict[pdb_info[0] + chain_suffixes[-1]] = [[], []]
    pdb_names_with_chain_suffixes = [
        pdb + chain_suffix
        for pdb, chain_suffix in zip(pdb_list, chain_suffixes)
    ]
    if cfg.chain_dict_json is not None:
        with open(cfg.chain_dict_json, 'r') as f:
            chain_dict = json.load(f)

    # Load various configuration dictionaries
    fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict, tied_positions_dict, bias_by_res_dict, omit_AAs_np = process_configs(cfg)
    constant = torch.tensor(omit_AAs_np, device=cfg.dev)

    # Setup Alphabet and Bias
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX-'
    if cfg.model.vocab == 21:
        alphabet = alphabet[:-1]  # remove gap character

    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]
    constant_bias = torch.tensor(bias_AAs_np, device=cfg.dev)

    print(f"Prepared to process {len(pdb_list)} PDBs.")

    # Ensure output directories exist for sequences and metrics
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Filepaths for outputs
    filename = os.path.join(cfg.out_dir, cfg.out_name + '.fasta')
    decoding_order_filename = os.path.join(cfg.out_dir, cfg.out_name + '_decoding_order.json')
    av_loss_filename = os.path.join(cfg.out_dir, cfg.out_name + '_av_loss.csv')
    if cfg.inference.write_pdb:
        pdb_out_dir = os.path.join(cfg.out_dir, cfg.out_name + '_pdbs')

    # Optimization check logic
    skip_calc = False
    existing_sequence_keys_by_pdb = {}

    if cfg.inference.optimization_mode:
        optimized_filename = os.path.join(cfg.out_dir, cfg.out_name + f'_optimized_{cfg.inference.optimization_mode}.fasta')
        
        if cfg.inference.optimize_fasta:
            if not os.path.isfile(cfg.inference.optimize_fasta):
                raise FileNotFoundError(
                    f"Tried to optimize sequences in {cfg.inference.optimize_fasta}, "
                    "but the file does not exist."
                )
            print(
                f"Found existing sequences at {cfg.inference.optimize_fasta}. "
                "Loading for optimization..."
            )
            with open(cfg.inference.optimize_fasta, 'r') as f:
                seqs_raw = f.readlines()
            # Parse .fasta
            existing_seqs = {pdb.strip('>').strip(): seq.strip() for pdb, seq in zip(seqs_raw[::2], seqs_raw[1::2])}
            existing_sequence_keys_by_pdb = index_external_sequence_keys(
                existing_seqs,
                pdb_names_with_chain_suffixes,
            )
            print(f'Saving optimized sequences to filename {optimized_filename}.')
            skip_calc = True
        elif cfg.inference.optimize_pdb:
            print(f"Optimizing existing sequences in pdb files in {cfg.input_dir}. Loading for optimization...")
            existing_seqs = {}
            for pdb, chain_info in zip(pdb_list, chain_suffixes):
                wt_info = parse_PDB_seq_only(os.path.join(cfg.input_dir, pdb + '.pdb'), skip_gaps=cfg.inference.skip_gaps) # Parse .pdb files
                if chain_info:
                    _, hidden_chains, vis_chains = chain_info.split('|')
                    chain_order = hidden_chains.split(':') + vis_chains.split(':')
                    wt_seq = ""
                    for chain in chain_order:
                        if chain: wt_seq += wt_info[f'seq_chain_{chain}']
                    pdb_name_with_chain_suffix = pdb + chain_info
                    existing_seqs[pdb_name_with_chain_suffix] = wt_seq
                    existing_sequence_keys_by_pdb[pdb_name_with_chain_suffix] = [pdb_name_with_chain_suffix]
                else:
                    existing_seqs[pdb] = wt_info['seq']
                    existing_sequence_keys_by_pdb[pdb] = [pdb]
            print(f'Saving optimized sequences to filename {optimized_filename}.')
            skip_calc = True
        else:
            print(f'Saving sequences to filename {filename} and saving optimized sequences to filename {optimized_filename}.')

        # Set up binding energy optimization if requested
        if cfg.inference.binding_energy_optimization in ["both", "only"]:
            assert cfg.inference.binding_energy_json is not None, "Chain separation information required for binding energy optimization"
            with open(cfg.inference.binding_energy_json, 'r') as f:
                binding_energy_chains = json.load(f)
            for pdb, chain_info in zip(pdb_list, chain_suffixes):
                assert pdb + chain_info in binding_energy_chains, f"Chain separation information required for {pdb + chain_info} binding energy optimization"

    else:
        print(f'Saving to filename {filename}.')

    # Containers to accumulate outputs and metrics across PDBs
    out_seqs = {}
    opt_seqs = {}
    best_seqs = {}

    av_losses = {'pdb': [], 'seq_loss': [], 'nsr': [], 'potts_loss': []}
    if cfg.inference.optimize_fasta and os.path.exists(decoding_order_filename):
        with open(decoding_order_filename, 'r') as f:
            decoding_orders = json.load(f)
    else:
        decoding_orders = {}

    # Iterate over PDBs
    for i_pdb, pdb in tqdm(enumerate(pdb_list)):
        input_pdb = os.path.join(cfg.input_dir, pdb + '.pdb')
        pdb_with_chain_suffix = pdb + chain_suffixes[i_pdb]
        # Parse PDB
        pdb_data = parse_PDB(input_pdb, chain_dict[pdb_with_chain_suffix][0] + chain_dict[pdb_with_chain_suffix][1], skip_gaps=cfg.inference.skip_gaps)
        # Featurize
        pdb_chain_dict = {pdb: chain_dict[pdb_with_chain_suffix]}
        X, S_true, mask, _, chain_mask, chain_encoding_all, _, _, _, _, chain_M_pos, omit_AA_mask, residue_idx, _, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds, bias_by_res, tied_beta, chain_lens = tied_featurize(
            [pdb_data[0]], cfg.dev, pdb_chain_dict, fixed_positions_dict, omit_AA_dict, 
            tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False, vocab=cfg.model.vocab
        )
        chain_cuts = [0, *accumulate(chain_lens)]

        pssm_log_odds_mask = (pssm_log_odds > cfg.inference.pssm_threshold).float()
        
        # Set seed if required
        if cfg.inference.fix_decoding_order:
            torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset)
        
        # Run Encoder
        h_V, E_idx, h_E, etab = model.run_encoder(X, mask, residue_idx, chain_encoding_all)
        sampled_sequence_keys = []
        
        # Skip sampling if we are just optimizing existing sequences
        if not skip_calc:
            # etab for Potts energy calculations
            etab_functional = etab_utils.functionalize_etab(etab.clone(), E_idx)
            etab_functional = torch.nn.functional.pad(etab_functional, pad, "constant", 0) # Add padding to account for 'X' and '-' tokens
            sample_records = []
            sample_seq_loss = []
            sample_nsr = []
            sample_nlcpl = []
            
            # Sampling Loop
            for sidx in range(cfg.inference.num_samples):
                if cfg.inference.fix_decoding_order:
                    torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset + sidx)
                
                randn = torch.randn(chain_mask.shape, device=X.device)

                # Decoder
                if tied_positions_dict is None or not tied_pos_list_of_lists_list[0]:
                    output_dict, all_probs = model.decoder(
                        h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=mask, 
                        temperature=cfg.inference.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, 
                        pssm_multi=cfg.inference.pssm_multi, pssm_log_odds_flag=bool(cfg.inference.pssm_log_odds_flag), 
                        pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(cfg.inference.pssm_bias_flag), 
                        bias_by_res=bias_by_res
                    )
                else:
                    output_dict, all_probs = model.tied_decoder(
                        h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=mask, 
                        temperature=cfg.inference.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, 
                        chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, 
                        pssm_multi=cfg.inference.pssm_multi, pssm_log_odds_flag=bool(cfg.inference.pssm_log_odds_flag), 
                        pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(cfg.inference.pssm_bias_flag), 
                        tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res
                    )

                # Metrics
                log_probs = torch.log(all_probs)
                mask_for_loss = mask * chain_mask
                _, av_seq_loss, nsr = loss_nll(S_true, torch.nan_to_num(log_probs, nan=0.1, posinf=0.1, neginf=0.1), chain_mask)
                nsr = torch.sum((nsr * mask_for_loss).float()) / torch.sum(mask_for_loss)
                sample_seq_loss.append(av_seq_loss.cpu().item())
                sample_nsr.append(nsr.cpu().item())
                av_nlcpl_loss, _, _ = nlcpl(etab, E_idx, S_true, chain_mask)
                sample_nlcpl.append(av_nlcpl_loss.cpu().item())

                # Convert to String
                seq_str = "".join(etab_utils.ints_to_seq_torch(output_dict['S'][0]))

                # Potts Energy
                seq_tensor = output_dict['S'][0].unsqueeze(0).to(dtype=torch.int64, device=E_idx.device)
                total_energy, _, _ = etab_utils.calc_eners(etab_functional, E_idx, seq_tensor.unsqueeze(1), None)
                total_energy = total_energy.squeeze().cpu().item()
                
                sample_records.append({
                    'sample_idx': sidx, 'seq': seq_str, 'energy': total_energy, 
                    'decoding_order': output_dict['decoding_order']
                })

            # Sort and Store
            sample_records = sorted(sample_records, key=lambda x: x['energy'])
            for k, rec in enumerate(sample_records):
                sidx = rec['sample_idx']
                sample_suffix = f"_{sidx}" if cfg.inference.num_samples != 1 else ''
                sequence_key = pdb_with_chain_suffix + sample_suffix
                sample_key = str(sidx) if sample_suffix else None
                out_seqs[sequence_key] = ':'.join(rec['seq'][a:b] for a, b in zip(chain_cuts, chain_cuts[1:]))
                sampled_sequence_keys.append(sequence_key)

                store_decoding_order(
                    decoding_orders,
                    pdb_with_chain_suffix,
                    sample_key,
                    rec['decoding_order'].squeeze().cpu().numpy().tolist(),
                )

                av_losses['pdb'].append(sequence_key)
                av_losses['seq_loss'].append(sample_seq_loss[sidx])
                av_losses['nsr'].append(sample_nsr[sidx])
                av_losses['potts_loss'].append(sample_nlcpl[sidx])

                if k == 0: # Save best sequence and sample number
                    best_seqs[pdb_with_chain_suffix] = (out_seqs[sequence_key], sidx)

        # Optimization Step (Optional)
        if cfg.inference.optimization_mode:
            # Setup for binding energy optimization
            if cfg.inference.binding_energy_optimization in ["both", "only"]:
                partitions = binding_energy_chains[pdb_with_chain_suffix]
                # Ensure chains are ordered correctly in partitions
                chain_order_indices = {chain: i_chain for i_chain, chain in enumerate(pdb_data[0]['chain_order'])}
                index_order_chains = {i_chain: chain for i_chain, chain in enumerate(pdb_data[0]['chain_order'])}
                for i_partition, partition in enumerate(partitions):
                    partition_indices = sorted([chain_order_indices[chain] for chain in partition])
                    partitions[i_partition] = [index_order_chains[chain_index] for chain_index in partition_indices]
                    
                # Get energy tables for separated chains
                partition_etabs = {}
                for i_p, partition in enumerate(partitions):
                    partition_etabs[i_p] = get_etab(model, pdb_data, cfg, partition)

                # Define mapping of residues to partition and partition interface mask
                partition_index = chain_to_partition_map(chain_encoding_all, pdb_data[0]['chain_order'], partitions)
                inter_mask = inter_partition_contact_mask(X[:,:,1], partition_index, cfg.inference.binding_energy_cutoff)
            else:
                partition_etabs, partition_index, inter_mask = None, None, None
            
            # Re-encode for optimization context
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            
            # Optimize sequences associated with this PDB
            source_seqs = existing_seqs if skip_calc else out_seqs
            current_pdb_keys = (
                existing_sequence_keys_by_pdb.get(pdb_with_chain_suffix, [])
                if skip_calc
                else sampled_sequence_keys
            )
            
            for key in current_pdb_keys:
                seq_to_opt = source_seqs[key].replace(':', '')
                suffix_key = key[len(pdb_with_chain_suffix):] if len(key) > len(pdb_with_chain_suffix) else ''
                sample_key = suffix_key[1:] if suffix_key.startswith("_") else None
                decoding_order = get_decoding_order(
                    decoding_orders,
                    pdb_with_chain_suffix,
                    sample_key,
                )

                if decoding_order is None:
                    if cfg.inference.fix_decoding_order:
                        suffix_add = int(sample_key) if sample_key is not None else 0
                        torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset + suffix_add)
                    randn = torch.randn(chain_mask.shape, device=X.device)
                    decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(randn))).squeeze().cpu().numpy().tolist()
                    store_decoding_order(
                        decoding_orders,
                        pdb_with_chain_suffix,
                        sample_key,
                        decoding_order,
                    )
                if tied_positions_dict is None or not tied_pos_list_of_lists_list[0]:
                    opt_seq = optimize_sequence(
                        seq_to_opt, etab, E_idx, mask*chain_M_pos, chain_mask, cfg.inference.optimization_mode, 
                        etab_utils.seq_to_ints, cfg.inference.optimization_temperature, constant, constant_bias, 
                        bias_by_res, cfg.inference.pssm_bias_flag, pssm_coef, pssm_bias, cfg.inference.pssm_multi,
                        cfg.inference.pssm_log_odds_flag, pssm_log_odds_mask, omit_AA_mask, model, h_E, h_EXV_encoder, h_V, 
                        decoding_order=decoding_order, partition_etabs=partition_etabs, partition_index=partition_index,
                        inter_mask=inter_mask, binding_optimization=cfg.inference.binding_energy_optimization, vocab=cfg.model.vocab
                    )
                else:
                    opt_seq = tied_optimize_sequence(
                        seq_to_opt, etab, E_idx, mask*chain_M_pos, chain_mask, cfg.inference.optimization_mode, 
                        etab_utils.seq_to_ints, cfg.inference.optimization_temperature, constant, constant_bias, 
                        bias_by_res, cfg.inference.pssm_bias_flag, pssm_coef, pssm_bias, cfg.inference.pssm_multi,
                        cfg.inference.pssm_log_odds_flag, pssm_log_odds_mask, omit_AA_mask, model, h_E, h_EXV_encoder, h_V, 
                        decoding_order=decoding_order, partition_etabs=partition_etabs, partition_index=partition_index,
                        inter_mask=inter_mask, binding_optimization=cfg.inference.binding_energy_optimization, vocab=cfg.model.vocab,
                        tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, tied_epistasis=cfg.inference.tied_epistasis
                    )

                opt_seq = etab_utils.ints_to_seq_torch(opt_seq)
                opt_seq = ':'.join(opt_seq[a:b] for a, b in zip(chain_cuts, chain_cuts[1:]))
                opt_seqs[key] = opt_seq

                if cfg.inference.num_samples == 1 or (not skip_calc and int(sample_key) == best_seqs[pdb_with_chain_suffix][1]): # Overwrite best sequence if on appropriate sample
                    if pdb_with_chain_suffix in best_seqs:
                        suffix = best_seqs[pdb_with_chain_suffix][1]
                    else:
                        suffix = 0
                    best_seqs[pdb_with_chain_suffix] = (opt_seq, suffix)

    print("Processing complete.")

    # Write Optimized Sequences (if any)
    if cfg.inference.optimization_mode:
        print(f"Writing optimized sequences to {optimized_filename}...")
        with open(optimized_filename, 'w') as f:
            for pdb_name, seq in opt_seqs.items():
                f.write('>' + pdb_name + '\n' + seq + '\n')

    # Write Decoding Orders
    if cfg.inference.optimization_mode or cfg.inference.fix_decoding_order:
        if not os.path.exists(decoding_order_filename) or not cfg.inference.optimize_fasta:
            print(f"Writing decoding orders to {decoding_order_filename}...")
            with open(decoding_order_filename, 'w') as f:
                json.dump(decoding_orders, f)

    # Write Sampled Sequences and Metrics (only if sampling occured)
    if not skip_calc:
        print(f"Writing sampled sequences to {filename}...")
        with open(filename, 'w') as f:
            for pdb_name, seq in out_seqs.items():
                f.write('>' + pdb_name + '\n' + seq + '\n')
                
        print(f"Writing metrics to {av_loss_filename}...")
        av_losses_df = pd.DataFrame(av_losses)
        av_losses_df.to_csv(av_loss_filename, index=None)

    # Write new .pdb files (if requested)
    if cfg.inference.write_pdb:
        print(f"Writing new .pdb files to {pdb_out_dir}")
        rewrite_pdb_sequences(best_seqs, cfg.input_dir, pdb_out_dir)

print("All outputs saved.")

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Run sampling using configuration supplied as an OmegaConf YAML file
    sample_seqs(args)
