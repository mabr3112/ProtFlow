'''
Module to provide utilities revolving around BioPython
'''

# Imports
import copy
import os
from typing import Union
import pandas as pd



# dependencies
import Bio
import Bio.PDB
from Bio.PDB.Structure import Structure
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils.ProtParam import ProteinAnalysis



# customs
from protslurm.residues import ResidueSelection

def load_structure_from_pdbfile(path_to_pdb: str, all_models = False, model: int = 0, quiet: bool = True, handle: str = "pose") -> Bio.PDB.Structure:
    """
    Load a structure from a PDB file using BioPython's PDBParser.

    This function parses a PDB file and returns a structure object. It allows 
    the option to load all models from the PDB file or a specific model.

    Parameters:
    path_to_pdb (str): Path to the PDB file to be parsed.
    all_models (bool, optional): If True, all models from the PDB file are 
                                 returned. If False, only the specified model 
                                 is returned. Defaults to False.
    model (int, optional): The index of the model to return. Only used if 
                           all_models is False. Defaults to 0 (first model).
    quiet (bool, optional): If True, suppresses output from the PDBParser. 
                            Defaults to True.
    handle (str, optional): String handle that is passed to the PDBParser's get_structure() method.

    Returns:
    Bio.PDB.Structure: The parsed structure object from the PDB file. If 
                       all_models is True, returns a Structure containing all 
                       models. Otherwise, returns a single Model object at the 
                       specified index.

    Raises:
    FileNotFoundError: If the specified PDB file does not exist.
    ValueError: If the specified model index is out of range for the PDB file.

    Example:
    To load the first model from a PDB file:
    >>> structure = load_structure_from_pdbfile("example.pdb")

    To load all models from a PDB file:
    >>> all_structures = load_structure_from_pdbfile("example.pdb", all_models=True)
    """
    # sanity
    if not os.path.isfile(path_to_pdb):
        raise FileNotFoundError(f"PDB file {path_to_pdb} not found!")
    if not path_to_pdb.endswith(".pdb"):
        raise ValueError(f"File must be .pdb file. File: {path_to_pdb}")

    # load poses
    pdb_parser = Bio.PDB.PDBParser(QUIET=quiet)
    if all_models:
        return pdb_parser.get_structure(handle, path_to_pdb)
    else:
        return pdb_parser.get_structure(handle, path_to_pdb)[model]

def save_structure_to_pdbfile(pose: Structure, save_path: str) -> None:
    '''Stores Bio.PDB.Structure at <save_path>'''
    io = Bio.PDB.PDBIO()
    io.set_structure(pose)
    io.save(save_path)

def superimpose_on_motif(mobile: Structure, target: Structure, mobile_atoms: ResidueSelection = None, target_atoms: ResidueSelection = None, atom_list: list[str] = None) -> Structure:
    '''Superimposes :mobile: onto :target: based on provided :mobile_atoms: and :target_atoms: If no atoms are given, superimposition is based on Structure CA.'''
    # prep inputs
    atom_list = atom_list or ["N", "CA", "O"]

    # if no motif is specified, superimpose on protein backbones.
    if (mobile_atoms is None and target_atoms is None):
        mobile_atms = get_atoms(mobile, atoms=atom_list)
        target_atms = get_atoms(target, atoms=atom_list)

    # collect atoms of motif. If only one of the motifs is specified, use the same motif for both target and mobile
    else:
        # in case heavy-atom superimposition is desired, pass 'all' for atom_list
        if atom_list == "all":
            atom_list = None
        mobile_atms = get_atoms_of_motif(mobile, mobile_atoms or target_atoms, atoms=atom_list)
        target_atms = get_atoms_of_motif(target, target_atoms or mobile_atoms, atoms=atom_list)

    # superimpose and return
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(target_atms, mobile_atms)
    super_imposer.apply(mobile)
    return mobile

def superimpose(mobile: Structure, target: Structure, mobile_atoms: list = None, target_atoms: list = None):
    '''Superimposes :mobile: onto :target: provided lists of atoms.'''
    # superimpose on protein Backbones if no atoms are provided
    atom_list = ["N", "CA", "O"]
    if (mobile_atoms is None and target_atoms is None):
        mobile_atoms = get_atoms(mobile, atoms=atom_list)
        target_atoms = get_atoms(target, atoms=atom_list)

    # superimpose and return
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(target_atoms, mobile_atoms)
    super_imposer.apply(mobile)
    return mobile

def get_atoms(structure: Structure, atoms: list[str], chains: list[str] = None) -> list:
    '''
    Extract specified atoms from specified chains in a given structure.
    
    Parameters:
    - structure (Bio.PDB.Structure): The structure from which atoms are to be extracted.
    - atoms (list of str): A list of atom names to extract.
    - chains (list of str, optional): A list of chain identifiers from which atoms will be extracted.
      If None, atoms will be extracted from all chains in the structure.
    
    Returns:
    - list: A list of Bio.PDB.Atom objects corresponding to the specified atoms.
    '''
    # Gather all chains from the structure
    chains = [structure[chain] for chain in chains] if chains else [chain for chain in structure]

    # loop over chains and residues and gather all atoms.
    atms_list = []
    for chain in chains:
        # Only select amino acids in each chain:
        residues = [res for res in chain if res.id[0] == " "]
        for residue in residues:
            for atom in atoms:
                atms_list.append(residue[atom])

    return atms_list

def get_atoms_of_motif(pose: Structure, motif: ResidueSelection, atoms: list[str] = None, excluded_atoms: list[str] = None, exclude_hydrogens: bool = True) -> list:
    '''Selects atoms from a pose based on a provided motif.'''
    # setup params
    if motif is None:
        return None
    if excluded_atoms is None:
        excluded_atoms = ["H", "NE1", "OXT"] # exclude hydrogens and terminal N and O because they lead to crashes when calculating all-atom RMSD

    # empty list to collect atoms into:
    out_atoms = []
    for chain, res_id in motif:
        if atoms:
            res_atoms = [pose[chain][(" ", res_id, " ")][atom] for atom in atoms]
        else:
            res_atoms = pose[chain][(" ", res_id, " ")].get_atoms()

        # filter out forbidden atoms
        res_atoms = [atom for atom in res_atoms if atom.name not in excluded_atoms]
        if exclude_hydrogens:
            res_atoms = [atom for atom in res_atoms if atom.element != "H"]

        # add atoms into aggregation list:
        for atom in res_atoms:
            out_atoms.append(atom)
    return out_atoms

def add_chain(target: Structure, reference: Structure, copy_chain: str, overwrite: bool = True) -> Structure:
    '''Adds chain :copy_chain: from :reference: into :target:'''
    if overwrite:
        if copy_chain in [chain.id for chain in target.get_chains()]:
            target.detach_child(copy_chain)
    target.add(reference[copy_chain])

    return target

######################## Bio.PDB.Structure.Structure functions ##########################################
def get_sequence_from_pose(pose: Structure, chain_sep:str=":") -> str:
    '''
    Extracts the sequence of peptides from a protein structure.

    Parameters:
    - pose (Bio.PDB.Structure.Structure): A BioPython Protein Data Bank (PDB) structure object containing the protein's atomic coordinates.
    - chain_sep (str, optional): Separator used to join the sequences of individual peptides. Default is ":".

    Returns:
    - str: The concatenated sequence of peptides, separated by the specified separator.

    Description:
    This function takes a BioPython PDB structure object 'pose' and extracts the sequences of individual peptides within the structure using the PPBuilder from BioPython. It then joins these sequences into a single string, using the 'chain_sep' as a separator. The resulting string represents the concatenated sequence of peptides in the protein structure.

    Example:
    >>> structure = Bio.PDB.PDBParser().get_structure("example", "example.pdb")
    >>> sequence = get_sequence_from_pose(structure, "-")
    >>> print(sequence)
    'MSTHRRRPQEAAGRVNRLPGTPLARAKYFYPKPGERKVEQTPWFAWDVTAGNEYEDTIEFRLEAEGKVGEVVEREDPDNGRGNFARFSLGLYGSKTQYRLPFTVEEVFHDLESVTQKDGFWNCTAFRTVQRLPRTRVAAELNPRAKAAASAVFTFQSQDVDAVANAVEACFAGFYEVVGVFVSNAVDGSVAGAQNFSQFCVGFRGGPRMLRQNRAPATFASAGNHPAKVLAACGLRYAA...
    '''
    # setup PPBuilder:
    ppb = Bio.PDB.PPBuilder()

    # collect sequence
    return chain_sep.join([str(x.get_sequence()) for x in ppb.build_peptides(pose)])

def renumber_pdb_by_residue_mapping(pose_path: str, residue_mapping: dict, out_pdb_path: str = None, keep_chain: str = "") -> str:
    '''Renumbers PDB file by a resdidue mapping: {(chain, res_id): (chain, res_id), ...}. ordering is {old: new, ...}.
    Stores pdb either at out_pdb_path if provided, or otherwise overwrites pose_path'''
    # change numbering
    pose = load_structure_from_pdbfile(pose_path)
    pose = renumber_pose_by_residue_mapping(pose=pose, residue_mapping=residue_mapping, keep_chain=keep_chain)

    # save pose
    path_to_output_structure = out_pdb_path or pose_path
    save_structure_to_pdbfile(pose, path_to_output_structure)
    return path_to_output_structure

def renumber_pose_by_residue_mapping(pose: Bio.PDB.Structure.Structure, residue_mapping: dict, keep_chain:str="") -> Bio.PDB.Structure.Structure:
    '''Renumbers a Biopython Structure object based on a residue mapping. {(chain, res_id): (chain, res_id), ...}. ordering is {old: new, ...}'''
    # deepcopy pose and detach all residues from chains
    out_pose = copy.deepcopy(pose)
    ch = [chain.id for chain in out_pose.get_chains() if chain.id != keep_chain]

    # remove all residues from old chains:
    for chain in ch:
        residues = [res.id for res in out_pose[chain].get_residues()]
        for resi in residues:
            out_pose[chain].detach_child(resi)

    # collect residues with renumbered ids and chains into one list:
    for (old_chain, old_id), (new_chain, new_id) in residue_mapping.items():
        # remove old residue from original pose
        res = pose[old_chain][(" ", old_id, " ")]
        pose[old_chain].detach_child((" ", old_id, " "))
        res.detach_parent()

        # set new residue ID
        res.id = (" ", new_id, " ")

        # add to appropriate chain (residue mapping) in out_pose
        out_pose[new_chain].add(res)

    # remove chains from pose that are empty:
    chain_ids = [x.id for x in out_pose] # for some reason, iterating over chains in struct directly does not work here...
    for chain_id in chain_ids:
        if not out_pose[chain_id].__dict__["child_dict"]:
            out_pose.detach_child(chain_id)

    return out_pose


######################## Bio.Seq functions ##########################################

def load_sequence_from_fasta(fasta:str, return_multiple_entries:bool=True):
    '''
    imports a fasta file and returns a single record if it is a single entry fasta or <return_multiple_entries> is False, otherwise return a record iterator
    '''
    records = SeqIO.parse(fasta, "fasta")
    if len([i for i in records]) == 1 or return_multiple_entries == False:
        return next(records)
    else:
        return records


def determine_protparams(seq:Union[str, Bio.SeqRecord.SeqRecord, Bio.Seq.Seq], pH:float=7):
    '''
    calculates protein features based on sequence. Returns a dataframe. See Bio.SeqUtils.ProtParam for further information.
    Included are:
        -num_amino_acids                    total number of amino acids in the sequence
        -molecular_weight                   molecular weight of amino acids in sequence in Da
        -aromaticity                        relative frequency of PHE + TRP + TYR
        -GRAVY                              gravy according to Kyte and Doolittle
        -instability_index                  Calculate the instability index according to Guruprasad et al 1990.
                                            Any value above 40 means the protein is unstable (has a short half life).
        -isoelectric_point                  isoelectric point based on sequence
        -molar_extinction_coefficient_red   molar extinction coefficient assuming cysteines are reduced.
        -molar_extinction_coefficient_ox    molar extinction coefficient assuming cysteines are oxidized, forming CYS-CYS-bond.
        -flexibility                        flexibility according to Vihinen, 1994.
        -secondary_structure_fraction       fraction of helix, turn and sheet. returns a list of the fraction of amino acids which tend to
                                            be in helix, turn or sheet. amino acids in helix: V, I, Y, F, W, L. amino acids in turn: N, P, G, S.
                                            amino acids in sheet: E, M, A, L. returns a tuple of three floats (helix, turn, sheet).
        -charge_at_ph_<pH>                  charge of a protein at given pH. default = 7
    '''

    if isinstance(seq, Bio.SeqRecord.SeqRecord):
        seq = seq.seq
    elif isinstance(seq, Bio.Seq.Seq):
        seq = seq.data
    elif isinstance(seq, str):
        seq = seq
    else:
        raise TypeError(f"Input must be a sequence, not {type(seq)}!")

    protparams = ProteinAnalysis(seq)
    data = {
        "sequence": seq,
        "num_amino_acids": protparams.count_amino_acids(),
        "molecular_weight": protparams.molecular_weight(),
        "aromaticity": protparams.aromaticity(),
        "GRAVY": protparams.gravy(),
        "instability_index": protparams.instability_index(),
        "isoelectric_point": protparams.isoelectric_point(),
        "molar_extinction_coefficient": protparams.molar_extinction_coefficient(),
        "flexibility": protparams.flexibility(),
        "secondary_structure_fraction": protparams.secondary_structure_fraction(),
        f"charge_at_pH_{pH}": protparams.charge_at_pH(pH=pH)
    }

    return pd.DataFrame(data)
    
    

