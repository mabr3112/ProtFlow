'''
Module to provide utilities revolving around BioPython
'''

# Imports
import os

# dependencies
import Bio
import Bio.PDB

def load_structure_from_pdbfile(path_to_pdb: str, all_models=False, model:int=0, quiet:bool=True, handle:str="pose") -> Bio.PDB.Structure:
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
    if not os.path.isfile(path_to_pdb): raise FileNotFoundError("PDB file {path_to_pdb} not found!")
    if not path_to_pdb.endswith(".pdb"): raise ValueError(f"File must be .pdb file. File: {path_to_pdb}")

    # load poses
    pdb_parser = Bio.PDB.PDBParser(QUIET=quiet)
    if all_models:
        return pdb_parser.get_structure(handle, path_to_pdb)
    else:
        return pdb_parser.get_structure(handle, path_to_pdb)[model]

def save_structure_to_pdbfile(pose: Bio.PDB.Structure.Structure, save_path: str) -> None:
    '''Stores Bio.PDB.Structure at <save_path>'''
    io = Bio.PDB.PDBIO()
    io.set_structure(pose)
    io.save(save_path)

######################## Bio.PDB.Structure.Structure functions ##########################################
def get_sequence_from_pose(pose: Bio.PDB.Structure.Structure, chain_sep:str=":") -> str:
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
