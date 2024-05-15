'''protslurm internal module to handle residue_selection and everything related to residues.'''
# imports
from collections import OrderedDict

class ResidueSelection:
    '''Class to represent selections of Residues.
    Selection of Residues is represented as a tuple with the hierarchy ((chain, residue_idx), ...)

    fast: parses the selection without any type checking. For when :selection: already has ResidueSelection Format.

    '''
    def __init__(self, selection: list = None, delim: str = ",", fast: bool = False):
        self.residues = parse_selection(selection, delim=delim, fast=fast)

    def __str__(self) -> str:
        return ", ".join([f"{chain}{str(resi)}" for chain, resi in self])

    def __iter__(self):
        return iter(self.residues)

    def __add__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(self.residues + (other - self).residues, fast=True)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ResidueSelection):
            return ResidueSelection(tuple(res for res in self.residues if res not in set(other.residues)), fast=True)
        return NotImplemented

    ####################################### INPUT ##############################################
    def from_selection(self, selection) -> "ResidueSelection":
        '''Construct ResidueSelection Class.'''
        return residue_selection(selection)

    ####################################### OUTPUT #############################################
    def to_string(self, delim: None = ",", ordering: str = None) -> str:
        '''Converts ResidueSelection to string.'''
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return delim.join([str(idx) + chain for chain, idx in self])
        if ordering.lower() == "pymol":
            return delim.join([chain + str(idx) for chain, idx in self])
        return delim.join([chain + str(idx) for chain, idx in self])

    def to_list(self, ordering: str = None) -> list[str]:
        '''Converts ResidueSelection to list'''
        ordering = ordering or ""
        if ordering.lower() == "rosetta":
            return [str(idx) + chain for chain, idx in self]
        if ordering.lower() == "pymol":
            return [chain + str(idx) for chain, idx in self]
        return [chain+str(idx) for chain, idx in self]

    def to_dict(self) -> dict:
        '''Converts a ResidueSelection to a dictionary. 
        Caution: Converting to a dictionary destroys the ordering of specific residues on the same chain in a motif!
        '''
        # collect list of chains and setup chains as dictionary keys
        chains = list(set([x[0] for x in self.residues]))
        out_d = {chain: [] for chain in chains}

        # aggregate all residues to the chains and return
        for (chain, res_id) in self.residues:
            out_d[chain].append(res_id)

        return out_d

def fast_parse_selection(input_selection: tuple[tuple[str, int]]) -> tuple[tuple[str, int]]:
    '''Fast selection parser for when :input_selection: is already in ResidueSelection.residues format.'''
    return input_selection

def parse_selection(input_selection, delim: str = ",", fast: bool = False) -> tuple[tuple[str,int]]:
    '''Parses selction into ResidueSelection formatted selection.'''
    #TODO: This implementation is safe from bugs, but not very efficient.
    if fast:
        return fast_parse_selection(input_selection)
    if isinstance(input_selection, str):
        return tuple(parse_residue(residue.strip()) for residue in input_selection.split(delim))
    if isinstance(input_selection, list) or isinstance(input_selection, tuple):
        if all(isinstance(residue, str) for residue in input_selection):
            return tuple(parse_residue(residue) for residue in input_selection)
        elif all(isinstance(residue, list) or isinstance(residue, tuple) for residue in input_selection):
            return tuple(parse_residue("".join([str(r) for r in residue])) for residue in input_selection)
    raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. Only str and list allowed.")

def parse_residue(residue_identifier: str) -> tuple[str,int]:
    '''parses singular residue identifier into a tuple (chain, residue_index).
    Currently only supports single letter chain identifiers!'''
    chain_first = False if residue_identifier[0].isdigit() else True

    # assemble residue tuple
    chain = residue_identifier[0] if chain_first else residue_identifier[-1]
    residue_index = residue_identifier[1:] if chain_first else residue_identifier[:-1]

    # Convert residue_index to int for accurate typing
    return (chain, int(residue_index))

def residue_selection(input_selection, delim: str = ",") -> ResidueSelection:
    '''Creates residue selection from selection of residues.'''
    return ResidueSelection(input_selection, delim=delim)

def from_dict(input_dict: dict) -> ResidueSelection:
    '''Creates ResidueSelection object from dictionary. The dictionary specifies a motif in this way: {chain: [residues], ...}'''
    return ResidueSelection([f"{chain}{resi}" for chain, res_l in input_dict.items() for resi in res_l])

def from_contig(input_contig: str) -> ResidueSelection:
    '''Creates ResidueSelection object from a contig.'''
    sel = []
    elements = [x.strip() for x in input_contig.split(",") if x]
    for element in elements:
        subsplit = element.split("-")
        if len(subsplit) > 1:
            sel += [element[0] + str(i) for i in range(int(subsplit[0][1:]), int(subsplit[-1])+1)]
        else:
            sel.append(element)
    return ResidueSelection(sel)

def reduce_to_unique(input_array: list|tuple) -> list|tuple:
    '''reduces input_array to it's unique elements while preserving order.'''
    return type(input_array)(OrderedDict.fromkeys(input_array))
