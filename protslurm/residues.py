'''protslurm internal module to handle residue_selection and everything related to residues.'''

class ResidueSelection:
    '''Class to represent selections of Residues.
    Selection of Residues is represented as a tuple with the hierarchy ((chain, residue_idx), ...)

    '''
    def __init__(self, selection: list, delim: str = ","):
        self.residues = parse_selection(selection, delim=delim)

    def __str__(self) -> str:
        return ", ".join([f"{chain}{str(resi)}" for chain, resi in self])

    def __iter__(self):
        return iter(self.residues)

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

def parse_selection(input_selection, delim: str = ",") -> tuple[tuple[str,int]]:
    '''Parses selction into ResidueSelection formatted selection.'''
    #TODO: This implementation is safe from bugs, but not very efficient.
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

#TODO @Adrian please write a contig parser for ResidueSelection construction: ResidueSelection(contig="A1-6,A8,A10-120,B1-9")
