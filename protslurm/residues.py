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

def parse_selection(input_selection, delim: str = ",") -> tuple[tuple[str,int]]:
    '''Parses selction into ResidueSelection formatted selection.'''
    if isinstance(input_selection, str):
        return tuple(parse_residue(residue.strip()) for residue in input_selection.split(delim))
    if isinstance(input_selection, list):
        return tuple(parse_residue(residue) for residue in input_selection)
    raise TypeError(f"Unsupported Input type for parameter 'input_selection' {type(input_selection)}. Only str and list allowed.")

def parse_residue(residue_identifier: str) -> tuple[str,int]:
    '''parses singular residue identifier into a tuple (chain, residue_index)'''
    chain_first = False if residue_identifier[0].isdigit() else True
    index_transition = None
    for i, char in enumerate(residue_identifier):
        # search for transition point between letter and number
        if char.isdigit() != residue_identifier[max(0, i-1)].isdigit():
            index_transition = i
            break

    # assemble residue tuple
    chain = residue_identifier[:index_transition] if chain_first else residue_identifier[index_transition:]
    residue_index = residue_identifier[index_transition:] if chain_first else residue_identifier[:index_transition]

    # Convert residue_index to int for accurate typing
    return (chain, int(residue_index))

def residue_selection(input_selection, delim: str = ",") -> ResidueSelection:
    '''Creates residue selection from selection of residues.'''
    return ResidueSelection(input_selection, delim=delim)
