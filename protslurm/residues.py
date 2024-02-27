'''protslurm internal module to handle residue_selection and everything related to residues.'''

class ResidueSelection:
    '''Class to represent selections of Residues.'''
    def __init__(self, pymol:str=None, rosetta:str=None):
        if pymol:
            self.selection = self.from_pymol(pymol)
        if rosetta:
            self.selection = self.from_rosetta(rosetta)

    ####################################### INPUT ##############################################
    def from_pymol(self, input) -> tuple:
        '''Method to construct a residue selection from a list of pymol-style residue indeces <chain><number>: [A5, A6, ...].'''
        raise NotImplementedError

    def from_rosetta(self, input) -> tuple:
        '''Method to construct a residue selection from a list of rosetta-style residue indeces <number><chain>: [5A, 6A, ...]'''
        raise NotImplementedError

    ####################################### OUTPUT #############################################

    def to_pymol(self, format:str) -> list:
        '''Prints Selection as a list in PyMol Format.'''
        raise NotImplementedError
    
    def to_rosetta(self, format:str) -> list:
        '''Returns Selection in Rosetta format as a type specified by 'format'. '''
        raise NotImplementedError