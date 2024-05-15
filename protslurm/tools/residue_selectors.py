'''Module to select Residues in a Poses class and add the resulting motif into Poses.df'''
# imports

# dependencies

# customs
from protslurm.poses import Poses
from protslurm.residues import ResidueSelection
from protslurm.utils.biopython_tools import load_structure_from_pdbfile

class ResidueSelector:
    '''Abstract class for ResidueSelectors. 
    All ResidueSelector classes must implement a select() method that selects residues of Poses.
    '''
    def __init__(self, poses: Poses = None):
        self.set_poses(poses)

    def set_poses(self, poses: Poses = None) -> None:
        '''Sets poses for ResidueSelector class.'''
        if not isinstance(poses, Poses):
            raise TypeError(f"Parameter :poses: must be of type Poses. type(poses) = {type(poses)}")
        self.poses = poses

    def select(self, prefix: str) -> None:
        '''
        Main function that selects residues in poses.
        Prefix is the name of the column that will be added to Poses.df
        Poses.df[prefix] holds the selected Residues as a ResidueSelection object.
        '''
        raise NotImplementedError

    def select_single(self, *args) -> ResidueSelection:
        '''
        Function that selects Residues for a single pose.
        Optimally, select_single is executed within the .select() method.
        Returns a ResidueSelection that contains the selected residues of the pose.
        '''
        raise NotImplementedError

class ChainSelector(ResidueSelector):
    '''Selects all residues of a given chain in Poses.'''
    def __init__(self, poses: Poses = None, chain: list = None, chains: list = None):
        if chain and chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {chain}, chains: {chains}")
        super().__init__(poses = poses)
        self.set_chains(chains)
        self.set_chain(chain)

    def set_chains(self, chains: list[str] = None) -> None:
        '''Sets chains for the .select() method.
        :chains: can also be given as a parameter to the .select() method directly. 
        This would overwrite all parameters given to the base class.
        '''
        if not chains:
            self.chains = None
        if not isinstance(chains, list):
            raise ValueError(f"Parameter :chains: must be a list containing the chains to select as single characters, e.g. chains=['A', 'C']")
        if not all(isinstance(chain, str) or len(chain) > 1 for chain in chains):
            raise ValueError(f"Parameter :chains: must be a list containing the chains to select as single characters, e.g. chains=['A', 'C']")
        self.chains = chains
        self.chain = None

    def set_chain(self, chain: str = None) -> None:
        '''Sets chain for the .select() method.
        :chain: can also be given as a parameter to the .select() method directly. 
        This would overwrite all parameters given to the base class.'''
        if not chain:
            self.chain = None
        if not isinstance(chain, str) or len(chain) > 1:
            raise ValueError(f"Parameter :chain: must be a string of a single character denoting the chain that should be selected. e.g. chain='B' ")
        self.chain = chain
        self.chains = None

    def select(self, prefix: str, poses: Poses = None, chain: str = None, chains: list[str] = None) -> None:
        '''Selects all residues of a given chain for all poses in a Poses object.
        Selected residues are added as ResidueSelection objects under the column :prefix: to Poses.df
        '''
        poses.check_prefix(prefix)

        # prep inputs
        chains = self.prep_chain_input(chain, chains)
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")

        # select Residues
        poses.df[prefix] = [self.select_single(pose_path=pose, chains=chains) for pose in poses.poses_list()]

    def select_single(self, pose_path: str, chains: list[str]) -> ResidueSelection: # pylint: disable=W0221
        '''Selects Residues of a given chain of poses and returns them as ResidueSelection object.'''
        pose = load_structure_from_pdbfile(pose_path)

        # check if chain is in chains:
        pose_chains = [chain.id for chain in pose.get_chains()]
        if not all(chain in pose_chains for chain in chains):
            raise KeyError(f"Some of your specified chains {chains} are not present in the pose. chains of pose: {pose_chains}")

        # for all chain in chains collect resis
        residues = []
        for chain in chains:
            resis = pose[chain].get_residues()
            residues += resis

        # convert selected Biopython residues to ResidueSelection object and return
        return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in residues])

    def prep_chain_input(self, chain: str = None, chains: list[str] = None) -> list[str]:
        '''Prepares chain input for chain_selection.
        This method makes sure to prioritize options set in method parameters over opteions in class attributes.
        This means that ChainSelector.select(chain="A") has higher priority than ChainSelector(chain="C") (attribute set to class).
        '''
        # error handling
        if chain and chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {chain}, chains: {chains}")
        if self.chain and self.chains:
            raise ValueError(f"Either chain, or chains can be set, but not both. chain: {self.chain}, chains: {self.chains}")
        if all(not param for param in [chain, chains, self.chain, self.chains]):
            raise ValueError(f"Set one of parameters :chain: or :chains: to select a chain with ChainSelector!")

        # handle priorities (method parameters over class parameters)
        class_chains = self.chains if self.chains else [self.chain]
        method_chains = chains if chains else [chain]
        return method_chains or class_chains

class TrueSelector(ResidueSelector):
    '''ResidueSelector that selects all residues of a pose.'''
    def __init__(self, poses: Poses = None):
        super().__init__(poses = poses)

    def select(self, prefix: str, poses: Poses = None):
        '''
        Selects all residues of a given pose for all poses in a Poses object.
        Selected residues are added as ResidueSelection objects under the column :prefix: to Poses.df.
        '''
        # prep inputs and run
        if not (poses := poses or self.poses):
            raise ValueError(f"You must set poses for your .select() method. Either with :poses: parameter of .select() or the ResidueSelector.set_poses() method to the class.")
        poses.check_prefix(prefix)
        poses[prefix] = [self.select_single(pose) for pose in poses.poses_list()]

    def select_single(self, pose_path: str) -> ResidueSelection: # pylint: disable=W0221
        '''Selects all residues in a pose and returns them as ResidueSelection object.'''
        pose = load_structure_from_pdbfile(pose_path)
        return ResidueSelection([residue.parent.id + str(residue.id[1]) for residue in pose.get_residues()])

class NotSelector(ResidueSelector):
    '''ResidueSelector that selects all residues except the ones specified by a residueselection.'''
    def __init__(self, poses: Poses = None, residue_selection: ResidueSelection|str = None, contig: str = None):
        super().__init__(poses)
        if residue_selection and contig:
            raise ValueError(f"NotSelector Class cannot be initialized with both parameters :contig: or :residue_selection: set.\n Either choose a residue_selection, or give the residue selectio as a contig, but not both.")
        self.set_residue_selection(residue_selection)
        self.set_contig(contig)

    def set_residue_selection(self, residue_selection: ResidueSelection = None) -> None:
        '''Sets residue_selection attribute for NotSelector class.'''
        if not residue_selection:
            residue_selection = None
        if isinstance(residue_selection, ResidueSelection) or isinstance(residue_selection, str):
            self.residue_selection = residue_selection
            self.contig = None

    def set_contig(self, contig: str) -> None:
        '''Sets contig attribute for NotSelector class.'''
        if not isinstance(contig, str):
            raise ValueError(f"Contig must be of type str. E.g.: contig='A1-7,A25-109,B45-50,C1,C3,C5")
        self.contig = contig
        self.residue_selection = None

    def select(self, prefix: str, poses: Poses = None, residue_selection: ResidueSelection|str = None, contig: str = None) -> None:
        '''Selects all residues except the ones specified in :residue_selection: or by :contig:
        Parameter :residue_selection: can be either a ResidueSelection object or a string pointing to a column in the poses.df that contains ResidueSelection objects.
        '''
        return None

    def select_single(self, pose_path: str, residue_selection: ResidueSelection) -> ResidueSelection:
        '''Selects all residues except the ones specified in :residue_selection: or by :contig:
        Parameter :residue_selection: can be either a ResidueSelection object or a string pointing to a column in the poses.df that contains ResidueSelection objects.
        '''
        
