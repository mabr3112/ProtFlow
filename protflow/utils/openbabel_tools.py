# TODO: write doc strings!!

"""
This module provides a collection of utilities for working with BioPython, specifically designed to facilitate the analysis and manipulation of protein structures and sequences. The functionalities included in this module allow users to load, save, and superimpose protein structures from PDB files, as well as extract and analyze sequences from these structures.

Overview
--------
The module encompasses a range of tools to handle various tasks related to protein structural data. Users can load structures from PDB files, save structures back to PDB files, and perform structural superimpositions based on specific atoms or motifs. Additionally, it provides methods to extract sequences from protein structures, renumber residues, and add chains to structures. For sequence analysis, it includes functionalities to load sequences from FASTA files and compute various protein properties using the `Bio.SeqUtils.ProtParam` class.

Examples
--------
Here are some examples of how to use the functions provided in this module:

1. Loading a structure from a PDB file:

    .. code-block:: python

        from biopython_tools import load_structure_from_pdbfile
        structure = load_structure_from_pdbfile("example.pdb")

2. Saving a structure to a PDB file:

    .. code-block:: python

        from biopython_tools import save_structure_to_pdbfile
        save_structure_to_pdbfile(structure, "output.pdb")

3. Superimposing one structure onto another:

    .. code-block:: python

        from biopython_tools import superimpose
        superimposed_structure = superimpose(mobile_structure, target_structure)

4. Extracting sequence from a protein structure:

    .. code-block:: python

        from biopython_tools import get_sequence_from_pose
        sequence = get_sequence_from_pose(structure)

5. Loading a sequence from a FASTA file:

    .. code-block:: python

        from biopython_tools import load_sequence_from_fasta
        sequence_record = load_sequence_from_fasta("example.fasta")

6. Calculating protein parameters from a sequence:

    .. code-block:: python

        from biopython_tools import determine_protparams
        parameters = determine_protparams(sequence_record.seq)

7. Extracting a ligand from a co-crystal structure:

    .. code-block:: python

        from openbabel_tools import split_complex
        sdf_path = split_complex("complex.pdb", "ligand.sdf", "out.pdb", name="COC")

These examples illustrate the primary capabilities of the module, showcasing how it can be utilized to streamline the process of working with protein structures and sequences in BioPython.

Authors
-------
Markus Braun, Adrian Tripp
"""
# Imports
import os
# dependencies

from openbabel import openbabel, pybel

# customs
def openbabel_fileconverter(input_file: str, output_format:str, output_file:str=None, input_format:str=None) -> str:
    '''converts files.'''

    file, ext = os.path.splitext(input_file)
    if not input_format:
        input_format = ext[1:]
    if not output_file:
        output_file = file + f".{output_format}"

    # Create an Open Babel molecule object
    mol = openbabel.OBMol()

    # Read the PDB file
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats(input_format,output_format)
    obConversion.ReadFile(mol, input_file)

    # Convert the molecule to the desired output format
    obConversion.SetOutFormat(output_format)
    obConversion.WriteFile(mol, output_file)
    return output_file


def split_complex(path: str, work_dir: str, name: str = "ligand") -> None:
    """
    Split a structure file into a HETATM-based SDF and an ATOM-only PDB.

    CIF inputs are converted to PDB first. Both outputs are written to ``work_dir``
    using the input file stem. The HETATM part gets a ``_ligand`` suffix. HETATOM is unsafe since boltz for example does put ligand on chain B and not mark them as hetatoms, maybe convert to cif and extract there as ligand?
    Water (HOH) is excluded from the SDF output.

    Parameters
    ----------
    path : str
        Path to the input file (``.pdb`` or ``.cif``).
    work_dir : str
        Directory where output files are written.
    name : str, optional
        Molecule title written into the SDF record. Defaults to ``"ligand"``.
        
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    ext = path.rsplit(".", 1)[-1].lower()

    if ext == "cif":
        path = openbabel_fileconverter(path, output_format="pdb", output_file=os.path.join(work_dir, f"{stem}.pdb"))
        print(f"Converted CIF to PDB: {path}")

    out_pdb = os.path.join(work_dir, f"{stem}.pdb")
    out_sdf = os.path.join(work_dir, f"{stem}_ligand.sdf")

    def _is_hetatm(a) -> bool:
        res = a.OBAtom.GetResidue()
        return res is not None and res.IsHetAtom(a.OBAtom) and res.GetName().strip() != "HOH"

    mol_lig = next(pybel.readfile("pdb", path))
    for atom in [a for a in mol_lig.atoms if not _is_hetatm(a)]:
        mol_lig.OBMol.DeleteAtom(atom.OBAtom)

    mol_prot = next(pybel.readfile("pdb", path))
    for atom in [a for a in mol_prot.atoms if _is_hetatm(a)]:
        mol_prot.OBMol.DeleteAtom(atom.OBAtom)

    mol_lig.title = name
    mol_lig.write("sdf", out_sdf, overwrite=True)
    mol_prot.write("pdb", out_pdb, overwrite=True)

    return None
