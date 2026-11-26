"""Heavy-atom permutations for standard protein residues, in Bio.PDB atom order."""
from itertools import product

# Each tuple of pairs is one coupled operation on one residue.
_RING = (("CD1", "CD2"), ("CE1", "CE2"))
_PAIRS = {
    "ASP": (("OD1", "OD2"),), "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),), "LEU": (("CD1", "CD2"),),
    "VAL": (("CG1", "CG2"),), "PHE": _RING, "TYR": _RING,
}
_HIS = {"ring": (("ND1", "CD2"), ("CE1", "NE2")), "nitrogens": (("ND1", "NE2"),)}
_AMIDES = {"ASN": (("OD1", "ND2"),), "GLN": (("OE1", "NE2"),)}


def protonated_carboxylate(residue):
    """A named acid proton distinguishes the two oxygen sites."""
    names = {"ASP": ("HD1", "HD2", "DD1", "DD2"), "GLU": ("HE1", "HE2", "DE1", "DE2")}
    return any(name in residue for name in names.get(residue.resname, ()))


def swap_groups(atoms, *, histidine=None, amide_flip=False, locked=()):
    """Return index/partner lists grouped by residue. Only realized conformers participate."""
    groups = {}
    incomplete = set()
    for i, atom in enumerate(atoms):
        res = atom.get_parent()
        if res in locked or protonated_carboxylate(res):
            continue
        if res.resname == "HIS" and histidine:
            pairs = _HIS[histidine]
        elif amide_flip and res.resname in _AMIDES:
            pairs = _AMIDES[res.resname]
        else:
            pairs = _PAIRS.get(res.resname, ())
        for a, b in pairs:
            if atom.name not in (a, b):
                continue
            other = res.child_dict.get(b if atom.name == a else a)
            label = atom.get_altloc()
            if other is not None and other.is_disordered() == 2:
                other = other.child_dict.get(label) if label.strip() else other.selected_child
            if other is None or (label.strip() and other.altloc.strip() and other.altloc != label):
                incomplete.add(res)
            else:
                groups.setdefault(res, []).append((i, other))
    return [v for res, v in groups.items() if res not in incomplete]


def atom_assignments(atoms, **options):
    """Yield literal first, then each complete residue-consistent alternative."""
    groups = swap_groups(atoms, **options)
    for flags in product((False, True), repeat=len(groups)):
        assigned = list(atoms)
        for use, group in zip(flags, groups):
            if use:
                for i, other in group:
                    assigned[i] = other
        yield assigned
