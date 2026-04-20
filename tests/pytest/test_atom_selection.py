from protflow.residues import AtomSelection


def _pdb_line(record, serial, atom_name, resname, chain, resseq, x, y, z, element):
    return (
        f"{record:<6}{serial:>5} {atom_name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}           {element:>2}"
    )


def _write_test_pdb(tmp_path):
    lines = [
        _pdb_line("ATOM", 1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        _pdb_line("ATOM", 2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 3, "C", "ALA", "A", 1, 2.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 4, "O", "ALA", "A", 1, 3.0, 0.0, 0.0, "O"),
        _pdb_line("ATOM", 5, "CB", "ALA", "A", 1, 1.0, 1.0, 0.0, "C"),
        _pdb_line("ATOM", 6, "N", "ASN", "A", 2, 4.0, 0.0, 0.0, "N"),
        _pdb_line("ATOM", 7, "CA", "ASN", "A", 2, 5.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 8, "C", "ASN", "A", 2, 6.0, 0.0, 0.0, "C"),
        _pdb_line("ATOM", 9, "O", "ASN", "A", 2, 7.0, 0.0, 0.0, "O"),
        _pdb_line("ATOM", 10, "CB", "ASN", "A", 2, 5.0, 1.0, 0.0, "C"),
        _pdb_line("ATOM", 11, "CG", "ASN", "A", 2, 5.0, 2.0, 0.0, "C"),
        _pdb_line("ATOM", 12, "OD1", "ASN", "A", 2, 4.0, 2.0, 0.0, "O"),
        _pdb_line("ATOM", 13, "ND2", "ASN", "A", 2, 6.0, 2.0, 0.0, "N"),
        _pdb_line("HETATM", 14, "C1", "LIG", "Z", 9, 0.0, 2.0, 0.0, "C"),
        _pdb_line("HETATM", 15, "O1", "LIG", "Z", 9, 1.0, 2.0, 0.0, "O"),
        "TER",
        "END",
    ]
    pdb_path = tmp_path / "selection_input.pdb"
    pdb_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pdb_path


def test_atom_selection_add_and_subtract_preserve_order():
    first = AtomSelection.from_list([("A", 1, "N"), ("A", 1, "CA")])
    second = AtomSelection.from_list([("A", 1, "CA"), ("A", 1, "C")])

    assert (first + second).to_tuple() == (("A", 1, "N"), ("A", 1, "CA"), ("A", 1, "C"))
    assert (first - second).to_tuple() == (("A", 1, "N"),)


def test_from_dict_parses_rfd3_input_selection_without_pose_when_atoms_explicit():
    selection = AtomSelection.from_dict({"A1-2": "BKBN", "B5": ["N", "CA"]})

    assert selection.to_tuple() == (
        ("A", 1, "N"),
        ("A", 1, "CA"),
        ("A", 1, "C"),
        ("A", 1, "O"),
        ("A", 2, "N"),
        ("A", 2, "CA"),
        ("A", 2, "C"),
        ("A", 2, "O"),
        ("B", 5, "N"),
        ("B", 5, "CA"),
    )


def test_from_dict_parses_nested_chain_residue_atom_mapping():
    selection = AtomSelection.from_dict({"A": {1: ["N", "CA"], 2: "C,O"}})

    assert selection.to_tuple() == (
        ("A", 1, "N"),
        ("A", 1, "CA"),
        ("A", 2, "C"),
        ("A", 2, "O"),
    )


def test_from_rfd3_contig_expands_structure_atoms_and_ligand_index(tmp_path):
    pdb_path = _write_test_pdb(tmp_path)

    selection = AtomSelection.from_rfd3_contig("10-20,A1-2,/0,Z9", pose=pdb_path)

    assert ("A", 1, "N") in selection.to_tuple()
    assert ("A", 2, "ND2") in selection.to_tuple()
    assert ("Z", ("H_LIG", 9, " "), "C1") in selection.to_tuple()
    assert ("Z", ("H_LIG", 9, " "), "O1") in selection.to_tuple()


def test_from_rfd3_input_spec_parses_selection_fields_and_ligand(tmp_path):
    pdb_path = _write_test_pdb(tmp_path)
    input_spec = {
        "input": str(pdb_path),
        "contig": "A1-2,/0,20",
        "unindex": "A2",
        "select_fixed_atoms": {"A1": "BKBN", "LIG": "C1,O1"},
        "select_hbond_donor": {"A2": "TIP"},
        "select_hotspots": False,
        "ligand": "LIG",
    }

    selections = AtomSelection.from_rfd3_input_spec(input_spec)

    assert selections["select_fixed_atoms"].to_tuple() == (
        ("A", 1, "N"),
        ("A", 1, "CA"),
        ("A", 1, "C"),
        ("A", 1, "O"),
        ("Z", ("H_LIG", 9, " "), "C1"),
        ("Z", ("H_LIG", 9, " "), "O1"),
    )
    assert selections["select_hbond_donor"].to_tuple() == (
        ("A", 2, "CB"),
        ("A", 2, "CG"),
        ("A", 2, "OD1"),
        ("A", 2, "ND2"),
    )
    assert selections["select_hotspots"].to_tuple() == ()
    assert selections["ligand"].to_tuple() == (
        ("Z", ("H_LIG", 9, " "), "C1"),
        ("Z", ("H_LIG", 9, " "), "O1"),
    )
