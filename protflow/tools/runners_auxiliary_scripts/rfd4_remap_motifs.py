"""Worker for parallel extraction of RFD4-Proteina output provenance maps.

Reads the standard annotations RFD4-Proteina writes into every generated CIF and
turns them into the input-to-output residue map that motif remapping needs.

Unlike the other auxiliary scripts in this directory, this worker does **not**
import ProtFlow: it runs under the RFD4-Proteina environment's own interpreter,
because it needs ``atomworks`` and ``rfproteina`` to read the annotations through
the vendor's documented API. Everything it needs arrives in the input JSON and
everything it returns goes out as plain JSON, so the two environments never have
to share a dependency.

Annotations consumed (see the RFD4-Proteina output-format reference):

``SA_SRC``
    Per-atom source component label naming the input residue each output atom
    came from; ``"."`` means the atom was generated rather than carried over.
``SA_CRDSRC``
    Same labels, restricted to coordinate-conditioned atoms.
``SA_IDXGRPSRC``
    Original unindexed ``IndexGroup`` of a mapped motif token; ``-1`` when the
    token was not an unindexed insertion.
``SA_UII``
    Per-atom boolean marking unindexed insertion targets.

A record may additionally ask for a *renumbered input*: given the design's own
input structure and a destination path, the worker rewrites that structure's
chain/residue numbering into the design's numbering and saves it, so the input
can be compared against the design without reconstructing the mapping by hand.

Input JSON: ``[{"description": str, "cif_location": str,
"input_location": str | None, "renumbered_input_path": str | None}, ...]``
Output JSON: one record per input, each carrying ``source_index_map`` plus the
derived motif/unindexed residue lists and, when requested, ``renumbered_input``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def _residue_label(chain_id, res_id) -> str:
    """Format one output residue as the ``<chain><res_id>`` label ProtFlow uses."""
    return f"{chain_id}{int(res_id)}"


def renumber_input_structure(input_location: str, source_index_map: dict, out_path: str) -> dict:
    """
    Rewrite an input structure into the numbering of one generated design.

    Parameters
    ----------
    input_location : str
        Path to the structure the design was generated from.
    source_index_map : dict
        ``{input_residue: output_residue}`` from :func:`build_provenance_record`.
    out_path : str
        Destination path. The suffix selects the format; ``.cif`` is written
        plain, ``.cif.gz`` / ``.cif.zst`` compressed.

    Returns
    -------
    dict
        ``{"renumbered_input": path, "n_renumbered": int, "n_unmapped": int}``.
        ``n_unmapped`` counts atoms whose residue had no map entry; those keep
        their original numbering, which is expected for ligands and for any
        residue the model did not carry over.
    """
    import numpy as np
    from atomworks.io.parser import parse
    from atomworks.io.utils.io_utils import to_cif_file
    from rfproteina.utils.parse_presets import ANNOTATED_CIF

    atom_array = parse(input_location, config=ANNOTATED_CIF)["assemblies"]["1"][0]

    chain_ids = np.array(atom_array.chain_id, dtype=object)
    res_ids = np.array(atom_array.res_id)

    new_chain_ids = list(atom_array.chain_id)
    new_res_ids = list(atom_array.res_id)
    n_renumbered = 0
    n_unmapped = 0

    for index, (chain_id, res_id) in enumerate(zip(chain_ids, res_ids)):
        target = source_index_map.get(f"{chain_id}{int(res_id)}")
        if target is None:
            n_unmapped += 1
            continue
        # output labels are "<chain><res_id>"; the chain id is the leading non-digit run
        split_at = 0
        while split_at < len(target) and not (target[split_at].isdigit() or target[split_at] == "-"):
            split_at += 1
        new_chain_ids[index] = target[:split_at]
        new_res_ids[index] = int(target[split_at:])
        n_renumbered += 1

    atom_array.chain_id = np.array(new_chain_ids)
    atom_array.res_id = np.array(new_res_ids, dtype=int)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    to_cif_file(atom_array, out_path)
    return {"renumbered_input": os.path.abspath(out_path), "n_renumbered": n_renumbered, "n_unmapped": n_unmapped}


def build_provenance_record(description: str, cif_location: str) -> dict:
    """
    Extract the provenance maps for one generated structure.

    Parameters
    ----------
    description : str
        Pose description, echoed back so results can be matched to score rows.
    cif_location : str
        Path to the generated annotated CIF. Plain, gzip and Zstandard files are
        all readable.

    Returns
    -------
    dict
        Record with the keys:

        ``description``
            Echoed input description.
        ``source_index_map``
            ``{input_residue: output_residue}``, e.g. ``{"B45": "B45"}``. Only
            source labels that resolve to exactly one output residue appear
            here, so the map is always safe to apply one-to-one.
        ``ambiguous_source_labels``
            Source labels that spanned more than one output residue and were
            therefore left out of ``source_index_map``.
        ``motif_residues``
            Output residues carrying coordinate conditioning (``SA_CRDSRC``).
        ``unindexed_residues``
            Output residues flagged as unindexed insertions (``SA_UII``).
        ``index_group_map``
            ``{output_residue: index_group}`` for residues whose
            ``SA_IDXGRPSRC`` is not ``-1``.
        ``n_source_atoms``
            Number of atoms carrying any source label, for sanity checking.
    """
    # imported here so --help works without the RFD4 environment loaded
    from atomworks.io.parser import parse
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY
    from rfproteina.utils.standard_annotations import SA_CRDSRC, SA_IDXGRPSRC, SA_SRC, SA_UII

    atom_array = parse(cif_location, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]

    def annotation(standard_annotation):
        """Return an annotation array, or None when the structure lacks it."""
        try:
            return standard_annotation.annotation(atom_array)
        except Exception:  # pylint: disable=W0718 - absent annotation is not an error
            return None

    src = annotation(SA_SRC)
    crd_src = annotation(SA_CRDSRC)
    idx_grp = annotation(SA_IDXGRPSRC)
    unindexed = annotation(SA_UII)

    chain_ids = atom_array.chain_id
    res_ids = atom_array.res_id

    # input residue label -> set of output residues it landed on
    source_targets: dict[str, set] = defaultdict(set)
    n_source_atoms = 0
    if src is not None:
        for label, chain_id, res_id in zip(src, chain_ids, res_ids):
            if label == ".":
                continue
            n_source_atoms += 1
            source_targets[str(label)].add(_residue_label(chain_id, res_id))

    source_index_map = {}
    ambiguous = []
    for label, targets in source_targets.items():
        if len(targets) == 1:
            source_index_map[label] = next(iter(targets))
        else:
            ambiguous.append(label)

    def residues_where(mask_values, predicate) -> list:
        """Collect unique output residues whose atoms satisfy *predicate*."""
        if mask_values is None:
            return []
        found = {
            _residue_label(chain_id, res_id)
            for value, chain_id, res_id in zip(mask_values, chain_ids, res_ids)
            if predicate(value)
        }
        return sorted(found)

    motif_residues = residues_where(crd_src, lambda value: value != ".")
    unindexed_residues = residues_where(unindexed, bool)

    index_group_map = {}
    if idx_grp is not None:
        for value, chain_id, res_id in zip(idx_grp, chain_ids, res_ids):
            if int(value) != -1:
                index_group_map[_residue_label(chain_id, res_id)] = int(value)

    return {
        "description": description,
        "source_index_map": source_index_map,
        "ambiguous_source_labels": sorted(ambiguous),
        "motif_residues": motif_residues,
        "unindexed_residues": unindexed_residues,
        "index_group_map": index_group_map,
        "n_source_atoms": n_source_atoms,
    }


def main(args):
    """Build provenance records for every entry in one worker input JSON."""
    with open(args.input_json, "r", encoding="UTF-8") as handle:
        records = json.load(handle)

    results = []
    for record in records:
        try:
            result = build_provenance_record(record["description"], record["cif_location"])
            if record.get("renumbered_input_path") and record.get("input_location"):
                result.update(
                    renumber_input_structure(
                        input_location=record["input_location"],
                        source_index_map=result["source_index_map"],
                        out_path=record["renumbered_input_path"],
                    )
                )
            results.append(result)
        except Exception as exc:  # pylint: disable=W0718 - one bad design must not sink the batch
            results.append({
                "description": record["description"],
                "source_index_map": {},
                "ambiguous_source_labels": [],
                "motif_residues": [],
                "unindexed_residues": [],
                "index_group_map": {},
                "n_source_atoms": 0,
                "renumbered_input": None,
                "error": f"{type(exc).__name__}: {exc}",
            })

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="UTF-8") as handle:
        json.dump(results, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_json", required=True, help="Worker input JSON containing provenance records.")
    parser.add_argument("--output_json", required=True, help="Worker output JSON path.")
    main(parser.parse_args())
