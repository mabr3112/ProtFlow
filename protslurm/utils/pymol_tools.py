'''
Module that contains functions relevant for working with pymol and writing pymol scripts
'''

def mutations_pymol_scriptwriter(out_path:str, reference:str, variant:str, mutations:list[int]) -> str:
    '''writes pymol script to visualize mutations'''
    reference_name = reference.rsplit("/", maxsplit=1)[-1].replace(".pdb", "")
    variant_name = variant.rsplit("/", maxsplit=1)[-1].replace(".pdb", "")
    cmds = [
        f"load {reference}, {reference_name}",
        f"load {variant}, {variant_name}",
        "hide everything",
        "set_color sblue, (0.2980392277240753, 0.3960784375667572, 0.698039233684539)",
        "set_color spurple, (0.5490196347236633, 0.24705882370471954, 0.6000000238418579)",
        f"spectrum count, spurple sblue, {variant_name}",
        f"show cartoon, {reference_name} or {variant_name}",
        f"select variant_mutations, resi {'+'.join([str(x) for x in mutations])} and {variant_name}",
        f"select wt_mutation_positions, resi {'+'.join([str(x) for x in mutations])} and {reference_name}",
        f"color grey60, {reference_name}"
    ]
    
    if len(mutations) > 5:
        cmds.append(f"cealign wt_mutation_positions, variant_mutations")
    else:
        cmds.append(f"align {reference_name}, {variant_name}")

    cmds += [
        "show sticks, variant_mutations",
        "show lines, wt_mutation_positions and sidechain",
        "hide everything, hydrogens",
        "color sand, variant_mutations",
        "color atomic, (not elem C)",
        "scene wt_comparison, store",
        f"disable {reference_name}",
        f"scene variant_mutations, store"
    ]

    with open(out_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join(cmds))

