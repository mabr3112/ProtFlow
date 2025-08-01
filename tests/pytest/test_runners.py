'''Module to test code from protflow.runners'''
# imports
import pytest

# dependencies

# custom
from protflow.runners import prepend_cmd, regex_expand_options_flags, expand_options_flags, options_flags_to_string

####################### variables ##############################
test_opts2 = "-out:path:all /home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/fastrelax -in:file:s /home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/output_pdbs/D6-A0-C19-B0_0001_0008.pdb -out:prefix r0005_ -out:file:scorefile r0005_D6-A0-C19-B0_0001_0008_score.json -out:file:scorefile_format json -parser:protocol=/home/mabr3112/ProtFlow/protflow/tools/runners_auxiliary_scripts//fastrelax_sap.xml -beta"
test_opts2_parsed = ({'out:path:all': '/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/fastrelax', 'in:file:s': '/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/output_pdbs/D6-A0-C19-B0_0001_0008.pdb', 'out:prefix': 'r0005_', 'out:file:scorefile': 'r0005_D6-A0-C19-B0_0001_0008_score.json', 'out:file:scorefile_format': 'json', 'parser:protocol': '/home/mabr3112/ProtFlow/protflow/tools/runners_auxiliary_scripts//fastrelax_sap.xml'}, {'beta'})
test_opts2_recompiled = " -out:path:all=/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/fastrelax -in:file:s=/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/output_pdbs/D6-A0-C19-B0_0001_0008.pdb -out:prefix=r0005_ -out:file:scorefile=r0005_D6-A0-C19-B0_0001_0008_score.json -out:file:scorefile_format=json -parser:protocol=/home/mabr3112/ProtFlow/protflow/tools/runners_auxiliary_scripts//fastrelax_sap.xml -beta"

esm_opts = "--fasta /home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/input_fastas/fasta_0007.fa --output_dir /home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/esm_preds --fake_flag1 --fake_flag2"
esm_opts_parsed = ({'fasta': '/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/input_fastas/fasta_0007.fa', 'output_dir': '/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/esm_preds'}, {'fake_flag1', 'fake_flag2'})
esm_opts_recompiled = "--fasta=/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/input_fastas/fasta_0007.fa --output_dir=/home/mabr3112/projects/riff_diff/rad/diffusion/nnewpot_d5_dw50/esm/esm_preds --fake_flag1 --fake_flag2"
####################### tests ##################################
@pytest.mark.parametrize("cmds, pre_cmd, expect", [
    (["A", "B", "C"], "pre", ['pre; A', 'pre; B', 'pre; C'])
])
def test_prepend_cmd(cmds, pre_cmd, expect):
    output = prepend_cmd(cmds, pre_cmd)
    assert output == expect

@pytest.mark.parametrize("options_str, sep, expect", [
    (test_opts2, "-", test_opts2_parsed),
    (esm_opts, "--", esm_opts_parsed)
])
def test_regex_expand_options_flags(options_str: str, sep: str, expect: str):
    output = regex_expand_options_flags(options_str, sep)
    assert output == expect

@pytest.mark.parametrize("opts, flags, sep, expect", [
    (test_opts2_parsed[0], test_opts2_parsed[1], "-", test_opts2_recompiled),
    (esm_opts_parsed[0], esm_opts_parsed[1], "--", esm_opts_recompiled)
])
def test_options_flags_to_string(opts: dict, flags: list, sep: str, expect: str):
    output = options_flags_to_string(options=opts, flags=flags, sep=sep)
    assert len(output.strip().split()) == len(expect.strip().split()) # workaround, because sets do not have order --> flags do not always occur in same order
    assert set(output.strip().split()) == set(expect.strip().split()) # workaround, because sets do not have order --> flags do not always occur in same order

@pytest.mark.parametrize("opts, sep, expect", [
    (esm_opts, "--", esm_opts_parsed)
])
def test_expand_options_flags(opts: str, sep: str, expect: str):
    output = expand_options_flags(options_str=opts, sep=sep)
    assert output == expect
