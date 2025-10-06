'''Module to test code from protflow.runners'''
# imports
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from protflow.runners import Runner


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

def test_came_from_collect_scores_true():
    try:
        def collect_scores():
            raise RuntimeError("boom")
        collect_scores()
    except RuntimeError as e:
        assert Runner._came_from_collect_scores(e) is True

def test_came_from_collect_scores_false():
    def not_collect():
        raise RuntimeError("nope")
    try:
        not_collect()
    except RuntimeError as e:
        assert Runner._came_from_collect_scores(e) is False

class DummyJobstarter:
    def __init__(self, last_error_message=""):
        self.last_error_message = last_error_message

def test_wrap_run_collect_scores_includes_tail_and_sets_cause():
    class R(Runner):
        @Runner._wrap_run_with_stderr_context
        def collect_scores(self):
            raise RuntimeError("raw fail")

    r = R()
    r.current_jobstarter = DummyJobstarter("tail-lines from stderr")

    with pytest.raises(Runner.CrashError) as ei:
        r.collect_scores()

    msg = str(ei.value)
    assert "collect_scores failed" in msg
    assert "=== JOB ERROR OUTPUT ===" in msg
    assert "tail-lines from stderr" in msg
    assert isinstance(ei.value.__cause__, RuntimeError)
    assert "raw fail" in repr(ei.value.__cause__)

def test_wrap_run_collect_scores_without_tail():
    class R(Runner):
        @Runner._wrap_run_with_stderr_context
        def collect_scores(self):
            raise RuntimeError("oops")

    r = R()
    r.current_jobstarter = DummyJobstarter("")  # no stderr tail

    with pytest.raises(Runner.CrashError) as ei:
        r.collect_scores()

    msg = str(ei.value)
    assert "collect_scores failed" in msg
    assert "=== JOB ERROR OUTPUT ===" not in msg

def test_wrap_run_non_collect_scores_exception_is_passthrough():
    class R(Runner):
        @Runner._wrap_run_with_stderr_context
        def some_other_method(self):
            raise ValueError("bad!")

    r = R()
    with pytest.raises(ValueError) as ei:
        r.some_other_method()
    assert "collect_scores failed" not in str(ei.value)

class DummyProcessError(Exception):
    pass


def test_wrap_run_process_error_is_wrapped_even_outside_collect_scores(monkeypatch):
    # Ensure the decorator sees our DummyProcessError as ProcessError
    import protflow.runners as runners_mod
    monkeypatch.setattr(runners_mod, "ProcessError", DummyProcessError, raising=True)

    class R(Runner):
        @Runner._wrap_run_with_stderr_context
        def run(self):
            # Not named collect_scores, but should still be wrapped due to ProcessError
            raise DummyProcessError("proc failed")

    r = R()
    r.current_jobstarter = DummyJobstarter("stderr tail here")

    with pytest.raises(Runner.CrashError) as ei:
        r.run()

    msg = str(ei.value)
    assert "collect_scores failed" in msg  # message template from your wrapper
    assert "stderr tail here" in msg
    assert isinstance(ei.value.__cause__, DummyProcessError)
