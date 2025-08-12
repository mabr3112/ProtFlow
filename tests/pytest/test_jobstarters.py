'''Code to test jobstarters.py'''
# dependencies
import re
import time
import pytest

# customs
import protflow
from protflow.jobstarters import split_list, add_timestamp

def test_split_list():
    tl1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    expect1 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    expect2 = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    expect3 = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    assert split_list(tl1, element_length=2) == expect1
    assert split_list(tl1, n_sublists=5) == expect1
    assert split_list(tl1, element_length=5) == expect2
    assert split_list(tl1, n_sublists=2) == expect2
    assert split_list(tl1, element_length=3) == expect3

    with pytest.raises(ValueError):
        split_list(tl1, element_length=2, n_sublists=2)

    with pytest.raises(ValueError):
        split_list(tl1)

def test_add_timestamp():
    ''''''
    # define output pattern
    pattern = r"A_\d*"

    # create outputs
    ts1 = add_timestamp("A")
    time.sleep(0.0001)
    ts2 = add_timestamp("A")

    # timestamps must be different
    assert ts1 != ts2

    # must match pattern
    assert re.match(pattern, ts1) is not None
    assert re.match(pattern, ts2) is not None

def test_baseclass_unimplemented_methods():
    jst = protflow.jobstarters.JobStarter()

    with pytest.raises(NotImplementedError):
        jst.start(cmds=[], jobname="", wait=False, output_path="")
    with pytest.raises(NotImplementedError):
        jst.wait_for_job(jobname="", interval=1)

    jst.set_max_cores(5)
    assert jst.max_cores == 5
