'''Code to test jobstarters.py'''
# dependencies
import pytest

# customs
from protflow.jobstarters import split_list

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
