import galselect.matching
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def index_ranges():
    return np.array([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7]])


def test_too_few_threads(index_ranges):
    with pytest.raises(ValueError):
        galselect.matching.compute_thread_index_range(index_ranges, 0)


def test_too_many_threads(index_ranges):
    thread_ranges = galselect.matching.compute_thread_index_range(
        index_ranges, len(index_ranges) + 1)
    assert len(index_ranges) == len(thread_ranges)


@pytest.mark.skip
def test_no_threads(index_ranges):
    pass


def test_output(index_ranges):
    thread_ranges = galselect.matching.compute_thread_index_range(
        index_ranges, 2)
    npt.assert_array_equal(thread_ranges, [[0, 4], [2, 7]])
    # identity case
    thread_ranges = galselect.matching.compute_thread_index_range(
        index_ranges, len(index_ranges))
    npt.assert_array_equal(thread_ranges, index_ranges)
