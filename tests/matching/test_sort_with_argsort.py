import galselect.matching
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def index_range_shuffled():
    return np.array([2, 3, 1, 4, 0]).astype(np.float_)


@pytest.fixture
def sort_order():
    return np.array([4, 2, 0, 1, 3]).astype(np.float_)


def test_sort_order(index_range_shuffled, sort_order):
    index, result = galselect.matching.sort_with_argsort(index_range_shuffled)
    npt.assert_array_equal(index, sort_order)
    npt.assert_array_equal(result, np.arange(len(sort_order)))
