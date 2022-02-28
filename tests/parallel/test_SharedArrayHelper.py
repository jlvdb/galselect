import numpy as np
import numpy.testing as npt
import pytest
from galselect.parallel import SharedArrayHelper


@pytest.fixture
def array_data():
    return np.linspace(1, 10, 10)


@pytest.fixture
def shared_array(array_data):
    return SharedArrayHelper(array_data)


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.float32, np.float64])
def test_init(array_data, dtype):
    SharedArrayHelper(array_data.astype(dtype))


def test__len__(array_data, shared_array):
    assert len(array_data) == len(shared_array)


def test_size(array_data, shared_array):
    assert array_data.size == shared_array.size


def test_get_data(array_data, shared_array):
    npt.assert_array_equal(array_data, shared_array.get_data())


def test_get_meta(array_data, shared_array):
    meta = shared_array.get_meta()
    assert shared_array.buffer is meta.buffer
    assert array_data.dtype == meta.dtype
    assert array_data.shape == meta.shape


def test_from_meta(array_data, shared_array):
    meta = shared_array.get_meta()
    sah = SharedArrayHelper.from_meta(meta)
    assert meta == sah.get_meta()
    npt.assert_array_equal(array_data, sah.get_data())
