import unittest

import numpy as np
import numpy.testing as npt

from galselect.parallel import SharedArrayHelper


class SharedArrayHelperTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data = np.linspace(1, 10, 10)
        self.sah = SharedArrayHelper(self.data)

    def test_init(self):
        for dtype in (np.int16, np.int32, np.int64, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                SharedArrayHelper(self.data.astype(dtype))

    def test__len__(self):
        self.assertEqual(len(self.data), len(self.sah))

    def test_size(self):
        self.assertEqual(self.data.size, self.sah.size)

    def test_get_data(self):
        npt.assert_array_equal(self.data, self.sah.get_data())

    def test_get_meta(self):
        meta = self.sah.get_meta()
        self.assertIs(self.sah.buffer, meta.buffer)
        self.assertEqual(self.data.dtype, meta.dtype)
        self.assertEqual(self.data.shape, meta.shape)

    def test_from_meta(self):
        meta = self.sah.get_meta()
        sah = SharedArrayHelper.from_meta(meta)
        self.assertTupleEqual(meta, sah.get_meta())
        npt.assert_array_equal(self.data, sah.get_data())
