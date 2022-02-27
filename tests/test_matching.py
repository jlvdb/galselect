from telnetlib import GA
import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from galselect import GalaxyMatcher, MatchingCatalogue
import galselect.matching


class SortWithArgsortTest(unittest.TestCase):

    def test_ordering(self):
        shuffled = np.array([2, 0, 3, 1])
        index, result = galselect.matching.sort_with_argsort(shuffled)
        npt.assert_array_equal(index, list(reversed(shuffled)))
        npt.assert_array_equal(result, list(range(4)))


class ComputeThreadIndexRangeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.index_ranges = np.array([
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7]])

    @unittest.expectedFailure
    def test_too_few_threads(self):
        galselect.matching.compute_thread_index_range(self.index_ranges, 0)

    def test_too_many_threads(self):
        thread_ranges = galselect.matching.compute_thread_index_range(
            self.index_ranges, len(self.index_ranges) + 1)
        self.assertEqual(len(self.index_ranges), len(thread_ranges))

    @unittest.skip
    def test_no_threads(self):
        pass

    def test_output(self):
        threads = 2
        with self.subTest(threads=threads):
            thread_ranges = galselect.matching.compute_thread_index_range(
                self.index_ranges, threads)
            npt.assert_array_equal(thread_ranges, [[0, 4], [2, 7]])
        threads = len(self.index_ranges)
        with self.subTest(threads=threads):
            thread_ranges = galselect.matching.compute_thread_index_range(
                self.index_ranges, threads)
            npt.assert_array_equal(thread_ranges, self.index_ranges)


class EuclideanDistanceTest(unittest.TestCase):
    pass


class GalaxyMatcherTest(unittest.TestCase):

    def setUp(self) -> None:
        self.redshifts = [0.5, 0.1, 0.2, 0.3, 0.8]
        self.featurenames = ["feature1", "feature2"]
        self.features = [
            np.array([3, 4, 123, 24, 52]),
            np.array([2.5, 3.0, 4.0, 2.0, 3.2])]
        self.medians = [24.0, 3.0]
        self.nmads = [21.0, 0.5]
        self.mock = pd.DataFrame({
            "redshift": self.redshifts,
            self.featurenames[0]: self.features[0],
            self.featurenames[1]: self.features[1]
        })
        self.mock["extra"] = ["a", "b", "c", "d", "e"]
        self.mock_cat = MatchingCatalogue(
            self.mock, "redshift", self.featurenames)

    def test_mock_sorted(self):
        gm = GalaxyMatcher(self.mock_cat, normalise=False)
        npt.assert_array_equal(
            gm._sorted_redshifts, [0.1, 0.2, 0.3, 0.5, 0.8])
        npt.assert_array_equal(
            gm._mock_features,
            np.column_stack(self.features)[[1, 2, 3, 0, 4]])
