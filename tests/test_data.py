import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from galselect import MatchingCatalogue


class MatchingCatalogueTest(unittest.TestCase):

    def setUp(self) -> None:
        self.medians = [3.2, 3.0]
        self.nmads = [1.0, 0.5]
        self.data = pd.DataFrame({
            "redshift": [0.1, 0.2, 0.3, 0.5, 0.8],
            "feature1": [1.5, 2.2, 3.7, 4.2, 3.2],
            "feature2": [2.5, 3.0, 4.0, 2.0, 3.2]
        })
        self.data["not numeric"] = "whatever"

    def test_init(self):
        MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])

    def test_init_no_feature(self):
        with self.assertRaisesRegex(ValueError, "no features"):
            MatchingCatalogue(
                self.data, "redshift", [])

    def test_init_no_weights(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        npt.assert_array_equal(mc._weights, np.ones(2))

    @unittest.expectedFailure
    def test_init_weight_mismatch(self):
        MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"],
            feature_weights=[2.0,])

    @unittest.expectedFailure
    def test_check_column_missing(self):
        MatchingCatalogue(
            self.data, "red", ["feature1", "feature2"])

    def test_check_column_not_numeric(self):
        with self.assertRaisesRegex(TypeError, "type of column"):
            MatchingCatalogue(
                self.data, "not numeric", ["feature1", "feature2"])
        with self.assertRaisesRegex(TypeError, "type of column"):
            MatchingCatalogue(
                self.data, "redshift", ["feature1", "not numeric"])

    def test_n_features(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        self.assertEqual(mc.n_features(), 2)

    def test_redshift(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        npt.assert_array_equal(mc.redshift, self.data["redshift"])

    def test_features(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1"])
        npt.assert_array_equal(
            mc.features, [[1.5], [2.2], [3.7], [4.2], [3.2]])
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        npt.assert_array_equal(
            mc.features,
            [[1.5, 2.5], [2.2, 3.0], [3.7, 4.0], [4.2, 2.0], [3.2, 3.2]])

    def test_compute_norm(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        offset, scale = mc._compute_norm()
        npt.assert_array_equal(offset, self.medians)
        npt.assert_array_equal(scale, self.nmads)

    def test_get_features_normed(self):
        mc = MatchingCatalogue(
            self.data.copy(), "redshift", ["feature1", "feature2"])
        features = mc.get_features(True)
        mc.data["feature1"] = features[:, 0]
        mc.data["feature2"] = features[:, 1]
        offset, scale = mc._compute_norm()
        npt.assert_array_equal(offset, np.zeros(2))
        npt.assert_array_equal(scale, np.ones(2))

    @unittest.skip
    def test_get_features_normed_external(self):
        mc1 = MatchingCatalogue(
            self.data.copy(), "redshift", ["feature1", "feature2"])
        mc2 = MatchingCatalogue(
            self.data.copy(), "redshift", ["feature2", "feature1"])
        features = mc1.get_features(mc2)

    @unittest.skip
    def test_get_features_normed_external_incompatible(self):
        mc1 = MatchingCatalogue(
            self.data.copy(), "redshift", ["feature1", "feature2"])
        mc2 = MatchingCatalogue(
            self.data.copy(), "redshift", ["feature2", "feature1"])
        features = mc1.get_features(mc2)
