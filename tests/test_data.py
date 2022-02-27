import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

from galselect import MatchingCatalogue
from galselect.data import FeaturesIncompatibleError


class MatchingCatalogueTest(unittest.TestCase):

    def setUp(self) -> None:
        self.featurenames = ["feature1", "feature2"]
        self.features = [
            np.array([3, 4, 123, 24, 52]),
            np.array([2.5, 3.0, 4.0, 2.0, 3.2])]
        self.medians = [24.0, 3.0]
        self.nmads = [21.0, 0.5]
        # create test data
        self.data = pd.DataFrame({
            "redshift": [0.5, 0.1, 0.2, 0.3, 0.8],
            self.featurenames[0]: self.features[0],
            self.featurenames[1]: self.features[1]
        })
        # add invalid data
        self.data["not numeric"] = "non numeric value"


class MatchingCatalogueInitTest(MatchingCatalogueTest):

    def setUp(self) -> None:
        super().setUp()

    def test_init(self):
        MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])

    def test_init_empty_feature(self):
        with self.assertRaises(ValueError):
            MatchingCatalogue(
                self.data, "redshift", [])

    def test_init_no_weights_provided(self):
        mc = MatchingCatalogue(
            self.data, "redshift", ["feature1", "feature2"])
        npt.assert_array_equal(mc._weights, np.ones(2))

    def test_init_weight_shape_mismatch(self):
        for weight_length in [1, 3]:
            with self.subTest(weight_length=weight_length):
                with self.assertRaises(ValueError):
                    MatchingCatalogue(
                        self.data, "redshift", ["feature1", "feature2"],
                        feature_weights=[2.0] * weight_length)

    def test_check_column_not_numeric(self):
        with self.assertRaises(TypeError):
            MatchingCatalogue(
                self.data, "not numeric", ["feature1", "feature2"])
        with self.assertRaises(TypeError):
            MatchingCatalogue(
                self.data, "redshift", ["feature1", "not numeric"])

    def test_n_features(self):
        for n_features in [1, 2]:
            with self.subTest(n_features=n_features):
                mc = MatchingCatalogue(
                    self.data, "redshift", self.featurenames[:n_features])
                self.assertEqual(mc.n_features(), n_features)


class MatchingCatalogueInternalTest(MatchingCatalogueTest):

    def setUp(self) -> None:
        super().setUp()
        self.mcs = {}
        for n_features in [1, 2]:
            self.mcs[n_features] = MatchingCatalogue(
                self.data, "redshift", self.featurenames[:n_features])

    def test__len__(self):
        self.assertTrue(len(self.mcs[1]) == len(self.data))

    def test_redshift(self):
        npt.assert_array_equal(self.mcs[2].redshift, self.data["redshift"])

    def test_features(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                npt.assert_array_equal(
                    mc.features, np.column_stack(self.features[:n_features]))

    def test_compute_norm(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                offset, scale = mc._compute_norm()
                npt.assert_array_equal(offset, self.medians[:n_features])
                npt.assert_array_equal(scale, self.nmads[:n_features])

    def test_compute_norm_weighted(self):
        for n_features in [1, 2]:
            with self.subTest(n_features=n_features):
                mc = MatchingCatalogue(
                    self.data, "redshift", self.featurenames[:n_features],
                    feature_weights=[2.0] * n_features)
                offset, scale = mc._compute_norm()
                npt.assert_array_equal(offset, (2*self.medians)[:n_features])
                npt.assert_array_equal(scale, self.nmads[:n_features])

    def test_get_features(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                npt.assert_array_equal(mc.features, mc.get_features())
                npt.assert_array_equal(mc.features, mc.get_features(False))

    def test_get_features_normed(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                features = mc.get_features(True)
                mc_new = MatchingCatalogue(
                    self.data.copy(), "redshift",
                    self.featurenames[:n_features])
                for i, n in enumerate(range(1, n_features+1)):
                    mc_new.data[f"feature{n}"] = features[:, i]
                offset, scale = mc_new._compute_norm()
                npt.assert_array_equal(offset, np.zeros(n_features))
                npt.assert_array_equal(scale, np.ones(n_features))


class MatchingCatalogueExternalTest(MatchingCatalogueInternalTest):

    def test_compatible(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                self.assertTrue(mc.compatible(mc))

    def test_not_compatible(self):
        self.assertFalse(self.mcs[1].compatible(self.mcs[2]))

    def test_get_features_normed_external(self):
        for n_features, mc in self.mcs.items():
            with self.subTest(n_features=n_features):
                features = mc.get_features(mc)
                mc_new = MatchingCatalogue(
                    self.data.copy(), "redshift",
                    self.featurenames[:n_features])
                for i, n in enumerate(range(1, n_features+1)):
                    mc_new.data[f"feature{n}"] = features[:, i]
                offset, scale = mc_new._compute_norm()
                npt.assert_array_equal(offset, np.zeros(n_features))
                npt.assert_array_equal(scale, np.ones(n_features))

    def test_get_features_normed_external_incompatible(self):
        with self.assertRaises(FeaturesIncompatibleError):
            self.mcs[1].get_features(self.mcs[2])

    def test_get_features_normed_external_invalid_type(self):
        with self.assertRaises(TypeError):
            self.mcs[1].get_features("invalid input")
