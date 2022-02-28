"""
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
"""