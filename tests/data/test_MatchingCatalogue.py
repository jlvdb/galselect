from textwrap import indent
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import scipy.stats
from galselect import MatchingCatalogue
from galselect.data import FeaturesIncompatibleError


class MockData:

    def __init__(self, n_features, weighted=False):
        self.n_features = n_features
        self.redshifts = np.array([0.5, 0.1, 0.2, 0.3, 0.8])
        self.features = [
            np.array([3,     4, 123,  24,  52]),
            np.array([2.5, 3.0, 4.0, 2.0, 3.2])][:n_features]
        self.names = [f"feature{n+1}" for n in range(n_features)]
        if weighted:
            self.weights = 2.0 + np.arange(n_features)
        else:
            self.weights = np.ones(n_features)
        self.median = np.median(self.features, axis=1)
        self.nmad = scipy.stats.median_abs_deviation(self.features, axis=1)
    
    @property
    def dataframe(self):
        df = pd.DataFrame({"redshift": self.redshifts})
        for key, value in zip(self.names, self.features):
            df[key] = value
        df["not numeric"] = "non numeric value"
        return df


@pytest.fixture
def mock_data():
    def make_mock_data(n_features=2, weighted=False):
        return MockData(n_features, weighted)
    return make_mock_data


@pytest.fixture
def mock_catalogue(mock_data):
    def make_mock_catalogue(n_features=2, weighted=False):
        data = mock_data(n_features, weighted)
        kwargs = {}
        if weighted:
            kwargs["feature_weights"] = data.weights
        return MatchingCatalogue(
            data.dataframe, "redshift", data.names, **kwargs)
    return make_mock_catalogue


def test_MatchingCatalogue_init(mock_data):
    data = mock_data()
    MatchingCatalogue(data.dataframe, "redshift", data.names)


def test_no_features_error(mock_data):
    data = mock_data()
    with pytest.raises(ValueError):
        MatchingCatalogue(data.dataframe, "redshift", [])


@pytest.mark.parametrize("n_features", [1, 2])
def test_weight(mock_data, n_features):
    data = mock_data(n_features, weighted=True)
    catalogue = MatchingCatalogue(
        data.dataframe, "redshift", data.names, data.weights)
    npt.assert_array_equal(catalogue._weights, data.weights)


@pytest.mark.parametrize("n_features", [1, 2])
def test_weight_default_value(mock_data, n_features):
    data = mock_data(n_features)
    catalogue = MatchingCatalogue(data.dataframe, "redshift", data.names)
    npt.assert_array_equal(catalogue._weights, np.ones(n_features))


@pytest.mark.parametrize("n_weights", [1, 3])
def test_weight_shape_mismatch(mock_data, n_weights):
    data = mock_data()
    with pytest.raises(ValueError):
        MatchingCatalogue(
            data.dataframe, "redshift", data.names,
            feature_weights=[2.0] * n_weights)


def test_check_redshit_not_numeric(mock_data):
    data = mock_data()
    with pytest.raises(TypeError):
        MatchingCatalogue(data.dataframe, "not numeric", data.names)


def test_check_feature_not_numeric(mock_data):
    data = mock_data()
    with pytest.raises(TypeError):
        MatchingCatalogue(
            data.dataframe, "redshift", [data.names[0], "not numeric"])


@pytest.mark.parametrize("n_features", [1, 2])
def test_n_features(mock_data, mock_catalogue, n_features):
    data = mock_data(n_features)
    catalogue = mock_catalogue(n_features)
    assert catalogue.n_features() == data.n_features


def test__len__(mock_data, mock_catalogue):
    data = mock_data()
    catalogue = mock_catalogue()
    assert len(catalogue) == len(data.dataframe)


def test_redshift(mock_data, mock_catalogue):
    data = mock_data()
    catalogue = mock_catalogue()
    npt.assert_array_equal(catalogue.redshift, data.dataframe["redshift"])


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_features(mock_data, mock_catalogue, n_features, weighted):
    data = mock_data(n_features, weighted)
    catalogue = mock_catalogue(n_features, weighted)
    npt.assert_array_equal(
        catalogue.features, np.column_stack(data.features))


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_compute_norm(mock_data, mock_catalogue, n_features, weighted):
    data = mock_data(n_features, weighted)
    catalogue = mock_catalogue(n_features, weighted)
    offset, scale = catalogue._compute_norm()
    npt.assert_array_equal(offset, data.median)
    npt.assert_array_equal(scale, data.nmad)


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_get_features(mock_data, mock_catalogue, n_features, weighted):
    data = mock_data(n_features, weighted)
    catalogue = mock_catalogue(n_features, weighted)
    npt.assert_array_equal(
        catalogue.features * data.weights,
        catalogue.get_features())
    npt.assert_array_equal(
        catalogue.features * data.weights,
        catalogue.get_features(False))


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_get_features_normed(mock_data, mock_catalogue, n_features, weighted):
    data = mock_data(n_features, weighted)
    catalogue = mock_catalogue(n_features, weighted)
    features = catalogue.get_features(True)
    # create a new catalogue from the obtined weights
    data.features = [feature for feature in features.T]
    normed_catalogue = MatchingCatalogue(
        data.dataframe, "redshift", data.names)
    normed_offset, normed_scale = normed_catalogue._compute_norm()
    # compare to normalisation parameters expected from normalised features
    npt.assert_array_equal(normed_offset, np.zeros(n_features))
    npt.assert_array_equal(normed_scale, data.weights)


@pytest.mark.parametrize("n_features", [1, 2])
def test_compatible(mock_catalogue, n_features):
    catalogue = mock_catalogue(n_features)
    assert catalogue.compatible(catalogue)


def test_not_compatible(mock_catalogue):
    catalogue1 = mock_catalogue(n_features=1)
    catalogue2 = mock_catalogue(n_features=2)
    assert not catalogue1.compatible(catalogue2)


@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_get_features_normed_external(
        mock_data, mock_catalogue, n_features, weighted):
    data = mock_data(n_features, weighted)
    catalogue = mock_catalogue(n_features, weighted)
    # normalise with the instance of itself
    features = catalogue.get_features(catalogue)
    # create a new catalogue from the obtined weights
    data.features = [feature for feature in features.T]
    normed_catalogue = MatchingCatalogue(
        data.dataframe, "redshift", data.names)
    normed_offset, normed_scale = normed_catalogue._compute_norm()
    # compare to normalisation parameters expected from normalised features
    npt.assert_array_equal(normed_offset, np.zeros(n_features))
    npt.assert_array_equal(normed_scale, data.weights)


def test_get_features_normed_external_incompatible(mock_catalogue):
    catalogue1 = mock_catalogue(n_features=1)
    catalogue2 = mock_catalogue(n_features=2)
    with pytest.raises(FeaturesIncompatibleError):
        catalogue1.get_features(catalogue2)


def test_get_features_normed_external_invalid_type(mock_catalogue):
    catalogue = mock_catalogue()
    with pytest.raises(TypeError):
        catalogue.get_features("invalid input")


def test_apply_mask(mock_data, mock_catalogue):
    data = mock_data(weighted=True)
    catalogue = mock_catalogue(weighted=True)
    mask = np.ones(len(catalogue), dtype="bool")
    mask[0] = False
    masked = catalogue.apply_mask(mask)
    npt.assert_array_equal(masked.redshift, catalogue.redshift[mask])
    npt.assert_array_equal(masked.features, catalogue.features[mask])
    # previous statements effectively test _redshift and _feature_names
    npt.assert_array_equal(catalogue._weights, masked._weights)
