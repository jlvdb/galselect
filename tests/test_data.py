import numpy as np
import numpy.testing as npt
import pandas as pd
import py
import pytest
import scipy.stats
from galselect import MatchingCatalogue
from galselect.data import FeaturesIncompatibleError


@pytest.fixture
def mock_redshifts():
    return np.array([0.5, 0.1, 0.2, 0.3, 0.8])


@pytest.fixture
def mock_feature_names():
    return [f"feature{n}" for n in (1, 2)]


@pytest.fixture
def mock_features():
    return [np.array([3,     4, 123,  24,  52]),
            np.array([2.5, 3.0, 4.0, 2.0, 3.2])]


@pytest.fixture
def mock_features_median(mock_features):
    return np.median(mock_features, axis=1)


@pytest.fixture
def mock_features_nmad(mock_features):
    return scipy.stats.median_abs_deviation(mock_features, axis=1)


@pytest.fixture
def mock_dataframe(mock_redshifts, mock_feature_names, mock_features):
    df = pd.DataFrame({"redshift": mock_redshifts})
    for key, value in zip(mock_feature_names, mock_features):
        df[key] = value
    df["not numeric"] = "non numeric value"
    return df


@pytest.fixture
def catalogue_instances(mock_dataframe, mock_feature_names):
    mcs = {
        n: MatchingCatalogue(
            mock_dataframe, "redshift", mock_feature_names[:n])
        for n in [1, 2]}
    return mcs


def test_MatchingCatalogue_init(mock_dataframe, mock_feature_names):
    MatchingCatalogue(mock_dataframe, "redshift", mock_feature_names)


def test_init_empty_feature(mock_dataframe):
    with pytest.raises(ValueError):
        MatchingCatalogue(mock_dataframe, "redshift", [])


def test_init_no_weights_provided(mock_dataframe, mock_feature_names):
    mc = MatchingCatalogue(
        mock_dataframe, "redshift", mock_feature_names)
    npt.assert_array_equal(mc._weights, np.ones(2))


@pytest.mark.parametrize("n_weights", [1, 3])
def test_init_weight_shape_mismatch(
        mock_dataframe, mock_feature_names, n_weights):
    with pytest.raises(ValueError):
        MatchingCatalogue(
            mock_dataframe, "redshift", mock_feature_names,
            feature_weights=[2.0] * n_weights)


def test_check_column_not_numeric(mock_dataframe, mock_feature_names):
    with pytest.raises(TypeError):
        MatchingCatalogue(
            mock_dataframe, "not numeric", mock_feature_names)
    with pytest.raises(TypeError):
        MatchingCatalogue(
            mock_dataframe, "redshift", [mock_feature_names[0], "not numeric"])


@pytest.mark.parametrize("n_features", [1, 2])
def test_n_features(mock_dataframe, mock_feature_names, n_features):
    mc = MatchingCatalogue(
        mock_dataframe, "redshift", mock_feature_names[:n_features])
    assert mc.n_features() == n_features


def test__len__(mock_dataframe, catalogue_instances):
    assert len(catalogue_instances[1]) == len(mock_dataframe)


def test_redshift(mock_dataframe, catalogue_instances):
    npt.assert_array_equal(
        catalogue_instances[2].redshift, mock_dataframe["redshift"])

@pytest.mark.parametrize("n_features", [1, 2])
def test_features(catalogue_instances, mock_features, n_features):
    npt.assert_array_equal(
        catalogue_instances[n_features].features,
        np.column_stack(mock_features[:n_features]))

@pytest.mark.parametrize("n_features", [1, 2])
def test_features_weighted(
        mock_dataframe, mock_feature_names, mock_features, n_features):
    mc = MatchingCatalogue(
        mock_dataframe, "redshift", mock_feature_names[:n_features],
        feature_weights=[2.0] * n_features)
    npt.assert_array_equal(
        mc.features,
        np.column_stack((2.0*np.array(mock_features))[:n_features]))

@pytest.mark.parametrize("n_features", [1, 2])
def test_compute_norm_weighted(
        mock_dataframe, mock_feature_names, mock_features_median,
        mock_features_nmad, n_features):
    mc = MatchingCatalogue(
        mock_dataframe, "redshift", mock_feature_names[:n_features],
        feature_weights=[2.0] * n_features)
    offset, scale = mc._compute_norm()
    npt.assert_array_equal(offset, (2*mock_features_median)[:n_features])
    npt.assert_array_equal(scale, (2*mock_features_nmad)[:n_features])


@pytest.mark.parametrize("n_features", [1, 2])
def test_get_features(catalogue_instances, n_features):
    mc = catalogue_instances[n_features]
    npt.assert_array_equal(mc.features, mc.get_features())
    npt.assert_array_equal(mc.features, mc.get_features(False))


@pytest.mark.parametrize("n_features", [1, 2])
def test_get_features_normed(
        catalogue_instances, mock_dataframe, mock_feature_names, n_features):
    features = catalogue_instances[n_features].get_features(True)
    mc_new = MatchingCatalogue(
        mock_dataframe.copy(), "redshift",
        mock_feature_names[:n_features])
    for i, n in enumerate(range(1, n_features+1)):
        mc_new.data[f"feature{n}"] = features[:, i]
    offset, scale = mc_new._compute_norm()
    npt.assert_array_equal(offset, np.zeros(n_features))
    npt.assert_array_equal(scale, np.ones(n_features))


@pytest.mark.parametrize("n_features", [1, 2])
def test_compatible(catalogue_instances, n_features):
    mc = catalogue_instances[n_features]
    assert mc.compatible(mc)


def test_not_compatible(catalogue_instances):
    mc1 = catalogue_instances[1]
    mc2 = catalogue_instances[2]
    assert not mc1.compatible(mc2)


@pytest.mark.parametrize("n_features", [1, 2])
def test_get_features_normed_external(
        catalogue_instances, mock_dataframe, mock_feature_names, n_features):
    mc = catalogue_instances[n_features]
    features = mc.get_features(mc)
    mc_new = MatchingCatalogue(
        mock_dataframe.copy(), "redshift",
        mock_feature_names[:n_features])
    for i, n in enumerate(range(1, n_features+1)):
        mc_new.data[f"feature{n}"] = features[:, i]
    offset, scale = mc_new._compute_norm()
    npt.assert_array_equal(offset, np.zeros(n_features))
    npt.assert_array_equal(scale, np.ones(n_features))


def test_get_features_normed_external_incompatible(catalogue_instances):
    mc1 = catalogue_instances[1]
    mc2 = catalogue_instances[2]
    with pytest.raises(FeaturesIncompatibleError):
        mc1.get_features(mc2)


def test_get_features_normed_external_invalid_type(catalogue_instances):
    mc1 = catalogue_instances[1]
    mc2 = catalogue_instances[2]
    with pytest.raises(TypeError):
        mc1.get_features("invalid input")
