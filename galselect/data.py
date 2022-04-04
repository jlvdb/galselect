from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats


MCType = TypeVar("MCType", bound="MatchingCatalogue")
NormaliseType = Union[bool, MCType]


class FeaturesIncompatibleError(Exception):
    pass


class MatchingCatalogue(object):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        redshift: str,
        feature_names: List[str],
        feature_weights: Union[List[float], np.ndarray] = None
    ) -> None:
        self.data = dataframe
        # check the redshift data
        self._check_column_numeric(redshift)
        self._redshift = redshift
        # check the feature data
        if len(feature_names) == 0:
            raise ValueError("empty list of features provided")
        self._feature_names = []
        for col in feature_names:
            self._check_column_numeric(col)
            self._feature_names.append(col)
        # check the optional weights
        if feature_weights is None:
            self._weights = np.ones(len(feature_names))
        elif len(feature_weights) != len(feature_names):
            raise ValueError("number of features and weights does not match")
        else:
            self._weights = feature_weights

    def _check_column_numeric(
        self,
        colname: str
    ) -> None:
        if not np.issubdtype(self.data[colname], np.number):
            raise TypeError(
                f"type of column '{colname}' is not numeric")

    def __len__(self) -> int:
        return len(self.data)

    def n_features(self) -> int:
        return len(self._feature_names)

    def compatible(
        self,
        other: MCType
    ) -> bool:
        return self.n_features() == other.n_features()

    def _compute_norm(self) -> Tuple[npt.NDArray, npt.NDArray]:
        features = self.features
        offset = np.median(features, axis=0)
        scale = scipy.stats.median_abs_deviation(features, axis=0)
        return offset, scale

    @property
    def redshift(self) -> npt.NDArray:
        return self.data[self._redshift].to_numpy()

    @property
    def features(self) -> npt.NDArray:
        features = np.column_stack([
            self.data[col] for col in self._feature_names])
        return features

    def get_features(
        self,
        normalise: Optional[NormaliseType] = None
    ) -> npt.NDArray:
        features = self.features
        if normalise is None or normalise is False:
            return features * self._weights
        else:
            if normalise is True:
                offset, scale = self._compute_norm()
            elif type(normalise) is type(self):
                if not self.compatible(normalise):
                    raise FeaturesIncompatibleError
                offset, scale = normalise._compute_norm()
            else:
                raise TypeError(f"invalid normalisation '{type(normalise)}'")
            return (features - offset) / scale * self._weights

    def apply_mask(
        self,
        mask: npt.NDArray
    ) -> MCType:
        masked = self.__class__.__new__(self.__class__)
        # set new data with mask applied
        masked.data = self.data[mask]
        # copy static attributes
        for attr in ("_redshift", "_feature_names", "_weights"):
            setattr(masked, attr, getattr(self, attr))
        return masked
