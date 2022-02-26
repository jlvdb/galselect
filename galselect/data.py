from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats


MCType = TypeVar("MCType", bound="MatchingCatalogue")
NormaliseType = Union[bool, MCType]


class MatchingCatalogue:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        redshift: str,
        feature_names: List[str],
        feature_weights: Union[List[float], np.ndarray] = None
    ) -> None:
        self.data = dataframe
        # check the redshift data
        self._check_column(redshift)
        self._redshift = redshift
        # check the feature data
        if len(feature_names) == 0:
            raise ValueError("no features provided")
        self._feature_names = []
        for col in feature_names:
            self._check_column(col)
            self._feature_names.append(col)
        # check the optional weights
        if feature_weights is None:
            self._weights = np.ones(self.n_features())
        elif len(feature_weights) != self.n_features():
            raise ValueError("number of features and weights does not match")
        else:
            self._weights = feature_weights

    def _check_column(
        self,
        colname: str
    ) -> None:
        if colname not in self.data:
            raise KeyError(f"column '{colname}' does not exist")
        if not np.issubdtype(self.data[colname], np.number):
            raise TypeError(
                f"type of column '{colname}' is not numeric")

    def __len__(self) -> int:
        return len(self.data)

    def n_features(self) -> int:
        return len(self._feature_names)

    def _compute_norm(self) -> Tuple[npt.NDArray, npt.NDArray]:
        offset = np.median(self.features, axis=0)
        scale = scipy.stats.median_abs_deviation(self.features, axis=0)
        return offset, scale

    @property
    def redshift(self) -> npt.NDArray:
        return self.data[self._redshift].to_numpy()

    @property
    def features(self) -> npt.NDArray:
        return np.column_stack([self.data[col] for col in self._feature_names])

    def get_features(
        self,
        normalise: Optional[NormaliseType] = None
    ) -> npt.NDArray:
        features = self.features
        if normalise is None:
            return features
        if normalise is not None:
            if normalise is True:
                offset, scale = self._compute_norm()
            elif type(normalise) is type(self):
                offset, scale = normalise._compute_norm()
            else:
                raise TypeError(f"invalid normalisation '{type(normalise)}'")
            return (features - offset) / scale
