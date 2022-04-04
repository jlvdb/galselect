import warnings
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .data import MCType


def sort_with_argsort(data: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    sort_index = np.argsort(data)
    sorted_data = np.asarray(data)[sort_index]
    return sort_index, sorted_data


def euclidean_distance(
    point: npt.ArrayLike,
    other: npt.ArrayLike
) -> npt.NDArray:
    dist_squares = (np.atleast_2d(other) - np.atleast_2d(point)) ** 2
    dist = np.sqrt(dist_squares.sum(axis=1))
    return dist


class GalaxyMatcher(object):

    def __init__(
        self,
        mock_data: MCType,
        normalise: Optional[bool] = True,
        duplicates: Optional[bool] = True
    ) -> None:
        self._mock = mock_data
        self._normalise = normalise
        self._duplicates = duplicates
        # create a sorted view of the redshifts with a mapping index
        self._mock_sort_index, self._mock_sorted_redshifts = sort_with_argsort(
            self._mock.redshift)
        # get the feature space and apply the redshift sorting
        features = self._mock.get_features(normalise)
        self._mock_features = features[self._mock_sort_index]

    def compute_mock_index_range(
        self,
        redshift: npt.NDArray,
        d_idx: Optional[int] = 10000
    ) -> npt.NDArray:
        # find the appropriate range of indices in the sorted mock redshifts
        # around the target redshift(s)
        idx = np.searchsorted(self._mock_sorted_redshifts, redshift)
        idx_lo = idx - d_idx // 2
        idx_hi = idx_lo + d_idx
        # clip at array limits
        limits = [0, len(self._sorted_redshifts) - 1]
        idx_range = np.empty_like(idx, shape=(len(redshift), 2))
        idx_range[:, 0] = np.maximum(idx_lo, limits[0])
        idx_range[:, 1] = np.minimum(idx_hi, limits[1])
        return idx_range

    def _single_match(
        self,
        mock_index_range: npt.ArrayLike,
        features: npt.ArrayLike,
    ):
        pass

    def match(
        self,
        data: MCType,
        d_idx: Optional[int] = 10000,
        clonecols: Optional[List[str]] = None,
        store_internal_distance: Optional[bool] = False,
        allow_duplicates: Optional[bool] = True
    ) -> pd.DataFrame:

        # check the redshift limits imposed by the simulation
        redshift_limits = [
            self._mock_sorted_redshifts[0],
            self._mock_sorted_redshifts[-1]]
        redshift_mask = (
            (data.redshift < redshift_limits[0]) |
            (data.redshift > redshift_limits[1]))
        n_out_of_range = np.count_nonzero(redshift_mask)
        if n_out_of_range > 0:
            msg = "{:} data entries are outside the mock redshift range "
            msg += "({:.3f}<=z<={:.3f}) and will be igorend"
            warnings.warn(msg.format(n_out_of_range, *redshift_limits))
        # remove objects that are out of range
        data = data.apply_mask(redshift_mask)

        # get the data features and apply the feature space normalisation
        if self._normalise:
            # apply from mock data, checks if feature types are compatible
            data_features = data.get_features(normalise=self._mock)
        else:
            data_features = data.get_features(normalise=False)
        # get data and sort by redshift, similar to the mock data
        data_sort_index, data_sorted_redshifts = sort_with_argsort(
            data.redshift)
        data_features = data_features[data_sort_index]
        
        # compute the range of mock data indices to consider for the matching
        mock_index_ranges = self.compute_mock_index_range(
            data_sorted_redshifts, d_idx)

        # For each data object find and match the mock object that is closest in
        # feature space. Experience has shown that brute force matching is
        # faster than a classical KD-tree due to the overhead when building the
        # tree and the typically small input catalogues (spectroscopic data).
        if allow_duplicates:
            pass
        else:
            self._mask = np.ones(len(self._mock), dtype="bool")
            
