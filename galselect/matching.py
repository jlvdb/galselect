import multiprocessing
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .data import MCType


def sort_with_argsort(data: npt.ArrayLike) -> Tuple[npt.NDArray, npt.NDArray]:
    sort_index = np.argsort(data)
    sorted_data = data[sort_index]
    return sort_index, sorted_data


def compute_thread_index_range(
    index_ranges: npt.NDArray,
    threads: int
) -> npt.NDArray:
    # check the number of threads if there are too few objects
    if threads is None:
        threads = multiprocessing.cpu_count()
    n_mock = len(index_ranges)
    if n_mock < threads:
        threads = n_mock
    # Distribute the mock index ranges over the thread. Since mock and data
    # are both sorted by redshift the index ranges are increasing
    # monotonically.
    splits = np.linspace(0, n_mock, threads + 1).astype(np.int_)
    thread_ranges = np.empty_like(index_ranges, shape=(threads, 2))
    for i, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
        thread_ranges[i] = (index_ranges[start, 0], index_ranges[end, 1])
    return thread_ranges


def euclidean_distance(
    point: npt.ArrayLike,
    other: npt.ArrayLike
) -> npt.NDArray:
    dist_squares = (np.atleast_2d(other) - np.atleast_2d(point)) ** 2
    dist = np.sqrt(dist_squares.sum(axis=1))
    return dist


class GalaxyMatcher:

    def __init__(
        self,
        mock_data: MCType,
        normalise: Optional[bool] = True,
        duplicates: Optional[bool] = True
    ) -> None:
        self._mock = mock_data
        self._mock_redshifts = self._mock.redshift
        self._normalise = normalise
        self._duplicates = duplicates
        # create a sorted view of the redshifts with a mapping index
        self._sort_index, self._sorted_redshifts = sort_with_argsort(
            self._mock_redshifts)
        # get the feature space and apply the redshift sorting
        features = self._mock.get_features(normalise)
        self._mock_features = features[self._sort_index]

    def compute_mock_index_range(
        self,
        redshift: npt.NDArray,
        d_idx: Optional[int] = 10000
    ) -> npt.NDArray:
        # find the appropriate range of indices in the sorted redshift array
        # around the target redshift(s)
        idx = np.searchsorted(self._sorted_redshifts, redshift)
        idx_lo = idx - d_idx // 2
        idx_hi = idx_lo + d_idx
        # clip array limits
        idx_range = np.empty_like(idx_lo, shape=(len(redshift), 2))
        idx_range[:, 0] = np.maximum(idx_lo, 0)
        idx_range[:, 1] = np.minimum(idx_hi, len(self._sorted_redshifts) - 1)
        return idx_range

    def match(
        self,
        data: MCType,
        d_idx: Optional[int] = 10000,
        clonecols: Optional[List[str]] = None,
        store_internal_distance: Optional[bool] = False,
        threads: Optional[int] = None
    ) -> pd.DataFrame:
        # sort data by redshift as well
        sort_index, sorted_redshifts = sort_with_argsort(data.redshift)
        # compute the range of mock data indices to consider for the matching
        mock_index_ranges = self.compute_mock_index_range(
            sorted_redshifts, d_idx)
        # compile a list of (overlapping) index ranges to apply in each thread
        thread_index_ranges = self.compute_parallel_chunks(
            mock_index_ranges, threads)
        threads = len(thread_index_ranges)
        # sort the data features
        data_features = data.get_features(self._normalise)[sort_index]

        # build a list of arguments for processings
        thread_args = []
        for _ in range(threads):
            args = [...]
            thread_args.append(args)
        # run the matching in parallel threads
        with multiprocessing.Pool(initializer=None, initargs=()) as pool:
            pool.starmap(id)
