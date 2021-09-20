import warnings

import numpy as np
import pandas as pd
import tqdm


class DataMatcher:
    """
    Match galaxy mock data to a galaxy data catalogue based on features such as
    magnitudes. The matching is done within a neighbourhood of nearest mock data
    redshift around the input data redshifts. The matching is unique such that
    each mock objects is used at most once. This essentially allows cloning the
    data catalogue using mock data with very similar properties.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with with galaxy mock data such as redshifts and the features
        used for matching.
    redshift_name : str
        Name of the redshift column in the mock data.
    feature_names : list of str
        List of names of the feature columns in the mock data.
    initial_mask : array_like of bool (optional)
        Mask array that with an initial selection of mock data entries that
        may be used for the matching, entries with False will be ingored
        during the matching.
    redshift_warning : float (optional)
        Issue warnings if the redshift neighbourhood of the selected mock data
        exceeds this threshold, e.g. due to a lock of objects in the redshift
        range or exhaustion of objects due to the uniqueness of the matching.
    """
    
    def __init__(
            self, data, redshift_name, feature_names,
            initial_mask=None, redshift_warning=0.05):
        if type(data) is not pd.DataFrame:
            raise TypeError(f"input data must be of type {type(pd.DataFrame)}")
        self.data = data
        self.z_warn = redshift_warning

        self._init_available_mask(initial_mask)
        self._init_feature_space(feature_names)
        self._init_redshifts(redshift_name)

    def _init_available_mask(self, initial_mask=None):
        """
        Create a mask for entries in the data table that can be used to match
        to the data catalogue. The mask is updated after every match to ensure
        that objects are used uniquely.

        Parameters:
        -----------
        initial_mask : array_like of bool (optional)
            Mask array that with an initial selection of mock data entries that
            may be used for the matching, entries with False will be ingored
            during the matching.
        """
        if initial_mask is None:
            self.available_mask = np.ones(len(self.data), "bool")
        else:
            self.available_mask = initial_mask

    def _init_redshifts(self, redshift_name):
        """
        Sort the mock redshift data in ascending order and create an index
        mapping from the sorted to originally ordered mock data.

        Parameters:
        -----------
        redshift_name : str
            Name of the redshift column in the mock data.
        """
        try:
            self.redshifts = np.array(self.data[redshift_name])
        except KeyError:
            raise KeyError(f"redshift column '{redshift_name}' not found")
        # sort redshifts to later select objects with similar redshifts
        self.z_sort_idx = np.argsort(self.redshifts)
        self.z_sorted = self.redshifts[self.z_sort_idx]
        self.z_min = self.z_sorted[0]
        self.z_max = self.z_sorted[-1]

    def _init_feature_space(self, feature_names):
        """
        Collect the feature space data from the mock data columns in a 2-dim
        array.

        Parameters:
        -----------
        feature_names : list of str
            List of names of the feature columns in the mock data.
        """
        for name in feature_names:
            if name not in self.data:
                raise KeyError(f"feature column '{name}' not found")
        self.feature_space = np.column_stack([
            self.data[name] for name in feature_names])
        self.feature_dim = len(feature_names)

    def close_redshifts(self, redshift, d_idx):
        """
        Find a number of objects in the mock data that are closest to a given
        input redshift.

        Parameters:
        -----------
        redshift : float
            Find the closest mock objects around this redshift.
        d_idx : int
            Number of mock objects closest to the input redshift.
        
        Returns:
        --------
        idx_mock : array_like of int
            Indices of entries in the mock data that selects the d_idx closest
            objects around the input redshift.
        dz : float
            Redshift range covered by the selected mock objects.
        """
        d_idx //= 2
        # check that the redshift is covered by the mock data
        if redshift < self.z_min or redshift > self.z_max:
            raise ValueError(f"input redshift z={redshift:.3f} out of range")
        # find the appropriate range of indices in the sorted redshift array
        # around the target redshift
        idx = np.searchsorted(self.z_sorted, redshift)
        idx_lo = np.maximum(idx - d_idx, 0)
        idx_hi = np.minimum(idx + d_idx, len(self.z_sorted) - 1)
        # map the selected indices back to the order in the mock data table
        idx_mock = self.z_sort_idx[idx_lo:idx_hi+1]
        dz = self.z_sorted[idx_hi] - self.z_sorted[idx_lo]
        return idx_mock, dz

    @staticmethod
    def euclidean_distance(point, other):
        """
        Compute the Euclidean distance between two points, a point and a set of
        points or two set of points with equal size.

        Parameters:
        -----------
        point : array_like
        other : array_like

        Returns:
        --------
        dist : array_like
        """
        dist_squares = (np.atleast_2d(other) - np.atleast_2d(point)) ** 2
        dist = np.sqrt(dist_squares.sum(axis=1))
        return dist

    def single_match(
            self, redshift, features, d_idx=10000, return_mock_distance=False):
        """
        Match a single data point in the feature space to the mock data with
        similar redshift. Matches are unique and every consequtive match is
        guaranteed to be a different mock data entry. The method my fail if the
        redshift is outside the redshift range of the mock data or all mock
        objects are exhausted in the redshift neighbourhood.

        Parameters:
        -----------
        redshift : float
            Match only the mock data closest to this redshift.
        features : array_like
            Features of the data that are compared to the mock data features.
        d_idx : int
            Size of the redshift neighbourhood, total number of mock objects
            that are considered for the feature space matching.
        return_mock_distance : bool
            Additionally calculate the distance of the match to the next point
            in the mock data in feature space.

        Returns:
        --------
        match_idx : int
            Index in the mock data table of the best match to the input data.
        meta : dict
            Meta data for the match such as the number of neighbours after
            masking (n_neigh), the distance in feature space between the data
            and the match (dist_data), and, if return_mock_distance is given,
            the distance of the match to the next point in the mock feature
            space (dist_mock).
        """
        dim = self.feature_dim
        if features.shape != (dim,):
            raise ValueError(
                f"dimensions of data features do not match (expected {dim})")
        # select the nearest objects in the mock data and its features used for
        # matching to the data
        neighbourhood_idx, z_range = self.close_redshifts(redshift, d_idx)
        if z_range > self.z_warn:
            warnings.warn(
                f"redshift range of neighbourhood exceeds dz={z_range:.3f}")
        mock_features = self.feature_space[neighbourhood_idx]
        available_mask = self.available_mask[neighbourhood_idx]
        n_candidates = np.count_nonzero(available_mask)
        if n_candidates == 0:
            raise ValueError(f"no unmasked entries within d_idx={d_idx:d}")

        # find nearest unmasked mock entry
        data_dist = self.euclidean_distance(
            features, mock_features[available_mask])
        idx = np.argmin(data_dist)  # nearest neighbour in feature space
        match_data_dist = data_dist[idx]
        match_idx = neighbourhood_idx[available_mask][idx]
        # mask the match and collect meta data for the match
        self.available_mask[match_idx] = False
        meta = {
            "n_neigh": n_candidates,
            "dist_data": match_data_dist}

        # optionally find the distance between the match and the next nearest
        # neighbour within mock data
        if return_mock_distance:
            match_features = self.feature_space[match_idx]
            mock_dist = self.euclidean_distance(match_features, mock_features)
            # store the distance to the closest neighbour which is not the match
            # itself
            meta["dist_mock"] = np.sort(mock_dist)[1]
        return match_idx, meta

    def match_catalog(
            self, redshifts, features, d_idx=10000, return_mock_distance=False,
            progress=False):
        """
        Create data catalouge by matching a data catalogue in the feature space
        to the mock data with in a neighbourhood of similar redshifts. Matches
        are unique and every consequtive match is guaranteed to be a different
        mock data entry. The method my fail if the redshift is outside the
        redshift range of the mock data or all mock objects are exhausted in the
        redshift neighbourhood.

        Parameters:
        -----------
        redshifts : float
            Match only the mock data closest to these data redshifts.
        features : array_like
            Features of the data that are compared to the mock data features.
        d_idx : int
            Size of the redshift neighbourhood, total number of mock objects
            that are considered for the feature space matching.
        return_mock_distance : bool
            Additionally calculate the distance of the match to the next point
            in the mock data in feature space.
        progress : bool
            Show a progressbar for the matching operation

        Returns:
        --------
        catalogue : pd.DataFrame
            Catalogue of matches from the mock data that are matched to the
            input data. Additional columns with match statistics are appended
            which contain the number of neighbours after masking (n_neigh), the
            distance in feature space between the data and the match
            (dist_data), and, if return_mock_distance is given, the distance of
            the match to the next point in the mock feature space (dist_mock).
        """
        idx_match = np.empty_like(redshifts, dtype=np.int)
        match_stats = {
            "n_neigh": np.empty_like(redshifts, dtype=np.int),
            "dist_data": np.empty_like(redshifts, dtype=np.float)}
        if return_mock_distance:
            match_stats["dist_mock"] = np.empty_like(redshifts, dtype=np.float)
        # iterate the input catalogue and collect the match index and statistics
        try:
            if progress:
                pbar = tqdm.tqdm(total=len(redshifts))
                step = 100
            for i, (redshift, entry) in enumerate(zip(redshifts, features)):
                idx, meta = self.single_match(
                    redshift, entry, d_idx, return_mock_distance)
                idx_match[i] = idx
                for key, val in meta.items():
                    match_stats[key][i] = val
                if i % step == 0 and progress:
                    pbar.update(step)
        finally:
            if progress:
                pbar.close()
        # construct the matched output catalogue
        catalogue = self.data.iloc[idx_match].copy(deep=True)
        for colname, values in match_stats.items():
            catalogue[colname] = values
        return catalogue
