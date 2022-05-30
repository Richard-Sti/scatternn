# Copyright (C) 2021 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Data frame selector for ML training and split routines."""

import numpy
from scipy.interpolate import interp1d

from sklearn.base import (BaseEstimator, TransformerMixin)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline


class DataFrameSelector(BaseEstimator, TransformerMixin):
    r"""
    A simple class to convert data from a numpy structured array into a
    sklearn-friendly format. This selector must be the first step of
    a Pipeline.

    Optionally applies :math:`\log_{10}` to the specified attributes.
    Parameters
    ----------
    attributes : (list of) str
        Attributes to be extracted from the structured array.
    log_attributes : (list of) str, optional
        Attributes to which log transform is applied. By default None.
    """

    def __init__(self, attributes, log_attributes=None):
        self._attributes = None
        self._log_attributes = None

        self.attributes = attributes
        self.log_attributes = log_attributes

    @staticmethod
    def _enforce_list_str(strings, name):
        """A support function to ensure `strings` is a list of strings."""
        if isinstance(strings, tuple):
            strings = list(strings)
        if not isinstance(strings, (list, str)):
            raise ValueError("'{}' must be a list or a single string"
                             .format(name))
        if isinstance(strings, str):
            strings = [strings]
        for attr in strings:
            if not isinstance(attr, str):
                raise ValueError("{} '{}' must be a string"
                                 .format(name, attr))
        return strings

    @property
    def attributes(self):
        """Attributes handled by this selector."""
        if self._attributes is None:
            raise ValueError("'attributes' not set.")
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        """Sets the attributes."""
        self._attributes = self._enforce_list_str(attributes, 'attribute')

    @property
    def log_attributes(self):
        """Returns the attributes which are to be log-transformed."""
        if self._log_attributes is None:
            return []
        return self._log_attributes

    @log_attributes.setter
    def log_attributes(self, attributes):
        """Sets the log attributes."""
        if attributes is None:
            return
        attributes = self._enforce_list_str(attributes, 'log_attribute')
        # Check that each attribute is in `attributes`
        for attr in attributes:
            if attr not in self.attributes:
                raise ValueError("Log attribute '{}' not found in attributes"
                                 .format(attr))
        self._log_attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Unpacks the data frame. Takes logarithmic transforms of beforehand
        specified parameters.
        """
        out = [None] * len(self.attributes)
        for i, par in enumerate(self.attributes):
            if par in self.log_attributes:
                out[i] = numpy.log10(X[par])
            else:
                out[i] = X[par]
        return numpy.vstack(out).T

    def inverse_transform(self, X):
        """Inverse transforms the data frame."""
        out = numpy.zeros(
                X.shape[0],
                dtype={'names': self.attributes,
                       'formats': ['float64'] * len(self.attributes)})
        for i, par in enumerate(self.attributes):
            if par in self.log_attributes:
                X[:, i] = 10**X[:, i]
            out[par] = X[:, i]
        return out


#
# =============================================================================
#
#                     Stratify split utilities
#
# =============================================================================
#


def stratify_split(data, features, targets, stratify_axis, log_stratify,
                   test_size=0.2, seed=42, Nbins=10, n_splits=50, weights=None,
                   ax_percentile=(0.5, 99.5)):
    """
    Performs stratified split on the input data structured array and
    return X_train, X_test, y_train, y_test structured arrays, and train and
    test set weights.

    The stratified split is performed along the specified target axis.

    Parameters
    ----------
    data : numpy.ndarray with named fields
        Input structured data array that contains both the features and
        target variable.
    features : (list of) str
        Feature attributes.
    targets : (list of) str
        Target attributes.
    stratify_axis : str
        Features/target along which to stratify.
    log_stratify : bool
        Whether to log-transform the target axis before performing the
        stratified split.
    test_size : float, optional
        A fractional size of the test set. Must be larger than 0 and smaller
        than 1. By default 0.2.
    seed : int
        Random seed.
    Nbins : int, optional
        Number of bins for stratified splitting. By default 10.
    n_splits : int, optional
        Number of re-shuffling & splitting iterations. By default 50.
    weights : str or numpy.array, optional
        Sample weights. Supported values are:
            ``uniform`` : Weights each sample equally.
            ``inverse`` : Weights each sample inversely proportionaly to the
                number of its occurences along the stratify axis.
        Weights are calculated by binning the data along the stratify axis
        (defined by the target), which depends on ``Nbins``.
        If input numpy.ndarray, then it must be one-dimensional and its size
        must match the number of data points. By default `None`.
    ax_percentile: len-2 tuple, optional
        Percentile range to estimate the bins for stratified split. By default
        (0.5, 99.5).

    Returns
    ------
    X_train : numpy.ndarray with named fields
        Train features.
    X_test : numpy.ndarray with named fields
        Test features.
    y_train : numpy.ndarray with named fields
        Train target.
    y_test : numpy.ndarray with named fields
        Test target.
    train_weights : numpy.ndarray
        Sample weights for the train set.  If weights is `None` is not
        returned.
    test_weights : numpy.ndarray
        Sample weights for the test set. If weights is `None` is not returned.
    """

    # Check the features inputs
    if isinstance(features, str):
        features = [features]
    elif not isinstance(features, list):
        raise ValueError("'features' must be a list or a string.")
    for feat in features:
        if not isinstance(feat, str):
            raise ValueError("Feature '{}' must be a string.".format(feat))
    # Check the targets
    if isinstance(targets, str):
        targets = [targets]
    elif not isinstance(targets, list):
        raise ValueError("'targets' must be a list or a string.")
    for targ in targets:
        if not isinstance(targ, str):
            raise ValueError("Target '{}' must be a string.".format(targ))
    # Check stratify axis
    if not isinstance(stratify_axis, str):
        raise ValueError("'stratify_axis' must be a string.")
    # And check the other inputs..
    if not isinstance(log_stratify, bool):
        raise ValueError("'log_stratify' must be a bool.")
    for p, v in zip(['Nbins', 'seed', 'n_splits'], [Nbins, seed, n_splits]):
        if not isinstance(v, int):
            raise ValueError("'{}' must be an integer.".format(p))
    if not 0.0 < test_size < 1.0:
        raise ValueError("'test_size' must be between 0 and 1.")
    if not (isinstance(ax_percentile, (list, tuple))
            and len(ax_percentile) == 2):
        raise ValueError("'ax_percentile' must be a len-2 list or tuple.")
    ax_percentile = list(ax_percentile)
    for val in ax_percentile:
        if not 0.0 < val < 100.:
            raise ValueError("'ax_percentile' must be between 0 and 100.")
    # Check the weights input
    if weights is None:
        pass
    elif isinstance(weights, numpy.ndarray):
        if weights.ndim != 1:
            raise ValueError("'weights' must be a 1D array.")
    elif isinstance(weights, str):
        if weights not in ['uniform', 'inverse']:
            raise ValueError("Unsupported 'weights' scheme.")
    else:
        raise ValueError("unknown type '{}' for 'weights'."
                         .format(type(weights)))

    # Enforce an increasing order
    if not ax_percentile[1] > ax_percentile[0]:
        ax_percentile = ax_percentile[::-1]
    # Stratify axis
    axis = data[stratify_axis]
    if log_stratify:
        axis = numpy.log10(axis)

    axmin, axmax = numpy.percentile(axis, ax_percentile)

    # Bins along which will perform stratify
    bins = numpy.linspace(axmin, axmax, Nbins)
    bands = numpy.digitize(axis, bins)

    # Optionally calculate the weights
    if weights is None:
        target_weights = None
    elif weights == 'inverse':
        # Count how many samples in each band and divide by the band width
        stat = numpy.zeros(Nbins + 1)
        bin_medians = numpy.zeros_like(stat)
        dx = bins[1] - bins[0]
        for i in range(Nbins + 1):
            mask = bands == i
            bin_medians[i] = numpy.median(axis[mask])
            stat[i] += numpy.sum(mask)
            if i == 0:
                stat[i] += numpy.sum(bands == i) / (bins[0] - axis.min())
            elif i == Nbins:
                stat[i] += numpy.sum(bands == i) / (axis.max() - bins[-1])
            else:
                stat[i] += numpy.sum(bands == i) / dx

        # Interpolate the counts normalised to the bin widths
        interp = interp1d(bin_medians, stat, bounds_error=False,
                          fill_value='extrapolate')
        target_weights = interp(axis)
        # We want the inverse abundance
        target_weights = numpy.abs(1./target_weights)
        # Normalise
        target_weights /= target_weights.sum()

    elif weights == 'uniform':
        target_weights = numpy.ones_like(axis)
    else:
        target_weights = weights / numpy.sum(weights)

    # Unpack the data into smaller structured arrays
    Nsamples = axis.size
    X = numpy.zeros(Nsamples, dtype={'names': features,
                                     'formats': [float] * len(features)})
    for feat in features:
        X[feat] = data[feat]

    y = numpy.zeros(Nsamples, dtype={'names': targets,
                                     'formats': [float] * len(targets)})
    for targ in targets:
        y[targ] = data[targ]

    # Perform the stratify split
    split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                   random_state=seed)
    for train_index, test_index in split.split(X, bands):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if target_weights is None:
            train_weights = None
            test_weights = None
        else:
            train_weights = target_weights[train_index]
            test_weights = target_weights[test_index]

    if weights is not None:
        return X_train, X_test, y_train, y_test, train_weights, test_weights

    return X_train, X_test, y_train, y_test


def apply_preprocess(train, test, pipeline):
    """
    Fit a pipeline on training data and then transform both train and test.

    Arguments
    ---------
    train: structured array
        Training data.
    test: structured array
        Test data.
    pipeline: :py:class:`sklearn.pipeline.Pipeline`
        Preprocessing pipeline.

    Returns
    -------
    train: structured array
        Transformed training data.
    test: structured array
        Tranformed test data.
    """
    train = pipeline.fit_transform(train)
    test = pipeline.transform(test)
    return train, test
