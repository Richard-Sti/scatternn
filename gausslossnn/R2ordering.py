# Copyright (C) 2022  Richard Stiskalek
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
from copy import deepcopy

import numpy
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


def parse_data(data, features, target, log_labels, test_feature=None):
    """
    Parse data and return `X` and `y` that can be given to sklearn models.

    Arguments
    ---------
    data : structured array
        Array containing the data to be extracted.
    features : list of str
        List of feature labels to be extracted from `data`.
    target : str
        Target label.
    log_labels : list of str
        List of labels that are to be log-transformed. If a feature or target
        are in this list returns their logarithm.
    test_feature : str, optional
        Test feature.

    Returns
    -------
    X : n-dimensional array
        Features array of shape (`Nsamples`, `Nfeatures`).
    y : 1-dimensional array
        Target array of shape (`Nsamples`, ).
    z : 1-dimensional array
        Test feature array of shape (`Nsamples`, ). Returned only if
        `test_feature` is not `None`.
    """
    if test_feature is not None and test_feature in features:
        raise ValueError("`test_feature` {} already in `features`."
                         .format(test_feature))

    X = numpy.array([numpy.log10(data[p]) if p in log_labels else data[p]
                     for p in features]).T
    y = data[target]
    # Check whether to log the target
    if target in log_labels:
        y = numpy.log10(y)
    if test_feature is None:
        return X, y
    # Get the test_feature and check whether to log it
    z = data[test_feature]
    if test_feature in log_labels:
        z = numpy.log10(z)
    return X, y, z


def incremental_importance(data, features, target, model, log_labels=None,
                           test_size=0.2, verbose=True, seed=42):
    """
    List features by their test set score in order that maximises the score
    increments.

    Arguments
    ---------
    data : structured array
        Array containing the data to be extracted.
    features : list of str
        List of feature labels.
    target : str
        Target label.
    model :  sklearn model
        Sklearn model that with `fit` and `score` methods.
    log_labels : list of str, optional
        List of labels that are log-transformed. By default `None`.
    test_size : float, optional
        Fractional test size to evaluate the scores of fitted models. By
        default 0.2.
    verbose : bool, optional
        Verbosity flag, by default `True`.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    out : dict
        Dictionary with items:
            ordered_features : list of str
                List of cumulative features.
            scores : 1-dimensional array
                Cumulative maximised scores.
            correlations : 1-dimensional array
                Correlation between the newly added feature that maximises
                the score increment and a model trained without it.
    """
    # If only a string passed in convert to a list
    if isinstance(features, str):
        features = [features]
    # Check that the requested labels are present in the data array
    for feat in features + [target]:
        if feat not in data.dtype.names:
            raise KeyError("Label `{}` not present in the `data` array."
                           .format(feat))
    # Will be popping of the  features list so make a copy
    features = features.copy()
    # If no log labels turn into an empty list
    if log_labels is None:
        log_labels = []

    # Initialise list to keep track of results
    sel_feats = []
    best_scores = []
    correlations = []

    # Best model on the previous iteration
    previous_model = None
    # Continue until all features have been ranked
    while len(features) > 0:
        scores = numpy.zeros(len(features))
        models = [None] * len(features)

        # Loop over the features and select one that maximises the test score
        for i, feat in enumerate(features):
            # Select the relevant columns from the data array
            X, y = parse_data(data, sel_feats + [feat], target, log_labels)
            # A simple test-train split
            Xtrain, Xtest, ytrain, ytest = train_test_split(
                X, y, test_size=test_size, random_state=seed)
            # Fit the model
            current_model = deepcopy(model)
            current_model.fit(Xtrain, ytrain)
            # Append the model score and the model
            scores[i] = current_model.score(Xtest, ytest)
            models[i] = current_model

        k = numpy.argmax(scores)
        best_feature = features.pop(k)

        # If selecting the first best feature calculate its correlation
        # the target. Otherwise calculate the residuals of the previous best
        # model with the newly added feature.
        if len(sel_feats) == 0:
            X, y = parse_data(data, [best_feature], target, log_labels)
            correlations.append(spearmanr(X.reshape(-1,), y))
            previous_model = models[k]
        else:
            X, y, z = parse_data(data, sel_feats, target, log_labels,
                                 best_feature)
            __, Xtest, __, ytest, __, ztest = train_test_split(
                X, y, z, test_size=test_size, random_state=seed)

            residuals = ytest - previous_model.predict(Xtest)
            correlations.append(spearmanr(residuals, ztest))
            # Get the previous best model
            previous_model = models[k]

        # Append the best selected feature and its score
        sel_feats.append(best_feature)
        best_scores.append(scores[k])

        if verbose:
            print("Iteration {}. Found ordering: {}, remaining: {}"
                  .format(len(sel_feats), sel_feats, features))

    return {"ordered_features": sel_feats,
            "scores": best_scores,
            "correlations": correlations}
