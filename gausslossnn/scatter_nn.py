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

import numpy
from copy import deepcopy
from warnings import warn
import os
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import (Input, Normalization, Dense, Add)
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import neural_structured_learning as nsl
from sklearn.metrics import r2_score

import joblib


#
# =============================================================================
#
#                           Gaussian loss NN
#
# =============================================================================
#


class GaussianLossNN:
    """
    An adversarial neural network 1-dimensional regressor with a Gaussian
    loss function that predicts both a mean and a standard deviation.

    Connects the weights to outputs linearly and through a deep NN whose
    outputs are summed, the wide & deep architecture.

    Recommended to use the default scaled exponential linear unit activation
    (SELU) and the Lecun Normal weights initialiser.

    Arguments
    ---------
    Ninputs: int
        Number of input features.
    checkpoint_dir: str
        Path to the directory with the checkpoint files `params.p` and
        `cp.ckpt`.
    deep_layers: list of int, optional
        Number of neurons within each deep layer.
        By default `[16, 16, 16, 16, 8]`.
    activation: str, optional
        Activation function. By default `selu`, the scaled exponential linear
        unit.
    initializer: str, optional
        Network weights initialiser, by default `LecunNormal`. Alternatively
        can be picked from Tensorflow's selection of initialisers.
    adv_multiplier: float, optinal
        Multiplier to adversarial regularization loss. By default 0.2.
    adv_step: float, optional
        Step size to find the adversarial sample. By default to 0.001.
    pgd_iters: int, optional
        Nnumber of attack iterations for Projected Gradient Descent (PGD)
        attack. Defaults to 3.
    seed: int, optional
        Random seed for setting the initial weights.
    """
    def __init__(self, Ninputs, checkpoint_dir,
                 deep_layers=[16, 16, 16, 16, 8], activation="selu",
                 initializer="LecunNormal", adv_multiplier=0.2, adv_step=0.001,
                 pgd_iters=3, seed=None):
        # Initialise the model
        self.model, self.adv_model = self._make_model(
            Ninputs, deep_layers, activation, initializer, adv_multiplier,
            adv_step, pgd_iters, seed)

        if not os.path.isdir(checkpoint_dir):
            raise ValueError("Invalid `checkpoint_dir` `{}`"
                             .format(checkpoint_dir))
        self.checkpoint_dir = checkpoint_dir

        self._params = {"Ninputs": Ninputs, "deep_layers": deep_layers,
                        "activation": activation, "initializer": initializer,
                        "adv_multiplier": adv_multiplier, "adv_step": adv_step,
                        "pgd_iters": pgd_iters, "seed": seed}

    def _make_model(self, Ninputs, deep_layers, activation, initializer,
                    adv_multiplier, adv_step, pgd_iters, seed):
        """Make the (adversarial) model."""
        # Weights initialiser
        if initializer == "LecunNormal":
            inits = tf.keras.initializers.LecunNormal(seed)
        elif not isinstance(initializer, str):
            inits = initializer(seed=seed)
        else:
            inits = initializer

        input_shape = (Ninputs, )

        # Linear part, directly connect to the output with no activation
        linear_input = Input(shape=input_shape, name="linear_input")
        linear = Dense(2, name="linear_layer")(linear_input)

        # Deep part, densely connected layers
        deep_input = Input(shape=input_shape, name="deep_input")
        # Normalising layer
        deep = Normalization()(deep_input)
        # Append the deep layers
        for i, layer in enumerate(deep_layers):
            deep = Dense(layer, activation=activation,
                         kernel_initializer=inits,
                         name="deep_{}".format(i + 1))(deep)
        # Need two output nodes: mean and variance
        deep = Dense(2, name="deep_final")(deep)

        # Connect wide and deep
        final_layer = Add(name="add_linear_deep")([linear, deep])
        # The distribution at the end of the NN. Softplus transform the std to
        # ensure positivity.
        lambda_dist = lambda t: tfd.Normal(
            loc=t[..., :1], scale=1e-6 + tf.math.softplus(0.1 * t[..., 1:]))
        # Append the distribution layer
        final_layer = tfp.layers.DistributionLambda(lambda_dist)(final_layer)

        # Generate the model
        model = tf.keras.models.Model(inputs=[linear_input, deep_input],
                                      outputs=final_layer)
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=adv_multiplier, adv_step_size=adv_step,
            pgd_iterations=pgd_iters)
        adv_model = nsl.keras.AdversarialRegularization(
            model, adv_config=adv_config)

        return model, adv_model

    @staticmethod
    def _hamiltonian_loss(x, dist):
        """The Hamiltonian (negative log likelihood) loss function."""
        return -dist.log_prob(x)

    def get_callbacks(self, patience):
        """
        Get the early stopping and checkpointing callbacks. Restores the best
        weights.

        Arguments
        ---------
        patience: int
            The patience, if the loss minimum does not change over this many
            epochs terminate the training.

        Returns
        -------
        cbs: list of callbacks
            The early stopping and model checkpoint callbacks.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "cp.ckpt")
        return [tf.keras.callbacks.EarlyStopping(
                    patience=patience, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path, save_weights_only=True,
                    verbose=0)
                ]

    def fit(self, Xtrain, ytrain, batch_size, optimizer="adamax", patience=50,
            epochs=500, validation_size=0.2):
        """
        Fit the NN with the given optimizer and save its training history and
        weights into the checkpoint folder.

        Arguments
        ---------
        Xtrain: 2-dimensional array
            Feature array.
        ytrain: 1-dimensional array
            Target array.
        batch_size: int
            The batch size.
        optimizer: keras optimizer, optional
            Optimizer to train the network. By default `adamax` with default TF
            parameters.
        patience: int, optional
            The patience, if the loss minimum does not change over this many
            epochs terminate the training. By default 50.
        epochs: int, optional
            Number of epochs to train the network, by defualt 500.
        validation_size: float, optional
            Fractional validation size.

        Returns
        -------
        None
        """
        # Compile the model
        self.adv_model.compile(optimizer=optimizer,
                               loss=self._hamiltonian_loss)
        # Data in a format to be given to the NN
        data = {"linear_input": Xtrain,
                "deep_input": Xtrain,
                "label": ytrain}

        callbacks = self.get_callbacks(patience)

        history = self.adv_model.fit(
            x=data, batch_size=batch_size, callbacks=callbacks, verbose=0,
            epochs=epochs, validation_split=validation_size)

        # Save the history and the params for reproducibility
        joblib.dump(history.history,
                    os.path.join(self.checkpoint_dir, 'history.p'))
        joblib.dump(self._params,
                    os.path.join(self.checkpoint_dir, 'params.p'))

    def predict(self, X, full=False):
        """
        Predict the mean and standard deviations for samples `X` (if `full`).

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        full: bool, optional
            Whether to also return the standard deviation.

        Returns
        -------
        out: n-dimensional array
            Predictions. If `full=False` returns a 1-dimensional array of
            means, otherwise a 2-dimensional array of shape (`Nsamples`, 2)
            where the 2nd column represent the mean and std, respectively.
        """
        yhat = self.model({"linear_input": X, "deep_input": X})
        mu = numpy.asarray(yhat.mean()).reshape(-1,)

        if full:
            std = numpy.asarray(yhat.stddev()).reshape(-1,)
            return numpy.vstack([mu, std]).T

        return mu

    def score_R2mean(self, X, y):
        r"""
        Calculate the :math:`R^2` score of mean predictions defined as

        .. math::
            R^2 = 1 - \frac{\sum_n (\mu_n - y_n)^2}{\sum_n (\mu_n - \hat{y})^2}

        where :math:`\mu_n, y_n` are the predicted mean and true values,
        respectively, of the :math:`n`th sample. :math:`\hat{y}` is the average
        of the true values.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.

        Returns
        -------
        R2: float
            The R2 score.
        """
        return r2_score(self.predict(X), y)

    def score_reduced_chi2(self, X, y):
        r"""
        Calculate the reduced :math:`\chi^2` score defined as

        .. math::
            \chi^2 = \frac{1}{N-2} \sum_{n} \frac{(\mu_n - y_n)^2}{\sigma_n^2}

        where :math:`\mu_n, \sigma_n, y_n` are the :math:`n`th predicted mean
        value, predicted uncertainty and true value, respectively. Lastly,
        :math:`N` is the number of samples.

        Values of :math:`\chi^2 \gg 1` indicates that the error variance is
        underestimated and :math:`\chi^2 < 1` indicates the error variance is
        overestimated.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.

        Returns
        -------
        chi2 : float
            The reduced :math:`\chi^2` value.
        """
        stats = self.predict(X, full=True)
        if y.ndim > 1 and y.shape[1] > 1:
            raise TypeError("`y` must be a 1D array.")
        else:
            y = y.reshape(-1,)

        return numpy.sum((stats[:, 0] - y)**2 / stats[:, 1]**2) / (y.size - 2)

    def predict_gradient(self, X):
        """
        Predict the gradient of the predictions with respect to the input
        features.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.

        Returns
        -------
        grad: 3-dimensional array
            Array of gradients of shape (2, `Nsamples`, `Nfeatures`). The first
            axis correspond to the gradient of the mean and standard deviation,
            respectively.
        """
        X = tf.convert_to_tensor(X)
        x_input = {"linear_input": X, "deep_input": X}
        # We will need separate tapes for mu and std
        with tf.GradientTape() as t_mu:
            t_mu.watch(x_input)
            mu_pred = self.model(x_input).mean()

        with tf.GradientTape() as t_std:
            t_std.watch(x_input)
            std_pred = self.model(x_input).stddev()

        # The linear and deep input gradients are the same so might as well
        # take this.
        mu_grad = t_mu.gradient(mu_pred, x_input)["linear_input"]
        std_grad = t_std.gradient(std_pred, x_input)["linear_input"]

        return numpy.stack([numpy.asarray(mu_grad), numpy.asarray(std_grad)])

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, optimizer):
        """
        Initialise from a checkpoint.

        Arguments
        ---------
        checkpoint_dir: str
            Path to the directory with the checkpoint files `params.p` and
            `cp.ckpt`.
        optimizer: keras optimizer
            Optimizer to train the network.

        Returns
        -------
        network: :py:class:`GaussianLossNN`
            The initialised model with loaded weights.
        """
        params = joblib.load(os.path.join(checkpoint_dir, "params.p"))
        network = cls(**params, checkpoint_dir=checkpoint_dir)
        network.adv_model.compile(optimizer=optimizer,
                                  loss=network._hamiltonian_loss)
        checkpoint_path = os.path.join(checkpoint_dir, "cp.ckpt")
        network.adv_model.load_weights(checkpoint_path)
        return network

    @classmethod
    def fit_directly(cls, Xtrain, ytrain, batch_size, checkpoint_dir, seed,
                     model_kwargs={}, optimizer="adamax", patience=50,
                     epochs=500, validation_size=0.2):
        """
        Initialise the model and directly fit it.

        Arguments
        ---------
        Xtrain: 2-dimensional array.
            Feature array.
        ytrain: 1-dimensional array.
            Target  batch_size: int
        batch_size: int
            The batch size.
        checkpoint_dir: str
            Path to the directory with the checkpoint files `params.p` and
            `cp.ckpt`.
        seed: int, optional
            Random seed for setting the initial weights.
        model_kwargs: dict
            Kwargs passed into :py:class:`GaussianLossNN`, except `Ninputs` and
            `checkpoint_dir`.
        optimizer: keras optimizer, optional
            Optimizer to train the network. By default `adamax` with default TF
            parameters.
        patience: int, optional
            The patience, if the loss minimum does not change over this many
            epochs terminate the training. By default 50.
        epochs: int, optional
            Number of epochs to train the network, by defualt 500.
        validation_size: float, optional
            Fractional validation size. array.

        Returns
        -------
        network: :py:class:`GaussianLossNN`
            The trained model.
        """
        # Deepcopy the kwargs
        model_kwargs = deepcopy(model_kwargs)

        # Do some input checking
        if model_kwargs.pop("Ninputs", None) is not None:
            warn("`Ninputs` inferred implicitly from `Xtrain`. "
                 "Ignoring the value in `model_kwargs`.")

        for par in ["checkpoint_dir", "seed"]:
            if model_kwargs.pop(par, None) is not None:
                warn("`{}` must be specified outside `model_kwargs`. "
                     "Ignoring the value in `model_kwargs`.".format(par))

        # Initiliase the model
        Ninputs = Xtrain.shape[1]
        network = cls(Ninputs, checkpoint_dir, **model_kwargs)
        # Fit it
        network.fit(Xtrain, ytrain, batch_size, optimizer, patience, epochs,
                    validation_size)
        return network


#
# =============================================================================
#
#                       Gaussian loss NN utilities
#
# =============================================================================
#


def make_checkpoint_dirs(base_path, Nensemble):
    """
    Create directories `ensemble_n` for checkpointing the models within the
    ensemble.

    Arguments
    ---------
    base_path: str
        Base checkpoint directory within which to create the subdirectories.
    Nensemble: int
        Size of the ensemble.

    Returns
    -------
    cdirs: list of str
        List of ensemble checkpointing subdirectories.
    """
    cdirs = [None] * Nensemble
    for i in range(Nensemble):
        cdir = os.path.join(base_path, "ensemble_{}".format(i))

        if not os.path.isdir(cdir):
            os.mkdir(cdir)
        cdirs[i] = cdir
    return cdirs


def get_random_seeds(N, seed):
    """
    Get a set of random seeds (unique integers).

    Arguments
    ---------
    N: int
        Number of seeds to get.
    seed: int
        Initial random seed.

    Returns
    -------
    seeds: list of int
        List of unique seeds.
    """
    rng = numpy.random.default_rng(seed)
    seeds = []
    while len(seeds) < N:
        rvs = rng.integers(0, 2**32)
        if rvs not in seeds:
            seeds.append(rvs)

    return seeds


class SummaryEnsembleGaussianLossNN:
    """
    Summary class for an ensemble of :py:class:`GaussianLossNN` models.

    Contains functionalities for checkings their convergence, ensemble
    predictions, ensemble scores and ensemble gradients of the predictions
    with respect to the features.
    """
    def __init__(self, base_checkpoint_dir, optimizer):
        self._converge_mask = None
        cdirs = glob(os.path.join(base_checkpoint_dir, "ensemble_*"))

        print("Found {} models in `{}`."
              .format(len(cdirs), base_checkpoint_dir))
        self._models = [GaussianLossNN.from_checkpoint(cdir, optimizer)
                        for cdir in cdirs]

    @property
    def Nensemble(self):
        """
        The ensemble size without models that considered not converged, if any.

        Returns
        -------
        Nensemble: int
            The ensemble size.
        """
        return numpy.sum(self.convergence_mask)

    @property
    def convergence_mask(self):
        """
        The boolean mask of converged models.

        Returns
        -------
        convergence_mask: 1-dimensional array of bools
            Convergence mask.
        """
        if self._converge_mask is None:
            return numpy.ones(len(self._models), dtype=bool)
        return self._converge_mask

    @property
    def models(self):
        """
        Return the list of models (that are considered converged.)

        Returns
        -------
        models: list
            List of models.
        """
        return [model for i, model in enumerate(self._models)
                if self.convergence_mask[i]]

    def predict(self, X, full=False):
        """
        Predict the mean and standard deviations for samples `X` (if `full`).

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        full: bool, optional
            Whether to also return the standard deviation.

        Returns
        -------
        out: n-dimensional array
            Predictions. If `full=False` returns a 1-dimensional array of
            means, otherwise a 2-dimensional array of shape (`Nsamples`, 2)
            where the 2nd column represent the mean and std, respectively.
        """
        N = X.shape[0]
        if full:
            shape = (self.Nensemble, N, 2)
        else:
            shape = (self.Nensemble, N)

        pred = numpy.full(shape, numpy.nan)

        for i, model in enumerate(self.models):
            pred[i, ...] = model.predict(X, full)

        return pred

    def predict_summary(self, X, full=False, bootstrap=False):
        """
        Predict the ensemble summary statistics.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        full: bool, optional
            Whether to also return the standard deviation, by default `False`.
        bootstrap: bool, optional
            Whether to also return the bootstrapped mean and standard
            deviation across the ensemble. By default `False`.

        Returns
        -------
        out : n-dimensional array
            Array of summary statistics.

                If `full=False` and `bootstrap=False` returns ensemble mean of
                means of shape (`Nsamples`, ).

                If `full=False` and `bootstrap=True` returns ensemble mean of
                means and its bootstrap of shape (`Nsamples`, 2).

                If `full=True` and `bootstrap=False` returns ensemble mean of
                means and ensemble averaged standard deviation of shape
                (`Nsamples`, 2).

                If `full=True` and `bootstrap=True` returns ensemble mean of
                means, the ensemble averaged standard deviation and their
                corresponding bootstraps in the last axis. The shape is
                (`Nsamples`, 2, 2).
        """
        pred = self.predict(X, True)

        mus = numpy.mean(pred[..., 0], axis=0)
        stds = numpy.mean(
            pred[..., 1]**2 + (pred[..., 0] - mus)**2, axis=0)**0.5

        mus_bootstrap = numpy.std(pred[..., 0], axis=0)
        stds_bootstrap = numpy.mean((pred[..., 1] - stds)**2, axis=0)**0.5

        out = numpy.full((mus.size, 2, 2), numpy.nan)
        out[:, 0, :] = numpy.vstack([mus, mus_bootstrap]).T
        out[:, 1, :] = numpy.vstack([stds, stds_bootstrap]).T

        if full and bootstrap:
            return out
        elif full and not bootstrap:
            return out[:, :, 0]
        elif not full and bootstrap:
            return out[:, 0, :]
        else:
            return out[:, 0, 0]

    def predict_gradient(self, X):
        """
        Predict gradient of the predicted mean and standard deviation with
        respect to the input features.


        Arguments
        ---------
        X: 2-dimensional array
            Feature array.

        Returns
        -------
        out: 4-dimensional array
            Array of gradients of shape (`Nensemble`, 2, `Nsamples`,
            `Nfeatures`). The second axis represents the mean and standard
            deviation, respectively.
        """
        out = numpy.full((self.Nensemble, 2, *X.shape), numpy.nan)
        for i, model in enumerate(self.models):
            out[i, ...] = model.predict_gradient(X)
        return out

    def predict_gradient_summary(self, X, bootstrap=False):
        """
        Predict the summary statistic across the ensemble of the gradient of
        the predicted mean and standard deviation.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.

        Returns
        -------
        out: 3- or 4-dimensional array
            Array of gradients.

                If `bootstrap=False` of shape (2, `Nsamples`,
                `Nfeatures`), where the first axis represents the gradient of
                the mean and standard deviation.

                Otherwise the output shape is (2, 2, `Nsamples`, `Nfeatures`),
                where the second axis of length 2 represents the bootstrap
                standard deviations.
        """
        pred = self.predict_gradient(X)
        mean_grad = numpy.mean(pred, axis=0)

        if bootstrap:
            out = numpy.full((2, 2, *X.shape), numpy.nan)
            out[:, 0, ...] = mean_grad
            out[:, 1, ...] = numpy.std(pred, axis=0)
            return out

        return mean_grad

    def score_R2mean(self, X, y, full=True):
        r"""
        Calculate for each model the :math:`R^2` score of mean predictions
        defined as

        .. math::
            R^2 = 1 - \frac{\sum_n (\mu_n - y_n)^2}{\sum_n (\mu_n - \hat{y})^2}

        where :math:`\mu_n, y_n` are the predicted mean and true values,
        respectively, of the :math:`n`th sample. :math:`\hat{y}` is the average
        of the true values.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.
        full: bool, optional
            Whether to calculate the score from the ensemble summary mean.

        Returns
        -------
        R2: list of floats or a float
            The R2 score(s).
        """
        if full:
            return [model.score_R2mean(X, y) for model in self.models]

        return r2_score(self.predict_summary(X), y)

    def score_reduced_chi2(self, X, y, full=True):
        r"""
        Calculate for each model the reduced :math:`\chi^2` score defined as

        .. math::
            \chi^2 = \frac{1}{N-2} \sum_{n} \frac{(\mu_n - y_n)^2}{\sigma_n^2}

        where :math:`\mu_n, \sigma_n, y_n` are the :math:`n`th predicted mean
        value, predicted uncertainty and true value, respectively. Lastly,
        :math:`N` is the number of samples.

        Values of :math:`\chi^2 \gg 1` indicates that the error variance is
        underestimated and :math:`\chi^2 < 1` indicates the error variance is
        overestimated.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.
        full: bool, optional
            Whether to calculate the score from the ensemble summary.

        Returns
        -------
        chi2 : float
            The reduced :math:`\chi^2` value.
        """
        if full:
            return [model.score_reduced_chi2(X, y) for model in self.models]

        stats = self.predict_summary(X, full=True)
        if y.ndim > 1 and y.shape[1] > 1:
            raise TypeError("`y` must be a 1D array.")
        else:
            y = y.reshape(-1,)

        return numpy.sum((stats[:, 0] - y)**2 / stats[:, 1]**2) / (y.size - 2)

    def enforce_R2s_convergence(self, X, y, minR2):
        """
        Calculate a mask of which models resulted in :math:`R^2` higher than
        `minR2` and store it internally.

        Arguments
        ---------
        X: 2-dimensional array
            Feature array.
        y: 1-dimensional array
            Target array.
        minR2: float
            Minimum :math:`R^2` value.
        """
        mask = numpy.asarray(self.score_R2mean(X, y)) > minR2
        warn("Masking {} models whose R2 score is below threshold {}."
             .format(numpy.sum(~mask), minR2))
        self._convergence_mask = mask
