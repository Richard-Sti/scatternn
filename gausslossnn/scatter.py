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
import numpy
from scipy.stats import (uniform, norm)


class BinningAssignment:
    """
    A supplementary data model that handles assignment of samples to bins.

    Arguments
    ---------
    x : 1-dimensional array
        Samples' x-values to be binned.
    y : 1-dimensional array
        Samples' y-values to be binned.
    dx : float
        The x-axis bin spacing.
    """

    def __init__(self, x, y, dx):
        self._x = None
        self._y = None
        self._dx = None
        self._mask = None
        self._Nbins = None
        self.set_data(x, y)
        self.dx = dx

    @property
    def x(self):
        """`x` data."""
        return self._x

    @property
    def y(self):
        """`y` data."""
        return self._y

    def set_data(self, x, y):
        """Sets `x` and `y` and checks their formatting"""
        if not all(isinstance(d, numpy.ndarray) for d in (x, y)):
            raise TypeError("`x` and `y` must be 1-dimensional arrays.")
        if not all(d.ndim == 1 for d in (x, y)):
            raise TypeError("`x` and `y` must be 1-dimensional arrays.")
        if x.size != y.size:
            raise TypeError("`x` and `y` must be of equal size.")
        self._x = x
        self._y = y

    @property
    def dx(self):
        """The x-axis bin spacing."""
        return self._dx

    @dx.setter
    def dx(self, dx):
        """Store the spacing and assign samples to bins."""
        if not dx > 0:
            raise ValueError("`dx` must be positive.")

        bins = numpy.arange(self.x.min(), self.x.max(), dx)
        # Assign the samples to bins by their x values
        mask = numpy.full_like(self.x, numpy.nan, dtype=int)
        for i in range(bins.size - 1):
            if i == 0:
                mask[(bins[i] <= self.x) & (self.x <= bins[i+1])] = i
            else:
                mask[(bins[i] < self.x) & (self.x <= bins[i+1])] = i

        self._dx = dx
        self._mask = mask
        self._Nbins = numpy.unique(self._mask).size

    @property
    def Nbins(self):
        """The number of bins."""
        return self._Nbins

    def __call__(self, n):
        """
        The samples belonging to the `n`th bin.

        Arguments
        ---------
        n : int
            The bin index of interest.

        Returns
        -------
        x : 1-dimensional array
            The `n`th bin's samples' x-values.
        y : 1-dimensional array
            The `n`th bin's samples' y-values.
        """
        if n < 0:
            raise ValueError("`n` must be greater or equal than 0.")
        if n > self.Nbins - 1:
            raise ValueError("`{}`th bin does not exist. There are {} bins"
                             .format(n, self.Nbins))
        binmask = self._mask == n
        return self.x[binmask], self.y[binmask]


class BivariateGaussianScatterPosterior:
    """
    The bivariate Gaussian scatter posterior parametrised by some mean and
    standard deviation. Returns the log likelihood and prior for `bin_samples`
    to have a some mean and standard deviation.

    The prior for mean is taken to be uniformly distributed between the minimum
    and maximum value and the prior on standard deviation is taken to be
    uniformly distributed between 0 and the absolute value of the maximum value
    within the bin.

    Arguments
    ---------
    bin_samples : 1-dimensional array
        Samples' `y-values` whose mean and standard deviation we wish to
        measure.
    """

    def __init__(self, bin_samples):
        self._bin_samples = None
        self._bin_samples = bin_samples

        self.params = ["mu", "std"]
        bounds = {"mu": [numpy.min(bin_samples), numpy.max(bin_samples)],
                  "std": [0, numpy.abs(numpy.max(bin_samples))]}
        self.prior = {p: uniform(loc=bnd[0], scale=bnd[1] - bnd[0])
                      for p, bnd in bounds.items()}

    @property
    def bin_samples(self):
        """The samples falling within a bin."""
        return self._bin_samples

    @bin_samples.setter
    def bin_samples(self, bin_samples):
        """Sets bin samples, ensuring it's an array."""
        if not isinstance(bin_samples, (list, numpy.nadarray)):
            raise TypeError("`bin_samples` must be a 1-dimensional")
        if isinstance(bin_samples, list):
            bin_samples = numpy.asarray(bin_samples)
        if bin_samples.ndim > 1:
            raise TypeError("`bin_samples` must be a 1-dimensional")
        self._bin_samples = bin_samples

    def prior_rvs(self, size=None):
        """Samples the uniform prior distributions."""
        return {p: self.prior[p].rvs(size=size) for p in self.params}

    def log_prior(self, **kwargs):
        """The log prior."""
        return sum(self.prior[p].logpdf(kwargs[p]) for p in self.params)

    def log_likelihood(self, **kwargs):
        """The log likelihood."""
        return numpy.sum(norm(loc=kwargs["mu"],
                              scale=kwargs["std"]).logpdf(self.bin_samples))

    def __call__(self, **kwargs):
        """Calls the log likelihood and prior."""
        lp = self.log_prior(**kwargs)
        if not numpy.isfinite(lp):
            ll = None
        else:
            ll = self.log_likelihood(**kwargs)
        # Check if there are any remaining parameters.
        return ll, lp
