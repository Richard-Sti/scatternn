# Copyright (C) 2020  Richard Stiskalek
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

__version__ = "0.1.3"
__author__ = "Richard Stiskalek"


from .scatter import (BinningAssignment, BivariateGaussianScatterPosterior)
from .R2ordering import incremental_importance, parse_data
from .preprocess import DataFrameSelector, stratify_split, apply_preprocess

from .scatter_nn import (GaussianLossNN, make_checkpoint_dirs,
                         SummaryEnsembleGaussianLossNN, get_random_seeds)
