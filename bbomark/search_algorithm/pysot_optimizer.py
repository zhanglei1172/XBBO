# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from copy import copy

import numpy as np
from poap.strategy import EvalRecord
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant

from bbomark.configspace.space import Configurations
from bbomark.core.feature_space import FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
'''
全【0，1】
'''
class PySOTOptimizer(AbstractOptimizer, FeatureSpace_uniform):
    primary_import = "pysot"

    def __init__(self, config_spaces):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self)
        self.dtypes_idx_map = self.space.dtypes_idx_map
        self.opt_name = 'pysot'
        # self.space_x = JointSpace(api_config)
        self.bounds = self.space.get_bounds()
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.create_opt_prob()  # Sets up the optimization problem (needs self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.batch_size = None
        self.history = []
        self.proposals = []

    def transform_sparseArray_to_optSpace(self, sparse_array):
        return [self.array_to_feature(x, self.dense_dimension) for x in sparse_array]

    def create_opt_prob(self):
        """Create an optimization problem object."""
        opt = OptimizationProblem()
        opt.lb = self.bounds.lb  # In warped space
        opt.ub = self.bounds.ub  # In warped space
        opt.dim = len(opt.lb)
        opt.cont_var = np.arange(opt.dim)
        opt.int_var = []
        assert len(opt.cont_var) + len(opt.int_var) == opt.dim
        opt.objfun = None
        self.opt = opt

    def start(self):
        """Starts a new pySOT run."""
        self.history = []
        self.proposals = []

        # Symmetric Latin hypercube design
        des_pts = max([self.batch_size, 2 * (self.opt.dim + 1)])
        slhd = SymmetricLatinHypercube(dim=self.opt.dim, num_pts=des_pts)

        # Warped RBF interpolant
        rbf = RBFInterpolant(
            dim=self.opt.dim,
            lb=self.opt.lb,
            ub=self.opt.ub,
            kernel=CubicKernel(),
            tail=LinearTail(self.opt.dim),
            eta=1e-4,
        )

        # Optimization strategy
        self.strategy = SRBFStrategy(
            max_evals=self.max_evals,
            opt_prob=self.opt,
            exp_design=slhd,
            surrogate=rbf,
            asynchronous=True,
            batch_size=1,
            use_restarts=True,
        )

    def suggest(self, n_suggestions=1):
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """

        if self.batch_size is None:  # First call to suggest
            self.batch_size = n_suggestions
            self.start()

        # Set the tolerances pretending like we are running batch
        d, p = float(self.opt.dim), float(n_suggestions)
        self.strategy.failtol = p * int(max(np.ceil(d / p), np.ceil(4 / p)))

        # Now we can make suggestions
        x_w = []
        features = []
        self.proposals = []
        for _ in range(n_suggestions):
            proposal = self.strategy.propose_action()
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()  # This triggers all the callbacks

            # It is possible that pySOT proposes a previously evaluated point
            # when all variables are integers, so we just abort in this case
            # since we have likely converged anyway. See PySOT issue #30.
            x_guess_data = (proposal.record.params)[0]  # From tuple to list
            x_array = self.feature_to_array(x_guess_data, self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            if x_unwarped in self.history:
                warnings.warn("pySOT proposed the same point twice")
                self.start()
                return self.suggest(n_suggestions=n_suggestions)

            # NOTE: Append unwarped to avoid rounding issues
            self.history.append(copy(x_unwarped))
            self.proposals.append(proposal)
            x_w.append(copy(x_unwarped))
            features.append(proposal)

        return x_w, features

    def _observe(self, feature, y):
        # Find the matching proposal and execute its callbacks
        idx = [feature == xx for xx in self.proposals]
        i = np.argwhere(idx)[0].item()  # Pick the first index if there are ties
        proposal = self.proposals[i]
        proposal.record.complete(y)
        self.proposals.pop(i)
        self.history.pop(i)

    def observe(self, features, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(features) == len(y)

        for x_, y_ in zip(features, y):
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(y_):
                self._observe(x_, y_)


opt_class = PySOTOptimizer
# feature_space = FeatureSpace_uniform