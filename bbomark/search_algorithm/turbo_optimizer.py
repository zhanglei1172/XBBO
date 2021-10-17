import glob
import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from matplotlib import pyplot as plt
from botorch.optim import optimize_acqf
import tqdm, random
from torch.quasirandom import SobolEngine
from bbomark.acquisition_function.ei import EI
from bbomark.configspace.feature_space import FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations

from bbomark.core.trials import Trials
from bbomark.surrogate.gaussian_process import GaussianProcessRegressor, GaussianProcessRegressorARD_gpy, \
    GaussianProcessRegressorARD_torch
from bbomark.surrogate.gp import train_gp

def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X

class TuRBO(AbstractOptimizer, FeatureSpace_uniform):
    '''
    reference: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_m.py
    '''

    def __init__(self,
                 config_spaces,
                 min_sample=8, # recommand 2*dim
                 num_tr=1,
                 max_cholesky_size=2000,
                 use_ard=True,
                 n_training_steps=50,
                 use_design=False
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.min_sample = min_sample
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.n_training_steps = n_training_steps

        self.trials = Trials()
        self.max_cholesky_size = max_cholesky_size
        self.n_cand = min(100 * self.dense_dimension, 5000)
        self.use_ard = use_ard
        self.num_tr = num_tr
        self.use_design = use_design
        self.succtol = 3
        self.n_evals = 0
        self.verbose = True
        self.candidates = []
        # Save the full history
        self.X = np.zeros((0, self.dense_dimension))
        self.fX = np.zeros((0, 1))
        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8
        self.hypers = [{} for _ in range(self.num_tr)]

        self._restart()


    def _restart(self):
        self._idx = np.zeros((0, 1), dtype=int)  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.num_tr, dtype=int)
        self.succcount = np.zeros(self.num_tr, dtype=int)
        self.length = self.length_init * np.ones(self.num_tr)

    def _create_candidates(self, X, fX, length, n_training_steps, hypers, batch_size):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (deepcopy(fX) - mu) / sigma

        # Figure out what device we are running on
        # if len(X) < self.min_cuda:
        #     device, dtype = torch.device("cpu"), torch.float64
        # else:
        #     device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X)  # .to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX.ravel())  # .to(device=device, dtype=dtype)
            gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dense_dimension, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dense_dimension, 1.0)
        mask = np.random.rand(self.n_cand, self.dense_dimension) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dense_dimension - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dense_dimension))
        X_cand[mask] = pert[mask]

        # # Figure out what device we are running on
        # if len(X_cand) < self.min_cuda:
        #     device, dtype = torch.device("cpu"), torch.float64
        # else:
        #     device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        # gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand)  # .to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(torch.Size([batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def suggest(self, n_suggestions=1):
        if not hasattr(self, 'failtol'):
            self.failtol = np.ceil(np.max([4.0 / n_suggestions, self.dense_dimension / n_suggestions]))

        for m in range(self.num_tr):
            if (self._idx == m).sum() < self.min_sample:
                # raise NotImplemented
                self._idx = np.vstack((self._idx, np.full((n_suggestions,1), m, dtype=int)))
                return self._random_suggest(n_suggestions)

        x_unwarpeds = []
        sas = []
        X_cand = np.zeros((self.num_tr, self.n_cand, self.dense_dimension))
        y_cand = np.inf * np.ones((self.num_tr, self.n_cand, n_suggestions))
        for m in range(self.num_tr):
            idx = np.where(self._idx == m)[0]
            X = self.X[idx]
            Y = self.fX[idx]
            X_cand[m, :, :], y_cand[m, :, :], self.hypers[m] = self._create_candidates(X, Y, self.length[m],
                                                                                       self.n_training_steps,
                                                                                       self.hypers[m],
                                                                                       n_suggestions)
        X_next = np.zeros((n_suggestions, self.dense_dimension))
        idx_next = np.zeros((n_suggestions, 1), dtype=int)
        for b in range(n_suggestions):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, b]), (self.num_tr, self.n_cand))
            X_next[b, :] = deepcopy(X_cand[i, j, :])
            idx_next[b, 0] = i
            assert np.isfinite(y_cand[i, j, b])  # Just to make sure we never select nan or inf
            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf
            x_array = self.feature_to_array(X_next[b, :], self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            x_unwarpeds.append(x_unwarped)
            sas.append(X_next[b, :])
        self._idx = np.vstack((self._idx, idx_next))
        self.trials.params_history.extend(x_unwarpeds)
        return x_unwarpeds, sas

    def observe(self, x, y):
        # print(y)
        batch_num = len(x)
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += batch_num
        # self.history_x = np.array(self.trials.history)
        # self.history_y = np.array(self.trials.history_y)
        X_next = np.array(x)
        fX_next = np.atleast_2d(y).T
        # Update trust regions
        for i in range(self.num_tr):
            if (self._idx == i).sum() <= self.min_sample:
                continue # don't update
            idx_i = np.where(self._idx[-batch_num:] == i)[0]
            if len(idx_i) > 0:
                self.hypers[i] = {}  # Remove model hypers
                fX_i = fX_next[idx_i]

                if self.verbose and (fX_i.min() < self.fX.min() - 1e-3 * math.fabs(
                        self.fX.min())):
                    n_evals, fbest = self.n_evals, fX_i.min()
                    print(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                    sys.stdout.flush()
                self._adjust_length(fX_i, i)

        # Update budget and append data
        self.n_evals += batch_num
        self.X = np.vstack((self.X, deepcopy(X_next)))
        self.fX = np.vstack((self.fX, deepcopy(fX_next)))
        # self._idx = np.vstack((self._idx, deepcopy(idx_next)))

        # Check if any TR needs to be restarted
        for i in range(self.num_tr):
            if self.length[i] < self.length_min:  # Restart trust region if converged
                idx_i = self._idx[:, 0] == i

                if self.verbose:
                    n_evals, fbest = self.n_evals, self.fX[idx_i, 0].min()
                    print(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")
                    sys.stdout.flush()

                # Reset length and counters, remove old data from trust region
                self.length[i] = self.length_init
                self.succcount[i] = 0
                self.failcount[i] = 0
                self._idx[idx_i, 0] = -1  # Remove points from trust region
                self.hypers[i] = {}  # Remove model hypers


    def _adjust_length(self, fX_next, i):
        assert i >= 0 and i <= self.num_tr - 1

        fX_min = self.fX[self._idx[:len(self.fX), 0] == i].min()  # Target value
        if fX_next.min() < fX_min - 1e-3 * math.fabs(fX_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0


    def _random_suggest(self, n_suggestions=1):
        sas = []
        x_unwarpeds = []
        if self.use_design:
            for n in range(n_suggestions):
                if len(self.candidates) == 0:
                    self.candidates = latin_hypercube(self.min_sample, self.dense_dimension)
                rm_id = np.random.randint(low=0, high=len(self.candidates))
                sas.append(self.candidates[rm_id])
                x_array = self.feature_to_array(sas[-1], self.sparse_dimension)
                x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
                x_unwarpeds.append(x_unwarped)
                self.candidates = np.delete(self.candidates, rm_id, axis=0)  # TODO
        else:
            x_unwarpeds = (self.space.sample_configuration(n_suggestions))
            for n in range(n_suggestions):
                array = Configurations.dictUnwarped_to_array(self.space, x_unwarpeds[n])
                sas.append(self.array_to_feature(array, self.dense_dimension))
        return x_unwarpeds, sas


opt_class = TuRBO
