import logging
import math
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace

from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design
# from xbbo.surrogate.gaussian_process import GaussianProcessRegressor, GaussianProcessRegressorARD_gpy, \
#     GaussianProcessRegressorARD_torch
from xbbo.utils.constants import MAXINT
from xbbo.surrogate.gp import train_gp
from . import alg_register

logger = logging.getLogger(__name__)


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

@alg_register.register('turbo')
class TuRBO(AbstractOptimizer):
    '''
    reference: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_m.py
    '''
    def __init__(
            self,
            space: DenseConfigurationSpace,
            seed: int = 42,
            initial_design: str = 'sobol',
            num_tr=1,
            #  total_limit: int = 10,
            **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.n_min_sample = kwargs.get('n_min_sample', 5)
        self.init_budget = self.n_min_sample  # self.initial_design.init_budget
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, init_budget=self.init_budget)
        self.initial_design_configs = [
            self.initial_design.select_configurations() for _ in range(num_tr)
        ]
        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)
        self.n_training_steps = kwargs.get("n_training_steps", 50)
        self.max_cholesky_size = kwargs.get("max_cholesky_size", 2000)
        self.n_candidates = min(100 * self.dense_dimension, 5000)
        self.use_ard = kwargs.get("use_ard", True)
        self.num_tr = num_tr
        self.succtol = 3
        self.n_evals = 0
        self.candidates = []
        # Save the full history
        self.X = np.zeros((0, self.dense_dimension))
        self.fX = np.zeros((0, 1))
        # Trust region sizes
        self.length_min = 0.5**7
        self.length_max = 1.6
        self.length_init = 0.8
        self.hypers = [{} for _ in range(self.num_tr)]

        self._restart()

    def _restart(self):
        self._idx = np.zeros(
            (0, 1), dtype=int
        )  # Track what trust region proposed what using an index vector
        self.failcount = np.zeros(self.num_tr, dtype=int)
        self.succcount = np.zeros(self.num_tr, dtype=int)
        self.length = self.length_init * np.ones(self.num_tr)

    def _create_candidates(self, X, fX, length, n_training_steps, hypers,
                           batch_size):
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
            y_torch = torch.tensor(
                fX.ravel())  # .to(device=device, dtype=dtype)
            gp = train_gp(train_x=X_torch,
                          train_y=y_torch,
                          use_ard=self.use_ard,
                          num_steps=n_training_steps,
                          hypers=hypers)

            # Save state dict
            hypers = gp.state_dict()

        # Create the trust region boundaries
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy(
        ).ravel()
        weights = weights / weights.mean(
        )  # This will make the next line more stable
        weights = weights / np.prod(np.power(
            weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = self.rng.randint(MAXINT)
        sobol = SobolEngine(self.dense_dimension, scramble=True, seed=seed)
        pert = sobol.draw(self.n_candidates).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dense_dimension, 1.0)
        mask = self.rng.rand(self.n_candidates,
                             self.dense_dimension) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind,
             self.rng.randint(0, self.dense_dimension - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones(
            (self.n_candidates, self.dense_dimension))
        X_cand[mask] = pert[mask]

        # # Figure out what device we are running on
        # if len(X_cand) < self.min_cuda:
        #     device, dtype = torch.device("cpu"), torch.float64
        # else:
        #     device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        # gp = gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(
                self.max_cholesky_size):
            X_cand_torch = torch.tensor(
                X_cand)  # .to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(
                torch.Size([batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch, gp

        # De-standardize the sampled values
        y_cand = mu + sigma * y_cand

        return X_cand, y_cand, hypers

    def suggest(self, n_suggestions=1):
        if not hasattr(self, 'failtol'):
            self.failtol = np.ceil(
                np.max([
                    4.0 / n_suggestions, self.dense_dimension / n_suggestions
                ]))

        for m in range(self.num_tr):
            if (self._idx == m).sum() < self.n_min_sample:
                # raise NotImplemented
                self._idx = np.vstack(
                    (self._idx, np.full((n_suggestions, 1), m, dtype=int)))
                return self._random_suggest(n_suggestions, m)

        X_cand = np.zeros(
            (self.num_tr, self.n_candidates, self.dense_dimension))
        y_cand = np.inf * np.ones(
            (self.num_tr, self.n_candidates, n_suggestions))
        for m in range(self.num_tr):
            indicator = np.where(self._idx == m)[0]
            X = np.asarray(self.trials.get_dense_array())[indicator]
            Y = np.asarray(self.trials._his_observe_value)[indicator]
            X_cand[m, :, :], y_cand[
                m, :, :], self.hypers[m] = self._create_candidates(
                    X, Y, self.length[m], self.n_training_steps,
                    self.hypers[m], n_suggestions)
        X_next = np.zeros((n_suggestions, self.dense_dimension))
        idx_next = np.zeros((n_suggestions, 1), dtype=int)
        trial_list = []
        for b in range(n_suggestions):
            i, j = np.unravel_index(np.argmin(y_cand[:, :, b]),
                                    (self.num_tr, self.n_candidates))
            idx_next[b, 0] = i
            X_next[b, :] = deepcopy(X_cand[i, j, :])
            assert np.isfinite(
                y_cand[i, j,
                       b])  # Just to make sure we never select nan or inf
            # Make sure we never pick this point again
            y_cand[i, j, :] = np.inf
            config = DenseConfiguration.from_dense_array(
                self.space, X_next[b, :])
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      dense_array=X_next[b, :],
                      origin='TuRBO-region-{}'.format(i),
                      region=i))
        self._idx = np.vstack((self._idx, idx_next))
        return trial_list

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)
        # Update trust regions
        f_min = min([np.inf] +
                    self.trials._his_observe_value[:-len(trial_list)])
        for i in range(self.num_tr):
            # indicator = [
            #     trial.region == i for trial in self.trials.traj_history
            # ]
            if (self._idx == i).sum() <= self.n_min_sample:
                continue  # don't update
            new_trial_idxs = np.where(self._idx[-len(trial_list):] == i)[0]
            if len(new_trial_idxs) > 0:
                self.hypers[i] = {}  # Remove model hypers
                next_fX_i = []
                for new_trial_idx in new_trial_idxs:
                    next_fX_i.append(trial_list[new_trial_idx].observe_value)

                if (min(next_fX_i) < f_min - 1e-3 * math.fabs(f_min)):
                    n_evals, fbest = self.trials.trials_num, f_min
                    logger.info(f"{n_evals}) New best @ TR-{i}: {fbest:.4}")
                self._adjust_length(
                    next_fX_i, i,
                    min([
                        trial.observe_value for trial in
                        self.trials.traj_history[:-len(trial_list)]
                        if trial.region == i
                    ]))

        # Check if any TR needs to be restarted
        f = np.asarray(self.trials._his_observe_value)
        for i in range(self.num_tr):
            if self.length[
                    i] < self.length_min:  # Restart trust region if converged
                # indicator = [
                #     trial.region == i for trial in self.trials.traj_history
                # ]
                idx_i = self._idx[:, 0] == i

                n_evals, fbest = self.trials.trials_num, f[idx_i].min()
                logger.info(f"{n_evals}) TR-{i} converged to: : {fbest:.4}")

                # Reset length and counters, remove old data from trust region
                self.length[i] = self.length_init
                self.succcount[i] = 0
                self.failcount[i] = 0
                self._idx[idx_i, 0] = -1  # Remove points from trust region
                self.hypers[i] = {}  # Remove model hypers

    def _adjust_length(self, fX_next, i, f_min):
        assert i >= 0 and i <= self.num_tr - 1

        # fX_min = self.fX[self._idx[:len(self.fX),
        #    0] == i].min()  # Target value
        if min(fX_next) < f_min - 1e-3 * math.fabs(f_min):
            self.succcount[i] += 1
            self.failcount[i] = 0
        else:
            self.succcount[i] = 0
            self.failcount[i] += len(
                fX_next)  # NOTE: Add size of the batch for this TR

        if self.succcount[i] == self.succtol:  # Expand trust region
            self.length[i] = min([2.0 * self.length[i], self.length_max])
            self.succcount[i] = 0
        elif self.failcount[
                i] >= self.failtol:  # Shrink trust region (we may have exceeded the failtol)
            self.length[i] /= 2.0
            self.failcount[i] = 0

    def _random_suggest(self, n_suggestions=1, region=0):
        trial_list = []
        for n in range(n_suggestions):
            config = self.initial_design_configs[region].pop(0)
            trial_list.append(
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      dense_array=config.get_dense_array(),
                      origion='turbo-design',
                      region=region), )
        return trial_list


opt_class = TuRBO
