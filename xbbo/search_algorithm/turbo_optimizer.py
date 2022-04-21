from collections import defaultdict
import logging
import math
from copy import deepcopy
import warnings
from scipy.stats.qmc import Sobol

import numpy as np
# from xbbo.acquisition_function.acq_optimizer import DesignBoundSearch
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace

from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design
from xbbo.surrogate.gaussian_process import GPR_sklearn


from xbbo.utils.constants import MAXINT
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


class TuRBO_state():
    def __init__(self,
                 surrogate_model,
                 marker,
                 bounds,
                 rng,
                 dim,
                 n_min_sample,
                 succ_tol=3,
                 length_max=1.6,
                 length_min=0.5**7,
                 length_init=0.8,
                 **kwargs) -> None:
        self.surrogate_model = surrogate_model
        # self.hyper = self.surrogate_models.hypers
        self.dim = dim
        self.marker = marker
        self.rng = rng
        self.bounds = bounds
        self.succ_tol = succ_tol
        self.length_min = length_min
        self.length_max = length_max
        self.length_init = length_init
        self.n_min_sample = n_min_sample
        self.sobol_gen = Sobol(d=self.dim,
                          scramble=True,
                          seed=self.rng.randint(MAXINT))
        self._restart()

    def _restart(self):
        # self.do_optimize = True
        self.length = self.length_init
        self.succ_count = 0
        self.fail_count = 0
        self.center = None
        self.center_value = np.inf

    def _train(self, X, Y):
        # self.surrogate_model.do_optimize = self.do_optimize
        self.surrogate_model.train(X, Y)
        # self.do_optimize = False

    def sample_y(self, X, size=1):
        mean, var = self.surrogate_model.predict_marginalized_over_instances(
            X, 'full_cov')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sample = self.rng.multivariate_normal(mean.ravel(), var,
                                            size=size).T  # (sample, N)
            except:
                sample = self.rng.multivariate_normal(mean.ravel(), np.diag(np.diag(var)),
                                            size=size).T # TODO
                
        return sample

    def update(self, trial: Trial, trials: Trials, obs_num: int):
        markers = np.array(trials.markers)
        idx = markers == self.marker
        if idx.sum() < self.n_min_sample:
            return
        # self.do_optimize = True
        if (not np.isfinite(self.center_value)
            ) or trial.observe_value < self.center_value - 1e-3 * math.fabs(
                self.center_value):
            self.center = self.to_unit_cube(trial.array)
            self.center_value = trial.observe_value
            logger.info(
                f"{trials.trials_num}) New best @ TR-{self.marker}: {self.center_value:.4}"
            )
            self.succ_count += 1
            self.fail_count = 0
        else:
            self.succ_count = 0
            self.fail_count += obs_num  # NOTE: Add size of the batch for this TR
        if self.succ_count == self.succ_tol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succ_count = 0
        elif self.fail_count >= self.fail_tol:  # Shrink trust region (we may have exceeded the fail_tol)
            self.length /= 2.0
            self.fail_count = 0

        # Check if any TR needs to be restarted

        if self.length < self.length_min:  # Restart trust region if converged
            # indicator = [
            #     trial.region == i for trial in self.trials.traj_history
            # ]

            logger.info(
                f"{trials.trials_num}) TR-{self.marker} converged to: : {self.center_value:.4}"
            )

            # Reset length and counters, remove old data from trust region
            self._restart()
            # Remove points from trust region
            markers[idx] = -1
            trials.markers = list(markers)

        X = self.to_unit_cube((trials.get_array())[idx])
        Y = np.array(trials._his_observe_value)[idx]
        self._train(X, Y)
    
    def _get_length_scale(self):
        ks = self.surrogate_model.kernel
        ele_end = 0
        ele_num = None
        for hp in ks.hyperparameters:
            ele_end += hp.n_elements
            if 'length_scale' in hp.name:
                ele_num = hp.n_elements
                break
        assert ele_num is not None
        return np.exp(ks.theta[ele_end-ele_num:ele_end]) # Be careful log theta

    def create_candidates(self, n_candidates):
        length = self.length
        weights = (self._get_length_scale())
        # Create the trust region boundaries
        x_center = self.center  #X[fX.argmin().item(), :][None, :]
        weights = weights / weights.mean(
        )  # This will make the next line more stable
        weights = weights / np.prod(np.power(
            weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        
        pert = self.sobol_gen.random(n_candidates)
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = self.rng.rand(n_candidates, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, self.rng.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((n_candidates, self.dim))
        X_cand[mask] = pert[mask]
        return X_cand

    def to_unit_cube(self, x):
        """Project to [0, 1]^d from hypercube with bounds lb and ub"""

        xx = (x - self.bounds.lb) / (self.bounds.ub - self.bounds.lb)
        return xx

    def from_unit_cube(self, x):
        """Project from [0, 1]^d to hypercube with bounds lb and ub"""
        xx = x * (self.bounds.ub - self.bounds.lb) + self.bounds.lb
        return xx


@alg_register.register('turbo')
class TuRBO(AbstractOptimizer):
    '''
    reference: https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_m.py
    '''
    def __init__(
            self,
            space: DenseConfigurationSpace,
            seed: int = 42,
            surrogate: str = 'gp',
            # acq_func: str = 'mc',
            # acq_opt: str = 'design',
            initial_design: str = 'sobol',
            num_tr=1,
            #  suggest_limit: int = np.inf,
            **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   **kwargs)
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        self.n_min_sample = kwargs.get('n_min_sample', 5)
        self.init_budget = self.n_min_sample  # self.initial_design.init_budget
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, init_budget=self.init_budget)

        self.initial_design_configs = [[] for _ in range(num_tr)]
        self.trials = Trials(dim=self.dimension)
        self.n_training_steps = kwargs.get("n_training_steps", 50)
        self.max_cholesky_size = kwargs.get("max_cholesky_size", 2000)
        self.dim = self.dimension
        self.n_candidates = 2**int(np.log2(min(100 * self.dim, 5000)))
        self.use_ard = kwargs.get("use_ard", True)
        self.num_tr = num_tr
        self.candidates = []

        if surrogate == 'gp':
            self.turbo_states = [
                TuRBO_state(GPR_sklearn(self.space, rng=self.rng),
                            i,
                            self.bounds,
                            self.rng,
                            self.dim,
                            n_min_sample=self.n_min_sample,
                            **kwargs) for i in range(num_tr)
            ]
            # self.surrogate_models = [
            #     GPR_sklearn(self.space, rng=self.rng) for _ in range(num_tr)
            # ]
        else:
            raise ValueError('surrogate {} not in {}'.format(
                surrogate, ['gp']))

    def _suggest(self, n_suggestions=1):
        markers = np.array(self.trials.markers)

        for m in range(self.num_tr):
            if not hasattr(self.turbo_states[m], 'fail_tol'):
                self.turbo_states[m].fail_tol = np.ceil(
                    np.max([4.0 / n_suggestions, self.dim / n_suggestions]))
            if (markers == m).sum() < self.n_min_sample:
                return self._init_suggest(n_suggestions, m)

        X_cand = np.empty((self.num_tr, self.n_candidates, self.dim))
        y_cand = np.full(
            (self.num_tr, self.n_candidates, n_suggestions), np.inf)
        for m in range(self.num_tr):
            cand = self.turbo_states[m].create_candidates(self.n_candidates)
            cand_y = self.turbo_states[m].sample_y(cand, size=n_suggestions)

            X_cand[m, :, :], y_cand[m, :, :] = cand, cand_y

        X_next = np.empty((n_suggestions, self.dim))
        trial_list = []
        for b in range(n_suggestions):
            marker, j = np.unravel_index(np.argmin(y_cand[:, :, b]),
                                         (self.num_tr, self.n_candidates))
            X_next[b, :] = (X_cand[marker, j, :])
            # X_next[b, :] = deepcopy(X_cand[marker, j, :])
            assert np.isfinite(
                y_cand[marker, j,
                       b])  # Just to make sure we never select nan or inf
            # Make sure we never pick this point again
            y_cand[marker, j, :] = np.inf
            array = self.turbo_states[0].from_unit_cube(X_next[b, :])
            config = DenseConfiguration.from_array(self.space, array)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=array,
                      origin='TuRBO-region-{}'.format(marker),
                    #   region=marker,
                      marker=marker))
        return trial_list

    def _observe(self, trial_list):
        update_markers = defaultdict(list)
        for trial in trial_list:
            self.trials.add_a_trial(trial)
            update_markers[trial.marker].append((trial.observe_value, trial))
        for marker in update_markers:
            best_trial = sorted(update_markers[marker],
                                key=lambda x: x[0])[0][1]
            self.turbo_states[marker].update(best_trial, self.trials,
                                             len(trial_list))

        # markers = self.trials.markers
        # for m in update_markers:
        # if (markers == m).sum() <= self.n_min_sample:
        #     continue  # don't train

    def _init_suggest(self, n_suggestions=1, region=0):
        trial_list = []
        for n in range(n_suggestions):
            if not self.initial_design_configs[region]:
                self.initial_design_configs[
                    region] = self.initial_design.select_configurations()
            config = self.initial_design_configs[region].pop(0)
            trial_list.append(
                Trial(
                    configuration=config,
                    config_dict=config.get_dictionary(),
                    array=config.get_array(sparse=False),
                    #   array=config.get_array(sparse=False),
                    origion='turbo-design',
                    # region=region,
                    marker=region))
        return trial_list


opt_class = TuRBO
