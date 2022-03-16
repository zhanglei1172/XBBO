import logging
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize, OptimizeResult
from sklearn.ensemble import RandomForestClassifier
from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design

from . import alg_register
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration

logger = logging.getLogger(__name__)


@alg_register.register('bore')
class BORE(AbstractOptimizer):
    '''
    ref: https://github.com/ltiao/bore
    '''
    def __init__(self,
                 space,
                 seed: int = 42,
                 random_rate: float = 0.1,
                 initial_design: str = 'sobol',
                 suggest_limit: int = np.inf,
                 classify: str = 'rf',
                 **kwargs):

        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='one-hot',
                                   encoding_ord='one-hot',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)

        # self.multi_start = multi_start(minimizer_fn=minimize)
        self.classifier = Classfify(classify=classify)
        if self.space.get_conditions():
            raise NotImplementedError(
                "BORE optimizer currently does not support conditional space!")

        self.dimension = self.space.get_dimensions()
        bounds = self.space.get_bounds()
        self.bounds = Bounds(bounds.lb, bounds.ub)  #(bounds.lb, bounds.ub)
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )

        self.trials = Trials(dim=self.dimension)
        self.random_rate = random_rate
        self.num_starts = kwargs.get("num_starts", 5)
        self.num_samples = kwargs.get("num_samples", 1024)
        self.method = kwargs.get("method", "L-BFGS-B")
        self.options = kwargs.get('options', dict(maxiter=1000, ftol=1e-9))
        self.quantile = kwargs.get("quantile", 0.25)

    def _suggest(self, n_suggestions=1):
        dataset_size = self.trials.trials_num

        # Insufficient training data
        if dataset_size < self.init_budget:
            logger.debug(f"Completed {dataset_size}/{self.init_budget}"
                         " initial runs. Suggesting random candidate...")
            return [
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      array=config.get_array(sparse=False),
                      origin='Random') for config in
                self.initial_design_configs[dataset_size:dataset_size +
                                            n_suggestions]
            ]
        # config_random = []
        # while len(config_random) < n_suggestions:
        #     config = self.space.sample_configuration(1)[0]
        #     if not self.trials.is_contain(config):
        #         config_random.append(config)

        # # epsilon-greedy exploration
        # if self.random_rate is not None and \
        #         self.rng.binomial(p=self.random_rate, n=1):
        #     logger.info("[Glob. maximum: skipped "
        #                 f"(prob={self.random_rate:.2f})] "
        #                 "Suggesting random candidate ...")
        #     return [
        #         Trial(configuration=config,
        #               config_dict=config.get_dictionary(),
        #               array=config.get_array(sparse=False),
        #               origin='Random') for config in config_random
        #     ]
        targets = self.trials.get_history()[0]
        tau = np.quantile(targets, q=self.quantile)
        z = np.less(targets, tau)
        self.classifier.fit(self.trials.get_array(), z)
        # # Create classifier (if retraining from scratch every iteration)
        # self._maybe_create_classifier()
        #
        # # Train classifier
        # self._update_classifier()
        X_init = self.rng.uniform(low=self.bounds.lb,
                                  high=self.bounds.ub,
                                  size=(self.num_samples,
                                        self.dimension))
        f_init = self.classifier.predict(X_init)  # to minimize

        trial_list = []
        for n_ in range(n_suggestions):
            results = []
            if self.num_starts > 0:
                ind = np.argpartition(f_init,
                                      kth=self.num_starts - 1,
                                      axis=None)
                for i in range(self.num_starts):
                    x0 = X_init[ind[i]]
                    result = minimize(self.classifier.predict,
                                      x0=x0,
                                      method=self.method,
                                      jac=False,
                                      bounds=self.bounds,
                                      options=self.options)
                    results.append(result)
                    # # TODO(LT): Make this message a customizable option.
                    # print(f"[Maximum {i+1:02d}: value={result.fun:.3f}] "
                    #         f"success: {result.success}, "
                    #         f"iterations: {result.nit:02d}, "
                    #         f"status: {result.status} ({result.message})")
            else:
                i = np.argmin(f_init, axis=None)
                result = OptimizeResult(x=X_init[i],
                                        fun=f_init[i],
                                        success=True)
                results.append(result)
            best_v = np.inf
            best_config = None
            for res in results:
                if (res.success or res.status == 1) and res.fun < best_v:
                    config = DenseConfiguration.from_array(
                        self.space, res.x)
                    if not self.trials.is_contain(config):
                        best_config = config
            assert best_config is not None
            trial_list.append(
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      array=config.get_array(sparse=False)))

        return trial_list

    def _observe(self, trial_list):
        # for xx, yy in zip(features, y):
        for trial in trial_list:
            self.trials.add_a_trial(trial)

    def _is_unique(self, res, rtol=1e-5, atol=1e-8):
        is_duplicate = any(
            np.allclose(x_prev, res.x, rtol=rtol, atol=atol)
            for x_prev in self.trials.get_array())
        if is_duplicate:
            logger.warn("Duplicate detected! Skipping...")
        return not is_duplicate


class Classfify():
    def __init__(self, classify: str = 'rf'):
        if classify == 'rf':
            self.model = RandomForestClassifier(n_estimators=25)
        else:
            raise NotImplementedError()

    def fit(self, X, z):
        self.model.fit(X, z.ravel())

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return 1 - self.model.predict_proba(x)[:, 1]


def from_bounds(bounds):

    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dim = len(low)
        assert dim == len(high), "lower and upper bounds sizes do not match!"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dim = len(bounds)

    return (low, high), dim


opt_class = BORE