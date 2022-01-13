import logging
import numpy as np
from scipy.optimize import minimize
from sklearn.utils import check_random_state
from scipy.optimize import Bounds
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
                 space: DenseConfiguration,
                 seed: int = 42,
                 random_rate: float = 0.1,
                 initial_design: str = 'sobol',
                 total_limit: int = 10,
                 classify: str = 'rf',
                 **kwargs):

        AbstractOptimizer.__init__(self, space, seed, **kwargs)

        # self.multi_start = multi_start(minimizer_fn=minimize)
        self.classifier = Classfify(classify=classify)
        if self.space.get_conditions():
            raise NotImplementedError(
                "BORE optimizer currently does not support conditional space!")

        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        bounds = self.space.get_bounds()
        self.bounds = Bounds(bounds.lb, bounds.ub)  #(bounds.lb, bounds.ub)
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=total_limit,**kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )

        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)
        self.random_rate = random_rate
        self.num_starts = kwargs.get("num_starts", 5)
        self.num_samples = kwargs.get("num_samples", 1024)
        self.method = kwargs.get("method", "L-BFGS-B")
        self.ftol = kwargs.get("ftol", 1e-9)
        self.max_iter = kwargs.get("max_iter", 1000)
        self.quantile = kwargs.get("quantile", 0.25)

    def suggest(self, n_suggestions=1):
        dataset_size = self.trials.trials_num

        # Insufficient training data
        if dataset_size < self.init_budget:
            logger.debug(f"Completed {dataset_size}/{self.init_budget}"
                         " initial runs. Suggesting random candidate...")
            return [
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      dense_array=config.get_dense_array(),
                      origin='Random') for config in
                self.initial_design_configs[dataset_size:dataset_size +
                                            n_suggestions]
            ]

        config_random = self.space.sample_configuration(n_suggestions)

        # epsilon-greedy exploration
        if self.random_rate is not None and \
                self.rng.binomial(p=self.random_rate, n=1):
            logger.info("[Glob. maximum: skipped "
                        f"(prob={self.random_rate:.2f})] "
                        "Suggesting random candidate ...")
            return [
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      dense_array=config.get_dense_array(),
                      origin='Random') for config in config_random
            ]
        targets = self.trials.get_history()[0]
        tau = np.quantile(targets, q=self.quantile)
        z = np.less(targets, tau)
        self.classifier.fit(self.trials.get_dense_array(), z)
        # # Create classifier (if retraining from scratch every iteration)
        # self._maybe_create_classifier()
        #
        # # Train classifier
        # self._update_classifier()

        # Maximize classifier wrt input
        logger.debug("Beginning multi-start maximization with "
                     f"{self.num_starts} starts...")
        loc = []
        for n_ in range(n_suggestions):
            opt = self.classifier.argmax(self.bounds,
                                         num_starts=self.num_starts,
                                         num_samples=self.num_samples,
                                         method=self.method,
                                         options=dict(maxiter=self.max_iter,
                                                      ftol=self.ftol),
                                         print_fn=logger.debug,
                                         filter_fn=self._is_unique,
                                         random_state=self.rng)
            if opt is None:
                # TODO(LT): It's actually important to report which of these
                # failures occurred...
                logger.warn("[Glob. maximum: not found!] Either optimization "
                            f"failed in all {self.num_starts} starts, or "
                            "all maxima found have been evaluated previously!"
                            " Suggesting random candidate...")
                return [
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          dense_array=config.get_dense_array(),
                          origin='Random') for config in config_random
                ]
            config = DenseConfiguration.from_dense_array(self.space, opt.x)
            loc.append(
                Trial(configuration=config,
                      config_dict=config.get_dictionary(),
                      dense_array=opt.x))
            logger.info(
                f"[Glob. maximum: value={-opt.fun.item():.3f} x={opt.x}]")

        return loc

    def observe(self, trial_list):
        # for xx, yy in zip(features, y):
        for trial in trial_list:
            self.trials.add_a_trial(trial)

    def _is_unique(self, res, rtol=1e-5, atol=1e-8):
        is_duplicate = any(
            np.allclose(x_prev, res.x, rtol=rtol, atol=atol)
            for x_prev in self.trials.get_dense_array())
        if is_duplicate:
            logger.warn("Duplicate detected! Skipping...")
        return not is_duplicate


class Classfify():
    def __init__(self, classify: str = 'rf'):
        self.minimize_multi_start = self._multi_start(minimizer_fn=minimize)
        if classify == 'rf':
            self.model = RandomForestClassifier(n_estimators=25)
        else:
            raise NotImplementedError()

    def _multi_start(self, minimizer_fn=minimize):
        def new_minimizer(fn,
                          bounds,
                          num_starts,
                          num_samples=None,
                          random_state=None,
                          *args,
                          **kwargs):
            """
            Minimize a function from multiple starting points.
            First, the function is evaluated at some number of points that are
            sampled uniformly at random from within the specified bound.
            Then, the minimizer is called on the function using the best points
            from the previous step as the starting points.

            Parameters
            ----------
            num_starts : int
                Number of starting points from which to run the minimizer on the
                function.
            num_samples : int, optional
                Number of points, sampled uniformly at random within the specified
                bound, to evaluate in order to determine the starting points
                (if not specified, defaults to `num_starts`).

            Returns
            -------
            results : list of `OptimizeResult`
                A list of `scipy.optimize.OptimizeResult` objects that encapsulate
                information about the optimization result.
            """
            random_state = check_random_state(random_state)

            assert "x0" not in kwargs, "`x0` should not be specified"
            # assert "jac" not in kwargs or kwargs["jac"], "`jac` must be true"

            if num_samples is None:
                num_samples = num_starts

            assert num_samples >= num_starts, \
                "number of random samples (`num_samples`) must be " \
                "greater than number of starting points (`num_starts`)"

            (low, high), dims = from_bounds(bounds)

            # TODO(LT): Allow alternative arbitary generator function callbacks
            # to support e.g. Gaussian sampling, low-discrepancy sequences, etc.
            X_init = random_state.uniform(low=low,
                                          high=high,
                                          size=(num_samples, dims))

            values = fn(X_init)
            ind = values.argsort()

            results = []
            for i in range(num_starts):
                x_init = X_init[ind[i]:ind[i] + 1, :]
                result = minimizer_fn(fn,
                                      x0=x_init,
                                      bounds=bounds,
                                      *args,
                                      **kwargs)
                results.append(result)

            return results

        return new_minimizer

    def fit(self, X, z):
        self.model.fit(X, z.ravel())

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return 1 - self.model.predict_log_proba(x)[:, 1]

    def _maxima(self,
                bounds,
                num_starts=5,
                num_samples=1024,
                method="L-BFGS-B",
                options=dict(maxiter=1000, ftol=1e-9),
                random_state=None):

        return self.minimize_multi_start(self.predict,
                                         bounds=bounds,
                                         num_starts=num_starts,
                                         num_samples=num_samples,
                                         random_state=random_state,
                                         method=method,
                                         jac=False,
                                         options=options)

    def argmax(self,
               bounds,
               print_fn=print,
               filter_fn=lambda res: True,
               *args,
               **kwargs):

        # Equivalent to:
        # res_best = min(filter(lambda res: res.success or res.status == 1,
        #                       self.maxima(bounds, *args, **kwargs)),
        #                key=lambda res: res.fun)
        res_best = None
        for i, res in enumerate(self._maxima(bounds, *args, **kwargs)):

            print_fn(f"[Maximum {i + 1:02d}: value={res.fun.item():.3f}] "
                     f"success: {res.success}, "
                     f"iterations: {res.nit:02d}, "
                     f"status: {res.status} ({res.message})")

            # TODO(LT): Create Enum type for these status codes `status == 1`
            # signifies maximum iteration reached, which we don't want to
            # treat as a failure condition.
            if (res.success or res.status == 1) and filter_fn(res):
                if res_best is None or res.fun < res_best.fun:
                    res_best = res

        return res_best


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