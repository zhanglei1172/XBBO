import numpy as np
from scipy.optimize import minimize
from sklearn.utils import check_random_state
from scipy.optimize import Bounds
from sklearn.ensemble import RandomForestClassifier

import logging

from bbomark.core.feature_space import FeatureSpace_uniform
from bbomark.core.abstract_optimizer import AbstractOptimizer
from bbomark.configspace.space import Configurations

class BORE(AbstractOptimizer, FeatureSpace_uniform):
    opt_name = 'BORE'
    def __init__(self,
                 config_spaces,
                 optimizer_kws = {},
                 num_random_init=10,
                 # num_starts=5,
                 random_rate=0.1,
                 seed=None,
                 logger=None,
                 ):

        # super(AbstractOptimizer, self).__init__(config_spaces)
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self)
        self.dtypes_idx_map = self.space.dtypes_idx_map
        # self.multi_start = multi_start(minimizer_fn=minimize)
        self.classifier = Classfify()

        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        bounds = self.space.get_bounds()
        self.bounds = Bounds(bounds.lb, bounds.ub) #(bounds.lb, bounds.ub)
        self.history = History()

        self.random_rate = random_rate
        self.random_state = np.random.RandomState(seed)
        self.logger=logger
        if self.logger is None:
            self.logger = logging.getLogger()
        self.num_random_init = num_random_init
        # self.num_starts = num_starts
        self.num_starts = optimizer_kws.get("num_starts", 5)
        self.num_samples = optimizer_kws.get("num_samples", 1024)
        self.method = optimizer_kws.get("method", "L-BFGS-B")
        self.ftol = optimizer_kws.get("ftol", 1e-9)
        self.max_iter = optimizer_kws.get("max_iter", 1000)


    def suggest(self, n_suggestions):
        dataset_size = len(self.history.targets)

        config_random = self.space.sample_configuration(n_suggestions)
        x_guess = [x_guess_config.get_dict_unwarped() for x_guess_config in config_random]
        features = [x_guess_config.get_array() for x_guess_config in config_random]


        # epsilon-greedy exploration
        if self.random_rate is not None and \
                self.random_state.binomial(p=self.random_rate, n=1):
            self.logger.info("[Glob. maximum: skipped "
                             f"(prob={self.random_rate:.2f})] "
                             "Suggesting random candidate ...")
            return (x_guess, features)

        # Insufficient training data
        if dataset_size < self.num_random_init:
            self.logger.debug(f"Completed {dataset_size}/{self.num_random_init}"
                              " initial runs. Suggesting random candidate...")
            return (x_guess, features)

        # # Create classifier (if retraining from scratch every iteration)
        # self._maybe_create_classifier()
        #
        # # Train classifier
        # self._update_classifier()

        # Maximize classifier wrt input
        self.logger.debug("Beginning multi-start maximization with "
                          f"{self.num_starts} starts...")
        loc = []
        for n_ in range(n_suggestions):
            opt = self.classifier.argmax(self.bounds,
                                    num_starts=self.num_starts,
                                    num_samples=self.num_samples,
                                    method=self.method,
                                    options=dict(maxiter=self.max_iter,
                                                 ftol=self.ftol),
                                    print_fn=self.logger.debug,
                                    filter_fn=self._is_unique,
                                    random_state=self.random_state)
            if opt is None:
                # TODO(LT): It's actually important to report which of these
                # failures occurred...
                self.logger.warn("[Glob. maximum: not found!] Either optimization "
                                 f"failed in all {self.num_starts} starts, or "
                                 "all maxima found have been evaluated previously!"
                                 " Suggesting random candidate...")
                return (x_guess, features)
            loc.append(opt.x)
            self.logger.info(f"[Glob. maximum: value={-opt.fun.item():.3f} x={opt.x}]")


        x_guess = [None] * n_suggestions
        for ii, xx in enumerate(loc):
            x_array = self.feature_to_array(xx, self.sparse_dimension)
            dict_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            x_guess[ii] = dict_unwarped

        # Delete classifier (if retraining from scratch every iteration)
        # self._maybe_delete_classifier()

        return (x_guess, loc)

    def observe(self, features, y):
        # for xx, yy in zip(features, y):
        self.history.tell(features, y)
        tau = np.quantile(self.history.targets, q=0.25)
        z = np.less(self.history.targets, tau)
        self.classifier.fit(self.history.features, z)

    def transform_sparseArray_to_optSpace(self, sparse_array):
        return [self.array_to_feature(x, self.dense_dimension) for x in sparse_array]

    def _is_unique(self, res):
        is_duplicate = self.history.is_duplicate(res.x)
        if is_duplicate:
            self.logger.warn("Duplicate detected! Skipping...")
        return not is_duplicate


class Classfify():

    def __init__(self):
        self.minimize_multi_start = self._multi_start(minimizer_fn=minimize)
        self.model = RandomForestClassifier(n_estimators=25)

    def _multi_start(self, minimizer_fn=minimize):

        def new_minimizer(fn, bounds, num_starts, num_samples=None,
                          random_state=None, *args, **kwargs):
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
            X_init = random_state.uniform(low=low, high=high, size=(num_samples, dims))

            values = fn(X_init)
            ind = values.argsort()

            results = []
            for i in range(num_starts):
                x_init = X_init[ind[i]:ind[i]+1, :]
                result = minimizer_fn(fn, x0=x_init, bounds=bounds, *args, **kwargs)
                results.append(result)

            return results

        return new_minimizer

    def fit(self, X, z):
        self.model.fit(X, z.ravel())

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return 1 - self.model.predict_log_proba(x)[:, 1]


    def _maxima(self, bounds, num_starts=5, num_samples=1024, method="L-BFGS-B",
               options=dict(maxiter=1000, ftol=1e-9), random_state=None):

        return self.minimize_multi_start(self.predict, bounds=bounds,
                                    num_starts=num_starts,
                                    num_samples=num_samples,
                                    random_state=random_state,
                                    method=method, jac=False, options=options)

    def argmax(self, bounds, print_fn=print, filter_fn=lambda res: True,
               *args, **kwargs):

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

class History():
    def __init__(self):
        self.features = []
        self.targets = []

    def tell(self, features, targets):
        self.features.extend(features)
        self.targets.extend(targets)

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol)
                   for x_prev in self.features)

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
# feature_space = FeatureSpace_uniform