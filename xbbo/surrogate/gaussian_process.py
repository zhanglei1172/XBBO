from typing import List
import typing
from scipy import optimize, stats
import sklearn
# from sklearn.gaussian_process import kernels
from sklearn.gaussian_process.kernels import Kernel, KernelOperator
# from scipy.linalg import solve_triangular, cholesky
import numpy as np
# import GPy
from sklearn import gaussian_process

from xbbo.surrogate.base import Surrogate, BaseGP
from xbbo.surrogate.gp_kernels import HammingKernel, Matern, ConstantKernel, WhiteKernel
from xbbo.surrogate.gp_prior import HorseshoePrior, LognormalPrior, Prior, SoftTopHatPrior, TophatPrior
from xbbo.utils.util import get_types

VERY_SMALL_NUMBER = 1e-10


class GaussianTransform:
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    :param y: shape (n, dim)
    """
    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.sorted = y.copy()
        self.sorted.sort(axis=0)

    @staticmethod
    def z_transform(series, values_sorted=None):
        # applies truncated ECDF then inverse Gaussian CDF.
        if values_sorted is None:
            values_sorted = sorted(series)

        def winsorized_delta(n):
            return 1.0 / (4.0 * n**0.25 * np.sqrt(np.pi * np.log(n)))

        delta = winsorized_delta(len(series))

        def quantile(values_sorted, values_to_insert, delta):
            res = np.searchsorted(values_sorted,
                                  values_to_insert) / len(values_sorted)
            return np.clip(res, a_min=delta, a_max=1 - delta)

        quantiles = quantile(values_sorted, series, delta)

        quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)

        return stats.norm.ppf(quantiles)

    def transform(self, y: np.array):
        """
        :param y: shape (n, dim)
        :return: shape (n, dim), distributed along a normal
        """
        assert y.shape[1] == self.dim
        # compute truncated quantile, apply gaussian inv cdf
        return np.stack([
            self.z_transform(y[:, i], self.sorted[:, i])
            for i in range(self.dim)
        ]).T


class StandardTransform:
    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)

    def transform(self, y: np.array):
        z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
        return z


class SEkernel():
    def __init__(self):
        self.initialize()

    def initialize(self):
        # self.sumF = 0.001
        # self.sumL = 0.001
        # self.sumY = 0.001
        self.sigma_f = 1
        self.sigma_l = 1  # TODO 之前设的是1
        self.sigma_y = 0.001

    def compute_kernel(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            x2 = np.atleast_2d(x2)
            x1 = np.atleast_2d(x1)
            # noise = np.diag([self.sigma_y**2 for _ in range(x1.shape[0])])
            noise = np.eye(x1.shape[0]) * self.sigma_y**2
        else:
            x2 = np.atleast_2d(x2)
            x1 = np.atleast_2d(x1)
            noise = 0
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(
            x2**2, 1) - 2 * (x1 @ x2.T)
        return self.sigma_f**2 * np.exp(
            -0.5 / self.sigma_l**2 * dist_matrix) + noise


class GaussianProcessRegressorARD_gpy(Surrogate):
    def __init__(self, dim, min_sample=3):
        super(GaussianProcessRegressorARD_gpy, self).__init__(dim, min_sample)
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.kernel = GPy.kern.Matern52(input_dim=dim, ARD=True)
        # self.kernel = GPy.kern.RBF(input_dim=self.dim,
        #                       variance=0.001,
        #                       lengthscale=0.5,
        #                       ARD=True)

        self.is_fited = False
        self.standardlize = False

    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.is_fited = True
        y = np.asarray(y)
        if self.standardlize:
            self.Y_mean = y.mean()
            self.Y_std = y.std()
        else:
            self.Y_mean = 0
            self.Y_std = 1

        y = (y - self.Y_mean) / self.Y_std
        self.gpr = GPy.models.gp_regression.GPRegression(x, y, self.kernel)
        self.gpr.optimize(max_iters=100)
        # self.kernel = self.gpr.kern

    def predict(self, newX):
        assert self.is_fited
        return np.squeeze(self.gpr.predict(
            np.atleast_2d(newX))[0]) * self.Y_std + self.Y_mean

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key in self.cached_mu_sigma:
            return self.cached_mu_sigma[key][0]
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu) * self.Y_std + self.Y_mean, np.squeeze(
                np.sqrt(std)) * self.Y_std

    def cached_predict_with_sigma(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_sigma[key] = self.predict_with_sigma(newX)
        return self.cached_mu_sigma[key]

    def predict_with_cov(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, cov = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu) * self.Y_std + self.Y_mean, np.squeeze(
                cov) * self.Y_std**2

    def cached_predict_with_cov(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_cov[key] = self.predict_with_cov(newX)
        return self.cached_mu_cov[key]


class GPR_sklearn(BaseGP):
    def __init__(
        self,
        cs,
        #  min_sample=3,
        # alpha=0,
        rng=np.random.RandomState(0),
        n_opt_restarts: int = 10,
        instance_features: typing.Optional[np.ndarray] = None,
        pca_components: typing.Optional[int] = None,
        **kwargs
    ):
        types, bounds = get_types(cs)
        # self.cached = {}

        super(GPR_sklearn, self).__init__(cs, types, bounds, rng,instance_features=instance_features,
            pca_components=pca_components,**kwargs)

        self.is_fited = False
        # self.alpha = alpha  # Fix RBF kernel error
        self.n_opt_restarts = n_opt_restarts
        self._n_ll_evals = 0
        self._set_has_conditions()

    def _get_kernel(self, ):
        cov_amp = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng),
        )

        cont_dims = np.where(np.array(self.types) == 0)[0]
        cat_dims = np.where(np.array(self.types) != 0)[0]

        if len(cont_dims) > 0:
            exp_kernel = Matern(
                np.ones([len(cont_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976))
                 for _ in range(len(cont_dims))],
                nu=2.5,
                operate_on=cont_dims,
            )

        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                np.ones([len(cat_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976))
                 for _ in range(len(cat_dims))],
                operate_on=cat_dims,
            )

        # assert (len(cont_dims) + len(cat_dims)) == len(
        #     scenario.cs.get_hyperparameters())

        noise_kernel = WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, rng=self.rng),
        )

        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only cont
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only cont
            kernel = cov_amp * ham_kernel + noise_kernel
        else:
            raise ValueError()
        # kernel = gaussian_process.kernels.ConstantKernel(
        #     constant_value=1  #, constant_value_bounds=(1e-4, 1e4)
        # ) * gaussian_process.kernels.RBF(
        #     length_scale=1  #, length_scale_bounds=(1e-4, 1e4)
        # )
        return kernel

    def _predict(self,
                X_test,
                cov_return_type: typing.Optional[str] = 'diagonal_cov'):
        '''
        return: \mu ,\sigma^2
        '''
        assert self.is_fited
        X_test = self._impute_inactive(X_test)
        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None

            if self.normalize_y:
                mu = self._untransform_y(mu)

        else:
            predict_kwargs = {'return_cov': False, 'return_std': True}
            if cov_return_type == 'full_cov':
                predict_kwargs = {'return_cov': True, 'return_std': False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)

            if cov_return_type != 'full_cov':
                var = var**2  # since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

            if self.normalize_y:
                mu, var = self._untransform_y(mu, var)

            if cov_return_type == 'diagonal_std':
                var = np.sqrt(
                    var)  # converting variance to std deviation if specified

        return mu, var

    def _get_gp(self) -> gaussian_process.GaussianProcessRegressor:
        return gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=False,
            optimizer=None,
            n_restarts_optimizer=
            -1,  # Do not use scikit-learn's optimization routine
            alpha=0,  # Governed by the kernel
            random_state=self.rng,
        )
    
    def _nll(self, theta: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """
        Returns the negative marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.
        (negative because we use scipy minimize for optimization)

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        self._n_ll_evals += 1

        try:
            lml, grad = self.gp.log_marginal_likelihood(theta, eval_gradient=True)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros(theta.shape)

        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.lnprob(theta[dim])
                grad[dim] += prior.gradient(theta[dim])

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml).all() or not np.all(np.isfinite(grad)):
            return 1e25, np.zeros(theta.shape)
        else:
            return -lml, -grad

    def _train(self, X: np.ndarray, y: np.ndarray, **kwargs):
        X = np.atleast_2d(X)
        X = self._impute_inactive(X)
        if self.normalize_y:
            y = self._normalize_y(y)
        if len(y.shape) == 1:
            self.n_objectives_ = 1
        else:
            self.n_objectives_ = y.shape[1]
        if self.n_objectives_ == 1:
            y = y.flatten()

        n_tries = 10
        for i in range(n_tries):
            try:
                self.gp = self._get_gp()  # new model
                self.gp.fit(X, y)
                break
            except np.linalg.LinAlgError as e:
                if i == n_tries:
                    raise e
                # Assume that the last entry of theta is the noise
                theta = np.exp(self.kernel.theta)
                theta[-1] += 1
                self.kernel.theta = np.log(theta)
        if self.do_optimize:
            self._all_priors = self._get_all_priors(add_bound_priors=False)
            self.hypers = self._optimize()
            self.gp.kernel.theta = self.hypers
            self.gp.fit(X, y)
        else:
            self.hypers = self.gp.kernel.theta
        self.is_fited = True

    def _get_all_priors(
        self,
        add_bound_priors: bool = True,
        add_soft_bounds: bool = False,
    ) -> List[List[Prior]]:
        # Obtain a list of all priors for each tunable hyperparameter of the kernel
        all_priors = []
        to_visit = []
        to_visit.append(self.gp.kernel.k1)
        to_visit.append(self.gp.kernel.k2)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param, KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                continue
            elif isinstance(current_param, Kernel):
                hps = current_param.hyperparameters
                assert len(hps) == 1
                hp = hps[0]
                if hp.fixed:
                    continue
                bounds = hps[0].bounds
                for i in range(hps[0].n_elements):
                    priors_for_hp = []
                    if current_param.prior is not None:
                        priors_for_hp.append(current_param.prior)
                    if add_bound_priors:
                        if add_soft_bounds:
                            priors_for_hp.append(
                                SoftTopHatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                    exponent=2,
                                ))
                        else:
                            priors_for_hp.append(
                                TophatPrior(
                                    lower_bound=bounds[i][0],
                                    upper_bound=bounds[i][1],
                                    rng=self.rng,
                                ))
                    all_priors.append(priors_for_hp)
        return all_priors

    def _optimize(self) -> np.ndarray:
        """
        Optimizes the marginal log likelihood and returns the best found
        hyperparameter configuration theta.

        Returns
        -------
        theta : np.ndarray(H)
            Hyperparameter vector that maximizes the marginal log likelihood
        """

        log_bounds = [(b[0], b[1]) for b in self.gp.kernel.bounds]

        # Start optimization from the previous hyperparameter configuration
        p0 = [self.gp.kernel.theta]
        if self.n_opt_restarts > 0:
            dim_samples = []

            prior = None  # type: typing.Optional[typing.Union[typing.List[Prior], Prior]]
            for dim, hp_bound in enumerate(log_bounds):
                prior = self._all_priors[dim]
                # Always sample from the first prior
                if isinstance(prior, list):
                    if len(prior) == 0:
                        prior = None
                    else:
                        prior = prior[0]
                prior = typing.cast(typing.Optional[Prior], prior)
                if prior is None:
                    try:
                        sample = self.rng.uniform(
                            low=hp_bound[0],
                            high=hp_bound[1],
                            size=(self.n_opt_restarts, ),
                        )
                    except OverflowError:
                        raise ValueError(
                            'OverflowError while sampling from (%f, %f)' %
                            (hp_bound[0], hp_bound[1]))
                    dim_samples.append(sample.flatten())
                else:
                    dim_samples.append(
                        prior.sample_from_prior(self.n_opt_restarts).flatten())
            p0 += list(np.vstack(dim_samples).transpose())

        theta_star = None
        f_opt_star = np.inf
        for i, start_point in enumerate(p0):
            theta, f_opt, _ = optimize.fmin_l_bfgs_b(self._nll,
                                                     start_point,
                                                     bounds=log_bounds)
            if f_opt < f_opt_star:
                f_opt_star = f_opt
                theta_star = theta
        return theta_star

    def _set_has_conditions(self) -> None:
        has_conditions = len(self.configspace.get_conditions()) > 0
        to_visit = []
        to_visit.append(self.kernel)
        while len(to_visit) > 0:
            current_param = to_visit.pop(0)
            if isinstance(current_param,
                          sklearn.gaussian_process.kernels.KernelOperator):
                to_visit.insert(0, current_param.k1)
                to_visit.insert(1, current_param.k2)
                current_param.has_conditions = has_conditions
            elif isinstance(current_param,
                            sklearn.gaussian_process.kernels.Kernel):
                current_param.has_conditions = has_conditions
            else:
                raise ValueError(current_param)


class GaussianProcessRegressorARD_sklearn(Surrogate):
    def __init__(self, dim, min_sample=3):
        super(GaussianProcessRegressorARD_sklearn,
              self).__init__(dim, min_sample)
        self.cached = {}
        kernel = gaussian_process.kernels.ConstantKernel(
            constant_value=1  #, constant_value_bounds=(1e-4, 1e4)
        ) * gaussian_process.kernels.RBF(
            length_scale=1  #, length_scale_bounds=(1e-4, 1e4)
        )
        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2)
        self.is_fited = False

    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.gpr.fit(x, y)
        self.is_fited = True

    def predict(self, newX):
        assert self.is_fited
        return self.gpr.predict(np.atleast_2d(newX))

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), return_std=True)
            return mu, std


class GaussianProcessRegressor(Surrogate):
    def __init__(self, dim, min_sample=3):
        super().__init__(dim, min_sample)
        self.kernel = SEkernel()
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.is_fited = False

    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.is_fited = True
        self.X = x
        kernel = self.kernel.compute_kernel(x)
        self.L = cholesky(kernel, lower=True)
        _part = solve_triangular(self.L, y, lower=True)
        self.KinvY = solve_triangular(self.L.T, _part, lower=False)

    def predict(self, newX):
        assert self.is_fited
        # Kstar = np.squeeze(self.kernel.compute_kernel(self.X, newX))
        Kstar = (self.kernel.compute_kernel(self.X, newX))
        return (Kstar.T @ self.KinvY).item()

    def cached_predict(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return np.squeeze(Kstar.T @ self.KinvY), np.sqrt(
                self.kernel.compute_kernel(newX) - _LinvKstar.T @ _LinvKstar)

    def cached_predict_with_sigma(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_sigma[key] = self.predict_with_sigma(newX)
        return self.cached_mu_sigma[key]

    def cached_predict_with_cov(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_cov:
            self.cached_mu_cov[key] = self.predict_with_cov(newX)
        return self.cached_mu_cov[key]

    def predict_with_cov(self, newX):
        assert self.is_fited
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return np.squeeze(
                Kstar.T @ self.KinvY), (self.kernel.compute_kernel(newX) -
                                        _LinvKstar.T @ _LinvKstar)


