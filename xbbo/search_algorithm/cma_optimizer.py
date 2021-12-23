import math
from typing import Optional, List, Tuple, cast

import numpy as np
import cma
from xbbo.configspace.feature_space import FeatureSpace_gaussian
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration
from xbbo.core.trials import Trials

_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

class CMAES(AbstractOptimizer, FeatureSpace_gaussian):

    def __init__(self,
                 config_spaces,):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.es = cma.CMAEvolutionStrategy([0.] * self.dense_dimension, 1.0)
        # self.hp_num = len(configs)

        self.trials = Trials()
        self.listx = []
        self.listy = []

    def suggest(self, n_suggestions=1):

        sas = []
        x_arrays = []
        for n in range(n_suggestions):
            new_individual = self.es.ask(1)[0]
            sas.append(new_individual)
            x_arrays.append(self.feature_to_array(np.asarray(new_individual), self.sparse_dimension))

        x = [DenseConfiguration.array_to_dict(self.space,
                                                  np.array(sa)) for sa in x_arrays]
        self.trials.params_history.extend(x)
        # self._num_suggestions += n_suggestions
        return x, sas

    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        self.listx += x
        self.listy += y
        if len(self.listx) >= self.es.popsize:
            self.es.tell(self.listx, self.listy)
            self.listx, self.listy = [], []

class CMA(AbstractOptimizer, FeatureSpace_gaussian):
    '''
    reference: https://github.com/CyberAgentAILab/cmaes/blob/main/cmaes/_cma.py
    '''

    def __init__(self,
                 config_spaces,):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # configs = self.space.get_hyperparameters()
        n_dim = self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        mean = np.zeros(n_dim)
        sigma = 1.0
        popsize = 4 + math.floor(3 * math.log(n_dim)) # (eq. 48)
        mu = popsize // 2


        # (eq.49)
        weights_prime = np.array(
            [
                math.log((popsize + 1) / 2) - math.log(i + 1)
                for i in range(popsize)
            ]
        )
        mu_eff = (np.sum(weights_prime[:mu]) ** 2) / np.sum(weights_prime[:mu] ** 2)
        mu_eff_minus = (np.sum(weights_prime[mu:]) ** 2) / np.sum(
            weights_prime[mu:] ** 2
        )

        # learning rate for the rank-one update
        alpha_cov = 2
        c1 = alpha_cov / ((n_dim + 1.3) ** 2 + mu_eff)
        # learning rate for the rank-μ update
        cmu = min(
            1 - c1 - 1e-8,  # 1e-8 is for large popsize.
            alpha_cov
            * (mu_eff - 2 + 1 / mu_eff)
            / ((n_dim + 2) ** 2 + alpha_cov * mu_eff / 2),
        )
        assert c1 <= 1 - cmu, "invalid learning rate for the rank-one update"
        assert cmu <= 1 - c1, "invalid learning rate for the rank-μ update"

        min_alpha = min(
            1 + c1 / cmu,  # eq.50
            1 + (2 * mu_eff_minus) / (mu_eff + 2),  # eq.51
            (1 - c1 - cmu) / (n_dim * cmu),  # eq.52
        )

        # (eq.53)
        positive_sum = np.sum(weights_prime[weights_prime > 0])
        negative_sum = np.sum(np.abs(weights_prime[weights_prime < 0]))
        weights = np.where(
            weights_prime >= 0,
            1 / positive_sum * weights_prime,
            min_alpha / negative_sum * weights_prime,
        )
        cm = 1  # (eq. 54)

        # learning rate for the cumulation for the step-size control (eq.55)
        c_sigma = (mu_eff + 2) / (n_dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n_dim + 1)) - 1) + c_sigma
        assert (
                c_sigma < 1
        ), "invalid learning rate for cumulation for the step-size control"

        # learning rate for cumulation for the rank-one update (eq.56)
        cc = (4 + mu_eff / n_dim) / (n_dim + 4 + 2 * mu_eff / n_dim)
        assert cc <= 1, "invalid learning rate for cumulation for the rank-one update"

        self._n_dim = n_dim
        self._popsize = popsize
        self._mu = mu
        self._mu_eff = mu_eff

        self._cc = cc
        self._c1 = c1
        self._cmu = cmu
        self._c_sigma = c_sigma
        self._d_sigma = d_sigma
        self._cm = cm

        # E||N(0, I)|| (p.28)
        self._chi_n = math.sqrt(self._n_dim) * (
                1.0 - (1.0 / (4.0 * self._n_dim)) + 1.0 / (21.0 * (self._n_dim ** 2))
        )

        self._weights = weights

        # evolution path
        self._p_sigma = np.zeros(n_dim)
        self._pc = np.zeros(n_dim)

        self._mean = mean


        self._C = np.eye(n_dim)


        self._sigma = sigma
        self._D: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None

        # bounds contains low and high of each parameter.
        self._bounds = np.stack([np.full_like(mean, -np.inf),np.full_like(mean, np.inf)], axis=1)
        self._n_max_resampling = 100

        self._g = 0

        # Termination criteria
        self._tolx = 1e-12 * sigma
        self._tolxup = 1e4
        self._tolfun = 1e-12
        self._tolconditioncov = 1e14

        self._funhist_term = 10 + math.ceil(30 * n_dim / popsize)
        self._funhist_values = np.empty(self._funhist_term * 2)

        self.trials = Trials()
        self.listx = []
        self.listy = []

    def suggest(self, n_suggestions=1):

        sas = []
        x_arrays = []
        for n in range(n_suggestions):
            new_individual = self.ask()
            sas.append(new_individual)
            x_arrays.append(self.feature_to_array(np.asarray(new_individual), self.sparse_dimension))

        x = [DenseConfiguration.array_to_dict(self.space,
                                                  np.array(sa)) for sa in x_arrays]
        self.trials.params_history.extend(x)
        # self._num_suggestions += n_suggestions
        return x, sas

    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        self.listx += x
        self.listy += y
        if len(self.listx) >= self._popsize:
            self.tell(list(zip(np.array(self.listx), self.listy)))
            self.listx, self.listy = [], []

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        for i in range(self._n_max_resampling):
            x = self._sample_solution()
            if self._is_feasible(x):
                return x
        x = self._sample_solution()
        x = self._repair_infeasible_params(x)
        return x

    def _eigen_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        self._C = np.dot(np.dot(B, np.diag(D ** 2)), B.T)

        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> np.ndarray:
        B, D = self._eigen_decomposition()
        z = np.random.randn(self._n_dim)  # ~ N(0, I)
        y = cast(np.ndarray, B.dot(np.diag(D))).dot(z)  # ~ N(0, C)
        x = self._mean + self._sigma * y  # ~ N(m, σ^2 C)
        return x

    def _is_feasible(self, param: np.ndarray) -> bool:
        if self._bounds is None:
            return True
        return cast(
            bool,
            np.all(param >= self._bounds[:, 0]) and np.all(param <= self._bounds[:, 1]),
        )  # Cast bool_ to bool.

    def _repair_infeasible_params(self, param: np.ndarray) -> np.ndarray:
        if self._bounds is None:
            return param

        # clip with lower and upper bound.
        param = np.where(param < self._bounds[:, 0], self._bounds[:, 0], param)
        param = np.where(param > self._bounds[:, 1], self._bounds[:, 1], param)
        return param

    def tell(self, solutions: List[Tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors"

        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # Stores 'best' and 'worst' values of the
        # last 'self._funhist_term' generations.
        funhist_idx = 2 * (self._g % self._funhist_term)
        self._funhist_values[funhist_idx] = solutions[0][1]
        self._funhist_values[funhist_idx + 1] = solutions[-1][1]

        # Sample new population of search_points, for k=1, ..., popsize
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None

        x_k = np.array([s[0] for s in solutions])  # ~ N(m, σ^2 C)
        y_k = (x_k - self._mean) / self._sigma  # ~ N(0, C)

        # Selection and recombination
        y_w = np.sum(y_k[: self._mu].T * self._weights[: self._mu], axis=1)  # eq.41
        self._mean += self._cm * self._sigma * y_w

        # Step-size control
        C_2 = cast(
            np.ndarray, cast(np.ndarray, B.dot(np.diag(1 / D))).dot(B.T)
        )  # C^(-1/2) = B D^(-1) B^T
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * C_2.dot(y_w)

        norm_p_sigma = np.linalg.norm(self._p_sigma)
        self._sigma *= np.exp(
            (self._c_sigma / self._d_sigma) * (norm_p_sigma / self._chi_n - 1)
        )
        self._sigma = min(self._sigma, _SIGMA_MAX)

        # Covariance matrix adaption
        h_sigma_cond_left = norm_p_sigma / math.sqrt(
            1 - (1 - self._c_sigma) ** (2 * (self._g + 1))
        )
        h_sigma_cond_right = (1.4 + 2 / (self._n_dim + 1)) * self._chi_n
        h_sigma = 1.0 if h_sigma_cond_left < h_sigma_cond_right else 0.0  # (p.28)

        # (eq.45)
        self._pc = (1 - self._cc) * self._pc + h_sigma * math.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # (eq.46)
        w_io = self._weights * np.where(
            self._weights >= 0,
            1,
            self._n_dim / (np.linalg.norm(C_2.dot(y_k.T), axis=0) ** 2 + _EPS),
        )

        delta_h_sigma = (1 - h_sigma) * self._cc * (2 - self._cc)  # (p.28)
        assert delta_h_sigma <= 1

        # (eq.47)
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )
        self._C = (
            (
                1
                + self._c1 * delta_h_sigma
                - self._c1
                - self._cmu * np.sum(self._weights)
            )
            * self._C
            + self._c1 * rank_one
            + self._cmu * rank_mu
        )


opt_class = CMA
