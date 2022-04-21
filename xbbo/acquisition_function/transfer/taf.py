from typing import Optional

import numpy as np
from scipy import stats

from xbbo.acquisition_function.base import AbstractAcquisitionFunction

class TAF_AcqFunc(AbstractAcquisitionFunction):
    def __init__(self, surrogate_model, base_models, rng, rho=None):
        self.par = 0.0
        self.base_models = base_models
        self.rng = rng
        self.rho = rho
        super().__init__(surrogate_model)

    def argmax(self, candidates):
        print('depreciate!')
        # best_ei = -1
        # # best_candidate = []
        # candidates_rm_id = []
        # y_hats = list(zip(*surrogate.predict_with_sigma(candidates)))
        scores = self.__call__(candidates)

        return candidates[self.rng.choice(np.where(scores == scores.max())[0])]

    def update_weight(self, w, rho=None):
        assert w.min() >= 0
        if rho:
            self.w = np.array(w, dtype='float')
            self.selfWeight = rho
        else:
            self.selfWeight = w[-1]
            self.w = np.array(w[:-1],dtype='float')
        #non_zero_weight_indices = (self.w**2 > 0).nonzero()[0]
        non_zero_weight_indices = self.w.nonzero()[0]
        self.pre_weight_model = []
        self.base_incuments = []
        for idx in non_zero_weight_indices:
            self.pre_weight_model.append(self.base_models[idx])
            self.base_incuments.append(self._base_incuments[idx])
        self.weight = self.w[non_zero_weight_indices]
        norm_coeff = (self.weight.sum() + self.selfWeight)
        self.weight /= norm_coeff
        self.selfWeight /= norm_coeff
        
        

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaetas]

        m, v = self.surrogate_model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.y_best is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.y_best - m - self.par) / s
            return (self.y_best - m -
                    self.par) * stats.norm.cdf(z) + s * stats.norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")
        # m, v = self.surrogate_model.predict_marginalized_over_instances(X)
        # denominator = self.selfWeight
        f *= self.selfWeight
        for d in range(len(self.weight)):
            f += self.weight[d] * (self.base_incuments[d] - self.pre_weight_model[d].predict(
                X, None)[0]).clip(0)
            # denominator += self.weight[d]

        return f



class TAF_with_RGPE_rank():
    def __init__(
        self,
        gps,
    ):
        self.gps = gps
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best):  # minimize
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    def argmax(self,
               y_best,
               surrogate,
               candidates,
               similarity,
               old_ybests,
               rho=0.75):
        best_ei = -1
        best_candidate = []
        candidates_rm_id = []
        for i, candidate in enumerate(candidates):
            # if not hasattr(surrogate, 'X'):
            if not surrogate.is_fited:
                y_hat = 0, 1000
            else:
                y_hat = surrogate.predict_with_sigma(candidate)
            denominator = rho
            ei = self._getEI(y_hat[0], y_hat[1], y_best).item() * rho
            for d in range(len(similarity)):
                # mu, sigma = self.gps[d].cached_predict_with_sigma(candidate)
                # ei += similarity[d] * max(np.random.normal(mu, sigma)-self.old_Ybests[d], 0)
                mu = self.gps[d].cached_predict(candidate)  # TODO
                ei += similarity[d] * max(mu - old_ybests[d], 0)
                # ei += similarity[d] * max(mu-y_best, 0)
                # mu = self.gps[d].cached_predict(candidate)
                # ei += similarity[d] * max(mu-self.old_Ybests[d][len_h], 0)
                denominator += similarity[d]
            ei /= denominator
            if ei > best_ei:
                best_ei = ei
                best_candidate = [candidate]
                candidates_rm_id = [i]
            elif ei == best_ei:
                best_candidate.append(candidate)
                candidates_rm_id.append(i)

        assert best_candidate
        idx = np.random.choice(len(best_candidate))
        # TODO: this
        # for d in range(len(old_ybests)):
        #     old_ybests[d] = min(old_ybests[d], self.gps[d].cached_predict(best_candidate[idx]))
        return (best_candidate)[idx], candidates_rm_id[idx]