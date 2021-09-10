import numpy as np
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import GP
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList

from bbomark.surrogate.base import Surrogate
from bbomark.surrogate.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessRegressorARD_sklearn,
    GaussianProcessRegressorARD_gpy, GaussianProcessRegressorARD_torch
)


class TST_surrogate_(GP, GPyTorchModel):
    num_outputs = 1
    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)
        # self.candidates = None
        # self.bandwidth = bandwidth

    def forward(self, x):
        weighted_means = []
        non_zero_weight_indices = (self.weights ** 2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        for m_id in range(non_zero_weight_indices.shape[0]):
            model = self.models[non_zero_weight_indices[m_id]]
            posterior = model.posterior(x)
            # posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
            posterior_mean = posterior.mean.squeeze(-1)
            # apply weight
            weight = non_zero_weights[m_id]
            weighted_means.append(weight * posterior_mean)
        mean_x = torch.stack(weighted_means).sum(dim=0)/non_zero_weights.sum()
        posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
        return MultivariateNormal(mean_x, posterior_cov)



class TST_surrogate(Surrogate):
    rho = 0.75

    def __init__(self, dim, bandwidth=0.1):
        super().__init__(dim, min_sample=3)

        # self.new_gp = GaussianProcessRegressor(dim)
        self.new_gp = GaussianProcessRegressorARD_gpy(dim)
        # self.candidates = None
        self.bandwidth = bandwidth
        self.is_fited = False
        # self.history_x = []
        # self.history_y = []

    def get_knowledge(self, old_D_x, old_D_y, new_D_x=None):
        self.old_D_num = len(old_D_x)
        self.gps = []
        for d in range(self.old_D_num):
            # self.gps.append(GaussianProcessRegressor(self.dim))
            self.gps.append(GaussianProcessRegressorARD_gpy(self.dim))
            self.gps[d].fit(old_D_x[d], old_D_y[d])
        if new_D_x is not None:
            candidates = new_D_x
        else:  #
            raise NotImplemented
        self.similarity = [self.rho for _ in range(self.old_D_num)]
        return candidates

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape[0] < self.min_sample:
            return
        self.is_fited = True
        self.new_gp.fit(x, y)
        self.similarity = [self.kendallTauCorrelation(d, x, y) for d in range(self.old_D_num)]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        denominator = self.rho
        mu, sigma = self.new_gp.predict_with_sigma(newX)
        mu *= self.rho
        for d in range(self.old_D_num):
            mu += self.similarity[d] * self.gps[d].cached_predict(newX)
            denominator += self.similarity[d]
        mu /= denominator
        if sigma == np.inf:
            sigma = 1000
        return mu, sigma

    def kendallTauCorrelation(self, d, x, y):
        '''
        计算第d个datasets与new datasets的 相关性
        (x, y) 为newdatasets上的history结果
        '''
        if y is None or len(y) < 2:
            return self.rho
        disordered_pairs = total_pairs = 0
        for i in range(len(y)):
            for j in range(i+1, len(y)): # FIXME
                if (y[i] < y[j] != self.gps[d].cached_predict(
                        x[i]) < self.gps[d].cached_predict(x[j])):
                    disordered_pairs += 1
                total_pairs += 1
        t = disordered_pairs / total_pairs / self.bandwidth
        return self.rho * (1 - t * t) if t < 1 else 0
