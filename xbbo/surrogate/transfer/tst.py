import typing
import numpy as np
# import torch
# from botorch.models.gpytorch import GPyTorchModel
# from gpytorch.distributions import MultivariateNormal
# from gpytorch.models import GP
# from gpytorch.likelihoods import LikelihoodList
# from torch.nn import ModuleList
from xbbo.surrogate.gaussian_process import GPR_sklearn
from xbbo.surrogate.transfer.base_surrogate import BaseModel
# from xbbo.surrogate.base import Surrogate
# from xbbo.utils.util import get_types
# from xbbo.surrogate.gaussian_process import (
#     GaussianProcessRegressor,
#     GaussianProcessRegressorARD_sklearn,
#     GaussianProcessRegressorARD_gpy, GaussianProcessRegressorARD_torch
# )
VERY_SMALL_NUMBER = 1e-10

# class TST_surrogate_(GP, GPyTorchModel):
# num_outputs = 1

# def __init__(self, models, weights):
#     super().__init__()
#     self.models = ModuleList(models)
#     for m in models:
#         if not hasattr(m, "likelihood"):
#             raise ValueError(
#                 "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
#             )
#     self.likelihood = LikelihoodList(*[m.likelihood for m in models])
#     self.weights = weights
#     self.to(weights)
#     # self.candidates = None
#     # self.bandwidth = bandwidth

# def forward(self, x):
#     weighted_means = []
#     non_zero_weight_indices = (self.weights**2 > 0).nonzero()
#     non_zero_weights = self.weights[non_zero_weight_indices]
#     for m_id in range(non_zero_weight_indices.shape[0]):
#         model = self.models[non_zero_weight_indices[m_id]]
#         posterior = model.posterior(x)
#         # posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
#         posterior_mean = posterior.mean.squeeze(-1)
#         # apply weight
#         weight = non_zero_weights[m_id]
#         weighted_means.append(weight * posterior_mean)
#     mean_x = torch.stack(weighted_means).sum(
#         dim=0) / non_zero_weights.sum()
#     posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(
#         2)
#     return MultivariateNormal(mean_x, posterior_cov)


class TST_surrogate(GPR_sklearn):
    def __init__(
            self,
            cs,
            #  min_sample=3,
            # alpha=0,
            base_models: typing.List[BaseModel],
            rng=np.random.RandomState(0),
            n_opt_restarts: int = 10,
            instance_features: typing.Optional[np.ndarray] = None,
            pca_components: typing.Optional[int] = None,
            **kwargs):
        super().__init__(cs,
                         rng,
                         n_opt_restarts,
                         instance_features=instance_features,
                         pca_components=pca_components,
                         **kwargs)
        self.base_models = base_models
        
    def target_model_predict(self, X: np.ndarray, cov_return_type: typing.Optional[str] = 'diagonal_cov') -> typing.Tuple[np.ndarray, np.ndarray]:
        return super()._predict(X, cov_return_type=cov_return_type)

    def update_weight(self, w, rho=None):
        assert w.min() >= 0
        if rho:
            self.w = np.array(w, dtype='float')
            self.selfWeight = rho
        else:
            self.selfWeight = w[-1]
            self.w = np.array(w[:-1], dtype='float')
        #non_zero_weight_indices = (self.w**2 > 0).nonzero()[0]
        non_zero_weight_indices = self.w.nonzero()[0]
        self.pre_weight_model = []
        for idx in non_zero_weight_indices:
            self.pre_weight_model.append(self.base_models[idx])
        self.weight = self.w[non_zero_weight_indices]
        norm_coeff = (self.weight.sum() + self.selfWeight)
        self.weight /= norm_coeff
        self.selfWeight /= norm_coeff

    def _predict(self,
                 X_test,
                 cov_return_type: typing.Optional[str] = 'diagonal_cov'):
        '''
        return: \mu ,\sigma^2
        '''
        assert self.is_fited
        # denominator = self.selfWeight

        X_test = self._impute_inactive(X_test)
        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None
            mu *= self.selfWeight
            for d in range(len(self.weight)):
                mu += self.weight[d] * self.pre_weight_model[
                    d]._predict_normalize(X_test, cov_return_type=None)[0]
                # denominator += self.weight[d]
            # mu /= denominator
            if self.normalize_y:
                mu = self._untransform_y(mu)

        else:
            predict_kwargs = {'return_cov': False, 'return_std': True}
            if cov_return_type == 'full_cov':
                predict_kwargs = {'return_cov': True, 'return_std': False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)
            mu *= self.selfWeight
            for d in range(len(self.weight)):
                mu += self.weight[d] * self.pre_weight_model[
                    d]._predict_normalize(X_test, cov_return_type=None)[0]
                # denominator += self.weight[d]
            # mu /= denominator

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

    # def kendallTauCorrelation(self, d, x, y):
    #     '''
    #     计算第d个datasets与new datasets的 相关性
    #     (x, y) 为newdatasets上的history结果
    #     '''
    #     if y is None or len(y) < 2:
    #         return self.rho
    #     disordered_pairs = total_pairs = 0
    #     for i in range(len(y)):
    #         for j in range(i+1, len(y)): # FIXME
    #             if (y[i] < y[j] != self.gps[d].cached_predict(
    #                     x[i]) < self.gps[d].cached_predict(x[j])):
    #                 disordered_pairs += 1
    #             total_pairs += 1
    #     t = disordered_pairs / total_pairs / self.bandwidth
    #     return self.rho * (1 - t * t) if t < 1 else 0
