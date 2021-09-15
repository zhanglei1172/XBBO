from typing import Optional

import numpy as np
import torch
from botorch.acquisition import AnalyticAcquisitionFunction, ScalarizedObjective, ExpectedImprovement
# from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from gpytorch.likelihoods import LikelihoodList
from scipy import stats
from torch import Tensor
from torch.nn import ModuleList
import torch.nn.functional as F


class TAF_(ExpectedImprovement):
    def __init__(
            self,
            model,
            best_f,
            base_models,
            base_model_best,
            weights,
            objective: Optional[ScalarizedObjective] = None,
            maximize: bool = False,
    ) -> None:
        # model = ModuleList(base_models)
        # model.likelihood = LikelihoodList(*[m.likelihood for m in model])
        super().__init__(model=model, best_f=best_f, objective=objective, maximize=maximize)
        # self.maximize = maximize
        self.weights = weights
        self.base_models = base_models
        self.base_model_best = base_model_best

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        improvement = []
        non_zero_weight_indices = (self.weights ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights[non_zero_weight_indices]
        for m_id in range(non_zero_weight_indices.shape[0] - 1):
            model = self.base_models[non_zero_weight_indices[m_id]]
            posterior = model.posterior(X)
            # posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
            posterior_mean = posterior.mean.squeeze()
            # apply weight
            weight = non_zero_weights[m_id]
            if self.maximize:
                improvement.append(
                    weight * F.relu(posterior_mean - self.base_model_best[non_zero_weight_indices[m_id]]))
            else:
                improvement.append(
                    weight * F.relu(self.base_model_best[non_zero_weight_indices[m_id]] - posterior_mean))
        others = torch.stack(improvement).sum(dim=0) / non_zero_weights.sum() if improvement else 0
        return self.weights[-1] * super().forward(X) + others


class TAF():
    def __init__(self, gps, ):
        self.gps = gps
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best):  # minimize
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    def argmax(self, y_best, surrogate, candidates, similarity, old_ybests, rho=0.75):
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

class TAF_with_RGPE_rank():
    def __init__(self, gps, ):
        self.gps = gps
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best):  # minimize
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    def argmax(self, y_best, surrogate, candidates, similarity, old_ybests, rho=0.75):
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