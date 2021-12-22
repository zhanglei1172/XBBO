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


class MoGP_(ExpectedImprovement):
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
        self.base_ei = []
        for i in range(len(base_models)):
            self.base_ei.append(ExpectedImprovement(base_models[i], base_model_best[i], maximize=maximize))
        self.weights = weights
        # self.base_models = base_models
        # self.base_model_best = base_model_best

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        Eimprovement = []
        non_zero_weight_indices = (self.weights ** 2 > 0).nonzero()[0]
        non_zero_weights = self.weights[non_zero_weight_indices]
        for m_id in range(non_zero_weight_indices.shape[0] - 1):
            # model = self.base_models[non_zero_weight_indices[m_id]]
            # posterior = model.posterior(X)
            # posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
            # posterior_mean = posterior.mean.squeeze()
            # apply weight
            weight = non_zero_weights[m_id]
            Eimprovement.append(
                weight * self.base_ei[non_zero_weight_indices[m_id]](X))
        others = torch.stack(Eimprovement).sum(dim=0) / non_zero_weights.sum() if Eimprovement else 0
        return self.weights[-1] * super().forward(X) + others
