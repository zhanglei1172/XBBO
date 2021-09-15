import glob
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from matplotlib import pyplot as plt
from botorch.optim import optimize_acqf
import tqdm, random

from bbomark.acquisition_function.ei import EI
from bbomark.configspace.feature_space import FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations

from bbomark.core.trials import Trials
from bbomark.surrogate import get_fitted_model
from bbomark.surrogate.gaussian_process import GaussianProcessRegressor, GaussianProcessRegressorARD_gpy, \
    GaussianProcessRegressorARD_torch
from bbomark.surrogate.tst import TST_surrogate_


class SMBO(AbstractOptimizer, FeatureSpace_uniform):

    def __init__(self,
                 config_spaces,
                 min_sample=4,
                 noise_std=0.01,
                 rho=0.75,
                 bandwidth=0.1,
                 mc_samples=256,
                 raw_samples=100
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.min_sample = min_sample
        self.candidates = None
        # self.avg_best_idx = avg_best_idx
        # self.meta_data_path = meta_data_path
        configs = self.space.get_hyperparameters()
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.hp_num = len(configs)
        self.bounds_tensor = torch.stack([
            torch.zeros(self.hp_num),
            torch.ones(self.hp_num)
        ])
        self.trials = Trials()
        # self.surrogate = GaussianProcessRegressor(self.hp_num)
        # self.surrogate = GaussianProcessRegressorARD_torch(self.hp_num, self.min_sample)
        self.acq_class = ExpectedImprovement
        self.noise_std = noise_std
        self.rho = rho
        self.bandwidth = bandwidth
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples

    def prepare(self, old_D_x_params, old_D_y, new_D_x_param, sort_idx=None, params=True):
        if params:
            old_D_x = []
            for insts_param in old_D_x_params:
                insts_feature = []
                for inst_param in insts_param:
                    array = Configurations.dictUnwarped_to_array(self.space, inst_param)
                    insts_feature.append(self.array_to_feature(array, self.dense_dimension))
                old_D_x.append(np.asarray(insts_feature))
            insts_feature = []
            if new_D_x_param:
                for inst_param in new_D_x_param:
                    array = Configurations.dictUnwarped_to_array(self.space, inst_param)
                    insts_feature.append(self.array_to_feature(array, self.dense_dimension))
                new_D_x = (np.asarray(insts_feature))
                self.candidates = new_D_x
            else:
                self.candidates = None
        else:
            old_D_x = []
            for insts_param in old_D_x_params:
                # insts_feature = []
                old_D_x.append(insts_param[:, sort_idx])

            if new_D_x_param is not None:
                new_D_x = new_D_x_param[:, sort_idx]
                self.candidates = new_D_x
            else:
                self.candidates = None

        self.old_D_num = len(old_D_x)
        self.gps = []
        for d in range(self.old_D_num):
            # self.gps.append(GaussianProcessRegressor())
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=50)
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=len())
            observed_idx = list(range(len(old_D_y[d])))
            # observed_idx = np.random.choice(len(old_D_y[d]), size=50, replace=False)
            x = torch.Tensor(old_D_x[d][observed_idx,:]) # TODO
            y = torch.Tensor(old_D_y[d][observed_idx])
            train_yvar = torch.full_like(y, self.noise_std ** 2)
            self.gps.append(get_fitted_model(x, y, train_yvar))
            # g = GaussianProcessRegressorARD_torch(self.hp_num)
            # self.gps.append(g.fit(x, y.squeeze()))
        # print(1)
        # if new_D_x is not None:
        #     candidates = new_D_x
        # else:  #
        #     raise NotImplemented
        # self.candidates = candidates


    def kendallTauCorrelation(self, base_model_means, y):
        if y is None or len(y) < 2:
            return torch.full(base_model_means.shape[0], self.rho)
        rank_loss = (base_model_means.unsqueeze(-1) < base_model_means.unsqueeze(-2)) ^ (
                y.unsqueeze(-1) < y.unsqueeze(-2))
        t = rank_loss.float().mean(dim=-1).mean(dim=-1) / self.bandwidth
        return (t < 1) * (1 - t * t) * self.rho
        # return self.rho * (1 - t * t) if t < 1 else 0

    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) < self.min_sample:
            # raise NotImplemented
            return self._random_suggest()
        else:
            x_unwarpeds = []
            sas = []
            for n in range(n_suggestions):
                surrogate = TST_surrogate_(self.gps + [self.target_model],
                                           torch.cat([self.similarity, torch.Tensor([self.rho])]))

                acq = self.acq_class(surrogate, self.y.min(), maximize=False)
                # acq = self.acq_class(self.surrogate.gpr,
                #                      self.surrogate.transform_outputs(
                #                          np.asarray(self.trials.history_y)[..., None]).min().astype(np.float32), maximize=False)
                if self.candidates is None:
                    # optimize
                    candidate, acq_value = optimize_acqf(
                        acq, bounds=self.bounds_tensor, q=1, num_restarts=5, raw_samples=self.raw_samples,
                    )
                    suggest_array = candidate[0].detach().cpu().numpy()
                    x_array = self.feature_to_array(suggest_array, self.sparse_dimension)
                    x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)

                    sas.append(suggest_array)
                    x_unwarpeds.append(x_unwarped)
                else:
                    with torch.no_grad():
                        ei = acq(torch.Tensor(self.candidates).unsqueeze(dim=-2))
                    rm_id = ei.argmax()
                    suggest_array = self.candidates[rm_id]
                    self.candidates = np.delete(self.candidates, rm_id, axis=0) # TODO
                    x_array = self.feature_to_array(suggest_array, self.sparse_dimension)
                    x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)

                    sas.append(suggest_array)
                    x_unwarpeds.append(x_unwarped)
        # x = [Configurations.array_to_dictUnwarped(self.space,
        #                                           np.asarray(sa)) for sa in sas]
        self.trials.params_history.extend(x_unwarpeds)
        return x_unwarpeds, sas

    def observe(self, x, y):
        # print(y)
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        if len(self.trials.history_y) < self.min_sample:
            return
        self.is_fited = True
        x = torch.Tensor(self.trials.history)
        self.y = torch.Tensor(self.trials.history_y).unsqueeze(-1)
        train_yvar = torch.full_like(self.y, self.noise_std ** 2)
        self.target_model = get_fitted_model(x, self.y, train_yvar)
        with torch.no_grad():
            base_model_means = []
            for d in range(len(self.gps)):
                base_model_means.append(self.gps[d].posterior(x).mean)
            base_model_means = torch.stack(base_model_means)  # [model, obs_num, 1]
        # target_model_mean = self.target_model.posterior(x).mean
        self.similarity = self.kendallTauCorrelation(base_model_means.squeeze(), self.y.squeeze())

    def _random_suggest_explore(self, n_suggestions=1):
        sas = []
        x_unwarpeds = []
        for n in range(n_suggestions):
            rm_id = np.random.choice(len(self.candidates))
            sas.append(self.candidates[rm_id])
            x_array = self.feature_to_array(sas[-1], self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            x_unwarpeds.append(x_unwarped)
            self.candidates = np.delete(self.candidates, rm_id, axis=0)
        return x_unwarpeds, sas

    def _random_suggest(self, n_suggestions=1):
        sas = []
        x_unwarpeds = []
        if self.candidates is not None:
            for n in range(n_suggestions):
                rm_id = np.random.randint(low=0, high=len(self.candidates))
                sas.append(self.candidates[rm_id])
                x_array = self.feature_to_array(sas[-1], self.sparse_dimension)
                x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
                x_unwarpeds.append(x_unwarped)
                self.candidates = np.delete(self.candidates, rm_id, axis=0) # TODO
        else:
            x_unwarpeds = (self.space.sample_configuration(n_suggestions))
            for n in range(n_suggestions):

                array = Configurations.dictUnwarped_to_array(self.space, x_unwarpeds[-1])
                sas.append(self.array_to_feature(array, self.dense_dimension))
        return x_unwarpeds, sas

opt_class = SMBO
