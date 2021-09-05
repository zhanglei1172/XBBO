import numpy as np

import GPy

from bbomark.surrogate.base import Surrogate
from bbomark.surrogate.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessRegressorARD_sklearn,
    GaussianProcessRegressorARD_gpy
)


class RGPE_surrogate(Surrogate):

    def __init__(self, dim, rank_sample_num):
        super().__init__(dim)
        self.sparse = True
        self.rank_sample_num = rank_sample_num
        # self.iter = 0
        self.dim = dim
        # self.new_gp = GaussianProcessRegressor(dim)
        self.new_gp = GaussianProcessRegressorARD_gpy(dim)
        # self.candidates = None
        # self.bandwidth = bandwidth
        # self.history_x = []
        # self.history_y = []

    def _compute_ranking_loss(self, f_samples, f_target):
        '''
        f_samples 'n_samples x (n) x n' -dim
        '''
        if f_samples.ndim == 3:  # for target model
            rank_loss = ((f_samples.diagonal(axis1=-2, axis2=-1)[:, :, None] < f_samples) ^ (
                    np.expand_dims(f_target, axis=-1) < np.expand_dims(f_target, axis=-2)
            )).sum(axis=-1).sum(axis=-1)
        else:
            rank_loss = ((np.expand_dims(f_samples, axis=-1) < np.expand_dims(f_samples, axis=-2)) ^ (
                    np.expand_dims(f_target, axis=-1) < np.expand_dims(f_target, axis=-2)
            )).sum(axis=-1).sum(axis=-1)
        return rank_loss

    def _get_min_index(self, array):
        best_model_idxs = np.zeros(array.shape[1], dtype=np.int64)
        is_best_model = (array == array.min(axis=0))
        # idxs = np.argwhere(is_best_model)
        # mod, samp = idxs[:,0], idxs[:, 1]
        # np.random.randint(np.bincount())
        for i in range(array.shape[1]):
            if is_best_model[-1, i]:
                best_model_idxs[i] = self.old_D_num
            else:
                best_model_idxs[i] = np.random.choice(np.argwhere(is_best_model[:,i]).ravel().tolist())
        return best_model_idxs

    def _get_weight(self, t_x, t_y):
        ranking_losses = []
        for d in range(self.old_D_num):
            mean, cov = self.gps[d].cached_predict_with_cov(t_x)
            ranking_losses.append(self._compute_ranking_loss(
                np.random.multivariate_normal(mean, cov, size=self.rank_sample_num), t_y))
        ranking_losses.append(self._compute_ranking_loss(self._get_loocv_preds(t_x, t_y), t_y))
        ranking_losses = np.array(ranking_losses)
        ranking_losses = self._delete_noise_knowledge(ranking_losses)
        # TODO argmin 多个最小处理
        # best_model_idxs = ranking_losses.argmin(axis=0)  # per sample都有一个best idx
        best_model_idxs = self._get_min_index(ranking_losses)

        # rank_weight = np.bincount(best_model_idxs, minlength=ranking_losses.shape[0]) / self.rank_sample_num
        rank_weight = np.bincount(best_model_idxs, minlength=ranking_losses.shape[0])

        return rank_weight / rank_weight.sum()

    def _get_loocv_preds(self, x, y):
        try_num = len(y)
        masks = ~np.eye(try_num, dtype=np.bool_)
        x_cv = [x[m] for m in masks]
        y_cv = [y[m] for m in masks]
        samples = []
        for i in range(len(y)):
            # kernel = self.new_gp.kernel.copy()
            model = GPy.models.gp_regression.GPRegression(x_cv[i], y_cv[i][:,None], kernel=self.new_gp.kernel)
            mean, cov = model.predict(x, full_cov=True, kern=None)  # TODO 使用kernel子块
            # model = GaussianProcessRegressor(self.dim)
            # model.fit(x_cv[i], y_cv[i][:, None])
            # mean, cov = model.predict_with_cov(x)
            samples.append(np.random.multivariate_normal(np.squeeze(mean), cov, self.rank_sample_num))
        return np.stack(samples, axis=1)

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
        return candidates

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self.new_gp.fit(x, y)
        self.rank_weight = self._get_weight(x, y)
        # self.iter += 1

    def predict_with_sigma(self, newX):
        models = [self.gps[d].cached_predict_with_sigma(newX) for d in range(self.old_D_num)]
        models.append(self.new_gp.predict_with_sigma(newX))
        models = np.asarray(models)
        mu = self.rank_weight.dot(models[:, 0])
        sigma = (self.rank_weight ** 2).dot(models[:, 1] ** 2)

        return mu, np.sqrt(sigma)

    def _delete_noise_knowledge(self, ranking_losses):
        if self.sparse:
            target_loss = ranking_losses[-1]
            threshold = np.percentile(target_loss, 95)
            mask_to_remove = np.median(ranking_losses, axis=1) > threshold  # 中位数样本loss 比 95% 的target model结果差
            idx_to_remove = np.argwhere(mask_to_remove).ravel()
            ranking_losses = np.delete(ranking_losses, idx_to_remove, axis=0)
            # ranking_losses[idx_to_remove] = self.rank_sample_num
            for idx in reversed(idx_to_remove): # TODO Remove permanent
                self.gps.pop(idx)
            self.old_D_num -= len(idx_to_remove)
        return ranking_losses
