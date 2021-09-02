import numpy as np


from bbomark.surrogate.base import Surrogate
from bbomark.surrogate.gaussian_process import (
    GaussianProcessRegressor,
    GaussianProcessRegressorARD_sklearn,
    GaussianProcessRegressorARD_gpy
)


class TST_surrogate(Surrogate):
    rho = 0.75

    def __init__(self, dim, bandwidth=0.1):
        super().__init__(dim)

        # self.new_gp = GaussianProcessRegressor()
        self.new_gp = GaussianProcessRegressorARD_gpy(dim)
        # self.candidates = None
        self.bandwidth = bandwidth
        # self.history_x = []
        # self.history_y = []

    def get_knowledge(self, old_D_x, old_D_y, new_D_x=None):
        self.old_D_num = len(old_D_x)
        self.gps = []
        for d in range(self.old_D_num):
            # self.gps.append(GaussianProcessRegressor())
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
        self.new_gp.fit(x, y)
        self.similarity = [self.kendallTauCorrelation(d, x, y) for d in range(self.old_D_num)]

    def predict_with_sigma(self, newX):
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
            for j in range(len(y)):
                if (y[i] < y[j] != self.gps[d].cached_predict(
                        x[i]) < self.gps[d].cached_predict(x[j])):
                    disordered_pairs += 1
                total_pairs += 1
        t = disordered_pairs / total_pairs / self.bandwidth
        return self.rho * (1 - t * t) if t < 1 else 0
