import numpy as np
from scipy import stats
from xbbo.acquisition_function.base import AbstractAcquisitionFunction

class EI():
    def __init__(self, surrogate, y_best):
        self.eta = 0.0
        self.surrogate = surrogate
        self.y_best = y_best

    def __call__(self, candidate): #
        mu, sigma = self.surrogate.predict_with_sigma(candidate)
        z = (self.y_best - mu - self.eta) / sigma
        ei = (self.y_best - mu -
              self.eta) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    # def argmax(self, y_best, surrogate, candidates):
    #     best_ei = -1
    #     best_candidate = []
    #     for candidate in candidates:
    #         y_hat = surrogate.predict(candidate)
    #         ei = self._getEI(y_hat[0], y_hat[1], y_best)
    #         if ei > best_ei:
    #             best_ei = ei
    #             best_candidate = [candidate]
    #         elif ei == best_ei:
    #             best_candidate.append(candidate)
    #     return np.random.choice(best_candidate)

    def argmax(self, candidates):
        best_ei = -1
        # best_candidate = []
        candidates_rm_id = []
        # y_hats = list(zip(*surrogate.predict_with_sigma(candidates)))
        for i, candidate in enumerate(candidates):

            ei = self.__call__(candidate)
            if ei > best_ei:
                best_ei = ei
                # best_candidate = [candidate]
                candidates_rm_id = [i]
            elif ei == best_ei:
                # best_candidate.append(candidate)
                candidates_rm_id.append(i)

        assert candidates_rm_id
        idx = np.random.choice(len(candidates_rm_id))
        return candidates_rm_id[idx]

class EI_AcqFunc(AbstractAcquisitionFunction):
    def __init__(self, surrogate_model, rng):
        self.par = 0.0
        self.rng = rng
        super().__init__(surrogate_model)

    def argmax(self, candidates):
        print('depreciate!')
        # best_ei = -1
        # # best_candidate = []
        # candidates_rm_id = []
        # y_hats = list(zip(*surrogate.predict_with_sigma(candidates)))
        scores = self.__call__(candidates)
        
        return candidates[self.rng.choice(np.where(scores==scores.max())[0])]

    def _compute(self, X: np.ndarray, **kwargs) -> np.ndarray:
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
            X = X[:, np.newaxis]

        m, v = self.surrogate_model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.y_best is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.y_best - m - self.par) / s
            return (self.y_best - m - self.par) * stats.norm.cdf(z) + s * stats.norm.pdf(z)

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

        return f

# class MC_AcqFunc(AbstractAcquisitionFunction):
#     def __init__(self, surrogate_model, rng):
#         self.sample_num = 10
#         self.rng = rng
#         super().__init__(surrogate_model)

#     def _compute(self, X: np.ndarray) -> np.ndarray:
#         """Computes the EI value and its derivatives.

#         Parameters
#         ----------
#         X: np.ndarray(N, D), The input points where the acquisition function
#             should be evaluated. The dimensionality of X is (N, D), with N as
#             the number of points to evaluate at and D is the number of
#             dimensions of one X.

#         Returns
#         -------
#         np.ndarray(N,1)
#             Expected Improvement of X
#         """
#         if len(X.shape) == 1:
#             X = X[:, np.newaxis]
#         m, v = self.surrogate_model.predict_marginalized_over_instances(X, 'full_cov')
#         f = self.rng.multivariate_normal(m, v, size=self.sample_num)

#         return f # shape: (sample_num, N)
