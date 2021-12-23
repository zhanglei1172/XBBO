import numpy as np
from scipy import stats

class EI():
    def __init__(self, surrogate, y_best):
        self.xi = 0.0
        self.surrogate = surrogate
        self.y_best = y_best

    def __call__(self, candidate): #
        mu, sigma = self.surrogate.predict_with_sigma(candidate)
        z = (self.y_best - mu - self.xi) / sigma
        ei = (self.y_best - mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
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

class EI_():
    def __init__(self, rng):
        self.xi = 0.0
        self.rng = rng
    
    def update(self, surrogate, y_best):
        self.surrogate = surrogate
        self.y_best = y_best

    def __call__(self, candidates): # minimize
        mu, var = self.surrogate.predict(candidates)
        sigma = np.sqrt(var)
        z = (self.y_best - mu - self.xi) / sigma
        ei = (self.y_best - mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
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
        # best_ei = -1
        # # best_candidate = []
        # candidates_rm_id = []
        # y_hats = list(zip(*surrogate.predict_with_sigma(candidates)))
        scores = self.__call__(candidates)
        
        return candidates[self.rng.choice(np.where(scores==scores.max())[0])]
