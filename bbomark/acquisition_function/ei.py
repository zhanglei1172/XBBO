import numpy as np
from scipy import stats

class EI():
    def __init__(self):
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best): #
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
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

    def argmax(self, y_best, surrogate, candidates):
        best_ei = -1
        best_candidate = []
        candidates_rm_id = []
        for i, candidate in enumerate(candidates):
            y_hat = surrogate.predict(candidate)
            ei = self._getEI(y_hat[0], y_hat[1], y_best)
            if ei > best_ei:
                best_ei = ei
                best_candidate = [candidate]
                candidates_rm_id = [i]
            elif ei == best_ei:
                best_candidate.append(candidate)
                candidates_rm_id.append(i)

        assert best_candidate
        idx = np.random.choice(len(best_candidate))
        return (best_candidate)[idx], candidates_rm_id[idx]
