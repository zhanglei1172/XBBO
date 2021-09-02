import numpy as np
from scipy import stats

class TAF():
    def __init__(self, gps, ):
        self.gps = gps
        self.xi = 0.1

    def _getEI(self, mu, sigma, y_best): # minimize
        z = (-y_best + mu - self.xi) / sigma
        ei = (-y_best + mu -
              self.xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei


    def argmax(self, y_best, surrogate, candidates, similarity, old_ybests,rho=0.75):
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
                mu = self.gps[d].cached_predict(candidate) # TODO
                ei += similarity[d] * max(mu-old_ybests[d], 0)
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
