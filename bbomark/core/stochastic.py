import numpy as np

class Category():
    '''
    every point has a instance of this class
    '''
    def __init__(self, choices, prior=None, prior_w=0):
        '''
        choice should be a int number
        '''
        self.choices = choices
        self.prior_w = prior_w
        if prior is None:
            self.prob = [1/self.choices for _ in range(self.choices)]
        else:
            self.prob = prior
        self._prior = self.prob

    def sample(self, p=None):
        return np.random.multinomial(1, self.prob if p is None else p, size=1)

    def log_pdf(self, x):
        return np.log(self.prob[x])

    def pdf(self, x):
        return self.prob[x]

    def random_sample(self):
        return np.random.choice(self.choices)

    def update(self, x, weights=None):
        counts = np.bincount(x, weights, minlength=self.choices)
        self.prob = (1 - self.prior_w) * counts + self.prior_w * self._prior


class Uniform():
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high

    def sample(self, low=None, high=None):
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        return np.random.uniform(low, high)

    def random_sample(self):
        return np.random.uniform(self.low, self.high)


class Normal():
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma

    def sample(self, mu=None, sigma=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        return np.random.normal(mu, sigma)

    def random_sample(self,  ):
        mu = self.mu
        sigma = self.sigma
        return np.random.normal(mu, sigma)
