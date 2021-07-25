import numpy as np


class Record:

    def __init__(self, n_suggestions, n_calls=None):
        self.n_calls = n_calls
        self.features = []
        self.targets = []
        self.budgets = []
        self.timing = {
            'suggest_time': [],
            'observe_time': [],
            'eval_time': [] # n_calls, n_suggestions
        }

    def __str__(self):
        return np.asarray(self.targets).__str__()

    def size(self):
        return len(self.targets)

    def append(self, x, y, timing, b=None):
        self.features.append(x)
        self.targets.append(y)
        self.timing.update(timing)
        if b is not None:
            self.budgets.append(b)

    # def load_feature_matrix(self):
    #     return np.vstack(self.features)
    #
    # def load_target_vector(self):
    #     return np.hstack(self.targets)
    #
    # def load_regression_data(self):
    #     X = self.load_feature_matrix()
    #     y = self.load_target_vector()
    #     return X, y
    #
    # def load_classification_data(self, gamma):
    #     X, y = self.load_regression_data()
    #     tau = np.quantile(y, q=gamma)
    #     z = np.less(y, tau)
    #     return X, z

    # def to_dataframe(self):
    #     frame = pd.DataFrame(data=self.features).assign(budget=self.budgets,
    #                                                     loss=self.targets)
    #     return frame

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        # Clever ways of doing this would involve data structs. like KD-trees
        # or locality sensitive hashing (LSH), but these are premature
        # optimizations at this point, especially since the `any` below does lazy
        # evaluation, i.e. is early stopped as soon as anything returns `True`.
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol)
                   for x_prev in self.features)
