from typing import Optional, Tuple, Union
import numpy as np

class Surrogate():
    def __init__(self, dim, min_sample):
        self.dim = dim
        self.min_sample = min_sample
        pass

    def predict(self, newX):
        pass

    def fit(self, x, y):
        pass

    def _normalize_y(self, y:np.ndarray) -> np.ndarray:
        self.mean_y = np.mean(y)
        self.std_y = np.std(y)
        if self.std_y == 0:
            self.std_y = 1
        return (y - self.mean_y) / self.std_y

    def _untransform_y(
        self,
        y: np.ndarray,
        var: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Transform zeromean unit standard deviation data into the regular space.

        This function should be used after a prediction with the Gaussian process which was trained on normalized data.

        Parameters
        ----------
        y : np.ndarray
            Normalized data.
        var : np.ndarray (optional)
            Normalized variance

        Returns
        -------
        np.ndarray on Tuple[np.ndarray, np.ndarray]
        """
        y = y * self.std_y_ + self.mean_y_
        if var is not None:
            var = var * self.std_y_ ** 2
            return y, var
        return y

    def _impute_inactive(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[~np.isfinite(X)] = -1
        return X