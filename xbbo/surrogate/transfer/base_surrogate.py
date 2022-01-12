import typing
import numpy as np

from xbbo.utils.constants import VERY_SMALL_NUMBER
from xbbo.surrogate.gaussian_process import GPR_sklearn

class BaseModel(GPR_sklearn):
    def __init__(
        self,
        cs,
        rng=np.random.RandomState(0),
        n_opt_restarts: int = 10,
        instance_features: typing.Optional[np.ndarray] = None,
        pca_components: typing.Optional[int] = None,
        **kwargs
    ):
        super().__init__(cs, rng, n_opt_restarts,instance_features=instance_features,
            pca_components=pca_components,**kwargs)
        self.cached = {}
        self.cached_normalize = {}

    def _predict_normalize(self, X_test, cov_return_type: typing.Optional[str] = 'diagonal_cov'):
        assert self.is_fited
        # key = hash(X_test.data.tobytes()+bytes(cov_return_type if cov_return_type else '', 'utf-8'))
        # if key in self.cached_normalize:
        #     return self.cached_normalize[key]
        
        X_test = self._impute_inactive(X_test)
        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None

            # if self.normalize_y:
            #     mu = self._untransform_y(mu)

        else:
            predict_kwargs = {'return_cov': False, 'return_std': True}
            if cov_return_type == 'full_cov':
                predict_kwargs = {'return_cov': True, 'return_std': False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)

            if cov_return_type != 'full_cov':
                var = var**2  # since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

            # if self.normalize_y:
            #     mu, var = self._untransform_y(mu, var)

            if cov_return_type == 'diagonal_std':
                var = np.sqrt(
                    var)  # converting variance to std deviation if specified
        # self.cached_normalize[key] = (mu, var)
        
        return mu, var
    
    def _predict(self, X_test, cov_return_type: typing.Optional[str] = 'diagonal_cov'):
        assert self.is_fited
        # key = hash(X_test.data.tobytes()+bytes(cov_return_type if cov_return_type else '', 'utf-8'))
        # if key in self.cached:
        #     return self.cached[key]
        
        X_test = self._impute_inactive(X_test)
        if cov_return_type is None:
            mu = self.gp.predict(X_test)
            var = None

            if self.normalize_y:
                mu = self._untransform_y(mu)

        else:
            predict_kwargs = {'return_cov': False, 'return_std': True}
            if cov_return_type == 'full_cov':
                predict_kwargs = {'return_cov': True, 'return_std': False}

            mu, var = self.gp.predict(X_test, **predict_kwargs)

            if cov_return_type != 'full_cov':
                var = var**2  # since we get standard deviation for faster computation

            # Clip negative variances and set them to the smallest
            # positive float value
            var = np.clip(var, VERY_SMALL_NUMBER, np.inf)

            if self.normalize_y:
                mu, var = self._untransform_y(mu, var)

            if cov_return_type == 'diagonal_std':
                var = np.sqrt(
                    var)  # converting variance to std deviation if specified
        # self.cached[key] = (mu, var)
        
        return mu, var   

    # def _cached_predict(self, X_test, cov_return_type: typing.Optional[str] = 'diagonal_cov'):
    #     key = hash(X_test.data.tobytes()+bytes(cov_return_type if cov_return_type else '', 'utf-8'))
    #     if key in self.cached:
    #         return self.cached[key]
    #     if key not in self.cached:
    #         self.cached[key] = self._predict(X_test, cov_return_type)
    #     return self.cached[key]
    

