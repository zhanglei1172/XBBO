from abc import ABC, abstractmethod
import numpy as np

class AbstractBaseModel(ABC):

    def __init__(self, **params):
        self.params = params
        pass

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def get_score(self, test_data, metric):
        '''
        score 指标 越大越好
        '''
        pass



class TestFunction(ABC):
    """Abstract base class for test functions in the benchmark. These do not need to be ML hyper-parameter tuning.
    """

    def __init__(self, seed=42):
        """Setup general test function for benchmark. We assume the test function knows the meta-data about the search
        configspace, but is also stateless to fit modeling assumptions. To keep stateless, it does not do things like count
        the number of function evaluations.
        """
        # This will need to be set before using other routines
        self.api_config = None
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def evaluate(self, params):
        """Abstract method to evaluate the function at a parameter setting.
        """

    def get_api_config(self):
        """Get the API config for this test problem.

        Returns
        -------
        api_config : dict(str, dict(str, object))
            The API config for the used model. See README for API description.
        """
        assert self.api_config is not None, "API config is not set."
        assert isinstance(self.api_config, dict), "API config is not a dict type"
        return self.api_config

    def _load_data(self, dataset, data_root=None):
        pass

    def _load_api_config(self):
        pass

    def _load_base_model(self):
        '''
        return a class
        '''
        pass