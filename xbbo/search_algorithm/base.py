from abc import ABC, abstractmethod
# from xbbo.configspace.space import Configurations

class AbstractOptimizer(ABC):
    """Abstract base class for the optimizers in the benchmark. This creates a common API across all packages.
    """

    # Every implementation package needs to specify this static variable, e.g., "primary_import=opentuner"
    primary_import = None

    def __init__(self, config_spaces, **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        # self.api_config = api_config
        self.space = config_spaces
        # self.feature_spaces = feature_spaces
        # if not (feature_spaces is None):
        #     self.feature_spaces.dtypes_idx_map = self.space.dtypes_idx_map
        # self.warp = warp
        # if logger is None:
        #     self.logger = logging.getLogger('xbbo')
        # else:
        #     self.logger = logger

    # # @abstractmethod
    # def transform_sparseArray_to_optSpace(self, sparse_array):
    #     return sparse_array

    @classmethod
    def get_version(cls):
        """Get the version for this optimizer.

        Returns
        -------
        version_str : str
            Version number of the optimizer. Usually, this is equivalent to ``package.__version__``.
        """
        assert (cls.primary_import is None) or isinstance(cls.primary_import, str)
        # Should use x.x.x as version if sub-class did not specify its primary import
        # version_str = "x.x.x" if cls.primary_import is None else version(cls.primary_import)
        version_str = "1.0.0"
        return version_str



    @abstractmethod
    def suggest(self, n_suggestions): # output [meta param]
        """Get a suggestion from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        x_guess_configs = self.space.sample_configuration(size=n_suggestions)
        x_guess = [x_guess_config.get_dictionary() for x_guess_config in x_guess_configs]
        return x_guess



    def observe(self, trials, y): # input [meta param]
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        features : list of features
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        pass
