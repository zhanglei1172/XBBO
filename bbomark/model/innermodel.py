from abc import ABC, abstractmethod
import warnings
import numpy as np

from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score, train_test_split


# from data import load_data
from bbomark.data.data import METRICS_LOOKUP, ProblemType, load_data
from bbomark.constants import ARG_DELIM, METRICS, MODEL_NAMES, VISIBLE_TO_OPT
from bbomark.utils.util import str_join_safe

# Using 3 would be faster, but 5 is the most realistic CV split (5-fold)
CV_SPLITS = 5

from .config import MODELS_REG, MODELS_CLF

# If both classifiers and regressors match MODEL_NAMES then the experiment
# launcher can simply go thru the cartesian product and do all combos.
assert sorted(MODELS_REG.keys()) == sorted(MODEL_NAMES)


class TestFunction(ABC):
    """Abstract base class for test functions in the benchmark. These do not need to be ML hyper-parameter tuning.
    """

    def __init__(self):
        """Setup general test function for benchmark. We assume the test function knows the meta-data about the search
        configspace, but is also stateless to fit modeling assumptions. To keep stateless, it does not do things like count
        the number of function evaluations.
        """
        # This will need to be set before using other routines
        self.api_config = None

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
        return self.api_config


class SklearnModel(TestFunction):
    """Test class for sklearn classifier/regressor CV score objective functions.
    """

    # Map our short names for metrics to the full length sklearn name
    _METRIC_MAP = {
        "nll": "neg_log_loss",
        "acc": "accuracy",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }

    # This can be static and constant for now
    objective_names = (VISIBLE_TO_OPT, "generalization")

    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):
        """Build class that wraps sklearn classifier/regressor CV score for use as an objective function.

        Parameters
        ----------
        model : str
            Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
            classification or regression.
        dataset : str
            Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
        metric : str
            Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
            classification or regression.
        shuffle_seed : int
            Random seed to use when splitting the data into train and validation in the cross-validation splits. This
            is needed in order to keep the split constant across calls. Otherwise there would be extra noise in the
            objective function for varying splits.
        data_root : str
            Root directory to look for all custom csv files.
        """
        TestFunction.__init__(self)
        data, target, problem_type = load_data(dataset, data_root=data_root)
        assert problem_type in (ProblemType.clf, ProblemType.reg)
        self.is_classifier = problem_type == ProblemType.clf

        # Do some validation on loaded data
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert data.ndim == 2 and target.ndim == 1
        assert data.shape[0] == target.shape[0]
        assert data.size > 0
        assert data.dtype == np.float_
        assert np.all(np.isfinite(data))  # also catch nan
        assert target.dtype == (np.int_ if self.is_classifier else np.float_)
        assert np.all(np.isfinite(target))  # also catch nan

        model_lookup = MODELS_CLF if self.is_classifier else MODELS_REG
        base_model, fixed_params, api_config = model_lookup[model]

        # New members for model
        self.base_model = base_model
        self.fixed_params = fixed_params
        self.api_config = api_config

        # Always shuffle your data to be safe. Use fixed seed for reprod.
        self.data_X, self.data_Xt, self.data_y, self.data_yt = train_test_split(
            data, target, test_size=0.2, random_state=shuffle_seed, shuffle=True
        )

        assert metric in METRICS, "Unknown metric %s" % metric
        assert metric in METRICS_LOOKUP[problem_type], "Incompatible metric %s with problem type %s" % (
            metric,
            problem_type,
        )
        self.scorer = get_scorer(SklearnModel._METRIC_MAP[metric])

    def evaluate(self, params):
        """Evaluate the sklearn CV objective at a particular parameter setting.

        Parameters
        ----------
        params : dict(str, object)
            The varying (non-fixed) parameter dict to the sklearn model.

        Returns
        -------
        cv_loss : float
            Average loss over CV splits for sklearn model when tested using the settings in params.
        """
        params = dict(params)  # copy to avoid modification of original
        params.update(self.fixed_params)  # add in fixed params

        # now build the skl object
        clf = self.base_model(**params)

        assert np.all(np.isfinite(self.data_X)), "all features must be finite"
        assert np.all(np.isfinite(self.data_y)), "all targets must be finite"

        # Do the x-val, ignore user warn since we expect BO to try weird stuff
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            S = cross_val_score(clf, self.data_X, self.data_y, scoring=self.scorer, cv=CV_SPLITS)
        # Take the mean score across all x-val splits
        cv_score = np.mean(S)

        # Now let's get the generalization error for same hypers
        clf = self.base_model(**params)
        clf.fit(self.data_X, self.data_y)
        generalization_score = self.scorer(clf, self.data_Xt, self.data_yt)

        # get_scorer makes everything a score not a loss, so we need to negate to get the loss back
        cv_loss = -cv_score
        assert np.isfinite(cv_loss), "loss not even finite"
        generalization_loss = -generalization_score
        assert np.isfinite(generalization_loss), "loss not even finite"

        # Unbox to basic float to keep it simple
        cv_loss = cv_loss.item()
        assert isinstance(cv_loss, float)
        generalization_loss = generalization_loss.item()
        assert isinstance(generalization_loss, float)

        # For now, score with same objective. We can later add generalization error
        return cv_loss, generalization_loss

    @staticmethod
    def test_case_str(model, dataset, scorer):
        """Generate the combined test case string from model, dataset, and scorer combination."""
        test_case = str_join_safe(ARG_DELIM, (model, dataset, scorer))
        return test_case

    @staticmethod
    def inverse_test_case_str(test_case):
        """Inverse of `test_case_str`."""
        model, dataset, scorer = test_case.split(ARG_DELIM)
        assert test_case == SklearnModel.test_case_str(model, dataset, scorer)
        return model, dataset, scorer


# class SklearnSurrogate(TestFunction):
#     """Test class for sklearn classifier/regressor CV score objective function surrogates.
#     """
#
#     # This can be static and constant for now
#     objective_names = (VISIBLE_TO_OPT, "generalization")
#
#     def __init__(self, model, dataset, scorer, path):
#         """Build class that wraps sklearn classifier/regressor CV score for use as an objective function surrogate.
#
#         Parameters
#         ----------
#         model : str
#             Which classifier to use, must be key in `MODELS_CLF` or `MODELS_REG` dict depending on if dataset is
#             classification or regression.
#         dataset : str
#             Which data set to use, must be key in `DATA_LOADERS` dict, or name of custom csv file.
#         scorer : str
#             Which sklearn scoring metric to use, in `SCORERS_CLF` list or `SCORERS_REG` dict depending on if dataset is
#             classification or regression.
#         path : str
#             Root directory to look for all pickle files.
#         """
#         TestFunction.__init__(self)
#
#         # Find the configspace class, we could consider putting this in pkl too
#         problem_type = get_problem_type(dataset)
#         assert problem_type in (ProblemType.clf, ProblemType.reg)
#         _, _, self.api_config = MODELS_CLF[model] if problem_type == ProblemType.clf else MODELS_REG[model]
#         self.configspace = JointSpace(self.api_config)
#
#         # Load the pre-trained model
#         fname = SklearnModel.test_case_str(model, dataset, scorer) + ".pkl"
#
#         if isinstance(path, bytes):
#             # This is for test-ability, we could use mock instead.
#             self.model = pkl.loads(path)
#         else:
#             path = os.path.join(path, fname)  # pragma: io
#             assert os.path.isfile(path), "Model file not found: %s" % path
#
#             with absopen(path, "rb") as f:  # pragma: io
#                 self.model = pkl.load(f)  # pragma: io
#         assert callable(getattr(self.model, "predict", None))
#
#     def evaluate(self, params):
#         """Evaluate the sklearn CV objective at a particular parameter setting.
#
#         Parameters
#         ----------
#         params : dict(str, object)
#             The varying (non-fixed) parameter dict to the sklearn model.
#
#         Returns
#         -------
#         overall_loss : float
#             Average loss over CV splits for sklearn model when tested using the settings in params.
#         """
#         x = self.configspace.warp([params])
#         y, = self.model.predict(x)
#
#         assert y.shape == (len(self.objective_names),)
#         assert y.dtype.kind == "f"
#
#         assert np.all(-np.inf < y)  # Will catch nan too
#         y = tuple(y.tolist())  # Make consistent with SklearnModel typing
#         return y
