import importlib

from .innermodel import SklearnModel
import bbomark.model.custom_model as custom_model
from bbomark.utils.util import chomp


def build_test_problem(model_name, dataset, scorer, path):
    """Build the class with the class to use an objective. Sort of a factory.

    Parameters
    ----------
    model_name : str
        Which sklearn model we are attempting to tune, must be an element of `constants.MODEL_NAMES`.
    dataset : str
        Which data set the model is being tuned to, which must be either a) an element of
        `constants.DATA_LOADER_NAMES`, or b) the name of a csv file in the `data_root` folder for a custom data set.
    scorer : str
        Which metric to use when evaluating the model. This must be an element of `sklearn_funcs.SCORERS_CLF` for
        classification models, or `sklearn_funcs.SCORERS_REG` for regression models.
    path : str or None
        Absolute path to folder containing custom data sets/pickle files with surrogate model.

    Returns
    -------
    prob : :class:`.sklearn_funcs.TestFunction`
        The test function to evaluate in experiments.
    """
    if model_name.endswith(".py"):
        # Requires IO to test these, so will add the pargma here. Maybe that points towards a possible design change.
        # raise NotImplementedError()
        model_name = chomp(model_name, ".py")  # pragma: io
        model = importlib.import_module('.'+model_name, __package__)
        prob = model.Model(model_name, dataset, scorer, data_root=path)
        # prob = SklearnSurrogate(model_name, dataset, scorer, path=path)  # pragma: io
    else:
        prob = SklearnModel(model_name, dataset, scorer, data_root=path)
    return prob