# import importlib

from xbbo.core.register import Register


problem_register = Register("all avaliable black box problem")

from . import offline_hp
from . import toy_function

# def build_test_problem(filename, cfg, seed=42):

#     if filename.endswith(".py"):
#         filename = filename[:-3]
#     # model = importlib.import_module('..custom_model.'+model_name, __package__)
#     model = importlib.import_module('.' + filename, __package__)
#     test_problem = model.Model(cfg, seed=seed, **dict(cfg.TEST_PROBLEM.kwargs))
#         # prob = SklearnSurrogate(model_name, dataset, scorer, path=path)  # pragma: io
#     return test_problem