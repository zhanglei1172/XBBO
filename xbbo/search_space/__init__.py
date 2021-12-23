import importlib


def build_test_problem(filename, cfg):

    if filename.endswith(".py"):
        filename = filename[:-3]
    # model = importlib.import_module('..custom_model.'+model_name, __package__)
    model = importlib.import_module('.' + filename, __package__)
    test_problem = model.Model(cfg, **dict(cfg.TEST_PROBLEM.kwargs))
        # prob = SklearnSurrogate(model_name, dataset, scorer, path=path)  # pragma: io
    return test_problem