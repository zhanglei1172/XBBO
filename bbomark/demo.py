import numpy as np
import time, json

from constants import ARG_DELIM, ITER, OBJECTIVE, SUGGEST
from config import CONFIG
from cmd_parse import CmdArgs
import cmd_parse as cmd
from innermodel import SklearnModel
import random_search as rs

def _get_opt_class(opt_name):
    """Load the relevant wrapper class based on this optimizer name.

    There is inherently a bit ugly, but is only called at the main() level before the inner workings get going. There
    are a few ways to do this with some pro and con:
    1) The way done here: based on the filename, load that module via conditional imports and if-else. cons:
        - uses conditional imports
        - must manually repeat yourself in the if-else, but these are checked in unit testing
    2) Import everything and then pick the right optimizer based on a dict of name_str -> class. cons:
        - loads every dependency no matter which is used so could be slow
        - also a stupid dependency might change global state in a way that corrupts experiments
    3) Use the wrapper file as the entry point and add that to setup.py. cons:
        - Will clutter the CLI namespace with one command for each wrapper
    4) Use importlib to import the specified file. cons:
        - Makes assumptions about relative path structure. For pip-installed packages, probably safer to let python
        find the file via import.
    This option (1) seems least objectionable. However, this function could easily be switched to use importlib without
    any changes elsewhere.
    """
    wrapper_file, _ = CONFIG[opt_name]

    if wrapper_file == "hyperopt_optimizer.py":
        import hyperopt_optimizer as opt
    elif wrapper_file == "nevergrad_optimizer.py":
        import nevergrad_optimizer as opt
    elif wrapper_file == "opentuner_optimizer.py":
        import opentuner_optimizer as opt
    elif wrapper_file == "pysot_optimizer.py":
        import pysot_optimizer as opt
    elif wrapper_file == "random_optimizer.py":
        import random_optimizer as opt
    elif wrapper_file == "scikit_optimizer.py":
        import scikit_optimizer as opt
    else:
        assert False, "CONFIG for built in optimizers has added a new optimizer, but not updated this function."

    opt_class = opt.opt_wrapper
    return opt_class

def _build_test_problem(model_name, dataset, scorer, path):
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
    if model_name.endswith("-surr"):
        # Requires IO to test these, so will add the pargma here. Maybe that points towards a possible design change.
        raise NotImplementedError()
        # model_name = chomp(model_name, "-surr")  # pragma: io
        # prob = SklearnSurrogate(model_name, dataset, scorer, path=path)  # pragma: io
    else:
        prob = SklearnModel(model_name, dataset, scorer, data_root=path)
    return prob



def run_study(optimizer, test_problem, n_calls, n_suggestions, n_obj=1, callback=None):
    """Run a study for a single optimizer on a single test problem.

    This function can be used for benchmarking on general stateless objectives (not just `sklearn`).

    Parameters
    ----------
    optimizer : :class:`.abstract_optimizer.AbstractOptimizer`
        Instance of one of the wrapper optimizers.
    test_problem : :class:`.sklearn_funcs.TestFunction`
        Instance of test function to attempt to minimize.
    n_calls : int
        How many iterations of minimization to run.
    n_suggestions : int
        How many parallel evaluation we run each iteration. Must be ``>= 1``.
    n_obj : int
        Number of different objectives measured, only objective 0 is seen by optimizer. Must be ``>= 1``.
    callback : callable
        Optional callback taking the current best function evaluation, and the number of iterations finished. Takes
        array of shape `(n_obj,)`.

    Returns
    -------
    function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
        Value of objective for each evaluation.
    timing_evals : (:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`)
        Tuple of 3 timing results: ``(suggest_time, eval_time, observe_time)`` with shapes ``(n_calls,)``,
        ``(n_calls, n_suggestions)``, and ``(n_calls,)``. These are the time to make each suggestion, the time for each
        evaluation of the objective function, and the time to make an observe call.
    suggest_log : list(list(dict(str, object)))
        Log of the suggestions corresponding to the `function_evals`.
    """
    assert n_suggestions >= 1, "batch size must be at least 1"
    assert n_obj >= 1, "Must be at least one objective"

    # space_for_validate = JointSpace(test_problem.get_api_config()) # TODO

    if callback is not None:
        # First do initial log at inf score, in case we don't even get to first eval before crash/job timeout
        callback(np.full((n_obj,), np.inf, dtype=float), 0)

    suggest_time = np.zeros(n_calls)
    observe_time = np.zeros(n_calls)
    eval_time = np.zeros((n_calls, n_suggestions))
    function_evals = np.zeros((n_calls, n_suggestions, n_obj))
    suggest_log = [None] * n_calls
    for ii in range(n_calls):
        tt = time()
        try:
            next_points = optimizer.suggest(n_suggestions) # TODO 1
        except Exception as e:
            # logger.warning("Failure in optimizer suggest. Falling back to random search.")
            # logger.exception(e, exc_info=True)
            print(json.dumps({"optimizer_suggest_exception": {ITER: ii}}))
            api_config = test_problem.get_api_config()
            next_points = rs.suggest_dict([], [], api_config, n_suggestions=n_suggestions)
        suggest_time[ii] = time() - tt

        # logger.info("suggestion time taken %f iter %d next_points %s" % (suggest_time[ii], ii, str(next_points)))
        assert len(next_points) == n_suggestions, "invalid number of suggestions provided by the optimizer"

        # We could put this inside the TestProblem class, but ok here for now.

        # try:
        #     space_for_validate.validate(next_points)  # Fails if suggestions outside allowed range
        # except Exception:
        #     raise ValueError("Optimizer suggestion is out of range.")

        for jj, next_point in enumerate(next_points):
            tt = time()
            try:
                f_current_eval = test_problem.evaluate(next_point) # TODO 2
            except Exception as e:
                # logger.warning("Failure in function eval. Setting to inf.")
                # logger.exception(e, exc_info=True)
                f_current_eval = np.full((n_obj,), np.inf, dtype=float)
            eval_time[ii, jj] = time() - tt
            assert np.shape(f_current_eval) == (n_obj,)

            suggest_log[ii] = next_points
            function_evals[ii, jj, :] = f_current_eval
            # logger.info(
            #     "function_evaluation time %f value %f suggestion %s"
            #     % (eval_time[ii, jj], f_current_eval[0], str(next_point))
            # )

        # Note: this could be inf in the event of a crash in f evaluation, the optimizer must be able to handle that.
        # Only objective 0 is seen by optimizer.
        eval_list = function_evals[ii, :, 0].tolist()

        if callback is not None:
            raise NotImplementedError()
            # idx_ii, idx_jj = argmin_2d(function_evals[: ii + 1, :, 0])
            # callback(function_evals[idx_ii, idx_jj, :], ii + 1)

        tt = time()
        try:
            optimizer.observe(next_points, eval_list) # TODO 3
        except Exception as e:
            # logger.warning("Failure in optimizer observe. Ignoring these observations.")
            # logger.exception(e, exc_info=True)
            print(json.dumps({"optimizer_observe_exception": {ITER: ii}}))
        observe_time[ii] = time() - tt

        # logger.info(
        #     "observation time %f, current best %f at iter %d"
        #     % (observe_time[ii], np.min(function_evals[: ii + 1, :, 0]), ii)
        # )

    return function_evals, (suggest_time, eval_time, observe_time), suggest_log


def load_optimizer_kwargs(optimizer_name, opt_root):  # pragma: io
    """Load the kwarg options for this optimizer being tested.

    This is part of the general experiment setup before a study.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer being tested. This optimizer name must be present in optimizer config file.
    opt_root : str
        Absolute path to folder containing the config file.

    Returns
    -------
    kwargs : dict(str, object)
        The kwargs setting to pass into the optimizer wrapper constructor.
    """
    if optimizer_name in CONFIG:
        _, kwargs = CONFIG[optimizer_name]
    else:
        settings = cmd.load_optimizer_settings(opt_root)
        assert optimizer_name in settings, "optimizer %s not found in settings file %s" % optimizer_name
        _, kwargs = settings[optimizer_name]
    return kwargs
def run_sklearn_study(
    opt_class, opt_kwargs, model_name, dataset, scorer, n_calls, n_suggestions, data_root=None, callback=None
):
    """Run a study for a single optimizer on a single `sklearn` model/data set combination.

    This routine is meant for benchmarking when tuning `sklearn` models, as opposed to the more general
    :func:`.run_study`.

    Parameters
    ----------
    opt_class : :class:`.abstract_optimizer.AbstractOptimizer`
        Type of wrapper optimizer must be subclass of :class:`.abstract_optimizer.AbstractOptimizer`.
    opt_kwargs : kwargs
        `kwargs` to use when instantiating the wrapper class.
    model_name : str
        Which sklearn model we are attempting to tune, must be an element of `constants.MODEL_NAMES`.
    dataset : str
        Which data set the model is being tuned to, which must be either a) an element of
        `constants.DATA_LOADER_NAMES`, or b) the name of a csv file in the `data_root` folder for a custom data set.
    scorer : str
        Which metric to use when evaluating the model. This must be an element of `sklearn_funcs.SCORERS_CLF` for
        classification models, or `sklearn_funcs.SCORERS_REG` for regression models.
    n_calls : int
        How many iterations of minimization to run.
    n_suggestions : int
        How many parallel evaluation we run each iteration. Must be ``>= 1``.
    data_root : str
        Absolute path to folder containing custom data sets. This may be ``None`` if no custom data sets are used.``
    callback : callable
        Optional callback taking the current best function evaluation, and the number of iterations finished. Takes
        array of shape `(n_obj,)`.

    Returns
    -------
    function_evals : :class:`numpy:numpy.ndarray` of shape (n_calls, n_suggestions, n_obj)
        Value of objective for each evaluation.
    timing_evals : (:class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`, :class:`numpy:numpy.ndarray`)
        Tuple of 3 timing results: ``(suggest_time, eval_time, observe_time)`` with shapes ``(n_calls,)``,
        ``(n_calls, n_suggestions)``, and ``(n_calls,)``. These are the time to make each suggestion, the time for each
        evaluation of the objective function, and the time to make an observe call.
    suggest_log : list(list(dict(str, object)))
        Log of the suggestions corresponding to the `function_evals`.
    """
    # Setup test function
    function_instance = _build_test_problem(model_name, dataset, scorer, data_root)

    # Setup optimizer
    api_config = function_instance.get_api_config() # 优化的hp
    optimizer_instance = opt_class(api_config, **opt_kwargs)

    # assert function_instance.objective_names == OBJECTIVE_NAMES
    # assert OBJECTIVE_NAMES[0] == cc.VISIBLE_TO_OPT
    n_obj = len(function_instance.objective_names)

    # Now actually do the experiment
    function_evals, timing, suggest_log = run_study(
        optimizer_instance, function_instance, n_calls, n_suggestions, n_obj=n_obj, callback=callback
    )
    return function_evals, timing, suggest_log

def experiment_main(opt_class, args=None):  # pragma: main
    """This is in effect the `main` routine for this experiment. However, it is called from the optimizer wrapper file
    so the class can be passed in. The optimizers are assumed to be outside the package, so the optimizer class can't
    be named from inside the main function without using hacky stuff like `eval`.
    """
    if args is None:
        description = "Run a study with one benchmark function and an optimizer"
        args = cmd.parse_args(cmd.experiment_parser(description))
    # args[CmdArgs.opt_rev] = opt_class.get_version()

    opt_kwargs = load_optimizer_kwargs(args[CmdArgs.optimizer], args[CmdArgs.optimizer_root])
    function_evals, timing, suggest_log = run_sklearn_study(
        opt_class,
        opt_kwargs,
        args[CmdArgs.classifier],
        args[CmdArgs.data],
        args[CmdArgs.metric],
        args[CmdArgs.n_calls],
        args[CmdArgs.n_suggest],
        data_root=args[CmdArgs.data_root],
        callback=None,
    )
    eval_ds = build_eval_ds(function_evals, OBJECTIVE_NAMES)
    time_ds = build_timing_ds(*timing)
    suggest_ds = build_suggest_ds(suggest_log)

def main():
    description = "Run a study with one benchmark function and an optimizer"
    args = cmd.parse_args(cmd.experiment_parser(description))
    opt_class = _get_opt_class(args[CmdArgs.optimizer])
    experiment_main(opt_class, args=args)

if __name__ == '__main__':
    main()
