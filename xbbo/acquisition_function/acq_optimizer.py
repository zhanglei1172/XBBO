import logging
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import numpy as np
import time
import scipy
from scipy.stats.qmc import Sobol


from xbbo.acquisition_function.base import AbstractAcquisitionFunction, AcquisitionFunctionMaximizer

from xbbo.configspace.space import DenseConfigurationSpace, DenseConfiguration, get_one_exchange_neighbourhood
from xbbo.core.trials import Trials
from xbbo.utils.constants import MAXINT
from xbbo.utils.util import get_types

logger = logging.getLogger(__name__)


class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~xbbo.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.configspace.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def _maximize(self,
                  trials: Trials,
                  num_points: int,
                  _sorted: bool = False,
                  **kwargs) -> List[Tuple[float, DenseConfiguration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        trials: ~xbbo.trials.trials.trials
            trials object
        stats: ~xbbo.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`xbbo.configspace.DenseConfiguration`).
        """
        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(
                size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class LocalSearch(AcquisitionFunctionMaximizer):
    """Implementation of xbbo's local search.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        Maximum number of iterations that the local search will perform

    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates

    """
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        config_space: DenseConfigurationSpace,
        rng: np.random.RandomState = np.random.RandomState(42),
        max_steps: Optional[int] = None,
        n_steps_plateau_walk: int = 10,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> List[Tuple[float, DenseConfiguration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        trials: ~xbbo.utils.history_container.Trials
            trials object
        stats: ~xbbo.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        ***kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found DenseConfiguration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        init_points = self._get_initial_points(num_points, trials)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, DenseConfiguration = self._one_iter(start_point, **kwargs)

            DenseConfiguration.origin = "Local Search"
            acq_configs.append((acq_val, DenseConfiguration))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        return acq_configs

    def _get_initial_points(self, num_points, trials):

        if trials.is_empty():
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = trials.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(
                min(len(configs_previous_runs_sorted), num_points))
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search]))

        return init_points

    def _one_iter(self, start_point: DenseConfiguration,
                  **kwargs) -> Tuple[float, DenseConfiguration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], **kwargs)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps)

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.randint(MAXINT))

            for neighbor in all_neighbors:
                s_time = time.time()
                acq_val = self.acquisition_function([neighbor], **kwargs)
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent:
                    logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and
                     local_search_steps == self.max_steps):
                logger.debug(
                    "Local search took %d steps and looked at %d "
                    "configurations. Computing the acquisition "
                    "value for one DenseConfiguration took %f seconds"
                    " on average.", local_search_steps, neighbors_looked_at,
                    np.mean(time_n))
                break

        return acq_val_incumbent, incumbent


class ScipyGlobalOptimizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy global optimizer. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def __init__(self,
                 acquisition_function: AbstractAcquisitionFunction,
                 config_space: DenseConfigurationSpace,
                 rng: np.random.RandomState = np.random.RandomState(42)):
        super().__init__(acquisition_function, config_space, rng)

        types, bounds = get_types(self.config_space)
        assert all(types == 0)
        self.bounds = bounds

    def maximize(self,
                 trials: Trials,
                 initial_config=None,
                 drop_self_duplicate: bool = False,
                 **kwargs) -> List[Tuple[float, DenseConfiguration]]:
        def negative_acquisition(x):
            # shape of x = (d,)
            return -self.acquisition_function(x,
                                              convert=False)[0]  # shape=(1,)

        acq_configs = []
        result = scipy.optimize.differential_evolution(
            func=negative_acquisition, bounds=self.bounds)
        if not result.success:
            logger.debug(
                'Scipy differential evolution optimizer failed. Info:\n%s' %
                (result, ))
        try:
            config = DenseConfiguration(self.config_space, vector=result.x)
            acq = self.acquisition_function(result.x, convert=False)
            acq_configs.append((acq, config))
        except Exception:
            pass

        if not acq_configs:  # empty
            logger.warning(
                'Scipy differential evolution optimizer failed. Return empty config list. Info:\n%s'
                % (result, ))
        configs = [config for _, config in acq_configs]
        return self.unique(configs=configs) if drop_self_duplicate else configs

    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> Iterable[Tuple[float, DenseConfiguration]]:
        raise NotImplementedError()


class RandomScipyOptimizer(AcquisitionFunctionMaximizer):
    """
    Use scipy.optimize with start points chosen by random search. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        config_space: DenseConfigurationSpace,
        rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)

        self.random_search = InterleavedLocalAndRandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng)
        self.scipy_optimizer = ScipyOptimizer(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng)

    def maximize(self,
                 trials: Trials,
                 num_points: int,
                 drop_self_duplicate: bool = False,
                 num_trials=10,
                 **kwargs) -> List[Tuple[float, DenseConfiguration]]:
        acq_configs = []

        initial_configs = self.random_search.maximize(trials, num_points,
                                                      **kwargs).challengers
        initial_acqs = self.acquisition_function(initial_configs)
        acq_configs.extend(zip(initial_acqs, initial_configs))

        success_count = 0
        for config in initial_configs[:num_trials]:
            scipy_configs = self.scipy_optimizer.maximize(
                trials, initial_config=config).challengers
            if not scipy_configs:  # empty
                continue
            scipy_acqs = self.acquisition_function(scipy_configs)
            acq_configs.extend(zip(scipy_acqs, scipy_configs))
            success_count += 1
        if success_count == 0:
            logger.warning(
                'None of Scipy optimizations are successful in RandomScipyOptimizer.'
            )

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        configs = [_[1] for _ in acq_configs]

        return self.unique(configs=configs) if drop_self_duplicate else configs

    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> Iterable[Tuple[float, DenseConfiguration]]:
        raise NotImplementedError()


class ScipyOptimizer(AcquisitionFunctionMaximizer):
    """
    Wraps scipy optimizer. Only on continuous dims.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional
    """
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        config_space: DenseConfigurationSpace,
        rng: Union[bool, np.random.RandomState] = None,
    ):
        super().__init__(acquisition_function, config_space, rng)

        types, bounds = get_types(
            self.config_space)  # todo: support constant hp in scipy optimizer
        assert all(
            types == 0
        ), 'Scipy optimizer (L-BFGS-B) only supports Integer and Float parameters.'
        self.bounds = bounds

        options = dict(disp=False, maxiter=1000)
        self.scipy_config = dict(tol=None, method='L-BFGS-B', options=options)

    def maximize(self,
                 trials: Trials,
                 initial_config=None,
                 drop_self_duplicate: bool = False,
                 **kwargs) -> List[Tuple[float, DenseConfiguration]]:
        def negative_acquisition(x):
            # shape of x = (d,)
            x = np.clip(x, 0.0, 1.0)  # fix numerical problem in L-BFGS-B
            return -self.acquisition_function(x,
                                              convert=False)[0]  # shape=(1,)

        if initial_config is None:
            initial_config = self.config_space.sample_configuration()
        init_point = initial_config.get_array(sparse=False)

        acq_configs = []
        result = scipy.optimize.minimize(fun=negative_acquisition,
                                         x0=init_point,
                                         bounds=self.bounds,
                                         **self.scipy_config)
        # if result.success:
        #     acq_configs.append((result.fun, DenseConfiguration(self.config_space, vector=result.x)))
        if not result.success:
            logger.debug('Scipy optimizer failed. Info:\n%s' % (result, ))
        try:
            x = np.clip(result.x, 0.0,
                        1.0)  # fix numerical problem in L-BFGS-B
            config = DenseConfiguration(self.config_space, vector=x)
            acq = self.acquisition_function(x, convert=False)
            acq_configs.append((acq, config))
        except Exception:
            pass

        if not acq_configs:  # empty
            logger.warning(
                'Scipy optimizer failed. Return empty config list. Info:\n%s' %
                (result, ))

        configs = [config for _, config in acq_configs]
        return self.unique(configs=configs) if drop_self_duplicate else configs

    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> Iterable[Tuple[float, DenseConfiguration]]:
        raise NotImplementedError()


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements xbbo's default acquisition function optimization.

    This acq_maximizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    acquisition_function : ~xbbo.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~xbbo.config_space.DenseConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearch] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearch] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [Local Search] number of local search iterations

    """
    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        config_space: DenseConfigurationSpace,
        rng: Union[bool, np.random.RandomState] = None,
        max_steps: Optional[int] = None,
        n_steps_plateau_walk: int = 10,
        n_sls_iterations: int = 10,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng)
        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk)
        self.n_sls_iterations = n_sls_iterations

        # =======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        # =======================================================================

    def maximize(self,
                 trials: Trials,
                 num_points: int,
                 drop_self_duplicate: bool = False,
                 **kwargs) -> Iterable[DenseConfiguration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        trials: ~xbbo.utils.history_container.Trials
            trials object
        num_points: int
            number of points to be sampled
        random_configuration_chooser: ~xbbo.acq_maximizer.random_configuration_chooser.RandomConfigurationChooser
            part of the returned ChallengerList such
            that we can interleave random configurations
            by a scheme defined by the random_configuration_chooser;
            random_configuration_chooser.next_smbo_iteration()
            is called at the end of this function
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[DenseConfiguration]
            to be concrete: ~xbbo.ei_optimization.ChallengerList
        """

        next_configs_by_local_search = self.local_search._maximize(
            trials, self.n_sls_iterations, **kwargs)

        # Get configurations sorted by EI
        new_kwargs = {"_sorted":True}
        new_kwargs.update(kwargs)
        next_configs_by_random_search_sorted = self.random_search._maximize(
            trials,
            num_points - len(next_configs_by_local_search),
            **new_kwargs)

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of xbbo. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (next_configs_by_random_search_sorted +
                                     next_configs_by_local_search)
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]]))
        configs = [_[1] for _ in next_configs_by_acq_value]
        return self.unique(configs=configs) if drop_self_duplicate else configs

    def _maximize(self, trials: Trials, num_points: int,
                  **kwargs) -> Iterable[Tuple[float, DenseConfiguration]]:
        raise NotImplementedError()



    # """Get candidate solutions via random sampling of configurations.

    # Parameters
    # ----------
    # acquisition_function : ~xbbo.optimizer.acquisition.AbstractAcquisitionFunction

    # config_space : ~xbbo.configspace.DenseConfigurationSpace

    # rng : np.random.RandomState or int, optional
    # """
    # def __init__(
    #     self,
    #     acquisition_function: AbstractAcquisitionFunction,
    #     config_space: DenseConfigurationSpace,
    #     rng: np.random.RandomState = np.random.RandomState(42),
    #     design_method: Optional[str] = 'sobol',
    # ):
    #     super().__init__(acquisition_function, config_space, rng)
    #     self.design_method = design_method
    #     types, bounds = get_types(self.config_space)
    #     self.bounds = bounds
    #     self.dim = len(self.bounds)

    # def _maximize(self,
    #               trials: Trials,
    #               num_points: int,
    #               _sorted: bool = False,
    #               **kwargs) -> List[Tuple[float, DenseConfiguration]]:
    #     """Design sampled configurations

    #     Parameters
    #     ----------
    #     trials: ~xbbo.trials.trials.trials
    #         trials object
    #     stats: ~xbbo.stats.stats.Stats
    #         current stats object
    #     num_points: int
    #         number of points to be sampled
    #     _sorted: bool
    #         whether random configurations are sorted according to acquisition function

    #     Returns
    #     -------
    #     iterable
    #         An iterable consistng of
    #         tuple(acqusition_value, :class:`xbbo.configspace.DenseConfiguration`).
    #     """
    #     X = trials.get_array()
    #     fX = np.asarray(trials._his_observe_value)
    #     length_scale = kwargs.get('length_scale', np.ones(self.dim))
    #     length = kwargs['length']
    #     weights = np.array(length_scale)
    #     # Create the trust region boundaries
    #     x_center = X[fX.argmin().item(), :][None, :]
    #     weights = weights / weights.mean(
    #     )  # This will make the next line more stable
    #     weights = weights / np.prod(np.power(
    #         weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
    #     lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
    #     ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

    #     # Draw a Sobolev sequence in [lb, ub]
    #     sobol_gen = Sobol(d=self.dim,
    #                       scramble=True,
    #                       seed=self.rng.randint(MAXINT))
    #     pert = sobol_gen.random(num_points)
    #     pert = lb + (ub - lb) * pert

    #     # Create a perturbation mask
    #     prob_perturb = min(20.0 / self.dim, 1.0)
    #     mask = self.rng.rand(num_points,
    #                          self.dim) <= prob_perturb
    #     ind = np.where(np.sum(mask, axis=1) == 0)[0]
    #     mask[ind,
    #          self.rng.randint(0, self.dim - 1, size=len(ind))] = 1

    #     # Create candidate points
    #     X_cand = x_center.copy() * np.ones(
    #         (num_points, self.dim))
    #     X_cand[mask] = pert[mask]
        
    #     acq_and_configs = []
    #     for n in num_points:
    #         config = DenseConfiguration.from_array(self.config_space, X_cand, use_dense=trials.use_dense)
    #         y_cand = self.acquisition_function(config)
    #         acq_and_configs.append((y_cand, config))