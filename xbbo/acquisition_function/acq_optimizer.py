from typing import Iterator, List, Optional, Tuple, Union
import numpy as np
import time
from xbbo.acquisition_function.base import AbstractAcquisitionFunction, AcquisitionFunctionMaximizer

from xbbo.configspace.space import DenseConfigurationSpace, DenseConfiguration, get_one_exchange_neighbourhood
from xbbo.core.trials import Trials
from xbbo.utils.constants import MAXINT


class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            trials: Trials,
            num_points: int,
            _sorted: bool = False,
    ) -> List[Tuple[float, DenseConfiguration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        trials: ~smac.trials.trials.trials
            trials object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.DenseConfiguration`).
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
    """Implementation of openbox's local search.

    Parameters
    ----------
    acquisition_function : ~openbox.acquisition_function.acquisition.AbstractAcquisitionFunction

    config_space : ~openbox.config_space.ConfigurationSpace

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
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(
            self,
            trials: Trials,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float, DenseConfiguration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        trials: ~openbox.utils.history_container.Trials
            trials object
        stats: ~openbox.stats.stats.Stats
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

        init_points = self._get_initial_points(
            num_points, trials)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, DenseConfiguration = self._one_iter(
                start_point, **kwargs)

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
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(
            self,
            start_point: DenseConfiguration,
            **kwargs
    ) -> Tuple[float, DenseConfiguration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = self.acquisition_function([incumbent], **kwargs)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                self.logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

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
                    # self.logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and
                     local_search_steps == self.max_steps):
                # self.logger.debug("Local search took %d steps and looked at %d "
                #                   "configurations. Computing the acquisition "
                #                   "value for one DenseConfiguration took %f seconds"
                #                   " on average.",
                #                   local_search_steps, neighbors_looked_at,
                #                   np.mean(time_n))
                break

        return acq_val_incumbent, incumbent
