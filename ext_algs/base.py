from abc import abstractmethod
import numpy as np
import ConfigSpace
from xbbo.core.trials import Trial, Trials


class Ext_opt():
    def __init__(self,
                 cs: ConfigSpace,
                 objective_function=None,
                 budget_bound=[0, np.inf],
                 seed=0,
                 **kwargs) -> None:
        '''
        ::objective_function: _call_obj(self, trial, **kwargs),
            trial: Trial or Configuration
            kwargs: kwargs[Key.BUDGET]...
            ::return: dict[Key.COST]...
        '''
        self._inner_opt = None
        self.cs = cs
        self.dim = len(self.cs.get_hyperparameter_names())
        self.min_budget, self.max_budget = budget_bound
        self.seed = seed
        self.objective_function = objective_function


    def optimize(self):
        self.trials : Trials = self._optimize()

    @abstractmethod
    def _optimize(self) -> Trials:
        pass
    