import numpy as np
from ext_algs.base import Ext_opt
from xbbo.core.trials import Trial, Trials
import ConfigSpace
from ConfigSpace import Configuration
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.constants import Key
import sys

sys.path.append('../git/DEHB')

from dehb import DEHB


class DEHB_opt(Ext_opt):
    def __init__(self,
                 cs: ConfigSpace,
                 objective_function=None,
                 budget_bound=[0, np.inf],
                 seed=0,
                 **kwargs) -> None:
        super().__init__(cs,
                         objective_function,
                         budget_bound=budget_bound,
                         seed=seed)

        np.random.seed(self.seed)
        self.trials = Trials(self.dim)

        def obj_func(config, budget, **kwargs):
            # config = Configuration(cs, config)
            dic = config.get_dictionary()
            res = objective_function(config, budget=budget, **kwargs)
            r = {
                "fitness": res[Key.FUNC_VALUE],
                Key.COST: res.get(Key.COST, budget),
                "info": {
                    Key.REGRET_TEST: res.get(Key.REGRET_TEST, 0),
                    Key.BUDGET: budget
                }
            }
            self.trials.add_a_trial(Trial(config,
                                     dic,
                                     observe_value=res[Key.FUNC_VALUE],
                                     info={
                                         Key.REGRET_TEST: r["info"][Key.REGRET_TEST],
                                         Key.REGRET_VAL: res[Key.FUNC_VALUE],
                                         Key.COST: r[Key.COST]
                                     }),
                               permit_duplicate=True)
            return r

        kwargs["eta"] = kwargs.get("eta", 3)
        kwargs["n_workers"] = kwargs.get("n_workers", 1)
        kwargs["brackets"] = kwargs.get("bracket_limit", 50)
        self._inner_opt = DEHB(
            f=obj_func,
            cs=cs,
            dimensions=len(cs.get_hyperparameter_names()),
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            # output_path=None,
            # if client is not None and of type Client, n_workers is ignored
            # if client is None, a Dask client with n_workers is set up
            client=None,
            **kwargs)
        self.kwargs = kwargs

    def _optimize(self):
        traj, runtime, history = self._inner_opt.run(
            # total_cost=args.runtime,
            # verbose=args.verbose,
            # arguments below are part of **kwargs shared across workers
            # train_set=train_set,
            # valid_set=valid_set,
            # test_set=test_set,
            # single_node_with_gpus=single_node_with_gpus,
            **self.kwargs)
        # self.his = self._inner_opt.runhistory
        return self.trials



opt_class = DEHB_opt