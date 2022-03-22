import numpy as np
from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from smac.initial_design.latin_hypercube_design import LHDesign
from ext_algs.base import Ext_opt

from xbbo.core.trials import Trial, Trials
from xbbo.utils.constants import Key

class SMAC_opt(Ext_opt):
    def __init__(self,
                 cs,
                 objective_function=None,
                 budget_bound=[0, np.inf],
                 n_iters=None,
                 seed=0,
                 **kwargs) -> None:
        super().__init__(cs,
                         objective_function,
                         budget_bound=budget_bound,
                         seed=seed)
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": n_iters,
            "cs": cs,
            "deterministic": "false",
            "initial_incumbent": "RANDOM",
            "output_dir": ""
        })
        np.random.seed(self.seed)
        def obj_func(trial, **kwargs):
            res = objective_function(trial, **kwargs)
            return res[Key.REGRET_VAL], res
            
        self._inner_opt = SMAC(scenario=scenario,
                               tae_runner=obj_func,
                               initial_design=LHDesign)

    def _optimize(self):
        self._inner_opt.optimize()
        self.his = self._inner_opt.runhistory
        return self._calc_trials(self.his.data)

    def _calc_trials(self, data):
        # valid_regret = []
        # test_regret = []
        # costs = []
        
        trials = Trials(self.dim)
        for k, v in data.items():
            curr_regret = v.cost
            config = self.his.ids_config[k.config_id]
            dic = config.get_dictionary()
            curr_test_regret = v.additional_info[Key.REGRET_TEST]
            trials.add_a_trial(
                Trial(config,
                      dic,
                      observe_value=curr_regret,
                      info={
                          Key.REGRET_TEST: curr_test_regret,
                          Key.REGRET_VAL: curr_regret,
                          Key.COST: v.additional_info[Key.COST]
                      }), permit_duplicate=True)
            # valid_regret.append(curr_regret)
            # test_regret.append(curr_test_regret)
            # if "cost" in v.additional_info:
            #     costs.append(v.additional_info["cost"])
            # else:
            #     costs.append(v.additional_info.get(Key.BUDGET))

        return trials

opt_class = SMAC_opt