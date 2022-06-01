import time
import numpy as np
from ext_algs.base import Ext_opt
from xbbo.core.trials import Trial, Trials
import ConfigSpace
from ConfigSpace import Configuration
from xbbo.core.constants import Key
import sys
from multiprocessing import Process


sys.path.append('../open-box')

from openbox.apps.multi_fidelity.mq_mfes import mqMFES
from openbox.apps.multi_fidelity.mq_mf_worker import mqmfWorker
# logging.basicConfig(level=logging.ERROR)

class Float(float):
    regret_test = None

class HB_opt(Ext_opt):
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
        self.port = kwargs.get("port", 13577)
        self.cs = cs
        self.trials = Trials(cs,len(cs.get_hyperparameter_names()))
        new_max_budget = self.max_budget / self.min_budget
        new_min_budget = 1
        old_min_budget = self.min_budget

        def obj(config, n_resource, extra_conf):
            budget = n_resource * old_min_budget
            res = objective_function(config, budget=budget, **kwargs)
            # config = Configuration(cs, dic)
            # res = {
            #     Key.FUNC_VALUE:res[Key.FUNC_VALUE],  # minimize
            #     Key.COST: res.get(Key.COST, n_resource),
            #     "info": {
            #     Key.REGRET_TEST: res.get(Key.REGRET_TEST, 0),
            #     Key.BUDGET: n_resource
            # }}
            # trials.append(Trial(config,
            #                          config.get_dictionary(),
            #                          observe_value=res[Key.FUNC_VALUE],
            #                          info={
            #                              Key.REGRET_TEST: res["info"][Key.REGRET_TEST],
            #                              Key.REGRET_VAL: res[Key.FUNC_VALUE],
            #                              Key.COST: res[Key.COST]
            #                          }))
            obs = Float(res[Key.FUNC_VALUE])
            obs.regret_test = res.get(Key.REGRET_TEST, 0)
            result = {
                 "objective_value":obs,
            }
            return result
        # def work():
        #     worker = mqmfWorker(obj, '127.0.0.1', self.port, authkey=b'abc')
        #     worker.run()

        self._inner_opt  = mqMFES(
            None, cs, new_max_budget, eta=kwargs.get("eta", 3),
            num_iter=kwargs.get('round_limit', 50), random_state=seed,
            method_id='-', restart_needed=True,
            time_limit_per_trial=999999,
            runtime_limit=np.inf,
            ip='127.0.0.1', port=self.port, authkey=b'abc'
        )
        self._inner_opt.iterate_r = (self._inner_opt.R * self._inner_opt.eta ** -np.linspace(
                    start=self._inner_opt.s_max, stop=0, num=self._inner_opt.s_max+1)).astype('int').tolist()
        self._inner_opt.target_x = {k:[] for k in self._inner_opt.iterate_r}
        self._inner_opt.target_y = {k:[] for k in self._inner_opt.iterate_r}
        map_old = self._inner_opt.weighted_surrogate.surrogate_r.copy()
        self._inner_opt.weighted_surrogate.surrogate_r = self._inner_opt.iterate_r.copy()
        self._inner_opt.weighted_surrogate.surrogate_container = {self._inner_opt.weighted_surrogate.surrogate_r[i]:self._inner_opt.weighted_surrogate.surrogate_container[map_old[i]] for i in range(len(map_old))}
        self._inner_opt.weighted_surrogate.surrogate_weight = {self._inner_opt.weighted_surrogate.surrogate_r[i]:self._inner_opt.weighted_surrogate.surrogate_weight[map_old[i]] for i in range(len(map_old))}
        self.p=Process(target=work, args=())
        self.p.start()

        self.kwargs = kwargs

    def _optimize(self):
        self._inner_opt.run()
        self.p.terminate()
        self._calc_trials(self._inner_opt.recorder)
        self._inner_opt.master_messager.workerQueue._manager.shutdown()
        self.p.join()
        self._inner_opt.master_messager.workerQueue._manager.join()
        # time.sleep(10)
        return self.trials

    def _calc_trials(self, data):
        # valid_regret = []
        # test_regret = []
        # costs = []
        
        for obs in data:
            r_info = obs["return_info"]
            curr_regret = r_info["loss"]
            config = obs["configuration"]
            dic = config.get_dictionary()
            curr_test_regret = curr_regret.regret_test
            self.trials.add_a_trial(
                Trial(config,
                      dic,
                      observe_value=curr_regret,
                      info={
                          Key.REGRET_TEST: curr_test_regret,
                          Key.REGRET_VAL: curr_regret,
                          Key.COST: r_info["n_iteration"] * self.min_budget
                      }), permit_duplicate=True)



opt_class = HB_opt