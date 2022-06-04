import numpy as np
from ext_algs.base import Ext_opt
from xbbo.core.trials import Trial, Trials
import ConfigSpace
from ConfigSpace import Configuration
from xbbo.core.constants import Key
import sys



sys.path.append('../HpBandSter')

import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB


from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns


# logging.basicConfig(level=logging.ERROR)


class MyWorker(Worker):

    def __init__(self, cs, trials, objective_function, run_id, id, nameserver, nameserver_port):
        super(MyWorker, self).__init__(run_id=run_id, id=id,
                                       nameserver=nameserver, nameserver_port=nameserver_port)
        self.objective_function = objective_function
        self.trials = trials
        self.cs = cs

    def compute(self, config, budget, **kwargs):
        config = Configuration(self.cs, config)
        dic = config.get_dictionary()
        res = self.objective_function(config, budget=budget, **kwargs)
        res = {
            "fitness": res[Key.FUNC_VALUE],
            "cost": res.get(Key.COST, budget),
            "info": {
                "test_loss": res.get(Key.REGRET_TEST, 0),
                "budget": budget
            }
        }
        self.trials.add_a_trial(Trial(config,
                                    dic,
                                    observe_value=res["fitness"],
                                    info={
                                        Key.REGRET_TEST: res["info"]["test_loss"],
                                        Key.REGRET_VAL: res["fitness"],
                                        Key.COST: res["cost"]
                                    }),
                            permit_duplicate=True)
        return ({
            'loss': float(res["fitness"]),
            'info': float(res["cost"])})

class BOHB_opt(Ext_opt):
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
        hb_run_id = '0'
        self.NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
        ns_host, ns_port = self.NS.start()
        num_workers = kwargs.get('num_works', 1)
        workers = []
        self.trials = Trials(cs,len(cs.get_hyperparameter_names()))
        for i in range(num_workers):
            w = MyWorker(cs, self.trials,objective_function=objective_function, run_id=hb_run_id, id=i,
                        nameserver=ns_host, nameserver_port=ns_port)
            w.run(background=True)
            workers.append(w)
        kwargs["eta"] = kwargs.get("eta", 3)
        # kwargs["n_workers"] = kwargs.get("n_workers", 1)
        # kwargs["brackets"] = kwargs.get("bracket_limit", 50)
        self._inner_opt = BOHB(configspace=cs, run_id=hb_run_id,
            min_budget=self.min_budget, max_budget=self.max_budget,
            nameserver=ns_host, nameserver_port=ns_port)

        self.kwargs = kwargs

    def _optimize(self):
        res =  self._inner_opt.run(n_iterations=self.kwargs.get('bracket_limit', 50))

        # Step 4: Shutdown
        # After the optimizer run, we must shutdown the master and the nameserver.
        self._inner_opt.shutdown(shutdown_workers=True)
        self.NS.shutdown()
        # self.his = self._inner_opt.runhistory
        return self.trials



opt_class = BOHB_opt