import numpy as np
from ext_algs.base import Ext_opt
from xbbo.core.trials import Trial, Trials
import ConfigSpace
from ConfigSpace import Configuration
from xbbo.core.constants import Key
import sys



sys.path.append('../git/HpBandSter')

import hpbandster.core.result as hpres

from hpbandster.optimizers import HyperBand as HyperBand


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
            Key.FUNC_VALUE: res[Key.FUNC_VALUE],
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
                                        Key.REGRET_TEST: res["info"][Key.REGRET_TEST],
                                        Key.REGRET_VAL: res[Key.FUNC_VALUE],
                                        Key.COST: res[Key.COST]
                                    }),
                            permit_duplicate=True)
        return ({
            'loss': float(res[Key.FUNC_VALUE]),
            'info': float(res[Key.COST])})

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
        self._inner_opt = HyperBand(configspace=cs, run_id=hb_run_id,
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



opt_class = HB_opt