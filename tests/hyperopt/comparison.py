from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from hyperopt_synthetic import run_one_exp as hyperopt_synthetic_opt
from xbbo_synthetic import run_one_exp as xbbo_synthetic_opt

max_call = 50
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    result_opts = defaultdict(list)
    for i in range(3):
        seed = rng.randint(1e5)
        # result_opts['hyperopt-rand'].append(hyperopt_synthetic_opt('rand', max_call,seed))
        result_opts['hyperopt-tpe'].append(hyperopt_synthetic_opt('tpe', max_call,seed))
        # result_opts['hyperopt-atpe'].append(hyperopt_synthetic_opt('atpe', max_call,seed))
        # result_opts['hyperopt-mix'].append(hyperopt_synthetic_opt('mix', max_call,seed))
        result_opts['hyperopt-anneal'].append(hyperopt_synthetic_opt('anneal', max_call,seed))
        result_opts['XBBO-tpe'].append(xbbo_synthetic_opt('tpe', max_call,seed))
        result_opts['XBBO-anneal'].append(xbbo_synthetic_opt('anneal',max_call,seed))
    plt.figure()
    for key in result_opts:
        plt.plot(range(1,max_call+1), np.mean(np.asarray(result_opts[key]),axis=0)[:], label=key)
    plt.ylim([-0.1,1000])
    plt.xlabel('# of Evaluate')
    plt.ylabel('OBJ')
    plt.title('Average of cumulate best on 3 seeds')
    plt.legend()
    plt.savefig('./out/comp_with_hyperopt.png')
    plt.show()

