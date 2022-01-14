import numpy as np
import matplotlib.pyplot as plt

from smac3_svm_cv import run_one_exp as smac3_svm_opt
from xbbo_svm_cv import run_one_exp as xbbo_svm_opt
from smac3_synthetic import run_one_exp as smac3_synthetic_opt
from xbbo_synthetic import run_one_exp as xbbo_synthetic_opt

if __name__ == "__main__":
    rng = np.random.RandomState(42)
    result_smac3 = []
    result_xbbo = []
    for i in range(3):
        seed = rng.randint(1e5)
        result_smac3.append(smac3_synthetic_opt(seed))
        result_xbbo.append(xbbo_synthetic_opt(seed))
    result_smac3 = np.array(result_smac3)
    result_xbbo = np.array(result_xbbo)
    plt.figure()
    plt.plot(range(1,31), np.mean(result_xbbo,axis=0)[:], label='XBBO')
    plt.plot(range(1,31), np.mean(result_smac3,axis=0)[:], label='SMAC3')
    plt.ylim([-0.1,1000])
    plt.xlabel('# of Evaluate')
    plt.ylabel('OBJ')
    plt.title('Average of cumulate best on 3 seeds')
    plt.legend()
    plt.savefig('./out/comp_with_smac3.png')
    plt.show()

    # rng = np.random.RandomState(42)
    # result_smac3 = []
    # result_xbbo = []
    # for i in range(3):
    #     seed = rng.randint(1e5)
    #     result_xbbo.append(xbbo_svm_opt(seed))
    #     result_smac3.append(smac3_svm_opt(seed))
    # result_smac3 = np.array(result_smac3)
    # result_xbbo = np.array(result_xbbo)
    # plt.figure()
    # plt.plot(range(1,51), np.mean(result_xbbo,axis=0)[:], label='XBBO')
    # plt.plot(range(1,51), np.mean(result_smac3,axis=0)[:], label='SMAC3')
    # plt.ylim([0,0.1])
    # plt.xlabel('# of Evaluate')
    # plt.ylabel('OBJ')
    # plt.title('Average of cumulate best on 3 seeds')
    # plt.legend()
    # plt.savefig('./out/comp_with_smac3_2.png')
    # plt.show()
