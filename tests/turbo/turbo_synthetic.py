from turbo import TurboM, Turbo1

import numpy as np

def branin(x):
    x1, x2 = x[0], x[1]
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y

alg_map = {
    'turbo-2': 2,
    'turbo-1': 1,
}

def run_one_exp(opt_name, max_call, seed):
    if opt_name == 'turbo-2':
        turbo = TurboM(
            f=branin,  # Handle to objective function
            lb=np.array([-5,0]),  # Numpy array specifying lower bounds
            ub=np.array([10,15]),  # Numpy array specifying upper bounds
            n_init=
            10,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=max_call,  # Maximum number of evaluations
            n_trust_regions=2,  # Number of trust regions
            batch_size=1,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )
    else:
        turbo = Turbo1(
            f=branin,  # Handle to objective function
            lb=np.array([-5,0]),  # Numpy array specifying lower bounds
            ub=np.array([10,15]),  # Numpy array specifying upper bounds
            n_init=
            10,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=max_call,  # Maximum number of evaluations
            batch_size=1,  # How large batch size TuRBO uses
            verbose=True,  # Print information from each batch
            use_ard=True,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
            n_training_steps=50,  # Number of steps of ADAM to learn the hypers
            min_cuda=1024,  # Run on the CPU for small datasets
            device="cpu",  # "cpu" or "cuda"
            dtype="float64",  # float64 or float32
        )

    turbo.optimize()

    fX = turbo.fX  # Observed values
    losses = np.asarray(fX)
    # return np.minimum.accumulate(losses)
    return losses


if __name__ == "__main__":
    from tests.xbbo_benchmark import benchmark
    benchmark(list(alg_map.keys()),
              run_one_exp,
              200,
              10,
              42,
              desc='TuRBO')
    # rng = np.random.RandomState(42)
    # best_vals = []
    # for _ in range(3):
    #     best_val = run_one_exp('tpe', 50, rng.randint(1e5))
    #     best_vals.append(best_val)
    # print(best_vals)