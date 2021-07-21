from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from bbomark.constants import MODEL_NAMES
# We should add cat variables into some of these configurations but a lot of
# the wrappers for the BO methods really have trouble with cat types.

# kNN
knn_cfg = {
    "n_neighbors": {"type": "int", "space": "linear", "range": (1, 25)},
    "p": {"type": "int", "space": "linear", "range": (1, 4)},
}

# SVM
svm_cfg = {
    "C": {"type": "real", "space": "log", "range": (1.0, 1e3)},
    "gamma": {"type": "real", "space": "log", "range": (1e-4, 1e-3)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
}

# DT
dt_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# RF
rf_cfg = {
    "max_depth": {"type": "int", "space": "linear", "range": (1, 15)},
    "max_features": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_split": {"type": "real", "space": "logit", "range": (0.01, 0.99)},
    "min_samples_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_weight_fraction_leaf": {"type": "real", "space": "logit", "range": (0.01, 0.49)},
    "min_impurity_decrease": {"type": "real", "space": "linear", "range": (0.0, 0.5)},
}

# MLP with ADAM
mlp_adam_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "beta_1": {"type": "real", "space": "logit", "range": (0.5, 0.99)},
    "beta_2": {"type": "real", "space": "logit", "range": (0.9, 1.0 - 1e-6)},
    "epsilon": {"type": "real", "space": "log", "range": (1e-9, 1e-6)},
}

# MLP with SGD
mlp_sgd_cfg = {
    "hidden_layer_sizes": {"type": "int", "space": "linear", "range": (50, 200)},
    "alpha": {"type": "real", "space": "log", "range": (1e-5, 1e1)},
    "batch_size": {"type": "int", "space": "linear", "range": (10, 250)},
    "learning_rate_init": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "power_t": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "momentum": {"type": "real", "space": "logit", "range": (0.001, 0.999)},
    "validation_fraction": {"type": "real", "space": "logit", "range": (0.1, 0.9)},
}

# AdaBoostClassifier
ada_cfg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

# lasso
lasso_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

# linear
linear_cfg = {
    "C": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "intercept_scaling": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
}

MODELS_CLF = {
    "kNN": (KNeighborsClassifier, {}, knn_cfg),
    "SVM": (SVC, {"kernel": "rbf", "probability": True}, svm_cfg),
    "DT": (DecisionTreeClassifier, {"max_leaf_nodes": None}, dt_cfg),
    "RF": (RandomForestClassifier, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "MLP-adam": (MLPClassifier, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-sgd": (
        MLPClassifier,
        {"solver": "sgd", "early_stopping": True, "learning_rate": "invscaling", "nesterovs_momentum": True},
        mlp_sgd_cfg,
    ),
    "ada": (AdaBoostClassifier, {}, ada_cfg),
    "lasso": (
        LogisticRegression,
        {"penalty": "l1", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        lasso_cfg,
    ),
    "linear": (
        LogisticRegression,
        {"penalty": "l2", "fit_intercept": True, "solver": "liblinear", "multi_class": "ovr"},
        linear_cfg,
    ),
}

# For now, we will assume the default is to go thru all classifiers
assert sorted(MODELS_CLF.keys()) == sorted(MODEL_NAMES)

ada_cfg_reg = {
    "n_estimators": {"type": "int", "space": "linear", "range": (10, 100)},
    "learning_rate": {"type": "real", "space": "log", "range": (1e-4, 1e1)},
}

lasso_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    "normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-5, 1e-1)},
    "positive": {"type": "bool"},
}

linear_cfg_reg = {
    "alpha": {"type": "real", "space": "log", "range": (1e-2, 1e2)},
    "fit_intercept": {"type": "bool"},
    "normalize": {"type": "bool"},
    "max_iter": {"type": "int", "space": "log", "range": (10, 5000)},
    "tol": {"type": "real", "space": "log", "range": (1e-4, 1e-1)},
}

MODELS_REG = {
    "kNN": (KNeighborsRegressor, {}, knn_cfg),
    "SVM": (SVR, {"kernel": "rbf"}, svm_cfg),
    "DT": (DecisionTreeRegressor, {"max_leaf_nodes": None}, dt_cfg),
    "RF": (RandomForestRegressor, {"n_estimators": 10, "max_leaf_nodes": None}, rf_cfg),
    "MLP-adam": (MLPRegressor, {"solver": "adam", "early_stopping": True}, mlp_adam_cfg),
    "MLP-sgd": (
        MLPRegressor,  # regression crashes often with relu
        {
            "activation": "tanh",
            "solver": "sgd",
            "early_stopping": True,
            "learning_rate": "invscaling",
            "nesterovs_momentum": True,
        },
        mlp_sgd_cfg,
    ),
    "ada": (AdaBoostRegressor, {}, ada_cfg_reg),
    "lasso": (Lasso, {}, lasso_cfg_reg),
    "linear": (Ridge, {"solver": "auto"}, linear_cfg_reg),
}